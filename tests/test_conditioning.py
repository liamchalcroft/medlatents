"""Tests for medlatents.conditioning.

Covers:
- Frozen encoder contract (shapes/dtypes, freezing) using a tiny stub encoder
  so no pretrained weights are downloaded.
- SliceWiseEncoder 3D->2D handling and aggregation modes.
- ConditioningBundle assembly, device/property helpers, CFG dropout, null bundle,
  from_batch key resolution, and combined-embedding fusion.
- collate_conditional_batch: keys, stacking, variable-length padding/truncation.
"""

from __future__ import annotations

import pytest
import torch

from medlatents.conditioning import (
    ConditionalBatchConfig,
    ConditioningBundle,
    ConditioningConfig,
    FrozenEncoder,
    SliceWiseEncoder,
    collate_conditional_batch,
)
from medlatents.conditioning.collate import (
    _stack_or_pad,
    create_conditional_collate_fn,
)
from medlatents.conditioning.encoders import create_encoder


# ---------------------------------------------------------------------------
# Tiny stub encoder: avoids any network / pretrained-weight download.
# ---------------------------------------------------------------------------
class StubEncoder(FrozenEncoder):
    """Deterministic patch encoder producing [B, num_patches, embed_dim]."""

    def __init__(self, embed_dim: int = 8, patch_size: int = 16, num_patches: int = 4):
        super().__init__()
        self._embed_dim = embed_dim
        self._patch_size = patch_size
        self._supports_3d = False
        self.num_patches = num_patches
        # A trivial learnable param so freeze() has something to act on.
        self.proj = torch.nn.Linear(1, embed_dim)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        # Map mean intensity of each (fake) patch through a linear layer.
        feat = x.flatten(1).mean(dim=1, keepdim=True)  # [B, 1]
        emb = self.proj(feat)  # [B, embed_dim]
        return emb.unsqueeze(1).expand(batch, self.num_patches, self._embed_dim).contiguous()


class TestFrozenEncoderContract:
    def test_stub_encoder_output_shape_and_dtype(self):
        torch.manual_seed(0)
        enc = StubEncoder(embed_dim=8, num_patches=4)
        x = torch.rand(3, 1, 16, 16)
        out = enc(x)
        assert out.shape == (3, 4, 8)
        assert out.dtype == torch.float32

    def test_freeze_disables_grad_and_sets_eval(self):
        enc = StubEncoder()
        enc.freeze()
        assert not enc.training
        assert all(not p.requires_grad for p in enc.parameters())

    def test_encoder_properties(self):
        enc = StubEncoder(embed_dim=12, patch_size=14)
        assert enc.embed_dim == 12
        assert enc.patch_size == 14
        assert enc.supports_3d is False

    def test_preprocess_default_is_identity(self):
        enc = StubEncoder()
        x = torch.rand(2, 1, 16, 16)
        assert torch.equal(enc.preprocess(x), x)


class TestSliceWiseEncoder:
    def test_2d_input_passthrough(self):
        enc = StubEncoder(embed_dim=8, num_patches=4)
        wrapper = SliceWiseEncoder(enc, aggregation="none")
        x = torch.rand(2, 1, 16, 16)
        out = wrapper(x)
        assert out.shape == (2, 4, 8)

    def test_3d_none_aggregation_flattens_depth(self):
        enc = StubEncoder(embed_dim=8, num_patches=4)
        wrapper = SliceWiseEncoder(enc, aggregation="none")
        d = 3
        x = torch.rand(2, 1, d, 16, 16)
        out = wrapper(x)
        # [batch, D*num_patches, embed_dim]
        assert out.shape == (2, d * 4, 8)

    def test_3d_mean_aggregation(self):
        enc = StubEncoder(embed_dim=8, num_patches=4)
        wrapper = SliceWiseEncoder(enc, aggregation="mean")
        x = torch.rand(2, 1, 3, 16, 16)
        out = wrapper(x)
        # mean over slices -> [batch, num_patches, embed_dim]
        assert out.shape == (2, 4, 8)

    def test_3d_attention_aggregation(self):
        torch.manual_seed(1)
        enc = StubEncoder(embed_dim=8, num_patches=4)
        wrapper = SliceWiseEncoder(enc, aggregation="attention")
        x = torch.rand(2, 1, 3, 16, 16)
        out = wrapper(x)
        # Attention aggregation collapses only the slice (D) dimension via learned
        # weights, leaving the patch dimension intact -> [batch, num_patches, embed_dim].
        # NOTE: possible doc mismatch -- the SliceWiseEncoder.forward docstring implies
        # the "attention" branch returns [batch, embed_dim], but it actually keeps the
        # patch axis and returns [batch, num_patches, embed_dim].
        assert out.shape == (2, 4, 8)
        assert torch.isfinite(out).all()

    def test_embed_dim_property_delegates(self):
        enc = StubEncoder(embed_dim=17)
        wrapper = SliceWiseEncoder(enc)
        assert wrapper.embed_dim == 17


class TestCreateEncoderFactory:
    def test_unknown_encoder_type_raises(self):
        with pytest.raises(ValueError):
            create_encoder("not-a-real-encoder")


# ---------------------------------------------------------------------------
# ConditioningConfig
# ---------------------------------------------------------------------------
class TestConditioningConfig:
    def test_defaults_validate(self):
        cfg = ConditioningConfig()
        cfg.validate()  # should not raise

    def test_class_requires_num_classes(self):
        cfg = ConditioningConfig(use_class=True, num_classes=0)
        with pytest.raises(ValueError):
            cfg.validate()

    def test_spatial_requires_channels(self):
        cfg = ConditioningConfig(use_spatial=True, spatial_channels=0)
        with pytest.raises(ValueError):
            cfg.validate()

    def test_segmentation_requires_classes(self):
        cfg = ConditioningConfig(use_segmentation=True, num_segmentation_classes=0)
        with pytest.raises(ValueError):
            cfg.validate()

    def test_metadata_requires_dim(self):
        cfg = ConditioningConfig(use_metadata=True, metadata_dim=0)
        with pytest.raises(ValueError):
            cfg.validate()

    def test_valid_full_config(self):
        cfg = ConditioningConfig(
            use_class=True,
            num_classes=10,
            use_spatial=True,
            spatial_channels=4,
            use_segmentation=True,
            num_segmentation_classes=5,
            use_metadata=True,
            metadata_dim=8,
        )
        cfg.validate()


# ---------------------------------------------------------------------------
# ConditioningBundle
# ---------------------------------------------------------------------------
class TestConditioningBundle:
    def test_batch_size_and_device(self):
        t = torch.rand(4)
        bundle = ConditioningBundle(timesteps=t)
        assert bundle.batch_size == 4
        assert bundle.device == t.device

    def test_to_moves_tensors(self):
        t = torch.rand(2)
        y = torch.randint(0, 5, (2,))
        bundle = ConditioningBundle(timesteps=t, class_labels=y)
        moved = bundle.to("cpu")
        assert moved.timesteps.device.type == "cpu"
        assert moved.class_labels.device.type == "cpu"

    def test_cfg_dropout_force_drop_uses_null_class(self):
        t = torch.rand(4)
        y = torch.tensor([1, 2, 3, 4])
        bundle = ConditioningBundle(timesteps=t, class_labels=y, null_class=0)
        force = torch.tensor([1, 0, 1, 0])  # drop indices 0 and 2
        out = bundle.apply_cfg_dropout(class_dropout_prob=1.0, force_drop_ids=force)
        assert out.class_labels[0].item() == 0
        assert out.class_labels[1].item() == 2
        assert out.class_labels[2].item() == 0
        assert out.class_labels[3].item() == 4

    def test_cfg_dropout_without_null_class_uses_max_plus_one(self):
        t = torch.rand(3)
        y = torch.tensor([0, 1, 2])
        bundle = ConditioningBundle(timesteps=t, class_labels=y, null_class=None)
        force = torch.tensor([1, 1, 1])
        out = bundle.apply_cfg_dropout(class_dropout_prob=1.0, force_drop_ids=force)
        # null token == max+1 == 3
        assert torch.all(out.class_labels == 3)

    def test_cfg_dropout_zero_prob_is_noop(self):
        t = torch.rand(4)
        y = torch.tensor([1, 2, 3, 4])
        bundle = ConditioningBundle(timesteps=t, class_labels=y)
        out = bundle.apply_cfg_dropout(class_dropout_prob=0.0)
        assert torch.equal(out.class_labels, y)

    def test_cfg_dropout_text_zeroing(self):
        t = torch.rand(2)
        text = torch.randn(2, 5, 8)
        pooled = torch.randn(2, 8)
        bundle = ConditioningBundle(
            timesteps=t, text_embeddings=text, text_pooled=pooled, null_text=None
        )
        # Force-drop both via prob 1.0 (text path uses random; use seed for determinism).
        torch.manual_seed(0)
        out = bundle.apply_cfg_dropout(text_dropout_prob=1.0)
        assert torch.all(out.text_embeddings == 0)
        assert torch.all(out.text_pooled == 0)

    def test_get_null_bundle_class(self):
        t = torch.rand(3)
        y = torch.tensor([1, 2, 3])
        bundle = ConditioningBundle(timesteps=t, class_labels=y, null_class=0)
        null = bundle.get_null_bundle()
        assert torch.all(null.class_labels == 0)
        assert null._is_null is True

    def test_get_null_bundle_drops_spatial_and_metadata(self):
        t = torch.rand(2)
        bundle = ConditioningBundle(
            timesteps=t,
            spatial_condition=torch.randn(2, 3, 8),
            segmentation_mask=torch.randint(0, 4, (2, 8)),
            metadata=torch.randn(2, 5),
        )
        null = bundle.get_null_bundle()
        assert null.spatial_condition is None
        assert null.segmentation_mask is None
        assert null.metadata is None

    def test_get_null_bundle_text_zeroing(self):
        t = torch.rand(2)
        text = torch.randn(2, 5, 8)
        bundle = ConditioningBundle(timesteps=t, text_embeddings=text, null_text=None)
        null = bundle.get_null_bundle()
        assert null.text_embeddings.shape == text.shape
        assert torch.all(null.text_embeddings == 0)

    def test_from_batch_resolves_aliased_keys(self):
        t = torch.rand(2)
        batch = {
            "timesteps": t,
            "y": torch.tensor([1, 2]),  # alias for class_labels
            "text_emb": torch.randn(2, 4, 8),  # alias for text_embeddings
            "condition": torch.randn(2, 3, 8),  # alias for spatial_condition
            "seg": torch.randint(0, 3, (2, 8)),  # alias for segmentation_mask
            "scanner_params": torch.randn(2, 5),  # alias for metadata
        }
        bundle = ConditioningBundle.from_batch(batch)
        assert bundle.class_labels is not None
        assert bundle.text_embeddings is not None
        assert bundle.spatial_condition is not None
        assert bundle.segmentation_mask is not None
        assert bundle.metadata is not None

    def test_from_batch_uses_config_null_class(self):
        t = torch.rand(2)
        cfg = ConditioningConfig(use_class=True, num_classes=10)
        batch = {"timesteps": t, "class_labels": torch.tensor([1, 2])}
        bundle = ConditioningBundle.from_batch(batch, config=cfg)
        assert bundle.null_class == 10

    def test_from_batch_requires_timesteps(self):
        with pytest.raises(ValueError):
            ConditioningBundle.from_batch({"class_labels": torch.tensor([1, 2])})

    def test_from_batch_explicit_timesteps(self):
        t = torch.rand(3)
        bundle = ConditioningBundle.from_batch(
            {"class_labels": torch.tensor([0, 1, 2])}, timesteps=t
        )
        assert torch.equal(bundle.timesteps, t)

    def test_get_combined_embedding_add(self):
        t_emb = torch.randn(2, 8)
        y_emb = torch.randn(2, 8)
        bundle = ConditioningBundle(timesteps=torch.rand(2))
        out = bundle.get_combined_embedding(t_emb, y_emb=y_emb, fusion="add")
        assert out.shape == (2, 8)
        assert torch.allclose(out, t_emb + y_emb, atol=1e-6)

    def test_get_combined_embedding_concat(self):
        t_emb = torch.randn(2, 8)
        y_emb = torch.randn(2, 8)
        bundle = ConditioningBundle(timesteps=torch.rand(2))
        out = bundle.get_combined_embedding(t_emb, y_emb=y_emb, fusion="concat")
        assert out.shape == (2, 16)

    def test_get_combined_embedding_invalid_fusion(self):
        bundle = ConditioningBundle(timesteps=torch.rand(2))
        with pytest.raises(ValueError):
            bundle.get_combined_embedding(torch.randn(2, 8), fusion="bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------
class TestCollate:
    def test_collate_stacks_tokens_and_class(self):
        batch = [
            {"tokens": torch.arange(4), "class": torch.tensor(0)},
            {"tokens": torch.arange(4) + 10, "class": torch.tensor(1)},
        ]
        out = collate_conditional_batch(batch)
        assert out["tokens"].shape == (2, 4)
        assert out["class_labels"].shape == (2,)
        assert out["class_labels"].tolist() == [0, 1]

    def test_collate_variable_length_padding(self):
        batch = [
            {"tokens": torch.tensor([1, 2, 3])},
            {"tokens": torch.tensor([4, 5])},
            {"tokens": torch.tensor([6, 7, 8, 9])},
        ]
        cfg = ConditionalBatchConfig(pad_token_id=0)
        out = collate_conditional_batch(batch, cfg)
        assert out["tokens"].shape == (3, 4)
        # First row keeps its values, padded with 0 at the end.
        assert out["tokens"][0].tolist() == [1, 2, 3, 0]
        assert out["tokens"][1].tolist() == [4, 5, 0, 0]
        assert out["tokens"][2].tolist() == [6, 7, 8, 9]

    def test_collate_all_modalities(self):
        cfg = ConditionalBatchConfig(
            tokens_key="tokens",
            class_key="class",
            text_embeddings_key="text",
            text_pooled_key="pooled",
            text_mask_key="mask",
            spatial_key="spatial",
            spatial_mask_key="spatial_mask",
            segmentation_key="seg",
            metadata_key="meta",
            raw_image_key="img",
        )
        batch = [
            {
                "tokens": torch.arange(4),
                "class": torch.tensor(0),
                "text": torch.randn(5, 8),
                "pooled": torch.randn(8),
                "mask": torch.ones(5),
                "spatial": torch.randn(3, 8),
                "spatial_mask": torch.ones(8),
                "seg": torch.randint(0, 3, (8,)),
                "meta": torch.randn(6),
                "img": torch.randn(1, 16, 16),
            }
            for _ in range(2)
        ]
        out = collate_conditional_batch(batch, cfg)
        assert out["tokens"].shape == (2, 4)
        assert out["class_labels"].shape == (2,)
        assert out["text_embeddings"].shape == (2, 5, 8)
        assert out["text_pooled"].shape == (2, 8)
        assert out["text_mask"].shape == (2, 5)
        assert out["spatial_condition"].shape == (2, 3, 8)
        assert out["spatial_mask"].shape == (2, 8)
        assert out["segmentation_mask"].shape == (2, 8)
        assert out["metadata"].shape == (2, 6)
        assert out["raw_image"].shape == (2, 1, 16, 16)

    def test_collate_omits_absent_modalities(self):
        batch = [{"tokens": torch.arange(3)} for _ in range(2)]
        # class_key defaults to "class" but is absent -> should be omitted.
        out = collate_conditional_batch(batch)
        assert "tokens" in out
        assert "class_labels" not in out

    def test_collate_max_seq_length_truncation_equal_lengths(self):
        batch = [{"tokens": torch.arange(6)} for _ in range(2)]
        cfg = ConditionalBatchConfig(max_seq_length=4)
        out = collate_conditional_batch(batch, cfg)
        assert out["tokens"].shape == (2, 4)

    def test_stack_or_pad_equal_lengths(self):
        tensors = [torch.arange(3), torch.arange(3) + 5]
        out = _stack_or_pad(tensors)
        assert out.shape == (2, 3)

    def test_stack_or_pad_truncates_variable(self):
        tensors = [torch.tensor([1, 2, 3, 4, 5]), torch.tensor([6, 7])]
        out = _stack_or_pad(tensors, pad_value=0, max_length=3)
        assert out.shape == (2, 3)
        assert out[0].tolist() == [1, 2, 3]
        assert out[1].tolist() == [6, 7, 0]

    def test_create_conditional_collate_fn(self):
        cfg = ConditionalBatchConfig(tokens_key="input_ids", class_key="label")
        fn = create_conditional_collate_fn(cfg)
        batch = [
            {"input_ids": torch.arange(4), "label": torch.tensor(0)},
            {"input_ids": torch.arange(4), "label": torch.tensor(1)},
        ]
        out = fn(batch)
        assert out["tokens"].shape == (2, 4)
        assert out["class_labels"].tolist() == [0, 1]
