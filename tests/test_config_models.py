"""Tests for medlatents.config_models.pydantic.

Verifies that the pydantic configuration models reject invalid architecture
dimensions, accept valid ones, and that the NANO/SMALL/BASE presets stay in
sync with medlatents.configs.MODEL_CONFIGS (the single source of truth).

All pydantic-specific tests are skipped when pydantic is not installed; the
PYDANTIC_AVAILABLE flag path itself is always checked.
"""

from __future__ import annotations

import pytest

from medlatents.config_models import pydantic as cm
from medlatents.config_models.pydantic import (
    BASE_CONFIG,
    NANO_CONFIG,
    PYDANTIC_AVAILABLE,
    SMALL_CONFIG,
)
from medlatents.configs import MODEL_CONFIGS

pydantic_only = pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="pydantic not installed")


# Expected dims per the task spec / MODEL_CONFIGS.
EXPECTED = {
    "nano": (192, 8, 3),
    "small": (384, 12, 6),
    "base": (768, 16, 12),
}


class TestPydanticAvailabilityFlag:
    def test_flag_is_bool(self):
        assert isinstance(PYDANTIC_AVAILABLE, bool)

    def test_fallback_types_when_unavailable(self):
        if PYDANTIC_AVAILABLE:
            pytest.skip("pydantic available; fallback dict path not active")
        # When pydantic is missing, the config classes degrade to ``dict`` and
        # presets become plain dicts.
        assert cm.ModelConfig is dict
        assert isinstance(NANO_CONFIG, dict)
        assert NANO_CONFIG["model"]["hidden_size"] == MODEL_CONFIGS["nano"]["hidden_size"]


class TestModelConfigsSourceOfTruth:
    """These checks hold regardless of pydantic availability."""

    def test_model_configs_match_spec(self):
        for name, (hidden, depth, heads) in EXPECTED.items():
            spec = MODEL_CONFIGS[name]
            assert spec["hidden_size"] == hidden
            assert spec["depth"] == depth
            assert spec["num_heads"] == heads


@pydantic_only
class TestValidation:
    def test_valid_model_config(self):
        cfg = cm.ModelConfig(
            model_type="autoreg",
            seq_length=128,
            vocab_size=256,
            hidden_size=64,
            depth=4,
            num_heads=4,
        )
        assert cfg.hidden_size == 64
        assert cfg.depth == 4
        assert cfg.num_heads == 4

    @pytest.mark.parametrize("bad", [0, -1, -8])
    def test_rejects_nonpositive_hidden_size(self, bad):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            cm.ModelConfig(
                model_type="autoreg",
                seq_length=128,
                vocab_size=256,
                hidden_size=bad,
                depth=4,
                num_heads=4,
            )

    @pytest.mark.parametrize("bad", [0, -1])
    def test_rejects_nonpositive_depth(self, bad):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            cm.ModelConfig(
                model_type="autoreg",
                seq_length=128,
                vocab_size=256,
                hidden_size=64,
                depth=bad,
                num_heads=4,
            )

    @pytest.mark.parametrize("bad", [0, -1])
    def test_rejects_nonpositive_num_heads(self, bad):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            cm.ModelConfig(
                model_type="autoreg",
                seq_length=128,
                vocab_size=256,
                hidden_size=64,
                depth=4,
                num_heads=bad,
            )

    def test_rejects_nonpositive_seq_length(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            cm.ModelConfig(
                model_type="autoreg",
                seq_length=0,
                vocab_size=256,
                hidden_size=64,
                depth=4,
                num_heads=4,
            )

    def test_rejects_nonpositive_vocab_size(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            cm.ModelConfig(
                model_type="autoreg",
                seq_length=128,
                vocab_size=0,
                hidden_size=64,
                depth=4,
                num_heads=4,
            )

    def test_rejects_unknown_model_type(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            cm.ModelConfig(
                model_type="not-a-model",  # type: ignore[arg-type]
                seq_length=128,
                vocab_size=256,
                hidden_size=64,
                depth=4,
                num_heads=4,
            )

    def test_training_config_rejects_nonpositive_epochs(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            cm.TrainingConfig(epochs=0)

    def test_training_config_rejects_nonpositive_lr(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            cm.TrainingConfig(learning_rate=0.0)

    def test_training_config_defaults(self):
        cfg = cm.TrainingConfig()
        assert cfg.epochs == 100
        assert cfg.batch_size == 32
        assert cfg.learning_rate == pytest.approx(1e-4)

    def test_experiment_config_nests_defaults(self):
        cfg = cm.ExperimentConfig(
            model=cm.ModelConfig(
                model_type="flow",
                seq_length=64,
                vocab_size=128,
                hidden_size=32,
                depth=2,
                num_heads=2,
            )
        )
        assert isinstance(cfg.training, cm.TrainingConfig)
        assert isinstance(cfg.optimizer, cm.OptimizerConfig)
        assert isinstance(cfg.scheduler, cm.SchedulerConfig)
        assert cfg.seed == 42
        assert cfg.device == "auto"


@pydantic_only
class TestPresets:
    @pytest.mark.parametrize(
        "preset,name",
        [(NANO_CONFIG, "nano"), (SMALL_CONFIG, "small"), (BASE_CONFIG, "base")],
    )
    def test_preset_dims_match_model_configs(self, preset, name):
        hidden, depth, heads = EXPECTED[name]
        assert preset.model.hidden_size == hidden
        assert preset.model.depth == depth
        assert preset.model.num_heads == heads
        # And they really derive from MODEL_CONFIGS.
        assert preset.model.hidden_size == MODEL_CONFIGS[name]["hidden_size"]
        assert preset.model.depth == MODEL_CONFIGS[name]["depth"]
        assert preset.model.num_heads == MODEL_CONFIGS[name]["num_heads"]

    def test_preset_model_types(self):
        assert NANO_CONFIG.model.model_type == "autoreg"
        assert SMALL_CONFIG.model.model_type == "maskgit"
        assert BASE_CONFIG.model.model_type == "diffusion"

    def test_presets_are_experiment_configs(self):
        assert isinstance(NANO_CONFIG, cm.ExperimentConfig)
        assert isinstance(SMALL_CONFIG, cm.ExperimentConfig)
        assert isinstance(BASE_CONFIG, cm.ExperimentConfig)
