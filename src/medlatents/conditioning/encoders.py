"""Frozen pretrained encoders for REPA and representation alignment.

Provides wrappers for visual encoders used as alignment targets during training:
- DINOv2: General vision encoder (facebookresearch/dinov2)
- SigLIP: Vision-language aligned encoder (google/siglip)
- Medical models: NeuroVFM, MedSigLIP for domain-specific alignment

These encoders extract semantic features from clean images that serve as
alignment targets for diffusion/flow model hidden states during REPA training.

Example:
    >>> encoder = FrozenDINOv2(device="cuda")
    >>> features = encoder(images)  # [B, N, D]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, cast

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrozenEncoder(ABC, nn.Module):
    """Base class for frozen pretrained encoders.

    All parameters are frozen and the encoder runs in eval mode.
    Subclasses implement specific encoder architectures.

    Attributes:
        embed_dim: Output embedding dimension
        patch_size: Patch size for vision transformers
        supports_3d: Whether encoder natively supports 3D inputs
    """

    def __init__(self) -> None:
        super().__init__()
        self._embed_dim: int = 0
        self._patch_size: int = 16
        self._supports_3d: bool = False

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def patch_size(self) -> int:
        return self._patch_size

    @property
    def supports_3d(self) -> bool:
        return self._supports_3d

    def freeze(self) -> None:
        """Freeze all parameters and set to eval mode."""
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    @abstractmethod
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patch-level features from images.

        Args:
            x: Input images [batch, C, H, W] or [batch, C, D, H, W]
               Normalized to appropriate range for the encoder.

        Returns:
            Patch features [batch, num_patches, embed_dim]
        """
        raise NotImplementedError

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess images for this encoder.

        Override in subclasses for encoder-specific preprocessing.

        Args:
            x: Input images in [0, 1] or [-1, 1] range

        Returns:
            Preprocessed images ready for forward_features
        """
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass with preprocessing."""
        x = self.preprocess(x)
        return self.forward_features(x)


class FrozenDINOv2(FrozenEncoder):
    """Frozen DINOv2 encoder for REPA alignment.

    DINOv2 provides strong semantic features that transfer well across domains.
    Supports: dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14

    Reference:
        Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision"
        https://arxiv.org/abs/2304.07193

    Args:
        model_name: DINOv2 variant (vits14, vitb14, vitl14, vitg14)
        device: Device to load model on
        trust_remote_code: Whether to trust remote code for custom implementations
    """

    # ImageNet normalization used by DINOv2
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        device: torch.device | str = "cpu",
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__()

        device_obj = torch.device(device)

        # Map short names to full names
        name_map = {
            "vits14": "dinov2_vits14",
            "vitb14": "dinov2_vitb14",
            "vitl14": "dinov2_vitl14",
            "vitg14": "dinov2_vitg14",
            "dinov2-vit-s": "dinov2_vits14",
            "dinov2-vit-b": "dinov2_vitb14",
            "dinov2-vit-l": "dinov2_vitl14",
            "dinov2-vit-g": "dinov2_vitg14",
        }
        full_name = name_map.get(model_name, model_name)

        # Try loading from torch hub first (official implementation)
        try:
            encoder_obj = torch.hub.load(
                "facebookresearch/dinov2",
                full_name,
                pretrained=True,
            )
            self.encoder = cast(nn.Module, encoder_obj)
        except Exception:
            # Fallback to HuggingFace transformers
            try:
                from transformers import ViTModel

                hf_name = f"facebook/dinov2-{full_name.replace('dinov2_', '')}"
                self.encoder = ViTModel.from_pretrained(
                    hf_name, trust_remote_code=trust_remote_code
                )

                # Use custom forward for HF models - remove CLS token
                def forward_features(x: torch.Tensor) -> torch.Tensor:
                    return self.encoder(x, output_hidden_states=True).last_hidden_state[:, 1:]  # type: ignore[attr-defined]

                self.encoder.forward_features = forward_features  # type: ignore[assignment]
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load DINOv2 model '{full_name}'. "
                    f"Tried torch.hub and transformers. Ensure you have internet access. "
                    f"Error: {e}"
                ) from e

        self.encoder = self.encoder.to(device_obj)  # type: ignore[arg-type]
        self._embed_dim = int(getattr(self.encoder, "embed_dim", 768))  # type: ignore[arg-type]
        self._patch_size = 14  # DINOv2 uses 14x14 patches
        self._supports_3d = False

        # Register normalization buffers
        self.register_buffer("mean", torch.tensor(self.MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(self.STD).view(1, 3, 1, 1))

        self.freeze()

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess images for DINOv2.

        Args:
            x: Images in [0, 1] range, shape [batch, C, H, W]
               If C=1, will be replicated to 3 channels.

        Returns:
            Normalized images ready for DINOv2
        """
        # Handle single-channel (grayscale/medical) images
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3:
            raise ValueError(f"Expected 1 or 3 channels, got {x.shape[1]}")

        # Resize to DINOv2-compatible size (multiple of 14)
        h, w = x.shape[-2:]
        target_h = (h // 14) * 14
        target_w = (w // 14) * 14
        if target_h != h or target_w != w:
            x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)

        # Normalize
        mean_buf = self.get_buffer("mean")
        std_buf = self.get_buffer("std")
        x = (x - mean_buf.to(x.device)) / std_buf.to(x.device)

        return x

    @torch.no_grad()
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patch features from DINOv2.

        Args:
            x: Preprocessed images [batch, 3, H, W]

        Returns:
            Patch tokens [batch, num_patches, embed_dim]
        """
        features = self.encoder.forward_features(x)  # type: ignore[attr-defined,operator]
        # DINOv2 returns dict with 'x_norm_patchtokens'
        if isinstance(features, dict):
            return features["x_norm_patchtokens"]  # type: ignore[return-value]
        # Older versions return tuple (cls, patches)
        if isinstance(features, tuple):
            return features[0][:, 1:]  # type: ignore[index,return-value]
        # Transformers ViT returns tensor with CLS at position 0
        return features[:, 1:]  # type: ignore[return-value]


class FrozenSigLIP(FrozenEncoder):
    """Frozen SigLIP encoder for REPA alignment.

    SigLIP provides vision-language aligned features, useful when
    text conditioning is also used.

    Reference:
        Zhai et al., "Sigmoid Loss for Language Image Pre-Training"
        https://arxiv.org/abs/2303.15343

    Args:
        model_name: SigLIP variant (e.g., 'siglip-base-patch16-224')
        device: Device to load model on
        trust_remote_code: Whether to trust remote code for custom models
    """

    # CLIP/SigLIP normalization
    MEAN = (0.48145466, 0.4578275, 0.40821073)
    STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        device: torch.device | str = "cpu",
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__()

        device_obj = torch.device(device)

        try:
            from transformers import SiglipVisionModel

            encoder = SiglipVisionModel.from_pretrained(
                model_name, trust_remote_code=trust_remote_code
            )
        except ImportError as e:
            raise ImportError(
                "SigLIP requires the 'transformers' library. Install with: pip install transformers"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load SigLIP model '{model_name}': {e}") from e

        self.encoder = encoder.to(device_obj)
        self._embed_dim = self.encoder.config.hidden_size
        self._patch_size = self.encoder.config.patch_size
        self._supports_3d = False

        self.register_buffer("mean", torch.tensor(self.MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(self.STD).view(1, 3, 1, 1))

        self.freeze()

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess images for SigLIP.

        Args:
            x: Images in [0, 1] range, shape [batch, C, H, W]

        Returns:
            Normalized images ready for SigLIP
        """
        # Handle single-channel images
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # Resize to expected size
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        mean_buf = self.get_buffer("mean")
        std_buf = self.get_buffer("std")
        x = (x - mean_buf.to(x.device)) / std_buf.to(x.device)

        return x

    @torch.no_grad()
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patch features from SigLIP.

        Args:
            x: Preprocessed images [batch, 3, 224, 224]

        Returns:
            Patch tokens [batch, num_patches, embed_dim]
        """
        outputs = self.encoder(pixel_values=x, output_hidden_states=False)
        hidden_states = outputs.last_hidden_state
        # SigLIP doesn't have CLS token, return all tokens
        return hidden_states


class FrozenNeuroVFM(FrozenEncoder):
    """Frozen NeuroVFM encoder for medical imaging REPA alignment.

    NeuroVFM is a neuroimaging foundation model pretrained on large-scale
    brain MRI datasets, providing domain-specific semantic features.

    Reference:
        https://huggingface.co/microsoft/NeuroVFM

    Args:
        model_name: NeuroVFM variant
        device: Device to load model on
        trust_remote_code: Whether to trust remote code
    """

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        model_name: str = "microsoft/NeuroVFM-base",
        device: torch.device | str = "cpu",
        trust_remote_code: bool = True,
    ) -> None:
        super().__init__()

        device_obj = torch.device(device)

        try:
            from transformers import AutoModel

            encoder = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
            )
        except ImportError as e:
            raise ImportError(
                "NeuroVFM requires the 'transformers' library. Install with: pip install transformers"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load NeuroVFM model '{model_name}': {e}") from e

        self.encoder = encoder.to(device_obj)  # type: ignore[arg-type]
        self._embed_dim = self.encoder.config.hidden_size
        self._patch_size = self.encoder.config.patch_size
        self._supports_3d = False

        self.register_buffer("mean", torch.tensor(self.MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(self.STD).view(1, 3, 1, 1))

        self.freeze()

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess images for NeuroVFM.

        Args:
            x: Images in [0, 1] range, shape [batch, C, H, W]

        Returns:
            Normalized images ready for NeuroVFM
        """
        # Handle single-channel medical images
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] == 3:
            pass
        else:
            raise ValueError(f"Expected 1 or 3 channels, got {x.shape[1]}")

        # Resize to multiple of patch_size
        h, w = x.shape[-2:]
        target_h = (h // self.patch_size) * self.patch_size
        target_w = (w // self.patch_size) * self.patch_size
        if target_h != h or target_w != w:
            x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)

        mean_buf = self.get_buffer("mean")
        std_buf = self.get_buffer("std")
        x = (x - mean_buf.to(x.device)) / std_buf.to(x.device)

        return x

    @torch.no_grad()
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patch features from NeuroVFM.

        Args:
            x: Preprocessed images [batch, 3, H, W]

        Returns:
            Patch tokens [batch, num_patches, embed_dim]
        """
        outputs = self.encoder(x, output_hidden_states=False)
        hidden_states = outputs.last_hidden_state
        # Remove CLS token if present
        return hidden_states[:, 1:]


class FrozenMedSigLIP(FrozenEncoder):
    """Frozen MedSigLIP encoder for medical imaging REPA alignment.

    MedSigLIP is a medical vision-language model based on SigLIP,
    pretrained on medical image-text pairs.

    Reference:
        https://huggingface.co/models?search=medsiglip

    Args:
        model_name: MedSigLIP variant
        device: Device to load model on
        trust_remote_code: Whether to trust remote code
    """

    MEAN = (0.48145466, 0.4578275, 0.40821073)
    STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(
        self,
        model_name: str = "Microsoft/med-siglip-base-patch16-256",
        device: torch.device | str = "cpu",
        trust_remote_code: bool = True,
    ) -> None:
        super().__init__()

        device_obj = torch.device(device)

        try:
            from transformers import AutoModel

            encoder = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
            )
        except ImportError as e:
            raise ImportError(
                "MedSigLIP requires the 'transformers' library. Install with: pip install transformers"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load MedSigLIP model '{model_name}': {e}") from e

        self.encoder = encoder.to(device_obj)
        self._embed_dim = self.encoder.config.hidden_size
        self._patch_size = self.encoder.config.patch_size
        self._supports_3d = False

        self.register_buffer("mean", torch.tensor(self.MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(self.STD).view(1, 3, 1, 1))

        self.freeze()

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess images for MedSigLIP.

        Args:
            x: Images in [0, 1] range, shape [batch, C, H, W]

        Returns:
            Normalized images ready for MedSigLIP
        """
        # Handle single-channel medical images
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] == 3:
            pass
        else:
            raise ValueError(f"Expected 1 or 3 channels, got {x.shape[1]}")

        # Resize to expected size (256 for most medical models)
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)

        mean_buf = self.get_buffer("mean")
        std_buf = self.get_buffer("std")
        x = (x - mean_buf.to(x.device)) / std_buf.to(x.device)

        return x

    @torch.no_grad()
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patch features from MedSigLIP.

        Args:
            x: Preprocessed images [batch, 3, 256, 256]

        Returns:
            Patch tokens [batch, num_patches, embed_dim]
        """
        outputs = self.encoder(x, output_hidden_states=False)
        hidden_states = outputs.last_hidden_state
        # MedSigLIP typically doesn't have CLS token
        return hidden_states


class FrozenMedicalEncoder(FrozenEncoder):
    """Base class for frozen medical imaging encoders.

    Provides common functionality for medical domain-specific encoders
    like NeuroVFM, MedSigLIP, or custom pretrained models.

    Subclasses should implement _load_encoder() and forward_features().
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.model_path = model_path
        self._device = torch.device(device)

    def adapt_2d_to_3d(
        self,
        x: torch.Tensor,
        mode: Literal["slice_wise", "projection", "central_slices"] = "slice_wise",
    ) -> torch.Tensor:
        """Adapt 3D volume for 2D encoder.

        Args:
            x: 3D volume [batch, C, D, H, W]
            mode: Adaptation strategy
                - 'slice_wise': Process each slice independently
                - 'projection': Max/mean projection along depth
                - 'central_slices': Use central N slices as RGB

        Returns:
            2D images [batch*D, C, H, W] or [batch, 3, H, W]
        """
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input for 3D adaptation, got {x.dim()}D")

        batch, c, d, h, w = x.shape

        if mode == "slice_wise":
            # Reshape to [batch*D, C, H, W]
            x = x.permute(0, 2, 1, 3, 4).reshape(batch * d, c, h, w)

        elif mode == "projection":
            # Max projection along depth
            x = x.max(dim=2)[0]  # [batch, C, H, W]

        elif mode == "central_slices":
            # Take central 3 slices as RGB channels
            center = d // 2
            if c == 1:
                slices = x[:, 0, center - 1 : center + 2, :, :]  # [batch, 3, H, W]
                x = slices
            else:
                x = x[:, :, center, :, :]  # Just take center slice

        return x


class SliceWiseEncoder(nn.Module):
    """Wrapper that applies 2D encoder slice-wise to 3D volumes.

    Enables using 2D pretrained encoders (DINOv2, SigLIP) on 3D medical images
    by processing each slice independently.

    Args:
        encoder: 2D frozen encoder
        aggregation: How to aggregate slice features ('none', 'mean', 'attention')
    """

    def __init__(
        self,
        encoder: FrozenEncoder,
        aggregation: Literal["none", "mean", "attention"] = "none",
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.aggregation = aggregation

        if aggregation == "attention":
            # Learnable attention over slices
            self.slice_attention = nn.Sequential(
                nn.Linear(encoder.embed_dim, 1),
                nn.Softmax(dim=1),
            )

    @property
    def embed_dim(self) -> int:
        return self.encoder.embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process 3D volume slice-wise.

        Args:
            x: 3D volume [batch, C, D, H, W] or 2D image [batch, C, H, W]

        Returns:
            Features [batch, D*num_patches, embed_dim] or [batch, num_patches, embed_dim]
        """
        if x.dim() == 4:
            # 2D input, process directly
            return self.encoder(x)

        batch, c, d, h, w = x.shape

        # Reshape to [batch*D, C, H, W]
        x = x.permute(0, 2, 1, 3, 4).reshape(batch * d, c, h, w)

        # Process through encoder
        features = self.encoder(x)  # [batch*D, num_patches, embed_dim]
        num_patches = features.shape[1]

        # Reshape back
        features = features.view(batch, d, num_patches, -1)

        if self.aggregation == "none":
            # Flatten D and patches dimensions
            return features.view(batch, d * num_patches, -1)

        elif self.aggregation == "mean":
            # Average over slices
            return features.mean(dim=1)

        elif self.aggregation == "attention":
            # Weighted average with learned attention
            # Attention over slice dimension
            weights = self.slice_attention(features.mean(dim=2))  # [batch, D, 1]
            return (features * weights.unsqueeze(-1)).sum(dim=1)

        return features.view(batch, d * num_patches, -1)


def create_encoder(
    encoder_type: str,
    device: torch.device | str = "cpu",
    **kwargs,
) -> FrozenEncoder:
    """Factory function to create frozen encoders.

    Args:
        encoder_type: Encoder type identifier
            - 'dinov2_vitb14', 'dinov2-vit-b', etc.
            - 'siglip-base-patch16-224'
            - 'neurovfm-base' or full model path
            - 'medsiglip-base' or full model path
        device: Device to load model on
        **kwargs: Additional arguments for specific encoders (e.g., trust_remote_code)

    Returns:
        Frozen encoder instance

    Example:
        >>> encoder = create_encoder("dinov2_vitb14", device="cuda")
        >>> encoder = create_encoder("google/siglip-base-patch16-224")
        >>> encoder = create_encoder("microsoft/NeuroVFM-base")
    """
    encoder_type_lower = encoder_type.lower()

    if "dinov2" in encoder_type_lower or "dino" in encoder_type_lower:
        return FrozenDINOv2(model_name=encoder_type, device=device, **kwargs)

    elif "siglip" in encoder_type_lower and "med" not in encoder_type_lower:
        return FrozenSigLIP(model_name=encoder_type, device=device, **kwargs)

    elif "neurovfm" in encoder_type_lower:
        return FrozenNeuroVFM(model_name=encoder_type, device=device, **kwargs)

    elif "medsiglip" in encoder_type_lower:
        return FrozenMedSigLIP(model_name=encoder_type, device=device, **kwargs)

    else:
        raise ValueError(
            f"Unknown encoder type: {encoder_type}. "
            f"Supported: dinov2_vitb14, dinov2-vit-b, siglip-base-patch16-224, "
            f"neurovfm-base, medsiglip-base (or full HuggingFace paths)"
        )


__all__ = [
    "FrozenEncoder",
    "FrozenDINOv2",
    "FrozenSigLIP",
    "FrozenNeuroVFM",
    "FrozenMedSigLIP",
    "FrozenMedicalEncoder",
    "SliceWiseEncoder",
    "create_encoder",
]
