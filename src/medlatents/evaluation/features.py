"""Feature extractors for medical imaging evaluation metrics."""

from __future__ import annotations

import logging
from typing import Literal

import torch
import torch.nn as nn
import torchvision.models as models
from jaxtyping import Float

logger = logging.getLogger(__name__)


class FeatureExtractor(nn.Module):
    """Base class for feature extractors."""

    def forward(
        self, x: Float[torch.Tensor, "batch channel height width"]
    ) -> Float[torch.Tensor, "batch features"]:
        """Extract features from images."""
        raise NotImplementedError


class InceptionV3Features(FeatureExtractor):
    """
    InceptionV3 feature extractor (standard FID).

    Uses pool3 features (2048-dimensional) from ImageNet-pretrained InceptionV3.
    Note: Suboptimal for medical images but provided for compatibility.
    """

    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device

        # Load pretrained InceptionV3
        inception = models.inception_v3(pretrained=True, transform_input=False)
        inception.eval()

        # Extract feature layers (up to pool3)
        self.layers = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        ).to(device)

        for param in self.layers.parameters():
            param.requires_grad = False

    def forward(
        self, x: Float[torch.Tensor, "batch channel height width"]
    ) -> Float[torch.Tensor, "batch features"]:
        """
        Extract InceptionV3 features.

        Args:
            x: Input images in range [-1, 1], shape [B, C, H, W]
               Must have C=3 (RGB). For grayscale, repeat channels.

        Returns:
            Features of shape [B, 2048]
        """
        # Handle grayscale by repeating to RGB
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # Resize to 299x299 (InceptionV3 input size)
        if x.shape[-2:] != (299, 299):
            x = torch.nn.functional.interpolate(
                x, size=(299, 299), mode="bilinear", align_corners=False
            )

        with torch.no_grad():
            features = self.layers(x)

        return features.squeeze(-1).squeeze(-1)


class RadImageNetFeatures(FeatureExtractor):
    """
    RadImageNet feature extractor for medical imaging.

    RadImageNet is pretrained on medical images and provides better
    features for medical imaging evaluation than ImageNet models.

    Note: Requires radiology-pretrained model weights. If not available,
    falls back to InceptionV3 with a warning.
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str = "cuda",
        fallback: bool = True,
    ):
        """
        Initialize RadImageNet feature extractor.

        Args:
            model_path: Path to RadImageNet pretrained weights
            device: Device to run on
            fallback: Whether to fallback to InceptionV3 if RadImageNet unavailable
        """
        super().__init__()
        self.device = device

        # Try to load RadImageNet model
        try:
            if model_path is None:
                raise FileNotFoundError("RadImageNet weights not provided")

            # Load RadImageNet-pretrained ResNet50
            self.model = models.resnet50(pretrained=False)
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            self.model.load_state_dict(state_dict)

            # Remove final classification layer
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            self.model.eval().to(device)

            for param in self.model.parameters():
                param.requires_grad = False

            self.feature_dim = 2048
            self.using_fallback = False

        except (FileNotFoundError, RuntimeError) as e:
            if fallback:
                logger.warning(
                    f"Warning: Could not load RadImageNet ({e}). Falling back to InceptionV3."
                )
                self.model = InceptionV3Features(device=device)
                self.feature_dim = 2048
                self.using_fallback = True
            else:
                raise

    def forward(
        self, x: Float[torch.Tensor, "batch channel height width"]
    ) -> Float[torch.Tensor, "batch features"]:
        """
        Extract RadImageNet features.

        Args:
            x: Input images, shape [B, C, H, W]

        Returns:
            Features of shape [B, feature_dim]
        """
        if self.using_fallback:
            return self.model(x)

        # Handle grayscale by repeating to RGB
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # Resize to 224x224 (ResNet input size)
        if x.shape[-2:] != (224, 224):
            x = torch.nn.functional.interpolate(
                x, size=(224, 224), mode="bilinear", align_corners=False
            )

        with torch.no_grad():
            features = self.model(x)

        return features.squeeze(-1).squeeze(-1)


def get_feature_extractor(
    extractor_type: Literal["inception", "radimagenet"] = "radimagenet",
    device: str = "cuda",
    **kwargs,
) -> FeatureExtractor:
    """
    Get feature extractor for FID calculation.

    Args:
        extractor_type: Type of feature extractor
        device: Device to run on
        **kwargs: Additional arguments for specific extractors

    Returns:
        Initialized feature extractor

    Example:
        >>> extractor = get_feature_extractor('radimagenet', device='cuda')
        >>> features = extractor(images)
    """
    if extractor_type == "inception":
        return InceptionV3Features(device=device)
    elif extractor_type == "radimagenet":
        return RadImageNetFeatures(device=device, **kwargs)
    else:
        raise ValueError(f"Unknown extractor type: {extractor_type}")
