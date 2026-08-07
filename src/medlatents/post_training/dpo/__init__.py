"""Direct Preference Optimization (DPO) for generative models.

This module implements DPO and its variants for aligning generative models
to human/AI preferences without explicit reward modeling.

Implementations:
- BaseDPOTrainer: Core DPO infrastructure with FSDP support
- DPOLoss: Flexible loss with sigmoid, hinge, IPO, KTO variants
- DiscreteDPOTrainer: DPO for discrete token models (MaskGIT, autoreg)
- DiffusionDPOTrainer: DPO for diffusion models (D3PM)
- FlowDPOTrainer: DPO for flow matching models
- SPOTrainer: Step-by-step Preference Optimization

References:
- DPO: "Direct Preference Optimization" (Rafailov et al., 2023)
- IPO: "A General Theoretical Paradigm" (Azar et al., 2023)
- KTO: "Kahneman-Tversky Optimization" (Ethayarajh et al., 2024)
- Diffusion-DPO: "Diffusion Model Alignment Using DPO" (Wallace et al., 2023)
- SPO: "Step-by-step Preference Optimization" (Liang et al., 2024, CVPR 2025)
"""

from .base import BaseDPOTrainer, DPOLoss, DPOOutput
from .diffusion import D3PMDPOTrainer, DiffusionDPOTrainer
from .discrete import (
    AutoregressiveDPOTrainer,
    DiscreteDPOTrainer,
    MaskGITDPOTrainer,
)
from .flow import DiscreteFlowDPOTrainer, FlowDPOTrainer
from .spo import SPOStepOutput, SPOTrainer, StepPreferenceModel

__all__ = [
    # Base
    "BaseDPOTrainer",
    "DPOLoss",
    "DPOOutput",
    # Discrete (MaskGIT, Autoregressive)
    "AutoregressiveDPOTrainer",
    "MaskGITDPOTrainer",
    "DiscreteDPOTrainer",
    # Diffusion (D3PM)
    "D3PMDPOTrainer",
    "DiffusionDPOTrainer",
    # Flow Matching
    "DiscreteFlowDPOTrainer",
    "FlowDPOTrainer",
    # SPO
    "SPOTrainer",
    "StepPreferenceModel",
    "SPOStepOutput",
]
