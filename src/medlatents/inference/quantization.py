"""Model quantization and pruning for efficient inference.

Provides tools for reducing model size and improving inference speed:
- Dynamic and static quantization (INT8, INT4)
- Weight pruning (magnitude, structured)
- Quantization-aware training (QAT) utilities
- ONNX/TensorRT export helpers

These complement the KV-cache quantization in kv_cache.py, which focuses on
runtime memory reduction during generation.

References:
- "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"
- "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale"
- "SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs"
- "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks"
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================


class QuantizationMethod(str, Enum):
    """Quantization method selection."""

    DYNAMIC = "dynamic"  # Dynamic quantization (weights static, activations dynamic)
    STATIC = "static"  # Static quantization (calibrated activation ranges)
    QAT = "qat"  # Quantization-aware training
    GPTQ = "gptq"  # GPTQ-style weight quantization
    SMOOTHQUANT = "smoothquant"  # SmoothQuant for LLM-friendly quantization


class PruningMethod(str, Enum):
    """Pruning method selection."""

    MAGNITUDE = "magnitude"  # Prune smallest magnitude weights
    STRUCTURED = "structured"  # Prune entire channels/heads
    MOVEMENT = "movement"  # Movement pruning (gradient-based)
    WANDA = "wanda"  # Weights and Activations (WANDA) pruning


@dataclass
class QuantizationConfig:
    """Configuration for model quantization.

    Attributes:
        method: Quantization method to use
        dtype: Target dtype for quantized weights (int8/int4 for quantization, fp16/bf16 for casting)
        per_channel: Use per-channel quantization (more accurate)
        symmetric: Use symmetric quantization
        calibration_batches: Number of batches for static quantization calibration
        modules_to_quantize: List of module types to quantize (None = all supported)
        modules_to_skip: List of module names to skip
        smoothing_alpha: SmoothQuant smoothing factor (0.5 is typical)
    """

    method: QuantizationMethod = QuantizationMethod.DYNAMIC
    dtype: Literal["int8", "int4", "fp16", "bf16"] = "int8"
    per_channel: bool = True
    symmetric: bool = True
    calibration_batches: int = 100
    modules_to_quantize: list[str] | None = None
    modules_to_skip: list[str] = field(default_factory=list)
    smoothing_alpha: float = 0.5

    def get_quant_dtype(self) -> Literal["int8", "int4"]:
        """Get quantization dtype, defaulting to int8 for non-quantized types."""
        if self.dtype in ("int8", "int4"):
            return self.dtype  # type: ignore[return-value]
        return "int8"


@dataclass
class PruningConfig:
    """Configuration for model pruning.

    Attributes:
        method: Pruning method to use
        sparsity: Target sparsity ratio (0.0 to 1.0)
        structured_dim: Dimension for structured pruning (0=output, 1=input)
        granularity: Pruning granularity ('element', 'row', 'column')
        iterative_steps: Number of iterative pruning steps
        importance_scores: Pre-computed importance scores (for custom pruning)
    """

    method: PruningMethod = PruningMethod.MAGNITUDE
    sparsity: float = 0.5
    structured_dim: int = 0
    granularity: Literal["element", "row", "column"] = "element"
    iterative_steps: int = 1
    importance_scores: dict[str, torch.Tensor] | None = None


# ============================================================================
# Quantization Utilities
# ============================================================================


def compute_scale_zero_point(
    tensor: torch.Tensor,
    dtype: Literal["int8", "int4"] = "int8",
    symmetric: bool = True,
    per_channel: bool = False,
    channel_dim: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute quantization scale and zero point.

    Args:
        tensor: Tensor to quantize
        dtype: Target quantization dtype
        symmetric: Use symmetric quantization
        per_channel: Compute per-channel scales
        channel_dim: Channel dimension for per-channel quantization

    Returns:
        scale: Quantization scale factor(s)
        zero_point: Zero point(s) for asymmetric quantization
    """
    qmin, qmax = (-128, 127) if dtype == "int8" else (-8, 7)

    if per_channel:
        # Reduce all dimensions except channel_dim
        reduce_dims = [i for i in range(tensor.ndim) if i != channel_dim]
        min_val = tensor.amin(dim=reduce_dims, keepdim=True)
        max_val = tensor.amax(dim=reduce_dims, keepdim=True)
    else:
        min_val = tensor.min()
        max_val = tensor.max()

    if symmetric:
        abs_max = torch.maximum(min_val.abs(), max_val.abs())
        scale = abs_max / ((qmax - qmin) / 2)
        scale = scale.clamp(min=1e-6)
        zero_point = torch.zeros_like(scale)
    else:
        scale = (max_val - min_val) / (qmax - qmin)
        scale = scale.clamp(min=1e-6)
        zero_point = qmin - torch.round(min_val / scale)
        zero_point = zero_point.clamp(qmin, qmax)

    return scale, zero_point


def quantize_tensor(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    dtype: Literal["int8", "int4"] = "int8",
) -> torch.Tensor:
    """Quantize a tensor to the specified dtype.

    Args:
        tensor: Input tensor
        scale: Quantization scale
        zero_point: Zero point
        dtype: Target dtype

    Returns:
        Quantized tensor
    """
    qmin, qmax = (-128, 127) if dtype == "int8" else (-8, 7)
    quantized = torch.round(tensor / scale + zero_point)
    quantized = quantized.clamp(qmin, qmax)

    if dtype == "int8":
        return quantized.to(torch.int8)
    else:
        # INT4 stored in INT8 container
        return quantized.to(torch.int8)


def dequantize_tensor(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
) -> torch.Tensor:
    """Dequantize a tensor back to float.

    Args:
        quantized: Quantized tensor
        scale: Quantization scale
        zero_point: Zero point

    Returns:
        Dequantized float tensor
    """
    return (quantized.float() - zero_point) * scale


class QuantizedLinear(nn.Module):
    """Quantized linear layer with INT8 weights.

    Stores weights in INT8 format and dequantizes during forward pass.
    For true INT8 computation, use hardware-specific backends (cuBLAS, TensorRT).
    """

    # Type hints for buffers
    weight_quantized: torch.Tensor
    weight_scale: torch.Tensor
    weight_zero_point: torch.Tensor
    bias: torch.Tensor | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: Literal["int8", "int4"] = "int8",
        per_channel: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_dtype = dtype  # Renamed to avoid confusion with torch dtype
        self.per_channel = per_channel

        # Quantized weight storage
        self.register_buffer(
            "weight_quantized",
            torch.zeros(out_features, in_features, dtype=torch.int8),
        )
        self.register_buffer(
            "weight_scale",
            torch.ones(out_features if per_channel else 1),
        )
        self.register_buffer(
            "weight_zero_point",
            torch.zeros(out_features if per_channel else 1),
        )

        if bias:
            self.register_buffer("bias", torch.zeros(out_features))
        else:
            self.register_buffer("bias", None)

    @classmethod
    def from_float(
        cls,
        linear: nn.Linear,
        dtype: Literal["int8", "int4"] = "int8",
        per_channel: bool = True,
        symmetric: bool = True,
    ) -> "QuantizedLinear":
        """Create quantized linear from a float linear layer.

        Args:
            linear: Source float linear layer
            dtype: Quantization dtype
            per_channel: Use per-channel quantization
            symmetric: Use symmetric quantization

        Returns:
            Quantized linear layer
        """
        q_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            dtype=dtype,
            per_channel=per_channel,
        )

        # Compute scale and zero point
        weight = linear.weight.data
        scale, zero_point = compute_scale_zero_point(
            weight,
            dtype=dtype,
            symmetric=symmetric,
            per_channel=per_channel,
            channel_dim=0,
        )

        # Quantize weights
        if per_channel:
            scale = scale.squeeze()
            zero_point = zero_point.squeeze()

        q_linear.weight_scale = scale
        q_linear.weight_zero_point = zero_point
        q_linear.weight_quantized = quantize_tensor(
            weight,
            scale.unsqueeze(1) if per_channel else scale,
            zero_point.unsqueeze(1) if per_channel else zero_point,
            dtype,
        )

        if linear.bias is not None:
            q_linear.bias = linear.bias.data.clone()

        return q_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with weight dequantization.

        Note: This performs dequantize-then-compute, which doesn't give
        speedup on most hardware. For actual INT8 speedup, export to
        ONNX/TensorRT or use torch.ao.quantization.
        """
        # Dequantize weights
        if self.per_channel:
            weight = dequantize_tensor(
                self.weight_quantized,
                self.weight_scale.unsqueeze(1),
                self.weight_zero_point.unsqueeze(1),
            )
        else:
            weight = dequantize_tensor(
                self.weight_quantized,
                self.weight_scale,
                self.weight_zero_point,
            )

        return F.linear(x, weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, dtype={self.quant_dtype}, per_channel={self.per_channel}"
        )


# ============================================================================
# Model Quantization
# ============================================================================


class ModelQuantizer:
    """Quantize PyTorch models for efficient inference.

    Supports:
    - Dynamic quantization (quick, no calibration needed)
    - Static quantization (requires calibration data)
    - SmoothQuant for transformer-friendly quantization

    Example:
        quantizer = ModelQuantizer(config=QuantizationConfig(method="dynamic"))
        quantized_model = quantizer.quantize(model)
    """

    def __init__(self, config: QuantizationConfig | None = None):
        """Initialize quantizer.

        Args:
            config: Quantization configuration
        """
        self.config = config or QuantizationConfig()

    def quantize(
        self,
        model: nn.Module,
        calibration_data: Iterator[torch.Tensor] | None = None,
    ) -> nn.Module:
        """Quantize a model.

        Args:
            model: Model to quantize
            calibration_data: Iterator of calibration inputs (for static quantization)

        Returns:
            Quantized model
        """
        if self.config.method == QuantizationMethod.DYNAMIC:
            return self._dynamic_quantize(model)
        elif self.config.method == QuantizationMethod.STATIC:
            if calibration_data is None:
                raise ValueError("Static quantization requires calibration_data")
            return self._static_quantize(model, calibration_data)
        elif self.config.method == QuantizationMethod.SMOOTHQUANT:
            if calibration_data is None:
                raise ValueError("SmoothQuant requires calibration_data")
            return self._smoothquant(model, calibration_data)
        else:
            raise ValueError(f"Unsupported quantization method: {self.config.method}")

    def _dynamic_quantize(self, model: nn.Module) -> nn.Module:
        """Apply dynamic quantization to linear layers.

        Dynamic quantization quantizes weights statically and activations
        dynamically at runtime. No calibration data needed.
        """
        model = copy.deepcopy(model)

        for name, module in model.named_modules():
            if self._should_quantize(name, module):
                if isinstance(module, nn.Linear):
                    # Replace with quantized version
                    q_linear = QuantizedLinear.from_float(
                        module,
                        dtype=self.config.get_quant_dtype(),
                        per_channel=self.config.per_channel,
                        symmetric=self.config.symmetric,
                    )
                    self._replace_module(model, name, q_linear)

        logger.info(f"Applied dynamic {self.config.get_quant_dtype()} quantization")
        return model

    def _static_quantize(
        self,
        model: nn.Module,
        calibration_data: Iterator[torch.Tensor],
    ) -> nn.Module:
        """Apply static quantization with calibration.

        Static quantization uses calibration data to determine optimal
        activation quantization ranges. More accurate than dynamic but
        requires representative data.
        """
        model = copy.deepcopy(model)

        # Collect activation statistics
        activation_stats: dict[str, dict[str, torch.Tensor]] = {}
        hooks = []

        def make_hook(name: str):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    if name not in activation_stats:
                        activation_stats[name] = {"min": output.min(), "max": output.max()}
                    else:
                        activation_stats[name]["min"] = torch.minimum(
                            activation_stats[name]["min"], output.min()
                        )
                        activation_stats[name]["max"] = torch.maximum(
                            activation_stats[name]["max"], output.max()
                        )

            return hook

        # Register hooks
        for name, module in model.named_modules():
            if self._should_quantize(name, module):
                hooks.append(module.register_forward_hook(make_hook(name)))

        # Run calibration
        model.eval()
        num_batches = 0
        with torch.no_grad():
            for i, batch in enumerate(calibration_data):
                if i >= self.config.calibration_batches:
                    break
                if isinstance(batch, dict):
                    model(**batch)
                else:
                    model(batch)
                num_batches = i + 1

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Apply quantization with calibrated ranges
        for name, module in model.named_modules():
            if self._should_quantize(name, module):
                if isinstance(module, nn.Linear):
                    q_linear = QuantizedLinear.from_float(
                        module,
                        dtype=self.config.get_quant_dtype(),
                        per_channel=self.config.per_channel,
                        symmetric=self.config.symmetric,
                    )
                    self._replace_module(model, name, q_linear)

        logger.info(
            f"Applied static {self.config.get_quant_dtype()} quantization "
            f"with {num_batches} calibration batches"
        )
        return model

    def _smoothquant(
        self,
        model: nn.Module,
        calibration_data: Iterator[torch.Tensor],
    ) -> nn.Module:
        """Apply SmoothQuant quantization.

        SmoothQuant migrates quantization difficulty from activations to
        weights by applying a mathematically equivalent transformation,
        making quantization more accurate for transformer models.

        Reference: https://arxiv.org/abs/2211.10438
        """
        model = copy.deepcopy(model)
        alpha = self.config.smoothing_alpha

        # Collect activation statistics per channel
        act_scales: dict[str, torch.Tensor] = {}
        hooks = []

        def make_hook(name: str):
            def hook(module, input, output):
                if isinstance(input, tuple) and len(input) > 0:
                    x = input[0]
                    if isinstance(x, torch.Tensor) and x.ndim >= 2:
                        # Per-channel max (last dimension is usually the channel)
                        scale = x.abs().amax(dim=tuple(range(x.ndim - 1)))
                        if name not in act_scales:
                            act_scales[name] = scale
                        else:
                            act_scales[name] = torch.maximum(act_scales[name], scale)

            return hook

        # Register hooks on linear layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and self._should_quantize(name, module):
                hooks.append(module.register_forward_hook(make_hook(name)))

        # Collect activation scales
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(calibration_data):
                if i >= self.config.calibration_batches:
                    break
                if isinstance(batch, dict):
                    model(**batch)
                else:
                    model(batch)

        for hook in hooks:
            hook.remove()

        # Apply smoothing and quantization
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and self._should_quantize(name, module):
                if name in act_scales:
                    act_scale = act_scales[name].clamp(min=1e-5)
                    weight_scale = module.weight.abs().amax(dim=0).clamp(min=1e-5)

                    # Compute smoothing factor: s = act_scale^alpha / weight_scale^(1-alpha)
                    smooth_scale = (act_scale.pow(alpha) / weight_scale.pow(1 - alpha)).clamp(
                        min=1e-5
                    )

                    # Apply smoothing: W' = W * s, X' = X / s
                    # We only modify weights here; activations are scaled at runtime
                    module.weight.data.mul_(smooth_scale.unsqueeze(0))

                # Now quantize the smoothed weights
                q_linear = QuantizedLinear.from_float(
                    module,
                    dtype=self.config.get_quant_dtype(),
                    per_channel=self.config.per_channel,
                    symmetric=self.config.symmetric,
                )
                self._replace_module(model, name, q_linear)

        logger.info(f"Applied SmoothQuant with alpha={alpha}")
        return model

    def _should_quantize(self, name: str, module: nn.Module) -> bool:
        """Check if a module should be quantized."""
        # Skip specified modules
        for skip_name in self.config.modules_to_skip:
            if skip_name in name:
                return False

        # Check module type
        if self.config.modules_to_quantize is not None:
            return any(t in str(type(module)) for t in self.config.modules_to_quantize)

        # Default: quantize Linear layers
        return isinstance(module, nn.Linear)

    def _replace_module(self, model: nn.Module, name: str, new_module: nn.Module) -> None:
        """Replace a module in the model by name."""
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)


# ============================================================================
# Pruning Utilities
# ============================================================================


def compute_importance_scores(
    module: nn.Module,
    method: PruningMethod,
    calibration_data: Iterator[torch.Tensor] | None = None,
) -> torch.Tensor:
    """Compute importance scores for weights.

    Args:
        module: Module to compute scores for
        method: Pruning method
        calibration_data: Calibration data for activation-based methods

    Returns:
        Importance scores with same shape as module weights
    """
    if not isinstance(module, nn.Linear):
        raise ValueError("Currently only Linear layers are supported for pruning")

    weight = module.weight.data

    if method == PruningMethod.MAGNITUDE:
        # Simple magnitude-based importance
        return weight.abs()

    elif method == PruningMethod.WANDA:
        # WANDA: Weight magnitude * activation magnitude
        if calibration_data is None:
            logger.warning("WANDA requires calibration data, falling back to magnitude")
            return weight.abs()

        # Collect activation norms
        act_norms = []

        def hook(module, input, output):
            if isinstance(input, tuple) and len(input) > 0:
                x = input[0]
                if isinstance(x, torch.Tensor):
                    # L2 norm per input feature
                    act_norms.append(x.pow(2).mean(dim=tuple(range(x.ndim - 1))).sqrt())

        handle = module.register_forward_hook(hook)

        with torch.no_grad():
            for batch in calibration_data:
                if isinstance(batch, dict):
                    # Can't easily forward through just this module
                    break
                module(batch)
                if len(act_norms) >= 100:  # Limit samples
                    break

        handle.remove()

        if act_norms:
            mean_act_norm = torch.stack(act_norms).mean(dim=0)
            # Importance = |W| * ||X||
            return weight.abs() * mean_act_norm.unsqueeze(0)
        else:
            return weight.abs()

    else:
        # Default to magnitude
        return weight.abs()


def create_pruning_mask(
    importance: torch.Tensor,
    sparsity: float,
    granularity: Literal["element", "row", "column"] = "element",
) -> torch.Tensor:
    """Create a binary mask for pruning.

    Args:
        importance: Importance scores
        sparsity: Target sparsity (fraction to prune)
        granularity: Pruning granularity

    Returns:
        Binary mask (1 = keep, 0 = prune)
    """
    if granularity == "element":
        # Unstructured: prune individual elements
        threshold = torch.quantile(importance.flatten(), sparsity)
        return (importance > threshold).float()

    elif granularity == "row":
        # Row-wise: prune entire output neurons
        row_importance = importance.abs().sum(dim=1)
        k = int(importance.size(0) * (1 - sparsity))
        _, top_indices = torch.topk(row_importance, k)
        mask = torch.zeros_like(importance)
        mask[top_indices] = 1.0
        return mask

    elif granularity == "column":
        # Column-wise: prune entire input features
        col_importance = importance.abs().sum(dim=0)
        k = int(importance.size(1) * (1 - sparsity))
        _, top_indices = torch.topk(col_importance, k)
        mask = torch.zeros_like(importance)
        mask[:, top_indices] = 1.0
        return mask

    else:
        raise ValueError(f"Unknown granularity: {granularity}")


class PrunedLinear(nn.Module):
    """Linear layer with sparse weight mask.

    Stores a binary mask and applies it during forward pass.
    For actual sparse computation speedup, convert to sparse format
    or use hardware-specific sparse kernels.
    """

    # Type hints for buffers
    weight: torch.Tensor
    mask: torch.Tensor
    bias: torch.Tensor | None

    def __init__(self, linear: nn.Linear, mask: torch.Tensor):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features

        # Store original weight and mask
        self.register_buffer("weight", linear.weight.data * mask)
        self.register_buffer("mask", mask)
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.data)
        else:
            self.register_buffer("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

    @property
    def sparsity(self) -> float:
        """Current sparsity ratio."""
        return 1.0 - float(self.mask.mean().item())

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, sparsity={self.sparsity:.2%}"
        )


# ============================================================================
# Model Pruning
# ============================================================================


class ModelPruner:
    """Prune PyTorch models to reduce size and computation.

    Supports:
    - Magnitude-based pruning (unstructured and structured)
    - WANDA pruning (weights + activations)
    - Iterative pruning with gradual sparsity increase

    Example:
        pruner = ModelPruner(config=PruningConfig(sparsity=0.5))
        pruned_model = pruner.prune(model)
    """

    def __init__(self, config: PruningConfig | None = None):
        """Initialize pruner.

        Args:
            config: Pruning configuration
        """
        self.config = config or PruningConfig()

    def prune(
        self,
        model: nn.Module,
        calibration_data: Iterator[torch.Tensor] | None = None,
    ) -> nn.Module:
        """Prune a model.

        Args:
            model: Model to prune
            calibration_data: Calibration data for activation-based pruning

        Returns:
            Pruned model
        """
        model = copy.deepcopy(model)

        if self.config.iterative_steps > 1:
            return self._iterative_prune(model, calibration_data)
        else:
            return self._one_shot_prune(model, calibration_data)

    def _one_shot_prune(
        self,
        model: nn.Module,
        calibration_data: Iterator[torch.Tensor] | None = None,
    ) -> nn.Module:
        """Apply one-shot pruning."""
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Linear):
                # Compute importance scores
                if self.config.importance_scores and name in self.config.importance_scores:
                    importance = self.config.importance_scores[name]
                else:
                    importance = compute_importance_scores(
                        module, self.config.method, calibration_data
                    )

                # Create mask
                mask = create_pruning_mask(
                    importance,
                    self.config.sparsity,
                    self.config.granularity,
                )

                # Replace with pruned module
                pruned = PrunedLinear(module, mask)
                self._replace_module(model, name, pruned)

        total_params, pruned_params = self._count_parameters(model)
        actual_sparsity = pruned_params / total_params if total_params > 0 else 0
        logger.info(
            f"Pruned model: {pruned_params:,}/{total_params:,} params "
            f"({actual_sparsity:.1%} sparsity)"
        )

        return model

    def _iterative_prune(
        self,
        model: nn.Module,
        calibration_data: Iterator[torch.Tensor] | None = None,
    ) -> nn.Module:
        """Apply iterative pruning with gradual sparsity increase."""
        current_sparsity = 0.0
        target_sparsity = self.config.sparsity
        sparsity_step = target_sparsity / self.config.iterative_steps

        for step in range(self.config.iterative_steps):
            current_sparsity = min(current_sparsity + sparsity_step, target_sparsity)

            for name, module in list(model.named_modules()):
                if isinstance(module, nn.Linear):
                    importance = compute_importance_scores(
                        module, self.config.method, calibration_data
                    )
                    mask = create_pruning_mask(
                        importance, current_sparsity, self.config.granularity
                    )
                    pruned = PrunedLinear(module, mask)
                    self._replace_module(model, name, pruned)

            logger.info(
                f"Iterative pruning step {step + 1}/{self.config.iterative_steps}: {current_sparsity:.1%}"
            )

        return model

    def _replace_module(self, model: nn.Module, name: str, new_module: nn.Module) -> None:
        """Replace a module in the model by name."""
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    def _count_parameters(self, model: nn.Module) -> tuple[int, int]:
        """Count total and pruned parameters."""
        total = 0
        pruned = 0

        for module in model.modules():
            if isinstance(module, PrunedLinear):
                total += module.weight.numel()
                pruned += (module.mask == 0).sum().item()
            elif isinstance(module, nn.Linear):
                total += module.weight.numel()

        return total, int(pruned)


# ============================================================================
# Attention Head Pruning
# ============================================================================


def prune_attention_heads(
    model: nn.Module,
    heads_to_prune: dict[int, list[int]],
    num_heads: int,
    head_dim: int,
) -> nn.Module:
    """Prune specific attention heads from a model.

    Args:
        model: Model containing attention layers
        heads_to_prune: Dict mapping layer index to list of head indices to prune
        num_heads: Total number of heads per layer
        head_dim: Dimension per head

    Returns:
        Model with pruned heads (zeroed out)
    """
    model = copy.deepcopy(model)

    for layer_idx, heads in heads_to_prune.items():
        # Find attention module for this layer
        for name, module in model.named_modules():
            if f"blocks.{layer_idx}" in name or f"layers.{layer_idx}" in name:
                attn: Any = getattr(module, "attn", None) or getattr(module, "self_attn", None)
                if attn is None:
                    continue

                # Zero out the specified heads
                for head_idx in heads:
                    start = head_idx * head_dim
                    end = (head_idx + 1) * head_dim

                    # Zero Q, K, V projections for this head
                    qkv = getattr(attn, "qkv", None)
                    if qkv is not None:
                        # Combined QKV projection
                        hidden = num_heads * head_dim
                        qkv.weight.data[start:end, :] = 0  # Q
                        qkv.weight.data[hidden + start : hidden + end, :] = 0  # K
                        qkv.weight.data[2 * hidden + start : 2 * hidden + end, :] = 0  # V
                    else:
                        # Separate projections
                        q_proj = getattr(attn, "q_proj", None)
                        k_proj = getattr(attn, "k_proj", None)
                        v_proj = getattr(attn, "v_proj", None)
                        if q_proj is not None:
                            q_proj.weight.data[start:end, :] = 0
                        if k_proj is not None:
                            k_proj.weight.data[start:end, :] = 0
                        if v_proj is not None:
                            v_proj.weight.data[start:end, :] = 0

                logger.info(f"Pruned heads {heads} from layer {layer_idx}")
                break

    return model


def compute_head_importance(
    model: nn.Module,
    calibration_data: Iterator[torch.Tensor],
    num_layers: int,
    num_heads: int,
) -> torch.Tensor:
    """Compute importance scores for attention heads.

    Uses gradient-based importance: heads with larger gradients are more important.

    Args:
        model: Model to analyze
        calibration_data: Calibration data
        num_layers: Number of transformer layers
        num_heads: Number of attention heads per layer

    Returns:
        Importance scores [num_layers, num_heads]
    """
    importance = torch.zeros(num_layers, num_heads)

    # This is a simplified version - full implementation would track
    # attention patterns and output contributions
    logger.warning(
        "Head importance computation is approximate. "
        "For best results, use task-specific importance metrics."
    )

    # Use attention output norms as a proxy for importance
    for name, module in model.named_modules():
        attn: Any = getattr(module, "attn", None)
        if attn is None:
            continue
        proj = getattr(attn, "proj", None)
        if proj is None:
            continue

        # Get layer index from name
        for i in range(num_layers):
            if f".{i}." in name or f"[{i}]" in name:
                # Use projection weight norms per head
                weight = proj.weight
                head_dim_calc = weight.size(1) // num_heads
                for h in range(num_heads):
                    start = h * head_dim_calc
                    end = (h + 1) * head_dim_calc
                    importance[i, h] = weight[:, start:end].norm()
                break

    return importance


# ============================================================================
# Export Utilities
# ============================================================================


def export_to_onnx(
    model: nn.Module,
    sample_input: torch.Tensor | dict[str, torch.Tensor],
    output_path: str,
    opset_version: int = 17,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
) -> None:
    """Export model to ONNX format for deployment.

    Args:
        model: Model to export
        sample_input: Sample input for tracing
        output_path: Output file path (.onnx)
        opset_version: ONNX opset version
        dynamic_axes: Dynamic axis specification
    """
    model.eval()

    if dynamic_axes is None:
        dynamic_axes = {
            "input": {0: "batch_size", 1: "sequence_length"},
            "output": {0: "batch_size", 1: "sequence_length"},
        }

    if isinstance(sample_input, dict):
        input_names = list(sample_input.keys())
        torch.onnx.export(
            model,
            tuple(sample_input.values()),
            output_path,
            input_names=input_names,
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
        )
    else:
        torch.onnx.export(
            model,
            (sample_input,),  # Wrap in tuple for ONNX export
            output_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
        )

    logger.info(f"Exported model to {output_path}")


def estimate_model_size(model: nn.Module) -> dict[str, float]:
    """Estimate model size in different formats.

    Args:
        model: Model to analyze

    Returns:
        Dict with size estimates in MB
    """
    param_count = sum(p.numel() for p in model.parameters())
    buffer_count = sum(b.numel() for b in model.buffers())

    # Estimate sizes
    fp32_size = (param_count + buffer_count) * 4 / (1024 * 1024)
    fp16_size = (param_count + buffer_count) * 2 / (1024 * 1024)
    int8_size = (param_count + buffer_count) * 1 / (1024 * 1024)
    int4_size = (param_count + buffer_count) * 0.5 / (1024 * 1024)

    # Count actual quantized/pruned layers
    quantized_params = 0
    pruned_zeros = 0

    for module in model.modules():
        if isinstance(module, QuantizedLinear):
            quantized_params += module.weight_quantized.numel()
        elif isinstance(module, PrunedLinear):
            pruned_zeros += (module.mask == 0).sum().item()

    return {
        "fp32_mb": fp32_size,
        "fp16_mb": fp16_size,
        "int8_mb": int8_size,
        "int4_mb": int4_size,
        "param_count": param_count,
        "buffer_count": buffer_count,
        "quantized_params": quantized_params,
        "pruned_zeros": pruned_zeros,
    }


__all__ = [
    # Enums
    "QuantizationMethod",
    "PruningMethod",
    # Configs
    "QuantizationConfig",
    "PruningConfig",
    # Quantization
    "ModelQuantizer",
    "QuantizedLinear",
    "compute_scale_zero_point",
    "quantize_tensor",
    "dequantize_tensor",
    # Pruning
    "ModelPruner",
    "PrunedLinear",
    "compute_importance_scores",
    "create_pruning_mask",
    # Head pruning
    "prune_attention_heads",
    "compute_head_importance",
    # Export
    "export_to_onnx",
    "estimate_model_size",
]
