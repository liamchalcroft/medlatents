"""Configuration models for MedLatents.

Pydantic is optional - install with: pip install -e ".[docs]"

The predefined ``NANO_CONFIG`` / ``SMALL_CONFIG`` / ``BASE_CONFIG`` presets derive
their architecture dimensions from :data:`medlatents.configs.MODEL_CONFIGS`, the
single source of truth for model sizes, so the pydantic and dict-based config
surfaces never disagree.
"""

from __future__ import annotations

from typing import Literal

from medlatents.configs import MODEL_CONFIGS

# Architecture dimensions come from the single source of truth in medlatents.configs.
_NANO = MODEL_CONFIGS["nano"]
_SMALL = MODEL_CONFIGS["small"]
_BASE = MODEL_CONFIGS["base"]

# Try to import Pydantic, but make it optional
try:
    from pydantic import BaseModel, Field

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

if PYDANTIC_AVAILABLE:

    class ModelConfig(BaseModel):
        """Base model configuration."""

        model_type: Literal["autoreg", "maskgit", "flow", "diffusion", "bayesian_flow"]
        seq_length: int = Field(gt=0, description="Sequence length")
        vocab_size: int = Field(gt=0, description="Vocabulary size")
        hidden_size: int = Field(gt=0, description="Hidden size")
        depth: int = Field(gt=0, description="Number of transformer layers")
        num_heads: int = Field(gt=0, description="Number of attention heads")
        gradient_checkpointing: bool = False
        chunk_size: int | None = Field(
            default=None, ge=0, description="Chunk size for long sequences"
        )
        allow_dynamic_seq_length: bool = False

    class TrainingConfig(BaseModel):
        """Training configuration."""

        epochs: int = Field(gt=0, default=100, description="Number of training epochs")
        batch_size: int = Field(gt=0, default=32, description="Batch size")
        learning_rate: float = Field(gt=0, default=1e-4, description="Learning rate")
        weight_decay: float = Field(ge=0, default=0.01, description="Weight decay")
        gradient_accumulation_steps: int = Field(
            ge=1, default=1, description="Gradient accumulation steps"
        )
        warmup_steps: int = Field(ge=0, default=1000, description="Warmup steps")
        max_grad_norm: float = Field(
            ge=0, default=1.0, description="Max gradient norm for clipping"
        )
        compile_model: bool = False
        compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "reduce-overhead"
        use_mixed_precision: bool = False

    class OptimizerConfig(BaseModel):
        """Optimizer configuration."""

        optimizer_type: Literal["adamw", "sgd", "adam"] = "adamw"
        betas: tuple[float, float] = (0.9, 0.999)
        eps: float = 1e-8
        amsgrad: bool = False

    class SchedulerConfig(BaseModel):
        """Learning rate scheduler configuration."""

        scheduler_type: Literal["cosine", "linear", "warmup_cosine"] = "cosine"
        num_warmup_steps: int = 1000
        min_lr: float = 0.0

    class ExperimentConfig(BaseModel):
        """Complete experiment configuration."""

        model: ModelConfig
        training: TrainingConfig = Field(default_factory=TrainingConfig)
        optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
        scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)

        seed: int = 42
        device: Literal["cuda", "cpu", "mps", "auto"] = "auto"
        checkpoint_dir: str = "checkpoints"
        log_dir: str = "logs"

    # Predefined configurations (dimensions from medlatents.configs.MODEL_CONFIGS)
    NANO_CONFIG = ExperimentConfig(
        model=ModelConfig(
            model_type="autoreg",
            seq_length=256,
            vocab_size=512,
            hidden_size=_NANO["hidden_size"],
            depth=_NANO["depth"],
            num_heads=_NANO["num_heads"],
        ),
    )

    SMALL_CONFIG = ExperimentConfig(
        model=ModelConfig(
            model_type="maskgit",
            seq_length=256,
            vocab_size=512,
            hidden_size=_SMALL["hidden_size"],
            depth=_SMALL["depth"],
            num_heads=_SMALL["num_heads"],
        ),
    )

    BASE_CONFIG = ExperimentConfig(
        model=ModelConfig(
            model_type="diffusion",
            seq_length=4096,
            vocab_size=8192,
            hidden_size=_BASE["hidden_size"],
            depth=_BASE["depth"],
            num_heads=_BASE["num_heads"],
        ),
    )
else:
    # Pydantic not available - provide fallback classes
    ModelConfig = dict
    TrainingConfig = dict
    OptimizerConfig = dict
    SchedulerConfig = dict
    ExperimentConfig = dict

    NANO_CONFIG = {
        "model": {
            "model_type": "autoreg",
            "seq_length": 256,
            "vocab_size": 512,
            "hidden_size": _NANO["hidden_size"],
            "depth": _NANO["depth"],
            "num_heads": _NANO["num_heads"],
        },
        "training": {},
        "optimizer": {},
        "scheduler": {},
        "seed": 42,
        "device": "auto",
        "checkpoint_dir": "checkpoints",
        "log_dir": "logs",
    }

    SMALL_CONFIG = {
        "model": {
            "model_type": "maskgit",
            "seq_length": 256,
            "vocab_size": 512,
            "hidden_size": _SMALL["hidden_size"],
            "depth": _SMALL["depth"],
            "num_heads": _SMALL["num_heads"],
        },
        "training": {},
        "optimizer": {},
        "scheduler": {},
        "seed": 42,
        "device": "auto",
        "checkpoint_dir": "checkpoints",
        "log_dir": "logs",
    }

    BASE_CONFIG = {
        "model": {
            "model_type": "diffusion",
            "seq_length": 4096,
            "vocab_size": 8192,
            "hidden_size": _BASE["hidden_size"],
            "depth": _BASE["depth"],
            "num_heads": _BASE["num_heads"],
        },
        "training": {},
        "optimizer": {},
        "scheduler": {},
        "seed": 42,
        "device": "auto",
        "checkpoint_dir": "checkpoints",
        "log_dir": "logs",
    }

__all__ = [
    "ModelConfig",
    "TrainingConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "ExperimentConfig",
    "NANO_CONFIG",
    "SMALL_CONFIG",
    "BASE_CONFIG",
    "PYDANTIC_AVAILABLE",
]
