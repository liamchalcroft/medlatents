"""Configuration classes for post-training methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class PostTrainingConfig:
    """Configuration for post-training methods.

    This config class supports all post-training methods in medlatents:
    - DPO variants (standard, IPO, KTO, SPO)
    - RL methods (DDPO, GRPO, GARDO)
    - Distillation (consistency, reflow)
    - Self-play (SPIN, RFT)

    The config is designed to be serializable for experiment tracking and
    supports sensible defaults that can be overridden as needed.

    Example:
        >>> config = PostTrainingConfig(method="dpo", beta=0.1, lr=1e-5)
        >>> config = PostTrainingConfig.from_preset("dpo_small")
    """

    # ===== Method Selection =====
    method: Literal[
        "dpo",
        "spo",
        "ipo",
        "kto",
        "ddpo",
        "grpo",
        "gardo",
        "consistency",
        "reflow",
        "self_play",
        "rft",
    ] = "dpo"

    # ===== Common Training Parameters =====
    lr: float = 1e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_steps: int = 1000
    warmup_steps: int = 100
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    mixed_precision: Literal["no", "fp16", "bf16"] = "bf16"
    seed: int | None = 42

    # ===== EMA Settings =====
    use_ema: bool = True
    ema_decay: float = 0.9999

    # ===== DPO-Specific Parameters =====
    beta: float = 0.1  # KL penalty weight
    loss_type: Literal["sigmoid", "hinge", "ipo", "kto", "disco"] = "sigmoid"
    reference_free: bool = False  # If True, don't use reference model
    label_smoothing: float = 0.0  # Soft labels for robustness

    # ===== SPO-Specific Parameters =====
    num_candidates: int = 4  # Candidates per denoising step
    step_preference_model: str | None = None  # Path or "train"

    # ===== RL-Specific Parameters =====
    clip_range: float = 0.2  # PPO/GRPO clipping
    kl_coeff: float = 0.1  # KL penalty coefficient
    num_samples_per_prompt: int = 4  # Samples for reward estimation
    normalize_rewards: bool = True  # Group-normalize rewards
    gae_lambda: float = 0.95  # GAE lambda for DDPO

    # ===== GARDO-Specific =====
    reward_threshold: float = 0.1  # Minimum reward improvement
    distance_penalty: float = 0.01  # Penalty for KL divergence

    # ===== Distillation Parameters =====
    teacher_path: str | None = None
    num_distillation_steps: int = 50
    huber_c: float = 0.00054  # Pseudo-Huber loss parameter
    discretization_steps: int = 18  # Consistency model discretization

    # ===== Reflow Parameters =====
    num_reflow_iterations: int = 2
    pairs_per_iteration: int = 10000

    # ===== Self-Play Parameters =====
    num_self_play_iterations: int = 3
    top_k_samples: int = 4  # For RFT

    # ===== Reward Model Settings =====
    reward_model_type: Literal["trained", "ai", "discriminator", "external"] = "ai"
    reward_model_path: str | None = None
    reward_model_config: dict[str, Any] = field(default_factory=dict)

    # ===== Logging and Checkpointing =====
    log_every: int = 10
    eval_every: int = 100
    save_every: int = 500
    logdir: str = "./post_training_logs"
    run_name: str | None = None
    wandb_project: str | None = None
    wandb_entity: str | None = None

    # ===== Distributed Training =====
    use_fsdp: bool = True
    fsdp_sharding_strategy: Literal["full", "shard_grad_op", "no_shard"] = "full"

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Validate method-specific requirements
        if self.method == "spo" and self.num_candidates < 2:
            raise ValueError("SPO requires at least 2 candidates per step")

        if self.method in ["ddpo", "grpo", "gardo"] and self.num_samples_per_prompt < 2:
            raise ValueError("RL methods require at least 2 samples per prompt")

        if self.method == "consistency" and self.teacher_path is None:
            raise ValueError("Consistency distillation requires a teacher model")

    @classmethod
    def from_preset(cls, preset: str, **overrides) -> "PostTrainingConfig":
        """Create config from a named preset.

        Available presets:
        - dpo_small: Quick DPO for testing
        - dpo_standard: Standard DPO settings
        - dpo_large: Large-scale DPO
        - spo_standard: Step-by-step preference optimization
        - ddpo_standard: DDPO with standard settings
        - grpo_efficient: Memory-efficient GRPO
        - consistency_fast: Fast consistency distillation

        Args:
            preset: Name of the preset
            **overrides: Parameters to override

        Returns:
            PostTrainingConfig with preset values
        """
        presets = {
            "dpo_small": {
                "method": "dpo",
                "lr": 1e-5,
                "batch_size": 2,
                "gradient_accumulation_steps": 2,
                "max_steps": 500,
                "beta": 0.1,
            },
            "dpo_standard": {
                "method": "dpo",
                "lr": 5e-6,
                "batch_size": 4,
                "gradient_accumulation_steps": 4,
                "max_steps": 2000,
                "beta": 0.1,
            },
            "dpo_large": {
                "method": "dpo",
                "lr": 1e-6,
                "batch_size": 8,
                "gradient_accumulation_steps": 8,
                "max_steps": 10000,
                "beta": 0.05,
                "warmup_steps": 500,
            },
            "spo_standard": {
                "method": "spo",
                "lr": 5e-6,
                "batch_size": 2,
                "gradient_accumulation_steps": 8,
                "max_steps": 3000,
                "beta": 0.1,
                "num_candidates": 4,
            },
            "ddpo_standard": {
                "method": "ddpo",
                "lr": 1e-6,
                "batch_size": 4,
                "gradient_accumulation_steps": 4,
                "max_steps": 5000,
                "clip_range": 1e-4,
                "kl_coeff": 0.1,
                "num_samples_per_prompt": 4,
            },
            "grpo_efficient": {
                "method": "grpo",
                "lr": 5e-6,
                "batch_size": 2,
                "gradient_accumulation_steps": 8,
                "max_steps": 3000,
                "clip_range": 0.2,
                "num_samples_per_prompt": 8,
            },
            "consistency_fast": {
                "method": "consistency",
                "lr": 1e-4,
                "batch_size": 8,
                "gradient_accumulation_steps": 2,
                "max_steps": 20000,
                "discretization_steps": 18,
            },
        }

        if preset not in presets:
            available = ", ".join(presets.keys())
            raise ValueError(f"Unknown preset '{preset}'. Available: {available}")

        config_dict = presets[preset].copy()
        config_dict.update(overrides)
        return cls(**config_dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PostTrainingConfig":
        """Create config from dictionary."""
        return cls(**d)


# ===== Convenience presets as module-level constants =====

DPO_SMALL = PostTrainingConfig.from_preset("dpo_small")
DPO_STANDARD = PostTrainingConfig.from_preset("dpo_standard")
DPO_LARGE = PostTrainingConfig.from_preset("dpo_large")
SPO_STANDARD = PostTrainingConfig.from_preset("spo_standard")
DDPO_STANDARD = PostTrainingConfig.from_preset("ddpo_standard")
GRPO_EFFICIENT = PostTrainingConfig.from_preset("grpo_efficient")


__all__ = [
    "PostTrainingConfig",
    "DPO_SMALL",
    "DPO_STANDARD",
    "DPO_LARGE",
    "SPO_STANDARD",
    "DDPO_STANDARD",
    "GRPO_EFFICIENT",
]
