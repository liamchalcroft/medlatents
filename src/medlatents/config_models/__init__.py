"""Configuration module for MedLatents."""

from .pydantic import (
    PYDANTIC_AVAILABLE,
)

if PYDANTIC_AVAILABLE:
    from .pydantic import (  # noqa: F401
        BASE_CONFIG,
        NANO_CONFIG,
        SMALL_CONFIG,
        ExperimentConfig,
        ModelConfig,
        OptimizerConfig,
        SchedulerConfig,
        TrainingConfig,
    )

__all__ = [
    "PYDANTIC_AVAILABLE",
]

if PYDANTIC_AVAILABLE:
    __all__.extend(
        [
            "ModelConfig",
            "TrainingConfig",
            "OptimizerConfig",
            "SchedulerConfig",
            "ExperimentConfig",
            "NANO_CONFIG",
            "SMALL_CONFIG",
            "BASE_CONFIG",
        ]
    )
