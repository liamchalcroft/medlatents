"""Training utilities for medlatents."""

from ..common.schedules import get_cosine_schedule_with_warmup
from .augmentation import (
    AdaptiveLossWeighting,
    CurriculumSampler,
    GradientNoiseInjection,
    MaskingSchedule,
    NoiseSchedule,
    ScheduledSampling,
)
from .checkpointing import load_checkpoint, resume_from_checkpoint, save_checkpoint
from .conditional import (
    ConditionalLossWrapper,
    ConditionalTrainingConfig,
    compute_conditional_loss,
    prepare_batch_with_conditioning,
)
from .continuous import ContinuousLatentTrainer
from .curriculum import (
    CompositeCurriculum,
    CurriculumConfig,
    CurriculumDataLoader,
    CurriculumSchedule,
    CurriculumScheduler,
    MaskingRatioCurriculum,
    MultiScaleCurriculum,
    NoiseLevelCurriculum,
    SequenceLengthCurriculum,
    TokenDifficultyCurriculum,
    create_curriculum_from_config,
)
from .dataloader import (
    AsyncPrefetcher,
    CUDAPrefetcher,
    DataLoaderConfig,
    MemoryMappedDataset,
    ShardedIterableDataset,
    collate_variable_length,
    create_optimized_dataloader,
)
from .discrete import DiscreteLatentTrainer
from .losses import (
    ContrastiveFlowMatchingLoss,
    VelocityContrastiveRegularization,
)
from .parallelism import (
    DistributedConfig,
    GradientAccumulator,
    all_reduce_mean,
    barrier,
    broadcast_object,
    cleanup_distributed,
    setup_distributed,
    sync_gradients,
    wrap_model,
    wrap_model_ddp,
    wrap_model_fsdp,
)
from .performance import (
    ZeroGradOptimizer,
    compile_model,
    create_fused_adamw,
    enable_cuda_optimizations,
    get_optimal_dtype,
    optimize_dataloader,
)
from .profiling import MemoryProfiler, TrainingProfiler, profile_training_step
from .repa import (
    AttentionAlignmentLoss,
    HASTEScheduler,
    MultiEncoderREPA,
    REPALoss,
    REPAProjection,
)
from .repa_e import (
    REPAEConfig,
    REPAELoss,
    REPAETrainer,
    VAEWrapper,
)
from .schedulers import get_mask_ratio

__all__ = [
    "DiscreteLatentTrainer",
    "ContinuousLatentTrainer",
    "get_cosine_schedule_with_warmup",
    "get_mask_ratio",
    "save_checkpoint",
    "load_checkpoint",
    "resume_from_checkpoint",
    "TrainingProfiler",
    "MemoryProfiler",
    "profile_training_step",
    # Performance optimizations
    "enable_cuda_optimizations",
    "create_fused_adamw",
    "compile_model",
    "get_optimal_dtype",
    "ZeroGradOptimizer",
    "optimize_dataloader",
    # Data loading
    "DataLoaderConfig",
    "AsyncPrefetcher",
    "CUDAPrefetcher",
    "MemoryMappedDataset",
    "ShardedIterableDataset",
    "create_optimized_dataloader",
    "collate_variable_length",
    # Parallelism
    "DistributedConfig",
    "setup_distributed",
    "cleanup_distributed",
    "wrap_model",
    "wrap_model_ddp",
    "wrap_model_fsdp",
    "sync_gradients",
    "GradientAccumulator",
    "all_reduce_mean",
    "broadcast_object",
    "barrier",
    # Augmentation utilities
    "CurriculumSampler",
    "ScheduledSampling",
    "NoiseSchedule",
    "MaskingSchedule",
    "GradientNoiseInjection",
    "AdaptiveLossWeighting",
    # Curriculum learning
    "CurriculumSchedule",
    "CurriculumConfig",
    "CurriculumScheduler",
    "SequenceLengthCurriculum",
    "MaskingRatioCurriculum",
    "NoiseLevelCurriculum",
    "TokenDifficultyCurriculum",
    "MultiScaleCurriculum",
    "CompositeCurriculum",
    "CurriculumDataLoader",
    "create_curriculum_from_config",
    # Conditional training (REPA, VeCoR, ConditioningBundle)
    "ConditionalTrainingConfig",
    "ConditionalLossWrapper",
    "prepare_batch_with_conditioning",
    "compute_conditional_loss",
    # REPA alignment
    "REPAProjection",
    "REPALoss",
    "HASTEScheduler",
    "AttentionAlignmentLoss",
    "MultiEncoderREPA",
    # Contrastive losses
    "VelocityContrastiveRegularization",
    "ContrastiveFlowMatchingLoss",
    # REPA-E (End-to-end VAE + DiT training)
    "REPAEConfig",
    "REPAELoss",
    "REPAETrainer",
    "VAEWrapper",
]
