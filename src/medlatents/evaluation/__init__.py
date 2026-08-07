"""Evaluation metrics for reconstruction, generation quality, memorization
probes, and FID bootstrap confidence intervals."""

from .classifier_utility import (
    build_grayscale_resnet18,
    classifier_fid,
    evaluate_auc,
    extract_classifier_features,
    extract_features,
    train_classifier,
)
from .factorial import (
    FactorialCell,
    decompose_balanced_three_factor,
    discrete_factorial_cells_from_results,
    mean_ranks_by_factor,
)
from .features import get_feature_extractor
from .fid import (
    bootstrap_fid_noise_floor,
    bootstrap_fid_real_vs_gen,
    fid_from_features,
)
from .generative import (
    FIDCalculator,
    PrecisionRecallCalculator,
    calculate_fid,
    calculate_nll,
    calculate_precision_recall,
)
from .memorization import (
    InceptionPool3FeatureExtractor,
    cosine_distance_matrix,
    memorization_metrics,
    memorization_pairs,
    nearest_neighbor_gallery_pairs,
)
from .reconstruction import calculate_psnr, calculate_ssim

__all__ = [
    "calculate_psnr",
    "calculate_ssim",
    "InceptionPool3FeatureExtractor",
    "cosine_distance_matrix",
    "memorization_metrics",
    "memorization_pairs",
    "nearest_neighbor_gallery_pairs",
    "fid_from_features",
    "bootstrap_fid_noise_floor",
    "bootstrap_fid_real_vs_gen",
    "FactorialCell",
    "decompose_balanced_three_factor",
    "discrete_factorial_cells_from_results",
    "mean_ranks_by_factor",
    "build_grayscale_resnet18",
    "train_classifier",
    "evaluate_auc",
    "extract_features",
    "extract_classifier_features",
    "classifier_fid",
    "get_feature_extractor",
    # Feature-based generative quality metrics (functional + accumulator APIs)
    "calculate_fid",
    "calculate_precision_recall",
    "calculate_nll",
    "FIDCalculator",
    "PrecisionRecallCalculator",
]
