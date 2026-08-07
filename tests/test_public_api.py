"""Public-API regression guard.

Imports every symbol advertised in the README and docs so that the documented
surface can never silently drift from the code. Each symbol must be importable
and either callable (functions/classes) or a non-None object.

If any of these fail, either the code moved/renamed a public symbol or the docs
promise something that no longer exists -- both are bugs worth catching in CI.
"""

from __future__ import annotations

import importlib


def _assert_callable(obj, name: str) -> None:
    assert obj is not None, f"{name} is None"
    assert callable(obj), f"{name} is not callable"


# ---------------------------------------------------------------------------
# Direct README import lines (copied verbatim so docs == code).
# ---------------------------------------------------------------------------
def test_sampling_guidance_imports():
    from medlatents.sampling import (
        GuidedSampler,
        constant_guidance,
        cosine_guidance,
        linear_guidance,
        triangular_guidance,
    )

    for obj, name in [
        (GuidedSampler, "GuidedSampler"),
        (cosine_guidance, "cosine_guidance"),
        (linear_guidance, "linear_guidance"),
        (constant_guidance, "constant_guidance"),
        (triangular_guidance, "triangular_guidance"),
    ]:
        _assert_callable(obj, name)


def test_evaluation_imports():
    from medlatents.evaluation import (
        FIDCalculator,
        PrecisionRecallCalculator,
        calculate_fid,
        calculate_nll,
        calculate_precision_recall,
    )

    for obj, name in [
        (FIDCalculator, "FIDCalculator"),
        (PrecisionRecallCalculator, "PrecisionRecallCalculator"),
        (calculate_fid, "calculate_fid"),
        (calculate_precision_recall, "calculate_precision_recall"),
        (calculate_nll, "calculate_nll"),
    ]:
        _assert_callable(obj, name)


def test_flow_matching_imports():
    from medlatents.flow_matching import RectifiedFlow, RectifiedFlowPP

    _assert_callable(RectifiedFlow, "RectifiedFlow")
    _assert_callable(RectifiedFlowPP, "RectifiedFlowPP")


def test_flow_matching_continuous_submodule_imports():
    # README references the continuous submodule path explicitly.
    from medlatents.flow_matching.continuous import RectifiedFlow, RectifiedFlowPP

    _assert_callable(RectifiedFlow, "continuous.RectifiedFlow")
    _assert_callable(RectifiedFlowPP, "continuous.RectifiedFlowPP")


def test_training_imports():
    from medlatents.training import MaskingSchedule

    _assert_callable(MaskingSchedule, "MaskingSchedule")


def test_autoregressive_imports():
    from medlatents.autoregressive import AutoregressiveTransformer, SpeculativeDecoder

    _assert_callable(SpeculativeDecoder, "SpeculativeDecoder")
    _assert_callable(AutoregressiveTransformer, "AutoregressiveTransformer")


def test_maskgit_imports():
    from medlatents.maskgit import MaskGIT

    _assert_callable(MaskGIT, "MaskGIT")


def test_sampling_maskgit_imports():
    from medlatents.sampling.maskgit import (
        KLASSGenerator,
        MaskGITScheduler,
        halton_schedule_1d,
    )

    _assert_callable(MaskGITScheduler, "MaskGITScheduler")
    _assert_callable(KLASSGenerator, "KLASSGenerator")
    _assert_callable(halton_schedule_1d, "halton_schedule_1d")


def test_bayesian_flow_imports():
    from medlatents.bayesian_flow import compute_entropy, exponential_schedule

    _assert_callable(compute_entropy, "compute_entropy")
    _assert_callable(exponential_schedule, "exponential_schedule")


def test_post_training_imports():
    # README "post-training stack" code block.
    from medlatents.post_training import (
        AutoregressiveDPOTrainer,
        PostTrainingConfig,
        PreferenceDataset,
    )

    _assert_callable(PostTrainingConfig, "PostTrainingConfig")
    _assert_callable(PreferenceDataset, "PreferenceDataset")
    _assert_callable(AutoregressiveDPOTrainer, "AutoregressiveDPOTrainer")


def test_top_level_model_imports():
    from medlatents import (
        AutoregressiveTransformer,
        ContinuousDiT,
        DiscreteDiT,
        MaskGIT,
    )

    for obj, name in [
        (AutoregressiveTransformer, "AutoregressiveTransformer"),
        (MaskGIT, "MaskGIT"),
        (DiscreteDiT, "DiscreteDiT"),
        (ContinuousDiT, "ContinuousDiT"),
    ]:
        _assert_callable(obj, name)


# ---------------------------------------------------------------------------
# Table-driven sweep: a single source of truth for the documented surface,
# so adding a doc symbol here guarantees an import-level regression check.
# ---------------------------------------------------------------------------
PUBLIC_SURFACE: dict[str, list[str]] = {
    "medlatents": [
        "AutoregressiveTransformer",
        "MaskGIT",
        "DiscreteDiT",
        "ContinuousDiT",
    ],
    "medlatents.sampling": [
        "GuidedSampler",
        "cosine_guidance",
        "linear_guidance",
        "constant_guidance",
        "triangular_guidance",
    ],
    "medlatents.sampling.maskgit": [
        "MaskGITScheduler",
        "KLASSGenerator",
        "halton_schedule_1d",
    ],
    "medlatents.evaluation": [
        "FIDCalculator",
        "PrecisionRecallCalculator",
        "calculate_fid",
        "calculate_precision_recall",
        "calculate_nll",
    ],
    "medlatents.flow_matching": ["RectifiedFlow", "RectifiedFlowPP"],
    "medlatents.flow_matching.continuous": ["RectifiedFlow", "RectifiedFlowPP"],
    "medlatents.training": ["MaskingSchedule"],
    "medlatents.autoregressive": ["SpeculativeDecoder", "AutoregressiveTransformer"],
    "medlatents.maskgit": ["MaskGIT"],
    "medlatents.bayesian_flow": ["compute_entropy", "exponential_schedule"],
    "medlatents.post_training": [
        "PostTrainingConfig",
        "PreferenceDataset",
        "AutoregressiveDPOTrainer",
    ],
}


def test_documented_surface_is_importable():
    """Every documented symbol resolves and is non-None."""
    missing: list[str] = []
    for module_name, symbols in PUBLIC_SURFACE.items():
        module = importlib.import_module(module_name)
        for symbol in symbols:
            obj = getattr(module, symbol, None)
            if obj is None:
                missing.append(f"{module_name}.{symbol}")
    assert not missing, f"Documented public symbols missing/None: {missing}"
