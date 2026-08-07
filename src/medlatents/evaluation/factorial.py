"""Factorial design summaries for tokenizer-generator result grids."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import log

import numpy as np


@dataclass(frozen=True)
class FactorialCell:
    """One balanced-factorial observation."""

    quantizer: str
    vocabulary: str
    generator: str
    value: float


def _unique(values: Iterable[str]) -> list[str]:
    return sorted(set(values))


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return ranks


def decompose_balanced_three_factor(
    cells: Iterable[FactorialCell],
    *,
    log_transform: bool = True,
) -> dict[str, dict[str, float]]:
    """Return a descriptive sum-of-squares decomposition for a balanced grid.

    The returned sums of squares are the orthogonal fixed-effect
    decomposition for a full factorial with one observation per
    (quantizer, vocabulary, generator) cell. With one observation per
    combination, the three-way term is a residual summary of unmodeled
    interaction and any run-to-run variability; it is not experimental
    noise and should not be used for inferential tests.
    """

    rows = list(cells)
    if not rows:
        raise ValueError("at least one cell is required")

    quantizers = _unique(row.quantizer for row in rows)
    vocabularies = _unique(row.vocabulary for row in rows)
    generators = _unique(row.generator for row in rows)
    shape = (len(quantizers), len(vocabularies), len(generators))
    values = np.full(shape, np.nan, dtype=float)
    q_idx = {name: i for i, name in enumerate(quantizers)}
    v_idx = {name: i for i, name in enumerate(vocabularies)}
    g_idx = {name: i for i, name in enumerate(generators)}

    for row in rows:
        value = float(row.value)
        if log_transform:
            if value <= 0:
                raise ValueError("log transform requires positive values")
            value = log(value)
        values[q_idx[row.quantizer], v_idx[row.vocabulary], g_idx[row.generator]] = value

    if np.isnan(values).any():
        raise ValueError("balanced decomposition requires a complete grid")

    grand = values.mean()
    mean_q = values.mean(axis=(1, 2), keepdims=True)
    mean_v = values.mean(axis=(0, 2), keepdims=True)
    mean_g = values.mean(axis=(0, 1), keepdims=True)
    mean_qv = values.mean(axis=2, keepdims=True)
    mean_qg = values.mean(axis=1, keepdims=True)
    mean_vg = values.mean(axis=0, keepdims=True)

    total = float(((values - grand) ** 2).sum())
    components = {
        "quantizer": float(np.prod(shape[1:]) * ((mean_q - grand) ** 2).sum()),
        "vocabulary": float(shape[0] * shape[2] * ((mean_v - grand) ** 2).sum()),
        "generator": float(np.prod(shape[:2]) * ((mean_g - grand) ** 2).sum()),
        "quantizer:vocabulary": float(shape[2] * ((mean_qv - mean_q - mean_v + grand) ** 2).sum()),
        "quantizer:generator": float(shape[1] * ((mean_qg - mean_q - mean_g + grand) ** 2).sum()),
        "vocabulary:generator": float(shape[0] * ((mean_vg - mean_v - mean_g + grand) ** 2).sum()),
    }
    components["three_way_residual"] = max(total - sum(components.values()), 0.0)

    return {
        name: {"sum_squares": ss, "fraction": ss / total if total else 0.0}
        for name, ss in components.items()
    }


def mean_ranks_by_factor(cells: Iterable[FactorialCell]) -> dict[str, dict[str, float]]:
    """Return mean FID ranks grouped by quantizer, vocabulary, and generator."""

    rows = list(cells)
    values = np.asarray([row.value for row in rows], dtype=float)
    ranks = _rankdata(values)
    grouped: dict[str, dict[str, list[float]]] = {
        "quantizer": {},
        "vocabulary": {},
        "generator": {},
    }
    for row, rank in zip(rows, ranks, strict=True):
        grouped["quantizer"].setdefault(row.quantizer, []).append(float(rank))
        grouped["vocabulary"].setdefault(row.vocabulary, []).append(float(rank))
        grouped["generator"].setdefault(row.generator, []).append(float(rank))
    return {
        factor: {level: float(np.mean(vals)) for level, vals in levels.items()}
        for factor, levels in grouped.items()
    }


def discrete_factorial_cells_from_results(
    results: Mapping[str, Mapping[str, float]],
    *,
    dataset: str = "chestmnist",
) -> list[FactorialCell]:
    """Parse 3x3 discrete tokenizer-generator cells from evaluation results."""

    generators = {
        "transformer": "AR",
        "maskgit": "MaskGIT",
        "flow": "DFM",
        "d3pm": "D3PM",
        "sedd": "SEDD",
        "bayesian_flow": "BFN",
    }
    cells: list[FactorialCell] = []
    for quantizer in ("VQ", "LFQ", "FSQ"):
        for vocabulary in ("1024", "2048", "4096"):
            for generator_key, generator_name in generators.items():
                name = f"{dataset}_{quantizer}{vocabulary}_{generator_key}"
                if name not in results:
                    continue
                cells.append(
                    FactorialCell(
                        quantizer=quantizer,
                        vocabulary=vocabulary,
                        generator=generator_name,
                        value=float(results[name]["fid"]),
                    )
                )
    return cells
