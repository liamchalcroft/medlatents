"""BFN solver subpackage.

Modules:
- base: BaseBFNSolver
- euler: EulerSolver, HeunSolver
- dpm: DPMSolver2, DPMSolver3
- exponential: ExponentialIntegrator, StochasticHeun
"""

from .base import BaseBFNSolver
from .dpm import DPMSolver2, DPMSolver3
from .euler import EulerSolver, HeunSolver
from .exponential import ExponentialIntegrator, StochasticHeun


def get_bfn_solver(
    solver_type: str,
    model,
    num_steps: int = 50,
    **kwargs,
):
    """Get a BFN solver by name.

    Args:
        solver_type: One of 'euler', 'heun', 'dpm2', 'dpm3', 'exponential', 'stochastic_heun'
        model: BayesianFlowTransformer model
        num_steps: Number of solver steps
        **kwargs: Additional solver arguments

    Returns:
        Solver instance
    """
    solvers = {
        "euler": EulerSolver,
        "heun": HeunSolver,
        "dpm2": DPMSolver2,
        "dpm3": DPMSolver3,
        "exponential": ExponentialIntegrator,
        "stochastic_heun": StochasticHeun,
    }

    if solver_type not in solvers:
        raise ValueError(f"Unknown solver: {solver_type}. Available: {list(solvers.keys())}")

    return solvers[solver_type](model, num_steps, **kwargs)


__all__ = [
    "BaseBFNSolver",
    "EulerSolver",
    "HeunSolver",
    "DPMSolver2",
    "DPMSolver3",
    "ExponentialIntegrator",
    "StochasticHeun",
    "get_bfn_solver",
]
