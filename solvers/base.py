"""Base classes for Omni-MATH solvers."""

from dataclasses import dataclass
from typing import Any


@dataclass
class SolverResponse:
    """
    Response from a solver.

    Attributes:
        solution: The generated mathematical solution
        metadata: Additional information about the generation (e.g., model, tokens used)
    """
    solution: str
    metadata: dict[str, Any]


class SolverBase:
    """
    Base class for mathematical problem solvers.

    Solvers take a problem statement and generate a solution.
    """

    def __init__(self, timeout: float = 15.0):
        """
        Initialize solver.

        Args:
            timeout: Maximum time in seconds for solution generation (default: 15.0)
        """
        self.timeout = timeout

    def __call__(self, problem: str) -> SolverResponse:
        """
        Generate a solution for the given problem.

        Args:
            problem: The mathematical problem statement (may include LaTeX)

        Returns:
            SolverResponse containing the solution and metadata
        """
        raise NotImplementedError
