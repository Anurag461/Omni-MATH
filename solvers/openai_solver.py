"""OpenAI API solver for Omni-MATH."""

import os
from typing import Any

from .base import SolverBase, SolverResponse


class OpenAISolver(SolverBase):
    """
    Solve mathematical problems using OpenAI API.

    Uses the OpenAI Python SDK to call models like GPT-4, o1, etc.
    Requires OPENAI_API_KEY environment variable to be set.
    """

    def __init__(
        self,
        *,
        model: str = "gpt-4",
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: float = 15.0,
    ):
        """
        Initialize OpenAI API solver.

        Args:
            model: Model to use (e.g., "gpt-4", "gpt-4-turbo", "o1-preview")
            system_prompt: Custom system prompt for problem solving
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Maximum tokens to generate
            timeout: Maximum time in seconds for API call (default: 15.0)
        """
        super().__init__(timeout=timeout)
        self.model = model
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.temperature = temperature
        self.max_tokens = max_tokens

        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY must be set. "
                "Set it with: export OPENAI_API_KEY='your-api-key'\n"
                "Get your API key from: https://platform.openai.com/api-keys"
            )

        # Import OpenAI client (lazy import to avoid requiring it if not used)
        try:
            from openai import OpenAI
            self.client = OpenAI(timeout=self.timeout)
        except ImportError:
            raise ImportError(
                "OpenAI Python SDK not installed. "
                "Install with: pip install openai"
            )

    def _default_system_prompt(self) -> str:
        """Default system prompt for mathematical problem solving."""
        return """You are an expert mathematician solving Olympiad-level mathematics problems.

Provide a detailed, step-by-step solution to the problem. Your solution should:
1. Clearly explain your reasoning
2. Show all mathematical work
3. Use proper mathematical notation (LaTeX where appropriate)
4. Conclude with the final answer"""

    def __call__(self, problem: str) -> SolverResponse:
        """
        Generate a solution using OpenAI API.

        Args:
            problem: The mathematical problem statement

        Returns:
            SolverResponse with the generated solution
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": problem},
        ]

        # Prepare API call parameters
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }

        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens

        # Call OpenAI API
        try:
            response = self.client.chat.completions.create(**params)
        except TimeoutError as e:
            raise RuntimeError(f"OpenAI API call timed out after {self.timeout}s: {e}")
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {e}")

        # Extract solution
        solution = response.choices[0].message.content or ""

        # Extract metadata
        metadata = {
            "solver": "openai_api",
            "model": self.model,
            "temperature": self.temperature,
        }

        # Add usage information
        if response.usage:
            metadata["usage"] = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            metadata["input_tokens"] = response.usage.prompt_tokens
            metadata["output_tokens"] = response.usage.completion_tokens

        # Add finish reason
        if response.choices[0].finish_reason:
            metadata["finish_reason"] = response.choices[0].finish_reason

        return SolverResponse(solution=solution, metadata=metadata)
