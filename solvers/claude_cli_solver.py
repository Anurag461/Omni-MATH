"""Claude Code CLI solver for Omni-MATH."""

import json
import os
import shutil
import subprocess

from .base import SolverBase, SolverResponse


class ClaudeCLISolver(SolverBase):
    """
    Solve mathematical problems using Claude Code CLI.

    Uses non-interactive mode: `claude code --output-format json "<prompt>"`
    Requires ANTHROPIC_API_KEY environment variable to be set.
    """

    def __init__(
        self,
        *,
        model: str = "sonnet",
        binary: str = "claude",
        system_prompt: str | None = None,
        timeout: float = 15.0,
    ):
        """
        Initialize Claude CLI solver.

        Args:
            model: Model to use (e.g., "sonnet", "opus", "haiku")
            binary: Path or name of claude binary
            system_prompt: Custom system prompt for problem solving
            timeout: Maximum time in seconds for CLI execution (default: 15.0)
        """
        super().__init__(timeout=timeout)
        self.api_key_name = "ANTHROPIC_API_KEY"
        self.model = model
        self.binary = binary
        self.system_prompt = system_prompt or self._default_system_prompt()

        if shutil.which(self.binary) is None:
            raise FileNotFoundError(
                f"Claude CLI binary '{self.binary}' not found on PATH. "
                "Install from: https://github.com/anthropics/claude-cli"
            )

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ValueError(
                f"{self.api_key_name} must be set. "
                f"Set it with: export {self.api_key_name}='your-api-key'\n"
                "Get your API key from: https://console.anthropic.com/"
            )

    def _default_system_prompt(self) -> str:
        """Default system prompt for mathematical problem solving."""
        return """You are an expert mathematician solving Olympiad-level mathematics problems.

Provide a detailed, step-by-step solution to the problem. Your solution should:
1. Clearly explain your reasoning
2. Show all mathematical work
3. Use proper mathematical notation (LaTeX where appropriate)
4. Conclude with the final answer

Problem:"""

    def _format_prompt(self, problem: str) -> str:
        """Format the problem with system prompt."""
        return f"{self.system_prompt}\n\n{problem}"

    def __call__(self, problem: str) -> SolverResponse:
        """
        Generate a solution using Claude CLI.

        Args:
            problem: The mathematical problem statement

        Returns:
            SolverResponse with the generated solution
        """
        prompt = self._format_prompt(problem)

        cmd: list[str] = [
            self.binary,
            "code",
            "--model",
            self.model,
            "--output-format",
            "json",
            prompt,
        ]

        try:
            completed = subprocess.run(
                cmd,
                text=True,
                capture_output=True,
                stdin=subprocess.DEVNULL,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"Claude CLI timed out after {self.timeout}s. "
                "Increase timeout with timeout parameter if needed."
            )

        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()

        if completed.returncode != 0:
            error_details = []
            error_details.append(f"Return code: {completed.returncode}")
            error_details.append(f"Command: {' '.join(cmd)}")
            if stderr:
                error_details.append(f"Stderr: {stderr}")
            if stdout:
                error_details.append(f"Stdout: {stdout}")
            error_msg = "\n".join(error_details)

            # Check for authentication errors
            stderr_lower = stderr.lower()
            if (
                "authentication" in stderr_lower
                or "login" in stderr_lower
                or "unauthorized" in stderr_lower
                or "api_key" in stderr_lower
                or "api key" in stderr_lower
            ):
                raise RuntimeError(
                    f"Claude CLI authentication failed: {error_msg}\n"
                    f"Please set {self.api_key_name} environment variable:\n"
                    f"  export {self.api_key_name}='your-api-key'\n"
                    "Get your API key from: https://console.anthropic.com/"
                )
            raise RuntimeError(f"Claude CLI failed: {error_msg}")

        if not stdout:
            raise RuntimeError("Claude CLI returned empty response")

        # Try to parse JSON output
        solution = None
        usage_info = None

        try:
            data = json.loads(stdout)
            # Extract response from JSON
            solution = (
                data.get("response")
                or data.get("output")
                or data.get("text")
                or data.get("content")
                or None
            )

            # Extract usage information if available
            if "usage" in data:
                usage_info = data["usage"]

            # Fallback to raw stdout if no response found
            if not solution:
                solution = stdout
        except json.JSONDecodeError:
            # If JSON parsing fails, use raw stdout
            solution = stdout

        metadata = {
            "solver": "claude_cli",
            "model": self.model,
            "returncode": completed.returncode,
            "stderr": stderr,
            "raw_stdout": stdout,
        }

        # Add usage info if found
        if usage_info:
            metadata["usage"] = usage_info
            metadata["input_tokens"] = usage_info.get("input_tokens")
            metadata["output_tokens"] = usage_info.get("output_tokens")

        return SolverResponse(solution=solution, metadata=metadata)
