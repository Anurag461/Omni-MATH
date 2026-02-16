"""Codex CLI solver for Omni-MATH."""

import json
import os
import shutil
import subprocess

from .base import SolverBase, SolverResponse


class CodexCLISolver(SolverBase):
    """
    Solve mathematical problems using Codex CLI.

    Uses non-interactive mode: `codex exec --json "<prompt>"`
    Requires OPENAI_API_KEY environment variable to be set.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        binary: str = "codex",
        system_prompt: str | None = None,
        timeout: float = 15.0,
    ):
        """
        Initialize Codex CLI solver.

        Args:
            model: Specific model to use (e.g., "gpt-5.1-codex-max")
            binary: Path or name of codex binary
            system_prompt: Custom system prompt for problem solving
            timeout: Maximum time in seconds for CLI execution (default: 15.0)
        """
        super().__init__(timeout=timeout)
        self.api_key_name = "OPENAI_API_KEY"
        self.model = model
        self.binary = binary
        self.system_prompt = system_prompt or self._default_system_prompt()

        if shutil.which(self.binary) is None:
            raise FileNotFoundError(
                f"Codex CLI binary '{self.binary}' not found on PATH. "
                "Install with: npm install -g @openai/codex or brew install --cask codex"
            )

        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError(
                f"{self.api_key_name} must be set. "
                f"Set it with: export {self.api_key_name}='your-api-key'\n"
                "Get your API key from: https://platform.openai.com/api-keys"
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
        Generate a solution using Codex CLI.

        Args:
            problem: The mathematical problem statement

        Returns:
            SolverResponse with the generated solution
        """
        prompt = self._format_prompt(problem)

        cmd: list[str] = [
            self.binary,
            "exec",
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
        ]

        if self.model:
            cmd.extend(["--model", self.model])

        cmd.extend(["--json", "--", prompt])

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
                f"Codex CLI timed out after {self.timeout}s. "
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
                    f"Codex CLI authentication failed: {error_msg}\n"
                    f"Please set {self.api_key_name} environment variable:\n"
                    f"  export {self.api_key_name}='your-api-key'\n"
                    "Get your API key from: https://platform.openai.com/api-keys"
                )
            raise RuntimeError(f"Codex CLI failed: {error_msg}")

        if not stdout:
            raise RuntimeError("Codex CLI returned empty response")

        # Parse JSONL output (newline-delimited JSON)
        solution = None
        last_agent_message = None
        usage_info = None

        lines = stdout.split("\n")

        # First pass: collect agent messages
        for line in lines:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                event_type = event.get("type", "")

                # Look for agent_message items
                if event_type == "item.completed":
                    item = event.get("item", {})
                    if item.get("type") == "agent_message":
                        text = item.get("text", "")
                        if text:
                            last_agent_message = text
            except json.JSONDecodeError:
                continue

        # Second pass: find usage information
        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
                if isinstance(parsed, dict) and "usage" in parsed:
                    usage_info = parsed["usage"]
                    break
            except json.JSONDecodeError:
                continue

        # Use the last agent message, or fallback to raw stdout
        solution = last_agent_message or stdout

        metadata = {
            "solver": "codex_cli",
            "model": self.model or "default",
            "returncode": completed.returncode,
            "stderr": stderr,
            "raw_stdout": stdout,
        }

        # Add usage info if found
        if usage_info:
            metadata["usage"] = usage_info
            metadata["input_tokens"] = usage_info.get("input_tokens")
            metadata["cached_input_tokens"] = usage_info.get("cached_input_tokens")
            metadata["output_tokens"] = usage_info.get("output_tokens")

        return SolverResponse(solution=solution, metadata=metadata)
