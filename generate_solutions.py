#!/usr/bin/env python3
"""
Generate solutions for Omni-MATH problems using various solvers.

This script takes problems from Omni-Math.jsonl and generates solutions using
the specified solver (Codex CLI, Claude CLI, OpenAI API, etc.).

The output is a JSONL file with the same format as the input, but with an
additional 'model_generation' field containing the generated solution.

Usage:
    python generate_solutions.py --solver codex-cli --output results.jsonl
    python generate_solutions.py --solver claude-cli --model opus --limit 10
    python generate_solutions.py --solver openai --model gpt-4 --input subset.jsonl
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

from solvers import (
    SolverBase,
    CodexCLISolver,
    ClaudeCLISolver,
    OpenAISolver,
)


def build_solvers(timeout: float = 15.0) -> dict[str, Any]:
    """
    Define all available solvers as factory functions.

    Args:
        timeout: Maximum time in seconds for each solver execution

    Returns:
        Dictionary mapping solver names to factory functions
    """
    return {
        # Codex CLI solvers
        "codex-cli": lambda: CodexCLISolver(timeout=timeout),
        "codex-gpt-5.1": lambda: CodexCLISolver(model="gpt-5.1", timeout=timeout),
        "codex-gpt-5.1-max": lambda: CodexCLISolver(model="gpt-5.1-codex-max", timeout=timeout),
        "codex-gpt-5.1-mini": lambda: CodexCLISolver(model="gpt-5.1-codex-mini", timeout=timeout),

        # Claude CLI solvers
        "claude-cli": lambda: ClaudeCLISolver(timeout=timeout),
        "claude-sonnet": lambda: ClaudeCLISolver(model="sonnet", timeout=timeout),
        "claude-opus": lambda: ClaudeCLISolver(model="opus", timeout=timeout),
        "claude-haiku": lambda: ClaudeCLISolver(model="haiku", timeout=timeout),

        # OpenAI API solvers
        "openai": lambda: OpenAISolver(timeout=timeout),
        "gpt-4": lambda: OpenAISolver(model="gpt-4", timeout=timeout),
        "gpt-4-turbo": lambda: OpenAISolver(model="gpt-4-turbo", timeout=timeout),
        "gpt-4o": lambda: OpenAISolver(model="gpt-4o", timeout=timeout),
        "gpt-5-mini-2025-08-07": lambda: OpenAISolver(model="gpt-5-mini-2025-08-07", temperature=1.0, timeout=timeout),
        "o1-preview": lambda: OpenAISolver(model="o1-preview", timeout=timeout),
        "o1-mini": lambda: OpenAISolver(model="o1-mini", timeout=timeout),
    }


def load_problems(input_file: Path) -> list[dict[str, Any]]:
    """
    Load problems from JSONL file.

    Args:
        input_file: Path to input JSONL file

    Returns:
        List of problem dictionaries
    """
    problems = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line))
    return problems


def save_result(output_file: Path, result: dict[str, Any]) -> None:
    """
    Append a single result to the output JSONL file.

    Args:
        output_file: Path to output file
        result: Result dictionary to write
    """
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def solve_single_problem(problem_data: dict[str, Any], solver: SolverBase) -> dict[str, Any]:
    """
    Solve a single problem (worker function for parallel execution).

    Args:
        problem_data: Problem dictionary
        solver: Solver instance to use

    Returns:
        Result dictionary with solution or error
    """
    problem = problem_data["problem"]

    try:
        # Generate solution
        response = solver(problem)

        # Create result with all original fields plus model_generation
        result = {
            **problem_data,
            "model_generation": response.solution,
            "solver_metadata": response.metadata,
        }
        return result

    except Exception as e:
        # Return error result
        result = {
            **problem_data,
            "model_generation": f"ERROR: {str(e)}",
            "solver_metadata": {"error": str(e)},
        }
        return result


def generate_solutions(
    input_file: Path,
    output_file: Path,
    solver: SolverBase,
    limit: int | None = None,
    skip: int = 0,
    parallel: int = 16,
) -> None:
    """
    Generate solutions for problems using the specified solver.

    Args:
        input_file: Path to input JSONL file with problems
        output_file: Path to output JSONL file for results
        solver: Solver instance to use
        limit: Maximum number of problems to solve (None for all)
        skip: Number of problems to skip at the beginning
        parallel: Number of parallel threads (default: 16)
    """
    # Load problems
    problems = load_problems(input_file)
    print(f"Loaded {len(problems)} problems from {input_file}")

    # Apply skip and limit
    if skip > 0:
        problems = problems[skip:]
        print(f"Skipping first {skip} problems")

    if limit is not None:
        problems = problems[:limit]
        print(f"Limiting to {limit} problems")

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Clear output file if it exists
    if output_file.exists():
        output_file.unlink()

    # Generate solutions
    print(f"Generating solutions with {solver.__class__.__name__}...")
    print(f"Using {parallel} parallel thread{'s' if parallel > 1 else ''}")

    if parallel == 1:
        # Sequential execution (original behavior)
        for problem_data in tqdm(problems, desc="Solving problems"):
            result = solve_single_problem(problem_data, solver)
            save_result(output_file, result)
    else:
        # Parallel execution with ordered output
        results = [None] * len(problems)  # Pre-allocate results list

        with ThreadPoolExecutor(max_workers=parallel) as executor:
            # Submit all tasks with their index
            future_to_idx = {
                executor.submit(solve_single_problem, problem_data, solver): idx
                for idx, problem_data in enumerate(problems)
            }

            # Collect results as they complete
            for future in tqdm(as_completed(future_to_idx), total=len(problems), desc="Solving problems"):
                idx = future_to_idx[future]
                results[idx] = future.result()

        # Write results in order
        for result in results:
            save_result(output_file, result)

    print(f"\nResults saved to {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate solutions for Omni-MATH problems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate solutions using Codex CLI for first 10 problems
  python generate_solutions.py --solver codex-cli --limit 10

  # Generate solutions using Claude Opus
  python generate_solutions.py --solver claude-opus --output claude_opus_results.jsonl

  # Generate solutions using OpenAI GPT-4
  python generate_solutions.py --solver gpt-4 --input custom_problems.jsonl

  # Resume from problem 100 onwards
  python generate_solutions.py --solver codex-cli --skip 100

Available solvers:
  Codex CLI: codex-cli, codex-gpt-5.1, codex-gpt-5.1-max, codex-gpt-5.1-mini
  Claude CLI: claude-cli, claude-sonnet, claude-opus, claude-haiku
  OpenAI API: openai, gpt-4, gpt-4-turbo, gpt-4o, o1-preview, o1-mini
        """,
    )

    parser.add_argument(
        "-s",
        "--solver",
        type=str,
        required=True,
        help="Solver to use (e.g., codex-cli, claude-opus, gpt-4)",
    )

    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path(__file__).parent / "Omni-Math.jsonl",
        help="Input JSONL file with problems (default: Omni-Math.jsonl)",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSONL file for results (default: <solver>_results.jsonl)",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of problems to solve (default: all)",
    )

    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Number of problems to skip at the beginning (default: 0)",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="Timeout in seconds for each solver execution (default: 15.0)",
    )

    parser.add_argument(
        "--parallel",
        type=int,
        default=16,
        help="Number of parallel threads to use (default: 16)",
    )

    args = parser.parse_args()

    # Build solvers with timeout
    solvers = build_solvers(timeout=args.timeout)

    # Get solver factory
    if args.solver not in solvers:
        print(f"Error: Unknown solver '{args.solver}'")
        print(f"\nAvailable solvers:")
        for name in sorted(solvers.keys()):
            print(f"  - {name}")
        return 1

    # Create solver instance
    try:
        solver = solvers[args.solver]()
    except Exception as e:
        print(f"Error initializing solver: {e}")
        return 1

    # Determine output file
    if args.output is None:
        output_file = Path(__file__).parent / f"{args.solver}_results.jsonl"
    else:
        output_file = args.output

    # Generate solutions
    try:
        generate_solutions(
            input_file=args.input,
            output_file=output_file,
            solver=solver,
            limit=args.limit,
            skip=args.skip,
            parallel=args.parallel,
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
