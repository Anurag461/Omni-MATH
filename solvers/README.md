# Omni-MATH Solvers

This directory contains solver implementations for the Omni-MATH benchmark. Solvers generate mathematical solutions from problem statements and can be based on CLI agents, APIs, or local models.

## Architecture

The solver architecture follows the **factory pattern** used in OpenAI's simple-evals:

- **`SolverBase`**: Abstract base class defining the solver interface
- **Solver Implementations**: Concrete implementations for different backends
- **Factory Pattern**: Solvers are registered as factory functions in `generate_solutions.py`

## Available Solvers

### CLI-based Solvers

#### Codex CLI (`CodexCLISolver`)
- Uses OpenAI's Codex CLI tool
- Requires: `npm install -g @openai/codex` or `brew install --cask codex`
- Environment: `OPENAI_API_KEY`
- Models: default, gpt-5.1, gpt-5.1-codex-max, gpt-5.1-codex-mini

#### Claude CLI (`ClaudeCLISolver`)
- Uses Anthropic's Claude Code CLI
- Requires: Claude CLI installation
- Environment: `ANTHROPIC_API_KEY`
- Models: sonnet, opus, haiku

### API-based Solvers

#### OpenAI API (`OpenAISolver`)
- Uses OpenAI Python SDK
- Requires: `pip install openai`
- Environment: `OPENAI_API_KEY`
- Models: gpt-4, gpt-4-turbo, gpt-4o, o1-preview, o1-mini

## Usage

### Basic Usage

```bash
# Generate solutions using Codex CLI
python generate_solutions.py --solver codex-cli --output codex_results.jsonl

# Generate solutions using Claude Opus
python generate_solutions.py --solver claude-opus --limit 10

# Generate solutions using GPT-4
python generate_solutions.py --solver gpt-4
```

### Advanced Usage

```bash
# Test on first 5 problems
python generate_solutions.py --solver codex-cli --limit 5

# Resume from problem 100
python generate_solutions.py --solver claude-sonnet --skip 100

# Use custom input file
python generate_solutions.py --solver gpt-4 --input custom_problems.jsonl
```

## Adding New Solvers

To add a new solver:

1. **Create solver class** inheriting from `SolverBase`:

```python
from .base import SolverBase, SolverResponse

class MySolver(SolverBase):
    def __init__(self, model: str = "default"):
        self.model = model
        # Initialize your solver

    def __call__(self, problem: str) -> SolverResponse:
        # Generate solution
        solution = self._solve(problem)

        metadata = {
            "solver": "my_solver",
            "model": self.model,
        }

        return SolverResponse(solution=solution, metadata=metadata)
```

2. **Register in `__init__.py`**:

```python
from .my_solver import MySolver

__all__ = [..., "MySolver"]
```

3. **Add factory in `generate_solutions.py`**:

```python
def build_solvers():
    return {
        ...
        "my-solver": lambda: MySolver(),
        "my-solver-pro": lambda: MySolver(model="pro"),
    }
```

## Integration with Evaluation Pipeline

After generating solutions, evaluate them using the existing Omni-Judge or GPT evaluation pipeline:

```bash
# 1. Generate solutions
python generate_solutions.py --solver codex-cli --output codex_results.jsonl

# 2. Evaluate with Omni-Judge
cd Omni-Judge_eval
python omni_judge.py -i ../codex_results.jsonl -m <model_path> -o codex_evaluated.jsonl

# 3. Get accuracy results
python get_result.py -i codex_evaluated.jsonl
```

## Design Principles

1. **Unified Interface**: All solvers implement the same `__call__(problem) -> SolverResponse` interface
2. **Lazy Initialization**: Solvers are created via factory functions for flexible configuration
3. **Error Handling**: Graceful error handling with informative messages
4. **Metadata Tracking**: Solutions include metadata (model, tokens, etc.) for analysis
5. **Interchangeable**: Easy to swap solvers without changing evaluation code

## Comparison with simple-evals

This architecture mirrors OpenAI's simple-evals design:

| Aspect | simple-evals | Omni-MATH |
|--------|--------------|-----------|
| Base class | `SamplerBase` | `SolverBase` |
| Response | `SamplerResponse` | `SolverResponse` |
| Input | `MessageList` | `str` (problem) |
| CLI agents | Codex, Gemini | Codex, Claude |
| Registration | Factory in eval script | Factory in `generate_solutions.py` |

Key difference: Omni-MATH uses a **two-stage pipeline** (solution generation → evaluation), while simple-evals does evaluation in one stage.
