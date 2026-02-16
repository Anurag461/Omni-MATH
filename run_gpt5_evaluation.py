#!/usr/bin/env python3
"""
Run GPT-5 nano evaluation on generated solutions.
Uses the exact same template as the original GPT evaluation.
"""

import json
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm


def load_template():
    """Load the GPT evaluation template."""
    template_path = Path(__file__).parent / "GPT_eval" / "gpt_evaluation_template.txt"
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()


def format_prompt(template: str, problem: str, reference_answer: str, solution: str) -> str:
    """Format the template with problem data."""
    prompt = template.replace("{{Problem}}", problem)
    prompt = prompt.replace("{{Reference Answer}}", reference_answer)
    prompt = prompt.replace("{{Solution}}", solution)
    return prompt


def call_gpt5_nano(prompt: str) -> str:
    """Call GPT-5 nano to evaluate the solution."""
    try:
        from openai import OpenAI
        client = OpenAI(timeout=30.0)
    except ImportError:
        raise ImportError("OpenAI Python SDK required. Install with: pip install openai")

    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set")

    try:
        response = client.chat.completions.create(
            model="gpt-5-nano-2025-08-07",
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,  # GPT-5 requires temperature=1.0
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"ERROR: {str(e)}"


def evaluate_single_solution(solution_data: dict, template: str) -> dict:
    """
    Evaluate a single solution (worker function for parallel execution).

    Args:
        solution_data: Solution dictionary
        template: Evaluation prompt template

    Returns:
        Result dictionary with evaluation
    """
    problem = solution_data["problem"]
    reference_answer = solution_data["answer"]
    student_solution = solution_data.get("model_generation", "")

    # Format prompt using template
    prompt = format_prompt(template, problem, reference_answer, student_solution)

    # Call GPT-5 nano
    evaluation = call_gpt5_nano(prompt)

    # Return in the format expected by get_result.py
    return {
        "original_json": json.dumps(solution_data, ensure_ascii=False),
        "gen": evaluation
    }


def main():
    parser = argparse.ArgumentParser(description="Run GPT-5 nano evaluation")
    parser.add_argument("-i", "--in-file", type=str, required=True, help="Input JSONL file with solutions")
    parser.add_argument("-o", "--out-file", type=str, required=True, help="Output JSONL file with evaluations")
    parser.add_argument(
        "--parallel",
        type=int,
        default=16,
        help="Number of parallel threads to use (default: 16)",
    )
    args = parser.parse_args()

    # Load template
    template = load_template()

    # Load solutions
    solutions = []
    with open(args.in_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                solutions.append(json.loads(line))

    print(f"Loaded {len(solutions)} solutions")
    print(f"Evaluating with gpt-5-nano-2025-08-07...")
    print(f"Using {args.parallel} parallel thread{'s' if args.parallel > 1 else ''}")

    # Evaluate solutions
    if args.parallel == 1:
        # Sequential execution
        results = []
        for solution_data in tqdm(solutions, desc="Evaluating"):
            result = evaluate_single_solution(solution_data, template)
            results.append(result)
    else:
        # Parallel execution with ordered output
        results = [None] * len(solutions)  # Pre-allocate results list

        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            # Submit all tasks with their index
            future_to_idx = {
                executor.submit(evaluate_single_solution, solution_data, template): idx
                for idx, solution_data in enumerate(solutions)
            }

            # Collect results as they complete
            for future in tqdm(as_completed(future_to_idx), total=len(solutions), desc="Evaluating"):
                idx = future_to_idx[future]
                results[idx] = future.result()

    # Write output
    with open(args.out_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"\nEvaluations saved to: {args.out_file}")
    print(f"\nNow run: cd GPT_eval && python get_result.py -i ../{args.out_file}")


if __name__ == "__main__":
    main()
