"""
LCB prompt builder for the release runner.

Only ``build_lcb_prompt`` is used, by ``benchmarks/run_lcb_canonical.py``.
It reproduces the CodeQwenInstruct prompt that
``lcb_runner.prompts.code_generation.format_prompt_generation`` builds for
LiveCodeBench release_v6, with an optional closed think block.
"""
from __future__ import annotations

from typing import Any


def build_lcb_prompt(problem: dict[str, Any], *, closed_think: bool = True) -> str:
    """Build the CodeQwenInstruct prompt for one LCB problem.

    ``closed_think=True`` appends an empty think block after the assistant
    tag so the model starts directly with code.
    """
    starter = problem.get("starter_code", "") or ""
    content = problem.get("question_content", "")
    prompt = (
        "You will be given a question (problem specification) and will "
        "generate a correct Python program that matches the specification "
        "and passes all tests. You will NOT return anything except for the program.\n\n"
        f"Question: {content}\n\n"
    )
    if starter.strip():
        prompt += "You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n"
        prompt += f"```python\n{starter}\n```\n\n<|im_end|>\n"
    else:
        prompt += "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    if closed_think:
        prompt += "<think>\n</think>\n\n"
    return prompt
