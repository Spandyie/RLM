"""
RLM (Recursive Language Model) Package

Based on "Recursive Language Models" by Zhang, Kraska, and Khattab (2025)
https://arxiv.org/abs/2512.24601

## How RLM Works

Instead of putting all context in the LLM prompt, RLM:
1. Stores context as a `context` variable in a Python REPL
2. LLM writes code to examine/search the context
3. LLM can call llm_query() for recursive sub-tasks
4. LLM calls FINAL() when it has the answer

## Example Flow

    User: "What are the main themes?"

    LLM writes:
        ```python
        print(len(context))
        print(context[:1000])
        ```

    Output: 50000
            [first 1000 chars...]

    LLM writes:
        ```python
        # Found it's about AI
        themes = []
        for section in context.split('##'):
            if section.strip():
                summary = llm_query(f"Theme of: {section[:500]}")
                themes.append(summary)
        FINAL("Main themes: " + ", ".join(themes))
        ```

    Result: "Main themes: AI safety, ethics, future predictions"

## Files

- base.py: Simple data classes (RLMStep, RLMResult)
- environment.py: Python REPL with context, llm_query, FINAL
- engine.py: Main loop that coordinates LLM and REPL
"""

from .base import RLMStep, RLMResult
from .environment import RLMEnvironment, FinalAnswer
from .engine import RLMEngine
from .summarizer import RecursiveSummarizer

__all__ = [
    "RLMStep",
    "RLMResult",
    "RLMEnvironment",
    "FinalAnswer",
    "RLMEngine",
    "RecursiveSummarizer",
]
