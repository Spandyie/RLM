"""
RLM Base Classes

Simple data structures for tracking RLM execution.
Based on "Recursive Language Models" by Zhang et al. (2025)
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RLMStep:
    """
    One step in the RLM execution.

    Types:
    - "code": Python code the LLM wrote
    - "output": Result from running that code
    - "llm_call": A recursive llm_query() call
    - "final": The final answer from FINAL()
    """
    step_type: str   # "code", "output", "llm_call", or "final"
    content: str     # The code, output, or answer
    depth: int = 0   # Recursion depth (0 = main, 1+ = sub-calls)

    def to_dict(self) -> dict:
        return {
            "step_type": self.step_type,
            "content": self.content,
            "depth": self.depth,
        }


@dataclass
class RLMResult:
    """
    Final result after RLM finishes.

    Contains the answer plus the full execution trace
    so you can see how the LLM reasoned through the problem.
    """
    query: str                      # Original question
    context_length: int             # How big was the context
    final_answer: str               # The answer from FINAL()
    steps: List[RLMStep]            # All execution steps
    total_llm_calls: int            # How many LLM calls were made
    success: bool = True            # Did it complete successfully?
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "context_length": self.context_length,
            "final_answer": self.final_answer,
            "steps": [s.to_dict() for s in self.steps],
            "total_llm_calls": self.total_llm_calls,
            "success": self.success,
            "error": self.error,
        }
