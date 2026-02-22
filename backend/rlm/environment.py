"""
RLM REPL Environment

This is where the LLM's code runs. It provides:
- context: The document text as a variable
- query: The user's question
- llm_query(prompt): Make a sub-LLM call
- FINAL(answer): Return the final answer

Think of it as a Python sandbox with special functions.
"""

import re
import io
import sys
from contextlib import redirect_stdout, redirect_stderr


class FinalAnswer(Exception):
    """Raised when FINAL() is called to signal we have the answer."""
    def __init__(self, answer: str):
        self.answer = answer


class RLMEnvironment:
    """
    A simple Python environment for RLM code execution.

    Example of what runs here:
        print(len(context))           # See context size
        print(context[:500])          # Peek at start
        chunks = context.split('\\n')  # Split into lines
        result = llm_query("summarize this")  # Sub-LLM call
        FINAL("The answer is...")     # Return answer
    """

    def __init__(self, context: str, query: str, llm_callback):
        """
        Set up the environment.

        Args:
            context: The document text (stored as `context` variable)
            query: User's question (stored as `query` variable)
            llm_callback: Function to call for llm_query(prompt)
        """
        self.context = context
        self.query = query
        self.llm_callback = llm_callback
        self.variables = {}  # User-created variables persist here

    def execute(self, code: str) -> str:
        """
        Run Python code and return the output.

        Args:
            code: Python code to execute

        Returns:
            Whatever was printed (stdout)

        Raises:
            FinalAnswer: When FINAL() is called
        """
        # Capture print output
        stdout = io.StringIO()

        # Build the namespace (what variables are available)
        namespace = {
            # The main variables
            'context': self.context,
            'query': self.query,

            # Special RLM functions
            'llm_query': self._llm_query,
            'FINAL': self._final,
            'FINAL_VAR': self._final_var,

            # Useful for text processing
            're': re,

            # Standard builtins
            'print': lambda *args: print(*args, file=stdout),
            'len': len,
            'str': str,
            'int': int,
            'list': list,
            'dict': dict,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'sorted': sorted,
            'min': min,
            'max': max,
            'sum': sum,
            'True': True,
            'False': False,
            'None': None,
        }

        # Add any variables from previous executions
        namespace.update(self.variables)

        try:
            # Run the code
            exec(code, namespace)

            # Save any new variables for next execution
            self._save_variables(namespace)

            return stdout.getvalue()

        except FinalAnswer:
            raise  # Let this propagate up

        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

    def _llm_query(self, prompt: str) -> str:
        """
        Make a recursive LLM call.
        This is available as llm_query() in the code.
        """
        return self.llm_callback(prompt)

    def _final(self, answer) -> None:
        """
        Return the final answer.
        This is available as FINAL() in the code.
        """
        raise FinalAnswer(str(answer))

    def _final_var(self, var_name: str) -> None:
        """
        Return a variable's value as the answer.
        This is available as FINAL_VAR() in the code.
        """
        if var_name in self.variables:
            raise FinalAnswer(str(self.variables[var_name]))
        else:
            raise FinalAnswer(f"Error: Variable '{var_name}' not found")

    def _save_variables(self, namespace: dict):
        """Save user-created variables for persistence."""
        skip = {'context', 'query', 'llm_query', 'FINAL', 'FINAL_VAR',
                're', 'print', 'len', 'str', 'int', 'list', 'dict',
                'range', 'enumerate', 'zip', 'sorted', 'min', 'max', 'sum',
                'True', 'False', 'None', '__builtins__'}

        for name, value in namespace.items():
            if name not in skip and not name.startswith('_'):
                self.variables[name] = value
