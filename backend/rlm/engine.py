"""
RLM Engine - The Main Loop

This is the heart of RLM. It runs this loop:

    1. Send prompt to LLM
    2. LLM writes Python code
    3. Execute code in REPL
    4. Send output back to LLM
    5. Repeat until FINAL() is called

The key insight: context is NOT in the prompt.
Instead, it's a variable the LLM accesses via code.
"""

import re
from typing import Optional
from .base import RLMStep, RLMResult
from .environment import RLMEnvironment, FinalAnswer


# This prompt teaches the LLM how to use the REPL
SYSTEM_PROMPT = '''You have access to a Python REPL to answer questions about documents.

## Available Variables
- `context` - The document text (may be very large)
- `query` - The user's question

## Available Functions
- `print(...)` - See output
- `llm_query(prompt)` - Ask a sub-question to another LLM
- `FINAL(answer)` - Return your final answer (REQUIRED when done)
- `re` - Regex module for searching

## Strategy
1. First, check the context size: `print(len(context))`
2. Peek at the content: `print(context[:1000])`
3. Search if needed: `print(re.findall(r'pattern', context))`
4. For complex tasks, use llm_query() on chunks
5. Call FINAL("your answer") when done

## Rules
- Write Python code in ```python blocks
- Always end with FINAL() when you have the answer
- Don't try to read all context at once if it's large

## Your Task
Context length: {context_len} characters
Question: {query}

Start by examining the context, then answer the question.
'''


class RLMEngine:
    """
    The main RLM engine that coordinates everything.

    Usage:
        engine = RLMEngine(llm_client)
        result = await engine.run("What is this about?", document_text)
        print(result.final_answer)
    """

    def __init__(self, llm_client, max_iterations: int = 10):
        """
        Args:
            llm_client: The Ollama client for LLM calls
            max_iterations: Max code execution loops (safety limit)
        """
        self.llm = llm_client
        self.max_iterations = max_iterations

    async def run(self, query: str, context: str) -> RLMResult:
        """
        Run RLM to answer a question about the context.

        Args:
            query: The user's question
            context: The document text

        Returns:
            RLMResult with the answer and execution trace
        """
        steps = []
        llm_calls = 0

        # Create the REPL environment
        # The llm_callback handles recursive llm_query() calls
        async def llm_callback(prompt: str) -> str:
            nonlocal llm_calls
            llm_calls += 1
            steps.append(RLMStep("llm_call", prompt[:200] + "..." if len(prompt) > 200 else prompt))
            return await self.llm.generate("Answer concisely: " + prompt)

        # Sync wrapper (needed because exec() is synchronous)
        def sync_callback(prompt: str) -> str:
            import asyncio
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(llm_callback(prompt))

        env = RLMEnvironment(context, query, sync_callback)

        # Build the initial prompt
        prompt = SYSTEM_PROMPT.format(
            context_len=len(context),
            query=query
        )

        try:
            # === THE MAIN RLM LOOP ===
            for iteration in range(self.max_iterations):

                # Step 1: Ask LLM for code
                llm_calls += 1
                response = await self.llm.generate(prompt)

                # Step 2: Extract code from response
                code = self._extract_code(response)

                if not code:
                    # No code found - maybe LLM gave direct answer
                    # Try to use it as the final answer
                    steps.append(RLMStep("final", response))
                    return RLMResult(
                        query=query,
                        context_length=len(context),
                        final_answer=response,
                        steps=steps,
                        total_llm_calls=llm_calls,
                    )

                # Record the code step
                steps.append(RLMStep("code", code))

                # Step 3: Execute the code
                try:
                    output = env.execute(code)
                    steps.append(RLMStep("output", output))

                except FinalAnswer as fa:
                    # FINAL() was called - we have our answer!
                    steps.append(RLMStep("final", fa.answer))
                    return RLMResult(
                        query=query,
                        context_length=len(context),
                        final_answer=fa.answer,
                        steps=steps,
                        total_llm_calls=llm_calls,
                    )

                # Step 4: Build prompt for next iteration
                prompt = f"""Previous code output:
```
{output if output else "(no output)"}
```

Continue analyzing. Remember to call FINAL(answer) when you have the answer.
Write your next code block:"""

            # Max iterations reached
            return RLMResult(
                query=query,
                context_length=len(context),
                final_answer="[Reached max iterations without final answer]",
                steps=steps,
                total_llm_calls=llm_calls,
                success=False,
                error="Max iterations reached",
            )

        except Exception as e:
            return RLMResult(
                query=query,
                context_length=len(context),
                final_answer="[Error: {}]".format(e),
                steps=steps,
                total_llm_calls=llm_calls,
                success=False,
                error=str(e),
            )

    def _extract_code(self, text: str) -> Optional[str]:
        """
        Extract Python code from LLM response.

        Looks for ```python ... ``` blocks.
        """
        # Try ```python ... ```
        match = re.search(r'```python\s*(.*?)```', text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try ``` ... ``` (no language specified)
        match = re.search(r'```\s*(.*?)```', text, re.DOTALL)
        if match:
            code = match.group(1).strip()
            # Basic check that it looks like Python
            if any(kw in code for kw in ['print', 'for', 'if', 'def', '=', 'FINAL']):
                return code

        return None
