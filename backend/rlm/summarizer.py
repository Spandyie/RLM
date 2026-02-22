"""
Recursive Summarizer

For very large documents, this summarizer:
1. Splits into chunks
2. Summarizes each chunk
3. Combines summaries
4. Repeats until one summary remains

This is a simpler pattern that doesn't use the full REPL,
but shows the recursive decomposition idea from RLM.
"""

from dataclasses import dataclass


@dataclass
class SummaryResult:
    """Result of summarization."""
    summary: str
    chunk_count: int
    levels: int  # How many recursive levels


class RecursiveSummarizer:
    """
    Summarize large documents by recursive chunking.

    Usage:
        summarizer = RecursiveSummarizer(llm_client)
        result = await summarizer.summarize(big_document)
        print(result.summary)
    """

    def __init__(self, llm_client, chunk_size: int = 2000):
        self.llm = llm_client
        self.chunk_size = chunk_size

    async def summarize(self, text: str) -> SummaryResult:
        """Recursively summarize a document."""

        # Split into chunks
        chunks = self._split(text)

        if not chunks:
            return SummaryResult("Empty document.", 0, 0)

        # Summarize each chunk
        summaries = []
        for chunk in chunks:
            summary = await self.llm.generate(
                f"Summarize this in 2-3 sentences:\n\n{chunk}"
            )
            summaries.append(summary.strip())

        # Recursively combine until we have one summary
        level = 1
        while len(summaries) > 1:
            level += 1
            combined = []
            for i in range(0, len(summaries), 3):
                group = summaries[i:i+3]
                text = "\n\n".join(group)
                merged = await self.llm.generate(
                    f"Combine these summaries into one:\n\n{text}"
                )
                combined.append(merged.strip())
            summaries = combined

        return SummaryResult(
            summary=summaries[0],
            chunk_count=len(chunks),
            levels=level,
        )

    def _split(self, text: str) -> list[str]:
        """Split text into chunks."""
        chunks = []
        paragraphs = text.split("\n\n")

        current = ""
        for para in paragraphs:
            if len(current) + len(para) > self.chunk_size:
                if current:
                    chunks.append(current.strip())
                current = para
            else:
                current = current + "\n\n" + para if current else para

        if current.strip():
            chunks.append(current.strip())

        return chunks
