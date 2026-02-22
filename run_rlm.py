# save as run_rlm.py in repo root
import asyncio
from backend.llm import OllamaClient
from backend.rlm import RLMEngine

async def main():
    llm = OllamaClient()
    engine = RLMEngine(llm_client=llm, max_iterations=5)

    query = "What is this document about?"
    context = "Your document text goes here..."

    result = await engine.run(query, context)
    print(result.final_answer)

asyncio.run(main())