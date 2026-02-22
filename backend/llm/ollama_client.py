"""
Ollama Client - Local LLM interface.

Connects to Ollama for text generation.
"""

import httpx
from typing import Optional, List, AsyncIterator


class OllamaClient:
    """Simple async client for Ollama."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1:8b",
        timeout: int = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
    ) -> str:
        """Generate text from a prompt."""
        model = model or self.model

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature},
                },
            )
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                raise ValueError(f"Ollama error: {data['error']}")

            return data.get("response", "")

    async def check_health(self) -> bool:
        """Check if Ollama is running."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False

    async def check_model_available(self, model: Optional[str] = None) -> bool:
        """Check if model is available."""
        model = model or self.model
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code != 200:
                    return False

                data = response.json()
                models = data.get("models", [])

                for m in models:
                    if m.get("name", "").startswith(model.split(":")[0]):
                        return True
                return False
        except Exception:
            return False

    async def list_models(self) -> List[dict]:
        """List available models."""
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            return response.json().get("models", [])
