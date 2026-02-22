"""
Configuration for the RLM Document Chatbot.

Based on "Recursive Language Models" by Zhang, Kraska, and Khattab (2025)
https://arxiv.org/abs/2512.24601

This module contains all configurable parameters for the application.
"""

from pydantic import BaseModel


class OllamaConfig(BaseModel):
    """Configuration for Ollama LLM."""

    base_url: str = "http://localhost:11434"
    model: str = "llama3.1:8b"
    timeout: int = 120  # seconds


class RLMConfig(BaseModel):
    """
    Configuration for the Recursive Language Model engine.

    RLM Parameters:
    - max_iterations: Maximum code execution loops before forcing FINAL()
    - max_depth: Maximum recursion depth for llm_query() calls
    - max_output_length: Truncate REPL output beyond this length
    """

    max_iterations: int = 10  # Max code execution loops
    max_depth: int = 3        # Max recursion depth
    max_output_length: int = 10000  # Truncate output beyond this


class DocumentConfig(BaseModel):
    """Configuration for document processing."""

    chunk_size: int = 1000  # Characters per chunk
    chunk_overlap: int = 200  # Overlap between chunks for context
    max_chunks_per_query: int = 5  # Max chunks to retrieve for a query


class ChromaConfig(BaseModel):
    """Configuration for ChromaDB vector store."""

    persist_directory: str = "./data/chroma"
    collection_name: str = "documents"


class Settings(BaseModel):
    """Main settings container."""

    ollama: OllamaConfig = OllamaConfig()
    rlm: RLMConfig = RLMConfig()
    document: DocumentConfig = DocumentConfig()
    chroma: ChromaConfig = ChromaConfig()

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Debug mode
    debug: bool = True


# Global settings instance
settings = Settings()
