"""Small, framework-free retrieval augmented generation package."""

from .generation import ExtractiveGenerator, OllamaGenerator, OpenAIGenerator
from .pipeline import RAGPipeline

__all__ = [
    "ExtractiveGenerator",
    "OllamaGenerator",
    "OpenAIGenerator",
    "RAGPipeline",
]

