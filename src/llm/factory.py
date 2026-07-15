from src.config import CONFIG
from src.llm.anthropic import AnthropicProvider
from src.llm.fallback import FallbackProvider
from src.llm.gemini import GeminiProvider
from src.llm.openai import OpenAIProvider

_PROVIDER_CLASSES = {
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "anthropic": AnthropicProvider,
}


def get_generator() -> FallbackProvider:
    """Generation may fall over (NFR-3) -- built from CONFIG.gen_chain."""
    chain = [_PROVIDER_CLASSES[name]() for name in CONFIG.gen_chain]
    return FallbackProvider(chain)


def get_judge() -> AnthropicProvider:
    """The judge is pinned and never falls over (NFR-2, NFR-3)."""
    return AnthropicProvider(model=CONFIG.judge_model)
