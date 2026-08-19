"""Model resolution for the Agno runtime.

Owns `get_model()` and the optional model-import shims. A provider is added by
registering a builder in `_MODEL_BUILDERS` (see `register_provider`), so new
providers do not need to touch `get_model` itself.

Environment variables read here:
  AGNO_MODEL_PROVIDER  which provider to use (default: gemini)
  AGNO_MODEL           model id for the selected provider
  GOOGLE_API_KEY / GEMINI_API_KEY / DEEPSEEK_API_KEY / OPENAI_API_KEY / VLLM_API_KEY
  VLLM_BASE_URL / OPENAI_BASE_URL
"""

import os
from typing import Callable, Dict, Optional

# Compatibility shim for the installed google-genai package: Agno expects `FileSearch` to exist,
# but the current package exposes it in a slightly different state depending on version.
try:
    import google.genai.types as google_types
    if not hasattr(google_types, "FileSearch"):
        class FileSearch:  # pragma: no cover - compatibility shim for current google-genai
            def __init__(self, **kwargs):
                self.kwargs = kwargs
        google_types.FileSearch = FileSearch
except Exception:
    google_types = None

try:
    from agno.models.google import Gemini
except Exception:
    Gemini = None

try:
    from agno.models.deepseek import DeepSeek
except Exception:
    DeepSeek = None

# Prefer Gemini when configured; otherwise fall back to OpenAI.
try:
    from agno.models.openai import OpenAIChat
except Exception:
    OpenAIChat = None


def _build_gemini(api_key: Optional[str], model_id: Optional[str], base_url: Optional[str]):
    api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY or GEMINI_API_KEY is not set. Export it before running a Gemini Agent call.")
    if Gemini is None:
        raise RuntimeError("Gemini support is unavailable in the current installation (google-genai / agno google integration).")
    model_id = model_id or os.getenv("AGNO_MODEL", "gemini-3.6-flash")
    return Gemini(id=model_id, api_key=api_key)


def _build_deepseek(api_key: Optional[str], model_id: Optional[str], base_url: Optional[str]):
    api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set. Export it before running a DeepSeek Agent call.")
    if DeepSeek is None:
        raise RuntimeError("DeepSeek support is unavailable in the current installation of Agno.")
    model_id = model_id or os.getenv("AGNO_MODEL", "deepseek-v4-flash")
    return DeepSeek(id=model_id, api_key=api_key)


def _build_openai_compatible(api_key: Optional[str], model_id: Optional[str], base_url: Optional[str]):
    base_url = base_url or os.getenv("VLLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    if not base_url:
        raise RuntimeError(
            "A base URL is required for the vllm/openai_compatible provider. "
            "Set VLLM_BASE_URL (e.g. http://localhost:8000/v1) or pass base_url explicitly."
        )
    if OpenAIChat is None:
        raise RuntimeError("OpenAI-compatible support is unavailable in the current installation of Agno.")
    api_key = api_key or os.getenv("VLLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY"
    model_id = model_id or os.getenv("AGNO_MODEL")
    if not model_id:
        raise RuntimeError("model_id (or AGNO_MODEL) is required for the vllm/openai_compatible provider — use the model name vLLM was served with, e.g. --served-model-name.")
    return OpenAIChat(id=model_id, api_key=api_key, base_url=base_url)


def _build_openai(api_key: Optional[str], model_id: Optional[str], base_url: Optional[str]):
    # Fallback to OpenAI (also honors OPENAI_BASE_URL for OpenAI-compatible proxies)
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Export it before running a real OpenAI Agent call.")
    if OpenAIChat is None:
        raise RuntimeError("OpenAI support is unavailable in the current installation of Agno.")
    model_id = model_id or os.getenv("AGNO_MODEL", "gpt-4o-mini")
    base_url = base_url or os.getenv("OPENAI_BASE_URL")
    return OpenAIChat(id=model_id, api_key=api_key, base_url=base_url)


# provider name (already lowercased/stripped) -> builder. Every alias gets its
# own entry rather than a nested alias table, so a lookup is one dict hit and
# adding a provider is one line. Anything not listed here falls through to
# _DEFAULT_MODEL_BUILDER (OpenAI), which is what the original if/elif chain's
# trailing fallback did.
_MODEL_BUILDERS: Dict[str, Callable[..., object]] = {
    "gemini": _build_gemini,
    "google": _build_gemini,
    "deepseek": _build_deepseek,
    "vllm": _build_openai_compatible,
    "openai_compatible": _build_openai_compatible,
    "custom": _build_openai_compatible,
}
_DEFAULT_MODEL_BUILDER = _build_openai


def register_provider(name: str, builder: Callable[..., object]) -> None:
    """Register (or override) a provider builder under `name`.

    A builder is called as builder(api_key=..., model_id=..., base_url=...) and
    returns a configured Agno model instance; it owns its own env-var fallbacks,
    its own "not installed" error message, and its own default model id.
    """
    _MODEL_BUILDERS[name.strip().lower()] = builder


def get_model(
    api_key: Optional[str] = None,
    provider: Optional[str] = None,
    model_id: Optional[str] = None,
    base_url: Optional[str] = None,
):
    """Resolve and return a configured model instance.

    `provider`/`model_id`/`base_url`/`api_key` let a caller (e.g. a per-request
    override from the panel's settings UI) pick a model without editing .env;
    each falls back to the matching environment variable when omitted.

    Priority:
    - 'gemini'/'google': Gemini via GOOGLE_API_KEY or GEMINI_API_KEY.
    - 'deepseek': DeepSeek via DEEPSEEK_API_KEY.
    - 'vllm'/'openai_compatible'/'custom': any OpenAI-compatible server (vLLM,
      LM Studio, Ollama's OpenAI shim, etc.) via VLLM_BASE_URL/OPENAI_BASE_URL.
      vLLM's OpenAI-compatible server typically does not check the API key, so
      any non-empty placeholder ("EMPTY") is accepted if one isn't configured.
    - otherwise: OpenAI via OPENAI_API_KEY (also honors OPENAI_BASE_URL if set,
      so it can point at an OpenAI-compatible proxy without the 'vllm' provider).
    """
    provider = (provider or os.getenv("AGNO_MODEL_PROVIDER", "gemini")).strip().lower()
    builder = _MODEL_BUILDERS.get(provider, _DEFAULT_MODEL_BUILDER)
    return builder(api_key=api_key, model_id=model_id, base_url=base_url)
