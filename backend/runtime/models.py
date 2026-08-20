"""Model resolution for the Agno runtime.

Two parallel factories live here, one per backend, deliberately sharing the
same shape (registry + `register_*` + env-var fallbacks + per-provider error
messages) so that "add a provider" is one entry in each:

- `get_model()` builds an **Agno** model (`agno.models.*`) for the native
  backend, from `_MODEL_BUILDERS`.
- `get_langchain_model()` builds a **LangChain** chat model
  (`langchain_*.Chat*`) for the LangGraph/DeepAgents backend, from
  `_LANGCHAIN_MODEL_BUILDERS`. Before it existed, `runtime.graph.
  build_deepagents_graph` hardcoded `ChatDeepSeek`, so that whole backend was
  DeepSeek-only.

Every model package is imported lazily inside its builder (or guarded by a
try/except at module import for the Agno side), because all of them are
optional dependencies: importing this module must keep working with none of
them installed.

Thinking/reasoning is *also* a request-side concern, not only a parsing one
(see runtime/reasoning.py for the parsing half): several providers emit no
thinking at all unless it is asked for. `resolve_thinking()` turns the
AGNO_THINKING/AGNO_THINKING_BUDGET knobs (or a per-request override) into a
`ThinkingConfig`, and each provider has a small adapter translating that into
whatever that provider's client expects — see `_THINKING_ADAPTERS`.

Environment variables read here:
  AGNO_MODEL_PROVIDER      which provider the native Agno backend uses (default: gemini)
  AGNO_LANGGRAPH_PROVIDER  which provider the LangGraph/DeepAgents backend uses (default: deepseek)
  AGNO_MODEL               model id for the selected provider
  AGNO_LANGGRAPH_MODEL     model id override for the LangGraph/DeepAgents backend only
  AGNO_THINKING            request thinking/reasoning output: 1/0 (unset = provider default)
  AGNO_THINKING_BUDGET     thinking token budget, for providers that take one
  GOOGLE_API_KEY / GEMINI_API_KEY / DEEPSEEK_API_KEY / OPENAI_API_KEY /
  ANTHROPIC_API_KEY / VLLM_API_KEY
  VLLM_BASE_URL / OPENAI_BASE_URL
"""

import os
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Optional

from .reasoning import DEEPAGENTS_PROVIDER as DEFAULT_LANGCHAIN_PROVIDER

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

try:
    from agno.models.anthropic import Claude
except Exception:
    Claude = None

# Prefer Gemini when configured; otherwise fall back to OpenAI.
try:
    from agno.models.openai import OpenAIChat
except Exception:
    OpenAIChat = None


#: provider name -> default model id, shared by the Agno-side and LangChain-side
#: builders so both backends default to the same model for a given provider.
#: The openai-compatible providers are deliberately absent: their model id is
#: whatever the server was started with, so there is nothing to default to.
_PROVIDER_DEFAULT_MODEL: Dict[str, str] = {
    "gemini": "gemini-3.6-flash",
    "google": "gemini-3.6-flash",
    "deepseek": "deepseek-v4-flash",
    "openai": "gpt-4o-mini",
    #: Agno's own Claude class still defaults to a claude-4-5 id; this points at
    #: the current Sonnet generation instead. Claude's model ids are not dated
    #: aliases here on purpose — an undated id tracks the latest snapshot.
    "anthropic": "claude-sonnet-5",
    "claude": "claude-sonnet-5",
}


# ---------------------------------------------------------------------------
# Thinking / reasoning: the request side
# ---------------------------------------------------------------------------
#
# runtime/reasoning.py answers "where does this model put its thinking in the
# response?". This half answers the other question: "does the model emit any
# thinking at all unless asked?" — and for several providers the answer is no,
# with a different opt-in shape each. DeepSeek is the exception that made this
# easy to miss: it streams `reasoning_content` with no request-side flag, which
# is why the original hardcoded ChatDeepSeek harness needed none of this.

#: Env vars for the thinking knobs. Unset means "don't ask for anything", which
#: reproduces the behavior that existed before this module knew about thinking.
THINKING_ENV_VAR = "AGNO_THINKING"
THINKING_BUDGET_ENV_VAR = "AGNO_THINKING_BUDGET"

#: Values of AGNO_THINKING (or a per-request `thinking=`) that mean on/off.
_THINKING_TRUE = ("1", "true", "yes", "on", "enable", "enabled")
_THINKING_FALSE = ("0", "false", "no", "off", "disable", "disabled", "none")
#: Values that mean "on, at this effort level" — OpenAI-style reasoning effort.
_THINKING_EFFORTS = ("minimal", "low", "medium", "high")


class ThinkingConfig(NamedTuple):
    """A provider-neutral request for thinking output.

    `enabled` is deliberately tri-state: None means "nothing configured", and
    that is NOT the same as False. None sends no provider-specific argument at
    all (the provider's own default applies, which is what every call did
    before this existed); False actively asks the provider to turn thinking
    off, where that provider has a way to express it.
    """

    enabled: Optional[bool] = None
    budget: Optional[int] = None
    effort: Optional[str] = None

    @property
    def is_set(self) -> bool:
        """True when the caller/env said something about thinking at all."""
        return self.enabled is not None


#: The "nothing configured" config — no provider argument is sent for it.
UNSET_THINKING = ThinkingConfig()


def _parse_thinking_flag(value) -> ThinkingConfig:
    """Parse one AGNO_THINKING-shaped value into a ThinkingConfig.

    Accepts bools, the usual truthy/falsy strings, and an OpenAI-style effort
    level ("low"/"medium"/"high"/"minimal"), which implies enabled=True. An
    unrecognized value is treated as "unset" rather than raising: this arrives
    from an env var and from request payloads, and a typo there should degrade
    to today's behavior instead of failing a whole turn.
    """
    if value is None:
        return UNSET_THINKING
    if isinstance(value, bool):
        return ThinkingConfig(enabled=value)
    text = str(value).strip().lower()
    if not text:
        return UNSET_THINKING
    if text in _THINKING_TRUE:
        return ThinkingConfig(enabled=True)
    if text in _THINKING_FALSE:
        return ThinkingConfig(enabled=False)
    if text in _THINKING_EFFORTS:
        return ThinkingConfig(enabled=True, effort=text)
    return UNSET_THINKING


def _parse_thinking_budget(value) -> Optional[int]:
    """Parse a token budget; an unparseable value means "no budget given"."""
    if value is None or isinstance(value, bool):
        return None
    try:
        budget = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return budget if budget >= 0 else None


def resolve_thinking(thinking=None, thinking_budget=None) -> ThinkingConfig:
    """Resolve the thinking knobs, request value first, then env, then unset.

    Returns UNSET_THINKING when neither a caller value nor AGNO_THINKING is
    present, and callers must then send no thinking argument at all — that is
    the "no behavior change when nothing is configured" guarantee.
    """
    config = _parse_thinking_flag(thinking)
    if not config.is_set:
        config = _parse_thinking_flag(os.getenv(THINKING_ENV_VAR))
    budget = _parse_thinking_budget(thinking_budget)
    if budget is None:
        budget = _parse_thinking_budget(os.getenv(THINKING_BUDGET_ENV_VAR))
    if budget is None:
        return config
    return config._replace(budget=budget)


#: Fallback budget for providers whose "enabled" shape requires a number.
#: Anthropic in particular rejects an enabled-thinking request without one.
_DEFAULT_THINKING_BUDGET = 2048


def _effort_for_budget(config: ThinkingConfig) -> str:
    """OpenAI takes an effort level rather than a token budget; map one onto
    the other so a single AGNO_THINKING_BUDGET setting means something for
    every provider."""
    if config.effort:
        return config.effort
    if config.budget is None:
        return "medium"
    if config.budget <= 0:
        return "minimal"
    if config.budget <= 2048:
        return "low"
    if config.budget <= 8192:
        return "medium"
    return "high"


# -- per-provider thinking adapters ----------------------------------------
#
# UNVERIFIED, AND DELIBERATELY ISOLATED HERE. None of these client packages are
# installed in the environment this was written in, so the exact keyword each
# one accepts could not be executed and confirmed — only the *shape* below is
# designed to be safe. Every one of these dicts is passed through
# `_construct_with_optional_kwargs`, which retries construction WITHOUT them if
# the client rejects them, so a wrong name here degrades to "no thinking
# requested" instead of breaking a turn. Confirm against:
#   Anthropic: langchain_anthropic.ChatAnthropic  (`thinking=` / betas)
#   OpenAI:    langchain_openai.ChatOpenAI        (`reasoning_effort=`, and the
#              Responses API `reasoning={"effort":..,"summary":"auto"}` +
#              `use_responses_api=True` for *visible* summaries — plain Chat
#              Completions reasoning models think without emitting any text)
#   Gemini:    langchain_google_genai.ChatGoogleGenerativeAI
#              (`thinking_budget=` / `include_thoughts=`)
# The same adapters are used for both the Agno-side and LangChain-side
# factories because both wrappers name these after the provider's own API.
#
# VERIFIED for the Agno side of Anthropic, by reading the installed
# agno.models.anthropic.claude source: `Claude` really does declare
# `thinking: Optional[Dict[str, Any]] = None` and forwards it untouched as
# `_request_params["thinking"]`, so the {"type": "enabled", "budget_tokens": N}
# dict below reaches Anthropic's API in exactly that shape. It also validates
# the id against a NON_THINKING_MODELS list, treating unknown/newer ids as
# thinking-capable. Its replies come back as `reasoning_content` (mapped from
# the response's `thinking` blocks), which is why the anthropic *style* is
# happy reading the same side channel DeepSeek uses. The LangChain-side names
# above remain unverified here — those packages are not installed.


def _thinking_anthropic(config: ThinkingConfig) -> Dict[str, Any]:
    if config.enabled:
        budget = config.budget if config.budget else _DEFAULT_THINKING_BUDGET
        return {"thinking": {"type": "enabled", "budget_tokens": budget}}
    return {"thinking": {"type": "disabled"}}


def _thinking_openai(config: ThinkingConfig) -> Dict[str, Any]:
    if config.enabled:
        return {"reasoning_effort": _effort_for_budget(config)}
    # There is no portable "reasoning off" switch: a reasoning model always
    # reasons, and a non-reasoning model never does. Sending nothing is the
    # honest translation of "off" here.
    return {}


def _thinking_gemini(config: ThinkingConfig) -> Dict[str, Any]:
    if config.enabled:
        kwargs: Dict[str, Any] = {"include_thoughts": True}
        if config.budget is not None:
            kwargs["thinking_budget"] = config.budget
        return kwargs
    return {"include_thoughts": False, "thinking_budget": 0}


def _thinking_none(config: ThinkingConfig) -> Dict[str, Any]:
    """Providers that need no request-side opt-in (DeepSeek: `reasoning_content`
    is streamed by the reasoning models with no extra flag) or that have no
    documented knob (arbitrary OpenAI-compatible servers)."""
    return {}


#: provider name -> ThinkingConfig translator. Missing providers get
#: `_thinking_none`, i.e. their thinking is whatever the model does by default.
_THINKING_ADAPTERS: Dict[str, Callable[[ThinkingConfig], Dict[str, Any]]] = {
    "anthropic": _thinking_anthropic,
    "claude": _thinking_anthropic,
    "openai": _thinking_openai,
    "gemini": _thinking_gemini,
    "google": _thinking_gemini,
    "deepseek": _thinking_none,
}


def thinking_kwargs(provider: str, config: Optional[ThinkingConfig]) -> Dict[str, Any]:
    """Provider-specific constructor kwargs for a resolved ThinkingConfig.

    Returns {} for an unset config, so "nothing configured" sends nothing.
    """
    if config is None or not config.is_set:
        return {}
    return _THINKING_ADAPTERS.get((provider or "").strip().lower(), _thinking_none)(config)


def _construct_with_optional_kwargs(factory, kwargs: Dict[str, Any], optional: Dict[str, Any], label: str):
    """Construct a model, retrying without `optional` if the client rejects it.

    This is the safety net under the unverified thinking kwargs above: a client
    version that doesn't know a keyword raises TypeError (plain classes) or a
    pydantic ValidationError (which subclasses ValueError) — either way the
    model is rebuilt without the thinking request rather than the whole turn
    failing. The warning is printed, not swallowed silently, because "I asked
    for thinking and got none" is otherwise very hard to diagnose.
    """
    if not optional:
        return factory(**kwargs)
    try:
        return factory(**kwargs, **optional)
    except (TypeError, ValueError) as exc:
        print(
            f"[runtime.models] {label} rejected thinking arguments {sorted(optional)} "
            f"({type(exc).__name__}: {exc}); continuing without them.",
            flush=True,
        )
        return factory(**kwargs)


def _build_gemini(api_key: Optional[str], model_id: Optional[str], base_url: Optional[str], thinking: Optional[ThinkingConfig] = None):
    api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY or GEMINI_API_KEY is not set. Export it before running a Gemini Agent call.")
    if Gemini is None:
        raise RuntimeError("Gemini support is unavailable in the current installation (google-genai / agno google integration).")
    model_id = model_id or os.getenv("AGNO_MODEL") or _PROVIDER_DEFAULT_MODEL["gemini"]
    return _construct_with_optional_kwargs(
        Gemini, {"id": model_id, "api_key": api_key}, thinking_kwargs("gemini", thinking), "agno.models.google.Gemini"
    )


def _build_deepseek(api_key: Optional[str], model_id: Optional[str], base_url: Optional[str], thinking: Optional[ThinkingConfig] = None):
    api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set. Export it before running a DeepSeek Agent call.")
    if DeepSeek is None:
        raise RuntimeError("DeepSeek support is unavailable in the current installation of Agno.")
    model_id = model_id or os.getenv("AGNO_MODEL") or _PROVIDER_DEFAULT_MODEL["deepseek"]
    return DeepSeek(id=model_id, api_key=api_key)


def _build_anthropic(api_key: Optional[str], model_id: Optional[str], base_url: Optional[str], thinking: Optional[ThinkingConfig] = None):
    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Export it before running an Anthropic Agent call.")
    if Claude is None:
        raise RuntimeError("Anthropic support is unavailable in the current installation of Agno (needs the `anthropic` package).")
    model_id = model_id or os.getenv("AGNO_MODEL") or _PROVIDER_DEFAULT_MODEL["anthropic"]
    return _construct_with_optional_kwargs(
        Claude, {"id": model_id, "api_key": api_key}, thinking_kwargs("anthropic", thinking), "agno.models.anthropic.Claude"
    )


def _build_openai_compatible(api_key: Optional[str], model_id: Optional[str], base_url: Optional[str], thinking: Optional[ThinkingConfig] = None):
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


def _build_openai(api_key: Optional[str], model_id: Optional[str], base_url: Optional[str], thinking: Optional[ThinkingConfig] = None):
    # Fallback to OpenAI (also honors OPENAI_BASE_URL for OpenAI-compatible proxies)
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Export it before running a real OpenAI Agent call.")
    if OpenAIChat is None:
        raise RuntimeError("OpenAI support is unavailable in the current installation of Agno.")
    model_id = model_id or os.getenv("AGNO_MODEL") or _PROVIDER_DEFAULT_MODEL["openai"]
    base_url = base_url or os.getenv("OPENAI_BASE_URL")
    return _construct_with_optional_kwargs(
        OpenAIChat,
        {"id": model_id, "api_key": api_key, "base_url": base_url},
        thinking_kwargs("openai", thinking),
        "agno.models.openai.OpenAIChat",
    )


# provider name (already lowercased/stripped) -> builder. Every alias gets its
# own entry rather than a nested alias table, so a lookup is one dict hit and
# adding a provider is one line. Anything not listed here falls through to
# _DEFAULT_MODEL_BUILDER (OpenAI), which is what the original if/elif chain's
# trailing fallback did.
_MODEL_BUILDERS: Dict[str, Callable[..., object]] = {
    "gemini": _build_gemini,
    "google": _build_gemini,
    "deepseek": _build_deepseek,
    "anthropic": _build_anthropic,
    "claude": _build_anthropic,
    "openai": _build_openai,
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

    A builder may additionally accept `thinking=` (a ThinkingConfig) to honor
    the AGNO_THINKING knobs; one that doesn't is still called the old way (see
    `_call_builder`), so builders registered before thinking existed keep
    working untouched.
    """
    _MODEL_BUILDERS[name.strip().lower()] = builder


#: alias -> canonical provider name. Both halves are real registry keys (every
#: alias has its own builder entry, see _MODEL_BUILDERS); this table exists only
#: so a UI listing providers doesn't show the same model family three times.
_PROVIDER_ALIASES: Dict[str, str] = {
    "google": "gemini",
    "claude": "anthropic",
    "openai_compatible": "vllm",
    "custom": "vllm",
}


def _without_aliases(names: Iterable[str]) -> List[str]:
    return sorted(name for name in names if name not in _PROVIDER_ALIASES)


def available_providers(include_aliases: bool = False) -> List[str]:
    """Provider names the native Agno backend can build, for diagnostics/UI.

    Aliases ('google' for 'gemini', ...) are dropped unless asked for, so a
    picker built from this shows one entry per model family.
    """
    return sorted(_MODEL_BUILDERS) if include_aliases else _without_aliases(_MODEL_BUILDERS)


def resolve_provider(provider: Optional[str] = None) -> str:
    """The provider name `get_model` would use — request value, then
    AGNO_MODEL_PROVIDER, then the historical 'gemini' default.

    Exposed separately because callers need the *resolved* name for things
    other than building the model: picking a reasoning style for it, and
    reporting the current configuration to the panel.
    """
    return (provider or os.getenv("AGNO_MODEL_PROVIDER", "gemini")).strip().lower()


def _resolve_model_id(model_id: Optional[str], provider: str, env_provider: str) -> Optional[str]:
    """Model id for `provider`, or None to let the builder pick its default.

    AGNO_MODEL names a model *for the configured provider* — "deepseek-v4-flash"
    is meaningless to Gemini. So when a caller explicitly asks for a different
    provider than the environment is configured for (the panel's model picker
    doing exactly that), AGNO_MODEL is deliberately not applied and the builder
    falls back to that provider's own default id instead of being handed a
    foreign one. An explicit `model_id` always wins, and when the requested
    provider *is* the configured one nothing changes.
    """
    if model_id:
        return model_id
    if provider != env_provider:
        # Not the configured provider: hand it its own default id (None for the
        # openai-compatible providers, where there is no sensible default and
        # AGNO_MODEL genuinely is the served model name).
        return _PROVIDER_DEFAULT_MODEL.get(provider)
    return None  # same provider: the builders read AGNO_MODEL themselves


def _call_builder(builder, *, api_key, model_id, base_url, thinking: ThinkingConfig):
    """Call a provider builder, passing `thinking` only when it is configured.

    Builders registered by third-party code (see `register_provider`) may still
    have the original 3-argument signature; when nothing asked for thinking
    there is nothing to pass anyway, and when something did, a builder that
    can't take it is called the old way rather than blowing up.
    """
    kwargs = {"api_key": api_key, "model_id": model_id, "base_url": base_url}
    if thinking.is_set:
        try:
            return builder(thinking=thinking, **kwargs)
        except TypeError as exc:
            if "thinking" not in str(exc):
                raise
    return builder(**kwargs)


def get_model(
    api_key: Optional[str] = None,
    provider: Optional[str] = None,
    model_id: Optional[str] = None,
    base_url: Optional[str] = None,
    thinking=None,
    thinking_budget=None,
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
    - 'anthropic'/'claude': Claude via ANTHROPIC_API_KEY.
    - otherwise: OpenAI via OPENAI_API_KEY (also honors OPENAI_BASE_URL if set,
      so it can point at an OpenAI-compatible proxy without the 'vllm' provider).

    `thinking`/`thinking_budget` request reasoning output where the provider
    needs to be asked (see resolve_thinking / _THINKING_ADAPTERS); leaving both
    unset — and AGNO_THINKING unset — sends no thinking argument at all, which
    is exactly what this function did before thinking was configurable.
    """
    env_provider = resolve_provider(None)
    provider = resolve_provider(provider)
    builder = _MODEL_BUILDERS.get(provider, _DEFAULT_MODEL_BUILDER)
    return _call_builder(
        builder,
        api_key=api_key,
        model_id=_resolve_model_id(model_id, provider, env_provider),
        base_url=base_url,
        thinking=resolve_thinking(thinking, thinking_budget),
    )


# ---------------------------------------------------------------------------
# The LangChain chat-model factory (LangGraph/DeepAgents backend)
# ---------------------------------------------------------------------------
#
# Same shape as the Agno-side factory above — a builder registry, per-builder
# env-var fallbacks, per-builder "not installed" messages — but it returns a
# `langchain_core.language_models.BaseChatModel` rather than an `agno.models.*`
# instance, because that is what `deepagents.create_deep_agent(model=...)` and
# any other LangGraph graph expect.
#
# Every import here is inside its builder: langchain packages are optional
# extras, and importing this module must not require any of them.

#: Env vars for the LangGraph/DeepAgents backend's model, kept separate from
#: AGNO_MODEL_PROVIDER/AGNO_MODEL on purpose: the two backends can be pointed at
#: different providers at once, and (more importantly) an existing .env that
#: sets AGNO_MODEL_PROVIDER for the native backend must not silently re-point
#: the DeepAgents harness, which was hardcoded to DeepSeek before this existed.
LANGCHAIN_PROVIDER_ENV_VAR = "AGNO_LANGGRAPH_PROVIDER"
LANGCHAIN_MODEL_ENV_VAR = "AGNO_LANGGRAPH_MODEL"


def _import_langchain(module_name: str, attr: str, pip_name: str):
    """Import one LangChain chat-model class, or raise the install hint.

    Mirrors the Agno side's "support is unavailable in the current
    installation" messages, but names the exact pip package to install, since
    each LangChain provider ships as its own distribution.
    """
    try:
        import importlib
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"{pip_name} is required for this provider on the LangGraph backend. "
            f"Run: pip install {pip_name}"
        ) from exc
    model_cls = getattr(module, attr, None)
    if model_cls is None:
        raise RuntimeError(f"'{attr}' was not found in '{module_name}' — is {pip_name} up to date?")
    return model_cls


def _build_lc_deepseek(api_key: Optional[str], model_id: Optional[str], base_url: Optional[str], thinking: Optional[ThinkingConfig] = None):
    api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set — required for the default DeepAgents/DeepSeek harness.")
    # langchain_deepseek.ChatDeepSeek, not plain langchain_openai.ChatOpenAI pointed
    # at DeepSeek's endpoint — verified empirically that ChatOpenAI's SSE parser
    # doesn't recognize DeepSeek's `reasoning_content` delta field and silently
    # drops it, so reasoning would never reach the thinking-block UI at all. Also
    # handles DeepSeek's thinking-mode request shape itself, no manual extra_body needed.
    ChatDeepSeek = _import_langchain("langchain_deepseek", "ChatDeepSeek", "langchain-deepseek")
    model_id = model_id or os.getenv("AGNO_MODEL") or _PROVIDER_DEFAULT_MODEL["deepseek"]
    return _construct_with_optional_kwargs(
        ChatDeepSeek,
        {"model": model_id, "api_key": api_key},
        thinking_kwargs("deepseek", thinking),
        "langchain_deepseek.ChatDeepSeek",
    )


def _build_lc_openai(api_key: Optional[str], model_id: Optional[str], base_url: Optional[str], thinking: Optional[ThinkingConfig] = None):
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Export it before running an OpenAI LangGraph call.")
    ChatOpenAI = _import_langchain("langchain_openai", "ChatOpenAI", "langchain-openai")
    model_id = model_id or os.getenv("AGNO_MODEL") or _PROVIDER_DEFAULT_MODEL["openai"]
    base_url = base_url or os.getenv("OPENAI_BASE_URL")
    kwargs: Dict[str, Any] = {"model": model_id, "api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return _construct_with_optional_kwargs(
        ChatOpenAI, kwargs, thinking_kwargs("openai", thinking), "langchain_openai.ChatOpenAI"
    )


def _build_lc_anthropic(api_key: Optional[str], model_id: Optional[str], base_url: Optional[str], thinking: Optional[ThinkingConfig] = None):
    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Export it before running an Anthropic LangGraph call.")
    ChatAnthropic = _import_langchain("langchain_anthropic", "ChatAnthropic", "langchain-anthropic")
    model_id = model_id or os.getenv("AGNO_MODEL") or _PROVIDER_DEFAULT_MODEL["anthropic"]
    kwargs: Dict[str, Any] = {"model": model_id, "api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return _construct_with_optional_kwargs(
        ChatAnthropic, kwargs, thinking_kwargs("anthropic", thinking), "langchain_anthropic.ChatAnthropic"
    )


def _build_lc_gemini(api_key: Optional[str], model_id: Optional[str], base_url: Optional[str], thinking: Optional[ThinkingConfig] = None):
    api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY or GEMINI_API_KEY is not set. Export it before running a Gemini LangGraph call.")
    ChatGoogleGenerativeAI = _import_langchain(
        "langchain_google_genai", "ChatGoogleGenerativeAI", "langchain-google-genai"
    )
    model_id = model_id or os.getenv("AGNO_MODEL") or _PROVIDER_DEFAULT_MODEL["gemini"]
    return _construct_with_optional_kwargs(
        ChatGoogleGenerativeAI,
        {"model": model_id, "google_api_key": api_key},
        thinking_kwargs("gemini", thinking),
        "langchain_google_genai.ChatGoogleGenerativeAI",
    )


def _build_lc_openai_compatible(api_key: Optional[str], model_id: Optional[str], base_url: Optional[str], thinking: Optional[ThinkingConfig] = None):
    base_url = base_url or os.getenv("VLLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    if not base_url:
        raise RuntimeError(
            "A base URL is required for the vllm/openai_compatible provider. "
            "Set VLLM_BASE_URL (e.g. http://localhost:8000/v1) or pass base_url explicitly."
        )
    ChatOpenAI = _import_langchain("langchain_openai", "ChatOpenAI", "langchain-openai")
    api_key = api_key or os.getenv("VLLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY"
    model_id = model_id or os.getenv("AGNO_MODEL")
    if not model_id:
        raise RuntimeError("model_id (or AGNO_MODEL) is required for the vllm/openai_compatible provider — use the model name vLLM was served with, e.g. --served-model-name.")
    # Note the ChatDeepSeek comment above: a self-hosted *reasoning* model
    # served over an OpenAI-compatible endpoint puts its thinking in
    # `reasoning_content`, which this wrapper may drop the same way. The
    # "openai" reasoning style reads both that key and typed reasoning blocks
    # (see runtime/reasoning.py), so whatever does survive is still classified.
    return _construct_with_optional_kwargs(
        ChatOpenAI,
        {"model": model_id, "api_key": api_key, "base_url": base_url},
        thinking_kwargs("openai_compatible", thinking),
        "langchain_openai.ChatOpenAI",
    )


#: provider name -> LangChain builder. Mirrors _MODEL_BUILDERS entry for entry
#: (one line per alias, no nested alias table). Unlike the Agno side there is no
#: catch-all fallback builder: an unknown provider name here would otherwise
#: silently build an OpenAI model for a DeepSeek-configured harness, so
#: get_langchain_model raises with the list of known names instead.
_LANGCHAIN_MODEL_BUILDERS: Dict[str, Callable[..., object]] = {
    "deepseek": _build_lc_deepseek,
    "openai": _build_lc_openai,
    "anthropic": _build_lc_anthropic,
    "claude": _build_lc_anthropic,
    "gemini": _build_lc_gemini,
    "google": _build_lc_gemini,
    "vllm": _build_lc_openai_compatible,
    "openai_compatible": _build_lc_openai_compatible,
    "custom": _build_lc_openai_compatible,
}


def register_langchain_provider(name: str, builder: Callable[..., object]) -> None:
    """Register (or override) a LangChain chat-model builder under `name`.

    A builder is called as builder(api_key=..., model_id=..., base_url=...,
    thinking=...) and returns a LangChain chat model; like the Agno-side
    builders it owns its env-var fallbacks, its "not installed" message and its
    default model id. `thinking` is only passed when configured (see
    `_call_builder`), so a 3-argument builder still works.
    """
    _LANGCHAIN_MODEL_BUILDERS[name.strip().lower()] = builder


def available_langchain_providers(include_aliases: bool = False) -> List[str]:
    """Provider names the LangGraph/DeepAgents backend can build (see
    `available_providers` for the alias handling)."""
    if include_aliases:
        return sorted(_LANGCHAIN_MODEL_BUILDERS)
    return _without_aliases(_LANGCHAIN_MODEL_BUILDERS)


def resolve_langchain_provider(provider: Optional[str] = None) -> str:
    """The provider name `get_langchain_model` would use — request value, then
    AGNO_LANGGRAPH_PROVIDER, then DeepSeek (what the harness always used).

    Exposed separately because `runtime.graph` needs the resolved name to pick
    the matching reasoning style for the model it is about to build.
    """
    return (provider or os.getenv(LANGCHAIN_PROVIDER_ENV_VAR) or DEFAULT_LANGCHAIN_PROVIDER).strip().lower()


def get_langchain_model(
    api_key: Optional[str] = None,
    provider: Optional[str] = None,
    model_id: Optional[str] = None,
    base_url: Optional[str] = None,
    thinking=None,
    thinking_budget=None,
):
    """Resolve and return a configured LangChain chat model.

    The LangGraph-backend counterpart of `get_model`, with the same
    request-value -> env-var -> default precedence for every argument.

    Priority:
    - 'deepseek' (the default): ChatDeepSeek via DEEPSEEK_API_KEY.
    - 'openai': ChatOpenAI via OPENAI_API_KEY (honors OPENAI_BASE_URL too).
    - 'anthropic'/'claude': ChatAnthropic via ANTHROPIC_API_KEY.
    - 'gemini'/'google': ChatGoogleGenerativeAI via GOOGLE_API_KEY/GEMINI_API_KEY.
    - 'vllm'/'openai_compatible'/'custom': ChatOpenAI pointed at
      VLLM_BASE_URL/OPENAI_BASE_URL (vLLM, LM Studio, Ollama's OpenAI shim...).

    `thinking`/`thinking_budget` are translated per provider (see
    `_THINKING_ADAPTERS`); unset means no thinking argument is sent, i.e. the
    behavior of the hardcoded ChatDeepSeek call this replaced.
    """
    env_provider = resolve_langchain_provider(None)
    provider = resolve_langchain_provider(provider)
    builder = _LANGCHAIN_MODEL_BUILDERS.get(provider)
    if builder is None:
        raise RuntimeError(
            f"Unknown LangGraph model provider '{provider}'. Known providers: "
            f"{', '.join(available_langchain_providers())}. Set {LANGCHAIN_PROVIDER_ENV_VAR} "
            f"(or pass model_provider in the chat request) to one of these."
        )
    return _call_builder(
        builder,
        api_key=api_key,
        model_id=model_id or os.getenv(LANGCHAIN_MODEL_ENV_VAR) or _resolve_model_id(None, provider, env_provider),
        base_url=base_url,
        thinking=resolve_thinking(thinking, thinking_budget),
    )
