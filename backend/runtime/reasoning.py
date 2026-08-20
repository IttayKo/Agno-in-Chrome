"""Pluggable "communication styles": how a given model family expresses
thinking vs. output text, in one place instead of scattered across the
streaming adapters.

Two very different sources of the same question feed into this module:

- a raw streaming chunk from a LangChain chat model, as seen by
  `runtime.graph._make_reasoning_langgraph_agent_class()._arun_adapter_stream`
  while walking `graph.astream_events(...)`, and
- an Agno run event (`RunContent`, `ReasoningContentDelta`, ...), as seen by
  `runtime.events.stream_agno_events`.

Both are normalized to the same tiny vocabulary — a stream of
`("text", str)` / `("thinking", str)` pairs — so neither adapter needs to know
a provider-specific field name. Adding a provider means registering a
`ReasoningStyle` here; no adapter changes.

Selection: `resolve_style(name_or_auto, provider)`. `"auto"` (or None, or an
unrecognized name) resolves from the provider; an explicit style name wins over
the provider. Callers thread the name down from `AGNO_REASONING_STYLE` / an
explicit `reasoning_style=` argument.
"""

import os
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

#: One normalized piece of a model's output: ("text", ...) is user-visible
#: answer text, ("thinking", ...) is reasoning/thinking content.
Piece = Tuple[str, str]

TEXT = "text"
THINKING = "thinking"

#: Env var used as the fallback when no explicit style is passed in.
STYLE_ENV_VAR = "AGNO_REASONING_STYLE"

#: The names of the Agno run events that carry model output; anything else
#: (tool events, run lifecycle) is not this module's business.
DEFAULT_EVENT_NAMES = ("ReasoningContentDelta", "RunContent")


def _as_dict(block) -> Optional[dict]:
    """A content block as a plain dict, or None if it isn't one.

    LangChain content blocks are TypedDicts (i.e. dicts) today, but some
    versions/providers hand back pydantic objects instead; converting those
    costs one attribute probe and avoids dropping their content on the floor.
    """
    if isinstance(block, dict):
        return block
    dump = getattr(block, "model_dump", None)
    if callable(dump):
        try:
            dumped = dump()
        except Exception:
            return None
        if isinstance(dumped, dict):
            return dumped
    return None


def _part_text(part) -> str:
    """Text of one nested part of a block (e.g. an OpenAI reasoning summary
    entry, which is either a bare string or a {"type": "summary_text", "text":
    ...} dict)."""
    if isinstance(part, str):
        return part
    part = _as_dict(part)
    if not part:
        return ""
    for key in ("text", "thinking", "thought", "reasoning_content", "content"):
        value = part.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _side_channel_text(chunk, keys: Tuple[str, ...]) -> str:
    """Thinking text carried outside `chunk.content`, under any of `keys`.

    Providers ship this in `additional_kwargs` (DeepSeek's `reasoning_content`
    being the original case) but not always: some LangChain versions surface it
    as an attribute on the chunk instead, so both are probed. Values may be a
    string, or a dict/list wrapping one (OpenAI's `reasoning` object with its
    `summary` list) — `_part_text` flattens those. Booleans are ignored on
    purpose: Gemini's `thought: True` marks text elsewhere, it is not text.
    """
    extra = getattr(chunk, "additional_kwargs", None)
    if not isinstance(extra, dict):
        extra = {}
    collected: List[str] = []
    for key in keys:
        for value in (extra.get(key), getattr(chunk, key, None)):
            if isinstance(value, bool) or value is None:
                continue
            if isinstance(value, str):
                if value:
                    collected.append(value)
                continue
            if isinstance(value, list):
                collected.extend(part for part in (_part_text(item) for item in value) if part)
                continue
            as_dict = _as_dict(value)
            if not as_dict:
                continue
            for nested_key in ("text", "thinking", "thought", "reasoning_content", "content"):
                nested = as_dict.get(nested_key)
                if isinstance(nested, str) and nested:
                    collected.append(nested)
                    break
            else:
                summary = as_dict.get("summary")
                if isinstance(summary, list):
                    collected.extend(part for part in (_part_text(item) for item in summary) if part)
    # Dedupe while preserving order: the same text often appears both as an
    # attribute and in additional_kwargs, and emitting it twice would duplicate
    # it in the panel's thinking block.
    seen = set()
    unique = []
    for text in collected:
        if text not in seen:
            seen.add(text)
            unique.append(text)
    return "".join(unique)


class ReasoningStyle:
    """How one model family separates thinking from answer text.

    Subclasses normally only need to override `name` and `chunk_reasoning`;
    the typed-content-block handling and the Agno-event handling below are
    shared, because they are provider-agnostic shapes rather than one
    provider's convention.
    """

    #: Registry key for this style.
    name = "plain"

    #: Agno run-event names this style claims from `stream_agno_events`.
    event_names: Tuple[str, ...] = DEFAULT_EVENT_NAMES

    # -- LangChain streaming chunks -----------------------------------------

    def chunk_reasoning(self, chunk) -> str:
        """Provider-specific thinking text carried *outside* `chunk.content`.

        The base implementation says "there is none" — that is the "plain"
        style. A provider that ships reasoning in a side channel (DeepSeek's
        `additional_kwargs`, say) overrides just this.
        """
        return ""

    #: Block `type` values that mean "this block is thinking, not answer text".
    #: Several are the same concept under different provider/version spellings
    #: (Anthropic streams "thinking" blocks whose deltas may arrive typed as
    #: "thinking_delta", and redacts some of them as "redacted_thinking").
    THINKING_BLOCK_TYPES = (
        "thinking", "thinking_delta", "redacted_thinking",
        "reasoning", "reasoning_delta", "reasoning_content", "thought",
    )

    #: Keys a thinking block may carry its text under, in priority order. The
    #: first two preserve the original behavior; the rest are the shapes other
    #: providers use ("thinking" for Anthropic blocks, "summary" for OpenAI
    #: Responses-API reasoning summaries, "thought"/"content" as catch-alls).
    BLOCK_TEXT_KEYS = ("text", "reasoning_content", "thinking", "thought", "summary", "content")

    def block_text(self, block: dict) -> str:
        """Best-effort text of one typed content block, whatever key holds it.

        Deliberately tolerant: the same provider moves this between versions
        (a plain "text" key, a nested "summary" list of summary_text parts,
        ...), and reading a key that isn't there is free, whereas assuming one
        spelling loses the thinking silently.
        """
        for key in self.BLOCK_TEXT_KEYS:
            value = block.get(key)
            if isinstance(value, str) and value:
                return value
            if isinstance(value, list):
                parts = [_part_text(part) for part in value]
                joined = "".join(part for part in parts if part)
                if joined:
                    return joined
        return ""

    def block_is_thinking(self, block: dict) -> bool:
        """Whether a typed content block holds thinking rather than answer text.

        Beyond the block `type`, this also honors Gemini's shape, where a
        thought is an ordinary text part flagged with `thought: True` rather
        than given a type of its own.
        """
        if block.get("type") in self.THINKING_BLOCK_TYPES:
            return True
        return block.get("thought") is True

    def content_block_pieces(self, blocks: Iterable) -> Iterator[Piece]:
        """Generic typed-content-block handling, shared by every style.

        Some providers (not DeepSeek via ChatDeepSeek, which uses
        additional_kwargs above) put thinking in typed content blocks.
        """
        for block in blocks:
            block = _as_dict(block)
            if block is None:
                continue
            text = self.block_text(block)
            if not text:
                continue
            if self.block_is_thinking(block):
                yield (THINKING, text)
            else:
                yield (TEXT, text)

    def chunk_pieces(self, chunk) -> Iterator[Piece]:
        """Normalize one LangChain chat-model streaming chunk.

        Text pieces are yielded first, in order; all of the chunk's thinking
        content — from the provider side channel and from any typed thinking
        blocks — is coalesced into a single trailing ("thinking", ...) piece.
        That ordering is deliberate and matches what the adapter emitted
        before this abstraction existed: one ReasoningContentDeltaEvent per
        chunk, after that chunk's RunContentEvents.
        """
        reasoning = self.chunk_reasoning(chunk)

        content = getattr(chunk, "content", None)
        if isinstance(content, str) and content:
            yield (TEXT, content)
        elif isinstance(content, list):
            for kind, text in self.content_block_pieces(content):
                if kind == THINKING:
                    reasoning += text
                else:
                    yield (TEXT, text)

        if reasoning:
            yield (THINKING, reasoning)

    # -- Agno run events ----------------------------------------------------

    def owns_event(self, event_name: Optional[str]) -> bool:
        """True when `event_pieces` knows how to normalize this event name."""
        return event_name in self.event_names

    def event_pieces(self, event) -> Iterator[Piece]:
        """Normalize one Agno run event into the same ("text"/"thinking") pieces.

        Only events `owns_event()` accepts are passed here; anything else
        yields nothing.
        """
        event_name = getattr(event, 'event', None)

        if event_name == 'ReasoningContentDelta':
            chunk = getattr(event, 'reasoning_content', None)
            if chunk:
                yield (THINKING, str(chunk))
            return

        if event_name == 'RunContent':
            piece = getattr(event, 'content', None)
            if piece:
                yield (TEXT, str(piece))
            reasoning_piece = getattr(event, 'reasoning_content', None)
            if reasoning_piece:
                yield (THINKING, str(reasoning_piece))
            return


class PlainReasoningStyle(ReasoningStyle):
    """Content only, no separate thinking channel — the base behavior."""

    name = "plain"


class DeepSeekReasoningStyle(ReasoningStyle):
    """DeepSeek via `langchain_deepseek.ChatDeepSeek`.

    Verified empirically (see `_make_reasoning_langgraph_agent_class`'s
    docstring for the full history): DeepSeek's reasoning survives the SSE
    parse only with `ChatDeepSeek`, and when it does it lands in
    `chunk.additional_kwargs['reasoning_content']` rather than in
    `chunk.content`.
    """

    name = "deepseek"

    def chunk_reasoning(self, chunk) -> str:
        return (getattr(chunk, "additional_kwargs", None) or {}).get("reasoning_content") or ""


class OpenAIReasoningStyle(ReasoningStyle):
    """OpenAI-style reasoning models (o-series/GPT-5-class), and the many
    OpenAI-compatible servers that imitate them.

    UNVERIFIED SHAPES, READ DEFENSIVELY: which of these actually arrives
    depends on the langchain-openai version and on whether the Chat Completions
    or the Responses API is in use, and none of those packages are installed in
    the environment this was written in. So all of them are probed rather than
    one being assumed:

    - `additional_kwargs['reasoning_content']` — what OpenAI-compatible servers
      (vLLM/DeepSeek-style) emit, and what some langchain-openai versions map
      Responses-API reasoning into;
    - `additional_kwargs['reasoning']` — the Responses API's reasoning object,
      whose visible text lives in a `summary` list of `summary_text` parts;
    - typed `{"type": "reasoning", ...}` content blocks, handled by the shared
      `content_block_pieces`.

    Note that plain Chat Completions reasoning models emit no thinking text at
    all — they reason internally and only bill for it. Nothing here can conjure
    that; `reasoning={"summary": "auto"}` on the Responses API is what makes a
    summary exist in the first place (see runtime/models.py's thinking adapter).
    """

    name = "openai"

    REASONING_KEYS = ("reasoning_content", "reasoning", "reasoning_summary", "thinking")

    def chunk_reasoning(self, chunk) -> str:
        return _side_channel_text(chunk, self.REASONING_KEYS)


class AnthropicReasoningStyle(ReasoningStyle):
    """Anthropic (Claude) extended thinking via `langchain_anthropic.ChatAnthropic`.

    Anthropic's own transport puts thinking in typed content blocks
    (`{"type": "thinking", "thinking": "..."}`, with `redacted_thinking` for
    the parts it withholds), which the shared `content_block_pieces` already
    classifies — that is the primary path here. The side-channel probe below is
    a belt-and-braces fallback for versions that lift thinking out of the block
    list into `additional_kwargs` instead.

    UNVERIFIED: langchain-anthropic is not installed here, so which of the two
    shapes a given version emits could not be executed and confirmed; both are
    read for that reason.
    """

    name = "anthropic"

    REASONING_KEYS = ("reasoning_content", "thinking", "reasoning")

    def chunk_reasoning(self, chunk) -> str:
        return _side_channel_text(chunk, self.REASONING_KEYS)


class GeminiReasoningStyle(ReasoningStyle):
    """Gemini thinking via `langchain_google_genai.ChatGoogleGenerativeAI`.

    Gemini's distinctive shape is that a thought is not a block type of its
    own: it is an ordinary text part carrying `thought: True`. That is handled
    in the shared `block_is_thinking`, so this style's own job is just the side
    channel some versions use instead.

    UNVERIFIED: langchain-google-genai is not installed here. Thinking also has
    to be *requested* for Gemini (`include_thoughts` / `thinking_budget`, see
    runtime/models.py) — without that the stream simply contains no thoughts,
    and no amount of parsing changes that.
    """

    name = "gemini"

    REASONING_KEYS = ("reasoning_content", "thought", "thinking", "reasoning")

    def chunk_reasoning(self, chunk) -> str:
        return _side_channel_text(chunk, self.REASONING_KEYS)


#: style name -> instance. Styles are stateless, so one shared instance each.
_STYLES: Dict[str, ReasoningStyle] = {}


def register_style(style: ReasoningStyle) -> ReasoningStyle:
    """Register (or replace) a style under its `name`."""
    _STYLES[style.name] = style
    return style


register_style(PlainReasoningStyle())
register_style(DeepSeekReasoningStyle())
register_style(OpenAIReasoningStyle())
register_style(AnthropicReasoningStyle())
register_style(GeminiReasoningStyle())

#: provider name -> style name, for `resolve_style("auto", provider)`. Keys are
#: the same provider names runtime/models.py builds (every alias listed, no
#: nested alias table), so "auto" works for whatever a caller/env named.
#: The openai-compatible providers map to the OpenAI style because that is what
#: the servers behind them imitate — and it reads `reasoning_content` too, which
#: is what a self-hosted reasoning model behind such a server emits.
_PROVIDER_STYLES: Dict[str, str] = {
    "deepseek": "deepseek",
    "openai": "openai",
    "anthropic": "anthropic",
    "claude": "anthropic",
    "gemini": "gemini",
    "google": "gemini",
    "vllm": "openai",
    "openai_compatible": "openai",
    "custom": "openai",
}

# The auto-resolution fallback is deliberately "deepseek", not "plain": before
# styles existed, the LangGraph adapter unconditionally read
# additional_kwargs['reasoning_content'] for *every* model (including a custom
# LANGGRAPH_GRAPH whose provider is unknown here), so anything else would be a
# behavior change rather than a refactor. Reading that key on a model that
# never sets it is a no-op, which is why the old code got away with it. As
# real per-provider styles are registered, narrow this by adding entries to
# _PROVIDER_STYLES rather than by flipping the fallback.
_AUTO_FALLBACK_STYLE = "deepseek"

#: The provider the default DeepAgents harness builds when nothing else is
#: configured (see `runtime.graph.build_deepagents_graph`, which used to pin
#: ChatDeepSeek unconditionally and now asks `runtime.models.get_langchain_model`
#: for a provider — defaulting to this one, so an unconfigured install still
#: comes up on DeepSeek exactly as it always did). `runtime.models` imports this
#: as DEFAULT_LANGCHAIN_PROVIDER rather than restating the name.
DEEPAGENTS_PROVIDER = "deepseek"


def available_styles() -> List[str]:
    """Registered style names, for diagnostics/UI."""
    return sorted(_STYLES)


def style_for_provider(provider: Optional[str]) -> ReasoningStyle:
    """The style a provider name implies, or the auto fallback."""
    key = (provider or "").strip().lower()
    return _STYLES[_PROVIDER_STYLES.get(key, _AUTO_FALLBACK_STYLE)]


def resolve_style(name_or_auto: Optional[str] = None, provider: Optional[str] = None) -> ReasoningStyle:
    """Pick a `ReasoningStyle`.

    `name_or_auto` is an explicit registered style name, "auto", or None. None
    falls back to the AGNO_REASONING_STYLE env var, then to "auto"; "auto"
    resolves from `provider` (see `style_for_provider`).

    An unrecognized name is treated as "auto" rather than raising: this value
    arrives from an env var today and from request payloads in a later phase,
    and a typo there should not fail a whole turn.
    """
    name = (name_or_auto or os.getenv(STYLE_ENV_VAR) or "auto").strip().lower()
    if name and name != "auto":
        style = _STYLES.get(name)
        if style is not None:
            return style
    return style_for_provider(provider)
