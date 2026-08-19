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

    def content_block_pieces(self, blocks: Iterable) -> Iterator[Piece]:
        """Generic typed-content-block handling, shared by every style.

        Some providers (not DeepSeek via ChatDeepSeek, which uses
        additional_kwargs above) put thinking in typed content blocks.
        """
        for block in blocks:
            if not isinstance(block, dict):
                continue
            text = block.get("text") or block.get("reasoning_content") or block.get("content")
            if not text:
                continue
            if block.get("type") in ("thinking", "reasoning", "reasoning_content"):
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


#: style name -> instance. Styles are stateless, so one shared instance each.
_STYLES: Dict[str, ReasoningStyle] = {}


def register_style(style: ReasoningStyle) -> ReasoningStyle:
    """Register (or replace) a style under its `name`."""
    _STYLES[style.name] = style
    return style


register_style(PlainReasoningStyle())
register_style(DeepSeekReasoningStyle())

#: provider name -> style name, for `resolve_style("auto", provider)`.
_PROVIDER_STYLES: Dict[str, str] = {
    "deepseek": "deepseek",
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

#: The provider the default DeepAgents harness builds (see
#: `runtime.graph.build_deepagents_graph`, which pins ChatDeepSeek).
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
