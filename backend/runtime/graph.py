"""The LangGraph/DeepAgents backend: loading a configured graph, building the
default DeepAgents harness, and the LangGraphAgent subclass that surfaces
reasoning content through a pluggable communication style.
"""

import os
from typing import Any, Dict, List, Optional

from .agent import _shared_db
from .browser_tools import _OriginState, _deepagents_browser_tools
from .context import (
    _asset_index,
    _make_load_asset_tool_for_deepagents,
    _mcp_tools_for_deepagents,
)
from .reasoning import DEEPAGENTS_PROVIDER, THINKING, ReasoningStyle, resolve_style


def _load_langgraph_graph(graph_path: Optional[str] = None):
    """Import and return a compiled LangGraph graph from a 'module.path:attr' string.

    This is the integration point for graphs Agno cannot run natively — most
    notably a DeepAgents agent, since `deepagents.create_deep_agent(...)` returns
    a LangGraph `CompiledStateGraph` rather than an Agno-compatible model/agent.
    Point LANGGRAPH_GRAPH at wherever that compiled graph is exposed, e.g.:

        # my_deep_agent.py
        from deepagents import create_deep_agent
        graph = create_deep_agent(tools=[...], instructions="...")

        # .env
        LANGGRAPH_GRAPH=my_deep_agent:graph

    Requires `langgraph` (and whatever built the graph, e.g. `deepagents`) to be
    installed — these are optional dependencies of this bridge, not hard requirements.
    """
    graph_path = graph_path or os.getenv("LANGGRAPH_GRAPH")
    if not graph_path:
        raise RuntimeError(
            "No LangGraph graph configured. Set LANGGRAPH_GRAPH=module.path:graph_variable "
            "(e.g. the CompiledStateGraph returned by deepagents.create_deep_agent()), "
            "or pass langgraph_graph in the chat request."
        )
    if ":" not in graph_path:
        raise RuntimeError("LANGGRAPH_GRAPH must be in 'module.path:graph_variable' form.")
    module_name, _, attr = graph_path.partition(":")
    try:
        import importlib
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"Could not import '{module_name}' for LANGGRAPH_GRAPH. Make sure langgraph "
            f"(and the package that builds your graph, e.g. deepagents) is installed and "
            f"the module is importable from where server.py runs. Original error: {exc}"
        ) from exc
    graph = getattr(module, attr, None)
    if graph is None:
        raise RuntimeError(f"'{attr}' was not found in module '{module_name}'.")
    return graph


async def build_deepagents_graph(
    origin: Optional[str] = None,
    include_browser_tools: bool = True,
    extra_tools: Optional[List[Any]] = None,
    extra_system_prompt: Optional[str] = None,
    skills: Optional[List[dict]] = None,
    knowledge: Optional[List[dict]] = None,
    mcp_servers: Optional[List[dict]] = None,
    tool_meta: Optional[Dict[str, dict]] = None,
    reasoning_style: Optional[str] = None,
):
    """Build the default DeepAgents graph, configured with DeepSeek — this is
    what AGNO_BACKEND=langgraph uses out of the box when no LANGGRAPH_GRAPH
    override is set, so switching the default backend to LangGraph actually
    works rather than requiring the user to hand-wire a graph first.

    create_deep_agent(...) in the installed deepagents version already returns
    a *compiled* graph (a CompiledStateGraph) — no separate .compile() call
    needed or wanted; calling .compile() again on an already-compiled graph is
    not part of its API.

    Browser tools are rebuilt fresh per call, bound to `origin` via closure
    (see _deepagents_browser_tools) — deepagents has no post-construction way
    to rebind a tool's captured origin, so (mirroring build_agent()) the whole
    graph is rebuilt per request rather than cached as a module-level singleton.

    Same generic context-injection params as build_agent (extra_tools/
    extra_system_prompt/skills/knowledge/mcp_servers/tool_meta) — this module
    has no idea any of it might have come from a domain manifest; that
    translation is server.py's job. This function is `async def` specifically
    because MCP tool loading here needs an upfront await (see
    _mcp_tools_for_deepagents) — DeepAgents has no lazy per-run auto-connect
    the way Agno's native MCPTools does.

    `reasoning_style` names the communication style used to tell this model's
    thinking apart from its answer text (see runtime/reasoning.py); None means
    AGNO_REASONING_STYLE and then "auto", which resolves from this harness's
    provider — DeepSeek, since the model below is pinned to ChatDeepSeek.

    Returns (graph, tool_meta, style) — tool_meta is passed straight through, for
    build_langgraph_agent to attach to the wrapping agent, and style is the
    resolved ReasoningStyle for the model this harness just built, which
    build_langgraph_agent hands to the streaming adapter. This function is the
    only place that knows which provider the default harness uses, so it is also
    the right place to answer "how does this model express thinking?".
    """
    try:
        from deepagents import create_deep_agent
        from langchain_deepseek import ChatDeepSeek
    except ImportError as exc:
        raise RuntimeError(
            "deepagents/langchain-deepseek are required for the default LangGraph backend. "
            "Run: pip install deepagents langgraph langchain-core langchain-deepseek"
        ) from exc

    origin = origin or os.getenv("BROWSER_ORIGIN", "https://example.com")

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set — required for the default DeepAgents/DeepSeek harness.")
    model_id = os.getenv("AGNO_MODEL", "deepseek-v4-flash")
    # langchain_deepseek.ChatDeepSeek, not plain langchain_openai.ChatOpenAI pointed
    # at DeepSeek's endpoint — verified empirically that ChatOpenAI's SSE parser
    # doesn't recognize DeepSeek's `reasoning_content` delta field and silently
    # drops it, so reasoning would never reach the thinking-block UI at all. Also
    # handles DeepSeek's thinking-mode request shape itself, no manual extra_body needed.
    llm_model = ChatDeepSeek(model=model_id, api_key=api_key)

    tools = _deepagents_browser_tools(_OriginState(origin)) if include_browser_tools else []
    if extra_tools:
        tools.extend(extra_tools)
    if mcp_servers:
        tools.extend(await _mcp_tools_for_deepagents(mcp_servers))
    if skills or knowledge:
        tools.append(_make_load_asset_tool_for_deepagents(skills, knowledge))

    system_prompt = (
        f"You are an autonomous browser-automation operations expert working against {origin}. "
        "Break goals down into concrete steps and use the browser tools to read and act on the "
        "page. Use your file/shell tools only for non-browser work."
    )
    if extra_system_prompt:
        system_prompt = f"{system_prompt}\n\n{extra_system_prompt}"
    asset_index = _asset_index(skills, knowledge)
    if asset_index:
        system_prompt = f"{system_prompt}\n\n{asset_index}"

    graph = create_deep_agent(
        model=llm_model,
        tools=tools,
        system_prompt=system_prompt,
        name="Deep Agent (DeepSeek)",
    )
    style = resolve_style(reasoning_style, DEEPAGENTS_PROVIDER)
    return graph, (tool_meta or {}), style


def _make_reasoning_langgraph_agent_class(style: Optional[ReasoningStyle] = None):
    """Build a `LangGraphAgent` subclass that also surfaces reasoning/"thinking"
    content, which Agno's own adapter drops on the floor.

    Defined lazily inside a function (rather than at module level) because it
    needs `agno.agents.langgraph.LangGraphAgent` as its base class, and that
    import is itself optional (see build_langgraph_agent) — importing it at
    module level would make agno_runtime.py fail to import entirely whenever
    langgraph/langchain-core aren't installed, even for callers who only ever
    use the native Agno backend.

    Agno's own `LangGraphAgent._arun_adapter_stream` only yields a
    RunContentEvent for `on_chat_model_stream` chunks whose `.content` is a
    plain string, silently dropping everything else — which is exactly where a
    reasoning model's thinking lives. Verified empirically: DeepSeek's
    reasoning_content only survives at all when using `langchain_deepseek.
    ChatDeepSeek` (not plain `langchain_openai.ChatOpenAI` pointed at
    DeepSeek's endpoint — that wrapper's SSE parser doesn't know the
    `reasoning_content` field and drops it before it ever reaches Agno's
    adapter), where it lands in `chunk.additional_kwargs['reasoning_content']`.
    This override adds that one branch on top of the parent's tool-call
    handling, instead of reimplementing the whole adapter.

    Which field a chunk's thinking actually lives in is NOT decided here: that
    is the `style` argument's job (a `ReasoningStyle` from runtime/reasoning.py,
    defaulting to AGNO_REASONING_STYLE/auto), so adding a model family means
    registering a style rather than editing this adapter.
    """
    default_style = style or resolve_style()
    from agno.agents.langgraph import LangGraphAgent

    class ReasoningLangGraphAgent(LangGraphAgent):
        async def _arun_adapter_stream(self, input, *, history=None, _lg_config_override=None, **kwargs):
            from uuid import uuid4
            from agno.agents.langgraph.utils import build_messages_with_history
            from agno.models.response import ToolExecution
            from agno.run.agent import ReasoningContentDeltaEvent, RunContentEvent, ToolCallCompletedEvent, ToolCallStartedEvent

            if self.graph is None:
                raise ValueError("No graph provided to LangGraphAgent")

            # Per-instance style wins: the class is built before the graph is
            # (so that a missing LangGraphAgent is reported first), but the
            # model — and therefore the style — is only known afterwards.
            style = getattr(self, "_reasoning_style", None) or default_style

            run_id = kwargs.get("run_id", str(uuid4()))
            graph_input = None if input is None else {self.input_key: build_messages_with_history(input, history)}
            config = self._build_config(kwargs, override=_lg_config_override)

            async for event in self.graph.astream_events(graph_input, config=config, version="v2"):
                kind = event.get("event")

                if kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if not chunk:
                        continue

                    # The style yields this chunk's text pieces first and then a
                    # single coalesced thinking piece — the same order (and the
                    # same one-ReasoningContentDeltaEvent-per-chunk shape) this
                    # adapter emitted when it read provider fields itself.
                    for piece_kind, text in style.chunk_pieces(chunk):
                        if piece_kind == THINKING:
                            yield ReasoningContentDeltaEvent(
                                run_id=run_id, agent_id=self.get_id(), agent_name=self.name or "", reasoning_content=text
                            )
                        else:
                            yield RunContentEvent(run_id=run_id, agent_id=self.get_id(), agent_name=self.name or "", content=text)

                elif kind == "on_tool_start":
                    tool_name = event.get("name", "unknown")
                    tool_input = event.get("data", {}).get("input", {})
                    tool_run_id = event.get("run_id", str(uuid4()))
                    yield ToolCallStartedEvent(
                        run_id=run_id,
                        agent_id=self.get_id(),
                        agent_name=self.name or "",
                        tool=ToolExecution(
                            tool_call_id=tool_run_id,
                            tool_name=tool_name,
                            tool_args=tool_input if isinstance(tool_input, dict) else {"input": tool_input},
                        ),
                    )

                elif kind == "on_tool_end":
                    tool_name = event.get("name", "unknown")
                    output = event.get("data", {}).get("output", "")
                    tool_run_id = event.get("run_id", str(uuid4()))
                    result_str = str(output.content) if hasattr(output, "content") else str(output)
                    yield ToolCallCompletedEvent(
                        run_id=run_id,
                        agent_id=self.get_id(),
                        agent_name=self.name or "",
                        tool=ToolExecution(tool_call_id=tool_run_id, tool_name=tool_name, result=result_str),
                    )

    return ReasoningLangGraphAgent


async def build_langgraph_agent(
    graph_path: Optional[str] = None,
    origin: Optional[str] = None,
    include_browser_tools: bool = True,
    extra_tools: Optional[List[Any]] = None,
    extra_system_prompt: Optional[str] = None,
    skills: Optional[List[dict]] = None,
    knowledge: Optional[List[dict]] = None,
    mcp_servers: Optional[List[dict]] = None,
    tool_meta: Optional[Dict[str, dict]] = None,
    reasoning_style: Optional[str] = None,
):
    """Build an Agno `LangGraphAgent` (extended with reasoning/thinking support —
    see _make_reasoning_langgraph_agent_class) wrapping a compiled LangGraph
    graph — going through Agno's own adapter rather than driving
    `graph.astream()` by hand.

    With no `graph_path`/LANGGRAPH_GRAPH override configured, this builds the
    default DeepAgents+DeepSeek harness (build_deepagents_graph), passing the
    same generic context-injection params straight through (see build_agent's
    docstring — this module still has no idea where any of it came from). To
    point at your own graph instead (in which case these params are ignored —
    a custom graph has no default-harness tool binding of its own):

        # my_graph.py
        graph = my_own_compiled_langgraph_graph

        # .env
        LANGGRAPH_GRAPH=my_graph:graph

    Using Agno's own adapter means this shares stream_agno_events' event parsing
    with the native backend (LangGraphAgent already emits Agno's native
    RunContent/ToolCallStarted/ToolCallCompleted events, and the subclass here
    adds ReasoningContentDelta) and shares the same session db (_shared_db) for
    conversation continuity — one history mechanism for both backends, not a
    second bespoke one.

    `reasoning_style` names the communication style the adapter uses to tell
    thinking apart from answer text (see runtime/reasoning.py). With the default
    harness the style comes back from build_deepagents_graph, which knows the
    provider it just pinned; with a custom LANGGRAPH_GRAPH the provider is
    unknowable here, so it resolves from the name/env/auto default alone.

    `async def` because build_deepagents_graph is (MCP tool loading for
    DeepAgents needs an upfront await — see _mcp_tools_for_deepagents).

    Requires `langgraph` and `langchain-core` (and, for the default harness,
    `deepagents` + `langchain-deepseek`) to be installed.
    """
    # The style the adapter falls back to: the best that can be resolved before
    # a model exists (an explicit name, AGNO_REASONING_STYLE, or the auto
    # default). The default harness knows better — build_deepagents_graph
    # returns the style for the model it actually pinned — and that one replaces
    # this on the agent instance below.
    style = resolve_style(reasoning_style, None)

    try:
        ReasoningLangGraphAgent = _make_reasoning_langgraph_agent_class(style)
    except ImportError as exc:
        raise RuntimeError(
            "agno.agents.langgraph.LangGraphAgent is unavailable — install its "
            "dependencies with: pip install langgraph langchain-core"
        ) from exc

    configured_path = graph_path or os.getenv("LANGGRAPH_GRAPH")
    resolved_tool_meta: Dict[str, dict] = {}
    if configured_path:
        graph = _load_langgraph_graph(configured_path)
    else:
        graph, resolved_tool_meta, style = await build_deepagents_graph(
            origin=origin,
            include_browser_tools=include_browser_tools,
            extra_tools=extra_tools,
            extra_system_prompt=extra_system_prompt,
            skills=skills,
            knowledge=knowledge,
            mcp_servers=mcp_servers,
            tool_meta=tool_meta,
            reasoning_style=reasoning_style,
        )

    agent = ReasoningLangGraphAgent(name="LangGraph Agent", graph=graph, db=_shared_db)
    agent._tool_meta = resolved_tool_meta
    # Read back by _arun_adapter_stream above and by stream_agno_events, so one
    # style normalizes both the raw model chunks and the Agno run events built
    # from them.
    agent._reasoning_style = style
    return agent
