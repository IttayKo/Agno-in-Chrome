"""Streaming: turn a run (native Agno agent or LangGraph/DeepAgents graph)
into the transport-agnostic event dicts server.py forwards over the websocket.
"""

from typing import Any, Dict, List, Optional

from agno.exceptions import RunCancelledException

from .graph import build_langgraph_agent
from .reasoning import THINKING, resolve_style


async def stream_agno_events(agent, prompt: str, run_id: Optional[str] = None, session_id: Optional[str] = None):
    """Run an Agno agent (a native `Agent`, or an `agno.agents.langgraph.LangGraphAgent`
    wrapping a compiled LangGraph/DeepAgents graph — both expose the same
    RunContent/ToolCallStarted/ToolCallCompleted event vocabulary) and yield
    transport-agnostic event dicts:

      {'type': 'assistant.thinking', 'chunk': '...'}
      {'type': 'assistant.delta', 'chunk': '...'}
      {'type': 'tool.event', 'phase': 'call'|'output', 'tool_call_id', 'tool', 'args', 'result'?, 'error'?}
      {'type': 'assistant.done'}
      {'type': 'error', 'error': '...'}

    This is transport-agnostic on purpose: server.py just forwards each dict as
    JSON over the websocket, so the same logic drives both backends and the
    frontend doesn't need to know which one produced a given event.

    `run_id` (native Agent only — see build_agent) lets a caller stop this
    specific run cooperatively via `Agent.cancel_run(run_id)` (see server.py).
    That matters over a hard asyncio.Task.cancel(): Agno checks for cancellation
    at its own internal checkpoints and raises RunCancelledException, which
    unwinds cleanly and still lets the (partial) turn get persisted to session
    history — a raw task cancellation bypasses that and silently drops the turn.
    LangGraphAgent has no equivalent hook, so `run_id` is only passed for a
    native Agent; passed unconditionally it would collide with a same-named
    kwarg LangGraphAgent.arun() generates internally.

    `session_id` is only needed for LangGraphAgent: unlike a native Agent (whose
    session_id is bound once at construction in build_agent), LangGraphAgent is
    meant to be reused across conversations and takes session_id per call.
    """
    try:
        collected_text: List[str] = []
        stream_used = False
        # Caller-supplied per-tool-name metadata (see build_agent's tool_meta
        # param) merged verbatim into trace events — this module has no idea
        # what any of it means (e.g. that {"source": "site"} denotes a
        # domain-declared tool is entirely server.py's convention); it just
        # mechanically attaches whatever was registered for a given tool name.
        tool_meta: Dict[str, dict] = getattr(agent, '_tool_meta', None) or {}
        # The agent's communication style, attached at build time by
        # build_langgraph_agent; falls back to the env/auto default for a
        # native Agent, whose run events every registered style normalizes
        # identically.
        style = getattr(agent, '_reasoning_style', None) or resolve_style()

        # stream_events=True is required for a native Agent to yield ToolCallStarted/
        # ToolCallCompleted/ReasoningContentDelta events at all — without it,
        # arun(stream=True) only yields RunContent (plain text) deltas even though tool
        # calls still happen underneath, which is why the trace UI would otherwise
        # never see them. LangGraphAgent ignores this flag and always includes tool
        # events, but harmlessly accepts the kwarg either way.
        arun_kwargs = {"stream": True, "stream_events": True}
        if run_id is not None:
            arun_kwargs["run_id"] = run_id
        if session_id is not None:
            arun_kwargs["session_id"] = session_id

        async for event in agent.arun(prompt, **arun_kwargs):
            stream_used = True
            event_name = getattr(event, 'event', None)

            # Which run events carry model output, and how thinking is told
            # apart from answer text, is the reasoning style's business (see
            # runtime/reasoning.py) — no provider-specific field names here.
            if style.owns_event(event_name):
                for piece_kind, text in style.event_pieces(event):
                    if piece_kind == THINKING:
                        yield {'type': 'assistant.thinking', 'chunk': text}
                    else:
                        collected_text.append(text)
                        yield {'type': 'assistant.delta', 'chunk': text}
                continue

            if event_name == 'ToolCallStarted':
                tool = getattr(event, 'tool', None)
                tool_name = getattr(tool, 'tool_name', None) or 'tool'
                args = getattr(tool, 'tool_args', None)
                meta = tool_meta.get(tool_name) or {}
                args_extra = meta.get('args_extra')
                if args_extra:
                    args = {**(args or {}), **args_extra}
                yield {
                    'type': 'tool.event',
                    'phase': 'call',
                    'tool_call_id': getattr(tool, 'tool_call_id', None),
                    'tool': tool_name,
                    'args': args,
                    'source': meta.get('source', 'model'),
                }
                continue

            if event_name in ('ToolCallCompleted', 'ToolCallError'):
                tool = getattr(event, 'tool', None)
                tool_name = getattr(tool, 'tool_name', None) or 'tool'
                error = getattr(event, 'error', None) or (getattr(tool, 'tool_call_error', None) and 'Tool call failed')
                meta = tool_meta.get(tool_name) or {}
                yield {
                    'type': 'tool.event',
                    'phase': 'output',
                    'tool_call_id': getattr(tool, 'tool_call_id', None),
                    'tool': tool_name,
                    'args': getattr(tool, 'tool_args', None),
                    'result': getattr(tool, 'result', None) if event_name == 'ToolCallCompleted' else None,
                    'error': str(error) if error else None,
                    'source': meta.get('source', 'model'),
                }
                continue

        if not stream_used:
            fallback_kwargs = {"session_id": session_id} if session_id is not None else {}
            result = await agent.arun(prompt, **fallback_kwargs)
            content = getattr(result, 'content', str(result) if result is not None else '')
            for i in range(0, len(content), 40):
                yield {'type': 'assistant.delta', 'chunk': content[i:i + 40]}

        yield {'type': 'assistant.done'}
    except RunCancelledException:
        # A cooperative stop (Agent.cancel_run) — not an error. Whatever the run
        # produced before the cancel point is already persisted by Agno itself.
        yield {'type': 'assistant.done'}
    except Exception as e:
        yield {'type': 'error', 'error': str(e)}


async def stream_langgraph_events(
    prompt: str,
    graph_path: Optional[str] = None,
    thread_id: Optional[str] = None,
    origin: Optional[str] = None,
    include_browser_tools: bool = True,
    extra_tools: Optional[List[Any]] = None,
    extra_system_prompt: Optional[str] = None,
    skills: Optional[List[dict]] = None,
    knowledge: Optional[List[dict]] = None,
    mcp_servers: Optional[List[dict]] = None,
    tool_meta: Optional[Dict[str, dict]] = None,
    reasoning_style: Optional[str] = None,
    model_provider: Optional[str] = None,
    model_id: Optional[str] = None,
    base_url: Optional[str] = None,
    thinking=None,
    thinking_budget=None,
):
    """Drive a DeepAgents/LangGraph graph through Agno's own LangGraphAgent
    adapter and yield the same event protocol as stream_agno_events — this is a
    thin wrapper, not a separate implementation, since LangGraphAgent already
    speaks Agno's native event vocabulary.

    `thread_id` doubles as the session_id passed to LangGraphAgent, so a
    conversation's history is looked up from the same _shared_db as the native
    Agno backend. `origin`/`include_browser_tools` and the generic context
    params only matter when no custom graph is configured (see
    build_langgraph_agent) — they select and bind the default DeepAgents
    harness's tools to the current tab/context.

    `reasoning_style` names the communication style used to tell thinking apart
    from answer text (see runtime/reasoning.py); None means AGNO_REASONING_STYLE
    and then "auto", i.e. resolve it from the graph's model provider.

    `model_provider`/`model_id`/`base_url`/`thinking`/`thinking_budget` select
    and configure the model the default DeepAgents harness runs on (request
    value -> env var -> default; see runtime.models.get_langchain_model). Like
    the context params they do nothing when a custom graph is configured, which
    built its own model already.

    Known limitation, inherent to Agno's LangGraphAgent adapter itself (not
    something this wrapper patches around): it has no cooperative
    Agent.cancel_run equivalent, so Stop falls back to a hard task cancel for
    this backend (see server.py) rather than the graceful stop used natively.
    Reasoning/thinking content, which the stock adapter dropped, *is* streamed
    here — build_langgraph_agent wraps the graph in the reasoning subclass (see
    runtime.graph._make_reasoning_langgraph_agent_class) that emits it.
    """
    try:
        agent = await build_langgraph_agent(
            graph_path,
            origin=origin,
            include_browser_tools=include_browser_tools,
            extra_tools=extra_tools,
            extra_system_prompt=extra_system_prompt,
            skills=skills,
            knowledge=knowledge,
            mcp_servers=mcp_servers,
            tool_meta=tool_meta,
            reasoning_style=reasoning_style,
            model_provider=model_provider,
            model_id=model_id,
            base_url=base_url,
            thinking=thinking,
            thinking_budget=thinking_budget,
        )
    except Exception as exc:
        yield {'type': 'error', 'error': str(exc)}
        return

    async for evt in stream_agno_events(agent, prompt, session_id=thread_id):
        yield evt
