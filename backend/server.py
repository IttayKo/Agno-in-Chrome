"""
FastAPI bridge for Agno <-> Chrome extension.
Provides:
- POST /agent/context to register per-origin system_message / skills / tools
- POST /agent/enqueue for local code to enqueue a browser action
- POST /agent/stream used by extension long-poll to fetch next action for a given origin
- POST /agent/confirm used by extension to confirm action_id results
- WebSocket /agent/ws to broadcast actions to connected clients

Also exposes helper Python functions that Agno tools can call directly: execute_local_cli, browser_click_coordinates, browser_scroll_viewport

This is an in-memory prototype; replace with persistent queues and auth for production.
"""

import asyncio
import os
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uuid
import time
import shlex
import subprocess
import httpx
from typing import Dict, Any, List, Optional

# Load .env at process start, not as a side effect of the first request that
# happens to lazily `from agno_runtime import ...`. AGNO_BACKEND (and every
# other env-configured default) is read via os.getenv() before that lazy
# import ever runs, so without this, the very first request on a freshly
# started server silently used the "agno" backend regardless of .env — only
# self-correcting once some request's lazy import loaded agno_runtime (which
# calls load_dotenv() at its own module level) as a side effect.
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)

app = FastAPI(title="Agno browser bridge")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Per-origin context storage
contexts: Dict[str, Dict[str, Any]] = {}

# Pending actions per origin (FIFO)
pending_actions: Dict[str, List[Dict[str, Any]]] = {}

# Mapping action_id -> asyncio.Event and result
action_events: Dict[str, asyncio.Event] = {}
action_results: Dict[str, Any] = {}

# Connected websocket clients: list of {'ws': WebSocket, 'origin': Optional[str],
# 'browser_id': str, 'connected_at': float}. Each entry is one physical browser
# instance's extension bridging into this server (typically just one, but the
# protocol supports several — e.g. more than one Chrome profile/window each
# running the extension). See list_connected_browsers/select_browser below.
ws_clients: List[Dict[str, Any]] = []

# When set, routes browser actions to this one connected browser regardless of
# origin subscription (see broadcast_action) — set via select_browser/POST
# /agent/browsers/select. None means "fall back to origin-based broadcast",
# the original (and still default) behavior.
selected_browser_id: Optional[str] = None

# In-memory per-session agent execution plan (see update_plan tool). Keyed by
# whatever id the caller passes (session_id, falling back to origin) — purely
# a display/bookkeeping aid for the panel UI and the model itself, no browser
# action involved.
agent_plans: Dict[str, Dict[str, Any]] = {}

class ContextPayload(BaseModel):
    origin: str
    system_message: Optional[str] = None
    skills: Optional[List[str]] = None
    tools: Optional[List[str]] = None

class EnqueuePayload(BaseModel):
    origin: str
    tool: str
    args: Dict[str, Any] = {}

class ConfirmPayload(BaseModel):
    action_id: str
    result: Any

class ChatPayload(BaseModel):
    prompt: str
    origin: Optional[str] = None
    url: Optional[str] = None              # full current-tab URL, for context_registry path-level matching
    include_browser_tools: bool = False
    backend: Optional[str] = None          # 'agno' (default) or 'langgraph'
    model_provider: Optional[str] = None   # e.g. 'openai' | 'gemini' | 'deepseek' | 'vllm'
    model_id: Optional[str] = None         # overrides AGNO_MODEL
    base_url: Optional[str] = None         # for 'vllm'/'openai_compatible' providers
    langgraph_graph: Optional[str] = None  # overrides LANGGRAPH_GRAPH, 'module.path:graph_var'
    thread_id: Optional[str] = None        # LangGraph checkpointer thread id
    session_id: Optional[str] = None       # stable per-conversation id so later turns see earlier ones

@app.get('/', response_class=HTMLResponse)
async def read_root():
    ui_path = Path(__file__).resolve().parent / 'ui' / 'index.html'
    return HTMLResponse(ui_path.read_text())

app.mount('/ui', StaticFiles(directory=str(Path(__file__).resolve().parent / 'ui')), name='ui')

@app.post('/agent/chat')
async def chat(payload: ChatPayload):
    backend = (payload.backend or os.getenv('AGNO_BACKEND', 'agno')).strip().lower()
    try:
        if backend == 'langgraph':
            from agno_runtime import stream_langgraph_events
            ctx_kwargs = await build_context_kwargs(payload.url, payload.origin, for_deepagents=True)
            response_parts: List[str] = []
            trace: List[Dict[str, Any]] = []
            async for evt in stream_langgraph_events(
                payload.prompt,
                graph_path=payload.langgraph_graph,
                thread_id=payload.thread_id or payload.session_id or payload.origin,
                origin=payload.origin,
                include_browser_tools=payload.include_browser_tools,
                extra_tools=ctx_kwargs.get('extra_tools'),
                extra_system_prompt=ctx_kwargs.get('extra_system_prompt'),
                skills=ctx_kwargs.get('skills'),
                knowledge=ctx_kwargs.get('knowledge'),
                mcp_servers=ctx_kwargs.get('mcp_servers'),
                tool_meta=ctx_kwargs.get('tool_meta'),
            ):
                if evt.get('type') == 'assistant.delta':
                    response_parts.append(evt.get('chunk') or '')
                elif evt.get('type') == 'tool.event':
                    trace.append({'role': 'tool', 'name': evt.get('tool'), 'args': evt.get('args'), 'content': evt.get('result') or evt.get('error')})
                elif evt.get('type') == 'error':
                    return JSONResponse({'status': 'error', 'error': evt.get('error')}, status_code=500)
            return {'response': ''.join(response_parts), 'tool_trace': trace, 'status': 'ok'}

        from agno_runtime import build_agent
        ctx_kwargs = await build_context_kwargs(payload.url, payload.origin, for_deepagents=False)
        agent = build_agent(
            origin=payload.origin,
            include_browser_tools=payload.include_browser_tools,
            model_provider=payload.model_provider,
            model_id=payload.model_id,
            base_url=payload.base_url,
            session_id=payload.session_id,
            extra_tools=ctx_kwargs.get('extra_tools'),
            extra_system_prompt=ctx_kwargs.get('extra_system_prompt'),
            skills=ctx_kwargs.get('skills'),
            knowledge=ctx_kwargs.get('knowledge'),
            mcp_servers=ctx_kwargs.get('mcp_servers'),
            tool_meta=ctx_kwargs.get('tool_meta'),
        )
        result = await agent.arun(payload.prompt)
        response = getattr(result, 'content', str(result))
        trace = []
        for message in getattr(result, 'messages', []):
            if getattr(message, 'tool_name', None):
                trace.append({
                    'role': 'tool',
                    'name': message.tool_name,
                    'args': getattr(message, 'tool_args', None),
                    'content': getattr(message, 'content', None),
                })
            elif getattr(message, 'role', None) == 'assistant' and getattr(message, 'content', None):
                trace.append({'role': 'assistant', 'content': message.content})
        return {'response': response, 'tool_trace': trace, 'status': 'ok'}
    except Exception as exc:
        return JSONResponse({'status': 'error', 'error': str(exc)}, status_code=500)

@app.post('/agent/context')
async def set_context(payload: ContextPayload):
    contexts[payload.origin] = {
        'system_message': payload.system_message,
        'skills': payload.skills or [],
        'tools': payload.tools or []
    }
    # ensure queue exists
    pending_actions.setdefault(payload.origin, [])
    return {'ok': True}

@app.get('/agent/context')
async def get_context(origin: str):
    return contexts.get(origin, {})

async def internal_enqueue_action(origin: str, tool: str, args: Dict[str, Any]):
    action_id = str(uuid.uuid4())
    action = { 'action_id': action_id, 'tool': tool, 'args': args, 'origin': origin }
    pending_actions.setdefault(origin, []).append(action)
    # create an event to wait for confirmation if caller wants to await
    action_events[action_id] = asyncio.Event()
    # broadcast to WS clients (best-effort)
    await broadcast_action(action)
    return action_id, action

@app.post('/agent/enqueue')
async def enqueue_action(payload: EnqueuePayload):
    action_id, _ = await internal_enqueue_action(payload.origin, payload.tool, payload.args)
    return {'action_id': action_id}

@app.post('/agent/stream')
async def stream_for_extension(req: Request):
    # Expect JSON body with optional origin
    try:
        body = await req.json()
    except Exception:
        body = {}
    origin = body.get('origin')
    if not origin:
        return JSONResponse({'action': 'wait'})
    queue = pending_actions.get(origin, [])
    if queue:
        action = queue.pop(0)
        return JSONResponse(action)
    return JSONResponse({'action': 'wait'})

@app.post('/agent/confirm')
async def confirm_result(payload: ConfirmPayload):
    action_results[payload.action_id] = payload.result
    ev = action_events.get(payload.action_id)
    if ev:
        ev.set()
    return {'ok': True}

@app.get('/agent/result')
async def get_result(action_id: str):
    if action_id in action_results:
        return {'status': 'done', 'result': action_results[action_id]}
    # Check if action is still pending
    if action_id in action_events:
        ev = action_events[action_id]
        if ev.is_set():
            return {'status': 'done', 'result': action_results.get(action_id)}
        return {'status': 'pending'}
    return {'status': 'unknown'}

@app.websocket('/agent/ws')
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    client = {'ws': ws, 'origin': None, 'browser_id': str(uuid.uuid4()), 'connected_at': time.time()}
    ws_clients.append(client)
    try:
        while True:
            data = await ws.receive_text()
            # Expect a subscription message like: {"subscribe": "https://example.com"}
            try:
                import json
                msg = json.loads(data)
                if isinstance(msg, dict) and 'subscribe' in msg:
                    client['origin'] = msg.get('subscribe')
                    await ws.send_text(json_dump({'subscribed': client['origin'], 'browser_id': client['browser_id']}))
                else:
                    print('WS recv (ignored):', msg)
            except Exception:
                print('WS recv (raw):', data)
    except WebSocketDisconnect:
        if client in ws_clients:
            ws_clients.remove(client)
        global selected_browser_id
        if selected_browser_id == client['browser_id']:
            selected_browser_id = None


@app.websocket('/agent/chat_ws')
async def chat_stream_ws(ws: WebSocket):
    """WebSocket endpoint for streaming chat responses and live tool traces.

    Protocol: client sends a JSON message with {prompt, origin, include_browser_tools}
    Server responds with JSON messages of form:
      {type: 'assistant.thinking', chunk: '...'}                                    reasoning/thinking delta
      {type: 'assistant.delta', chunk: '...'}                                       final output text delta
      {type: 'assistant.done'}
      {type: 'tool.event', phase: 'call', tool_call_id, tool, args}                 tool invocation started
      {type: 'tool.event', phase: 'output', tool_call_id, tool, args, result, error} tool invocation finished
      {type: 'error', error: '...'}

    Event classification is driven by Agno's own run-event taxonomy
    (agno.run.agent.RunEvent) rather than guessing from ad-hoc attributes, so
    thinking, tool calls, tool outputs, and final output text are always
    distinguishable instead of being merged into one generic content stream.
    """
    await ws.accept()
    try:
        init_msg = await ws.receive_text()
        import json
        payload = json.loads(init_msg)
        prompt = payload.get('prompt')
        origin = payload.get('origin')
        url = payload.get('url')
        include_browser_tools = payload.get('include_browser_tools', False)
        backend = (payload.get('backend') or os.getenv('AGNO_BACKEND', 'agno')).strip().lower()
        model_provider = payload.get('model_provider')
        model_id = payload.get('model_id')
        base_url = payload.get('base_url')
        graph_path = payload.get('langgraph_graph')
        session_id = payload.get('session_id')
        thread_id = payload.get('thread_id') or session_id or origin
        run_id = str(uuid.uuid4())
        print(f"[chat_ws] session_id={session_id!r} origin={origin!r} run_id={run_id!r} prompt={prompt[:60]!r}", flush=True)

        async def run_and_stream():
            try:
                await ws.send_text(json.dumps({'type': 'assistant.start'}))

                if backend == 'langgraph':
                    from agno_runtime import stream_langgraph_events
                    ctx_kwargs = await build_context_kwargs(url, origin, for_deepagents=True)
                    event_source = stream_langgraph_events(
                        prompt,
                        graph_path=graph_path,
                        thread_id=thread_id,
                        origin=origin,
                        include_browser_tools=include_browser_tools,
                        extra_tools=ctx_kwargs.get('extra_tools'),
                        extra_system_prompt=ctx_kwargs.get('extra_system_prompt'),
                        skills=ctx_kwargs.get('skills'),
                        knowledge=ctx_kwargs.get('knowledge'),
                        mcp_servers=ctx_kwargs.get('mcp_servers'),
                        tool_meta=ctx_kwargs.get('tool_meta'),
                    )
                else:
                    from agno_runtime import build_agent, stream_agno_events
                    ctx_kwargs = await build_context_kwargs(url, origin, for_deepagents=False)
                    agent = build_agent(
                        origin=origin,
                        include_browser_tools=include_browser_tools,
                        model_provider=model_provider,
                        model_id=model_id,
                        base_url=base_url,
                        session_id=session_id,
                        extra_tools=ctx_kwargs.get('extra_tools'),
                        extra_system_prompt=ctx_kwargs.get('extra_system_prompt'),
                        skills=ctx_kwargs.get('skills'),
                        knowledge=ctx_kwargs.get('knowledge'),
                        mcp_servers=ctx_kwargs.get('mcp_servers'),
                        tool_meta=ctx_kwargs.get('tool_meta'),
                    )
                    # run_id (Agno backend only) is what lets a later {"cancel": true}
                    # message stop this specific run cooperatively via Agent.cancel_run
                    # below, instead of hard-cancelling the task.
                    event_source = stream_agno_events(agent, prompt, run_id=run_id)

                async for evt in event_source:
                    await ws.send_text(json.dumps(evt))
                    if evt.get('type') in ('assistant.delta', 'assistant.thinking'):
                        await asyncio.sleep(0.02)
            except Exception as e:
                try:
                    await ws.send_text(json.dumps({'type': 'error', 'error': str(e)}))
                except Exception:
                    pass

        stream_task = asyncio.create_task(run_and_stream())

        try:
            # Keep the websocket open until client disconnects or asks to cancel.
            while True:
                msg = await ws.receive_text()
                try:
                    parsed = json.loads(msg)
                except Exception:
                    continue
                if isinstance(parsed, dict) and parsed.get('cancel'):
                    if backend == 'langgraph':
                        # No equivalent cooperative-cancel hook wired up for the
                        # LangGraph adapter yet — fall back to a hard stop.
                        stream_task.cancel()
                    else:
                        # Agent.cancel_run is cooperative: Agno checks for it at its own
                        # internal checkpoints (including around tool calls) and raises
                        # RunCancelledException, which unwinds cleanly and still lets
                        # Agno persist whatever the run produced up to that point. A raw
                        # stream_task.cancel() bypasses that path entirely and was
                        # silently dropping the interrupted turn from session history.
                        from agno.agent import Agent
                        Agent.cancel_run(run_id)
        finally:
            if not stream_task.done():
                stream_task.cancel()
    except WebSocketDisconnect:
        return
    except Exception as exc:
        try:
            await ws.send_text(json.dumps({'type': 'error', 'error': str(exc)}))
        except Exception:
            pass
        return

async def broadcast_action(action: Dict[str, Any]):
    to_remove = []
    # A selected browser (see select_browser) takes an action regardless of its
    # own origin subscription — the caller explicitly chose this connection.
    # Otherwise, fall back to the original origin-based broadcast filter.
    targets = ws_clients
    if selected_browser_id:
        targets = [c for c in ws_clients if c.get('browser_id') == selected_browser_id]
    for client in targets:
        try:
            if not selected_browser_id and client.get('origin') and client.get('origin') != action.get('origin'):
                continue
            await client['ws'].send_text(json_dump(action))
        except Exception:
            to_remove.append(client)
    for c in to_remove:
        if c in ws_clients:
            ws_clients.remove(c)

def json_dump(obj: Any) -> str:
    import json
    return json.dumps(obj)

# Helper functions for tools
async def execute_local_cli(command: str, timeout: int = 30) -> str:
    """Run a local CLI command and return stdout/stderr"""
    proc = await asyncio.create_subprocess_shell(command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        return 'timeout'
    out = (stdout.decode() if stdout else '') + (('\n' + stderr.decode()) if stderr else '')
    return out.strip()

async def _await_action_result(action_id: str, timeout: int = 30) -> Any:
    ev = action_events.get(action_id)
    if not ev:
        # No event was created; create and wait
        action_events[action_id] = asyncio.Event()
        ev = action_events[action_id]
    try:
        await asyncio.wait_for(ev.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        return {'status': 'timeout'}
    return action_results.get(action_id)

async def browser_click_coordinates(origin: str, x: int, y: int, timeout: int = 30, **kwargs: Any) -> Any:
    action_id, _ = await internal_enqueue_action(origin, 'click', {'x': x, 'y': y})
    return await _await_action_result(action_id, timeout=timeout)

async def browser_scroll_viewport(
    origin: str,
    deltaY: Optional[int] = None,
    timeout: int = 30,
    **kwargs: Any,
) -> Any:
    if deltaY is None:
        deltaY = kwargs.get('delta_y')
    if deltaY is None:
        deltaY = kwargs.get('y_delta')
    if deltaY is None:
        deltaY = kwargs.get('y')
    if deltaY is None:
        deltaY = kwargs.get('delta')
    if deltaY is None:
        raise ValueError('scroll amount is required')
    action_id, _ = await internal_enqueue_action(origin, 'scroll', {'deltaY': deltaY})
    return await _await_action_result(action_id, timeout=timeout)


async def browser_tool_call(origin: str, tool: str, args: Dict[str, Any], timeout: int = 30) -> Any:
    """Enqueue a browser-side action for the given origin and wait for the result."""
    if not origin:
        raise ValueError('origin is required for browser tools')
    action_id, _ = await internal_enqueue_action(origin, tool, args)
    return await _await_action_result(action_id, timeout=timeout)


async def browser_navigate(origin: str, url: str, timeout: int = 30, **kwargs: Any) -> Any:
    return await browser_tool_call(origin, 'navigate', {'url': url}, timeout=timeout)


async def browser_tabs_context(origin: str, timeout: int = 30) -> Any:
    return await browser_tool_call(origin, 'tabs_context_mcp', {}, timeout=timeout)


async def browser_tabs_create(origin: str, url: str, timeout: int = 30) -> Any:
    return await browser_tool_call(origin, 'tabs_create_mcp', {'url': url}, timeout=timeout)


async def browser_tabs_close(origin: str, tab_id: int, timeout: int = 30) -> Any:
    return await browser_tool_call(origin, 'tabs_close_mcp', {'tab_id': tab_id}, timeout=timeout)


async def browser_get_page_text(origin: str, timeout: int = 30) -> Any:
    return await browser_tool_call(origin, 'get_page_text', {}, timeout=timeout)


async def browser_read_page(origin: str, selector: Optional[str] = None, timeout: int = 30) -> Any:
    return await browser_tool_call(origin, 'read_page', {'selector': selector}, timeout=timeout)


async def browser_find(origin: str, description: str, timeout: int = 30) -> Any:
    return await browser_tool_call(origin, 'find', {'description': description}, timeout=timeout)


async def browser_read_console_messages(origin: str, timeout: int = 30) -> Any:
    return await browser_tool_call(origin, 'read_console_messages', {}, timeout=timeout)


async def browser_read_network_requests(origin: str, timeout: int = 30) -> Any:
    return await browser_tool_call(origin, 'read_network_requests', {}, timeout=timeout)


async def browser_form_input(origin: str, selector: str, value: str, field_type: str = 'text', timeout: int = 30) -> Any:
    return await browser_tool_call(origin, 'form_input', {'selector': selector, 'value': value, 'field_type': field_type}, timeout=timeout)


async def browser_javascript_tool(origin: str, code: str, timeout: int = 30) -> Any:
    return await browser_tool_call(origin, 'javascript_tool', {'code': code}, timeout=timeout)


async def call_manifest_http_tool(origin: str, endpoint: str, method: str, args: dict) -> Any:
    """Execute a domain-declared 'http' tool. Restricted to the manifest's own
    origin as a basic SSRF guard — a manifest shouldn't be able to declare a
    tool that makes this server call an arbitrary internal/external address."""
    if not endpoint or not endpoint.startswith(origin):
        return {"status": "error", "error": "endpoint must be same-origin as the manifest that declared it"}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            if (method or "POST").upper() == "GET":
                resp = await client.get(endpoint, params=args or {})
            else:
                resp = await client.post(endpoint, json=args or {})
            try:
                return resp.json()
            except Exception:
                return resp.text
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _wrap_browser_js_code(code: str, args: dict) -> str:
    """A context-registry 'browser_tool' entry's code may reference call-time
    params — expose them as a local `args` object rather than requiring the
    domain author to hand-write parameter substitution."""
    import json as _json
    return f"const args = {_json.dumps(args or {})};\n{code or ''}"


def _build_context_tools_for_agno(origin: str, tools_cfg: List[dict], browser_tools_cfg: List[dict]):
    """Build live Agno Function objects for a matched context-registry
    template's plugins.tools ('http', executed server-side via
    call_manifest_http_tool) and plugins.browser_tools ('browser_js', executed
    in the attached tab via browser_javascript_tool). Every such tool is
    auto-enabled (this project's trust model for domain-supplied tools) but
    tagged source:"site" in tool_meta so the chat trace UI shows it distinctly
    — browser_js tools additionally carry their source code in args_extra so
    the trace can show exactly what ran."""
    from agno.tools.function import Function
    from agno_runtime import normalize_tool_schema

    built: List[Any] = []
    meta: Dict[str, dict] = {}

    for t in tools_cfg or []:
        name = t.get('name')
        if not name:
            continue
        endpoint, method = t.get('endpoint'), t.get('method', 'POST')
        schema = normalize_tool_schema(t.get('parameters'))

        async def _entrypoint(_endpoint=endpoint, _method=method, **kwargs):
            return await call_manifest_http_tool(origin, _endpoint, _method, kwargs)

        built.append(Function(name=name, description=t.get('description') or name, parameters=schema, entrypoint=_entrypoint))
        meta[name] = {'source': 'site'}

    for t in browser_tools_cfg or []:
        name = t.get('name')
        if not name:
            continue
        code = t.get('code') or ''
        schema = normalize_tool_schema(t.get('parameters'))

        async def _entrypoint(_code=code, **kwargs):
            return await browser_javascript_tool(origin, _wrap_browser_js_code(_code, kwargs))

        built.append(Function(name=name, description=t.get('description') or name, parameters=schema, entrypoint=_entrypoint))
        meta[name] = {'source': 'site', 'args_extra': {'_site_code': code}}

    return built, meta


async def _build_context_tools_for_deepagents(origin: str, tools_cfg: List[dict], browser_tools_cfg: List[dict]):
    """Same as _build_context_tools_for_agno but returns LangChain
    StructuredTools for the DeepAgents/LangGraph backend, which has no
    signature-introspection fallback the way Agno does — args_schema is built
    explicitly via agno_runtime.json_schema_to_pydantic."""
    from langchain_core.tools import StructuredTool
    from agno_runtime import normalize_tool_schema, json_schema_to_pydantic

    built: List[Any] = []
    meta: Dict[str, dict] = {}

    for t in tools_cfg or []:
        name = t.get('name')
        if not name:
            continue
        endpoint, method = t.get('endpoint'), t.get('method', 'POST')
        schema = normalize_tool_schema(t.get('parameters'))
        args_model = json_schema_to_pydantic(f'{name}_Args', schema)

        async def _run(_endpoint=endpoint, _method=method, **kwargs):
            return await call_manifest_http_tool(origin, _endpoint, _method, kwargs)

        built.append(StructuredTool.from_function(coroutine=_run, name=name, description=t.get('description') or name, args_schema=args_model))
        meta[name] = {'source': 'site'}

    for t in browser_tools_cfg or []:
        name = t.get('name')
        if not name:
            continue
        code = t.get('code') or ''
        schema = normalize_tool_schema(t.get('parameters'))
        args_model = json_schema_to_pydantic(f'{name}_Args', schema)

        async def _run(_code=code, **kwargs):
            return await browser_javascript_tool(origin, _wrap_browser_js_code(_code, kwargs))

        built.append(StructuredTool.from_function(coroutine=_run, name=name, description=t.get('description') or name, args_schema=args_model))
        meta[name] = {'source': 'site', 'args_extra': {'_site_code': code}}

    return built, meta


async def build_context_kwargs(url: Optional[str], origin: Optional[str], for_deepagents: bool = False) -> Dict[str, Any]:
    """Translate a matched context_registry template (see context_registry.py)
    into the generic context-injection kwargs agno_runtime.build_agent /
    build_deepagents_graph accept. This — not agno_runtime.py — is where the
    "there's a URL-template registry" concept lives, per the project's
    constraint that the runtime module stay usable by any caller with its own
    way of sourcing tools/mcps/skills."""
    match_url = url or origin
    if not match_url:
        return {'matched': False, 'matched_patterns': []}

    from context_registry import match_context
    ctx = match_context(match_url)
    if not ctx.get('matched'):
        return {'matched': False, 'matched_patterns': []}

    effective_origin = origin or match_url
    if for_deepagents:
        tools, meta = await _build_context_tools_for_deepagents(effective_origin, ctx['tools'], ctx['browser_tools'])
    else:
        tools, meta = _build_context_tools_for_agno(effective_origin, ctx['tools'], ctx['browser_tools'])

    return {
        'matched': True,
        'matched_patterns': ctx['matched_patterns'],
        'extra_tools': tools or None,
        'tool_meta': meta or None,
        'extra_system_prompt': ctx.get('instructions'),
        'skills': ctx.get('skills') or None,
        'knowledge': ctx.get('knowledge') or None,
        'mcp_servers': ctx.get('mcps') or None,
    }


@app.get('/agent/context_summary')
async def context_summary(url: Optional[str] = None, origin: Optional[str] = None):
    """Cheap, no-tool-building lookup for the extension to show a "context
    changed" note in the chat when navigating to a new page — a plain
    match_context() call, not the full build_context_kwargs() (which
    instantiates Function/StructuredTool objects, wasted work here)."""
    match_url = url or origin
    if not match_url:
        return {'matched': False, 'matched_patterns': [], 'tool_names': []}
    from context_registry import match_context
    ctx = match_context(match_url)
    tool_names = [t.get('name') for t in (ctx.get('tools') or []) + (ctx.get('browser_tools') or []) if t.get('name')]
    return {
        'matched': ctx.get('matched', False),
        'matched_patterns': ctx.get('matched_patterns', []),
        'tool_names': tool_names,
        'has_instructions': bool(ctx.get('instructions')),
        'skill_names': [s.get('name') for s in (ctx.get('skills') or []) if s.get('name')],
        'knowledge_names': [k.get('name') for k in (ctx.get('knowledge') or []) if k.get('name')],
    }


async def browser_resize_window(origin: str, width: int, height: int, timeout: int = 30) -> Any:
    return await browser_tool_call(origin, 'resize_window', {'width': width, 'height': height}, timeout=timeout)


async def browser_gif_creator(origin: str, duration_ms: int = 3000, title: Optional[str] = None, timeout: int = 30) -> Any:
    return await browser_tool_call(origin, 'gif_creator', {'duration_ms': duration_ms, 'title': title}, timeout=timeout)


async def browser_shortcuts_list(origin: str, timeout: int = 30) -> Any:
    return await browser_tool_call(origin, 'shortcuts_list', {}, timeout=timeout)


async def browser_shortcuts_execute(origin: str, shortcut_name: str, timeout: int = 30) -> Any:
    return await browser_tool_call(origin, 'shortcuts_execute', {'shortcut_name': shortcut_name}, timeout=timeout)


async def browser_batch(origin: str, operations: List[Dict[str, Any]], timeout: int = 30) -> Any:
    return await browser_tool_call(origin, 'browser_batch', {'operations': operations}, timeout=timeout)


async def browser_computer(origin: str, action: str, **kwargs: Any) -> Any:
    return await browser_tool_call(origin, 'computer', {'action': action, **kwargs}, timeout=30)


async def browser_switch_browser(origin: str, tab_id: int, timeout: int = 30) -> Any:
    return await browser_tool_call(origin, 'switch_browser', {'tab_id': tab_id}, timeout=timeout)


async def browser_upload_image(
    origin: str,
    selector: str,
    image_base64: Optional[str] = None,
    image_url: Optional[str] = None,
    filename: Optional[str] = None,
    mime_type: Optional[str] = None,
    timeout: int = 60,
) -> Any:
    # Downloading the image to disk before handing it to DOM.setFileInputFiles can
    # take longer than the other, purely in-page actions, hence the higher default timeout.
    return await browser_tool_call(origin, 'upload_image', {
        'selector': selector,
        'image_base64': image_base64,
        'image_url': image_url,
        'filename': filename,
        'mime_type': mime_type,
    }, timeout=timeout)


def list_connected_browsers_data() -> List[Dict[str, Any]]:
    return [
        {
            'browser_id': c['browser_id'],
            'origin': c.get('origin'),
            'connected_at': c.get('connected_at'),
            'selected': c['browser_id'] == selected_browser_id,
        }
        for c in ws_clients
    ]


async def browser_list_connected() -> Any:
    return list_connected_browsers_data()


async def browser_select(browser_id: str) -> Any:
    global selected_browser_id
    if not any(c['browser_id'] == browser_id for c in ws_clients):
        return {'status': 'error', 'error': f"No connected browser with id '{browser_id}'"}
    selected_browser_id = browser_id
    return {'status': 'ok', 'browser_id': browser_id}


@app.get('/agent/browsers')
async def get_connected_browsers():
    return {'browsers': list_connected_browsers_data()}


class SelectBrowserPayload(BaseModel):
    browser_id: str


@app.post('/agent/browsers/select')
async def select_browser_endpoint(payload: SelectBrowserPayload):
    return await browser_select(payload.browser_id)


class UpdatePlanPayload(BaseModel):
    plan_id: str
    plan: List[Dict[str, Any]]
    explanation: Optional[str] = None


async def update_plan(plan_id: str, plan: List[Dict[str, Any]], explanation: Optional[str] = None) -> Any:
    """Store/replace the agent's current execution plan for `plan_id` (a
    session_id, or an origin when no session_id is available). `plan` is a
    list of {"step": str, "status": "pending"|"in_progress"|"completed"}."""
    agent_plans[plan_id] = {'plan': plan, 'explanation': explanation}
    return {'status': 'ok', 'plan_id': plan_id, 'plan': plan, 'explanation': explanation}


@app.post('/agent/plan')
async def update_plan_endpoint(payload: UpdatePlanPayload):
    return await update_plan(payload.plan_id, payload.plan, payload.explanation)


@app.get('/agent/plan')
async def get_plan(plan_id: str):
    return agent_plans.get(plan_id, {'plan': [], 'explanation': None})


# Simple CLI to run server: uvicorn server:app --reload --port 8000
if __name__ == '__main__':
    import uvicorn
    uvicorn.run('server:app', host='127.0.0.1', port=8000, reload=True)
