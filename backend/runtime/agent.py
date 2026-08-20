"""The native Agno backend: the shared session db, the local (non-browser)
tools, `build_agent`, and the plain `run_prompt` convenience entrypoint.
"""

import asyncio
import os
from typing import Any, Dict, List, Optional

from agno.agent import Agent
from agno.tools.function import Function
from agno.db.sqlite import SqliteDb

from .browser_tools import (
    _OriginState,
    build_agno_browser_functions,
    execute_local_cli,
)
from .context import _asset_index, _make_load_asset_tool_for_agno, _mcp_tools_for_agno
from .models import get_model, resolve_provider
from .reasoning import resolve_style

# Agno only adds prior turns to context when the agent has a db to read history from
# (add_history_to_context=True is a silent no-op without one). This is a module-level
# singleton, shared across every build_agent() call/process request, so a *new* Agent
# object per request still resumes the right conversation as long as the caller passes
# the same session_id — no need to keep Agent objects alive/cached.
#
# A SQLite file (not InMemoryDb) is used deliberately: uvicorn's --reload restarts the
# whole process on every source-file edit during development, which would silently wipe
# an in-memory store and look exactly like "the agent forgot everything" from the panel's
# side. A local file survives that.
#
# The path joins the *parent* directory of this package so the file stays where it
# always was, next to server.py (backend/agno_sessions.db), rather than moving into
# backend/runtime/ when this module moved here.
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_shared_db = SqliteDb(db_file=os.path.join(_BACKEND_DIR, "agno_sessions.db"))


async def calculate_product(a: int, b: int) -> str:
    """Multiply two integers and return the exact result."""
    return f"{a} * {b} = {a * b}"


async def execute_local_shell(command: str, timeout: int = 30) -> str:
    """Run a local shell command. This is the same capability that the extension bridge exposes for native/system tools."""
    if execute_local_cli is not None:
        return await execute_local_cli(command, timeout=timeout)
    proc = await asyncio.create_subprocess_shell(command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    out = (stdout.decode() if stdout else "") + ("\n" + stderr.decode() if stderr else "")
    return out.strip()


def build_agent(
    origin: Optional[str] = None,
    include_browser_tools: bool = False,
    api_key: Optional[str] = None,
    model_provider: Optional[str] = None,
    model_id: Optional[str] = None,
    base_url: Optional[str] = None,
    session_id: Optional[str] = None,
    extra_tools: Optional[List[Any]] = None,
    extra_system_prompt: Optional[str] = None,
    skills: Optional[List[dict]] = None,
    knowledge: Optional[List[dict]] = None,
    mcp_servers: Optional[List[dict]] = None,
    tool_meta: Optional[Dict[str, dict]] = None,
    reasoning_style: Optional[str] = None,
    thinking=None,
    thinking_budget=None,
) -> Agent:
    """Generic context-injection surface for this Agno agent: `extra_tools`
    (pre-built Function/Toolkit objects), `extra_system_prompt` (appended text),
    `skills`/`knowledge` (progressive-disclosure content, exposed via an
    auto-added load_asset tool), `mcp_servers` (connected automatically by
    Agno's own Agent.arun — see _mcp_tools_for_agno), and `tool_meta`
    (tool name -> arbitrary caller metadata, merged verbatim into trace events
    by stream_agno_events — e.g. {"source": "site"}).

    This function deliberately has NO knowledge of where any of this came
    from — no origin-manifest fetching, no reaching into server.py's cache.
    That's extension-specific glue and lives in server.py; this module stays
    usable by any caller with its own way of sourcing tools/mcps/skills (see
    the module docstring: "usable independently of the browser extension").

    `thinking`/`thinking_budget` ask the provider for reasoning output where it
    has to be asked (see runtime/models.py); unset sends nothing, as before.
    `reasoning_style` names the communication style stream_agno_events uses for
    this agent's run events (see runtime/reasoning.py); None resolves it from
    AGNO_REASONING_STYLE and then from the model provider selected here.
    """
    origin = origin or os.getenv("BROWSER_ORIGIN", "https://example.com")
    origin_state = _OriginState(origin)
    provider = resolve_provider(model_provider)
    model = get_model(
        api_key=api_key,
        provider=provider,
        model_id=model_id,
        base_url=base_url,
        thinking=thinking,
        thinking_budget=thinking_budget,
    )

    tools = [
        Function(
            name="calculate_product",
            description="Multiply two integers and return the result.",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "First integer"},
                    "b": {"type": "integer", "description": "Second integer"},
                },
                "required": ["a", "b"],
            },
            entrypoint=calculate_product,
        ),
        Function(
            name="execute_local_shell",
            description="Execute a local shell command on the host machine and return stdout/stderr.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 30},
                },
                "required": ["command"],
            },
            entrypoint=execute_local_shell,
        ),
    ]

    if include_browser_tools:
        tools.extend(build_agno_browser_functions(origin_state, session_id or origin))

    if extra_tools:
        tools.extend(extra_tools)
    if mcp_servers:
        tools.extend(_mcp_tools_for_agno(mcp_servers))
    if skills or knowledge:
        tools.append(_make_load_asset_tool_for_agno(skills, knowledge))

    system_message = (
        f"You are running in a browser-aware environment for {origin}. "
        "Use the local tools for non-browser computation and host execution. "
        "Use browser tools only when the user explicitly wants page interaction."
    )
    if extra_system_prompt:
        system_message = f"{system_message}\n\n{extra_system_prompt}"
    asset_index = _asset_index(skills, knowledge)
    if asset_index:
        system_message = f"{system_message}\n\n{asset_index}"

    agent = Agent(
        model=model,
        tools=tools,
        system_message=system_message,
        instructions=[
            "Use calculate_product for arithmetic.",
            "Use execute_local_shell for local host commands.",
            "Use browser_scroll_viewport or browser_click_coordinates only for explicit browser actions.",
        ],
        markdown=True,
        debug_mode=False,
        add_datetime_to_context=True,
        db=_shared_db,
        session_id=session_id,
        add_history_to_context=True,
    )
    agent._tool_meta = tool_meta or {}
    # Read back by stream_agno_events. Every registered style normalizes a
    # native Agent's run events identically (they all share event_pieces), so
    # this is inert today for the native backend — it exists so that a request
    # can carry a style for both backends through the same field, and so a
    # style that *does* need to special-case Agno events later has somewhere to
    # be attached.
    agent._reasoning_style = resolve_style(reasoning_style, provider)
    return agent


async def run_prompt(prompt: str, origin: Optional[str] = None, include_browser_tools: Optional[bool] = None) -> str:
    if include_browser_tools is None:
        include_browser_tools = os.getenv("AGNO_USE_BROWSER_TOOLS", "0") == "1"
    agent = build_agent(origin=origin, include_browser_tools=include_browser_tools)
    result = await agent.arun(prompt)
    return getattr(result, 'content', str(result))


async def _demo() -> None:
    prompt = "Calculate the value of 583 * 22 and then tell me the result."
    print(await run_prompt(prompt, origin="https://example.com"))
