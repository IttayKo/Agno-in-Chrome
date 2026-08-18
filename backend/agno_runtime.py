"""Generic Agno runtime that is usable independently of the browser extension.

The Chrome extension is just one adapter instance on top of the same runtime.
This module defines the actual model/tool layer and can be used either:
- directly from Python in local workflows
- via the browser bridge when browser-specific tools are needed

Environment variables:
  OPENAI_API_KEY  required for real model calls
  AGNO_MODEL      defaults to gpt-4o-mini
  BROWSER_ORIGIN  defaults to https://example.com
  AGNO_USE_BROWSER_TOOLS  set to 1 to include browser tools by default
"""

import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv
from agno.agent import Agent
from agno.tools.function import Function
from agno.db.sqlite import SqliteDb
from agno.exceptions import RunCancelledException

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)

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
_shared_db = SqliteDb(db_file=os.path.join(os.path.dirname(__file__), "agno_sessions.db"))

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

try:
    from server import browser_scroll_viewport, browser_click_coordinates, execute_local_cli
    HAS_SERVER_MODULE = True
except Exception:
    HAS_SERVER_MODULE = False
    browser_scroll_viewport = None
    browser_click_coordinates = None
    execute_local_cli = None


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

    if provider in {"gemini", "google"}:
        api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY or GEMINI_API_KEY is not set. Export it before running a Gemini Agent call.")
        if Gemini is None:
            raise RuntimeError("Gemini support is unavailable in the current installation (google-genai / agno google integration).")
        model_id = model_id or os.getenv("AGNO_MODEL", "gemini-3.6-flash")
        return Gemini(id=model_id, api_key=api_key)

    if provider == "deepseek":
        api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set. Export it before running a DeepSeek Agent call.")
        if DeepSeek is None:
            raise RuntimeError("DeepSeek support is unavailable in the current installation of Agno.")
        model_id = model_id or os.getenv("AGNO_MODEL", "deepseek-v4-flash")
        return DeepSeek(id=model_id, api_key=api_key)

    if provider in {"vllm", "openai_compatible", "custom"}:
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

    # Fallback to OpenAI (also honors OPENAI_BASE_URL for OpenAI-compatible proxies)
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Export it before running a real OpenAI Agent call.")
    if OpenAIChat is None:
        raise RuntimeError("OpenAI support is unavailable in the current installation of Agno.")
    model_id = model_id or os.getenv("AGNO_MODEL", "gpt-4o-mini")
    base_url = base_url or os.getenv("OPENAI_BASE_URL")
    return OpenAIChat(id=model_id, api_key=api_key, base_url=base_url)


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


async def browser_scroll_tool(
    origin: Optional[str] = None,
    delta_y: Optional[int] = None,
    y_delta: Optional[int] = None,
    y: Optional[int] = None,
    delta: Optional[int] = None,
    deltaY: Optional[int] = None,
) -> str:
    """Request the browser bridge to scroll by a Y delta on the current origin.

    Models vary across providers; accept multiple aliases: delta_y, y_delta, y, delta, deltaY.
    """
    if origin is None:
        origin = os.getenv("BROWSER_ORIGIN", "https://example.com")
    if delta_y is None:
        delta_y = y_delta
    if delta_y is None:
        delta_y = y
    if delta_y is None:
        delta_y = delta
    if delta_y is None:
        delta_y = deltaY
    if delta_y is None:
        raise ValueError("scroll amount is required (delta_y / y_delta / y)")

    if browser_scroll_viewport is not None:
        result = await browser_scroll_viewport(origin, delta_y, timeout=30)
        return f"Browser scroll on {origin}: {result}"

    async with httpx.AsyncClient() as client:
        res = await client.post("http://127.0.0.1:8000/agent/enqueue", json={
            "origin": origin,
            "tool": "scroll",
            "args": {"deltaY": delta_y},
        })
        action_id = res.json().get("action_id")
        for _ in range(30):
            r = await client.get("http://127.0.0.1:8000/agent/result", params={"action_id": action_id})
            data = r.json()
            if data.get("status") == "done":
                return f"Browser scroll on {origin}: {data.get('result')}"
            await asyncio.sleep(1)
        return f"Browser scroll triggered on {origin}, but no confirmation was received within timeout."


async def browser_click_tool(origin: Optional[str] = None, x: Optional[int] = None, y: Optional[int] = None) -> str:
    """Request the browser bridge to click at an x/y coordinate on the current origin."""
    if origin is None:
        origin = os.getenv("BROWSER_ORIGIN", "https://example.com")
    if x is None or y is None:
        raise ValueError("x and y are required")

    if browser_click_coordinates is not None:
        result = await browser_click_coordinates(origin, x, y, timeout=30)
        return f"Browser click on {origin} at ({x}, {y}): {result}"

    async with httpx.AsyncClient() as client:
        res = await client.post("http://127.0.0.1:8000/agent/enqueue", json={
            "origin": origin,
            "tool": "click",
            "args": {"x": x, "y": y},
        })
        action_id = res.json().get("action_id")
        for _ in range(30):
            r = await client.get("http://127.0.0.1:8000/agent/result", params={"action_id": action_id})
            data = r.json()
            if data.get("status") == "done":
                return f"Browser click on {origin} at ({x}, {y}): {data.get('result')}"
            await asyncio.sleep(1)
        return f"Browser click triggered on {origin}, but no confirmation was received within timeout."


async def browser_navigate_tool(origin: Optional[str] = None, url: Optional[str] = None, **kwargs) -> str:
    if origin is None:
        origin = os.getenv("BROWSER_ORIGIN", "https://example.com")
    if not url:
        raise ValueError("url is required")
    try:
        from server import browser_navigate
        result = await browser_navigate(origin, url, timeout=30)
        return f"Navigated to {url} on {origin}: {result}"
    except Exception:
        async with httpx.AsyncClient() as client:
            res = await client.post("http://127.0.0.1:8000/agent/enqueue", json={
                "origin": origin,
                "tool": "navigate",
                "args": {"url": url},
            })
            action_id = res.json().get("action_id")
            for _ in range(30):
                r = await client.get("http://127.0.0.1:8000/agent/result", params={"action_id": action_id})
                data = r.json()
                if data.get("status") == "done":
                    return f"Navigated to {url} on {origin}: {data.get('result')}"
                await asyncio.sleep(1)
            return f"Navigation to {url} was triggered on {origin}, but no confirmation was received."


async def browser_tabs_context_tool(origin: Optional[str] = None) -> str:
    if origin is None:
        origin = os.getenv("BROWSER_ORIGIN", "https://example.com")
    try:
        from server import browser_tabs_context
        result = await browser_tabs_context(origin, timeout=30)
        return f"Tabs context for {origin}: {result}"
    except Exception:
        return f"Tabs context request for {origin} was queued but not confirmed."


async def browser_tabs_create_tool(origin: Optional[str] = None, url: Optional[str] = None) -> str:
    if origin is None:
        origin = os.getenv("BROWSER_ORIGIN", "https://example.com")
    if not url:
        url = "https://example.com"
    try:
        from server import browser_tabs_create
        result = await browser_tabs_create(origin, url, timeout=30)
        return f"Created tab {url} on {origin}: {result}"
    except Exception:
        return f"Tab creation for {url} on {origin} was queued but not confirmed."


async def browser_tabs_close_tool(origin: Optional[str] = None, tab_id: Optional[int] = None) -> str:
    if origin is None:
        origin = os.getenv("BROWSER_ORIGIN", "https://example.com")
    if tab_id is None:
        raise ValueError("tab_id is required")
    try:
        from server import browser_tabs_close
        result = await browser_tabs_close(origin, tab_id, timeout=30)
        return f"Closed tab {tab_id} on {origin}: {result}"
    except Exception:
        return f"Tab close request for {tab_id} on {origin} was queued but not confirmed."


async def browser_get_page_text_tool(origin: Optional[str] = None) -> str:
    if origin is None:
        origin = os.getenv("BROWSER_ORIGIN", "https://example.com")
    try:
        from server import browser_get_page_text
        result = await browser_get_page_text(origin, timeout=30)
        return f"Page text for {origin}: {result}"
    except Exception:
        return f"Page text request for {origin} was queued but not confirmed."


async def browser_read_page_tool(origin: Optional[str] = None, selector: Optional[str] = None) -> str:
    if origin is None:
        origin = os.getenv("BROWSER_ORIGIN", "https://example.com")
    try:
        from server import browser_read_page
        result = await browser_read_page(origin, selector=selector, timeout=30)
        return f"Read page for {origin}: {result}"
    except Exception:
        return f"Read page request for {origin} was queued but not confirmed."


async def browser_find_tool(origin: Optional[str] = None, description: Optional[str] = None) -> str:
    if origin is None:
        origin = os.getenv("BROWSER_ORIGIN", "https://example.com")
    if not description:
        raise ValueError("description is required")
    try:
        from server import browser_find
        result = await browser_find(origin, description, timeout=30)
        return f"Find result for '{description}' on {origin}: {result}"
    except Exception:
        return f"Find request for '{description}' on {origin} was queued but not confirmed."


async def browser_read_console_messages_tool(origin: Optional[str] = None) -> str:
    if origin is None:
        origin = os.getenv("BROWSER_ORIGIN", "https://example.com")
    try:
        from server import browser_read_console_messages
        result = await browser_read_console_messages(origin, timeout=30)
        return f"Console messages for {origin}: {result}"
    except Exception:
        return f"Console log request for {origin} was queued but not confirmed."


async def browser_read_network_requests_tool(origin: Optional[str] = None) -> str:
    if origin is None:
        origin = os.getenv("BROWSER_ORIGIN", "https://example.com")
    try:
        from server import browser_read_network_requests
        result = await browser_read_network_requests(origin, timeout=30)
        return f"Network requests for {origin}: {result}"
    except Exception:
        return f"Network capture request for {origin} was queued but not confirmed."


async def browser_form_input_tool(origin: Optional[str] = None, selector: Optional[str] = None, value: Optional[str] = None, field_type: str = 'text') -> str:
    if origin is None:
        origin = os.getenv("BROWSER_ORIGIN", "https://example.com")
    if not selector or value is None:
        raise ValueError("selector and value are required")
    try:
        from server import browser_form_input
        result = await browser_form_input(origin, selector, value, field_type=field_type, timeout=30)
        return f"Set selector {selector} on {origin} to {value!r}: {result}"
    except Exception:
        return f"Form input request for {selector} on {origin} was queued but not confirmed."


async def browser_javascript_tool_fn(origin: Optional[str] = None, code: Optional[str] = None) -> str:
    if origin is None:
        origin = os.getenv("BROWSER_ORIGIN", "https://example.com")
    if not code:
        raise ValueError("code is required")
    try:
        from server import browser_javascript_tool
        result = await browser_javascript_tool(origin, code, timeout=30)
        return f"JS execution on {origin}: {result}"
    except Exception:
        return f"JS execution for {origin} was queued but not confirmed."


async def browser_resize_window_tool(origin: Optional[str] = None, width: Optional[int] = None, height: Optional[int] = None) -> str:
    if origin is None:
        origin = os.getenv("BROWSER_ORIGIN", "https://example.com")
    if width is None or height is None:
        raise ValueError("width and height are required")
    try:
        from server import browser_resize_window
        result = await browser_resize_window(origin, width, height, timeout=30)
        return f"Resized window to {width}x{height} on {origin}: {result}"
    except Exception:
        return f"Resize request to {width}x{height} on {origin} was queued but not confirmed."


async def browser_gif_creator_tool(origin: Optional[str] = None, duration_ms: int = 3000, title: Optional[str] = None) -> str:
    if origin is None:
        origin = os.getenv("BROWSER_ORIGIN", "https://example.com")
    try:
        from server import browser_gif_creator
        result = await browser_gif_creator(origin, duration_ms=duration_ms, title=title, timeout=30)
        return f"GIF capture for {origin}: {result}"
    except Exception:
        return f"GIF capture request for {origin} was queued but not confirmed."


async def browser_shortcuts_list_tool(origin: Optional[str] = None) -> str:
    if origin is None:
        origin = os.getenv("BROWSER_ORIGIN", "https://example.com")
    try:
        from server import browser_shortcuts_list
        result = await browser_shortcuts_list(origin, timeout=30)
        return f"Shortcuts list for {origin}: {result}"
    except Exception:
        return f"Shortcuts list request for {origin} was queued but not confirmed."


async def browser_shortcuts_execute_tool(origin: Optional[str] = None, shortcut_name: Optional[str] = None) -> str:
    if origin is None:
        origin = os.getenv("BROWSER_ORIGIN", "https://example.com")
    if not shortcut_name:
        raise ValueError("shortcut_name is required")
    try:
        from server import browser_shortcuts_execute
        result = await browser_shortcuts_execute(origin, shortcut_name, timeout=30)
        return f"Executed shortcut '{shortcut_name}' on {origin}: {result}"
    except Exception:
        return f"Shortcut '{shortcut_name}' request for {origin} was queued but not confirmed."


async def browser_batch_tool(origin: Optional[str] = None, operations: Optional[List[dict]] = None) -> str:
    if origin is None:
        origin = os.getenv("BROWSER_ORIGIN", "https://example.com")
    if operations is None:
        operations = []
    try:
        from server import browser_batch
        result = await browser_batch(origin, operations, timeout=30)
        return f"Batch browser actions for {origin}: {result}"
    except Exception:
        return f"Batch browser actions for {origin} were queued but not confirmed."


async def computer_tool(origin: Optional[str] = None, action: Optional[str] = None, **kwargs) -> str:
    if origin is None:
        origin = os.getenv("BROWSER_ORIGIN", "https://example.com")
    if not action:
        raise ValueError("action is required")
    try:
        from server import browser_computer
        result = await browser_computer(origin, action, **kwargs)
        return f"Computer action '{action}' on {origin}: {result}"
    except Exception:
        return f"Computer action '{action}' for {origin} was queued but not confirmed."


class _OriginState:
    """Mutable box for "the origin this turn's browser tools currently target."

    A turn can call `navigate` and then, in the SAME turn, call another
    browser tool expecting it to act on the page it just navigated to. Origin
    used to be a frozen closure value (see _bind_origin's old docstring) —
    every tool in a turn was permanently bound to whatever origin the request
    started with. That's provably wrong: the extension re-subscribes its
    action-broadcast WS channel to the new origin as soon as a same-tab
    navigation is detected (see background.js's onUpdated listener), often
    before the model's next same-turn tool call goes out, so that next call
    -- still targeting the OLD origin -- gets silently dropped by the
    server's exact-origin broadcast filter and times out with no real signal
    about the new page. This box is shared by every browser tool built for
    one turn; `navigate`'s entrypoint (see build_agent/_deepagents_browser_tools)
    is the only writer, updating it the moment a navigation succeeds so every
    later tool call in the same turn targets the page that's actually loaded."""

    __slots__ = ("origin",)

    def __init__(self, origin: str):
        self.origin = origin


def _origin_from_url(url: Optional[str]) -> Optional[str]:
    """scheme://netloc for a URL, or None if it can't be parsed as one."""
    if not url:
        return None
    try:
        from urllib.parse import urlsplit
        parts = urlsplit(url)
        if parts.scheme and parts.netloc:
            return f"{parts.scheme}://{parts.netloc}"
    except Exception:
        pass
    return None


def _bind_origin(fn, origin_state: "_OriginState"):
    """Bind a browser tool's origin to the real per-request tab, instead of
    exposing it as a model-supplied parameter. Reads `origin_state.origin` at
    CALL time (not build time) so a same-turn `navigate` can update it for
    every tool called after it — see _OriginState's docstring.

    A JSON Schema "default" is documentation, not an enforced fallback — OpenAI-
    style function calling never fills in an omitted argument from it. Every
    browser_*_tool below defaults its own `origin` param to None and then falls
    back to BROWSER_ORIGIN/"https://example.com" whenever the model didn't bother
    to pass one (which is most of the time, since origin is routing plumbing the
    model has no reason to know). That silently pointed every browser action at
    the placeholder origin instead of the actual attached tab. Wrapping the
    entrypoint and removing "origin" from its schema (done in build_agent) makes
    it impossible for the model to get this wrong.

    A plain closure (not functools.partial) is used deliberately: Agno's schema
    auto-inspection walks the entrypoint's signature, and partial objects don't
    introspect the same way plain functions do.

    Models aren't strictly bound to the declared schema — e.g. DeepSeek would
    sometimes call a zero-argument tool with a stray `{"kwargs": {}}`, which a
    naive `**kwargs` pass-through forwards straight into the real function and
    blows up with "unexpected keyword argument 'kwargs'". Unknown keys are
    dropped here instead, unless the wrapped function itself declares its own
    `**kwargs` (a few tools use that for scroll-delta aliases), in which case
    everything is passed through as before.
    """
    import inspect
    sig = inspect.signature(fn)
    accepts_var_keyword = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    known_params = set(sig.parameters) - {"origin"}

    async def wrapper(**kwargs):
        if not accepts_var_keyword:
            kwargs = {k: v for k, v in kwargs.items() if k in known_params}
        return await fn(origin=origin_state.origin, **kwargs)
    wrapper.__name__ = getattr(fn, '__name__', 'tool')
    wrapper.__doc__ = fn.__doc__
    return wrapper


def _bind_navigate(origin_state: "_OriginState"):
    """Like _bind_origin(browser_navigate_tool, origin_state), but also
    updates origin_state.origin to the destination's origin the moment
    navigation succeeds, and says so in the returned message — the model
    doesn't need a second same-turn tool call just to find out where it
    ended up (see _OriginState's docstring for why that second call was
    silently failing)."""

    async def navigate(url: Optional[str] = None) -> str:
        if not url:
            raise ValueError("url is required")
        result = await browser_navigate_tool(origin=origin_state.origin, url=url)
        new_origin = _origin_from_url(url)
        if new_origin:
            origin_state.origin = new_origin
        return f"{result} Now targeting {origin_state.origin} — later tool calls this turn act on this page."

    return navigate


# ---------------------------------------------------------------------------
# Generic context-injection helpers, used by build_agent/build_deepagents_graph
# below. Deliberately agnostic to where any of this content comes from — a
# domain-matched config from a URL-template registry, a manifest, a hardcoded
# config, anything. That sourcing decision belongs to whichever extension-
# specific layer calls into this module (server.py, for the browser bridge);
# see build_agent's docstring for why that separation matters.
# ---------------------------------------------------------------------------

def normalize_tool_schema(schema: Optional[dict]) -> dict:
    """JSON Schema hardening for a dynamically-declared tool's params,
    generically useful for any caller building tools with a generic **kwargs
    entrypoint: unconditionally sets additionalProperties unequal-to-baseline
    so Agno's params_set_by_user check never falls back to auto-inspecting the
    entrypoint's signature (see build_agent's zero-property tool schemas
    earlier in this file for the original bug this class of fix addresses)."""
    schema = dict(schema or {})
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    schema.setdefault("required", [])
    schema["additionalProperties"] = False
    return schema


def json_schema_to_pydantic(model_name: str, schema: dict):
    """Minimal JSON Schema -> Pydantic model converter, just for manifest tool
    params (string/integer/number/boolean/array/object). Used to give LangChain/
    DeepAgents tools an explicit args_schema instead of relying on Python
    signature introspection over the manifest entrypoints' generic **kwargs —
    that inference has already shown a documented failure mode on the Agno side
    (see _bind_origin's docstring); LangChain's would hit the same class of issue."""
    from pydantic import Field, create_model

    type_map = {"string": str, "integer": int, "number": float, "boolean": bool, "array": list, "object": dict}
    properties = (schema or {}).get("properties") or {}
    required = set((schema or {}).get("required") or [])
    fields: Dict[str, Any] = {}
    for prop_name, prop_schema in properties.items():
        py_type = type_map.get((prop_schema or {}).get("type"), str)
        description = (prop_schema or {}).get("description")
        if prop_name in required:
            fields[prop_name] = (py_type, Field(..., description=description))
        else:
            fields[prop_name] = (Optional[py_type], Field(None, description=description))
    return create_model(model_name, **fields)


def _asset_index(skills: Optional[List[dict]], knowledge: Optional[List[dict]]) -> str:
    """Build a name+description index for a generic list of skills/knowledge
    dicts ({"name","description","content"}), for appending to the system
    message. Full `content` is NOT included — that's progressive disclosure via
    the load_asset tool (see _make_load_asset_tool_*), so a caller handing in a
    lot of content doesn't cost tokens on every turn regardless of whether the
    model ends up using it."""
    def _index(label, items):
        lines = []
        for item in (items or []):
            if not isinstance(item, dict) or not item.get("name"):
                continue
            desc = item.get("description") or ""
            lines.append(f"- {item['name']}: {desc}" if desc else f"- {item['name']}")
        if not lines:
            return None
        return f"Available {label} (call load_asset to read the full content):\n" + "\n".join(lines)

    parts = [b for b in (_index("skills", skills), _index("knowledge", knowledge)) if b]
    return "\n\n".join(parts)


async def _lookup_asset(skills: Optional[List[dict]], knowledge: Optional[List[dict]], kind: str, name: str) -> str:
    bucket = skills if kind == "skill" else knowledge if kind == "knowledge" else None
    if bucket is None:
        return f"Unknown kind '{kind}'; use 'skill' or 'knowledge'."
    for item in bucket:
        if isinstance(item, dict) and item.get("name") == name:
            return item.get("content") or "(no content)"
    return f"No {kind} named '{name}' found."


_LOAD_ASSET_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["skill", "knowledge"]},
        "name": {"type": "string", "description": "the asset's name, from the system-prompt index"},
    },
    "required": ["kind", "name"],
    "additionalProperties": False,
}
_LOAD_ASSET_DESCRIPTION = "Load the full content of a skill or knowledge item (see the index in the system prompt)."


def _make_load_asset_tool_for_agno(skills: Optional[List[dict]], knowledge: Optional[List[dict]]) -> Function:
    async def load_asset(kind: str, name: str) -> str:
        return await _lookup_asset(skills, knowledge, kind, name)

    return Function(name="load_asset", description=_LOAD_ASSET_DESCRIPTION, parameters=dict(_LOAD_ASSET_SCHEMA), entrypoint=load_asset)


def _make_load_asset_tool_for_deepagents(skills: Optional[List[dict]], knowledge: Optional[List[dict]]):
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field

    class LoadAssetArgs(BaseModel):
        kind: str = Field(..., description="'skill' or 'knowledge'")
        name: str = Field(..., description="the asset's name, from the system-prompt index")

    async def load_asset(kind: str, name: str) -> str:
        return await _lookup_asset(skills, knowledge, kind, name)

    return StructuredTool.from_function(coroutine=load_asset, name="load_asset", description=_LOAD_ASSET_DESCRIPTION, args_schema=LoadAssetArgs)


def _mcp_connection_defaults(server_def: dict) -> dict:
    """Normalize a generic {"name","url","transport","headers"?} MCP server
    connection dict — callers (e.g. server.py translating a domain manifest, or
    anyone else with their own way of sourcing MCP servers) only need to supply
    these fields; this fills in a sane default transport."""
    return {
        "name": server_def.get("name") or server_def.get("url", "mcp"),
        "url": server_def.get("url"),
        "transport": server_def.get("transport") or "streamable-http",
        "headers": server_def.get("headers"),
    }


def _mcp_tools_for_agno(mcp_servers: Optional[List[dict]]) -> List[Any]:
    """Build (unconnected) Agno MCPTools instances for a generic list of MCP
    server connection dicts. Agno's own Agent.arun() connects these
    automatically per-run (see agno.agent._tools.aget_tools /
    _init.connect_mcp_tools) — no manual connect()/close() needed here."""
    if not mcp_servers:
        return []
    from agno.tools.mcp import MCPTools

    tools = []
    for server_def in mcp_servers:
        if not isinstance(server_def, dict) or not server_def.get("url"):
            continue
        conn = _mcp_connection_defaults(server_def)
        tools.append(MCPTools(name=conn["name"], url=conn["url"], transport=conn["transport"]))
    return tools


async def _mcp_tools_for_deepagents(mcp_servers: Optional[List[dict]]) -> List[Any]:
    """Load real MCP server tools as LangChain BaseTool objects for the
    DeepAgents harness. Unlike Agno's MCPTools, langchain-mcp-adapters has no
    lazy per-run auto-connect — get_tools() must be awaited up front, which is
    why this (and build_deepagents_graph/build_langgraph_agent, which call it)
    is async: DeepAgents needs a plain list of already-usable tools at
    construction time, not a Toolkit it can connect lazily."""
    if not mcp_servers:
        return []
    from langchain_mcp_adapters.client import MultiServerMCPClient

    connections = {}
    for server_def in mcp_servers:
        if not isinstance(server_def, dict) or not server_def.get("url"):
            continue
        conn = _mcp_connection_defaults(server_def)
        # langchain-mcp-adapters spells the transport with an underscore
        # ("streamable_http"), Agno's MCPTools with a hyphen ("streamable-http")
        # — normalize here so callers only ever write one convention.
        transport = "streamable_http" if conn["transport"] in ("streamable-http", "streamable_http") else conn["transport"]
        entry: Dict[str, Any] = {"transport": transport, "url": conn["url"]}
        if conn.get("headers"):
            entry["headers"] = conn["headers"]
        connections[conn["name"]] = entry

    if not connections:
        return []
    client = MultiServerMCPClient(connections)
    return await client.get_tools()


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
    """
    origin = origin or os.getenv("BROWSER_ORIGIN", "https://example.com")
    origin_state = _OriginState(origin)
    model = get_model(api_key=api_key, provider=model_provider, model_id=model_id, base_url=base_url)

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
        tools.extend([
            Function(
                name="navigate",
                description="Navigate the active browser tab to a URL.",
                parameters={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to visit, including the protocol"},
                    },
                    "required": ["url"],
                },
                entrypoint=_bind_navigate(origin_state),
            ),
            Function(
                name="tabs_context_mcp",
                description="List open tabs and their metadata in the current browser window.",
                parameters={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
                entrypoint=_bind_origin(browser_tabs_context_tool, origin_state),
            ),
            Function(
                name="tabs_create_mcp",
                description="Open a new tab at a URL.",
                parameters={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL for the new tab"},
                    },
                    "required": ["url"],
                },
                entrypoint=_bind_origin(browser_tabs_create_tool, origin_state),
            ),
            Function(
                name="tabs_close_mcp",
                description="Close a browser tab by tab id.",
                parameters={
                    "type": "object",
                    "properties": {
                        "tab_id": {"type": "integer", "description": "Tab id to close"},
                    },
                    "required": ["tab_id"],
                },
                entrypoint=_bind_origin(browser_tabs_close_tool, origin_state),
            ),
            Function(
                name="get_page_text",
                description="Extract plain text from the current page body.",
                parameters={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
                entrypoint=_bind_origin(browser_get_page_text_tool, origin_state),
            ),
            Function(
                name="read_page",
                description="Read a lightweight DOM snapshot or visible page structure.",
                parameters={
                    "type": "object",
                    "properties": {
                        "selector": {"type": "string", "description": "Optional CSS selector for a specific region of the page"},
                    },
                    "required": [],
                },
                entrypoint=_bind_origin(browser_read_page_tool, origin_state),
            ),
            Function(
                name="find",
                description="Find UI elements by a natural-language description, label, text, or CSS selector hint.",
                parameters={
                    "type": "object",
                    "properties": {
                        "description": {"type": "string", "description": "Text or description used to locate the target element"},
                    },
                    "required": ["description"],
                },
                entrypoint=_bind_origin(browser_find_tool, origin_state),
            ),
            Function(
                name="read_console_messages",
                description="Read runtime console logs and errors from the current page.",
                parameters={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
                entrypoint=_bind_origin(browser_read_console_messages_tool, origin_state),
            ),
            Function(
                name="read_network_requests",
                description="Read recent network requests and fetch/XHR calls from the current page.",
                parameters={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
                entrypoint=_bind_origin(browser_read_network_requests_tool, origin_state),
            ),
            Function(
                name="form_input",
                description="Fill a form field or input element on the page by selector and value.",
                parameters={
                    "type": "object",
                    "properties": {
                        "selector": {"type": "string", "description": "CSS selector or form element locator"},
                        "value": {"type": "string", "description": "Text value to enter"},
                        "field_type": {"type": "string", "description": "Type of field, usually text, email, search, textarea, or select", "default": "text"},
                    },
                    "required": ["selector", "value"],
                },
                entrypoint=_bind_origin(browser_form_input_tool, origin_state),
            ),
            Function(
                name="javascript_tool",
                description=(
                    "Run JavaScript in the page and return its value — same semantics as pasting the "
                    "code into the DevTools console: the value of the last expression is returned even "
                    "without an explicit `return`, and top-level `await` works directly. No need to wrap "
                    "the snippet in a function; write plain statements, e.g. "
                    "`document.querySelectorAll('a').length` or "
                    "`const r = await fetch('/api'); await r.json()`."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "JavaScript to run in the page context"},
                    },
                    "required": ["code"],
                },
                entrypoint=_bind_origin(browser_javascript_tool_fn, origin_state),
            ),
            Function(
                name="browser_scroll_viewport",
                description="Scroll the current browser tab by a Y delta in pixels.",
                parameters={
                    "type": "object",
                    "properties": {
                        "delta_y": {"type": "integer", "description": "Number of pixels to scroll"},
                        "y_delta": {"type": "integer", "description": "Alternate alias for delta_y supported by some models", "default": None},
                        "y": {"type": "integer", "description": "Short alias used by some models for the scroll amount", "default": None},
                        "delta": {"type": "integer", "description": "Generic alias for the scroll amount", "default": None},
                        "deltaY": {"type": "integer", "description": "CamelCase alias for the scroll amount", "default": None},
                    },
                    "required": ["delta_y"],
                },
                entrypoint=_bind_origin(browser_scroll_tool, origin_state),
            ),
            Function(
                name="browser_click_coordinates",
                description="Click the browser page at coordinate x/y.",
                parameters={
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer", "description": "X coordinate"},
                        "y": {"type": "integer", "description": "Y coordinate"},
                    },
                    "required": ["x", "y"],
                },
                entrypoint=_bind_origin(browser_click_tool, origin_state),
            ),
            Function(
                name="resize_window",
                description="Resize the current browser window to a specific width and height.",
                parameters={
                    "type": "object",
                    "properties": {
                        "width": {"type": "integer", "description": "Window width in pixels"},
                        "height": {"type": "integer", "description": "Window height in pixels"},
                    },
                    "required": ["width", "height"],
                },
                entrypoint=_bind_origin(browser_resize_window_tool, origin_state),
            ),
            Function(
                name="gif_creator",
                description="Create a lightweight screen capture or GIF-like artifact from the current browser session.",
                parameters={
                    "type": "object",
                    "properties": {
                        "duration_ms": {"type": "integer", "description": "Duration for the GIF-like capture", "default": 3000},
                        "title": {"type": "string", "description": "Optional title for the capture metadata"},
                    },
                    "required": [],
                },
                entrypoint=_bind_origin(browser_gif_creator_tool, origin_state),
            ),
            Function(
                name="shortcuts_list",
                description="List available browser automation shortcuts that can be executed on the active page.",
                parameters={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
                entrypoint=_bind_origin(browser_shortcuts_list_tool, origin_state),
            ),
            Function(
                name="shortcuts_execute",
                description="Execute a named browser shortcut or macro.",
                parameters={
                    "type": "object",
                    "properties": {
                        "shortcut_name": {"type": "string", "description": "Name of the shortcut to execute"},
                    },
                    "required": ["shortcut_name"],
                },
                entrypoint=_bind_origin(browser_shortcuts_execute_tool, origin_state),
            ),
            Function(
                name="browser_batch",
                description="Apply a short batch of browser actions in order to the current page.",
                parameters={
                    "type": "object",
                    "properties": {
                        "operations": {"type": "array", "description": "List of browser actions to execute in order", "items": {"type": "object"}},
                    },
                    "required": ["operations"],
                },
                entrypoint=_bind_origin(browser_batch_tool, origin_state),
            ),
            Function(
                name="computer",
                description="Perform generic computer-like browser actions such as click, type, scroll, keypress, hover, and drag.",
                parameters={
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "description": "Action to perform, such as click, type, scroll, keypress, hover, or drag"},
                        "x": {"type": "integer", "description": "X coordinate for click/hover/drag actions"},
                        "y": {"type": "integer", "description": "Y coordinate for click/hover/drag actions"},
                        "text": {"type": "string", "description": "Text to type for type actions"},
                        "delta_y": {"type": "integer", "description": "Vertical scroll delta in pixels"},
                        "key": {"type": "string", "description": "Key to press for keypress actions"},
                    },
                    "required": ["action"],
                },
                entrypoint=_bind_origin(computer_tool, origin_state),
            ),
        ])

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
    return agent


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

            if event_name == 'ReasoningContentDelta':
                chunk = getattr(event, 'reasoning_content', None)
                if chunk:
                    yield {'type': 'assistant.thinking', 'chunk': str(chunk)}
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

            if event_name == 'RunContent':
                piece = getattr(event, 'content', None)
                if piece:
                    collected_text.append(str(piece))
                    yield {'type': 'assistant.delta', 'chunk': str(piece)}
                reasoning_piece = getattr(event, 'reasoning_content', None)
                if reasoning_piece:
                    yield {'type': 'assistant.thinking', 'chunk': str(reasoning_piece)}
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


def _deepagents_browser_tools(origin_state: "_OriginState") -> List[Any]:
    """Browser tools for the DeepAgents harness, bound via closure to a shared
    mutable _OriginState (origin is routing plumbing, not something the model
    should have to supply) — the same binding strategy _bind_origin uses for
    the native Agno backend, including the same same-turn navigate handling
    (see _OriginState's docstring). These are thin wrappers around the exact
    same server.py browser_*
    helpers the native backend calls, so behavior is identical across both
    backends. Plain typed async functions work directly as deepagents/LangChain
    tools — their schema is inferred from the signature and docstring, no
    @tool decorator or manual JSON schema needed.
    """
    from server import (
        browser_navigate, browser_computer, browser_get_page_text, browser_read_page,
        browser_find, browser_javascript_tool, browser_read_console_messages,
        browser_read_network_requests, browser_form_input,
    )

    async def navigate(url: str) -> str:
        """Navigate the active browser tab to a URL, including the protocol."""
        result = await browser_navigate(origin_state.origin, url, timeout=30)
        new_origin = _origin_from_url(url)
        if new_origin:
            origin_state.origin = new_origin
        return f"Navigated to {url}: {result} Now targeting {origin_state.origin} — later tool calls this turn act on this page."

    async def computer(action: str, x: int = 0, y: int = 0, text: str = "", delta_y: int = 0, key: str = "Enter") -> str:
        """Perform a browser action: click, type, scroll, keypress, or hover.

        action: one of "click", "type", "scroll", "keypress", "hover".
        x, y: coordinates, for click/hover.
        text: text to type, for the "type" action.
        delta_y: vertical scroll amount in pixels, for the "scroll" action.
        key: key name (e.g. "Enter", "Tab"), for the "keypress" action.
        """
        kwargs: Dict[str, Any] = {}
        if action in ("click", "hover"):
            kwargs = {"x": x, "y": y}
        elif action == "type":
            kwargs = {"text": text}
        elif action == "scroll":
            kwargs = {"delta_y": delta_y}
        elif action == "keypress":
            kwargs = {"key": key}
        return str(await browser_computer(origin_state.origin, action, **kwargs))

    async def get_page_text() -> str:
        """Extract the plain visible text of the current page body."""
        return str(await browser_get_page_text(origin_state.origin, timeout=30))

    async def read_page(selector: str = "") -> str:
        """Read a lightweight snapshot (title, text, size) of the page, or of one element if selector is given."""
        return str(await browser_read_page(origin_state.origin, selector=selector or None, timeout=30))

    async def find(description: str) -> str:
        """Find clickable/interactive elements on the page matching a natural-language description."""
        return str(await browser_find(origin_state.origin, description, timeout=30))

    async def javascript_tool(code: str) -> str:
        """Run JavaScript in the page and return its value — console-style completion value, top-level await works."""
        return str(await browser_javascript_tool(origin_state.origin, code, timeout=30))

    async def read_console_messages() -> str:
        """Read recent browser console logs and errors from the current page."""
        return str(await browser_read_console_messages(origin_state.origin, timeout=30))

    async def read_network_requests() -> str:
        """Read recent network requests (fetch/XHR) made by the current page."""
        return str(await browser_read_network_requests(origin_state.origin, timeout=30))

    async def form_input(selector: str, value: str, field_type: str = "text") -> str:
        """Set the value of a form field (input, textarea, or select) identified by a CSS selector."""
        return str(await browser_form_input(origin_state.origin, selector, value, field_type=field_type, timeout=30))

    return [
        navigate, computer, get_page_text, read_page, find,
        javascript_tool, read_console_messages, read_network_requests, form_input,
    ]


async def build_deepagents_graph(
    origin: Optional[str] = None,
    include_browser_tools: bool = True,
    extra_tools: Optional[List[Any]] = None,
    extra_system_prompt: Optional[str] = None,
    skills: Optional[List[dict]] = None,
    knowledge: Optional[List[dict]] = None,
    mcp_servers: Optional[List[dict]] = None,
    tool_meta: Optional[Dict[str, dict]] = None,
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

    Returns (graph, tool_meta) — tool_meta is passed straight through, for
    build_langgraph_agent to attach to the wrapping agent.
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
    return graph, (tool_meta or {})


def _make_reasoning_langgraph_agent_class():
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
    """
    from agno.agents.langgraph import LangGraphAgent

    class ReasoningLangGraphAgent(LangGraphAgent):
        async def _arun_adapter_stream(self, input, *, history=None, _lg_config_override=None, **kwargs):
            from uuid import uuid4
            from agno.agents.langgraph.utils import build_messages_with_history
            from agno.models.response import ToolExecution
            from agno.run.agent import ReasoningContentDeltaEvent, RunContentEvent, ToolCallCompletedEvent, ToolCallStartedEvent

            if self.graph is None:
                raise ValueError("No graph provided to LangGraphAgent")

            run_id = kwargs.get("run_id", str(uuid4()))
            graph_input = None if input is None else {self.input_key: build_messages_with_history(input, history)}
            config = self._build_config(kwargs, override=_lg_config_override)

            async for event in self.graph.astream_events(graph_input, config=config, version="v2"):
                kind = event.get("event")

                if kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if not chunk:
                        continue

                    reasoning = (getattr(chunk, "additional_kwargs", None) or {}).get("reasoning_content") or ""

                    content = getattr(chunk, "content", None)
                    if isinstance(content, str) and content:
                        yield RunContentEvent(run_id=run_id, agent_id=self.get_id(), agent_name=self.name or "", content=content)
                    elif isinstance(content, list):
                        # Some providers (not DeepSeek via ChatDeepSeek, which uses
                        # additional_kwargs above) put thinking in typed content blocks.
                        for block in content:
                            if not isinstance(block, dict):
                                continue
                            text = block.get("text") or block.get("reasoning_content") or block.get("content")
                            if not text:
                                continue
                            if block.get("type") in ("thinking", "reasoning", "reasoning_content"):
                                reasoning += text
                            else:
                                yield RunContentEvent(run_id=run_id, agent_id=self.get_id(), agent_name=self.name or "", content=text)

                    if reasoning:
                        yield ReasoningContentDeltaEvent(
                            run_id=run_id, agent_id=self.get_id(), agent_name=self.name or "", reasoning_content=reasoning
                        )

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

    `async def` because build_deepagents_graph is (MCP tool loading for
    DeepAgents needs an upfront await — see _mcp_tools_for_deepagents).

    Requires `langgraph` and `langchain-core` (and, for the default harness,
    `deepagents` + `langchain-deepseek`) to be installed.
    """
    try:
        ReasoningLangGraphAgent = _make_reasoning_langgraph_agent_class()
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
        graph, resolved_tool_meta = await build_deepagents_graph(
            origin=origin,
            include_browser_tools=include_browser_tools,
            extra_tools=extra_tools,
            extra_system_prompt=extra_system_prompt,
            skills=skills,
            knowledge=knowledge,
            mcp_servers=mcp_servers,
            tool_meta=tool_meta,
        )
    agent = ReasoningLangGraphAgent(name="LangGraph Agent", graph=graph, db=_shared_db)
    agent._tool_meta = resolved_tool_meta
    return agent


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

    Known limitations, both inherent to Agno's LangGraphAgent adapter itself
    (not something this wrapper patches around): it doesn't stream reasoning/
    thinking content the way the native backend does, and it has no cooperative
    Agent.cancel_run equivalent, so Stop falls back to a hard task cancel for
    this backend (see server.py) rather than the graceful stop used natively.
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
        )
    except Exception as exc:
        yield {'type': 'error', 'error': str(exc)}
        return

    async for evt in stream_agno_events(agent, prompt, session_id=thread_id):
        yield evt


async def run_prompt(prompt: str, origin: Optional[str] = None, include_browser_tools: Optional[bool] = None) -> str:
    if include_browser_tools is None:
        include_browser_tools = os.getenv("AGNO_USE_BROWSER_TOOLS", "0") == "1"
    agent = build_agent(origin=origin, include_browser_tools=include_browser_tools)
    result = await agent.arun(prompt)
    return getattr(result, 'content', str(result))


async def _demo() -> None:
    prompt = "Calculate the value of 583 * 22 and then tell me the result."
    print(await run_prompt(prompt, origin="https://example.com"))


if __name__ == "__main__":
    asyncio.run(_demo())
