"""Browser tools: the extension-facing half of the runtime.

Everything here is a thin wrapper around the `browser_*` helper coroutines
server.py exposes (which enqueue a CDP action for the Chrome extension and
await its result), plus the per-turn origin binding that keeps a turn's tool
calls pointed at the page that is actually loaded.

Two builders sit at the bottom, one per backend, both taking the same shared
`_OriginState`:
  - `build_agno_browser_functions(origin_state, plan_id)` -> Agno `Function`s
  - `build_deepagents_browser_tools(origin_state)` -> plain async callables
    usable as deepagents/LangChain tools
"""

import asyncio
import os
from typing import Any, Dict, List, Optional

import httpx
from agno.tools.function import Function

try:
    from server import browser_scroll_viewport, browser_click_coordinates, execute_local_cli
    HAS_SERVER_MODULE = True
except Exception:
    HAS_SERVER_MODULE = False
    browser_scroll_viewport = None
    browser_click_coordinates = None
    execute_local_cli = None


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


async def browser_switch_browser_tool(origin: Optional[str] = None, tab_id: Optional[int] = None) -> str:
    if origin is None:
        origin = os.getenv("BROWSER_ORIGIN", "https://example.com")
    if tab_id is None:
        raise ValueError("tab_id is required")
    try:
        from server import browser_switch_browser
        result = await browser_switch_browser(origin, tab_id, timeout=30)
        return f"Switched active tab to {tab_id}: {result}"
    except Exception:
        return f"Switch-browser request for tab {tab_id} was queued but not confirmed."


async def browser_upload_image_tool(
    origin: Optional[str] = None,
    selector: Optional[str] = None,
    image_base64: Optional[str] = None,
    image_url: Optional[str] = None,
    filename: Optional[str] = None,
    mime_type: Optional[str] = None,
) -> str:
    if origin is None:
        origin = os.getenv("BROWSER_ORIGIN", "https://example.com")
    if not selector:
        raise ValueError("selector is required")
    if not image_base64 and not image_url:
        raise ValueError("image_base64 (with optional mime_type) or image_url is required")
    try:
        from server import browser_upload_image
        result = await browser_upload_image(
            origin, selector, image_base64=image_base64, image_url=image_url,
            filename=filename, mime_type=mime_type, timeout=60,
        )
        return f"Uploaded image via {selector} on {origin}: {result}"
    except Exception:
        return f"Image upload for {selector} on {origin} was queued but not confirmed."


async def browser_list_connected_tool() -> str:
    try:
        from server import browser_list_connected
        result = await browser_list_connected()
        return f"Connected browsers: {result}"
    except Exception:
        return "Connected-browsers listing was requested but not available."


async def browser_select_tool(browser_id: Optional[str] = None) -> str:
    if not browser_id:
        raise ValueError("browser_id is required")
    try:
        from server import browser_select
        result = await browser_select(browser_id)
        return f"Selected browser {browser_id}: {result}"
    except Exception:
        return f"Browser selection for {browser_id} was queued but not confirmed."


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


def _bind_plain(fn):
    """Like _bind_origin, but for tools that don't need a per-turn origin at all
    — list_connected_browsers/select_browser operate on the bridge's connection
    registry, not a specific page. Still drops stray kwargs some models send
    instead of blowing up (see _bind_origin's docstring for why that matters)."""
    import inspect
    sig = inspect.signature(fn)
    accepts_var_keyword = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    known_params = set(sig.parameters)

    async def wrapper(**kwargs):
        if not accepts_var_keyword:
            kwargs = {k: v for k, v in kwargs.items() if k in known_params}
        return await fn(**kwargs)
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


def _bind_switch_browser(origin_state: "_OriginState"):
    """Like _bind_navigate, but for switch_browser: re-pointing the CDP
    debugger at a different tab_id also changes which page later tool calls
    this turn should target, so origin_state.origin is updated from the
    switched-to tab's URL the moment the switch confirms."""

    async def switch_browser(tab_id: Optional[int] = None) -> str:
        if tab_id is None:
            raise ValueError("tab_id is required")
        result = await browser_switch_browser_tool(origin=origin_state.origin, tab_id=tab_id)
        new_origin = _origin_from_url(_result_url(result))
        if new_origin:
            origin_state.origin = new_origin
        return f"{result} Now targeting {origin_state.origin} — later tool calls this turn act on this page."

    return switch_browser


def _bind_update_plan(plan_id: str):
    """update_plan's entrypoint, bound to a fixed plan_id (session_id, or origin
    when no session_id was given — see build_agent) rather than exposing that
    routing detail as a model-supplied parameter, the same reasoning _bind_origin
    uses for origin itself."""

    async def update_plan(plan: Optional[List[dict]] = None, explanation: Optional[str] = None) -> str:
        if not plan:
            raise ValueError("plan is required")
        try:
            from server import update_plan as _update_plan
            result = await _update_plan(plan_id, plan, explanation)
            return f"Plan updated: {result}"
        except Exception:
            return "Plan update was requested but not confirmed."

    return update_plan


def _result_url(result_text: str) -> Optional[str]:
    """Best-effort extraction of a 'url': '...' value out of a tool result's
    string repr (switch_browser's underlying result is a dict rendered via
    f-string, e.g. "...: {'status': 'ok', 'tabId': 3, 'url': 'https://...', ...}"),
    so origin_state can be updated without needing a second round trip."""
    import re
    match = re.search(r"'url':\s*'([^']*)'", result_text or "")
    return match.group(1) if match else None



def build_agno_browser_functions(origin_state: "_OriginState", plan_id: str) -> List[Function]:
    """The browser half of the native Agno agent's tool list.

    `runtime.agent.build_agent` extends its local-tool list with these when
    include_browser_tools is set. `origin_state` is the per-turn mutable origin
    box every tool here is bound to (see _OriginState); `plan_id` is what
    update_plan is bound to — session_id, or origin when no session_id was
    given (see build_agent).
    """
    return [
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
        Function(
            name="switch_browser",
            description="Switch the active browser tab that all other browser tools target, without opening or closing any tab.",
            parameters={
                "type": "object",
                "properties": {
                    "tab_id": {"type": "integer", "description": "Id of the tab to make active, from tabs_context_mcp"},
                },
                "required": ["tab_id"],
            },
            entrypoint=_bind_switch_browser(origin_state),
        ),
        Function(
            name="upload_image",
            description="Upload an image through a webpage's file input, given a CSS selector and either base64 image data or an image URL.",
            parameters={
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "description": "CSS selector of the file input element"},
                    "image_base64": {"type": "string", "description": "Base64-encoded image data (used with mime_type)"},
                    "image_url": {"type": "string", "description": "URL of an image to fetch and upload instead of image_base64"},
                    "filename": {"type": "string", "description": "Filename to save the image as before uploading"},
                    "mime_type": {"type": "string", "description": "MIME type for image_base64, e.g. image/png", "default": "image/png"},
                },
                "required": ["selector"],
            },
            entrypoint=_bind_origin(browser_upload_image_tool, origin_state),
        ),
        Function(
            name="list_connected_browsers",
            description="List the browser instances currently connected to this bridge, and which one (if any) is selected as the routing target.",
            parameters={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            entrypoint=_bind_plain(browser_list_connected_tool),
        ),
        Function(
            name="select_browser",
            description="Route subsequent browser tool calls to one specific connected browser instance, when more than one is connected.",
            parameters={
                "type": "object",
                "properties": {
                    "browser_id": {"type": "string", "description": "Id of the connected browser, from list_connected_browsers"},
                },
                "required": ["browser_id"],
            },
            entrypoint=_bind_plain(browser_select_tool),
        ),
        Function(
            name="update_plan",
            description="Present or update the agent's step-by-step execution plan before carrying it out.",
            parameters={
                "type": "object",
                "properties": {
                    "plan": {
                        "type": "array",
                        "description": "Ordered list of plan steps",
                        "items": {
                            "type": "object",
                            "properties": {
                                "step": {"type": "string", "description": "Description of this step"},
                                "status": {"type": "string", "enum": ["pending", "in_progress", "completed"], "description": "Current status of this step"},
                            },
                            "required": ["step", "status"],
                        },
                    },
                    "explanation": {"type": "string", "description": "Optional short explanation of the plan or why it changed"},
                },
                "required": ["plan"],
            },
            entrypoint=_bind_update_plan(plan_id),
        ),
    ]


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
        browser_read_network_requests, browser_form_input, browser_switch_browser,
        browser_upload_image, browser_list_connected, browser_select, update_plan as _update_plan,
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

    async def switch_browser(tab_id: int) -> str:
        """Switch the active browser tab that all other browser tools target, without opening or closing any tab."""
        result = await browser_switch_browser(origin_state.origin, tab_id, timeout=30)
        new_origin = _origin_from_url(_result_url(str(result)))
        if new_origin:
            origin_state.origin = new_origin
        return f"{result} Now targeting {origin_state.origin} — later tool calls this turn act on this page."

    async def upload_image(
        selector: str,
        image_base64: str = "",
        image_url: str = "",
        filename: str = "",
        mime_type: str = "image/png",
    ) -> str:
        """Upload an image through a webpage's file input, given a CSS selector and either base64 image data or an image URL."""
        if not image_base64 and not image_url:
            return "image_base64 (with optional mime_type) or image_url is required"
        return str(await browser_upload_image(
            origin_state.origin, selector,
            image_base64=image_base64 or None, image_url=image_url or None,
            filename=filename or None, mime_type=mime_type or None, timeout=60,
        ))

    async def list_connected_browsers() -> str:
        """List the browser instances currently connected to this bridge, and which one (if any) is selected as the routing target."""
        return str(await browser_list_connected())

    async def select_browser(browser_id: str) -> str:
        """Route subsequent browser tool calls to one specific connected browser instance, when more than one is connected."""
        return str(await browser_select(browser_id))

    async def update_plan(plan: List[dict], explanation: str = "") -> str:
        """Present or update the agent's step-by-step execution plan before carrying it out.

        plan: ordered list of {"step": str, "status": "pending"|"in_progress"|"completed"}.
        """
        return str(await _update_plan(origin_state.origin, plan, explanation or None))

    return [
        navigate, computer, get_page_text, read_page, find,
        javascript_tool, read_console_messages, read_network_requests, form_input,
        switch_browser, upload_image, list_connected_browsers, select_browser, update_plan,
    ]


#: Public name for the deepagents/LangChain-side builder, paired with
#: build_agno_browser_functions above. Same function; the underscore-prefixed
#: name is kept because several docstrings in this package refer to it.
build_deepagents_browser_tools = _deepagents_browser_tools
