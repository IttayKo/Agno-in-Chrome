"""Back-compat facade for the runtime, which now lives in the `runtime` package.

Nothing is implemented here: this module only re-exports what `runtime/` builds,
so existing `from agno_runtime import ...` callers (server.py, agno_agent.py,
anything outside this repo) keep working unchanged. New code should import from
`runtime` directly — see runtime/__init__.py for the layout.

Environment variables:
  OPENAI_API_KEY  required for real model calls
  AGNO_MODEL      defaults to gpt-4o-mini
  BROWSER_ORIGIN  defaults to https://example.com
  AGNO_USE_BROWSER_TOOLS  set to 1 to include browser tools by default
"""

import asyncio

# The documented public API.
from runtime import (  # noqa: F401
    build_agent,
    build_deepagents_graph,
    build_langgraph_agent,
    get_model,
    json_schema_to_pydantic,
    normalize_tool_schema,
    run_prompt,
    stream_agno_events,
    stream_langgraph_events,
)

# The communication-style seam (runtime/reasoning.py).
from runtime.reasoning import ReasoningStyle, register_style, resolve_style  # noqa: F401

# Everything else that used to be a module-level name here, re-exported so that
# older imports (including the underscore-prefixed internals a few call sites
# and tests reach for) keep resolving.
from runtime.models import DeepSeek, Gemini, OpenAIChat, google_types  # noqa: F401
from runtime.agent import (  # noqa: F401
    _demo,
    _shared_db,
    calculate_product,
    execute_local_shell,
)
from runtime.browser_tools import (  # noqa: F401
    HAS_SERVER_MODULE,
    _bind_navigate,
    _bind_origin,
    _bind_plain,
    _bind_switch_browser,
    _bind_update_plan,
    _deepagents_browser_tools,
    _origin_from_url,
    _OriginState,
    _result_url,
    browser_batch_tool,
    browser_click_coordinates,
    browser_click_tool,
    browser_find_tool,
    browser_form_input_tool,
    browser_get_page_text_tool,
    browser_gif_creator_tool,
    browser_javascript_tool_fn,
    browser_list_connected_tool,
    browser_navigate_tool,
    browser_read_console_messages_tool,
    browser_read_network_requests_tool,
    browser_read_page_tool,
    browser_resize_window_tool,
    browser_scroll_tool,
    browser_scroll_viewport,
    browser_select_tool,
    browser_shortcuts_execute_tool,
    browser_shortcuts_list_tool,
    browser_switch_browser_tool,
    browser_tabs_close_tool,
    browser_tabs_context_tool,
    browser_tabs_create_tool,
    browser_upload_image_tool,
    build_agno_browser_functions,
    build_deepagents_browser_tools,
    computer_tool,
    execute_local_cli,
)
from runtime.context import (  # noqa: F401
    _asset_index,
    _LOAD_ASSET_DESCRIPTION,
    _LOAD_ASSET_SCHEMA,
    _lookup_asset,
    _make_load_asset_tool_for_agno,
    _make_load_asset_tool_for_deepagents,
    _mcp_connection_defaults,
    _mcp_tools_for_agno,
    _mcp_tools_for_deepagents,
)
from runtime.graph import (  # noqa: F401
    _load_langgraph_graph,
    _make_reasoning_langgraph_agent_class,
)

__all__ = [
    "get_model",
    "build_agent",
    "run_prompt",
    "stream_agno_events",
    "build_deepagents_graph",
    "build_langgraph_agent",
    "stream_langgraph_events",
    "normalize_tool_schema",
    "json_schema_to_pydantic",
]


if __name__ == "__main__":
    asyncio.run(_demo())
