"""Generic Agno runtime that is usable independently of the browser extension.

The Chrome extension is just one adapter instance on top of the same runtime.
This package defines the actual model/tool layer and can be used either:
- directly from Python in local workflows
- via the browser bridge when browser-specific tools are needed

Layout:
  models.py         get_model() and the optional model-import shims
  reasoning.py      pluggable "communication styles" (thinking vs. answer text)
  browser_tools.py  the browser_*_tool wrappers, origin binding, tool builders
  context.py        generic context injection (skills/knowledge/MCP/schemas)
  agent.py          native Agno backend: build_agent, run_prompt, session db
  graph.py          LangGraph/DeepAgents backend
  events.py         streaming: runs -> transport-agnostic event dicts

`agno_runtime.py` next to this package re-exports the same public API, so
existing `from agno_runtime import ...` callers keep working.

Environment variables:
  OPENAI_API_KEY  required for real model calls
  AGNO_MODEL      defaults to gpt-4o-mini
  BROWSER_ORIGIN  defaults to https://example.com
  AGNO_USE_BROWSER_TOOLS  set to 1 to include browser tools by default
  AGNO_REASONING_STYLE    communication style override (default: auto)
  AGNO_MODEL_PROVIDER     provider for the native Agno backend (default: gemini)
  AGNO_LANGGRAPH_PROVIDER provider for the LangGraph/DeepAgents backend (default: deepseek)
  AGNO_THINKING           request thinking output where the provider needs asking
  AGNO_THINKING_BUDGET    thinking token budget for providers that take one
"""

import os

from dotenv import load_dotenv

# The .env lives next to server.py, one level up from this package.
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"), override=False)

from .models import (  # noqa: E402
    available_langchain_providers,
    available_providers,
    get_langchain_model,
    get_model,
    register_langchain_provider,
    register_provider,
    resolve_thinking,
)
from .reasoning import (  # noqa: E402
    ReasoningStyle,
    available_styles,
    register_style,
    resolve_style,
)
from .context import json_schema_to_pydantic, normalize_tool_schema  # noqa: E402
from .agent import build_agent, run_prompt  # noqa: E402
from .graph import build_deepagents_graph, build_langgraph_agent  # noqa: E402
from .events import stream_agno_events, stream_langgraph_events  # noqa: E402

__all__ = [
    "get_model",
    "get_langchain_model",
    "available_providers",
    "available_langchain_providers",
    "available_styles",
    "register_provider",
    "register_langchain_provider",
    "resolve_thinking",
    "build_agent",
    "run_prompt",
    "stream_agno_events",
    "build_deepagents_graph",
    "build_langgraph_agent",
    "stream_langgraph_events",
    "normalize_tool_schema",
    "json_schema_to_pydantic",
    "ReasoningStyle",
    "resolve_style",
    "register_style",
]
