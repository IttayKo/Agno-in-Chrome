"""Generic context-injection helpers shared by both backends.

Kept free of any browser/extension knowledge on purpose — see the block
comment below.
"""

from typing import Any, Dict, List, Optional

from agno.tools.function import Function


# ---------------------------------------------------------------------------
# Generic context-injection helpers, used by build_agent (runtime/agent.py) and
# build_deepagents_graph (runtime/graph.py). Deliberately agnostic to where any
# of this content comes from — a domain-matched config from a URL-template
# registry, a manifest, a hardcoded config, anything. That sourcing decision
# belongs to whichever extension-specific layer calls into this module
# (server.py, for the browser bridge); see build_agent's docstring for why that
# separation matters.
# ---------------------------------------------------------------------------

def normalize_tool_schema(schema: Optional[dict]) -> dict:
    """JSON Schema hardening for a dynamically-declared tool's params,
    generically useful for any caller building tools with a generic **kwargs
    entrypoint: unconditionally sets additionalProperties unequal-to-baseline
    so Agno's params_set_by_user check never falls back to auto-inspecting the
    entrypoint's signature (see the zero-property tool schemas in
    runtime/browser_tools.py's build_agno_browser_functions for the original bug
    this class of fix addresses)."""
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
    (see _bind_origin's docstring in runtime/browser_tools.py); LangChain's would hit the same class of issue."""
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
