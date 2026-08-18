"""URL-template -> agent context registry.

A small local table mapping URL glob patterns to a context config
(instructions, skills, knowledge, tools, browser_tools, mcps). server.py
matches the current URL against this registry on every chat request and
translates a match into the generic context-injection params
agno_runtime.build_agent/build_deepagents_graph accept.

Deliberately NOT in agno_runtime.py: that module is meant to stay usable by
any consumer with its own way of sourcing tools/mcps/skills, with no built-in
notion of "there's a URL-template registry." This module — and the
translation in server.py that reads it — is the one place that concept exists.

Reuses the same SQLite file Agno's own session history already lives in
(agno_sessions.db), in a separate table, rather than introducing a second
database file.
"""

import fnmatch
import json
import os
import sqlite3
from typing import Any, Dict, List

_DB_PATH = os.path.join(os.path.dirname(__file__), "agno_sessions.db")


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(_DB_PATH)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS url_context_templates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern TEXT NOT NULL,
            config TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    return con


def add_template(pattern: str, config: dict) -> int:
    """Insert a URL template -> config mapping. `config` shape:
    {"instructions": "...", "plugins": {"skills": [...], "knowledge": [...],
    "tools": [...], "browser_tools": [...], "mcps": [...]}}. All fields
    optional. Returns the new row's id (see remove_template)."""
    con = _connect()
    try:
        cur = con.execute(
            "INSERT INTO url_context_templates (pattern, config) VALUES (?, ?)",
            (pattern, json.dumps(config)),
        )
        con.commit()
        return cur.lastrowid
    finally:
        con.close()


def remove_template(template_id: int) -> None:
    con = _connect()
    try:
        con.execute("DELETE FROM url_context_templates WHERE id = ?", (template_id,))
        con.commit()
    finally:
        con.close()


def list_templates() -> List[Dict[str, Any]]:
    con = _connect()
    try:
        rows = con.execute("SELECT id, pattern, config, created_at FROM url_context_templates ORDER BY id").fetchall()
        return [{"id": r[0], "pattern": r[1], "config": json.loads(r[2]), "created_at": r[3]} for r in rows]
    finally:
        con.close()


def _pattern_matches(pattern: str, url: str) -> bool:
    """A pattern with no wildcard and no path component (just an origin, e.g.
    "https://chatgpt.com") matches every URL under that origin regardless of
    path — writing the bare origin is enough, no need for a trailing "/*".
    Anything else (has a wildcard, or an explicit path with no wildcard) is
    matched literally via glob-style '*' against the full URL — an explicit
    exact path with no wildcard therefore matches only that exact URL."""
    after_scheme = pattern.split("://", 1)[-1]
    if "*" not in pattern and "/" not in after_scheme:
        return url == pattern or url.startswith(pattern + "/") or url.startswith(pattern + "?")
    return fnmatch.fnmatch(url, pattern)


def _specificity(pattern: str) -> int:
    """Rough ordering key: longer, more literal (fewer wildcards) patterns
    count as more specific. Used only to order merged configs so a narrow
    page-level template's instructions are appended after (read as a
    refinement of) a broader domain-level one — doesn't otherwise affect
    correctness, since all matches are merged, none are dropped."""
    return len(pattern) - pattern.count("*") * 2


def match_context(url: str) -> Dict[str, Any]:
    """Match `url` against every stored template and merge all matches into
    one context dict. Returns matched=False with empty lists when nothing
    matches — the expected common case for most URLs, and cheap (a handful of
    in-memory pattern checks, no network call)."""
    templates = [t for t in list_templates() if _pattern_matches(t["pattern"], url)]
    templates.sort(key=lambda t: _specificity(t["pattern"]))

    instructions_parts: List[str] = []
    skills: List[dict] = []
    knowledge: List[dict] = []
    tools: List[dict] = []
    browser_tools: List[dict] = []
    mcps: List[dict] = []

    for t in templates:
        config = t["config"] or {}
        if config.get("instructions"):
            instructions_parts.append(str(config["instructions"]))
        plugins = config.get("plugins") or {}
        skills.extend(plugins.get("skills") or [])
        knowledge.extend(plugins.get("knowledge") or [])
        tools.extend(plugins.get("tools") or [])
        browser_tools.extend(plugins.get("browser_tools") or [])
        mcps.extend(plugins.get("mcps") or [])

    return {
        "matched": bool(templates),
        "matched_patterns": [t["pattern"] for t in templates],
        "instructions": "\n\n".join(instructions_parts) or None,
        "skills": skills,
        "knowledge": knowledge,
        "tools": tools,
        "browser_tools": browser_tools,
        "mcps": mcps,
    }
