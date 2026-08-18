"""Seed example URL -> agent context templates.

This replaces the abandoned GET /agents manifest design. The new approach
uses a local SQLite-backed registry (backend/context_registry.py) mapping URL
glob patterns to context configs (instructions, skills, tools, browser_tools).

The backend matches the current tab's URL against this registry server-side and
injects the matched config directly — no fetching from the domain.

This script seeds one example template for https://example.com with a skill,
an http tool, and a browser_js tool. It's idempotent: existing templates with
the same pattern are replaced before seeding.

Run it as:  cd backend && .venv/bin/python3 examples/example_site_with_agents.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import context_registry


def _template_by_pattern(pattern: str):
    """Find an existing template by pattern, return None if not found."""
    for t in context_registry.list_templates():
        if t["pattern"] == pattern:
            return t
    return None


if __name__ == "__main__":
    pattern = "https://example.com"
    config = {
        "instructions": "Help user on example.com with context-aware answers.",
        "plugins": {
            "skills": [
                {
                    "name": "example_skill",
                    "description": "Example skill for this domain",
                    "content": "This is a reference skill demonstrating how skills are seeded into the registry.",
                }
            ],
            "tools": [
                {
                    "type": "http",
                    "name": "site_echo",
                    "description": "Echo back provided text (reference http tool, endpoint is fake)",
                    "parameters": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                    "endpoint": "http://127.0.0.1:9001/api/echo",
                    "method": "POST",
                }
            ],
            "browser_tools": [
                {
                    "type": "browser_js",
                    "name": "site_get_title",
                    "description": "Get the current page title via JavaScript",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                    "code": "return 'SITEJS:' + document.title;",
                }
            ],
        },
    }

    # Idempotent: remove any existing template with the same pattern first
    existing = _template_by_pattern(pattern)
    if existing:
        context_registry.remove_template(existing["id"])
        print(f"Removed existing template (ID {existing['id']}) for pattern: {pattern}")

    # Insert the new template
    template_id = context_registry.add_template(pattern, config)
    print(f"Seeded template ID {template_id} for pattern: {pattern}")

    # Verify by matching against the pattern
    matched = context_registry.match_context("https://example.com/some/page")
    print(f"\nVerification — match_context('https://example.com/some/page'):")
    print(f"  matched: {matched['matched']}")
    print(f"  matched_patterns: {matched['matched_patterns']}")
    print(f"  instructions: {matched['instructions']}")
    print(f"  skills: {[s['name'] for s in matched['skills']]}")
    print(f"  tools: {[t['name'] for t in matched['tools']]}")
    print(f"  browser_tools: {[b['name'] for b in matched['browser_tools']]}")
