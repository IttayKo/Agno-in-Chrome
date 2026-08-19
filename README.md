Agno Bridge — Chrome MV3 extension + FastAPI bridge

Quick start

1. Install Python dependencies using a Python 3.12+ virtual environment:
   cd backend && python3.12 -m venv .venv
   backend/.venv/bin/pip install agno fastapi uvicorn python-dotenv httpx websockets sqlalchemy openai google-genai deepagents langgraph langchain-core langchain-openai langchain-deepseek

2. Start the backend:
   cd backend && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python3 server.py
   or
   cd backend && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python3 -m uvicorn server:app --reload --port 8000
   (PYTHONDONTWRITEBYTECODE=1 is needed to prevent Chrome from refusing to load the extension due to __pycache__ directories.)

3. Load the extension in Chrome:
   - Open chrome://extensions/
   - Enable Developer mode
   - Click "Load unpacked" and select the `extension/` subfolder (/Users/ittaykohavi/Documents/Agno Browser Extension/extension)

4. Open a test tab (e.g., https://example.com) and click the extension icon to attach the debugger.
   The extension will send the current origin to the backend and open a WebSocket or long-poll for actions.

5. Run the example agent to exercise local and browser tools:
   cd backend && .venv/bin/python3 agno_agent.py

Note: If `__pycache__` ever appears in the backend directory (e.g., from running Python without PYTHONDONTWRITEBYTECODE=1), delete it with `rm -rf backend/__pycache__` before the extension will load in Chrome again.

Browser tool registry
- The runtime now exposes a full browser tool registry for extension-driven page automation, including:
  - navigate, tabs_context_mcp, tabs_create_mcp, tabs_close_mcp, switch_browser
  - get_page_text, read_page, find, read_console_messages, read_network_requests
  - form_input, javascript_tool, resize_window, gif_creator, shortcuts_list, shortcuts_execute
  - browser_batch and computer (generic click/type/scroll/keypress/hover actions)
  - upload_image (uploads a base64/URL image through a page's file input)
  - update_plan (records the agent's step-by-step plan, readable via GET /agent/plan?plan_id=...)
- switch_browser re-points the attached CDP debugger at a different tab id without opening or closing anything — unlike tabs_create_mcp/tabs_close_mcp.
- list_connected_browsers / select_browser operate at the connection level, not per-page: every browser-extension instance that opens a WebSocket to this bridge gets its own browser_id (see GET /agent/browsers), and select_browser can pin all subsequent browser actions to one specific connection when more than one is connected — useful if more than one Chrome profile/window is bridged to the same server at once.

Notes and security
- The prototype uses in-memory queues and no authentication — bind to localhost only and run in a controlled dev environment.
- For production, add auth, persistent queues, and origin validation.

Agno model configuration
- The browser bridge and the Agno model are separate layers.
- The extension only executes browser actions via CDP.
- The general-purpose runtime is `backend/agno_runtime.py`, which can be used outside the extension too.
- The Agno `Agent` chooses the model (for example `OpenAIChat(id="gpt-4o-mini")`) and the tool list.
- In this installed Agno version, `show_tool_calls=True` is not a supported constructor field; use `debug_mode=True` or inspect the tool output from `Agent.arun` instead.
- Copy `backend/.env.example` to `backend/.env` and fill in required API keys before running:
  ```
  cp backend/.env.example backend/.env
  # Then edit backend/.env and add your actual API keys for GOOGLE_API_KEY, GEMINI_API_KEY, and DEEPSEEK_API_KEY
  ```

DeepAgents backend (default)
- By default, the chat backend runs a DeepAgents agent (from the `deepagents` package), controlled by `AGNO_BACKEND=langgraph` in `backend/.env`.
- It uses DeepSeek as the LLM via `langchain_deepseek.ChatDeepSeek` (not `langchain_openai.ChatOpenAI` pointed at DeepSeek's endpoint — that wrapper silently drops DeepSeek's `reasoning_content` field, so the thinking block would never show), built fresh per request in `backend/agno_runtime.build_deepagents_graph()`.
- It is driven through Agno's `agno.agents.langgraph.LangGraphAgent` adapter (see `backend/agno_runtime.build_langgraph_agent()`), not a hand-rolled LangGraph driver — so it shares the same conversation history storage (`backend/agno_sessions.db`, a SQLite file) as the native Agno backend.
- It has the same browser tools as the native backend: navigate, computer (click/scroll/type/keypress), get_page_text, read_page, find, javascript_tool, read_console_messages, read_network_requests, and form_input.
- Requires `DEEPSEEK_API_KEY` to be set in `backend/.env`.
- To switch back to the native Agno backend (single-shot tool calling, no DeepAgents planning), set `AGNO_BACKEND=agno` in `backend/.env`, or pass `"backend": "agno"` in an individual chat request.
- To point at a different LangGraph graph instead of the built-in DeepAgents harness, set `LANGGRAPH_GRAPH=module.path:graph_variable` in `backend/.env` (e.g., `LANGGRAPH_GRAPH=my_graphs:my_custom_graph`).

Files
- `extension/manifest.json` — Chrome MV3 manifest
- `extension/background.js` — Service worker module that attaches to tab via chrome.debugger and executes queued CDP actions
- `backend/server.py` — FastAPI bridge with endpoints for context, enqueue, stream, confirm, ws, and helper tool functions
- `backend/agno_runtime.py` — Generic reusable Agno runtime usable outside the browser extension; includes local tools and browser tools
- `backend/agno_agent.py` — Extension-focused example script that invokes the generic runtime for a browser flow

Contact
This scaffold was created by Copilot CLI runtime in VS Code. Modify and extend as needed.
