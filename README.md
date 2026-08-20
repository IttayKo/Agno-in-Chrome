Agno Bridge — Chrome MV3 extension + FastAPI bridge

Quick start

1. Install Python dependencies using a Python 3.12+ virtual environment:
   cd backend && python3.12 -m venv .venv
   backend/.venv/bin/pip install agno fastapi uvicorn python-dotenv httpx websockets sqlalchemy openai google-genai deepagents langgraph langchain-core langchain-openai langchain-deepseek
   Add the package for any other provider you want the LangGraph backend to use: `langchain-anthropic` (Anthropic), `langchain-google-genai` (Gemini). Every one of these is optional and imported lazily — the native Agno backend runs with none of them installed.

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
- By default it uses DeepSeek as the LLM via `langchain_deepseek.ChatDeepSeek` (not `langchain_openai.ChatOpenAI` pointed at DeepSeek's endpoint — that wrapper silently drops DeepSeek's `reasoning_content` field, so the thinking block would never show), built fresh per request in `backend/agno_runtime.build_deepagents_graph()`. Any other provider can be selected instead — see "Providers, reasoning styles and thinking" below.
- It is driven through Agno's `agno.agents.langgraph.LangGraphAgent` adapter (see `backend/agno_runtime.build_langgraph_agent()`), not a hand-rolled LangGraph driver — so it shares the same conversation history storage (`backend/agno_sessions.db`, a SQLite file) as the native Agno backend.
- It gets a subset of the native backend's browser tools — the same underlying `server.py` helpers, bound the same way, but 14 of the 24 rather than all of them: navigate, computer (click/scroll/type/keypress/hover), get_page_text, read_page, find, javascript_tool, read_console_messages, read_network_requests, form_input, switch_browser, upload_image, list_connected_browsers, select_browser, update_plan.
- Not exposed to the DeepAgents harness (native backend only): tabs_context_mcp, tabs_create_mcp, tabs_close_mcp, browser_scroll_viewport, browser_click_coordinates, resize_window, gif_creator, shortcuts_list, shortcuts_execute, browser_batch. It also has no calculate_product/execute_local_shell, since DeepAgents brings its own file/shell tools. See `backend/runtime/browser_tools.py` (`_deepagents_browser_tools` vs `build_agno_browser_functions`) for the two lists.
- Requires `DEEPSEEK_API_KEY` to be set in `backend/.env` (or the API key of whichever provider you point it at).
- To switch back to the native Agno backend (single-shot tool calling, no DeepAgents planning), set `AGNO_BACKEND=agno` in `backend/.env`, or pass `"backend": "agno"` in an individual chat request.
- To point at a different LangGraph graph instead of the built-in DeepAgents harness, set `LANGGRAPH_GRAPH=module.path:graph_variable` in `backend/.env` (e.g., `LANGGRAPH_GRAPH=my_graphs:my_custom_graph`).

Providers, reasoning styles and thinking
- Three things are configured independently: which **backend** runs the turn, which **provider/model** it runs on, and which **reasoning style** is used to tell the model's thinking apart from its answer text.
- The two backends have separate provider settings on purpose, so they can differ and so an existing `AGNO_MODEL_PROVIDER` never silently re-points the DeepAgents harness (which was DeepSeek-only before this):

  | What | Native Agno backend | LangGraph/DeepAgents backend |
  | --- | --- | --- |
  | Backend | `AGNO_BACKEND=agno` | `AGNO_BACKEND=langgraph` |
  | Provider | `AGNO_MODEL_PROVIDER` (default `gemini`) | `AGNO_LANGGRAPH_PROVIDER` (default `deepseek`) |
  | Model id | `AGNO_MODEL` | `AGNO_LANGGRAPH_MODEL`, else `AGNO_MODEL` |
  | Client class | `agno.models.*` via `runtime.models.get_model()` | `langchain_*.Chat*` via `runtime.models.get_langchain_model()` |

- Providers accepted by both: `deepseek`, `openai`, `anthropic` (alias `claude`), `gemini` (alias `google`), `vllm` (aliases `openai_compatible`, `custom`, for any OpenAI-compatible server via `VLLM_BASE_URL`/`OPENAI_BASE_URL`). Each reads its own key: `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`/`GEMINI_API_KEY`, `VLLM_API_KEY`.
- `AGNO_MODEL` names a model *for the configured provider*. When a request explicitly selects a different provider (the panel's picker doing exactly that), `AGNO_MODEL` is deliberately not applied to it — that provider's own default model id is used instead, since "deepseek-v4-flash" means nothing to Gemini.
- Reasoning styles (`backend/runtime/reasoning.py`) decide where thinking is read from: `deepseek` (`additional_kwargs['reasoning_content']`), `openai` (`reasoning_content`, the Responses API `reasoning.summary`, typed `reasoning` blocks), `anthropic` (typed `thinking`/`redacted_thinking` blocks), `gemini` (text parts flagged `thought: true`), `plain` (no thinking channel). `AGNO_REASONING_STYLE=auto` (the default) picks the one matching the selected provider; set it explicitly only for a custom `LANGGRAPH_GRAPH`, whose provider this bridge cannot know.
- Thinking also has to be *requested* from some providers, which is what `AGNO_THINKING` (`1`/`0`, or `low`/`medium`/`high`) and `AGNO_THINKING_BUDGET` do: Anthropic gets a `thinking` budget, Gemini gets `include_thoughts`/`thinking_budget`, OpenAI gets a reasoning effort, DeepSeek needs nothing (it streams `reasoning_content` unasked). Leaving them unset sends no thinking argument at all, i.e. the provider's own default. If a client library rejects the argument, the model is rebuilt without it and a warning is printed rather than the turn failing.
- Per-request overrides beat env vars, which beat the defaults above. `POST /agent/chat` and the `/agent/chat_ws` websocket both accept `backend`, `model_provider`, `model_id`, `base_url`, `reasoning_style`, `thinking`, `thinking_budget` — and both backends honor them (previously only the native one did).
- `GET /agent/models` reports what this install can be pointed at (providers per backend, registered styles, and the defaults currently in effect). The panel's two dropdowns populate from it and remember the choice in `chrome.storage.local`; leaving them on "Default …"/"Auto style" sends nothing and keeps the backend's own configuration.
- Adding a provider is one entry per factory (`register_provider` / `register_langchain_provider` in `backend/runtime/models.py`), plus a `ReasoningStyle` and a `_PROVIDER_STYLES` entry in `backend/runtime/reasoning.py` if it expresses thinking in a new way. No adapter or endpoint changes.

Files
- `extension/manifest.json` — Chrome MV3 manifest
- `extension/background.js` — Service worker module that attaches to tab via chrome.debugger and executes queued CDP actions
- `backend/server.py` — FastAPI bridge with endpoints for context, enqueue, stream, confirm, ws, and helper tool functions
- `backend/agno_runtime.py` — Generic reusable Agno runtime usable outside the browser extension; includes local tools and browser tools
- `backend/agno_agent.py` — Extension-focused example script that invokes the generic runtime for a browser flow

Contact
This scaffold was created by Copilot CLI runtime in VS Code. Modify and extend as needed.
