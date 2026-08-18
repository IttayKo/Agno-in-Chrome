document.addEventListener('DOMContentLoaded', () => {
  const messagesEl = document.getElementById('messages');
  const sendBtn = document.getElementById('sendBtn');
  const clearBtn = document.getElementById('clearBtn');
  const openUiBtn = document.getElementById('openUiBtn');
  const quickCmd = document.getElementById('quickCmd');
  const browserToolsInput = document.getElementById('browserToolsInput');
  const statusBox = document.getElementById('statusBox');
  const statusText = statusBox ? statusBox.querySelector('.status-text') : null;
  const modelSelect = document.getElementById('modelSelect');

  // The assistant text node currently receiving stream deltas, if any, plus the
  // raw markdown accumulated for it so far (re-rendered on every delta).
  // Cleared whenever a tool call interrupts the stream so the next delta
  // starts a fresh bubble instead of splicing into stale text.
  let currentAssistantContent = null;
  let currentAssistantRaw = '';
  // The reasoning/thinking block currently receiving deltas, if any.
  let currentThinking = null;
  let currentThinkingRaw = '';
  // In-flight tool calls keyed by tool_call_id, correlating a 'call' event with its 'output'.
  const toolCallsById = new Map();
  // The socket for the in-flight request, if any — lets the Stop button actually
  // cancel generation server-side instead of just closing the client's ears.
  let activeSocket = null;
  // A stable id for this conversation, sent with every message so the backend
  // agent sees prior turns instead of starting fresh each time. Regenerated
  // when the user clears the conversation.
  let sessionId = crypto.randomUUID();
  // The tab/URL the previous message was sent against, so the next message can
  // tell the model when the user has switched tabs or navigated in between —
  // otherwise the model keeps reasoning about a page that's no longer on screen.
  let lastKnownTabId = null;
  let lastKnownUrl = null;
  // Names of context-registry tools/skills the model was last told it has for
  // the current page — compared on every page change so the tab-change note
  // can also say when its available tools/skills changed, not just the URL.
  let lastKnownContextKey = null;

  function setSendMode(mode) {
    if (!sendBtn) return;
    sendBtn.dataset.mode = mode;
    sendBtn.textContent = mode === 'stop' ? 'Stop' : 'Send';
    sendBtn.disabled = false;
  }

  // ---- minimal, dependency-free markdown -> sanitized HTML ----
  // Agno agents are run with markdown=True; escape all raw text before ever
  // introducing markup so nothing from the model/page can inject HTML.
  function escapeHtml(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function renderInline(text) {
    let out = escapeHtml(text);
    out = out.replace(/`([^`]+)`/g, (m, code) => `<code>${code}</code>`);
    out = out.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    out = out.replace(/(^|[^*])\*([^*\s][^*]*)\*(?!\*)/g, '$1<em>$2</em>');
    out = out.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
    return out;
  }

  function renderMarkdown(md) {
    const lines = String(md || '').split('\n');
    let html = '';
    let inCode = false;
    let codeBuf = [];
    let listType = null;
    let listBuf = [];

    function flushList() {
      if (listType) {
        html += `<${listType}>${listBuf.join('')}</${listType}>`;
        listType = null;
        listBuf = [];
      }
    }

    for (const line of lines) {
      const fence = line.match(/^```(\w*)\s*$/);
      if (fence) {
        if (!inCode) { inCode = true; codeBuf = []; flushList(); }
        else { html += `<pre class="md-code"><code>${escapeHtml(codeBuf.join('\n'))}</code></pre>`; inCode = false; }
        continue;
      }
      if (inCode) { codeBuf.push(line); continue; }

      const ol = line.match(/^\s*\d+\.\s+(.*)$/);
      const ul = line.match(/^\s*[-*+]\s+(.*)$/);
      if (ol) { if (listType !== 'ol') { flushList(); listType = 'ol'; } listBuf.push(`<li>${renderInline(ol[1])}</li>`); continue; }
      if (ul) { if (listType !== 'ul') { flushList(); listType = 'ul'; } listBuf.push(`<li>${renderInline(ul[1])}</li>`); continue; }
      flushList();

      const h = line.match(/^(#{1,6})\s+(.*)$/);
      if (h) { html += `<h${h[1].length}>${renderInline(h[2])}</h${h[1].length}>`; continue; }

      if (line.trim() === '') continue;
      html += `<p>${renderInline(line)}</p>`;
    }
    if (inCode) html += `<pre class="md-code"><code>${escapeHtml(codeBuf.join('\n'))}</code></pre>`;
    flushList();
    return html;
  }

  function setStatus(state, text) {
    if (!statusBox) return;
    statusBox.dataset.state = state;
    if (statusText) statusText.textContent = text;
  }

  function isNearBottom() {
    if (!messagesEl) return true;
    return messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight < 60;
  }

  function scrollToBottom(force) {
    if (!messagesEl) return;
    if (force || isNearBottom()) messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function clearEmptyHint() {
    const hint = messagesEl ? messagesEl.querySelector('.empty-hint') : null;
    if (hint) hint.remove();
  }

  function showEmptyHint() {
    if (!messagesEl) return;
    clearEmptyHint();
    const hint = document.createElement('div');
    hint.className = 'empty-hint';
    hint.textContent = 'Ask Agno to do something on the current tab — try "scroll down 400" or "calculate 583 * 22".';
    messagesEl.appendChild(hint);
  }

  function addMessage(role, content) {
    if (!messagesEl) return null;
    clearEmptyHint();
    const wasNearBottom = isNearBottom();
    const el = document.createElement('div');
    el.className = 'msg ' + (role || 'assistant');
    const body = document.createElement('div');
    body.className = 'content';
    if (role === 'assistant') body.innerHTML = renderMarkdown(content || '');
    else body.textContent = content || '';
    el.appendChild(body);
    messagesEl.appendChild(el);
    scrollToBottom(wasNearBottom);
    return body;
  }

  function summarize(value) {
    if (value === undefined || value === null) return '';
    let s;
    if (typeof value === 'string') s = value;
    else { try { s = JSON.stringify(value); } catch (e) { s = String(value); } }
    return s.length > 140 ? s.slice(0, 140) + '…' : s;
  }

  function isErrorResult(result) {
    if (result && typeof result === 'object') {
      return result.status === 'error' || !!result.error;
    }
    return typeof result === 'string' && /^error/i.test(result.trim());
  }

  function createTraceRow(tool, args, source) {
    if (!messagesEl) return null;
    clearEmptyHint();
    const wasNearBottom = isNearBottom();

    source = source || 'model';

    // Extract _site_code from args if present
    let siteCode = null;
    let displayArgs = args;
    if (args && typeof args === 'object' && '_site_code' in args) {
      siteCode = args._site_code;
      displayArgs = Object.keys(args).length > 1 ? Object.fromEntries(Object.entries(args).filter(([k]) => k !== '_site_code')) : {};
    }

    const trace = document.createElement('div');
    trace.className = 'trace';
    trace.dataset.status = 'running';
    // Auto-expand if this is a site JS tool with code
    trace.dataset.open = siteCode ? 'true' : 'false';

    const row = document.createElement('button');
    row.className = 'trace-row';
    row.type = 'button';
    row.setAttribute('aria-expanded', siteCode ? 'true' : 'false');

    const dot = document.createElement('span');
    dot.className = 'trace-dot';

    const name = document.createElement('span');
    name.className = 'trace-name';
    name.textContent = tool || 'tool';

    // Add source badge if source is 'site'
    if (source === 'site') {
      const badge = document.createElement('span');
      badge.className = 'trace-badge';
      badge.textContent = 'site';
      row.appendChild(dot);
      row.appendChild(name);
      row.appendChild(badge);
    } else {
      row.appendChild(dot);
      row.appendChild(name);
    }

    const summary = document.createElement('span');
    summary.className = 'trace-summary';
    summary.textContent = summarize(displayArgs) || 'running…';

    const time = document.createElement('span');
    time.className = 'trace-time';
    time.textContent = '0ms';

    const chevron = document.createElement('span');
    chevron.className = 'trace-chevron';
    chevron.textContent = '⌄';

    row.appendChild(summary);
    row.appendChild(time);
    row.appendChild(chevron);

    const bodyEl = document.createElement('div');
    bodyEl.className = 'trace-body';

    const argsBlock = document.createElement('div');
    argsBlock.className = 'trace-block';
    const argsLabel = document.createElement('span');
    argsLabel.className = 'trace-label';
    argsLabel.textContent = 'args';
    const argsJson = document.createElement('pre');
    argsJson.className = 'trace-json';
    argsJson.textContent = (() => { try { return JSON.stringify(displayArgs ?? {}, null, 2); } catch (e) { return String(displayArgs); } })();
    argsBlock.appendChild(argsLabel);
    argsBlock.appendChild(argsJson);

    const resultBlock = document.createElement('div');
    resultBlock.className = 'trace-block';
    resultBlock.hidden = true;
    const resultLabel = document.createElement('span');
    resultLabel.className = 'trace-label';
    resultLabel.textContent = 'result';
    const resultJson = document.createElement('pre');
    resultJson.className = 'trace-json';
    resultBlock.appendChild(resultLabel);
    resultBlock.appendChild(resultJson);

    bodyEl.appendChild(argsBlock);

    // Add code block if _site_code is present
    if (siteCode) {
      const codeBlock = document.createElement('div');
      codeBlock.className = 'trace-block';
      const codeLabel = document.createElement('span');
      codeLabel.className = 'trace-label';
      codeLabel.textContent = 'code';
      const codePre = document.createElement('pre');
      codePre.className = 'trace-json';
      codePre.textContent = siteCode;
      codeBlock.appendChild(codeLabel);
      codeBlock.appendChild(codePre);
      bodyEl.appendChild(codeBlock);
    }

    bodyEl.appendChild(resultBlock);

    row.addEventListener('click', () => {
      const open = trace.dataset.open !== 'true';
      trace.dataset.open = String(open);
      row.setAttribute('aria-expanded', String(open));
    });

    trace.appendChild(row);
    trace.appendChild(bodyEl);
    messagesEl.appendChild(trace);
    scrollToBottom(wasNearBottom);

    const startedAt = Date.now();
    const timerId = setInterval(() => {
      time.textContent = (Date.now() - startedAt) + 'ms';
    }, 100);

    return { trace, summary, time, resultBlock, resultJson, startedAt, timerId };
  }

  function finishTrace(record, status, result) {
    if (!record) return;
    clearInterval(record.timerId);
    record.trace.dataset.status = status;
    record.time.textContent = (Date.now() - record.startedAt) + 'ms';
    record.summary.textContent = summarize(result) || (status === 'error' ? 'failed' : 'done');
    if (result !== undefined && result !== null) {
      record.resultBlock.hidden = false;
      try { record.resultJson.textContent = JSON.stringify(result, null, 2); }
      catch (e) { record.resultJson.textContent = String(result); }
    }
  }

  function handleToolEvent(msg) {
    // A tool call interrupts the current stream bubble — clear its blinking cursor
    // before dropping the reference, or it's stuck blinking on stale text forever.
    if (currentAssistantContent) currentAssistantContent.classList.remove('is-streaming');
    currentAssistantContent = null;
    currentAssistantRaw = '';
    finalizeThinking();
    const tool = msg.tool || 'tool';
    const source = msg.source || 'model';

    if (msg.phase === 'call') {
      const record = createTraceRow(tool, msg.args, source);
      toolCallsById.set(msg.tool_call_id || `${tool}:${Date.now()}`, record);
      return;
    }

    // phase === 'output' — resolve the call this belongs to by id; a call event
    // for it may not have arrived (or may lack an id), so fall back gracefully.
    let record = msg.tool_call_id ? toolCallsById.get(msg.tool_call_id) : null;
    if (record) toolCallsById.delete(msg.tool_call_id);
    else record = createTraceRow(tool, msg.args, source);

    if (msg.error) {
      finishTrace(record, 'error', { error: msg.error });
    } else {
      const status = isErrorResult(msg.result) ? 'error' : 'done';
      finishTrace(record, status, msg.result);
    }
  }

  // ---- thinking / reasoning block: a collapsible console-row like the tool
  // trace, but representing the model's reasoning rather than a browser action.
  function createThinkingBlock() {
    if (!messagesEl) return null;
    clearEmptyHint();
    const wasNearBottom = isNearBottom();

    const block = document.createElement('div');
    block.className = 'thinking';
    block.dataset.open = 'true';

    const row = document.createElement('button');
    row.className = 'thinking-row';
    row.type = 'button';
    row.setAttribute('aria-expanded', 'true');

    const label = document.createElement('span');
    label.className = 'thinking-label';
    label.textContent = 'Thinking';

    const chevron = document.createElement('span');
    chevron.className = 'trace-chevron';
    chevron.textContent = '⌄';

    row.appendChild(label);
    row.appendChild(chevron);

    const body = document.createElement('div');
    body.className = 'thinking-body is-streaming';

    row.addEventListener('click', () => {
      const open = block.dataset.open !== 'true';
      block.dataset.open = String(open);
      row.setAttribute('aria-expanded', String(open));
    });

    block.appendChild(row);
    block.appendChild(body);
    messagesEl.appendChild(block);
    scrollToBottom(wasNearBottom);

    return { block, body, row };
  }

  function finalizeThinking() {
    if (!currentThinking) return;
    currentThinking.body.classList.remove('is-streaming');
    currentThinking.block.dataset.open = 'false';
    currentThinking.row.setAttribute('aria-expanded', 'false');
    currentThinking = null;
    currentThinkingRaw = '';
  }

  // Cheap lookup of what the context-registry has configured for a URL, so the
  // tab-change note can mention it. Best-effort: any failure (backend down,
  // slow) just means the note skips the context clause, never blocks sending.
  async function fetchContextSummary(url) {
    if (!url) return null;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 1500);
    try {
      const resp = await fetch(`http://127.0.0.1:8000/agent/context_summary?url=${encodeURIComponent(url)}`, { signal: controller.signal });
      if (!resp.ok) return null;
      return await resp.json();
    } catch (e) {
      return null;
    } finally {
      clearTimeout(timeoutId);
    }
  }

  async function sendPrompt() {
    const displayPrompt = quickCmd ? quickCmd.value.trim() : '';
    if (!displayPrompt) return;
    if (quickCmd) {
      quickCmd.value = '';
      quickCmd.disabled = true;
    }
    setSendMode('stop');
    if (currentAssistantContent) currentAssistantContent.classList.remove('is-streaming');
    currentAssistantContent = null;
    currentAssistantRaw = '';
    finalizeThinking();
    toolCallsById.clear();

    addMessage('user', displayPrompt);
    setStatus('sending', 'Sending…');

    // Ensure background.js is attached + subscribed to whichever tab is
    // *currently* active before this message goes out. background.js's
    // chrome.tabs.onActivated listener reacts to tab switches on its own
    // schedule, with no coordination with sendPrompt() — a fast switch-then-
    // send could race it: this prompt would carry the new tab's origin while
    // background.js is still attached/subscribed to the old one, so the
    // server's broadcast filter (exact origin match) silently drops every
    // action for the new origin and tool calls just hang until they time out.
    // Re-running the same handshake here closes that race; it's cheap when
    // nothing actually changed (attachToTab/subscribeOrigin both no-op if
    // already correct).
    try {
      await chrome.runtime.sendMessage({ type: 'agno:ready' });
    } catch (e) { /* e.g. current tab isn't attachable (chrome://, etc.) — send anyway */ }

    let origin = null;
    let currentTabId = null;
    let currentUrl = null;
    try {
      const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
      const tab = tabs && tabs[0];
      if (tab) {
        currentTabId = tab.id;
        currentUrl = tab.url || null;
        if (tab.url) {
          try { origin = new URL(tab.url).origin; } catch (e) { origin = null; }
        }
      }
    } catch (e) {
      console.warn('Could not resolve active tab origin', e);
    }

    // Tell the model when the page it's about to act on isn't the one it last
    // saw — background.js re-attaches automatically on a tab switch, but the
    // model itself has no way to know the switch happened unless told. Also
    // mention when the context-registry tools/skills available on this page
    // changed, since that's silently re-resolved per-message on the backend
    // with no other narrative cue to the model that it happened.
    let prompt = displayPrompt;
    if (currentTabId !== null) {
      const target = currentUrl || origin || 'a different page';
      const tabChanged = lastKnownTabId !== null && currentTabId !== lastKnownTabId;
      const pageNavigated = !tabChanged && lastKnownUrl !== null && currentUrl !== lastKnownUrl && lastKnownTabId === currentTabId;
      if (tabChanged || pageNavigated) {
        let note = tabChanged
          ? `The active browser tab changed since your last message — you are now on ${target}.`
          : `The page navigated since your last message — you are now on ${target}.`;

        const summary = await fetchContextSummary(currentUrl);
        if (summary) {
          const contextKey = summary.matched
            ? JSON.stringify({ p: summary.matched_patterns, t: summary.tool_names, s: summary.skill_names, k: summary.knowledge_names, i: summary.has_instructions })
            : null;
          if (contextKey !== lastKnownContextKey) {
            if (summary.matched) {
              const names = [...(summary.tool_names || []), ...(summary.skill_names || [])];
              note += names.length
                ? ` This page also has extra tools/skills available: ${names.join(', ')}.`
                : ` This page also has extra instructions loaded for it.`;
            } else if (lastKnownContextKey !== null) {
              note += ` This page has no extra tools or instructions configured (the previous page's no longer apply).`;
            }
          }
          lastKnownContextKey = contextKey;
        }
        prompt = `[${note}]\n\n${displayPrompt}`;
      }
      lastKnownTabId = currentTabId;
      lastKnownUrl = currentUrl;
    }

    function releaseComposer() {
      if (quickCmd) { quickCmd.disabled = false; quickCmd.focus(); }
      setSendMode('send');
      activeSocket = null;
    }

    const includeBrowserTools = browserToolsInput ? browserToolsInput.checked : false;

    // Resolves with the open, sent WebSocket on success, or null if it never
    // managed to open/send in time — callers should fall back to HTTP in that case.
    function openChatStream() {
      return new Promise((resolve) => {
        let settled = false;
        const settle = (result) => { if (!settled) { settled = true; resolve(result); } };

        let socket;
        try {
          socket = new WebSocket('ws://127.0.0.1:8000/agent/chat_ws');
        } catch (e) {
          settle(null);
          return;
        }

        const openTimeout = setTimeout(() => {
          if (socket.readyState !== WebSocket.OPEN) {
            try { socket.close(); } catch (e) {}
            settle(null);
          }
        }, 2500);

        socket.onopen = () => {
          clearTimeout(openTimeout);
          try {
            socket.send(JSON.stringify({ prompt, origin, url: currentUrl, include_browser_tools: includeBrowserTools, session_id: sessionId }));
            activeSocket = socket;
            settle(socket);
          } catch (e) {
            settle(null);
          }
        };

        socket.onmessage = (evt) => {
          let msg;
          try { msg = JSON.parse(evt.data); } catch (e) { return; }

          if (msg.type === 'assistant.start') {
            setStatus('streaming', 'Responding…');
            currentAssistantContent = null;
            currentAssistantRaw = '';
          } else if (msg.type === 'assistant.thinking') {
            if (!currentThinking) {
              // Thinking interrupting active text output: close that bubble so
              // resumed text after this thinking block starts a fresh one below it,
              // instead of appending back into the bubble that came before.
              if (currentAssistantContent) currentAssistantContent.classList.remove('is-streaming');
              currentAssistantContent = null;
              currentAssistantRaw = '';
              currentThinking = createThinkingBlock();
            }
            currentThinkingRaw += msg.chunk;
            currentThinking.body.textContent = currentThinkingRaw;
            scrollToBottom(false);
          } else if (msg.type === 'assistant.delta') {
            finalizeThinking();
            if (!currentAssistantContent) {
              currentAssistantContent = addMessage('assistant', '');
              currentAssistantContent.classList.add('is-streaming');
              currentAssistantRaw = '';
            }
            currentAssistantRaw += msg.chunk;
            currentAssistantContent.innerHTML = renderMarkdown(currentAssistantRaw);
            scrollToBottom(false);
          } else if (msg.type === 'assistant.done') {
            finalizeThinking();
            if (currentAssistantContent) currentAssistantContent.classList.remove('is-streaming');
            currentAssistantContent = null;
            currentAssistantRaw = '';
            setStatus('done', 'Ready');
            releaseComposer();
            try { socket.close(); } catch (e) {}
          } else if (msg.type === 'tool.event') {
            handleToolEvent(msg);
          } else if (msg.type === 'error') {
            addMessage('assistant', `Error: ${msg.error}`);
            setStatus('error', 'Error');
            releaseComposer();
            try { socket.close(); } catch (e) {}
          }
        };

        socket.onerror = () => {
          clearTimeout(openTimeout);
          if (settled) {
            addMessage('assistant', 'Stream connection error.');
            setStatus('error', 'Error');
            releaseComposer();
          }
          settle(null);
        };
        socket.onclose = () => {
          if (sendBtn && sendBtn.disabled) releaseComposer();
        };
      });
    }

    const stream = await openChatStream();
    if (!stream) {
      try {
        const resp = await fetch('http://127.0.0.1:8000/agent/chat', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt, origin, url: currentUrl, include_browser_tools: includeBrowserTools, session_id: sessionId })
        });
        const data = await resp.json();
        const assistantText = (data && data.response) ? data.response : (data && data.error ? `Error: ${data.error}` : 'No response');
        addMessage('assistant', assistantText);
        setStatus('done', 'Ready');
      } catch (e) {
        addMessage('assistant', 'Send failed: backend unreachable at 127.0.0.1:8000. Is server.py running?');
        setStatus('error', 'Backend unreachable');
      } finally {
        releaseComposer();
      }
    }
  }

  function clearConversation() {
    if (!messagesEl) return;
    messagesEl.innerHTML = '';
    currentAssistantContent = null;
    currentAssistantRaw = '';
    currentThinking = null;
    currentThinkingRaw = '';
    toolCallsById.clear();
    sessionId = crypto.randomUUID(); // start a genuinely fresh backend conversation, not just a fresh view of the old one
    lastKnownTabId = null;
    lastKnownUrl = null;
    lastKnownContextKey = null;
    showEmptyHint();
  }

  // Stops generation client-side (drop the connection) and server-side (cancel
  // the running agent task), so a stopped browser-tool run doesn't keep clicking
  // or scrolling the page after the user asked it to stop.
  function stopGeneration() {
    if (activeSocket) {
      try { activeSocket.send(JSON.stringify({ cancel: true })); } catch (e) {}
      try { activeSocket.close(); } catch (e) {}
      activeSocket = null;
    }
    finalizeThinking();
    if (currentAssistantContent) currentAssistantContent.classList.remove('is-streaming');
    currentAssistantContent = null;
    currentAssistantRaw = '';
    setStatus('idle', 'Stopped');
    if (quickCmd) { quickCmd.disabled = false; quickCmd.focus(); }
    setSendMode('send');
  }

  if (clearBtn) clearBtn.addEventListener('click', clearConversation);
  if (openUiBtn) openUiBtn.addEventListener('click', () => {
    chrome.tabs.create({ url: 'http://127.0.0.1:8000/' });
  });
  if (modelSelect) modelSelect.addEventListener('change', () => {
    console.log('Model selection is informational only; set AGNO_MODEL_PROVIDER on the backend to change models.');
  });

  if (quickCmd) quickCmd.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (sendBtn && !sendBtn.disabled && sendBtn.dataset.mode !== 'stop') sendBtn.click();
    }
  });

  if (sendBtn) sendBtn.addEventListener('click', () => {
    if (sendBtn.dataset.mode === 'stop') stopGeneration();
    else sendPrompt();
  });
  setSendMode('send');

  // Bootstrap the background bridge (WS + tab sync) as soon as the panel opens,
  // since chrome.sidePanel's openPanelOnActionClick suppresses action.onClicked.
  // Always auto-attaches to the current tab — there is no manual attach button.
  chrome.runtime.sendMessage({ type: 'agno:ready' }).then((resp) => {
    if (resp && resp.ok) {
      setStatus('attached', resp.tabId ? `Attached (tab ${resp.tabId})` : 'Attached');
    } else if (resp && resp.error) {
      setStatus('error', resp.error);
    }
  }).catch(() => {});

  showEmptyHint();
});
