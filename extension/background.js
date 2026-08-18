// background.js (service worker module)
// Attaches to the active tab via chrome.debugger and bridges it to the local agent server.
//
// IMPORTANT: chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true }) makes Chrome
// open the side panel natively and SUPPRESSES chrome.action.onClicked. Bootstrapping (debugger
// attach, /agent/ws connection, tab sync) must not live only inside onClicked or it never runs
// when the panel is opened via the toolbar/side-panel UI instead of a raw action click. Instead,
// the panel itself pings this service worker with an 'agno:ready' message on load, and the WS
// bridge connects unconditionally as soon as this script starts.

const CDP_VERSION = '1.3';
let attachedDebuggee = null;
let ws = null;
let wsReconnectTimer = null;
let pollAbortController = null;
let debuggerListenersRegistered = false;

try {
  if (chrome && chrome.sidePanel) {
    chrome.runtime.onInstalled.addListener(() => {
      chrome.sidePanel.setOptions({ path: 'sidepanel.html' }).catch((e) => console.warn('sidePanel.setOptions failed', e));
      chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true }).catch((e) => console.warn('sidePanel.setPanelBehavior failed', e));
    });
    chrome.sidePanel.setOptions({ path: 'sidepanel.html' }).catch(() => {});
    chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true }).catch(() => {});
  }
} catch (e) {
  console.debug('sidePanel API not available', e);
}

function getOriginFromUrl(url) {
  try { return new URL(url).origin; } catch (e) { return null; }
}

async function sendContextForOrigin(origin, info = {}) {
  if (!origin) return;
  try {
    const payload = {
      origin,
      system_message: info.system_message || `Browser context for ${origin}`,
      skills: info.skills || [],
      tools: info.tools || []
    };
    await fetch('http://127.0.0.1:8000/agent/context', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
  } catch (e) {
    console.warn('Failed to send context', e);
  }
}

function isAttachableUrl(url) {
  if (!url) return false;
  // chrome://, chrome-extension://, edge://, about:, devtools:// etc. can't be debugged.
  return /^https?:\/\//i.test(url) || /^file:\/\//i.test(url);
}

// Set once the user has attached at least once (via the panel opening, the
// Attach button, or the action click fallback). Auto-follow only kicks in
// after that — switching tabs before the user has ever engaged with the
// bridge should not silently start attaching the debugger to whatever they
// happen to be browsing.
let followActiveTab = false;

function setupGlobalTabTracking() {
  // Previously, tab-follow was scoped to a single tabId (set up once per
  // attach) and never fired for a *different* tab becoming active, so
  // switching tabs left the debugger silently attached to the old one while
  // the panel resolved a fresh origin for whatever tab was now active —
  // browser tool calls would then either fail outright or act on the wrong
  // tab. This listens globally and re-attaches to whichever tab is active.
  chrome.tabs.onActivated.addListener(async (activeInfo) => {
    if (!followActiveTab) return;
    if (attachedDebuggee && attachedDebuggee.tabId === activeInfo.tabId) return;
    try {
      const tab = await chrome.tabs.get(activeInfo.tabId);
      if (!tab || !isAttachableUrl(tab.url)) return;
      await initializeForTab(activeInfo.tabId);
    } catch (e) {
      console.warn('Auto-attach on tab switch failed', e);
    }
  });

  // Re-sync context/subscription when the *currently attached* tab navigates
  // (the chrome.debugger *session* stays attached across a same-tab
  // navigation, but a cross-origin navigation can swap the underlying
  // renderer process, which silently resets CDP domain-enable state —
  // Runtime.consoleAPICalled/Network.requestWillBeSent stop firing, breaking
  // read_console_messages/read_network_requests, even though core commands
  // like Input.dispatchMouseEvent/Runtime.evaluate don't depend on any domain
  // being enabled and keep working regardless). Re-enabling domains and
  // resetting capture state here keeps that in sync with the new page too,
  // not just on a fresh attach.
  chrome.tabs.onUpdated.addListener((updatedTabId, changeInfo, tab) => {
    if (!attachedDebuggee || updatedTabId !== attachedDebuggee.tabId) return;
    if (changeInfo.status === 'complete' && tab.url) {
      resetBrowserCaptureState();
      ensureDebuggerState(attachedDebuggee).catch((e) => console.warn('Debugger state re-bootstrap after navigation failed', e));
      const origin = getOriginFromUrl(tab.url);
      sendContextForOrigin(origin);
      subscribeOrigin(origin);
    }
  });
}

function registerDebuggerListenersOnce() {
  if (debuggerListenersRegistered) return;
  debuggerListenersRegistered = true;

  chrome.debugger.onEvent.addListener((source, method, params) => {
    if (!attachedDebuggee || source.tabId !== attachedDebuggee.tabId) return;
    if (method === 'Runtime.consoleAPICalled') {
      const args = (params && params.args) || [];
      browserCapture.console.push({
        type: 'console',
        timestamp: Date.now(),
        text: args.map(arg => serializeValue(arg.value ?? arg)).join(' '),
        method: 'console'
      });
      if (browserCapture.console.length > 100) browserCapture.console.shift();
    }
    if (method === 'Network.requestWillBeSent') {
      const req = (params && params.request) || {};
      browserCapture.network.push({
        type: 'request',
        timestamp: Date.now(),
        method: req.method || 'GET',
        url: req.url || '',
        headers: req.headers || {}
      });
      if (browserCapture.network.length > 100) browserCapture.network.shift();
    }
  });

  chrome.debugger.onDetach.addListener((source, reason) => {
    if (attachedDebuggee && source.tabId === attachedDebuggee.tabId) {
      console.warn('Debugger detached:', reason);
      attachedDebuggee = null;
      resetBrowserCaptureState();
    }
  });
}

function attachToTab(tabId) {
  return new Promise((resolve, reject) => {
    registerDebuggerListenersOnce();

    if (attachedDebuggee && attachedDebuggee.tabId === tabId) {
      resolve(attachedDebuggee);
      return;
    }

    const debuggee = { tabId };
    const finishAttach = () => {
      chrome.debugger.attach(debuggee, CDP_VERSION, () => {
        const err = chrome.runtime.lastError;
        if (err) {
          console.error('chrome.debugger.attach error:', err.message || JSON.stringify(err));
          return reject(new Error(err.message || JSON.stringify(err)));
        }
        attachedDebuggee = debuggee;
        resetBrowserCaptureState();
        ensureDebuggerState(debuggee).catch((e) => console.warn('Debugger state bootstrap failed', e));
        console.log('Attached debugger to tab', tabId);
        resolve(debuggee);
      });
    };

    if (attachedDebuggee && attachedDebuggee.tabId !== tabId) {
      chrome.debugger.detach(attachedDebuggee, () => finishAttach());
      attachedDebuggee = null;
    } else {
      finishAttach();
    }
  });
}

function sendCDPCommand(debuggee, method, params = {}) {
  return new Promise((resolve, reject) => {
    if (!debuggee) return reject(new Error('No debuggee attached'));
    try {
      chrome.debugger.sendCommand(debuggee, method, params, (result) => {
        const err = chrome.runtime.lastError;
        if (err) return reject(err);
        resolve(result);
      });
    } catch (e) {
      reject(e);
    }
  });
}

async function doClick(x, y) {
  if (!attachedDebuggee) throw new Error('Not attached');
  await sendCDPCommand(attachedDebuggee, 'Input.dispatchMouseEvent', { type: 'mousePressed', x, y, button: 'left', clickCount: 1 });
  await sendCDPCommand(attachedDebuggee, 'Input.dispatchMouseEvent', { type: 'mouseReleased', x, y, button: 'left', clickCount: 1 });
}

async function doScroll(deltaY) {
  if (!attachedDebuggee) throw new Error('Not attached');
  await sendCDPCommand(attachedDebuggee, 'Input.dispatchMouseEvent', { type: 'mouseWheel', x: 0, y: 0, deltaX: 0, deltaY });
}

async function doType(text) {
  if (!attachedDebuggee) throw new Error('Not attached');
  for (const ch of text) {
    await sendCDPCommand(attachedDebuggee, 'Input.dispatchKeyEvent', { type: 'char', text: ch });
  }
}

const browserCapture = { console: [], network: [] };
function resetBrowserCaptureState() {
  browserCapture.console = [];
  browserCapture.network = [];
}

function serializeValue(value) {
  if (value === undefined) return 'undefined';
  if (typeof value === 'string') return value;
  if (typeof value === 'number' || typeof value === 'boolean' || value === null) return value;
  try { return JSON.stringify(value); } catch (e) { return String(value); }
}

async function ensureDebuggerState(debuggee) {
  if (!debuggee) return;
  try {
    await sendCDPCommand(debuggee, 'Page.enable');
    await sendCDPCommand(debuggee, 'Runtime.enable');
    await sendCDPCommand(debuggee, 'Log.enable');
    await sendCDPCommand(debuggee, 'Network.enable');
    await sendCDPCommand(debuggee, 'DOM.enable');
  } catch (e) {
    console.warn('Failed to enable debugger domains', e);
  }
}

async function evaluateInPage(expression, opts = {}) {
  if (!attachedDebuggee) throw new Error('Not attached');
  const response = await sendCDPCommand(attachedDebuggee, 'Runtime.evaluate', {
    expression,
    returnByValue: true,
    awaitPromise: true,
    // replMode is what Chrome DevTools' own console input uses: top-level `await`
    // works without wrapping the code in an async function, and the value of the
    // last expression is the result even with no explicit `return` — the same
    // completion-value semantics as pasting a snippet into the console.
    replMode: !!opts.replMode,
  });
  if (response && response.exceptionDetails) {
    const ex = response.exceptionDetails;
    const desc = (ex.exception && (ex.exception.description || ex.exception.value)) || ex.text || 'Evaluation failed';
    throw new Error(String(desc));
  }
  return response && response.result ? response.result.value : undefined;
}

async function handleAction(action) {
  const { action_id, tool, args = {} } = action;
  try {
    let result = null;

    if (tool === 'click') {
      await doClick(args.x, args.y);
      result = `clicked:${args.x},${args.y}`;
    } else if (tool === 'scroll') {
      const deltaY = args.deltaY ?? args.delta_y ?? args.y_delta ?? args.y ?? 0;
      await doScroll(deltaY);
      result = `scrolled:${deltaY}`;
    } else if (tool === 'type') {
      await doType(args.text || '');
      result = `typed:${(args.text || '').slice(0, 100)}`;
    } else if (tool === 'navigate') {
      const url = args.url;
      await sendCDPCommand(attachedDebuggee, 'Page.navigate', { url });
      result = { status: 'ok', url };
    } else if (tool === 'tabs_context_mcp') {
      result = await chrome.tabs.query({ currentWindow: true });
    } else if (tool === 'tabs_create_mcp') {
      const tab = await chrome.tabs.create({ url: args.url || 'https://example.com' });
      result = { status: 'ok', tabId: tab.id, url: tab.url };
    } else if (tool === 'tabs_close_mcp') {
      await chrome.tabs.remove(args.tab_id);
      result = { status: 'ok', tab_id: args.tab_id };
    } else if (tool === 'get_page_text') {
      result = await evaluateInPage('document.body ? document.body.innerText : ""');
    } else if (tool === 'read_page') {
      const selector = args.selector || 'body';
      result = await evaluateInPage(`(() => { const el = document.querySelector(${JSON.stringify(selector)}) || document.body; return { title: document.title, selector: ${JSON.stringify(selector)}, text: el.innerText || '', htmlLength: (el.innerHTML || '').length }; })()`);
    } else if (tool === 'find') {
      result = await evaluateInPage(`(() => {
        const description = ${JSON.stringify(args.description || '')};
        const candidates = [...document.querySelectorAll('button, a, input, textarea, select, [role="button"], [role="link"], [role="textbox"], [type="submit"]')];
        const normalized = s => (s || '').toLowerCase().replace(/\s+/g, ' ').trim();
        const matches = candidates
          .map((el, idx) => ({
            idx,
            text: normalized(el.innerText || el.value || el.getAttribute('aria-label') || el.getAttribute('placeholder') || ''),
            tag: el.tagName,
            id: el.id || '',
            href: el.href || '',
            selector: el.tagName + (el.id ? '#' + el.id : '') + (el.className ? '.' + String(el.className).split(/\s+/).join('.') : '')
          }))
          .filter(item => !description || item.text.includes(normalized(description)) || (item.selector && item.selector.toLowerCase().includes(normalized(description))));
        return matches.slice(0, 10);
      })()`);
    } else if (tool === 'read_console_messages') {
      result = browserCapture.console.slice(-50);
    } else if (tool === 'read_network_requests') {
      result = browserCapture.network.slice(-50);
    } else if (tool === 'form_input') {
      const selector = args.selector;
      const value = args.value || '';
      const fieldType = args.field_type || 'text';
      result = await evaluateInPage(`(() => {
        const el = document.querySelector(${JSON.stringify(selector)});
        if (!el) return { status: 'not_found', selector: ${JSON.stringify(selector)} };
        if (el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement || el instanceof HTMLSelectElement) {
          el.value = ${JSON.stringify(value)};
          el.dispatchEvent(new Event('input', { bubbles: true }));
          el.dispatchEvent(new Event('change', { bubbles: true }));
          return { status: 'ok', selector: ${JSON.stringify(selector)}, fieldType: ${JSON.stringify(fieldType)}, value: ${JSON.stringify(value)} };
        }
        el.textContent = ${JSON.stringify(value)};
        return { status: 'ok', selector: ${JSON.stringify(selector)}, fieldType: ${JSON.stringify(fieldType)}, value: ${JSON.stringify(value)} };
      })()`);
    } else if (tool === 'javascript_tool') {
      // Evaluate the model's code directly — no wrapping. Models write this a few
      // different ways (a bare final expression with no `return`, an IIFE like
      // `(() => {...; return X;})()` as a statement whose own return value isn't
      // otherwise captured, or code using top-level `await`), and replMode (see
      // evaluateInPage) gives all of those the same completion-value + top-level-await
      // semantics as pasting the snippet into the DevTools console, which is what
      // wrapping it in an extra function can't do without picking one specific style
      // and breaking the others.
      const code = args.code || '';
      try {
        const value = await evaluateInPage(code, { replMode: true });
        result = { status: 'ok', value };
      } catch (e) {
        result = { status: 'error', error: e && e.message ? e.message : String(e) };
      }
    } else if (tool === 'resize_window') {
      const { width, height } = args;
      const current = await chrome.windows.getCurrent();
      await chrome.windows.update(current.id, { width, height });
      result = { status: 'ok', width, height };
    } else if (tool === 'gif_creator') {
      result = {
        status: 'stub',
        message: 'GIF capture is not available in the prototype; implement a screen recorder or screenshot pipeline if you need a real GIF.',
        duration_ms: args.duration_ms || 3000,
        title: args.title || null
      };
    } else if (tool === 'shortcuts_list') {
      result = [
        { name: 'scroll_down', description: 'scroll the page by 400 pixels' },
        { name: 'scroll_up', description: 'scroll the page by -400 pixels' },
        { name: 'home', description: 'scroll to the top of the page' }
      ];
    } else if (tool === 'shortcuts_execute') {
      const shortcut = (args.shortcut_name || '').toLowerCase();
      if (shortcut === 'scroll_down') { await doScroll(400); result = { status: 'ok', shortcut }; }
      else if (shortcut === 'scroll_up') { await doScroll(-400); result = { status: 'ok', shortcut }; }
      else if (shortcut === 'home') { await evaluateInPage('window.scrollTo(0,0)'); result = { status: 'ok', shortcut }; }
      else { result = { status: 'unknown_shortcut', shortcut }; }
    } else if (tool === 'browser_batch') {
      const operations = Array.isArray(args.operations) ? args.operations : [];
      const results = [];
      for (const op of operations) {
        if (!op || !op.tool) continue;
        results.push(await handleAction({ action_id: `${action_id}:${results.length}`, tool: op.tool, args: op.args || {} }));
      }
      result = results;
    } else if (tool === 'computer') {
      const actionName = (args.action || '').toLowerCase();
      if (actionName === 'click') {
        await doClick(args.x || 0, args.y || 0);
        result = { status: 'ok', action: 'click', x: args.x || 0, y: args.y || 0 };
      } else if (actionName === 'scroll') {
        const deltaY = args.delta_y || 0;
        await doScroll(deltaY);
        result = { status: 'ok', action: 'scroll', delta_y: deltaY };
      } else if (actionName === 'type') {
        await doType(args.text || '');
        result = { status: 'ok', action: 'type', text: (args.text || '').slice(0, 100) };
      } else if (actionName === 'keypress') {
        const key = args.key || 'Enter';
        await sendCDPCommand(attachedDebuggee, 'Input.dispatchKeyEvent', { type: 'keyDown', key });
        await sendCDPCommand(attachedDebuggee, 'Input.dispatchKeyEvent', { type: 'keyUp', key });
        result = { status: 'ok', action: 'keypress', key };
      } else if (actionName === 'hover') {
        await sendCDPCommand(attachedDebuggee, 'Input.dispatchMouseEvent', { type: 'mouseMoved', x: args.x || 0, y: args.y || 0, buttons: 0 });
        result = { status: 'ok', action: 'hover', x: args.x || 0, y: args.y || 0 };
      } else if (actionName === 'drag') {
        result = { status: 'stub', action: 'drag', x: args.x || 0, y: args.y || 0 };
      } else {
        result = { status: 'unknown_action', action: actionName };
      }
    } else {
      console.warn('Unknown tool', tool);
      result = { status: 'unknown_tool', tool };
    }

    await fetch('http://127.0.0.1:8000/agent/confirm', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action_id, result })
    });
  } catch (err) {
    console.error('Error executing action', err);
    await fetch('http://127.0.0.1:8000/agent/confirm', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action_id, result: { status: 'error', error: err && err.message ? err.message : String(err) } })
    });
  }
}

// ---- action broadcast channel (server -> extension) ----

// The server only forwards a broadcast action to clients subscribed to its exact
// origin (see server.py broadcast_action). The subscription was previously sent
// once at socket-open time and never refreshed, so it went stale the moment the
// user switched tabs or navigated — every subsequent browser tool call would be
// silently dropped by the server's origin filter and time out client-side after
// 30s with no trace of what happened. Re-subscribing on every tab change (and
// whenever the panel actively attaches/re-syncs) keeps it pinned to whichever
// tab is actually being driven.
let subscribedOrigin = null;
// The most recently *requested* origin, tracked independently of connection
// state. Callers (initializeForTab, the tab-switch/nav listeners) previously
// called subscribeOrigin() before connectWebSocket() had necessarily produced
// an OPEN socket — MV3 service workers are ephemeral and `ws` resets to null
// on every restart, so that request would silently be dropped (subscribeOrigin
// no-ops without an OPEN socket) with nothing to retry it. ws.onopen used to
// paper over this by independently re-querying the active tab, but that races
// against later tab switches and can subscribe to a stale origin. Now every
// caller just updates pendingOrigin, and onopen converges to *that* instead of
// re-deriving its own answer — single source of truth, no lost requests.
let pendingOrigin = null;

function subscribeOrigin(origin) {
  if (origin) pendingOrigin = origin;
  const target = pendingOrigin;
  if (!target || !ws || ws.readyState !== WebSocket.OPEN) return;
  if (target === subscribedOrigin) return;
  try {
    ws.send(JSON.stringify({ subscribe: target }));
    subscribedOrigin = target;
  } catch (e) { /* ignore */ }
}

function connectWebSocket() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
  clearTimeout(wsReconnectTimer);
  subscribedOrigin = null;

  try {
    ws = new WebSocket('ws://127.0.0.1:8000/agent/ws');
  } catch (e) {
    scheduleReconnect();
    return;
  }

  ws.onopen = async () => {
    console.log('WebSocket to agent opened');
    if (pendingOrigin) {
      subscribeOrigin(pendingOrigin);
      return;
    }
    // No request has ever been made yet (very first connect) — fall back to
    // whatever tab is currently active.
    try {
      const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
      const tab = tabs && tabs[0];
      const origin = tab ? getOriginFromUrl(tab.url) : null;
      if (origin) subscribeOrigin(origin);
    } catch (e) { /* ignore */ }
  };
  ws.onmessage = (evt) => {
    try {
      const payload = JSON.parse(evt.data);
      if (payload && payload.action_id) handleAction(payload);
    } catch (e) {
      console.error('Malformed WS payload', e);
    }
  };
  ws.onclose = () => { ws = null; scheduleReconnect(); };
  ws.onerror = () => { try { ws.close(); } catch (e) {} };
}

function scheduleReconnect() {
  clearTimeout(wsReconnectTimer);
  wsReconnectTimer = setTimeout(connectWebSocket, 3000);
}

async function pollLoop() {
  if (pollAbortController) pollAbortController.abort();
  pollAbortController = new AbortController();
  const signal = pollAbortController.signal;

  while (!signal.aborted) {
    try {
      const res = await fetch('http://127.0.0.1:8000/agent/stream', { method: 'POST', signal });
      if (!res.ok) { await new Promise(r => setTimeout(r, 1000)); continue; }
      const data = await res.json();
      if (data && data.action_id) await handleAction(data);
      else await new Promise(r => setTimeout(r, 500));
    } catch (e) {
      if (e.name === 'AbortError') break;
      await new Promise(r => setTimeout(r, 1000));
    }
  }
}

// ---- bootstrap: attach debugger + sync context + open action channel for a tab ----

async function initializeForTab(tabId) {
  if (!tabId) return { ok: false, error: 'No active tab' };
  try {
    await attachToTab(tabId);
  } catch (e) {
    return { ok: false, error: e && e.message ? e.message : String(e) };
  }

  followActiveTab = true;
  connectWebSocket();

  try {
    const tabInfo = await chrome.tabs.get(tabId);
    const origin = getOriginFromUrl(tabInfo && tabInfo.url);
    await sendContextForOrigin(origin);
    subscribeOrigin(origin);
  } catch (e) {
    console.warn('Failed to send initial context', e);
  }

  setTimeout(() => {
    if (!ws || ws.readyState !== WebSocket.OPEN) pollLoop();
  }, 800);

  return { ok: true, tabId };
}

async function currentActiveTabId() {
  const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  return tabs && tabs[0] && tabs[0].id;
}

// Panel pages (side panel / devtools panel) ping this on load, since opening the
// panel via chrome.sidePanel does not fire chrome.action.onClicked.
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (!msg || !msg.type) return;

  if (msg.type === 'agno:ready') {
    currentActiveTabId()
      .then((tabId) => initializeForTab(tabId))
      .then((res) => sendResponse(res))
      .catch((e) => sendResponse({ ok: false, error: e && e.message ? e.message : String(e) }));
    return true; // async response
  }

  if (msg.type === 'agno:attach') {
    currentActiveTabId()
      .then((tabId) => initializeForTab(tabId))
      .then((res) => sendResponse(res))
      .catch((e) => sendResponse({ ok: false, error: e && e.message ? e.message : String(e) }));
    return true; // async response
  }
});

// Fallback: if the icon click ever does fire onClicked (e.g. openPanelOnActionClick
// unset), bootstrap the same way.
chrome.action.onClicked.addListener(async (tab) => {
  try {
    let tabId = tab && tab.id;
    if (!tabId) tabId = await currentActiveTabId();
    if (!tabId) throw new Error('No active tabId available');
    await initializeForTab(tabId);
  } catch (err) {
    console.error('Failed to bootstrap from action click:', err);
  }
});

// Keep the action channel alive as soon as the service worker starts, independent
// of any panel/tab interaction.
connectWebSocket();
setupGlobalTabTracking();

self.addEventListener('unload', () => {
  try { if (attachedDebuggee) chrome.debugger.detach(attachedDebuggee); } catch (e) {}
  try { if (ws) ws.close(); } catch (e) {}
  if (pollAbortController) pollAbortController.abort();
});
