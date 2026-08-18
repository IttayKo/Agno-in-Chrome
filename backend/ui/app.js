const form = document.getElementById('chatForm');
const promptInput = document.getElementById('prompt');
const messagesEl = document.getElementById('messages');
const traceListEl = document.getElementById('traceList');
const originInput = document.getElementById('origin');
const browserToolsInput = document.getElementById('browserTools');
const clearChatBtn = document.getElementById('clearChat');

function addMessage(role, content) {
  const row = document.createElement('div');
  row.className = `message ${role}`;
  const label = document.createElement('span');
  label.className = 'role';
  label.textContent = role;
  const body = document.createElement('div');
  body.textContent = content || '(empty)';
  row.appendChild(label);
  row.appendChild(body);
  messagesEl.appendChild(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function addTrace(label, content) {
  const item = document.createElement('li');
  item.className = 'trace-item';
  const badge = document.createElement('small');
  badge.textContent = label;
  const body = document.createElement('div');
  body.textContent = content || '(no data)';
  item.appendChild(badge);
  item.appendChild(body);
  traceListEl.appendChild(item);
}

function clearTrace() {
  traceListEl.innerHTML = '';
}

clearChatBtn.addEventListener('click', () => {
  messagesEl.innerHTML = '';
  clearTrace();
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();

  const prompt = promptInput.value.trim();
  if (!prompt) return;

  addMessage('user', prompt);
  addTrace('request', prompt);
  promptInput.value = '';

  try {
    const response = await fetch('/agent/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt,
        origin: originInput.value || undefined,
        include_browser_tools: browserToolsInput.checked,
      }),
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || 'Agent request failed');
    }

    const traceItems = Array.isArray(payload.tool_trace) ? payload.tool_trace : [];
    const responseText = payload.response || '(no response)';
    addMessage('assistant', responseText);
    addTrace('response', responseText);
    for (const item of traceItems) {
      const label = item.name || item.role || 'trace';
      const body = item.content || JSON.stringify(item.args || {}) || '(trace item)';
      addTrace(label, body);
    }
  } catch (error) {
    addMessage('assistant', `Error: ${error.message}`);
    addTrace('error', error.message);
  }
});
