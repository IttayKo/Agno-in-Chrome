// content_script.js
// Listens for messages from the background script to inject or remove a right-side panel iframe
(() => {
  const PANEL_ID = 'agno-sidepanel-root';
  const IFRAME_ID = 'agno-sidepanel-iframe';

  function createPanel() {
    if (document.getElementById(PANEL_ID)) return;
    const root = document.createElement('div');
    root.id = PANEL_ID;
    root.style.position = 'fixed';
    root.style.top = '0';
    root.style.right = '0';
    root.style.height = '100vh';
    root.style.width = '380px';
    root.style.maxWidth = '40%';
    root.style.zIndex = '2147483647';
    root.style.boxShadow = '0 8px 30px rgba(0,0,0,0.6)';

    // header bar
    const header = document.createElement('div');
    header.style.height = '44px';
    header.style.display = 'flex';
    header.style.alignItems = 'center';
    header.style.justifyContent = 'space-between';
    header.style.padding = '8px';
    header.style.background = 'linear-gradient(90deg, rgba(12,18,36,0.95), rgba(12,18,36,0.9))';

    const title = document.createElement('div');
    title.style.color = '#e6eef6';
    title.style.fontWeight = '700';
    title.textContent = 'Agno';

    const closeBtn = document.createElement('button');
    closeBtn.textContent = '✕';
    closeBtn.style.background = 'transparent';
    closeBtn.style.color = '#e6eef6';
    closeBtn.style.border = 'none';
    closeBtn.style.cursor = 'pointer';
    closeBtn.style.fontSize = '16px';
    closeBtn.onclick = () => removePanel();

    header.appendChild(title);
    header.appendChild(closeBtn);
    root.appendChild(header);

    const iframe = document.createElement('iframe');
    iframe.id = IFRAME_ID;
    iframe.src = chrome.runtime.getURL('sidepanel.html');
    iframe.style.border = 'none';
    iframe.style.width = '100%';
    iframe.style.height = 'calc(100% - 44px)';
    iframe.style.background = 'transparent';

    root.appendChild(iframe);
    document.documentElement.appendChild(root);
    document.body.style.marginRight = (parseInt(getComputedStyle(document.body).marginRight || '0') + 380) + 'px';
  }

  function removePanel() {
    const el = document.getElementById(PANEL_ID);
    if (el) el.remove();
    document.body.style.marginRight = '';
  }

  function togglePanel() {
    if (document.getElementById(PANEL_ID)) removePanel(); else createPanel();
  }

  // listen for background messages
  chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (!msg || !msg.action) return;
    if (msg.action === 'toggle_sidepanel') {
      try { togglePanel(); } catch (e) { console.error('toggle sidepanel', e); }
      sendResponse({ ok: true });
    }
  });

  // expose a global for manual toggling if needed
  window.__agno_toggle_sidepanel = togglePanel;
})();
