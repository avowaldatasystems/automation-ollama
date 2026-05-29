import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Employee DB Agent", layout="wide")

# Hide Streamlit chrome
st.markdown("""
<style>
    #MainMenu, header, footer { display: none !important; }
    [data-testid="stToolbar"] { display: none !important; }
    [data-testid="stDecoration"] { display: none !important; }
    .block-container { padding: 0 !important; margin: 0 !important; max-width: 100vw !important; }
    .stApp { background: #0f0f17 !important; }
</style>
""", unsafe_allow_html=True)

# ── Point this at the port your existing FastAPI runs on ──
API_PORT = 8000   # change if yours runs on a different port

components.html(f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{
    width: 100%; height: 100%;
    background: transparent;
    font-family: 'DM Sans', sans-serif;
    overflow: hidden;
  }}

  /* ── FAB ── */
  .fab {{
    position: fixed; bottom: 20px; right: 20px;
    width: 58px; height: 58px; border-radius: 50%;
    background: linear-gradient(135deg, #4f8ef7, #1a56db);
    border: none; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    font-size: 24px;
    box-shadow: 0 4px 20px rgba(79,142,247,0.55);
    transition: transform .2s, box-shadow .2s;
    z-index: 9999;
  }}
  .fab:hover {{ transform: scale(1.08); }}
  .fab:active {{ transform: scale(0.93); }}

  /* ── Option Popup ── */
  .opt-popup {{
    position: fixed; bottom: 88px; right: 20px;
    width: 268px;
    background: #1c1c30;
    border: 1px solid rgba(255,255,255,0.11);
    border-radius: 18px; padding: 18px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.6);
    z-index: 9998; display: none;
    animation: popIn .18s ease;
  }}
  .opt-popup.open {{ display: block; }}
  @keyframes popIn {{
    from {{ opacity:0; transform: translateY(10px) scale(.97); }}
    to   {{ opacity:1; transform: translateY(0) scale(1); }}
  }}
  .pop-title {{ font-size: 14px; font-weight: 700; color: #fff; margin-bottom: 3px; }}
  .pop-sub   {{ font-size: 11px; color: rgba(255,255,255,.4); margin-bottom: 14px; }}
  .divider   {{ height: 1px; background: rgba(255,255,255,.07); margin-bottom: 12px; }}
  .opt-btn {{
    display: flex; align-items: center; gap: 11px;
    width: 100%; padding: 11px 13px;
    border-radius: 11px;
    border: 1px solid rgba(255,255,255,.07);
    background: rgba(255,255,255,.03);
    cursor: pointer; color: #fff; margin-bottom: 8px;
    transition: background .15s, border-color .15s, transform .1s;
    text-align: left;
  }}
  .opt-btn:hover {{ background: rgba(79,142,247,.14); border-color: rgba(79,142,247,.35); transform: translateX(2px); }}
  .opt-btn:last-child {{ margin-bottom: 0; }}
  .oico {{
    width: 34px; height: 34px; border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    font-size: 17px; flex-shrink: 0;
  }}
  .oico.c {{ background: rgba(79,142,247,.18); }}
  .oico.v {{ background: rgba(255,165,0,.14); }}
  .olabel {{ font-size: 13px; font-weight: 600; color: #fff; }}
  .odesc  {{ font-size: 10.5px; color: rgba(255,255,255,.38); margin-top: 1px; }}
  .badge {{
    margin-left: auto;
    background: rgba(239,68,68,.18); color: #f87171;
    border: 1px solid rgba(239,68,68,.28); border-radius: 20px;
    font-size: 9.5px; font-weight: 700; padding: 2px 7px;
  }}

  /* ── Chat Window ── */
  .chat-win {{
    position: fixed; bottom: 88px; right: 20px;
    width: 340px; height: 480px;
    background: #fff; border-radius: 18px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    display: none; flex-direction: column;
    overflow: hidden; z-index: 9997;
    animation: popIn .2s ease;
  }}
  .chat-win.open {{ display: flex; }}

  .chat-header {{
    background: linear-gradient(90deg, #e8192c, #3b2bbf);
    padding: 14px 16px;
    display: flex; align-items: center; gap: 10px;
    flex-shrink: 0;
  }}
  .avatar {{
    width: 34px; height: 34px; border-radius: 50%;
    background: rgba(255,255,255,.22);
    display: flex; align-items: center; justify-content: center; font-size: 18px;
  }}
  .h-title {{ font-size: 14px; font-weight: 700; color: #fff; }}
  .h-sub   {{ font-size: 10.5px; color: rgba(255,255,255,.65); margin-top: 1px; }}
  .close-btn {{
    margin-left: auto; background: transparent; border: none;
    color: rgba(255,255,255,.8); font-size: 20px; cursor: pointer;
    line-height: 1; padding: 2px; transition: color .15s;
  }}
  .close-btn:hover {{ color: #fff; }}

  .chat-messages {{
    flex: 1; overflow-y: auto;
    padding: 14px 12px;
    display: flex; flex-direction: column; gap: 10px;
    background: #f4f6f9;
  }}
  .chat-messages::-webkit-scrollbar {{ width: 4px; }}
  .chat-messages::-webkit-scrollbar-thumb {{ background: #ccc; border-radius: 4px; }}

  .bubble {{
    max-width: 82%; padding: 9px 13px;
    border-radius: 16px; font-size: 13px; line-height: 1.48;
    word-break: break-word;
  }}
  .bubble.bot {{
    background: #fff; color: #1a1a2e;
    align-self: flex-start; border-bottom-left-radius: 4px;
    box-shadow: 0 1px 4px rgba(0,0,0,.08);
  }}
  .bubble.user {{
    background: linear-gradient(135deg, #4f8ef7, #1a56db);
    color: #fff; align-self: flex-end; border-bottom-right-radius: 4px;
  }}
  .bubble.error {{
    background: #fff0f0; color: #c0392b;
    align-self: flex-start; border-bottom-left-radius: 4px;
    box-shadow: 0 1px 4px rgba(0,0,0,.08);
    font-size: 12px;
  }}

  /* Typing dots */
  .typing {{ display: flex; gap: 4px; align-items: center; padding: 4px 2px; }}
  .dot {{
    width: 7px; height: 7px; border-radius: 50%; background: #aaa;
    animation: blink 1.2s infinite;
  }}
  .dot:nth-child(2) {{ animation-delay: .2s; }}
  .dot:nth-child(3) {{ animation-delay: .4s; }}
  @keyframes blink {{ 0%,80%,100%{{opacity:.2;}} 40%{{opacity:1;}} }}

  .chat-input-row {{
    display: flex; align-items: center; gap: 8px;
    padding: 10px 12px;
    border-top: 1px solid #e8eaed; background: #fff; flex-shrink: 0;
  }}
  .chat-input-row input {{
    flex: 1; border: 1px solid #dadce0; border-radius: 22px;
    padding: 8px 14px; font-size: 13px; outline: none;
    font-family: 'DM Sans', sans-serif; transition: border-color .15s;
  }}
  .chat-input-row input:focus {{ border-color: #4f8ef7; }}
  .send-btn {{
    width: 36px; height: 36px; border-radius: 50%;
    background: linear-gradient(135deg, #4f8ef7, #1a56db);
    border: none; cursor: pointer; color: #fff;
    display: flex; align-items: center; justify-content: center;
    font-size: 15px; flex-shrink: 0; transition: transform .15s;
  }}
  .send-btn:hover {{ transform: scale(1.08); }}
  .send-btn:disabled {{ opacity: .5; cursor: default; transform: none; }}

  /* ── Toast ── */
  .toast {{
    position: fixed; bottom: 88px; right: 20px;
    background: #1c1c30; border: 1px solid rgba(239,68,68,.28);
    border-radius: 13px; padding: 13px 16px;
    display: flex; align-items: center; gap: 10px;
    box-shadow: 0 8px 32px rgba(0,0,0,.45);
    z-index: 10000; opacity: 0; transform: translateY(8px);
    transition: opacity .2s, transform .2s; pointer-events: none;
  }}
  .toast.show {{ opacity: 1; transform: translateY(0); }}
  .toast-title {{ color: #fff; font-size: 13px; font-weight: 600; }}
  .toast-sub   {{ color: rgba(255,255,255,.42); font-size: 11px; margin-top: 2px; }}
</style>
</head>
<body>

<!-- Toast -->
<div class="toast" id="toast">
  <span style="font-size:20px">🎙️</span>
  <div>
    <div class="toast-title">Not Yet!</div>
    <div class="toast-sub">Voice feature coming soon</div>
  </div>
</div>

<!-- Option Popup -->
<div class="opt-popup" id="optPopup">
  <div class="pop-title">👨‍💼 Employee Database Agent</div>
  <div class="pop-sub">How would you like to interact?</div>
  <div class="divider"></div>
  <button class="opt-btn" id="chatOptBtn">
    <div class="oico c">💬</div>
    <div><div class="olabel">Chat</div><div class="odesc">Type your questions</div></div>
  </button>
  <button class="opt-btn" id="voiceOptBtn">
    <div class="oico v">🎙️</div>
    <div><div class="olabel">Voice</div><div class="odesc">Speak your questions</div></div>
    <span class="badge">Soon</span>
  </button>
</div>

<!-- Chat Window -->
<div class="chat-win" id="chatWin">
  <div class="chat-header">
    <div class="avatar">👨‍💼</div>
    <div>
      <div class="h-title">Employee DB Agent</div>
      <div class="h-sub">Ask me anything about employees</div>
    </div>
    <button class="close-btn" id="closeChatBtn">✕</button>
  </div>
  <div class="chat-messages" id="chatMessages">
    <div class="bubble bot">Hi! Ask me anything about your employees 👋</div>
  </div>
  <div class="chat-input-row">
    <input type="text" id="chatInput" placeholder="Type a message..." autocomplete="off"/>
    <button class="send-btn" id="sendBtn">➤</button>
  </div>
</div>

<!-- FAB -->
<button class="fab" id="fab">🤖</button>

<script>
  const API_URL = 'http://localhost:{API_PORT}/chat';

  const fab          = document.getElementById('fab');
  const optPopup     = document.getElementById('optPopup');
  const chatWin      = document.getElementById('chatWin');
  const chatMsgs     = document.getElementById('chatMessages');
  const chatInput    = document.getElementById('chatInput');
  const sendBtn      = document.getElementById('sendBtn');
  const closeChatBtn = document.getElementById('closeChatBtn');
  const chatOptBtn   = document.getElementById('chatOptBtn');
  const voiceOptBtn  = document.getElementById('voiceOptBtn');
  const toast        = document.getElementById('toast');

  let popupOpen   = false;
  let chatOpen    = false;
  let chatHistory = [];
  let busy        = false;

  // ── FAB ──
  fab.addEventListener('click', e => {{
    e.stopPropagation();
    if (chatOpen) {{ closeChat(); return; }}
    popupOpen = !popupOpen;
    optPopup.classList.toggle('open', popupOpen);
    fab.textContent = popupOpen ? '✕' : '🤖';
  }});

  // ── Open chat ──
  chatOptBtn.addEventListener('click', e => {{
    e.stopPropagation();
    optPopup.classList.remove('open');
    popupOpen = false;
    fab.textContent = '✕';
    chatWin.classList.add('open');
    chatOpen = true;
    chatInput.focus();
    scrollBottom();
  }});

  // ── Voice ──
  voiceOptBtn.addEventListener('click', e => {{
    e.stopPropagation();
    optPopup.classList.remove('open');
    popupOpen = false;
    fab.textContent = '🤖';
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), 2800);
  }});

  // ── Close ──
  closeChatBtn.addEventListener('click', e => {{ e.stopPropagation(); closeChat(); }});
  function closeChat() {{
    chatWin.classList.remove('open');
    chatOpen = false;
    fab.textContent = '🤖';
  }}

  // ── Outside click ──
  document.addEventListener('click', () => {{
    if (popupOpen) {{
      optPopup.classList.remove('open');
      popupOpen = false;
      fab.textContent = '🤖';
    }}
  }});

  // ── Send ──
  sendBtn.addEventListener('click', sendMessage);
  chatInput.addEventListener('keydown', e => {{ if (e.key === 'Enter' && !busy) sendMessage(); }});

  async function sendMessage() {{
    const text = chatInput.value.trim();
    if (!text || busy) return;

    busy = true;
    chatInput.value = '';
    sendBtn.disabled = true;

    addBubble('user', text);
    const typingEl = addTyping();

    try {{
      const res = await fetch(API_URL, {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ question: text, chat_history: chatHistory }})
      }});

      if (!res.ok) throw new Error('Server error ' + res.status);

      const data = await res.json();
      typingEl.remove();
      addBubble('bot', data.answer);
      chatHistory = data.chat_history;

    }} catch (err) {{
      typingEl.remove();
      addBubble('error', '⚠️ Could not reach server. Is your FastAPI running on port {API_PORT}?');
      console.error(err);
    }}

    busy = false;
    sendBtn.disabled = false;
    chatInput.focus();
  }}

  function addBubble(role, text) {{
    const div = document.createElement('div');
    div.className = 'bubble ' + role;
    div.textContent = text;
    chatMsgs.appendChild(div);
    scrollBottom();
    return div;
  }}

  function addTyping() {{
    const wrap = document.createElement('div');
    wrap.className = 'bubble bot';
    wrap.innerHTML = '<div class="typing"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>';
    chatMsgs.appendChild(wrap);
    scrollBottom();
    return wrap;
  }}

  function scrollBottom() {{ chatMsgs.scrollTop = chatMsgs.scrollHeight; }}
</script>
</body>
</html>
""", height=700, scrolling=False)