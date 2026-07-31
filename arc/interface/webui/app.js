// Wires the orb, the ⌘S binding and the SSE stream together.
//
// Until arc/voice/ exists there is no microphone, so `level` is fed by a small
// synthetic envelope while ARC is speaking. That keeps the UI reviewable on its own
// and means the swap to real audio is one function, not a rewrite — see setLevelSource.

import { AppState, ACTIVITY } from './state.js';
import { InputManager } from './input.js';
import { OrbRenderer } from './orb.js';

const state = new AppState();
new InputManager(state);
const orb = new OrbRenderer(document.getElementById('stage'), state);

const el = {
  mute: document.getElementById('mute'),
  status: document.getElementById('status'),
  transcript: document.getElementById('transcript'),
  insp: document.getElementById('inspector'),
  cat: document.getElementById('insp-cat'),
  name: document.getElementById('insp-name'),
  st: document.getElementById('insp-state'),
  args: document.getElementById('insp-args'),
  resWrap: document.getElementById('insp-result-wrap'),
  res: document.getElementById('insp-result'),
  close: document.getElementById('insp-close'),
};

/* ── indicator ──────────────────────────────────────────────────────────── */
state.on('change', (s) => {
  el.mute.classList.toggle('on', s.listening);
  el.mute.firstChild.nodeValue = s.listening ? '● LISTENING ' : '● MUTED ';
  el.status.textContent =
    s.activity === ACTIVITY.THINKING ? 'thinking' :
    s.activity === ACTIVITY.SPEAKING ? 'speaking' : '';
});

/* ── inspector ──────────────────────────────────────────────────────────── */
state.on('inspect', (tool) => {
  if (!tool) { el.insp.hidden = true; return; }
  el.insp.hidden = false;
  el.cat.textContent = tool.category;
  el.cat.style.color = `var(--cat-${tool.category}, var(--cat-general))`;
  el.name.textContent = tool.name;
  el.st.textContent = tool.state === 'running'
    ? 'running…'
    : `${tool.state} · ${Math.round((tool.ended - tool.started))} ms`;
  el.st.dataset.s = tool.state;
  el.args.textContent = JSON.stringify(tool.args, null, 2);
  const has = tool.result != null;
  el.resWrap.hidden = !has;
  if (has) {
    el.res.textContent =
      typeof tool.result === 'string' ? tool.result : JSON.stringify(tool.result, null, 2);
  }
});
el.close.addEventListener('click', () => state.select(null));

/* ── level source ───────────────────────────────────────────────────────── */
// Replace this with the microphone / synthesiser tap once arc/voice/ lands.
let levelSource = null;
export function setLevelSource(fn) { levelSource = fn; }

let synth = 0;
setInterval(() => {
  if (levelSource) { state.setLevel(levelSource()); return; }
  // Placeholder: a plausible speech envelope only while ARC is speaking.
  const speaking = state.activity === ACTIVITY.SPEAKING;
  const t = performance.now() / 1000;
  synth = speaking
    ? Math.max(0, 0.45 + 0.35 * Math.sin(t * 11) * Math.sin(t * 3.1) + 0.12 * Math.sin(t * 23))
    : synth * 0.85;
  state.setLevel(Math.min(1, synth));
}, 33);

/* ── streaming ──────────────────────────────────────────────────────────── */
let reply = '';

/** Send one turn and consume the SSE stream. */
export async function send(message) {
  reply = '';
  el.transcript.textContent = '';
  state.setActivity(ACTIVITY.THINKING);

  const res = await fetch('/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message }),
  });
  if (!res.ok || !res.body) {
    state.setActivity(ACTIVITY.IDLE);
    el.status.textContent = `error ${res.status}`;
    return;
  }

  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });

    // SSE frames are separated by a blank line.
    let cut;
    while ((cut = buf.indexOf('\n\n')) !== -1) {
      const frame = buf.slice(0, cut);
      buf = buf.slice(cut + 2);
      handleFrame(frame);
    }
  }
  state.setActivity(ACTIVITY.IDLE);
}

function handleFrame(frame) {
  let event = 'message';
  const data = [];
  for (const line of frame.split('\n')) {
    if (line.startsWith('event:')) event = line.slice(6).trim();
    else if (line.startsWith('data:')) data.push(line.slice(5).trim());
  }
  if (!data.length) return;

  let payload;
  try { payload = JSON.parse(data.join('\n')); } catch { return; }

  switch (event) {
    case 'token':
      if (state.activity !== ACTIVITY.SPEAKING) state.setActivity(ACTIVITY.SPEAKING);
      reply += payload.text;
      el.transcript.textContent = reply;
      break;
    case 'tool_start':
      payload._id = state.startTool(payload.name, payload.category, payload.arguments);
      toolIds.set(payload.call_id ?? payload.name, payload._id);
      break;
    case 'tool_end': {
      const id = toolIds.get(payload.call_id ?? payload.name);
      if (id != null) {
        state.endTool(id, { ok: payload.ok !== false, result: payload.result });
        // Linger so an instant tool call is still visible, then clear.
        setTimeout(() => state.dropTool(id), 4000);
      }
      break;
    }
    case 'state':
      state.setActivity(ACTIVITY[payload.activity] ?? ACTIVITY.IDLE);
      break;
    case 'error':
      el.status.textContent = payload.error || 'error';
      break;
  }
}

const toolIds = new Map();

// Exposed so the REPL, a future voice loop, or the console can drive a turn.
window.arc = { send, state, orb, setLevelSource };
