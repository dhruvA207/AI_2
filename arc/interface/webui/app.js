// Wires the orb, the ⌘S binding and the SSE streams together.
//
// The server is the authority on microphone state, not this page. Both can change it
// — ⌘S here, `arc voice` or another window there — so the page pushes its own changes
// and adopts the server's, guarded by `lastSentListening` so an adopted change is
// never echoed back. Without that guard the two flip the mic open and closed forever.

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

/* ── level ──────────────────────────────────────────────────────────────── */
// While the mic is open the level is real: Python taps the PCM buffers and pushes
// RMS over /events. While ARC is *speaking* there is no such tap — the Gemini audio is
// played by afplay straight to the output device — so the orb runs on a synthesised
// envelope for that half. It is decoration there, not data, and is deliberately kept
// separate from micLevel so the two can never be confused.
let micLevel = 0;
let synth = 0;

setInterval(() => {
  const speaking = state.activity === ACTIVITY.SPEAKING;
  if (state.listening && !speaking) {
    state.setLevel(micLevel);
    synth = 0;
    return;
  }
  const t = performance.now() / 1000;
  synth = speaking
    ? Math.max(0, 0.45 + 0.35 * Math.sin(t * 11) * Math.sin(t * 3.1) + 0.12 * Math.sin(t * 23))
    : synth * 0.85;
  state.setLevel(Math.min(1, synth));
}, 33);

let lastSentListening = false;
let answersItself = false;

/* ── the long-lived event stream ────────────────────────────────────────── */
// Carries mic level and transcripts. Separate from /chat/stream because it outlives
// any one turn — the level flows while you are still speaking, before a turn exists.
function openEvents() {
  const es = new EventSource('/events');
  es.addEventListener('level', (e) => { micLevel = JSON.parse(e.data).level; });
  // The server is the authority on whether the mic is open. Adopt its state without
  // posting a toggle back, or the two flip the microphone back and forth forever.
  es.addEventListener('voice', (e) => {
    const { listening } = JSON.parse(e.data);
    if (listening === state.listening) return;
    lastSentListening = listening;
    state.setPermanentUnmute(listening);
  });
  // Tool activity arrives here too, because a task can be started from anywhere —
  // the CLI, another window, a voice turn. The orbs track ARC, not this page.
  for (const kind of ['tool_start', 'tool_end', 'state']) {
    es.addEventListener(kind, (e) => handleEvent(kind, JSON.parse(e.data)));
  }
  es.addEventListener('transcript', (e) => {
    const { text, final } = JSON.parse(e.data);
    // Partials are replacements, not additions — appending them makes the text stutter.
    el.transcript.innerHTML = '';
    const span = document.createElement('span');
    if (!final) span.className = 'partial';
    span.textContent = text;
    el.transcript.appendChild(span);
    // In live mode Gemini has already answered out loud; posting the transcript
    // would run a second reply locally and the two would talk over each other.
    if (final && text.trim() && !answersItself) send(text.trim());
  });
  es.onerror = () => { es.close(); setTimeout(openEvents, 1500); };
}
openEvents();

/* ── microphone ─────────────────────────────────────────────────────────── */
// ⌘S flips UI state instantly so the indicator never lags the keypress; the server is
// told afterwards and corrects us if it disagrees (e.g. the on-device refusal).
state.on('change', async (s) => {
  if (s.listening === lastSentListening) return;
  lastSentListening = s.listening;
  try {
    const r = await fetch('/voice/toggle', { method: 'POST' });
    const d = await r.json();
    if (d.error) {
      el.status.textContent = d.error;
      lastSentListening = false;
      state.setPermanentUnmute(false);
    } else if (d.listening !== s.listening) {
      lastSentListening = d.listening;
      state.setPermanentUnmute(d.listening);
    }
  } catch {
    el.status.textContent = 'voice unavailable';
  }
});
// Adopt the server's actual microphone state on load. AppState rests muted, but the
// server outlives the page: reloading while the mic was open otherwise leaves the
// indicator reading MUTED over a live microphone, which is the one thing this
// indicator exists to never do.
(async function syncVoice() {
  try {
    const st = await (await fetch('/voice/status')).json();
    if (!st.available) {
      el.status.textContent = 'voice unavailable';
      return;
    }
    answersItself = !!st.answers_itself;
    lastSentListening = st.listening;          // set first, so this does not toggle
    state.setPermanentUnmute(st.listening);
  } catch {
    /* server not up yet; the indicator stays muted, which is the safe default */
  }
})();

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
  handleEvent(event, payload);
}

/** One handler for both streams: /chat/stream frames and /events. */
function handleEvent(event, payload) {
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
        // Deliberately NOT dropped here. The agent runs tools one at a time, so
        // clearing each a few seconds after it finishes means you only ever see one
        // orb — the constellation of what ARC used never exists. They stay until the
        // task finishes, and are cleared together below.
      }
      break;
    }
    case 'state':
      state.setActivity(ACTIVITY[payload.activity] ?? ACTIVITY.IDLE);
      break;
    case 'done':
      // The task is over: let the finished constellation sit for a moment so it can
      // still be clicked, then clear it.
      for (const tool of state.activeTools()) {
        const id = tool.id;
        setTimeout(() => state.dropTool(id), 6000);
      }
      break;
    case 'error':
      el.status.textContent = payload.error || 'error';
      break;
  }
}

const toolIds = new Map();

// Exposed so a turn can be driven from the console or a future REPL bridge.
window.arc = { send, state, orb };
