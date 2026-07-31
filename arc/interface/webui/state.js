// Mute/listen state machine + tool tracking, with a tiny event emitter.
//
// Ported from ~/projects/Jarvis/webui/js/state/AppState.js. Two things carried over
// deliberately:
//
//   * It **rests muted**. ARC has unrestricted access to this machine (BRIEF §0.3),
//     so the microphone is never hot at startup. `listening` stays derived rather
//     than stored, which is what lets a temporary (push-to-talk) unmute and a
//     permanent (⌘S) unmute coexist without the two getting out of sync.
//   * `activity` is kept separate from `listening`. JARVIS's hologram splits these
//     because they answer different questions — is *your* mic open, versus is *the
//     engine* busy — and the orb maps them to different visual channels.

export const ACTIVITY = {
  IDLE: 'IDLE',
  THINKING: 'THINKING',
  SPEAKING: 'SPEAKING',
};

let nextToolId = 1;

export class AppState {
  constructor() {
    this.muted = true;
    this.permanentUnmute = false;
    this.tempUnmute = false;
    this.activity = ACTIVITY.IDLE;

    // Amplitude, 0..1, smoothed by the orb rather than here — this is the raw
    // drive signal from either your microphone or ARC's own synthesised speech.
    this.level = 0;

    // Live tool calls. Empty by default: the resting view is one orb and nothing
    // else, and a side orb only exists while there is something to look at.
    this.tools = new Map();
    this.selectedTool = null;

    this._listeners = {};
  }

  on(evt, cb) { (this._listeners[evt] ||= []).push(cb); return this; }
  emit(evt, ...a) { (this._listeners[evt] || []).forEach((cb) => cb(...a)); }

  get listening() { return this.permanentUnmute || this.tempUnmute; }

  _sync() { this.muted = !this.listening; this.emit('change', this); }

  startTempUnmute() { if (!this.tempUnmute) { this.tempUnmute = true; this._sync(); } }
  stopTempUnmute() { if (this.tempUnmute) { this.tempUnmute = false; this._sync(); } }

  setPermanentUnmute(v) {
    this.permanentUnmute = v;
    this.emit(v ? 'unmute' : 'mute');
    this._sync();
  }

  // ⌘S toggle: flip the mic between muted and listening.
  toggleTalk() { this.setPermanentUnmute(!this.permanentUnmute); }

  setActivity(a) {
    if (this.activity === a) return;
    this.activity = a;
    this.emit('change', this);
  }

  setLevel(v) { this.level = v; }

  // ── tools ────────────────────────────────────────────────────────────────
  // A tool call spawns a side orb; it lives until `endTool` and then lingers
  // briefly so a call that returns instantly is still visible.

  startTool(name, category, args) {
    const id = nextToolId++;
    this.tools.set(id, {
      id, name,
      category: category || 'general',
      args: args || {},
      state: 'running',
      result: null,
      started: performance.now(),
      ended: null,
    });
    this.emit('tools', this);
    return id;
  }

  endTool(id, { ok = true, result = null } = {}) {
    const t = this.tools.get(id);
    if (!t) return;
    t.state = ok ? 'ok' : 'error';
    t.result = result;
    t.ended = performance.now();
    this.emit('tools', this);
    if (this.selectedTool === id) this.emit('inspect', t);
  }

  dropTool(id) {
    if (!this.tools.delete(id)) return;
    if (this.selectedTool === id) this.select(null);
    this.emit('tools', this);
  }

  select(id) {
    this.selectedTool = id;
    this.emit('inspect', id == null ? null : this.tools.get(id));
  }

  activeTools() { return [...this.tools.values()]; }
}
