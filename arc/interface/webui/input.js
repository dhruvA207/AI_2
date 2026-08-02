// Keyboard handling:
//   • ⌘S (Cmd+S) → TOGGLE the mic (press = unmute/listen, press again = mute)
//   • Esc        → close the tool inspector
//
// Ported from ~/projects/Jarvis/webui/js/state/InputManager.js. The three details
// below are the entire reason that file works, and each one is a bug if dropped:
//
//   1. ⌘S is a TOGGLE, not hold-to-talk. macOS treats ⌘-modified keys as one-shot
//      menu commands inside a webview, so a "held" ⌘S fires press+instant-release
//      and the mic would only flicker open. A toggle stays open reliably.
//   2. Match on `e.code === 'KeyS'` (the physical key), not `e.key` — on macOS
//      Option+S remaps `e.key` to 'ß', so key-based matching silently misses.
//   3. preventDefault() on ⌘S blocks the browser's Save dialog, and the `e.repeat`
//      guard stops key-repeat from flapping the mic while the key is held down.

export class InputManager {
  constructor(state) {
    this.state = state;
    addEventListener('keydown', (e) => this._down(e));
  }

  _down(e) {
    // Escape is barge-in. It has to be a key rather than simply talking over ARC:
    // with no echo cancellation the microphone is deafened while the speakers play,
    // so speech cannot signal an interruption. Also closes the tool inspector.
    if (e.key === 'Escape') {
      fetch('/voice/interrupt', { method: 'POST' }).catch(() => {});
      this.state.select(null);
      return;
    }
    if (e.code !== 'KeyS') return;

    if (e.metaKey || e.ctrlKey) {
      e.preventDefault();      // block the browser Save dialog
      if (e.repeat) return;    // ignore auto-repeat while held
      this.state.toggleTalk();
    }
  }
}
