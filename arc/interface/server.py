"""Local HTTP server — the warm process.

Two of §7's items turn out to be the same feature. "Model warm-loading" and "a local
HTTP/WebSocket API for a future GUI" both want one thing: a process that holds the model
and memory resident so nothing pays to load them again.

Measured cost of *not* having it: 1.98s to load the model and 0.18s for the embedder,
on every single ``arc do`` or ``arc chat``. Against a task that then runs for ten
seconds, that is a fifth of the wall clock spent re-reading files that were already in
memory a moment ago.

**Binds to 127.0.0.1 only.** ARC has unrestricted access to this machine (§0.3), so an
HTTP endpoint that reaches it must not be reachable from the network. This is not a
configurable setting — a bind address is exactly the kind of thing that gets loosened
"temporarily" and left that way.

Built on ``http.server`` rather than a framework. The surface is five endpoints, and §7
says to treat dependencies as a liability.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from arc import __version__
from arc.config import Config
from arc.errors import ArcError
from arc.log import get_logger
from arc.paths import arc_home

_log = get_logger(__name__)

DEFAULT_PORT = 8787

#: Kept in step with the REPL's prompt in arc/interface/chat.py. Both paths inject
#: memories the same way, so both need the same instructions about how to use them.
DEFAULT_SYSTEM = (
    "You are ARC, a local-first assistant on Dhruv's machine, with memory of past "
    "conversations. Use any memories below naturally; never copy their bracketed "
    "provenance markers into your reply, and cite a source URL when you rely on one. "
    "Be concise."
)

#: Loopback only, and deliberately not configurable. See the module docstring.
BIND_HOST = "127.0.0.1"

#: The web UI ships inside the package so `arc serve` has something to serve without a
#: build step or a node toolchain. Everything in here is hand-written ES modules.
WEBUI_DIR = Path(__file__).parent / "webui"

#: Deliberately a closed allow-list rather than mimetypes.guess_type. The directory
#: only ever holds these four kinds of file, and an unknown extension should 404 rather
#: than be served with a guessed type.
_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
}


#: Where the server records that it is running, so the CLI can find it.
def endpoint_file() -> Any:
    """Path to the file advertising a running server."""
    return arc_home() / "server.json"


class Runtime:
    """The resident model, memory, and tools.

    Everything expensive is loaded once here and reused. Loading is lazy and guarded by
    a lock: two concurrent requests must not both spend two seconds loading the same
    model, and the second must not proceed with a half-initialised one.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.started_at = time.time()
        self._lock = threading.Lock()
        self._model: Any = None
        self._memory: Any = None
        self._voice: Any = None
        self.requests = 0
        #: One generation at a time. The MLX backend is not reentrant, and voice makes
        #: overlapping turns easy to trigger, so this serialises them instead of letting
        #: two streams interleave into the same model.
        self.generation = threading.Lock()
        #: Most recent microphone amplitude, 0..1. Polled by the UI rather than pushed
        #: per sample: it updates ~90 times a second and only the latest value matters.
        self.level = 0.0
        #: Event queues, one per open /events stream.
        self.listeners: list[Any] = []

    @property
    def model(self) -> Any:
        """The chat model, loaded on first use and kept warm afterwards."""
        with self._lock:
            if self._model is None:
                from arc.model import router

                started = time.perf_counter()
                self._model = router.load_model(self.config, "chat")
                _log.info("model warm", extra={"seconds": round(time.perf_counter() - started, 2)})
            return self._model

    @property
    def voice(self) -> Any:
        """The voice session, built on first use.

        Building it does *not* open the microphone — ``VoiceSession`` rests muted, and
        only ``start_listening`` (⌘S) opens anything. Returns None when speech support
        is unavailable so the UI degrades to text instead of erroring.
        """
        with self._lock:
            if self._voice is None and self.config.get("voice.enabled", True):
                from arc import voice as voice_mod

                if str(self.config.get("voice.mode", "local")).lower() == "live":
                    return self._live_session()
                if not voice_mod.available():
                    return None
                from arc.audit import AuditLogger

                audit = (
                    AuditLogger(
                        fsync=bool(self.config.get("audit.fsync", False)),
                        max_field_chars=int(self.config.get("audit.max_field_chars", 4000)),
                    )
                    if self.config.get("audit.enabled", True)
                    else None
                )
                self._voice = voice_mod.build(
                    self.config,
                    audit=audit,
                    on_transcript=self._push_transcript,
                    on_level=self._push_level,
                )
            return self._voice

    def _live_session(self) -> Any:
        """Build the Gemini Live session. Caller holds the lock.

        Not a drop-in for the local path: Live *answers as well as speaks*, so ARC's
        own model, memory and tools take no part in the turn. That is why it is behind
        ``voice.mode`` rather than being the default — it is much faster and it is a
        different assistant.
        """
        from arc.audit import AuditLogger
        from arc.voice.gemini import load_api_key
        from arc.voice.live import LiveSession

        key = load_api_key()
        if not key:
            _log.warning("voice.mode is 'live' but no Gemini API key was found")
            return None

        audit = (
            AuditLogger(
                fsync=bool(self.config.get("audit.fsync", False)),
                max_field_chars=int(self.config.get("audit.max_field_chars", 4000)),
            )
            if self.config.get("audit.enabled", True)
            else None
        )
        section = self.config.section("voice")
        self._voice = _LiveAdapter(
            LiveSession(
                key,
                voice=str(section.get("voice", "Iapetus")),
                silence_ms=int(section.get("silence_ms", 400)),
                system_prompt=DEFAULT_SYSTEM,
                on_transcript=lambda text, final: self._push_transcript_text(text, final),
                on_level=self._push_level,
                on_state=self._push_state,
                audit=audit,
            )
        )
        return self._voice

    def _push_state(self, activity: str) -> None:
        for queue in list(self.listeners):
            queue.append(("state", {"activity": activity}))

    def _push_transcript_text(self, text: str, final: bool) -> None:
        for queue in list(self.listeners):
            queue.append(("transcript", {"text": text, "final": final}))

    def _push_transcript(self, transcript: Any) -> None:
        for queue in list(self.listeners):
            queue.append(("transcript", {"text": transcript.text, "final": transcript.is_final}))

    def _push_level(self, level: float) -> None:
        # Coalesced rather than queued: levels arrive ~90 times a second and only the
        # most recent one matters, so a slow client must not accumulate a backlog.
        self.level = level

    @property
    def memory(self) -> Any:
        """The memory service, or None when memory is disabled."""
        with self._lock:
            if self._memory is None and self.config.get("memory.enabled", True):
                from arc.memory.service import MemoryService

                self._memory = MemoryService.from_config(self.config)
            return self._memory

    def status(self) -> dict[str, Any]:
        """What is loaded and how long it has been up."""
        return {
            "version": __version__,
            "uptime_seconds": round(time.time() - self.started_at, 1),
            "model_loaded": self._model is not None,
            "model": self._model.name if self._model is not None else None,
            "memory_loaded": self._memory is not None,
            "requests_served": self.requests,
        }

    def close(self) -> None:
        """Release the memory database."""
        if self._memory is not None:
            self._memory.close()


class _LiveAdapter:
    """Presents a ``LiveSession`` through the same surface as ``VoiceSession``.

    So ``/voice/status``, ``/voice/toggle`` and ⌘S do not have to know which mode is
    running. ``toggle`` opens the WebSocket on first use and then only gates the
    microphone — reconnecting per turn would put a handshake in front of every reply,
    which is the latency this mode exists to remove.
    """

    def __init__(self, session: Any) -> None:
        self._session: Any = session
        self.recognizer = _Named(f"gemini-live:{session._model.rsplit('/', 1)[-1]}", False)
        self.synthesizer = _Named(
            session.name if hasattr(session, "name") else "gemini-live", False
        )

    @property
    def listening(self) -> bool:
        return self._session.is_open and not self._session.muted

    @property
    def speaking(self) -> bool:
        return bool(self._session.is_speaking)

    def toggle(self) -> bool:
        if not self._session.is_open:
            self._session.open()
        self._session.set_muted(not self._session.muted)
        return self.listening

    def start_listening(self) -> None:
        if not self._session.is_open:
            self._session.open()
        self._session.set_muted(False)

    def stop_listening(self) -> None:
        self._session.set_muted(True)

    def interrupt(self) -> None:
        self._session.set_muted(True)
        self._session.set_muted(False)

    def reset(self) -> None:
        return None

    def feed(self, text: str) -> None:
        """No-op: Gemini speaks its own answer, so nothing is fed to a synthesiser."""
        return None

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self._session.close()


class _Named:
    """Minimal stand-in so status reporting works for both modes."""

    def __init__(self, name: str, on_device: bool) -> None:
        self.name = name
        self.on_device = on_device


class _Handler(BaseHTTPRequestHandler):
    """Routes requests to the resident runtime."""

    runtime: Runtime

    # Silence the default stderr logging; ARC has its own structured log.
    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def _send(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        """Send a non-JSON body (the web UI's static files)."""
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, path: str) -> None:
        """Serve one file from ``WEBUI_DIR``.

        ``resolve()`` then a containment check, rather than trusting the URL: a path
        like ``/ui/../../../etc/passwd`` normalises away only after resolution, and the
        server runs with ARC's own (unrestricted) privileges.
        """
        relative = path[len("/ui/") :] if path.startswith("/ui/") else "index.html"
        if not relative:
            relative = "index.html"

        target = (WEBUI_DIR / relative).resolve()
        root = WEBUI_DIR.resolve()
        if root not in target.parents and target != root:
            self._send({"error": "not found"}, 404)
            return

        content_type = _CONTENT_TYPES.get(target.suffix)
        if content_type is None or not target.is_file():
            self._send({"error": "not found"}, 404)
            return

        self._send_bytes(target.read_bytes(), content_type)

    def _open_stream(self) -> None:
        """Begin an SSE response.

        ``ThreadingHTTPServer`` with ``daemon_threads`` handles each request on its own
        thread, so holding this one open for the length of a generation does not block
        anything else.
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

    def _event(self, event: str, payload: dict[str, Any]) -> None:
        """Write one SSE frame and flush it.

        Flushing matters: the whole point is that tokens appear as they are produced,
        and a buffered stream that arrives all at once is indistinguishable from the
        non-streaming endpoint.
        """
        body = f"event: {event}\ndata: {json.dumps(payload, default=str)}\n\n"
        self.wfile.write(body.encode("utf-8"))
        self.wfile.flush()

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            parsed = json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def do_GET(self) -> None:
        route = urlparse(self.path)
        query = parse_qs(route.query)
        self.runtime.requests += 1

        try:
            if route.path in ("/health", "/status"):
                self._send(self.runtime.status())
            elif route.path == "/" or route.path.startswith("/ui/"):
                self._serve_static(route.path)
            elif route.path == "/voice/status":
                self._send(self._voice_status())
            elif route.path == "/events":
                self._handle_events()
            elif route.path == "/memory/search":
                self._handle_memory_search(query)
            elif route.path == "/tools":
                from arc.tools import registry

                self._send({"tools": registry.describe()})
            else:
                self._send({"error": f"no such endpoint: {route.path}"}, 404)
        except ArcError as exc:
            self._send({"error": str(exc)}, 400)
        except Exception as exc:  # pragma: no cover - defensive
            _log.exception("request failed")
            self._send({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def do_POST(self) -> None:
        route = urlparse(self.path)
        self.runtime.requests += 1

        try:
            body = self._body()
            if route.path == "/chat":
                self._handle_chat(body)
            elif route.path == "/chat/stream":
                self._handle_chat_stream(body)
            elif route.path == "/do/stream":
                self._handle_task_stream(body)
            elif route.path == "/do":
                self._handle_task(body)
            elif route.path == "/voice/toggle":
                self._handle_voice_toggle()
            elif route.path == "/voice/interrupt":
                self._handle_voice_interrupt()
            elif route.path == "/memory/add":
                self._handle_memory_add(body)
            else:
                self._send({"error": f"no such endpoint: {route.path}"}, 404)
        except ArcError as exc:
            self._send({"error": str(exc)}, 400)
        except Exception as exc:  # pragma: no cover - defensive
            _log.exception("request failed")
            self._send({"error": f"{type(exc).__name__}: {exc}"}, 500)

    # ── Endpoints ───────────────────────────────────────────────────────────────

    def _compose(self, text: str, body: dict[str, Any]) -> tuple[list[Any], Any, str]:
        """Build the prompt for one turn.

        Shared by ``/chat`` and ``/chat/stream`` rather than duplicated: the memory
        guidance below is load-bearing, and two copies would drift.
        """
        from arc.model.base import Message

        memory = self.runtime.memory
        session_id = str(body.get("session_id") or "http")
        messages: list[Message] = []

        # Memories are rendered with provenance markers like "[episodic, 2026-07-30]".
        # Without instructions the model treats those as a style to imitate and starts
        # appending them to its own replies — observed answering "pong [episodic,
        # 2026-07-30]". The guidance is not optional decoration.
        sections = [str(body.get("system") or DEFAULT_SYSTEM)]

        if memory is not None:
            hits = memory.recall(text)
            if hits:
                from arc.memory.working import WorkingMemory

                working = WorkingMemory.for_model(self.runtime.model)
                sections.append(working.render_memories(working.pack_memories(hits)))

        messages.append(Message(role="system", content="\n\n".join(s for s in sections if s)))
        messages.append(Message(role="user", content=text))
        return messages, memory, session_id

    def _handle_chat(self, body: dict[str, Any]) -> None:
        """One conversational turn, with memory recall and write-back."""
        text = str(body.get("message", "")).strip()
        if not text:
            self._send({"error": "message is required"}, 400)
            return

        messages, memory, session_id = self._compose(text, body)
        completion = self.runtime.model.generate(
            messages,
            max_tokens=int(body.get("max_tokens", 1024)),
            temperature=float(body.get("temperature", 0.7)),
        )

        if memory is not None:
            memory.remember_turn("user", text, session_id=session_id)
            memory.remember_turn("assistant", completion.text, session_id=session_id)

        self._send(
            {
                "reply": completion.text,
                "finish_reason": completion.finish_reason,
                "usage": {
                    "prompt_tokens": completion.usage.prompt_tokens,
                    "completion_tokens": completion.usage.completion_tokens,
                },
            }
        )

    def _handle_chat_stream(self, body: dict[str, Any]) -> None:
        """One conversational turn, streamed token by token over SSE.

        This exists for the web UI. At ~14 tok/s a forty-token reply takes about three
        seconds, so waiting for the whole thing before showing anything is the
        difference between a conversation and a progress bar.
        """
        text = str(body.get("message", "")).strip()
        if not text:
            self._send({"error": "message is required"}, 400)
            return

        messages, memory, session_id = self._compose(text, body)

        self._open_stream()
        self._event("state", {"activity": "THINKING"})

        # Speak sentence by sentence as generation proceeds, unless the caller opted
        # out. At ~14 tok/s, waiting for the whole reply means seconds of silence.
        session = self.runtime.voice if body.get("speak", True) else None
        if session is not None:
            session.reset()

        # A second turn arriving mid-generation is dropped rather than queued: by the
        # time the first finishes the second is stale, and queueing them is how you end
        # up answering a question nobody is still waiting for.
        if not self.runtime.generation.acquire(blocking=False):
            self._event("error", {"error": "busy"})
            self._event("state", {"activity": "IDLE"})
            return

        parts: list[str] = []
        finish = "stop"
        try:
            for token in self.runtime.model.stream(
                messages,
                max_tokens=int(body.get("max_tokens", 1024)),
                temperature=float(body.get("temperature", 0.7)),
            ):
                if token.text:
                    parts.append(token.text)
                    self._event("token", {"text": token.text})
                    if session is not None:
                        session.feed(token.text)
                if token.finish_reason is not None:
                    finish = token.finish_reason
            if session is not None:
                session.flush()
        except BrokenPipeError:
            # The browser navigated away mid-generation. Nothing to report to, and the
            # turn is incomplete, so it is deliberately not written to memory.
            _log.info("stream client disconnected")
            return
        except Exception as exc:  # pragma: no cover - defensive
            _log.exception("stream failed")
            self._event("error", {"error": f"{type(exc).__name__}: {exc}"})
            return
        finally:
            # Every exit above returns early, so releasing anywhere else would leak the
            # lock and wedge every later turn behind a generation that already ended.
            self.runtime.generation.release()

        reply = "".join(parts)
        if memory is not None and reply:
            memory.remember_turn("user", text, session_id=session_id)
            memory.remember_turn("assistant", reply, session_id=session_id)

        self._event("done", {"finish_reason": finish, "reply": reply})
        self._event("state", {"activity": "IDLE"})

    # ── voice ───────────────────────────────────────────────────────────────────

    def _voice_status(self) -> dict[str, Any]:
        session = self.runtime.voice
        if session is None:
            return {"available": False, "listening": False, "reason": "speech support unavailable"}
        mode = str(self.runtime.config.get("voice.mode", "local")).lower()
        return {
            "available": True,
            "mode": mode,
            "listening": session.listening,
            "speaking": session.speaking,
            "recognizer": session.recognizer.name,
            "on_device": session.recognizer.on_device,
            "synthesizer": session.synthesizer.name,
            # In live mode Gemini answers directly, so the UI must NOT post the
            # transcript to /chat/stream — doing so would run a second, local reply
            # on top of the one already being spoken.
            "answers_itself": mode == "live",
        }

    def _handle_voice_toggle(self) -> None:
        """Open or close the microphone. This is what ⌘S calls."""
        session = self.runtime.voice
        if session is None:
            self._send({"error": "speech support unavailable"}, 400)
            return
        try:
            listening = session.toggle()
            # Tell every open page. Without this the UI only learns the mic state at
            # load, so a change made anywhere else leaves the two disagreeing — and
            # since the UI posts a toggle whenever its own state changes, the
            # disagreement turns into the two of them flipping the mic back and forth.
            for queue in list(self.runtime.listeners):
                queue.append(("voice", {"listening": listening}))
        except ArcError as exc:
            # The on-device refusal lands here, and it must reach the UI verbatim
            # rather than as a generic failure.
            self._send({"error": str(exc)}, 400)
            return
        self._send({"listening": listening})

    def _handle_voice_interrupt(self) -> None:
        session = self.runtime.voice
        if session is not None:
            session.interrupt()
        self._send({"ok": True})

    def _handle_events(self) -> None:
        """Long-lived SSE stream carrying microphone level and transcripts.

        Separate from ``/chat/stream`` because it outlives any one turn: the level has
        to keep flowing while you are speaking, which is *before* there is a turn.
        """
        from collections import deque

        queue: deque[tuple[str, dict[str, Any]]] = deque(maxlen=64)
        self.runtime.listeners.append(queue)

        self._open_stream()
        last_level = -1.0
        last_ping = time.time()
        try:
            while True:
                while queue:
                    event, payload = queue.popleft()
                    self._event(event, payload)

                level = round(self.runtime.level, 3)
                if level != last_level:
                    self._event("level", {"level": level})
                    last_level = level

                # Keepalive. While the microphone is closed the level never changes, so
                # nothing is written and the connection sits idle — which WebKit drops,
                # logged server-side as ConnectionResetError. The page reconnects, but
                # every event in the gap is lost: a four-tool task showed one orb
                # because the other three arrived while nobody was listening. A comment
                # frame is ignored by EventSource and keeps the socket alive.
                now = time.time()
                if now - last_ping >= 10:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    last_ping = now

                # ~30 Hz: fast enough for the orb, slow enough to stay cheap.
                time.sleep(0.033)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the page went away
        finally:
            if queue in self.runtime.listeners:
                self.runtime.listeners.remove(queue)

    def _handle_task(self, body: dict[str, Any]) -> None:
        """Run a multi-step agent task."""
        from arc.agent.loop import Agent
        from arc.tools import registry

        task = str(body.get("task", "")).strip()
        if not task:
            self._send({"error": "task is required"}, 400)
            return

        agent = Agent(
            self.runtime.model,
            registry,
            memory=self.runtime.memory,
            max_steps=int(body.get("max_steps", 12)),
            dry_run=bool(body.get("dry_run", False)),
        )
        self._send(agent.run(task).to_dict())

    def _handle_task_stream(self, body: dict[str, Any]) -> None:
        """Run an agent task, reporting each tool call as it happens.

        ``/do`` returns one JSON blob when everything has finished, which for a
        multi-step task is up to a minute of apparent nothing. This reports each call
        at the moment it starts, so the UI can show the work rather than a spinner.
        """
        from arc.agent.loop import Agent, Step
        from arc.tools import registry

        task = str(body.get("task", "")).strip()
        if not task:
            self._send({"error": "task is required"}, 400)
            return

        if not self.runtime.generation.acquire(blocking=False):
            self._send({"error": "busy"}, 409)
            return

        self._open_stream()
        self._event("state", {"activity": "THINKING"})

        # Sent to this response *and* broadcast to every open /events stream. The UI
        # should show what ARC is doing whoever asked it — a task started from the CLI
        # or another window is still ARC working, and an orb that only appears for the
        # client that happened to make the request is a status display that lies.
        def emit(event: str, payload: dict[str, Any]) -> None:
            self._event(event, payload)
            for queue in list(self.runtime.listeners):
                queue.append((event, payload))

        def started(step: Step) -> None:
            emit(
                "tool_start",
                {
                    "call_id": step.number,
                    "name": step.tool,
                    "category": registry.category_of(step.tool or ""),
                    "arguments": step.arguments,
                },
            )

        def finished(step: Step) -> None:
            observation = step.observation
            emit(
                "tool_end",
                {
                    "call_id": step.number,
                    "name": step.tool,
                    "ok": bool(observation.ok) if observation is not None else False,
                    "result": observation.render() if observation is not None else None,
                },
            )

        try:
            agent = Agent(
                self.runtime.model,
                registry,
                memory=self.runtime.memory,
                max_steps=int(body.get("max_steps", 12)),
                dry_run=bool(body.get("dry_run", False)),
                on_tool_start=started,
                on_step=finished,
            )
            result = agent.run(task)
        except BrokenPipeError:
            _log.info("task stream client disconnected")
            return
        except Exception as exc:  # pragma: no cover - defensive
            _log.exception("task stream failed")
            self._event("error", {"error": f"{type(exc).__name__}: {exc}"})
            return
        finally:
            self.runtime.generation.release()

        session = self.runtime.voice if body.get("speak", True) else None
        if session is not None and result.answer:
            session.reset()
            session.feed(result.answer)
            session.flush()

        self._event("done", {"reply": result.answer, "tools_used": result.tools_used})
        self._event("state", {"activity": "IDLE"})

    def _handle_memory_search(self, query: dict[str, list[str]]) -> None:
        """Hybrid search over memory."""
        memory = self.runtime.memory
        if memory is None:
            self._send({"error": "memory is disabled"}, 400)
            return

        text = (query.get("q") or [""])[0]
        limit = int((query.get("limit") or ["10"])[0])
        hits = memory.retriever.search(text, limit=limit) if text else []
        self._send({"query": text, "results": [hit.to_dict() for hit in hits]})

    def _handle_memory_add(self, body: dict[str, Any]) -> None:
        """Store a fact directly."""
        memory = self.runtime.memory
        if memory is None:
            self._send({"error": "memory is disabled"}, 400)
            return

        text = str(body.get("text", "")).strip()
        if not text:
            self._send({"error": "text is required"}, 400)
            return
        self._send({"id": memory.semantic.add_fact(text, source="api")})


def write_endpoint(port: int) -> None:
    """Advertise a running server so the CLI can find and reuse it."""
    target = endpoint_file()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({"host": BIND_HOST, "port": port, "pid": _pid()}), encoding="utf-8"
    )


def clear_endpoint() -> None:
    """Remove the advertisement. Safe when it is already gone."""
    endpoint_file().unlink(missing_ok=True)


def _pid() -> int:
    import os

    return os.getpid()


def running_endpoint() -> tuple[str, int] | None:
    """Return a live server's address, or None.

    Checks that the advertised process still exists, because a crashed server leaves
    its endpoint file behind and the CLI would otherwise hang trying to reach it.
    """
    import os

    target = endpoint_file()
    if not target.is_file():
        return None
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        os.kill(int(data["pid"]), 0)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        clear_endpoint()
        return None
    return (str(data["host"]), int(data["port"]))


def serve(config: Config, *, port: int = DEFAULT_PORT, preload: bool = True) -> int:
    """Run the server until interrupted."""
    runtime = Runtime(config)

    handler = type("BoundHandler", (_Handler,), {"runtime": runtime})
    server = ThreadingHTTPServer((BIND_HOST, port), handler)
    # Daemon threads so Ctrl-C is not held up by an in-flight request.
    server.daemon_threads = True

    if preload:
        # Pay the cost now rather than on the first request, which is the entire point
        # of a warm process.
        started = time.perf_counter()
        _ = runtime.model
        _ = runtime.memory
        print(f"warm in {time.perf_counter() - started:.1f}s", flush=True)

    write_endpoint(port)
    print(f"ARC listening on http://{BIND_HOST}:{port} (loopback only)", flush=True)

    # The HTTP server runs on a background thread so the *main* thread is free to pump
    # the macOS run loop. Speech results are delivered on the main queue, so without
    # this the microphone captures audio and no transcript ever arrives — levels move,
    # nothing is transcribed, and nothing errors. Everything else works either way, so
    # the pump only runs when speech support is actually present.
    thread = threading.Thread(target=server.serve_forever, name="arc-http", daemon=True)
    thread.start()

    stop = threading.Event()
    try:
        from arc import voice as voice_mod

        if config.get("voice.enabled", True) and voice_mod.available():
            voice_mod.pump(stop)
        else:
            while not stop.wait(0.5):
                pass
    except KeyboardInterrupt:
        print("\nshutting down", flush=True)
    finally:
        stop.set()
        clear_endpoint()
        server.shutdown()
        runtime.close()
    return 0
