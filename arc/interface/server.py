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
    "You are ARC, a local-first assistant running on Dhruv's own machine. "
    "You have persistent memory of past conversations. When memories are provided "
    "below, use them naturally — do not announce that you are recalling something, "
    "and never copy the bracketed provenance markers into your reply. "
    "If a memory carries a source URL, cite it when you rely on it. Be concise."
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
        self.requests = 0

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
            elif route.path == "/do":
                self._handle_task(body)
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
                if token.finish_reason is not None:
                    finish = token.finish_reason
        except BrokenPipeError:
            # The browser navigated away mid-generation. Nothing to report to, and the
            # turn is incomplete, so it is deliberately not written to memory.
            _log.info("stream client disconnected")
            return
        except Exception as exc:  # pragma: no cover - defensive
            _log.exception("stream failed")
            self._event("error", {"error": f"{type(exc).__name__}: {exc}"})
            return

        reply = "".join(parts)
        if memory is not None and reply:
            memory.remember_turn("user", text, session_id=session_id)
            memory.remember_turn("assistant", reply, session_id=session_id)

        self._event("done", {"finish_reason": finish, "reply": reply})
        self._event("state", {"activity": "IDLE"})

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

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down", flush=True)
    finally:
        clear_endpoint()
        server.shutdown()
        runtime.close()
    return 0
