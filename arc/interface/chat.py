"""The chat REPL.

Phase 2's deliverable: talking to a local model. **Stateless by design** — history lives
only in this process, and closing the REPL discards it. Phase 3 replaces that with real
memory, and doing it properly there is better than half-doing it here with a JSON file
we would then have to migrate.

Streaming is the default because a 4B model on a laptop produces its first token in
well under a second but a full answer in several. Waiting for the whole completion
makes a fast model feel slow.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field

from arc.audit import AuditLogger
from arc.config import Config
from arc.errors import ArcError, ModelError
from arc.log import get_logger
from arc.model.base import LanguageModel, Message

_log = get_logger(__name__)

_BANNER = """\
ARC chat — {model}
context {context} tokens · {backend}

/help for commands, /exit to quit, Ctrl-C to interrupt generation
"""

_HELP = """\
  /help              show this
  /clear             forget the conversation so far
  /system <text>     set the system prompt (clears history)
  /tokens            show context use for the current conversation
  /model             show the loaded model and its capabilities
  /exit, /quit       leave
"""


@dataclass
class Session:
    """One REPL conversation.

    Holds the transcript and the system prompt. Deliberately not persisted: Phase 2 is
    stateless and pretending otherwise would invite building on a foundation Phase 3
    replaces.
    """

    model: LanguageModel
    system_prompt: str | None = None
    history: list[Message] = field(default_factory=list)

    def messages(self) -> list[Message]:
        """Return the full message list, system prompt first."""
        if self.system_prompt:
            return [Message(role="system", content=self.system_prompt), *self.history]
        return list(self.history)

    def token_use(self) -> tuple[int, int]:
        """Return (tokens used, context limit)."""
        return self.model.count_message_tokens(self.messages()), self.model.context_length

    def clear(self) -> None:
        """Drop the transcript, keeping the system prompt."""
        self.history.clear()


def _print_help() -> None:
    print(_HELP)


def _handle_command(line: str, session: Session) -> bool:
    """Run a slash command. Returns False when the REPL should exit."""
    parts = line.split(maxsplit=1)
    command = parts[0].lower()
    argument = parts[1] if len(parts) > 1 else ""

    if command in {"/exit", "/quit"}:
        return False

    if command == "/help":
        _print_help()
    elif command == "/clear":
        session.clear()
        print("conversation cleared")
    elif command == "/system":
        if not argument:
            print(f"system prompt: {session.system_prompt or '(none)'}")
        else:
            session.system_prompt = argument
            session.clear()
            print("system prompt set; conversation cleared")
    elif command == "/tokens":
        used, limit = session.token_use()
        pct = (used / limit * 100) if limit else 0
        print(f"{used} / {limit} tokens ({pct:.1f}% of context)")
    elif command == "/model":
        caps = session.model.capabilities
        print(f"{session.model.name}")
        print(f"  context      {session.model.context_length}")
        print(f"  tool calling {'native' if caps.native_tool_calling else 'prompted fallback'}")
        print(f"  json mode    {caps.json_mode}")
        print(f"  thinking     {caps.thinking}")
        print(f"  vision       {caps.vision}")
    else:
        print(f"unknown command {command!r} — /help for the list")

    return True


def _stream_reply(session: Session, config: Config, audit: AuditLogger | None) -> str:
    """Stream one assistant turn to stdout, returning the full text.

    Ctrl-C during generation aborts the *turn*, not the REPL. Whatever was produced
    before the interrupt is kept, because a partial answer is usually still worth
    reading and discarding it silently would be worse.
    """
    max_tokens = int(config.get("models.generation.max_tokens", 2048))
    temperature = float(config.get("models.generation.temperature", 0.7))

    chunks: list[str] = []
    started = time.perf_counter()
    interrupted = False

    try:
        for token in session.model.stream(
            session.messages(), max_tokens=max_tokens, temperature=temperature
        ):
            chunks.append(token.text)
            sys.stdout.write(token.text)
            sys.stdout.flush()
    except KeyboardInterrupt:
        interrupted = True
        print("\n[interrupted]")
    except ModelError as exc:
        print(f"\n[generation failed: {exc}]")
        raise

    elapsed = time.perf_counter() - started
    text = "".join(chunks)

    if not interrupted:
        print()

    if audit is not None:
        count = session.model.count_tokens(text) if text else 0
        audit.record(
            "chat.turn",
            status="ok",
            tool="chat",
            args={
                "model": session.model.name,
                "chars_in": sum(len(m.content) for m in session.messages()),
            },
            result={
                "chars_out": len(text),
                "tokens_out": count,
                "seconds": round(elapsed, 2),
                "tokens_per_second": round(count / elapsed, 1) if elapsed > 0 and count else None,
                "interrupted": interrupted,
            },
        )

    return text


def run(
    model: LanguageModel,
    config: Config,
    *,
    backend: str = "unknown",
    system_prompt: str | None = None,
    audit: AuditLogger | None = None,
) -> int:
    """Run the REPL until the user exits. Returns a process exit code."""
    session = Session(model=model, system_prompt=system_prompt)

    print(_BANNER.format(model=model.name, context=model.context_length, backend=backend))

    while True:
        try:
            line = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            # Ctrl-D or Ctrl-C at the prompt (rather than mid-generation) exits.
            print()
            return 0

        if not line:
            continue

        if line.startswith("/"):
            if not _handle_command(line, session):
                return 0
            continue

        session.history.append(Message(role="user", content=line))

        print("\narc > ", end="", flush=True)
        try:
            reply = _stream_reply(session, config, audit)
        except ArcError:
            # The turn failed; drop it so the transcript does not carry a user message
            # with no answer into the next request.
            session.history.pop()
            continue

        if reply:
            session.history.append(Message(role="assistant", content=reply))
        else:
            session.history.pop()
        print()
