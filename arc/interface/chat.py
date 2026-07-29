"""The chat REPL.

Talks to the local model, with persistent memory behind it. Conversations survive
restarts: every turn is written to the memory store, and each new question pulls back
whatever past memories are relevant before the model sees it.

Recalled memories are injected into the *system* message rather than replayed as fake
prior turns. A fabricated exchange would be indistinguishable from something the user
actually said, and the model would start attributing its own recollections to them.

Streaming is the default because a 4B model on a laptop produces its first token in
well under a second but a full answer in several. Waiting for the whole completion
makes a fast model feel slow.
"""

from __future__ import annotations

import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from arc.audit import AuditLogger
from arc.config import Config
from arc.errors import ArcError, ModelError
from arc.log import get_logger
from arc.memory.service import MemoryService
from arc.memory.working import WorkingMemory
from arc.model.base import LanguageModel, Message

_log = get_logger(__name__)

_DEFAULT_SYSTEM = (
    "You are ARC, a local-first assistant running on Dhruv's own machine. "
    "You have persistent memory of past conversations. When memories are provided "
    "below, use them naturally — do not announce that you are recalling something. "
    "If a memory carries a source URL, cite it when you rely on it. Be concise."
)

_BANNER = """\
ARC chat — {model}
context {context} tokens · {backend}

/help for commands, /exit to quit, Ctrl-C to interrupt generation
"""

_HELP = """\
  /help              show this
  /clear             forget the conversation so far (this session only)
  /system <text>     set the system prompt (clears history)
  /tokens            show context use for the current conversation
  /model             show the loaded model and its capabilities
  /memory            show what is in long-term memory
  /recall <query>    search memory directly
  /why               explain which memories informed the last reply
  /exit, /quit       leave
"""


@dataclass
class Session:
    """One REPL conversation.

    The transcript in ``history`` is this process's working set. Long-term persistence
    is the memory service's job: turns are written there as they happen, and relevant
    past memories are pulled back in per turn. Closing the REPL no longer discards the
    conversation.
    """

    model: LanguageModel
    system_prompt: str | None = None
    history: list[Message] = field(default_factory=list)
    memory: MemoryService | None = None
    session_id: str = ""
    #: Memories injected for the current turn, kept so `/why` can explain them.
    last_recall: list[Any] = field(default_factory=list)

    def messages(self) -> list[Message]:
        """Return the full message list, system prompt first."""
        if self.system_prompt:
            return [Message(role="system", content=self.system_prompt), *self.history]
        return list(self.history)

    def messages_with_memory(self, query: str) -> list[Message]:
        """Assemble the prompt, injecting recalled memories and known preferences.

        Memory goes into the *system* message rather than as a fake prior turn. A
        pretend exchange would be indistinguishable from something the user actually
        said, and the model would start attributing its own recollections to them.
        """
        if self.memory is None:
            self.history.append(Message(role="user", content=query))
            return self.messages()

        hits = self.memory.recall(query)
        procedures = self.memory.applicable_procedures(query)
        self.last_recall = list(hits)

        working = WorkingMemory.for_model(self.model)
        packed = working.pack_memories(hits)

        sections = [self.system_prompt or _DEFAULT_SYSTEM]
        if procedures:
            sections.append(working.render_procedures(procedures))
        if packed:
            sections.append(working.render_memories(packed))

        self.history.append(Message(role="user", content=query))
        return [
            Message(role="system", content="\n\n".join(s for s in sections if s)),
            *self.history,
        ]

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
    elif command == "/memory":
        if session.memory is None:
            print("memory is disabled for this session")
        else:
            stats = session.memory.stats()
            print(f"{stats['live_memories']} memories ({stats['by_layer']})")
            print(f"  embedder  {stats['embedder']}")
            print(f"  entities  {stats['entities']}, relations {stats['relations']}")
            print(f"  file      {stats['path']} ({stats['size_mb']} MB)")
    elif command == "/why":
        # "Why did it say that?" is the first question when memory misbehaves, and
        # answering it needs the retrieval provenance, not just the result.
        if not session.last_recall:
            print("no memories were used in the last turn")
        else:
            print(f"{len(session.last_recall)} memories informed the last reply:")
            for hit in session.last_recall:
                strategies = ", ".join(hit.sources)
                print(f"  [{hit.score:.4f} via {strategies}] {hit.record.content[:70]}")
    elif command == "/recall":
        if session.memory is None:
            print("memory is disabled for this session")
        elif not argument:
            print("usage: /recall <query>")
        else:
            for hit in session.memory.recall(argument, limit=8):
                print(f"  [{hit.record.layer}] {hit.record.content[:70]}")
    else:
        print(f"unknown command {command!r} — /help for the list")

    return True


def _stream_reply(
    session: Session,
    config: Config,
    audit: AuditLogger | None,
    prompt: list[Message] | None = None,
) -> str:
    """Stream one assistant turn to stdout, returning the full text.

    Ctrl-C during generation aborts the *turn*, not the REPL. Whatever was produced
    before the interrupt is kept, because a partial answer is usually still worth
    reading and discarding it silently would be worse.
    """
    max_tokens = int(config.get("models.generation.max_tokens", 2048))
    temperature = float(config.get("models.generation.temperature", 0.7))
    messages = prompt if prompt is not None else session.messages()

    chunks: list[str] = []
    started = time.perf_counter()
    interrupted = False

    try:
        for token in session.model.stream(messages, max_tokens=max_tokens, temperature=temperature):
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
                "chars_in": sum(len(m.content) for m in messages),
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
    memory: MemoryService | None = None,
) -> int:
    """Run the REPL until the user exits. Returns a process exit code."""
    session_id = uuid.uuid4().hex[:12]
    session = Session(
        model=model, system_prompt=system_prompt, memory=memory, session_id=session_id
    )

    if memory is not None:
        memory.store.start_session(session_id, model=model.name)

    print(_BANNER.format(model=model.name, context=model.context_length, backend=backend))
    if memory is not None:
        stats = memory.stats()
        print(f"memory: {stats['live_memories']} memories, {stats['entities']} entities\n")

    try:
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

            # Appends the user turn to history and builds the prompt, injecting
            # recalled memories into the system message.
            prompt = session.messages_with_memory(line)

            print("\narc > ", end="", flush=True)
            try:
                reply = _stream_reply(session, config, audit, prompt)
            except ArcError:
                # The turn failed; drop it so the transcript does not carry a user
                # message with no answer into the next request.
                session.history.pop()
                continue

            if reply:
                session.history.append(Message(role="assistant", content=reply))
                if memory is not None:
                    # Persisted only after a successful reply, so a failed turn does
                    # not leave a dangling question in long-term memory.
                    memory.remember_turn("user", line, session_id=session_id)
                    memory.remember_turn("assistant", reply, session_id=session_id)
            else:
                session.history.pop()
            print()
    finally:
        if memory is not None:
            memory.store.end_session(session_id)
