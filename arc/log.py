"""Structured logging.

The brief asks for JSONL rather than print statements. Two handlers, deliberately
different: a machine-readable JSONL file for anything we will later grep or analyse,
and a terse human-readable console stream, because a CLI that prints JSON at you is
unpleasant to actually use.

Named ``log`` rather than ``logging`` to avoid any ambiguity with the stdlib module
of that name when reading imports.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from arc.paths import log_dir

#: Fields the stdlib puts on every LogRecord. Anything outside this set was attached
#: by a caller via `extra=` and should be preserved in the JSON output.
_STANDARD_FIELDS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)

_CONSOLE_FORMAT = "%(levelname)-7s %(name)s: %(message)s"


class JsonlFormatter(logging.Formatter):
    """Render a log record as one JSON object per line.

    Caller-supplied ``extra=`` fields are merged in at the top level so that
    structured context survives into the file instead of being flattened into the
    message string.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key not in _STANDARD_FIELDS and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        # default=str so an unserializable value degrades to its repr rather than
        # throwing inside the logger, which would lose the very message being logged.
        return json.dumps(payload, default=str)


def setup(
    *,
    level: str = "info",
    console: bool = True,
    console_level: str = "warning",
    to_file: bool = True,
) -> Path | None:
    """Configure root logging. Returns the JSONL path, or None if file output is off.

    Idempotent: existing handlers are cleared first, so calling this twice in one
    process (tests, or a REPL) does not produce duplicated output.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    root.setLevel(_parse_level(level))

    path: Path | None = None
    if to_file:
        directory = log_dir()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"arc-{datetime.now(UTC):%Y-%m-%d}.jsonl"
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(JsonlFormatter())
        file_handler.setLevel(_parse_level(level))
        root.addHandler(file_handler)

    if console:
        # stderr, so stdout stays clean for command output that may be piped.
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(logging.Formatter(_CONSOLE_FORMAT))
        stream.setLevel(_parse_level(console_level))
        root.addHandler(stream)

    return path


def _parse_level(name: str) -> int:
    """Translate a config level name to a logging constant, defaulting to INFO."""
    resolved = logging.getLevelNamesMapping().get(name.strip().upper())
    return resolved if resolved is not None else logging.INFO


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger.

    Thin wrapper so modules do not import ``logging`` directly and we keep one place
    to change if the backend ever changes.
    """
    return logging.getLogger(name)
