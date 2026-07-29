"""Filesystem tools.

**Unrestricted by design** (§0.3). These read and write anywhere the user account can,
with no permission prompts and no deny-list. That is deliberate and not up for
negotiation in this codebase; the safeguards the brief asked for instead are the audit
log, the kill switch, and ``--dry-run``.

``config/policy.yaml`` has the plumbing to narrow this later, defaulting to permissive.

Every path goes through ``pathlib`` and is expanded and resolved, so ``~`` works and
the audit log records the real target rather than whatever relative fragment the model
happened to emit.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from arc.errors import ToolError
from arc.log import get_logger
from arc.tools.registry import tool

_log = get_logger(__name__)

#: Reading a multi-gigabyte file into the context window helps nobody. Truncation is
#: reported in the output so the model knows it saw a fragment.
_MAX_READ_CHARS = 100_000

#: Cap on how many paths a listing or search returns, for the same reason.
_MAX_ENTRIES = 500


def resolve(path: str) -> Path:
    """Expand and resolve a user-supplied path.

    ``strict=False`` because writes legitimately target files that do not exist yet.
    """
    return Path(path).expanduser().resolve(strict=False)


@tool(category="filesystem")
def read_file(path: str, max_chars: int = _MAX_READ_CHARS) -> str:
    """Read a text file and return its contents.

    Args:
        path: File to read. ``~`` is expanded.
        max_chars: Stop after this many characters.
    """
    target = resolve(path)
    if not target.exists():
        raise ToolError(f"no such file: {target}")
    if target.is_dir():
        raise ToolError(f"{target} is a directory; use list_directory")

    try:
        # errors="replace" so a stray non-UTF-8 byte in an otherwise readable file
        # degrades one character instead of failing the whole read.
        content = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ToolError(f"could not read {target}: {exc}") from exc

    if len(content) > max_chars:
        return content[:max_chars] + f"\n\n[truncated at {max_chars} of {len(content)} chars]"
    return content


@tool(category="filesystem", mutating=True)
def write_file(path: str, content: str, append: bool = False) -> str:
    """Write text to a file, creating parent directories as needed.

    Args:
        path: File to write.
        content: Text to write.
        append: Append instead of overwriting.
    """
    target = resolve(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a" if append else "w", encoding="utf-8") as handle:
            handle.write(content)
    except OSError as exc:
        raise ToolError(f"could not write {target}: {exc}") from exc

    verb = "appended to" if append else "wrote"
    return f"{verb} {target} ({len(content)} chars)"


@tool(category="filesystem")
def list_directory(path: str = ".", pattern: str = "*", recursive: bool = False) -> str:
    """List directory contents.

    Args:
        path: Directory to list.
        pattern: Glob pattern to filter by.
        recursive: Descend into subdirectories.
    """
    target = resolve(path)
    if not target.is_dir():
        raise ToolError(f"not a directory: {target}")

    try:
        matches = target.rglob(pattern) if recursive else target.glob(pattern)
        entries = sorted(matches)[:_MAX_ENTRIES]
    except OSError as exc:
        raise ToolError(f"could not list {target}: {exc}") from exc

    if not entries:
        return f"{target} is empty (or nothing matched {pattern!r})"

    lines = []
    for entry in entries:
        try:
            if entry.is_dir():
                lines.append(f"{entry.name}/")
            else:
                lines.append(f"{entry.name}  ({entry.stat().st_size} bytes)")
        except OSError:
            # A broken symlink or a file removed mid-listing must not abort the whole
            # listing.
            lines.append(f"{entry.name}  (unreadable)")

    return f"{target} ({len(entries)} entries):\n" + "\n".join(lines)


@tool(category="filesystem")
def search_files(path: str, query: str, pattern: str = "*", max_results: int = 50) -> str:
    """Search file contents for a string, returning matching lines.

    Args:
        path: Directory to search.
        query: Text to look for. Case-insensitive.
        pattern: Which files to search, as a glob.
        max_results: Stop after this many matching lines.
    """
    root = resolve(path)
    if not root.is_dir():
        raise ToolError(f"not a directory: {root}")

    needle = query.lower()
    results: list[str] = []

    for candidate in root.rglob(pattern):
        if len(results) >= max_results:
            break
        if not candidate.is_file():
            continue
        try:
            text = candidate.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue  # Unreadable file; keep searching the rest.

        for number, line in enumerate(text.splitlines(), start=1):
            if needle in line.lower():
                results.append(f"{candidate}:{number}: {line.strip()[:200]}")
                if len(results) >= max_results:
                    break

    if not results:
        return f"no matches for {query!r} under {root}"
    return "\n".join(results)


@tool(category="filesystem", mutating=True)
def move_path(source: str, destination: str) -> str:
    """Move or rename a file or directory.

    Args:
        source: Path to move.
        destination: Where to move it.
    """
    src, dst = resolve(source), resolve(destination)
    if not src.exists():
        raise ToolError(f"no such path: {src}")

    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    except OSError as exc:
        raise ToolError(f"could not move {src} to {dst}: {exc}") from exc
    return f"moved {src} to {dst}"


@tool(category="filesystem", mutating=True)
def copy_path(source: str, destination: str) -> str:
    """Copy a file or directory.

    Args:
        source: Path to copy.
        destination: Where to copy it.
    """
    src, dst = resolve(source), resolve(destination)
    if not src.exists():
        raise ToolError(f"no such path: {src}")

    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    except OSError as exc:
        raise ToolError(f"could not copy {src} to {dst}: {exc}") from exc
    return f"copied {src} to {dst}"


@tool(category="filesystem", mutating=True)
def delete_path(path: str, recursive: bool = False) -> str:
    """Delete a file or directory permanently.

    Args:
        path: Path to delete.
        recursive: Required to delete a non-empty directory.
    """
    target = resolve(path)
    if not target.exists():
        raise ToolError(f"no such path: {target}")

    try:
        if target.is_dir():
            if not recursive and any(target.iterdir()):
                # Not a permission gate — the agent can pass recursive=True freely.
                # It exists so a single wrong argument does not take a whole tree with
                # it, which is the difference between one mistake and a bad afternoon.
                raise ToolError(
                    f"{target} is not empty; pass recursive=true to delete it and its contents"
                )
            shutil.rmtree(target) if recursive else target.rmdir()
        else:
            target.unlink()
    except OSError as exc:
        raise ToolError(f"could not delete {target}: {exc}") from exc
    return f"deleted {target}"


@tool(category="filesystem")
def path_info(path: str) -> str:
    """Report whether a path exists and what it is.

    Args:
        path: Path to inspect.
    """
    target = resolve(path)
    if not target.exists():
        return f"{target} does not exist"

    try:
        stat = target.stat()
    except OSError as exc:
        raise ToolError(f"could not stat {target}: {exc}") from exc

    kind = "directory" if target.is_dir() else "file"
    return f"{target}\n  kind: {kind}\n  size: {stat.st_size} bytes\n  modified: {stat.st_mtime}"
