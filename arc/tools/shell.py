"""Shell and code execution.

**Arbitrary command execution, unrestricted** (§0.3). No allow-list, no deny-list, no
confirmation. The brief is explicit that this is deliberate and not to be negotiated;
the audit log and the kill switch are the tools for understanding and stopping it.

The one thing enforced here is a **timeout**, and that is not a safety measure — it is
the difference between an agent that recovers from a hung command and one that waits
forever holding the whole loop. A timeout is reported back to the model as an
observation it can act on.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

from arc.errors import ToolError
from arc.log import get_logger
from arc.paths import arc_home
from arc.tools.registry import tool

_log = get_logger(__name__)

#: Default seconds before a command is killed. Generous enough for a build, short
#: enough that a hung process does not strand the agent.
DEFAULT_TIMEOUT = 120.0

#: Command output beyond this is truncated before it reaches the model. A `find /`
#: would otherwise consume the whole context window in one observation.
_MAX_OUTPUT_CHARS = 20_000


def scratch_dir() -> Path:
    """Return the directory for generated code, creating it if needed."""
    target = arc_home() / "scratch"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _truncate(text: str, label: str) -> str:
    """Clip oversized output, saying so."""
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text
    return text[:_MAX_OUTPUT_CHARS] + f"\n[{label} truncated at {_MAX_OUTPUT_CHARS} chars]"


def _format(result: subprocess.CompletedProcess[str]) -> str:
    """Render a completed process for the model.

    Both streams are returned, labelled. Many tools write useful information to stderr
    while succeeding, so discarding it on a zero exit would throw away the answer.
    """
    parts = [f"exit code: {result.returncode}"]
    if result.stdout.strip():
        parts.append("stdout:\n" + _truncate(result.stdout.rstrip(), "stdout"))
    if result.stderr.strip():
        parts.append("stderr:\n" + _truncate(result.stderr.rstrip(), "stderr"))
    if len(parts) == 1:
        parts.append("(no output)")
    return "\n\n".join(parts)


@tool(category="shell", mutating=True)
def run_command(
    command: str, working_directory: str = ".", timeout: float = DEFAULT_TIMEOUT
) -> str:
    """Run a shell command and return its exit code and output.

    Args:
        command: Command line to execute.
        working_directory: Directory to run it in.
        timeout: Seconds before the command is killed.
    """
    cwd = Path(working_directory).expanduser().resolve(strict=False)
    if not cwd.is_dir():
        raise ToolError(f"working directory does not exist: {cwd}")

    try:
        # shell=True is the point of this tool: the model writes pipelines, globs, and
        # redirections, and parsing those ourselves would be reimplementing a shell.
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        # An observation, not an exception: the model may want to retry with a longer
        # timeout or a narrower command.
        return f"command timed out after {timeout}s: {command}"
    except OSError as exc:
        raise ToolError(f"could not run {command!r}: {exc}") from exc

    return _format(result)


@tool(category="shell")
def which(program: str) -> str:
    """Report the full path of an executable, or that it is not installed.

    Args:
        program: Executable name to look up.
    """
    import shutil as _shutil

    found = _shutil.which(program)
    return found if found else f"{program} is not on PATH"


@tool(category="code", mutating=True)
def run_python(code: str, timeout: float = DEFAULT_TIMEOUT) -> str:
    """Run Python code in a scratch directory and return its output.

    Args:
        code: Python source to execute.
        timeout: Seconds before execution is killed.
    """
    scratch = scratch_dir()
    script = scratch / "snippet.py"

    try:
        script.write_text(code, encoding="utf-8")
    except OSError as exc:
        raise ToolError(f"could not write scratch script: {exc}") from exc

    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=scratch,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            # Unbuffered, so output survives a timeout kill instead of dying in the
            # pipe buffer — the run that hangs is the one you most want output from.
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    except subprocess.TimeoutExpired:
        return f"python execution timed out after {timeout}s"
    except OSError as exc:
        raise ToolError(f"could not run python: {exc}") from exc

    return _format(result)


@tool(category="shell")
def current_directory() -> str:
    """Report the current working directory."""
    return str(Path.cwd())


@tool(category="shell")
def environment_variable(name: str) -> str:
    """Read an environment variable.

    Args:
        name: Variable name.
    """
    value = os.environ.get(name)
    return value if value is not None else f"{name} is not set"


def quote(argument: str) -> str:
    """Shell-quote a string, for tools that build command lines."""
    return shlex.quote(argument)
