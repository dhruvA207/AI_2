"""Tests for the tool registry, the built-in tools, and the executor."""

from __future__ import annotations

from pathlib import Path

import pytest

from arc.agent.executor import Executor
from arc.errors import ToolError
from arc.tools import filesystem, shell
from arc.tools.registry import ToolRegistry, build_parameters

# ── Registry and schema generation ──────────────────────────────────────────────


def test_schema_is_derived_from_hints_and_docstring() -> None:
    """Derived, not declared, so it cannot drift from the implementation."""
    registry = ToolRegistry()

    @registry.register
    def sample(path: str, count: int = 5, flag: bool = False) -> str:
        """Do a thing.

        Args:
            path: Where to do it.
            count: How many times.
        """
        return path

    tool = registry.get("sample")
    assert tool.description == "Do a thing."
    properties = tool.parameters["properties"]
    assert properties["path"] == {"type": "string", "description": "Where to do it."}
    assert properties["count"]["type"] == "integer"
    assert properties["flag"]["type"] == "boolean"
    assert tool.parameters["required"] == ["path"]


def test_defaults_are_advertised() -> None:
    registry = ToolRegistry()

    @registry.register
    def sample(count: int = 7) -> str:
        """Summary."""
        return str(count)

    assert registry.get("sample").parameters["properties"]["count"]["default"] == 7


def test_scalar_type_mapping() -> None:
    """bool is a subclass of int, so a naive subclass check types every flag as an
    integer and the model starts sending 1 and 0 for booleans."""

    def make(a: str, b: int, c: float, d: bool) -> str:
        """Summary."""
        return ""

    properties = build_parameters(make)["properties"]
    assert properties["a"]["type"] == "string"
    assert properties["b"]["type"] == "integer"
    assert properties["c"]["type"] == "number"
    assert properties["d"]["type"] == "boolean"


def test_optional_is_unwrapped() -> None:
    """`X | None` describes X, optional — not a union the model has to reason about."""

    def make(value: int | None = None) -> str:
        """Summary."""
        return str(value)

    assert build_parameters(make)["properties"]["value"]["type"] == "integer"


def test_list_becomes_an_array_with_item_type() -> None:
    def make(values: list[str]) -> str:
        """Summary."""
        return str(values)

    schema = build_parameters(make)["properties"]["values"]
    assert schema["type"] == "array"
    assert schema["items"]["type"] == "string"


def test_unannotated_parameter_falls_back_to_string() -> None:
    def make(value) -> str:  # type: ignore[no-untyped-def]
        """Summary."""
        return str(value)

    assert build_parameters(make)["properties"]["value"]["type"] == "string"


def test_docstring_is_required() -> None:
    """It is what the model reads to decide whether to call the tool."""
    registry = ToolRegistry()
    with pytest.raises(ToolError, match="needs a docstring"):

        @registry.register
        def undocumented(x: str) -> str:
            return x


def test_unknown_tool_error_lists_alternatives() -> None:
    registry = ToolRegistry()

    @registry.register
    def known(x: str) -> str:
        """Summary."""
        return x

    with pytest.raises(ToolError, match="known"):
        registry.get("unknown")


def test_registry_membership_and_length() -> None:
    registry = ToolRegistry()

    @registry.register
    def sample(x: str) -> str:
        """Summary."""
        return x

    assert "sample" in registry
    assert len(registry) == 1


def test_decorator_preserves_the_function() -> None:
    """The decorated function must stay directly callable and correctly typed."""
    registry = ToolRegistry()

    @registry.register
    def double(value: int) -> int:
        """Summary."""
        return value * 2

    assert double(21) == 42


def test_prompt_rendering_marks_optional_arguments() -> None:
    """This is the whole tool interface for a model without native tool calling."""
    registry = ToolRegistry()

    @registry.register
    def sample(path: str, deep: bool = False) -> str:
        """Do a thing.

        Args:
            path: Where.
        """
        return path

    rendered = registry.render_prompt()
    assert "sample: Do a thing." in rendered
    assert "path: string" in rendered
    assert "deep?: boolean" in rendered


def test_categories_filter_schemas() -> None:
    registry = ToolRegistry()

    @registry.register(category="a")
    def first(x: str) -> str:
        """Summary."""
        return x

    @registry.register(category="b")
    def second(x: str) -> str:
        """Summary."""
        return x

    assert [s.name for s in registry.schemas(categories=["a"])] == ["first"]


def test_builtin_tools_are_registered() -> None:
    from arc.tools import registry as default_registry

    for name in ("read_file", "write_file", "run_command", "run_python", "list_directory"):
        assert name in default_registry


def test_mutating_tools_are_flagged() -> None:
    """--dry-run depends on this classification being right."""
    from arc.tools import registry as default_registry

    assert default_registry.get("write_file").mutating is True
    assert default_registry.get("read_file").mutating is False


# ── Filesystem tools ────────────────────────────────────────────────────────────


def test_read_and_write_roundtrip(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "file.txt"
    filesystem.write_file(str(target), "hello")
    assert filesystem.read_file(str(target)) == "hello"


def test_write_creates_parent_directories(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c.txt"
    filesystem.write_file(str(target), "x")
    assert target.is_file()


def test_append_mode(tmp_path: Path) -> None:
    target = tmp_path / "f.txt"
    filesystem.write_file(str(target), "one")
    filesystem.write_file(str(target), "two", append=True)
    assert filesystem.read_file(str(target)) == "onetwo"


def test_read_missing_file_is_a_tool_error(tmp_path: Path) -> None:
    with pytest.raises(ToolError, match="no such file"):
        filesystem.read_file(str(tmp_path / "absent"))


def test_read_directory_suggests_the_right_tool(tmp_path: Path) -> None:
    with pytest.raises(ToolError, match="list_directory"):
        filesystem.read_file(str(tmp_path))


def test_read_truncates_and_says_so(tmp_path: Path) -> None:
    target = tmp_path / "big.txt"
    target.write_text("x" * 5000, encoding="utf-8")
    output = filesystem.read_file(str(target), max_chars=100)
    assert "truncated" in output
    assert len(output) < 500


def test_read_survives_invalid_utf8(tmp_path: Path) -> None:
    """One bad byte must degrade a character, not fail the whole read."""
    target = tmp_path / "bin.txt"
    target.write_bytes(b"good\xff\xfebad")
    assert "good" in filesystem.read_file(str(target))


def test_list_directory(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    output = filesystem.list_directory(str(tmp_path))
    assert "a.txt" in output
    assert "sub/" in output


def test_list_directory_with_pattern(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    (tmp_path / "b.md").write_text("x", encoding="utf-8")
    output = filesystem.list_directory(str(tmp_path), pattern="*.txt")
    assert "a.txt" in output
    assert "b.md" not in output


def test_list_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(ToolError, match="not a directory"):
        filesystem.list_directory(str(tmp_path / "absent"))


def test_search_files(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("needle here\nother", encoding="utf-8")
    (tmp_path / "b.txt").write_text("nothing", encoding="utf-8")
    output = filesystem.search_files(str(tmp_path), "NEEDLE")
    assert "a.txt:1" in output
    assert "b.txt" not in output


def test_search_reports_no_matches(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    assert "no matches" in filesystem.search_files(str(tmp_path), "absent")


def test_move_and_copy(tmp_path: Path) -> None:
    src = tmp_path / "src.txt"
    src.write_text("data", encoding="utf-8")

    filesystem.copy_path(str(src), str(tmp_path / "copy.txt"))
    assert (tmp_path / "copy.txt").read_text(encoding="utf-8") == "data"

    filesystem.move_path(str(src), str(tmp_path / "moved.txt"))
    assert not src.exists()
    assert (tmp_path / "moved.txt").is_file()


def test_delete_file(tmp_path: Path) -> None:
    target = tmp_path / "f.txt"
    target.write_text("x", encoding="utf-8")
    filesystem.delete_path(str(target))
    assert not target.exists()


def test_delete_nonempty_directory_needs_recursive(tmp_path: Path) -> None:
    """One wrong argument should not take a whole tree with it."""
    directory = tmp_path / "d"
    directory.mkdir()
    (directory / "f.txt").write_text("x", encoding="utf-8")

    with pytest.raises(ToolError, match="recursive"):
        filesystem.delete_path(str(directory))

    filesystem.delete_path(str(directory), recursive=True)
    assert not directory.exists()


def test_path_info(tmp_path: Path) -> None:
    target = tmp_path / "f.txt"
    target.write_text("hello", encoding="utf-8")
    assert "file" in filesystem.path_info(str(target))
    assert "does not exist" in filesystem.path_info(str(tmp_path / "absent"))


# ── Shell tools ─────────────────────────────────────────────────────────────────


def test_run_command_captures_output_and_exit_code(tmp_path: Path) -> None:
    output = shell.run_command("echo hello", working_directory=str(tmp_path))
    assert "exit code: 0" in output
    assert "hello" in output


def test_run_command_reports_nonzero_exit(tmp_path: Path) -> None:
    assert "exit code: 3" in shell.run_command("exit 3", working_directory=str(tmp_path))


def test_run_command_captures_stderr(tmp_path: Path) -> None:
    """Many tools write useful information to stderr while succeeding."""
    output = shell.run_command("echo oops >&2", working_directory=str(tmp_path))
    assert "stderr" in output
    assert "oops" in output


def test_run_command_timeout_is_an_observation(tmp_path: Path) -> None:
    """A timeout is something the model can act on, not an exception."""
    output = shell.run_command("sleep 5", working_directory=str(tmp_path), timeout=0.3)
    assert "timed out" in output


def test_run_command_rejects_a_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(ToolError, match="working directory"):
        shell.run_command("echo x", working_directory=str(tmp_path / "absent"))


def test_run_python(arc_home_tmp: Path) -> None:
    assert "42" in shell.run_python("print(6 * 7)")


def test_run_python_reports_errors(arc_home_tmp: Path) -> None:
    output = shell.run_python("raise ValueError('boom')")
    assert "ValueError" in output
    assert "exit code: 1" in output


def test_which() -> None:
    assert "not on PATH" in shell.which("definitely-not-a-real-binary-xyz")


# ── Executor ────────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_registry() -> ToolRegistry:
    registry = ToolRegistry()

    @registry.register
    def echo(text: str, times: int = 1) -> str:
        """Repeat text."""
        return text * times

    @registry.register(mutating=True)
    def mutate(target: str) -> str:
        """Change something."""
        return f"changed {target}"

    @registry.register
    def explode() -> str:
        """Always fails."""
        raise ToolError("deliberate failure")

    @registry.register
    def crash() -> str:
        """Raises an unexpected error."""
        raise RuntimeError("bug in the tool")

    return registry


def test_executor_runs_a_tool(sample_registry: ToolRegistry) -> None:
    observation = Executor(sample_registry).execute("echo", {"text": "hi"})
    assert observation.ok
    assert observation.output == "hi"


def test_executor_reports_unknown_tools_without_raising(sample_registry: ToolRegistry) -> None:
    """A tool error is information for the model, not a crash."""
    observation = Executor(sample_registry).execute("nope", {})
    assert not observation.ok
    assert "no tool named" in observation.output


def test_executor_reports_missing_arguments(sample_registry: ToolRegistry) -> None:
    observation = Executor(sample_registry).execute("echo", {})
    assert not observation.ok
    assert "missing required argument" in observation.output


def test_executor_reports_unknown_arguments(sample_registry: ToolRegistry) -> None:
    observation = Executor(sample_registry).execute("echo", {"text": "x", "bogus": 1})
    assert not observation.ok
    assert "unknown argument" in observation.output


def test_executor_catches_tool_errors(sample_registry: ToolRegistry) -> None:
    observation = Executor(sample_registry).execute("explode", {})
    assert not observation.ok
    assert "deliberate failure" in observation.output


def test_executor_catches_unexpected_exceptions(sample_registry: ToolRegistry) -> None:
    """A bug in a tool must not take down the loop, and its type must be visible."""
    observation = Executor(sample_registry).execute("crash", {})
    assert not observation.ok
    assert "RuntimeError" in observation.output


@pytest.mark.parametrize(
    ("value", "expected"), [("true", True), ("yes", True), ("false", False), ("0", False)]
)
def test_executor_coerces_string_booleans(value: str, expected: bool) -> None:
    """Models routinely send "true" for a bool; rejecting that is useless pedantry."""
    registry = ToolRegistry()

    @registry.register
    def flag(enabled: bool) -> str:
        """Summary."""
        return str(enabled)

    observation = Executor(registry).execute("flag", {"enabled": value})
    assert observation.output == str(expected)


def test_executor_coerces_string_numbers() -> None:
    registry = ToolRegistry()

    @registry.register
    def add(value: int) -> str:
        """Summary."""
        return str(value + 1)

    assert Executor(registry).execute("add", {"value": "41"}).output == "42"


def test_dry_run_skips_mutating_tools(sample_registry: ToolRegistry) -> None:
    observation = Executor(sample_registry, dry_run=True).execute("mutate", {"target": "x"})
    assert observation.ok
    assert observation.dry_run
    assert "[dry-run]" in observation.output


def test_dry_run_still_runs_read_only_tools(sample_registry: ToolRegistry) -> None:
    """A dry run that cannot even look at things tells you nothing."""
    observation = Executor(sample_registry, dry_run=True).execute("echo", {"text": "hi"})
    assert observation.output == "hi"
    assert not observation.dry_run


def test_observations_are_audited(sample_registry: ToolRegistry, arc_home_tmp: Path) -> None:
    from arc.audit import AuditLogger

    audit = AuditLogger()
    Executor(sample_registry, audit=audit).execute("echo", {"text": "hi"})
    records = audit.read_recent()
    assert records[-1]["event"] == "tool.echo"


def test_failed_observations_are_audited_as_errors(
    sample_registry: ToolRegistry, arc_home_tmp: Path
) -> None:
    from arc.audit import AuditLogger

    audit = AuditLogger()
    Executor(sample_registry, audit=audit).execute("explode", {})
    assert audit.read_recent()[-1]["status"] == "error"


def test_observation_render_marks_failure(sample_registry: ToolRegistry) -> None:
    observation = Executor(sample_registry).execute("explode", {})
    assert "(failed)" in observation.render()
