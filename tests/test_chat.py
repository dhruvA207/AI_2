"""Tests for the chat REPL."""

from __future__ import annotations

from pathlib import Path

import pytest

from arc.config import Config
from arc.interface.chat import Session, _handle_command, run
from arc.model.base import Message, ModelCapabilities
from tests.fakes import ExplodingModel, FakeModel


def a_config(directory: Path) -> Config:
    """A Config with generation defaults."""
    (directory / "models.yaml").write_text(
        "generation:\n  max_tokens: 64\n  temperature: 0.5\n", encoding="utf-8"
    )
    return Config.load(directory=directory, use_env=False)


def test_session_prepends_system_prompt() -> None:
    session = Session(model=FakeModel(), system_prompt="be terse")
    session.history.append(Message(role="user", content="hi"))
    assert [m.role for m in session.messages()] == ["system", "user"]


def test_session_without_system_prompt() -> None:
    session = Session(model=FakeModel())
    session.history.append(Message(role="user", content="hi"))
    assert [m.role for m in session.messages()] == ["user"]


def test_clear_keeps_system_prompt() -> None:
    session = Session(model=FakeModel(), system_prompt="be terse")
    session.history.append(Message(role="user", content="hi"))
    session.clear()
    assert session.history == []
    assert session.system_prompt == "be terse"


def test_token_use_reports_against_context() -> None:
    session = Session(model=FakeModel(context_length=1024))
    session.history.append(Message(role="user", content="one two three"))
    used, limit = session.token_use()
    assert limit == 1024
    assert used > 0


def test_exit_commands_stop_the_loop() -> None:
    session = Session(model=FakeModel())
    assert _handle_command("/exit", session) is False
    assert _handle_command("/quit", session) is False


def test_clear_command(capsys: pytest.CaptureFixture[str]) -> None:
    session = Session(model=FakeModel())
    session.history.append(Message(role="user", content="hi"))
    assert _handle_command("/clear", session) is True
    assert session.history == []
    assert "cleared" in capsys.readouterr().out


def test_system_command_sets_and_clears(capsys: pytest.CaptureFixture[str]) -> None:
    """Changing the system prompt mid-conversation must reset history, not blend."""
    session = Session(model=FakeModel())
    session.history.append(Message(role="user", content="hi"))
    _handle_command("/system you are a pirate", session)
    assert session.system_prompt == "you are a pirate"
    assert session.history == []


def test_system_command_without_argument_shows_current(
    capsys: pytest.CaptureFixture[str],
) -> None:
    session = Session(model=FakeModel(), system_prompt="be terse")
    _handle_command("/system", session)
    assert "be terse" in capsys.readouterr().out


def test_model_command_reports_capabilities(capsys: pytest.CaptureFixture[str]) -> None:
    model = FakeModel(capabilities=ModelCapabilities(max_context=99, native_tool_calling=True))
    _handle_command("/model", Session(model=model))
    assert "native" in capsys.readouterr().out


def test_model_command_reports_prompted_fallback(capsys: pytest.CaptureFixture[str]) -> None:
    _handle_command("/model", Session(model=FakeModel()))
    assert "prompted fallback" in capsys.readouterr().out


def test_tokens_command(capsys: pytest.CaptureFixture[str]) -> None:
    _handle_command("/tokens", Session(model=FakeModel()))
    assert "context" in capsys.readouterr().out


def test_unknown_command_is_reported(capsys: pytest.CaptureFixture[str]) -> None:
    assert _handle_command("/nonsense", Session(model=FakeModel())) is True
    assert "unknown command" in capsys.readouterr().out


def test_run_exits_on_command(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("builtins.input", lambda _="": "/exit")
    assert run(FakeModel(), a_config(config_dir)) == 0


def test_run_streams_a_reply(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    lines = iter(["hello", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _="": next(lines))
    assert run(FakeModel("well hello there"), a_config(config_dir)) == 0
    assert "well hello there" in capsys.readouterr().out


def test_run_keeps_transcript_across_turns(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    lines = iter(["first", "second", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _="": next(lines))
    model = FakeModel("ok")
    run(model, a_config(config_dir))
    # Second call should carry the first exchange plus the new question.
    assert [m.role for m in model.calls[1]] == ["user", "assistant", "user"]


def test_run_ignores_blank_input(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    lines = iter(["", "   ", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _="": next(lines))
    model = FakeModel()
    run(model, a_config(config_dir))
    assert model.calls == []


def test_run_exits_cleanly_on_eof(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def raise_eof(_: str = "") -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", raise_eof)
    assert run(FakeModel(), a_config(config_dir)) == 0


def test_failed_turn_drops_the_user_message(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A user message with no answer must not poison the next request."""
    lines = iter(["this will fail", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _="": next(lines))
    model = ExplodingModel()
    run(model, a_config(config_dir))
    assert "generation failed" in capsys.readouterr().out
