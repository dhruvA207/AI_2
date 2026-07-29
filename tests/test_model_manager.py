"""Tests for model management: status, selection, and removal.

``pull`` is not tested against the network — the suite must not download gigabytes.
Its guard clauses are covered instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from arc.config import Config
from arc.errors import ConfigError, ModelError
from arc.model.manager import is_downloaded, local_path, pull, remove, status_for, use
from arc.model.registry import parse_entry
from arc.paths import arc_home, models_dir

REGISTRY_YAML = """
registry:
  a:
    backend: mlx
    repo: org/a
    licence: Apache-2.0
    licence_verified: "2026-07-29"
    context_length: 4096
    approx_size_gb: 2.0
  b:
    backend: llamacpp
    repo: org/b
    licence: MIT
    licence_verified: "2026-07-29"
    context_length: 8192
active:
  chat: a
"""


def a_config(directory: Path) -> Config:
    """Build a Config from the two-entry registry above."""
    (directory / "models.yaml").write_text(REGISTRY_YAML, encoding="utf-8")
    return Config.load(directory=directory, use_env=False)


def make_weights(key: str, filename: str = "model.safetensors") -> Path:
    """Create a plausible weights file so is_downloaded() sees the model."""
    path = models_dir() / key
    path.mkdir(parents=True, exist_ok=True)
    target = path / filename
    target.write_bytes(b"x" * 2048)
    return target


def test_not_downloaded_when_directory_missing(arc_home_tmp: Path, config_dir: Path) -> None:
    entry = parse_entry(
        "a",
        {
            "backend": "mlx",
            "repo": "o/a",
            "licence": "MIT",
            "licence_verified": "x",
            "context_length": 8,
        },
    )
    assert is_downloaded(entry) is False


def test_not_downloaded_when_directory_is_empty(arc_home_tmp: Path) -> None:
    """An interrupted download leaves a directory; reporting that as ready would lie."""
    (models_dir() / "a").mkdir(parents=True)
    entry = parse_entry(
        "a",
        {
            "backend": "mlx",
            "repo": "o/a",
            "licence": "MIT",
            "licence_verified": "x",
            "context_length": 8,
        },
    )
    assert is_downloaded(entry) is False


@pytest.mark.parametrize("filename", ["model.safetensors", "model.gguf", "weights.npz"])
def test_downloaded_when_weights_present(arc_home_tmp: Path, filename: str) -> None:
    make_weights("a", filename)
    entry = parse_entry(
        "a",
        {
            "backend": "mlx",
            "repo": "o/a",
            "licence": "MIT",
            "licence_verified": "x",
            "context_length": 8,
        },
    )
    assert is_downloaded(entry) is True


def test_status_lists_everything_sorted(arc_home_tmp: Path, config_dir: Path) -> None:
    statuses = status_for(a_config(config_dir))
    assert [s.entry.key for s in statuses] == ["a", "b"]


def test_status_marks_active_role(arc_home_tmp: Path, config_dir: Path) -> None:
    statuses = {s.entry.key: s for s in status_for(a_config(config_dir))}
    assert statuses["a"].active_for == ["chat"]
    assert statuses["b"].active_for == []


def test_status_reports_disk_size(arc_home_tmp: Path, config_dir: Path) -> None:
    make_weights("a")
    statuses = {s.entry.key: s for s in status_for(a_config(config_dir))}
    assert statuses["a"].downloaded is True
    assert statuses["a"].size_on_disk_gb is not None
    assert statuses["b"].size_on_disk_gb is None


def test_use_writes_machine_local_override(arc_home_tmp: Path, config_dir: Path) -> None:
    """Switching models must not dirty the committed config."""
    target = use(a_config(config_dir), "b")
    assert target == arc_home() / "config.yaml"
    written = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert written["models"]["active"]["chat"] == "b"


def test_use_preserves_unrelated_keys(arc_home_tmp: Path, config_dir: Path) -> None:
    """~/.arc/config.yaml is a general override file, not ours alone."""
    override = arc_home() / "config.yaml"
    override.write_text(
        yaml.safe_dump({"logging": {"level": "debug"}, "models": {"active": {"vision": "v"}}}),
        encoding="utf-8",
    )
    use(a_config(config_dir), "b")
    written = yaml.safe_load(override.read_text(encoding="utf-8"))
    assert written["logging"]["level"] == "debug"
    assert written["models"]["active"]["vision"] == "v"
    assert written["models"]["active"]["chat"] == "b"


def test_use_rejects_unknown_model(arc_home_tmp: Path, config_dir: Path) -> None:
    with pytest.raises(ConfigError, match="unknown model"):
        use(a_config(config_dir), "nope")


def test_use_rejects_unknown_role(arc_home_tmp: Path, config_dir: Path) -> None:
    with pytest.raises(ConfigError, match="unknown role"):
        use(a_config(config_dir), "a", "telepathy")


def test_use_sets_other_roles(arc_home_tmp: Path, config_dir: Path) -> None:
    target = use(a_config(config_dir), "b", "embedding")
    written = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert written["models"]["active"]["embedding"] == "b"


def test_corrupt_override_is_caught_at_load(arc_home_tmp: Path, config_dir: Path) -> None:
    """Config.load reads ~/.arc/config.yaml, so it rejects the bad file first."""
    (arc_home() / "config.yaml").write_text("{ broken", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        a_config(config_dir)


def test_use_guards_against_a_corrupt_override(arc_home_tmp: Path, config_dir: Path) -> None:
    """And use() guards independently, for a file corrupted after config was loaded."""
    config = a_config(config_dir)
    (arc_home() / "config.yaml").write_text("{ broken", encoding="utf-8")
    with pytest.raises(ConfigError, match="could not parse"):
        use(config, "b")


def test_use_rejects_non_mapping_models_key(arc_home_tmp: Path, config_dir: Path) -> None:
    config = a_config(config_dir)
    (arc_home() / "config.yaml").write_text("models: 5\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not a mapping"):
        use(config, "b")


def test_pull_rejects_unknown_model(arc_home_tmp: Path, config_dir: Path) -> None:
    with pytest.raises(ConfigError, match="unknown model"):
        pull(a_config(config_dir), "nope")


def test_pull_is_a_noop_when_already_present(arc_home_tmp: Path, config_dir: Path) -> None:
    """Must not hit the network when the weights are already on disk."""
    make_weights("a")
    assert pull(a_config(config_dir), "a") == local_path("a")


def test_remove_deletes_weights(arc_home_tmp: Path, config_dir: Path) -> None:
    make_weights("a")
    remove(a_config(config_dir), "a")
    assert not local_path("a").exists()


def test_remove_reports_when_not_downloaded(arc_home_tmp: Path, config_dir: Path) -> None:
    with pytest.raises(ModelError, match="not downloaded"):
        remove(a_config(config_dir), "a")


def test_remove_rejects_unknown_model(arc_home_tmp: Path, config_dir: Path) -> None:
    with pytest.raises(ConfigError, match="unknown model"):
        remove(a_config(config_dir), "nope")
