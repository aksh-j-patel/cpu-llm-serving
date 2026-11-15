import hashlib
import json
from pathlib import Path

import pytest

from bench import models, paths


def test_git_blob_sha1_matches_git(tmp_path: Path):
    # Known values from `git hash-object`
    empty = tmp_path / "empty"
    empty.write_bytes(b"")
    hello = tmp_path / "hello"
    hello.write_bytes(b"hello\n")
    assert models.git_blob_sha1(empty) == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
    assert models.git_blob_sha1(hello) == "ce013625030ba8dba906f756967f9e9ca394464a"


def _spec(tmp_path: Path, content: bytes, lfs: bool) -> models.FileSpec:
    f = tmp_path / "m" / "file.bin"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(content)
    return models.FileSpec(
        model="m", repo="org/repo", revision="abc", filename="file.bin", size=len(content),
        sha256=hashlib.sha256(content).hexdigest() if lfs else None,
        git_sha1=None if lfs else models.git_blob_sha1(f))


@pytest.mark.parametrize("lfs", [True, False])
def test_verify_ok_then_cached(tmp_path: Path, lfs: bool):
    spec = _spec(tmp_path, b"model weights", lfs)
    assert models.verify_file(spec, tmp_path) == (True, "ok")
    assert models.verify_file(spec, tmp_path) == (True, "ok (cached)")


def test_verify_detects_corruption(tmp_path: Path):
    spec = _spec(tmp_path, b"model weights", lfs=True)
    models.verify_file(spec, tmp_path)
    (tmp_path / "m" / "file.bin").write_bytes(b"model WEIGHTS")  # same size, new mtime
    ok, reason = models.verify_file(spec, tmp_path)
    assert not ok and "hash mismatch" in reason


def test_verify_size_and_missing(tmp_path: Path):
    spec = _spec(tmp_path, b"abc", lfs=True)
    (tmp_path / "m" / "file.bin").write_bytes(b"abcd")
    assert models.verify_file(spec, tmp_path)[1].startswith("size")
    (tmp_path / "m" / "file.bin").unlink()
    assert models.verify_file(spec, tmp_path) == (False, "missing")


def test_repo_config_sets_resolve():
    cfg = models.load_config()
    smoke = models.resolve_set("smoke", cfg)
    quick = models.resolve_set("quick", cfg)
    full = models.resolve_set("full", cfg)
    assert set(smoke) < set(quick) < set(full) == set(cfg["models"])
    assert len(full) == len(set(full))


def test_set_errors():
    cfg = {"models": {"a": {}}, "sets": {"x": ["@y"], "y": ["@x"], "bad": ["zzz"]}}
    with pytest.raises(models.ModelError, match="includes itself"):
        models.resolve_set("x", cfg)
    with pytest.raises(models.ModelError, match="unknown model"):
        models.resolve_set("bad", cfg)
    with pytest.raises(models.ModelError, match="unknown model set"):
        models.resolve_set("nope", cfg)


def test_lock_file_shape_if_present():
    if not paths.MODELS_LOCK.exists():
        pytest.skip("models.lock.json not generated yet (run `make lock-models`)")
    lock = json.loads(paths.MODELS_LOCK.read_text())
    cfg = models.load_config()
    for key, m in cfg["models"].items():
        entry = lock["models"][key]
        assert entry["repo"] == m["repo"] and len(entry["revision"]) == 40
        assert [f["filename"] for f in entry["files"]] == m["files"]
        for f in entry["files"]:
            assert f["size"] > 0 and (f["sha256"] or f["git_sha1"])
