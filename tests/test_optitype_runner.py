"""Tests for the OptiType runner's execution guards.

These cover the two ways a real whole-exome run gets silently ruined: a
wall-clock ceiling short enough to kill it partway through, and a paired
FASTQ whose mate is invisible inside the container's single mount.
"""

from __future__ import annotations

import subprocess

import pytest

from cognisom.genomics import optitype_hla
from cognisom.genomics.optitype_hla import (
    DEFAULT_OPTITYPE_TIMEOUT,
    _run_optitype_docker,
    optitype_timeout,
)


class TestOptitypeTimeout:
    """The ceiling has to survive a whole-exome razers3 run."""

    def test_default_accommodates_whole_exome(self, monkeypatch):
        monkeypatch.delenv("COGNISOM_OPTITYPE_TIMEOUT", raising=False)
        # The bug this replaces was 600s, which cannot finish WES.
        assert optitype_timeout() == DEFAULT_OPTITYPE_TIMEOUT
        assert DEFAULT_OPTITYPE_TIMEOUT >= 3600

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("COGNISOM_OPTITYPE_TIMEOUT", "43200")
        assert optitype_timeout() == 43200

    @pytest.mark.parametrize("bad", ["", "abc", "0", "-1", "3.5"])
    def test_unusable_override_falls_back(self, monkeypatch, bad):
        monkeypatch.setenv("COGNISOM_OPTITYPE_TIMEOUT", bad)
        assert optitype_timeout() == DEFAULT_OPTITYPE_TIMEOUT


class TestPairedFastqMount:
    """Only one directory is mounted, so both mates must live in it."""

    def test_split_directories_rejected(self, tmp_path):
        r1 = tmp_path / "a" / "s_R1.fastq.gz"
        r2 = tmp_path / "b" / "s_R2.fastq.gz"
        for f in (r1, r2):
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(b"")

        with pytest.raises(ValueError, match="share a directory"):
            _run_optitype_docker(str(r1), str(tmp_path / "out"), str(r2))

    def test_both_mates_passed_to_container(self, tmp_path, monkeypatch):
        r1 = tmp_path / "s_R1.fastq.gz"
        r2 = tmp_path / "s_R2.fastq.gz"
        for f in (r1, r2):
            f.write_bytes(b"")

        seen = {}

        def fake_run(cmd, **kwargs):
            seen["cmd"] = cmd
            seen["timeout"] = kwargs.get("timeout")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(subprocess, "run", fake_run)
        out = tmp_path / "out"
        out.mkdir()
        (out / "s_result.tsv").write_text("")

        _run_optitype_docker(str(r1), str(out), str(r2))

        cmd = seen["cmd"]
        # Both mates are arguments to -i, in order, before --dna.
        i = cmd.index("-i")
        assert cmd[i + 1] == "/data/s_R1.fastq.gz"
        assert cmd[i + 2] == "/data/s_R2.fastq.gz"
        assert cmd[i + 3] == "--dna"
        assert seen["timeout"] == DEFAULT_OPTITYPE_TIMEOUT

    def test_timeout_reports_the_ceiling(self, tmp_path, monkeypatch):
        r1 = tmp_path / "s_R1.fastq.gz"
        r1.write_bytes(b"")

        def fake_run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))

        monkeypatch.setattr(subprocess, "run", fake_run)
        monkeypatch.setenv("COGNISOM_OPTITYPE_TIMEOUT", "120")

        with pytest.raises(RuntimeError, match="120s ceiling"):
            _run_optitype_docker(str(r1), str(tmp_path))


class TestContainerInvocation:
    """The image reference and command shape the runner depends on."""

    def test_image_is_the_bioconda_build(self):
        # fred2/optitype stops at release-v1.3.1; :1.3.5 there is a 404.
        assert "biocontainers" in optitype_hla.OPTITYPE_IMAGE
        assert "1.3.5" in optitype_hla.OPTITYPE_IMAGE

    def test_pipeline_script_named_explicitly(self, tmp_path, monkeypatch):
        """The image entrypoint is a conda wrapper, not OptiTypePipeline.py."""
        r1 = tmp_path / "s_R1.fastq.gz"
        r1.write_bytes(b"")
        seen = {}

        def fake_run(cmd, **kwargs):
            seen["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(subprocess, "run", fake_run)
        out = tmp_path / "out"
        out.mkdir()
        (out / "s_result.tsv").write_text("")

        _run_optitype_docker(str(r1), str(out))
        cmd = seen["cmd"]

        # The command word must immediately follow the image reference.
        assert cmd[cmd.index(optitype_hla.OPTITYPE_IMAGE) + 1] == "OptiTypePipeline.py"
        # And OptiType needs its config to locate razers3.
        assert cmd[cmd.index("-c") + 1] == optitype_hla.OPTITYPE_CONFIG
        # Results must not land root-owned on the host.
        assert "--user" in cmd
