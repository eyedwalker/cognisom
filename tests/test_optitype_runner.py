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


class TestResultParsing:
    """Locating allele columns by name, not by position."""

    # Verbatim from OptiType 1.3.5 on SRR7890874 (HCC1395BL, 8M read pairs).
    REAL_RESULT = (
        "\tA1\tA2\tB1\tB2\tC1\tC2\tReads\tObjective\n"
        "0\tA*29:02\tA*29:02\tB*45:01\tB*45:01\tC*06:02\tC*06:02\t245.0\t245.0\n"
    )

    def _write(self, tmp_path, text):
        f = tmp_path / "r_result.tsv"
        f.write_text(text)
        return str(f)

    def test_all_six_alleles_survive(self, tmp_path):
        """The row index used to be consumed as an allele, dropping C2."""
        alleles = optitype_hla._parse_optitype_result(
            self._write(tmp_path, self.REAL_RESULT)
        )

        assert alleles == [
            "HLA-A*29:02", "HLA-A*29:02",
            "HLA-B*45:01", "HLA-B*45:01",
            "HLA-C*06:02", "HLA-C*06:02",
        ]
        # Both HLA-C copies, not one.
        assert sum(a.startswith("HLA-C") for a in alleles) == 2

    def test_heterozygous_typing(self, tmp_path):
        text = (
            "\tA1\tA2\tB1\tB2\tC1\tC2\tReads\tObjective\n"
            "0\tA*02:01\tA*03:01\tB*07:02\tB*44:02\tC*05:01\tC*07:02\t9\t9\n"
        )
        assert optitype_hla._parse_optitype_result(self._write(tmp_path, text)) == [
            "HLA-A*02:01", "HLA-A*03:01",
            "HLA-B*07:02", "HLA-B*44:02",
            "HLA-C*05:01", "HLA-C*07:02",
        ]

    def test_column_reordering_is_tolerated(self, tmp_path):
        text = (
            "\tC1\tC2\tA1\tA2\tB1\tB2\tReads\n"
            "0\tC*05:01\tC*07:02\tA*02:01\tA*03:01\tB*07:02\tB*44:02\t9\n"
        )
        # Returned in canonical A, B, C order regardless of file order.
        assert optitype_hla._parse_optitype_result(self._write(tmp_path, text))[0] \
            == "HLA-A*02:01"

    def test_uncalled_locus_is_reported(self, tmp_path, caplog):
        text = (
            "\tA1\tA2\tB1\tB2\tC1\tC2\tReads\n"
            "0\tA*02:01\tA*03:01\tB*07:02\tB*44:02\tC*05:01\t\t9\n"
        )
        with caplog.at_level("WARNING"):
            alleles = optitype_hla._parse_optitype_result(
                self._write(tmp_path, text)
            )

        assert len(alleles) == 5
        assert "C2" in caplog.text

    def test_missing_columns_rejected(self, tmp_path):
        text = "\tA1\tA2\tReads\n0\tA*02:01\tA*03:01\t9\n"
        with pytest.raises(RuntimeError, match="missing columns"):
            optitype_hla._parse_optitype_result(self._write(tmp_path, text))


class TestDropoutSignature:
    """All three class-I loci homozygous is usually thin coverage, not biology."""

    def _write(self, tmp_path, text):
        f = tmp_path / "d_result.tsv"
        f.write_text(text)
        return str(f)

    def test_all_homozygous_is_flagged(self, tmp_path, caplog):
        """Exactly the call OptiType made on SRR7890874 at 8M read pairs.

        Two of those three homozygous calls disagreed with the published
        type for the line, so the warning is load-bearing.
        """
        text = (
            "\tA1\tA2\tB1\tB2\tC1\tC2\tReads\tObjective\n"
            "0\tA*29:02\tA*29:02\tB*45:01\tB*45:01\tC*06:02\tC*06:02\t245.0\t245.0\n"
        )
        with caplog.at_level("WARNING"):
            optitype_hla._parse_optitype_result(self._write(tmp_path, text))

        assert "dropout" in caplog.text
        assert "245 reads" in caplog.text

    def test_heterozygous_typing_is_not_flagged(self, tmp_path, caplog):
        text = (
            "\tA1\tA2\tB1\tB2\tC1\tC2\tReads\tObjective\n"
            "0\tA*29:02\tA*29:02\tB*08:01\tB*45:01\tC*06:02\tC*07:01\t900\t900\n"
        )
        with caplog.at_level("WARNING"):
            optitype_hla._parse_optitype_result(self._write(tmp_path, text))

        assert caplog.text == ""

    def test_well_supported_homozygosity_is_not_called_dropout(
        self, tmp_path, caplog,
    ):
        """The merged 40M-pair call: unusual, but 1198 reads back it."""
        text = (
            "\tA1\tA2\tB1\tB2\tC1\tC2\tReads\tObjective\n"
            "0\tA*29:02\tA*29:02\tB*45:01\tB*45:01\tC*06:02\tC*06:02"
            "\t1198.0\t1198.0\n"
        )
        with caplog.at_level("WARNING"):
            optitype_hla._parse_optitype_result(self._write(tmp_path, text))

        assert "1198 reads" in caplog.text
        assert "support is adequate" in caplog.text
        # Advice to add reads would be stale at this depth.
        assert "Re-run with more reads" not in caplog.text

    def test_thin_support_is_flagged_even_when_heterozygous(
        self, tmp_path, caplog,
    ):
        text = (
            "\tA1\tA2\tB1\tB2\tC1\tC2\tReads\tObjective\n"
            "0\tA*29:02\tA*29:02\tB*08:01\tB*45:01\tC*06:02\tC*07:01\t60\t60\n"
        )
        with caplog.at_level("WARNING"):
            optitype_hla._parse_optitype_result(self._write(tmp_path, text))

        assert "only 60 reads" in caplog.text


class TestFailureExplanation:
    """razers3 dying for memory surfaces as a missing BAM three stages later."""

    OOM_STDERR = (
        '[E::hts_open_format] Failed to open file '
        '"/out/2026_08_22_23_32_36/2026_08_22_23_32_36_1.bam" : '
        "No such file or directory\n"
        "FileNotFoundError: [Errno 2] Could not open alignment file"
    )

    def test_missing_bam_points_at_memory(self):
        msg = optitype_hla._explain_docker_failure(self.OOM_STDERR)
        assert "razers3" in msg and "memory" in msg
        assert "oom-kill" in msg
        # The original text is preserved, not swallowed.
        assert "hts_open_format" in msg

    def test_unrelated_failure_passes_through(self):
        msg = optitype_hla._explain_docker_failure("config.ini not found")
        assert msg == "OptiType Docker failed: config.ini not found"
        assert "razers3" not in msg
