"""Measured HLA types must stay distinguishable from invented ones."""

from __future__ import annotations

import pytest

from cognisom.genomics.hla_typer import (
    MEASURED_HLA_TYPINGS,
    SYNTHETIC_HLA_PROFILES,
    HLATyper,
    MeasuredTyping,
)


class TestMeasuredTyping:
    def test_a_class_i_typing_has_six_alleles(self):
        with pytest.raises(ValueError, match="6 alleles"):
            MeasuredTyping(
                alleles=("HLA-A*29:02", "HLA-A*29:02"),
                source="truncated", reads=1,
            )

    def test_registries_do_not_overlap(self):
        """A sample is either measured or invented, never quietly both."""
        assert not (set(MEASURED_HLA_TYPINGS) & set(SYNTHETIC_HLA_PROFILES))

    def test_every_measured_entry_carries_its_provenance(self):
        for pid, typing in MEASURED_HLA_TYPINGS.items():
            assert typing.source, f"{pid} has no source"
            assert typing.reads > 0, f"{pid} has no read support"
            assert len(typing.alleles) == 6


class TestTypingResolution:
    def test_measured_type_is_used_and_marked_patient_specific(self):
        typer = HLATyper()
        alleles = typer.type_from_variants([], patient_id="SEQC2")

        assert alleles == list(MEASURED_HLA_TYPINGS["SEQC2"].alleles)
        assert typer.typing_method == HLATyper.METHOD_OPTITYPE
        assert typer.is_patient_specific is True

    def test_measured_beats_synthetic_for_the_same_id(self, monkeypatch):
        """If an id ever appears in both, the real reads win."""
        measured = MeasuredTyping(
            alleles=(
                "HLA-A*01:01", "HLA-A*02:01", "HLA-B*07:02",
                "HLA-B*08:01", "HLA-C*07:01", "HLA-C*07:02",
            ),
            source="reads", reads=900,
        )
        monkeypatch.setitem(MEASURED_HLA_TYPINGS, "COGNISOM-DEMO-001", measured)
        typer = HLATyper()

        assert typer.type_from_variants(
            [], patient_id="COGNISOM-DEMO-001"
        ) == list(measured.alleles)
        assert typer.typing_method == HLATyper.METHOD_OPTITYPE

    def test_unknown_patient_is_not_patient_specific(self):
        typer = HLATyper()
        typer.type_from_variants([], patient_id="nobody-in-particular")

        assert typer.is_patient_specific is False
