"""The flagship demo labelled a population default "Patient HLA Type".

HCC1395 has a real typing in this repository: OptiType on the matched
normal (SRR7890874, HCC1395BL, WXS) gives HLA-A*29:02, B*45:01, C*06:02,
homozygous at all three loci. The SEQC2 demo displays A*02:01, A*03:01,
B*07:02, B*08:01, C*07:01, C*07:02 -- a population-frequency default
that shares NO allele with it.

Everything downstream inherits that: every peptide-HLA restriction on
screen, the vaccine designed from them on page 31, and the exported JSON
that carries the alleles onward. The product's own pipeline guide says
"Wrong HLA = wrong neoantigen predictions = wrong vaccine targets" on a
different page from the one doing the work.

``hla_is_patient_specific`` and ``hla_typing_method`` were on
PatientProfile the whole time. Nothing read them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEMO = REPO_ROOT / "cognisom" / "genomics" / "seqc2_precomputed.json"
PAGES = REPO_ROOT / "cognisom" / "dashboard" / "_pages"


@pytest.fixture(scope="module")
def demo():
    return json.loads(DEMO.read_text())


def test_the_demo_declares_that_it_was_not_typed(demo):
    assert demo["hla_is_patient_specific"] is False
    assert demo["hla_typing_method"] == "population_default"


def test_the_demo_records_what_the_real_typing_is(demo):
    """The measured alleles are in this repository. A reader comparing
    the two should not have to go and find them."""
    assert set(demo["hla_measured_alleles"]) == {
        "HLA-A*29:02", "HLA-B*45:01", "HLA-C*06:02"}
    assert "SRR7890874" in demo["hla_note"]


def test_the_displayed_alleles_really_do_share_none_with_the_measured(demo):
    """If this ever stops being true the warnings may be relaxed. It is
    the whole reason they exist."""
    from cognisom.genomics.hla_typer import MEASURED_HLA_TYPINGS

    measured = {a for a in MEASURED_HLA_TYPINGS["SEQC2"].alleles}
    displayed = set(demo["hla_alleles"])
    assert not (measured & displayed), (
        f"the demo now overlaps the measured typing: {measured & displayed}")


def test_the_profile_is_built_with_the_flag(demo):
    """app.py constructs PatientProfile directly for this demo -- it
    never calls HLATyper -- so the flag has to be passed explicitly or it
    silently defaults to False with nothing displaying it."""
    text = (REPO_ROOT / "cognisom" / "dashboard" / "app.py").read_text()
    assert "hla_is_patient_specific=bool(_data.get(" in text
    assert 'hla_typing_method=_data.get(' in text


def test_the_vaccine_page_warns_when_the_patient_was_not_typed():
    text = (PAGES / "31_neoantigen_vaccine.py").read_text()
    assert 'st.markdown(f"**Patient HLA Type:** {hla_display}")' not in \
        text.split("if _typed:")[0], \
        "the unconditional 'Patient HLA Type' heading is back"
    assert "population default" in text
    assert "hla_is_patient_specific" in text
    assert "wrong vaccine targets" in text.lower()


def test_the_caveat_travels_in_the_exported_file():
    """A download that carries the alleles without the method lets the
    error leave the product."""
    text = (PAGES / "31_neoantigen_vaccine.py").read_text()
    export = text[text.index("export_data = {"):]
    export = export[:export.index("}")]
    assert "hla_is_patient_specific" in export
    assert "hla_typing_method" in export


def test_the_clinical_report_does_not_present_a_default_as_a_genotype():
    text = (PAGES / "33_clinical_report.py").read_text()
    assert "hla_is_patient_specific" in text
    assert "population default" in text


def test_a_typed_patient_is_not_warned_about():
    """The warning must be conditional, or it becomes noise that a
    reader learns to skip."""
    text = (PAGES / "31_neoantigen_vaccine.py").read_text()
    assert "if _typed:" in text and "else:" in text
