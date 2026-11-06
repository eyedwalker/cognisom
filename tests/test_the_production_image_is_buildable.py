"""The two lines that decide whether an image reaches ECR at all.

`build-and-push` is `needs: test`, so for two months a red test job meant
the build was *skipped* and nothing noticed. When the test job finally went
green on 2026-11-06 the build ran for the first time since 2026-09-08 --
and died after 2 h 26 m inside `pip install -r requirements.txt` with
``resolvelib.resolvers.ResolutionTooDeep: 200000``, having walked
scikit-image backwards one release at a time.

The requirement set was never in conflict. The CI test job installs the
same requirements.txt on the same python 3.11 / linux x86_64 in 89 seconds,
because actions/setup-python ships pip 26.2.2 while python:3.11-slim bundles
pip 24.0. So the subject of these tests is the *installer*, not the
dependencies -- the thing no requirements pin can express and no local build
can observe, since neither Docker nor the tensorflow-cpu marker is live on
the dev machine.

The second test guards the trap that fixing the first one armed: an
rdkit-pypi install that could not previously run, because the install above
it always failed first.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

#: Built by .github/workflows/deploy.yml and by deploy/deploy-apps-server.sh.
#: This is the image production runs.
PROD_DOCKERFILE = ROOT / "Dockerfile.prod"

#: The pip that python:3.11-slim bundles, and cannot resolve requirements.txt.
BUNDLED_PIP_THAT_FAILS = (24, 0)

#: Measured 2026-11-06: the version actions/setup-python shipped in the test
#: job that resolved this exact requirements.txt in 89 s. Recorded, not
#: asserted -- the tests below assert the property (pinned, and newer than
#: the resolver that fails), so a later bump does not have to edit a test.
MEASURED_WORKING_PIP = "26.2.2"


def _dockerfiles():
    return sorted(ROOT.glob("Dockerfile*"))


def test_the_production_image_pins_a_pip_that_can_resolve_requirements():
    """A bundled pip 24.0 cannot build this image, and says so after 2.5 h."""
    text = PROD_DOCKERFILE.read_text()

    match = re.search(r"pip install[^\n]*--upgrade\s+pip==(\d+)\.(\d+)", text)
    assert match, (
        "Dockerfile.prod must upgrade pip to a pinned version before "
        "installing requirements.txt. Without it the bundled pip 24.0 "
        "backtracks for 2 h 26 m and fails with ResolutionTooDeep, and no "
        f"image reaches ECR. The measured working pin is {MEASURED_WORKING_PIP}."
    )

    pinned = (int(match.group(1)), int(match.group(2)))
    assert pinned > BUNDLED_PIP_THAT_FAILS, (
        f"pip=={match.group(0)} is not newer than the bundled pip "
        f"{'.'.join(map(str, BUNDLED_PIP_THAT_FAILS))} that fails to resolve "
        "requirements.txt, so pinning it changes nothing"
    )


def test_pip_is_upgraded_before_the_requirements_install_not_after():
    """Ordering is the whole point: an upgrade after the failing install
    never executes."""
    text = PROD_DOCKERFILE.read_text()

    upgrade = text.find("--upgrade pip==")
    install = text.find("-r requirements.txt")

    assert upgrade != -1 and install != -1
    assert upgrade < install, (
        "the pip upgrade appears after `pip install -r requirements.txt`, "
        "so it runs only if the install that needs it already succeeded"
    )


@pytest.mark.parametrize("dockerfile", _dockerfiles(), ids=lambda p: p.name)
def test_no_image_installs_rdkit_pypi_over_the_official_rdkit(dockerfile):
    """rdkit-pypi is not a fallback; it is a four-year-old overwrite.

    requirements.txt declares the official ``rdkit`` distribution (CI
    resolves 2026.9.1). ``rdkit-pypi`` is the unofficial build that stopped
    publishing after 2022.9.5 -- but it still has a cp311 manylinux wheel,
    so `pip install rdkit-pypi` succeeds, and that wheel ships the SAME
    top-level ``rdkit/`` package (verified 2026-11-06 against
    rdkit_pypi-2022.9.5-cp311-cp311-manylinux_2_17_x86_64.whl, whose
    entries are ``rdkit/``, ``rdkit_pypi.libs/`` and the dist-info).

    So it overwrites site-packages/rdkit with 2022.09 code while pip's
    metadata goes on reporting rdkit 2026.9.1 -- an `|| echo` guard makes
    that silent rather than safe. Dockerfile.prod carried exactly this
    line, and it had never run only because the install above it always
    failed first.
    """
    text = dockerfile.read_text()
    offenders = [
        line.strip()
        for line in text.splitlines()
        if "rdkit-pypi" in line and not line.lstrip().startswith("#")
    ]
    assert not offenders, (
        f"{dockerfile.name} installs rdkit-pypi, which overwrites the "
        f"official rdkit package with 2022.09 code: {offenders}"
    )


def test_requirements_declares_the_official_rdkit():
    """The reason no fallback is warranted, stated where it can fail.

    Without this, deleting the rdkit line from requirements.txt would make
    the test above pass while leaving the image with no rdkit at all.
    """
    text = (ROOT / "requirements.txt").read_text()
    declared = [
        line.strip()
        for line in text.splitlines()
        if re.match(r"^\s*rdkit\b", line)
    ]
    assert declared, "requirements.txt no longer declares the official rdkit"
    assert not any("rdkit-pypi" in line for line in declared), (
        "requirements.txt declares rdkit-pypi; the official distribution is "
        "`rdkit`"
    )
