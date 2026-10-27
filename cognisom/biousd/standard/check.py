"""biousd-check: OpenUSD's validation framework and Bio-USD's checks, one report.

    biousd-check [--strict] [--keywords K1,K2] scene.usdc ...
    python -m biousd.check scene.usdc

A usdchecker-style command that works on a plain ``pip install usd-core``
(24.11 or later binds ``pxr.UsdValidation``; 25.11 and 26.11 measured):

1. every validator the ``UsdValidationRegistry`` knows in this process,
   minus the usdz-only ones unless the file is a package -- the selection
   ``usdchecker`` makes (pxr/usdValidation/bin/usdchecker/usdchecker.cpp)
   -- through one ``ValidationContext``. On usd-core 26.05 or later that
   includes Bio-USD's own validators (:mod:`.validators`), registered
   here first; on earlier wheels it cannot, and the next step is where
   the same checks run;
2. :func:`~.validate.validate`, the Bio-USD conformance checks.

Output is one line per finding, so a CI job can grep it. Exit status: 0
clean, 1 any error (or any warning with ``--strict``), 2 the stage could
not be opened. Apple's bundled ``usdchecker`` (0.25.2) loads no external
plugin at all, so it is a "still opens cleanly" check and nothing more;
this command is how the Bio-USD checks reach a framework report on every
platform.

WHAT A WHEEL DOES NOT SHIP
--------------------------
The usd-core 25.11 wheel ships no shader definitions (no ``usdShaders``
plugin, no ``shaderDefs.usda`` anywhere under ``pxr``), so the first
``usdShadeValidators:*`` validator to ask Sdr for one raised
``Tf.ErrorException`` out of ``ValidationContext.Validate`` -- a
traceback instead of an exit status for every stage holding a
``UsdShade.Shader``, which is every stage Cognisom writes (measured:
21 validators raise, the same context minus ``usdShadeValidators:*``
returns cleanly). And the discovery fails only ONCE per process: a
second ``check_file`` in the same process raised nothing and instead
reported ``MissingShaderIdInRegistry`` for every standard shader, which
is the wheel's defect, not the file's. So the question is asked up
front, once, of Sdr itself (:func:`sdr_resolves_standard_shaders`:
can it resolve ``UsdPreviewSurface``? ``None`` on 25.11, a node on
26.11), and where it cannot, the Sdr-dependent validators are left out
of every pass and the report says so; 26.11 ships the definitions and
runs them all. The probe makes the 25.11 discovery plugin print one
``Coding Error (secondary thread)`` line to stderr; that is the wheel's
complaint about itself, once per process, and not an exit status.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .validate import Report, validate
from .validators import PLUGIN_NAME, register_validators, registration_available
from .vocabulary import VERSION

USDZ_KEYWORD = "UsdzValidators"
#: The validators that need Sdr shader definitions, which a wheel may lack.
SDR_DEPENDENT_PREFIX = "usdShadeValidators:"
#: The shader every wheel with definitions resolves; the probe asks for it.
STANDARD_SHADER = "UsdPreviewSurface"

_sdr_state: dict = {}


def sdr_resolves_standard_shaders() -> bool:
    """Whether this process's Sdr registry resolves ``UsdPreviewSurface``.

    Asked once per process and cached, because the discovery that fails
    on a wheel without ``shaderDefs.usda`` fails loudly only the first
    time. ``False`` on the usd-core 25.11 wheel, ``True`` on 26.11.
    """
    if "ok" not in _sdr_state:
        try:
            from pxr import Sdr
            node = Sdr.Registry().GetShaderNodeByIdentifier(STANDARD_SHADER)
            _sdr_state["ok"] = node is not None
        except Exception:                          # noqa: BLE001 - Tf errors, no Sdr
            _sdr_state["ok"] = False
    return _sdr_state["ok"]


@dataclass
class Finding:
    level: str              # "error" | "warning" | "info"
    identifier: str         # e.g. usdValidation:StageMetadataChecker.MissingDefaultPrim
    message: str
    sites: List[str] = field(default_factory=list)


@dataclass
class CheckResult:
    path: str
    status: int = 0                          # 0 / 1 / 2
    framework_validators: int = 0            # how many ran
    framework: List[Finding] = field(default_factory=list)
    #: Bio-USD's own validators were among the framework's
    biousd_in_framework: bool = False
    #: when they were not, why (the wheel, usually)
    biousd_framework_reason: str = ""
    #: why the framework pass did not run at all
    framework_note: str = ""
    #: validators dropped from the pass because they raised, and why
    framework_skipped: List[str] = field(default_factory=list)
    framework_skip_reason: str = ""
    report: Optional[Report] = None
    error: str = ""                          # why the stage could not be opened

    @property
    def framework_codes(self) -> List[str]:
        """The Bio-USD codes behind the framework's ``biousdValidators`` findings."""
        from .validators import code_for

        return [code_for(f.identifier) for f in self.framework
                if f.identifier.startswith(PLUGIN_NAME + ":")]


def _site_text(site) -> str:
    if site.GetProperty():
        return str(site.GetProperty().GetPath())
    if site.GetPrim():
        return str(site.GetPrim().GetPath())
    layer = site.GetLayer()
    return layer.identifier if layer else "(stage)"


def _framework_findings(stage, path: str, keywords: Optional[List[str]], result: CheckResult):
    """Fill ``result``'s framework fields: findings, count, what was skipped and why."""
    try:
        from pxr import Tf, UsdValidation
    except ImportError:
        result.framework_note = ("pxr.UsdValidation is not available (usd-core 24.11 or "
                                 "later binds it)")
        return
    reg = UsdValidation.ValidationRegistry()
    if keywords:
        metadata = list(reg.GetValidatorMetadataForKeywords(keywords))
    else:
        metadata = list(reg.GetAllValidatorMetadata())
        if not path.lower().endswith(".usdz"):
            metadata = [m for m in metadata if USDZ_KEYWORD not in m.GetKeywords()]
    if not registration_available()[0]:
        # Advertised (by the env-var route) but unloadable here; the
        # registry would skip them silently and the count would lie.
        metadata = [m for m in metadata if not m.name.startswith(PLUGIN_NAME + ":")]
    if not sdr_resolves_standard_shaders():
        # No shader definitions on this wheel (see the module docstring):
        # the Sdr-dependent validators would raise, or report every
        # standard shader as unknown. Leave them out and say so.
        sdr = [m for m in metadata if m.name.startswith(SDR_DEPENDENT_PREFIX)]
        metadata = [m for m in metadata if not m.name.startswith(SDR_DEPENDENT_PREFIX)]
        result.framework_skipped = sorted(m.name for m in sdr)
        if sdr:
            result.framework_skip_reason = (
                f"{SDR_DEPENDENT_PREFIX}* need shader definitions (shaderDefs.usda) this "
                f"wheel does not ship -- Sdr cannot resolve {STANDARD_SHADER}")
    levels = {UsdValidation.ValidationErrorType.Error: "error",
              UsdValidation.ValidationErrorType.Warn: "warning",
              UsdValidation.ValidationErrorType.Info: "info"}
    try:
        errors = UsdValidation.ValidationContext(metadata, True).Validate(stage)
    except Tf.ErrorException as exc:
        # Not the file's fault; the pass is reported as not run, with the
        # reason, and the exit status comes from the Bio-USD pass alone.
        first = str(exc).strip().splitlines()[0] if str(exc).strip() else repr(exc)
        result.framework_note = f"ValidationContext raised: {first[:200]}"
        return
    result.framework = [Finding(levels.get(err.GetType(), str(err.GetType())),
                                str(err.GetIdentifier()), err.GetMessage(),
                                [_site_text(s) for s in err.GetSites()])
                        for err in errors]
    result.framework_validators = len(metadata)


def check_file(path: str, *, strict: bool = False,
               keywords: Optional[List[str]] = None) -> CheckResult:
    """Run both validators on one file. Does not print."""
    from pxr import Usd

    result = CheckResult(path=path)
    try:
        stage = Usd.Stage.Open(path)
        if not stage:
            raise RuntimeError("Usd.Stage.Open returned null")
    except Exception as exc:                   # noqa: BLE001 - reported
        result.status, result.error = 2, str(exc)
        return result

    result.biousd_in_framework = register_validators()
    if not result.biousd_in_framework:
        result.biousd_framework_reason = (
            registration_available()[1] or "register_validators() did not register them")
    _framework_findings(stage, path, keywords, result)
    for finding in result.framework:
        if finding.level == "error" or (strict and finding.level == "warning"):
            result.status = max(result.status, 1)

    result.report = validate(path)
    if any(i.code == "stage.open" for i in result.report.issues):
        result.status = 2
    elif not result.report.ok or (strict and result.report.issues):
        result.status = max(result.status, 1)
    return result


def describe(result: CheckResult) -> str:
    lines = [result.path]
    if result.error:
        lines.append(f"  could not open the stage: {result.error}")
        lines.append("  => COULD NOT BE READ")
        return "\n".join(lines)
    if result.framework_note:
        lines.append(f"  OpenUSD validation framework: not run ({result.framework_note})")
    else:
        lines.append(f"  OpenUSD validation framework: {result.framework_validators} "
                     f"validator(s), {len(result.framework)} finding(s)"
                     + (f", {PLUGIN_NAME} among them" if result.biousd_in_framework else ""))
        if result.framework_skipped:
            lines.append(f"    skipped {len(result.framework_skipped)} validator(s): "
                         f"{result.framework_skip_reason}")
        if not result.biousd_in_framework:
            lines.append(f"    {PLUGIN_NAME} not in the framework: "
                         f"{result.biousd_framework_reason}")
    for f in result.framework:
        lines.append(f"    {f.level.upper():7s} {f.identifier:60s} {f.message}  "
                     f"[{', '.join(f.sites)}]")
    report = result.report
    assert report is not None
    lines.append(f"  Bio-USD {VERSION}: {report.populations} population(s), "
                 f"{report.cells:,} cells, {len(report.errors)} error(s), "
                 f"{len(report.issues) - len(report.errors)} warning(s)"
                 + (" (the same checks as the framework's biousdValidators above)"
                    if result.biousd_in_framework else ""))
    for issue in report.issues:
        lines.append(f"    {issue.level.upper():7s} biousd:{issue.code:52s} {issue.message}")
    verdict = {0: "CONFORMS", 1: "DOES NOT CONFORM", 2: "COULD NOT BE READ"}[result.status]
    lines.append(f"  => {verdict}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 2
    strict = "--strict" in args
    keywords: Optional[List[str]] = None
    paths: List[str] = []
    it = iter(a for a in args if a != "--strict")
    for a in it:
        if a == "--keywords":
            keywords = next(it, "").split(",")
        else:
            paths.append(a)
    if not paths:
        print(__doc__)
        return 2
    worst = 0
    for path in paths:
        result = check_file(path, strict=strict, keywords=keywords)
        print(describe(result))
        worst = max(worst, result.status)
    return worst


if __name__ == "__main__":                        # pragma: no cover
    raise SystemExit(main())
