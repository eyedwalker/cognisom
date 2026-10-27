"""Regenerate ``schema.usda`` and ``plugin/`` from the vocabulary.

    python -m biousd.build_plugin            # rewrite the package's copies
    python -m biousd.build_plugin --check    # exit 1 if they are stale

Four files are generated, and a test holds the checked-in copies
byte-identical to a fresh build:

    schema.usda                               <- api_schema.render_schema_usda()
    plugin/generatedSchema.usda               <- usdGenSchema (codeless)
    plugin/plugInfo.json                      <- usdGenSchema, then this module
    plugin/biousdValidators/plugInfo.json     <- validators.render_plug_info()

usdGenSchema, in codeless mode, needs two files the usd-core 25.11 wheel
does not ship (measured: the wheel has no ``usd/schema.usda`` source and
no ``codegenTemplates``; the 26.x wheels have both):

- ``usd/schema.usda``, which defines ``/APISchemaBase``, the class this
  schema inherits. usdGenSchema resolves ``@usd/schema.usda@`` through
  ``ArDefaultResolver`` search paths that it sets itself from the
  registered schema plugins' resource directories -- which is why
  ``PXR_AR_DEFAULT_SEARCH_PATH`` does not help (``SetDefaultSearchPath``
  overwrites it, verified) -- so instead the relative identifier is
  anchored: a ``usd/`` directory beside the schema in the build
  directory, holding a copy of the file, resolves on every wheel.
- ``codegenTemplates/plugInfo.json``, the template the plugin file is
  rendered from; passed with ``-t``.

Both are vendored under ``vendor/openusd/`` from OpenUSD v25.11 (see
NOTICE) and used when the wheel lacks them. The 26.8 and 26.11 wheels
produced a byte-identical ``generatedSchema.usda`` from the same input.

What usdGenSchema writes is then finished here, as CMake's
``_plugInfo_subst`` would in an OpenUSD build: the three ``@PLUG_INFO_*@``
placeholders become ``"."``, ``"."`` and ``""`` (flat layout, plugInfo
beside the schema, no library), the three leading ``#`` comment lines go
(the wheel's own plugInfo files carry none; the registry tolerates them,
strict JSON readers do not), and the ``Info.Profiles`` block comes from
:mod:`.profiles`. The result is re-serialised with sorted keys so that
two wheels' templates cannot differ in the output.

The validators plugin is deliberately NOT pulled in from here. The first
build gave this file ``"Includes": ["biousdValidators/"]``, so every
process that registered the schema -- ``import biousd``, or
``PXR_PLUGINPATH_NAME`` in a tool nobody controls -- advertised the
``"Type": "python"`` validators too, and the first framework client to
run all validators raised ``Tf.ErrorException`` (four ``PlugPlugin::_Load
... failed``, ``ModuleNotFoundError: biousdValidators``) on both wheels
unless ``plugin/`` happened to be on ``PYTHONPATH``. The schema plugin
now stands alone, and :mod:`.validators` registers its own directory
with Plug in-process, where it can also make the import resolve.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple

from .api_schema import LIBRARY_NAME, TF_TYPE, package_dir, render_schema_usda
from .profiles import CAPABILITY_CELLS, render_plug_info_profiles
from .validators import PLUGIN_NAME as VALIDATORS_PLUGIN, render_plug_info

#: The files this module owns, relative to the package directory.
GENERATED: Tuple[str, ...] = (
    "schema.usda",
    os.path.join("plugin", "plugInfo.json"),
    os.path.join("plugin", "generatedSchema.usda"),
    os.path.join("plugin", VALIDATORS_PLUGIN, "plugInfo.json"),
)


class BuildError(RuntimeError):
    """usdGenSchema could not run or did not produce what this expects."""


def vendored_paths() -> Tuple[str, str]:
    """(dir holding ``schema.usda``, codegenTemplates dir) from ``vendor/openusd``."""
    root = os.path.join(package_dir(), "vendor", "openusd")
    return os.path.join(root, "usd"), os.path.join(root, "codegenTemplates")


def wheel_paths() -> Tuple[Optional[str], Optional[str]]:
    """The same two, from the installed ``usd`` plugin's resources, where it ships them."""
    try:
        from pxr import Plug
    except ImportError:
        return None, None
    plugin = Plug.Registry().GetPluginWithName("usd")
    if plugin is None:
        return None, None
    res = plugin.resourcePath
    usd = os.path.join(res, "usd")
    templates = os.path.join(res, "codegenTemplates")
    return (usd if os.path.isfile(os.path.join(usd, "schema.usda")) else None,
            templates if os.path.isfile(os.path.join(templates, "plugInfo.json")) else None)


def _finish_plug_info(text: str) -> str:
    lines = [l for l in text.splitlines() if not l.lstrip().startswith("#")]
    try:
        info = json.loads("\n".join(lines))
    except ValueError as exc:
        raise BuildError(f"usdGenSchema's plugInfo.json is not JSON: {exc}") from exc
    plugins = info.get("Plugins") or []
    ours = [p for p in plugins if p.get("Name") == LIBRARY_NAME]
    if len(ours) != 1:
        raise BuildError(f"expected one plugin named {LIBRARY_NAME!r}, found {len(ours)}")
    plugin = ours[0]
    plugin["Root"] = "."
    plugin["ResourcePath"] = "."
    plugin["LibraryPath"] = ""
    if plugin.get("Type") != "resource":
        raise BuildError(f"plugin Type is {plugin.get('Type')!r}, not 'resource': the "
                         "schema was not generated codeless")
    types = plugin.get("Info", {}).get("Types", {})
    if TF_TYPE not in types:
        raise BuildError(f"Types has no {TF_TYPE}: {sorted(types)}")
    # What schema.usda's extraPlugInfo declares, set here because the
    # block is stripped before usdGenSchema runs (see _without_extra_plug_info).
    implied = types[TF_TYPE].get("impliesCapabilities", [CAPABILITY_CELLS])
    if list(implied) != [CAPABILITY_CELLS]:
        raise BuildError(f"impliesCapabilities came out as {implied!r}")
    types[TF_TYPE]["impliesCapabilities"] = [CAPABILITY_CELLS]
    plugin["Info"]["Profiles"] = render_plug_info_profiles()
    # No "Includes": the validators plugin is registered by .validators,
    # in-process and only where it can load (see the module docstring).
    info.pop("Includes", None)
    return json.dumps(info, indent=4, sort_keys=True) + "\n"


_EXTRA_PLUG_INFO = re.compile(r"\n[ \t]*dictionary extraPlugInfo = \{.*?\n[ \t]*\}", re.S)


def _without_extra_plug_info(schema_text: str) -> str:
    """The schema as usdGenSchema 25.11 can take it.

    The ``string[] impliesCapabilities`` in ``extraPlugInfo`` reaches
    usdGenSchema as a ``Vt.StringArray``, which 25.11's ``json.dumps``
    refuses ("Object of type StringArray is not JSON serializable");
    26.x converts it. The block is removed from the working copy on every
    wheel and :func:`_finish_plug_info` sets the key, so the output is
    the same whichever usdGenSchema built it.
    """
    stripped, n = _EXTRA_PLUG_INFO.subn("", schema_text)
    if n != 1:
        raise BuildError(f"expected one extraPlugInfo block in schema.usda, found {n}")
    return stripped


def run_usd_gen_schema(schema_text: str, *, verbose: bool = False) -> Tuple[str, str]:
    """(plugInfo.json text, generatedSchema.usda text) from usdGenSchema, in a temp dir."""
    usd_dir, templates = wheel_paths()
    vendored_usd, vendored_templates = vendored_paths()
    usd_dir = usd_dir or vendored_usd
    templates = templates or vendored_templates
    work = tempfile.mkdtemp(prefix="biousd-build-")
    try:
        with open(os.path.join(work, "schema.usda"), "w") as fh:
            fh.write(_without_extra_plug_info(schema_text))
        os.mkdir(os.path.join(work, "usd"))
        shutil.copyfile(os.path.join(usd_dir, "schema.usda"),
                        os.path.join(work, "usd", "schema.usda"))
        out = os.path.join(work, "plugin")
        cmd = [sys.executable, "-m", "pxr.Usd.usdGenSchema", "-t", templates,
               "schema.usda", out]
        run = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
        if verbose:
            print(run.stdout, end="")
            print(run.stderr, end="", file=sys.stderr)
        if run.returncode != 0:
            raise BuildError(f"usdGenSchema exited {run.returncode}:\n{run.stdout}{run.stderr}")
        try:
            with open(os.path.join(out, "plugInfo.json")) as fh:
                plug_info = fh.read()
            with open(os.path.join(out, "generatedSchema.usda")) as fh:
                generated = fh.read()
        except OSError as exc:
            raise BuildError(f"usdGenSchema wrote no plugin: {exc}\n{run.stdout}") from exc
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return plug_info, generated


def build(out_dir: Optional[str] = None, *, verbose: bool = False) -> Dict[str, str]:
    """Write the four generated files under ``out_dir`` (default: the package).

    Returns ``{relative path: text}``. Raises :class:`BuildError` when
    usdGenSchema cannot run (for example without ``jinja2``, which it
    imports) or produces something other than the codeless plugin this
    package ships.
    """
    out_dir = out_dir or package_dir()
    schema_text = render_schema_usda()
    plug_info, generated = run_usd_gen_schema(schema_text, verbose=verbose)
    files = {
        GENERATED[0]: schema_text,
        GENERATED[1]: _finish_plug_info(plug_info),
        GENERATED[2]: generated,
        GENERATED[3]: render_plug_info(),
    }
    for rel, text in files.items():
        path = os.path.join(out_dir, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(text)
    return files


def stale(verbose: bool = False) -> List[str]:
    """The generated files whose checked-in copy differs from a fresh build."""
    fresh = build(tempfile.mkdtemp(prefix="biousd-check-"), verbose=verbose)
    out = []
    for rel, text in fresh.items():
        path = os.path.join(package_dir(), rel)
        try:
            with open(path) as fh:
                current = fh.read()
        except OSError:
            current = None
        if current != text:
            out.append(rel)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ("-h", "--help"):
        print(__doc__)
        return 2
    verbose = "--verbose" in args
    try:
        if "--check" in args:
            bad = stale(verbose=verbose)
            if bad:
                print("stale: " + ", ".join(bad) + " (run python -m biousd.build_plugin)")
                return 1
            print("plugin is up to date with vocabulary.py")
            return 0
        for rel in build(verbose=verbose):
            print(f"wrote {os.path.join(package_dir(), rel)}")
        return 0
    except BuildError as exc:
        print(f"build failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":                        # pragma: no cover
    raise SystemExit(main())
