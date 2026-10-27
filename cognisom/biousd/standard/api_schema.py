"""The ``BioCellAPI`` applied API schema: codeless, generated, optional.

Bio-USD 0.1 declared its primvars in prose and in :mod:`.vocabulary`,
and a tool loading a file saw seven custom attributes with no type
information beyond what was authored. A *codeless* API schema
(``skipCodeGeneration = true``) declares them to USD itself with nothing
to compile: ``schema.usda`` is rendered from the vocabulary by
:func:`render_schema_usda`, usdGenSchema turns it into
``plugin/plugInfo.json`` + ``plugin/generatedSchema.usda``
(:mod:`.build_plugin`), and :func:`register_schema` hands that directory
to ``Plug.Registry``. With the plugin loaded, ``primvars:bio:cell:phase``
on an instancer that applies the schema has ``allowedTokens``, a doc
string and ``varying`` as its interpolation fallback; without it the
file is exactly what it was.

What was measured (usd-core 25.11 and 26.11; scripts and logs in the
project's scratch area) and what it decided here:

- **Order.** The schema registry is built once per process, the first
  time anything needs a prim definition -- opening a stage with typed
  prims does it -- and a plugin registered after that is never seen:
  ``RegisterPlugins`` returns the plugin and ``FindAppliedAPIPrimDefinition``
  still answers ``None``. So the package registers at import, and a
  caller must ``import biousd`` before opening a stage, or set
  ``PXR_PLUGINPATH_NAME`` to :func:`plugin_path`. A late registration
  logs a WARNING once rather than failing, because the file is still
  readable and the validator still runs; only the fallbacks are missing.
- **Asking is building.** The first draft asked the registry at import
  whether it knew the schema, which BUILT it, so ``import biousd``
  imposed the same trap on its hosts: a second schema plugin registered
  with Plug after the import was never known (measured: registered
  first, both known; imported first, the other one unknown). Now
  :func:`register_schema` touches Plug only, :data:`schema_registered`
  is that Plug-level fact, and :func:`schema_is_known` asks the registry
  when a caller asks -- that is where the late-registration WARNING is
  logged, once. Not a matter of cost: registering takes about 20 ms
  with the ``pxr`` import it needs and the registry build about 3 ms;
  the build now simply happens where it always did, at the first stage,
  after every plugin the host meant to register.
- **Advisory, not enforced.** ``apiSchemaCanOnlyApplyTo = [PointInstancer]``
  makes ``CanApplyAPI`` say no on an Xform, but ``ApplyAPI`` and
  ``AddAppliedSchema`` apply anyway, with no error; the validator's
  ``schema.misapplied`` is the enforcement.
- **A writer needs no plugin.** ``UsdPrim.AddAppliedSchema("BioCellAPI")``
  authors ``apiSchemas`` whether or not the name is known, which is how
  :mod:`.write` applies it and how a 25.x process writes a file a 26.x
  process reads with the schema.
- **Without the plugin** a file carrying ``apiSchemas`` opens silently;
  ``GetAppliedSchemas()`` drops the unknown name, ``GetPrimTypeInfo().
  GetAppliedAPISchemas()`` keeps it, and ``CanApplyAPI`` raises. The
  reader uses the second and never the third.
- **The reader regression.** With the plugin, a schema-declared primvar
  that nobody authored is *defined* (``HasPrimvar`` is ``IsDefined``),
  so the 0.1 reader saw it, read ``None``, and the validator turned an
  optional primvar's absence into ``primvar.length`` with exit 1 on a
  conforming file. :mod:`.read` now gates on authored primvars.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from .profiles import CAPABILITY_CELLS
from .vocabulary import (
    API_SCHEMA, CARRIER_PRIM_TYPE, CELL_PRIMVARS, LIBRARY_NAME, METERS_PER_UNIT,
    REQUIRED_PRIMVARS, VERSION)

__all__ = [
    "API_SCHEMA", "LIBRARY_NAME", "TF_TYPE", "ENV_NO_PLUGIN", "render_schema_usda",
    "schema_path", "plugin_path", "register_schema", "schema_is_known", "apply_api",
    "applied_api_schemas",
]

log = logging.getLogger(__name__)

#: usdGenSchema names the TfType after the library (proper-cased) and the
#: class; ``plugInfo.json`` declares it and aliases it to the identifier.
TF_TYPE = LIBRARY_NAME[:1].upper() + LIBRARY_NAME[1:] + API_SCHEMA

#: Set to ``1`` to stop ``import biousd`` registering the plugin.
ENV_NO_PLUGIN = "BIOUSD_NO_PLUGIN"

_SCHEMA_FILE = "schema.usda"
_PLUGIN_DIR = "plugin"


def package_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def schema_path() -> str:
    """The checked-in usdGenSchema input, rendered from the vocabulary."""
    return os.path.join(package_dir(), _SCHEMA_FILE)


def plugin_path() -> str:
    """The directory ``Plug.Registry`` (or ``PXR_PLUGINPATH_NAME``) takes.

    Flat layout: ``plugInfo.json`` and ``generatedSchema.usda`` side by
    side, ``Root`` and ``ResourcePath`` both ``"."``. The schema and the
    profiles only: the validators plugin sits in ``biousdValidators/``
    beneath it and is registered separately, where it can load
    (:mod:`.validators`).
    """
    return os.path.join(package_dir(), _PLUGIN_DIR)


# ── rendering ───────────────────────────────────────────────────────


def _usda_string(text: str) -> str:
    """A triple-quoted usda string literal."""
    return '"""' + text.replace("\\", "\\\\").replace('"""', '\\"\\"\\"') + '"""'


def _attribute_block(p) -> str:
    # usdGenSchema keeps only the FIRST sentence of a doc string as the
    # runtime-visible customData ``userDocBrief`` and strips the rest
    # from generatedSchema.usda, so the summary sentence goes first.
    extras: List[str] = []
    if p.units:
        extras.append(f"Units: {p.units}.")
    if p.minimum is not None:
        extras.append(f"Minimum {p.minimum:g}.")
    if p.name == "bio:cell:ontologyTerm":
        extras.append("A CURIE matching ^(CL|NCIT):[0-9A-Za-z]+$, or the empty string.")
    extras.append("Required for Bio-USD conformance." if p.required else
                  "Optional; when present it must have this type and one value per instance.")
    doc = " ".join([p.description] + extras)
    lines = [f"    {p.usd_type} primvars:{p.name} ("]
    if p.tokens:
        lines.append("        allowedTokens = [" + ", ".join(f'"{t}"' for t in p.tokens) + "]")
    lines.append(f"        doc = {_usda_string(doc)}")
    # Declaring the interpolation here makes the schema supply it as the
    # FALLBACK for a prim that applies the API but did not author it. A
    # consumer without the plugin sees no fallback, which is why the
    # writer authors it explicitly on every primvar regardless.
    lines.append(f'        interpolation = "{p.interpolation}"')
    lines.append("    )")
    return "\n".join(lines)


def render_schema_usda() -> str:
    """``schema.usda`` for usdGenSchema, from the vocabulary. Nothing typed twice."""
    class_doc = (
        f"A population of biological cells carried by a {CARRIER_PRIM_TYPE} "
        f"(Bio-USD {VERSION}). Each instance is one cell, and the properties "
        "declared here are the convention's per-cell primvars, one value per "
        "instance (interpolation 'varying'). Required for conformance: "
        + ", ".join(f"primvars:{n}" for n in REQUIRED_PRIMVARS)
        + f". The convention also requires stage metersPerUnit = {METERS_PER_UNIT:g} "
        "and a 'bio:version' entry in the root layer's customLayerData; a schema "
        "cannot express those, so the biousd validator still checks them. "
        "Generated from vocabulary.py; do not edit."
    )
    out = [
        "#usda 1.0",
        "(",
        f'    "Bio-USD {VERSION}: codeless applied API schema. GENERATED from '
        'vocabulary.py by biousd.api_schema.render_schema_usda; do not edit."',
        "    subLayers = [",
        "        @usd/schema.usda@",
        "    ]",
        ")",
        "",
        'over "GLOBAL" (',
        "    customData = {",
        f'        string libraryName = "{LIBRARY_NAME}"',
        "        bool skipCodeGeneration = true",
        "        bool useLiteralIdentifier = true",
        "    }",
        ")",
        "{",
        "}",
        "",
        f'class "{API_SCHEMA}" (',
        "    inherits = </APISchemaBase>",
        f"    doc = {_usda_string(class_doc)}",
        "    customData = {",
        '        token apiSchemaType = "singleApply"',
        f'        token[] apiSchemaCanOnlyApplyTo = ["{CARRIER_PRIM_TYPE}"]',
        # usdGenSchema 26.x copies extraPlugInfo into the plugin's Types
        # entry, where UsdProfiles (26.08+) reads impliesCapabilities.
        # usdGenSchema 25.11 cannot serialise the string[] it becomes
        # ("Object of type StringArray is not JSON serializable"), so
        # build_plugin strips this block from its working copy and sets
        # the key itself; the declaration stays here for anyone who runs
        # usdGenSchema 26.x on this file directly, with the same result.
        "        dictionary extraPlugInfo = {",
        f'            string[] impliesCapabilities = ["{CAPABILITY_CELLS}"]',
        "        }",
        "    }",
        ")",
        "{",
    ]
    out.extend(_attribute_block(p) for p in CELL_PRIMVARS)
    out += ["}", ""]
    return "\n".join(out)


# ── registration ────────────────────────────────────────────────────

_state = {"registered": False, "warned_late": False, "warned_missing": False}


def schema_is_known() -> bool:
    """Whether this process's schema registry defines ``BioCellAPI``.

    Asking BUILDS the registry if nothing has yet, and a registry once
    built never learns a plugin registered later -- so this is not asked
    at import, only when a caller asks. The first ``False`` after a
    successful :func:`register_schema` is the late-registration case and
    logs one WARNING; the answer is still ``False``, because the file is
    readable and the validator runs without the schema.
    """
    try:
        from pxr import Usd
    except ImportError:
        return False
    known = Usd.SchemaRegistry().FindAppliedAPIPrimDefinition(API_SCHEMA) is not None
    if not known and _state["registered"] and not _state["warned_late"]:
        _state["warned_late"] = True
        log.warning(
            "Bio-USD: the %s schema plugin was registered after this process built "
            "its USD schema registry, so %s is unknown here (files still open and "
            "validate; the schema's fallbacks and documentation are missing). Import "
            "biousd, or call biousd.register_schema(), before any stage is opened; "
            "or set PXR_PLUGINPATH_NAME=%s", LIBRARY_NAME, API_SCHEMA, plugin_path())
    return known


def register_schema() -> bool:
    """Hand the plugin to ``Plug.Registry``; ``True`` when Plug has it afterwards.

    Idempotent, about 3 ms, and it never builds the schema registry --
    so a host may register its own schema plugins after ``import biousd``
    and still have them known. ``False``, with one ERROR, when the plugin
    files are missing (a checkout before ``python -m biousd.build_plugin``
    has run), and ``False`` quietly when ``pxr`` is absent. Whether the
    registry then KNOWS the schema is :func:`schema_is_known`'s answer,
    which depends on whether a stage was opened before this call.
    """
    if _state["registered"]:
        return True
    try:
        from pxr import Plug, Tf
    except ImportError:
        log.debug("usd-core is not installed; the %s plugin was not registered", LIBRARY_NAME)
        return False
    try:
        plugins = Plug.Registry().RegisterPlugins(plugin_path())
    except Tf.ErrorException as exc:          # a malformed plugInfo.json
        log.error("the %s plugin at %s did not register: %s", LIBRARY_NAME,
                  plugin_path(), exc)
        return False
    if not plugins and Plug.Registry().GetPluginWithName(LIBRARY_NAME) is None:
        if not _state["warned_missing"]:
            _state["warned_missing"] = True
            log.error("no %s plugin at %s (plugInfo.json missing; run "
                      "python -m biousd.build_plugin)", LIBRARY_NAME, plugin_path())
        return False
    _state["registered"] = True
    return True


def applied_api_schemas(prim) -> List[str]:
    """The API schemas authored on ``prim``, known to the registry or not.

    ``GetAppliedSchemas()`` silently drops a name the registry does not
    define, which is ``BioCellAPI`` in any process without the plugin;
    the prim type info keeps what the file says.
    """
    return [str(s) for s in prim.GetPrimTypeInfo().GetAppliedAPISchemas()]


def apply_api(prim) -> bool:
    """Apply ``BioCellAPI`` to a PointInstancer prim; no plugin needed.

    Uses ``AddAppliedSchema``, which authors ``apiSchemas`` without the
    registry having to know the name. Returns ``False`` without writing
    for an invalid prim or one that is not a PointInstancer, because USD
    would apply it anyway and the validator would then reject the file.
    """
    from pxr import UsdGeom

    if not prim or not prim.IsValid():
        return False
    if not prim.IsA(UsdGeom.PointInstancer):
        log.warning("%s is a %s, not a %s; %s not applied", prim.GetPath(),
                    prim.GetTypeName(), CARRIER_PRIM_TYPE, API_SCHEMA)
        return False
    if API_SCHEMA in applied_api_schemas(prim):
        return True
    return bool(prim.AddAppliedSchema(API_SCHEMA))
