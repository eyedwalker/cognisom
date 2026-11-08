"""Bio-USD: the standard, its code, and how to use it. No login.

Why the specification is not typed into this page
-------------------------------------------------
Bio-USD's normative text is generated from ``vocabulary.py``
(``render_spec_markdown``), the same module the writer and the validator
read. This page renders that function's output, so the specification a
visitor reads here cannot disagree with the code that checks files.

Why the example runs when the page loads
----------------------------------------
The first code block on this page is executed against the package this
deployment ships, and the validator's real verdict is printed under it.
If the example ever stops working, the page shows the failure instead of
a transcript of an old success.

Like the demo page it calls no gate, so it must stay read-only: the only
thing it writes is the example file, into a temporary directory that is
deleted at once.
"""

import tempfile
from pathlib import Path

import streamlit as st

from cognisom.biousd import standard as biousd
from cognisom.dashboard.page_config import safe_set_page_config

safe_set_page_config(page_title="Cognisom | Bio-USD",
                     page_icon="\U0001f9ec", layout="wide")

REPO = "https://github.com/eyedwalker/biousd"
CONCEPT_DOI = "10.5281/zenodo.23603569"
VERSION_DOI = "10.5281/zenodo.23700359"
INSTALL = f'pip install "git+{REPO}@v{biousd.VERSION}"'

_EXAMPLES = Path(biousd.__file__).resolve().parent / "examples"

# The exact code shown in the first example, executed below.
WRITE_EXAMPLE = '''\
import biousd

biousd.write_cells("scene.usdc", [
    {"position": (12.4, 8.1, 50.0), "cell_type": "cancer",
     "ontology_term": "NCIT:C2919", "scale_um": 12.0, "phase": "S"},
    {"position": (31.0, 9.7, 50.0), "cell_type": "immune",
     "ontology_term": "CL:0000625", "scale_um": 8.0},
])

table = biousd.read_cells("scene.usdc")
print(table.n, table["bio:cell:ontologyTerm"])
'''

MINIMAL_USDA = '''\
#usda 1.0
(
    customLayerData = { string "bio:version" = "0.2.0" }
    defaultPrim = "Cells"
    metersPerUnit = 1e-06
    upAxis = "Y"
)
def PointInstancer "Cells" (
    prepend apiSchemas = ["BioCellAPI"]            # optional, new in 0.2.0
)
{
    def Sphere "Prototype"                         # unit diameter: the instance scale is the cell's
    {
        double radius = 0.5
    }
    rel prototypes = </Cells/Prototype>
    point3f[] positions = [(12.4, 8.1, 50.0), (31.0, 9.7, 50.0)]
    int[] protoIndices = [0, 0]
    float3[] scales = [(12, 12, 12), (8, 8, 8)]
    int[] primvars:bio:cell:id = [0, 1] (interpolation = "varying")
    token[] primvars:bio:cell:type = ["cancer", "immune"] (interpolation = "varying")
    token[] primvars:bio:cell:ontologyTerm = ["NCIT:C2919", "CL:0000625"] (interpolation = "varying")
}
'''

PLAIN_USD_EXAMPLE = '''\
from pxr import Usd, UsdGeom          # OpenUSD only; no Bio-USD code

stage = Usd.Stage.Open("scene.usdc")
for prim in stage.Traverse():
    if prim.IsA(UsdGeom.PointInstancer):
        cells = UsdGeom.PointInstancer(prim)
        primvars = UsdGeom.PrimvarsAPI(prim)
        print(primvars.GetPrimvar("bio:cell:type").Get())
        print(primvars.GetPrimvar("bio:cell:ontologyTerm").Get())
        print(cells.GetPositionsAttr().Get())       # micrometres
        print(cells.GetScalesAttr().Get())          # diameters, micrometres
'''

COGNISOM_EXAMPLE = '''\
from cognisom.dashboard.engine_runner import EngineRunner
from cognisom.visualization.run_usd import write_run

runner = EngineRunner(scenario="Baseline", duration=2.0)
runner.build()
runner.run()

# One PointInstancer, time-sampled: positions, sizes, lineage, cell-cycle
# phase and viability for every cell at every recorded frame.
write_run("run.usdc", runner)
'''


@st.cache_data(show_spinner=False)
def _run_the_example(code: str, version: str):
    """Execute the first example and validate what it wrote.

    ``version`` is part of the cache key so a new package version re-runs.
    Returns (printed line, validator verdict, error or None).
    """
    import contextlib
    import io
    import os

    out = io.StringIO()
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        try:
            os.chdir(tmp)
            # The example says `import biousd`. Inside Cognisom the package
            # is cognisom.biousd.standard, so the namespace is handed it
            # under that name and the one import line is dropped -- rather
            # than aliasing it in sys.modules, which would change what
            # every later `import biousd` in this server process gets.
            assert code.startswith("import biousd\n"), code[:40]
            body = code[len("import biousd\n"):]
            namespace = {"__name__": "__biousd_example__", "biousd": biousd}
            with contextlib.redirect_stdout(out):
                exec(compile(body, "<example>", "exec"), namespace)
            report = biousd.validate("scene.usdc")
            return out.getvalue().strip(), report.describe().replace(
                tmp + os.sep, ""), None
        except Exception as exc:                            # noqa: BLE001
            return out.getvalue().strip(), "", f"{type(exc).__name__}: {exc}"
        finally:
            os.chdir(cwd)


# ── Header ───────────────────────────────────────────────────────────

st.title("Bio-USD")
st.markdown(
    f"**Biological cells on OpenUSD, as a convention anyone can write to.** "
    f"Draft {biousd.VERSION} · Apache 2.0 · "
    f"[source]({REPO}) · DOI [{CONCEPT_DOI}](https://doi.org/{CONCEPT_DOI})"
)
st.markdown(
    "A Bio-USD file is an ordinary OpenUSD stage. Cells are instances of a "
    "`PointInstancer`, and what each cell *is* rides along as per-cell "
    "primvars. It adds no prim types, so every OpenUSD tool already opens "
    "it — usdview, NVIDIA Omniverse and Isaac Sim, Houdini, Blender. And "
    "because each cell can carry a **Cell Ontology** term, a scene can be "
    "joined to CZ CELLxGENE, the Human Cell Atlas and Tabula Sapiens, which "
    "annotate their cells with the same identifiers."
)
st.caption(
    "It is the format Cognisom writes its simulations in: the ray-traced "
    "video on the [roadmap page](/?page=roadmap) was rendered from one."
)

st.divider()

# ── The whole idea ───────────────────────────────────────────────────

st.subheader("The whole idea, in one file")
st.markdown(
    "Two primvars are required — an id and a coarse type. Five more are "
    "typed when present: ontology term, cell-cycle phase, age, viability "
    "and relative volume. Positions and diameters are in micrometres. This "
    "block is a conforming file as it stands."
)
st.code(MINIMAL_USDA, language="text")

# ── Install ──────────────────────────────────────────────────────────

st.subheader("Install")
st.code(INSTALL, language="bash")
st.caption(
    "Python 3.10 or later; it brings `usd-core` with it. Not yet on PyPI, "
    "so install from the tagged release on GitHub."
)

# ── Write one, read one ──────────────────────────────────────────────

st.subheader("Write a file, read it back")
st.code(WRITE_EXAMPLE, language="python")

printed, verdict, error = _run_the_example(WRITE_EXAMPLE, biousd.VERSION)
if error:
    st.error(
        "This example failed when the page ran it, so it is shown failing "
        f"rather than as an old success: `{error}`"
    )
else:
    st.markdown("**Run on this server when the page loaded:**")
    st.code(f"{printed}\n\n$ biousd-validate scene.usdc\n{verdict}",
            language="text")

# ── Any OpenUSD reader ───────────────────────────────────────────────

st.subheader("Read it with plain OpenUSD")
st.markdown(
    "Nothing about the format needs Bio-USD's own code to read. This is "
    "OpenUSD's Python API alone:"
)
st.code(PLAIN_USD_EXAMPLE, language="python")

# ── Validate ─────────────────────────────────────────────────────────

st.subheader("Check a file")
st.code(
    "$ biousd-validate scene.usdc\n"
    f"scene.usdc: CONFORMS to Bio-USD {biousd.VERSION} -- 1 population(s), "
    "2 cells, 0 error(s), 0 warning(s)\n\n"
    "$ biousd-check scene.usdc      # OpenUSD's validation framework, then Bio-USD\n",
    language="bash",
)
st.markdown(
    "`biousd-validate` checks exactly what the specification says and "
    "nothing else. `biousd-check` runs OpenUSD's own validation framework "
    "first and gives one report. Both exit 0 (conforms), 1 (does not) or "
    "2 (could not be read), so they drop into CI. Every finding has a "
    "code, listed in the specification below."
)

# ── Ontology terms ───────────────────────────────────────────────────

st.subheader("Which ontology term?")
st.code(
    "from biousd.terms import curie_for\n\n"
    "curie_for(\"Macrophage M1\")   # 'CL:0000863'\n"
    "curie_for(\"Macrophage M2\")   # 'CL:0000890'\n"
    "curie_for(\"blancmange\")      # None -- an unknown name is not guessed\n",
    language="python",
)
st.markdown(
    "A starter map from 43 cell identities to Cell Ontology and NCI "
    "Thesaurus terms, every one resolved by **exact** label match. A fuzzy "
    "search returned four wrong tissues with full confidence, which is why "
    "that rule exists."
)

# ── From other tools ─────────────────────────────────────────────────

st.subheader("From PhysiCell")
st.code(
    f'pip install "biousd[physicell] @ git+{REPO}@v{biousd.VERSION}"\n'
    "biousd-from-physicell output00000001.xml output00000001.usdc "
    "--type 0=normal --type 1=cancer",
    language="bash",
)
st.markdown(
    "A PhysiCell / MultiCellDS snapshot converts directly; cell-type names "
    "are not in PhysiCell's output, so you supply them. Bio-USD is meant to "
    "sit beside MultiCellDS, not replace it: the ontology term is the join "
    "between the two."
)

st.subheader("From a Cognisom simulation")
st.code(COGNISOM_EXAMPLE, language="python")

# ── Examples to download ─────────────────────────────────────────────

st.subheader("Example files")
_FILES = (
    ("minimal_200_cells.usda", "200 cells, human-readable text"),
    ("tissue_slab_10k_cells.usdc", "a 10,000-cell tissue slab"),
    ("coloured_by_age_2k_cells.usdc", "2,000 cells coloured by a per-cell field"),
)
cols = st.columns(len(_FILES))
for col, (name, what) in zip(cols, _FILES):
    path = _EXAMPLES / name
    with col:
        if path.exists():
            st.download_button(
                f"{name}", data=path.read_bytes(), file_name=name,
                mime="application/octet-stream", use_container_width=True)
            st.caption(f"{what} · {path.stat().st_size / 1024:,.0f} KB")
st.caption("All three were written by the reference writer, apply the "
           "schema, and pass the validator. Open them in usdview, "
           "Omniverse or Blender.")

st.divider()

# ── The specification ────────────────────────────────────────────────

st.subheader(f"The specification, version {biousd.VERSION}")
st.caption(
    "Rendered from `vocabulary.py`, the module the writer and the validator "
    "read — so this text cannot disagree with the code that checks files."
)
with st.expander("Read the full specification", expanded=False):
    st.markdown(biousd.render_spec_markdown())

# ── Status ───────────────────────────────────────────────────────────

st.subheader("Status, honestly")
st.markdown(
    """
- **Draft.** One implementation writes it today: Cognisom. It becomes a
  standard when someone else writes to it, not before.
- **The ontology join is the point.** `bio:cell:type` exists for renderers
  that only want a label; `bio:cell:ontologyTerm` is what makes a file mean
  something outside the tool that wrote it.
- **Not a simulation format.** It says what cells are and where, not how
  they got there.
- **The schema is optional.** `BioCellAPI` is a codeless applied API schema
  that documents the primvars to tools that load it. A file is conforming
  with or without it, which is why the files open everywhere.
"""
)
st.markdown(
    f"**Feedback:** open an issue at [{REPO.removeprefix('https://')}]({REPO}), "
    "or write a file and run the validator on it. The questions worth "
    "answering first: is the required set right, is the coarse type list "
    "useful or a nuisance, and what is missing for your tool to read it?"
)
st.markdown(
    f"**Cite:** Bio-USD, DOI [{CONCEPT_DOI}](https://doi.org/{CONCEPT_DOI}) "
    f"(all versions) or [{VERSION_DOI}](https://doi.org/{VERSION_DOI}) "
    f"(version {biousd.VERSION}). Please cite the version you wrote to."
)
