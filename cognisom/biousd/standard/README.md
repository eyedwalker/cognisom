# Bio-USD

**Biological cells on OpenUSD, as a convention anyone can write to.**
Draft v0.1.0 · Apache 2.0 · [SPEC.md](SPEC.md)

A Bio-USD file is an ordinary OpenUSD stage. Cells are instances of a
`PointInstancer`; what each cell *is* rides along as per-instance primvars.
Because it adds no prim types, every OpenUSD tool already reads it —
usdview, Omniverse, Houdini, Blender, Mol* via a loader — and because each
cell may carry a **Cell Ontology** term, an ontology-aware tool can join a
scene to CZ CELLxGENE, the Human Cell Atlas and Tabula Sapiens, which
annotate their 150 million cells with the same identifiers.

```
#usda 1.0
(
    customLayerData = { string "bio:version" = "0.1.0" }
    metersPerUnit = 1e-06
    upAxis = "Y"
)
def PointInstancer "Cells"
{
    point3f[] positions = [(12.4, 8.1, 50.0), (31.0, 9.7, 50.0)]
    int[] protoIndices = [0, 0]
    float3[] scales = [(12, 12, 12), (8, 8, 8)]
    int[] primvars:bio:cell:id = [0, 1]                  (interpolation = "varying")
    token[] primvars:bio:cell:type = ["cancer", "immune"] (interpolation = "varying")
    token[] primvars:bio:cell:ontologyTerm = ["NCIT:C2919", "CL:0000625"]
}
```

That is the whole idea. Two primvars are required — an id and a coarse
type — and five more are typed when present: ontology term, cycle phase,
age, viability, relative volume. Positions and diameters are in
micrometres. [SPEC.md](SPEC.md) is the normative list and is generated from
[`vocabulary.py`](vocabulary.py), so the code and the prose cannot drift.

## Write one, check one

```python
import biousd                     # or: from cognisom.biousd import standard as biousd

biousd.write_cells("scene.usdc", [
    {"position": (12.4, 8.1, 50.0), "cell_type": "cancer",
     "ontology_term": "NCIT:C2919", "scale_um": 12.0, "phase": "S"},
    {"position": (31.0, 9.7, 50.0), "cell_type": "immune",
     "ontology_term": "CL:0000625", "scale_um": 8.0},
])
table = biousd.read_cells("scene.usdc")
print(table.n, table["bio:cell:ontologyTerm"])
```

```
$ python -m biousd.validate scene.usdc
scene.usdc: CONFORMS to Bio-USD 0.1.0 -- 1 population(s), 2 cells, 0 error(s), 0 warning(s)
```

The validator checks exactly what the spec says and nothing else; its exit
status is 0, 1 or 2, so it drops into CI.

## Status, honestly

- **Draft.** One implementation writes it (Cognisom, a prostate-cancer
  digital twin); nobody else does yet. It becomes a standard when someone
  else writes to it, not before.
- **The ontology join is the point.** `bio:cell:type` exists for renderers
  that only have a token; `bio:cell:ontologyTerm` is what makes a file
  mean something outside the tool that wrote it. Terms are resolved by
  exact label match only — a fuzzy search returned four wrong tissues
  with full confidence.
- **No prim types.** A codeless USD schema may follow; nothing here
  requires one, and that is deliberate: it is why the files open everywhere.
- **Not a simulation format.** It says what cells are and where, not how
  they got there. It is meant to sit beside MultiCellDS, the community
  standard for multicellular data, not replace it; the ontology term is
  the join between the two.
- **Prototypes are free-form.** The minimal writer draws spheres; a tool
  with real morphology authors its own prototypes at unit diameter and
  sets `protoIndices`. The instance scale is then the cell's diameter.

## Examples

`examples/` holds three files written by the reference writer, with the
script that made them: a 200-cell scene as readable `.usda`, a 10,000-cell
tissue slab, and a 2,000-cell scene coloured by a per-cell field.

## Feedback

Open an issue, or write a file and run the validator on it. The questions
worth answering first: is the required set right, is the coarse type list
useful or a nuisance, and what is missing for your tool to read it.

## Citation

See [CITATION.cff](CITATION.cff). Please cite the version you wrote to.
