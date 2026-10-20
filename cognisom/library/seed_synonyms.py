"""Give every entity the other names it is known by.

The ``synonyms`` field has been in the schema since the library was
written, and **all 294 entities had it empty**. The cost is not cosmetic:
the library's own records refer to molecules by name, and a reference
only resolves if the name used happens to be the name stored. Of 151
molecules named in marker and secretion lists, 16 resolved -- 11%. One
of those sixteen, ``CD62L``, resolved only because it sits inside a
parenthetical in ``L-selectin (CD62L)``.

So ``IFNg`` in a marker list cannot reach the ``IFN-gamma`` entity that
exists, and adding the genuinely missing entities would leave most
references dangling anyway.

DERIVED, NOT INVENTED
---------------------
Every synonym here comes from something the entity already carries:

  * the parenthetical in its own name -- ``LFA-1 (CD11a/CD18)`` yields
    LFA-1, CD11a and CD18, and the slash is split because that name
    declares a heterodimer of two CD numbers;
  * the stem with the parenthetical removed;
  * its gene symbol and UniProt accession from ``properties``;
  * punctuation variants, because the same molecule is written IFN-gamma,
    IFN-g, IFNg and IFNgamma across these records, and a reader should
    not have to guess which.

Nothing is looked up and nothing is guessed. A synonym that cannot be
derived from the record is not added, because a wrong alias silently
merges two molecules and that is worse than a reference that fails to
resolve -- a failure to resolve is visible.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Dict, Iterable, List, Sequence, Set, Tuple

log = logging.getLogger(__name__)

#: Greek letters as they get written in these records.
_GREEK = {
    "alpha": ["a", "α"], "beta": ["b", "β"], "gamma": ["g", "γ"],
    "delta": ["d", "δ"], "kappa": ["k", "κ"], "lambda": ["l", "λ"],
    "epsilon": ["e", "ε"], "zeta": ["z", "ζ"], "mu": ["u", "μ"],
}

#: Names whose parenthetical is prose rather than an alias, so splitting
#: it would manufacture molecules that do not exist.
_NOT_ALIASES = re.compile(
    r"^(pentameric|secretory|human|mouse|recombinant|soluble|membrane|"
    r"intact|total|free|homophilic|inducible|displayed|gelatinase [ab]|"
    r"p\d+/\w+|beta-2 microglobulin)$", re.I)


def _punctuation_variants(name: str) -> Set[str]:
    """IFN-gamma, IFN gamma, IFNgamma, IFN-g, IFNg -- one molecule."""
    out: Set[str] = set()
    base = name.strip()
    if not base:
        return out

    forms = {base}
    # hyphen / space / nothing
    for sep in ("-", " ", "_"):
        if sep in base:
            forms.add(base.replace(sep, ""))
            forms.add(base.replace(sep, "-"))
            forms.add(base.replace(sep, " "))

    for form in list(forms):
        lowered = form.lower()
        for spelled, shorts in _GREEK.items():
            start = lowered.find(spelled)
            if start < 0:
                continue
            # Substitute the letter; do NOT truncate at it. Cutting the
            # string at "kappa" turns NF-kappaB into NF-k, which is not a
            # name for anything -- the B is part of the molecule.
            stem, tail = form[:start], form[start + len(spelled):]
            for short in shorts:
                out.add(f"{stem}{short}{tail}")
                out.add(f"{stem.rstrip('- ')}{short}{tail}")
    out |= forms
    return {x.strip() for x in out if x.strip() and x.strip() != base}


def derive(name: str, properties: Dict[str, object] | None = None) -> List[str]:
    """Every other name this entity is known by, from its own record."""
    properties = properties or {}
    out: Set[str] = set()

    stem = re.sub(r"\s*\(.*?\)\s*", " ", name).strip()
    if stem and stem != name:
        out.add(stem)

    for inner in re.findall(r"\(([^)]*)\)", name):
        inner = inner.strip()
        if not inner or _NOT_ALIASES.match(inner):
            continue
        # A slash in an alias is a heterodimer or a dual name: both halves
        # are real names for this entity.
        parts = [p.strip() for p in inner.split("/")] if "/" in inner else [inner]
        for part in parts:
            if part and not _NOT_ALIASES.match(part):
                out.add(part)

    for key in ("gene_symbol", "uniprot_id", "hgnc_symbol", "drugbank_id",
                "chembl_id", "pubchem_cid"):
        value = properties.get(key)
        if isinstance(value, str) and value.strip():
            out.add(value.strip())

    for form in list(out) + [stem or name]:
        out |= _punctuation_variants(form)

    out.discard(name)
    out.discard("")
    # Keep them stable and readable.
    return sorted(out, key=lambda s: (len(s), s.lower()))


def ambiguous_forms(rows: Sequence[Tuple[str, str]]) -> Dict[str, List[str]]:
    """Name forms that more than one entity would claim.

    A synonym that two entities answer to is not a synonym: it cannot
    resolve, and whichever entity happens to be indexed first silently
    becomes the answer. Three kinds show up in this library and all three
    are real rather than derivation bugs:

      * a gene and the mutations in it -- ``AR`` is claimed by the gene
        AR and by AR_T878A and AR_amplification, which carry it as their
        gene_symbol. A mutation is not its gene.
      * a shared subunit -- ``CD18`` is in both LFA-1 (CD11a/CD18) and
        MAC-1 (CD11b/CD18), because the beta-2 integrin chain really is
        in both. CD18 is its own molecule, not an alias of either.
      * a locus and its alleles -- ``HLA-A`` is claimed by HLA-A*02:01
        and HLA-A*03:01. You cannot resolve a locus to one allele.

    So they are dropped rather than special-cased. The rule is one rule
    and it is honest: a form that cannot identify one entity is not a
    name for one.
    """
    import collections
    owner = collections.defaultdict(set)
    for name, blob in rows:
        data = json.loads(blob or "{}")
        owner[name.strip().lower()].add(name)
        for form in derive(name, data.get("properties") or {}):
            owner[form.strip().lower()].add(name)
    return {form: sorted(names) for form, names in owner.items()
            if len(names) > 1}


def seed(store, *, dry_run: bool = False) -> Tuple[int, int, int]:
    """Attach derived synonyms to every entity that has none.

    Returns ``(entities given synonyms, synonyms added, forms dropped as
    ambiguous)``. The third number is returned rather than logged so a
    caller cannot take the first two and present the pass as complete.

    An entity that already carries synonyms is left alone -- hand
    curation beats derivation, and overwriting it would be the worse
    failure.
    """
    rows = [tuple(r) for r in store._conn.execute(
        "SELECT name, data_json FROM entities")]
    ambiguous = ambiguous_forms(rows)
    touched = added = dropped = 0
    for name, blob in rows:
        data = json.loads(blob or "{}")
        if data.get("synonyms"):
            continue
        derived = derive(name, data.get("properties") or {})
        synonyms = [s for s in derived
                    if s.strip().lower() not in ambiguous]
        dropped += len(derived) - len(synonyms)
        if not synonyms:
            continue
        touched += 1
        added += len(synonyms)
        if dry_run:
            continue
        entity = store.find_entity_by_name(name)
        if entity is None:
            continue
        entity.synonyms = synonyms
        store.update_entity(entity, changed_by="seed_synonyms")
    return touched, added, dropped


def resolve_index(store) -> Dict[str, str]:
    """Every name form in the library, lowercased, to its entity name.

    This is what a reference is looked up through, so it is built from
    the same forms :func:`derive` produces rather than a second set that
    could disagree with them.
    """
    index: Dict[str, str] = {}
    rows = [tuple(r) for r in store._conn.execute(
        "SELECT name, data_json FROM entities")]
    ambiguous = ambiguous_forms(rows)
    for name, blob in rows:
        data = json.loads(blob or "{}")
        forms = {name} | set(data.get("synonyms") or [])
        forms |= set(derive(name, data.get("properties") or {}))
        for form in forms:
            key = str(form).strip().lower()
            if key in ambiguous and key != name.strip().lower():
                continue
            index.setdefault(key, name)
    return index
