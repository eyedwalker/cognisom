"""Database query tools — public genomic / protein / literature APIs."""

from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

from .tools import Tool, ToolResult

log = logging.getLogger(__name__)

# ── helpers ──────────────────────────────────────────────────────────


def _ncbi_key_params() -> str:
    """Return NCBI api_key, tool, and email query params if available."""
    key = os.environ.get("NCBI_API_KEY", "")
    params = "&tool=cognisom&email=support@cognisom.com"
    if key:
        params += f"&api_key={key}"
    return params


def _http_get(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Cognisom/2.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode()


def _http_get_json(url: str, timeout: int = 30) -> Any:
    return json.loads(_http_get(url, timeout=timeout))


# ── NCBI Gene ────────────────────────────────────────────────────────


class NCBIGeneTool(Tool):
    """Search NCBI Gene database for gene info, aliases, summaries."""

    name = "ncbi_gene"
    description = "Look up a gene by name or ID — returns summary, aliases, chromosome location, and function."
    parameters = {"query": "Gene name or NCBI Gene ID (e.g. 'TP53' or '7157')"}

    def run(self, *, query: str = "", **kw) -> ToolResult:
        try:
            # Step 1: search
            search_url = (
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
                f"?db=gene&term={urllib.parse.quote(query)}[Gene Name]+AND+Homo+sapiens[Organism]"
                f"&retmax=1&retmode=json{_ncbi_key_params()}"
            )
            sr = _http_get_json(search_url)
            ids = sr.get("esearchresult", {}).get("idlist", [])
            if not ids:
                return ToolResult(tool_name=self.name, success=True, data={"message": f"No gene found for '{query}'"})

            gene_id = ids[0]

            # Step 2: summary
            sum_url = (
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                f"?db=gene&id={gene_id}&retmode=json{_ncbi_key_params()}"
            )
            sd = _http_get_json(sum_url)
            doc = sd.get("result", {}).get(gene_id, {})

            info = {
                "gene_id": gene_id,
                "symbol": doc.get("name", query),
                "full_name": doc.get("description", ""),
                "organism": doc.get("organism", {}).get("scientificname", ""),
                "chromosome": doc.get("chromosome", ""),
                "aliases": doc.get("otheraliases", ""),
                "summary": doc.get("summary", ""),
                "gene_type": doc.get("genetype", ""),
                "maplocation": doc.get("maplocation", ""),
            }
            return ToolResult(tool_name=self.name, success=True, data=info)
        except Exception as exc:
            return ToolResult(tool_name=self.name, success=False, error=str(exc))


# ── UniProt ──────────────────────────────────────────────────────────


class UniProtTool(Tool):
    """Search UniProt for protein information, sequence, GO terms."""

    name = "uniprot"
    description = "Look up a protein by gene name — returns sequence, function, GO terms, pathways."
    parameters = {"query": "Gene name or UniProt accession (e.g. 'TP53' or 'P04637')"}

    def run(self, *, query: str = "", **kw) -> ToolResult:
        try:
            search_url = (
                "https://rest.uniprot.org/uniprotkb/search"
                f"?query={urllib.parse.quote(query)}+AND+organism_id:9606"
                "&format=json&fields=accession,gene_names,protein_name,sequence,"
                "go_p,go_f,go_c,cc_function,cc_pathway,length&size=1"
            )
            req = urllib.request.Request(search_url, headers={"User-Agent": "Cognisom/1.0", "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())

            results = data.get("results", [])
            if not results:
                return ToolResult(tool_name=self.name, success=True, data={"message": f"No protein found for '{query}'"})

            entry = results[0]

            # Extract GO terms
            go_terms = []
            for section in ("go_p", "go_f", "go_c"):
                for aspect_list in [entry.get(section, [])]:
                    if isinstance(aspect_list, list):
                        for go in aspect_list:
                            if isinstance(go, dict):
                                go_terms.append(f"{go.get('id', '')} - {go.get('name', '')}")

            # Extract function comments
            functions = []
            for comment in entry.get("cc_function", []) if isinstance(entry.get("cc_function"), list) else []:
                if isinstance(comment, dict):
                    for txt in comment.get("texts", []):
                        if isinstance(txt, dict):
                            functions.append(txt.get("value", ""))

            seq_obj = entry.get("sequence", {})

            info = {
                "accession": entry.get("primaryAccession", ""),
                "protein_name": _nested_get(entry, "proteinDescription", "recommendedName", "fullName", "value"),
                "gene_names": [g.get("geneName", {}).get("value", "") for g in entry.get("genes", [])],
                "organism": "Homo sapiens",
                "length": seq_obj.get("length", 0),
                "sequence": seq_obj.get("value", "")[:200] + "…" if len(seq_obj.get("value", "")) > 200 else seq_obj.get("value", ""),
                "function": "; ".join(functions)[:500],
                "go_terms": go_terms[:15],
                "pathways": [p.get("name", "") for p in entry.get("cc_pathway", [])] if isinstance(entry.get("cc_pathway"), list) else [],
            }
            return ToolResult(tool_name=self.name, success=True, data=info)
        except Exception as exc:
            return ToolResult(tool_name=self.name, success=False, error=str(exc))


def _nested_get(d: dict, *keys: str) -> str:
    """Safely traverse nested dicts."""
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, {})
        else:
            return ""
    return d if isinstance(d, str) else ""


# ── PDB Search ───────────────────────────────────────────────────────


class PDBSearchTool(Tool):
    """Search RCSB PDB for protein structures."""

    name = "pdb_search"
    description = "Find 3D structures by gene name or keyword — returns PDB IDs, titles, resolution, methods."
    parameters = {"query": "Gene name or keyword (e.g. 'TP53' or 'prostate cancer')"}

    def run(self, *, query: str = "", max_results: int = 5, **kw) -> ToolResult:
        try:
            # RCSB search API
            search_body = json.dumps({
                "query": {
                    "type": "terminal",
                    "service": "full_text",
                    "parameters": {"value": query},
                },
                "return_type": "entry",
                "request_options": {"paginate": {"start": 0, "rows": max_results}},
            })
            req = urllib.request.Request(
                "https://search.rcsb.org/rcsbsearch/v2/query",
                data=search_body.encode(),
                headers={"Content-Type": "application/json", "User-Agent": "Cognisom/1.0"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())

            pdb_ids = [r["identifier"] for r in data.get("result_set", [])]
            if not pdb_ids:
                return ToolResult(tool_name=self.name, success=True, data={"message": f"No structures found for '{query}'"})

            # Fetch summaries for each PDB
            structures = []
            for pid in pdb_ids:
                try:
                    info_url = f"https://data.rcsb.org/rest/v1/core/entry/{pid}"
                    entry = _http_get_json(info_url)
                    struct = entry.get("rcsb_entry_info", {})
                    structures.append({
                        "pdb_id": pid,
                        "title": entry.get("struct", {}).get("title", ""),
                        "method": struct.get("experimental_method", ""),
                        "resolution": struct.get("resolution_combined", [None])[0],
                        "polymer_count": struct.get("polymer_entity_count", 0),
                        "release_date": entry.get("rcsb_accession_info", {}).get("initial_release_date", ""),
                    })
                except Exception:
                    structures.append({"pdb_id": pid, "error": "Could not fetch details"})

            return ToolResult(tool_name=self.name, success=True, data=structures)
        except Exception as exc:
            return ToolResult(tool_name=self.name, success=False, error=str(exc))


# ── cBioPortal ───────────────────────────────────────────────────────


class CBioPortalTool(Tool):
    """Query cBioPortal for cancer genomics data (mutations, CNA, expression)."""

    name = "cbioportal"
    description = "Get cancer mutation frequency, copy-number alterations, and expression for a gene across studies."
    parameters = {
        "gene": "Gene symbol (e.g. 'TP53')",
        "study": "cBioPortal study ID (default: 'prad_tcga' — prostate adenocarcinoma)",
    }

    CBIO_BASE = "https://www.cbioportal.org/api"

    def run(self, *, gene: str = "", study: str = "prad_tcga", **kw) -> ToolResult:
        try:
            # Get molecular profiles for study
            profiles_url = f"{self.CBIO_BASE}/molecular-profiles?studyId={study}"
            profiles = _http_get_json(profiles_url)
            mutation_profile = None
            for p in profiles:
                if "mutations" in p.get("molecularAlterationType", "").lower():
                    mutation_profile = p["molecularProfileId"]
                    break

            # Get mutations for gene
            mutations = []
            if mutation_profile:
                mut_url = (
                    f"{self.CBIO_BASE}/molecular-profiles/{mutation_profile}/mutations"
                    f"?sampleListId={study}_all&entrezGeneId=0"
                )
                # Use gene symbol search instead
                genes_url = f"{self.CBIO_BASE}/genes/{gene}"
                try:
                    gene_info = _http_get_json(genes_url)
                    entrez_id = gene_info.get("entrezGeneId", 0)

                    mut_url = (
                        f"{self.CBIO_BASE}/molecular-profiles/{mutation_profile}/mutations"
                        f"?sampleListId={study}_all&entrezGeneId={entrez_id}"
                    )
                    mutations_raw = _http_get_json(mut_url)
                    # Summarise
                    mut_types = {}
                    for m in mutations_raw:
                        mt = m.get("mutationType", "Unknown")
                        mut_types[mt] = mut_types.get(mt, 0) + 1
                    mutations = [{"type": k, "count": v} for k, v in sorted(mut_types.items(), key=lambda x: -x[1])]
                except Exception:
                    pass

            # Study metadata
            study_url = f"{self.CBIO_BASE}/studies/{study}"
            try:
                study_info = _http_get_json(study_url)
                study_name = study_info.get("name", study)
                sample_count = study_info.get("allSampleCount", 0)
            except Exception:
                study_name = study
                sample_count = 0

            data = {
                "gene": gene,
                "study": study,
                "study_name": study_name,
                "sample_count": sample_count,
                "total_mutations": sum(m["count"] for m in mutations),
                "mutation_types": mutations,
            }
            return ToolResult(tool_name=self.name, success=True, data=data)
        except Exception as exc:
            return ToolResult(tool_name=self.name, success=False, error=str(exc))


# ── PubMed Literature ────────────────────────────────────────────────


class PubMedSearchTool(Tool):
    """Search PubMed for research articles."""

    name = "pubmed_search"
    description = "Search PubMed literature — returns titles, abstracts, DOIs, authors."
    parameters = {
        "query": "Search query (e.g. 'TP53 prostate cancer immunotherapy')",
        "max_results": "Number of results (default 5)",
    }

    def run(self, *, query: str = "", max_results: int = 5, **kw) -> ToolResult:
        try:
            # Search
            search_url = (
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
                f"?db=pubmed&term={urllib.parse.quote(query)}"
                f"&retmax={max_results}&retmode=json&sort=date{_ncbi_key_params()}"
            )
            sr = _http_get_json(search_url)
            ids = sr.get("esearchresult", {}).get("idlist", [])
            if not ids:
                return ToolResult(tool_name=self.name, success=True, data=[])

            # Fetch details
            fetch_url = (
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
                f"?db=pubmed&id={','.join(ids)}&retmode=xml{_ncbi_key_params()}"
            )
            xml_text = _http_get(fetch_url)
            root = ET.fromstring(xml_text)

            articles = []
            for art in root.findall(".//PubmedArticle"):
                medline = art.find("MedlineCitation")
                if medline is None:
                    continue
                article = medline.find("Article")
                if article is None:
                    continue

                title = article.findtext("ArticleTitle", "")
                abstract_parts = article.findall(".//AbstractText")
                abstract = " ".join(a.text or "" for a in abstract_parts)[:500]

                # Authors
                authors = []
                for a in article.findall(".//Author"):
                    ln = a.findtext("LastName", "")
                    fn = a.findtext("ForeName", "")
                    if ln:
                        authors.append(f"{ln} {fn}".strip())

                # DOI
                doi = ""
                for eid in article.findall(".//ELocationID"):
                    if eid.get("EIdType") == "doi":
                        doi = eid.text or ""

                pmid = medline.findtext("PMID", "")

                # Date
                pub_date = article.find(".//PubDate")
                year = pub_date.findtext("Year", "") if pub_date is not None else ""

                articles.append({
                    "pmid": pmid,
                    "title": title,
                    "authors": authors[:5],
                    "abstract": abstract,
                    "doi": doi,
                    "year": year,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
                })

            return ToolResult(tool_name=self.name, success=True, data=articles)
        except Exception as exc:
            return ToolResult(tool_name=self.name, success=False, error=str(exc))
