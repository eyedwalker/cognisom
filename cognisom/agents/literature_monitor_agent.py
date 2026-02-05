"""
Literature Monitor Agent (Phase 7)
===================================

Autonomous agent that monitors scientific literature for new publications
relevant to prostate cancer research and Cognisom's knowledge base.

Sources:
- PubMed (via NCBI E-utilities)
- bioRxiv (via API)
- arXiv (for computational biology)

The agent identifies papers that may require updates to:
- Entity library (new genes, proteins, drugs, pathways)
- Simulation parameters (new kinetic data)
- Validation benchmarks (new experimental data)

Usage::

    from cognisom.agents import LiteratureMonitorAgent

    agent = LiteratureMonitorAgent()
    report = agent.scan_recent(days=7)
    print(report.summary())
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta

log = logging.getLogger(__name__)


@dataclass
class Publication:
    """A scientific publication."""
    source: str = ""        # "pubmed", "biorxiv", "arxiv"
    pub_id: str = ""        # PMID, DOI, or arXiv ID
    title: str = ""
    abstract: str = ""
    authors: List[str] = field(default_factory=list)
    journal: str = ""
    publication_date: str = ""
    doi: str = ""
    url: str = ""
    relevance_score: float = 0.0
    matched_terms: List[str] = field(default_factory=list)
    category: str = ""      # "gene", "drug", "mechanism", "clinical", "method"


@dataclass
class LiteratureReport:
    """Results of a literature scan."""
    timestamp: float = 0.0
    days_scanned: int = 7
    sources_checked: List[str] = field(default_factory=list)
    total_papers: int = 0
    relevant_papers: int = 0
    high_priority: int = 0
    publications: List[Publication] = field(default_factory=list)
    suggested_updates: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    elapsed_sec: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()

    def summary(self) -> str:
        status = "OK" if not self.errors else f"ERRORS ({len(self.errors)})"
        lines = [
            f"Literature Monitor {status}",
            f"  Period: {self.days_scanned} days",
            f"  Sources: {', '.join(self.sources_checked)}",
            f"  Total papers: {self.total_papers}",
            f"  Relevant: {self.relevant_papers} (High priority: {self.high_priority})",
            f"  Elapsed: {self.elapsed_sec:.1f}s",
        ]
        if self.suggested_updates:
            lines.append("  Suggested updates:")
            for s in self.suggested_updates[:5]:
                lines.append(f"    - {s}")
        for err in self.errors[:3]:
            lines.append(f"  [ERROR] {err}")
        return "\n".join(lines)


class LiteratureMonitorAgent:
    """Autonomous agent for monitoring scientific literature.

    Scans PubMed, bioRxiv, and arXiv for prostate cancer publications.
    """

    # Core search terms
    CORE_TERMS = [
        "prostate cancer",
        "prostate adenocarcinoma",
        "PRAD",
        "castration resistant prostate cancer",
        "CRPC",
    ]

    # Gene/pathway terms to monitor
    GENE_TERMS = [
        "androgen receptor",
        "AR gene prostate",
        "PTEN prostate",
        "TP53 prostate cancer",
        "BRCA2 prostate",
        "ERG fusion",
        "TMPRSS2",
    ]

    # Drug/treatment terms
    DRUG_TERMS = [
        "enzalutamide",
        "abiraterone",
        "docetaxel prostate",
        "PARP inhibitor prostate",
        "checkpoint inhibitor prostate",
        "ADT androgen deprivation",
    ]

    # Methodology terms
    METHOD_TERMS = [
        "single cell RNA-seq prostate",
        "spatial transcriptomics prostate",
        "digital twin cancer",
        "tumor simulation",
        "agent-based model cancer",
    ]

    # High-priority journals for prostate cancer
    PRIORITY_JOURNALS = {
        "Cancer Cell",
        "Nature Cancer",
        "Cancer Discovery",
        "Clinical Cancer Research",
        "European Urology",
        "Journal of Clinical Oncology",
        "Nature Medicine",
        "Cell",
        "Nature",
        "Science",
    }

    def __init__(self, cache_dir: str = "data/literature_cache") -> None:
        """Initialize the literature monitor.

        Args:
            cache_dir: Directory for caching search results
        """
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._seen_ids: Set[str] = set()
        self._load_seen()

    def scan_recent(self, days: int = 7) -> LiteratureReport:
        """Scan for recent publications across all sources."""
        t0 = time.time()
        report = LiteratureReport(days_scanned=days)

        # Scan each source
        pubmed_pubs = self._scan_pubmed(days)
        report.sources_checked.append("PubMed")
        report.total_papers += len(pubmed_pubs)
        report.publications.extend(pubmed_pubs)

        biorxiv_pubs = self._scan_biorxiv(days)
        report.sources_checked.append("bioRxiv")
        report.total_papers += len(biorxiv_pubs)
        report.publications.extend(biorxiv_pubs)

        arxiv_pubs = self._scan_arxiv(days)
        report.sources_checked.append("arXiv")
        report.total_papers += len(arxiv_pubs)
        report.publications.extend(arxiv_pubs)

        # Filter for relevance and categorize
        for pub in report.publications:
            self._score_relevance(pub)
            self._categorize(pub)
            if pub.relevance_score >= 0.5:
                report.relevant_papers += 1
            if pub.relevance_score >= 0.8 or pub.journal in self.PRIORITY_JOURNALS:
                report.high_priority += 1

        # Sort by relevance
        report.publications.sort(key=lambda p: p.relevance_score, reverse=True)

        # Generate suggested updates
        report.suggested_updates = self._generate_suggestions(report.publications)

        # Save seen IDs
        for pub in report.publications:
            self._seen_ids.add(pub.pub_id)
        self._save_seen()

        report.elapsed_sec = time.time() - t0
        return report

    def scan_pubmed(self, days: int = 7) -> LiteratureReport:
        """Scan PubMed only."""
        t0 = time.time()
        report = LiteratureReport(days_scanned=days)
        report.sources_checked.append("PubMed")

        pubs = self._scan_pubmed(days)
        report.publications = pubs
        report.total_papers = len(pubs)

        for pub in pubs:
            self._score_relevance(pub)
            if pub.relevance_score >= 0.5:
                report.relevant_papers += 1
            if pub.relevance_score >= 0.8:
                report.high_priority += 1

        report.elapsed_sec = time.time() - t0
        return report

    # ── PubMed Methods ───────────────────────────────────────────────

    def _scan_pubmed(self, days: int) -> List[Publication]:
        """Search PubMed for recent publications."""
        import urllib.request
        import urllib.parse
        import xml.etree.ElementTree as ET

        publications = []

        # Build search query
        terms = self.CORE_TERMS[:3]  # Use core terms
        query = " OR ".join(f'"{t}"[Title/Abstract]' for t in terms)
        date_filter = f" AND {days}[dp]"

        encoded_query = urllib.parse.quote(query + date_filter)
        search_url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
            f"db=pubmed&term={encoded_query}&retmax=50&retmode=xml"
        )

        try:
            with urllib.request.urlopen(search_url, timeout=15) as resp:
                xml_data = resp.read().decode()
                root = ET.fromstring(xml_data)

                # Extract PMIDs
                pmids = [id_elem.text for id_elem in root.findall(".//Id")]
                log.info("PubMed search found %d papers", len(pmids))

                # Fetch details for each PMID
                for pmid in pmids[:20]:  # Limit to avoid rate limiting
                    if pmid in self._seen_ids:
                        continue
                    pub = self._fetch_pubmed_details(pmid)
                    if pub:
                        publications.append(pub)

        except Exception as e:
            log.warning("PubMed search failed: %s", e)

        return publications

    def _fetch_pubmed_details(self, pmid: str) -> Optional[Publication]:
        """Fetch publication details from PubMed."""
        import urllib.request
        import xml.etree.ElementTree as ET

        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid}&retmode=xml"

        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                xml_data = resp.read().decode()
                root = ET.fromstring(xml_data)

                article = root.find(".//PubmedArticle")
                if article is None:
                    return None

                title_elem = article.find(".//ArticleTitle")
                abstract_elem = article.find(".//AbstractText")
                journal_elem = article.find(".//Journal/Title")

                return Publication(
                    source="pubmed",
                    pub_id=pmid,
                    title=title_elem.text if title_elem is not None else "",
                    abstract=abstract_elem.text if abstract_elem is not None else "",
                    journal=journal_elem.text if journal_elem is not None else "",
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                )

        except Exception as e:
            log.warning("Failed to fetch PMID %s: %s", pmid, e)
            return None

    # ── bioRxiv Methods ──────────────────────────────────────────────

    def _scan_biorxiv(self, days: int) -> List[Publication]:
        """Search bioRxiv for recent preprints."""
        import urllib.request
        import json
        from datetime import datetime, timedelta

        publications = []

        # bioRxiv API date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        url = f"https://api.biorxiv.org/details/biorxiv/{start_str}/{end_str}/0/50"

        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                data = json.loads(resp.read().decode())
                papers = data.get("collection", [])

                # Filter for prostate cancer related
                for paper in papers:
                    title = paper.get("title", "").lower()
                    abstract = paper.get("abstract", "").lower()

                    is_relevant = any(
                        term.lower() in title or term.lower() in abstract
                        for term in self.CORE_TERMS
                    )

                    if is_relevant:
                        doi = paper.get("doi", "")
                        if doi in self._seen_ids:
                            continue

                        pub = Publication(
                            source="biorxiv",
                            pub_id=doi,
                            title=paper.get("title", ""),
                            abstract=paper.get("abstract", ""),
                            authors=paper.get("authors", "").split("; "),
                            publication_date=paper.get("date", ""),
                            doi=doi,
                            url=f"https://www.biorxiv.org/content/{doi}",
                        )
                        publications.append(pub)

                log.info("bioRxiv search found %d relevant papers", len(publications))

        except Exception as e:
            log.warning("bioRxiv search failed: %s", e)

        return publications

    # ── arXiv Methods ────────────────────────────────────────────────

    def _scan_arxiv(self, days: int) -> List[Publication]:
        """Search arXiv for recent papers in q-bio and cs.CE."""
        import urllib.request
        import urllib.parse
        import xml.etree.ElementTree as ET

        publications = []

        # arXiv search query
        query = urllib.parse.quote('all:"prostate cancer" OR all:"tumor simulation"')
        url = f"http://export.arxiv.org/api/query?search_query={query}&max_results=20&sortBy=submittedDate&sortOrder=descending"

        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                xml_data = resp.read().decode()
                root = ET.fromstring(xml_data)

                ns = {"atom": "http://www.w3.org/2005/Atom"}
                for entry in root.findall("atom:entry", ns):
                    arxiv_id = entry.find("atom:id", ns)
                    title = entry.find("atom:title", ns)
                    summary = entry.find("atom:summary", ns)

                    if arxiv_id is not None:
                        pub_id = arxiv_id.text.split("/")[-1]
                        if pub_id in self._seen_ids:
                            continue

                        pub = Publication(
                            source="arxiv",
                            pub_id=pub_id,
                            title=title.text.strip() if title is not None else "",
                            abstract=summary.text.strip() if summary is not None else "",
                            url=arxiv_id.text,
                        )
                        publications.append(pub)

                log.info("arXiv search found %d papers", len(publications))

        except Exception as e:
            log.warning("arXiv search failed: %s", e)

        return publications

    # ── Relevance Scoring ────────────────────────────────────────────

    def _score_relevance(self, pub: Publication) -> None:
        """Score publication relevance to Cognisom knowledge base."""
        text = (pub.title + " " + pub.abstract).lower()
        matched = []
        score = 0.0

        # Core terms (high weight)
        for term in self.CORE_TERMS:
            if term.lower() in text:
                matched.append(term)
                score += 0.2

        # Gene terms
        for term in self.GENE_TERMS:
            if term.lower() in text:
                matched.append(term)
                score += 0.15

        # Drug terms
        for term in self.DRUG_TERMS:
            if term.lower() in text:
                matched.append(term)
                score += 0.15

        # Method terms (computational methods are very relevant)
        for term in self.METHOD_TERMS:
            if term.lower() in text:
                matched.append(term)
                score += 0.25

        # Priority journal boost
        if pub.journal in self.PRIORITY_JOURNALS:
            score += 0.2

        pub.relevance_score = min(score, 1.0)
        pub.matched_terms = matched

    def _categorize(self, pub: Publication) -> None:
        """Categorize publication by topic."""
        text = (pub.title + " " + pub.abstract).lower()

        if any(t.lower() in text for t in self.METHOD_TERMS):
            pub.category = "method"
        elif any(t.lower() in text for t in self.DRUG_TERMS):
            pub.category = "drug"
        elif any(t.lower() in text for t in self.GENE_TERMS):
            pub.category = "gene"
        elif "clinical trial" in text or "patient" in text:
            pub.category = "clinical"
        else:
            pub.category = "mechanism"

    def _generate_suggestions(self, publications: List[Publication]) -> List[str]:
        """Generate suggested updates based on publications."""
        suggestions = []

        for pub in publications[:10]:  # Top 10 most relevant
            if pub.relevance_score < 0.5:
                continue

            if pub.category == "gene":
                suggestions.append(
                    f"Review gene data: '{pub.title[:60]}...' may have new gene findings"
                )
            elif pub.category == "drug":
                suggestions.append(
                    f"Check drug library: '{pub.title[:60]}...' may have new drug data"
                )
            elif pub.category == "method":
                suggestions.append(
                    f"Methodology update: '{pub.title[:60]}...' may have simulation improvements"
                )

        return suggestions[:5]  # Limit suggestions

    # ── Persistence ──────────────────────────────────────────────────

    def _load_seen(self) -> None:
        """Load seen publication IDs."""
        seen_file = self._cache_dir / "seen_ids.txt"
        if seen_file.exists():
            self._seen_ids = set(seen_file.read_text().strip().split("\n"))

    def _save_seen(self) -> None:
        """Save seen publication IDs."""
        seen_file = self._cache_dir / "seen_ids.txt"
        # Keep only last 1000 IDs
        recent = list(self._seen_ids)[-1000:]
        seen_file.write_text("\n".join(recent))
