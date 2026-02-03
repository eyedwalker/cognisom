"""
Feed Source Adapters
====================

Each source fetches research articles from a specific API and normalizes
them into FeedItem objects.
"""

import logging
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)


class FeedSource(ABC):
    """Base class for research feed sources."""
    name: str = "unknown"

    @abstractmethod
    def fetch(self, query: Optional[str] = None,
              max_items: int = 20) -> List[dict]:
        """Fetch articles and return list of dicts with standard fields.

        Each dict must have: title, authors, summary, url, source,
        published_date (str), tags (list), doi (optional).
        """
        ...


class PubMedSource(FeedSource):
    """Fetch recent articles from PubMed via NCBI E-utilities (free API)."""

    name = "PubMed"
    BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    DEFAULT_QUERY = (
        "(prostate cancer genomics) OR (drug discovery AI) OR "
        "(single cell RNA-seq cancer) OR (protein structure prediction drug) OR "
        "(immune checkpoint therapy) OR (cancer metastasis simulation)"
    )

    def fetch(self, query: Optional[str] = None,
              max_items: int = 15) -> List[dict]:
        q = query or self.DEFAULT_QUERY
        items = []
        try:
            # Step 1: Search for PMIDs
            search_url = f"{self.BASE}/esearch.fcgi"
            search_r = requests.get(search_url, params={
                "db": "pubmed", "term": q, "retmax": max_items,
                "sort": "date", "retmode": "json",
            }, timeout=15)
            search_r.raise_for_status()
            pmids = search_r.json().get("esearchresult", {}).get("idlist", [])

            if not pmids:
                return items

            # Step 2: Fetch article details
            fetch_url = f"{self.BASE}/efetch.fcgi"
            fetch_r = requests.get(fetch_url, params={
                "db": "pubmed", "id": ",".join(pmids),
                "retmode": "xml",
            }, timeout=30)
            fetch_r.raise_for_status()

            root = ET.fromstring(fetch_r.text)
            for article_el in root.findall(".//PubmedArticle"):
                items.append(self._parse_article(article_el))

        except Exception as e:
            logger.error(f"PubMed fetch error: {e}")

        return items

    def _parse_article(self, el) -> dict:
        medline = el.find(".//MedlineCitation")
        article = medline.find(".//Article") if medline is not None else None

        title = ""
        if article is not None:
            title_el = article.find("ArticleTitle")
            title = title_el.text or "" if title_el is not None else ""

        authors = []
        if article is not None:
            for auth in article.findall(".//Author"):
                last = auth.findtext("LastName", "")
                first = auth.findtext("ForeName", "")
                if last:
                    authors.append(f"{last} {first}".strip())

        abstract = ""
        if article is not None:
            abs_el = article.find(".//Abstract/AbstractText")
            abstract = abs_el.text or "" if abs_el is not None else ""

        pmid = medline.findtext(".//PMID", "") if medline is not None else ""

        # Date
        date_el = medline.find(".//DateCompleted") if medline is not None else None
        if date_el is None and medline is not None:
            date_el = medline.find(".//DateRevised")
        pub_date = ""
        if date_el is not None:
            y = date_el.findtext("Year", "")
            m = date_el.findtext("Month", "01")
            d = date_el.findtext("Day", "01")
            pub_date = f"{y}-{m.zfill(2)}-{d.zfill(2)}"

        # DOI
        doi = ""
        if article is not None:
            for eid in article.findall(".//ELocationID"):
                if eid.get("EIdType") == "doi":
                    doi = eid.text or ""

        # Tags from MeSH
        tags = []
        if medline is not None:
            for mesh in medline.findall(".//MeshHeading/DescriptorName"):
                if mesh.text:
                    tags.append(mesh.text)

        return {
            "title": title,
            "authors": authors[:5],
            "summary": abstract[:500],
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "source": "PubMed",
            "published_date": pub_date,
            "tags": tags[:6],
            "doi": doi,
        }


class BioRxivSource(FeedSource):
    """Fetch recent preprints from bioRxiv REST API."""

    name = "bioRxiv"
    BASE = "https://api.biorxiv.org/details/biorxiv"

    def fetch(self, query: Optional[str] = None,
              max_items: int = 15) -> List[dict]:
        items = []
        try:
            # bioRxiv API returns by date range
            end = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            url = f"{self.BASE}/{start}/{end}/0/{max_items}"

            r = requests.get(url, timeout=15)
            r.raise_for_status()
            data = r.json()

            for paper in data.get("collection", []):
                # Filter by relevance if query provided
                title = paper.get("title", "")
                abstract = paper.get("abstract", "")
                category = paper.get("category", "")

                if query:
                    text = f"{title} {abstract} {category}".lower()
                    if not any(kw in text for kw in query.lower().split()):
                        continue

                items.append({
                    "title": title,
                    "authors": self._parse_authors(paper.get("authors", "")),
                    "summary": abstract[:500],
                    "url": f"https://www.biorxiv.org/content/{paper.get('doi', '')}",
                    "source": "bioRxiv",
                    "published_date": paper.get("date", ""),
                    "tags": [category] if category else [],
                    "doi": paper.get("doi", ""),
                })

        except Exception as e:
            logger.error(f"bioRxiv fetch error: {e}")

        return items[:max_items]

    @staticmethod
    def _parse_authors(authors_str: str) -> List[str]:
        if not authors_str:
            return []
        return [a.strip() for a in authors_str.split(";") if a.strip()][:5]


class ArXivSource(FeedSource):
    """Fetch recent papers from arXiv Atom API (q-bio categories)."""

    name = "arXiv"
    BASE = "http://export.arxiv.org/api/query"

    DEFAULT_QUERY = (
        "cat:q-bio.GN OR cat:q-bio.BM OR cat:q-bio.QM OR "
        "all:cancer+genomics OR all:drug+discovery+protein OR "
        "all:single+cell+RNA"
    )

    def fetch(self, query: Optional[str] = None,
              max_items: int = 15) -> List[dict]:
        q = query or self.DEFAULT_QUERY
        items = []
        try:
            r = requests.get(self.BASE, params={
                "search_query": q,
                "start": 0,
                "max_results": max_items,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }, timeout=15)
            r.raise_for_status()

            # Parse Atom XML
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            root = ET.fromstring(r.text)

            for entry in root.findall("atom:entry", ns):
                title = entry.findtext("atom:title", "", ns).strip().replace("\n", " ")
                summary = entry.findtext("atom:summary", "", ns).strip().replace("\n", " ")
                published = entry.findtext("atom:published", "", ns)[:10]

                authors = []
                for author in entry.findall("atom:author", ns):
                    name = author.findtext("atom:name", "", ns)
                    if name:
                        authors.append(name)

                link = ""
                for link_el in entry.findall("atom:link", ns):
                    if link_el.get("type") == "text/html":
                        link = link_el.get("href", "")
                        break
                if not link:
                    link = entry.findtext("atom:id", "", ns)

                categories = []
                for cat in entry.findall("atom:category", ns):
                    term = cat.get("term", "")
                    if term:
                        categories.append(term)

                items.append({
                    "title": title,
                    "authors": authors[:5],
                    "summary": summary[:500],
                    "url": link,
                    "source": "arXiv",
                    "published_date": published,
                    "tags": categories[:4],
                    "doi": "",
                })

        except Exception as e:
            logger.error(f"arXiv fetch error: {e}")

        return items
