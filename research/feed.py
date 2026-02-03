"""
Research Feed Aggregator
========================

Merges articles from multiple sources, ranks by relevance, and caches results.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .sources import FeedSource, PubMedSource, BioRxivSource, ArXivSource

logger = logging.getLogger(__name__)


@dataclass
class FeedItem:
    """Normalized research article from any source."""
    title: str
    authors: List[str]
    summary: str
    url: str
    source: str             # "PubMed", "bioRxiv", "arXiv"
    published_date: str     # "YYYY-MM-DD"
    tags: List[str] = field(default_factory=list)
    doi: str = ""
    relevance_score: float = 0.0


# Keywords for relevance scoring (weighted)
RELEVANCE_KEYWORDS = {
    # High relevance (weight 3)
    "prostate cancer": 3, "cancer simulation": 3, "drug discovery": 3,
    "single cell": 3, "immune checkpoint": 3, "metastasis": 3,
    "protein structure prediction": 3,
    # Medium relevance (weight 2)
    "cancer genomics": 2, "protein folding": 2, "molecular docking": 2,
    "gene expression": 2, "scRNA-seq": 2, "epigenetic": 2,
    "circadian": 2, "morphogen": 2, "nvidia": 2, "bionemo": 2,
    # Base relevance (weight 1)
    "cancer": 1, "tumor": 1, "genomic": 1, "protein": 1,
    "cell cycle": 1, "apoptosis": 1, "angiogenesis": 1,
    "immunotherapy": 1, "AI": 1, "deep learning": 1,
}


class ResearchFeed:
    """Aggregates research from multiple sources with caching."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.sources: List[FeedSource] = []
        self._cache_dir = cache_dir or str(
            Path(__file__).parent.parent.parent / "data" / "feed_cache"
        )
        self._register_default_sources()

    def _register_default_sources(self):
        self.sources = [
            PubMedSource(),
            BioRxivSource(),
            ArXivSource(),
        ]

    def register_source(self, source: FeedSource):
        self.sources.append(source)

    def fetch_all(self, max_items: int = 50,
                  query: Optional[str] = None) -> List[FeedItem]:
        """Fetch from all sources, merge, rank, and return.

        Args:
            max_items: Maximum total items to return.
            query: Optional custom search query (overrides source defaults).

        Returns:
            List of FeedItem sorted by relevance then date.
        """
        all_items = []

        per_source = max(5, max_items // max(len(self.sources), 1))

        for source in self.sources:
            try:
                raw = source.fetch(query=query, max_items=per_source)
                for r in raw:
                    item = FeedItem(
                        title=r.get("title", ""),
                        authors=r.get("authors", []),
                        summary=r.get("summary", ""),
                        url=r.get("url", ""),
                        source=r.get("source", source.name),
                        published_date=r.get("published_date", ""),
                        tags=r.get("tags", []),
                        doi=r.get("doi", ""),
                    )
                    item.relevance_score = self._score_relevance(item)
                    all_items.append(item)
            except Exception as e:
                logger.error(f"Error fetching from {source.name}: {e}")

        # Sort: relevance descending, then date descending
        all_items.sort(
            key=lambda x: (x.relevance_score, x.published_date),
            reverse=True,
        )

        result = all_items[:max_items]

        # Cache
        self._save_cache(result)

        return result

    def get_cached(self, max_age_hours: int = 24) -> List[FeedItem]:
        """Return cached feed items if fresh enough.

        Args:
            max_age_hours: Maximum age of cache in hours.

        Returns:
            List of FeedItem from cache, or empty list if stale/missing.
        """
        cache_path = os.path.join(self._cache_dir, "feed_cache.json")
        if not os.path.exists(cache_path):
            return []

        mtime = os.path.getmtime(cache_path)
        age_hours = (time.time() - mtime) / 3600
        if age_hours > max_age_hours:
            return []

        try:
            with open(cache_path) as f:
                data = json.load(f)
            return [FeedItem(**item) for item in data]
        except Exception as e:
            logger.error(f"Cache read error: {e}")
            return []

    def search(self, query: str, max_items: int = 30) -> List[FeedItem]:
        """Search with a custom query across all sources."""
        return self.fetch_all(max_items=max_items, query=query)

    def _score_relevance(self, item: FeedItem) -> float:
        """Score relevance based on keyword matching."""
        text = f"{item.title} {item.summary} {' '.join(item.tags)}".lower()
        score = 0.0
        for keyword, weight in RELEVANCE_KEYWORDS.items():
            if keyword.lower() in text:
                score += weight
        return score

    def _save_cache(self, items: List[FeedItem]):
        """Save feed items to JSON cache."""
        try:
            os.makedirs(self._cache_dir, exist_ok=True)
            cache_path = os.path.join(self._cache_dir, "feed_cache.json")
            with open(cache_path, "w") as f:
                json.dump([asdict(item) for item in items], f, indent=2)
        except Exception as e:
            logger.error(f"Cache write error: {e}")
