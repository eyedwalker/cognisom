"""
Journal & Forum Subscription Manager
=====================================

Aggregates RSS/Atom feeds and API endpoints from academic journals,
preprint servers, forums, clinical databases, and conference sources.
All sources are free and require no authentication for basic access.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


# ── Data classes ─────────────────────────────────────────────────────


@dataclass
class SubscriptionSource:
    """A single subscribable source."""

    name: str
    category: str  # "journal", "preprint", "forum", "database", "conference", "blog"
    url: str  # RSS/Atom URL or API endpoint
    source_type: str  # "rss", "atom", "json_api"
    description: str = ""
    enabled: bool = True


@dataclass
class FeedEntry:
    """A single entry from any subscribed source."""

    title: str
    url: str
    source_name: str
    category: str
    published: str  # ISO date string
    summary: str = ""
    authors: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


# ── Default sources catalogue ────────────────────────────────────────

DEFAULT_SOURCES: List[SubscriptionSource] = [
    # ── Journals ──
    SubscriptionSource(
        "Nature Cancer", "journal",
        "https://www.nature.com/natcancer.rss", "rss",
        "Springer Nature — cancer biology & translational research",
    ),
    SubscriptionSource(
        "Nature Medicine", "journal",
        "https://www.nature.com/nm.rss", "rss",
        "Springer Nature — clinical and translational medicine",
    ),
    SubscriptionSource(
        "Nature Biotechnology", "journal",
        "https://www.nature.com/nbt.rss", "rss",
        "Springer Nature — biotechnology & bioengineering",
    ),
    SubscriptionSource(
        "Cancer Research", "journal",
        "https://aacrjournals.org/rss/site_1000011/1000008.xml", "rss",
        "AACR — major cancer research journal",
    ),
    SubscriptionSource(
        "Clinical Cancer Research", "journal",
        "https://aacrjournals.org/rss/site_1000013/1000009.xml", "rss",
        "AACR — clinical and translational cancer research",
    ),
    SubscriptionSource(
        "Cell", "journal",
        "http://www.cell.com/cell/current.rss", "rss",
        "Cell Press — flagship life science journal",
    ),
    SubscriptionSource(
        "Cancer Cell", "journal",
        "http://www.cell.com/cancer-cell/current.rss", "rss",
        "Cell Press — cancer biology",
    ),
    SubscriptionSource(
        "Science", "journal",
        "https://www.science.org/action/showFeed?type=etoc&feed=rss&jc=science", "rss",
        "AAAS Science — interdisciplinary research",
    ),
    SubscriptionSource(
        "PNAS", "journal",
        "https://www.pnas.org/action/showFeed?type=etoc&feed=rss&jc=PNAS", "rss",
        "National Academy of Sciences proceedings",
    ),
    SubscriptionSource(
        "Genome Biology", "journal",
        "https://genomebiology.biomedcentral.com/articles/most-recent/rss.xml", "rss",
        "BioMed Central — genomics & systems biology (open access)",
    ),
    SubscriptionSource(
        "J Proteome Research", "journal",
        "https://pubs.acs.org/action/showFeed?type=axatoc&feed=rss&jc=jprobs", "rss",
        "ACS — proteomics research",
    ),
    SubscriptionSource(
        "J Medicinal Chemistry", "journal",
        "https://pubs.acs.org/action/showFeed?type=axatoc&feed=rss&jc=jmcmar", "rss",
        "ACS — drug design & medicinal chemistry",
    ),
    # ── Preprints ──
    SubscriptionSource(
        "medRxiv (Oncology)", "preprint",
        "http://connect.medrxiv.org/medrxiv_xml.php?subject=Oncology", "atom",
        "Clinical oncology preprints",
    ),
    SubscriptionSource(
        "bioRxiv (Cancer Biology)", "preprint",
        "https://connect.biorxiv.org/biorxiv_xml.php?subject=Cancer+Biology", "atom",
        "Cancer biology preprints",
    ),
    SubscriptionSource(
        "arXiv q-bio", "preprint",
        "https://rss.arxiv.org/rss/q-bio", "rss",
        "Quantitative biology preprints (genomics, biomolecules, etc.)",
    ),
    # ── Forums ──
    SubscriptionSource(
        "r/bioinformatics", "forum",
        "https://www.reddit.com/r/bioinformatics/new/.rss?limit=25", "rss",
        "Reddit bioinformatics community",
    ),
    SubscriptionSource(
        "r/genomics", "forum",
        "https://www.reddit.com/r/genomics/new/.rss?limit=25", "rss",
        "Reddit genomics community",
    ),
    SubscriptionSource(
        "BioStars", "forum",
        "https://www.biostars.org/feeds/tag/cancer+genomics/", "rss",
        "BioStars Q&A — cancer & genomics tags",
    ),
    SubscriptionSource(
        "SEQanswers", "forum",
        "http://feeds.feedburner.com/seqanswers-all", "rss",
        "Sequencing community forum",
    ),
    # ── Databases ──
    SubscriptionSource(
        "ClinicalTrials.gov (Prostate Cancer)", "database",
        "https://clinicaltrials.gov/api/v2/studies?query.cond=prostate+cancer&filter.overallStatus=RECRUITING&format=json&pageSize=20", "json_api",
        "Recruiting prostate cancer clinical trials",
    ),
    SubscriptionSource(
        "FDA Drug Approvals", "database",
        "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/drugs/rss.xml", "rss",
        "FDA new drug approvals and safety alerts",
    ),
    # ── Blogs ──
    SubscriptionSource(
        "NVIDIA Developer Blog (Healthcare)", "blog",
        "https://developer.nvidia.com/blog/tag/healthcare-and-lifesciences/feed/", "rss",
        "NVIDIA healthcare & life sciences blog posts",
    ),
]


# ── RSS/Atom parser ──────────────────────────────────────────────────


def _fetch_url(url: str, timeout: int = 20) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Cognisom/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _parse_rss(xml_text: str, source: SubscriptionSource) -> List[FeedEntry]:
    """Parse an RSS 2.0 feed."""
    entries = []
    try:
        root = ET.fromstring(xml_text)
        for item in root.findall(".//item"):
            title = item.findtext("title", "").strip()
            link = item.findtext("link", "").strip()
            desc = item.findtext("description", "").strip()
            pub_date = item.findtext("pubDate", "")

            # Try to parse date
            date_str = _normalize_date(pub_date)

            # Authors from dc:creator
            ns = {"dc": "http://purl.org/dc/elements/1.1/"}
            creator = item.findtext("dc:creator", "", ns).strip()
            authors = [creator] if creator else []

            # Categories as tags
            tags = [c.text.strip() for c in item.findall("category") if c.text]

            entries.append(FeedEntry(
                title=title,
                url=link,
                source_name=source.name,
                category=source.category,
                published=date_str,
                summary=desc[:500],
                authors=authors,
                tags=tags[:10],
            ))
    except ET.ParseError as exc:
        log.warning("RSS parse error for %s: %s", source.name, exc)
    return entries


def _parse_atom(xml_text: str, source: SubscriptionSource) -> List[FeedEntry]:
    """Parse an Atom 1.0 feed."""
    entries = []
    try:
        root = ET.fromstring(xml_text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        for entry in root.findall(".//atom:entry", ns) or root.findall(".//entry"):
            title = ""
            link = ""
            summary = ""
            published = ""
            authors = []

            # Title
            title_el = entry.find("atom:title", ns) or entry.find("title")
            if title_el is not None and title_el.text:
                title = title_el.text.strip()

            # Link
            link_el = entry.find("atom:link", ns) or entry.find("link")
            if link_el is not None:
                link = link_el.get("href", link_el.text or "").strip()

            # Summary
            sum_el = entry.find("atom:summary", ns) or entry.find("summary")
            if sum_el is not None and sum_el.text:
                summary = sum_el.text.strip()[:500]

            # Published
            pub_el = entry.find("atom:published", ns) or entry.find("published") or entry.find("atom:updated", ns) or entry.find("updated")
            if pub_el is not None and pub_el.text:
                published = _normalize_date(pub_el.text)

            # Authors
            for a in entry.findall("atom:author/atom:name", ns) or entry.findall("author/name"):
                if a.text:
                    authors.append(a.text.strip())

            entries.append(FeedEntry(
                title=title,
                url=link,
                source_name=source.name,
                category=source.category,
                published=published,
                summary=summary,
                authors=authors,
            ))
    except ET.ParseError as exc:
        log.warning("Atom parse error for %s: %s", source.name, exc)
    return entries


def _parse_clinicaltrials_json(json_text: str, source: SubscriptionSource) -> List[FeedEntry]:
    """Parse ClinicalTrials.gov API v2 JSON response."""
    entries = []
    try:
        data = json.loads(json_text)
        studies = data.get("studies", [])
        for study in studies:
            proto = study.get("protocolSection", {})
            id_mod = proto.get("identificationModule", {})
            status_mod = proto.get("statusModule", {})
            desc_mod = proto.get("descriptionModule", {})

            nct_id = id_mod.get("nctId", "")
            title = id_mod.get("briefTitle", id_mod.get("officialTitle", ""))
            status = status_mod.get("overallStatus", "")
            brief = desc_mod.get("briefSummary", "")[:500]
            start_date = status_mod.get("startDateStruct", {}).get("date", "")

            entries.append(FeedEntry(
                title=f"[{nct_id}] {title}",
                url=f"https://clinicaltrials.gov/study/{nct_id}",
                source_name=source.name,
                category=source.category,
                published=start_date,
                summary=brief,
                tags=[status],
            ))
    except (json.JSONDecodeError, KeyError) as exc:
        log.warning("ClinicalTrials parse error: %s", exc)
    return entries


def _normalize_date(date_str: str) -> str:
    """Best-effort conversion of various date formats to YYYY-MM-DD."""
    if not date_str:
        return ""
    # ISO 8601
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str[:19], fmt[:19]).strftime("%Y-%m-%d")
        except ValueError:
            continue
    # RFC 822 (RSS)
    try:
        from email.utils import parsedate_to_datetime
        return parsedate_to_datetime(date_str).strftime("%Y-%m-%d")
    except Exception:
        pass
    return date_str[:10]


# ── Subscription Manager ─────────────────────────────────────────────


class SubscriptionManager:
    """Manages journal, forum, and database subscriptions.

    Usage::

        mgr = SubscriptionManager(cache_dir="data/subscriptions")
        entries = mgr.fetch_all()            # fetches from all enabled sources
        entries = mgr.fetch_category("journal")
        mgr.enable("Nature Cancer")
        mgr.disable("SEQanswers")
        mgr.add_custom_source(SubscriptionSource(...))
    """

    def __init__(self, cache_dir: str = "data/subscriptions") -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._sources: List[SubscriptionSource] = list(DEFAULT_SOURCES)
        self._load_user_config()

    # ── Source management ────────────────────────────────────────────

    @property
    def sources(self) -> List[SubscriptionSource]:
        return self._sources

    @property
    def enabled_sources(self) -> List[SubscriptionSource]:
        return [s for s in self._sources if s.enabled]

    def categories(self) -> List[str]:
        return sorted({s.category for s in self._sources})

    def enable(self, name: str) -> None:
        for s in self._sources:
            if s.name == name:
                s.enabled = True
                self._save_user_config()
                return

    def disable(self, name: str) -> None:
        for s in self._sources:
            if s.name == name:
                s.enabled = False
                self._save_user_config()
                return

    def add_custom_source(self, source: SubscriptionSource) -> None:
        self._sources.append(source)
        self._save_user_config()

    def remove_source(self, name: str) -> None:
        self._sources = [s for s in self._sources if s.name != name]
        self._save_user_config()

    # ── Fetching ─────────────────────────────────────────────────────

    def fetch_all(self, max_per_source: int = 20) -> List[FeedEntry]:
        """Fetch entries from all enabled sources."""
        all_entries: List[FeedEntry] = []
        for src in self.enabled_sources:
            try:
                entries = self._fetch_source(src)
                all_entries.extend(entries[:max_per_source])
            except Exception as exc:
                log.warning("Failed to fetch %s: %s", src.name, exc)
            time.sleep(0.3)  # be polite

        # Sort by date descending
        all_entries.sort(key=lambda e: e.published or "", reverse=True)
        return all_entries

    def fetch_category(self, category: str, max_per_source: int = 20) -> List[FeedEntry]:
        """Fetch entries from all enabled sources in a category."""
        all_entries: List[FeedEntry] = []
        for src in self.enabled_sources:
            if src.category != category:
                continue
            try:
                entries = self._fetch_source(src)
                all_entries.extend(entries[:max_per_source])
            except Exception as exc:
                log.warning("Failed to fetch %s: %s", src.name, exc)
            time.sleep(0.3)

        all_entries.sort(key=lambda e: e.published or "", reverse=True)
        return all_entries

    def fetch_source(self, name: str) -> List[FeedEntry]:
        """Fetch entries from a single named source."""
        for src in self._sources:
            if src.name == name:
                return self._fetch_source(src)
        return []

    def _fetch_source(self, src: SubscriptionSource) -> List[FeedEntry]:
        """Fetch and parse a single source."""
        raw = _fetch_url(src.url)

        if src.source_type == "rss":
            return _parse_rss(raw, src)
        elif src.source_type == "atom":
            return _parse_atom(raw, src)
        elif src.source_type == "json_api":
            if "clinicaltrials.gov" in src.url:
                return _parse_clinicaltrials_json(raw, src)
            # Generic JSON — return raw entries
            return []
        return []

    # ── Caching ──────────────────────────────────────────────────────

    def get_cached(self, max_age_hours: int = 6) -> Optional[List[FeedEntry]]:
        """Return cached entries if fresh enough."""
        cache_file = self._cache_dir / "subscriptions_cache.json"
        if not cache_file.exists():
            return None
        age = time.time() - cache_file.stat().st_mtime
        if age > max_age_hours * 3600:
            return None
        try:
            data = json.loads(cache_file.read_text())
            return [FeedEntry(**item) for item in data]
        except Exception:
            return None

    def save_cache(self, entries: List[FeedEntry]) -> None:
        """Persist entries to cache."""
        cache_file = self._cache_dir / "subscriptions_cache.json"
        data = [
            {
                "title": e.title, "url": e.url, "source_name": e.source_name,
                "category": e.category, "published": e.published,
                "summary": e.summary, "authors": e.authors, "tags": e.tags,
            }
            for e in entries
        ]
        cache_file.write_text(json.dumps(data, indent=2))

    # ── User config persistence ──────────────────────────────────────

    def _load_user_config(self) -> None:
        """Load user enable/disable preferences and custom sources."""
        config_file = self._cache_dir / "subscription_config.json"
        if not config_file.exists():
            return
        try:
            cfg = json.loads(config_file.read_text())
            disabled = set(cfg.get("disabled", []))
            for s in self._sources:
                if s.name in disabled:
                    s.enabled = False

            for custom in cfg.get("custom_sources", []):
                self._sources.append(SubscriptionSource(**custom))
        except Exception as exc:
            log.warning("Could not load subscription config: %s", exc)

    def _save_user_config(self) -> None:
        """Save user preferences."""
        config_file = self._cache_dir / "subscription_config.json"
        disabled = [s.name for s in self._sources if not s.enabled]
        custom = [
            {
                "name": s.name, "category": s.category, "url": s.url,
                "source_type": s.source_type, "description": s.description,
                "enabled": s.enabled,
            }
            for s in self._sources
            if s.name not in {d.name for d in DEFAULT_SOURCES}
        ]
        config_file.write_text(json.dumps({
            "disabled": disabled,
            "custom_sources": custom,
        }, indent=2))
