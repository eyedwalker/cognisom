"""Research Innovation Feed – latest biotech, cancer & genomics papers."""

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st
from cognisom.dashboard.page_config import safe_set_page_config
from datetime import datetime, timedelta

safe_set_page_config(page_title="Research Feed | Cognisom", page_icon="📰", layout="wide")

from cognisom.auth.middleware import streamlit_page_gate
user = streamlit_page_gate("6_research_feed")

from cognisom.research import ResearchFeed, PubMedSource, BioRxivSource, ArXivSource

# ── Initialise feed ─────────────────────────────────────────────────

CACHE_DIR = Path(_project_root) / "data" / "research_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_resource
def get_feed() -> ResearchFeed:
    return ResearchFeed(cache_dir=str(CACHE_DIR))


feed = get_feed()

# ── Page header ─────────────────────────────────────────────────────

st.title("Research Innovation Feed")
st.markdown(
    "Live stream of the latest **prostate-cancer**, **drug-discovery**, "
    "**genomics**, and **protein-structure** research from PubMed, bioRxiv & arXiv."
)

# ── Sidebar controls ────────────────────────────────────────────────

st.sidebar.header("Feed Filters")

sources_enabled = {
    "PubMed": st.sidebar.checkbox("PubMed", value=True),
    "bioRxiv": st.sidebar.checkbox("bioRxiv", value=True),
    "arXiv": st.sidebar.checkbox("arXiv", value=True),
}

st.sidebar.markdown("---")
keyword = st.sidebar.text_input(
    "Keyword filter",
    placeholder="e.g. BRCA1, immunotherapy, docking",
    help="Leave blank for the default cancer / genomics / drug-discovery query",
)

st.sidebar.markdown("---")
max_items = st.sidebar.slider("Max articles", 10, 100, 40, step=10)

st.sidebar.markdown("---")
min_relevance = st.sidebar.slider(
    "Min relevance score", 0.0, 1.0, 0.0, step=0.05,
    help="Higher values surface articles most related to Cognisom's focus areas",
)

force_refresh = st.sidebar.button("🔄  Refresh Now")

# ── Fetch / cache ───────────────────────────────────────────────────


@st.cache_data(ttl=3600, show_spinner="Fetching latest research…")
def _fetch(query: str, n: int, _ts: str):
    """Wrapper so Streamlit can cache the result for 1 hour.

    ``_ts`` is a timestamp string rounded to the hour so that
    pressing *Refresh Now* busts the cache.
    """
    return feed.fetch_all(max_items=n, query=query or None)


# Round to hour for normal caching; force_refresh bumps to current second
cache_ts = datetime.utcnow().strftime("%Y-%m-%d-%H") if not force_refresh else datetime.utcnow().isoformat()

items = _fetch(keyword, max_items, cache_ts)

# ── Post-fetch filtering ────────────────────────────────────────────

# Filter by enabled sources
items = [it for it in items if sources_enabled.get(it.source, True)]

# Filter by minimum relevance
if min_relevance > 0:
    items = [it for it in items if it.relevance_score >= min_relevance]

# ── Summary metrics ─────────────────────────────────────────────────

cols = st.columns(4)
cols[0].metric("Total articles", len(items))
source_counts = {}
for it in items:
    source_counts[it.source] = source_counts.get(it.source, 0) + 1
cols[1].metric("PubMed", source_counts.get("PubMed", 0))
cols[2].metric("bioRxiv", source_counts.get("bioRxiv", 0))
cols[3].metric("arXiv", source_counts.get("arXiv", 0))

st.markdown("---")

# ── Source colour badge helper ──────────────────────────────────────

_BADGE_COLOURS = {
    "PubMed": "#2196F3",
    "bioRxiv": "#E91E63",
    "arXiv": "#FF9800",
}


def _badge(source: str) -> str:
    colour = _BADGE_COLOURS.get(source, "#607D8B")
    return (
        f'<span style="background:{colour};color:#fff;padding:2px 8px;'
        f'border-radius:10px;font-size:0.8em;font-weight:600">{source}</span>'
    )


def _relevance_bar(score: float) -> str:
    pct = int(score * 100)
    colour = "#4CAF50" if pct >= 60 else "#FF9800" if pct >= 30 else "#9E9E9E"
    return (
        f'<div style="background:#e0e0e0;border-radius:4px;height:8px;width:120px;display:inline-block">'
        f'<div style="background:{colour};height:8px;border-radius:4px;width:{pct}%"></div></div>'
        f' <span style="font-size:0.8em">{pct}%</span>'
    )


# ── Render feed ─────────────────────────────────────────────────────

if not items:
    st.info("No articles match your filters. Try broadening the keyword or lowering the relevance threshold.")
else:
    for idx, item in enumerate(items):
        with st.container():
            # Header row: badge + title
            st.markdown(
                f"{_badge(item.source)}  **{item.title}**",
                unsafe_allow_html=True,
            )

            # Meta row
            authors_str = ", ".join(item.authors[:4])
            if len(item.authors) > 4:
                authors_str += f" + {len(item.authors) - 4} more"
            meta_parts = []
            if authors_str:
                meta_parts.append(authors_str)
            if item.published_date:
                meta_parts.append(f"📅 {item.published_date}")
            if item.doi:
                meta_parts.append(f"DOI: `{item.doi}`")
            st.caption(" · ".join(meta_parts))

            # Relevance bar
            st.markdown(
                f"Relevance: {_relevance_bar(item.relevance_score)}",
                unsafe_allow_html=True,
            )

            # Tags
            if item.tags:
                tag_html = " ".join(
                    f'<span style="background:#e3f2fd;color:#1565C0;padding:1px 6px;'
                    f'border-radius:8px;font-size:0.75em;margin-right:4px">{t}</span>'
                    for t in item.tags[:8]
                )
                st.markdown(tag_html, unsafe_allow_html=True)

            # Expandable abstract
            with st.expander("Abstract / Summary"):
                st.write(item.summary if item.summary else "_No abstract available._")
                if item.url:
                    st.markdown(f"[Open full article →]({item.url})")

            st.markdown("---")

# ── Export ───────────────────────────────────────────────────────────

if items:
    import json

    export_data = [
        {
            "title": it.title,
            "authors": it.authors,
            "source": it.source,
            "published_date": it.published_date,
            "url": it.url,
            "doi": it.doi,
            "tags": it.tags,
            "relevance_score": round(it.relevance_score, 3),
            "summary": it.summary,
        }
        for it in items
    ]

    st.download_button(
        "📥  Export feed as JSON",
        data=json.dumps(export_data, indent=2),
        file_name=f"cognisom_research_feed_{datetime.utcnow():%Y%m%d}.json",
        mime="application/json",
    )

# Footer
from cognisom.dashboard.footer import render_footer
render_footer()
