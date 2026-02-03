"""Subscriptions – manage journal, forum & database feeds."""

import sys
import json
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Subscriptions | Cognisom", page_icon="📬", layout="wide")

from cognisom.auth.middleware import streamlit_page_gate
user = streamlit_page_gate("8_subscriptions")

from cognisom.research.subscriptions import SubscriptionManager, SubscriptionSource

# ── Initialise manager ──────────────────────────────────────────────

CACHE_DIR = Path(_project_root) / "data" / "subscriptions"


@st.cache_resource
def get_manager() -> SubscriptionManager:
    return SubscriptionManager(cache_dir=str(CACHE_DIR))


mgr = get_manager()

# ── Page header ─────────────────────────────────────────────────────

st.title("Subscriptions")
st.markdown(
    "Monitor **journals**, **preprints**, **forums**, **databases**, and **blogs** — "
    "all relevant to Cognisom's focus on prostate cancer, drug discovery, genomics & AI."
)

# ── Tabs ────────────────────────────────────────────────────────────

tab_feed, tab_manage, tab_add = st.tabs(["Feed", "Manage Sources", "Add Custom Source"])

# ====================================================================
# TAB 1: Live Feed
# ====================================================================

with tab_feed:
    st.subheader("Latest from your subscriptions")

    # Sidebar filters
    st.sidebar.header("Subscription Filters")
    categories = mgr.categories()
    cat_filter = {}
    for cat in categories:
        cat_filter[cat] = st.sidebar.checkbox(cat.title(), value=True, key=f"cat_{cat}")

    keyword_filter = st.sidebar.text_input(
        "Keyword",
        placeholder="e.g. prostate, BRCA1, docking",
        key="sub_kw",
    )

    max_items = st.sidebar.slider("Max items per source", 5, 50, 15, key="sub_max")
    force_refresh = st.sidebar.button("Refresh Now")

    # Fetch
    @st.cache_data(ttl=3600, show_spinner="Fetching subscriptions…")
    def _fetch_subs(max_per: int, _ts: str):
        entries = mgr.fetch_all(max_per_source=max_per)
        mgr.save_cache(entries)
        return entries

    cache_ts = datetime.utcnow().strftime("%Y-%m-%d-%H") if not force_refresh else datetime.utcnow().isoformat()

    # Try cache first for speed
    entries = mgr.get_cached(max_age_hours=1)
    if entries is None or force_refresh:
        entries = _fetch_subs(max_items, cache_ts)

    # Apply filters
    if entries:
        entries = [e for e in entries if cat_filter.get(e.category, True)]
        if keyword_filter:
            kw = keyword_filter.lower()
            entries = [
                e for e in entries
                if kw in e.title.lower() or kw in e.summary.lower()
                or any(kw in t.lower() for t in e.tags)
            ]

    # Metrics
    cols = st.columns(len(categories) + 1)
    cols[0].metric("Total", len(entries) if entries else 0)
    for i, cat in enumerate(categories):
        count = sum(1 for e in entries if e.category == cat) if entries else 0
        cols[i + 1].metric(cat.title(), count)

    st.markdown("---")

    # Category colour badges
    _CAT_COLORS = {
        "journal": "#1976D2",
        "preprint": "#E91E63",
        "forum": "#4CAF50",
        "database": "#FF9800",
        "conference": "#9C27B0",
        "blog": "#607D8B",
    }

    if not entries:
        st.info("No entries match your filters. Try enabling more categories or broadening the keyword.")
    else:
        for entry in entries:
            with st.container():
                color = _CAT_COLORS.get(entry.category, "#607D8B")
                badge = (
                    f'<span style="background:{color};color:#fff;padding:2px 8px;'
                    f'border-radius:10px;font-size:0.75em;font-weight:600">{entry.source_name}</span>'
                )
                st.markdown(f"{badge}  **{entry.title}**", unsafe_allow_html=True)

                meta = []
                if entry.authors:
                    meta.append(", ".join(entry.authors[:3]))
                if entry.published:
                    meta.append(f"Published: {entry.published}")
                if meta:
                    st.caption(" · ".join(meta))

                if entry.tags:
                    tag_html = " ".join(
                        f'<span style="background:#e3f2fd;color:#1565C0;padding:1px 6px;'
                        f'border-radius:8px;font-size:0.7em">{t}</span>'
                        for t in entry.tags[:5]
                    )
                    st.markdown(tag_html, unsafe_allow_html=True)

                with st.expander("Summary"):
                    st.write(entry.summary if entry.summary else "_No summary available._")
                    if entry.url:
                        st.markdown(f"[Open →]({entry.url})")

                st.markdown("---")

    # Export
    if entries:
        export = [
            {
                "title": e.title, "url": e.url, "source": e.source_name,
                "category": e.category, "published": e.published,
                "summary": e.summary, "authors": e.authors, "tags": e.tags,
            }
            for e in entries
        ]
        st.download_button(
            "Export feed as JSON",
            data=json.dumps(export, indent=2),
            file_name=f"subscriptions_{datetime.utcnow():%Y%m%d}.json",
            mime="application/json",
            key="dl_subs",
        )

# ====================================================================
# TAB 2: Manage Sources
# ====================================================================

with tab_manage:
    st.subheader("Manage Subscription Sources")
    st.markdown("Enable or disable individual sources. Changes are saved automatically.")

    for cat in mgr.categories():
        st.markdown(f"### {cat.title()}")
        cat_sources = [s for s in mgr.sources if s.category == cat]
        for src in cat_sources:
            col1, col2, col3 = st.columns([0.5, 3, 1])
            with col1:
                new_val = st.checkbox(
                    "Enable",
                    value=src.enabled,
                    key=f"toggle_{src.name}",
                    label_visibility="collapsed",
                )
                if new_val != src.enabled:
                    if new_val:
                        mgr.enable(src.name)
                    else:
                        mgr.disable(src.name)
            with col2:
                st.markdown(f"**{src.name}**")
                st.caption(src.description)
            with col3:
                st.caption(src.source_type.upper())

# ====================================================================
# TAB 3: Add Custom Source
# ====================================================================

with tab_add:
    st.subheader("Add Custom Source")
    st.markdown("Add any RSS/Atom feed or JSON API endpoint to your subscriptions.")

    with st.form("add_source_form"):
        name = st.text_input("Source name", placeholder="e.g. My Lab Blog")
        url = st.text_input("RSS/Atom/API URL", placeholder="https://example.com/feed.xml")
        category = st.selectbox("Category", ["journal", "preprint", "forum", "database", "conference", "blog"])
        source_type = st.selectbox("Feed type", ["rss", "atom", "json_api"])
        description = st.text_input("Description (optional)", placeholder="Brief description of this source")

        submitted = st.form_submit_button("Add Source", type="primary")
        if submitted:
            if not name or not url:
                st.error("Name and URL are required.")
            else:
                mgr.add_custom_source(SubscriptionSource(
                    name=name,
                    category=category,
                    url=url,
                    source_type=source_type,
                    description=description,
                ))
                st.success(f"Added **{name}** to subscriptions!")
                st.rerun()

# Footer
from cognisom.dashboard.footer import render_footer
render_footer()
