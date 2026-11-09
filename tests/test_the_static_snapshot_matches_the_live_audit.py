"""The crawlable copy of the audit says the same thing as the live page.

The capability audit is a Streamlit page, and Streamlit renders in the
browser. MEASURED on the live site: a plain fetch of `/?page=roadmap`
returns 10,626 bytes whose entire content is `<title>Streamlit</title>`
and a `<noscript>` tag -- no capability, no number, not even the page's
own name. `robots.txt` and `sitemap.xml` return that same shell, so a
crawler asking for plain text gets HTML and discards it. For a document
whose purpose is to be read, cited and linked, that is an own goal, and
an outside reviewer hit it before reading a word of the content.

`scripts/build_static_snapshot.py` writes a standalone copy. The risk a
second copy introduces is DRIFT -- a mirror that quietly stops matching
is worse than no mirror -- so both are generated from
`cognisom.dashboard.capabilities` and this file asserts that everything
in the module reaches the file, that the counts agree, and that the
things a crawler actually reads are present and correct.
"""

from __future__ import annotations

import html as _html
import re
import subprocess
import sys
from pathlib import Path

import pytest

from cognisom.dashboard.capabilities import (
    ABSENT, BUILT, LAYERS, OUTCOME_MODEL_NOTE, PARTIAL, SCALE_NOTE, TARGETS,
    all_rows, counts,
)

REPO = Path(__file__).resolve().parent.parent
BUILDER = REPO / "scripts" / "build_static_snapshot.py"
BASE = "https://cognisom.com"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Run the real generator, so the test exercises what ships."""
    out = tmp_path_factory.mktemp("static")
    done = subprocess.run(
        [sys.executable, str(BUILDER), "--out", str(out), "--base-url", BASE],
        cwd=REPO, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stdout + done.stderr
    return {p.name: p.read_text() for p in out.iterdir()}


def test_the_generator_writes_the_three_files_a_crawler_needs(built):
    assert set(built) == {"audit.html", "robots.txt", "sitemap.xml"}


# ── no drift from the module ─────────────────────────────────────────

def test_every_capability_reaches_the_static_copy(built):
    text = _html.unescape(built["audit.html"])
    missing = [row.name for _, row in all_rows() if row.name not in text]
    assert not missing, f"absent from the static audit: {missing}"


def test_every_layer_and_its_question_reach_it(built):
    text = _html.unescape(built["audit.html"])
    for layer in LAYERS:
        assert layer.title in text, layer.title
        assert layer.question in text, layer.question


def test_the_counts_agree_with_the_module(built):
    n = counts()
    text = built["audit.html"]
    for status in (BUILT, PARTIAL, ABSENT):
        assert f"<strong>{n[status]}</strong>" in text, status
    # and the total is every row, so no row was silently dropped
    assert sum(n.values()) == len(all_rows())


def test_the_targets_keep_their_order(built):
    """The sequencing IS the plan, so a mirror that reorders it misstates
    the plan even if every item is present."""
    text = _html.unescape(built["audit.html"])
    # Scoped to the ordered list, because several target titles also name
    # a layer ("Scale and time", "Validation") and one is mentioned in the
    # counts paragraph above it -- a bare index() would find that first
    # and report an ordering fault that is not there.
    start = text.index("What we are building toward")
    end = text.index("Everything, layer by layer")
    section = text[start:end]
    positions = [section.index(title) for title, _ in TARGETS]
    assert positions == sorted(positions), (
        "the static copy lists the goals in a different order than the "
        "module does")


def test_the_two_standing_caveats_are_in_the_static_copy(built):
    """The outcome model is a separate model, and built is not validated.
    Both are the kind of thing a mirror drops first."""
    text = _html.unescape(built["audit.html"])
    assert "does not call the cell-level engine" in text
    assert "Built is not validated" in text
    for fragment in ("second, separate model", "2.1 seconds per simulated"):
        assert fragment in text, fragment
    # the notes are rendered, not pasted with their markup showing
    assert "**" not in text.replace("**bold**", "")


def test_every_built_row_links_its_evidence_at_a_commit(built):
    text = built["audit.html"]
    for _, row in all_rows():
        for path in row.evidence:
            assert f"/blob/" in text and Path(path).name in text, path
    # pinned to a commit, not to a moving branch, so a reader checking
    # the claim sees the file as it was when the claim was made
    assert re.search(r"/blob/[0-9a-f]{40}/", text), (
        "evidence links are not pinned to a commit")


# ── what a crawler actually reads ────────────────────────────────────

def test_the_page_has_a_real_title_and_description(built):
    text = built["audit.html"]
    title = re.search(r"<title>(.*?)</title>", text, re.S)
    assert title and "Streamlit" not in title.group(1)
    assert "Cognisom" in title.group(1)
    desc = re.search(r'<meta name="description" content="(.*?)"', text, re.S)
    assert desc and 80 < len(desc.group(1)) < 320, (
        "a description that short or long is not useful in a result list")


def test_it_needs_no_javascript_and_no_network(built):
    """The whole point. A copy that itself needs a script, or pulls a
    stylesheet from a CDN, fails for the reader it exists for."""
    text = built["audit.html"]
    assert "<script" not in text.lower()
    assert "<noscript" not in text.lower()
    for host in ("http://", "cdn.", "googleapis", "unpkg"):
        body = text.split("</head>", 1)[-1]
        assert host not in body.replace("https://", ""), host
    assert "<style>" in text, "the styling must be inline, not fetched"


def test_it_declares_itself_a_copy_and_points_at_the_live_page(built):
    text = _html.unescape(built["audit.html"])
    assert "static" in text and "?page=roadmap" in text
    assert f'rel="canonical"' in built["audit.html"]


def test_it_is_legible_on_a_phone(built):
    text = built["audit.html"]
    assert 'name="viewport"' in text
    # the tables collapse rather than scroll sideways at phone width
    assert "@media (max-width:40rem)" in text


def test_robots_is_plain_text_that_names_the_sitemap(built):
    robots = built["robots.txt"]
    assert not robots.lstrip().startswith("<"), (
        "robots.txt must not be HTML -- serving the app shell here is the "
        "bug this replaces")
    assert "User-agent: *" in robots
    assert f"Sitemap: {BASE}/sitemap.xml" in robots
    assert "/audit.html" in robots
    # the websocket endpoint and the media route are not for crawlers
    assert "Disallow: /_stcore/" in robots


def test_the_sitemap_is_valid_xml_and_leads_with_the_audit(built):
    import xml.etree.ElementTree as ET

    root = ET.fromstring(built["sitemap.xml"])
    ns = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    locs = [u.findtext("s:loc", namespaces=ns) for u in root.findall("s:url", ns)]
    assert locs and locs[0] == f"{BASE}/audit.html"
    assert f"{BASE}/" in locs
    for loc in locs:
        assert loc.startswith(BASE)


def test_the_disclaimer_survives_into_the_static_copy(built):
    """Research-use-only is not optional, and a mirror is exactly where
    it goes missing."""
    text = _html.unescape(built["audit.html"])
    assert "research platform" in text
    assert "not a clinical finding" in text or "Nothing here is a clinical" in text
