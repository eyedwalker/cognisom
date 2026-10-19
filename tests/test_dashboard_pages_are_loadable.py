"""Every page the navigation offers must at least parse.

``42_patent_pipeline.py`` shipped with ``from collections import Counter``
placed above ``from __future__ import annotations``, which Python rejects: a
``__future__`` import has to be the first statement after the docstring. The
page raised SyntaxError the moment a user clicked it, and nothing noticed,
because the dashboard pages are loaded by ``st.navigation`` at runtime and no
test imported them.

Parsing is a deliberately low bar -- these modules execute Streamlit calls at
import time, so actually importing them outside a Streamlit runtime is not
meaningful. But a page that cannot be parsed cannot be rendered under any
circumstances, and that is worth one cheap test over a surface of 43 pages.

The second test is the one that keeps this honest: it reads the page list out
of ``app.py`` so a page added to the navigation is covered automatically
rather than when somebody remembers to extend a list here.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PAGES_DIR = REPO / "cognisom" / "dashboard" / "_pages"
APP = REPO / "cognisom" / "dashboard" / "app.py"


def _declared_pages() -> list[tuple[str, Path]]:
    """(title, path) for every page app.py registers with st.Page."""
    src = APP.read_text()
    out = []
    for stem, title in re.findall(r'_pages_dir / "([^"]+\.py)"\), title="([^"]+)"', src):
        out.append((title, PAGES_DIR / stem))
    return out


def test_app_declares_pages_at_all():
    """Guards the regex: a silent zero would make the suite below vacuous."""
    declared = _declared_pages()

    assert len(declared) >= 40, (
        f"only {len(declared)} pages parsed out of app.py; the st.Page call "
        "shape has changed and these tests are no longer checking anything")


@pytest.mark.parametrize(
    "title,path", _declared_pages(), ids=[t for t, _ in _declared_pages()]
)
def test_a_navigable_page_parses(title: str, path: Path):
    assert path.exists(), f"{title} points at {path}, which does not exist"

    source = path.read_text()
    try:
        ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        pytest.fail(
            f"{title} ({path.name}) cannot be parsed, so clicking it raises "
            f"instead of rendering: line {exc.lineno}: {exc.msg}")


def test_future_imports_come_first(request):
    """The specific mistake, checked across every page at once.

    ast.parse already catches it, but this names the rule in the failure
    message, which is the difference between a two-minute fix and a puzzle.
    """
    offenders = []
    for path in sorted(PAGES_DIR.glob("*.py")):
        lines = path.read_text().splitlines()
        future_at = next((i for i, l in enumerate(lines)
                          if l.startswith("from __future__ import")), None)
        if future_at is None:
            continue
        for i, line in enumerate(lines[:future_at]):
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")) and "__future__" not in stripped:
                offenders.append(f"{path.name}: {stripped!r} on line {i + 1} "
                                 f"precedes the __future__ import on line {future_at + 1}")
                break

    assert not offenders, (
        "a __future__ import must be the first statement after the module "
        "docstring:\n  " + "\n  ".join(offenders))
