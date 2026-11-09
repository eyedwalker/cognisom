#!/usr/bin/env python3
"""A crawlable, JavaScript-free copy of the capability audit.

Why this exists
---------------
``/?page=roadmap`` is a Streamlit page, and Streamlit renders entirely in
the browser. Measured on the live site: a plain fetch of that URL returns
10,626 bytes whose whole content is ``<title>Streamlit</title>`` and a
``<noscript>`` tag. Not one capability, not one number, and not even the
page's own name. So the audit is invisible to search engines, to anything
fetching it without JavaScript, and to an archive -- for a document whose
entire purpose is to be read, cited and linked, that is an own goal.

This writes a standalone HTML file with the same content, plus a
``robots.txt`` and a ``sitemap.xml`` (the live ones currently return
Streamlit's JavaScript shell, so a crawler gets HTML where it asked for
plain text and ignores it).

Why it cannot drift from the live page
--------------------------------------
Both are generated from ``cognisom.dashboard.capabilities``. There is no
second copy of the content to fall out of date, and
``tests/test_the_static_snapshot_matches_the_live_audit.py`` asserts that
every row, note and count present in the module reaches this file.

Usage
-----
    python scripts/build_static_snapshot.py [--out DIR] [--base-url URL]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import html
import os
import subprocess
import sys
from pathlib import Path

# Run as a script from a checkout, like the other entries in scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognisom.dashboard.capabilities import (  # noqa: E402
    ABSENT, BUILT, LAYERS, OUTCOME_MODEL_NOTE, PARTIAL, SCALE_NOTE,
    STATUS_MEANING, TARGETS, counts,
)

REPO_URL = "https://github.com/eyedwalker/cognisom"
DEFAULT_BASE = "https://cognisom.com"
#: Where the static copy is served from. Kept as one constant because the
#: sitemap, the canonical link and the tests all have to agree on it.
SNAPSHOT_PATH = "/audit.html"

_LABEL = {BUILT: "Built", PARTIAL: "Partial", ABSENT: "Not yet"}
_CLASS = {BUILT: "built", PARTIAL: "partial", ABSENT: "absent"}

STYLE = """
:root{--bg:#fbfaf8;--fg:#1b1a18;--muted:#5d5a55;--rule:#e2ddd5;
--built:#1d6b3f;--partial:#8a5a10;--absent:#6a6560;--link:#1a4fa0;
--card:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#15161a;--fg:#ececea;
--muted:#a6a39d;--rule:#2d2f36;--built:#6fcf97;--partial:#e0a84a;
--absent:#9a958e;--link:#8ab4f8;--card:#1c1e24}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
-webkit-text-size-adjust:100%}
main{max-width:62rem;margin:0 auto;padding:2.5rem max(1rem,4vw) 5rem}
h1{font-size:clamp(1.7rem,4.4vw,2.6rem);line-height:1.15;margin:0 0 .6rem;
letter-spacing:-.02em;text-wrap:balance}
h2{font-size:clamp(1.25rem,2.6vw,1.6rem);margin:3rem 0 .3rem;
padding-top:1.4rem;border-top:1px solid var(--rule);letter-spacing:-.01em}
h3{font-size:1.05rem;margin:2rem 0 .2rem}
p,li{color:var(--fg)}
.lede{font-size:1.1rem;color:var(--muted);max-width:46rem}
.note{background:var(--card);border:1px solid var(--rule);
border-left:3px solid var(--partial);padding:1rem 1.1rem;margin:1.5rem 0;
border-radius:.35rem}
.counts{list-style:none;padding:0;margin:1.2rem 0}
.counts li{margin:.35rem 0}
.q{color:var(--muted);font-style:italic;margin:.2rem 0 1rem}
table{width:100%;border-collapse:collapse;margin:.8rem 0 2rem;font-size:.94rem}
th,td{text-align:left;vertical-align:top;padding:.6rem .7rem;
border-bottom:1px solid var(--rule)}
th{font-size:.78rem;text-transform:uppercase;letter-spacing:.06em;
color:var(--muted);font-weight:600}
td.cap{width:30%}td.ev{width:22%;font-size:.85rem}
.badge{display:inline-block;font-size:.72rem;font-weight:700;
text-transform:uppercase;letter-spacing:.05em;padding:.1rem .4rem;
border-radius:.25rem;border:1px solid currentColor;margin-bottom:.3rem}
.built{color:var(--built)}.partial{color:var(--partial)}.absent{color:var(--absent)}
a{color:var(--link);text-decoration-thickness:1px;text-underline-offset:2px}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.88em;
background:var(--card);border:1px solid var(--rule);border-radius:.2rem;
padding:.05rem .3rem}
footer{margin-top:3rem;padding-top:1.2rem;border-top:1px solid var(--rule);
color:var(--muted);font-size:.86rem}
.vids{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,18rem),1fr));
gap:1.2rem;margin:1.5rem 0}
.vids figure{margin:0}
.vids video{width:100%;height:auto;border-radius:.3rem;border:1px solid var(--rule)}
.vids figcaption{font-size:.85rem;color:var(--muted);margin-top:.4rem}
@media (max-width:40rem){
  table,thead,tbody,th,td,tr{display:block}
  thead{display:none}
  tr{border-bottom:1px solid var(--rule);padding:.7rem 0}
  td{border:0;padding:.15rem 0}
  td.cap,td.ev{width:auto}
  td.ev{font-size:.82rem;color:var(--muted)}
}
"""


def _git_sha() -> str:
    sha = os.environ.get("COGNISOM_GIT_SHA", "")
    if sha and sha != "dev":
        return sha
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], text=True,
                              capture_output=True, timeout=20).stdout.strip()
    except Exception:                                        # noqa: BLE001
        return ""


def _md_to_html(text: str) -> str:
    """The small subset of Markdown these notes use: **bold** and `code`.

    Written out rather than pulled in, because one dependency for two
    constructs is not worth it, and the notes are authored in this file's
    own repository so the subset is known.
    """
    out, bold = [], False
    i = 0
    esc = html.escape(text)
    while i < len(esc):
        if esc.startswith("**", i):
            out.append("</strong>" if bold else "<strong>")
            bold = not bold
            i += 2
        elif esc[i] == "`":
            end = esc.find("`", i + 1)
            if end == -1:
                out.append("`")
                i += 1
            else:
                out.append("<code>" + esc[i + 1:end] + "</code>")
                i = end + 1
        else:
            out.append(esc[i])
            i += 1
    return "".join(out)


def _evidence_links(paths, sha: str) -> str:
    if not paths:
        return "&mdash;"
    ref = sha or "main"
    return ", ".join(
        f'<a href="{REPO_URL}/blob/{ref}/{html.escape(p)}">'
        f'{html.escape(Path(p).name)}</a>' for p in paths)


def build_html(base_url: str, sha: str) -> str:
    n = counts()
    today = _dt.date.today().isoformat()
    short = sha[:7] if sha else "working tree"

    parts = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        "<title>What Cognisom simulates, and what it can&rsquo;t yet "
        "&mdash; capability audit</title>",
        '<meta name="description" content="A row-by-row audit of what the '
        'Cognisom molecular digital twin simulates today: '
        f'{n[BUILT]} capabilities built and held by a named test, '
        f'{n[PARTIAL]} partial, {n[ABSENT]} not yet modelled, each linked to '
        'the test in a public Apache-2.0 repository.">',
        f'<link rel="canonical" href="{base_url}{SNAPSHOT_PATH}">',
        f"<style>{STYLE}</style>",
        "</head><body><main>",
        "<h1>What Cognisom simulates, and what it can&rsquo;t yet</h1>",
        '<p class="lede">The goal is a simulation of human tissue that '
        "responds to cancer and to immunotherapy the way a real "
        "patient&rsquo;s tumour does, built from the patient&rsquo;s own "
        "genome, cell by cell, with every number traceable to a "
        "measurement. This is the honest distance between that goal and "
        "what runs today.</p>",
        '<p class="q">Cognisom is a research platform. Nothing here is a '
        "clinical finding, a diagnosis, or a treatment recommendation.</p>",
        '<p><strong>This is a static, JavaScript-free copy</strong> of the '
        f'audit at <a href="{base_url}/?page=roadmap">{base_url}'
        "/?page=roadmap</a>, generated from the same source so the two "
        "cannot disagree. The interactive page carries the videos and the "
        "Bio-USD documentation.</p>",
    ]

    # videos, by reference: the static copy stays light and still shows them
    parts += [
        '<div class="vids">',
        '<figure><video controls preload="metadata" '
        f'src="{base_url}/media/demo_run.mp4"></video>'
        "<figcaption><strong>One run, plotted.</strong> Twenty-four "
        "simulated hours of a tumour and its immune infiltrate, with the "
        "counts beside it.</figcaption></figure>",
        '<figure><video controls preload="metadata" '
        f'src="{base_url}/media/rtx_demo.mp4"></video>'
        "<figcaption><strong>The same run, ray-traced</strong> in NVIDIA "
        "Isaac Sim from the Bio-USD file the run exported."
        "</figcaption></figure>",
        "</div>",
        '<p class="q">If the players above are empty, the videos are served '
        f'by the interactive page: <a href="{base_url}/?page=demo">'
        f"{base_url}/?page=demo</a>.</p>",
    ]

    parts += [
        "<h2>Where it stands</h2>",
        '<ul class="counts">',
        f'<li><span class="badge built">Built</span> <strong>{n[BUILT]}</strong>'
        f" &mdash; {html.escape(STATUS_MEANING[BUILT])}</li>",
        f'<li><span class="badge partial">Partial</span> <strong>{n[PARTIAL]}</strong>'
        f" &mdash; {html.escape(STATUS_MEANING[PARTIAL])}</li>",
        f'<li><span class="badge absent">Not yet</span> <strong>{n[ABSENT]}</strong>'
        f" &mdash; {html.escape(STATUS_MEANING[ABSENT])}</li>",
        "</ul>",
        "<p><strong>Built is not validated, and these are not comparable "
        "counts.</strong> Whether the simulation <em>predicts</em> anything "
        "is a separate question, answered under Validation below: "
        "<strong>one</strong> falsifiable prediction has held against "
        "outside human data, and one pre-specified outcome test was "
        "null.</p>",
        f'<div class="note">{_md_to_html(OUTCOME_MODEL_NOTE)}</div>',
        "<h2>What we are building toward</h2>",
        '<p class="q">In order, with what each one needs first. The '
        "sequencing is the plan; the tables below are the inventory it is "
        "drawn from.</p>",
        "<ol>",
    ]
    for title, body in TARGETS:
        parts.append(f"<li><strong>{html.escape(title)}.</strong> "
                     f"{_md_to_html(body)}</li>")
    parts += ["</ol>", f"<p>{_md_to_html(SCALE_NOTE)}</p>",
              "<h2>Everything, layer by layer</h2>",
              '<p class="q">Each built row links to the test, or the '
              "analysis script, in the public repository that holds it. A "
              "test fails the build if a built row cites nothing, or cites "
              "a file that defines no test.</p>"]

    for layer in LAYERS:
        parts.append(f"<h3>{html.escape(layer.title)}</h3>")
        parts.append(f'<p class="q">{html.escape(layer.question)}</p>')
        parts.append("<table><thead><tr><th>Capability</th>"
                     "<th>What it does, and its limits</th>"
                     "<th>Evidence</th></tr></thead><tbody>")
        for row in layer.rows:
            parts.append(
                "<tr>"
                f'<td class="cap"><span class="badge {_CLASS[row.status]}">'
                f'{_LABEL[row.status]}</span><br><strong>'
                f"{html.escape(row.name)}</strong></td>"
                f"<td>{_md_to_html(row.note) if row.note else '&mdash;'}</td>"
                f'<td class="ev">{_evidence_links(row.evidence, sha)}</td>'
                "</tr>")
        parts.append("</tbody></table>")

    commit = (f'<a href="{REPO_URL}/commit/{sha}"><code>{short}</code></a>'
              if sha else "<code>working tree</code>")
    parts += [
        "<footer>",
        f"<p>Generated from <code>cognisom/dashboard/capabilities.py</code> "
        f"at commit {commit} on {today}, in "
        f'<a href="{REPO_URL}">github.com/eyedwalker/cognisom</a> '
        "(Apache&nbsp;2.0). Every built row&rsquo;s evidence links to the "
        "test that holds it at that commit, so this audit can be "
        "re-checked, and its movement over time can be read from the "
        "repository&rsquo;s history.</p>",
        f'<p><a href="{base_url}/">Cognisom</a> &middot; '
        f'<a href="{base_url}/?page=demo">recorded demo</a> &middot; '
        f'<a href="{base_url}/?page=biousd">Bio-USD standard</a></p>',
        "</footer></main></body></html>",
    ]
    return "\n".join(parts)


def build_robots(base_url: str) -> str:
    return (
        "# cognisom.com -- research platform, not for clinical use.\n"
        "# The application is a client-rendered Streamlit app, so a\n"
        "# crawler gets no content from it. The capability audit is\n"
        f"# published as a static page at {SNAPSHOT_PATH}.\n"
        "User-agent: *\n"
        "Allow: /\n"
        f"Allow: {SNAPSHOT_PATH}\n"
        "# Nothing useful is served under these, and they need an account.\n"
        "Disallow: /_stcore/\n"
        "Disallow: /media/\n"
        f"\nSitemap: {base_url}/sitemap.xml\n"
    )


def build_sitemap(base_url: str) -> str:
    today = _dt.date.today().isoformat()
    urls = [(SNAPSHOT_PATH, "1.0"), ("/", "0.8"),
            ("/?page=demo", "0.6"), ("/?page=biousd", "0.6")]
    body = "\n".join(
        f"  <url><loc>{html.escape(base_url + p)}</loc>"
        f"<lastmod>{today}</lastmod>"
        f"<priority>{pr}</priority></url>" for p, pr in urls)
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
            f"{body}\n</urlset>\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="static",
                    help="directory to write into (default: static/)")
    ap.add_argument("--base-url", default=DEFAULT_BASE)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sha = _git_sha()

    written = []
    for name, text in (("audit.html", build_html(args.base_url, sha)),
                       ("robots.txt", build_robots(args.base_url)),
                       ("sitemap.xml", build_sitemap(args.base_url))):
        path = out / name
        path.write_text(text)
        written.append((name, len(text)))

    n = counts()
    print(f"commit {sha[:7] or 'working tree'}; "
          f"{n[BUILT]} built / {n[PARTIAL]} partial / {n[ABSENT]} not yet")
    for name, size in written:
        print(f"  {out / name}  {size:,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
