"""The deploy script reclaims disk on a SHARED box without touching anything
that is not a superseded cognisom image.

apps-server's 40 G root volume hosts other apps (genesis, guidance,
walkerconway, ...). On 2026-11-06 a pull died with "no space left on device"
at 95% full because every deploy left a ~4.3 G image behind and nothing ever
removed one. The fix prunes -- and a prune with the wrong keep-set on a shared
production box deletes someone's rollback, or another app's image.

These tests run the REAL script against fake `docker`, `df` and `aws`
executables placed first on PATH. The fixture is the box's actual state on
2026-11-07: a live container, its -prev rollback, a stopped container holding
an old image on purpose, a named rollback tag, and other apps' images that no
container references.
"""
from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "deploy" / "deploy-apps-server.sh"
W = "780457123717.dkr.ecr.us-west-2.amazonaws.com/cognisom"
E = "780457123717.dkr.ecr.us-east-1.amazonaws.com/cognisom"

# repo tag id size_gb
LIVE = ("cognisom", "latest", "sha256:live", 4)
LIVE_ECR = (W, "68a9ea8", "sha256:live", 4)
PREV = (W, "ee2d3cc", "sha256:prev", 4)              # held by -prev container
STALE = (W, "c5dc48c", "sha256:stale", 4)            # nothing holds it -> goes
HELD = (E, "b6e205ef", "sha256:held", 4)             # stopped preregion container
ROLLBACK = ("cognisom", "rollback-20260820", "sha256:rb", 3)
DANGLING = ("cognisom", "<none>", "sha256:dangling", 1)  # untagged cognisom -> goes
OTHER_APP = ("genesis-backend", "latest", "sha256:genesis", 2)   # unreferenced!
OTHER_ECR = ("780457123717.dkr.ecr.us-west-2.amazonaws.com/walkerconway", "old",
             "sha256:wc", 2)                          # unreferenced, ECR, not ours
NOT_QUITE = ("cognisom-tools", "latest", "sha256:tools", 1)  # name only CONTAINS it

CONTAINERS = [
    ("c-live", "sha256:live"),
    ("c-prev", "sha256:prev"),
    ("c-preregion", "sha256:held"),
    ("c-guidance", "sha256:guidance"),
]

EXPECTED_GONE = {f"{W}:c5dc48c", "sha256:dangling"}

DOCKER = r'''#!/bin/bash
F="$SHIM_FIXTURE"
echo "docker $*" >> "$F/calls.log"
cmd="$1"; shift
case "$cmd" in
  info) echo /var/lib/docker ;;
  # the real one reads the password from stdin; exiting without reading
  # SIGPIPEs `aws ... | docker login` under pipefail
  login) cat >/dev/null ;;
  ps) awk '{print $1}' "$F/containers.txt" ;;
  inspect)
    shift 2
    for c in "$@"; do awk -v c="$c" '$1==c {print $2}' "$F/containers.txt"; done ;;
  images)
    case "$3" in
      *':{{.Tag}}'*) awk '{print $1":"$2" "$3}' "$F/images.txt" ;;
      *) awk '{print $1" "$2" "$3}' "$F/images.txt" ;;
    esac ;;
  image)
    # the ref is the LAST argument: the script calls it with and without --format
    id=$(awk -v r="${!#}" '$1":"$2==r {print $3}' "$F/images.txt")
    [ -n "$id" ] || exit 1
    echo "$id" ;;
  rmi)
    for a in "$@"; do
      case "$a" in -f|--force) echo "FORCED-RMI" >> "$F/calls.log"; exit 3 ;; esac
    done
    ref="$1"
    id=$(awk -v r="$ref" '$1":"$2==r || $3==r {print $3; exit}' "$F/images.txt")
    [ -n "$id" ] || exit 1
    # real docker refuses to delete an image any container (even stopped) uses
    if awk -v i="$id" '$2==i {f=1} END {exit !f}' "$F/containers.txt"; then exit 1; fi
    size=$(awk -v i="$id" '$3==i {print $4; exit}' "$F/images.txt")
    awk -v r="$ref" '!($1":"$2==r || $3==r)' "$F/images.txt" > "$F/images.tmp"
    mv "$F/images.tmp" "$F/images.txt"
    if ! awk -v i="$id" '$3==i {f=1} END {exit !f}' "$F/images.txt"; then
      echo $(( $(cat "$F/free.txt") + size )) > "$F/free.txt"
    fi ;;
  *) : ;;
esac
'''

DF = r'''#!/bin/bash
echo "Filesystem 1G-blocks Used Available Capacity Mounted"
echo "/dev/nvme0n1p1 40G 38G $(cat "$SHIM_FIXTURE/free.txt")G 95% /"
'''


def _box(tmp_path: Path, free_gb: int, stale_size: int = 4):
    fx = tmp_path / "fx"
    bin_ = tmp_path / "bin"
    fx.mkdir()
    bin_.mkdir()
    stale = STALE[:3] + (stale_size,)
    images = [LIVE, LIVE_ECR, PREV, stale, HELD, ROLLBACK, DANGLING,
              OTHER_APP, OTHER_ECR, NOT_QUITE]
    (fx / "images.txt").write_text("".join(f"{r} {t} {i} {s}\n" for r, t, i, s in images))
    (fx / "containers.txt").write_text("".join(f"{c} {i}\n" for c, i in CONTAINERS))
    (fx / "free.txt").write_text(f"{free_gb}\n")
    (fx / "calls.log").write_text("")
    shims = {"docker": DOCKER, "df": DF,
             "aws": "#!/bin/bash\necho token\n",
             # step 1 creates and chowns /opt/cognisom; not this test's business
             "mkdir": "#!/bin/bash\nexit 0\n", "chown": "#!/bin/bash\nexit 0\n"}
    for name, body in shims.items():
        p = bin_ / name
        p.write_text(body)
        p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    env = dict(os.environ, PATH=f"{bin_}:{os.environ['PATH']}", SHIM_FIXTURE=str(fx))
    env.pop("COGNISOM_MIN_FREE_GB", None)
    return fx, env


def _run(env, *args):
    return subprocess.run(["bash", str(SCRIPT), *args], env=env,
                          capture_output=True, text=True, timeout=60)


def _calls(fx: Path) -> list[str]:
    return fx.joinpath("calls.log").read_text().splitlines()


def _removed(fx: Path) -> set[str]:
    return {c.split(" ", 2)[2] for c in _calls(fx) if c.startswith("docker rmi ")}


def test_a_dry_run_names_exactly_the_superseded_images_and_removes_nothing(tmp_path):
    fx, env = _box(tmp_path, free_gb=12)
    r = _run(env, "--reclaim-only", "--dry-run")
    assert r.returncode == 0, r.stdout + r.stderr
    named = set(re.findall(r"would remove (\S+)", r.stdout))
    assert named == EXPECTED_GONE, r.stdout
    assert _removed(fx) == set(), "a dry run called docker rmi"


def test_a_reclaim_removes_those_and_nothing_else(tmp_path):
    fx, env = _box(tmp_path, free_gb=2)
    r = _run(env, "--reclaim-only")
    assert r.returncode == 0, r.stdout + r.stderr
    assert _removed(fx) == EXPECTED_GONE, r.stdout
    assert "FORCED-RMI" not in _calls(fx)
    left = fx.joinpath("images.txt").read_text()
    for survivor, why in [
        (LIVE, "the live site"), (PREV, "the -prev rollback"),
        (HELD, "a stopped container someone kept"), (ROLLBACK, "a named rollback tag"),
        (OTHER_APP, "another app's image"), (OTHER_ECR, "another app's ECR image"),
        (NOT_QUITE, "a repo whose name merely contains 'cognisom'"),
    ]:
        assert f"{survivor[0]} {survivor[1]} " in left, f"deleted {why}: {survivor}"


def test_a_short_disk_aborts_before_the_running_site_is_touched(tmp_path):
    # 2 G free, and the only reclaimable cognisom image is 4 G -> 6 G < 8 G
    fx, env = _box(tmp_path, free_gb=2, stale_size=4)
    r = _run(env, "--pull", "deadbeef")
    assert r.returncode != 0
    assert "has NOT been touched" in r.stdout
    calls = _calls(fx)
    for verb in ("docker pull", "docker stop", "docker rename", "docker run", "docker build"):
        assert not any(c.startswith(verb) for c in calls), f"{verb} ran on a short disk"
    assert f"docker rmi {W}:c5dc48c" in calls, "it gave up without trying to reclaim"


def test_a_short_disk_is_reclaimed_before_the_pull(tmp_path):
    # 2 G free, but removing the stale image frees 10 G -> 12 G, enough
    fx, env = _box(tmp_path, free_gb=2, stale_size=10)
    r = _run(env, "--pull", "deadbeef", "--no-restart")
    assert r.returncode == 0, r.stdout + r.stderr
    calls = _calls(fx)
    pull = next(i for i, c in enumerate(calls) if c.startswith("docker pull"))
    rmi = calls.index(f"docker rmi {W}:c5dc48c")
    assert rmi < pull, "reclaimed AFTER pulling -- the pull would already have failed"


def test_a_dry_run_of_a_deploy_is_refused(tmp_path):
    fx, env = _box(tmp_path, free_gb=12)
    r = _run(env, "--pull", "deadbeef", "--dry-run")
    assert r.returncode != 0
    assert "only applies to --reclaim-only" in r.stdout
    assert _calls(fx) == [], "it did something before refusing"


@pytest.mark.parametrize("forbidden", [
    r"rmi\s+(-f|--force)",          # docker refusing is the last guard
    r"system\s+prune",              # other apps' images, volumes, networks
    r"image\s+prune",               # other apps' dangling layers
    r"builder\s+prune",             # other apps' build cache
])
def test_the_script_never_reaches_outside_cognisom(forbidden):
    code = "\n".join(l for l in SCRIPT.read_text().splitlines()
                     if not l.lstrip().startswith("#"))
    assert not re.search(forbidden, code), f"deploy script uses {forbidden!r}"


POLICY = SCRIPT.parent / "ecr-lifecycle-policy.json"


def test_the_ecr_policy_cannot_strip_the_image_from_under_a_tag():
    """Every tag is an OCI index; its real image is an UNTAGGED child. A preview
    on 2026-11-07 showed ECR excludes children of kept indexes from a
    COUNT-based untagged rule. That is the only untagged form measured, so it is
    the only one allowed -- an age-based untagged rule was not observed."""
    import json
    rules = json.loads(POLICY.read_text())["rules"]
    for r in rules:
        assert len(r["description"]) <= 255, "ECR rejects descriptions over 255 chars"
        sel = r["selection"]
        assert sel["tagStatus"] != "any", "an 'any' rule ignores tags entirely"
        if sel["tagStatus"] == "untagged":
            assert sel["countType"] == "imageCountMoreThan", (
                "only the count-based untagged rule was measured to spare index "
                "children; re-run a lifecycle PREVIEW before changing it")
    tagged = [r for r in rules if r["selection"]["tagStatus"] == "tagged"]
    assert tagged and all(r["selection"]["countNumber"] >= 5 for r in tagged), (
        "keep enough tagged images to roll back with --pull <sha>")
