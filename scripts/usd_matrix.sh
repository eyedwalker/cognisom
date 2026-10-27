#!/usr/bin/env bash
# Run the USD-touching tests under both OpenUSD versions, locally.
#
#   scripts/usd_matrix.sh                    # usd-core 26.11 (the venv's pin), then 25.11
#   scripts/usd_matrix.sh --list             # print the test files, one per line, and exit
#   scripts/usd_matrix.sh --only 25.11       # one side only (26.11 or 25.11)
#   scripts/usd_matrix.sh -- -x -k schema    # anything after -- goes to pytest
#
# WHY TWO. requirements.txt pins usd-core 26.11 (since 2026-10-27), which
# production installs, and the suite normally sees nothing else. Bio-USD is
# published for other people, some of whom are on an older OpenUSD: before
# 26.05 (PR #4015) validators cannot be registered from Python, and before
# 26.08 there is no UsdProfiles. The package must behave there -- reporting
# that its checks could not be registered rather than implying they ran --
# and only the older side exercises those guards. The nightly workflow's
# `usd-25` job does the same in CI.
#
# HOW THE SECOND USD GETS IN WITHOUT A SECOND ENVIRONMENT. The older
# usd-core is installed with `pip install --no-deps --target` into a
# directory that holds only `pxr`, and that directory is PREPENDED to
# PYTHONPATH, so `from pxr import Usd` resolves there while numpy, scipy
# and everything else still come from the venv. USD_OLD_SITE names an
# existing such directory; otherwise one is created under
# ${XDG_CACHE_HOME:-$HOME/.cache}/cognisom/usd-core-25.11 (one download).
#
# Nothing is written under the repository: pytest runs with
# -p no:cacheprovider and the logs go to a temporary directory whose path
# is printed at the end.

set -euo pipefail

cd "$(dirname "$0")/.."

# One list, used here and by .github/workflows/nightly-tests.yml (--list).
USD_TEST_FILES=(
    tests/test_biousd_writes_real_usd.py
    tests/test_a_run_scrubs_in_time.py
    tests/test_each_lineage_has_its_own_shape.py
    tests/test_an_exported_population_is_not_spheres.py
    tests/test_a_composed_scene_exports_as_real_usd.py
    tests/test_cell_usd_export.py
    tests/test_a_cut_scene_shows_the_lesion.py
    tests/test_usda_files_are_loadable.py
    tests/test_a_large_scene_stays_legible_and_drawable.py
    tests/test_morphology_usd.py
    tests/test_molecular_usd.py
    tests/test_every_entity_renders.py
    tests/test_cognisom_stages_carry_the_schema.py
    tests/test_biousd_is_publishable.py
    tests/test_biousd_api_schema.py
    tests/test_biousd_profiles.py
    tests/test_biousd_validators.py
    tests/test_biousd_reads_a_physicell_snapshot.py
)

#: The version the venv is pinned to (requirements.txt), run first.
USD_PINNED_VERSION="${USD_PINNED_VERSION:-26.11}"
#: The older one, fetched into its own pxr-only directory.
USD_OLD_VERSION="${USD_OLD_VERSION:-25.11}"
USD_OLD_SITE="${USD_OLD_SITE:-${XDG_CACHE_HOME:-$HOME/.cache}/cognisom/usd-core-$USD_OLD_VERSION}"
if [ -z "${PYTHON:-}" ]; then
    if [ -x ./venv/bin/python ]; then PYTHON=./venv/bin/python; else PYTHON=python3; fi
fi

only=""
pytest_args=()
while [ $# -gt 0 ]; do
    case "$1" in
        --list)
            printf '%s\n' "${USD_TEST_FILES[@]}"
            exit 0 ;;
        --only)
            only="$2"; shift 2 ;;
        --)
            shift; pytest_args=("$@"); break ;;
        -h|--help)
            sed -n '2,30p' "$0"; exit 0 ;;
        *)
            echo "usd_matrix.sh: unknown argument $1 (try --help)" >&2; exit 2 ;;
    esac
done
case "$only" in
    ""|"$USD_PINNED_VERSION"|"$USD_OLD_VERSION") ;;
    *) echo "usd_matrix.sh: --only takes $USD_PINNED_VERSION or $USD_OLD_VERSION" >&2; exit 2 ;;
esac

logdir="$(mktemp -d "${TMPDIR:-/tmp}/usd_matrix.XXXXXX")"
declare -a summary=()
status=0

usd_version() {
    "$PYTHON" -c "from pxr import Usd; print('.'.join(map(str, Usd.GetVersion()[1:])))"
}

run_side() {
    # $1: label, remaining: environment assignments for this run
    local label="$1"; shift
    local log="$logdir/$label.log"
    local found
    found="$(env "$@" "$PYTHON" -c "from pxr import Usd; print('.'.join(map(str, Usd.GetVersion()[1:])))")"
    echo "== usd-core $label (pxr reports $found) =="
    if [ "$found" != "$label" ]; then
        echo "usd_matrix.sh: expected usd-core $label on this side, pxr reports $found" >&2
        summary+=("usd-core $label: NOT RUN (pxr reports $found)")
        status=1
        return
    fi
    # ${arr[@]+"${arr[@]}"}: an EMPTY array expanded as "${arr[@]}" is an
    # "unbound variable" under set -u on bash 3.2, which is what macOS
    # ships; this idiom expands to nothing there and to the elements
    # elsewhere.
    set +e
    env "$@" COGNISOM_ALLOW_PWM_FALLBACK=1 "$PYTHON" -m pytest -q -p no:cacheprovider \
        "${USD_TEST_FILES[@]}" ${pytest_args[@]+"${pytest_args[@]}"} 2>&1 | tee "$log"
    local rc="${PIPESTATUS[0]}"
    set -e
    local line
    line="$(grep -E '^[0-9]+ (passed|failed|error)|^no tests ran|^[0-9]+ (error|skipped)' "$log" | tail -n 1 || true)"
    [ -n "$line" ] || line="(no pytest summary line; see $log)"
    summary+=("usd-core $label: $line")
    [ "$rc" -eq 0 ] || status=1
}

if [ -z "$only" ] || [ "$only" = "$USD_PINNED_VERSION" ]; then
    run_side "$USD_PINNED_VERSION" PYTHONPATH="${PYTHONPATH:-}"
fi

if [ -z "$only" ] || [ "$only" = "$USD_OLD_VERSION" ]; then
    if [ ! -d "$USD_OLD_SITE/pxr" ]; then
        echo "== installing usd-core $USD_OLD_VERSION into $USD_OLD_SITE (pxr only, one download) =="
        "$PYTHON" -m pip install --quiet --disable-pip-version-check --no-deps \
            --target "$USD_OLD_SITE" "usd-core==$USD_OLD_VERSION"
    fi
    run_side "$USD_OLD_VERSION" PYTHONPATH="$USD_OLD_SITE${PYTHONPATH:+:$PYTHONPATH}"
fi

echo
echo "== summary (logs in $logdir) =="
printf '%s\n' ${summary[@]+"${summary[@]}"}
exit "$status"
