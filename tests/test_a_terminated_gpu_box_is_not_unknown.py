"""A GPU box that no longer exists says so, instead of saying nothing.

The box this project rendered on until September 2026 -- a g6e.2xlarge
with an L40S, ``i-0ac9eb88c1b046163``, on Elastic IP 52.32.247.131 --
has been terminated and its address released. The lifecycle code went on
asking EC2 about it, caught the resulting error, and reported "unknown".

"Unknown" and "gone" call for opposite responses. Unknown means wait and
retry; gone means no amount of waiting will help and something has to be
reconfigured. Reporting the second as the first is how a dead instance id
stayed in three source files without anyone noticing.
"""

from __future__ import annotations

import sys
import types

import pytest

from cognisom.infrastructure import gpu_connector


class _NotFound(Exception):
    """Shaped like botocore's ClientError for a missing instance."""

    def __str__(self):
        return ("An error occurred (InvalidInstanceID.NotFound) when calling "
                "the DescribeInstances operation: The instance ID "
                "'i-0ac9eb88c1b046163' does not exist")


def _fake_boto3(error=None, state="running"):
    """A boto3 stub. ``error`` is raised by describe_instances."""
    calls = []

    class _Client:
        def describe_instances(self, InstanceIds=None):      # noqa: N803
            calls.append(("describe", tuple(InstanceIds or ())))
            if error is not None:
                raise error
            return {"Reservations": [{"Instances": [
                {"State": {"Name": state}}]}]}

        def start_instances(self, InstanceIds=None):         # noqa: N803
            calls.append(("start", tuple(InstanceIds or ())))
            return {}

    module = types.ModuleType("boto3")
    module.client = lambda service, region_name=None: _Client()
    module._calls = calls
    return module


@pytest.fixture
def boto3_stub(monkeypatch):
    """Install a boto3 stub, and make sure we are not ON the GPU box."""
    monkeypatch.setattr(gpu_connector, "IS_GPU_INSTANCE", False)

    def install(**kwargs):
        module = _fake_boto3(**kwargs)
        monkeypatch.setitem(sys.modules, "boto3", module)
        return module

    return install


# ── the state ─────────────────────────────────────────────────────


def test_a_missing_instance_is_missing_not_unknown(boto3_stub):
    boto3_stub(error=_NotFound())
    assert gpu_connector.get_gpu_instance_state() == gpu_connector.STATE_MISSING
    assert gpu_connector.STATE_MISSING != "unknown"


def test_a_question_that_could_not_be_asked_is_still_unknown(boto3_stub):
    """No credentials, no network, no boto3: that IS unknown, and must
    not be reported as a terminated instance."""
    boto3_stub(error=RuntimeError("Unable to locate credentials"))
    assert gpu_connector.get_gpu_instance_state() == "unknown"


@pytest.mark.parametrize("state", ["running", "stopped", "pending", "stopping"])
def test_a_live_instance_reports_its_own_state(boto3_stub, state):
    boto3_stub(state=state)
    assert gpu_connector.get_gpu_instance_state() == state


# ── starting it ───────────────────────────────────────────────────


def test_starting_a_missing_instance_fails_and_says_what_to_change(boto3_stub):
    """The old code fell through to start_instances for anything that was
    not explicitly running or pending, so a terminated box produced an
    EC2 error rather than an answer."""
    module = boto3_stub(error=_NotFound())
    ok, message = gpu_connector.start_gpu_instance()
    assert ok is False
    assert "does not exist" in message
    assert "GPU_INSTANCE_ID" in message, "it names the setting to change"
    assert not any(call[0] == "start" for call in module._calls), (
        "it must not ask EC2 to start an instance that is gone")


def test_a_stopped_instance_is_actually_started(boto3_stub):
    module = boto3_stub(state="stopped")
    ok, message = gpu_connector.start_gpu_instance()
    assert ok is True
    assert any(call[0] == "start" for call in module._calls)
    assert "starting" in message.lower()


def test_a_running_instance_is_not_started_again(boto3_stub):
    module = boto3_stub(state="running")
    ok, _ = gpu_connector.start_gpu_instance()
    assert ok is True
    assert not any(call[0] == "start" for call in module._calls)


# ── one copy of the id ────────────────────────────────────────────


def test_only_gpu_connector_hardcodes_an_instance_id():
    """It was hardcoded in FIVE files, all naming the dead box, so fixing
    one left the rest pointing at nothing. Two of the five were found by
    this test and not by reading the code.

    This scans the source rather than comparing values, because Python
    interns short strings: two separate hardcoded copies of the same
    literal compare equal and even identical, and would pass a value
    check while remaining two copies.
    """
    import os
    import re

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    allowed = os.path.join("cognisom", "infrastructure", "gpu_connector.py")
    pattern = re.compile(r"""["']i-[0-9a-f]{8,}["']""")

    offenders = []
    for dirpath, dirnames, filenames in os.walk(os.path.join(root, "cognisom")):
        dirnames[:] = [d for d in dirnames
                       if d not in ("__pycache__", "node_modules")]
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            full = os.path.join(dirpath, filename)
            if full.endswith(allowed):
                continue
            with open(full, encoding="utf-8", errors="replace") as handle:
                for number, line in enumerate(handle, 1):
                    if pattern.search(line) and "GPU_INSTANCE_ID" in line:
                        offenders.append(f"{os.path.relpath(full, root)}:{number}")
    assert not offenders, (
        "these set GPU_INSTANCE_ID to a hardcoded instance id instead of "
        f"importing it from gpu_connector: {offenders}")


def test_only_the_configured_instance_is_ever_asked_about(boto3_stub):
    module = boto3_stub(state="stopped")
    gpu_connector.start_gpu_instance()
    asked = {call[1] for call in module._calls}
    assert asked == {(gpu_connector.GPU_INSTANCE_ID,)}


def test_the_id_is_configurable_without_editing_source():
    """The default is stale on purpose; the override is the supported way
    to point this at a live box."""
    import importlib
    import os

    original = os.environ.get("GPU_INSTANCE_ID")
    os.environ["GPU_INSTANCE_ID"] = "i-0123456789abcdef0"
    try:
        reloaded = importlib.reload(gpu_connector)
        assert reloaded.GPU_INSTANCE_ID == "i-0123456789abcdef0"
    finally:
        if original is None:
            del os.environ["GPU_INSTANCE_ID"]
        else:
            os.environ["GPU_INSTANCE_ID"] = original
        importlib.reload(gpu_connector)
