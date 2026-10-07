"""Command line entry point: version, compute, converge, replay, replay-dv,
replay-ls and help.

This layer contains only argument checking, file reading, JSON parsing and
JSON serialization.  All topology validation, protocol computation and
failure-state progression live in the I/O-free
:mod:`packet_routing_sim.core` package.
"""
import json
import sys

from . import __version__
from .core import (
    InvalidScenario,
    InvalidTopology,
    compute_topology,
    converge_topology,
    replay_dv_validated,
    replay_ls_validated,
    replay_validated,
    validate_topology,
)

COMPUTE_USAGE = "usage: python3 -m packet_routing_sim compute TOPOLOGY.json"
CONVERGE_USAGE = "usage: python3 -m packet_routing_sim converge TOPOLOGY.json"
REPLAY_USAGE = (
    "usage: python3 -m packet_routing_sim replay TOPOLOGY.json SCENARIO.json"
)
REPLAY_DV_USAGE = (
    "usage: python3 -m packet_routing_sim replay-dv TOPOLOGY.json SCENARIO.json"
)
REPLAY_LS_USAGE = (
    "usage: python3 -m packet_routing_sim replay-ls TOPOLOGY.json SCENARIO.json"
)

USAGE = """usage: python3 -m packet_routing_sim <command>

commands:
  version                               print the package version
  compute TOPOLOGY.json                 compute shortest-path forwarding tables
                                        from a static undirected topology
  converge TOPOLOGY.json                show distance-vector convergence round
                                        by round from a static undirected topology
  replay TOPOLOGY.json SCENARIO.json    replay the static link-state failure
                                        timeline from a scenario
  replay-dv TOPOLOGY.json SCENARIO.json show distance-vector failure
                                        convergence round by round
  replay-ls TOPOLOGY.json SCENARIO.json show link-state flooding round by round
  help                                  print this message
"""


# Sentinel for "reading/parsing failed and the error was already reported";
# an integer would collide with a JSON document that legitimately decodes to
# one (e.g. 42).
_READ_FAILED = object()


def _read_json(path, unreadable_message, invalid_message):
    """Read and decode ``path``.

    Report ``unreadable_message`` (with the path) when the file cannot be
    read and ``invalid_message`` when it is not valid JSON.  Return the
    decoded value, or ``_READ_FAILED`` after reporting.
    """
    try:
        with open(path, "rb") as stream:
            raw = stream.read()
    except OSError:
        print(unreadable_message.format(path=path), file=sys.stderr)
        return _READ_FAILED
    try:
        return json.loads(raw)
    except ValueError:
        print(invalid_message, file=sys.stderr)
        return _READ_FAILED


def _emit(result):
    """Serialize a core result in the published wire format."""
    sys.stdout.write(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


def run_compute(path):
    topology = _read_json(
        path,
        "cannot read topology: {path}",
        "invalid topology",
    )
    if topology is _READ_FAILED:
        return 2
    try:
        result = compute_topology(topology)
    except InvalidTopology:
        print("invalid topology", file=sys.stderr)
        return 2
    _emit(result)
    return 0


def run_converge(path):
    topology = _read_json(
        path,
        "cannot read topology: {path}",
        "invalid topology",
    )
    if topology is _READ_FAILED:
        return 2
    try:
        result = converge_topology(topology)
    except InvalidTopology:
        print("invalid topology", file=sys.stderr)
        return 2
    _emit(result)
    return 0


def _run_replay(topology_path, scenario_path, replay_validated):
    """Shared file/error/serialization boundary for every replay command.

    Topology problems are always reported before the scenario is even read
    or validated.  Structural scenario failures and illegal state
    transitions share the one ``invalid scenario`` report.
    """
    topology = _read_json(
        topology_path,
        "cannot read topology: {path}",
        "invalid topology",
    )
    if topology is _READ_FAILED:
        return 2
    try:
        topo = validate_topology(topology)
    except InvalidTopology:
        print("invalid topology", file=sys.stderr)
        return 2

    scenario = _read_json(
        scenario_path,
        "cannot read scenario: {path}",
        "invalid scenario",
    )
    if scenario is _READ_FAILED:
        return 2
    try:
        result = replay_validated(topo, scenario)
    except InvalidScenario:
        print("invalid scenario", file=sys.stderr)
        return 2
    _emit(result)
    return 0


def run_replay(topology_path, scenario_path):
    return _run_replay(topology_path, scenario_path, replay_validated)


def run_replay_dv(topology_path, scenario_path):
    return _run_replay(topology_path, scenario_path, replay_dv_validated)


def run_replay_ls(topology_path, scenario_path):
    return _run_replay(topology_path, scenario_path, replay_ls_validated)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    command = args[0] if args else "help"
    if command == "version":
        print(__version__)
        return 0
    if command == "compute":
        if len(args) != 2:
            print(COMPUTE_USAGE, file=sys.stderr)
            return 2
        return run_compute(args[1])
    if command == "converge":
        if len(args) != 2:
            print(CONVERGE_USAGE, file=sys.stderr)
            return 2
        return run_converge(args[1])
    if command == "replay":
        if len(args) != 3:
            print(REPLAY_USAGE, file=sys.stderr)
            return 2
        return run_replay(args[1], args[2])
    if command == "replay-dv":
        if len(args) != 3:
            print(REPLAY_DV_USAGE, file=sys.stderr)
            return 2
        return run_replay_dv(args[1], args[2])
    if command == "replay-ls":
        if len(args) != 3:
            print(REPLAY_LS_USAGE, file=sys.stderr)
            return 2
        return run_replay_ls(args[1], args[2])
    if command in {"help", "-h", "--help"}:
        print(USAGE, end="")
        return 0
    print(f"unknown command: {command}", file=sys.stderr)
    print(USAGE, end="", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
