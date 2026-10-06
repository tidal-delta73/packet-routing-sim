"""Command line entry point: version, compute, converge, replay and help.

This layer only checks arguments, reads files, parses JSON, calls the pure
simulation core in :mod:`packet_routing_sim.core` and serializes results.
It contains no topology, protocol or failure-state logic of its own.
"""
import json
import sys

from . import __version__
from .core import (
    InvalidScenario,
    InvalidTopology,
    compute,
    converge,
    simulate_replay,
)

COMPUTE_USAGE = "usage: python3 -m packet_routing_sim compute TOPOLOGY.json"
CONVERGE_USAGE = "usage: python3 -m packet_routing_sim converge TOPOLOGY.json"
REPLAY_USAGE = (
    "usage: python3 -m packet_routing_sim replay TOPOLOGY.json SCENARIO.json"
)

USAGE = """usage: python3 -m packet_routing_sim <command>

commands:
  version                       print the package version
  compute TOPOLOGY.json         compute shortest-path forwarding tables
                                from a static undirected topology
  converge TOPOLOGY.json        show distance-vector convergence round
                                by round from a static undirected topology
  help                          print this message
"""

_READ_FAILED = object()


def load_json(path, read_error, parse_error):
    """Read and JSON-decode the file at path.

    Returns the decoded value, or the ``_READ_FAILED`` sentinel after
    reporting ``read_error``/``parse_error``; the caller maps the sentinel
    to exit code 2.
    """
    try:
        with open(path, "rb") as stream:
            raw = stream.read()
    except OSError:
        print(read_error, file=sys.stderr)
        return _READ_FAILED
    try:
        return json.loads(raw)
    except ValueError:
        print(parse_error, file=sys.stderr)
        return _READ_FAILED


def run_compute(path):
    topology = load_json(
        path, f"cannot read topology: {path}", "invalid topology"
    )
    if topology is _READ_FAILED:
        return 2
    try:
        result = compute(topology)
    except InvalidTopology:
        print("invalid topology", file=sys.stderr)
        return 2
    sys.stdout.write(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return 0


def run_converge(path):
    topology = load_json(
        path, f"cannot read topology: {path}", "invalid topology"
    )
    if topology is _READ_FAILED:
        return 2
    try:
        result = converge(topology)
    except InvalidTopology:
        print("invalid topology", file=sys.stderr)
        return 2
    sys.stdout.write(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return 0


def run_replay(topology_path, scenario_path):
    topology = load_json(
        topology_path,
        f"cannot read topology: {topology_path}",
        "invalid topology",
    )
    if topology is _READ_FAILED:
        return 2

    scenario = load_json(
        scenario_path,
        f"cannot read scenario: {scenario_path}",
        "invalid scenario",
    )
    if scenario is _READ_FAILED:
        return 2

    try:
        result = simulate_replay(topology, scenario)
    except InvalidTopology:
        print("invalid topology", file=sys.stderr)
        return 2
    except InvalidScenario:
        # Covers illegal state transitions as well as malformed scenarios.
        print("invalid scenario", file=sys.stderr)
        return 2
    sys.stdout.write(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return 0


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
    if command in {"help", "-h", "--help"}:
        print(USAGE, end="")
        return 0
    print(f"unknown command: {command}", file=sys.stderr)
    print(USAGE, end="", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
