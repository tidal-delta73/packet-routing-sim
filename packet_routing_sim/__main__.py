"""Command line entry point for packet-routing-sim."""
import heapq
import json
import sys

from . import __version__

USAGE = """usage: python3 -m packet_routing_sim <command>

commands:
  version                       print the package version
  compute TOPOLOGY.json         compute shortest-path forwarding tables from a
                                static undirected topology snapshot (JSON)
  help                          print this message
"""

COMPUTE_USAGE = (
    "usage: python3 -m packet_routing_sim compute TOPOLOGY.json\n"
)


class InvalidTopology(Exception):
    """Raised when the topology document fails structural validation."""


def _parse_topology(data):
    """Validate a decoded topology document.

    Returns (nodes, adjacency); raises InvalidTopology on any mismatch.
    """
    if not isinstance(data, dict):
        raise InvalidTopology
    nodes_raw = data.get("nodes")
    links_raw = data.get("links")
    if not isinstance(nodes_raw, list) or not isinstance(links_raw, list):
        raise InvalidTopology
    if len(nodes_raw) == 0:
        raise InvalidTopology

    nodes = []
    declared = set()
    for node in nodes_raw:
        if not isinstance(node, str) or node == "":
            raise InvalidTopology
        if node in declared:
            raise InvalidTopology
        declared.add(node)
        nodes.append(node)

    adjacency = {node: {} for node in nodes}
    pairs = set()
    for link in links_raw:
        if not isinstance(link, dict):
            raise InvalidTopology
        if "from" not in link or "to" not in link or "metric" not in link:
            raise InvalidTopology
        left = link["from"]
        right = link["to"]
        metric = link["metric"]
        if not isinstance(left, str) or not isinstance(right, str):
            raise InvalidTopology
        # bool is a subclass of int: reject it before the int check.
        if isinstance(metric, bool) or not isinstance(metric, int) or metric <= 0:
            raise InvalidTopology
        if left == right or left not in declared or right not in declared:
            raise InvalidTopology
        pair = (left, right) if left < right else (right, left)
        if pair in pairs:
            raise InvalidTopology
        pairs.add(pair)
        adjacency[left][right] = metric
        adjacency[right][left] = metric

    return nodes, adjacency


def _shortest_paths(source, adjacency):
    """Dijkstra from source.

    Returns (dist, first_hop) for every reachable node. On ties in total
    metric, the first hop is the smallest next-hop node name among all
    equal-cost paths, derived from the shortest-path DAG after distances
    are settled.
    """
    dist = {source: 0}
    queue = [(0, source)]
    while queue:
        current_dist, u = heapq.heappop(queue)
        if current_dist != dist[u]:
            continue
        for v in sorted(adjacency[u]):
            new_dist = current_dist + adjacency[u][v]
            if new_dist < dist.get(v, float("inf")):
                dist[v] = new_dist
                heapq.heappush(queue, (new_dist, v))

    # Positive metrics make distance order a topological order of the
    # shortest-path DAG, so every predecessor is settled first.
    first_hop = {source: None}
    for v in sorted(dist, key=lambda node: (dist[node], node)):
        if v == source:
            continue
        candidates = []
        for u, weight in adjacency[v].items():
            if u in dist and dist[u] + weight == dist[v]:
                candidates.append(v if u == source else first_hop[u])
        first_hop[v] = min(candidates)

    return dist, first_hop


def _run_compute(path):
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except OSError:
        print(f"cannot read topology: {path}", file=sys.stderr)
        return 2

    try:
        data = json.loads(raw)
        nodes, adjacency = _parse_topology(data)
    except (UnicodeDecodeError, json.JSONDecodeError, InvalidTopology):
        print("invalid topology", file=sys.stderr)
        return 2

    ordered_nodes = sorted(nodes)
    routers = {}
    for source in ordered_nodes:
        dist, first_hop = _shortest_paths(source, adjacency)
        entries = {}
        for destination in ordered_nodes:
            if destination == source:
                entries[destination] = {"nextHop": None, "metric": 0}
            elif destination in dist:
                entries[destination] = {
                    "nextHop": first_hop[destination],
                    "metric": dist[destination],
                }
            else:
                entries[destination] = {"nextHop": None, "metric": None}
        routers[source] = entries

    output = json.dumps({"routers": routers}, ensure_ascii=False, indent=2)
    sys.stdout.write(output + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    command = args[0] if args else "help"
    if command == "version":
        print(__version__)
        return 0
    if command == "compute":
        if len(args) != 2:
            sys.stderr.write(COMPUTE_USAGE)
            return 2
        return _run_compute(args[1])
    if command in {"help", "-h", "--help"}:
        print(USAGE, end="")
        return 0
    print(f"unknown command: {command}", file=sys.stderr)
    print(USAGE, end="", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
