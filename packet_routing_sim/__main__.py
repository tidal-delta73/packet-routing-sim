"""Command line entry point: version, compute, converge and help."""
import heapq
import json
import sys

from . import __version__

COMPUTE_USAGE = "usage: python3 -m packet_routing_sim compute TOPOLOGY.json"
CONVERGE_USAGE = "usage: python3 -m packet_routing_sim converge TOPOLOGY.json"

USAGE = """usage: python3 -m packet_routing_sim <command>

commands:
  version                       print the package version
  compute TOPOLOGY.json         compute shortest-path forwarding tables
                                from a static undirected topology
  converge TOPOLOGY.json        show distance-vector convergence round
                                by round from local knowledge to stable
                                forwarding tables
  help                          print this message
"""


def parse_topology(topology):
    """Return (nodes, adjacency) for a validated topology, or None if invalid."""
    if not isinstance(topology, dict):
        return None
    nodes_raw = topology.get("nodes")
    links_raw = topology.get("links")
    if not isinstance(nodes_raw, list) or not isinstance(links_raw, list):
        return None

    nodes = []
    declared = set()
    for node in nodes_raw:
        if not isinstance(node, str) or node == "" or node in declared:
            return None
        declared.add(node)
        nodes.append(node)

    adjacency = {node: {} for node in nodes}
    pairs = set()
    for link in links_raw:
        if not isinstance(link, dict):
            return None
        if "from" not in link or "to" not in link or "metric" not in link:
            return None
        source, target, metric = link["from"], link["to"], link["metric"]
        if not isinstance(source, str) or not isinstance(target, str):
            return None
        if source not in declared or target not in declared:
            return None
        if source == target:
            return None
        if isinstance(metric, bool) or not isinstance(metric, int) or metric <= 0:
            return None
        pair = (source, target) if source < target else (target, source)
        if pair in pairs:
            return None
        pairs.add(pair)
        adjacency[source][target] = metric
        adjacency[target][source] = metric

    return nodes, adjacency


def forwarding_table(source, ordered_nodes, adjacency):
    """Dijkstra from source; ties on total metric break to the smaller first hop."""
    distances = {source: 0}
    next_hops = {}
    # Label is (total metric, first hop name); "" is a sentinel for the source
    # and sorts before any real (necessarily non-empty) node name.
    heap = [(0, "", source)]
    settled = set()
    while heap:
        distance, first_hop, node = heapq.heappop(heap)
        if node in settled:
            continue
        settled.add(node)
        distances[node] = distance
        next_hops[node] = first_hop
        for neighbor in sorted(adjacency[node]):
            if neighbor in settled:
                continue
            metric = adjacency[node][neighbor]
            candidate = distance + metric
            candidate_hop = neighbor if node == source else first_hop
            heapq.heappush(heap, (candidate, candidate_hop, neighbor))

    table = {}
    for destination in ordered_nodes:
        if destination == source:
            table[destination] = {"nextHop": None, "metric": 0}
        elif destination in distances:
            table[destination] = {
                "nextHop": next_hops[destination],
                "metric": distances[destination],
            }
        else:
            table[destination] = {"nextHop": None, "metric": None}
    return table


def initial_table(source, ordered_nodes, adjacency):
    """Round-0 distance vector: only the router itself and direct neighbors."""
    table = {}
    for destination in ordered_nodes:
        if destination == source:
            table[destination] = {"nextHop": None, "metric": 0}
        elif destination in adjacency[source]:
            table[destination] = {
                "nextHop": destination,
                "metric": adjacency[source][destination],
            }
        else:
            table[destination] = {"nextHop": None, "metric": None}
    return table


def dv_step(source, ordered_nodes, adjacency, previous):
    """One synchronous Bellman-Ford update from the previous round's vectors."""
    table = {}
    for destination in ordered_nodes:
        if destination == source:
            table[destination] = {"nextHop": None, "metric": 0}
            continue
        best_metric = None
        best_hop = None
        for neighbor in adjacency[source]:
            neighbor_metric = previous[neighbor][destination]["metric"]
            if neighbor_metric is None:
                continue
            candidate_metric = adjacency[source][neighbor] + neighbor_metric
            if (
                best_metric is None
                or candidate_metric < best_metric
                or (candidate_metric == best_metric and neighbor < best_hop)
            ):
                best_metric = candidate_metric
                best_hop = neighbor
        table[destination] = {"nextHop": best_hop, "metric": best_metric}
    return table


def load_topology(path):
    """Read and validate a topology file; return (nodes, adjacency) or exit code 2."""
    try:
        with open(path, "rb") as stream:
            raw = stream.read()
    except OSError:
        print(f"cannot read topology: {path}", file=sys.stderr)
        return 2

    try:
        topology = json.loads(raw)
    except ValueError:
        print("invalid topology", file=sys.stderr)
        return 2

    parsed = parse_topology(topology)
    if parsed is None:
        print("invalid topology", file=sys.stderr)
        return 2
    return parsed


def run_compute(path):
    loaded = load_topology(path)
    if isinstance(loaded, int):
        return loaded
    nodes, adjacency = loaded

    ordered_nodes = sorted(nodes)
    routers = {
        source: forwarding_table(source, ordered_nodes, adjacency)
        for source in ordered_nodes
    }
    output = json.dumps({"routers": routers}, indent=2, ensure_ascii=False)
    sys.stdout.write(output + "\n")
    return 0


def run_converge(path):
    loaded = load_topology(path)
    if isinstance(loaded, int):
        return loaded
    nodes, adjacency = loaded

    ordered_nodes = sorted(nodes)
    routers = {
        source: initial_table(source, ordered_nodes, adjacency)
        for source in ordered_nodes
    }
    rounds = [{"round": 0, "routers": routers}]
    while True:
        next_routers = {
            source: dv_step(source, ordered_nodes, adjacency, routers)
            for source in ordered_nodes
        }
        if next_routers == routers:
            break
        rounds.append({"round": len(rounds), "routers": next_routers})
        routers = next_routers

    output = {
        "protocol": "distance-vector",
        "convergenceRound": rounds[-1]["round"],
        "rounds": rounds,
    }
    sys.stdout.write(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
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
    if command in {"help", "-h", "--help"}:
        print(USAGE, end="")
        return 0
    print(f"unknown command: {command}", file=sys.stderr)
    print(USAGE, end="", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
