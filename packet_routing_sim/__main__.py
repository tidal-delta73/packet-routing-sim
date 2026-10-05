"""Command line entry point: version, compute and help."""
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
                                by round from a static undirected topology
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


def load_topology(path):
    """Read and validate the topology at path.

    Returns (nodes, adjacency), or an exit code (2) after reporting the
    matching error if the file cannot be read, parsed, or validated.
    """
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
    parsed = load_topology(path)
    if isinstance(parsed, int):
        return parsed
    nodes, adjacency = parsed

    ordered_nodes = sorted(nodes)
    routers = {
        source: forwarding_table(source, ordered_nodes, adjacency)
        for source in ordered_nodes
    }
    output = json.dumps({"routers": routers}, indent=2, ensure_ascii=False)
    sys.stdout.write(output + "\n")
    return 0


def initial_distance_vectors(ordered_nodes, adjacency):
    """Round 0: each router knows only itself and its directly connected links."""
    vectors = {}
    for router in ordered_nodes:
        table = {}
        for destination in ordered_nodes:
            if destination == router:
                table[destination] = {"nextHop": None, "metric": 0}
            elif destination in adjacency[router]:
                table[destination] = {
                    "nextHop": destination,
                    "metric": adjacency[router][destination],
                }
            else:
                table[destination] = {"nextHop": None, "metric": None}
        vectors[router] = table
    return vectors


def distance_vector_round(previous, ordered_nodes, adjacency):
    """Synchronously update every router from the previous round's advertisements."""
    current = {}
    for router in ordered_nodes:
        table = {}
        for destination in ordered_nodes:
            if destination == router:
                table[destination] = {"nextHop": None, "metric": 0}
                continue
            best_metric = None
            best_hop = None
            for neighbor in sorted(adjacency[router]):
                advertised = previous[neighbor][destination]
                if advertised["metric"] is None:
                    continue
                candidate = adjacency[router][neighbor] + advertised["metric"]
                # Neighbors are iterated in name order, so the first best
                # candidate keeps the smaller next-hop name on ties.
                if best_metric is None or candidate < best_metric:
                    best_metric = candidate
                    best_hop = neighbor
            if best_metric is None:
                table[destination] = {"nextHop": None, "metric": None}
            else:
                table[destination] = {"nextHop": best_hop, "metric": best_metric}
        current[router] = table
    return current


def run_converge(path):
    parsed = load_topology(path)
    if isinstance(parsed, int):
        return parsed
    nodes, adjacency = parsed

    ordered_nodes = sorted(nodes)
    vectors = initial_distance_vectors(ordered_nodes, adjacency)
    rounds = [{"round": 0, "routers": vectors}]
    while True:
        updated = distance_vector_round(vectors, ordered_nodes, adjacency)
        if updated == vectors:
            break
        vectors = updated
        rounds.append({"round": len(rounds), "routers": vectors})

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
