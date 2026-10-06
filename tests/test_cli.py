"""Differential and property tests for the public ``compute``/``converge`` CLI.

These tests exercise only the documented command-line surface
(`python -m packet_routing_sim <command> TOPOLOGY.json`) and never import or
call ``forwarding_table`` / ``distance_vector_round``: every expected value is
either hand-written for a small topology or produced by an independently
written Floyd-Warshall / Bellman-Ford reference in this module, so a defect in
the implementation cannot simultaneously corrupt the answer under test and
the assertion about it.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Run every subprocess under several hash seeds: dict iteration order (and thus
# any accidental dependence on it) must never change the published output.
# PRSIM_TEST_SEEDS="0" restricts the set so local mutation runs stay quick.
HASH_SEEDS = tuple(
    int(seed)
    for seed in os.environ.get("PRSIM_TEST_SEEDS", "0,1,42,987654321").split(",")
)

VERSION = "0.1.0"
COMPUTE_USAGE = "usage: python3 -m packet_routing_sim compute TOPOLOGY.json\n"
CONVERGE_USAGE = "usage: python3 -m packet_routing_sim converge TOPOLOGY.json\n"
REPLAY_USAGE = (
    "usage: python3 -m packet_routing_sim replay TOPOLOGY.json SCENARIO.json\n"
)
REPLAY_DV_USAGE = (
    "usage: python3 -m packet_routing_sim replay-dv TOPOLOGY.json SCENARIO.json\n"
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


# ---------------------------------------------------------------------------
# CLI invocation and topology helpers
# ---------------------------------------------------------------------------

def link(source, target, metric):
    return {"from": source, "to": target, "metric": metric}


def run_cli(args, seed=0):
    """Invoke the package as a module; return the completed process."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONHASHSEED"] = str(seed)
    return subprocess.run(
        [sys.executable, "-m", "packet_routing_sim", *args],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )


def write_topology(directory, topology, name="topology.json"):
    path = Path(directory) / name
    path.write_text(json.dumps(topology), encoding="utf-8")
    return path


def invoke_both(directory, topology, seed=0):
    """Return ((compute_json, compute_bytes), (converge_json, converge_bytes))."""
    path = write_topology(directory, topology)
    compute_proc = run_cli(["compute", str(path)], seed=seed)
    converge_proc = run_cli(["converge", str(path)], seed=seed)
    assert compute_proc.returncode == 0, compute_proc.stderr
    assert converge_proc.returncode == 0, converge_proc.stderr
    assert compute_proc.stderr == b"" and converge_proc.stderr == b""
    return (
        (json.loads(compute_proc.stdout), compute_proc.stdout),
        (json.loads(converge_proc.stdout), converge_proc.stdout),
    )


# ---------------------------------------------------------------------------
# Independent reference implementation (test-side oracle)
# ---------------------------------------------------------------------------

def build_adjacency(topology):
    """Build an undirected adjacency map straight from the declared links."""
    nodes = sorted(topology["nodes"])
    adjacency = {node: {} for node in nodes}
    for edge in topology["links"]:
        a, b, w = edge["from"], edge["to"], edge["metric"]
        adjacency[a][b] = w
        adjacency[b][a] = w
    return nodes, adjacency


def floyd_warshall(nodes, adjacency):
    """All-pairs shortest path lengths, written independently of the SUT."""
    inf = float("inf")
    dist = {i: {j: inf for j in nodes} for i in nodes}
    for node in nodes:
        dist[node][node] = 0
    for a in nodes:
        for b, w in adjacency[a].items():
            # The topology validates at most one link per undirected pair.
            dist[a][b] = w
    for k in nodes:
        dk = dist[k]
        for i in nodes:
            di = dist[i]
            through_base = di[k]
            if through_base == inf:
                continue
            for j in nodes:
                candidate = through_base + dk[j]
                if candidate < di[j]:
                    di[j] = candidate
    return dist


def oracle_tables(topology):
    """Expected {(router, destination): (nextHop, metric)} via Floyd-Warshall.

    On equal total cost the lexicographically smallest direct neighbor wins,
    matching the published tie-break rule.
    """
    nodes, adjacency = build_adjacency(topology)
    dist = floyd_warshall(nodes, adjacency)
    expected = {}
    for source in nodes:
        for dest in nodes:
            if source == dest:
                expected[source, dest] = (None, 0)
                continue
            if dist[source][dest] == float("inf"):
                expected[source, dest] = (None, None)
                continue
            winners = [
                neighbor
                for neighbor in adjacency[source]
                if dist[neighbor][dest] != float("inf")
                and adjacency[source][neighbor] + dist[neighbor][dest]
                == dist[source][dest]
            ]
            hop = min(winners)
            expected[source, dest] = (hop, dist[source][dest])
    return nodes, expected


def normalize(routers):
    """Convert JSON routing tables into {(r, d): (nextHop, metric)} tuples."""
    return {
        (r, d): (entry["nextHop"], entry["metric"])
        for r, row in routers.items()
        for d, entry in row.items()
    }


def synchronous_update(previous, nodes, adjacency):
    """One independent Bellman-Ford synchronous round from a prior snapshot.

    Only the *previous* round's neighbor advertisements are read; nothing from
    the round being computed. Ties keep the smaller neighbor name because
    neighbors are scanned in sorted order with a strict `<` comparison.
    """
    current = {}
    for router in nodes:
        for dest in nodes:
            if dest == router:
                current[router, dest] = (None, 0)
                continue
            best_cost = None
            best_hop = None
            for neighbor in sorted(adjacency[router]):
                advertised_cost = previous[neighbor, dest][1]
                if advertised_cost is None:
                    continue
                cost = adjacency[router][neighbor] + advertised_cost
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_hop = neighbor
            if best_cost is None:
                current[router, dest] = (None, None)
            else:
                current[router, dest] = (best_hop, best_cost)
    return current


def round_zero(nodes, adjacency):
    """The only legal round-0 view: self plus direct links."""
    tables = {}
    for router in nodes:
        for dest in nodes:
            if dest == router:
                tables[router, dest] = (None, 0)
            elif dest in adjacency[router]:
                tables[router, dest] = (dest, adjacency[router][dest])
            else:
                tables[router, dest] = (None, None)
    return tables


# ---------------------------------------------------------------------------
# Fixed, reproducible fixtures
# ---------------------------------------------------------------------------

EMPTY = {"nodes": [], "links": []}

SINGLE = {"nodes": ["solo"], "links": []}

# Two nontrivial components plus a fully isolated node.
COMPONENTS = {
    "nodes": ["a1", "a2", "b1", "b2", "iso"],
    "links": [link("a1", "a2", 2), link("b1", "b2", 5)],
}

CHAIN3 = {
    "nodes": ["A", "B", "C"],
    "links": [link("A", "B", 5), link("B", "C", 7)],
}

CHAIN4 = {
    "nodes": ["A", "B", "C", "D"],
    "links": [link("A", "B", 1), link("B", "C", 2), link("C", "D", 4)],
}

# Weights 1/2/3/4 create genuine equal-cost routes (B<->D costs 5 both ways).
RING = {
    "nodes": ["A", "B", "C", "D"],
    "links": [
        link("A", "B", 1),
        link("B", "C", 2),
        link("C", "D", 3),
        link("D", "A", 4),
    ],
}

# Unit diamond: every A<->D and B<->C route has two equal-cost alternatives.
DIAMOND = {
    "nodes": ["A", "B", "C", "D"],
    "links": [
        link("A", "B", 1),
        link("A", "C", 1),
        link("B", "D", 1),
        link("C", "D", 1),
    ],
}

# Same tie situation with non-unit weights, deliberately declared with nodes
# reversed and links in mixed directions/order: output must not depend on it.
TIECOST = {
    "nodes": ["T", "n2", "n1", "S"],
    "links": [
        link("T", "n2", 3),
        link("n1", "T", 3),
        link("n2", "S", 2),
        link("S", "n1", 2),
    ],
}

# Equal-cost square (both directions around cost 4) plus an isolated node.
SQUARE = {
    "nodes": ["A", "B", "C", "D"],
    "links": [
        link("A", "B", 2),
        link("B", "C", 2),
        link("C", "D", 2),
        link("D", "A", 2),
    ],
}

CHAIN3_WITH_ISOLATE = {
    "nodes": ["A", "B", "C", "Z"],
    "links": [link("A", "B", 5), link("B", "C", 7)],
}

# Hand-computed expected tables for the small topologies. `None`/`None` marks
# an unreachable destination; `None`/0 marks the router's own entry.
def _table(rows):
    return {
        (router, dest): value
        for router, row in rows.items()
        for dest, value in row.items()
    }


HAND_EXPECTED = {
    "single": _table({"solo": {"solo": (None, 0)}}),
    "components": _table(
        {
            "a1": {
                "a1": (None, 0),
                "a2": ("a2", 2),
                "b1": (None, None),
                "b2": (None, None),
                "iso": (None, None),
            },
            "a2": {
                "a1": ("a1", 2),
                "a2": (None, 0),
                "b1": (None, None),
                "b2": (None, None),
                "iso": (None, None),
            },
            "b1": {
                "a1": (None, None),
                "a2": (None, None),
                "b1": (None, 0),
                "b2": ("b2", 5),
                "iso": (None, None),
            },
            "b2": {
                "a1": (None, None),
                "a2": (None, None),
                "b1": ("b1", 5),
                "b2": (None, 0),
                "iso": (None, None),
            },
            "iso": {
                "a1": (None, None),
                "a2": (None, None),
                "b1": (None, None),
                "b2": (None, None),
                "iso": (None, 0),
            },
        }
    ),
    "chain3": _table(
        {
            "A": {"A": (None, 0), "B": ("B", 5), "C": ("B", 12)},
            "B": {"A": ("A", 5), "B": (None, 0), "C": ("C", 7)},
            "C": {"A": ("B", 12), "B": ("B", 7), "C": (None, 0)},
        }
    ),
    "chain4": _table(
        {
            "A": {"A": (None, 0), "B": ("B", 1), "C": ("B", 3), "D": ("B", 7)},
            "B": {"A": ("A", 1), "B": (None, 0), "C": ("C", 2), "D": ("C", 6)},
            "C": {"A": ("B", 3), "B": ("B", 2), "C": (None, 0), "D": ("D", 4)},
            "D": {"A": ("C", 7), "B": ("C", 6), "C": ("C", 4), "D": (None, 0)},
        }
    ),
    "ring": _table(
        {
            "A": {"A": (None, 0), "B": ("B", 1), "C": ("B", 3), "D": ("D", 4)},
            "B": {"A": ("A", 1), "B": (None, 0), "C": ("C", 2), "D": ("A", 5)},
            "C": {"A": ("B", 3), "B": ("B", 2), "C": (None, 0), "D": ("D", 3)},
            "D": {"A": ("A", 4), "B": ("A", 5), "C": ("C", 3), "D": (None, 0)},
        }
    ),
    "diamond": _table(
        {
            "A": {"A": (None, 0), "B": ("B", 1), "C": ("C", 1), "D": ("B", 2)},
            "B": {"A": ("A", 1), "B": (None, 0), "C": ("A", 2), "D": ("D", 1)},
            "C": {"A": ("A", 1), "B": ("A", 2), "C": (None, 0), "D": ("D", 1)},
            "D": {"A": ("B", 2), "B": ("B", 1), "C": ("C", 1), "D": (None, 0)},
        }
    ),
    "tiecost": _table(
        {
            "S": {"S": (None, 0), "T": ("n1", 5), "n1": ("n1", 2), "n2": ("n2", 2)},
            "T": {"S": ("n1", 5), "T": (None, 0), "n1": ("n1", 3), "n2": ("n2", 3)},
            "n1": {"S": ("S", 2), "T": ("T", 3), "n1": (None, 0), "n2": ("S", 4)},
            "n2": {"S": ("S", 2), "T": ("T", 3), "n1": ("S", 4), "n2": (None, 0)},
        }
    ),
    "square": _table(
        {
            "A": {"A": (None, 0), "B": ("B", 2), "C": ("B", 4), "D": ("D", 2)},
            "B": {"A": ("A", 2), "B": (None, 0), "C": ("C", 2), "D": ("A", 4)},
            "C": {"A": ("B", 4), "B": ("B", 2), "C": (None, 0), "D": ("D", 2)},
            "D": {"A": ("A", 2), "B": ("A", 4), "C": ("C", 2), "D": (None, 0)},
        }
    ),
}

# Every fixture gets the full differential/property treatment under every seed.
ALL_TOPOLOGIES = [
    ("empty", EMPTY, None),
    ("single", SINGLE, "single"),
    ("components", COMPONENTS, "components"),
    ("chain3", CHAIN3, "chain3"),
    ("chain4", CHAIN4, "chain4"),
    ("ring", RING, "ring"),
    ("diamond", DIAMOND, "diamond"),
    ("tiecost", TIECOST, "tiecost"),
    ("square", SQUARE, "square"),
    ("chain3_with_isolate", CHAIN3_WITH_ISOLATE, None),
]

# Semantically equivalent rewordings of the diamond: node declaration order,
# link direction, and link permutation must not alter a single output byte.
EQUIVALENT_VARIANTS = [
    DIAMOND,
    {"nodes": ["D", "C", "B", "A"], "links": DIAMOND["links"]},
    {
        "nodes": ["A", "B", "C", "D"],
        "links": [
            link("B", "A", 1),
            link("C", "A", 1),
            link("D", "B", 1),
            link("D", "C", 1),
        ],
    },
    {
        "nodes": ["A", "B", "C", "D"],
        "links": list(reversed(DIAMOND["links"])),
    },
    {
        "nodes": ["D", "C", "B", "A"],
        "links": [
            link("D", "C", 1),
            link("D", "B", 1),
            link("C", "A", 1),
            link("B", "A", 1),
        ],
    },
]


# ---------------------------------------------------------------------------
# Differential and structural properties
# ---------------------------------------------------------------------------

class TestComputeConvergeDifferential(unittest.TestCase):
    def _check_table_shape_and_semantics(self, name, topology, routers):
        nodes, adjacency = build_adjacency(topology)
        _, expected = oracle_tables(topology)
        self.assertEqual(set(routers), set(nodes))
        for router in nodes:
            self.assertEqual(set(routers[router]), set(nodes))
            for dest in nodes:
                entry = routers[router][dest]
                self.assertEqual(set(entry), {"nextHop", "metric"})
                hop, metric = entry["nextHop"], entry["metric"]
                # The table must match the independent Floyd-Warshall oracle.
                self.assertEqual((hop, metric), expected[router, dest], name)
                if router == dest:
                    self.assertEqual((hop, metric), (None, 0), name)
                if metric is None:
                    # Unreachable: both fields are JSON null.
                    self.assertIsNone(hop, name)
                else:
                    self.assertIsInstance(metric, int, name)
                    self.assertNotIsInstance(metric, bool, name)
                    if router != dest:
                        # A reachable next hop must be a direct neighbor.
                        self.assertIn(hop, adjacency[router], name)
                        self.assertGreaterEqual(metric, adjacency[router][hop], name)
                    else:
                        self.assertIsNone(hop, name)
                        self.assertEqual(metric, 0, name)

    def test_compute_matches_hand_and_oracle_and_converge_under_every_seed(self):
        for name, topology, hand_key in ALL_TOPOLOGIES:
            with self.subTest(topology=name), tempfile.TemporaryDirectory() as tmp:
                compute_byte_variants = set()
                converge_byte_variants = set()
                for seed in HASH_SEEDS:
                    with self.subTest(seed=seed):
                        (compute, compute_raw), (converge, converge_raw) = invoke_both(
                            tmp, topology, seed
                        )
                        # Published stdout must not depend on hash randomization.
                        compute_byte_variants.add(compute_raw)
                        converge_byte_variants.add(converge_raw)

                        self.assertEqual(set(compute), {"routers"})
                        self.assertEqual(
                            set(converge),
                            {"protocol", "convergenceRound", "rounds"},
                        )
                        self.assertEqual(converge["protocol"], "distance-vector")

                        # Differential core: the final DV snapshot is exactly
                        # the link-state forwarding table.
                        final = converge["rounds"][-1]
                        self.assertEqual(
                            final["routers"],
                            compute["routers"],
                            name,
                        )
                        self.assertEqual(
                            converge["convergenceRound"],
                            final["round"],
                            name,
                        )

                        self._check_table_shape_and_semantics(
                            name, topology, compute["routers"]
                        )
                        self._check_table_shape_and_semantics(
                            name, topology, final["routers"]
                        )

                        if hand_key is not None:
                            hand = HAND_EXPECTED[hand_key]
                            self.assertEqual(
                                normalize(compute["routers"]), hand, name
                            )
                            self.assertEqual(
                                normalize(final["routers"]), hand, name
                            )

                self.assertEqual(len(compute_byte_variants), 1, name)
                self.assertEqual(len(converge_byte_variants), 1, name)

    def test_adding_isolated_node_only_adds_unreachable_entries(self):
        # Byte-identity across hash seeds is covered above, so a single seed
        # suffices for this structural before/after comparison.
        with tempfile.TemporaryDirectory() as tmp:
            (base_compute, _), (base_converge, _) = invoke_both(tmp, CHAIN3)
            (ext_compute, _), (ext_converge, _) = invoke_both(
                tmp, CHAIN3_WITH_ISOLATE
            )
            originals = ["A", "B", "C"]
            final_ext = ext_converge["rounds"][-1]["routers"]
            final_base = base_converge["rounds"][-1]["routers"]

            # Existing routers: every old destination keeps its route.
            for router in originals:
                for dest in originals:
                    self.assertEqual(
                        ext_compute["routers"][router][dest],
                        base_compute["routers"][router][dest],
                    )
                    self.assertEqual(
                        final_ext[router][dest], final_base[router][dest]
                    )
                # The only addition for old routers is an unreachable Z.
                self.assertEqual(
                    ext_compute["routers"][router]["Z"],
                    {"nextHop": None, "metric": None},
                )

            # Z itself: self entry only, everything else unreachable.
            zrow = ext_compute["routers"]["Z"]
            self.assertEqual(zrow["Z"], {"nextHop": None, "metric": 0})
            for dest in originals:
                self.assertEqual(zrow[dest], {"nextHop": None, "metric": None})

    def test_equivalent_inputs_produce_byte_identical_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            for seed in HASH_SEEDS:
                with self.subTest(seed=seed):
                    compute_outs, converge_outs = [], []
                    for index, variant in enumerate(EQUIVALENT_VARIANTS):
                        path = write_topology(tmp, variant, f"v{index}.json")
                        compute_outs.append(
                            run_cli(["compute", str(path)], seed).stdout
                        )
                        converge_outs.append(
                            run_cli(["converge", str(path)], seed).stdout
                        )
                    self.assertEqual(len(set(compute_outs)), 1)
                    self.assertEqual(len(set(converge_outs)), 1)

    def test_equal_cost_always_picks_smaller_neighbor_name(self):
        # Hand expectations already encode the tie-break; assert the decisive
        # ties explicitly for both protocols so the rule cannot regress
        # silently behind the full-table comparison.
        ties = {
            "ring": {("A", "C"): "B", ("B", "D"): "A", ("C", "A"): "B", ("D", "B"): "A"},
            "diamond": {
                ("A", "D"): "B",
                ("D", "A"): "B",
                ("B", "C"): "A",
                ("C", "B"): "A",
            },
            "tiecost": {("S", "T"): "n1", ("T", "S"): "n1"},
            "square": {
                ("A", "C"): "B",
                ("C", "A"): "B",
                ("B", "D"): "A",
                ("D", "B"): "A",
            },
        }
        for name, topology, _ in [
            t for t in ALL_TOPOLOGIES if t[0] in ties
        ]:
            with self.subTest(topology=name), tempfile.TemporaryDirectory() as tmp:
                (compute, _), (converge, _) = invoke_both(tmp, topology)
                final = converge["rounds"][-1]["routers"]
                for (source, dest), hop in ties[name].items():
                    self.assertEqual(
                        compute["routers"][source][dest]["nextHop"], hop
                    )
                    self.assertEqual(final[source][dest]["nextHop"], hop)


# ---------------------------------------------------------------------------
# Convergence trajectory properties
# ---------------------------------------------------------------------------

class TestConvergenceTrajectory(unittest.TestCase):
    def validate_trace(self, name, topology, converge):
        nodes, adjacency = build_adjacency(topology)
        rounds = converge["rounds"]

        # Rounds are numbered 0..N with no gaps or duplicates.
        self.assertEqual([snapshot["round"] for snapshot in rounds], list(range(len(rounds))))
        self.assertEqual(converge["convergenceRound"], rounds[-1]["round"], name)

        snapshots = [normalize(snapshot["routers"]) for snapshot in rounds]
        for snapshot in rounds:
            self.assertEqual(set(snapshot), {"round", "routers"})

        # Round 0: self plus direct neighbors, nothing else.
        self.assertEqual(snapshots[0], round_zero(nodes, adjacency), name)

        # Every later snapshot must follow from the previous one by one
        # synchronous Bellman-Ford update (neighbors' same-round results are
        # unavailable by construction of synchronous_update), and recorded
        # adjacent snapshots must genuinely differ.
        for index in range(1, len(rounds)):
            derived = synchronous_update(snapshots[index - 1], nodes, adjacency)
            self.assertEqual(snapshots[index], derived, f"{name} round {index}")
            self.assertNotEqual(
                snapshots[index], snapshots[index - 1], f"{name} round {index}"
            )

        # The final recorded snapshot is already a fixed point, so
        # convergenceRound is the first round at which the state is stable.
        self.assertEqual(
            synchronous_update(snapshots[-1], nodes, adjacency),
            snapshots[-1],
            name,
        )

    def test_trace_properties_for_every_topology_and_seed(self):
        for name, topology, _ in ALL_TOPOLOGIES:
            with self.subTest(topology=name), tempfile.TemporaryDirectory() as tmp:
                path = write_topology(tmp, topology)
                for seed in HASH_SEEDS:
                    with self.subTest(seed=seed):
                        proc = run_cli(["converge", str(path)], seed)
                        self.assertEqual(proc.returncode, 0, proc.stderr)
                        self.assertEqual(proc.stderr, b"")
                        converge = json.loads(proc.stdout)
                        self.validate_trace(name, topology, converge)

    def test_disconnected_and_single_node_converge_at_round_zero(self):
        for name, topology in (
            ("empty", EMPTY),
            ("single", SINGLE),
            ("components", COMPONENTS),
        ):
            with self.subTest(topology=name), tempfile.TemporaryDirectory() as tmp:
                path = write_topology(tmp, topology)
                proc = run_cli(["converge", str(path)])
                self.assertEqual(proc.returncode, 0, proc.stderr)
                converge = json.loads(proc.stdout)
                self.assertEqual(converge["convergenceRound"], 0)
                self.assertEqual(len(converge["rounds"]), 1)

    def test_chain_converges_in_diameter_rounds(self):
        # A chain needs (diameter-1) advertisement exchanges: round 0 already
        # contains the one-hop view, so a 3-node chain stabilizes at round 1
        # and a 4-node chain at round 2.
        with tempfile.TemporaryDirectory() as tmp:
            c3 = json.loads(run_cli(["converge", str(write_topology(tmp, CHAIN3, "c3.json"))]).stdout)
            c4 = json.loads(run_cli(["converge", str(write_topology(tmp, CHAIN4, "c4.json"))]).stdout)
            self.assertEqual(c3["convergenceRound"], 1)
            self.assertEqual(c4["convergenceRound"], 2)


# ---------------------------------------------------------------------------
# Replay: independent oracle and scenario helpers
# ---------------------------------------------------------------------------

def replay_oracle_routers(topology, down_nodes, down_links):
    """Expected {(router, destination): (nextHop, metric)} after failures.

    Written independently of the SUT: Floyd-Warshall over the subgraph of
    active nodes and links, with down routers' rows and down or unreachable
    destinations forced to (None, None).
    """
    nodes, adjacency = build_adjacency(topology)
    disabled = {frozenset(pair) for pair in down_links}
    active = {node: {} for node in nodes}
    for source in nodes:
        if source in down_nodes:
            continue
        for target, metric in adjacency[source].items():
            if target in down_nodes or frozenset((source, target)) in disabled:
                continue
            active[source][target] = metric
    dist = floyd_warshall(nodes, active)
    expected = {}
    for source in nodes:
        for dest in nodes:
            if source in down_nodes:
                expected[source, dest] = (None, None)
            elif source == dest:
                expected[source, dest] = (None, 0)
            elif dest in down_nodes or dist[source][dest] == float("inf"):
                expected[source, dest] = (None, None)
            else:
                winners = [
                    neighbor
                    for neighbor in active[source]
                    if active[source][neighbor] + dist[neighbor][dest]
                    == dist[source][dest]
                ]
                expected[source, dest] = (min(winners), dist[source][dest])
    return expected


def write_scenario(directory, scenario, name="scenario.json"):
    path = Path(directory) / name
    path.write_text(json.dumps(scenario), encoding="utf-8")
    return path


def invoke_replay(directory, topology, scenario, seed=0):
    topo_path = write_topology(directory, topology)
    scenario_path = write_scenario(directory, scenario)
    return run_cli(
        ["replay", str(topo_path), str(scenario_path)], seed=seed
    )


# Exercises every action plus the independent tracking of link-down state:
# the C-D link is disabled while C is down, so C's recovery must not revive
# it, and the reversed-direction link-up must clear exactly that record.
REPLAY_SCENARIO = [
    {"time": 3, "action": "link-down", "from": "A", "to": "B"},
    {"time": 5, "action": "node-down", "node": "C"},
    {"time": 8, "action": "link-down", "from": "D", "to": "C"},
    {"time": 9, "action": "node-up", "node": "C"},
    {"time": 12, "action": "link-up", "from": "C", "to": "D"},
    {"time": 15, "action": "link-up", "from": "B", "to": "A"},
]


class TestReplay(unittest.TestCase):
    def _apply(self, event, down_nodes, down_links):
        action = event["action"]
        if action == "node-down":
            down_nodes.add(event["node"])
        elif action == "node-up":
            down_nodes.discard(event["node"])
        else:
            pair = frozenset((event["from"], event["to"]))
            if action == "link-down":
                down_links.add(pair)
            else:
                down_links.discard(pair)

    def test_timeline_matches_oracle_step_by_step(self):
        for name, topology, _ in ALL_TOPOLOGIES:
            nodes = topology["nodes"]
            if not nodes:
                continue
            # Drive every fixture through the same scenario shape: disable a
            # declared link and a node, then recover both. Skip fixtures
            # without the nodes/links the shared scenario references.
            with self.subTest(topology=name), tempfile.TemporaryDirectory() as tmp:
                scenario = []
                down_nodes, down_links = set(), set()
                time = 1
                if topology["links"]:
                    first = topology["links"][0]
                    scenario.append(
                        {"time": time, "action": "link-down",
                         "from": first["from"], "to": first["to"]}
                    )
                    time += 1
                victim = nodes[0]
                scenario.append({"time": time, "action": "node-down", "node": victim})
                time += 1
                scenario.append({"time": time, "action": "node-up", "node": victim})
                time += 1
                if topology["links"]:
                    first = topology["links"][0]
                    scenario.append(
                        {"time": time, "action": "link-up",
                         "from": first["to"], "to": first["from"]}
                    )
                proc = invoke_replay(tmp, topology, {"events": scenario})
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertEqual(proc.stderr, b"")
                replay = json.loads(proc.stdout)
                self.assertEqual(set(replay), {"protocol", "timeline"})
                self.assertEqual(replay["protocol"], "link-state")
                timeline = replay["timeline"]
                self.assertEqual(len(timeline), len(scenario) + 1)

                self.assertEqual(set(timeline[0]), {"event", "routers"})
                self.assertIsNone(timeline[0]["event"])
                for index, event in enumerate(scenario, start=1):
                    entry = timeline[index]
                    self.assertEqual(set(entry), {"time", "event", "routers"})
                    self.assertEqual(entry["time"], event["time"])
                    self.assertEqual(entry["event"], event)

                for index, entry in enumerate(timeline):
                    if index > 0:
                        self._apply(scenario[index - 1], down_nodes, down_links)
                    self.assertEqual(
                        normalize(entry["routers"]),
                        replay_oracle_routers(topology, down_nodes, down_links),
                        f"{name} timeline[{index}]",
                    )

    def test_baseline_equals_compute_and_key_order_is_sorted(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = invoke_replay(tmp, DIAMOND, {"events": REPLAY_SCENARIO})
            self.assertEqual(proc.returncode, 0, proc.stderr)
            replay = json.loads(proc.stdout)
            compute = json.loads(
                run_cli(["compute", str(write_topology(tmp, DIAMOND, "t2.json"))]).stdout
            )
            self.assertEqual(replay["timeline"][0]["routers"], compute["routers"])
            nodes = sorted(DIAMOND["nodes"])
            for entry in replay["timeline"]:
                self.assertEqual(list(entry["routers"]), nodes)
                for row in entry["routers"].values():
                    self.assertEqual(list(row), nodes)

    def test_down_router_row_and_down_destination_are_null(self):
        scenario = {"events": [{"time": 4, "action": "node-down", "node": "B"}]}
        with tempfile.TemporaryDirectory() as tmp:
            proc = invoke_replay(tmp, CHAIN3, scenario)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            routers = json.loads(proc.stdout)["timeline"][1]["routers"]
            for dest in ("A", "B", "C"):
                self.assertEqual(
                    routers["B"][dest], {"nextHop": None, "metric": None}
                )
                self.assertEqual(
                    routers[dest]["B"], {"nextHop": None, "metric": None}
                )
            self.assertEqual(routers["A"]["A"], {"nextHop": None, "metric": 0})
            # C is now unreachable from A (and vice versa) with B down.
            self.assertEqual(routers["A"]["C"], {"nextHop": None, "metric": None})
            self.assertEqual(routers["C"]["A"], {"nextHop": None, "metric": None})

    def test_link_down_state_survives_node_recovery(self):
        scenario = {
            "events": [
                {"time": 1, "action": "link-down", "from": "A", "to": "B"},
                {"time": 2, "action": "node-down", "node": "B"},
                {"time": 3, "action": "node-up", "node": "B"},
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            proc = invoke_replay(tmp, CHAIN3, scenario)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            routers = json.loads(proc.stdout)["timeline"][3]["routers"]
            # B recovered, but A-B was link-down'd, so A and B stay split.
            self.assertEqual(routers["A"]["B"], {"nextHop": None, "metric": None})
            self.assertEqual(routers["A"]["C"], {"nextHop": None, "metric": None})
            self.assertEqual(routers["B"]["C"], {"nextHop": "C", "metric": 7})

    def test_empty_events_and_byte_identity_across_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs = set()
            for seed in HASH_SEEDS:
                proc = invoke_replay(tmp, RING, {"events": []}, seed=seed)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertEqual(proc.stderr, b"")
                outputs.add(proc.stdout)
            self.assertEqual(len(outputs), 1)
            replay = json.loads(outputs.pop())
            self.assertEqual(len(replay["timeline"]), 1)
            self.assertIsNone(replay["timeline"][0]["event"])

    def test_full_scenario_byte_identity_across_seeds_and_declaration_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs = set()
            for seed in HASH_SEEDS:
                for index, variant in enumerate(EQUIVALENT_VARIANTS):
                    topo_path = write_topology(tmp, variant, f"v{index}.json")
                    scen_path = write_scenario(
                        tmp, {"events": REPLAY_SCENARIO}, f"s{index}.json"
                    )
                    proc = run_cli(
                        ["replay", str(topo_path), str(scen_path)], seed=seed
                    )
                    self.assertEqual(proc.returncode, 0, proc.stderr)
                    outputs.add(proc.stdout)
            self.assertEqual(len(outputs), 1)


class TestReplayInvalidScenarios(unittest.TestCase):
    def _assert_invalid(self, directory, scenario):
        proc = invoke_replay(directory, CHAIN3, scenario)
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stdout, b"")
        self.assertEqual(proc.stderr, b"invalid scenario\n")

    def test_invalid_scenarios(self):
        node_down = {"time": 1, "action": "node-down", "node": "A"}
        link_down = {"time": 1, "action": "link-down", "from": "A", "to": "B"}
        invalid = [
            [],
            "just a string",
            42,
            {},
            {"events": {}},
            {"events": "x"},
            {"events": ["x"]},
            {"events": [None]},
            # Missing or mistyped fields.
            {"events": [{"action": "node-down", "node": "A"}]},
            {"events": [{"time": 1}]},
            {"events": [{"time": 1, "action": 7, "node": "A"}]},
            {"events": [{"time": 1, "action": "explode", "node": "A"}]},
            {"events": [{"time": 1, "action": "node-down"}]},
            {"events": [{"time": 1, "action": "node-down", "node": 3}]},
            {"events": [{"time": 1, "action": "link-down", "from": "A"}]},
            {"events": [{"time": 1, "action": "link-down", "from": "A", "to": 2}]},
            # Bad times: zero, negative, non-integer, boolean, duplicate,
            # regressed.
            {"events": [dict(node_down, time=0)]},
            {"events": [dict(node_down, time=-2)]},
            {"events": [dict(node_down, time=1.5)]},
            {"events": [dict(node_down, time="3")]},
            {"events": [dict(node_down, time=True)]},
            {"events": [dict(node_down, time=2), dict(link_down, time=2)]},
            {"events": [dict(node_down, time=5), dict(link_down, time=3)]},
            # Unknown objects.
            {"events": [dict(node_down, node="ZZ")]},
            {"events": [dict(link_down, to="ZZ")]},
            {"events": [dict(link_down, **{"from": "A", "to": "C"})]},  # no link
            {"events": [dict(link_down, **{"from": "A", "to": "A"})]},  # no link
            # Illegal transitions.
            {"events": [{"time": 1, "action": "node-up", "node": "A"}]},
            {
                "events": [
                    dict(node_down, time=1),
                    dict(node_down, time=2),
                ]
            },
            {"events": [{"time": 1, "action": "link-up", "from": "A", "to": "B"}]},
            {
                "events": [
                    dict(link_down, time=1),
                    # Reversed direction still names the same link.
                    dict(link_down, time=2, **{"from": "B", "to": "A"}),
                ]
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for index, scenario in enumerate(invalid):
                with self.subTest(case=index):
                    self._assert_invalid(tmp, scenario)

    def test_unreadable_scenario(self):
        with tempfile.TemporaryDirectory() as tmp:
            topo_path = write_topology(tmp, CHAIN3)
            missing = str(Path(tmp) / "does-not-exist.json")
            proc = run_cli(["replay", str(topo_path), missing])
            self.assertEqual(proc.returncode, 2)
            self.assertEqual(proc.stdout, b"")
            self.assertEqual(
                proc.stderr, f"cannot read scenario: {missing}\n".encode()
            )

    def test_unparseable_scenario(self):
        with tempfile.TemporaryDirectory() as tmp:
            topo_path = write_topology(tmp, CHAIN3)
            scen_path = Path(tmp) / "broken.json"
            scen_path.write_text("{not valid json", encoding="utf-8")
            proc = run_cli(["replay", str(topo_path), str(scen_path)])
            self.assertEqual(proc.returncode, 2)
            self.assertEqual(proc.stdout, b"")
            self.assertEqual(proc.stderr, b"invalid scenario\n")

    def test_topology_errors_take_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing_topo = str(Path(tmp) / "missing.json")
            missing_scen = str(Path(tmp) / "missing2.json")
            proc = run_cli(["replay", missing_topo, missing_scen])
            self.assertEqual(proc.returncode, 2)
            self.assertEqual(proc.stdout, b"")
            self.assertEqual(
                proc.stderr, f"cannot read topology: {missing_topo}\n".encode()
            )

            bad_topo = write_topology(tmp, {"nodes": ["A", "A"], "links": []})
            scen_path = write_scenario(tmp, {"events": []})
            proc = run_cli(["replay", str(bad_topo), str(scen_path)])
            self.assertEqual(proc.returncode, 2)
            self.assertEqual(proc.stdout, b"")
            self.assertEqual(proc.stderr, b"invalid topology\n")

    def test_argument_count_errors(self):
        for args in (
            ["replay"],
            ["replay", "a.json"],
            ["replay", "a.json", "b.json", "c.json"],
        ):
            with self.subTest(args=args):
                proc = run_cli(args)
                self.assertEqual(proc.returncode, 2)
                self.assertEqual(proc.stdout, b"")
                self.assertEqual(proc.stderr, REPLAY_USAGE.encode())


# ---------------------------------------------------------------------------
# replay-dv: independent oracle, count-to-infinity trace, errors
# ---------------------------------------------------------------------------


DV_CHAIN = {
    "nodes": ["A", "B", "C"],
    "links": [link("A", "B", 1), link("B", "C", 1)],
}


def invoke_replay_dv(directory, topology, scenario, seed=0):
    topo_path = write_topology(directory, topology)
    scenario_path = write_scenario(directory, scenario)
    return run_cli(
        ["replay-dv", str(topo_path), str(scenario_path)], seed=seed
    )


def dv_round_zero_after(previous, nodes, active, down_nodes):
    """Independent re-derivation of replay-dv's post-event round 0.

    Online routers keep their prior advertisements, dropping only routes
    whose next hop ceased to be an available direct neighbor; available
    direct links always readvertise; a recovered router starts with no
    reachable prior entries, leaving only itself and direct neighbors.
    """
    table = {}
    for router in nodes:
        if router in down_nodes:
            table[router] = {dest: (None, None) for dest in nodes}
            continue
        row = {}
        for dest in nodes:
            if dest == router:
                row[dest] = (None, 0)
            elif dest in active[router]:
                row[dest] = (dest, active[router][dest])
            else:
                hop, metric = previous[router, dest]
                if hop is None or hop not in active[router]:
                    row[dest] = (None, None)
                else:
                    row[dest] = (hop, metric)
        table[router] = row
    return {(r, d): table[r][d] for r in nodes for d in nodes}


def dv_synchronous(previous, nodes, active, down_nodes, infinity_metric):
    """Independent synchronous Bellman-Ford round with infinity clamping."""
    current = {}
    for router in nodes:
        if router in down_nodes:
            current.update({(router, d): (None, None) for d in nodes})
            continue
        for dest in nodes:
            if dest == router:
                current[router, dest] = (None, 0)
                continue
            best_cost, best_hop = None, None
            for neighbor in sorted(active[router]):
                advertised_cost = previous[neighbor, dest][1]
                if advertised_cost is None:
                    continue
                cost = active[router][neighbor] + advertised_cost
                if cost >= infinity_metric:
                    continue
                if best_cost is None or cost < best_cost:
                    best_cost, best_hop = cost, neighbor
            current[router, dest] = (
                (None, None) if best_cost is None else (best_hop, best_cost)
            )
    return current


def dv_expected_timeline(topology, scenario):
    """Build the full normalized expected replay-dv timeline independently."""
    nodes, adjacency = build_adjacency(topology)
    infinity_metric = scenario["infinityMetric"]
    down_nodes, down_links = set(), set()

    def active_map():
        disabled = {frozenset(p) for p in down_links}
        active = {node: {} for node in nodes}
        for source in nodes:
            if source in down_nodes:
                continue
            for target, metric in adjacency[source].items():
                if target in down_nodes:
                    continue
                if frozenset((source, target)) in disabled:
                    continue
                active[source][target] = metric
        return active

    def converge(initial):
        rounds = [initial]
        current = initial
        while True:
            updated = dv_synchronous(
                current, nodes, active_map(), down_nodes, infinity_metric
            )
            if updated == current:
                return rounds, current
            current = updated
            rounds.append(current)

    # Fault-free baseline round 0 then synchronous convergence.
    vectors = round_zero(nodes, active_map())
    baseline_rounds, vectors = converge(vectors)
    timeline = [(None, baseline_rounds)]

    for event in scenario["events"]:
        if event["action"] == "node-down":
            down_nodes.add(event["node"])
        elif event["action"] == "node-up":
            down_nodes.discard(event["node"])
        elif event["action"] == "link-down":
            down_links.add(frozenset((event["from"], event["to"])))
        else:
            down_links.discard(frozenset((event["from"], event["to"])))
        vectors = dv_round_zero_after(vectors, nodes, active_map(), down_nodes)
        rounds, vectors = converge(vectors)
        timeline.append((event["time"], rounds))
    return timeline


class TestReplayDV(unittest.TestCase):
    def _apply_state(self, event, down_nodes, down_links):
        action = event["action"]
        if action == "node-down":
            down_nodes.add(event["node"])
        elif action == "node-up":
            down_nodes.discard(event["node"])
        elif action == "link-down":
            down_links.add(frozenset((event["from"], event["to"])))
        else:
            down_links.discard(frozenset((event["from"], event["to"])))

    def test_envelope_and_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = invoke_replay_dv(
                tmp, DV_CHAIN, {"infinityMetric": 8, "events": []}
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stderr, b"")
            result = json.loads(proc.stdout)
            self.assertEqual(
                set(result), {"protocol", "infinityMetric", "timeline"}
            )
            self.assertEqual(result["protocol"], "distance-vector")
            self.assertEqual(result["infinityMetric"], 8)
            baseline = result["timeline"][0]
            self.assertEqual(
                set(baseline), {"event", "convergenceRound", "rounds"}
            )
            self.assertIsNone(baseline["event"])
            self.assertEqual(
                [r["round"] for r in baseline["rounds"]], [0, 1]
            )
            self.assertEqual(baseline["convergenceRound"], 1)

    def test_timeline_matches_independent_oracle(self):
        scenario = {
            "infinityMetric": 30,
            "events": list(REPLAY_SCENARIO),
        }
        with tempfile.TemporaryDirectory() as tmp:
            proc = invoke_replay_dv(tmp, DIAMOND, scenario)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            result = json.loads(proc.stdout)
            expected = dv_expected_timeline(DIAMOND, scenario)
            self.assertEqual(len(result["timeline"]), len(expected))
            for index, (entry, (expected_time, expected_rounds)) in enumerate(
                zip(result["timeline"], expected)
            ):
                actual_rounds = entry["rounds"]
                self.assertEqual(
                    [r["round"] for r in actual_rounds],
                    list(range(len(actual_rounds))),
                    index,
                )
                self.assertEqual(
                    entry["convergenceRound"],
                    actual_rounds[-1]["round"],
                    index,
                )
                for actual, want in zip(actual_rounds, expected_rounds):
                    self.assertEqual(
                        normalize(actual["routers"]), want, (index, actual["round"])
                    )
                if expected_time is None:
                    self.assertIsNone(entry["event"])
                else:
                    self.assertEqual(entry["time"], expected_time)
                    self.assertEqual(
                        entry["event"], scenario["events"][index - 1]
                    )

    def test_hand_traced_count_to_infinity(self):
        scenario = {
            "infinityMetric": 8,
            "events": [{"time": 1, "action": "node-down", "node": "C"}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            proc = invoke_replay_dv(tmp, DV_CHAIN, scenario)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            rounds = json.loads(proc.stdout)["timeline"][1]["rounds"]
            metrics = [
                (r["routers"]["A"]["C"], r["routers"]["B"]["C"])
                for r in rounds
            ]
            null = {"nextHop": None, "metric": None}
            self.assertEqual(
                metrics,
                [
                    ({"nextHop": "B", "metric": 2}, null),
                    (null, {"nextHop": "A", "metric": 3}),
                    ({"nextHop": "B", "metric": 4}, null),
                    (null, {"nextHop": "A", "metric": 5}),
                    ({"nextHop": "B", "metric": 6}, null),
                    (null, {"nextHop": "A", "metric": 7}),
                    (null, null),
                ],
            )
            self.assertEqual(
                json.loads(proc.stdout)["timeline"][1]["convergenceRound"], 6
            )

    def test_node_and_link_recovery_round_trips_to_shortest_paths(self):
        scenario = {
            "infinityMetric": 30,
            "events": [
                {"time": 1, "action": "node-down", "node": "C"},
                {"time": 2, "action": "node-up", "node": "C"},
                {"time": 3, "action": "link-down", "from": "A", "to": "B"},
                {"time": 4, "action": "link-up", "from": "B", "to": "A"},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            proc = invoke_replay_dv(tmp, DIAMOND, scenario)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            result = json.loads(proc.stdout)
            # Recovered link readvertises as a direct route at round 0.
            recovered = result["timeline"][4]["rounds"][0]["routers"]
            self.assertEqual(
                recovered["A"]["B"], {"nextHop": "B", "metric": 1}
            )
            final = result["timeline"][-1]["rounds"][-1]["routers"]
            compute = json.loads(
                run_cli(
                    ["compute", str(write_topology(tmp, DIAMOND, "t.json"))]
                ).stdout
            )
            self.assertEqual(final, compute["routers"])

    def test_down_router_whole_row_is_null(self):
        scenario = {
            "infinityMetric": 8,
            "events": [{"time": 1, "action": "node-down", "node": "B"}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            proc = invoke_replay_dv(tmp, DV_CHAIN, scenario)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            for snapshot in json.loads(proc.stdout)["timeline"][1]["rounds"]:
                for dest in ("A", "B", "C"):
                    self.assertEqual(
                        snapshot["routers"]["B"][dest],
                        {"nextHop": None, "metric": None},
                    )

    def test_empty_topology_and_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = invoke_replay_dv(
                tmp, EMPTY, {"infinityMetric": 1, "events": []}
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            result = json.loads(proc.stdout)
            self.assertEqual(result["infinityMetric"], 1)
            self.assertEqual(
                result["timeline"],
                [
                    {
                        "event": None,
                        "convergenceRound": 0,
                        "rounds": [{"round": 0, "routers": {}}],
                    }
                ],
            )

    def test_byte_identity_across_seeds_and_declaration_order(self):
        scenario = {"infinityMetric": 30, "events": list(REPLAY_SCENARIO)}
        with tempfile.TemporaryDirectory() as tmp:
            outputs = set()
            for seed in HASH_SEEDS:
                for index, variant in enumerate(EQUIVALENT_VARIANTS):
                    topo_path = write_topology(tmp, variant, f"v{index}.json")
                    scen_path = write_scenario(
                        tmp, scenario, f"s{index}.json"
                    )
                    proc = run_cli(
                        ["replay-dv", str(topo_path), str(scen_path)],
                        seed=seed,
                    )
                    self.assertEqual(proc.returncode, 0, proc.stderr)
                    outputs.add(proc.stdout)
            self.assertEqual(len(outputs), 1)


class TestReplayDVInvalidScenarios(unittest.TestCase):
    def _assert_invalid(self, directory, scenario):
        proc = invoke_replay_dv(directory, CHAIN3, scenario)
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stdout, b"")
        self.assertEqual(proc.stderr, b"invalid scenario\n")

    def test_invalid_infinity_metric(self):
        invalid = [
            {},
            {"events": []},
            {"infinityMetric": None, "events": []},
            {"infinityMetric": True, "events": []},
            {"infinityMetric": 0, "events": []},
            {"infinityMetric": -5, "events": []},
            {"infinityMetric": 1.5, "events": []},
            {"infinityMetric": "8", "events": []},
            # CHAIN3's largest link metric is 7: must be strictly greater.
            {"infinityMetric": 7, "events": []},
            {"infinityMetric": 1, "events": []},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for index, scenario in enumerate(invalid):
                with self.subTest(case=index):
                    self._assert_invalid(tmp, scenario)

    def test_event_errors_match_replay_conventions(self):
        cases = [
            {"infinityMetric": 50, "events": [None]},
            {
                "infinityMetric": 50,
                "events": [
                    {"time": 1, "action": "node-up", "node": "A"}
                ],
            },
            {
                "infinityMetric": 50,
                "events": [
                    {"time": 1, "action": "link-down", "from": "A", "to": "Z"}
                ],
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for index, scenario in enumerate(cases):
                with self.subTest(case=index):
                    self._assert_invalid(tmp, scenario)

    def test_unreadable_and_unparseable_scenario(self):
        with tempfile.TemporaryDirectory() as tmp:
            topo_path = write_topology(tmp, CHAIN3)
            missing = str(Path(tmp) / "nope.json")
            proc = run_cli(["replay-dv", str(topo_path), missing])
            self.assertEqual(proc.returncode, 2)
            self.assertEqual(proc.stdout, b"")
            self.assertEqual(
                proc.stderr, f"cannot read scenario: {missing}\n".encode()
            )
            broken = Path(tmp) / "broken.json"
            broken.write_text("{bad", encoding="utf-8")
            proc = run_cli(["replay-dv", str(topo_path), str(broken)])
            self.assertEqual(proc.returncode, 2)
            self.assertEqual(proc.stdout, b"")
            self.assertEqual(proc.stderr, b"invalid scenario\n")

    def test_topology_errors_take_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing_topo = str(Path(tmp) / "missing.json")
            proc = run_cli(["replay-dv", missing_topo, "also-missing.json"])
            self.assertEqual(proc.returncode, 2)
            self.assertEqual(proc.stdout, b"")
            self.assertEqual(
                proc.stderr, f"cannot read topology: {missing_topo}\n".encode()
            )
            bad_topo = write_topology(tmp, {"nodes": ["A", "A"], "links": []})
            scen = write_scenario(tmp, {"infinityMetric": 1, "events": []})
            proc = run_cli(["replay-dv", str(bad_topo), str(scen)])
            self.assertEqual(proc.returncode, 2)
            self.assertEqual(proc.stdout, b"")
            self.assertEqual(proc.stderr, b"invalid topology\n")

    def test_argument_count_errors(self):
        for args in (
            ["replay-dv"],
            ["replay-dv", "a.json"],
            ["replay-dv", "a.json", "b.json", "c.json"],
        ):
            with self.subTest(args=args):
                proc = run_cli(args)
                self.assertEqual(proc.returncode, 2)
                self.assertEqual(proc.stdout, b"")
                self.assertEqual(proc.stderr, REPLAY_DV_USAGE.encode())


# ---------------------------------------------------------------------------
# Existing CLI contract that must not change
# ---------------------------------------------------------------------------

class TestCLIContract(unittest.TestCase):
    def test_version(self):
        proc = run_cli(["version"])
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(proc.stdout, f"{VERSION}\n".encode())
        self.assertEqual(proc.stderr, b"")

    def test_help_forms(self):
        for args in ([], ["help"], ["-h"], ["--help"]):
            with self.subTest(args=args):
                proc = run_cli(args)
                self.assertEqual(proc.returncode, 0)
                self.assertEqual(proc.stdout, USAGE.encode())
                self.assertEqual(proc.stderr, b"")

    def test_unknown_command(self):
        proc = run_cli(["frobnicate"])
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stdout, b"")
        self.assertEqual(
            proc.stderr, f"unknown command: frobnicate\n{USAGE}".encode()
        )

    def test_argument_count_errors(self):
        cases = [
            (["compute"], COMPUTE_USAGE),
            (["compute", "a.json", "b.json"], COMPUTE_USAGE),
            (["converge"], CONVERGE_USAGE),
            (["converge", "a.json", "b.json"], CONVERGE_USAGE),
        ]
        for args, usage in cases:
            with self.subTest(args=args):
                proc = run_cli(args)
                self.assertEqual(proc.returncode, 2)
                self.assertEqual(proc.stdout, b"")
                self.assertEqual(proc.stderr, usage.encode())

    def test_unreadable_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = str(Path(tmp) / "does-not-exist.json")
            for command in ("compute", "converge"):
                with self.subTest(command=command):
                    proc = run_cli([command, missing])
                    self.assertEqual(proc.returncode, 2)
                    self.assertEqual(proc.stdout, b"")
                    self.assertEqual(
                        proc.stderr,
                        f"cannot read topology: {missing}\n".encode(),
                    )

    def test_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.json"
            path.write_text("{not valid json", encoding="utf-8")
            for command in ("compute", "converge"):
                with self.subTest(command=command):
                    proc = run_cli([command, str(path)])
                    self.assertEqual(proc.returncode, 2)
                    self.assertEqual(proc.stdout, b"")
                    self.assertEqual(proc.stderr, b"invalid topology\n")

    def test_invalid_topologies(self):
        invalid = [
            [],
            "just a string",
            42,
            {},
            {"nodes": "A", "links": []},
            {"nodes": [], "links": {}},
            {"nodes": [1], "links": []},
            {"nodes": [""], "links": []},
            {"nodes": ["A", "A"], "links": []},
            {"nodes": ["A"]},
            {"links": []},
            {"nodes": ["A"], "links": [{"from": "A", "to": "B"}]},
            {"nodes": ["A"], "links": ["not-a-link"]},
            {"nodes": ["A", "B"], "links": [{"from": "A", "to": "B"}]},
            {"nodes": ["A", "B"], "links": [{"from": "A", "to": "B", "metric": 0}]},
            {"nodes": ["A", "B"], "links": [{"from": "A", "to": "B", "metric": -3}]},
            {"nodes": ["A", "B"], "links": [{"from": "A", "to": "B", "metric": 1.5}]},
            {"nodes": ["A", "B"], "links": [{"from": "A", "to": "B", "metric": "1"}]},
            {"nodes": ["A", "B"], "links": [{"from": "A", "to": "B", "metric": True}]},
            {"nodes": ["A"], "links": [{"from": "A", "to": "A", "metric": 1}]},
            {"nodes": ["A", "B"], "links": [{"from": "B", "to": "unknown", "metric": 1}]},
            {
                "nodes": ["A", "B"],
                "links": [link("A", "B", 1), link("B", "A", 1)],
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for index, topology in enumerate(invalid):
                path = write_topology(tmp, topology, f"invalid-{index}.json")
                for command in ("compute", "converge"):
                    with self.subTest(command=command, case=index):
                        proc = run_cli([command, str(path)])
                        self.assertEqual(proc.returncode, 2)
                        self.assertEqual(proc.stdout, b"")
                        self.assertEqual(proc.stderr, b"invalid topology\n")


if __name__ == "__main__":
    unittest.main()
