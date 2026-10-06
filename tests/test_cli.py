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
