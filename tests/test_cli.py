"""Differential and property tests for the public compute/converge CLI.

Every expectation in this suite is either hand computed from the fixed
topologies below or derived by an independent test-side re-derivation of the
documented distance-vector rules.  The package's ``forwarding_table`` and
``distance_vector_round`` functions are never used to produce expected
values, so an implementation defect cannot contaminate both sides of an
assertion.  The suite only exercises the documented command line interface:
JSON on standard output, exit codes, and standard error text.
"""

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from packet_routing_sim import __version__  # noqa: E402

USAGE = """usage: python3 -m packet_routing_sim <command>

commands:
  version                       print the package version
  compute TOPOLOGY.json         compute shortest-path forwarding tables
                                from a static undirected topology
  converge TOPOLOGY.json        show distance-vector convergence round
                                by round from a static undirected topology
  help                          print this message
"""
COMPUTE_USAGE = "usage: python3 -m packet_routing_sim compute TOPOLOGY.json\n"
CONVERGE_USAGE = "usage: python3 -m packet_routing_sim converge TOPOLOGY.json\n"


def link(source, target, metric):
    return {"from": source, "to": target, "metric": metric}


def entry(next_hop, metric):
    return {"nextHop": next_hop, "metric": metric}


# --------------------------------------------------------------------------
# Fixed, reproducible test topologies.
# --------------------------------------------------------------------------

SINGLE = {"nodes": ["solo"], "links": []}

CHAIN = {
    "nodes": ["a", "b", "c", "d"],
    "links": [link("a", "b", 1), link("b", "c", 2), link("c", "d", 3)],
}

RING = {
    "nodes": ["a", "b", "c", "d"],
    "links": [
        link("a", "b", 1),
        link("b", "c", 1),
        link("c", "d", 1),
        link("d", "a", 1),
    ],
}

# Diamond with two equal-cost s->t paths; ties must prefer next hop "a".
EQUAL_COST = {
    "nodes": ["s", "a", "b", "t"],
    "links": [
        link("s", "a", 1),
        link("s", "b", 1),
        link("a", "t", 2),
        link("b", "t", 2),
    ],
}

# Two connected components plus one fully isolated node.
DISCONNECTED = {
    "nodes": ["a", "b", "c", "d", "e"],
    "links": [link("a", "b", 1), link("c", "d", 2)],
}

CHAIN_WITH_ISOLATED = {
    "nodes": ["a", "b", "c", "d", "iso"],
    "links": CHAIN["links"],
}

# Semantically identical topology declared in different ways: node order,
# link direction, and link order must not affect the output.
EQUIV_VARIANTS = [
    {"nodes": ["x", "y", "z"], "links": [link("x", "y", 5), link("y", "z", 7)]},
    {"nodes": ["z", "y", "x"], "links": [link("x", "y", 5), link("y", "z", 7)]},
    {"nodes": ["x", "y", "z"], "links": [link("y", "x", 5), link("z", "y", 7)]},
    {"nodes": ["x", "y", "z"], "links": [link("y", "z", 7), link("x", "y", 5)]},
    {"nodes": ["z", "x", "y"], "links": [link("z", "y", 7), link("y", "x", 5)]},
]

TOPOLOGIES = {
    "single": SINGLE,
    "chain": CHAIN,
    "ring": RING,
    "equal_cost": EQUAL_COST,
    "disconnected": DISCONNECTED,
    "chain_with_isolated": CHAIN_WITH_ISOLATED,
    "equiv_base": EQUIV_VARIANTS[0],
}

UNREACHABLE = entry(None, None)

# Hand-computed forwarding tables for the small topologies.
EXPECTED_ROUTERS = {
    "single": {
        "solo": {"solo": entry(None, 0)},
    },
    "chain": {
        "a": {"a": entry(None, 0), "b": entry("b", 1), "c": entry("b", 3), "d": entry("b", 6)},
        "b": {"a": entry("a", 1), "b": entry(None, 0), "c": entry("c", 2), "d": entry("c", 5)},
        "c": {"a": entry("b", 3), "b": entry("b", 2), "c": entry(None, 0), "d": entry("d", 3)},
        "d": {"a": entry("c", 6), "b": entry("c", 5), "c": entry("c", 3), "d": entry(None, 0)},
    },
    "ring": {
        "a": {"a": entry(None, 0), "b": entry("b", 1), "c": entry("b", 2), "d": entry("d", 1)},
        "b": {"a": entry("a", 1), "b": entry(None, 0), "c": entry("c", 1), "d": entry("a", 2)},
        "c": {"a": entry("b", 2), "b": entry("b", 1), "c": entry(None, 0), "d": entry("d", 1)},
        "d": {"a": entry("a", 1), "b": entry("a", 2), "c": entry("c", 1), "d": entry(None, 0)},
    },
    "equal_cost": {
        "a": {"a": entry(None, 0), "b": entry("s", 2), "s": entry("s", 1), "t": entry("t", 2)},
        "b": {"a": entry("s", 2), "b": entry(None, 0), "s": entry("s", 1), "t": entry("t", 2)},
        "s": {"a": entry("a", 1), "b": entry("b", 1), "s": entry(None, 0), "t": entry("a", 3)},
        "t": {"a": entry("a", 2), "b": entry("b", 2), "s": entry("a", 3), "t": entry(None, 0)},
    },
    "disconnected": {
        "a": {"a": entry(None, 0), "b": entry("b", 1), "c": UNREACHABLE, "d": UNREACHABLE, "e": UNREACHABLE},
        "b": {"a": entry("a", 1), "b": entry(None, 0), "c": UNREACHABLE, "d": UNREACHABLE, "e": UNREACHABLE},
        "c": {"a": UNREACHABLE, "b": UNREACHABLE, "c": entry(None, 0), "d": entry("d", 2), "e": UNREACHABLE},
        "d": {"a": UNREACHABLE, "b": UNREACHABLE, "c": entry("c", 2), "d": entry(None, 0), "e": UNREACHABLE},
        "e": {"a": UNREACHABLE, "b": UNREACHABLE, "c": UNREACHABLE, "d": UNREACHABLE, "e": entry(None, 0)},
    },
}

# Hand-computed first stable round for each topology.
EXPECTED_CONVERGENCE_ROUNDS = {
    "single": 0,
    "chain": 2,
    "ring": 1,
    "equal_cost": 1,
    "disconnected": 0,
    "chain_with_isolated": 2,
    "equiv_base": 1,
}


# --------------------------------------------------------------------------
# Test-side re-derivation of the documented distance-vector rules.
# --------------------------------------------------------------------------

def adjacency_of(topology):
    adjacency = {node: {} for node in topology["nodes"]}
    for item in topology["links"]:
        adjacency[item["from"]][item["to"]] = item["metric"]
        adjacency[item["to"]][item["from"]] = item["metric"]
    return adjacency


def expected_round_zero(nodes, adjacency):
    """Round 0 by specification: each router knows itself and direct links."""
    vectors = {}
    for router in nodes:
        table = {}
        for destination in nodes:
            if destination == router:
                table[destination] = entry(None, 0)
            elif destination in adjacency[router]:
                table[destination] = entry(destination, adjacency[router][destination])
            else:
                table[destination] = entry(None, None)
        vectors[router] = table
    return vectors


def synchronous_step(previous, nodes, adjacency):
    """One synchronous distance-vector round derived only from ``previous``.

    Every router's new table is computed exclusively from its neighbors'
    previous-round advertisements; ties on total metric keep the
    lexicographically smaller neighbor name.
    """
    current = {}
    for router in nodes:
        table = {}
        for destination in nodes:
            if destination == router:
                table[destination] = entry(None, 0)
                continue
            best_metric = None
            best_hop = None
            for neighbor in sorted(adjacency[router]):
                advertised = previous[neighbor][destination]
                if advertised["metric"] is None:
                    continue
                candidate = adjacency[router][neighbor] + advertised["metric"]
                if best_metric is None or candidate < best_metric:
                    best_metric = candidate
                    best_hop = neighbor
            table[destination] = entry(best_hop, best_metric)
        current[router] = table
    return current


# --------------------------------------------------------------------------
# Shared CLI helpers.
# --------------------------------------------------------------------------

class CliTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def run_cli(self, *args, hashseed=None):
        env = dict(os.environ)
        if hashseed is not None:
            env["PYTHONHASHSEED"] = hashseed
        return subprocess.run(
            [sys.executable, "-m", "packet_routing_sim", *args],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
        )

    def write_file(self, name, content):
        path = os.path.join(self.tmpdir.name, name)
        mode = "wb" if isinstance(content, bytes) else "w"
        with open(path, mode) as handle:
            handle.write(content)
        return path

    def topology_path(self, topology, name="topology.json"):
        return self.write_file(name, json.dumps(topology))

    def command_json(self, command, topology, name="topology.json", hashseed=None):
        result = self.run_cli(command, self.topology_path(topology, name), hashseed=hashseed)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, b"")
        return json.loads(result.stdout)

    def compute_json(self, topology, name="topology.json", hashseed=None):
        return self.command_json("compute", topology, name, hashseed)

    def converge_json(self, topology, name="topology.json", hashseed=None):
        return self.command_json("converge", topology, name, hashseed)


# --------------------------------------------------------------------------
# compute: hand-computed tables and structural properties.
# --------------------------------------------------------------------------

class ComputeTableTests(CliTestCase):
    def test_tables_match_hand_computed_expectations(self):
        for name, expected in EXPECTED_ROUTERS.items():
            with self.subTest(topology=name):
                output = self.compute_json(TOPOLOGIES[name])
                self.assertEqual(set(output), {"routers"})
                self.assertEqual(output["routers"], expected)

    def test_equal_cost_ties_choose_smaller_direct_next_hop(self):
        routers = self.compute_json(EQUAL_COST)["routers"]
        self.assertEqual(routers["s"]["t"], entry("a", 3))
        self.assertEqual(routers["t"]["s"], entry("a", 3))


class TableStructureTests(CliTestCase):
    def check_table_structure(self, routers, topology):
        nodes = set(topology["nodes"])
        adjacency = adjacency_of(topology)
        self.assertEqual(set(routers), nodes)
        for router, table in routers.items():
            self.assertEqual(set(table), nodes)
            for destination, item in table.items():
                self.assertEqual(set(item), {"nextHop", "metric"})
                if destination == router:
                    self.assertIsNone(item["nextHop"])
                    self.assertEqual(item["metric"], 0)
                elif item["metric"] is None:
                    self.assertIsNone(item["nextHop"])
                else:
                    self.assertIsInstance(item["metric"], int)
                    self.assertGreater(item["metric"], 0)
                    self.assertIn(item["nextHop"], adjacency[router])

    def test_compute_tables_cover_all_declared_destinations(self):
        for name, topology in TOPOLOGIES.items():
            with self.subTest(topology=name):
                self.check_table_structure(self.compute_json(topology)["routers"], topology)

    def test_every_converge_snapshot_covers_all_declared_destinations(self):
        for name, topology in TOPOLOGIES.items():
            with self.subTest(topology=name):
                output = self.converge_json(topology)
                for snapshot in output["rounds"]:
                    self.check_table_structure(snapshot["routers"], topology)


class IsolatedNodeTests(CliTestCase):
    def test_adding_isolated_node_preserves_existing_routes(self):
        base_routers = self.compute_json(CHAIN, "base.json")["routers"]
        extended_routers = self.compute_json(CHAIN_WITH_ISOLATED, "extended.json")["routers"]
        for router in CHAIN["nodes"]:
            for destination in CHAIN["nodes"]:
                self.assertEqual(extended_routers[router][destination],
                                 base_routers[router][destination])
            self.assertEqual(extended_routers[router]["iso"], UNREACHABLE)
        for destination in CHAIN["nodes"]:
            self.assertEqual(extended_routers["iso"][destination], UNREACHABLE)
        self.assertEqual(extended_routers["iso"]["iso"], entry(None, 0))

    def test_adding_isolated_node_preserves_converged_routes(self):
        base_final = self.converge_json(CHAIN, "base.json")["rounds"][-1]["routers"]
        extended_final = self.converge_json(CHAIN_WITH_ISOLATED, "extended.json")["rounds"][-1]["routers"]
        for router in CHAIN["nodes"]:
            for destination in CHAIN["nodes"]:
                self.assertEqual(extended_final[router][destination],
                                 base_final[router][destination])
            self.assertEqual(extended_final[router]["iso"], UNREACHABLE)
        for destination in CHAIN["nodes"]:
            self.assertEqual(extended_final["iso"][destination], UNREACHABLE)


# --------------------------------------------------------------------------
# Differential: both public paths agree on the converged state.
# --------------------------------------------------------------------------

class DifferentialTests(CliTestCase):
    def test_converged_state_matches_compute(self):
        for name, topology in TOPOLOGIES.items():
            with self.subTest(topology=name):
                compute = self.compute_json(topology)
                converge = self.converge_json(topology)
                self.assertEqual(set(converge), {"protocol", "convergenceRound", "rounds"})
                self.assertEqual(converge["protocol"], "distance-vector")
                final = converge["rounds"][-1]
                self.assertEqual(converge["convergenceRound"], final["round"])
                self.assertEqual(final["routers"], compute["routers"])

    def test_converged_ties_choose_smaller_direct_next_hop(self):
        final = self.converge_json(EQUAL_COST)["rounds"][-1]["routers"]
        self.assertEqual(final["s"]["t"], entry("a", 3))
        self.assertEqual(final["t"]["s"], entry("a", 3))


# --------------------------------------------------------------------------
# converge: the recorded trajectory itself.
# --------------------------------------------------------------------------

class ConvergenceTrajectoryTests(CliTestCase):
    def test_round_numbers_are_sequential_and_convergence_round_is_last(self):
        for name, topology in TOPOLOGIES.items():
            with self.subTest(topology=name):
                output = self.converge_json(topology)
                rounds = output["rounds"]
                self.assertEqual([s["round"] for s in rounds], list(range(len(rounds))))
                self.assertEqual(output["convergenceRound"], rounds[-1]["round"])
                self.assertEqual(output["convergenceRound"], EXPECTED_CONVERGENCE_ROUNDS[name])

    def test_round_zero_holds_self_and_direct_neighbors_only(self):
        for name, topology in TOPOLOGIES.items():
            with self.subTest(topology=name):
                output = self.converge_json(topology)
                nodes = sorted(topology["nodes"])
                expected = expected_round_zero(nodes, adjacency_of(topology))
                self.assertEqual(output["rounds"][0]["routers"], expected)

    def test_each_round_derives_synchronously_from_previous_round(self):
        for name, topology in TOPOLOGIES.items():
            with self.subTest(topology=name):
                output = self.converge_json(topology)
                nodes = sorted(topology["nodes"])
                adjacency = adjacency_of(topology)
                rounds = output["rounds"]
                for index in range(1, len(rounds)):
                    expected = synchronous_step(rounds[index - 1]["routers"], nodes, adjacency)
                    self.assertEqual(rounds[index]["routers"], expected)

    def test_recorded_adjacent_snapshots_always_differ(self):
        for name, topology in TOPOLOGIES.items():
            with self.subTest(topology=name):
                rounds = self.converge_json(topology)["rounds"]
                for index in range(1, len(rounds)):
                    self.assertNotEqual(rounds[index]["routers"], rounds[index - 1]["routers"])

    def test_last_recorded_snapshot_is_stable(self):
        # Together with "every recorded adjacent pair differs" and synchronous
        # derivation from round 0, a fixed point at the last recorded snapshot
        # proves convergenceRound names the first stable round.
        for name, topology in TOPOLOGIES.items():
            with self.subTest(topology=name):
                output = self.converge_json(topology)
                nodes = sorted(topology["nodes"])
                adjacency = adjacency_of(topology)
                final = output["rounds"][-1]["routers"]
                self.assertEqual(synchronous_step(final, nodes, adjacency), final)


# --------------------------------------------------------------------------
# Determinism: equivalent inputs and hash randomization.
# --------------------------------------------------------------------------

class DeterminismTests(CliTestCase):
    def test_equivalent_declarations_produce_byte_identical_output(self):
        compute_outputs = []
        converge_outputs = []
        for index, variant in enumerate(EQUIV_VARIANTS):
            path = self.topology_path(variant, f"topology{index}.json")
            compute = self.run_cli("compute", path)
            converge = self.run_cli("converge", path)
            self.assertEqual(compute.returncode, 0, compute.stderr)
            self.assertEqual(converge.returncode, 0, converge.stderr)
            compute_outputs.append(compute.stdout)
            converge_outputs.append(converge.stdout)
        for output in compute_outputs:
            self.assertEqual(output, compute_outputs[0])
        for output in converge_outputs:
            self.assertEqual(output, converge_outputs[0])

    def test_output_is_stable_across_hash_randomization(self):
        for name, topology in TOPOLOGIES.items():
            with self.subTest(topology=name):
                path = self.topology_path(topology)
                for command in ("compute", "converge"):
                    outputs = [
                        self.run_cli(command, path, hashseed=seed).stdout
                        for seed in ("0", "1", "42")
                    ]
                    self.assertEqual(outputs[0], outputs[1])
                    self.assertEqual(outputs[1], outputs[2])


# --------------------------------------------------------------------------
# Existing CLI conventions: version, help, and error behavior.
# --------------------------------------------------------------------------

class CliBehaviorTests(CliTestCase):
    def test_version(self):
        result = self.run_cli("version")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, (__version__ + "\n").encode())
        self.assertEqual(result.stderr, b"")

    def test_help_variants(self):
        for args in (("help",), ("-h",), ("--help",), ()):
            with self.subTest(args=args):
                result = self.run_cli(*args)
                self.assertEqual(result.returncode, 0)
                self.assertEqual(result.stdout, USAGE.encode())
                self.assertEqual(result.stderr, b"")

    def test_unknown_command(self):
        result = self.run_cli("frobnicate")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, b"")
        self.assertEqual(result.stderr,
                         b"unknown command: frobnicate\n" + USAGE.encode())

    def test_command_argument_count_errors(self):
        cases = [
            (("compute",), COMPUTE_USAGE),
            (("compute", "a.json", "b.json"), COMPUTE_USAGE),
            (("converge",), CONVERGE_USAGE),
            (("converge", "a.json", "b.json"), CONVERGE_USAGE),
        ]
        for args, usage in cases:
            with self.subTest(args=args):
                result = self.run_cli(*args)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, b"")
                self.assertEqual(result.stderr, usage.encode())

    def test_unreadable_topology_file(self):
        missing = os.path.join(self.tmpdir.name, "missing.json")
        for command in ("compute", "converge"):
            with self.subTest(command=command):
                result = self.run_cli(command, missing)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, b"")
                self.assertEqual(result.stderr,
                                 f"cannot read topology: {missing}\n".encode())

    def test_invalid_json(self):
        path = self.write_file("broken.json", b"{not json")
        for command in ("compute", "converge"):
            with self.subTest(command=command):
                result = self.run_cli(command, path)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, b"")
                self.assertEqual(result.stderr, b"invalid topology\n")

    def test_invalid_topologies(self):
        invalid = {
            "not_an_object": [1, 2],
            "missing_keys": {},
            "nodes_not_a_list": {"nodes": {}, "links": []},
            "links_not_a_list": {"nodes": ["a"], "links": {}},
            "empty_node_name": {"nodes": [""], "links": []},
            "non_string_node": {"nodes": [1], "links": []},
            "duplicate_node": {"nodes": ["a", "a"], "links": []},
            "link_not_an_object": {"nodes": ["a", "b"], "links": [["a", "b", 1]]},
            "link_missing_metric": {"nodes": ["a", "b"], "links": [{"from": "a", "to": "b"}]},
            "link_unknown_node": {"nodes": ["a"], "links": [link("a", "b", 1)]},
            "self_link": {"nodes": ["a"], "links": [link("a", "a", 1)]},
            "zero_metric": {"nodes": ["a", "b"], "links": [link("a", "b", 0)]},
            "negative_metric": {"nodes": ["a", "b"], "links": [link("a", "b", -1)]},
            "float_metric": {"nodes": ["a", "b"], "links": [link("a", "b", 1.5)]},
            "bool_metric": {"nodes": ["a", "b"], "links": [link("a", "b", True)]},
            "string_metric": {"nodes": ["a", "b"], "links": [link("a", "b", "3")]},
            "duplicate_link": {"nodes": ["a", "b"],
                               "links": [link("a", "b", 1), link("a", "b", 2)]},
            "duplicate_link_reversed": {"nodes": ["a", "b"],
                                        "links": [link("a", "b", 1), link("b", "a", 2)]},
        }
        for name, topology in invalid.items():
            path = self.topology_path(topology, f"{name}.json")
            for command in ("compute", "converge"):
                with self.subTest(case=name, command=command):
                    result = self.run_cli(command, path)
                    self.assertEqual(result.returncode, 2)
                    self.assertEqual(result.stdout, b"")
                    self.assertEqual(result.stderr, b"invalid topology\n")


if __name__ == "__main__":
    unittest.main()
