"""Acceptance tests for the pure simulation core (``packet_routing_sim.core``).

These tests call the core directly with decoded Python values and never touch
the file system, standard streams or the command line. Every expected routing
value comes from an independently written Floyd-Warshall / Bellman-Ford
reference in this module, so a defect in the implementation cannot corrupt both
the answer and the assertion about it.

Coverage:
  * static routing semantics (sorted routers/destinations, smaller-name tie
    break, null unreachable/down entries) under declaration-order shuffles;
  * distance-vector convergence trajectory (round 0 self + direct links,
    only changed rounds recorded, fixed point, diameter timing);
  * replay timeline against a step-by-step oracle, including link-down state
    surviving node recovery, reversed link endpoints, multiple fail/recover
    cycles, isolated nodes, empty topology and equal-cost ties;
  * distinguishable, deterministic validation results (InvalidTopology vs
    InvalidScenario vs IllegalTransition);
  * side-effect freedom: inputs stay equal, repeat calls independent, and
    mutating one returned value cannot pollute a later call;
  * the core module's purity boundary (no file/stream/argv/env access).
"""
import copy
import inspect
import json
import unittest

from packet_routing_sim import core
from packet_routing_sim.core import (
    IllegalTransition,
    InvalidScenario,
    InvalidTopology,
    NetworkState,
    apply_event,
    compute,
    converge,
    initial_distance_vectors,
    distance_vector_round,
    forwarding_table,
    routers_snapshot,
    simulate_replay,
    validate_scenario,
    validate_topology,
)


def link(source, target, metric):
    return {"from": source, "to": target, "metric": metric}


# ---------------------------------------------------------------------------
# Independent reference implementation (test-side oracle)
# ---------------------------------------------------------------------------

def build_adjacency(topology):
    nodes = sorted(topology["nodes"])
    adjacency = {node: {} for node in nodes}
    for edge in topology["links"]:
        a, b, w = edge["from"], edge["to"], edge["metric"]
        adjacency[a][b] = w
        adjacency[b][a] = w
    return nodes, adjacency


def floyd_warshall(nodes, adjacency):
    inf = float("inf")
    dist = {i: {j: inf for j in nodes} for i in nodes}
    for node in nodes:
        dist[node][node] = 0
    for a in nodes:
        for b, w in adjacency[a].items():
            dist[a][b] = w
    for k in nodes:
        dk = dist[k]
        for i in nodes:
            di = dist[i]
            if di[k] == inf:
                continue
            for j in nodes:
                candidate = di[k] + dk[j]
                if candidate < di[j]:
                    di[j] = candidate
    return dist


def oracle_tables(topology):
    """{(router, destination): (nextHop, metric)} via Floyd-Warshall + tie rule."""
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
            expected[source, dest] = (min(winners), dist[source][dest])
    return nodes, expected


def normalize(routers):
    return {
        (r, d): (entry["nextHop"], entry["metric"])
        for r, row in routers.items()
        for d, entry in row.items()
    }


def synchronous_update(previous, nodes, adjacency):
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


def replay_oracle_routers(topology, down_nodes, down_links):
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

EMPTY = {"nodes": [], "links": []}
SINGLE = {"nodes": ["solo"], "links": []}

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

RING = {
    "nodes": ["A", "B", "C", "D"],
    "links": [
        link("A", "B", 1),
        link("B", "C", 2),
        link("C", "D", 3),
        link("D", "A", 4),
    ],
}

DIAMOND = {
    "nodes": ["A", "B", "C", "D"],
    "links": [
        link("A", "B", 1),
        link("A", "C", 1),
        link("B", "D", 1),
        link("C", "D", 1),
    ],
}

TIECOST = {
    "nodes": ["T", "n2", "n1", "S"],
    "links": [
        link("T", "n2", 3),
        link("n1", "T", 3),
        link("n2", "S", 2),
        link("S", "n1", 2),
    ],
}

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

ALL_TOPOLOGIES = [
    ("empty", EMPTY),
    ("single", SINGLE),
    ("components", COMPONENTS),
    ("chain3", CHAIN3),
    ("chain4", CHAIN4),
    ("ring", RING),
    ("diamond", DIAMOND),
    ("tiecost", TIECOST),
    ("square", SQUARE),
    ("chain3_with_isolate", CHAIN3_WITH_ISOLATE),
]

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

REPLAY_SCENARIO = [
    {"time": 3, "action": "link-down", "from": "A", "to": "B"},
    {"time": 5, "action": "node-down", "node": "C"},
    {"time": 8, "action": "link-down", "from": "D", "to": "C"},
    {"time": 9, "action": "node-up", "node": "C"},
    {"time": 12, "action": "link-up", "from": "C", "to": "D"},
    {"time": 15, "action": "link-up", "from": "B", "to": "A"},
]


# Multiple fail/recover cycles on both a node and a link, deliberately mixing
# endpoint order to prove the undirected pair identity.
MULTI_CYCLE_SCENARIO = [
    {"time": 1, "action": "node-down", "node": "B"},
    {"time": 2, "action": "node-up", "node": "B"},
    {"time": 3, "action": "link-down", "from": "A", "to": "B"},
    {"time": 4, "action": "link-up", "from": "B", "to": "A"},
    {"time": 5, "action": "link-down", "from": "B", "to": "A"},
    {"time": 6, "action": "node-down", "node": "B"},
    {"time": 7, "action": "link-up", "from": "A", "to": "B"},
    {"time": 8, "action": "node-up", "node": "B"},
    {"time": 9, "action": "node-down", "node": "A"},
    {"time": 10, "action": "node-up", "node": "A"},
]


# ---------------------------------------------------------------------------
# compute / converge semantics
# ---------------------------------------------------------------------------

class TestComputeSemantics(unittest.TestCase):
    def test_matches_oracle_for_every_topology(self):
        for name, topology in ALL_TOPOLOGIES:
            with self.subTest(topology=name):
                _, expected = oracle_tables(topology)
                routers = compute(topology)["routers"]
                self.assertEqual(normalize(routers), expected, name)
                # Routers and destinations are in stable name order.
                nodes = sorted(topology["nodes"])
                self.assertEqual(list(routers), nodes)
                for row in routers.values():
                    self.assertEqual(list(row), nodes)

    def test_equal_cost_picks_smaller_neighbor_name(self):
        ties = {
            "ring": {("A", "C"): "B", ("B", "D"): "A",
                     ("C", "A"): "B", ("D", "B"): "A"},
            "diamond": {
                ("A", "D"): "B", ("D", "A"): "B",
                ("B", "C"): "A", ("C", "B"): "A",
            },
            "tiecost": {("S", "T"): "n1", ("T", "S"): "n1"},
            "square": {
                ("A", "C"): "B", ("C", "A"): "B",
                ("B", "D"): "A", ("D", "B"): "A",
            },
        }
        for name, topology in ALL_TOPOLOGIES:
            if name not in ties:
                continue
            routers = compute(topology)["routers"]
            for (source, dest), hop in ties[name].items():
                self.assertEqual(
                    routers[source][dest]["nextHop"], hop, f"{name} {source}->{dest}"
                )

    def test_isolated_and_empty_entries(self):
        empty = compute(EMPTY)
        self.assertEqual(empty, {"routers": {}})
        self.assertEqual(
            compute(SINGLE)["routers"],
            {"solo": {"solo": {"nextHop": None, "metric": 0}}},
        )
        routers = compute(CHAIN3_WITH_ISOLATE)["routers"]
        for router in ("A", "B", "C"):
            self.assertEqual(
                routers[router]["Z"], {"nextHop": None, "metric": None}
            )
        self.assertEqual(routers["Z"]["Z"], {"nextHop": None, "metric": 0})
        for dest in ("A", "B", "C"):
            self.assertEqual(
                routers["Z"][dest], {"nextHop": None, "metric": None}
            )

    def test_order_independence_is_deterministic(self):
        serialized = {
            json.dumps(compute(variant), sort_keys=False, ensure_ascii=False)
            for variant in EQUIVALENT_VARIANTS
        }
        self.assertEqual(len(serialized), 1)


class TestConvergeSemantics(unittest.TestCase):
    def validate_trace(self, name, topology, result):
        nodes, adjacency = build_adjacency(topology)
        rounds = result["rounds"]
        self.assertEqual(result["protocol"], "distance-vector")
        self.assertEqual(
            [snapshot["round"] for snapshot in rounds], list(range(len(rounds)))
        )
        self.assertEqual(result["convergenceRound"], rounds[-1]["round"])
        snapshots = [normalize(snapshot["routers"]) for snapshot in rounds]
        self.assertEqual(snapshots[0], round_zero(nodes, adjacency), name)
        for index in range(1, len(rounds)):
            self.assertEqual(
                snapshots[index],
                synchronous_update(snapshots[index - 1], nodes, adjacency),
                f"{name} round {index}",
            )
            self.assertNotEqual(
                snapshots[index], snapshots[index - 1], f"{name} round {index}"
            )
        self.assertEqual(
            synchronous_update(snapshots[-1], nodes, adjacency),
            snapshots[-1],
            name,
        )

    def test_trace_for_every_topology(self):
        for name, topology in ALL_TOPOLOGIES:
            with self.subTest(topology=name):
                result = converge(topology)
                self.validate_trace(name, topology, result)

    def test_final_round_equals_static_compute(self):
        for name, topology in ALL_TOPOLOGIES:
            with self.subTest(topology=name):
                self.assertEqual(
                    converge(topology)["rounds"][-1]["routers"],
                    compute(topology)["routers"],
                    name,
                )

    def test_disconnected_topologies_converge_at_round_zero(self):
        for name, topology in (
            ("empty", EMPTY),
            ("single", SINGLE),
            ("components", COMPONENTS),
        ):
            with self.subTest(topology=name):
                result = converge(topology)
                self.assertEqual(result["convergenceRound"], 0)
                self.assertEqual(len(result["rounds"]), 1)

    def test_chain_converges_in_diameter_rounds(self):
        self.assertEqual(converge(CHAIN3)["convergenceRound"], 1)
        self.assertEqual(converge(CHAIN4)["convergenceRound"], 2)

    def test_order_independence_is_deterministic(self):
        serialized = {
            json.dumps(converge(variant), sort_keys=False, ensure_ascii=False)
            for variant in EQUIVALENT_VARIANTS
        }
        self.assertEqual(len(serialized), 1)


# ---------------------------------------------------------------------------
# replay semantics
# ---------------------------------------------------------------------------

class TestReplaySemantics(unittest.TestCase):
    def _track(self, event, down_nodes, down_links):
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

    def _check_timeline(self, topology, events, name):
        result = simulate_replay(topology, {"events": events})
        self.assertEqual(result["protocol"], "link-state")
        timeline = result["timeline"]
        self.assertEqual(len(timeline), len(events) + 1)
        self.assertEqual(set(timeline[0]), {"event", "routers"})
        self.assertIsNone(timeline[0]["event"])
        for index, event in enumerate(events, start=1):
            entry = timeline[index]
            self.assertEqual(set(entry), {"time", "event", "routers"})
            self.assertEqual(entry["time"], event["time"])
            # Timeline carries the event back verbatim.
            self.assertEqual(entry["event"], event)

        down_nodes, down_links = set(), set()
        for index, entry in enumerate(timeline):
            if index > 0:
                self._track(events[index - 1], down_nodes, down_links)
            self.assertEqual(
                normalize(entry["routers"]),
                replay_oracle_routers(topology, down_nodes, down_links),
                f"{name} timeline[{index}]",
            )
        # Baseline snapshot is exactly the static compute result.
        self.assertEqual(timeline[0]["routers"], compute(topology)["routers"])

    def test_full_scenario_matches_oracle(self):
        self._check_timeline(DIAMOND, REPLAY_SCENARIO, "diamond")

    def test_every_fixture_with_link_and_node_cycle(self):
        for name, topology in ALL_TOPOLOGIES:
            if not topology["nodes"]:
                continue
            with self.subTest(topology=name):
                events = []
                time = 1
                if topology["links"]:
                    first = topology["links"][0]
                    events.append({"time": time, "action": "link-down",
                                   "from": first["from"], "to": first["to"]})
                    time += 1
                victim = topology["nodes"][0]
                events.append({"time": time, "action": "node-down",
                               "node": victim})
                time += 1
                events.append({"time": time, "action": "node-up",
                               "node": victim})
                time += 1
                if topology["links"]:
                    first = topology["links"][0]
                    events.append({"time": time, "action": "link-up",
                                   "from": first["to"], "to": first["from"]})
                self._check_timeline(topology, events, name)

    def test_multiple_fail_recover_cycles(self):
        self._check_timeline(CHAIN4, MULTI_CYCLE_SCENARIO, "multi-cycle")

    def test_link_down_survives_node_recovery(self):
        topology = CHAIN3
        events = [
            {"time": 1, "action": "link-down", "from": "A", "to": "B"},
            {"time": 2, "action": "node-down", "node": "B"},
            {"time": 3, "action": "node-up", "node": "B"},
        ]
        timeline = simulate_replay(topology, {"events": events})["timeline"]
        routers = timeline[3]["routers"]
        # B recovered, but A-B was explicitly link-down'd: stay split.
        self.assertEqual(routers["A"]["B"], {"nextHop": None, "metric": None})
        self.assertEqual(routers["A"]["C"], {"nextHop": None, "metric": None})
        # B-C is unaffected.
        self.assertEqual(routers["B"]["C"], {"nextHop": "C", "metric": 7})

    def test_reversed_endpoints_name_same_link(self):
        events_down = [
            {"time": 1, "action": "link-down", "from": "A", "to": "B"},
        ]
        events_reversed = [
            {"time": 1, "action": "link-down", "from": "B", "to": "A"},
        ]
        down = simulate_replay(CHAIN3, {"events": events_down})["timeline"]
        rev = simulate_replay(CHAIN3, {"events": events_reversed})["timeline"]
        self.assertEqual(
            [entry["routers"] for entry in down],
            [entry["routers"] for entry in rev],
        )
        # A link-up written with reversed endpoints clears exactly the record.
        events = [
            {"time": 1, "action": "link-down", "from": "A", "to": "B"},
            {"time": 2, "action": "link-up", "from": "B", "to": "A"},
        ]
        timeline = simulate_replay(CHAIN3, {"events": events})["timeline"]
        self.assertEqual(timeline[2]["routers"], compute(CHAIN3)["routers"])

    def test_down_router_row_and_destination_null(self):
        events = [{"time": 4, "action": "node-down", "node": "B"}]
        routers = simulate_replay(CHAIN3, {"events": events})["timeline"][1][
            "routers"
        ]
        for dest in ("A", "B", "C"):
            self.assertEqual(routers["B"][dest], {"nextHop": None, "metric": None})
            self.assertEqual(routers[dest]["B"], {"nextHop": None, "metric": None})
        self.assertEqual(routers["A"]["A"], {"nextHop": None, "metric": 0})
        self.assertEqual(routers["A"]["C"], {"nextHop": None, "metric": None})

    def test_empty_topology_and_empty_events(self):
        result = simulate_replay(EMPTY, {"events": []})
        self.assertEqual(
            result,
            {"protocol": "link-state", "timeline": [{"event": None, "routers": {}}]},
        )
        result = simulate_replay(SINGLE, {"events": []})
        self.assertEqual(len(result["timeline"]), 1)
        self.assertIsNone(result["timeline"][0]["event"])

    def test_isolated_node_through_node_failure(self):
        events = [
            {"time": 1, "action": "node-down", "node": "iso"},
            {"time": 2, "action": "node-up", "node": "iso"},
        ]
        timeline = simulate_replay(COMPONENTS, {"events": events})["timeline"]
        for entry in timeline:
            for other in ("a1", "a2", "b1", "b2"):
                self.assertEqual(
                    entry["routers"]["iso"][other],
                    {"nextHop": None, "metric": None},
                )
                self.assertEqual(
                    entry["routers"][other]["iso"],
                    {"nextHop": None, "metric": None},
                )
        # After recovery the isolated row is back to the self entry only.
        self.assertEqual(
            timeline[2]["routers"]["iso"]["iso"], {"nextHop": None, "metric": 0}
        )

    def test_order_independence_is_deterministic(self):
        scenario = {"events": REPLAY_SCENARIO}
        serialized = set()
        for variant in EQUIVALENT_VARIANTS:
            serialized.add(
                json.dumps(simulate_replay(variant, scenario), ensure_ascii=False)
            )
        self.assertEqual(len(serialized), 1)


# ---------------------------------------------------------------------------
# Validation: distinguishable, deterministic results
# ---------------------------------------------------------------------------

class TestValidation(unittest.TestCase):
    INVALID_TOPOLOGIES = [
        [], "just a string", 42, {},
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
        {"nodes": ["A", "B"],
         "links": [{"from": "B", "to": "unknown", "metric": 1}]},
        {"nodes": ["A", "B"],
         "links": [link("A", "B", 1), link("B", "A", 1)]},
    ]

    def test_invalid_topologies_raise_distinct_type(self):
        for index, topology in enumerate(self.INVALID_TOPOLOGIES):
            with self.subTest(case=index):
                with self.assertRaises(InvalidTopology):
                    compute(topology)
                with self.assertRaises(InvalidTopology):
                    converge(topology)
                with self.assertRaises(InvalidTopology):
                    simulate_replay(topology, {"events": []})

    def test_exception_hierarchy_is_distinguishable(self):
        # The three failure kinds must be catchable separately by the caller.
        self.assertTrue(issubclass(IllegalTransition, InvalidScenario))
        self.assertFalse(issubclass(IllegalTransition, InvalidTopology))
        self.assertFalse(issubclass(InvalidTopology, InvalidScenario))
        self.assertFalse(issubclass(InvalidScenario, InvalidTopology))

    def test_invalid_scenarios_are_distinct_from_topology(self):
        node_down = {"time": 1, "action": "node-down", "node": "A"}
        link_down = {"time": 1, "action": "link-down", "from": "A", "to": "B"}
        invalid = [
            [], "just a string", 42, {},
            {"events": {}}, {"events": "x"},
            {"events": ["x"]}, {"events": [None]},
            {"events": [{"action": "node-down", "node": "A"}]},
            {"events": [{"time": 1}]},
            {"events": [{"time": 1, "action": 7, "node": "A"}]},
            {"events": [{"time": 1, "action": "explode", "node": "A"}]},
            {"events": [{"time": 1, "action": "node-down"}]},
            {"events": [{"time": 1, "action": "node-down", "node": 3}]},
            {"events": [{"time": 1, "action": "link-down", "from": "A"}]},
            {"events": [{"time": 1, "action": "link-down",
                         "from": "A", "to": 2}]},
            {"events": [dict(node_down, time=0)]},
            {"events": [dict(node_down, time=-2)]},
            {"events": [dict(node_down, time=1.5)]},
            {"events": [dict(node_down, time="3")]},
            {"events": [dict(node_down, time=True)]},
            {"events": [dict(node_down, time=2), dict(link_down, time=2)]},
            {"events": [dict(node_down, time=5), dict(link_down, time=3)]},
            {"events": [dict(node_down, node="ZZ")]},
            {"events": [dict(link_down, to="ZZ")]},
            {"events": [dict(link_down, **{"from": "A", "to": "C"})]},
            {"events": [dict(link_down, **{"from": "A", "to": "A"})]},
        ]
        for index, scenario in enumerate(invalid):
            with self.subTest(case=index):
                with self.assertRaises(InvalidScenario):
                    simulate_replay(CHAIN3, scenario)
                with self.assertRaises(InvalidScenario):
                    validated = validate_topology(CHAIN3)
                    validate_scenario(scenario, validated)

    def test_illegal_transitions_are_their_own_subtype(self):
        cases = [
            {"events": [{"time": 1, "action": "node-up", "node": "A"}]},
            {"events": [
                {"time": 1, "action": "node-down", "node": "A"},
                {"time": 2, "action": "node-down", "node": "A"},
            ]},
            {"events": [
                {"time": 1, "action": "link-up", "from": "A", "to": "B"}
            ]},
            {"events": [
                {"time": 1, "action": "link-down", "from": "A", "to": "B"},
                {"time": 2, "action": "link-down", "from": "B", "to": "A"},
            ]},
        ]
        for index, scenario in enumerate(cases):
            with self.subTest(case=index):
                with self.assertRaises(IllegalTransition):
                    simulate_replay(CHAIN3, scenario)
                # Subtype relationship: still an InvalidScenario for the CLI.
                with self.assertRaises(InvalidScenario):
                    simulate_replay(CHAIN3, scenario)

    def test_validation_is_deterministic(self):
        # Repeated validation of the same bad input always raises the same type.
        for _ in range(5):
            with self.assertRaises(InvalidTopology):
                validate_topology({"nodes": ["A", "A"], "links": []})
            with self.assertRaises(IllegalTransition):
                validated = validate_topology(CHAIN3)
                validate_scenario(
                    {"events": [
                        {"time": 1, "action": "node-up", "node": "A"}]},
                    validated,
                )

    def test_apply_event_enforces_transitions_directly(self):
        state = NetworkState(validate_topology(CHAIN3))
        apply_event(state, {"time": 1, "action": "node-down", "node": "B"})
        with self.assertRaises(IllegalTransition):
            apply_event(state, {"time": 2, "action": "node-down", "node": "B"})
        apply_event(state, {"time": 3, "action": "node-up", "node": "B"})
        with self.assertRaises(IllegalTransition):
            apply_event(state, {"time": 4, "action": "node-up", "node": "B"})
        apply_event(
            state, {"time": 5, "action": "link-down", "from": "B", "to": "C"}
        )
        with self.assertRaises(InvalidScenario):
            apply_event(
                state, {"time": 6, "action": "link-down", "from": "A", "to": "X"}
            )
        with self.assertRaises(InvalidScenario):
            apply_event(state, {"time": 7, "action": "teleport", "node": "A"})


# ---------------------------------------------------------------------------
# Side-effect freedom
# ---------------------------------------------------------------------------

class TestPurityAndIsolation(unittest.TestCase):
    def test_inputs_remain_equal_after_calls(self):
        for topology in (DIAMOND, CHAIN3_WITH_ISOLATE, EMPTY):
            snapshot = copy.deepcopy(topology)
            compute(topology)
            converge(topology)
            simulate_replay(topology, {"events": REPLAY_SCENARIO}
                            if topology is DIAMOND else {"events": []})
            self.assertEqual(topology, snapshot)

        scenario = {"events": list(REPLAY_SCENARIO)}
        scenario_snapshot = copy.deepcopy(scenario)
        simulate_replay(DIAMOND, scenario)
        self.assertEqual(scenario, scenario_snapshot)

    def test_repeated_calls_are_independent(self):
        topology = CHAIN3
        scenario_break = {"events": [
            {"time": 1, "action": "node-down", "node": "B"},
        ]}
        first = simulate_replay(topology, scenario_break)
        second = simulate_replay(topology, {"events": []})
        third = simulate_replay(topology, scenario_break)
        # A fault scenario must not leak into a later fault-free call.
        self.assertEqual(
            second["timeline"][0]["routers"], compute(topology)["routers"]
        )
        self.assertEqual(len(second["timeline"]), 1)
        self.assertEqual(first, third)

    def test_mutating_return_value_cannot_pollute_later_calls(self):
        before = compute(DIAMOND)
        result = compute(DIAMOND)
        result["routers"]["A"]["A"] = {"nextHop": "EVIL", "metric": -1}
        result["routers"]["Z"] = {}
        self.assertEqual(compute(DIAMOND), before)

        replay = simulate_replay(DIAMOND, {"events": REPLAY_SCENARIO})
        replay["timeline"][1]["routers"]["A"]["B"] = {
            "nextHop": "EVIL", "metric": -9
        }
        replay["timeline"][1]["event"]["action"] = "tampered"
        again = simulate_replay(DIAMOND, {"events": list(REPLAY_SCENARIO)})
        self.assertNotIn("EVIL", json.dumps(again))
        self.assertEqual(again["timeline"][1]["event"]["action"], "link-down")

    def test_network_state_clones_are_independent(self):
        topology = validate_topology(CHAIN3)
        state = NetworkState(topology)
        clone = state.clone()
        apply_event(
            clone, {"time": 1, "action": "link-down", "from": "A", "to": "B"}
        )
        self.assertNotIn(("A", "B"), state.down_links)
        self.assertIn(("A", "B"), clone.down_links)
        # Snapshot generation does not mutate topology adjacency.
        adjacency_snapshot = copy.deepcopy(topology.adjacency)
        routers_snapshot(clone)
        routers_snapshot(state)
        self.assertEqual(topology.adjacency, adjacency_snapshot)

    def test_direct_protocol_helpers_are_pure(self):
        topology = validate_topology(CHAIN4)
        ordered = topology.ordered
        snapshot = copy.deepcopy(topology.adjacency)
        table = forwarding_table("A", ordered, topology.adjacency)
        table["A"] = {"nextHop": "EVIL", "metric": 0}
        self.assertEqual(topology.adjacency["A"].get("A"), None)
        vectors = initial_distance_vectors(ordered, topology.adjacency)
        advanced = distance_vector_round(vectors, ordered, topology.adjacency)
        advanced["A"]["D"] = {"nextHop": "EVIL", "metric": 1}
        self.assertEqual(topology.adjacency, snapshot)
        # Round helper output does not alias its input tables.
        self.assertIsNot(advanced, vectors)
        for router in ordered:
            for dest in ordered:
                self.assertIsNot(advanced[router][dest], vectors[router][dest])

    def test_core_has_no_io_boundary_imports_or_calls(self):
        source = inspect.getsource(core)
        for forbidden in ("import sys", "import os", "import io",
                          "open(", "input(", "print(", "sys.stdout",
                          "sys.stderr", "sys.argv", "os.environ",
                          "__file__"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
