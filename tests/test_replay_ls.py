"""Acceptance tests for the ``replay-ls`` I/O-free core entry point.

These tests call :func:`replay_ls_scenario` with decoded JSON values and pin
down the auditable link-state semantics:

* round-0 neighbor discovery: sequence-1 LSAs, neighbors name-sorted,
  installed only into the origin's own database;
* synchronous flooding over currently usable links, accepting only a
  strictly higher sequence per origin, until every database is stable;
* per-database SPF where only mutually declared, metric-agreeing links
  participate, with the smaller next-hop name winning equal-cost ties;
* failure handling: changed online endpoints originate the next sequence,
  down routers freeze their database and get a ``null`` router row, remote
  routers keep stale routes until a replacement LSA arrives, and recovered
  routers restart with an empty database on their historical sequence;
* validation reuse, purity and byte determinism.

The whole timeline is additionally checked against an independently
written flood/SPF oracle in this module.
"""
import copy
import heapq
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from packet_routing_sim.core import (
    InvalidScenario,
    InvalidStateTransition,
    InvalidTopology,
    compute_topology,
    replay_ls_scenario,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
HASH_SEEDS = tuple(
    int(seed)
    for seed in os.environ.get("PRSIM_TEST_SEEDS", "0,1,42,987654321").split(",")
)


def link(source, target, metric):
    return {"from": source, "to": target, "metric": metric}


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
DIAMOND = {
    "nodes": ["A", "B", "C", "D"],
    "links": [
        link("A", "B", 1),
        link("A", "C", 1),
        link("B", "D", 1),
        link("C", "D", 1),
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


# ---------------------------------------------------------------------------
# Independently written oracle: LSA flooding plus per-database SPF
# ---------------------------------------------------------------------------


def oracle_adjacency(topology, down_nodes, down_links):
    nodes = sorted(topology["nodes"])
    full = {node: {} for node in nodes}
    for edge in topology["links"]:
        full[edge["from"]][edge["to"]] = edge["metric"]
        full[edge["to"]][edge["from"]] = edge["metric"]
    disabled = {frozenset(pair) for pair in down_links}
    active = {node: {} for node in nodes}
    for source in nodes:
        if source in down_nodes:
            continue
        for target, metric in full[source].items():
            if target in down_nodes:
                continue
            if frozenset((source, target)) in disabled:
                continue
            active[source][target] = metric
    return nodes, full, active


def oracle_originate(router, sequence, active):
    return {
        "seq": sequence,
        "neighbors": [
            {"router": neighbor, "metric": active[router][neighbor]}
            for neighbor in sorted(active[router])
        ],
    }


def oracle_flood(dbs, nodes, active, down_nodes):
    """One synchronous flood: read only each neighbor's start-of-round DB."""
    current = {router: dict(dbs[router]) for router in nodes}
    for receiver in nodes:
        if receiver in down_nodes:
            continue
        for neighbor in sorted(active[receiver]):
            for origin in sorted(dbs[neighbor]):
                advertised = dbs[neighbor][origin]
                known = current[receiver].get(origin)
                if known is None or advertised["seq"] > known["seq"]:
                    current[receiver][origin] = copy.deepcopy(advertised)
    return current


def oracle_graph(database):
    declared = {
        origin: {item["router"]: item["metric"] for item in lsa["neighbors"]}
        for origin, lsa in database.items()
    }
    graph = {origin: {} for origin in sorted(declared)}
    for source in sorted(declared):
        for target, metric in sorted(declared[source].items()):
            if target not in declared:
                continue
            if declared[target].get(source) != metric:
                continue
            graph[source][target] = metric
            graph[target][source] = metric
    return graph


def oracle_dijkstra(source, nodes, graph):
    """Smallest first-hop name wins equal-cost ties (labels carry the hop)."""
    distances = {source: 0}
    first_hops = {}
    settled = set()
    heap = [(0, "", source)]
    while heap:
        distance, first_hop, node = heapq.heappop(heap)
        if node in settled:
            continue
        settled.add(node)
        distances[node] = distance
        first_hops[node] = first_hop
        for neighbor in sorted(graph[node]):
            if neighbor in settled:
                continue
            hop = neighbor if node == source else first_hop
            heapq.heappush(
                heap, (distance + graph[node][neighbor], hop, neighbor)
            )
    result = {}
    for destination in nodes:
        if destination == source:
            result[destination] = {"nextHop": None, "metric": 0}
        elif destination in distances:
            result[destination] = {
                "nextHop": first_hops[destination],
                "metric": distances[destination],
            }
        else:
            result[destination] = {"nextHop": None, "metric": None}
    return result


def oracle_view(dbs, nodes, active, down_nodes):
    routers = {}
    for router in nodes:
        if router in down_nodes:
            routers[router] = None
        else:
            routers[router] = oracle_dijkstra(
                router, nodes, oracle_graph(dbs[router])
            )
    return {"databases": copy.deepcopy(dbs), "routers": routers}


def oracle_converge(dbs, nodes, active, down_nodes):
    """Flood to a fixed point, publishing round 0 and each changed round."""
    rounds = [{"round": 0, **oracle_view(dbs, nodes, active, down_nodes)}]
    while True:
        updated = oracle_flood(dbs, nodes, active, down_nodes)
        if updated == dbs:
            return rounds, dbs
        dbs = updated
        rounds.append(
            {"round": len(rounds),
             **oracle_view(dbs, nodes, active, down_nodes)}
        )


def oracle_timeline(topology, scenario):
    """The complete expected replay-ls timeline, built from scratch."""
    down_nodes, down_links = set(), set()
    nodes, _full, active = oracle_adjacency(topology, down_nodes, down_links)

    numbers = {router: 1 for router in nodes}
    dbs = {
        router: {router: oracle_originate(router, 1, active)}
        for router in nodes
    }
    rounds, dbs = oracle_converge(dbs, nodes, active, down_nodes)
    timeline = [
        {"event": None,
         "convergenceRound": rounds[-1]["round"],
         "rounds": rounds}
    ]
    previous_active = active

    for event in scenario["events"]:
        action = event["action"]
        recovered = ()
        if action == "node-down":
            down_nodes.add(event["node"])
        elif action == "node-up":
            down_nodes.discard(event["node"])
            recovered = (event["node"],)
        elif action == "link-down":
            down_links.add(frozenset((event["from"], event["to"])))
        else:
            down_links.discard(frozenset((event["from"], event["to"])))

        _nodes, _full, active = oracle_adjacency(
            topology, down_nodes, down_links
        )
        dbs = {router: dict(dbs[router]) for router in nodes}
        for router in recovered:
            dbs[router] = {}
        for router in nodes:
            changed = set(previous_active[router]) != set(active[router])
            if router in down_nodes:
                continue
            if router in recovered or changed:
                numbers[router] += 1
                dbs[router][router] = oracle_originate(
                    router, numbers[router], active
                )

        rounds, dbs = oracle_converge(dbs, nodes, active, down_nodes)
        timeline.append(
            {
                "time": event["time"],
                "event": event,
                "convergenceRound": rounds[-1]["round"],
                "rounds": rounds,
            }
        )
        previous_active = active
    return {"protocol": "link-state", "timeline": timeline}


# ---------------------------------------------------------------------------
# Envelope and baseline behavior
# ---------------------------------------------------------------------------


class TestReplayLSEnvelope(unittest.TestCase):
    def test_empty_topology(self):
        self.assertEqual(
            replay_ls_scenario(EMPTY, {"events": []}),
            {
                "protocol": "link-state",
                "timeline": [
                    {
                        "event": None,
                        "convergenceRound": 0,
                        "rounds": [
                            {"round": 0, "databases": {}, "routers": {}}
                        ],
                    }
                ],
            },
        )

    def test_single_node_round_zero_is_stable(self):
        result = replay_ls_scenario(SINGLE, {"events": []})
        baseline = result["timeline"][0]
        self.assertEqual(baseline["convergenceRound"], 0)
        self.assertEqual(len(baseline["rounds"]), 1)
        round_zero = baseline["rounds"][0]
        self.assertEqual(
            round_zero["databases"],
            {"solo": {"solo": {"seq": 1, "neighbors": []}}},
        )
        self.assertEqual(
            round_zero["routers"],
            {"solo": {"solo": {"nextHop": None, "metric": 0}}},
        )

    def test_isolated_components_converge_at_round_zero(self):
        result = replay_ls_scenario(COMPONENTS, {"events": []})
        baseline = result["timeline"][0]
        # The two connected pairs exchange LSAs in round 1; the fully
        # isolated node never learns anything past its own LSA.
        self.assertEqual(baseline["convergenceRound"], 1)
        databases = baseline["rounds"][0]["databases"]
        # Every router installs only its own sequence-1 LSA at round 0.
        for router in ("a1", "a2", "b1", "b2", "iso"):
            self.assertEqual(set(databases[router]), {router})
            self.assertEqual(databases[router][router]["seq"], 1)
        self.assertEqual(
            databases["iso"]["iso"]["neighbors"], []
        )
        self.assertEqual(
            databases["a1"]["a1"]["neighbors"],
            [{"router": "a2", "metric": 2}],
        )
        converged = baseline["rounds"][1]["databases"]
        self.assertEqual(set(converged["a1"]), {"a1", "a2"})
        self.assertEqual(set(converged["b2"]), {"b1", "b2"})
        self.assertEqual(set(converged["iso"]), {"iso"})
        # Neighbors inside a declared LSA are name-sorted.
        multi = replay_ls_scenario(
            {"nodes": ["z", "a", "m"],
             "links": [link("z", "m", 1), link("a", "z", 1)]},
            {"events": []},
        )
        z_lsa = multi["timeline"][0]["rounds"][0]["databases"]["z"]["z"]
        self.assertEqual(
            [item["router"] for item in z_lsa["neighbors"]], ["a", "m"]
        )


class TestBaselineFlooding(unittest.TestCase):
    def test_chain_round_progression(self):
        result = replay_ls_scenario(CHAIN3, {"events": []})
        rounds = result["timeline"][0]["rounds"]
        self.assertEqual([r["round"] for r in rounds], [0, 1, 2])
        self.assertEqual(result["timeline"][0]["convergenceRound"], 2)

        dbs = [r["databases"] for r in rounds]
        # Round 0: only self LSAs installed.
        self.assertEqual(set(dbs[0]["A"]), {"A"})
        self.assertEqual(set(dbs[0]["B"]), {"B"})
        self.assertEqual(set(dbs[0]["C"]), {"C"})
        # Round 1: direct neighbors' LSAs exchanged.
        self.assertEqual(set(dbs[1]["A"]), {"A", "B"})
        self.assertEqual(set(dbs[1]["B"]), {"A", "B", "C"})
        self.assertEqual(set(dbs[1]["C"]), {"B", "C"})
        # Round 2: every database holds every LSA.
        for router in ("A", "B", "C"):
            self.assertEqual(set(dbs[2][router]), {"A", "B", "C"})

    def test_round_zero_routers_know_only_self(self):
        result = replay_ls_scenario(CHAIN3, {"events": []})
        routers = result["timeline"][0]["rounds"][0]["routers"]
        for router in ("A", "B", "C"):
            for dest in ("A", "B", "C"):
                entry = routers[router][dest]
                if router == dest:
                    self.assertEqual(entry, {"nextHop": None, "metric": 0})
                else:
                    self.assertEqual(entry, {"nextHop": None, "metric": None})

    def test_converged_baseline_routers_equal_compute(self):
        for topology in (CHAIN3, DIAMOND, SQUARE, COMPONENTS):
            with self.subTest(topology=topology):
                result = replay_ls_scenario(topology, {"events": []})
                final = result["timeline"][0]["rounds"][-1]["routers"]
                self.assertEqual(final, compute_topology(topology)["routers"])

    def test_equal_cost_breaks_to_smaller_hop_at_each_router(self):
        result = replay_ls_scenario(DIAMOND, {"events": []})
        final = result["timeline"][0]["rounds"][-1]["routers"]
        self.assertEqual(final["A"]["D"]["nextHop"], "B")
        self.assertEqual(final["D"]["A"]["nextHop"], "B")
        self.assertEqual(final["B"]["C"]["nextHop"], "A")
        self.assertEqual(final["C"]["B"]["nextHop"], "A")

    def test_each_round_routers_derive_from_that_routers_own_database(self):
        # Round 1 on the chain: A knows A and B LSAs only, so C is absent
        # from A's graph and stays unreachable even though B already knows C.
        result = replay_ls_scenario(CHAIN3, {"events": []})
        round_one = result["timeline"][0]["rounds"][1]["routers"]
        self.assertEqual(round_one["A"]["B"], {"nextHop": "B", "metric": 5})
        self.assertEqual(round_one["A"]["C"], {"nextHop": None, "metric": None})
        self.assertEqual(round_one["B"]["C"], {"nextHop": "C", "metric": 7})


# ---------------------------------------------------------------------------
# Failure/recovery semantics
# ---------------------------------------------------------------------------


class TestFailureSemantics(unittest.TestCase):
    def test_node_down_freezes_database_and_nulls_router_row(self):
        result = replay_ls_scenario(
            CHAIN3,
            {"events": [{"time": 1, "action": "node-down", "node": "B"}]},
        )
        entry = result["timeline"][1]
        # With B isolated nothing more can flood: stable immediately.
        self.assertEqual(entry["convergenceRound"], 0)
        self.assertEqual(len(entry["rounds"]), 1)
        snapshot = entry["rounds"][0]
        # A and C each originate sequence 2; B keeps its frozen database.
        self.assertEqual(snapshot["databases"]["A"]["A"]["seq"], 2)
        self.assertEqual(snapshot["databases"]["A"]["A"]["neighbors"], [])
        self.assertEqual(snapshot["databases"]["C"]["C"]["seq"], 2)
        self.assertEqual(snapshot["databases"]["C"]["C"]["neighbors"], [])
        frozen_b = snapshot["databases"]["B"]
        self.assertEqual(set(frozen_b), {"A", "B", "C"})
        self.assertEqual(frozen_b["B"]["seq"], 1)
        # The down router's whole row is null in every (here: one) round.
        self.assertIsNone(snapshot["routers"]["B"])
        # Online endpoints lose every route: their own fresh LSAs declare no
        # neighbors, so no bidirectional edge survives in their graph.
        self.assertEqual(
            snapshot["routers"]["A"],
            {
                "A": {"nextHop": None, "metric": 0},
                "B": {"nextHop": None, "metric": None},
                "C": {"nextHop": None, "metric": None},
            },
        )
        self.assertEqual(snapshot["routers"]["C"]["A"],
                         {"nextHop": None, "metric": None})

    def test_remote_routers_keep_old_routes_until_new_lsa_arrives(self):
        # The A-B link fails on the diamond.  C and D are not endpoints, so
        # at round 0 their databases are unchanged and they keep routing
        # over the old topology, including via A-B.
        scenario = {
            "events": [
                {"time": 1, "action": "link-down", "from": "A", "to": "B"}
            ]
        }
        result = replay_ls_scenario(DIAMOND, scenario)
        rounds = result["timeline"][1]["rounds"]
        round_zero = rounds[0]
        self.assertEqual(round_zero["databases"]["A"]["A"]["seq"], 2)
        self.assertEqual(round_zero["databases"]["B"]["B"]["seq"], 2)
        # C/D still hold everyone's sequence-1 LSAs.
        for router in ("C", "D"):
            for origin in ("A", "B", "C", "D"):
                self.assertEqual(
                    round_zero["databases"][router][origin]["seq"], 1
                )
        self.assertEqual(round_zero["routers"]["C"]["A"]["nextHop"], "A")
        self.assertEqual(round_zero["routers"]["D"]["B"]["nextHop"], "B")
        # Endpoints themselves reroute at once from their own fresh LSAs:
        # A reaches D through C and B reaches C through D.
        self.assertEqual(round_zero["routers"]["A"]["D"],
                         {"nextHop": "C", "metric": 2})
        self.assertEqual(round_zero["routers"]["B"]["C"],
                         {"nextHop": "D", "metric": 2})
        # After flooding, the whole network routes on diamond-minus-A-B:
        # the direct link is gone from every database, but A and B stay
        # reachable through the two-hop C-D detour (metric 3).
        final = rounds[-1]
        self.assertEqual(
            final["routers"]["A"]["B"], {"nextHop": "C", "metric": 3}
        )
        self.assertEqual(final["routers"]["A"]["D"],
                         {"nextHop": "C", "metric": 2})
        self.assertEqual(final["routers"]["C"]["B"],
                         {"nextHop": "D", "metric": 2})
        final_databases = final["databases"]
        for router in ("A", "B", "C", "D"):
            a_neighbors = {
                item["router"]
                for item in final_databases[router]["A"]["neighbors"]
            }
            b_neighbors = {
                item["router"]
                for item in final_databases[router]["B"]["neighbors"]
            }
            self.assertNotIn("B", a_neighbors)
            self.assertNotIn("A", b_neighbors)

    def test_only_mutual_metric_agreeing_lsas_form_edges(self):
        from packet_routing_sim.core.routing import (
            forwarding_table,
            link_state_graph,
        )

        def lsa(origin, neighbors, sequence=1):
            return (
                origin,
                sequence,
                tuple((name, metric) for name, metric in neighbors),
            )

        # A declares A-B(1) and A-C(1); B does not declare A at all;
        # C declares A with a *different* metric (2).  Neither declaration
        # is mutual+consistent, so A must end up with no usable edges.
        database = {
            "A": lsa("A", [("B", 1), ("C", 1)]),
            "B": lsa("B", [("D", 1)]),
            "C": lsa("C", [("A", 2)]),
            "D": lsa("D", [("B", 1)]),
        }
        graph = link_state_graph(database)
        self.assertEqual(graph["A"], {})
        self.assertNotIn("A", graph["B"])
        self.assertNotIn("A", graph["C"])
        # The mutually declared, metric-agreeing B-D edge survives both
        # directions, while the bad edges never appear anywhere.
        self.assertEqual(graph["B"], {"D": 1})
        self.assertEqual(graph["D"], {"B": 1})
        table = forwarding_table("A", ("A", "B", "C", "D"), graph)
        self.assertEqual(table["B"], {"nextHop": None, "metric": None})
        self.assertEqual(table["C"], {"nextHop": None, "metric": None})
        self.assertEqual(table["D"], {"nextHop": None, "metric": None})

    def test_node_recovery_restarts_empty_database_on_next_sequence(self):
        scenario = {
            "events": [
                {"time": 1, "action": "node-down", "node": "B"},
                {"time": 2, "action": "node-up", "node": "B"},
            ]
        }
        result = replay_ls_scenario(CHAIN3, scenario)
        recovery = result["timeline"][2]
        round_zero = recovery["rounds"][0]
        # B restarted with an empty database and originates sequence 2
        # (continuing its historical sequence, not restarting at 1).
        self.assertEqual(set(round_zero["databases"]["B"]), {"B"})
        self.assertEqual(round_zero["databases"]["B"]["B"]["seq"], 2)
        # A and C react to B's return with sequence 3.
        self.assertEqual(round_zero["databases"]["A"]["A"]["seq"], 3)
        self.assertEqual(round_zero["databases"]["C"]["C"]["seq"], 3)
        # A and C still hold B's stale sequence-1 LSA at round 0.  Their
        # new LSAs declare B again and the stale B LSA declares them with
        # the same metric, so the bidirectional-consistency edge forms at
        # once: A already reaches B directly (B's own new LSA is not yet
        # needed for the edge, but is needed for B's view outward).
        self.assertEqual(
            round_zero["routers"]["A"]["B"], {"nextHop": "B", "metric": 5}
        )
        self.assertEqual(
            round_zero["routers"]["C"]["B"], {"nextHop": "B", "metric": 7}
        )
        # B itself sees only itself at round 0: its fresh database has no
        # neighbor LSAs yet, so no edge can be established from its view.
        self.assertEqual(
            round_zero["routers"]["B"],
            {
                "A": {"nextHop": None, "metric": None},
                "B": {"nextHop": None, "metric": 0},
                "C": {"nextHop": None, "metric": None},
            },
        )
        # Final convergence restores the fault-free forwarding tables.
        final = recovery["rounds"][-1]["routers"]
        self.assertEqual(final, compute_topology(CHAIN3)["routers"])
        for router in ("A", "B", "C"):
            self.assertEqual(
                recovery["rounds"][-1]["databases"][router]["B"]["seq"], 2
            )

    def test_explicit_link_down_survives_node_recovery(self):
        scenario = {
            "events": [
                {"time": 1, "action": "link-down", "from": "A", "to": "B"},
                {"time": 2, "action": "node-down", "node": "B"},
                {"time": 3, "action": "node-up", "node": "B"},
            ]
        }
        result = replay_ls_scenario(CHAIN3, scenario)
        final = result["timeline"][-1]["rounds"][-1]["routers"]
        self.assertEqual(final["A"]["B"], {"nextHop": None, "metric": None})
        self.assertEqual(final["A"]["C"], {"nextHop": None, "metric": None})
        self.assertEqual(final["B"]["C"], {"nextHop": "C", "metric": 7})

    def test_link_down_and_up_round_trips(self):
        scenario = {
            "events": [
                {"time": 1, "action": "link-down", "from": "B", "to": "A"},
                {"time": 2, "action": "link-up", "from": "A", "to": "B"},
            ]
        }
        result = replay_ls_scenario(CHAIN3, scenario)
        final = result["timeline"][-1]["rounds"][-1]
        self.assertEqual(final["routers"], compute_topology(CHAIN3)["routers"])
        # The restored link is announced in sequence-3 LSAs by both ends.
        for router in ("A", "B", "C"):
            self.assertEqual(final["databases"][router]["A"]["seq"], 3)
            self.assertEqual(final["databases"][router]["B"]["seq"], 3)

    def test_down_router_row_is_null_in_every_round(self):
        # On the diamond C's down LSA-free isolation still leaves an
        # A-B-D path, so flooding after node-down takes real rounds; the
        # down router's row must be null throughout them.
        scenario = {
            "events": [{"time": 1, "action": "node-down", "node": "C"}]
        }
        result = replay_ls_scenario(DIAMOND, scenario)
        for snapshot in result["timeline"][1]["rounds"]:
            self.assertIsNone(snapshot["routers"]["C"])
        final = result["timeline"][1]["rounds"][-1]["routers"]
        # C is unreachable; the A-B-D path survives.
        self.assertEqual(final["C"], None)
        self.assertEqual(final["A"]["D"], {"nextHop": "B", "metric": 2})

    def test_events_are_echoed_verbatim(self):
        events = [
            {"time": 3, "action": "link-down", "from": "A", "to": "B"},
            {"time": 9, "action": "node-down", "node": "C"},
        ]
        timeline = replay_ls_scenario(DIAMOND, {"events": events})["timeline"]
        self.assertEqual(len(timeline), 3)
        for index, event in enumerate(events, start=1):
            self.assertEqual(timeline[index]["time"], event["time"])
            self.assertEqual(timeline[index]["event"], event)


# ---------------------------------------------------------------------------
# Differential check against the independent oracle
# ---------------------------------------------------------------------------


class TestOracleDifferential(unittest.TestCase):
    SCENARIO = {
        "events": [
            {"time": 3, "action": "link-down", "from": "A", "to": "B"},
            {"time": 5, "action": "node-down", "node": "C"},
            {"time": 8, "action": "link-down", "from": "D", "to": "C"},
            {"time": 9, "action": "node-up", "node": "C"},
            {"time": 12, "action": "link-up", "from": "C", "to": "D"},
            {"time": 15, "action": "link-up", "from": "B", "to": "A"},
        ]
    }

    def test_full_diamond_timeline_matches_oracle(self):
        actual = replay_ls_scenario(DIAMOND, self.SCENARIO)
        expected = oracle_timeline(DIAMOND, self.SCENARIO)
        self.assertEqual(actual, expected)

    def test_every_fixture_matches_oracle(self):
        scenario = {"events": []}
        for topology in (EMPTY, SINGLE, COMPONENTS, CHAIN3, SQUARE):
            with self.subTest(topology=topology):
                self.assertEqual(
                    replay_ls_scenario(topology, scenario),
                    oracle_timeline(topology, scenario),
                )

    def test_node_failure_cycle_matches_oracle_on_every_fixture(self):
        # Only the first node is a meaningful victim on the chain/square;
        # components exercises an isolated-side node as well.
        for topology in (CHAIN3, DIAMOND, SQUARE, COMPONENTS):
            victim = sorted(topology["nodes"])[0]
            scenario = {
                "events": [
                    {"time": 1, "action": "node-down", "node": victim},
                    {"time": 2, "action": "node-up", "node": victim},
                ]
            }
            with self.subTest(topology=topology):
                self.assertEqual(
                    replay_ls_scenario(topology, scenario),
                    oracle_timeline(topology, scenario),
                )


# ---------------------------------------------------------------------------
# Validation reuse
# ---------------------------------------------------------------------------


class TestValidation(unittest.TestCase):
    def test_invalid_scenarios_raise_invalid_scenario(self):
        invalid = [
            [], "x", 42, {},
            {"events": {}}, {"events": "x"}, {"events": ["x"]},
            {"events": [None]},
            {"events": [{"action": "node-down", "node": "A"}]},
            {"events": [{"time": 1}]},
            {"events": [{"time": 1, "action": 7, "node": "A"}]},
            {"events": [{"time": 1, "action": "explode", "node": "A"}]},
            {"events": [{"time": 0, "action": "node-down", "node": "A"}]},
            {"events": [{"time": 1.5, "action": "node-down", "node": "A"}]},
            {"events": [{"time": 1, "action": "node-down", "node": "ZZ"}]},
            {"events": [{"time": 1, "action": "link-down",
                         "from": "A", "to": "C"}]},
            {
                "events": [
                    {"time": 1, "action": "node-down", "node": "A"},
                    {"time": 1, "action": "node-down", "node": "B"},
                ]
            },
        ]
        for index, scenario in enumerate(invalid):
            with self.subTest(case=index):
                with self.assertRaises(InvalidScenario):
                    replay_ls_scenario(CHAIN3, scenario)

    def test_illegal_transitions_raise_distinguishable_error(self):
        cases = [
            {"events": [{"time": 1, "action": "node-up", "node": "A"}]},
            {"events": [{"time": 1, "action": "link-up",
                         "from": "A", "to": "B"}]},
            {
                "events": [
                    {"time": 1, "action": "node-down", "node": "B"},
                    {"time": 2, "action": "node-down", "node": "B"},
                ]
            },
        ]
        for index, scenario in enumerate(cases):
            with self.subTest(case=index):
                with self.assertRaises(InvalidStateTransition):
                    replay_ls_scenario(CHAIN3, scenario)
                with self.assertRaises(InvalidScenario):
                    replay_ls_scenario(CHAIN3, scenario)

    def test_invalid_topology_takes_precedence(self):
        with self.assertRaises(InvalidTopology):
            replay_ls_scenario(
                {"nodes": ["A", "A"], "links": []}, {"events": []}
            )


# ---------------------------------------------------------------------------
# Purity
# ---------------------------------------------------------------------------


class TestPurity(unittest.TestCase):
    SCENARIO = {
        "events": [
            {"time": 1, "action": "link-down", "from": "A", "to": "B"},
            {"time": 2, "action": "node-down", "node": "B"},
            {"time": 3, "action": "node-up", "node": "B"},
            {"time": 4, "action": "link-up", "from": "B", "to": "A"},
        ]
    }

    def test_inputs_remain_equal(self):
        topology_before = copy.deepcopy(CHAIN3)
        scenario_before = copy.deepcopy(self.SCENARIO)
        replay_ls_scenario(CHAIN3, self.SCENARIO)
        self.assertEqual(CHAIN3, topology_before)
        self.assertEqual(self.SCENARIO, scenario_before)

    def test_no_carry_over_between_calls(self):
        faulted = replay_ls_scenario(
            CHAIN3,
            {"events": [{"time": 1, "action": "node-down", "node": "B"}]},
        )
        self.assertIsNone(faulted["timeline"][1]["rounds"][0]["routers"]["B"])
        fresh = replay_ls_scenario(CHAIN3, {"events": []})
        self.assertEqual(
            fresh["timeline"][0]["rounds"][-1]["routers"],
            compute_topology(CHAIN3)["routers"],
        )
        self.assertEqual(
            replay_ls_scenario(CHAIN3, copy.deepcopy(self.SCENARIO)),
            replay_ls_scenario(CHAIN3, copy.deepcopy(self.SCENARIO)),
        )

    def test_mutating_result_cannot_pollute_later_calls(self):
        scenario = {
            "events": [{"time": 1, "action": "node-down", "node": "C"}]
        }
        first = replay_ls_scenario(CHAIN3, scenario)
        first["timeline"][1]["rounds"][0]["databases"]["B"]["B"] = "HACK"
        first["timeline"][1]["event"]["node"] = "A"
        second = replay_ls_scenario(CHAIN3, copy.deepcopy(scenario))
        # Only B (the endpoint adjacent to the down node) re-originates;
        # A's adjacency is unchanged so it keeps sequence 1.
        self.assertEqual(
            second["timeline"][1]["rounds"][0]["databases"]["B"]["B"]["seq"],
            2,
        )
        self.assertEqual(
            second["timeline"][1]["rounds"][0]["databases"]["A"]["A"]["seq"],
            1,
        )
        self.assertEqual(second["timeline"][1]["event"]["node"], "C")

    def test_echoed_event_is_not_aliased_to_input(self):
        result = replay_ls_scenario(CHAIN3, self.SCENARIO)
        self.assertIsNot(
            result["timeline"][1]["event"], self.SCENARIO["events"][0]
        )

    def test_result_is_json_serialisable_and_key_order_sorted(self):
        result = replay_ls_scenario(DIAMOND, self.SCENARIO)
        json.loads(json.dumps(result))
        nodes = ["A", "B", "C", "D"]
        for entry in result["timeline"]:
            for snapshot in entry["rounds"]:
                self.assertEqual(list(snapshot["databases"]), nodes)
                for database in snapshot["databases"].values():
                    self.assertEqual(list(database), sorted(database))
                    for lsa in database.values():
                        self.assertEqual(list(lsa), ["seq", "neighbors"])
                routers = snapshot["routers"]
                self.assertEqual(list(routers), nodes)
                for row in routers.values():
                    if row is not None:
                        self.assertEqual(list(row), nodes)


# ---------------------------------------------------------------------------
# Determinism across hash seeds and declaration order
# ---------------------------------------------------------------------------

_DETERMINISM_SNIPPET = """
import json, os, sys
sys.path.insert(0, %r)
from packet_routing_sim.core import replay_ls_scenario
sys.stdout.write(json.dumps(
    replay_ls_scenario(json.loads(os.environ["PRSIM_TOPO"]),
                       json.loads(os.environ["PRSIM_SCEN"])),
    sort_keys=True))
""" % str(REPO_ROOT)


def run_in_subprocess(topology, scenario, seed):
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(seed)
    env["PRSIM_TOPO"] = json.dumps(topology)
    env["PRSIM_SCEN"] = json.dumps(scenario)
    proc = subprocess.run(
        [sys.executable, "-c", _DETERMINISM_SNIPPET],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


class TestDeterminism(unittest.TestCase):
    SCENARIO = {
        "events": [
            {"time": 3, "action": "link-down", "from": "A", "to": "B"},
            {"time": 5, "action": "node-down", "node": "C"},
            {"time": 8, "action": "link-down", "from": "D", "to": "C"},
            {"time": 9, "action": "node-up", "node": "C"},
            {"time": 12, "action": "link-up", "from": "C", "to": "D"},
            {"time": 15, "action": "link-up", "from": "B", "to": "A"},
        ]
    }
    VARIANTS = [
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
        {"nodes": ["A", "B", "C", "D"],
         "links": list(reversed(DIAMOND["links"]))},
    ]

    def test_byte_identical_across_seeds_and_declaration_order(self):
        outputs = set()
        for seed in HASH_SEEDS:
            for variant in self.VARIANTS:
                outputs.add(run_in_subprocess(variant, self.SCENARIO, seed))
        self.assertEqual(len(outputs), 1)


if __name__ == "__main__":
    unittest.main()
