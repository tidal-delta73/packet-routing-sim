"""Acceptance tests for the I/O-free simulation core.

These tests call :mod:`packet_routing_sim.core` directly with decoded JSON
values (plain dicts/lists), never files or streams.  They pin down:

* no side effects: inputs stay equal, repeated calls share no fault state,
  and mutating one result cannot corrupt a later computation;
* distinguishable deterministic validation results (invalid topology vs
  invalid scenario vs illegal state transition);
* the published routing semantics: sorted names, smaller-hop tie breaks,
  null reachability markers, round-0/converged distance-vector shape, and a
  replay baseline identical to compute;
* failure semantics: isolated nodes, the empty topology, equal-cost paths,
  link-down surviving a node restart, reversed link endpoints, and repeated
  failure/recovery cycles;
* byte-identical results across hash seeds and declaration orders.
"""
import copy
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
    NetworkState,
    apply_event,
    compute_topology,
    converge_topology,
    replay_dv_scenario,
    replay_scenario,
    validate_dv_scenario,
    validate_scenario,
    validate_topology,
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
# Basic semantics on the validated core inputs
# ---------------------------------------------------------------------------


class TestCoreSemantics(unittest.TestCase):
    def test_empty_topology(self):
        self.assertEqual(compute_topology(EMPTY), {"routers": {}})
        converge = converge_topology(EMPTY)
        self.assertEqual(
            converge,
            {
                "protocol": "distance-vector",
                "convergenceRound": 0,
                "rounds": [{"round": 0, "routers": {}}],
            },
        )
        replay = replay_scenario(EMPTY, {"events": []})
        self.assertEqual(
            replay,
            {"protocol": "link-state", "timeline": [{"event": None, "routers": {}}]},
        )

    def test_single_node(self):
        routers = compute_topology(SINGLE)["routers"]
        self.assertEqual(routers["solo"]["solo"], {"nextHop": None, "metric": 0})
        self.assertEqual(converge_topology(SINGLE)["convergenceRound"], 0)

    def test_isolated_node_is_fully_unreachable(self):
        routers = compute_topology(COMPONENTS)["routers"]
        self.assertEqual(
            routers["iso"]["iso"], {"nextHop": None, "metric": 0}
        )
        for other in ("a1", "a2", "b1", "b2"):
            self.assertEqual(
                routers["iso"][other], {"nextHop": None, "metric": None}
            )
            self.assertEqual(
                routers[other]["iso"], {"nextHop": None, "metric": None}
            )
        # The connected component still routes normally.
        self.assertEqual(routers["a1"]["a2"], {"nextHop": "a2", "metric": 2})

    def test_equal_cost_picks_smaller_next_hop_name(self):
        routers = compute_topology(DIAMOND)["routers"]
        self.assertEqual(routers["A"]["D"]["nextHop"], "B")
        self.assertEqual(routers["D"]["A"]["nextHop"], "B")
        self.assertEqual(routers["B"]["C"]["nextHop"], "A")
        self.assertEqual(routers["C"]["B"]["nextHop"], "A")
        square = compute_topology(SQUARE)["routers"]
        self.assertEqual(square["A"]["C"]["nextHop"], "B")
        self.assertEqual(square["C"]["A"]["nextHop"], "B")

    def test_converge_round_zero_then_only_changed_rounds(self):
        result = converge_topology(CHAIN3)
        rounds = result["rounds"]
        self.assertEqual([r["round"] for r in rounds], [0, 1])
        self.assertEqual(result["convergenceRound"], 1)
        round_zero = rounds[0]["routers"]
        # Round 0: self and directly connected neighbors only.
        self.assertEqual(round_zero["A"]["A"], {"nextHop": None, "metric": 0})
        self.assertEqual(round_zero["A"]["B"], {"nextHop": "B", "metric": 5})
        self.assertEqual(round_zero["A"]["C"], {"nextHop": None, "metric": None})
        self.assertEqual(round_zero["C"]["A"], {"nextHop": None, "metric": None})
        # The converged snapshot equals the link-state answer.
        self.assertEqual(
            rounds[-1]["routers"], compute_topology(CHAIN3)["routers"]
        )
        # Running one more update would change nothing (fixed point).
        from packet_routing_sim.core.routing import distance_vector_round

        ordered = ("A", "B", "C")
        adjacency = validate_topology(CHAIN3).adjacency()
        self.assertEqual(
            distance_vector_round(rounds[-1]["routers"], ordered, adjacency),
            rounds[-1]["routers"],
        )

    def test_replay_baseline_equals_compute(self):
        replay = replay_scenario(DIAMOND, {"events": []})
        baseline = replay["timeline"][0]
        self.assertIsNone(baseline["event"])
        self.assertEqual(
            baseline["routers"], compute_topology(DIAMOND)["routers"]
        )

    def test_timeline_carries_events_verbatim(self):
        events = [
            {"time": 3, "action": "link-down", "from": "A", "to": "B"},
            {"time": 7, "action": "node-down", "node": "C"},
        ]
        timeline = replay_scenario(DIAMOND, {"events": events})["timeline"]
        self.assertEqual(len(timeline), 3)
        for index, event in enumerate(events, start=1):
            self.assertEqual(timeline[index]["time"], event["time"])
            self.assertEqual(timeline[index]["event"], event)

    def test_down_router_row_and_down_destination_are_null(self):
        replay = replay_scenario(
            CHAIN3, {"events": [{"time": 1, "action": "node-down", "node": "B"}]}
        )
        routers = replay["timeline"][1]["routers"]
        for dest in ("A", "B", "C"):
            self.assertEqual(routers["B"][dest], {"nextHop": None, "metric": None})
            self.assertEqual(routers[dest]["B"], {"nextHop": None, "metric": None})
        self.assertEqual(routers["A"]["C"], {"nextHop": None, "metric": None})
        self.assertEqual(routers["A"]["A"], {"nextHop": None, "metric": 0})


# ---------------------------------------------------------------------------
# Failure/recovery state rules
# ---------------------------------------------------------------------------


class TestFailureStateRules(unittest.TestCase):
    def test_link_down_survives_node_recovery(self):
        scenario = {
            "events": [
                {"time": 1, "action": "link-down", "from": "A", "to": "B"},
                {"time": 2, "action": "node-down", "node": "B"},
                {"time": 3, "action": "node-up", "node": "B"},
            ]
        }
        routers = replay_scenario(CHAIN3, scenario)["timeline"][3]["routers"]
        # B is back, but the explicitly disabled A-B link stays disabled.
        self.assertEqual(routers["A"]["B"], {"nextHop": None, "metric": None})
        self.assertEqual(routers["A"]["C"], {"nextHop": None, "metric": None})
        self.assertEqual(routers["B"]["C"], {"nextHop": "C", "metric": 7})

    def test_reversed_endpoints_name_the_same_link(self):
        # down A->B, up B->A must match and fully restore the baseline.
        scenario = {
            "events": [
                {"time": 1, "action": "link-down", "from": "A", "to": "B"},
                {"time": 2, "action": "link-up", "from": "B", "to": "A"},
            ]
        }
        timeline = replay_scenario(CHAIN3, scenario)["timeline"]
        self.assertEqual(
            timeline[2]["routers"], compute_topology(CHAIN3)["routers"]
        )

    def test_multiple_failures_and_recoveries(self):
        scenario = {
            "events": [
                {"time": 1, "action": "node-down", "node": "B"},
                {"time": 2, "action": "node-down", "node": "C"},
                {"time": 3, "action": "link-down", "from": "A", "to": "B"},
                {"time": 4, "action": "node-up", "node": "C"},
                {"time": 5, "action": "node-up", "node": "B"},
                {"time": 6, "action": "node-down", "node": "C"},
                {"time": 7, "action": "node-up", "node": "C"},
                {"time": 8, "action": "link-up", "from": "B", "to": "A"},
            ]
        }
        timeline = replay_scenario(CHAIN3, scenario)["timeline"]
        self.assertEqual(len(timeline), 9)
        # After every recovery and the matching link-up, the network is whole
        # again: final snapshot equals the fault-free forwarding tables.
        self.assertEqual(
            timeline[-1]["routers"], compute_topology(CHAIN3)["routers"]
        )
        # Mid-scenario: both B and C down leaves A completely isolated.
        routers = timeline[2]["routers"]
        self.assertEqual(routers["A"]["B"], {"nextHop": None, "metric": None})
        self.assertEqual(routers["A"]["C"], {"nextHop": None, "metric": None})

    def test_each_snapshot_is_derived_from_explicit_state(self):
        topo = validate_topology(CHAIN3)
        state = NetworkState.initial(topo)
        events = validate_scenario(
            topo,
            {"events": [{"time": 1, "action": "node-down", "node": "B"}]},
        )
        state = apply_event(state, events[0])
        # The state is explicit: B is down; incident links unusable.
        self.assertFalse(state.is_node_up("B"))
        self.assertTrue(state.is_node_up("A"))
        self.assertFalse(state.is_link_up("A", "B"))
        self.assertFalse(state.is_link_up("B", "C"))
        active = state.active_adjacency()
        self.assertEqual(active["A"], {})
        self.assertEqual(active["B"], {})
        self.assertEqual(active["C"], {})

    def test_apply_event_returns_new_state_without_mutating_old(self):
        topo = validate_topology(CHAIN3)
        before = NetworkState.initial(topo)
        events = validate_scenario(
            topo,
            {"events": [{"time": 1, "action": "link-down", "from": "A", "to": "B"}]},
        )
        after = apply_event(before, events[0])
        self.assertEqual(before.down_links, frozenset())
        self.assertEqual(after.down_links, frozenset({("A", "B")}))


# ---------------------------------------------------------------------------
# Distance-vector failure replay
# ---------------------------------------------------------------------------


DV_CHAIN = {
    "nodes": ["A", "B", "C"],
    "links": [link("A", "B", 5), link("B", "C", 7)],
}
DV_DIAMOND = {
    "nodes": ["A", "B", "C", "D"],
    "links": [
        link("A", "B", 1),
        link("A", "C", 1),
        link("B", "D", 1),
        link("C", "D", 1),
    ],
}
UNREACHABLE = {"nextHop": None, "metric": None}


def dv_entries(item):
    """Flatten one replay-dv timeline item to {round: {(r, d): entry}}."""
    return {
        snapshot["round"]: {
            (router, dest): entry
            for router, row in snapshot["routers"].items()
            for dest, entry in row.items()
        }
        for snapshot in item["rounds"]
    }


class TestDistanceVectorReplay(unittest.TestCase):
    def test_top_level_and_item_shape(self):
        result = replay_dv_scenario(
            DV_CHAIN,
            {
                "infinityMetric": 100,
                "events": [
                    {"time": 1, "action": "link-down", "from": "B", "to": "C"}
                ],
            },
        )
        self.assertEqual(set(result), {"protocol", "infinityMetric", "timeline"})
        self.assertEqual(result["protocol"], "distance-vector")
        self.assertEqual(result["infinityMetric"], 100)
        timeline = result["timeline"]
        self.assertEqual(len(timeline), 2)
        self.assertEqual(set(timeline[0]), {"event", "convergenceRound", "rounds"})
        self.assertIsNone(timeline[0]["event"])
        self.assertNotIn("time", timeline[0])
        event_item = timeline[1]
        self.assertEqual(
            set(event_item), {"time", "event", "convergenceRound", "rounds"}
        )
        for item in timeline:
            rounds = item["rounds"]
            self.assertEqual(
                [snapshot["round"] for snapshot in rounds],
                list(range(len(rounds))),
            )
            self.assertEqual(item["convergenceRound"], rounds[-1]["round"])
            for snapshot in rounds:
                self.assertEqual(set(snapshot), {"round", "routers"})

    def test_baseline_round_zero_starts_from_self_and_direct_neighbors(self):
        item = replay_dv_scenario(
            DV_CHAIN, {"infinityMetric": 100, "events": []}
        )["timeline"][0]
        round_zero = item["rounds"][0]["routers"]
        self.assertEqual(round_zero["A"]["A"], {"nextHop": None, "metric": 0})
        self.assertEqual(round_zero["A"]["B"], {"nextHop": "B", "metric": 5})
        self.assertEqual(round_zero["A"]["C"], UNREACHABLE)
        self.assertEqual(round_zero["C"]["A"], UNREACHABLE)

    def test_baseline_converges_to_shortest_paths_like_converge(self):
        replay = replay_dv_scenario(
            DV_CHAIN, {"infinityMetric": 100, "events": []}
        )
        converge = converge_topology(DV_CHAIN)
        baseline = replay["timeline"][0]
        self.assertEqual(baseline["convergenceRound"], 1)
        self.assertEqual(
            baseline["rounds"][-1]["routers"],
            converge["rounds"][-1]["routers"],
        )

    def test_empty_and_isolated_topologies_converge_at_round_zero(self):
        empty = replay_dv_scenario(
            EMPTY, {"infinityMetric": 1, "events": []}
        )
        self.assertEqual(
            empty,
            {
                "protocol": "distance-vector",
                "infinityMetric": 1,
                "timeline": [
                    {
                        "event": None,
                        "convergenceRound": 0,
                        "rounds": [{"round": 0, "routers": {}}],
                    }
                ],
            },
        )
        isolated = replay_dv_scenario(
            {"nodes": ["x", "y"], "links": []},
            {"infinityMetric": 3, "events": []},
        )["timeline"][0]
        self.assertEqual(isolated["convergenceRound"], 0)
        routers = isolated["rounds"][0]["routers"]
        self.assertEqual(routers["x"]["x"], {"nextHop": None, "metric": 0})
        self.assertEqual(routers["x"]["y"], UNREACHABLE)

    def test_counting_to_infinity_trace_on_link_failure(self):
        result = replay_dv_scenario(
            DV_CHAIN,
            {
                "infinityMetric": 100,
                "events": [
                    {"time": 1, "action": "link-down", "from": "B", "to": "C"}
                ],
            },
        )
        item = result["timeline"][1]
        rounds = dv_entries(item)
        # Round 0: B's direct route via the failed neighbor C is gone at
        # once; A still believes its route via B (metric 12).
        self.assertEqual(rounds[0][("A", "C")], {"nextHop": "B", "metric": 12})
        self.assertEqual(rounds[0][("B", "C")], UNREACHABLE)
        # Alternating advertisements inflate the metric by one link cost
        # per round: odd rounds B points at A, even rounds A at B.
        for index in range(1, 18):
            if index % 2 == 1:
                self.assertEqual(
                    rounds[index][("A", "C")],
                    UNREACHABLE,
                    f"round {index}",
                )
                self.assertEqual(
                    rounds[index][("B", "C")],
                    {"nextHop": "A", "metric": 12 + 5 * index},
                    f"round {index}",
                )
            else:
                self.assertEqual(
                    rounds[index][("A", "C")],
                    {"nextHop": "B", "metric": 12 + 5 * index},
                    f"round {index}",
                )
                self.assertEqual(
                    rounds[index][("B", "C")], UNREACHABLE, f"round {index}"
                )
        # Round 18: the last candidate (102) reaches infinity, both give up.
        self.assertEqual(rounds[18][("A", "C")], UNREACHABLE)
        self.assertEqual(rounds[18][("B", "C")], UNREACHABLE)
        self.assertEqual(item["convergenceRound"], 18)
        self.assertEqual(len(item["rounds"]), 19)

    def test_node_failure_matches_link_failure_counting_and_nulls_row(self):
        result = replay_dv_scenario(
            DV_CHAIN,
            {
                "infinityMetric": 16,
                "events": [
                    {"time": 1, "action": "node-down", "node": "C"}
                ],
            },
        )
        item = result["timeline"][1]
        rounds = dv_entries(item)
        # B's route via the now-failed direct neighbor C is null at round 0,
        # while A retains its prior advertisement via B.
        self.assertEqual(rounds[0][("B", "C")], UNREACHABLE)
        self.assertEqual(rounds[0][("A", "C")], {"nextHop": "B", "metric": 12})
        # The failed router's whole row -- including its self entry -- is
        # unreachable in every recorded round.
        for snapshot in item["rounds"]:
            for entry in snapshot["routers"]["C"].values():
                self.assertEqual(entry, UNREACHABLE)
        # infinityMetric 16: A adopts nothing at round 1 (5+12 = 17 >= 16),
        # B likewise has no usable advertisement, so round 1 is the fixpoint.
        self.assertEqual(item["convergenceRound"], 1)
        self.assertEqual(
            dv_entries(item)[1][("A", "C")], UNREACHABLE
        )

    def test_recovered_router_cold_starts_with_only_direct_neighbors(self):
        result = replay_dv_scenario(
            DV_CHAIN,
            {
                "infinityMetric": 100,
                "events": [
                    {"time": 1, "action": "node-down", "node": "C"},
                    {"time": 2, "action": "node-up", "node": "C"},
                ],
            },
        )
        item = result["timeline"][2]
        round_zero = item["rounds"][0]["routers"]
        # C knows only itself and its currently usable direct neighbor B.
        self.assertEqual(round_zero["C"]["C"], {"nextHop": None, "metric": 0})
        self.assertEqual(round_zero["C"]["B"], {"nextHop": "B", "metric": 7})
        self.assertEqual(round_zero["C"]["A"], UNREACHABLE)
        # B's direct route over the recovered adjacency is back at once;
        # A still has to hear the news through B.
        self.assertEqual(round_zero["B"]["C"], {"nextHop": "C", "metric": 7})
        self.assertEqual(round_zero["A"]["C"], UNREACHABLE)
        # One synchronous exchange later the network reconverges.
        final = item["rounds"][-1]["routers"]
        self.assertEqual(item["convergenceRound"], 1)
        self.assertEqual(final["A"]["C"], {"nextHop": "B", "metric": 12})
        self.assertEqual(final["C"]["A"], {"nextHop": "B", "metric": 12})

    def test_recovered_link_reinstalls_direct_routes_at_both_ends(self):
        result = replay_dv_scenario(
            DV_CHAIN,
            {
                "infinityMetric": 100,
                "events": [
                    {"time": 1, "action": "link-down", "from": "B", "to": "C"},
                    {"time": 2, "action": "link-up", "from": "C", "to": "B"},
                ],
            },
        )
        round_zero = result["timeline"][2]["rounds"][0]["routers"]
        self.assertEqual(round_zero["B"]["C"], {"nextHop": "C", "metric": 7})
        self.assertEqual(round_zero["C"]["B"], {"nextHop": "B", "metric": 7})

    def test_explicit_link_down_survives_node_recovery(self):
        result = replay_dv_scenario(
            DV_CHAIN,
            {
                "infinityMetric": 100,
                "events": [
                    {"time": 1, "action": "link-down", "from": "A", "to": "B"},
                    {"time": 2, "action": "node-down", "node": "B"},
                    {"time": 3, "action": "node-up", "node": "B"},
                ],
            },
        )
        item = result["timeline"][3]
        routers = item["rounds"][-1]["routers"]
        # B is back (cold) but the explicitly disabled A-B link stays down,
        # so A stays isolated and B only reaches C.
        self.assertEqual(routers["A"]["B"], UNREACHABLE)
        self.assertEqual(routers["A"]["C"], UNREACHABLE)
        self.assertEqual(routers["B"]["A"], UNREACHABLE)
        self.assertEqual(routers["B"]["C"], {"nextHop": "C", "metric": 7})

    def test_tied_finite_metrics_pick_smaller_next_hop_name(self):
        item = replay_dv_scenario(
            DV_DIAMOND, {"infinityMetric": 10, "events": []}
        )["timeline"][0]
        final = item["rounds"][-1]["routers"]
        self.assertEqual(final["A"]["D"], {"nextHop": "B", "metric": 2})
        self.assertEqual(final["D"]["A"], {"nextHop": "B", "metric": 2})

    def test_events_are_echoed_verbatim_and_chain_in_order(self):
        events = [
            {"time": 3, "action": "link-down", "from": "A", "to": "B"},
            {"time": 7, "action": "node-down", "node": "C"},
        ]
        timeline = replay_dv_scenario(
            DV_DIAMOND, {"infinityMetric": 50, "events": events}
        )["timeline"]
        self.assertEqual(len(timeline), 3)
        for index, event in enumerate(events, start=1):
            self.assertEqual(timeline[index]["time"], event["time"])
            self.assertEqual(timeline[index]["event"], event)

    def test_infinity_metric_validation(self):
        valid_topo = DV_CHAIN  # link metrics 5 and 7; maximum 7
        for bad in (
            {},  # missing
            None,
            True,
            False,
            0,
            -1,
            1.5,
            "8",
            [8],
            7,   # equal to the largest link metric: not strictly greater
            5,   # below a link metric
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(InvalidScenario):
                    replay_dv_scenario(valid_topo, {"infinityMetric": bad})
        # Strictly above every link metric is accepted; one above is enough.
        events, infinity = validate_dv_scenario(
            validate_topology(valid_topo),
            {"infinityMetric": 8, "events": []},
        )
        self.assertEqual(infinity, 8)
        self.assertEqual(events, ())
        # A linkless topology accepts infinityMetric 1.
        replay_dv_scenario(
            {"nodes": ["solo"], "links": []},
            {"infinityMetric": 1, "events": []},
        )

    def test_event_rules_and_transitions_are_still_enforced(self):
        with self.assertRaises(InvalidStateTransition):
            replay_dv_scenario(
                DV_CHAIN,
                {
                    "infinityMetric": 8,
                    "events": [
                        {"time": 1, "action": "node-up", "node": "A"}
                    ],
                },
            )
        with self.assertRaises(InvalidStateTransition):
            replay_dv_scenario(
                DV_CHAIN,
                {
                    "infinityMetric": 8,
                    "events": [
                        {"time": 1, "action": "link-down", "from": "A", "to": "B"},
                        {"time": 2, "action": "link-down", "from": "B", "to": "A"},
                    ],
                },
            )
        with self.assertRaises(InvalidScenario):
            replay_dv_scenario(
                DV_CHAIN,
                {
                    "infinityMetric": 8,
                    "events": [
                        {"time": 1, "action": "explode", "node": "A"}
                    ],
                },
            )

    def test_topology_errors_take_precedence_over_scenario_errors(self):
        with self.assertRaises(InvalidTopology):
            replay_dv_scenario(
                {"nodes": ["A", "A"], "links": []},
                {"infinityMetric": True},
            )

    def test_link_state_replay_ignores_infinity_field(self):
        # The new optional field must not change link-state replay at all.
        without = replay_scenario(DV_CHAIN, {"events": []})
        with_field = replay_scenario(
            DV_CHAIN, {"events": [], "infinityMetric": "ignored"}
        )
        self.assertEqual(with_field, without)

    def test_inputs_are_not_mutated_and_state_does_not_leak(self):
        topology = copy.deepcopy(DV_CHAIN)
        scenario = {
            "infinityMetric": 100,
            "events": [
                {"time": 1, "action": "node-down", "node": "C"}
            ],
        }
        topo_before = copy.deepcopy(topology)
        scen_before = copy.deepcopy(scenario)
        replay_dv_scenario(topology, scenario)
        self.assertEqual(topology, topo_before)
        self.assertEqual(scenario, scen_before)

        faulted = replay_dv_scenario(topology, scenario)
        again = replay_dv_scenario(
            topology, {"infinityMetric": 100, "events": []}
        )
        # A fresh call starts fault-free and reconverges to shortest paths.
        baseline_final = again["timeline"][0]["rounds"][-1]["routers"]
        self.assertEqual(
            baseline_final["A"]["C"], {"nextHop": "B", "metric": 12}
        )
        self.assertEqual(
            replay_dv_scenario(topology, copy.deepcopy(scenario)), faulted
        )

    def test_mutating_a_result_cannot_pollute_later_calls(self):
        scenario = {
            "infinityMetric": 100,
            "events": [
                {"time": 1, "action": "link-down", "from": "B", "to": "C"}
            ],
        }
        first = replay_dv_scenario(DV_CHAIN, scenario)
        first["timeline"][1]["rounds"][0]["routers"]["A"]["C"] = {
            "nextHop": "HACKED",
            "metric": -999,
        }
        first["timeline"][1]["event"]["from"] = "Z"
        second = replay_dv_scenario(DV_CHAIN, scenario)
        self.assertEqual(
            second["timeline"][1]["rounds"][0]["routers"]["A"]["C"],
            {"nextHop": "B", "metric": 12},
        )
        self.assertEqual(second["timeline"][1]["event"]["from"], "B")
        # The echoed event is a copy, not the caller's dict.
        self.assertIsNot(
            second["timeline"][1]["event"], scenario["events"][0]
        )

    def test_result_is_plain_json_serialisable(self):
        result = replay_dv_scenario(
            DV_CHAIN,
            {
                "infinityMetric": 100,
                "events": [
                    {"time": 1, "action": "link-down", "from": "B", "to": "C"}
                ],
            },
        )
        json.loads(json.dumps(result))


# ---------------------------------------------------------------------------
# Distinguishable validation results
# ---------------------------------------------------------------------------


class TestValidation(unittest.TestCase):
    def assert_invalid_topology(self, value):
        with self.assertRaises(InvalidTopology):
            compute_topology(value)
        with self.assertRaises(InvalidTopology):
            converge_topology(value)

    def test_invalid_topologies_raise_invalid_topology(self):
        invalid = [
            [], "just a string", 42, {},
            {"nodes": "A", "links": []},
            {"nodes": [], "links": {}},
            {"nodes": [1], "links": []},
            {"nodes": [""], "links": []},
            {"nodes": ["A", "A"], "links": []},
            {"nodes": ["A"]},
            {"links": []},
            {"nodes": ["A"], "links": ["not-a-link"]},
            {"nodes": ["A", "B"], "links": [{"from": "A", "to": "B"}]},
            {"nodes": ["A", "B"], "links": [{"from": "A", "to": "B", "metric": 0}]},
            {"nodes": ["A", "B"], "links": [{"from": "A", "to": "B", "metric": -1}]},
            {"nodes": ["A", "B"], "links": [{"from": "A", "to": "B", "metric": 1.5}]},
            {"nodes": ["A", "B"], "links": [{"from": "A", "to": "B", "metric": "1"}]},
            {"nodes": ["A", "B"], "links": [{"from": "A", "to": "B", "metric": True}]},
            {"nodes": ["A"], "links": [{"from": "A", "to": "A", "metric": 1}]},
            {"nodes": ["A", "B"],
             "links": [link("A", "B", 1), link("B", "A", 1)]},
        ]
        for index, value in enumerate(invalid):
            with self.subTest(case=index):
                self.assert_invalid_topology(value)

    def test_invalid_scenarios_raise_invalid_scenario(self):
        valid_topo = CHAIN3
        invalid = [
            [], "x", 42, {},
            {"events": {}}, {"events": "x"}, {"events": ["x"]},
            {"events": [None]},
            {"events": [{"action": "node-down", "node": "A"}]},
            {"events": [{"time": 1}]},
            {"events": [{"time": 1, "action": 7, "node": "A"}]},
            {"events": [{"time": 1, "action": "explode", "node": "A"}]},
            {"events": [{"time": 1, "action": "node-down"}]},
            {"events": [{"time": 1, "action": "node-down", "node": 3}]},
            {"events": [{"time": 1, "action": "node-down", "node": "ZZ"}]},
            {"events": [{"time": 1, "action": "link-down", "from": "A"}]},
            {"events": [{"time": 1, "action": "link-down", "from": "A", "to": 2}]},
            {"events": [{"time": 0, "action": "node-down", "node": "A"}]},
            {"events": [{"time": -2, "action": "node-down", "node": "A"}]},
            {"events": [{"time": 1.5, "action": "node-down", "node": "A"}]},
            {"events": [{"time": "3", "action": "node-down", "node": "A"}]},
            {"events": [{"time": True, "action": "node-down", "node": "A"}]},
            {
                "events": [
                    {"time": 1, "action": "node-down", "node": "A"},
                    {"time": 1, "action": "node-down", "node": "B"},
                ]
            },
            {
                "events": [
                    {"time": 5, "action": "node-down", "node": "A"},
                    {"time": 3, "action": "node-down", "node": "B"},
                ]
            },
            {"events": [{"time": 1, "action": "link-down", "from": "A", "to": "C"}]},
        ]
        for index, value in enumerate(invalid):
            with self.subTest(case=index):
                with self.assertRaises(InvalidScenario):
                    replay_scenario(valid_topo, value)

    def test_illegal_transitions_are_distinguishably_invalid(self):
        # InvalidStateTransition is separately catchable yet still an
        # InvalidScenario so the command layer reports one message.
        self.assertTrue(issubclass(InvalidStateTransition, InvalidScenario))
        self.assertFalse(issubclass(InvalidStateTransition, InvalidTopology))

        transition_cases = [
            {"events": [{"time": 1, "action": "node-up", "node": "A"}]},
            {
                "events": [
                    {"time": 1, "action": "node-down", "node": "A"},
                    {"time": 2, "action": "node-down", "node": "A"},
                ]
            },
            {"events": [{"time": 1, "action": "link-up", "from": "A", "to": "B"}]},
            {
                "events": [
                    {"time": 1, "action": "link-down", "from": "A", "to": "B"},
                    {"time": 2, "action": "link-down", "from": "B", "to": "A"},
                ]
            },
        ]
        for index, value in enumerate(transition_cases):
            with self.subTest(case=index):
                with self.assertRaises(InvalidStateTransition):
                    replay_scenario(CHAIN3, value)
                # And it is reported through the broader type as well.
                with self.assertRaises(InvalidScenario):
                    replay_scenario(CHAIN3, value)


# ---------------------------------------------------------------------------
# Purity: no input mutation, no state carry-over, output isolation
# ---------------------------------------------------------------------------


class TestCorePurity(unittest.TestCase):
    def test_inputs_remain_equal_after_every_entry_point(self):
        for topology in (EMPTY, SINGLE, COMPONENTS, DIAMOND):
            topo_before = copy.deepcopy(topology)
            compute_topology(topology)
            converge_topology(topology)
            self.assertEqual(topology, topo_before)

        scenario = {
            "events": [
                {"time": 1, "action": "link-down", "from": "A", "to": "B"},
                {"time": 2, "action": "node-down", "node": "B"},
                {"time": 3, "action": "node-up", "node": "B"},
                {"time": 4, "action": "link-up", "from": "B", "to": "A"},
            ]
        }
        topo_before = copy.deepcopy(CHAIN3)
        scen_before = copy.deepcopy(scenario)
        replay_scenario(CHAIN3, scenario)
        self.assertEqual(CHAIN3, topo_before)
        self.assertEqual(scenario, scen_before)

    def test_repeated_calls_share_no_fault_state(self):
        fault = {
            "events": [{"time": 1, "action": "node-down", "node": "B"}]
        }
        faulted = replay_scenario(CHAIN3, fault)
        self.assertIsNotNone(faulted["timeline"][-1]["routers"]["A"]["B"])
        self.assertIsNone(
            faulted["timeline"][-1]["routers"]["A"]["B"]["nextHop"]
        )
        # A fresh replay starts fault-free even though the prior call failed B.
        again = replay_scenario(CHAIN3, {"events": []})
        self.assertEqual(
            again["timeline"][0]["routers"],
            compute_topology(CHAIN3)["routers"],
        )
        # Identical inputs give structurally identical answers every time.
        self.assertEqual(
            replay_scenario(CHAIN3, fault),
            replay_scenario(CHAIN3, copy.deepcopy(fault)),
        )

    def test_mutating_a_result_cannot_pollute_later_calls(self):
        first = compute_topology(DIAMOND)
        first["routers"]["A"]["D"] = {"nextHop": "HACKED", "metric": -999}
        second = compute_topology(DIAMOND)
        self.assertEqual(
            second["routers"]["A"]["D"], {"nextHop": "B", "metric": 2}
        )

        replay1 = replay_scenario(
            CHAIN3,
            {"events": [{"time": 1, "action": "node-down", "node": "B"}]},
        )
        replay1["timeline"][1]["routers"]["A"]["B"] = {"nextHop": "X", "metric": 1}
        replay1["timeline"][1]["event"]["node"] = "C"
        replay2 = replay_scenario(
            CHAIN3,
            {"events": [{"time": 1, "action": "node-down", "node": "B"}]},
        )
        self.assertEqual(
            replay2["timeline"][1]["routers"]["A"]["B"],
            {"nextHop": None, "metric": None},
        )
        self.assertEqual(replay2["timeline"][1]["event"]["node"], "B")

    def test_returned_results_do_not_alias_each_other_or_input(self):
        scenario = {
            "events": [{"time": 1, "action": "node-down", "node": "B"}]
        }
        result = replay_scenario(CHAIN3, scenario)
        # The echoed event is a copy, not the caller's dict.
        self.assertIsNot(
            result["timeline"][1]["event"], scenario["events"][0]
        )
        scenario["events"][0]["node"] = "C"
        self.assertEqual(result["timeline"][1]["event"]["node"], "B")

    def test_outputs_are_plain_json_serialisable_data(self):
        for value in (
            compute_topology(DIAMOND),
            converge_topology(CHAIN3),
            replay_scenario(
                DIAMOND,
                {"events": [{"time": 1, "action": "node-down", "node": "B"}]},
            ),
        ):
            json.loads(json.dumps(value))

    def test_core_module_performs_no_stream_or_file_io_on_import(self):
        # The core package must not even import sys-level I/O machinery for
        # its logic; importing it must have no stdio side effects.
        source_dir = REPO_ROOT / "packet_routing_sim" / "core"
        for path in source_dir.glob("*.py"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("open(", text, path)
            self.assertNotIn("sys.stdout", text, path)
            self.assertNotIn("sys.stderr", text, path)
            self.assertNotIn("sys.argv", text, path)
            self.assertNotIn("os.environ", text, path)


# ---------------------------------------------------------------------------
# Determinism across hash seeds and declaration order
# ---------------------------------------------------------------------------

_DETERMINISM_SNIPPET = """
import json, os, sys
sys.path.insert(0, %r)
from packet_routing_sim import core
topology = json.loads(os.environ["PRSIM_TOPO"])
scenario = json.loads(os.environ["PRSIM_SCEN"])
out = {}
out["compute"] = core.compute_topology(topology)
out["converge"] = core.converge_topology(topology)
out["replay"] = core.replay_scenario(topology, scenario)
sys.stdout.write(json.dumps(out, sort_keys=True))
""" % str(REPO_ROOT)


def run_core_in_subprocess(topology, scenario, seed):
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
        {
            "nodes": ["A", "B", "C", "D"],
            "links": list(reversed(DIAMOND["links"])),
        },
    ]
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

    def test_byte_identical_across_seeds_and_declaration_order(self):
        outputs = set()
        for seed in HASH_SEEDS:
            for variant in self.VARIANTS:
                outputs.add(
                    run_core_in_subprocess(variant, self.SCENARIO, seed)
                )
        self.assertEqual(len(outputs), 1)

    def test_published_key_order_is_sorted_in_process(self):
        routers = compute_topology(DIAMOND)["routers"]
        self.assertEqual(list(routers), ["A", "B", "C", "D"])
        for row in routers.values():
            self.assertEqual(list(row), ["A", "B", "C", "D"])


if __name__ == "__main__":
    unittest.main()
