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
    replay_ls_scenario,
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
# Unit-metric chain so small infinity thresholds still leave the baseline
# converged and expose the full count-to-infinity progression.
DV_CHAIN = {
    "nodes": ["A", "B", "C"],
    "links": [link("A", "B", 1), link("B", "C", 1)],
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
# Link-state flooding replay (replay-ls core)
# ---------------------------------------------------------------------------


CHAIN4_LS = {
    "nodes": ["A", "B", "C", "D"],
    "links": [link("A", "B", 1), link("B", "C", 1), link("C", "D", 1)],
}


def _db_view(entry_round):
    """{router: {originator: (sequence, neighbors)}} for one round."""
    return {
        router: None
        if database is None
        else {
            originator: (lsa["sequence"], tuple(lsa["neighbors"]))
            for originator, lsa in database.items()
        }
        for router, database in entry_round["databases"].items()
    }


class TestReplayLinkState(unittest.TestCase):
    def test_envelope_and_baseline_shape(self):
        result = replay_ls_scenario(
            CHAIN3, {"events": [{"time": 1, "action": "node-down", "node": "B"}]}
        )
        self.assertEqual(set(result), {"protocol", "timeline"})
        self.assertEqual(result["protocol"], "link-state")
        timeline = result["timeline"]
        self.assertEqual(len(timeline), 2)

        baseline = timeline[0]
        self.assertEqual(set(baseline), {"event", "convergenceRound", "rounds"})
        self.assertIsNone(baseline["event"])
        for snapshot in baseline["rounds"]:
            self.assertEqual(set(snapshot), {"round", "databases", "routers"})

        event_entry = timeline[1]
        self.assertEqual(
            set(event_entry), {"time", "event", "convergenceRound", "rounds"}
        )
        self.assertEqual(event_entry["time"], 1)
        self.assertEqual(
            event_entry["event"],
            {"time": 1, "action": "node-down", "node": "B"},
        )

    def test_baseline_round_zero_is_self_origination_only(self):
        baseline = replay_ls_scenario(CHAIN3, {"events": []})["timeline"][0]
        round_zero = baseline["rounds"][0]
        self.assertEqual(
            _db_view(round_zero),
            {
                "A": {"A": (1, ("B",))},
                "B": {"B": (1, ("A", "C"))},
                "C": {"C": (1, ("B",))},
            },
        )
        # Every LSA has exactly the two published fields and sorted neighbors.
        for database in round_zero["databases"].values():
            for lsa in database.values():
                self.assertEqual(set(lsa), {"sequence", "neighbors"})
        # Round 0 each router can reach only itself (it has no other LSA).
        self.assertEqual(
            round_zero["routers"]["A"]["C"], {"nextHop": None, "metric": None}
        )
        self.assertEqual(
            round_zero["routers"]["B"]["A"], {"nextHop": None, "metric": None}
        )

    def test_lsas_propagate_one_hop_per_round(self):
        baseline = replay_ls_scenario(CHAIN4_LS, {"events": []})["timeline"][0]
        self.assertEqual(baseline["convergenceRound"], 3)
        views = _db_view
        ownership = [
            {router: set(db) for router, db in views(snapshot).items()}
            for snapshot in baseline["rounds"]
        ]
        self.assertEqual(
            ownership[0],
            {"A": {"A"}, "B": {"B"}, "C": {"C"}, "D": {"D"}},
        )
        self.assertEqual(
            ownership[1],
            {
                "A": {"A", "B"},
                "B": {"A", "B", "C"},
                "C": {"B", "C", "D"},
                "D": {"C", "D"},
            },
        )
        self.assertEqual(
            ownership[2],
            {
                "A": {"A", "B", "C"},
                "B": {"A", "B", "C", "D"},
                "C": {"A", "B", "C", "D"},
                "D": {"B", "C", "D"},
            },
        )
        self.assertEqual(
            ownership[3],
            {router: {"A", "B", "C", "D"} for router in "ABCD"},
        )
        # The converged per-database SPF result is the static link-state view.
        self.assertEqual(
            baseline["rounds"][-1]["routers"],
            compute_topology(CHAIN4_LS)["routers"],
        )

    def test_rounds_are_contiguous_and_each_changes_to_a_fixed_point(self):
        for topology in (EMPTY, SINGLE, COMPONENTS, CHAIN3, DIAMOND, CHAIN4_LS):
            result = replay_ls_scenario(topology, {"events": []})
            entry = result["timeline"][0]
            numbers = [snapshot["round"] for snapshot in entry["rounds"]]
            self.assertEqual(numbers, list(range(len(numbers))), topology)
            self.assertEqual(entry["convergenceRound"], numbers[-1])
            for previous, current in zip(entry["rounds"], entry["rounds"][1:]):
                self.assertNotEqual(previous["databases"], current["databases"])

    def test_disconnected_topologies_converge_at_round_zero(self):
        # A topology whose nodes have no links at all is fixed at round 0.
        isolated = {"nodes": ["x", "y"], "links": []}
        entry = replay_ls_scenario(isolated, {"events": []})["timeline"][0]
        self.assertEqual(entry["convergenceRound"], 0)

        # COMPONENTS has two-node links: those exchange at round 1, but the
        # fully isolated node never learns anyone else's LSA.
        entry = replay_ls_scenario(COMPONENTS, {"events": []})["timeline"][0]
        self.assertEqual(entry["convergenceRound"], 1)
        round_zero = entry["rounds"][0]
        self.assertEqual(
            _db_view(round_zero)["iso"], {"iso": (1, ())}
        )
        self.assertEqual(
            _db_view(round_zero)["a1"], {"a1": (1, ("a2",))}
        )
        # Nothing is learned across components even at the fixed point, and
        # the isolated node's database never gains another originator.
        final_view = _db_view(entry["rounds"][-1])
        self.assertEqual(set(final_view["iso"]), {"iso"})
        self.assertEqual(set(final_view["a1"]), {"a1", "a2"})
        empty = replay_ls_scenario(EMPTY, {"events": []})["timeline"][0]
        self.assertEqual(
            empty,
            {
                "event": None,
                "convergenceRound": 0,
                "rounds": [{"round": 0, "databases": {}, "routers": {}}],
            },
        )

    def test_link_down_floods_and_drops_the_stale_route(self):
        scenario = {
            "events": [{"time": 1, "action": "link-down", "from": "A", "to": "B"}]
        }
        entry = replay_ls_scenario(CHAIN3, scenario)["timeline"][1]
        # A and B bump their own sequence at round 0; C is untouched.
        round_zero = entry["rounds"][0]
        self.assertEqual(
            _db_view(round_zero)["A"]["A"], (2, ())
        )
        self.assertEqual(
            _db_view(round_zero)["B"]["B"], (2, ("C",))
        )
        self.assertEqual(
            _db_view(round_zero)["C"]["C"], (1, ("B",))
        )
        # C did not change adjacency: it keeps the old view and at round 0
        # still routes to A through B using the stale mutual LSAs.
        self.assertEqual(
            round_zero["routers"]["C"]["A"], {"nextHop": "B", "metric": 12}
        )
        # One flooding round later C has B's seq-2 LSA; the A-B edge is gone
        # from every consistent view and the stale route disappears.
        self.assertEqual(entry["convergenceRound"], 1)
        final = entry["rounds"][1]
        self.assertEqual(_db_view(final)["C"]["B"], (2, ("C",)))
        self.assertEqual(
            final["routers"]["C"]["A"], {"nextHop": None, "metric": None}
        )
        self.assertEqual(
            final["routers"]["C"]["B"], {"nextHop": "B", "metric": 7}
        )
        self.assertEqual(
            final["routers"]["A"]["B"], {"nextHop": None, "metric": None}
        )

    def test_node_down_keeps_null_database_and_router_rows(self):
        scenario = {
            "events": [{"time": 1, "action": "node-down", "node": "B"}]
        }
        entry = replay_ls_scenario(CHAIN3, scenario)["timeline"][1]
        for snapshot in entry["rounds"]:
            self.assertIsNone(snapshot["databases"]["B"])
            self.assertIsNone(snapshot["routers"]["B"])
            # Online routers keep a (stale) database entry for B at first.
        round_zero = entry["rounds"][0]
        self.assertEqual(_db_view(round_zero)["A"]["A"], (2, ()))
        self.assertEqual(_db_view(round_zero)["C"]["C"], (2, ()))
        # A and C are now isolated; nothing can be flooded to either side.
        self.assertEqual(entry["convergenceRound"], 0)
        self.assertEqual(
            round_zero["routers"]["A"]["C"], {"nextHop": None, "metric": None}
        )

    def test_recovered_node_continues_sequence_and_relearns_neighbors(self):
        scenario = {
            "events": [
                {"time": 1, "action": "node-down", "node": "B"},
                {"time": 2, "action": "node-up", "node": "B"},
            ]
        }
        timeline = replay_ls_scenario(CHAIN3, scenario)["timeline"]
        recovery_round_zero = timeline[2]["rounds"][0]
        # B restarts with an empty database but its own LSA continues at 2.
        self.assertEqual(
            _db_view(recovery_round_zero)["B"], {"B": (2, ("A", "C"))}
        )
        # Neighbors still carry B's old seq-1 LSA at round 0, so A can keep a
        # stale end-to-end route built from the old view; B itself knows
        # nobody yet.
        self.assertEqual(
            recovery_round_zero["routers"]["A"]["C"],
            {"nextHop": "B", "metric": 12},
        )
        self.assertEqual(
            recovery_round_zero["routers"]["B"]["A"],
            {"nextHop": None, "metric": None},
        )
        # After flooding all databases agree and tables match the baseline.
        final = timeline[2]["rounds"][-1]
        self.assertEqual(
            _db_view(final)["A"],
            {"A": (3, ("B",)), "B": (2, ("A", "C")), "C": (3, ("B",))},
        )
        self.assertEqual(
            final["routers"], compute_topology(CHAIN3)["routers"]
        )

    def test_repeated_down_up_keeps_incrementing_from_history(self):
        scenario = {
            "events": [
                {"time": 1, "action": "node-down", "node": "B"},
                {"time": 2, "action": "node-up", "node": "B"},
                {"time": 3, "action": "node-down", "node": "B"},
                {"time": 4, "action": "node-up", "node": "B"},
            ]
        }
        timeline = replay_ls_scenario(CHAIN3, scenario)["timeline"]
        self.assertEqual(_db_view(timeline[2]["rounds"][0])["B"]["B"], (2, ("A", "C")))
        self.assertEqual(_db_view(timeline[4]["rounds"][0])["B"]["B"], (3, ("A", "C")))

    def test_explicit_link_down_survives_node_recovery(self):
        scenario = {
            "events": [
                {"time": 1, "action": "link-down", "from": "A", "to": "B"},
                {"time": 2, "action": "node-down", "node": "B"},
                {"time": 3, "action": "node-up", "node": "B"},
            ]
        }
        timeline = replay_ls_scenario(CHAIN3, scenario)["timeline"]
        final = timeline[3]["rounds"][-1]
        view = _db_view(final)
        # B bumped its sequence on the link-down (seq 2) and again on
        # recovery (seq 3); its recovered LSA names only C.
        self.assertEqual(view["B"]["B"], (3, ("C",)))
        # Both A and B keep stale pre-failure LSAs for each other because no
        # withdrawal crosses the disabled link: A holds B's baseline LSA and
        # B holds A's.
        self.assertEqual(view["A"]["B"], (1, ("A", "C")))
        self.assertEqual(view["B"]["A"], (1, ("B",)))
        # B's own current LSA no longer names A, so the stale declarations
        # never form a mutual edge and neither side routes across it.
        self.assertEqual(
            final["routers"]["A"]["B"], {"nextHop": None, "metric": None}
        )
        self.assertEqual(
            final["routers"]["A"]["C"], {"nextHop": None, "metric": None}
        )
        self.assertEqual(
            final["routers"]["B"]["C"], {"nextHop": "C", "metric": 7}
        )

    def test_reversed_link_endpoints_round_trip(self):
        scenario = {
            "events": [
                {"time": 1, "action": "link-down", "from": "A", "to": "B"},
                {"time": 2, "action": "link-up", "from": "B", "to": "A"},
            ]
        }
        timeline = replay_ls_scenario(CHAIN3, scenario)["timeline"]
        final = timeline[2]["rounds"][-1]
        self.assertEqual(final["routers"], compute_topology(CHAIN3)["routers"])
        # The restored link is re-declared by seq-2 LSAs on both endpoints.
        self.assertEqual(_db_view(final)["A"]["A"], (3, ("B",)))
        self.assertEqual(_db_view(final)["B"]["B"], (3, ("A", "C")))

    def test_equal_cost_picks_smaller_next_hop(self):
        timeline = replay_ls_scenario(DIAMOND, {"events": []})["timeline"]
        final = timeline[0]["rounds"][-1]
        self.assertEqual(final["routers"]["A"]["D"]["nextHop"], "B")
        self.assertEqual(final["routers"]["D"]["A"]["nextHop"], "B")
        self.assertEqual(final["routers"]["B"]["C"]["nextHop"], "A")

    def test_events_are_echoed_verbatim_in_order(self):
        events = [
            {"time": 3, "action": "link-down", "from": "A", "to": "B"},
            {"time": 9, "action": "node-down", "node": "C"},
        ]
        timeline = replay_ls_scenario(DIAMOND, {"events": events})["timeline"]
        self.assertEqual(len(timeline), 3)
        for index, event in enumerate(events, start=1):
            self.assertEqual(timeline[index]["time"], event["time"])
            self.assertEqual(timeline[index]["event"], event)


class TestReplayLinkStateValidation(unittest.TestCase):
    def test_invalid_topology_raises_invalid_topology(self):
        with self.assertRaises(InvalidTopology):
            replay_ls_scenario({"nodes": ["A", "A"], "links": []}, {"events": []})

    def test_invalid_scenarios_raise_invalid_scenario(self):
        invalid = [
            [], "x", 42, {},
            {"events": {}}, {"events": [None]},
            {"events": [{"time": 1, "action": "explode", "node": "A"}]},
            {"events": [{"time": 0, "action": "node-down", "node": "A"}]},
            {"events": [{"time": 1, "action": "link-down", "from": "A", "to": "Z"}]},
        ]
        for index, scenario in enumerate(invalid):
            with self.subTest(case=index):
                with self.assertRaises(InvalidScenario):
                    replay_ls_scenario(CHAIN3, scenario)

    def test_illegal_transitions_raise_distinguishably(self):
        cases = [
            {"events": [{"time": 1, "action": "node-up", "node": "A"}]},
            {"events": [{"time": 1, "action": "link-up", "from": "A", "to": "B"}]},
            {
                "events": [
                    {"time": 1, "action": "node-down", "node": "A"},
                    {"time": 2, "action": "node-down", "node": "A"},
                ]
            },
        ]
        for index, scenario in enumerate(cases):
            with self.subTest(case=index):
                with self.assertRaises(InvalidStateTransition):
                    replay_ls_scenario(CHAIN3, scenario)

    def test_topology_errors_take_precedence(self):
        with self.assertRaises(InvalidTopology):
            replay_ls_scenario(
                {"nodes": ["A", "A"], "links": []},
                {"events": [{"time": 1, "action": "explode"}]},
            )


class TestReplayLinkStatePurity(unittest.TestCase):
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
        for snapshot in faulted["timeline"][1]["rounds"]:
            self.assertIsNone(snapshot["routers"]["B"])
        fresh = replay_ls_scenario(CHAIN3, {"events": []})
        baseline = fresh["timeline"][0]
        self.assertEqual(baseline["convergenceRound"], 2)
        # A fresh baseline starts at sequence 1 for every originator.
        for database in baseline["rounds"][-1]["databases"].values():
            for lsa in database.values():
                self.assertEqual(lsa["sequence"], 1)
        self.assertEqual(
            replay_ls_scenario(CHAIN3, copy.deepcopy(self.SCENARIO)),
            replay_ls_scenario(CHAIN3, copy.deepcopy(self.SCENARIO)),
        )

    def test_mutating_result_cannot_pollute_later_calls(self):
        scenario = {
            "events": [{"time": 1, "action": "node-down", "node": "B"}]
        }
        first = replay_ls_scenario(CHAIN3, scenario)
        first["timeline"][1]["rounds"][0]["databases"]["A"]["A"] = {
            "sequence": 999, "neighbors": ["HACK"],
        }
        first["timeline"][1]["event"]["node"] = "C"
        second = replay_ls_scenario(CHAIN3, copy.deepcopy(scenario))
        self.assertEqual(
            _db_view(second["timeline"][1]["rounds"][0])["A"]["A"], (2, ())
        )
        self.assertEqual(second["timeline"][1]["event"]["node"], "B")

    def test_echoed_event_is_not_aliased_to_input(self):
        result = replay_ls_scenario(
            CHAIN3,
            {"events": [{"time": 1, "action": "node-down", "node": "B"}]},
        )
        scenario = {"events": [{"time": 1, "action": "node-down", "node": "B"}]}
        self.assertIsNot(
            result["timeline"][1]["event"],
            {"time": 1, "action": "node-down", "node": "B"},
        )

    def test_result_is_json_serialisable_and_sorted(self):
        result = replay_ls_scenario(CHAIN3, copy.deepcopy(self.SCENARIO))
        json.loads(json.dumps(result))
        for entry in result["timeline"]:
            for snapshot in entry["rounds"]:
                self.assertEqual(list(snapshot["databases"]), ["A", "B", "C"])
                online = [r for r in snapshot["routers"]]
                self.assertEqual(online, ["A", "B", "C"])
                for router, database in snapshot["databases"].items():
                    if database is not None:
                        self.assertEqual(list(database), sorted(database))


# ---------------------------------------------------------------------------
# Distance-vector failure replay (replay-dv core)
# ---------------------------------------------------------------------------


def _metric_view(routers):
    """{router: {destination: metric}} from a nested routing snapshot."""
    return {
        router: {dest: entry["metric"] for dest, entry in row.items()}
        for router, row in routers.items()
    }


class TestDistanceVectorReplay(unittest.TestCase):
    INF8_SCENARIO = {
        "infinityMetric": 8,
        "events": [{"time": 1, "action": "node-down", "node": "C"}],
    }

    def test_output_shape_and_baseline(self):
        result = replay_dv_scenario(DV_CHAIN, self.INF8_SCENARIO)
        self.assertEqual(set(result), {"protocol", "infinityMetric", "timeline"})
        self.assertEqual(result["protocol"], "distance-vector")
        self.assertEqual(result["infinityMetric"], 8)
        timeline = result["timeline"]
        self.assertEqual(len(timeline), 2)

        baseline = timeline[0]
        self.assertEqual(set(baseline), {"event", "convergenceRound", "rounds"})
        self.assertIsNone(baseline["event"])
        self.assertEqual(
            [snapshot["round"] for snapshot in baseline["rounds"]], [0, 1]
        )
        self.assertEqual(baseline["convergenceRound"], 1)
        for snapshot in baseline["rounds"]:
            self.assertEqual(set(snapshot), {"round", "routers"})

        event_entry = timeline[1]
        self.assertEqual(
            set(event_entry),
            {"time", "event", "convergenceRound", "rounds"},
        )
        self.assertEqual(event_entry["time"], 1)
        self.assertEqual(
            event_entry["event"],
            {"time": 1, "action": "node-down", "node": "C"},
        )

    def test_baseline_with_large_infinity_matches_converge_and_compute(self):
        result = replay_dv_scenario(CHAIN3, {"infinityMetric": 100, "events": []})
        baseline = result["timeline"][0]
        converged = baseline["rounds"][-1]["routers"]
        self.assertEqual(converged, converge_topology(CHAIN3)["rounds"][-1]["routers"])
        self.assertEqual(converged, compute_topology(CHAIN3)["routers"])
        self.assertEqual(baseline["convergenceRound"], 1)
        self.assertEqual(
            [snapshot["round"] for snapshot in baseline["rounds"]], [0, 1]
        )

    def test_empty_and_isolated_topologies(self):
        result = replay_dv_scenario(
            EMPTY, {"infinityMetric": 1, "events": []}
        )
        self.assertEqual(
            result["timeline"][0],
            {"event": None, "convergenceRound": 0,
             "rounds": [{"round": 0, "routers": {}}]},
        )
        result = replay_dv_scenario(
            COMPONENTS, {"infinityMetric": 9, "events": []}
        )
        baseline = result["timeline"][0]
        self.assertEqual(baseline["convergenceRound"], 0)
        round_zero = baseline["rounds"][0]["routers"]
        self.assertEqual(
            round_zero["iso"],
            {
                "a1": {"nextHop": None, "metric": None},
                "a2": {"nextHop": None, "metric": None},
                "b1": {"nextHop": None, "metric": None},
                "b2": {"nextHop": None, "metric": None},
                "iso": {"nextHop": None, "metric": 0},
            },
        )

    def test_round_zero_then_count_to_infinity_after_node_down(self):
        result = replay_dv_scenario(DV_CHAIN, self.INF8_SCENARIO)
        rounds = result["timeline"][1]["rounds"]
        self.assertEqual(result["timeline"][1]["convergenceRound"], 6)
        metrics = [
            (snapshot["routers"]["A"]["C"]["metric"],
             snapshot["routers"]["B"]["C"]["metric"])
            for snapshot in rounds
        ]
        self.assertEqual(
            metrics,
            [
                (2, None),      # round 0: A keeps stale via B; B loses via C
                (None, 3),      # B adopts A's old metric 2 + 1
                (4, None),      # A adopts B's 3 + 1
                (None, 5),
                (6, None),
                (None, 7),
                (None, None),   # metric 8 == infinity becomes unreachable
            ],
        )
        # Alternating next hops on the rising metrics.
        self.assertEqual(
            rounds[1]["routers"]["B"]["C"], {"nextHop": "A", "metric": 3}
        )
        self.assertEqual(
            rounds[2]["routers"]["A"]["C"], {"nextHop": "B", "metric": 4}
        )
        self.assertEqual(
            rounds[5]["routers"]["B"]["C"], {"nextHop": "A", "metric": 7}
        )
        # Final snapshot is a fixed point under another synchronous update.
        from packet_routing_sim.core.routing import distance_vector_round_infinity

        topo = validate_topology(DV_CHAIN)
        final = rounds[-1]["routers"]
        active = {"A": {"B": 1}, "B": {"A": 1}, "C": {}}
        again = distance_vector_round_infinity(
            final, topo.nodes, active, frozenset({"C"}), 8
        )
        self.assertEqual(again, final)

    def test_three_router_counting_alternates_with_tie_to_smaller_name(self):
        # A-B-C-D unit chain; C-D fails while D stays up.  A keeps
        # advertising too, and B's equal candidates via A and via C tie to
        # the smaller neighbor name A.
        chain4 = {
            "nodes": ["A", "B", "C", "D"],
            "links": [
                link("A", "B", 1),
                link("B", "C", 1),
                link("C", "D", 1),
            ],
        }
        scenario = {
            "infinityMetric": 7,
            "events": [
                {"time": 1, "action": "link-down", "from": "C", "to": "D"}
            ],
        }
        rounds = replay_dv_scenario(chain4, scenario)["timeline"][1]["rounds"]
        toward_d = [
            (
                snapshot["routers"]["A"]["D"]["metric"],
                snapshot["routers"]["B"]["D"]["metric"],
                snapshot["routers"]["C"]["D"]["metric"],
            )
            for snapshot in rounds
        ]
        self.assertEqual(
            toward_d,
            [
                (3, 2, None),
                (3, 4, 3),
                (5, 4, 5),
                (5, 6, 5),
                (None, 6, None),
                (None, None, None),
            ],
        )
        self.assertEqual(
            rounds[1]["routers"]["B"]["D"], {"nextHop": "A", "metric": 4}
        )
        self.assertEqual(
            rounds[2]["routers"]["C"]["D"], {"nextHop": "B", "metric": 5}
        )

    def test_infinity_threshold_terminates_early(self):
        scenario = {
            "infinityMetric": 3,
            "events": [{"time": 1, "action": "node-down", "node": "C"}],
        }
        entry = replay_dv_scenario(DV_CHAIN, scenario)["timeline"][1]
        self.assertEqual(entry["convergenceRound"], 1)
        final = entry["rounds"][-1]["routers"]
        self.assertEqual(final["A"]["C"], {"nextHop": None, "metric": None})
        self.assertEqual(final["B"]["C"], {"nextHop": None, "metric": None})

    def test_rounds_are_contiguous_and_each_changes(self):
        result = replay_dv_scenario(DV_CHAIN, self.INF8_SCENARIO)
        for entry in result["timeline"]:
            numbers = [snapshot["round"] for snapshot in entry["rounds"]]
            self.assertEqual(numbers, list(range(len(numbers))))
            self.assertEqual(entry["convergenceRound"], numbers[-1])
            for previous, current in zip(entry["rounds"], entry["rounds"][1:]):
                self.assertNotEqual(previous["routers"], current["routers"])

    def test_down_router_row_is_unreachable_in_every_round(self):
        result = replay_dv_scenario(DV_CHAIN, self.INF8_SCENARIO)
        rounds = result["timeline"][1]["rounds"]
        for snapshot in rounds:
            # The down router's own row is unreachable for every
            # destination in every round, including itself.
            for dest in ("A", "B", "C"):
                self.assertEqual(
                    snapshot["routers"]["C"][dest],
                    {"nextHop": None, "metric": None},
                    (snapshot["round"], dest),
                )
        # Online routers count up toward the down destination; once the
        # phase converges, every one of them reports it unreachable.
        final = rounds[-1]["routers"]
        for router in ("A", "B"):
            self.assertEqual(
                final[router]["C"], {"nextHop": None, "metric": None}
            )
        # But at round 0 stale reachability is deliberately retained (here
        # A still routes via B), which is what fuels the counting.
        self.assertEqual(
            rounds[0]["routers"]["A"]["C"], {"nextHop": "B", "metric": 2}
        )
        self.assertEqual(
            rounds[0]["routers"]["B"]["C"], {"nextHop": None, "metric": None}
        )

    def test_node_recovery_round_zero_is_fresh_and_converges(self):
        scenario = {
            "infinityMetric": 50,
            "events": [
                {"time": 1, "action": "node-down", "node": "C"},
                {"time": 2, "action": "node-up", "node": "C"},
            ],
        }
        timeline = replay_dv_scenario(CHAIN3, scenario)["timeline"]
        recovered_round_zero = timeline[2]["rounds"][0]["routers"]["C"]
        self.assertEqual(
            recovered_round_zero,
            {
                "A": {"nextHop": None, "metric": None},
                "B": {"nextHop": "B", "metric": 7},
                "C": {"nextHop": None, "metric": 0},
            },
        )
        final = timeline[2]["rounds"][-1]["routers"]
        self.assertEqual(final["A"]["C"], {"nextHop": "B", "metric": 12})
        self.assertEqual(final["B"]["C"], {"nextHop": "C", "metric": 7})
        self.assertEqual(final["C"]["A"], {"nextHop": "B", "metric": 12})

    def test_link_recovery_restores_direct_routes_at_round_zero(self):
        scenario = {
            "infinityMetric": 8,
            "events": [
                {"time": 1, "action": "link-down", "from": "B", "to": "A"},
                {"time": 2, "action": "link-up", "from": "A", "to": "B"},
            ],
        }
        timeline = replay_dv_scenario(CHAIN3, scenario)["timeline"]
        round_zero = timeline[2]["rounds"][0]["routers"]
        self.assertEqual(round_zero["A"]["B"], {"nextHop": "B", "metric": 5})
        self.assertEqual(round_zero["B"]["A"], {"nextHop": "A", "metric": 5})

    def test_explicit_link_down_survives_node_recovery(self):
        scenario = {
            "infinityMetric": 9,
            "events": [
                {"time": 1, "action": "link-down", "from": "A", "to": "B"},
                {"time": 2, "action": "node-down", "node": "B"},
                {"time": 3, "action": "node-up", "node": "B"},
            ],
        }
        final = replay_dv_scenario(CHAIN3, scenario)["timeline"][3]["rounds"][-1]["routers"]
        self.assertEqual(final["A"]["B"], {"nextHop": None, "metric": None})
        self.assertEqual(final["A"]["C"], {"nextHop": None, "metric": None})
        self.assertEqual(final["B"]["C"], {"nextHop": "C", "metric": 7})

    def test_reversed_link_endpoints_round_trip(self):
        scenario = {
            "infinityMetric": 50,
            "events": [
                {"time": 1, "action": "link-down", "from": "A", "to": "B"},
                {"time": 2, "action": "link-up", "from": "B", "to": "A"},
            ],
        }
        timeline = replay_dv_scenario(CHAIN3, scenario)["timeline"]
        final = timeline[-1]["rounds"][-1]["routers"]
        self.assertEqual(final, compute_topology(CHAIN3)["routers"])

    def test_events_are_echoed_verbatim_in_order(self):
        events = [
            {"time": 3, "action": "link-down", "from": "A", "to": "B"},
            {"time": 9, "action": "node-down", "node": "C"},
        ]
        scenario = {"infinityMetric": 50, "events": events}
        timeline = replay_dv_scenario(CHAIN3, scenario)["timeline"]
        self.assertEqual(len(timeline), 3)
        for index, event in enumerate(events, start=1):
            self.assertEqual(timeline[index]["time"], event["time"])
            self.assertEqual(timeline[index]["event"], event)

    def test_long_paths_unreachable_when_infinity_is_small(self):
        # infinity must exceed every *link* metric, not the network diameter:
        # metric-10 links with infinity 11 make every 2+ hop route null.
        chain = {
            "nodes": ["n0", "n1", "n2", "n3"],
            "links": [
                link("n0", "n1", 10),
                link("n1", "n2", 10),
                link("n2", "n3", 10),
            ],
        }
        result = replay_dv_scenario(chain, {"infinityMetric": 11, "events": []})
        final = result["timeline"][0]["rounds"][-1]["routers"]
        self.assertEqual(
            _metric_view(final)["n0"],
            {"n0": 0, "n1": 10, "n2": None, "n3": None},
        )


class TestDistanceVectorReplayValidation(unittest.TestCase):
    def test_validate_dv_scenario_returns_events_and_infinity(self):
        events, infinity_metric, hold_down_rounds, poison_reverse = (
            validate_dv_scenario(
                validate_topology(CHAIN3),
                {"infinityMetric": 8,
                 "events": [{"time": 1, "action": "node-down", "node": "C"}]},
            )
        )
        self.assertEqual(infinity_metric, 8)
        self.assertIsNone(hold_down_rounds)
        self.assertFalse(poison_reverse)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].node, "C")

    def test_invalid_infinity_metric_values(self):
        invalid = [
            {},
            {"events": []},
            {"infinityMetric": None, "events": []},
            {"infinityMetric": True, "events": []},
            {"infinityMetric": False, "events": []},
            {"infinityMetric": 0, "events": []},
            {"infinityMetric": -4, "events": []},
            {"infinityMetric": 1.5, "events": []},
            {"infinityMetric": "8", "events": []},
            {"infinityMetric": [], "events": []},
            {"infinityMetric": {}, "events": []},
            # CHAIN3's largest link metric is 7: must be strictly greater.
            {"infinityMetric": 7, "events": []},
            {"infinityMetric": 2, "events": []},
        ]
        for index, scenario in enumerate(invalid):
            with self.subTest(case=index):
                with self.assertRaises(InvalidScenario):
                    replay_dv_scenario(CHAIN3, scenario)

    def test_infinity_compared_against_maximum_link_metric(self):
        topology = {
            "nodes": ["X", "Y"],
            "links": [link("X", "Y", 5)],
        }
        with self.assertRaises(InvalidScenario):
            replay_dv_scenario(topology, {"infinityMetric": 5, "events": []})
        with self.assertRaises(InvalidScenario):
            replay_dv_scenario(topology, {"infinityMetric": 4, "events": []})
        result = replay_dv_scenario(
            topology, {"infinityMetric": 6, "events": []}
        )
        self.assertEqual(result["infinityMetric"], 6)

    def test_empty_topology_accepts_any_positive_infinity(self):
        result = replay_dv_scenario(
            EMPTY, {"infinityMetric": 1, "events": []}
        )
        self.assertEqual(result["infinityMetric"], 1)

    def test_event_rules_and_transitions_are_still_enforced(self):
        base = {"infinityMetric": 50}
        malformed = [
            [], "x", 42,
            {"infinityMetric": 50, "events": "x"},
            {"infinityMetric": 50, "events": [None]},
            {"infinityMetric": 50,
             "events": [{"time": 0, "action": "node-down", "node": "A"}]},
            {"infinityMetric": 50,
             "events": [{"time": 1, "action": "explode", "node": "A"}]},
        ]
        for index, scenario in enumerate(malformed):
            with self.subTest(case=index):
                with self.assertRaises(InvalidScenario):
                    replay_dv_scenario(CHAIN3, scenario)
        with self.assertRaises(InvalidStateTransition):
            replay_dv_scenario(
                CHAIN3,
                {"infinityMetric": 50,
                 "events": [{"time": 1, "action": "node-up", "node": "A"}]},
            )

    def test_topology_errors_take_precedence(self):
        with self.assertRaises(InvalidTopology):
            replay_dv_scenario(
                {"nodes": ["A", "A"], "links": []},
                {"infinityMetric": 1},
            )


class TestDistanceVectorReplayPurity(unittest.TestCase):
    SCENARIO = {
        "infinityMetric": 50,
        "events": [
            {"time": 1, "action": "link-down", "from": "A", "to": "B"},
            {"time": 2, "action": "node-down", "node": "B"},
            {"time": 3, "action": "node-up", "node": "B"},
            {"time": 4, "action": "link-up", "from": "B", "to": "A"},
        ],
    }

    def test_inputs_remain_equal(self):
        topology_before = copy.deepcopy(CHAIN3)
        scenario_before = copy.deepcopy(self.SCENARIO)
        replay_dv_scenario(CHAIN3, self.SCENARIO)
        self.assertEqual(CHAIN3, topology_before)
        self.assertEqual(self.SCENARIO, scenario_before)

    def test_no_carry_over_between_calls(self):
        faulted = replay_dv_scenario(
            CHAIN3,
            {"infinityMetric": 50,
             "events": [{"time": 1, "action": "node-down", "node": "B"}]},
        )
        self.assertIsNone(
            faulted["timeline"][1]["rounds"][-1]["routers"]["A"]["B"]["nextHop"]
        )
        fresh = replay_dv_scenario(
            CHAIN3, {"infinityMetric": 50, "events": []}
        )
        self.assertEqual(
            fresh["timeline"][0]["rounds"][-1]["routers"],
            converge_topology(CHAIN3)["rounds"][-1]["routers"],
        )
        self.assertEqual(
            replay_dv_scenario(CHAIN3, copy.deepcopy(self.SCENARIO)),
            replay_dv_scenario(CHAIN3, copy.deepcopy(self.SCENARIO)),
        )

    def test_mutating_result_cannot_pollute_later_calls(self):
        scenario = {
            "infinityMetric": 50,
            "events": [{"time": 1, "action": "node-down", "node": "C"}],
        }
        first = replay_dv_scenario(CHAIN3, scenario)
        first["timeline"][1]["rounds"][0]["routers"]["A"]["C"] = {
            "nextHop": "HACK", "metric": -1,
        }
        first["timeline"][1]["event"]["node"] = "A"
        second = replay_dv_scenario(CHAIN3, copy.deepcopy(scenario))
        self.assertEqual(
            second["timeline"][1]["rounds"][0]["routers"]["A"]["C"],
            {"nextHop": "B", "metric": 12},
        )
        self.assertEqual(second["timeline"][1]["event"]["node"], "C")

    def test_echoed_event_is_not_aliased_to_input(self):
        scenario = {
            "infinityMetric": 50,
            "events": [{"time": 1, "action": "node-down", "node": "C"}],
        }
        result = replay_dv_scenario(CHAIN3, scenario)
        self.assertIsNot(result["timeline"][1]["event"], scenario["events"][0])

    def test_result_is_json_serialisable(self):
        result = replay_dv_scenario(CHAIN3, copy.deepcopy(self.SCENARIO))
        json.loads(json.dumps(result))
        for entry in result["timeline"]:
            self.assertEqual(list(entry["rounds"][0]["routers"]), ["A", "B", "C"])
            for snapshot in entry["rounds"]:
                for row in snapshot["routers"].values():
                    self.assertEqual(list(row), ["A", "B", "C"])


# ---------------------------------------------------------------------------
# Distance-vector replay with route hold-down (replay-dv holdDownRounds)
# ---------------------------------------------------------------------------


def _holddown_view(entry):
    """[{round: {router: {dest: remaining}}}] from one timeline entry."""
    return [
        {router: dict(mapping) for router, mapping in snapshot["holdDowns"].items()}
        for snapshot in entry["rounds"]
    ]


class TestDistanceVectorHoldDown(unittest.TestCase):
    INF8 = 8

    def test_root_echoes_holddown_and_every_round_carries_map(self):
        scenario = {
            "infinityMetric": self.INF8,
            "holdDownRounds": 2,
            "events": [{"time": 1, "action": "node-down", "node": "C"}],
        }
        result = replay_dv_scenario(DV_CHAIN, scenario)
        self.assertEqual(
            set(result),
            {"protocol", "infinityMetric", "holdDownRounds", "timeline"},
        )
        self.assertEqual(result["holdDownRounds"], 2)
        # Baseline: fault free, so every holdDowns map is empty.
        baseline = result["timeline"][0]
        for snapshot in baseline["rounds"]:
            self.assertEqual(set(snapshot), {"round", "routers", "holdDowns"})
            self.assertEqual(
                snapshot["holdDowns"], {"A": {}, "B": {}, "C": {}}
            )
        entry = result["timeline"][1]
        for snapshot in entry["rounds"]:
            self.assertEqual(set(snapshot), {"round", "routers", "holdDowns"})
            self.assertEqual(set(snapshot["holdDowns"]), {"A", "B", "C"})

    def test_omitted_or_zero_is_byte_compatible_with_legacy(self):
        events = [{"time": 1, "action": "node-down", "node": "C"}]
        legacy = replay_dv_scenario(
            DV_CHAIN, {"infinityMetric": self.INF8, "events": events}
        )
        explicit_zero = replay_dv_scenario(
            DV_CHAIN,
            {"infinityMetric": self.INF8, "holdDownRounds": 0, "events": events},
        )
        self.assertEqual(legacy, explicit_zero)
        self.assertNotIn("holdDownRounds", legacy)
        for entry in legacy["timeline"]:
            for snapshot in entry["rounds"]:
                self.assertNotIn("holdDowns", snapshot)

    def test_node_down_holds_then_releases_to_unreachable(self):
        scenario = {
            "infinityMetric": self.INF8,
            "holdDownRounds": 2,
            "events": [{"time": 1, "action": "node-down", "node": "C"}],
        }
        entry = replay_dv_scenario(DV_CHAIN, scenario)["timeline"][1]
        rounds = entry["rounds"]
        null = {"nextHop": None, "metric": None}
        # Round 0: both A (via failed hop B) and B (direct neighbor C gone)
        # start a 2-round timer; every held route advertises null/null.
        self.assertEqual(rounds[0]["routers"]["A"]["C"], null)
        self.assertEqual(rounds[0]["routers"]["B"]["C"], null)
        self.assertEqual(rounds[0]["holdDowns"]["A"], {"C": 2})
        self.assertEqual(rounds[0]["holdDowns"]["B"], {"C": 2})
        # One full update round later: still held, remainder one.
        self.assertEqual(rounds[1]["routers"]["A"]["C"], null)
        self.assertEqual(rounds[1]["holdDowns"]["A"], {"C": 1})
        # The round after the remainder reached zero has an empty map and the
        # destination simply stays unreachable (no alternative exists).
        self.assertEqual(rounds[2]["holdDowns"], {"A": {}, "B": {}, "C": {}})
        self.assertEqual(rounds[2]["routers"]["A"]["C"], null)
        self.assertEqual(entry["convergenceRound"], 2)

    def test_alternative_advertisement_is_refused_then_adopted(self):
        # A reaches D through B and C at equal cost (tie -> B).  When the A-B
        # link fails, the surviving A-C-D alternative must not be adopted
        # until after the hold-down window.
        scenario = {
            "infinityMetric": self.INF8,
            "holdDownRounds": 2,
            "events": [
                {"time": 1, "action": "link-down", "from": "A", "to": "B"}
            ],
        }
        entry = replay_dv_scenario(DIAMOND, scenario)["timeline"][1]
        rounds = entry["rounds"]
        null = {"nextHop": None, "metric": None}
        # Round 0: A's route via failed hop B is null and timed.  A's own
        # route to the now-non-neighbor B is invalidated and timed as well.
        self.assertEqual(rounds[0]["routers"]["A"]["D"], null)
        self.assertEqual(rounds[0]["holdDowns"]["A"], {"B": 2, "D": 2})
        # The alternative via C (metric 2) is already advertised this round
        # but is refused throughout the window.
        self.assertEqual(rounds[1]["routers"]["A"]["D"], null)
        self.assertEqual(rounds[1]["holdDowns"]["A"], {"B": 1, "D": 1})
        self.assertEqual(rounds[2]["routers"]["A"]["D"], null)
        self.assertEqual(rounds[2]["holdDowns"]["A"], {})
        # First normal-selection round after release adopts C.
        self.assertEqual(
            rounds[3]["routers"]["A"]["D"], {"nextHop": "C", "metric": 2}
        )
        self.assertEqual(rounds[3]["holdDowns"]["A"], {})
        # convergenceRound points at the last recorded (changed) round.
        self.assertEqual(entry["convergenceRound"], 4)
        self.assertEqual(
            rounds[4]["routers"]["A"]["D"], {"nextHop": "C", "metric": 2}
        )

    def test_direct_neighbor_recovery_clears_timer_immediately(self):
        # C fails (B starts a hold-down for C), then C recovers while the
        # window could still be live for a later-destined scenario; here the
        # direct link B-C makes B adopt C at round 0 of recovery.
        scenario = {
            "infinityMetric": 50,
            "holdDownRounds": 3,
            "events": [
                {"time": 1, "action": "node-down", "node": "C"},
                {"time": 2, "action": "node-up", "node": "C"},
            ],
        }
        timeline = replay_dv_scenario(CHAIN3, scenario)["timeline"]
        recovery_zero = timeline[2]["rounds"][0]["routers"]
        # Direct neighbor again: adopted at the direct metric at once, no map.
        self.assertEqual(
            recovery_zero["B"]["C"], {"nextHop": "C", "metric": 7}
        )
        self.assertEqual(
            timeline[2]["rounds"][0]["holdDowns"]["B"], {}
        )

    def test_repeated_unreachable_advertisement_does_not_extend_window(self):
        # With a genuine alternative present, confirm the published remainder
        # decreases exactly one per round regardless of continued null ads.
        scenario = {
            "infinityMetric": self.INF8,
            "holdDownRounds": 3,
            "events": [
                {"time": 1, "action": "link-down", "from": "A", "to": "B"}
            ],
        }
        entry = replay_dv_scenario(DIAMOND, scenario)["timeline"][1]
        remainders = [
            entry["rounds"][i]["holdDowns"]["A"].get("D")
            for i in range(4)
        ]
        self.assertEqual(remainders, [3, 2, 1, None])

    def test_down_router_row_stays_null_and_carries_no_timer(self):
        scenario = {
            "infinityMetric": self.INF8,
            "holdDownRounds": 2,
            "events": [{"time": 1, "action": "node-down", "node": "C"}],
        }
        rounds = replay_dv_scenario(DV_CHAIN, scenario)["timeline"][1]["rounds"]
        for snapshot in rounds:
            self.assertEqual(snapshot["holdDowns"]["C"], {})
            for dest in ("A", "B", "C"):
                self.assertEqual(
                    snapshot["routers"]["C"][dest],
                    {"nextHop": None, "metric": None},
                )

    def test_rounds_recorded_until_map_and_vectors_both_stable(self):
        scenario = {
            "infinityMetric": self.INF8,
            "holdDownRounds": 2,
            "events": [{"time": 1, "action": "node-down", "node": "C"}],
        }
        entry = replay_dv_scenario(DV_CHAIN, scenario)["timeline"][1]
        numbers = [snapshot["round"] for snapshot in entry["rounds"]]
        self.assertEqual(numbers, list(range(len(numbers))))
        self.assertEqual(entry["convergenceRound"], numbers[-1])
        # Last recorded round has no live timers and is a fixed point under
        # another synchronous hold-down update.
        from packet_routing_sim.core.routing import distance_vector_holddown_round

        topo = validate_topology(DV_CHAIN)
        final = entry["rounds"][-1]
        active = {"A": {"B": 1}, "B": {"A": 1}, "C": {}}
        empty = {"A": {}, "B": {}, "C": {}}
        again_vectors, again_holddowns = distance_vector_holddown_round(
            final["routers"],
            topo.nodes,
            active,
            frozenset({"C"}),
            self.INF8,
            empty,
            2,
        )
        self.assertEqual(again_vectors, final["routers"])
        self.assertEqual(again_holddowns, empty)

    def test_invalid_holddown_values(self):
        invalid = [True, False, -1, -100, 1.5, "2", [2], {"x": 2}, None]
        # Note: a *present* JSON null is invalid; an absent key stays legacy.
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(InvalidScenario):
                    replay_dv_scenario(
                        DV_CHAIN,
                        {
                            "infinityMetric": self.INF8,
                            "holdDownRounds": value,
                            "events": [],
                        },
                    )

    def test_holddown_does_not_change_other_entry_points(self):
        # replay / replay-ls ignore holdDownRounds entirely.
        scenario = {
            "infinityMetric": self.INF8,
            "holdDownRounds": 2,
            "events": [{"time": 1, "action": "node-down", "node": "C"}],
        }
        plain = {"events": scenario["events"]}
        self.assertEqual(
            replay_scenario(DV_CHAIN, scenario),
            replay_scenario(DV_CHAIN, plain),
        )
        self.assertEqual(
            replay_ls_scenario(DV_CHAIN, scenario),
            replay_ls_scenario(DV_CHAIN, plain),
        )

    def test_timers_do_not_leak_across_events_and_convergence_extends(self):
        # A richer multi-event scenario: every event starts from converged
        # vectors with no live timers, and the final state still matches the
        # static shortest paths once everything is restored.
        scenario = {
            "infinityMetric": 30,
            "holdDownRounds": 2,
            "events": [
                {"time": 3, "action": "link-down", "from": "A", "to": "B"},
                {"time": 5, "action": "node-down", "node": "C"},
                {"time": 9, "action": "node-up", "node": "C"},
                {"time": 15, "action": "link-up", "from": "B", "to": "A"},
            ],
        }
        result = replay_dv_scenario(DIAMOND, scenario)
        timeline = result["timeline"]
        # Each entry runs to convergence, so its last recorded round has no
        # live timer: nothing can leak into the next event's round zero.
        for entry in timeline[1:]:
            for router, mapping in entry["rounds"][-1]["holdDowns"].items():
                self.assertEqual(mapping, {}, router)
        # Final converged forwarding tables equal the static topology.
        final = timeline[-1]["rounds"][-1]["routers"]
        self.assertEqual(final, compute_topology(DIAMOND)["routers"])
        self.assertEqual(
            timeline[-1]["rounds"][-1]["holdDowns"],
            {"A": {}, "B": {}, "C": {}, "D": {}},
        )

    def test_holddown_purity_no_carry_over_or_input_mutation(self):
        scenario = {
            "infinityMetric": self.INF8,
            "holdDownRounds": 2,
            "events": [{"time": 1, "action": "node-down", "node": "C"}],
        }
        scenario_before = copy.deepcopy(scenario)
        topo_before = copy.deepcopy(DV_CHAIN)
        first = replay_dv_scenario(DV_CHAIN, scenario)
        self.assertEqual(scenario, scenario_before)
        self.assertEqual(DV_CHAIN, topo_before)
        # Mutating one result cannot affect an identical later call.
        first["timeline"][1]["rounds"][0]["holdDowns"]["A"]["C"] = 999
        second = replay_dv_scenario(DV_CHAIN, copy.deepcopy(scenario))
        self.assertEqual(
            second["timeline"][1]["rounds"][0]["holdDowns"]["A"]["C"], 2
        )
        self.assertEqual(
            replay_dv_scenario(DV_CHAIN, copy.deepcopy(scenario)),
            second,
        )


_DV_DETERMINISM_SNIPPET = """
import json, os, sys
sys.path.insert(0, %r)
from packet_routing_sim.core import replay_dv_scenario
sys.stdout.write(json.dumps(
    replay_dv_scenario(json.loads(os.environ["PRSIM_TOPO"]),
                       json.loads(os.environ["PRSIM_SCEN"])),
    sort_keys=True))
""" % str(REPO_ROOT)

def run_dv_core_in_subprocess(topology, scenario, seed):
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(seed)
    env["PRSIM_TOPO"] = json.dumps(topology)
    env["PRSIM_SCEN"] = json.dumps(scenario)
    proc = subprocess.run(
        [sys.executable, "-c", _DV_DETERMINISM_SNIPPET],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


class TestDistanceVectorReplayDeterminism(unittest.TestCase):
    TOPOLOGY = {
        "nodes": ["A", "B", "C", "D"],
        "links": [
            link("A", "B", 1),
            link("B", "C", 2),
            link("C", "D", 3),
            link("D", "A", 4),
        ],
    }
    SCENARIO = {
        "infinityMetric": 30,
        "events": [
            {"time": 3, "action": "link-down", "from": "A", "to": "B"},
            {"time": 5, "action": "node-down", "node": "C"},
            {"time": 8, "action": "link-down", "from": "D", "to": "C"},
            {"time": 9, "action": "node-up", "node": "C"},
            {"time": 12, "action": "link-up", "from": "C", "to": "D"},
            {"time": 15, "action": "link-up", "from": "B", "to": "A"},
        ],
    }
    VARIANTS = [
        TOPOLOGY,
        {"nodes": ["D", "C", "B", "A"], "links": TOPOLOGY["links"]},
        {
            "nodes": ["A", "B", "C", "D"],
            "links": [
                link("B", "A", 1),
                link("C", "B", 2),
                link("D", "C", 3),
                link("A", "D", 4),
            ],
        },
        {
            "nodes": ["A", "B", "C", "D"],
            "links": list(reversed(TOPOLOGY["links"])),
        },
    ]

    def test_byte_identical_across_seeds_and_declaration_order(self):
        outputs = set()
        for seed in HASH_SEEDS:
            for variant in self.VARIANTS:
                outputs.add(
                    run_dv_core_in_subprocess(variant, self.SCENARIO, seed)
                )
        self.assertEqual(len(outputs), 1)


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
out["replay_ls"] = core.replay_ls_scenario(topology, scenario)
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
