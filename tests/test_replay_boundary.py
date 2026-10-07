"""Regression tests for the unified replay event-progression boundary.

These tests pin the behavior that the shared timeline engine
(:mod:`packet_routing_sim.core.replay`) guarantees for every replay entry
point (``replay``, ``replay-ls`` and ``replay-dv``):

* the fault-free baseline always occupies timeline position 0 with
  ``event: null`` and no ``time``;
* empty event lists yield exactly that baseline;
* each later entry is produced by applying exactly one validated event to
  the preceding state, echoes ``time`` and the raw event verbatim (extra
  fields and the original key order included), and timeline order follows
  event order;
* consecutive node/link failures and recoveries (including a link
  explicitly disabled across a node restart, and reversed endpoint names)
  progress from inherited protocol state;
* link-state databases/sequence history and distance-vector converged
  vectors inherit *within* one call, while repeated calls start fresh and
  never share mutable results or inputs;
* outputs are independent of node/link declaration order and of the hash
  seed;
* the single error boundary is preserved: ``InvalidTopology`` /
  ``InvalidScenario`` / ``InvalidStateTransition`` for every entry point.
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
    replay_dv_scenario,
    replay_ls_scenario,
    replay_scenario,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
HASH_SEEDS = tuple(
    int(seed)
    for seed in os.environ.get("PRSIM_TEST_SEEDS", "0,1,42,987654321").split(",")
)


def link(source, target, metric):
    return {"from": source, "to": target, "metric": metric}


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

NODE_CYCLE = {"events": [
    {"time": 1, "action": "node-down", "node": "B"},
    {"time": 2, "action": "node-up", "node": "B"},
    {"time": 3, "action": "node-down", "node": "B"},
    {"time": 4, "action": "node-up", "node": "B"},
]}
LINK_CYCLE = {"events": [
    {"time": 1, "action": "link-down", "from": "B", "to": "A"},
    {"time": 2, "action": "link-up", "from": "A", "to": "B"},
    {"time": 3, "action": "link-down", "from": "A", "to": "B"},
    {"time": 4, "action": "link-up", "from": "B", "to": "A"},
]}
# Explicit link-down must outlive the endpoint's own downtime, and its
# recovery may name the endpoints in the reverse direction.
LINK_SURVIVES_NODE = {"events": [
    {"time": 1, "action": "link-down", "from": "A", "to": "B"},
    {"time": 2, "action": "node-down", "node": "B"},
    {"time": 3, "action": "node-up", "node": "B"},
    {"time": 4, "action": "link-up", "from": "B", "to": "A"},
]}

RECOVERY_PAIR = [
    {"time": 1, "action": "node-down", "node": "B"},
    {"time": 2, "action": "node-up", "node": "B"},
]

REPLAY_ENTRIES = (
    ("replay", replay_scenario, lambda events: {"events": events}),
    ("replay-ls", replay_ls_scenario, lambda events: {"events": events}),
    ("replay-dv", replay_dv_scenario,
     lambda events: {"infinityMetric": 50, "events": events}),
)


# ---------------------------------------------------------------------------
# Baseline position, empty events, verbatim echoing
# ---------------------------------------------------------------------------


class TestTimelineBoundary(unittest.TestCase):
    def test_empty_events_yield_only_the_baseline_for_every_entry(self):
        for name, entry, wrap in REPLAY_ENTRIES:
            with self.subTest(entry=name):
                result = entry(CHAIN3, wrap([]))
                self.assertEqual(result["protocol"], (
                    "distance-vector" if name == "replay-dv" else "link-state"
                ))
                timeline = result["timeline"]
                self.assertEqual(len(timeline), 1)
                baseline = timeline[0]
                self.assertIsNone(baseline["event"])
                self.assertNotIn("time", baseline)
                self.assertEqual(list(baseline)[0], "event")
                if name == "replay":
                    self.assertEqual(set(baseline), {"event", "routers"})
                else:
                    self.assertEqual(
                        set(baseline),
                        {"event", "convergenceRound", "rounds"},
                    )

    def test_timeline_order_and_one_entry_per_event(self):
        events = [
            {"time": 3, "action": "link-down", "from": "A", "to": "B"},
            {"time": 7, "action": "node-down", "node": "B"},
            {"time": 9, "action": "node-up", "node": "B"},
            {"time": 12, "action": "link-up", "from": "B", "to": "A"},
        ]
        for name, entry, wrap in REPLAY_ENTRIES:
            with self.subTest(entry=name):
                timeline = entry(CHAIN3, wrap(events))["timeline"]
                self.assertEqual(len(timeline), len(events) + 1)
                self.assertIsNone(timeline[0]["event"])
                self.assertEqual(
                    [item["time"] for item in timeline[1:]], [3, 7, 9, 12]
                )
                for item, raw in zip(timeline[1:], events):
                    self.assertEqual(item["event"], raw)
                    self.assertEqual(
                        list(item)[:2], ["time", "event"]
                    )

    def test_event_is_echoed_verbatim_with_extra_field_and_key_order(self):
        # An unrecognized field travels through untouched; JSON key order is
        # the decoded object's insertion order, which the echo must preserve.
        weird = {"z": 1, "action": "node-down", "time": 2, "node": "B",
                 "extra": ["x", "y"]}
        events = [weird]
        for name, entry, wrap in REPLAY_ENTRIES:
            with self.subTest(entry=name):
                scenario = wrap(events)
                echoed = entry(CHAIN3, scenario)["timeline"][1]["event"]
                self.assertEqual(echoed, weird)
                self.assertEqual(list(echoed), list(weird))
                self.assertEqual(
                    json.dumps(echoed, sort_keys=False),
                    json.dumps(weird, sort_keys=False),
                )
                # The echo is a copy, never the caller's own dict.
                self.assertIsNot(echoed, scenario["events"][0])

    def test_round_numbering_is_contiguous_from_zero(self):
        for name, entry, wrap in REPLAY_ENTRIES:
            with self.subTest(entry=name):
                result = entry(CHAIN3, wrap(RECOVERY_PAIR))
                for item in result["timeline"]:
                    if "rounds" not in item:
                        continue
                    numbers = [snapshot["round"] for snapshot in item["rounds"]]
                    self.assertEqual(numbers, list(range(len(numbers))))
                    self.assertEqual(
                        item["convergenceRound"], numbers[-1]
                    )


# ---------------------------------------------------------------------------
# Consecutive failures and recoveries through the single progression edge
# ---------------------------------------------------------------------------


class TestConsecutiveTransitions(unittest.TestCase):
    def test_plain_replay_returns_to_baseline_after_each_cycle(self):
        # Node and link cycles are whole again at entry 2 and again at the
        # end; the link-survives-node case stays split at entry 2 (the link
        # is still explicitly down) and only returns after its link-up.
        whole_again_early = {"node", "link"}
        for label, events in (
            ("node", NODE_CYCLE["events"]),
            ("link", LINK_CYCLE["events"]),
            ("link-survives-node", LINK_SURVIVES_NODE["events"]),
        ):
            with self.subTest(scenario=label):
                timeline = replay_scenario(CHAIN3, {"events": events})[
                    "timeline"
                ]
                baseline = timeline[0]["routers"]
                if label in whole_again_early:
                    self.assertEqual(timeline[2]["routers"], baseline)
                self.assertEqual(timeline[-1]["routers"], baseline)

    def test_explicit_link_down_remains_after_node_recovery(self):
        timeline = replay_scenario(CHAIN3, LINK_SURVIVES_NODE)["timeline"]
        # Entry 3: B is back, but the A-B link stays explicitly disabled.
        routers = timeline[3]["routers"]
        self.assertEqual(routers["A"]["B"], {"nextHop": None, "metric": None})
        self.assertEqual(routers["A"]["C"], {"nextHop": None, "metric": None})
        self.assertEqual(routers["B"]["C"], {"nextHop": "C", "metric": 7})

    def test_reversed_endpoints_round_trip_for_each_protocol(self):
        events = [
            {"time": 1, "action": "link-down", "from": "A", "to": "B"},
            {"time": 2, "action": "link-up", "from": "B", "to": "A"},
        ]
        for name, entry, wrap in REPLAY_ENTRIES:
            with self.subTest(entry=name):
                result = entry(CHAIN3, wrap(events))
                final = result["timeline"][-1]
                if name == "replay":
                    self.assertEqual(
                        final["routers"], result["timeline"][0]["routers"]
                    )
                else:
                    self.assertEqual(
                        final["rounds"][-1]["routers"],
                        result["timeline"][0]["rounds"][-1]["routers"],
                    )

    def test_illegal_transitions_still_cross_the_single_boundary(self):
        bad_sequences = [
            [{"time": 1, "action": "node-up", "node": "A"}],
            [{"time": 1, "action": "link-up", "from": "A", "to": "B"}],
            [
                {"time": 1, "action": "node-down", "node": "A"},
                {"time": 2, "action": "node-down", "node": "A"},
            ],
            [
                {"time": 1, "action": "link-down", "from": "A", "to": "B"},
                {"time": 2, "action": "link-down", "from": "B", "to": "A"},
            ],
        ]
        for name, entry, wrap in REPLAY_ENTRIES:
            for index, events in enumerate(bad_sequences):
                with self.subTest(entry=name, case=index):
                    with self.assertRaises(InvalidStateTransition):
                        entry(CHAIN3, wrap(events))


# ---------------------------------------------------------------------------
# Protocol state inheritance within one call
# ---------------------------------------------------------------------------


def _ls_sequences(timeline_entry):
    """The highest sequence each router holds for its own LSA at round 0."""
    view = {}
    databases = timeline_entry["rounds"][0]["databases"]
    for router, database in databases.items():
        if database is not None and router in database:
            view[router] = database[router]["sequence"]
    return view


class TestLinkStateInheritance(unittest.TestCase):
    def test_sequence_history_continues_across_events_in_one_call(self):
        timeline = replay_ls_scenario(CHAIN3, NODE_CYCLE)["timeline"]
        # Baseline: sequence 1 everywhere.
        self.assertEqual(
            _ls_sequences(timeline[0]), {"A": 1, "B": 1, "C": 1}
        )
        # First down: B goes away; A and C bump to 2 (adjacency changed).
        self.assertEqual(_ls_sequences(timeline[1]), {"A": 2, "C": 2})
        # B recovers and continues its own history at 2; neighbors bump to 3.
        self.assertEqual(
            _ls_sequences(timeline[2]), {"A": 3, "B": 2, "C": 3}
        )
        # Second down/up keeps incrementing from the carried history.
        self.assertEqual(_ls_sequences(timeline[3]), {"A": 4, "C": 4})
        self.assertEqual(
            _ls_sequences(timeline[4]), {"A": 5, "B": 3, "C": 5}
        )

    def test_converged_databases_carry_between_entries(self):
        # A link-down at time 1 is flooded; by the last round of entry 1 all
        # databases agree on the new LSAs.  Entry 2's round zero must start
        # from that converged view (C already holds B's bumped LSA), proving
        # inheritance rather than re-discovery.
        events = [
            {"time": 1, "action": "link-down", "from": "A", "to": "B"},
            {"time": 2, "action": "link-up", "from": "A", "to": "B"},
        ]
        timeline = replay_ls_scenario(CHAIN3, {"events": events})["timeline"]
        converged = timeline[1]["rounds"][-1]["databases"]
        inherited = timeline[2]["rounds"][0]["databases"]
        for router in ("A", "B", "C"):
            for originator, lsa in converged[router].items():
                # Only re-originating endpoints (A, B) jump a sequence at
                # entry 2; every other held LSA is inherited verbatim.
                if originator in ("A", "B"):
                    continue
                self.assertEqual(inherited[router][originator], lsa)


class TestDistanceVectorInheritance(unittest.TestCase):
    def test_converged_vectors_seed_the_next_event_round_zero(self):
        # After the first event every entry converges; the second event's
        # round zero is built from the first entry's converged vectors, so a
        # recovered direct link is usable at once and convergence is exact.
        events = [
            {"time": 1, "action": "node-down", "node": "C"},
            {"time": 2, "action": "node-up", "node": "C"},
        ]
        scenario = {"infinityMetric": 50, "events": events}
        timeline = replay_dv_scenario(CHAIN3, scenario)["timeline"]
        recovery_zero = timeline[2]["rounds"][0]["routers"]
        self.assertEqual(
            recovery_zero["C"]["B"], {"nextHop": "B", "metric": 7}
        )
        final = timeline[2]["rounds"][-1]["routers"]
        self.assertEqual(
            final, timeline[0]["rounds"][-1]["routers"]
        )

    def test_holddown_timers_never_carry_into_the_next_event(self):
        scenario = {
            "infinityMetric": 8,
            "holdDownRounds": 3,
            "events": [
                {"time": 1, "action": "link-down", "from": "A", "to": "B"},
                {"time": 2, "action": "node-down", "node": "C"},
                {"time": 3, "action": "node-up", "node": "C"},
                {"time": 4, "action": "link-up", "from": "B", "to": "A"},
            ],
        }
        timeline = replay_dv_scenario(DIAMOND, scenario)["timeline"]
        # Every entry runs to convergence, which requires all live timers to
        # have expired before the next event's round zero is built.
        for item in timeline[1:]:
            last_holddowns = item["rounds"][-1]["holdDowns"]
            for mapping in last_holddowns.values():
                self.assertEqual(mapping, {})
        # Final state is the fault-free shortest-path baseline.
        self.assertEqual(
            timeline[-1]["rounds"][-1]["routers"],
            timeline[0]["rounds"][-1]["routers"],
        )

    def test_holddown_countdown_is_published_round_by_round(self):
        scenario = {
            "infinityMetric": 8,
            "holdDownRounds": 2,
            "events": [{"time": 1, "action": "node-down", "node": "C"}],
        }
        rounds = replay_dv_scenario(
            {"nodes": ["A", "B", "C"],
             "links": [link("A", "B", 1), link("B", "C", 1)]},
            scenario,
        )["timeline"][1]["rounds"]
        self.assertEqual(rounds[0]["holdDowns"]["B"], {"C": 2})
        self.assertEqual(rounds[1]["holdDowns"]["B"], {"C": 1})
        self.assertEqual(rounds[2]["holdDowns"]["B"], {})
        self.assertEqual(rounds[-1]["round"], 2)


# ---------------------------------------------------------------------------
# No leakage between calls and no shared mutable data
# ---------------------------------------------------------------------------


class TestNoLeakageBetweenCalls(unittest.TestCase):
    FAULT_EVENTS = [
        {"time": 1, "action": "link-down", "from": "A", "to": "B"},
        {"time": 2, "action": "node-down", "node": "B"},
    ]

    def _fault_scenario(self, name):
        if name == "replay-dv":
            return {"infinityMetric": 50, "events": self.FAULT_EVENTS}
        return {"events": self.FAULT_EVENTS}

    def _empty_scenario(self, name):
        if name == "replay-dv":
            return {"infinityMetric": 50, "events": []}
        return {"events": []}

    def test_repeated_interleaved_calls_all_start_fault_free(self):
        # Interleave faulted and empty replays across all three entries: the
        # empty baseline must never see a failure from a neighboring call.
        seen_baselines = {}
        for _ in range(3):
            for name, entry, _wrap in REPLAY_ENTRIES:
                entry(CHAIN3, self._fault_scenario(name))
                fresh = entry(CHAIN3, self._empty_scenario(name))
                baseline = fresh["timeline"][0]
                if name == "replay":
                    key = json.dumps(baseline, sort_keys=True)
                else:
                    key = json.dumps(baseline["rounds"][-1], sort_keys=True)
                seen_baselines.setdefault(name, key)
                self.assertEqual(
                    json.dumps(baseline if name == "replay"
                               else baseline["rounds"][-1], sort_keys=True),
                    seen_baselines[name],
                )

    def test_identical_inputs_produce_equal_outputs_every_time(self):
        for name, entry, _wrap in REPLAY_ENTRIES:
            with self.subTest(entry=name):
                first = entry(CHAIN3, self._fault_scenario(name))
                second = entry(
                    copy.deepcopy(CHAIN3),
                    copy.deepcopy(self._fault_scenario(name)),
                )
                self.assertEqual(first, second)

    def test_mutating_one_result_cannot_reach_another(self):
        for name, entry, _wrap in REPLAY_ENTRIES:
            with self.subTest(entry=name):
                first = entry(CHAIN3, self._fault_scenario(name))
                # Corrupt every mutable surface the engine hands back.
                first["timeline"][1]["event"]["node"] = "HACK"
                if name == "replay":
                    first["timeline"][1]["routers"]["A"] = "HACK"
                else:
                    snapshot = first["timeline"][1]["rounds"][0]
                    if name == "replay-ls":
                        snapshot["databases"]["A"] = "HACK"
                    else:
                        snapshot["routers"]["A"] = "HACK"
                        if "holdDowns" in snapshot:
                            snapshot["holdDowns"]["A"]["Z"] = 99
                second = entry(
                    copy.deepcopy(CHAIN3),
                    copy.deepcopy(self._fault_scenario(name)),
                )
                target = second["timeline"][1]
                self.assertNotEqual(target["event"].get("node"), "HACK")
                self.assertNotEqual(
                    target["routers"] if name == "replay"
                    else target["rounds"][0][
                        "databases" if name == "replay-ls" else "routers"
                    ]["A"],
                    "HACK",
                )

    def test_entries_do_not_alias_each_others_round_structures(self):
        for name in ("replay-ls", "replay-dv"):
            entry = replay_ls_scenario if name == "replay-ls" \
                else replay_dv_scenario
            with self.subTest(entry=name):
                scenario = (
                    {"infinityMetric": 50, "events": self.FAULT_EVENTS}
                    if name == "replay-dv"
                    else {"events": self.FAULT_EVENTS}
                )
                timeline = entry(CHAIN3, scenario)["timeline"]
                baseline_payload = (
                    timeline[0]["rounds"][-1]["routers"]
                )
                event_payload = timeline[1]["rounds"][0]["routers"]
                for router in baseline_payload:
                    if baseline_payload[router] is None:
                        continue
                    for destination in baseline_payload[router]:
                        before = baseline_payload[router][destination]
                        after = event_payload[router]
                        if after is not None and destination in after:
                            self.assertIsNot(
                                after[destination], before,
                                (router, destination),
                            )

    def test_inputs_are_not_modified(self):
        for name, entry, _wrap in REPLAY_ENTRIES:
            with self.subTest(entry=name):
                topology = copy.deepcopy(CHAIN3)
                scenario = self._fault_scenario(name)
                topology_before = copy.deepcopy(topology)
                scenario_before = copy.deepcopy(scenario)
                entry(topology, scenario)
                self.assertEqual(topology, topology_before)
                self.assertEqual(scenario, scenario_before)


# ---------------------------------------------------------------------------
# Declaration-order and hash-seed stability
# ---------------------------------------------------------------------------


CHAIN3_VARIANTS = [
    CHAIN3,
    {"nodes": ["C", "B", "A"], "links": CHAIN3["links"]},
    {"nodes": ["A", "B", "C"],
     "links": [link("B", "A", 5), link("C", "B", 7)]},
    {"nodes": ["A", "B", "C"],
     "links": [link("C", "B", 7), link("B", "A", 5)]},
    {"nodes": ["B", "A", "C"],
     "links": [link("A", "B", 5), link("C", "B", 7)]},
]


class TestDeclarationOrderStability(unittest.TestCase):
    EVENTS = [
        {"time": 3, "action": "link-down", "from": "A", "to": "B"},
        {"time": 5, "action": "node-down", "node": "B"},
        {"time": 9, "action": "node-up", "node": "B"},
        {"time": 12, "action": "link-up", "from": "B", "to": "A"},
    ]

    def test_all_three_entries_are_equal_across_declaration_orders(self):
        for name, entry, wrap in REPLAY_ENTRIES:
            with self.subTest(entry=name):
                reference = json.dumps(
                    entry(CHAIN3_VARIANTS[0], wrap(self.EVENTS)),
                    sort_keys=True,
                )
                for variant in CHAIN3_VARIANTS[1:]:
                    self.assertEqual(
                        json.dumps(
                            entry(variant, wrap(self.EVENTS)),
                            sort_keys=True,
                        ),
                        reference,
                    )

    def test_holddown_output_is_order_independent(self):
        def wrap(events):
            return {"infinityMetric": 50, "holdDownRounds": 2,
                    "events": events}

        reference = json.dumps(
            replay_dv_scenario(CHAIN3_VARIANTS[0], wrap(self.EVENTS)),
            sort_keys=True,
        )
        for variant in CHAIN3_VARIANTS[1:]:
            self.assertEqual(
                json.dumps(
                    replay_dv_scenario(variant, wrap(self.EVENTS)),
                    sort_keys=True,
                ),
                reference,
            )


_DETERMINISM_SNIPPET = """
import json, os, sys
sys.path.insert(0, %r)
from packet_routing_sim.core import (
    replay_scenario, replay_ls_scenario, replay_dv_scenario)
topology = json.loads(os.environ["PRSIM_TOPO"])
scenario = json.loads(os.environ["PRSIM_SCEN"])
out = {
    "replay": replay_scenario(topology, {"events": scenario["events"]}),
    "replay_ls": replay_ls_scenario(topology, {"events": scenario["events"]}),
    "replay_dv": replay_dv_scenario(topology, scenario),
}
sys.stdout.write(json.dumps(out, sort_keys=True))
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


class TestHashSeedStability(unittest.TestCase):
    SCENARIO = {
        "infinityMetric": 50,
        "holdDownRounds": 2,
        "events": [
            {"time": 3, "action": "link-down", "from": "A", "to": "B"},
            {"time": 5, "action": "node-down", "node": "B"},
            {"time": 9, "action": "node-up", "node": "B"},
            {"time": 12, "action": "link-up", "from": "B", "to": "A"},
        ],
    }

    def test_byte_identical_across_seeds_and_orders(self):
        outputs = set()
        for seed in HASH_SEEDS:
            for variant in CHAIN3_VARIANTS:
                outputs.add(run_in_subprocess(variant, self.SCENARIO, seed))
        self.assertEqual(len(outputs), 1)


# ---------------------------------------------------------------------------
# The one error boundary, auditable through every entry point
# ---------------------------------------------------------------------------


class TestErrorBoundary(unittest.TestCase):
    def test_topology_structure_errors_raise_invalid_topology(self):
        bad_topology = {"nodes": ["A", "A"], "links": []}
        for name, entry, wrap in REPLAY_ENTRIES:
            with self.subTest(entry=name):
                with self.assertRaises(InvalidTopology):
                    entry(bad_topology, wrap([]))

    def test_scenario_structure_errors_raise_invalid_scenario(self):
        bad_scenarios = [
            {"events": "nope"},
            {"events": [{"time": 0, "action": "node-down", "node": "A"}]},
            {"events": [{"time": 1, "action": "explode", "node": "A"}]},
            {"events": [{"time": 1, "action": "node-down", "node": "Z"}]},
        ]
        for name, entry, wrap in REPLAY_ENTRIES:
            for index, raw in enumerate(bad_scenarios):
                with self.subTest(entry=name, case=index):
                    # Reuse the scenario as given (plain form); replay-dv
                    # additionally requires a valid infinityMetric.
                    scenario = raw
                    if name == "replay-dv":
                        scenario = copy.deepcopy(raw)
                        scenario["infinityMetric"] = 50
                    with self.assertRaises(InvalidScenario):
                        entry(CHAIN3, scenario)

    def test_dv_requires_infinity_metric(self):
        with self.assertRaises(InvalidScenario):
            replay_dv_scenario(CHAIN3, {"events": []})
        # replay and replay-ls must not care about it at all.
        replay_scenario(CHAIN3, {"infinityMetric": 1, "events": []})
        replay_ls_scenario(CHAIN3, {"infinityMetric": 1, "events": []})


if __name__ == "__main__":
    unittest.main()
