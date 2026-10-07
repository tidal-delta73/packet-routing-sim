"""Regression tests for the unified replay event-advancement boundary.

These tests pin the behavior that the behavior-preserving replay refactor
centralized in :mod:`packet_routing_sim.core.replay` -- one timeline engine
(:func:`packet_routing_sim.core.replay.run_timeline`) driven by three
protocol drivers:

* the empty-event baseline shape and its position for every entry point;
* consecutive node/link failures and recoveries folding through the single
  event boundary, including explicit link-down surviving a node restart and
  reversed endpoint identification;
* per-protocol state inheritance between entries (link-state databases and
  per-node sequence history; distance-vector converged vectors and
  hold-down timer expiry) without state leaking between calls;
* byte-identical output under node/link declaration-order changes and
  different hash seeds.

Routing values themselves are covered differentially in ``test_core.py`` and
``test_cli.py``; this module focuses on the shared boundary and its drivers.
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
from packet_routing_sim.core.replay import (
    DistanceVectorProtocol,
    LinkStateProtocol,
    ReplayProtocol,
    StaticLinkStateProtocol,
    run_timeline,
)
from packet_routing_sim.core.topology import validate_topology

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
RING = {
    "nodes": ["A", "B", "C", "D"],
    "links": [
        link("A", "B", 1),
        link("B", "C", 2),
        link("C", "D", 3),
        link("D", "A", 4),
    ],
}

CYCLE_EVENTS = [
    {"time": 1, "action": "link-down", "from": "A", "to": "B"},
    {"time": 2, "action": "node-down", "node": "B"},
    {"time": 3, "action": "node-up", "node": "B"},
    {"time": 4, "action": "link-up", "from": "B", "to": "A"},
]
# A scenario accepted by all three entries: replay/replay-ls simply ignore
# the distance-vector-only fields.
DV_OPTION_SCENARIO = {
    "infinityMetric": 30,
    "holdDownRounds": 3,
    "events": list(CYCLE_EVENTS),
}


def _db_view(round_snapshot):
    """{router: {originator: (sequence, neighbors)}} for one LS round."""
    return {
        router: None
        if database is None
        else {
            originator: (lsa["sequence"], tuple(lsa["neighbors"]))
            for originator, lsa in database.items()
        }
        for router, database in round_snapshot["databases"].items()
    }


# ---------------------------------------------------------------------------
# The shared boundary: baseline position, entry assembly and event echo
# ---------------------------------------------------------------------------


class TestUnifiedBoundaryShape(unittest.TestCase):
    ENTRIES = (
        ("replay", replay_scenario),
        ("replay-ls", replay_ls_scenario),
        ("replay-dv", replay_dv_scenario),
    )

    def test_empty_events_emit_only_the_fault_free_baseline(self):
        for name, entry in self.ENTRIES:
            with self.subTest(entry=name):
                result = entry(CHAIN3, copy.deepcopy(DV_OPTION_SCENARIO) | {"events": []})
                timeline = result["timeline"]
                self.assertEqual(len(timeline), 1)
                self.assertIsNone(timeline[0]["event"])
                self.assertNotIn("time", timeline[0])

    def test_baseline_is_always_position_zero_and_events_follow_in_order(self):
        events = [
            {"time": 3, "action": "link-down", "from": "A", "to": "B"},
            {"time": 9, "action": "node-down", "node": "C"},
        ]
        for name, entry in self.ENTRIES:
            with self.subTest(entry=name):
                scenario = (
                    {"events": events}
                    if name != "replay-dv"
                    else {"infinityMetric": 30, "events": events}
                )
                timeline = entry(CHAIN3, scenario)["timeline"]
                self.assertEqual(len(timeline), 3)
                self.assertEqual(
                    set(timeline[0]),
                    (
                        {"event", "routers"}
                        if name == "replay"
                        else {"event", "convergenceRound", "rounds"}
                    ),
                )
                for index, event in enumerate(events, start=1):
                    # Key insertion order is itself part of the wire format.
                    self.assertEqual(
                        list(timeline[index])[:2], ["time", "event"], name
                    )
                    self.assertEqual(timeline[index]["time"], event["time"])
                    self.assertEqual(timeline[index]["event"], event)
                    self.assertIsNot(timeline[index]["event"], event)

    def test_published_key_order_is_stable(self):
        scenario = copy.deepcopy(DV_OPTION_SCENARIO)
        # Static entry.
        static = replay_scenario(RING, scenario)
        self.assertEqual(list(static), ["protocol", "timeline"])
        self.assertEqual(list(static["timeline"][1]), ["time", "event", "routers"])
        # Auditable link-state entry.
        ls_result = replay_ls_scenario(RING, scenario)
        self.assertEqual(list(ls_result), ["protocol", "timeline"])
        self.assertEqual(
            list(ls_result["timeline"][1]),
            ["time", "event", "convergenceRound", "rounds"],
        )
        self.assertEqual(
            list(ls_result["timeline"][1]["rounds"][0]),
            ["round", "databases", "routers"],
        )
        # Distance-vector entry with hold-down fields in their old positions.
        dv = replay_dv_scenario(RING, scenario)
        self.assertEqual(
            list(dv),
            ["protocol", "infinityMetric", "holdDownRounds", "timeline"],
        )
        self.assertEqual(
            list(dv["timeline"][1]),
            ["time", "event", "convergenceRound", "rounds"],
        )
        self.assertEqual(
            list(dv["timeline"][1]["rounds"][0]),
            ["round", "routers", "holdDowns"],
        )
        legacy = replay_dv_scenario(
            RING, {"infinityMetric": 30, "events": scenario["events"]}
        )
        self.assertEqual(list(legacy), ["protocol", "infinityMetric", "timeline"])
        self.assertEqual(
            list(legacy["timeline"][1]["rounds"][0]), ["round", "routers"]
        )

    def test_engine_and_three_drivers_are_distinct_boundaries(self):
        # Every replay builds through a fresh driver instance exposing the
        # shared boundary; the abstract boundary itself stays unimplemented.
        topo = validate_topology(CHAIN3)
        self.assertIsInstance(StaticLinkStateProtocol(), ReplayProtocol)
        self.assertIsInstance(LinkStateProtocol(), ReplayProtocol)
        self.assertIsInstance(DistanceVectorProtocol(), ReplayProtocol)
        with self.assertRaises(NotImplementedError):
            run_timeline(topo, ReplayProtocol(), {"events": []})

    def test_one_driver_drives_the_engine_for_all_three_protocols(self):
        # A recording driver proves the engine calls baseline once and
        # advance once per event, in time order, from a fault-free state.
        topo = validate_topology(CHAIN3)

        class RecordingProtocol(ReplayProtocol):
            def __init__(self):
                self.calls = []

            def validate(self, topo, scenario):
                from packet_routing_sim.core.scenario import validate_scenario

                return validate_scenario(topo, copy.deepcopy(scenario))

            def baseline(self, state):
                self.calls.append(("baseline", state.down_nodes, state.down_links))
                return {"mark": "base"}, "seed"

            def advance(self, event, previous_state, state, inherited):
                self.calls.append(
                    (event.action, state.down_nodes, state.down_links)
                )
                self.inherited_seen = inherited
                return {"mark": event.time}, f"after-{event.time}"

            def result(self, timeline):
                return {"timeline": timeline}

        driver = RecordingProtocol()
        events = [
            {"time": 1, "action": "node-down", "node": "B"},
            {"time": 2, "action": "node-up", "node": "B"},
        ]
        result = run_timeline(topo, driver, {"events": events})
        self.assertEqual(
            driver.calls,
            [
                ("baseline", frozenset(), frozenset()),
                ("node-down", frozenset({"B"}), frozenset()),
                ("node-up", frozenset(), frozenset()),
            ],
        )
        # Inheritance is threaded through the engine untouched between steps.
        self.assertEqual(driver.inherited_seen, "after-1")
        self.assertEqual(
            [entry.get("mark") for entry in result["timeline"]],
            ["base", 1, 2],
        )
        self.assertIsNone(result["timeline"][0]["event"])


# ---------------------------------------------------------------------------
# Consecutive failures/recoveries through the single event boundary
# ---------------------------------------------------------------------------


class TestEventAdvancement(unittest.TestCase):
    def test_full_cycle_ends_back_at_the_fault_free_baseline(self):
        for name, entry, scenario in (
            ("replay", replay_scenario, {"events": CYCLE_EVENTS}),
            ("replay-ls", replay_ls_scenario, {"events": CYCLE_EVENTS}),
            (
                "replay-dv",
                replay_dv_scenario,
                {"infinityMetric": 30, "events": CYCLE_EVENTS},
            ),
        ):
            with self.subTest(entry=name):
                timeline = entry(CHAIN3, scenario)["timeline"]
                self.assertEqual(len(timeline), 5)
                final = (
                    timeline[-1]["routers"]
                    if name == "replay"
                    else timeline[-1]["rounds"][-1]["routers"]
                )
                self.assertEqual(
                    final, replay_scenario(CHAIN3, {"events": []})["timeline"][0]["routers"]
                )

    def test_explicit_link_down_stays_disabled_across_node_recovery(self):
        scenario = {
            "events": [
                {"time": 1, "action": "link-down", "from": "A", "to": "B"},
                {"time": 2, "action": "node-down", "node": "B"},
                {"time": 3, "action": "node-up", "node": "B"},
            ]
        }
        null = {"nextHop": None, "metric": None}
        static = replay_scenario(CHAIN3, scenario)["timeline"][-1]["routers"]
        self.assertEqual(static["A"]["B"], null)
        self.assertEqual(static["B"]["C"], {"nextHop": "C", "metric": 7})

        dv = replay_dv_scenario(
            CHAIN3, {"infinityMetric": 9, **scenario}
        )["timeline"][-1]["rounds"][-1]["routers"]
        self.assertEqual(dv["A"]["B"], null)
        self.assertEqual(dv["B"]["C"], {"nextHop": "C", "metric": 7})

        ls_final = replay_ls_scenario(CHAIN3, scenario)["timeline"][-1]["rounds"][-1]
        self.assertEqual(ls_final["routers"]["A"]["B"], null)
        self.assertEqual(ls_final["routers"]["B"]["C"], {"nextHop": "C", "metric": 7})

    def test_reversed_endpoints_identify_one_link_in_every_protocol(self):
        scenario = {
            "events": [
                {"time": 1, "action": "link-down", "from": "A", "to": "B"},
                {"time": 2, "action": "link-up", "from": "B", "to": "A"},
            ]
        }
        baseline = replay_scenario(CHAIN3, {"events": []})["timeline"][0]["routers"]
        self.assertEqual(
            replay_scenario(CHAIN3, scenario)["timeline"][-1]["routers"], baseline
        )
        self.assertEqual(
            replay_ls_scenario(CHAIN3, scenario)["timeline"][-1]["rounds"][-1]["routers"],
            baseline,
        )
        dv = replay_dv_scenario(CHAIN3, {"infinityMetric": 50, **scenario})
        self.assertEqual(
            dv["timeline"][-1]["rounds"][-1]["routers"], baseline
        )

    def test_repeated_down_up_cycles_advance_every_time(self):
        scenario = {
            "events": [
                {"time": 1, "action": "node-down", "node": "B"},
                {"time": 2, "action": "node-up", "node": "B"},
                {"time": 3, "action": "node-down", "node": "B"},
                {"time": 4, "action": "node-up", "node": "B"},
            ]
        }
        timeline = replay_ls_scenario(CHAIN3, scenario)["timeline"]
        self.assertEqual(len(timeline), 5)
        # Consecutive recoveries must keep advancing from inherited history.
        self.assertEqual(
            _db_view(timeline[2]["rounds"][0])["B"]["B"], (2, ("A", "C"))
        )
        self.assertEqual(
            _db_view(timeline[4]["rounds"][0])["B"]["B"], (3, ("A", "C"))
        )
        # A second identical cycle is legal state progression and echoes back.
        dv = replay_dv_scenario(DV_CHAIN, {"infinityMetric": 50, **scenario})
        self.assertEqual(
            [entry["event"]["time"] for entry in dv["timeline"][1:]],
            [1, 2, 3, 4],
        )


# ---------------------------------------------------------------------------
# Protocol state inheritance between timeline entries
# ---------------------------------------------------------------------------


class TestLinkStateInheritance(unittest.TestCase):
    def test_databases_and_sequence_history_carry_between_entries(self):
        scenario = {
            "events": [
                {"time": 1, "action": "node-down", "node": "B"},
                {"time": 2, "action": "node-up", "node": "B"},
            ]
        }
        recovery_round_zero = replay_ls_scenario(
            CHAIN3, scenario
        )["timeline"][2]["rounds"][0]
        view = _db_view(recovery_round_zero)
        # The recovered node keeps no database but its own new LSA continues
        # the historical sequence (2, not 1)...
        self.assertEqual(view["B"], {"B": (2, ("A", "C"))})
        # ...while neighbors inherited the converged pre-failure database, so
        # A still holds B's stale sequence-1 LSA at round zero and routes on.
        self.assertEqual(view["A"]["B"], (1, ("A", "C")))
        self.assertEqual(
            recovery_round_zero["routers"]["A"]["C"],
            {"nextHop": "B", "metric": 12},
        )

    def test_link_event_bumps_only_changed_endpoints_from_inherited_state(self):
        scenario = {
            "events": [
                {"time": 1, "action": "link-down", "from": "A", "to": "B"}
            ]
        }
        entry = replay_ls_scenario(CHAIN3, scenario)["timeline"][1]
        round_zero = entry["rounds"][0]
        view = _db_view(round_zero)
        self.assertEqual(view["A"]["A"], (2, ()))
        self.assertEqual(view["B"]["B"], (2, ("C",)))
        # C inherited its old database and sequence untouched.
        self.assertEqual(view["C"]["C"], (1, ("B",)))
        self.assertEqual(view["C"]["B"], (1, ("A", "C")))


class TestDistanceVectorInheritance(unittest.TestCase):
    def test_converged_vectors_carry_into_the_next_event_round_zero(self):
        scenario = {
            "infinityMetric": 8,
            "events": [
                {"time": 1, "action": "node-down", "node": "C"},
                {"time": 2, "action": "node-up", "node": "C"},
            ],
        }
        timeline = replay_dv_scenario(DV_CHAIN, scenario)["timeline"]
        # The failure entry counts to infinity while inheriting per event.
        self.assertEqual(timeline[1]["convergenceRound"], 6)
        # At recovery A starts from its inherited *unreachable* route, so C
        # comes back one synchronous hop later than B's direct adoption.
        recovery = timeline[2]
        round_zero = recovery["rounds"][0]["routers"]
        self.assertEqual(
            round_zero["B"]["C"], {"nextHop": "C", "metric": 1}
        )
        self.assertEqual(
            round_zero["A"]["C"], {"nextHop": None, "metric": None}
        )
        self.assertEqual(recovery["convergenceRound"], 1)
        self.assertEqual(
            recovery["rounds"][1]["routers"]["A"]["C"],
            {"nextHop": "B", "metric": 2},
        )

    def test_holddown_timers_expire_between_events_and_never_leak(self):
        scenario = {
            "infinityMetric": 8,
            "holdDownRounds": 2,
            "events": [
                {"time": 1, "action": "link-down", "from": "A", "to": "B"},
                {"time": 5, "action": "link-up", "from": "B", "to": "A"},
            ],
        }
        timeline = replay_dv_scenario(DIAMOND, scenario)["timeline"]
        for index, entry in enumerate(timeline[1:], start=1):
            last_maps = entry["rounds"][-1]["holdDowns"]
            self.assertEqual(
                last_maps,
                {"A": {}, "B": {}, "C": {}, "D": {}},
                f"timers leaked out of timeline entry {index}",
            )
        # The failure entry really used suppression: adopted only after the
        # window; the recovery entry converges straight back to shortest
        # paths from the inherited converged vectors.
        failure_rounds = timeline[1]["rounds"]
        self.assertEqual(failure_rounds[0]["holdDowns"]["A"]["D"], 2)
        self.assertEqual(failure_rounds[1]["holdDowns"]["A"]["D"], 1)
        self.assertEqual(failure_rounds[2]["holdDowns"]["A"], {})
        final = timeline[2]["rounds"][-1]["routers"]
        self.assertEqual(
            final, replay_scenario(DIAMOND, {"events": []})["timeline"][0]["routers"]
        )


# ---------------------------------------------------------------------------
# No state leakage across calls; input and output isolation
# ---------------------------------------------------------------------------


class TestNoStatelessLeakage(unittest.TestCase):
    ENTRIES = (
        ("replay", replay_scenario),
        ("replay-ls", replay_ls_scenario),
        ("replay-dv", replay_dv_scenario),
    )

    def _scenario(self, name):
        events = [
            {"time": 1, "action": "link-down", "from": "A", "to": "B"},
            {"time": 2, "action": "node-down", "node": "B"},
            {"time": 3, "action": "node-up", "node": "B"},
            {"time": 4, "action": "link-up", "from": "B", "to": "A"},
        ]
        if name == "replay-dv":
            return {"infinityMetric": 30, "holdDownRounds": 2, "events": events}
        return {"events": events}

    def test_repeated_calls_with_and_without_events_are_stateless(self):
        for name, entry in self.ENTRIES:
            with self.subTest(entry=name):
                topology = DV_CHAIN if name == "replay-dv" else CHAIN3
                fault_events = [
                    {"time": 1, "action": "node-down", "node": "B"}
                ]
                fault = (
                    {"infinityMetric": 8, "events": fault_events}
                    if name == "replay-dv"
                    else {"events": fault_events}
                )
                first = entry(topology, copy.deepcopy(fault))
                second = entry(topology, copy.deepcopy(fault))
                self.assertEqual(first, second)
                # A no-event replay afterwards starts fault-free: the fault
                # did not leak into this call's baseline.
                empty = (
                    {"infinityMetric": 8, "events": []}
                    if name == "replay-dv"
                    else {"events": []}
                )
                baseline_only = entry(topology, empty)["timeline"]
                self.assertEqual(len(baseline_only), 1)
                published = (
                    baseline_only[0]["routers"]
                    if name == "replay"
                    else baseline_only[0]["rounds"][-1]["routers"]
                )
                if name == "replay-ls":
                    # A fresh call's LS databases restart at sequence 1.
                    for database in baseline_only[0]["rounds"][-1][
                        "databases"
                    ].values():
                        for lsa in database.values():
                            self.assertEqual(lsa["sequence"], 1)
                # A second interleaved empty call produces the same baseline.
                again = entry(topology, copy.deepcopy(empty))["timeline"]
                published_again = (
                    again[0]["routers"]
                    if name == "replay"
                    else again[0]["rounds"][-1]["routers"]
                )
                self.assertEqual(published_again, published)
                self.assertEqual(entry(topology, copy.deepcopy(fault)), first)

    def test_inputs_are_not_modified(self):
        for name, entry in self.ENTRIES:
            with self.subTest(entry=name):
                topology = copy.deepcopy(DIAMOND)
                scenario = self._scenario(name)
                topology_before = copy.deepcopy(topology)
                scenario_before = copy.deepcopy(scenario)
                entry(topology, scenario)
                self.assertEqual(topology, topology_before)
                self.assertEqual(scenario, scenario_before)

    def test_mutating_one_result_cannot_reach_another(self):
        scenario = {"events": [{"time": 1, "action": "node-down", "node": "B"}]}
        first = replay_ls_scenario(CHAIN3, scenario)
        first["timeline"][1]["rounds"][0]["databases"]["A"]["A"] = {
            "sequence": 999,
            "neighbors": ["HACK"],
        }
        second = replay_ls_scenario(CHAIN3, copy.deepcopy(scenario))
        self.assertEqual(
            _db_view(second["timeline"][1]["rounds"][0])["A"]["A"], (2, ())
        )

        dv_scenario = {
            "infinityMetric": 8,
            "holdDownRounds": 2,
            "events": [{"time": 1, "action": "node-down", "node": "C"}],
        }
        first_dv = replay_dv_scenario(DV_CHAIN, dv_scenario)
        first_dv["timeline"][1]["rounds"][0]["holdDowns"]["A"]["C"] = 999
        first_dv["timeline"][1]["rounds"][0]["routers"]["A"]["C"] = {
            "nextHop": "HACK",
            "metric": 0,
        }
        second_dv = replay_dv_scenario(DV_CHAIN, copy.deepcopy(dv_scenario))
        self.assertEqual(
            second_dv["timeline"][1]["rounds"][0]["holdDowns"]["A"]["C"], 2
        )
        self.assertEqual(
            second_dv["timeline"][1]["rounds"][0]["routers"]["A"]["C"]["metric"],
            None,
        )


# ---------------------------------------------------------------------------
# Unique error boundary through the shared engine
# ---------------------------------------------------------------------------


class TestReplayBoundaryErrors(unittest.TestCase):
    def test_invalid_topology_still_raises_invalid_topology(self):
        bad_topology = {"nodes": ["A", "A"], "links": []}
        for entry in (replay_scenario, replay_ls_scenario, replay_dv_scenario):
            with self.subTest(entry=entry.__name__):
                with self.assertRaises(InvalidTopology):
                    entry(bad_topology, DV_OPTION_SCENARIO)

    def test_invalid_scenario_structure_still_raises_invalid_scenario(self):
        for entry, scenario in (
            (replay_scenario, {"events": [{"time": 1, "action": "explode"}]}),
            (replay_ls_scenario, {"events": [{"time": 0, "action": "node-down", "node": "A"}]}),
            (replay_dv_scenario, {"events": []}),
            (replay_dv_scenario, {"infinityMetric": 1, "events": []}),
        ):
            with self.subTest(entry=entry.__name__):
                with self.assertRaises(InvalidScenario):
                    entry(CHAIN3, scenario)

    def test_illegal_transitions_still_raise_the_distinguishable_subtype(self):
        cases = [
            {"events": [{"time": 1, "action": "node-up", "node": "A"}]},
            {"events": [
                {"time": 1, "action": "node-down", "node": "A"},
                {"time": 2, "action": "node-down", "node": "A"},
            ]},
            {"events": [{"time": 1, "action": "link-up", "from": "A", "to": "B"}]},
            {"events": [
                {"time": 1, "action": "link-down", "from": "A", "to": "B"},
                {"time": 2, "action": "link-down", "from": "B", "to": "A"},
            ]},
        ]
        for entry in (replay_scenario, replay_ls_scenario):
            for scenario in cases:
                with self.subTest(entry=entry.__name__, scenario=scenario):
                    with self.assertRaises(InvalidStateTransition):
                        entry(CHAIN3, scenario)
        with self.assertRaises(InvalidStateTransition):
            replay_dv_scenario(
                CHAIN3,
                {"infinityMetric": 30, "events": cases[0]["events"]},
            )

    def test_partial_validation_leaves_no_partial_state(self):
        # An illegal event late in the sequence raises and leaves subsequent
        # calls unaffected.
        scenario = {
            "events": [
                {"time": 1, "action": "node-down", "node": "B"},
                {"time": 2, "action": "node-down", "node": "B"},
            ]
        }
        with self.assertRaises(InvalidStateTransition):
            replay_ls_scenario(CHAIN3, scenario)
        self.assertEqual(
            replay_ls_scenario(CHAIN3, {"events": []}),
            replay_ls_scenario(CHAIN3, {"events": []}),
        )


# ---------------------------------------------------------------------------
# Byte stability across declaration order and hash seeds
# ---------------------------------------------------------------------------

_DETERMINISM_SNIPPET = """
import json, os, sys
sys.path.insert(0, %r)
from packet_routing_sim.core import (
    replay_scenario, replay_ls_scenario, replay_dv_scenario,
)
topology = json.loads(os.environ["PRSIM_TOPO"])
scenario = json.loads(os.environ["PRSIM_SCEN"])
out = {
    "replay": replay_scenario(topology, scenario),
    "replay_ls": replay_ls_scenario(topology, scenario),
    "replay_dv": replay_dv_scenario(topology, scenario),
}
sys.stdout.write(json.dumps(out, sort_keys=True))
""" % str(REPO_ROOT)


def run_boundary_in_subprocess(topology, scenario, seed):
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


class TestBoundaryDeterminism(unittest.TestCase):
    SCENARIO = {
        "infinityMetric": 30,
        "holdDownRounds": 3,
        "events": list(CYCLE_EVENTS),
    }
    VARIANTS = [
        RING,
        {"nodes": ["D", "C", "B", "A"], "links": RING["links"]},
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
            "links": list(reversed(RING["links"])),
        },
        {
            "nodes": ["D", "C", "B", "A"],
            "links": [
                link("A", "B", 1),
                link("B", "C", 2),
                link("C", "D", 3),
                link("D", "A", 4),
            ],
        },
    ]

    def test_byte_identical_across_seeds_and_declaration_order(self):
        outputs = set()
        for seed in HASH_SEEDS:
            for variant in self.VARIANTS:
                outputs.add(
                    run_boundary_in_subprocess(variant, self.SCENARIO, seed)
                )
        self.assertEqual(len(outputs), 1)

    def test_in_process_results_match_across_declaration_order(self):
        reference = None
        for variant in self.VARIANTS:
            packed = json.dumps(
                {
                    "replay": replay_scenario(variant, self.SCENARIO),
                    "replay_ls": replay_ls_scenario(variant, self.SCENARIO),
                    "replay_dv": replay_dv_scenario(variant, self.SCENARIO),
                },
                sort_keys=True,
            )
            if reference is None:
                reference = packed
            self.assertEqual(packed, reference)


if __name__ == "__main__":
    unittest.main()
