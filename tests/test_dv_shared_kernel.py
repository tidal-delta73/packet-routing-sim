"""Regression tests for the shared distance-vector round-selection kernel.

The distance-vector computation kernel is now shared by the three public DV
entry points:

* ``converge``                       - threshold-free cold-start convergence;
* ``replay-dv`` (normal mode)        - event inheritance plus the
  ``infinityMetric`` cutoff;
* ``replay-dv`` with positive
  ``holdDownRounds``                  - the same selection guarded by hold-down
  timers.

These tests pin the *shared per-round semantics* independently of the
implementation:

* candidate filtering (``null`` advertisements never qualify), the infinity
  cutoff (a candidate at or above the threshold is unreachable), the
  smaller-next-hop tie break, the always-zero self route and the entirely
  unreachable row of a down router;
* round zero starts from self plus direct links for ``converge`` and from the
  inherited converged vectors after an event for the replay modes;
* only rounds whose routing table actually changes are recorded (hold-down
  additionally records rounds in which only a timer changes), and the last
  recorded round is a fixed point;
* the three entries produce the same stable forwarding tables wherever their
  surrounding rules coincide (fault-free convergence, and -- since hold-down
  keeps current direct routes pinned to their direct metric -- any event
  sequence on a tree, where no indirect route can beat a direct one);
* hold-down invalidation closure, countdown, immediate direct-recovery
  adoption and joint vectors/timers stability;
* validation precedence/error distinctions, input immutability, result
  isolation and hash-seed/declaration-order independence.

Every expected trajectory below is produced by a small Bellman-Ford /
hold-down oracle written in this module, never by importing the kernel's
round functions, so a defect cannot corrupt the expectation and the result
under test at once.
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
    compute_topology,
    converge_topology,
    replay_dv_scenario,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
HASH_SEEDS = tuple(
    int(seed)
    for seed in os.environ.get("PRSIM_TEST_SEEDS", "0,1").split(",")
)


def link(source, target, metric):
    return {"from": source, "to": target, "metric": metric}


DV_CHAIN = {
    "nodes": ["A", "B", "C"],
    "links": [link("A", "B", 1), link("B", "C", 1)],
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
# A tree with a branching center; every surviving path after a failure is
# the unique direct path, so normal and hold-down convergence must agree on
# every entry of every legal event sequence.
BRANCH_TREE = {
    "nodes": ["A", "B", "C", "D", "E", "F"],
    "links": [
        link("C", "A", 2),
        link("C", "B", 4),
        link("C", "D", 1),
        link("D", "E", 3),
        link("D", "F", 6),
    ],
}
STAR = {
    "nodes": ["h", "l1", "l2", "l3"],
    "links": [
        link("h", "l1", 2),
        link("h", "l2", 5),
        link("h", "l3", 9),
    ],
}
TREES = [DV_CHAIN, CHAIN3, BRANCH_TREE, STAR]

# Cyclic topology containing a direct edge (A-D, 9) that is worse than the
# surviving indirect path after the A-B failure: used to pin the documented
# hold-down behavior of keeping a current direct neighbor pinned at the
# direct metric throughout convergence.
PINNED_DIRECT = {
    "nodes": ["A", "B", "C", "D"],
    "links": [
        link("A", "B", 1),
        link("A", "C", 5),
        link("A", "D", 9),
        link("B", "C", 1),
        link("B", "D", 1),
        link("C", "D", 1),
    ],
}

NULL = {"nextHop": None, "metric": None}


# ---------------------------------------------------------------------------
# Independent oracle
# ---------------------------------------------------------------------------


def normalized(routers):
    """Published router rows as ``{(r, d): (nextHop, metric)}`` tuples."""
    if routers is None:
        return None
    return {
        (router, dest): (entry["nextHop"], entry["metric"])
        for router, row in routers.items()
        for dest, entry in row.items()
    }


def oracle_normalized(vectors):
    """The oracle's nested rows as the same flat tuple map."""
    return {
        (router, dest): entry
        for router, row in vectors.items()
        for dest, entry in row.items()
    }


def active_map(topology, down_nodes, down_links):
    """Adjacency of usable links from an explicit failure state."""
    nodes = sorted(topology["nodes"])
    adjacency = {node: {} for node in nodes}
    for edge in topology["links"]:
        a, b, m = edge["from"], edge["to"], edge["metric"]
        if a in down_nodes or b in down_nodes:
            continue
        pair = tuple(sorted((a, b)))
        if pair in down_links:
            continue
        adjacency[a][b] = m
        adjacency[b][a] = m
    return adjacency


def cold_round_zero(nodes, adjacency):
    """The converge round zero: self plus direct links, nothing else."""
    vectors = {}
    for router in nodes:
        vectors[router] = {}
        for dest in nodes:
            if dest == router:
                vectors[router][dest] = (None, 0)
            elif dest in adjacency[router]:
                vectors[router][dest] = (dest, adjacency[router][dest])
            else:
                vectors[router][dest] = (None, None)
    return vectors


def failure_round_zero(previous, nodes, adjacency, down_nodes):
    """Plain post-event round zero inherited from prior converged vectors."""
    vectors = {}
    for router in nodes:
        if router in down_nodes:
            vectors[router] = {dest: (None, None) for dest in nodes}
            continue
        row = {}
        for dest in nodes:
            if dest == router:
                row[dest] = (None, 0)
            elif dest in adjacency[router]:
                row[dest] = (dest, adjacency[router][dest])
            else:
                hop, metric = previous[router][dest]
                if hop is None or hop not in adjacency[router]:
                    row[dest] = (None, None)
                else:
                    row[dest] = (hop, metric)
        vectors[router] = row
    return vectors


def best_route(router, dest, previous, adjacency, infinity_metric):
    """One shared Bellman-Ford selection, independent of the SUT kernel."""
    best_cost, best_hop = None, None
    for neighbor in sorted(adjacency[router]):
        advertised = previous[neighbor][dest][1]
        if advertised is None:
            continue
        cost = adjacency[router][neighbor] + advertised
        if infinity_metric is not None and cost >= infinity_metric:
            continue
        if best_cost is None or cost < best_cost:
            best_cost, best_hop = cost, neighbor
    return (None, None) if best_cost is None else (best_hop, best_cost)


def plain_round(previous, nodes, adjacency, down_nodes, infinity_metric):
    """One synchronous thresholded round (or one down-router row set)."""
    current = {}
    for router in nodes:
        if router in down_nodes:
            current[router] = {dest: (None, None) for dest in nodes}
            continue
        row = {}
        for dest in nodes:
            if dest == router:
                row[dest] = (None, 0)
            else:
                row[dest] = best_route(
                    router, dest, previous, adjacency, infinity_metric
                )
        current[router] = row
    return current


def run_to_fixed(round_zero, nodes, adjacency, down_nodes, infinity_metric):
    """Record round zero and every changed thresholded round."""
    rounds = [round_zero]
    current = round_zero
    while True:
        updated = plain_round(
            current, nodes, adjacency, down_nodes, infinity_metric
        )
        if updated == current:
            return rounds, current
        current = updated
        rounds.append(current)


def holddown_round_zero(previous, nodes, adjacency, down_nodes, hold_down):
    """Post-event round zero under hold-down: closure nulls and open timers."""
    current = failure_round_zero(previous, nodes, adjacency, down_nodes)
    # Monotone closure: a retained finite route whose selected hop now
    # advertises the destination unreachable is itself unreachable.
    while True:
        updated = {}
        for router in nodes:
            if router in down_nodes:
                updated[router] = {dest: (None, None) for dest in nodes}
                continue
            row = {}
            for dest in nodes:
                if dest == router:
                    row[dest] = (None, 0)
                elif dest in adjacency[router]:
                    row[dest] = (dest, adjacency[router][dest])
                else:
                    hop, metric = current[router][dest]
                    if hop is None or hop not in adjacency[router]:
                        row[dest] = (None, None)
                    elif current[hop][dest][1] is None:
                        row[dest] = (None, None)
                    else:
                        row[dest] = (hop, metric)
            updated[router] = row
        if updated == current:
            break
        current = updated

    timers = {router: {} for router in nodes}
    if hold_down > 0:
        for router in nodes:
            if router in down_nodes:
                continue
            for dest in nodes:
                if dest == router or dest in adjacency[router]:
                    continue
                prior_hop = previous[router][dest][0]
                if prior_hop is not None and current[router][dest][1] is None:
                    timers[router][dest] = hold_down
    return current, timers


def holddown_round(previous, timers, nodes, adjacency, down_nodes,
                   infinity_metric, hold_down):
    """One synchronous hold-down round; returns ``(vectors, next_timers)``."""
    current = {}
    next_timers = {router: {} for router in nodes}
    for router in nodes:
        if router in down_nodes:
            current[router] = {dest: (None, None) for dest in nodes}
            continue
        row = {}
        for dest in nodes:
            if dest == router:
                row[dest] = (None, 0)
                continue
            if dest in adjacency[router]:
                # A current direct neighbor is always adopted at its direct
                # metric and any timer for it is cleared immediately.
                row[dest] = (dest, adjacency[router][dest])
                continue
            remaining = timers[router].get(dest)
            selected = previous[router][dest][0]
            if selected is not None:
                usable = (
                    selected in adjacency[router]
                    and previous[selected][dest][1] is not None
                    and adjacency[router][selected]
                    + previous[selected][dest][1]
                    < infinity_metric
                )
            else:
                usable = None
            invalidated = remaining is None and selected is not None and not usable
            if invalidated:
                row[dest] = (None, None)
                next_timers[router][dest] = hold_down
            elif remaining is not None:
                row[dest] = (None, None)
                if remaining - 1 > 0:
                    next_timers[router][dest] = remaining - 1
            else:
                row[dest] = best_route(
                    router, dest, previous, adjacency, infinity_metric
                )
        current[router] = row
    return current, next_timers


def public_timers(timers, nodes):
    """Sorted fresh copy of the timer map, matching the published shape."""
    return {
        router: {dest: timers[router][dest] for dest in sorted(timers[router])}
        for router in nodes
    }


def run_holddown(previous, nodes, adjacency, down_nodes,
                 infinity_metric, hold_down):
    """Full hold-down convergence: rounds of (vectors, public timers)."""
    vectors, timers = holddown_round_zero(
        previous, nodes, adjacency, down_nodes, hold_down
    )
    rounds = [(vectors, public_timers(timers, nodes))]
    while True:
        updated, next_timers = holddown_round(
            vectors, timers, nodes, adjacency, down_nodes,
            infinity_metric, hold_down,
        )
        next_public = public_timers(next_timers, nodes)
        if updated == vectors and next_public == rounds[-1][1]:
            return rounds, vectors
        vectors, timers = updated, next_timers
        rounds.append((vectors, next_public))


def fold_state(topology, events):
    """Fold raw events into ``(down_nodes, down_links)``; assume validated."""
    down_nodes, down_links = set(), set()
    for event in events:
        if event["action"] == "node-down":
            down_nodes.add(event["node"])
        elif event["action"] == "node-up":
            down_nodes.discard(event["node"])
        elif event["action"] == "link-down":
            down_links.add(tuple(sorted((event["from"], event["to"]))))
        else:
            down_links.discard(tuple(sorted((event["from"], event["to"]))))
    return frozenset(down_nodes), frozenset(down_links)


def normal_timeline(topology, infinity_metric, events):
    """The independent oracle's full normal replay-dv timeline."""
    nodes = sorted(topology["nodes"])
    adjacency = active_map(topology, frozenset(), frozenset())
    rounds, vectors = run_to_fixed(
        cold_round_zero(nodes, adjacency), nodes, adjacency,
        frozenset(), infinity_metric,
    )
    timeline = [(None, rounds)]
    for index, event in enumerate(events, start=1):
        down_nodes, down_links = fold_state(topology, events[:index])
        adjacency = active_map(topology, down_nodes, down_links)
        round_zero = failure_round_zero(
            vectors, nodes, adjacency, down_nodes
        )
        rounds, vectors = run_to_fixed(
            round_zero, nodes, adjacency, down_nodes, infinity_metric
        )
        timeline.append((event["time"], rounds))
    return timeline


def holddown_timeline(topology, infinity_metric, hold_down, events):
    """The independent oracle's full hold-down replay-dv timeline."""
    nodes = sorted(topology["nodes"])
    adjacency = active_map(topology, frozenset(), frozenset())
    # The fault-free baseline invalidates no finite route: its rounds are the
    # plain convergence rounds paired with empty timer maps.
    baseline_rounds, vectors = run_holddown(
        cold_round_zero(nodes, adjacency),
        nodes, adjacency, frozenset(), infinity_metric, hold_down,
    )
    timeline = [
        (None, [(vectors_round, {router: {} for router in nodes})
                for vectors_round, _timers in baseline_rounds])
    ]
    for index, event in enumerate(events, start=1):
        down_nodes, down_links = fold_state(topology, events[:index])
        adjacency = active_map(topology, down_nodes, down_links)
        rounds, vectors = run_holddown(
            vectors, nodes, adjacency, down_nodes,
            infinity_metric, hold_down,
        )
        timeline.append((event["time"], rounds))
    return timeline


# ---------------------------------------------------------------------------
# Shared selection semantics on every entry point
# ---------------------------------------------------------------------------


class TestSharedSelectionSemantics(unittest.TestCase):
    def assert_rounds_match_oracle(self, entry, expected_rounds, with_timers):
        actual_rounds = entry["rounds"]
        self.assertEqual(
            [snapshot["round"] for snapshot in actual_rounds],
            list(range(len(actual_rounds))),
        )
        self.assertEqual(
            entry["convergenceRound"], actual_rounds[-1]["round"]
        )
        self.assertEqual(len(actual_rounds), len(expected_rounds))
        for snapshot, (expected_vectors, *rest) in zip(
            actual_rounds, expected_rounds
        ):
            self.assertEqual(
                normalized(snapshot["routers"]),
                oracle_normalized(expected_vectors),
            )
            if with_timers:
                expected_timers = rest[0]
                self.assertEqual(snapshot["holdDowns"], expected_timers)

    def test_converge_matches_threshold_free_oracle(self):
        for topology in (DV_CHAIN, CHAIN3, DIAMOND, BRANCH_TREE, STAR):
            with self.subTest(topology=topology):
                nodes = sorted(topology["nodes"])
                adjacency = active_map(topology, frozenset(), frozenset())
                expected, _vectors = run_to_fixed(
                    cold_round_zero(nodes, adjacency),
                    nodes, adjacency, frozenset(), None,
                )
                result = converge_topology(topology)
                self.assert_rounds_match_oracle(
                    result,
                    [(vectors,) for vectors in expected],
                    with_timers=False,
                )
                # The threshold-free fixed point is the static shortest-path
                # answer, tie-breaking on the smaller next-hop name.
                self.assertEqual(
                    result["rounds"][-1]["routers"],
                    compute_topology(topology)["routers"],
                )

    def test_normal_replay_matches_thresholded_oracle(self):
        scenarios = [
            (DV_CHAIN, 8,
             [{"time": 1, "action": "node-down", "node": "C"}]),
            (DIAMOND, 8,
             [{"time": 1, "action": "link-down", "from": "A", "to": "B"}]),
            (CHAIN3, 50,
             [{"time": 1, "action": "node-down", "node": "C"},
              {"time": 2, "action": "node-up", "node": "C"}]),
            (DIAMOND, 30,
             [{"time": 3, "action": "link-down", "from": "A", "to": "B"},
              {"time": 5, "action": "node-down", "node": "C"},
              {"time": 9, "action": "node-up", "node": "C"},
              {"time": 15, "action": "link-up", "from": "B", "to": "A"}]),
            (BRANCH_TREE, 20,
             [{"time": 1, "action": "link-down", "from": "C", "to": "D"},
              {"time": 2, "action": "node-down", "node": "A"},
              {"time": 3, "action": "node-up", "node": "A"},
              {"time": 4, "action": "link-up", "from": "D", "to": "C"}]),
        ]
        for topology, infinity_metric, events in scenarios:
            with self.subTest(topology=topology, events=events):
                result = replay_dv_scenario(
                    topology,
                    {"infinityMetric": infinity_metric, "events": events},
                )
                expected = normal_timeline(
                    topology, infinity_metric, events
                )
                self.assertEqual(len(result["timeline"]), len(expected))
                for entry, (_time, expected_rounds) in zip(
                    result["timeline"], expected
                ):
                    self.assert_rounds_match_oracle(
                        entry,
                        [(vectors,) for vectors in expected_rounds],
                        with_timers=False,
                    )

    def test_holddown_replay_matches_independent_holddown_oracle(self):
        scenarios = [
            (DV_CHAIN, 8, 2,
             [{"time": 1, "action": "node-down", "node": "C"}]),
            (DIAMOND, 8, 2,
             [{"time": 1, "action": "link-down", "from": "A", "to": "B"}]),
            (CHAIN3, 50, 3,
             [{"time": 1, "action": "node-down", "node": "C"},
              {"time": 2, "action": "node-up", "node": "C"}]),
            (DV_CHAIN, 3, 2,
             [{"time": 1, "action": "node-down", "node": "C"}]),
            (DIAMOND, 30, 2,
             [{"time": 3, "action": "link-down", "from": "A", "to": "B"},
              {"time": 5, "action": "node-down", "node": "C"},
              {"time": 9, "action": "node-up", "node": "C"},
              {"time": 15, "action": "link-up", "from": "B", "to": "A"}]),
        ]
        for topology, infinity_metric, hold_down, events in scenarios:
            with self.subTest(topology=topology, events=events):
                result = replay_dv_scenario(
                    topology,
                    {"infinityMetric": infinity_metric,
                     "holdDownRounds": hold_down, "events": events},
                )
                expected = holddown_timeline(
                    topology, infinity_metric, hold_down, events
                )
                self.assertEqual(len(result["timeline"]), len(expected))
                for entry, (_time, expected_rounds) in zip(
                    result["timeline"], expected
                ):
                    self.assert_rounds_match_oracle(
                        entry, expected_rounds, with_timers=True
                    )

    def test_equal_cost_selects_smaller_next_hop_in_every_mode(self):
        events = []
        converge_final = converge_topology(DIAMOND)["rounds"][-1]["routers"]
        normal_final = replay_dv_scenario(
            DIAMOND, {"infinityMetric": 50, "events": events}
        )["timeline"][0]["rounds"][-1]["routers"]
        held_final = replay_dv_scenario(
            DIAMOND,
            {"infinityMetric": 50, "holdDownRounds": 3, "events": events},
        )["timeline"][0]["rounds"][-1]["routers"]
        for table in (converge_final, normal_final, held_final):
            self.assertEqual(table["A"]["D"]["nextHop"], "B")
            self.assertEqual(table["D"]["A"]["nextHop"], "B")
            self.assertEqual(table["B"]["C"]["nextHop"], "A")
            self.assertEqual(table["C"]["B"]["nextHop"], "A")

    def test_low_infinity_truncates_long_candidates(self):
        # Unit links, threshold 2: a two-hop total of exactly 2 is at the
        # threshold and must already be unreachable at the fault-free
        # baseline; convergence still runs synchronously from round 0.
        result = replay_dv_scenario(
            DV_CHAIN, {"infinityMetric": 2, "events": []}
        )
        entry = result["timeline"][0]
        round_zero = entry["rounds"][0]["routers"]
        self.assertEqual(round_zero["A"]["C"], NULL)
        self.assertEqual(round_zero["B"]["C"], {"nextHop": "C", "metric": 1})
        final = entry["rounds"][-1]["routers"]
        self.assertEqual(final["A"]["B"], {"nextHop": "B", "metric": 1})
        self.assertEqual(final["A"]["C"], NULL)
        self.assertEqual(final["B"]["A"], {"nextHop": "A", "metric": 1})
        # Threshold 3 truncates only totals >= 3: the two-hop route (2)
        # converges normally.
        result3 = replay_dv_scenario(
            DV_CHAIN, {"infinityMetric": 3, "events": []}
        )
        final3 = result3["timeline"][0]["rounds"][-1]["routers"]
        self.assertEqual(
            final3["A"]["C"], {"nextHop": "B", "metric": 2}
        )

    def test_down_router_row_is_unreachable_in_every_mode_and_round(self):
        events = [{"time": 1, "action": "node-down", "node": "C"}]
        for scenario in (
            {"infinityMetric": 8, "events": events},
            {"infinityMetric": 8, "holdDownRounds": 2, "events": events},
        ):
            with self.subTest(scenario=scenario):
                entry = replay_dv_scenario(
                    DV_CHAIN, scenario
                )["timeline"][1]
                for snapshot in entry["rounds"]:
                    for dest in ("A", "B", "C"):
                        self.assertEqual(
                            snapshot["routers"]["C"][dest], NULL
                        )
                    if "holdDowns" in snapshot:
                        self.assertEqual(snapshot["holdDowns"]["C"], {})

    def test_only_changed_rounds_are_recorded(self):
        # Hold-down records a timer-only change, and a round identical in both
        # vectors and timers is the unrecorded fixed point.
        entry = replay_dv_scenario(
            DV_CHAIN,
            {"infinityMetric": 8, "holdDownRounds": 2,
             "events": [{"time": 1, "action": "node-down", "node": "C"}]},
        )["timeline"][1]
        rounds = entry["rounds"]
        for previous, current in zip(rounds, rounds[1:]):
            self.assertTrue(
                previous["routers"] != current["routers"]
                or previous["holdDowns"] != current["holdDowns"]
            )
        self.assertEqual(entry["convergenceRound"], rounds[-1]["round"])
        self.assertEqual(rounds[-1]["holdDowns"],
                         {"A": {}, "B": {}, "C": {}})


# ---------------------------------------------------------------------------
# Cross-entry stable-table agreement under the applicable conditions
# ---------------------------------------------------------------------------


def sequence_is_legal(topology, events):
    """Fold the transition rules; True iff every event applies in sequence."""
    names = set(topology["nodes"])
    links = {
        tuple(sorted((edge["from"], edge["to"])))
        for edge in topology["links"]
    }
    down_nodes, down_links = set(), set()
    for event in events:
        if event["action"].startswith("node"):
            node = event["node"]
            if node not in names:
                return False
            if event["action"] == "node-down":
                if node in down_nodes:
                    return False
                down_nodes.add(node)
            else:
                if node not in down_nodes:
                    return False
                down_nodes.discard(node)
        else:
            pair = tuple(sorted((event["from"], event["to"])))
            if pair not in links:
                return False
            if event["action"] == "link-down":
                if pair in down_links:
                    return False
                down_links.add(pair)
            else:
                if pair not in down_links:
                    return False
                down_links.discard(pair)
    return True


TREE_SEQUENCES = [
    [{"time": 1, "action": "node-down", "node": "B"}],
    [{"time": 1, "action": "link-down", "from": "C", "to": "D"},
     {"time": 2, "action": "link-up", "from": "D", "to": "C"}],
    [{"time": 1, "action": "node-down", "node": "A"},
     {"time": 2, "action": "node-down", "node": "E"},
     {"time": 3, "action": "node-up", "node": "A"}],
    [{"time": 1, "action": "link-down", "from": "C", "to": "A"},
     {"time": 2, "action": "node-down", "node": "D"},
     {"time": 3, "action": "node-up", "node": "D"},
     {"time": 4, "action": "link-up", "from": "A", "to": "C"}],
    [{"time": 1, "action": "node-down", "node": "l2"}],
]


class TestStableTableAgreement(unittest.TestCase):
    def test_fault_free_baseline_agrees_across_all_three_entries(self):
        # With a threshold above every path length, converge, normal
        # replay-dv and hold-down replay-dv converge to one stable table, and
        # it is the static compute answer.
        for topology in (DV_CHAIN, CHAIN3, DIAMOND, BRANCH_TREE, STAR):
            with self.subTest(topology=topology):
                converge = converge_topology(topology)["rounds"][-1]["routers"]
                normal = replay_dv_scenario(
                    topology, {"infinityMetric": 1000, "events": []}
                )["timeline"][0]["rounds"][-1]["routers"]
                held = replay_dv_scenario(
                    topology,
                    {"infinityMetric": 1000, "holdDownRounds": 3,
                     "events": []},
                )["timeline"][0]["rounds"][-1]["routers"]
                static = compute_topology(topology)["routers"]
                self.assertEqual(converge, static)
                self.assertEqual(normal, static)
                self.assertEqual(held, static)

    def test_normal_and_holddown_agree_after_every_event_on_trees(self):
        # On a tree the unique path to a current direct neighbor is the
        # direct link itself, so hold-down's direct-neighbor pinning can
        # never diverge from ordinary best-route selection: the two modes
        # must end every timeline entry at identical stable tables.
        for topology in TREES:
            legal = [
                sequence
                for sequence in TREE_SEQUENCES
                if sequence_is_legal(topology, sequence)
            ]
            for events in legal:
                for infinity_metric in (8, 20, 100):
                    max_link = max(
                        (edge["metric"] for edge in topology["links"]),
                        default=0,
                    )
                    if infinity_metric <= max_link:
                        continue
                    for hold_down in (1, 2, 3):
                        with self.subTest(
                            topology=topology["nodes"],
                            events=events,
                            infinity=infinity_metric,
                            hold=hold_down,
                        ):
                            normal = replay_dv_scenario(
                                topology,
                                {"infinityMetric": infinity_metric,
                                 "events": events},
                            )
                            held = replay_dv_scenario(
                                topology,
                                {"infinityMetric": infinity_metric,
                                 "holdDownRounds": hold_down,
                                 "events": events},
                            )
                            for normal_entry, held_entry in zip(
                                normal["timeline"], held["timeline"]
                            ):
                                self.assertEqual(
                                    normal_entry["rounds"][-1]["routers"],
                                    held_entry["rounds"][-1]["routers"],
                                )

    def test_full_restoration_returns_to_the_converged_table(self):
        restore_sequences = [
            [{"time": 1, "action": "node-down", "node": "C"},
             {"time": 2, "action": "node-up", "node": "C"}],
            [{"time": 1, "action": "link-down", "from": "A", "to": "B"},
             {"time": 2, "action": "link-up", "from": "B", "to": "A"}],
            [{"time": 3, "action": "link-down", "from": "A", "to": "B"},
             {"time": 5, "action": "node-down", "node": "C"},
             {"time": 9, "action": "node-up", "node": "C"},
             {"time": 15, "action": "link-up", "from": "B", "to": "A"}],
        ]
        for topology in (DV_CHAIN, CHAIN3, DIAMOND):
            for events in restore_sequences:
                if not sequence_is_legal(topology, events):
                    continue
                baseline = converge_topology(
                    topology
                )["rounds"][-1]["routers"]
                for hold_down in (None, 1, 3):
                    with self.subTest(
                        topology=topology["nodes"],
                        events=events,
                        hold=hold_down,
                    ):
                        scenario = {
                            "infinityMetric": 1000, "events": events
                        }
                        if hold_down is not None:
                            scenario["holdDownRounds"] = hold_down
                        result = replay_dv_scenario(topology, scenario)
                        self.assertEqual(
                            result["timeline"][-1]["rounds"][-1]["routers"],
                            baseline,
                        )

    def test_holddown_keeps_current_direct_route_pinned(self):
        # Preserved existing behavior: while converging under hold-down, a
        # current direct neighbor stays adopted at the direct metric even
        # once an indirect advertisement becomes smaller, whereas the normal
        # mode eventually selects the smaller indirect route.  Both modes
        # still agree on every non-direct destination they reach.
        events = [
            {"time": 1, "action": "link-down", "from": "A", "to": "B"}
        ]
        normal = replay_dv_scenario(
            PINNED_DIRECT, {"infinityMetric": 100, "events": events}
        )["timeline"][1]["rounds"][-1]["routers"]
        held = replay_dv_scenario(
            PINNED_DIRECT,
            {"infinityMetric": 100, "holdDownRounds": 2, "events": events},
        )["timeline"][1]["rounds"][-1]["routers"]
        # Indirect A-C-D = 6 beats the direct A-D = 9 under normal selection.
        self.assertEqual(
            normal["A"]["D"], {"nextHop": "C", "metric": 6}
        )
        # Hold-down keeps the current direct route pinned at its metric.
        self.assertEqual(
            held["A"]["D"], {"nextHop": "D", "metric": 9}
        )


# ---------------------------------------------------------------------------
# Hold-down timing semantics
# ---------------------------------------------------------------------------


class TestHolddownTiming(unittest.TestCase):
    def test_window_opens_counts_down_and_releases_once(self):
        entry = replay_dv_scenario(
            DV_CHAIN,
            {"infinityMetric": 8, "holdDownRounds": 3,
             "events": [{"time": 1, "action": "node-down", "node": "C"}]},
        )["timeline"][1]
        rounds = entry["rounds"]
        self.assertEqual(
            [rounds[i]["holdDowns"]["B"].get("C") for i in range(4)],
            [3, 2, 1, None],
        )
        for snapshot in rounds[:3]:
            self.assertEqual(snapshot["routers"]["B"]["C"], NULL)
        # No alternative exists, so release leaves the destination null.
        self.assertEqual(rounds[-1]["routers"]["B"]["C"], NULL)

    def test_alternative_is_refused_inside_window_and_adopted_after(self):
        entry = replay_dv_scenario(
            DIAMOND,
            {"infinityMetric": 8, "holdDownRounds": 2,
             "events": [
                 {"time": 1, "action": "link-down", "from": "A", "to": "B"}
             ]},
        )["timeline"][1]
        rounds = entry["rounds"]
        for snapshot in rounds[:3]:
            self.assertEqual(snapshot["routers"]["A"]["D"], NULL)
        self.assertEqual(
            rounds[3]["routers"]["A"]["D"],
            {"nextHop": "C", "metric": 2},
        )
        self.assertEqual(entry["convergenceRound"], 4)

    def test_direct_recovery_adopts_at_once_and_clears_timer(self):
        timeline = replay_dv_scenario(
            CHAIN3,
            {"infinityMetric": 50, "holdDownRounds": 3,
             "events": [
                 {"time": 1, "action": "node-down", "node": "C"},
                 {"time": 2, "action": "node-up", "node": "C"},
             ]},
        )["timeline"]
        recovery_zero = timeline[2]["rounds"][0]
        self.assertEqual(
            recovery_zero["routers"]["B"]["C"],
            {"nextHop": "C", "metric": 7},
        )
        self.assertEqual(recovery_zero["holdDowns"]["B"], {})
        for snapshot in timeline[2]["rounds"]:
            self.assertNotIn("C", snapshot["holdDowns"]["B"])

    def test_repeated_unreachability_never_extends_window(self):
        entry = replay_dv_scenario(
            DIAMOND,
            {"infinityMetric": 8, "holdDownRounds": 3,
             "events": [
                 {"time": 1, "action": "link-down", "from": "A", "to": "B"}
             ]},
        )["timeline"][1]
        remainders = [
            entry["rounds"][i]["holdDowns"]["A"].get("D")
            for i in range(4)
        ]
        self.assertEqual(remainders, [3, 2, 1, None])

    def test_timers_do_not_leak_between_events(self):
        events = [
            {"time": 3, "action": "link-down", "from": "A", "to": "B"},
            {"time": 5, "action": "node-down", "node": "C"},
            {"time": 9, "action": "node-up", "node": "C"},
            {"time": 15, "action": "link-up", "from": "B", "to": "A"},
        ]
        result = replay_dv_scenario(
            DIAMOND,
            {"infinityMetric": 30, "holdDownRounds": 2, "events": events},
        )
        for entry in result["timeline"][1:]:
            for mapping in entry["rounds"][-1]["holdDowns"].values():
                self.assertEqual(mapping, {})

    def test_zero_or_omitted_holddown_is_byte_compatible(self):
        events = [{"time": 1, "action": "node-down", "node": "C"}]
        omitted = replay_dv_scenario(
            DV_CHAIN, {"infinityMetric": 8, "events": events}
        )
        explicit_zero = replay_dv_scenario(
            DV_CHAIN,
            {"infinityMetric": 8, "holdDownRounds": 0, "events": events},
        )
        self.assertEqual(
            json.dumps(omitted, sort_keys=True),
            json.dumps(explicit_zero, sort_keys=True),
        )
        self.assertNotIn("holdDownRounds", omitted)
        for entry in omitted["timeline"]:
            for snapshot in entry["rounds"]:
                self.assertNotIn("holdDowns", snapshot)


# ---------------------------------------------------------------------------
# Validation distinctions and precedence remain unchanged
# ---------------------------------------------------------------------------


class TestValidationUnchanged(unittest.TestCase):
    def test_error_types_remain_distinguishable(self):
        self.assertTrue(
            issubclass(InvalidStateTransition, InvalidScenario)
        )
        with self.assertRaises(InvalidTopology):
            replay_dv_scenario(
                {"nodes": ["A", "A"], "links": []},
                {"infinityMetric": 8, "events": []},
            )
        with self.assertRaises(InvalidScenario):
            replay_dv_scenario(
                DV_CHAIN, {"infinityMetric": 1, "events": []}
            )
        with self.assertRaises(InvalidStateTransition):
            replay_dv_scenario(
                DV_CHAIN,
                {"infinityMetric": 8,
                 "events": [
                     {"time": 1, "action": "node-down", "node": "C"},
                     {"time": 2, "action": "node-down", "node": "C"},
                 ]},
            )

    def test_topology_errors_take_precedence(self):
        with self.assertRaises(InvalidTopology):
            replay_dv_scenario(
                {"nodes": ["A", "A"], "links": []},
                {"infinityMetric": "bad"},
            )

    def test_cli_reports_invalid_scenario_text_and_code(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get(
            "PYTHONPATH", ""
        )
        code = (
            "import json, tempfile, os, sys;"
            "from packet_routing_sim.__main__ import main;"
            f"topo = tempfile.NamedTemporaryFile('w', suffix='.json', delete=False);"
            "json.dump("
            + repr({"nodes": ["A", "B", "C"],
                    "links": [link("A", "B", 1), link("B", "C", 1)]})
            + ", topo); topo.close();"
            "scen = tempfile.NamedTemporaryFile('w', suffix='.json', delete=False);"
            "json.dump({'infinityMetric': 8, 'holdDownRounds': True,"
            "'events': []}, scen); scen.close();"
            "sys.exit(main(['replay-dv', topo.name, scen.name]))"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stdout, b"")
        self.assertEqual(proc.stderr, b"invalid scenario\n")


# ---------------------------------------------------------------------------
# Purity: input immutability, result isolation, no shared mutable structures
# ---------------------------------------------------------------------------


class TestSharedKernelPurity(unittest.TestCase):
    SCENARIO = {
        "infinityMetric": 30,
        "holdDownRounds": 2,
        "events": [
            {"time": 3, "action": "link-down", "from": "A", "to": "B"},
            {"time": 5, "action": "node-down", "node": "C"},
            {"time": 9, "action": "node-up", "node": "C"},
            {"time": 15, "action": "link-up", "from": "B", "to": "A"},
        ],
    }

    def test_inputs_are_not_modified(self):
        for topology in (DV_CHAIN, DIAMOND):
            topology_before = copy.deepcopy(topology)
            scenario_before = copy.deepcopy(self.SCENARIO)
            replay_dv_scenario(topology, self.SCENARIO)
            self.assertEqual(topology, topology_before)
            self.assertEqual(self.SCENARIO, scenario_before)
        converge_input = copy.deepcopy(DIAMOND)
        converge_topology(converge_input)
        self.assertEqual(converge_input, DIAMOND)

    def test_mutating_one_result_cannot_reach_another(self):
        first = replay_dv_scenario(DIAMOND, copy.deepcopy(self.SCENARIO))
        first["timeline"][1]["rounds"][0]["routers"]["A"]["D"] = {
            "nextHop": "HACK", "metric": -1
        }
        first["timeline"][1]["rounds"][0]["holdDowns"]["A"]["D"] = 999
        first["timeline"][1]["event"]["node"] = "HACK"
        second = replay_dv_scenario(DIAMOND, copy.deepcopy(self.SCENARIO))
        snapshot = second["timeline"][1]["rounds"][0]
        self.assertEqual(snapshot["routers"]["A"]["D"], NULL)
        self.assertEqual(snapshot["holdDowns"]["A"].get("D"), 2)
        self.assertNotEqual(
            second["timeline"][1]["event"].get("node"), "HACK"
        )

    def test_rounds_and_entries_do_not_share_mutable_data(self):
        result = replay_dv_scenario(DIAMOND, copy.deepcopy(self.SCENARIO))
        timeline = result["timeline"]
        baseline_final = timeline[0]["rounds"][-1]["routers"]
        event_zero = timeline[1]["rounds"][0]["routers"]
        for router in baseline_final:
            for dest in baseline_final[router]:
                self.assertIsNot(
                    baseline_final[router][dest],
                    event_zero[router][dest],
                    (router, dest),
                )
        # Mutating an early round leaves the converged round untouched.
        entry = timeline[1]
        entry["rounds"][0]["routers"]["A"]["B"] = {
            "nextHop": "HACK", "metric": 0
        }
        self.assertNotEqual(
            entry["rounds"][-1]["routers"]["A"].get("B"),
            {"nextHop": "HACK", "metric": 0},
        )

    def test_repeated_calls_share_no_carried_state(self):
        fault = {
            "infinityMetric": 8, "holdDownRounds": 2,
            "events": [{"time": 1, "action": "node-down", "node": "C"}],
        }
        faulted = replay_dv_scenario(DV_CHAIN, fault)
        self.assertEqual(
            faulted["timeline"][-1]["rounds"][-1]["routers"]["A"]["C"], NULL
        )
        fresh = replay_dv_scenario(
            DV_CHAIN,
            {"infinityMetric": 8, "holdDownRounds": 2, "events": []},
        )
        self.assertEqual(
            fresh["timeline"][0]["rounds"][-1]["routers"],
            converge_topology(DV_CHAIN)["rounds"][-1]["routers"],
        )


# ---------------------------------------------------------------------------
# Declaration-order and hash-seed independence
# ---------------------------------------------------------------------------


_DETERMINISM_SNIPPET = """
import json, os, sys
sys.path.insert(0, %r)
from packet_routing_sim.core import (
    converge_topology, replay_dv_scenario)
topology = json.loads(os.environ["PRSIM_TOPO"])
scenario = json.loads(os.environ["PRSIM_SCEN"])
out = {
    "converge": converge_topology(topology),
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
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


class TestDeterminism(unittest.TestCase):
    SCENARIO = {
        "infinityMetric": 30,
        "holdDownRounds": 2,
        "events": [
            {"time": 3, "action": "link-down", "from": "A", "to": "B"},
            {"time": 5, "action": "node-down", "node": "C"},
            {"time": 9, "action": "node-up", "node": "C"},
            {"time": 15, "action": "link-up", "from": "B", "to": "A"},
        ],
    }
    VARIANTS = [
        DIAMOND,
        {"nodes": ["D", "C", "B", "A"], "links": DIAMOND["links"]},
        {"nodes": ["A", "B", "C", "D"], "links": [
            link("B", "A", 1), link("C", "A", 1),
            link("D", "B", 1), link("D", "C", 1)]},
        {"nodes": ["A", "B", "C", "D"],
         "links": list(reversed(DIAMOND["links"]))},
    ]

    def test_byte_identical_across_seeds_and_declaration_orders(self):
        outputs = set()
        for seed in HASH_SEEDS:
            for variant in self.VARIANTS:
                outputs.add(
                    run_in_subprocess(variant, self.SCENARIO, seed)
                )
        self.assertEqual(len(outputs), 1)


if __name__ == "__main__":
    unittest.main()
