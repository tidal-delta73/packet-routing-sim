"""Regression tests for the unified distance-vector round-selection kernel.

The distance-vector core has three public ways to run but one shared set of
per-round semantics.  These tests pin that shared contract after the kernel
refactor:

* the three distance-vector entry points -- ``converge``, ``replay-dv`` in
  its plain (omitted/zero ``holdDownRounds``) form and ``replay-dv`` with a
  positive ``holdDownRounds`` -- reach the same stable forwarding tables
  whenever they are asked about the same network state; hold-down may delay
  adoption but can never change the converged answer;
* every path picks the route by smallest total metric, breaks ties to the
  smaller next-hop name, leaves a down router's whole row unreachable,
  clamps candidates at ``infinityMetric`` (where one applies), starts
  ``converge`` from a self/direct-neighbor round 0, inherits the previous
  converged vectors across replay events, and records only rounds in which
  the routing table (or, under hold-down, a timer) actually changes;
* hold-down keeps the documented invalidation, countdown, direct-recovery
  and dual-fixed-point rules;
* the published shape gains no new field, validation precedence and the
  InvalidTopology / InvalidScenario / InvalidStateTransition distinction
  (including the CLI status code and stderr text) are unchanged;
* inputs are never mutated, results never share mutable structures, and the
  serialized answer is independent of declaration order and hash seed.

Every expected stable table in this module is produced by an independently
written Floyd-Warshall oracle in the test file itself, never by reading the
sut's internals, so a defect in the implementation cannot also forge the
expected answer.
"""
import copy
import json
import os
import random
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
    for seed in os.environ.get("PRSIM_TEST_SEEDS", "0,1,42,987654321").split(",")
)


def link(source, target, metric):
    return {"from": source, "to": target, "metric": metric}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


EMPTY = {"nodes": [], "links": []}
EDGELESS = {"nodes": ["x", "y", "z"], "links": []}
COMPONENTS = {
    "nodes": ["a1", "a2", "b1", "b2", "iso"],
    "links": [link("a1", "a2", 2), link("b1", "b2", 5)],
}
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
# Metric-10 edges: a tight (but linkwise legal) infinity threshold truncates
# every multi-hop route.
LONG_METRIC_CHAIN = {
    "nodes": ["n0", "n1", "n2", "n3"],
    "links": [
        link("n0", "n1", 10),
        link("n1", "n2", 10),
        link("n2", "n3", 10),
    ],
}


# ---------------------------------------------------------------------------
# Independent oracle (Floyd-Warshall + the published clamp/tie rules)
# ---------------------------------------------------------------------------


INF = float("inf")


def _adjacency(topology, down_nodes=frozenset(), down_links=frozenset()):
    """Undirected adjacency of the links usable in a given availability state."""
    nodes = sorted(topology["nodes"])
    adjacency = {node: {} for node in nodes}
    for edge in topology["links"]:
        a, b, w = edge["from"], edge["to"], edge["metric"]
        pair = (a, b) if a < b else (b, a)
        if a in down_nodes or b in down_nodes or pair in frozenset(down_links):
            continue
        adjacency[a][b] = w
        adjacency[b][a] = w
    return nodes, adjacency


def _floyd(nodes, adjacency):
    dist = {i: {j: INF for j in nodes} for i in nodes}
    for node in nodes:
        dist[node][node] = 0
    for a in nodes:
        for b, w in adjacency[a].items():
            dist[a][b] = w
    for k in nodes:
        for i in nodes:
            via = dist[i][k]
            if via == INF:
                continue
            for j in nodes:
                candidate = via + dist[k][j]
                if candidate < dist[i][j]:
                    dist[i][j] = candidate
    return dist


def oracle_stable(topology, down_nodes=frozenset(), down_links=frozenset(),
                  infinity_metric=None):
    """Expected stable ``{(router, dest): (nextHop, metric)}`` tables.

    Shortest lengths come from Floyd-Warshall; ``infinity_metric`` truncates
    a route whose total is at or above it, a down router's whole row is
    unreachable, and equal-cost choices take the lexicographically smallest
    neighbor lying on a shortest path.
    """
    nodes, active = _adjacency(topology, down_nodes, down_links)
    dist = _floyd(nodes, active)
    tables = {}
    for router in nodes:
        if router in down_nodes:
            tables.update({(router, d): (None, None) for d in nodes})
            continue
        for dest in nodes:
            if dest == router:
                tables[router, dest] = (None, 0)
            elif dest in down_nodes or dist[router][dest] == INF:
                tables[router, dest] = (None, None)
            else:
                metric = dist[router][dest]
                if infinity_metric is not None and metric >= infinity_metric:
                    tables[router, dest] = (None, None)
                    continue
                winners = [
                    neighbor
                    for neighbor in active[router]
                    if dist[neighbor][dest] != INF
                    and active[router][neighbor] + dist[neighbor][dest]
                    == metric
                ]
                tables[router, dest] = (min(winners), metric)
    return tables


def oracle_dv_entry_finals(topology, events, infinity_metric,
                           hold_down=None):
    """Final normalized tables of every timeline entry, independently derived.

    Replays the whole distance-vector timeline with rules written only for
    these tests: the fault-free baseline converges with plain clamped
    Bellman-Ford in every mode (hold-down publishes empty maps on exactly
    those rounds); each later event inherits the previous converged vectors,
    builds failure round zero (with the transitive invalidation closure when
    ``hold_down`` is positive), and then iterates either plain clamped
    updates or the hold-down update (direct neighbor always adopted,
    suppression countdown, fresh-window invalidation) to a fixed point.
    ``hold_down`` of ``None`` is the plain mode.
    """
    nodes, full = _adjacency(topology)

    def fold_state(down, disabled):
        active = {node: {} for node in nodes}
        for router in nodes:
            if router in down:
                continue
            for neighbor, weight in full[router].items():
                pair = (router, neighbor) if router < neighbor \
                    else (neighbor, router)
                if neighbor not in down and pair not in disabled:
                    active[router][neighbor] = weight
        return active

    def direct_only(active, down):
        view = {}
        for router in nodes:
            if router in down:
                view.update({(router, d): (None, None) for d in nodes})
                continue
            for dest in nodes:
                if dest == router:
                    view[router, dest] = (None, 0)
                elif dest in active[router]:
                    view[router, dest] = (dest, active[router][dest])
                else:
                    view[router, dest] = (None, None)
        return view

    def plain_step(previous, active, down):
        return oracle_synchronous(
            previous, nodes, active, down, infinity_metric
        )

    def fixed_point(start, step):
        current = start
        for _ in range(10000):
            updated = step(current)
            if updated == current:
                return current
            current = updated
        raise AssertionError("oracle did not converge")

    down, disabled = set(), set()
    active = fold_state(down, disabled)
    converged = fixed_point(
        direct_only(active, frozenset()),
        lambda view: plain_step(view, active, frozenset()),
    )
    finals = [converged]

    for event in events:
        action = event["action"]
        if action == "node-down":
            down.add(event["node"])
        elif action == "node-up":
            down.discard(event["node"])
        elif action == "link-down":
            a, b = event["from"], event["to"]
            disabled.add((a, b) if a < b else (b, a))
        else:
            a, b = event["from"], event["to"]
            disabled.discard((a, b) if a < b else (b, a))
        downs = frozenset(down)
        active = fold_state(down, disabled)

        # Failure round zero: inherit, drop routes through a lost hop, always
        # re-adopt current direct neighbors.
        round_zero = {}
        for router in nodes:
            if router in downs:
                round_zero.update(
                    {(router, d): (None, None) for d in nodes}
                )
                continue
            for dest in nodes:
                if dest == router:
                    round_zero[router, dest] = (None, 0)
                elif dest in active[router]:
                    round_zero[router, dest] = (
                        dest, active[router][dest]
                    )
                else:
                    hop, metric = converged[router, dest]
                    round_zero[router, dest] = (
                        (hop, metric) if hop in active[router]
                        else (None, None)
                    )

        if hold_down is None:
            converged = fixed_point(
                round_zero,
                lambda view: plain_step(view, active, downs),
            )
            finals.append(converged)
            continue

        # Transitive round-zero invalidation closure for hold-down.
        current = round_zero
        while True:
            updated = {}
            for router in nodes:
                if router in downs:
                    updated.update(
                        {(router, d): current[router, d] for d in nodes}
                    )
                    continue
                for dest in nodes:
                    if dest == router or dest in active[router]:
                        updated[router, dest] = current[router, dest]
                        continue
                    hop, metric = current[router, dest]
                    if (hop is None or hop not in active[router]
                            or current[hop, dest][1] is None):
                        updated[router, dest] = (None, None)
                    else:
                        updated[router, dest] = current[router, dest]
            if updated == current:
                break
            current = updated

        timers = {}
        for router in nodes:
            if router in downs:
                continue
            for dest in nodes:
                if dest == router or dest in active[router]:
                    continue
                if (converged[router, dest][0] is not None
                        and current[router, dest][1] is None):
                    timers[router, dest] = hold_down

        def hold_step(state):
            vectors, live = state
            next_vectors, next_live = {}, {}
            for router in nodes:
                if router in downs:
                    next_vectors.update(
                        {(router, d): (None, None) for d in nodes}
                    )
                    continue
                for dest in nodes:
                    if dest == router:
                        next_vectors[router, dest] = (None, 0)
                        continue
                    if dest in active[router]:
                        next_vectors[router, dest] = (
                            dest, active[router][dest]
                        )
                        continue
                    remaining = live.get((router, dest))
                    selected = vectors[router, dest][0]
                    usable = (
                        selected in active[router]
                        and vectors[selected, dest][1] is not None
                        and active[router][selected]
                        + vectors[selected, dest][1] < infinity_metric
                    )
                    if remaining is None and selected is not None \
                            and not usable:
                        next_vectors[router, dest] = (None, None)
                        next_live[router, dest] = hold_down
                    elif remaining is not None:
                        next_vectors[router, dest] = (None, None)
                        if remaining - 1 > 0:
                            next_live[router, dest] = remaining - 1
                    else:
                        best_cost, best_hop = None, None
                        for neighbor in sorted(active[router]):
                            advertised = vectors[neighbor, dest][1]
                            if advertised is None:
                                continue
                            cost = active[router][neighbor] + advertised
                            if cost >= infinity_metric:
                                continue
                            if best_cost is None or cost < best_cost:
                                best_cost, best_hop = cost, neighbor
                        next_vectors[router, dest] = (
                            (None, None) if best_cost is None
                            else (best_hop, best_cost)
                        )
            return next_vectors, next_live

        converged, _timers = fixed_point(
            (current, timers), hold_step
        )
        finals.append(converged)
    return finals


def oracle_synchronous(previous, nodes, active, down_nodes, infinity_metric):
    """One independent synchronous Bellman-Ford update with clamp."""
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
                advertised = previous[neighbor, dest][1]
                if advertised is None:
                    continue
                cost = active[router][neighbor] + advertised
                if infinity_metric is not None and cost >= infinity_metric:
                    continue
                if best_cost is None or cost < best_cost:
                    best_cost, best_hop = cost, neighbor
            current[router, dest] = (
                (None, None) if best_cost is None else (best_hop, best_cost)
            )
    return current


def normalize(routers):
    """Nested JSON routing snapshot -> ``{(router, dest): (nextHop, metric)}``."""
    return {
        (router, dest): (entry["nextHop"], entry["metric"])
        for router, row in routers.items()
        for dest, entry in row.items()
    }


def fold_events(events):
    """Track availability independently of the sut while folding raw events."""
    down_nodes, down_links = set(), set()
    for event in events:
        if event["action"] == "node-down":
            down_nodes.add(event["node"])
        elif event["action"] == "node-up":
            down_nodes.discard(event["node"])
        elif event["action"] == "link-down":
            a, b = event["from"], event["to"]
            down_links.add((a, b) if a < b else (b, a))
        else:
            a, b = event["from"], event["to"]
            down_links.discard((a, b) if a < b else (b, a))
    return frozenset(down_nodes), frozenset(down_links)


# ---------------------------------------------------------------------------
# The three distance-vector entries agree on stable forwarding tables
# ---------------------------------------------------------------------------


class TestStableTableEquivalence(unittest.TestCase):
    def assert_stable_matches_oracle(
        self, topology, *, down_nodes=frozenset(), down_links=frozenset(),
        infinity_metric=None,
    ):
        expected = oracle_stable(
            topology, down_nodes, down_links, infinity_metric
        )
        # Plain replay-dv and hold-down replay-dv must converge to exactly
        # the same tables; hold-down changes timing, never the destination.
        for hold_down in (1, 2, 3):
            result = replay_dv_scenario(
                topology,
                {
                    "infinityMetric": infinity_metric
                    if infinity_metric is not None
                    else 10 ** 6,
                    "holdDownRounds": hold_down,
                    "events": [],
                },
            )
            final = result["timeline"][0]["rounds"][-1]["routers"]
            self.assertEqual(normalize(final), expected, hold_down)

    def test_fault_free_entries_agree_on_every_fixture(self):
        # converge (no clamp), plain replay-dv with a generous threshold,
        # replay-dv with holdDownRounds 0 and positive, and compute all
        # publish the same fault-free stable tables.
        for topology in (
            EMPTY, EDGELESS, COMPONENTS, CHAIN3, DV_CHAIN, DIAMOND, RING,
            LONG_METRIC_CHAIN,
        ):
            with self.subTest(topology=topology):
                converged = converge_topology(topology)["rounds"][-1]["routers"]
                self.assertEqual(
                    normalize(converged), oracle_stable(topology)
                )
                if topology["nodes"]:
                    self.assertEqual(
                        converged, compute_topology(topology)["routers"]
                    )
                plain = replay_dv_scenario(
                    topology, {"infinityMetric": 10 ** 6, "events": []}
                )["timeline"][0]["rounds"][-1]["routers"]
                self.assertEqual(normalize(plain), oracle_stable(topology))
                explicit_zero = replay_dv_scenario(
                    topology,
                    {"infinityMetric": 10 ** 6, "holdDownRounds": 0,
                     "events": []},
                )["timeline"][0]["rounds"][-1]["routers"]
                self.assertEqual(normalize(explicit_zero), oracle_stable(topology))
                self.assert_stable_matches_oracle(topology)

    def test_equivalence_after_full_failure_recovery_cycles(self):
        # With every object restored, each DV entry's last timeline entry
        # ends on the fault-free stable table regardless of hold-down.
        events = [
            {"time": 1, "action": "link-down", "from": "A", "to": "B"},
            {"time": 2, "action": "node-down", "node": "B"},
            {"time": 3, "action": "node-down", "node": "D"},
            {"time": 4, "action": "node-up", "node": "B"},
            {"time": 5, "action": "node-up", "node": "D"},
            {"time": 6, "action": "link-up", "from": "B", "to": "A"},
        ]
        baseline = converge_topology(DIAMOND)["rounds"][-1]["routers"]
        for scenario in (
            {"infinityMetric": 30, "events": events},
            {"infinityMetric": 30, "holdDownRounds": 0, "events": events},
            {"infinityMetric": 30, "holdDownRounds": 1, "events": events},
            {"infinityMetric": 30, "holdDownRounds": 3, "events": events},
        ):
            with self.subTest(scenario=scenario):
                timeline = replay_dv_scenario(DIAMOND, scenario)["timeline"]
                self.assertEqual(
                    timeline[-1]["rounds"][-1]["routers"], baseline
                )
                # Positive hold-down convergence exhausts every timer, so
                # nothing carries into the next event's round zero.
                if scenario.get("holdDownRounds"):
                    for entry in timeline[1:]:
                        for router, mapping in entry["rounds"][-1][
                            "holdDowns"
                        ].items():
                            self.assertEqual(mapping, {}, router)

    def test_equivalence_holds_at_intermediate_faulty_states(self):
        # The plain final rounds are exactly the independently computed
        # clamped shortest paths, in every mode's trajectory.  Hold-down can
        # delay adoption (and pins a currently direct route even when a
        # learned alternative would be shorter -- its documented round
        # rule), so its faulty-state tables are checked against the separate
        # hold-down timeline oracle.  In the three states below every
        # surviving direct link is itself shortest, so plain and hold-down
        # still land on the same tables.
        for events in (
            [{"time": 1, "action": "node-down", "node": "C"}],
            [{"time": 1, "action": "link-down", "from": "B", "to": "D"}],
            [
                {"time": 1, "action": "link-down", "from": "A", "to": "B"},
                {"time": 2, "action": "node-down", "node": "D"},
            ],
        ):
            down_nodes, down_links = fold_events(events)
            shortest = oracle_stable(
                DIAMOND, down_nodes, down_links, infinity_metric=8
            )
            for hold_down in (None, 0, 1, 2, 4):
                with self.subTest(events=events, hold_down=hold_down):
                    scenario = {
                        "infinityMetric": 8,
                        "events": events,
                    }
                    if hold_down is not None:
                        scenario["holdDownRounds"] = hold_down
                    result = replay_dv_scenario(DIAMOND, scenario)
                    final = result["timeline"][-1]["rounds"][-1]["routers"]
                    self.assertEqual(normalize(final), shortest)
                    # The independent hold-aware timeline replay agrees too.
                    want_finals = oracle_dv_entry_finals(
                        DIAMOND, events, 8,
                        hold_down if hold_down else None,
                    )
                    self.assertEqual(
                        normalize(final), want_finals[-1]
                    )

    def test_holddown_pins_a_worse_direct_link_but_plain_prefers_shortest(self):
        # A-F is a direct metric-5 link but the learned A-B-F path costs 2.
        # Plain replay-dv routes strictly by total metric (via B), in every
        # entry.  Hold-down's synchronous rule adopts a current direct
        # neighbor at its direct metric at *any* round: the fault-free
        # baseline is still built from plain rounds (it publishes empty
        # holdDowns maps), but the first event entry runs the hold-down
        # rounds and then pins the direct F/5 -- even for an unrelated
        # failure elsewhere.  This is the pre-refactor documented rule and
        # must be preserved byte for byte.
        topology = {
            "nodes": ["A", "B", "F", "G"],
            "links": [
                link("A", "B", 1),
                link("B", "F", 1),
                link("A", "F", 5),
                link("A", "G", 1),
            ],
        }
        events = [{"time": 1, "action": "node-down", "node": "G"}]
        via_b = {"nextHop": "B", "metric": 2}
        direct_f = {"nextHop": "F", "metric": 5}

        held = replay_dv_scenario(
            topology,
            {"infinityMetric": 45, "holdDownRounds": 1, "events": []},
        )
        # Baseline uses plain rounds: shortest via B.
        self.assertEqual(
            held["timeline"][0]["rounds"][-1]["routers"]["A"]["F"], via_b
        )
        held_event = replay_dv_scenario(
            topology,
            {"infinityMetric": 45, "holdDownRounds": 1, "events": events},
        )
        self.assertEqual(
            held_event["timeline"][1]["rounds"][-1]["routers"]["A"]["F"],
            direct_f,
        )
        # Plain mode never pins: the unrelated failure leaves via B in place.
        plain = replay_dv_scenario(
            topology, {"infinityMetric": 45, "events": events}
        )
        self.assertEqual(
            plain["timeline"][1]["rounds"][-1]["routers"]["A"]["F"], via_b
        )
        # The independent hold-aware timeline replay predicts the same pin.
        want = oracle_dv_entry_finals(topology, events, 45, 1)
        self.assertEqual(
            normalize(held_event["timeline"][1]["rounds"][-1]["routers"]),
            want[-1],
        )


# ---------------------------------------------------------------------------
# Connected convergence through the shared kernel
# ---------------------------------------------------------------------------


class TestConnectedConvergence(unittest.TestCase):
    def test_converge_starts_from_self_and_direct_neighbors(self):
        rounds = converge_topology(CHAIN3)["rounds"]
        self.assertEqual([r["round"] for r in rounds], [0, 1])
        round_zero = rounds[0]["routers"]
        self.assertEqual(round_zero["A"]["A"], {"nextHop": None, "metric": 0})
        self.assertEqual(round_zero["A"]["B"], {"nextHop": "B", "metric": 5})
        self.assertEqual(round_zero["A"]["C"], {"nextHop": None, "metric": None})
        self.assertEqual(round_zero["C"]["A"], {"nextHop": None, "metric": None})

    def test_converge_rounds_follow_one_synchronous_update_each(self):
        topology = RING
        nodes, adjacency = _adjacency(topology)
        rounds = converge_topology(topology)["rounds"]
        snapshots = [normalize(r["routers"]) for r in rounds]
        for previous, current in zip(snapshots, snapshots[1:]):
            self.assertEqual(
                current,
                oracle_synchronous(
                    previous, nodes, adjacency, frozenset(), None
                ),
            )
        # The recorded tail is a fixed point.
        self.assertEqual(
            oracle_synchronous(
                snapshots[-1], nodes, adjacency, frozenset(), None
            ),
            snapshots[-1],
        )

    def test_only_changed_rounds_are_recorded(self):
        for topology in (CHAIN3, DV_CHAIN, DIAMOND, RING, COMPONENTS, EMPTY):
            with self.subTest(topology=topology):
                result = converge_topology(topology)
                rounds = result["rounds"]
                self.assertEqual(
                    [r["round"] for r in rounds], list(range(len(rounds)))
                )
                self.assertEqual(
                    result["convergenceRound"], rounds[-1]["round"]
                )
                for previous, current in zip(rounds, rounds[1:]):
                    self.assertNotEqual(previous["routers"], current["routers"])


# ---------------------------------------------------------------------------
# Disconnected components
# ---------------------------------------------------------------------------


class TestDisconnected(unittest.TestCase):
    def test_edgeless_topology_converges_at_round_zero_everywhere(self):
        results = (
            converge_topology(EDGELESS),
            replay_dv_scenario(EDGELESS, {"infinityMetric": 9, "events": []}),
            replay_dv_scenario(
                EDGELESS,
                {"infinityMetric": 9, "holdDownRounds": 2, "events": []},
            ),
        )
        for result in results:
            rounds = (
                result["rounds"]
                if "rounds" in result
                else result["timeline"][0]["rounds"]
            )
            self.assertEqual(len(rounds), 1)
            # Both shapes point convergenceRound at the single round 0: the
            # converge root carries it directly, replay carries it per entry.
            convergence_round = (
                result["convergenceRound"]
                if "convergenceRound" in result
                else result["timeline"][0]["convergenceRound"]
            )
            self.assertEqual(convergence_round, 0)

    def test_isolated_router_row_is_wholly_unreachable(self):
        expected = oracle_stable(COMPONENTS)
        converged = converge_topology(COMPONENTS)["rounds"][-1]["routers"]
        self.assertEqual(normalize(converged), expected)
        for other in ("a1", "a2", "b1", "b2"):
            self.assertEqual(
                converged["iso"][other], {"nextHop": None, "metric": None}
            )
            self.assertEqual(
                converged[other]["iso"], {"nextHop": None, "metric": None}
            )
        # It stays unreachable in every recorded replay round and in both
        # replay-dv modes after a remote failure.
        scenario = {
            "infinityMetric": 9,
            "events": [{"time": 1, "action": "node-down", "node": "b2"}],
        }
        for with_holddown in (False, True):
            raw = dict(scenario)
            if with_holddown:
                raw["holdDownRounds"] = 2
            timeline = replay_dv_scenario(COMPONENTS, raw)["timeline"]
            for entry in timeline:
                for snapshot in entry["rounds"]:
                    for other in ("a1", "a2", "b1", "b2"):
                        self.assertEqual(
                            snapshot["routers"]["iso"][other],
                            {"nextHop": None, "metric": None},
                        )

    def test_down_router_row_is_unreachable_in_every_mode(self):
        events = [{"time": 1, "action": "node-down", "node": "C"}]
        for scenario in (
            {"infinityMetric": 8, "events": events},
            {"infinityMetric": 8, "holdDownRounds": 2, "events": events},
        ):
            rounds = replay_dv_scenario(DV_CHAIN, scenario)["timeline"][1]["rounds"]
            for snapshot in rounds:
                for dest in ("A", "B", "C"):
                    self.assertEqual(
                        snapshot["routers"]["C"][dest],
                        {"nextHop": None, "metric": None},
                    )


# ---------------------------------------------------------------------------
# Equal-cost paths: one tie rule shared by every path
# ---------------------------------------------------------------------------


class TestEqualCostTieBreak(unittest.TestCase):
    def test_smaller_next_hop_in_all_three_entries(self):
        ties = {
            ("A", "D"): "B",
            ("D", "A"): "B",
            ("B", "C"): "A",
            ("C", "B"): "A",
        }
        stable = {
            "converge": converge_topology(DIAMOND)["rounds"][-1]["routers"],
            "replay-dv": replay_dv_scenario(
                DIAMOND, {"infinityMetric": 50, "events": []}
            )["timeline"][0]["rounds"][-1]["routers"],
            "replay-dv-holddown": replay_dv_scenario(
                DIAMOND,
                {"infinityMetric": 50, "holdDownRounds": 2, "events": []},
            )["timeline"][0]["rounds"][-1]["routers"],
        }
        for label, routers in stable.items():
            for (router, dest), hop in ties.items():
                self.assertEqual(
                    routers[router][dest]["nextHop"], hop, (label, router, dest)
                )

    def test_alternative_after_release_uses_same_tie_kernel(self):
        # A-B fails on the diamond.  A's equal alternative via C is refused
        # throughout the window; the first post-release selection round runs
        # the shared kernel and adopts C.  B (one hop further from the
        # alternative) catches up one round later.
        events = [{"time": 1, "action": "link-down", "from": "B", "to": "D"}]
        scenario = {
            "infinityMetric": 20, "holdDownRounds": 2, "events": events,
        }
        entry = replay_dv_scenario(DIAMOND, scenario)["timeline"][1]
        rounds = entry["rounds"]
        null = {"nextHop": None, "metric": None}

        # Round 0 transitively invalidates the stale A-via-B and D-via-C-less
        # routes: B is still A's neighbor but itself advertises D unreachable.
        self.assertEqual(rounds[0]["routers"]["A"]["D"], null)
        self.assertEqual(rounds[0]["routers"]["B"]["D"], null)
        self.assertEqual(rounds[0]["routers"]["D"]["B"], null)
        self.assertEqual(rounds[0]["holdDowns"]["A"], {"D": 2})
        self.assertEqual(rounds[0]["holdDowns"]["B"], {"D": 2})

        # The surviving C alternative is advertised at round 0 but refused
        # while the window is live and counts down exactly one per round.
        self.assertEqual(rounds[1]["routers"]["A"]["D"], null)
        self.assertEqual(rounds[1]["holdDowns"]["A"], {"D": 1})
        self.assertEqual(rounds[2]["holdDowns"]["A"], {})
        self.assertEqual(rounds[2]["routers"]["A"]["D"], null)
        self.assertEqual(
            rounds[3]["routers"]["A"]["D"], {"nextHop": "C", "metric": 2}
        )
        self.assertEqual(
            rounds[4]["routers"]["B"]["D"], {"nextHop": "A", "metric": 3}
        )
        # Stable tables match the independent oracle for the failed state.
        down_nodes, down_links = fold_events(events)
        expected = oracle_stable(
            DIAMOND, down_nodes, down_links, infinity_metric=20
        )
        self.assertEqual(normalize(rounds[-1]["routers"]), expected)
        self.assertEqual(entry["convergenceRound"], rounds[-1]["round"])


# ---------------------------------------------------------------------------
# Low infinity threshold
# ---------------------------------------------------------------------------


class TestInfinityThreshold(unittest.TestCase):
    def test_tight_threshold_truncates_multihop_routes_in_both_replay_modes(self):
        # Infinity 11 is legal against metric-10 links but kills 2-hop routes.
        expected = oracle_stable(
            LONG_METRIC_CHAIN, infinity_metric=11
        )
        for scenario in (
            {"infinityMetric": 11, "events": []},
            {"infinityMetric": 11, "holdDownRounds": 0, "events": []},
            {"infinityMetric": 11, "holdDownRounds": 2, "events": []},
        ):
            result = replay_dv_scenario(LONG_METRIC_CHAIN, scenario)
            final = result["timeline"][0]["rounds"][-1]["routers"]
            self.assertEqual(normalize(final), expected)
        # converge has no threshold: it still publishes the full paths.
        converged = converge_topology(
            LONG_METRIC_CHAIN
        )["rounds"][-1]["routers"]
        self.assertEqual(
            converged["n0"]["n3"], {"nextHop": "n1", "metric": 30}
        )

    def test_threshold_terminates_failure_convergence_early(self):
        scenario = {
            "infinityMetric": 3,
            "events": [{"time": 1, "action": "node-down", "node": "C"}],
        }
        for with_holddown in (False, True):
            raw = dict(scenario)
            if with_holddown:
                raw["holdDownRounds"] = 2
            entry = replay_dv_scenario(DV_CHAIN, raw)["timeline"][1]
            final = entry["rounds"][-1]["routers"]
            self.assertEqual(
                final["A"]["C"], {"nextHop": None, "metric": None}
            )
            self.assertEqual(
                final["B"]["C"], {"nextHop": None, "metric": None}
            )
            expected = oracle_stable(
                DV_CHAIN, frozenset("C"), infinity_metric=3
            )
            self.assertEqual(normalize(final), expected)

    def test_candidate_at_threshold_is_unreachable_not_adopted(self):
        # A-B=1, B-C=1: with infinity 2, B reaches C directly at metric 1,
        # but A's candidate via B totals 2 == infinity and must be refused.
        result = replay_dv_scenario(
            DV_CHAIN, {"infinityMetric": 2, "events": []}
        )
        final = result["timeline"][0]["rounds"][-1]["routers"]
        self.assertEqual(final["B"]["C"], {"nextHop": "C", "metric": 1})
        self.assertEqual(final["A"]["C"], {"nextHop": None, "metric": None})

    def test_plain_replay_rounds_follow_one_clamped_synchronous_step(self):
        topology = RING
        nodes, baseline_active = _adjacency(topology)
        events = [{"time": 1, "action": "link-down", "from": "A", "to": "B"}]
        down_nodes, down_links = fold_events(events)
        _down, event_active = _adjacency(topology, down_nodes, down_links)
        timeline = replay_dv_scenario(
            topology, {"infinityMetric": 7, "events": events}
        )["timeline"]
        for index, entry in enumerate(timeline):
            active = baseline_active if index == 0 else event_active
            downs = frozenset() if index == 0 else down_nodes
            normalized = [normalize(s["routers"]) for s in entry["rounds"]]
            for previous, current in zip(normalized, normalized[1:]):
                self.assertEqual(
                    current,
                    oracle_synchronous(
                        previous, nodes, active, downs, 7
                    ),
                    (index,),
                )
            # The recorded tail is itself a clamped fixed point.
            self.assertEqual(
                oracle_synchronous(
                    normalized[-1], nodes, active, downs, 7
                ),
                normalized[-1],
                (index,),
            )


# ---------------------------------------------------------------------------
# Consecutive failures and recoveries with vector inheritance
# ---------------------------------------------------------------------------


RECOVERY_SCENARIO_EVENTS = [
    {"time": 1, "action": "node-down", "node": "B"},
    {"time": 2, "action": "node-down", "node": "C"},
    {"time": 3, "action": "link-down", "from": "A", "to": "B"},
    {"time": 4, "action": "node-up", "node": "C"},
    {"time": 5, "action": "node-up", "node": "B"},
    {"time": 6, "action": "node-down", "node": "C"},
    {"time": 7, "action": "node-up", "node": "C"},
    {"time": 8, "action": "link-up", "from": "B", "to": "A"},
]


class TestConsecutiveFailureRecovery(unittest.TestCase):
    def test_final_state_returns_to_baseline_in_every_mode(self):
        baseline = oracle_stable(CHAIN3)
        for hold_down in (None, 0, 1, 3):
            scenario = {"infinityMetric": 50,
                        "events": RECOVERY_SCENARIO_EVENTS}
            if hold_down is not None:
                scenario["holdDownRounds"] = hold_down
            result = replay_dv_scenario(CHAIN3, scenario)
            self.assertEqual(len(result["timeline"]), 9)
            final = result["timeline"][-1]["rounds"][-1]["routers"]
            self.assertEqual(normalize(final), baseline)

    def test_every_entry_matches_the_independent_oracle(self):
        rng = random.Random(2024)
        for case in range(60):
            topology, events, infinity_metric = _random_legal_case(rng)
            down_nodes, down_links = fold_events(events)
            final_shortest = oracle_stable(
                topology, down_nodes, down_links, infinity_metric
            )
            for hold_down in (None, 1, rng.choice([2, 3, 5])):
                scenario = {
                    "infinityMetric": infinity_metric,
                    "events": events,
                }
                if hold_down is not None:
                    scenario["holdDownRounds"] = hold_down
                result = replay_dv_scenario(topology, scenario)
                timeline = result["timeline"]
                # The independent per-entry timeline replay predicts the
                # fixed point of every single entry: plain clamped BF rounds,
                # or plain baseline rounds plus hold-down event rounds (which
                # may pin a currently direct but worse route).
                want_finals = oracle_dv_entry_finals(
                    topology, events, infinity_metric, hold_down
                )
                self.assertEqual(len(timeline), len(want_finals), case)
                for entry, want in zip(timeline, want_finals):
                    self.assertEqual(
                        normalize(entry["rounds"][-1]["routers"]),
                        want, (case, hold_down, entry["event"]),
                    )
                if hold_down is None:
                    # Plain mode's final entry is exactly the clamped
                    # shortest-path table of the final availability state.
                    self.assertEqual(
                        normalize(timeline[-1]["rounds"][-1]["routers"]),
                        final_shortest, case,
                    )
                rounds = timeline[-1]["rounds"]
                self.assertEqual(
                    [r["round"] for r in rounds],
                    list(range(len(rounds))),
                )
                for previous, current in zip(rounds, rounds[1:]):
                    if hold_down is None:
                        # Plain mode records a round only when a route
                        # actually changes.
                        self.assertNotEqual(
                            previous["routers"], current["routers"]
                        )
                    else:
                        # Hold-down mode additionally records a round in
                        # which only the countdown map moves.
                        self.assertTrue(
                            previous["routers"] != current["routers"]
                            or previous["holdDowns"]
                            != current["holdDowns"]
                        )
                if hold_down is not None:
                    # Convergence exhausts every timer before the next event.
                    for mapping in rounds[-1]["holdDowns"].values():
                        self.assertEqual(mapping, {})
            # converge stays the clamp-free answer for the same topology.
            self.assertEqual(
                normalize(converge_topology(topology)["rounds"][-1]["routers"]),
                oracle_stable(topology),
            )

    def test_recovered_link_is_directly_advertised_at_round_zero(self):
        events = [
            {"time": 1, "action": "link-down", "from": "B", "to": "A"},
            {"time": 2, "action": "link-up", "from": "A", "to": "B"},
        ]
        for hold_down in (None, 2):
            scenario = {"infinityMetric": 8, "events": events}
            if hold_down is not None:
                scenario["holdDownRounds"] = hold_down
            round_zero = replay_dv_scenario(
                CHAIN3, scenario
            )["timeline"][2]["rounds"][0]["routers"]
            self.assertEqual(
                round_zero["A"]["B"], {"nextHop": "B", "metric": 5}
            )
            self.assertEqual(
                round_zero["B"]["A"], {"nextHop": "A", "metric": 5}
            )


def _random_legal_case(rng):
    """Generate one topology plus a legal, independently tracked event list."""
    n = rng.randint(1, 6)
    nodes = [chr(ord("A") + i) for i in range(n)]
    pairs = [(nodes[i], nodes[j]) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(pairs)
    links = [
        {"from": a, "to": b, "metric": rng.randint(1, 5)}
        for a, b in pairs[: rng.randint(0, len(pairs))]
    ]
    topology = {"nodes": nodes, "links": links}
    down_nodes, down_links = set(), set()
    events = []
    time = 0
    for _ in range(rng.randint(0, 7)):
        options = []
        for node in nodes:
            options.append(
                ("node-up" if node in down_nodes else "node-down", node, None)
            )
        for edge in links:
            a, b = edge["from"], edge["to"]
            pair = (a, b) if a < b else (b, a)
            options.append(
                ("link-up" if pair in down_links else "link-down", a, b)
            )
        action, a, b = rng.choice(options)
        time += rng.randint(1, 3)
        if action.startswith("node"):
            events.append({"time": time, "action": action, "node": a})
            (down_nodes.add if action == "node-down"
             else down_nodes.discard)(a)
        else:
            events.append(
                {"time": time, "action": action, "from": a, "to": b}
            )
            pair = (a, b) if a < b else (b, a)
            (down_links.add if action == "link-down"
             else down_links.discard)(pair)
    max_metric = max((edge["metric"] for edge in links), default=0)
    return topology, events, max_metric + rng.choice([1, 2, 6, 40])


# ---------------------------------------------------------------------------
# Hold-down specifics layered on the shared kernel
# ---------------------------------------------------------------------------


class TestHoldDownSemantics(unittest.TestCase):
    def test_round_zero_opens_timers_only_for_lost_finite_routes(self):
        events = [{"time": 1, "action": "node-down", "node": "C"}]
        scenario = {
            "infinityMetric": 8, "holdDownRounds": 2, "events": events,
        }
        round_zero = replay_dv_scenario(
            DV_CHAIN, scenario
        )["timeline"][1]["rounds"][0]
        self.assertEqual(round_zero["holdDowns"]["A"], {"C": 2})
        self.assertEqual(round_zero["holdDowns"]["B"], {"C": 2})
        self.assertEqual(round_zero["holdDowns"]["C"], {})
        # Self and surviving direct neighbors never carry a timer.
        self.assertEqual(
            round_zero["routers"]["A"]["B"], {"nextHop": "B", "metric": 1}
        )

    def test_direct_recovery_clears_timer_at_once(self):
        events = [
            {"time": 1, "action": "node-down", "node": "C"},
            {"time": 2, "action": "node-up", "node": "C"},
        ]
        scenario = {
            "infinityMetric": 50, "holdDownRounds": 3, "events": events,
        }
        recovery_zero = replay_dv_scenario(
            CHAIN3, scenario
        )["timeline"][2]["rounds"][0]
        self.assertEqual(
            recovery_zero["routers"]["B"]["C"], {"nextHop": "C", "metric": 7}
        )
        self.assertEqual(recovery_zero["holdDowns"]["B"], {})

    def test_later_invalidation_starts_a_fresh_window(self):
        # B-D fails (B's direct route to D is timed); one round later nothing
        # extends the window, and repeated null advertisements leave the
        # countdown strictly monotone even though no alternative exists for B.
        events = [{"time": 1, "action": "link-down", "from": "B", "to": "D"}]
        scenario = {
            "infinityMetric": 20, "holdDownRounds": 3, "events": events,
        }
        rounds = replay_dv_scenario(
            DIAMOND, scenario
        )["timeline"][1]["rounds"]
        self.assertEqual(rounds[0]["holdDowns"]["B"], {"D": 3})
        self.assertEqual(rounds[1]["holdDowns"]["B"], {"D": 2})
        self.assertEqual(rounds[2]["holdDowns"]["B"], {"D": 1})
        self.assertEqual(rounds[3]["holdDowns"]["B"], {})
        # Window published while held; gone the round the count reaches zero.
        for index in range(3):
            self.assertEqual(
                rounds[index]["routers"]["B"]["D"],
                {"nextHop": None, "metric": None},
            )

    def test_timers_and_vectors_must_both_be_stable_to_converge(self):
        # The last recorded round has empty maps and is itself a fixed point:
        # another synchronous hold-down update changes nothing.
        from packet_routing_sim.core.routing import distance_vector_holddown_round

        events = [{"time": 1, "action": "link-down", "from": "A", "to": "B"}]
        scenario = {
            "infinityMetric": 8, "holdDownRounds": 2, "events": events,
        }
        entry = replay_dv_scenario(DIAMOND, scenario)["timeline"][1]
        final = entry["rounds"][-1]
        nodes, active = _adjacency(
            DIAMOND, frozenset(), {("A", "B")}
        )
        empty = {node: {} for node in nodes}
        again_vectors, again_timers = distance_vector_holddown_round(
            final["routers"], nodes, active, frozenset(), 8, empty, 2
        )
        self.assertEqual(again_vectors, final["routers"])
        self.assertEqual(again_timers, empty)
        self.assertEqual(entry["convergenceRound"], final["round"])

    def test_held_destination_advertises_null_in_every_live_round(self):
        events = [{"time": 1, "action": "node-down", "node": "C"}]
        scenario = {
            "infinityMetric": 8, "holdDownRounds": 2, "events": events,
        }
        rounds = replay_dv_scenario(
            DV_CHAIN, scenario
        )["timeline"][1]["rounds"]
        for snapshot in rounds[:-1]:
            for router, mapping in snapshot["holdDowns"].items():
                for dest in mapping:
                    self.assertEqual(
                        snapshot["routers"][router][dest],
                        {"nextHop": None, "metric": None},
                    )


# ---------------------------------------------------------------------------
# Published surface stays exactly as documented (no new fields)
# ---------------------------------------------------------------------------


class TestPublishedShape(unittest.TestCase):
    def test_converge_document_shape(self):
        result = converge_topology(CHAIN3)
        self.assertEqual(
            set(result), {"protocol", "convergenceRound", "rounds"}
        )
        self.assertEqual(result["protocol"], "distance-vector")
        for snapshot in result["rounds"]:
            self.assertEqual(set(snapshot), {"round", "routers"})

    def test_omitted_and_zero_holddown_are_byte_compatible(self):
        events = [{"time": 1, "action": "node-down", "node": "C"}]
        legacy = replay_dv_scenario(
            DV_CHAIN, {"infinityMetric": 8, "events": events}
        )
        explicit_zero = replay_dv_scenario(
            DV_CHAIN,
            {"infinityMetric": 8, "holdDownRounds": 0, "events": events},
        )
        self.assertEqual(legacy, explicit_zero)
        self.assertNotIn("holdDownRounds", legacy)
        for entry in legacy["timeline"]:
            for snapshot in entry["rounds"]:
                self.assertNotIn("holdDowns", snapshot)

    def test_positive_holddown_shape_contains_only_the_documented_maps(self):
        scenario = {
            "infinityMetric": 8, "holdDownRounds": 2,
            "events": [{"time": 1, "action": "node-down", "node": "C"}],
        }
        result = replay_dv_scenario(DV_CHAIN, scenario)
        self.assertEqual(
            set(result),
            {"protocol", "infinityMetric", "holdDownRounds", "timeline"},
        )
        self.assertEqual(result["holdDownRounds"], 2)
        for entry in result["timeline"]:
            self.assertEqual(
                set(entry), {"event", "convergenceRound", "rounds"}
                if "event" in entry and "time" not in entry
                else {"time", "event", "convergenceRound", "rounds"}
            )
            for snapshot in entry["rounds"]:
                self.assertEqual(
                    set(snapshot), {"round", "routers", "holdDowns"}
                )
                self.assertEqual(
                    set(snapshot["holdDowns"]), set(result["timeline"][0]
                                                    ["rounds"][0]["routers"])
                )


# ---------------------------------------------------------------------------
# Validation precedence and the distinguishable error types
# ---------------------------------------------------------------------------


class TestValidationContract(unittest.TestCase):
    def test_error_hierarchy_remains_distinguishable(self):
        self.assertTrue(issubclass(InvalidStateTransition, InvalidScenario))
        self.assertFalse(issubclass(InvalidStateTransition, InvalidTopology))

    def test_converge_reports_topology_errors_as_invalid_topology(self):
        bad_topologies = (
            {"nodes": ["A", "A"], "links": []},
            {"nodes": [], "links": [{"from": "A", "to": "B", "metric": 1}]},
            {"nodes": "A", "links": []},
        )
        for bad in bad_topologies:
            with self.assertRaises(InvalidTopology):
                converge_topology(bad)
            # A topology defect must never surface as an InvalidScenario.
            try:
                converge_topology(bad)
            except InvalidScenario:
                self.fail("topology defect surfaced as InvalidScenario")
            except InvalidTopology:
                pass

    def test_replay_dv_scenario_validation_order(self):
        # Topology problems win even when the scenario is also unusable.
        with self.assertRaises(InvalidTopology):
            replay_dv_scenario(
                {"nodes": ["A", "A"], "links": []},
                {"infinityMetric": 1},
            )
        with self.assertRaises(InvalidScenario):
            replay_dv_scenario(CHAIN3, {})
        with self.assertRaises(InvalidScenario):
            replay_dv_scenario(
                CHAIN3, {"infinityMetric": 5, "events": []}
            )
        with self.assertRaises(InvalidStateTransition):
            replay_dv_scenario(
                CHAIN3,
                {"infinityMetric": 8,
                 "events": [{"time": 1, "action": "node-up", "node": "A"}]},
            )

    def test_cli_status_codes_and_stderr_text_are_unchanged(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get(
            "PYTHONPATH", ""
        )
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            def run(*args):
                return subprocess.run(
                    [sys.executable, "-m", "packet_routing_sim", *args],
                    cwd=REPO_ROOT, env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    timeout=60,
                )

            bad_topo = tmp_path / "bad.json"
            bad_topo.write_text(json.dumps({"nodes": ["A", "A"], "links": []}))
            proc = run("converge", str(bad_topo))
            self.assertEqual(proc.returncode, 2)
            self.assertEqual(proc.stdout, b"")
            self.assertEqual(proc.stderr, b"invalid topology\n")

            topo = tmp_path / "topo.json"
            topo.write_text(json.dumps(CHAIN3))

            bad_inf = tmp_path / "bad_inf.json"
            bad_inf.write_text(json.dumps({"infinityMetric": 1, "events": []}))
            proc = run("replay-dv", str(topo), str(bad_inf))
            self.assertEqual(proc.returncode, 2)
            self.assertEqual(proc.stdout, b"")
            self.assertEqual(proc.stderr, b"invalid scenario\n")

            bad_transition = tmp_path / "bad_transition.json"
            bad_transition.write_text(json.dumps(
                {"infinityMetric": 8,
                 "events": [{"time": 1, "action": "node-up", "node": "A"}]}
            ))
            proc = run("replay-dv", str(topo), str(bad_transition))
            self.assertEqual(proc.returncode, 2)
            self.assertEqual(proc.stderr, b"invalid scenario\n")

            bad_holddown = tmp_path / "bad_holddown.json"
            bad_holddown.write_text(json.dumps(
                {"infinityMetric": 8, "holdDownRounds": True, "events": []}
            ))
            proc = run("replay-dv", str(topo), str(bad_holddown))
            self.assertEqual(proc.returncode, 2)
            self.assertEqual(proc.stderr, b"invalid scenario\n")


# ---------------------------------------------------------------------------
# Purity: inputs immutable, outputs isolated
# ---------------------------------------------------------------------------


class TestKernelPurity(unittest.TestCase):
    SCENARIO_EVENTS = [
        {"time": 1, "action": "link-down", "from": "A", "to": "B"},
        {"time": 2, "action": "node-down", "node": "C"},
        {"time": 3, "action": "node-up", "node": "C"},
        {"time": 4, "action": "link-up", "from": "B", "to": "A"},
    ]

    def _scenarios(self):
        return [
            {"infinityMetric": 30, "events": self.SCENARIO_EVENTS},
            {"infinityMetric": 30, "holdDownRounds": 0,
             "events": self.SCENARIO_EVENTS},
            {"infinityMetric": 30, "holdDownRounds": 2,
             "events": self.SCENARIO_EVENTS},
        ]

    def test_inputs_are_never_modified(self):
        for topology in (DIAMOND, CHAIN3):
            before = copy.deepcopy(topology)
            converge_topology(topology)
            self.assertEqual(topology, before)
        for scenario in self._scenarios():
            topology_before = copy.deepcopy(DIAMOND)
            scenario_before = copy.deepcopy(scenario)
            replay_dv_scenario(DIAMOND, scenario)
            self.assertEqual(DIAMOND, topology_before)
            self.assertEqual(scenario, scenario_before)

    def test_distinct_null_entries_and_rounds_never_share_objects(self):
        result = converge_topology(DIAMOND)
        rounds = result["rounds"]
        # Equal null/null cells are separate objects, both within a round...
        first_round = rounds[0]["routers"]
        self.assertIsNot(
            first_round["A"]["C"], first_round["A"]["D"]
        )
        # ...and across adjacent rounds even where the value is unchanged.
        if len(rounds) > 1:
            self.assertIsNot(
                rounds[0]["routers"]["A"]["A"],
                rounds[1]["routers"]["A"]["A"],
            )

    def test_timeline_entries_do_not_alias_each_other(self):
        scenario = {
            "infinityMetric": 30, "holdDownRounds": 2,
            "events": self.SCENARIO_EVENTS,
        }
        timeline = replay_dv_scenario(DIAMOND, scenario)["timeline"]
        baseline = timeline[0]["rounds"][-1]["routers"]
        after_event = timeline[1]["rounds"][0]["routers"]
        for router in baseline:
            for dest in baseline[router]:
                self.assertIsNot(
                    baseline[router][dest], after_event[router][dest]
                )

    def test_mutating_a_result_cannot_reach_a_later_call(self):
        scenario = {
            "infinityMetric": 30, "holdDownRounds": 2,
            "events": [{"time": 1, "action": "node-down", "node": "C"}],
        }
        first = replay_dv_scenario(DIAMOND, scenario)
        first["timeline"][1]["rounds"][0]["routers"]["A"]["C"] = "HACK"
        first["timeline"][1]["rounds"][0]["holdDowns"]["A"]["Z"] = 99
        second = replay_dv_scenario(DIAMOND, copy.deepcopy(scenario))
        snapshot = second["timeline"][1]["rounds"][0]
        self.assertNotEqual(snapshot["routers"]["A"]["C"], "HACK")
        self.assertNotIn("Z", snapshot["holdDowns"]["A"])

        first_converge = converge_topology(DIAMOND)
        first_converge["rounds"][0]["routers"]["A"] = "HACK"
        again = converge_topology(DIAMOND)
        self.assertNotEqual(again["rounds"][0]["routers"]["A"], "HACK")


# ---------------------------------------------------------------------------
# Declaration-order and hash-seed stability for every DV entry
# ---------------------------------------------------------------------------


_DETERMINISM_SNIPPET = """
import json, os, sys
sys.path.insert(0, %r)
from packet_routing_sim.core import converge_topology, replay_dv_scenario
topology = json.loads(os.environ["PRSIM_TOPO"])
scenario = json.loads(os.environ["PRSIM_SCEN"])
events = scenario["events"]
out = {
    "converge": converge_topology(topology),
    "dv_legacy": replay_dv_scenario(
        topology, {"infinityMetric": scenario["infinityMetric"],
                   "events": events}),
    "dv_zero": replay_dv_scenario(
        topology, {"infinityMetric": scenario["infinityMetric"],
                   "holdDownRounds": 0, "events": events}),
    "dv_hold": replay_dv_scenario(topology, scenario),
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
    TOPOLOGY = RING
    SCENARIO = {
        "infinityMetric": 30,
        "holdDownRounds": 2,
        "events": [
            {"time": 1, "action": "link-down", "from": "A", "to": "B"},
            {"time": 2, "action": "node-down", "node": "C"},
            {"time": 3, "action": "node-up", "node": "C"},
            {"time": 4, "action": "link-up", "from": "B", "to": "A"},
            {"time": 5, "action": "link-down", "from": "C", "to": "D"},
            {"time": 6, "action": "link-up", "from": "D", "to": "C"},
        ],
    }
    VARIANTS = [
        TOPOLOGY,
        {"nodes": ["D", "C", "B", "A"], "links": TOPOLOGY["links"]},
        {"nodes": list("ABCD"),
         "links": [
             link("B", "A", 1), link("C", "B", 2),
             link("D", "C", 3), link("A", "D", 4),
         ]},
        {"nodes": list("ABCD"),
         "links": list(reversed(TOPOLOGY["links"]))},
    ]

    def test_byte_identical_across_seeds_and_declaration_orders(self):
        outputs = set()
        for seed in HASH_SEEDS:
            for variant in self.VARIANTS:
                outputs.add(run_in_subprocess(variant, self.SCENARIO, seed))
        self.assertEqual(len(outputs), 1)


if __name__ == "__main__":
    unittest.main()
