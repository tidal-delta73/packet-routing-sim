"""Regression tests for the optional replay-dv poison-reverse mode.

Poison reverse is a single optional boolean on the ``replay-dv`` scenario:
``poisonReverse``.  These tests pin the contract from the spec:

* omitted or ``false`` takes the legacy path byte-for-byte and the root
  object gains no field; ``true`` echoes ``poisonReverse: true`` at the root
  and changes only how each synchronous exchange's advertisements are
  generated -- the local routing table still stores the selected next hop
  and metric, but a sender advertises a destination as unreachable only to
  the neighbor that is itself the selected next hop for it;
* the self/direct round 0 (baseline and post-event immediate invalidation,
  node-recovery initialization and link-recovery direct adoption) is
  unchanged; poison reverse takes effect from the next synchronous
  exchange; rounds are recorded only while they change and
  ``convergenceRound`` points at the stable round;
* poison reverse combines with ``holdDownRounds``: receiver-specific
  advertisements feed the existing invalidation, countdown, direct-clear
  and suppression semantics, and convergence needs both surfaces stable;
* a non-boolean ``poisonReverse`` is the core's sole InvalidScenario and
  the CLI's sole ``invalid scenario`` / status 2; topology precedence, the
  other replay entries, ``compute`` and ``converge`` are unaffected;
* results stay independent of declaration order, link direction, link
  ordering and the Python hash seed; inputs are never mutated and repeated
  calls share no protocol state.

Every expected trajectory is produced by an independently written
synchronous Bellman-Ford oracle in this file (per-receiver advertisements
included), never by reading the implementation's internals.
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
    InvalidTopology,
    compute_topology,
    converge_topology,
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


DV_CHAIN = {
    "nodes": ["A", "B", "C"],
    "links": [link("A", "B", 1), link("B", "C", 1)],
}
# Triangle with a slow direct A-C edge: the textbook count-to-infinity case
# after C goes down.
TRIANGLE = {
    "nodes": ["A", "B", "C"],
    "links": [link("A", "B", 1), link("B", "C", 1), link("A", "C", 10)],
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


# ---------------------------------------------------------------------------
# Independent synchronous oracle with per-receiver poison reverse
# ---------------------------------------------------------------------------
#
# A view is {(router, destination): (nextHop, metric)}.  Each update reads a
# neighbor's advertisement exactly as that neighbor generated it for the
# receiver: with poison reverse a selected route is poisoned (null) back at
# the very neighbor it routes through.


def _adjacency(topology, down_nodes=frozenset(), down_links=frozenset()):
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


def _seen_metric(view, neighbor, destination, receiver, poison):
    """The metric ``neighbor`` advertised to ``receiver`` for ``destination``."""
    hop, metric = view[neighbor, destination]
    if poison and hop == receiver:
        return None
    return metric


def _sync_step(view, nodes, active, down, infinity, poison):
    """One clamped synchronous Bellman-Ford round under the oracle's rules."""
    current = {}
    for router in nodes:
        if router in down:
            current.update({(router, d): (None, None) for d in nodes})
            continue
        for dest in nodes:
            if dest == router:
                current[router, dest] = (None, 0)
                continue
            best_cost, best_hop = None, None
            for neighbor in sorted(active[router]):
                advertised = _seen_metric(
                    view, neighbor, dest, router, poison
                )
                if advertised is None:
                    continue
                cost = active[router][neighbor] + advertised
                if infinity is not None and cost >= infinity:
                    continue
                if best_cost is None or cost < best_cost:
                    best_cost, best_hop = cost, neighbor
            current[router, dest] = (
                (None, None) if best_cost is None else (best_hop, best_cost)
            )
    return current


def _direct_only(nodes, active, down):
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


def _failure_round_zero(converged, nodes, active, down):
    round_zero = {}
    for router in nodes:
        if router in down:
            round_zero.update({(router, d): (None, None) for d in nodes})
            continue
        for dest in nodes:
            if dest == router:
                round_zero[router, dest] = (None, 0)
            elif dest in active[router]:
                round_zero[router, dest] = (dest, active[router][dest])
            else:
                hop, metric = converged[router, dest]
                round_zero[router, dest] = (
                    (hop, metric) if hop in active[router] else (None, None)
                )
    return round_zero


def _recorded_plain(round_zero, nodes, active, down, infinity, poison):
    """Record round 0 and each later changed plain round to a fixed point."""
    recorded = [round_zero]
    view = round_zero
    while True:
        updated = _sync_step(view, nodes, active, down, infinity, poison)
        if updated == view:
            return recorded
        view = updated
        recorded.append(view)


def _holddown_round_zero_vectors(round_zero, nodes, active, down):
    current = round_zero
    while True:
        updated = {}
        for router in nodes:
            if router in down:
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
            return current
        current = updated


def _public_holddowns(live, nodes):
    return {
        router: {
            dest: live[router, dest]
            for dest in sorted(d for r, d in live if r == router)
        }
        for router in nodes
    }


def _hold_step(view, live, nodes, active, down, infinity, poison, hold_down):
    next_view, next_live = {}, {}
    for router in nodes:
        if router in down:
            next_view.update({(router, d): (None, None) for d in nodes})
            continue
        for dest in nodes:
            if dest == router:
                next_view[router, dest] = (None, 0)
                continue
            if dest in active[router]:
                next_view[router, dest] = (dest, active[router][dest])
                continue
            remaining = live.get((router, dest))
            selected = view[router, dest][0]
            if selected is not None:
                advertised = _seen_metric(
                    view, selected, dest, router, poison
                )
                usable = (
                    advertised is not None
                    and active[router][selected] + advertised < infinity
                )
            else:
                usable = False
            if remaining is None and selected is not None and not usable:
                next_view[router, dest] = (None, None)
                next_live[router, dest] = hold_down
            elif remaining is not None:
                next_view[router, dest] = (None, None)
                if remaining - 1 > 0:
                    next_live[router, dest] = remaining - 1
            else:
                best_cost, best_hop = None, None
                for neighbor in sorted(active[router]):
                    advertised = _seen_metric(
                        view, neighbor, dest, router, poison
                    )
                    if advertised is None:
                        continue
                    cost = active[router][neighbor] + advertised
                    if cost >= infinity:
                        continue
                    if best_cost is None or cost < best_cost:
                        best_cost, best_hop = cost, neighbor
                next_view[router, dest] = (
                    (None, None) if best_cost is None
                    else (best_hop, best_cost)
                )
    return next_view, next_live


def _recorded_holddown(
    round_zero, live_zero, nodes, active, down, infinity, poison, hold_down
):
    """Record hold-down rounds while vectors OR timers move."""
    views = [round_zero]
    timers = [_public_holddowns(live_zero, nodes)]
    view, live = round_zero, dict(live_zero)
    while True:
        next_view, next_live = _hold_step(
            view, live, nodes, active, down, infinity, poison, hold_down
        )
        next_public = _public_holddowns(next_live, nodes)
        if next_view == view and next_public == timers[-1]:
            return views, timers
        view, live = next_view, next_live
        views.append(view)
        timers.append(next_public)


def oracle_replay(topology, events, infinity, poison=False, hold_down=None):
    """Full expected replay: one dict per timeline entry.

    Each entry holds ``"views"`` (recorded normalized rounds) and, when
    hold-down is on, ``"timers"`` (published hold-down maps per recorded
    round).  The baseline builds from a direct-only round 0 -- emulating the
    documented replay-dv rule, under which even the hold-down mode's
    fault-free baseline runs plain rounds carrying empty hold-down maps;
    event entries inherit the previous converged view, build failure round
    zero (with the transitive hold-down closure when hold-down is positive),
    then iterate.
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

    down, disabled = set(), set()
    active = fold_state(down, disabled)
    seed = _direct_only(nodes, active, frozenset())
    if hold_down is None:
        baseline_views = _recorded_plain(
            seed, nodes, active, frozenset(), infinity, poison
        )
        entries = [{"views": baseline_views, "timers": None}]
    else:
        # The fault-free hold-down baseline is the plain poisoned/plain
        # exchange with empty published hold-down maps on every round.
        baseline_views = _recorded_plain(
            seed, nodes, active, frozenset(), infinity, poison
        )
        entries = [{
            "views": baseline_views,
            "timers": [
                {router: {} for router in nodes}
                for _ in baseline_views
            ],
        }]
    converged = entries[-1]["views"][-1]

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
        round_zero = _failure_round_zero(converged, nodes, active, downs)

        if hold_down is None:
            recorded = _recorded_plain(
                round_zero, nodes, active, downs, infinity, poison
            )
            entries.append({"views": recorded, "timers": None})
            converged = recorded[-1]
            continue

        closed = _holddown_round_zero_vectors(round_zero, nodes, active, downs)
        live_zero = {}
        for router in nodes:
            if router in downs:
                continue
            for dest in nodes:
                if dest == router or dest in active[router]:
                    continue
                if (converged[router, dest][0] is not None
                        and closed[router, dest][1] is None):
                    live_zero[router, dest] = hold_down
        views, timers = _recorded_holddown(
            closed, live_zero, nodes, active, downs, infinity, poison,
            hold_down,
        )
        entries.append({"views": views, "timers": timers})
        converged = views[-1]
    return entries


def normalize(routers):
    return {
        (router, dest): (entry["nextHop"], entry["metric"])
        for router, row in routers.items()
        for dest, entry in row.items()
    }


# ---------------------------------------------------------------------------
# The whole poison-reverse trajectory matches the independent oracle
# ---------------------------------------------------------------------------


def _random_legal_case(rng):
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


class TestTrajectoryMatchesOracle(unittest.TestCase):
    def assert_entry_matches_oracle(
        self, topology, events, infinity, *, poison, hold_down
    ):
        scenario = {"infinityMetric": infinity, "events": events}
        if hold_down is not None:
            scenario["holdDownRounds"] = hold_down
        if poison:
            scenario["poisonReverse"] = True
        result = replay_dv_scenario(topology, scenario)
        expected_entries = oracle_replay(
            topology, events, infinity, poison=poison, hold_down=hold_down
        )
        timeline = result["timeline"]
        self.assertEqual(len(timeline), len(expected_entries))
        for index, (entry, expected) in enumerate(
            zip(timeline, expected_entries)
        ):
            snapshots = entry["rounds"]
            exp_views = expected["views"]
            exp_timers = expected["timers"]
            held = exp_timers is not None
            self.assertEqual(
                [s["round"] for s in snapshots],
                list(range(len(snapshots))),
                index,
            )
            self.assertEqual(len(snapshots), len(exp_views), index)
            for number, (snapshot, want) in enumerate(zip(snapshots, exp_views)):
                self.assertEqual(
                    normalize(snapshot["routers"]), want, (index, number)
                )
                if held:
                    self.assertEqual(
                        snapshot["holdDowns"], exp_timers[number],
                        (index, number),
                    )
            self.assertEqual(entry["convergenceRound"], snapshots[-1]["round"])

    def test_random_plain_poison_trajectories(self):
        rng = random.Random(2026)
        for case in range(80):
            topology, events, infinity = _random_legal_case(rng)
            self.assert_entry_matches_oracle(
                topology, events, infinity, poison=True, hold_down=None
            )
            self.assert_entry_matches_oracle(
                topology, events, infinity, poison=False, hold_down=None
            )

    def test_random_poison_plus_holddown_trajectories(self):
        rng = random.Random(777)
        for case in range(80):
            topology, events, infinity = _random_legal_case(rng)
            for hold_down in (1, rng.choice([2, 3, 5])):
                self.assert_entry_matches_oracle(
                    topology, events, infinity,
                    poison=True, hold_down=hold_down,
                )
                self.assert_entry_matches_oracle(
                    topology, events, infinity,
                    poison=False, hold_down=hold_down,
                )

    def test_poison_collapses_the_count_to_infinity_loop(self):
        # The textbook triangle: with C down, A and B count to infinity in
        # the legacy run; poison reverse flushes both routes in one exchange.
        events = [{"time": 1, "action": "node-down", "node": "C"}]
        scenario = {"infinityMetric": 100, "events": events}
        legacy = replay_dv_scenario(TRIANGLE, scenario)["timeline"][1]
        poisoned = replay_dv_scenario(
            TRIANGLE, {**scenario, "poisonReverse": True}
        )["timeline"][1]
        self.assertGreater(legacy["convergenceRound"], 50)
        self.assertEqual(poisoned["convergenceRound"], 1)
        null = {"nextHop": None, "metric": None}
        # Round 0 is identical: A still holds the stale metric-2 route via B.
        self.assertEqual(
            poisoned["rounds"][0]["routers"]["A"]["C"],
            {"nextHop": "B", "metric": 2},
        )
        self.assertEqual(
            poisoned["rounds"][0]["routers"],
            legacy["rounds"][0]["routers"],
        )
        # The first poisoned exchange kills it: B's route via C is gone, so
        # B poisons C back toward A whose round-zero route still used B.
        self.assertEqual(poisoned["rounds"][1]["routers"]["A"]["C"], null)
        self.assertEqual(poisoned["rounds"][1]["routers"]["B"]["C"], null)

    def test_only_selected_hop_is_poisoned_other_neighbors_keep_metric(self):
        # After B-D fails, B keeps no route to D while A and C still do.
        # Plain failure round zero keeps A's stale via-B route (only routes
        # through a hop that vanished are invalidated; B is still A's
        # neighbor); the first poisoned exchange then drops it -- B
        # advertised D to A specifically as unreachable, because A was B's own
        # selected path back toward D before the failure.  C's own route to
        # D is direct, so its advertisement to A keeps its metric and the
        # equal alternative via C is adopted.
        events = [{"time": 1, "action": "link-down", "from": "B", "to": "D"}]
        scenario = {
            "infinityMetric": 20, "poisonReverse": True, "events": events,
        }
        entry = replay_dv_scenario(DIAMOND, scenario)["timeline"][1]
        self.assertEqual(entry["convergenceRound"], 2)
        rounds = entry["rounds"]
        self.assertEqual(
            rounds[0]["routers"]["A"]["D"], {"nextHop": "B", "metric": 2}
        )
        # A's equal alternative via C is offered unpolluted and adopted.
        self.assertEqual(
            rounds[1]["routers"]["A"]["D"], {"nextHop": "C", "metric": 2}
        )
        self.assertEqual(
            rounds[2]["routers"]["B"]["D"], {"nextHop": "A", "metric": 3}
        )


# ---------------------------------------------------------------------------
# Poison reverse changes timing, never the stable forwarding answer
# ---------------------------------------------------------------------------


class TestStableAnswerUnchanged(unittest.TestCase):
    def test_stable_tables_match_plain_mode_on_random_cases(self):
        rng = random.Random(31337)
        for _case in range(60):
            topology, events, infinity = _random_legal_case(rng)
            plain = replay_dv_scenario(
                topology, {"infinityMetric": infinity, "events": events}
            )
            poisoned = replay_dv_scenario(
                topology,
                {"infinityMetric": infinity, "poisonReverse": True,
                 "events": events},
            )
            for plain_entry, poison_entry in zip(
                plain["timeline"], poisoned["timeline"]
            ):
                self.assertEqual(
                    plain_entry["rounds"][-1]["routers"],
                    poison_entry["rounds"][-1]["routers"],
                )

    def test_fault_free_baseline_converges_to_shortest_paths(self):
        for topology in (DV_CHAIN, TRIANGLE, DIAMOND):
            with self.subTest(topology=topology):
                poisoned = replay_dv_scenario(
                    topology,
                    {"infinityMetric": 10 ** 6, "poisonReverse": True,
                     "events": []},
                )
                self.assertEqual(
                    poisoned["timeline"][0]["rounds"][-1]["routers"],
                    compute_topology(topology)["routers"],
                )


# ---------------------------------------------------------------------------
# Published shape: the field exists exactly when, and only where, specified
# ---------------------------------------------------------------------------


class TestPublishedShape(unittest.TestCase):
    EVENTS = [{"time": 1, "action": "node-down", "node": "C"}]

    def test_omitted_and_false_are_byte_compatible_with_legacy(self):
        legacy = replay_dv_scenario(
            DV_CHAIN, {"infinityMetric": 8, "events": self.EVENTS}
        )
        explicit_false = replay_dv_scenario(
            DV_CHAIN,
            {"infinityMetric": 8, "poisonReverse": False,
             "events": self.EVENTS},
        )
        self.assertEqual(explicit_false, legacy)
        self.assertNotIn("poisonReverse", legacy)
        for entry in legacy["timeline"]:
            self.assertNotIn("poisonReverse", entry)
            for snapshot in entry["rounds"]:
                self.assertNotIn("poisonReverse", snapshot)

    def test_true_echoes_only_one_root_field(self):
        result = replay_dv_scenario(
            DV_CHAIN,
            {"infinityMetric": 8, "poisonReverse": True,
             "events": self.EVENTS},
        )
        self.assertEqual(
            set(result),
            {"protocol", "infinityMetric", "poisonReverse", "timeline"},
        )
        self.assertIs(result["poisonReverse"], True)
        # Published key order: option fields ahead of the timeline.
        self.assertEqual(list(result)[-1], "timeline")
        for entry in result["timeline"]:
            self.assertNotIn("poisonReverse", entry)
            for snapshot in entry["rounds"]:
                self.assertEqual(set(snapshot), {"round", "routers"})

    def test_true_with_holddown_keeps_both_fields_and_round_maps(self):
        result = replay_dv_scenario(
            DV_CHAIN,
            {"infinityMetric": 8, "poisonReverse": True, "holdDownRounds": 2,
             "events": self.EVENTS},
        )
        self.assertEqual(
            set(result),
            {
                "protocol", "infinityMetric", "holdDownRounds",
                "poisonReverse", "timeline",
            },
        )
        self.assertEqual(list(result)[-2:], ["poisonReverse", "timeline"])
        for entry in result["timeline"]:
            for snapshot in entry["rounds"]:
                self.assertEqual(
                    set(snapshot), {"round", "routers", "holdDowns"}
                )

    def test_holddown_without_poison_still_lacks_the_field(self):
        result = replay_dv_scenario(
            DV_CHAIN,
            {"infinityMetric": 8, "holdDownRounds": 2, "events": self.EVENTS},
        )
        self.assertNotIn("poisonReverse", result)


# ---------------------------------------------------------------------------
# Round-zero boundary: poison reverse begins only at the first exchange
# ---------------------------------------------------------------------------


class TestRoundZeroBoundary(unittest.TestCase):
    def test_event_round_zero_is_identical_with_and_without_poison(self):
        rng = random.Random(4242)
        for _case in range(50):
            topology, events, infinity = _random_legal_case(rng)
            if not events:
                continue
            plain = replay_dv_scenario(
                topology, {"infinityMetric": infinity, "events": events}
            )
            poisoned = replay_dv_scenario(
                topology,
                {"infinityMetric": infinity, "poisonReverse": True,
                 "events": events},
            )
            for plain_entry, poison_entry in zip(
                plain["timeline"][1:], poisoned["timeline"][1:]
            ):
                self.assertEqual(
                    plain_entry["rounds"][0]["routers"],
                    poison_entry["rounds"][0]["routers"],
                )

    def test_recovered_link_is_direct_at_round_zero_under_poison(self):
        events = [
            {"time": 1, "action": "link-down", "from": "A", "to": "B"},
            {"time": 2, "action": "link-up", "from": "A", "to": "B"},
        ]
        round_zero = replay_dv_scenario(
            DV_CHAIN,
            {"infinityMetric": 8, "poisonReverse": True, "events": events},
        )["timeline"][2]["rounds"][0]["routers"]
        self.assertEqual(round_zero["A"]["B"], {"nextHop": "B", "metric": 1})
        self.assertEqual(round_zero["B"]["A"], {"nextHop": "A", "metric": 1})

    def test_baseline_round_zero_is_self_and_direct_only(self):
        rounds = replay_dv_scenario(
            DV_CHAIN,
            {"infinityMetric": 9, "poisonReverse": True, "events": []},
        )["timeline"][0]["rounds"]
        zero = rounds[0]["routers"]
        self.assertEqual(zero["A"]["A"], {"nextHop": None, "metric": 0})
        self.assertEqual(zero["A"]["B"], {"nextHop": "B", "metric": 1})
        self.assertEqual(zero["A"]["C"], {"nextHop": None, "metric": None})


# ---------------------------------------------------------------------------
# Poison reverse combined with hold-down
# ---------------------------------------------------------------------------


class TestPoisonHolddownCombination(unittest.TestCase):
    def test_poison_runs_through_the_holddown_exchange_kernel(self):
        # Chain, C down: hold-down round zero transitively invalidates A's
        # stale via-B route (B itself advertises C unreachable), so both
        # routers' windows are already open at round 0; the poisoned
        # synchronous exchange then simply keeps every stale candidate
        # poisoned while both windows count down together.  This pins that
        # poison reverse feeds the same selection/invalidation kernel the
        # timers wrap.
        events = [{"time": 1, "action": "node-down", "node": "C"}]
        result = replay_dv_scenario(
            DV_CHAIN,
            {"infinityMetric": 8, "holdDownRounds": 2, "poisonReverse": True,
             "events": events},
        )["timeline"][1]
        rounds = result["rounds"]
        null = {"nextHop": None, "metric": None}
        self.assertEqual(rounds[0]["holdDowns"]["A"], {"C": 2})
        self.assertEqual(rounds[0]["holdDowns"]["B"], {"C": 2})
        for snapshot in rounds:
            self.assertEqual(snapshot["routers"]["A"]["C"], null)
            self.assertEqual(snapshot["routers"]["B"]["C"], null)
        self.assertEqual(rounds[1]["holdDowns"]["A"], {"C": 1})
        self.assertEqual(rounds[1]["holdDowns"]["B"], {"C": 1})
        self.assertEqual(rounds[2]["holdDowns"]["A"], {})
        self.assertEqual(rounds[2]["holdDowns"]["B"], {})
        self.assertEqual(result["convergenceRound"], 2)

    def test_poison_adds_no_rounds_or_timers_to_holddown_trajectories(self):
        # Because hold-down round zero already closes over every route whose
        # selected hop advertises the destination unreachable, a converged
        # forwarding forest never contains a mutual finite dependency for
        # poison reverse to break: for these failure trajectories the
        # poisoned and plain exchanges reach the same round sequence.  The
        # independent oracle in this module checks the same trajectories
        # directly; here the two SUT modes must agree entry for entry.
        rng = random.Random(9001)
        for _case in range(40):
            topology, events, infinity = _random_legal_case(rng)
            hold_down = rng.choice([1, 2, 3, 5])
            plain = replay_dv_scenario(
                topology,
                {"infinityMetric": infinity, "holdDownRounds": hold_down,
                 "events": events},
            )["timeline"]
            poisoned = replay_dv_scenario(
                topology,
                {"infinityMetric": infinity, "holdDownRounds": hold_down,
                 "poisonReverse": True, "events": events},
            )["timeline"]
            self.assertEqual(plain, poisoned)

    def test_convergence_empties_every_timer(self):
        events = [{"time": 1, "action": "link-down", "from": "B", "to": "D"}]
        result = replay_dv_scenario(
            DIAMOND,
            {"infinityMetric": 20, "holdDownRounds": 3, "poisonReverse": True,
             "events": events},
        )["timeline"][1]
        final = result["rounds"][-1]
        for mapping in final["holdDowns"].values():
            self.assertEqual(mapping, {})
        self.assertEqual(result["convergenceRound"], final["round"])

    def test_direct_recovery_clears_timer_under_poison(self):
        events = [
            {"time": 1, "action": "node-down", "node": "C"},
            {"time": 2, "action": "node-up", "node": "C"},
        ]
        recovery_zero = replay_dv_scenario(
            {"nodes": ["A", "B", "C"],
             "links": [link("A", "B", 5), link("B", "C", 7)]},
            {"infinityMetric": 50, "holdDownRounds": 3,
             "poisonReverse": True, "events": events},
        )["timeline"][2]["rounds"][0]
        self.assertEqual(
            recovery_zero["routers"]["B"]["C"],
            {"nextHop": "C", "metric": 7},
        )
        self.assertEqual(recovery_zero["holdDowns"]["B"], {})


# ---------------------------------------------------------------------------
# Validation: JSON booleans only, topology precedence, other entries ignore
# ---------------------------------------------------------------------------


class TestValidationContract(unittest.TestCase):
    def test_non_boolean_poison_reverse_is_invalid_scenario(self):
        bad_values = ("true", 1, 0, None, [], {}, 2, "yes")
        for value in bad_values:
            with self.subTest(value=value):
                with self.assertRaises(InvalidScenario):
                    replay_dv_scenario(
                        DV_CHAIN,
                        {"infinityMetric": 8, "poisonReverse": value,
                         "events": []},
                    )

    def test_booleans_are_accepted(self):
        replay_dv_scenario(
            DV_CHAIN,
            {"infinityMetric": 8, "poisonReverse": True, "events": []},
        )
        replay_dv_scenario(
            DV_CHAIN,
            {"infinityMetric": 8, "poisonReverse": False, "events": []},
        )

    def test_topology_errors_take_precedence(self):
        bad_topology = {"nodes": ["A", "A"], "links": []}
        for value in ("true", True, False):
            with self.subTest(value=value):
                with self.assertRaises(InvalidTopology):
                    replay_dv_scenario(
                        bad_topology,
                        {"infinityMetric": 8, "poisonReverse": value,
                         "events": []},
                    )

    def test_other_entries_and_compute_converge_ignore_the_option(self):
        # Accepted silently and behavior unchanged in every other entry.
        for scenario in (
            {"events": [], "poisonReverse": True},
            {"events": [], "poisonReverse": "garbage"},
        ):
            static = replay_scenario(DIAMOND, scenario)
            ls = replay_ls_scenario(DIAMOND, scenario)
            plain = replay_scenario(
                DIAMOND, {"events": scenario["events"]}
            )
            plain_ls = replay_ls_scenario(
                DIAMOND, {"events": scenario["events"]}
            )
            self.assertEqual(static, plain)
            self.assertEqual(ls, plain_ls)
        # compute/converge take no scenario at all; sanity only.
        compute_topology(DIAMOND)
        converge_topology(DIAMOND)

    def test_cli_reports_invalid_scenario_status_2(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get(
            "PYTHONPATH", ""
        )
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            topo = tmp_path / "topo.json"
            topo.write_text(json.dumps(DV_CHAIN))
            bad = tmp_path / "bad.json"
            bad.write_text(json.dumps(
                {"infinityMetric": 8, "poisonReverse": "yes", "events": []}
            ))
            proc = subprocess.run(
                [sys.executable, "-m", "packet_routing_sim",
                 "replay-dv", str(topo), str(bad)],
                cwd=REPO_ROOT, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=60,
            )
            self.assertEqual(proc.returncode, 2)
            self.assertEqual(proc.stdout, b"")
            self.assertEqual(proc.stderr, b"invalid scenario\n")

            good = tmp_path / "good.json"
            good.write_text(json.dumps(
                {"infinityMetric": 8, "poisonReverse": True, "events": []}
            ))
            proc = subprocess.run(
                [sys.executable, "-m", "packet_routing_sim",
                 "replay-dv", str(topo), str(good)],
                cwd=REPO_ROOT, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=60,
            )
            self.assertEqual(proc.returncode, 0)
            document = json.loads(proc.stdout)
            self.assertIs(document["poisonReverse"], True)


# ---------------------------------------------------------------------------
# Purity and determinism
# ---------------------------------------------------------------------------


class TestPurityAndDeterminism(unittest.TestCase):
    SCENARIO = {
        "infinityMetric": 30,
        "poisonReverse": True,
        "holdDownRounds": 2,
        "events": [
            {"time": 1, "action": "link-down", "from": "A", "to": "B"},
            {"time": 2, "action": "node-down", "node": "C"},
            {"time": 3, "action": "node-up", "node": "C"},
            {"time": 4, "action": "link-up", "from": "B", "to": "A"},
        ],
    }
    TOPOLOGY = DIAMOND
    VARIANTS = [
        TOPOLOGY,
        {"nodes": ["D", "C", "B", "A"], "links": TOPOLOGY["links"]},
        {"nodes": list("ABCD"),
         "links": [
             link("B", "A", 1), link("C", "A", 1),
             link("D", "B", 1), link("D", "C", 1),
         ]},
        {"nodes": list("ABCD"),
         "links": list(reversed(TOPOLOGY["links"]))},
    ]

    def test_inputs_are_never_modified(self):
        topology_before = copy.deepcopy(self.TOPOLOGY)
        scenario_before = copy.deepcopy(self.SCENARIO)
        replay_dv_scenario(self.TOPOLOGY, self.SCENARIO)
        self.assertEqual(self.TOPOLOGY, topology_before)
        self.assertEqual(self.SCENARIO, scenario_before)

    def test_repeated_calls_share_no_state(self):
        first = replay_dv_scenario(self.TOPOLOGY, self.SCENARIO)
        first["timeline"][1]["rounds"][0]["routers"]["A"]["D"] = "HACK"
        second = replay_dv_scenario(
            copy.deepcopy(self.TOPOLOGY), copy.deepcopy(self.SCENARIO)
        )
        self.assertNotEqual(
            second["timeline"][1]["rounds"][0]["routers"]["A"]["D"], "HACK"
        )
        self.assertEqual(
            second,
            replay_dv_scenario(self.TOPOLOGY, self.SCENARIO),
        )

    def test_byte_identical_across_hash_seeds_and_orders(self):
        env_base = dict(os.environ)
        env_base["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env_base.get(
            "PYTHONPATH", ""
        )
        outputs = set()
        for seed in HASH_SEEDS:
            for variant in self.VARIANTS:
                env = dict(env_base)
                env["PYTHONHASHSEED"] = str(seed)
                env["PRSIM_TOPO"] = json.dumps(variant)
                env["PRSIM_SCEN"] = json.dumps(self.SCENARIO)
                proc = subprocess.run(
                    [
                        sys.executable, "-c",
                        "import json, os, sys\n"
                        "from packet_routing_sim.core import "
                        "replay_dv_scenario\n"
                        "topo = json.loads(os.environ['PRSIM_TOPO'])\n"
                        "scen = json.loads(os.environ['PRSIM_SCEN'])\n"
                        "sys.stdout.write(json.dumps("
                        "replay_dv_scenario(topo, scen), sort_keys=True))",
                    ],
                    env=env, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, timeout=60,
                )
                self.assertEqual(proc.returncode, 0, proc.stderr)
                outputs.add(proc.stdout)
        self.assertEqual(len(outputs), 1)


if __name__ == "__main__":
    unittest.main()
