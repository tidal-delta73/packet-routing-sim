"""Regression tests for the optional replay-dv poison-reverse mode.

Poison reverse is a single optional JSON boolean on the ``replay-dv``
scenario; it changes no command, topology, event or forwarding-entry
format.  These tests pin its contract:

* omission and an explicit ``false`` are byte-for-byte the legacy document
  (the root gains no field); ``true`` echoes ``poisonReverse: true`` and
  nothing else changes shape; ``replay``, ``replay-ls``, ``compute`` and
  ``converge`` keep ignoring the option;
* the stored local table keeps its selected next hop and metric; only the
  advertisement toward the neighbor the route was selected through is
  rewritten to unreachable, so the chain count-to-infinity collapses while
  a redundant alternative still converges to the same shortest tables;
* the event round zero (immediate invalidation, node recovery, link
  recovery) is identical with and without the option -- poison reverse
  starts at the following synchronous exchange;
* every recorded plain-mode round matches an independently written
  receiver-specific Bellman-Ford oracle, including the infinity cutoff,
  the smaller-next-hop tie break, down-router rows and "only changed
  rounds";
* combined with ``holdDownRounds`` the receiver-specific advertisements
  feed the existing invalidation/selection checks, the published
  ``routers``/``holdDowns`` shapes and the dual fixed point are unchanged;
* a non-boolean value is the sole ``InvalidScenario``, reported by the CLI
  as ``invalid scenario`` with status 2, topology errors still win;
* inputs are not mutated, calls share no state, and output is independent
  of declaration order and hash seed.
"""
import copy
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from packet_routing_sim.core import (
    InvalidScenario,
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
NODE_C_DOWN = [{"time": 1, "action": "node-down", "node": "C"}]
BD_DOWN = [{"time": 1, "action": "link-down", "from": "B", "to": "D"}]


def normalize(routers):
    """Nested JSON routing snapshot -> ``{(router, dest): (nextHop, metric)}``."""
    return {
        (router, dest): (entry["nextHop"], entry["metric"])
        for router, row in routers.items()
        for dest, entry in row.items()
    }


# ---------------------------------------------------------------------------
# Independent poison-reverse Bellman-Ford oracle
# ---------------------------------------------------------------------------


INF = float("inf")


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


def oracle_stable(topology, down_nodes, down_links, infinity_metric):
    """Clamped shortest-path tables of one availability state."""
    nodes, active = _adjacency(topology, down_nodes, down_links)
    dist = _floyd(nodes, active)
    tables = {}
    for router in nodes:
        for dest in nodes:
            if router in down_nodes:
                tables[router, dest] = (None, None)
            elif dest == router:
                tables[router, dest] = (None, 0)
            elif dest in down_nodes or dist[router][dest] == INF:
                tables[router, dest] = (None, None)
            else:
                metric = dist[router][dest]
                if metric >= infinity_metric:
                    tables[router, dest] = (None, None)
                else:
                    winners = [
                        neighbor
                        for neighbor in active[router]
                        if dist[neighbor][dest] != INF
                        and active[router][neighbor] + dist[neighbor][dest]
                        == metric
                    ]
                    tables[router, dest] = (min(winners), metric)
    return tables


def oracle_round_zero(previous, nodes, active, down_nodes):
    """The shared post-event round zero, written independently of the SUT."""
    table = {}
    for router in nodes:
        if router in down_nodes:
            table.update({(router, d): (None, None) for d in nodes})
            continue
        for dest in nodes:
            if dest == router:
                table[router, dest] = (None, 0)
            elif dest in active[router]:
                table[router, dest] = (dest, active[router][dest])
            else:
                hop, metric = previous[router, dest]
                table[router, dest] = (
                    (hop, metric) if hop in active[router] else (None, None)
                )
    return table


def oracle_pr_round(previous, nodes, active, down_nodes, infinity_metric):
    """One synchronous update with receiver-specific poison reverse.

    The only departure from a plain Bellman-Ford round: a neighbor is shown
    unreachable for a destination while that neighbor is the sender's own
    selected next hop toward it.  Everything else (synchronous reads, the
    infinity cutoff, sorted-neighbor tie break) is the shared rule set.
    """
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
                hop, advertised = previous[neighbor, dest]
                if hop == router:
                    # The neighbor routes through us: it poisons this ad.
                    advertised = None
                if advertised is None:
                    continue
                cost = active[router][neighbor] + advertised
                if cost >= infinity_metric:
                    continue
                if best_cost is None or cost < best_cost:
                    best_cost, best_hop = cost, neighbor
            current[router, dest] = (
                (None, None) if best_cost is None else (best_hop, best_cost)
            )
    return current


def oracle_pr_timeline(topology, scenario):
    """Recorded poison-reverse rounds for every timeline entry."""
    infinity_metric = scenario["infinityMetric"]
    nodes, full = _adjacency(topology)

    def fold_active(down, disabled):
        active = {node: {} for node in nodes}
        for router in nodes:
            if router in down:
                continue
            for neighbor, weight in full[router].items():
                pair = (router, neighbor) if router < neighbor else (
                    neighbor, router
                )
                if neighbor not in down and pair not in disabled:
                    active[router][neighbor] = weight
        return active

    def record(round_zero, active, downs):
        recorded = [round_zero]
        current = round_zero
        while True:
            updated = oracle_pr_round(
                current, nodes, active, downs, infinity_metric
            )
            if updated == current:
                return recorded
            current = updated
            recorded.append(current)

    down, disabled = set(), set()
    active = fold_active(down, disabled)
    start = {
        (r, d): (
            (None, 0)
            if r == d
            else ((d, active[r][d]) if d in active[r] else (None, None))
        )
        for r in nodes
        for d in nodes
    }
    timeline = [record(start, active, frozenset())]
    for event in scenario["events"]:
        if event["action"] == "node-down":
            down.add(event["node"])
        elif event["action"] == "node-up":
            down.discard(event["node"])
        elif event["action"] == "link-down":
            a, b = event["from"], event["to"]
            disabled.add((a, b) if a < b else (b, a))
        else:
            a, b = event["from"], event["to"]
            disabled.discard((a, b) if a < b else (b, a))
        downs = frozenset(down)
        active = fold_active(down, disabled)
        round_zero = oracle_round_zero(
            timeline[-1][-1], nodes, active, downs
        )
        timeline.append(record(round_zero, active, downs))
    return timeline


# ---------------------------------------------------------------------------
# Document shape and byte compatibility
# ---------------------------------------------------------------------------


class TestPublishedShape(unittest.TestCase):
    def test_omitted_and_false_are_byte_compatible_with_legacy(self):
        legacy = replay_dv_scenario(
            DV_CHAIN, {"infinityMetric": 8, "events": NODE_C_DOWN}
        )
        explicit_false = replay_dv_scenario(
            DV_CHAIN,
            {"infinityMetric": 8, "poisonReverse": False, "events": NODE_C_DOWN},
        )
        self.assertEqual(legacy, explicit_false)
        self.assertNotIn("poisonReverse", legacy)
        for entry in legacy["timeline"]:
            self.assertNotIn("poisonReverse", entry)
            for snapshot in entry["rounds"]:
                self.assertEqual(set(snapshot), {"round", "routers"})

    def test_true_echoes_only_the_root_field(self):
        result = replay_dv_scenario(
            DV_CHAIN,
            {"infinityMetric": 8, "poisonReverse": True, "events": NODE_C_DOWN},
        )
        self.assertEqual(
            set(result),
            {"protocol", "infinityMetric", "poisonReverse", "timeline"},
        )
        self.assertIs(result["poisonReverse"], True)
        for entry in result["timeline"]:
            self.assertNotIn("poisonReverse", entry)
            for snapshot in entry["rounds"]:
                self.assertEqual(set(snapshot), {"round", "routers"})

    def test_true_combines_with_holddown_echo_and_round_shape(self):
        result = replay_dv_scenario(
            DV_CHAIN,
            {
                "infinityMetric": 8,
                "poisonReverse": True,
                "holdDownRounds": 2,
                "events": NODE_C_DOWN,
            },
        )
        self.assertEqual(
            set(result),
            {
                "protocol",
                "infinityMetric",
                "holdDownRounds",
                "poisonReverse",
                "timeline",
            },
        )
        for entry in result["timeline"]:
            for snapshot in entry["rounds"]:
                self.assertEqual(
                    set(snapshot), {"round", "routers", "holdDowns"}
                )

    def test_other_replay_entries_ignore_the_option(self):
        for wrap, entry in (
            (replay_scenario, lambda: {"events": NODE_C_DOWN}),
            (replay_ls_scenario, lambda: {"events": NODE_C_DOWN}),
        ):
            plain = wrap(DIAMOND, {"events": BD_DOWN})
            with_option = wrap(
                DIAMOND,
                {
                    "events": BD_DOWN,
                    "poisonReverse": "ignored",
                    "holdDownRounds": -3,
                    "infinityMetric": 1,
                },
            )
            self.assertEqual(plain, with_option)


# ---------------------------------------------------------------------------
# Hand-traced poison-reverse behavior
# ---------------------------------------------------------------------------


class TestPoisonReverseSemantics(unittest.TestCase):
    def test_chain_count_to_infinity_collapses_to_one_exchange(self):
        entry = replay_dv_scenario(
            DV_CHAIN,
            {"infinityMetric": 8, "poisonReverse": True, "events": NODE_C_DOWN},
        )["timeline"][1]
        null = {"nextHop": None, "metric": None}
        # Round zero is the unchanged immediate-invalidation rule: A still
        # holds its stale route via B; B already lost C.
        rounds = entry["rounds"]
        self.assertEqual(
            rounds[0]["routers"]["A"]["C"], {"nextHop": "B", "metric": 2}
        )
        self.assertEqual(rounds[0]["routers"]["B"]["C"], null)
        # The first poisoned exchange removes every stale route at once, so
        # the legacy six-round count to infinity never happens.
        self.assertEqual([snapshot["round"] for snapshot in rounds], [0, 1])
        self.assertEqual(entry["convergenceRound"], 1)
        self.assertEqual(rounds[1]["routers"]["A"]["C"], null)
        self.assertEqual(rounds[1]["routers"]["B"]["C"], null)

    def test_redundant_alternative_is_delayed_one_round_but_kept(self):
        # Without poison reverse A switches to C and B learns via A in the
        # very same exchange (convergence round 1).  With it, A's round-0
        # route to D still points through B, so A advertises D unreachable
        # back to B: B must wait one more round while A itself adopts C.
        scenarios = (
            ("legacy", {"infinityMetric": 20, "events": BD_DOWN}),
            (
                "poison",
                {"infinityMetric": 20, "poisonReverse": True, "events": BD_DOWN},
            ),
        )
        traces = {}
        for label, scenario in scenarios:
            entry = replay_dv_scenario(DIAMOND, scenario)["timeline"][1]
            traces[label] = entry
            rounds = entry["rounds"]
            self.assertEqual(
                rounds[-1]["routers"]["A"]["D"],
                {"nextHop": "C", "metric": 2},
            )
            self.assertEqual(
                rounds[-1]["routers"]["B"]["D"],
                {"nextHop": "A", "metric": 3},
            )
        self.assertEqual(traces["legacy"]["convergenceRound"], 1)
        self.assertEqual(traces["poison"]["convergenceRound"], 2)
        poison_rounds = traces["poison"]["rounds"]
        self.assertEqual(
            poison_rounds[1]["routers"]["A"]["D"],
            {"nextHop": "C", "metric": 2},
        )
        self.assertEqual(
            poison_rounds[1]["routers"]["B"]["D"],
            {"nextHop": None, "metric": None},
        )
        self.assertEqual(
            poison_rounds[2]["routers"]["B"]["D"],
            {"nextHop": "A", "metric": 3},
        )
        # Both modes publish the same stable tables.
        self.assertEqual(
            traces["legacy"]["rounds"][-1]["routers"],
            traces["poison"]["rounds"][-1]["routers"],
        )

    def test_round_zero_is_identical_with_and_without_poison(self):
        for hold_down in (None, 2):
            scenario = {
                "infinityMetric": 20,
                "events": NODE_C_DOWN + [
                    {"time": 2, "action": "link-down", "from": "A", "to": "B"},
                    {"time": 3, "action": "node-up", "node": "C"},
                    {"time": 4, "action": "link-up", "from": "B", "to": "A"},
                ],
            }
            if hold_down is not None:
                scenario["holdDownRounds"] = hold_down
            legacy = replay_dv_scenario(DV_CHAIN, scenario)["timeline"]
            poisoned = replay_dv_scenario(
                DV_CHAIN, {**scenario, "poisonReverse": True}
            )["timeline"]
            for legacy_entry, poison_entry in zip(legacy[1:], poisoned[1:]):
                self.assertEqual(
                    legacy_entry["rounds"][0],
                    poison_entry["rounds"][0],
                    hold_down,
                )

    def test_fault_free_baseline_is_unchanged(self):
        for topology, infinity_metric in ((DIAMOND, 20), (DV_CHAIN, 8)):
            legacy = replay_dv_scenario(
                topology, {"infinityMetric": infinity_metric, "events": []}
            )["timeline"][0]
            poisoned = replay_dv_scenario(
                topology,
                {"infinityMetric": infinity_metric, "poisonReverse": True,
                 "events": []},
            )["timeline"][0]
            # Poison reverse cannot alter a shortest-path fixed point: the
            # whole baseline trajectory, not just its final round, matches.
            self.assertEqual(legacy["rounds"], poisoned["rounds"])
            self.assertEqual(
                legacy["convergenceRound"], poisoned["convergenceRound"]
            )

    def test_every_poison_round_matches_the_independent_oracle(self):
        rng = random.Random(2026)
        for case in range(60):
            topology, events, infinity_metric = _random_legal_case(rng)
            scenario = {
                "infinityMetric": infinity_metric,
                "poisonReverse": True,
                "events": events,
            }
            timeline = replay_dv_scenario(topology, scenario)["timeline"]
            expected_entries = oracle_pr_timeline(topology, scenario)
            self.assertEqual(
                len(timeline), len(expected_entries), case
            )
            # Poison reverse changes timing, never the answer: every entry's
            # stable table is the clamped Floyd-Warshall shortest-path table
            # of its availability state.
            states = _fold_availability(events)
            for index, (entry, expected_rounds) in enumerate(
                zip(timeline, expected_entries)
            ):
                down_nodes, down_links = states[index]
                self.assertEqual(
                    normalize(entry["rounds"][-1]["routers"]),
                    oracle_stable(
                        topology, down_nodes, down_links, infinity_metric
                    ),
                    case,
                )
                rounds = entry["rounds"]
                self.assertEqual(
                    len(rounds), len(expected_rounds), case
                )
                self.assertEqual(
                    [snapshot["round"] for snapshot in rounds],
                    list(range(len(rounds))),
                    case,
                )
                for snapshot, expected in zip(rounds, expected_rounds):
                    self.assertEqual(
                        normalize(snapshot["routers"]), expected, case
                    )
                    self.assertIsInstance(snapshot["routers"], dict)
                self.assertEqual(
                    entry["convergenceRound"], rounds[-1]["round"], case
                )
                # Recorded adjacent rounds genuinely differ, and the tail is
                # a fixed point under another poisoned exchange.
                for previous, current in zip(rounds, rounds[1:]):
                    self.assertNotEqual(
                        previous["routers"], current["routers"], case
                    )


# ---------------------------------------------------------------------------
# Combination with hold-down
# ---------------------------------------------------------------------------


class TestPoisonReverseWithHoldDown(unittest.TestCase):
    def test_chain_keeps_the_holddown_window_and_its_round_shape(self):
        scenario = {
            "infinityMetric": 8,
            "holdDownRounds": 2,
            "poisonReverse": True,
            "events": NODE_C_DOWN,
        }
        entry = replay_dv_scenario(DV_CHAIN, scenario)["timeline"][1]
        null = {"nextHop": None, "metric": None}
        rounds = entry["rounds"]
        self.assertEqual(rounds[0]["routers"]["A"]["C"], null)
        self.assertEqual(rounds[0]["holdDowns"]["A"], {"C": 2})
        self.assertEqual(rounds[0]["holdDowns"]["B"], {"C": 2})
        self.assertEqual(rounds[1]["holdDowns"]["A"], {"C": 1})
        self.assertEqual(rounds[2]["holdDowns"]["A"], {})
        self.assertEqual(entry["convergenceRound"], 2)
        for snapshot in rounds:
            self.assertEqual(
                set(snapshot), {"round", "routers", "holdDowns"}
            )

    def test_timers_feed_on_poisoned_prior_advertisements(self):
        # Constructed to exercise the layering: at the first exchange the
        # selected hop's poisoned advertisement is the unusability signal the
        # hold-down checks read.  The final state is the clamped shortest
        # paths regardless of which suppression mechanism is active.
        scenario = {
            "infinityMetric": 20,
            "holdDownRounds": 2,
            "poisonReverse": True,
            "events": BD_DOWN,
        }
        entry = replay_dv_scenario(DIAMOND, scenario)["timeline"][1]
        final = entry["rounds"][-1]
        self.assertEqual(
            final["routers"]["A"]["D"], {"nextHop": "C", "metric": 2}
        )
        self.assertEqual(
            final["routers"]["B"]["D"], {"nextHop": "A", "metric": 3}
        )
        for mapping in final["holdDowns"].values():
            self.assertEqual(mapping, {})

    def test_trajectory_matches_holddown_without_poison_across_cases(self):
        # Poison reverse changes only the advertisements read by the
        # hold-down invalidation/selection checks; the suppression windows
        # dominate, so on these cases the whole timeline (rounds, routers and
        # holdDowns, only the root echo aside) is identical.
        rng = random.Random(4242)
        for case in range(40):
            topology, events, infinity_metric = _random_legal_case(rng)
            hold_down = rng.choice([1, 2, 5])
            base = {
                "infinityMetric": infinity_metric,
                "holdDownRounds": hold_down,
                "events": events,
            }
            held = replay_dv_scenario(topology, base)
            poisoned = replay_dv_scenario(
                topology, {**base, "poisonReverse": True}
            )
            self.assertEqual(
                held["timeline"], poisoned["timeline"], case
            )
            self.assertEqual(
                held["timeline"][-1]["rounds"][-1]["routers"],
                poisoned["timeline"][-1]["rounds"][-1]["routers"],
                case,
            )


# ---------------------------------------------------------------------------
# Validation contract
# ---------------------------------------------------------------------------


class TestValidation(unittest.TestCase):
    def test_non_boolean_values_are_invalid_scenario(self):
        for value in (1, 0, -1, "true", None, [], {}, 1.0):
            with self.subTest(value=value):
                with self.assertRaises(InvalidScenario):
                    replay_dv_scenario(
                        DV_CHAIN,
                        {
                            "infinityMetric": 8,
                            "poisonReverse": value,
                            "events": [],
                        },
                    )

    def test_true_and_false_are_accepted(self):
        for value in (True, False):
            replay_dv_scenario(
                DV_CHAIN,
                {"infinityMetric": 8, "poisonReverse": value, "events": []},
            )

    def test_topology_errors_take_precedence(self):
        bad_topology = {"nodes": ["A", "A"], "links": []}
        with self.assertRaises(InvalidTopology):
            replay_dv_scenario(
                bad_topology,
                {"infinityMetric": 1, "poisonReverse": "x", "events": []},
            )

    def test_cli_reports_invalid_scenario_with_status_2(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get(
            "PYTHONPATH", ""
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            topo_path = tmp_path / "topo.json"
            topo_path.write_text(json.dumps(DV_CHAIN))
            scenario_path = tmp_path / "scenario.json"
            scenario_path.write_text(
                json.dumps(
                    {
                        "infinityMetric": 8,
                        "poisonReverse": "yes",
                        "events": [],
                    }
                )
            )
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "packet_routing_sim",
                    "replay-dv",
                    str(topo_path),
                    str(scenario_path),
                ],
                cwd=REPO_ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
            )
            self.assertEqual(proc.returncode, 2)
            self.assertEqual(proc.stdout, b"")
            self.assertEqual(proc.stderr, b"invalid scenario\n")

            # A boolean true is accepted through the same unchanged command.
            scenario_path.write_text(
                json.dumps(
                    {
                        "infinityMetric": 8,
                        "poisonReverse": True,
                        "events": [],
                    }
                )
            )
            accepted = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "packet_routing_sim",
                    "replay-dv",
                    str(topo_path),
                    str(scenario_path),
                ],
                cwd=REPO_ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
            )
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            document = json.loads(accepted.stdout)
            self.assertIs(document["poisonReverse"], True)


# ---------------------------------------------------------------------------
# Purity and isolation
# ---------------------------------------------------------------------------


class TestPurity(unittest.TestCase):
    SCENARIO = {
        "infinityMetric": 30,
        "poisonReverse": True,
        "events": [
            {"time": 1, "action": "link-down", "from": "A", "to": "B"},
            {"time": 2, "action": "node-down", "node": "C"},
            {"time": 3, "action": "node-up", "node": "C"},
            {"time": 4, "action": "link-up", "from": "B", "to": "A"},
        ],
    }

    def test_input_is_never_modified(self):
        topology_before = copy.deepcopy(DIAMOND)
        scenario_before = copy.deepcopy(self.SCENARIO)
        replay_dv_scenario(DIAMOND, self.SCENARIO)
        self.assertEqual(DIAMOND, topology_before)
        self.assertEqual(self.SCENARIO, scenario_before)

    def test_repeated_calls_share_no_protocol_state(self):
        first = replay_dv_scenario(DIAMOND, self.SCENARIO)
        first["timeline"][1]["rounds"][0]["routers"]["A"]["B"] = "HACK"
        second = replay_dv_scenario(
            copy.deepcopy(DIAMOND), copy.deepcopy(self.SCENARIO)
        )
        self.assertNotEqual(
            second["timeline"][1]["rounds"][0]["routers"]["A"]["B"], "HACK"
        )
        # Every call starts fault free.
        empty = replay_dv_scenario(
            DIAMOND,
            {"infinityMetric": 30, "poisonReverse": True, "events": []},
        )
        self.assertEqual(
            empty["timeline"][0]["rounds"][-1]["routers"],
            first["timeline"][0]["rounds"][-1]["routers"],
        )


# ---------------------------------------------------------------------------
# Determinism across declaration order and hash seed
# ---------------------------------------------------------------------------


_DETERMINISM_SNIPPET = """
import json, os, sys
sys.path.insert(0, %r)
from packet_routing_sim.core import replay_dv_scenario
topology = json.loads(os.environ["PRSIM_TOPO"])
scenario = json.loads(os.environ["PRSIM_SCEN"])
sys.stdout.write(json.dumps(
    replay_dv_scenario(topology, scenario), sort_keys=True))
""" % str(REPO_ROOT)


def _run_in_subprocess(topology, scenario, seed):
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

    def test_byte_identical_across_seeds_and_declaration_orders(self):
        outputs = set()
        for seed in HASH_SEEDS:
            for variant in self.VARIANTS:
                outputs.add(
                    _run_in_subprocess(variant, self.SCENARIO, seed)
                )
        self.assertEqual(len(outputs), 1)


# ---------------------------------------------------------------------------
# Random legal-case generator (mirrors the kernel tests' event folding)
# ---------------------------------------------------------------------------


def _fold_availability(events):
    """Availability state ``[(down_nodes, down_links)]`` for each entry."""
    down_nodes, down_links = set(), set()
    states = [(frozenset(), frozenset())]
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
        states.append((frozenset(down_nodes), frozenset(down_links)))
    return states


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
            (down_nodes.add if action == "node-down" else down_nodes.discard)(a)
        else:
            events.append(
                {"time": time, "action": action, "from": a, "to": b}
            )
            pair = (a, b) if a < b else (b, a)
            (down_links.add if action == "link-down" else down_links.discard)(
                pair
            )
    max_metric = max((edge["metric"] for edge in links), default=0)
    return topology, events, max_metric + rng.choice([1, 2, 6, 40])


if __name__ == "__main__":
    unittest.main()
