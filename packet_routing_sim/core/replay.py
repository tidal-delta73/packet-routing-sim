"""Timeline orchestration shared by every scenario replay entry point.

This module is the *only* place that knows how a replay unfolds over time,
and it deliberately contains no validation and no routing algorithm:

* :mod:`topology` validates the topology;
* :mod:`scenario` validates structure, references, times and transitions and
  hands back immutable :class:`~scenario.Event` records in time order;
* :mod:`state` owns the failure/recovery transition rules
  (:func:`state.apply_event`);
* :mod:`routing` and :mod:`linkstate` own the protocol computations.

What is unified here is the event-progression boundary every replay crosses:

1. start from the fault-free :class:`~state.NetworkState` and ask the runner
   for the baseline entry's payload and for its own carried protocol state;
2. apply exactly one validated event to get the next network state;
3. hand the runner the previous and current network state plus the carried
   state it itself produced, and take back the entry payload and the carried
   state for the next step;
4. assemble the timeline entry with one fixed envelope and key order,
   echoing the raw event verbatim.

A runner is protocol-specific and isolated: the engine never inspects its
carried value, so distance-vector vectors, link-state databases/sequence
history and the snapshot runner's ``None`` can never touch one another.
Every call builds fresh state here, so no fault or protocol state ever leaks
between calls.
"""
from .linkstate import (
    active_neighbor_sets,
    baseline_databases,
    changed_endpoints,
    event_databases,
    link_state_convergence,
)
from .routing import (
    distance_vector_failure_convergence,
    distance_vector_failure_round_zero,
    distance_vector_holddown_convergence,
    distance_vector_holddown_round_zero,
    initial_distance_vectors,
    link_state_snapshot,
)
from .state import NetworkState, apply_event

# ---------------------------------------------------------------------------
# Timeline assembly
# ---------------------------------------------------------------------------


def _rounds_payload(rounds):
    """The shared payload of a rounds-bearing timeline entry.

    Insertion order is the published key order: ``convergenceRound`` then
    ``rounds``.
    """
    return {"convergenceRound": rounds[-1]["round"], "rounds": rounds}


def _baseline_entry(payload):
    """The fault-free entry: ``event: null`` ahead of the runner payload."""
    entry = {"event": None}
    entry.update(payload)
    return entry


def _event_entry(event, payload):
    """One event entry: ``time`` and the raw event ahead of runner payload."""
    entry = {"time": event.time, "event": event.raw}
    entry.update(payload)
    return entry


def run_timeline(topology, events, runner):
    """Fold ``events`` into a timeline via ``runner``.

    The runner supplies the baseline payload and carried state from the
    fault-free state, then one payload/carried pair per event.  Within the
    timeline engine this is the only progression path and the sole builder
    of timeline entries; scenario validation folds the same transition
    rules separately, ahead of the replay.
    """
    state = NetworkState.initial(topology)
    payload, carried = runner.baseline(topology, state)
    timeline = [_baseline_entry(payload)]
    for event in events:
        previous_state = state
        state = apply_event(state, event)
        payload, carried = runner.step(
            event, state, previous_state, carried
        )
        timeline.append(_event_entry(event, payload))
    return timeline


# ---------------------------------------------------------------------------
# Runners: one small adapter per protocol.  Each keeps its carried state
# private to itself and uses only the pure functions of its protocol module.
# ---------------------------------------------------------------------------


class StaticSnapshotRunner:
    """The plain ``replay`` runner: a fresh static forwarding snapshot."""

    def _snapshot(self, topology, state):
        return link_state_snapshot(
            topology.nodes, state.active_adjacency(), state.down_nodes
        )

    def baseline(self, topology, state):
        return {"routers": self._snapshot(topology, state)}, None

    def step(self, event, state, previous_state, carried):
        return {"routers": self._snapshot(state.topology, state)}, None


class LinkStateRunner:
    """The ``replay-ls`` runner: databases and sequence history carry over."""

    def baseline(self, topology, state):
        ordered_nodes = topology.nodes
        adjacency = state.active_adjacency()
        neighbors = active_neighbor_sets(ordered_nodes, adjacency)
        databases, sequences = baseline_databases(ordered_nodes, neighbors)
        rounds, databases = link_state_convergence(
            databases,
            ordered_nodes,
            topology,
            adjacency,
            state.down_nodes,
        )
        return _rounds_payload(rounds), (databases, sequences)

    def step(self, event, state, previous_state, carried):
        topology = state.topology
        ordered_nodes = topology.nodes
        databases, sequences = carried
        adjacency = state.active_adjacency()
        new_neighbors = active_neighbor_sets(ordered_nodes, adjacency)
        old_neighbors = active_neighbor_sets(
            ordered_nodes, previous_state.active_adjacency()
        )
        recovered = previous_state.down_nodes - state.down_nodes
        changed = changed_endpoints(
            ordered_nodes,
            old_neighbors,
            new_neighbors,
            state.down_nodes,
            recovered,
        )
        databases, sequences = event_databases(
            databases,
            ordered_nodes,
            new_neighbors,
            state.down_nodes,
            changed,
            sequences,
        )
        rounds, databases = link_state_convergence(
            databases,
            ordered_nodes,
            topology,
            adjacency,
            state.down_nodes,
        )
        return _rounds_payload(rounds), (databases, sequences)


def _rounds_with_empty_holddowns(rounds, ordered_nodes):
    """Fresh copy of legacy-style rounds carrying an empty ``holdDowns`` map."""
    return [
        {
            "round": snapshot["round"],
            "routers": snapshot["routers"],
            "holdDowns": {router: {} for router in ordered_nodes},
        }
        for snapshot in rounds
    ]


class DistanceVectorLegacyRunner:
    """The hold-down-free ``replay-dv`` runner; converged vectors carry over."""

    def __init__(self, infinity_metric, poison_reverse=False):
        self.infinity_metric = infinity_metric
        self.poison_reverse = poison_reverse

    def baseline(self, topology, state):
        adjacency = state.active_adjacency()
        rounds = distance_vector_failure_convergence(
            initial_distance_vectors(topology.nodes, adjacency),
            topology.nodes,
            adjacency,
            state.down_nodes,
            self.infinity_metric,
            self.poison_reverse,
        )
        return _rounds_payload(rounds), rounds[-1]["routers"]

    def step(self, event, state, previous_state, vectors):
        topology = state.topology
        adjacency = state.active_adjacency()
        # Round zero is the event's immediate invalidation/re-adoption rule;
        # poison reverse shapes only the synchronous exchanges after it.
        round_zero = distance_vector_failure_round_zero(
            vectors, topology.nodes, adjacency, state.down_nodes
        )
        rounds = distance_vector_failure_convergence(
            round_zero,
            topology.nodes,
            adjacency,
            state.down_nodes,
            self.infinity_metric,
            self.poison_reverse,
        )
        return _rounds_payload(rounds), rounds[-1]["routers"]


class DistanceVectorHolddownRunner:
    """``replay-dv`` with route hold-down; only converged vectors carry over.

    The fault-free baseline invalidates no finite route, so its rounds are the
    legacy rounds with empty ``holdDowns`` maps.  Each later event builds
    round zero from the previous converged vectors (every timer has expired
    by convergence) and runs the synchronous hold-down convergence.  With
    poison reverse enabled the baseline plain rounds and every event's later
    synchronous exchanges use receiver-specific advertisements, while round
    zero stays the shared immediate-invalidation rule.
    """

    def __init__(self, infinity_metric, hold_down_rounds, poison_reverse=False):
        self.infinity_metric = infinity_metric
        self.hold_down_rounds = hold_down_rounds
        self.poison_reverse = poison_reverse

    def baseline(self, topology, state):
        adjacency = state.active_adjacency()
        legacy_rounds = distance_vector_failure_convergence(
            initial_distance_vectors(topology.nodes, adjacency),
            topology.nodes,
            adjacency,
            state.down_nodes,
            self.infinity_metric,
            self.poison_reverse,
        )
        rounds = _rounds_with_empty_holddowns(legacy_rounds, topology.nodes)
        return _rounds_payload(rounds), legacy_rounds[-1]["routers"]

    def step(self, event, state, previous_state, vectors):
        topology = state.topology
        adjacency = state.active_adjacency()
        round_zero, holddowns_zero = distance_vector_holddown_round_zero(
            vectors,
            topology.nodes,
            adjacency,
            state.down_nodes,
            self.hold_down_rounds,
        )
        rounds, vectors = distance_vector_holddown_convergence(
            round_zero,
            holddowns_zero,
            topology.nodes,
            adjacency,
            state.down_nodes,
            self.infinity_metric,
            self.hold_down_rounds,
            self.poison_reverse,
        )
        return _rounds_payload(rounds), vectors


# ---------------------------------------------------------------------------
# Protocol envelopes
# ---------------------------------------------------------------------------


def static_replay(topology, events):
    """Plain link-state snapshot timeline from validated events."""
    return {
        "protocol": "link-state",
        "timeline": run_timeline(topology, events, StaticSnapshotRunner()),
    }


def link_state_replay(topology, events):
    """Auditable link-state flooding timeline from validated events."""
    return {
        "protocol": "link-state",
        "timeline": run_timeline(topology, events, LinkStateRunner()),
    }


def distance_vector_replay(
    topology,
    events,
    infinity_metric,
    hold_down_rounds=None,
    poison_reverse=False,
):
    """Distance-vector timeline from validated events.

    Omitted ``hold_down_rounds`` (``None``) takes the legacy path; an explicit
    ``0`` disables suppression and shares it byte-for-byte.  A positive value
    adds the echoed root field and per-round ``holdDowns`` maps.  Poison
    reverse (``poison_reverse``) is likewise absent from the root document
    unless explicitly enabled, in which case the root echoes
    ``poisonReverse: true``; the timeline entry and round shapes are
    otherwise unchanged.
    """
    if not hold_down_rounds:
        result = {
            "protocol": "distance-vector",
            "infinityMetric": infinity_metric,
        }
        runner = DistanceVectorLegacyRunner(infinity_metric, poison_reverse)
    else:
        result = {
            "protocol": "distance-vector",
            "infinityMetric": infinity_metric,
            "holdDownRounds": hold_down_rounds,
        }
        runner = DistanceVectorHolddownRunner(
            infinity_metric, hold_down_rounds, poison_reverse
        )
    if poison_reverse:
        result["poisonReverse"] = True
    result["timeline"] = run_timeline(topology, events, runner)
    return result
