"""Pure simulation core: validated topology in, plain routing data out.

Public entry points:

* :func:`compute_topology`    -> ``{"routers": ...}``
* :func:`converge_topology`   -> ``{"protocol", "convergenceRound", "rounds"}``
* :func:`replay_scenario`     -> ``{"protocol", "timeline"}`` (link-state)
* :func:`replay_ls_scenario`  -> ``{"protocol", "timeline"}`` with auditable
  neighbor discovery, LSA flooding and per-database SPF rounds
* :func:`replay_dv_scenario`  -> ``{"protocol", "infinityMetric",
  "timeline"}`` (distance-vector)

Each entry point takes already-decoded JSON values (plain ``dict``/``list``/
``str``/``int``) and returns fresh plain-Python data.  The core performs no
file or stream I/O, reads no command-line arguments or environment, uses no
module-level mutable state, and never depends on dict insertion order:
nodes are sorted by name and every neighbor scan is explicit.

Inputs are deep-copied on entry and outputs are built fresh, so an input
object compares equal before and after a call, mutating one call's result
cannot affect a later call, and no fault state ever leaks between calls --
every replay starts from the fault-free state.

Invalid input raises a distinguishable error:

* :class:`InvalidTopology` for a malformed topology;
* :class:`InvalidScenario` for a malformed scenario;
* :class:`InvalidStateTransition` (an ``InvalidScenario`` subclass) for a
  well-formed event that requests an illegal transition.
"""
import copy

from .errors import (
    InvalidScenario,
    InvalidStateTransition,
    InvalidTopology,
    SimulationError,
)
from .linkstate import (
    active_neighbor_sets,
    baseline_databases,
    changed_endpoints,
    event_databases,
    link_state_convergence,
)
from .routing import (
    distance_vector_convergence,
    distance_vector_failure_convergence,
    distance_vector_failure_round_zero,
    distance_vector_holddown_convergence,
    distance_vector_holddown_round,
    distance_vector_holddown_round_zero,
    forwarding_table,
    initial_distance_vectors,
    link_state_snapshot,
)
from .scenario import validate_dv_scenario, validate_scenario
from .state import NetworkState, apply_event
from .topology import Topology, validate_topology

__all__ = [
    "compute_topology",
    "converge_topology",
    "replay_scenario",
    "replay_validated",
    "replay_ls_scenario",
    "replay_ls_validated",
    "replay_dv_scenario",
    "replay_dv_validated",
    "distance_vector_holddown_round",
    "distance_vector_holddown_round_zero",
    "distance_vector_holddown_convergence",
    "holddowns_public",
    "snapshot_state",
    "validate_topology",
    "validate_scenario",
    "validate_dv_scenario",
    "Topology",
    "NetworkState",
    "apply_event",
    "SimulationError",
    "InvalidTopology",
    "InvalidScenario",
    "InvalidStateTransition",
]


def _validated(topology):
    """Validate a defensive copy so caller data can never be mutated."""
    return validate_topology(copy.deepcopy(topology))


def snapshot_state(state):
    """Build one link-state routers snapshot from an explicit network state."""
    ordered_nodes = state.topology.nodes
    active_adjacency = state.active_adjacency()
    return link_state_snapshot(ordered_nodes, active_adjacency, state.down_nodes)


def compute_topology(topology):
    """Static link-state forwarding tables for a decoded topology."""
    topo = _validated(topology)
    ordered_nodes = topo.nodes
    adjacency = topo.adjacency()
    routers = {
        source: forwarding_table(source, ordered_nodes, adjacency)
        for source in ordered_nodes
    }
    return {"routers": routers}


def converge_topology(topology):
    """Distance-vector convergence rounds for a decoded topology."""
    topo = _validated(topology)
    rounds = distance_vector_convergence(topo.nodes, topo.adjacency())
    return {
        "protocol": "distance-vector",
        "convergenceRound": rounds[-1]["round"],
        "rounds": rounds,
    }


def replay_validated(topo, scenario):
    """Link-state failure timeline from an already-validated topology.

    Lets a caller (the command layer) enforce topology-before-scenario
    validation precedence.  Only the scenario is validated here.
    """
    events = validate_scenario(topo, copy.deepcopy(scenario))

    state = NetworkState.initial(topo)
    timeline = [{"event": None, "routers": snapshot_state(state)}]
    for event in events:
        state = apply_event(state, event)
        timeline.append(
            {
                "time": event.time,
                "event": event.raw,
                "routers": snapshot_state(state),
            }
        )
    return {"protocol": "link-state", "timeline": timeline}


def replay_scenario(topology, scenario):
    """Link-state failure timeline for a decoded topology and scenario.

    Always begins from the fault-free state; failure state never carries over
    from a previous call.  Each step applies one event to an immutable state
    and generates a fresh routing snapshot from the resulting explicit node
    and link availability.
    """
    return replay_validated(_validated(topology), scenario)


def replay_ls_validated(topo, scenario):
    """Auditable link-state timeline from an already-validated topology.

    Mirrors :func:`replay_validated` (topology already validated, only the
    scenario validated here) but replays the protocol itself: every timeline
    entry carries ``rounds`` (each with ``round``, ``databases`` and
    ``routers``) from round 0 through a flooding fixed point, plus the
    final ``convergenceRound``.  The baseline entry has ``event: null``;
    event entries echo ``time`` and the raw event verbatim.

    Converged databases and per-node sequence history carry from one entry
    to the next, while every *call* starts from the fault-free state.
    """
    events = validate_scenario(topo, copy.deepcopy(scenario))

    ordered_nodes = topo.nodes
    state = NetworkState.initial(topo)
    baseline_neighbors = active_neighbor_sets(
        ordered_nodes, state.active_adjacency()
    )
    databases, sequences = baseline_databases(ordered_nodes, baseline_neighbors)
    baseline_rounds, databases = link_state_convergence(
        databases,
        ordered_nodes,
        topo,
        state.active_adjacency(),
        state.down_nodes,
    )
    timeline = [
        {
            "event": None,
            "convergenceRound": baseline_rounds[-1]["round"],
            "rounds": baseline_rounds,
        }
    ]

    for event in events:
        previous_state = state
        state = apply_event(state, event)
        recovered = previous_state.down_nodes - state.down_nodes
        active_adjacency = state.active_adjacency()
        new_neighbors = active_neighbor_sets(ordered_nodes, active_adjacency)
        old_neighbors = active_neighbor_sets(
            ordered_nodes, previous_state.active_adjacency()
        )
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
            topo,
            active_adjacency,
            state.down_nodes,
        )
        timeline.append(
            {
                "time": event.time,
                "event": event.raw,
                "convergenceRound": rounds[-1]["round"],
                "rounds": rounds,
            }
        )
    return {"protocol": "link-state", "timeline": timeline}


def replay_ls_scenario(topology, scenario):
    """Auditable link-state timeline for a decoded topology and scenario.

    Always begins from the fault-free state; neither databases, sequence
    counters nor failure state carry over from a previous call, and inputs
    are deep-copied so caller data and returned data never alias.
    """
    return replay_ls_validated(_validated(topology), scenario)


def replay_dv_validated(topo, scenario):
    """Distance-vector failure timeline from an already-validated topology.

    Mirrors :func:`replay_validated` so the command layer can enforce
    topology-before-scenario precedence; here the scenario must also carry
    an ``infinityMetric`` and may carry ``holdDownRounds``.

    Without ``holdDownRounds`` the result is byte-for-byte the legacy
    document::

        {"protocol": "distance-vector", "infinityMetric": ..., "timeline": ...}

    With ``holdDownRounds`` the root echoes it and every round of every
    timeline entry additionally carries ``holdDowns`` (a router ->
    destination -> remaining-rounds map, ``{}`` when no timer is live)::

        {"protocol": "distance-vector", "infinityMetric": ...,
         "holdDownRounds": ..., "timeline": ...}

    Every timeline entry holds its converged synchronous rounds (round 0
    plus each later changed round) and a ``convergenceRound`` pointer, which
    under hold-down points at the last round in which either a forwarding
    vector or a hold-down timer still changed.  The baseline entry has
    ``event: null``; event entries echo ``time`` and the raw event verbatim.
    """
    events, infinity_metric, hold_down_rounds = validate_dv_scenario(
        topo, copy.deepcopy(scenario)
    )

    ordered_nodes = topo.nodes
    state = NetworkState.initial(topo)
    baseline_adjacency = state.active_adjacency()
    baseline_rounds = distance_vector_failure_convergence(
        initial_distance_vectors(ordered_nodes, baseline_adjacency),
        ordered_nodes,
        baseline_adjacency,
        state.down_nodes,
        infinity_metric,
    )

    if not hold_down_rounds:
        # Omission preserves the legacy result; an explicit 0 is defined to
        # disable suppression entirely, so it shares the identical path.
        return _replay_dv_legacy(
            topo, events, ordered_nodes, infinity_metric, baseline_rounds, state
        )
    return _replay_dv_holddown(
        topo,
        events,
        ordered_nodes,
        infinity_metric,
        hold_down_rounds,
        baseline_rounds,
        state,
    )


def _replay_dv_legacy(
    topo, events, ordered_nodes, infinity_metric, baseline_rounds, state
):
    """The original hold-down-free replay, unchanged in every byte."""
    timeline = [
        {
            "event": None,
            "convergenceRound": baseline_rounds[-1]["round"],
            "rounds": baseline_rounds,
        }
    ]

    vectors = baseline_rounds[-1]["routers"]
    for event in events:
        state = apply_event(state, event)
        active_adjacency = state.active_adjacency()
        round_zero = distance_vector_failure_round_zero(
            vectors, ordered_nodes, active_adjacency, state.down_nodes
        )
        rounds = distance_vector_failure_convergence(
            round_zero,
            ordered_nodes,
            active_adjacency,
            state.down_nodes,
            infinity_metric,
        )
        vectors = rounds[-1]["routers"]
        timeline.append(
            {
                "time": event.time,
                "event": event.raw,
                "convergenceRound": rounds[-1]["round"],
                "rounds": rounds,
            }
        )
    return {
        "protocol": "distance-vector",
        "infinityMetric": infinity_metric,
        "timeline": timeline,
    }


def _rounds_with_empty_holddowns(rounds, ordered_nodes):
    """Fresh copy of legacy-style rounds carrying an empty ``holdDowns`` map."""
    empty = {router: {} for router in ordered_nodes}
    return [
        {
            "round": snapshot["round"],
            "routers": snapshot["routers"],
            "holdDowns": {router: dict(empty[router]) for router in ordered_nodes},
        }
        for snapshot in rounds
    ]


def _replay_dv_holddown(
    topo,
    events,
    ordered_nodes,
    infinity_metric,
    hold_down_rounds,
    baseline_rounds,
    state,
):
    """Replay in which failure-invalidated routes are held down briefly.

    The fault-free baseline never invalidates a finite route, so its rounds
    carry empty hold-down maps and exactly the legacy forwarding vectors.
    Each event after the baseline builds its round zero from the previous
    converged vectors (whose hold-down timers have all expired by then) and
    runs the synchronous hold-down convergence.
    """
    timeline = [
        {
            "event": None,
            "convergenceRound": baseline_rounds[-1]["round"],
            "rounds": _rounds_with_empty_holddowns(baseline_rounds, ordered_nodes),
        }
    ]

    vectors = baseline_rounds[-1]["routers"]
    for event in events:
        state = apply_event(state, event)
        active_adjacency = state.active_adjacency()
        round_zero, holddowns_zero = distance_vector_holddown_round_zero(
            vectors,
            ordered_nodes,
            active_adjacency,
            state.down_nodes,
            hold_down_rounds,
        )
        rounds, vectors = distance_vector_holddown_convergence(
            round_zero,
            holddowns_zero,
            ordered_nodes,
            active_adjacency,
            state.down_nodes,
            infinity_metric,
            hold_down_rounds,
        )
        timeline.append(
            {
                "time": event.time,
                "event": event.raw,
                "convergenceRound": rounds[-1]["round"],
                "rounds": rounds,
            }
        )
    return {
        "protocol": "distance-vector",
        "infinityMetric": infinity_metric,
        "holdDownRounds": hold_down_rounds,
        "timeline": timeline,
    }


def replay_dv_scenario(topology, scenario):
    """Distance-vector failure timeline for a decoded topology and scenario.

    Like :func:`replay_scenario`, every call starts from the fault-free
    state and neither reads nor retains state from a previous call.
    """
    return replay_dv_validated(_validated(topology), scenario)
