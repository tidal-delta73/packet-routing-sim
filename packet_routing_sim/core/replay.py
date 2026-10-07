"""Unified failure-timeline replay over one shared event boundary.

The three replay entry points (static link-state ``replay``, auditable
link-state ``replay-ls`` and distance-vector ``replay-dv``) differ only in
what each timeline entry publishes and in the protocol state carried from
one converged entry to the next.  Everything around that is the same shape,
so it lives here once:

* :func:`run_timeline` is the single event-advancing boundary.  The driver
  first validates a defensive copy of the scenario; every call then starts
  from the fault-free :class:`~packet_routing_sim.core.state.NetworkState`,
  publishes exactly one baseline entry (``event`` is ``None``), and folds the
  validated events in time order with
  :func:`~packet_routing_sim.core.state.apply_event`.  After each event the
  driver sees the states on both sides of the transition and returns the
  entry's published payload plus its own next inherited state.
* A :class:`ReplayProtocol` driver owns *only* protocol-specific concerns:
  scenario validation extras, the fault-free baseline, one post-event
  advance, and root-object assembly.  Its inherited state (converged
  forwarding vectors, or link-state databases plus sequence history) is
  opaque to the engine, which keeps protocol-specific data isolated.

Timeline entry fields (``event``/``time`` placement, baseline position,
event echoing) are assembled by the engine, not the drivers, so the three
entries cannot drift apart.

The module performs no file or stream I/O and holds no shared mutable state:
a driver is created per call, inputs are deep-copied on validation, and all
published structures are built fresh.
"""
import copy

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
from .scenario import validate_dv_scenario, validate_scenario
from .state import NetworkState, apply_event


def run_timeline(topo, protocol, scenario):
    """Validate ``scenario`` and fold its events from a fault-free state.

    ``protocol`` is a fresh :class:`ReplayProtocol`.  Its ``validate`` method
    decodes a defensive scenario copy into the validated event tuple (and may
    record protocol-specific validated options on itself).  The engine then
    publishes the driver's baseline entry first (``event`` is ``None``) and,
    for every event, applies it to an immutable network state and asks the
    driver for the next entry payload and inherited state.  Finally the
    driver packages the root result object.
    """
    events = protocol.validate(topo, scenario)

    state = NetworkState.initial(topo)
    baseline_payload, inherited = protocol.baseline(state)
    timeline = [{"event": None, **baseline_payload}]

    for event in events:
        previous_state = state
        state = apply_event(state, event)
        payload, inherited = protocol.advance(
            event, previous_state, state, inherited
        )
        timeline.append({"time": event.time, "event": event.raw, **payload})
    return protocol.result(timeline)


class ReplayProtocol:
    """Protocol-specific boundary for a timeline replay.

    A driver receives fresh availability states and returns plain-Python
    published data.  Whatever it needs converged from one entry to the next
    is carried as its own opaque ``inherited`` value; the engine never
    inspects it.
    """

    def validate(self, topo, scenario):
        """Decode a defensive scenario copy; return the validated events."""
        raise NotImplementedError

    def baseline(self, state):
        """Return ``(published_payload, inherited)`` for fault-free state."""
        raise NotImplementedError

    def advance(self, event, previous_state, state, inherited):
        """Return ``(published_payload, inherited)`` after one event.

        ``previous_state`` is the availability state immediately before the
        event and ``state`` immediately after it; most protocols only need
        ``state``, while link-state needs the transition to spot recovered
        nodes and changed adjacencies.
        """
        raise NotImplementedError

    def result(self, timeline):
        """Package the finished timeline into the entry point's root object."""
        raise NotImplementedError


def static_snapshot(state):
    """Per-node static shortest-path tables from an availability state."""
    ordered_nodes = state.topology.nodes
    return link_state_snapshot(
        ordered_nodes, state.active_adjacency(), state.down_nodes
    )


def _convergence_payload(rounds):
    """The payload shared by every auditable protocol entry."""
    return {"convergenceRound": rounds[-1]["round"], "rounds": rounds}


class StaticLinkStateProtocol(ReplayProtocol):
    """The original ``replay``: one static shortest-path snapshot per entry."""

    def validate(self, topo, scenario):
        return validate_scenario(topo, copy.deepcopy(scenario))

    def baseline(self, state):
        return {"routers": static_snapshot(state)}, None

    def advance(self, event, previous_state, state, inherited):
        return {"routers": static_snapshot(state)}, None

    def result(self, timeline):
        return {"protocol": "link-state", "timeline": timeline}


class LinkStateProtocol(ReplayProtocol):
    """Auditable ``replay-ls``: neighbor discovery, flooding, per-DB SPF.

    The inherited value is ``(converged_databases, sequence_history)``: both
    start at the fault-free baseline and carry between entries so a recovered
    node continues its own historical sequence.  The timeline engine never
    sees either object.
    """

    def validate(self, topo, scenario):
        return validate_scenario(topo, copy.deepcopy(scenario))

    def baseline(self, state):
        topo = state.topology
        ordered_nodes = topo.nodes
        initial_neighbors = active_neighbor_sets(
            ordered_nodes, state.active_adjacency()
        )
        databases, sequences = baseline_databases(
            ordered_nodes, initial_neighbors
        )
        rounds, databases = link_state_convergence(
            databases,
            ordered_nodes,
            topo,
            state.active_adjacency(),
            state.down_nodes,
        )
        return _convergence_payload(rounds), (databases, sequences)

    def advance(self, event, previous_state, state, inherited):
        databases, sequences = inherited
        topo = state.topology
        ordered_nodes = topo.nodes
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
        return _convergence_payload(rounds), (databases, sequences)

    def result(self, timeline):
        return {"protocol": "link-state", "timeline": timeline}


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


class DistanceVectorProtocol(ReplayProtocol):
    """Synchronous ``replay-dv`` with infinity and optional hold-down.

    Validated options (``infinity_metric``, ``hold_down_rounds``) are recorded
    on this per-call instance by :meth:`validate`; the inherited value is just
    the previous entry's converged forwarding vectors, since every entry runs
    to a fixed point and hold-down timers have all expired there.
    """

    def __init__(self):
        self.infinity_metric = None
        self.hold_down_rounds = None

    @property
    def holddown_enabled(self):
        # Omission (None) keeps the legacy document; an explicit 0 disables
        # suppression and deliberately shares that identical path.
        return bool(self.hold_down_rounds)

    def validate(self, topo, scenario):
        events, infinity_metric, hold_down_rounds = validate_dv_scenario(
            topo, copy.deepcopy(scenario)
        )
        self.infinity_metric = infinity_metric
        self.hold_down_rounds = hold_down_rounds
        return events

    def baseline(self, state):
        ordered_nodes = state.topology.nodes
        baseline_adjacency = state.active_adjacency()
        rounds = distance_vector_failure_convergence(
            initial_distance_vectors(ordered_nodes, baseline_adjacency),
            ordered_nodes,
            baseline_adjacency,
            state.down_nodes,
            self.infinity_metric,
        )
        if self.holddown_enabled:
            # The fault-free baseline never invalidates a finite route, so
            # its rounds carry empty hold-down maps over legacy vectors.
            payload = _convergence_payload(
                _rounds_with_empty_holddowns(rounds, ordered_nodes)
            )
        else:
            payload = _convergence_payload(rounds)
        return payload, rounds[-1]["routers"]

    def advance(self, event, previous_state, state, inherited):
        vectors = inherited
        ordered_nodes = state.topology.nodes
        active_adjacency = state.active_adjacency()
        if self.holddown_enabled:
            round_zero, holddowns_zero = distance_vector_holddown_round_zero(
                vectors,
                ordered_nodes,
                active_adjacency,
                state.down_nodes,
                self.hold_down_rounds,
            )
            rounds, vectors = distance_vector_holddown_convergence(
                round_zero,
                holddowns_zero,
                ordered_nodes,
                active_adjacency,
                state.down_nodes,
                self.infinity_metric,
                self.hold_down_rounds,
            )
        else:
            round_zero = distance_vector_failure_round_zero(
                vectors,
                ordered_nodes,
                active_adjacency,
                state.down_nodes,
            )
            rounds = distance_vector_failure_convergence(
                round_zero,
                ordered_nodes,
                active_adjacency,
                state.down_nodes,
                self.infinity_metric,
            )
        return _convergence_payload(rounds), rounds[-1]["routers"]

    def result(self, timeline):
        result = {
            "protocol": "distance-vector",
            "infinityMetric": self.infinity_metric,
        }
        if self.holddown_enabled:
            result["holdDownRounds"] = self.hold_down_rounds
        result["timeline"] = timeline
        return result
