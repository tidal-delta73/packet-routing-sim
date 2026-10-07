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
every replay starts from the fault-free state.  The three replay timelines
share the single event boundary in :mod:`packet_routing_sim.core.replay`;
only their protocol drivers differ.

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
from .replay import (
    DistanceVectorProtocol,
    LinkStateProtocol,
    StaticLinkStateProtocol,
    run_timeline,
)
from .routing import (
    distance_vector_convergence,
    distance_vector_holddown_convergence,
    distance_vector_holddown_round,
    distance_vector_holddown_round_zero,
    forwarding_table,
    holddowns_public,
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
    return link_state_snapshot(
        ordered_nodes, state.active_adjacency(), state.down_nodes
    )


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
    return run_timeline(topo, StaticLinkStateProtocol(), scenario)


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
    ``routers``) from round 0 through a flooding fixed point, plus the final
    ``convergenceRound``.  The baseline entry has ``event: null``; event
    entries echo ``time`` and the raw event verbatim.

    Converged databases and per-node sequence history carry from one entry to
    the next, while every *call* starts from the fault-free state.
    """
    return run_timeline(topo, LinkStateProtocol(), scenario)


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
    topology-before-scenario precedence; here the scenario must also carry an
    ``infinityMetric`` and may carry ``holdDownRounds``.

    Without ``holdDownRounds`` the result is byte-for-byte the legacy
    document::

        {"protocol": "distance-vector", "infinityMetric": ..., "timeline": ...}

    With ``holdDownRounds`` the root echoes it and every round of every
    timeline entry additionally carries ``holdDowns`` (a router ->
    destination -> remaining-rounds map, ``{}`` when no timer is live)::

        {"protocol": "distance-vector", "infinityMetric": ...,
         "holdDownRounds": ..., "timeline": ...}

    Every timeline entry holds its converged synchronous rounds (round 0 plus
    each later changed round) and a ``convergenceRound`` pointer, which under
    hold-down points at the last round in which either a forwarding vector or
    a hold-down timer still changed.  The baseline entry has ``event: null``;
    event entries echo ``time`` and the raw event verbatim.
    """
    return run_timeline(topo, DistanceVectorProtocol(), scenario)


def replay_dv_scenario(topology, scenario):
    """Distance-vector failure timeline for a decoded topology and scenario.

    Like :func:`replay_scenario`, every call starts from the fault-free state
    and neither reads nor retains state from a previous call.
    """
    return replay_dv_validated(_validated(topology), scenario)
