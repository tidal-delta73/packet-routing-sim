"""Pure simulation core: validated topology in, plain routing data out.

Public entry points:

* :func:`compute_topology`  -> ``{"routers": ...}``
* :func:`converge_topology` -> ``{"protocol", "convergenceRound", "rounds"}``
* :func:`replay_scenario`   -> ``{"protocol", "timeline"}``

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
from .routing import (
    distance_vector_convergence,
    forwarding_table,
    link_state_snapshot,
)
from .scenario import validate_scenario
from .state import NetworkState, apply_event
from .topology import Topology, validate_topology

__all__ = [
    "compute_topology",
    "converge_topology",
    "replay_scenario",
    "replay_validated",
    "snapshot_state",
    "validate_topology",
    "validate_scenario",
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
