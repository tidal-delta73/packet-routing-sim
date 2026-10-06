"""Scenario validation: decode failure events into validated event records.

Validation covers structure and field types, references into the topology,
strictly increasing positive integer times, and legal state transitions
(no repeated disables, no restore of an object that is not disabled).  The
transition check is performed by folding :func:`state.apply_event` over the
events from a fault-free state, so replay and any future protocol reuse one
authoritative set of event rules rather than a copied second copy.
"""
from dataclasses import dataclass

from .errors import InvalidScenario
from .state import (
    LINK_ACTIONS,
    NODE_ACTIONS,
    NetworkState,
    apply_event,
)
from .topology import canonical_pair


@dataclass(frozen=True)
class Event:
    """One validated failure/recovery event.

    ``node`` is set for node events, ``pair`` (the canonical undirected
    endpoint tuple) for link events.  ``raw`` is a shallow copy of the
    decoded event object, preserving its original field order so timeline
    output can echo the event byte-for-byte without aliasing caller data.
    """

    time: int
    action: str
    raw: dict
    node: object = None
    pair: object = None


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def validate_scenario(topology, scenario):
    """Validate a decoded scenario against a validated topology.

    Return a tuple of immutable :class:`Event` records in time order.  Raise
    :class:`InvalidScenario` for malformed structure or values, and its
    subclass :class:`InvalidStateTransition` for an otherwise well-formed
    event that cannot be applied from the preceding state.
    """
    if not isinstance(scenario, dict):
        raise InvalidScenario("scenario must be an object")
    events_raw = scenario.get("events")
    if not isinstance(events_raw, list):
        raise InvalidScenario("events must be an array")

    node_set = topology.node_set
    known_links = topology.link_pairs

    events = []
    state = NetworkState.initial(topology)
    last_time = 0
    for event_raw in events_raw:
        if not isinstance(event_raw, dict):
            raise InvalidScenario("event must be an object")
        time = event_raw.get("time")
        action = event_raw.get("action")
        if not _is_int(time):
            raise InvalidScenario("event time must be an integer")
        if time <= 0 or time <= last_time:
            raise InvalidScenario("event times must strictly increase")
        if not isinstance(action, str):
            raise InvalidScenario("event action must be a string")

        if action in NODE_ACTIONS:
            node = event_raw.get("node")
            if not isinstance(node, str) or node not in node_set:
                raise InvalidScenario("node event must name a declared node")
            event = Event(
                time=time, action=action, raw=dict(event_raw), node=node
            )
        elif action in LINK_ACTIONS:
            source = event_raw.get("from")
            target = event_raw.get("to")
            if not isinstance(source, str) or not isinstance(target, str):
                raise InvalidScenario("link event must name from and to")
            pair = canonical_pair(source, target)
            if pair not in known_links:
                raise InvalidScenario("link event must name a declared link")
            event = Event(
                time=time, action=action, raw=dict(event_raw), pair=pair
            )
        else:
            raise InvalidScenario(f"unknown action: {action}")

        # Fold the transition rules themselves; an illegal sequence raises
        # InvalidStateTransition (a distinguishable InvalidScenario).
        state = apply_event(state, event)
        events.append(event)
        last_time = time

    return tuple(events)
