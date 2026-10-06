"""Explicit network availability state and failure/recovery event rules.

This module is deliberately independent of routing computation: it knows only
which nodes and links are available.  Routing snapshots are generated from a
:class:`NetworkState` by the routing module, so a future distance-vector
failure replay can reuse the exact same topology and event rules.

A state is immutable; applying an event returns a new state.  Link-down
status is tracked independently of node status: a link explicitly disabled
while an endpoint is down stays disabled after that endpoint recovers, and
link endpoints are unordered (``from``/``to`` swapped name the same link).
"""
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from .errors import InvalidScenario, InvalidStateTransition
from .topology import canonical_pair

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from .topology import Topology

NODE_DOWN = "node-down"
NODE_UP = "node-up"
LINK_DOWN = "link-down"
LINK_UP = "link-up"
NODE_ACTIONS = (NODE_DOWN, NODE_UP)
LINK_ACTIONS = (LINK_DOWN, LINK_UP)


@dataclass(frozen=True)
class NetworkState:
    """Which nodes and links are currently unavailable.

    ``down_nodes`` holds node names; ``down_links`` holds canonical
    undirected endpoint pairs.  Both are frozensets, making a state safe to
    share and reuse.  Node status never implies link status here: incident
    links of a down node are simply ignored while deriving the active
    topology, while ``down_links`` keeps the explicitly disabled set.
    """

    topology: "Topology"
    down_nodes: frozenset = frozenset()
    down_links: frozenset = frozenset()

    @classmethod
    def initial(cls, topology):
        """The fault-free state for a validated topology."""
        return cls(topology=topology)

    def is_node_up(self, node):
        return node not in self.down_nodes

    def is_link_up(self, source, target):
        pair = canonical_pair(source, target)
        return (
            pair not in self.down_links
            and source not in self.down_nodes
            and target not in self.down_nodes
        )

    def active_adjacency(self):
        """Return a fresh undirected adjacency map of usable links only.

        Rows exist for every declared node; rows of down nodes and rows'
        entries towards a down endpoint are empty/omitted.  Iteration order
        of the returned dicts is not meaningful on its own: routing
        computations always sort neighbors explicitly.
        """
        adjacency = {node: {} for node in self.topology.nodes}
        for (source, target), metric in self.topology.links:
            if self.is_link_up(source, target):
                adjacency[source][target] = metric
                adjacency[target][source] = metric
        return adjacency


def apply_event(state, event):
    """Return the state after one validated event; never mutate ``state``.

    Structural validity (field types, time order, object references) is the
    scenario validator's job; this function enforces the state-transition
    rules and raises :class:`InvalidStateTransition` when an action cannot be
    taken from the current state (repeated disable, or restore without a
    matching disable).  Reference guards are repeated defensively so the
    function stays safe when called directly.
    """
    action = event.action
    topology = state.topology

    if action in NODE_ACTIONS:
        node = event.node
        if node not in topology.node_set:
            raise InvalidScenario("event references an undeclared node")
        down = set(state.down_nodes)
        if action == NODE_DOWN:
            if node in down:
                raise InvalidStateTransition(f"node already down: {node}")
            down.add(node)
        else:
            if node not in down:
                raise InvalidStateTransition(f"node is not down: {node}")
            down.discard(node)
        return replace(state, down_nodes=frozenset(down))

    if action in LINK_ACTIONS:
        pair = event.pair
        if pair not in topology.link_pairs:
            raise InvalidScenario("event references an undeclared link")
        down = set(state.down_links)
        if action == LINK_DOWN:
            if pair in down:
                raise InvalidStateTransition(f"link already down: {pair}")
            down.add(pair)
        else:
            if pair not in down:
                raise InvalidStateTransition(f"link is not down: {pair}")
            down.discard(pair)
        return replace(state, down_links=frozenset(down))

    raise InvalidScenario(f"unknown action: {action}")
