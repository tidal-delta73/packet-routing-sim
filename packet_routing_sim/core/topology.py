"""Topology validation and the validated, immutable topology object.

A :class:`Topology` is produced solely by :func:`validate_topology` from a
decoded JSON value.  It stores nodes in stable name order and links keyed by
their canonical undirected endpoint pair, so every later computation is
independent of declaration order and of any dict iteration order.
"""
from dataclasses import dataclass

from .errors import InvalidTopology


@dataclass(frozen=True)
class Topology:
    """A validated undirected weighted topology.

    ``nodes`` is sorted by name; ``links`` maps canonical pair
    ``(smaller, larger)`` to its positive integer metric.  Both fields use
    only immutable types.
    """

    nodes: tuple
    links: frozenset

    @property
    def node_set(self):
        return set(self.nodes)

    @property
    def link_pairs(self):
        return frozenset(pair for pair, _metric in self.links)

    def adjacency(self):
        """Return a fresh ``{node: {neighbor: metric}}`` undirected map."""
        adjacency = {node: {} for node in self.nodes}
        for (source, target), metric in self.links:
            adjacency[source][target] = metric
            adjacency[target][source] = metric
        return adjacency


def _is_positive_int(value):
    # bool is a subclass of int; JSON true/false must not count as a metric.
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def canonical_pair(source, target):
    """Order-independent key for an undirected link endpoint pair."""
    return (source, target) if source < target else (target, source)


def validate_topology(topology):
    """Validate a decoded topology value; return an immutable :class:`Topology`.

    Raises :class:`InvalidTopology` for any malformed structure, value,
    unknown endpoint reference, self-loop, non-positive/non-integer metric
    or duplicate undirected link.
    """
    if not isinstance(topology, dict):
        raise InvalidTopology("topology must be an object")
    nodes_raw = topology.get("nodes")
    links_raw = topology.get("links")
    if not isinstance(nodes_raw, list) or not isinstance(links_raw, list):
        raise InvalidTopology("nodes and links must be arrays")

    nodes = []
    declared = set()
    for node in nodes_raw:
        if not isinstance(node, str) or node == "" or node in declared:
            raise InvalidTopology("illegal node declaration")
        declared.add(node)
        nodes.append(node)

    links = set()
    seen_pairs = set()
    for link in links_raw:
        if not isinstance(link, dict):
            raise InvalidTopology("link must be an object")
        if "from" not in link or "to" not in link or "metric" not in link:
            raise InvalidTopology("link requires from, to and metric")
        source, target, metric = link["from"], link["to"], link["metric"]
        if not isinstance(source, str) or not isinstance(target, str):
            raise InvalidTopology("link endpoints must be strings")
        if source not in declared or target not in declared:
            raise InvalidTopology("link references an undeclared node")
        if source == target:
            raise InvalidTopology("self-loops are not allowed")
        if not _is_positive_int(metric):
            raise InvalidTopology("metric must be a positive integer")
        pair = canonical_pair(source, target)
        if pair in seen_pairs:
            raise InvalidTopology("duplicate undirected link")
        seen_pairs.add(pair)
        links.add((pair, metric))

    return Topology(nodes=tuple(sorted(nodes)), links=frozenset(links))
