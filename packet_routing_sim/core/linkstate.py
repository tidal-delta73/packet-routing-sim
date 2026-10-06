"""Link-state neighbor discovery, LSA flooding and per-database SPF.

This module models the link-state replay as deterministic synchronous rounds
over an explicit availability state (which nodes and links are up, supplied
by :mod:`packet_routing_sim.core.state`).  It performs no I/O of any kind and
depends on no dict insertion order: nodes and neighbors are always scanned in
sorted name order.

An LSA is ``{"sequence": <int>, "neighbors": [<sorted names>]}`` and a link-
state database maps an originator name to the newest LSA held for it.  An
incoming LSA is accepted only when its sequence number is strictly greater
than the one already held for the same originator.  The link metric is the
single undirected topology metric, hence identical from both endpoints; a
link participates in SPF only while both endpoints' newest held LSAs still
declare each other.

Round model
-----------

* Round 0 is built explicitly.  At the fault-free start every online router
  discovers its available direct neighbors, originates a sequence-1 LSA with
  those neighbors sorted by name, and installs only its own LSA into its own
  database.  After an event the converged databases are inherited; online
  endpoints whose adjacency changed bump their local sequence (a recovered
  node continues from its own historical sequence) and install the new LSA,
  everyone else keeps the old view, and a recovered node otherwise starts
  with an empty database.  Down routers originate nothing and hold no
  database.
* Each later round every online router synchronously relays the newest LSAs
  it held at the start of the round over each currently available link; a
  receiver accepts only a strictly higher-sequence LSA from the same
  originator.  Rounds are recorded while any database changes, stopping at a
  fixed point.

Every online router independently runs Dijkstra over its own current
database view, so a router that has not yet learned a newer LSA can briefly
keep an old route.  Equal-cost paths keep the smaller next-hop name.
"""
from .routing import forwarding_table


def make_lsa(sequence, neighbors):
    """Build one JSON-shaped local LSA with sorted neighbor names."""
    return {"sequence": sequence, "neighbors": sorted(neighbors)}


def copy_lsa(lsa):
    """Fresh copy of an LSA whose neighbor list shares nothing with the source."""
    return {"sequence": lsa["sequence"], "neighbors": list(lsa["neighbors"])}


def active_neighbor_sets(ordered_nodes, active_adjacency):
    """``{node: frozenset(its currently available neighbors)}``."""
    return {
        node: frozenset(active_adjacency[node]) for node in ordered_nodes
    }


def changed_endpoints(
    ordered_nodes, old_neighbors, new_neighbors, down_nodes, recovered=frozenset()
):
    """Online nodes whose discovered adjacency changed across an event.

    This is every currently online node with a different available-neighbor
    set, plus every node that just recovered (even one with no live
    neighbors, so it still originates an LSA continuing its own historical
    sequence).  A node that went down is excluded: it originates nothing.
    """
    changed = {node for node in recovered if node not in down_nodes}
    for node in ordered_nodes:
        if node in down_nodes:
            continue
        if old_neighbors.get(node, frozenset()) != new_neighbors[node]:
            changed.add(node)
    return changed


def baseline_databases(ordered_nodes, new_neighbors):
    """Fault-free round zero and the initial per-node sequence history.

    Every node originates a sequence-1 LSA and installs only its own LSA.
    """
    databases = {}
    sequences = {}
    for node in ordered_nodes:
        databases[node] = {node: make_lsa(1, new_neighbors[node])}
        sequences[node] = 1
    return databases, sequences


def event_databases(
    previous, ordered_nodes, new_neighbors, down_nodes, changed, sequences
):
    """Round-zero databases immediately after one failure/recovery event.

    Online routers inherit the previous converged database (a recovered node
    starts with an empty one).  Each online endpoint whose adjacency changed
    originates its historical sequence plus one and installs that LSA as its
    own entry.  Returns the updated sequence history alongside; the history
    survives a node's downtime so a recovery continues the numbering.
    """
    databases = {}
    for node in ordered_nodes:
        if node in down_nodes:
            continue
        databases[node] = {
            originator: copy_lsa(lsa)
            for originator, lsa in previous.get(node, {}).items()
        }

    next_sequences = dict(sequences)
    for node in sorted(changed):
        sequence = next_sequences.get(node, 0) + 1
        next_sequences[node] = sequence
        databases[node][node] = make_lsa(sequence, new_neighbors[node])
    return databases, next_sequences


def flood_round(previous, ordered_nodes, active_adjacency, down_nodes):
    """One synchronous flooding exchange from the previous round's databases.

    Each online router keeps its own previous database and accepts, from
    every currently available neighbor, strictly higher-sequence LSAs than
    the one it holds for that originator.  Every send reads only
    ``previous``, so no LSA travels more than one hop in a round.
    """
    current = {}
    for node in ordered_nodes:
        if node in down_nodes:
            continue
        current[node] = {
            originator: copy_lsa(lsa)
            for originator, lsa in previous[node].items()
        }
    for receiver in ordered_nodes:
        if receiver in down_nodes:
            continue
        held = current[receiver]
        for neighbor in sorted(active_adjacency[receiver]):
            for originator in sorted(previous[neighbor]):
                lsa = previous[neighbor][originator]
                known = held.get(originator)
                if known is None or lsa["sequence"] > known["sequence"]:
                    held[originator] = copy_lsa(lsa)
    return current


def _view_adjacency(database, topology):
    """Edges usable for SPF from one router's current database view.

    A declared undirected link participates only while the newest held LSAs
    of both endpoints still name each other; the topology metric is the
    single shared metric the two declarations agree on.  A one-sided stale
    LSA therefore contributes no edge.
    """
    declared = {}
    for originator, lsa in database.items():
        declared[originator] = frozenset(lsa["neighbors"])
    adjacency = {node: {} for node in declared}
    for (source, target), metric in topology.links:
        if source not in declared or target not in declared:
            continue
        if target not in declared[source] or source not in declared[target]:
            continue
        adjacency[source][target] = metric
        adjacency[target][source] = metric
    return adjacency


def public_round(round_number, databases, ordered_nodes, topology):
    """Build one public round: ``round``, sorted ``databases``, ``routers``.

    Every online router computes its forwarding table from its own database
    view alone; a down router has neither a database nor a forwarding row,
    both reported as ``null``.  All returned structures are fresh.
    """
    public_databases = {}
    routers = {}
    for node in ordered_nodes:
        if node not in databases:
            public_databases[node] = None
            routers[node] = None
            continue
        view = databases[node]
        public_databases[node] = {
            originator: copy_lsa(view[originator])
            for originator in sorted(view)
        }
        adjacency = _view_adjacency(view, topology)
        routers[node] = forwarding_table(node, ordered_nodes, adjacency)
    return {"round": round_number, "databases": public_databases, "routers": routers}


def link_state_convergence(round_zero, ordered_nodes, topology, active_adjacency, down_nodes):
    """Record round 0 and every later round in which a database changes.

    Returns ``(rounds, final_databases)`` so a replay can inherit the fixed
    point as the next event's round-zero basis.  Flooding only ever raises a
    held sequence toward a finite ceiling (one bounded counter per
    originator, exchanged inside finite components), so the scan always
    reaches a fixed point.
    """
    rounds = [public_round(0, round_zero, ordered_nodes, topology)]
    databases = round_zero
    while True:
        updated = flood_round(
            databases, ordered_nodes, active_adjacency, down_nodes
        )
        if updated == databases:
            break
        databases = updated
        rounds.append(
            public_round(len(rounds), databases, ordered_nodes, topology)
        )
    return rounds, databases
