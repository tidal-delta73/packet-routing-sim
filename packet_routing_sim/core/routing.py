"""Routing computations over an explicit topology/availability state.

Two protocols live here:

* link-state shortest-path forwarding tables (Dijkstra) used by ``compute``
  and every replay snapshot;
* synchronous distance-vector rounds used by ``converge``.

Both are pure functions of the supplied adjacency map and ordered node
tuple.  Neighbors are always iterated in sorted name order and equal-metric
choices use a strict comparison, so ties deterministically select the
smaller next-hop name regardless of dict insertion order or hash seed.
"""
import heapq


def forwarding_table(source, ordered_nodes, adjacency):
    """Dijkstra from ``source``; ties on total metric pick the smaller hop."""
    distances = {source: 0}
    next_hops = {}
    # Label is (total metric, first hop name); "" is a sentinel for the source
    # and sorts before any real (necessarily non-empty) node name.
    heap = [(0, "", source)]
    settled = set()
    while heap:
        distance, first_hop, node = heapq.heappop(heap)
        if node in settled:
            continue
        settled.add(node)
        distances[node] = distance
        next_hops[node] = first_hop
        for neighbor in sorted(adjacency[node]):
            if neighbor in settled:
                continue
            metric = adjacency[node][neighbor]
            candidate = distance + metric
            candidate_hop = neighbor if node == source else first_hop
            heapq.heappush(heap, (candidate, candidate_hop, neighbor))

    table = {}
    for destination in ordered_nodes:
        if destination == source:
            table[destination] = {"nextHop": None, "metric": 0}
        elif destination in distances:
            table[destination] = {
                "nextHop": next_hops[destination],
                "metric": distances[destination],
            }
        else:
            table[destination] = {"nextHop": None, "metric": None}
    return table


def link_state_snapshot(ordered_nodes, active_adjacency, down_nodes):
    """Forwarding tables for every node from an availability state.

    Up routers are computed against the subgraph of usable links; a down
    router's whole row is ``null``/``null``.  Destinations that are down or
    unreachable likewise come out ``null``/``null`` from Dijkstra.
    """
    routers = {}
    for source in ordered_nodes:
        if source in down_nodes:
            routers[source] = {
                destination: {"nextHop": None, "metric": None}
                for destination in ordered_nodes
            }
        else:
            routers[source] = forwarding_table(
                source, ordered_nodes, active_adjacency
            )
    return routers


def initial_distance_vectors(ordered_nodes, adjacency):
    """Round 0: each router knows only itself and its directly connected links."""
    vectors = {}
    for router in ordered_nodes:
        table = {}
        for destination in ordered_nodes:
            if destination == router:
                table[destination] = {"nextHop": None, "metric": 0}
            elif destination in adjacency[router]:
                table[destination] = {
                    "nextHop": destination,
                    "metric": adjacency[router][destination],
                }
            else:
                table[destination] = {"nextHop": None, "metric": None}
        vectors[router] = table
    return vectors


def distance_vector_round(previous, ordered_nodes, adjacency):
    """Synchronously update every router from the previous round's ads."""
    current = {}
    for router in ordered_nodes:
        table = {}
        for destination in ordered_nodes:
            if destination == router:
                table[destination] = {"nextHop": None, "metric": 0}
                continue
            best_metric = None
            best_hop = None
            for neighbor in sorted(adjacency[router]):
                advertised = previous[neighbor][destination]
                if advertised["metric"] is None:
                    continue
                candidate = adjacency[router][neighbor] + advertised["metric"]
                # Neighbors are iterated in name order, so the first best
                # candidate keeps the smaller next-hop name on ties.
                if best_metric is None or candidate < best_metric:
                    best_metric = candidate
                    best_hop = neighbor
            if best_metric is None:
                table[destination] = {"nextHop": None, "metric": None}
            else:
                table[destination] = {"nextHop": best_hop, "metric": best_metric}
        current[router] = table
    return current


def distance_vector_convergence(ordered_nodes, adjacency):
    """Return every changed synchronous round from round 0 to a fixed point."""
    vectors = initial_distance_vectors(ordered_nodes, adjacency)
    rounds = [{"round": 0, "routers": vectors}]
    while True:
        updated = distance_vector_round(vectors, ordered_nodes, adjacency)
        if updated == vectors:
            break
        vectors = updated
        rounds.append({"round": len(rounds), "routers": vectors})
    return rounds


def _unreachable_table(ordered_nodes):
    """A fresh all-``null``/``null`` row, as advertised by a down router."""
    return {
        destination: {"nextHop": None, "metric": None}
        for destination in ordered_nodes
    }


def distance_vector_round_bounded(
    previous, ordered_nodes, adjacency, infinity, down_nodes=frozenset()
):
    """Synchronous update like :func:`distance_vector_round`, with two rules
    specific to failure replay:

    * a candidate metric at or above ``infinity`` means unreachable, so the
      route is recorded as ``null``/``null`` (counting to infinity) instead
      of being adopted;
    * every down router's whole row stays ``null``/``null``.

    Only ``previous`` (the prior round) and the currently usable neighbors in
    ``adjacency`` are read; there is no split horizon or poison reverse.
    """
    current = {}
    for router in ordered_nodes:
        if router in down_nodes:
            current[router] = _unreachable_table(ordered_nodes)
            continue
        table = {}
        for destination in ordered_nodes:
            if destination == router:
                table[destination] = {"nextHop": None, "metric": 0}
                continue
            best_metric = None
            best_hop = None
            for neighbor in sorted(adjacency[router]):
                advertised = previous[neighbor][destination]
                if advertised["metric"] is None:
                    continue
                candidate = adjacency[router][neighbor] + advertised["metric"]
                if candidate >= infinity:
                    continue
                # Neighbors are iterated in name order, so the first best
                # candidate keeps the smaller next-hop name on ties.
                if best_metric is None or candidate < best_metric:
                    best_metric = candidate
                    best_hop = neighbor
            if best_metric is None:
                table[destination] = {"nextHop": None, "metric": None}
            else:
                table[destination] = {"nextHop": best_hop, "metric": best_metric}
        current[router] = table
    return current


def distance_vector_event_round(
    previous,
    ordered_nodes,
    active_adjacency,
    down_nodes,
    recovered_nodes,
):
    """Build round 0 immediately after a failure/recovery event.

    The post-event rules are:

    * a router that is down advertises an entirely unreachable row;
    * a router that just recovered starts cold: it knows only itself and
      its currently usable direct neighbors;
    * every other online router keeps the routes it advertised in the
      previous phase's converged state, except that a route whose next hop
      is a neighbor that is no longer directly usable becomes unreachable
      immediately (a directly usable neighbor route is always known at
      once, which also re-installs routes over a recovered link).

    Everything else is left to the following synchronous rounds: routes
    learned via still-available neighbors are retained even when their
    destination is the failed node, so counting to infinity unfolds round
    by round exactly as advertised.  ``previous`` is never mutated.
    """
    current = {}
    cold_views = initial_distance_vectors(ordered_nodes, active_adjacency)
    for router in ordered_nodes:
        if router in down_nodes:
            current[router] = _unreachable_table(ordered_nodes)
            continue
        available = active_adjacency[router]
        if router in recovered_nodes:
            current[router] = cold_views[router]
            continue
        table = {}
        for destination in ordered_nodes:
            if destination == router:
                table[destination] = {"nextHop": None, "metric": 0}
            elif destination in available:
                # A directly connected, usable link is always known at once;
                # this also re-installs a route over a recovered link.
                table[destination] = {
                    "nextHop": destination,
                    "metric": available[destination],
                }
            else:
                prior = previous[router][destination]
                hop = prior["nextHop"]
                if prior["metric"] is None or hop not in available:
                    # No prior advertisement, or one via a now-failed direct
                    # neighbor: unreachable immediately, before any new ads.
                    table[destination] = {"nextHop": None, "metric": None}
                else:
                    table[destination] = {
                        "nextHop": hop,
                        "metric": prior["metric"],
                    }
        current[router] = table
    return current


def distance_vector_bounded_convergence(
    ordered_nodes, initial, adjacency, infinity, down_nodes=frozenset()
):
    """Record round 0 and every changed bounded round up to a fixed point.

    Unlike :func:`distance_vector_convergence`, round 0 is supplied by the
    caller (the fault-free initial view or an event-adjusted view), updates
    treat metrics at or above ``infinity`` as unreachable, and down routers'
    rows stay entirely unreachable.  The bounded update guarantees the
    counting-to-infinity process terminates.
    """
    vectors = initial
    rounds = [{"round": 0, "routers": vectors}]
    while True:
        updated = distance_vector_round_bounded(
            vectors, ordered_nodes, adjacency, infinity, down_nodes
        )
        if updated == vectors:
            break
        vectors = updated
        rounds.append({"round": len(rounds), "routers": vectors})
    return rounds
