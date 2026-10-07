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


def _unreachable_row(ordered_nodes):
    """A fresh row advertising every destination (including self) as down."""
    return {
        destination: {"nextHop": None, "metric": None}
        for destination in ordered_nodes
    }


def distance_vector_failure_round_zero(
    previous, ordered_nodes, active_adjacency, down_nodes
):
    """Round 0 immediately after a failure/recovery event.

    Every online router keeps the routes it advertised in the previous
    converged state, except:

    * a route whose next hop is no longer an available direct neighbor (the
      failed or disconnected neighbor) becomes unreachable;
    * a currently available direct neighbor always advertises the direct
      route again, so a recovered link is usable at once;
    * a down router's whole row is unreachable.

    A recovering router is handed a prior row of its own that is entirely
    unreachable, which leaves it knowing only itself and its currently
    available direct neighbors.
    """
    vectors = {}
    for router in ordered_nodes:
        if router in down_nodes:
            vectors[router] = _unreachable_row(ordered_nodes)
            continue
        table = {}
        prior = previous[router]
        for destination in ordered_nodes:
            if destination == router:
                table[destination] = {"nextHop": None, "metric": 0}
            elif destination in active_adjacency[router]:
                table[destination] = {
                    "nextHop": destination,
                    "metric": active_adjacency[router][destination],
                }
            else:
                entry = prior[destination]
                hop = entry["nextHop"]
                if hop is None or hop not in active_adjacency[router]:
                    table[destination] = {"nextHop": None, "metric": None}
                else:
                    # A copy, so retained advertisements never alias caller
                    # data or later rounds.
                    table[destination] = dict(entry)
        vectors[router] = table
    return vectors


def distance_vector_round_infinity(
    previous, ordered_nodes, adjacency, down_nodes, infinity_metric
):
    """Synchronously update from prior ads with an infinity threshold.

    Same scan/tie rules as :func:`distance_vector_round`, with no split
    horizon or poison reverse: every available neighbor's previous-round
    advertisement is read.  A candidate metric at or above
    ``infinity_metric`` counts as unreachable.  Down routers keep an
    entirely unreachable row.
    """
    current = {}
    for router in ordered_nodes:
        if router in down_nodes:
            current[router] = _unreachable_row(ordered_nodes)
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
                if candidate >= infinity_metric:
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


def distance_vector_failure_convergence(
    round_zero, ordered_nodes, active_adjacency, down_nodes, infinity_metric
):
    """Record round 0 and every later changed round up to a fixed point.

    After a cost increase a stale route's metric rises by at least the link
    cost along its dependency cycle until it reaches the threshold and is
    flushed as unreachable (the count-to-infinity process), so the scan
    always terminates; a surviving finite alternative wins earlier.
    """
    rounds = [{"round": 0, "routers": round_zero}]
    vectors = round_zero
    while True:
        updated = distance_vector_round_infinity(
            vectors,
            ordered_nodes,
            active_adjacency,
            down_nodes,
            infinity_metric,
        )
        if updated == vectors:
            break
        vectors = updated
        rounds.append({"round": len(rounds), "routers": vectors})
    return rounds


# ---------------------------------------------------------------------------
# Distance-vector replay with optional route hold-down
# ---------------------------------------------------------------------------
#
# Hold-down keeps per (online router, destination) timers held *beside* the
# forwarding tables; the tables themselves keep the published
# ``{"nextHop", "metric"}`` shape.  A timer map stores only positive remaining
# round counts, so an absent/zero entry means "not held".  Timers are an
# internal simulation detail: the public per-round ``holdDowns`` view is built
# from them by :func:`public_hold_downs`.
#
# While a destination is held it is advertised as unreachable and finite
# advertisements from other neighbors cannot reinstall a route; a route whose
# selected next hop is unavailable, or whose selected next hop now advertises
# the destination unreachable, starts (or, without extending, rejoins) a
# timer at the post-failure round zero.  A destination that is currently a
# direct neighbor is always adopted at once at the direct metric and clears
# any timer.


def empty_hold_down_timers(ordered_nodes):
    """Fresh zero timer map for every declared router."""
    return {router: {} for router in ordered_nodes}


def public_hold_downs(timers, ordered_nodes):
    """Build the public per-router ``{destination: positive remaining}`` view.

    Every declared router is present; a router with no active timer (and
    every down router) maps to an empty object.  Destinations are emitted in
    node order so serialization is independent of dict insertion order.
    """
    return {
        router: {
            destination: timers.get(router, {})[destination]
            for destination in ordered_nodes
            if timers.get(router, {}).get(destination, 0) > 0
        }
        for router in ordered_nodes
    }


def distance_vector_hold_down_round_zero(
    previous,
    ordered_nodes,
    active_adjacency,
    down_nodes,
    hold_down_rounds,
    previous_timers,
):
    """Round 0 immediately after an event, with route hold-down enabled.

    The plain round zero is exactly :func:`distance_vector_failure_round_zero`:
    every online router applies only its own local change, keeping a prior
    finite entry whose selected next hop is still an available direct
    neighbor.  Hold-down then additionally invalidates a kept finite route
    whose selected next hop itself advertises the destination unreachable in
    this same round zero -- the withdrawal that would otherwise start
    count-to-infinity one round later.  Every such lost finite route on an
    online router starts a fresh timer of ``hold_down_rounds``; a destination
    whose timer is already running keeps its remaining rounds (a repeated
    unreachable advertisement never extends it).  Becoming a direct neighbor
    adopts the direct metric at once and clears the timer.  Down routers keep
    an entirely unreachable row and hold no timers.
    """
    plain = distance_vector_failure_round_zero(
        previous, ordered_nodes, active_adjacency, down_nodes
    )
    vectors = {}
    timers = {}
    for router in ordered_nodes:
        if router in down_nodes:
            vectors[router] = plain[router]
            timers[router] = {}
            continue
        table = {}
        row_timers = {}
        prior_timers = previous_timers.get(router, {})
        prior = previous[router]
        for destination in ordered_nodes:
            if destination == router:
                table[destination] = {"nextHop": None, "metric": 0}
                continue
            remaining = prior_timers.get(destination, 0)
            if destination in active_adjacency[router]:
                # The direct route is always trusted at once and clears any
                # outstanding timer for the destination.  A fresh copy keeps
                # this table independent of the caller's.
                table[destination] = dict(plain[router][destination])
                continue
            prior_entry = prior[destination]
            hop = prior_entry["nextHop"]
            finite_route_lost = (
                hop is not None
                and (
                    hop not in active_adjacency[router]
                    or plain[hop][destination]["metric"] is None
                )
            )
            if remaining > 0:
                # Already held: stay unreachable and keep the remaining
                # rounds; a renewed withdrawal must not extend the timer.
                table[destination] = {"nextHop": None, "metric": None}
                row_timers[destination] = remaining
            elif finite_route_lost:
                table[destination] = {"nextHop": None, "metric": None}
                if hold_down_rounds > 0:
                    row_timers[destination] = hold_down_rounds
            else:
                # Retained finite advertisement (or an already-unreachable
                # destination with no timer); a fresh copy never aliases.
                table[destination] = dict(plain[router][destination])
        vectors[router] = table
        timers[router] = row_timers
    return vectors, timers


def distance_vector_hold_down_round(
    previous,
    ordered_nodes,
    adjacency,
    down_nodes,
    infinity_metric,
    timers,
):
    """One synchronous update with the same scan rules, honoring hold-down.

    A held destination stays unreachable regardless of every neighbor's
    finite advertisement, and its remaining count drops by one; the round
    after the count reaches zero it rejoins normal selection.  Direct
    adoption on a recovered link is handled entirely at round zero --
    between events the adjacency is static, so a held destination never
    newly becomes a direct neighbor here -- and every non-held
    destination, a direct neighbor included, runs the ordinary Bellman-
    Ford scan of :func:`distance_vector_round_infinity` exactly, including
    the infinity threshold and the smaller-next-hop tie rule.  A direct
    neighbor is therefore still free to lose to a shorter indirect route.
    """
    current = {}
    next_timers = {}
    for router in ordered_nodes:
        if router in down_nodes:
            current[router] = _unreachable_row(ordered_nodes)
            next_timers[router] = {}
            continue
        table = {}
        row_timers = {}
        prior_timers = timers.get(router, {})
        for destination in ordered_nodes:
            if destination == router:
                table[destination] = {"nextHop": None, "metric": 0}
                continue
            remaining = prior_timers.get(destination, 0)
            if remaining > 0:
                table[destination] = {"nextHop": None, "metric": None}
                if remaining - 1 > 0:
                    row_timers[destination] = remaining - 1
                continue
            best_metric = None
            best_hop = None
            for neighbor in sorted(adjacency[router]):
                advertised = previous[neighbor][destination]
                if advertised["metric"] is None:
                    continue
                candidate = adjacency[router][neighbor] + advertised["metric"]
                if candidate >= infinity_metric:
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
        next_timers[router] = row_timers
    return current, next_timers


def distance_vector_hold_down_convergence(
    round_zero,
    initial_timers,
    ordered_nodes,
    active_adjacency,
    down_nodes,
    infinity_metric,
):
    """Record round 0 onward until both tables and hold-down timers are fixed.

    Returns ``(rounds, final_vectors, final_timers)`` so a replay can inherit
    the fixed point and any still-active timers as the next event's basis.
    Each public round is ``{"round", "routers", "holdDowns"}``; recording
    continues while either the forwarding tables or any remaining timer
    changes, so a round that only counts a timer down is still auditable.
    """
    rounds = [
        {
            "round": 0,
            "routers": round_zero,
            "holdDowns": public_hold_downs(initial_timers, ordered_nodes),
        }
    ]
    vectors = round_zero
    timers = initial_timers
    while True:
        updated, updated_timers = distance_vector_hold_down_round(
            vectors,
            ordered_nodes,
            active_adjacency,
            down_nodes,
            infinity_metric,
            timers,
        )
        if updated == vectors and updated_timers == timers:
            break
        vectors = updated
        timers = updated_timers
        rounds.append(
            {
                "round": len(rounds),
                "routers": vectors,
                "holdDowns": public_hold_downs(timers, ordered_nodes),
            }
        )
    return rounds, vectors, timers
