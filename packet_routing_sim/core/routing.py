"""Routing computations over an explicit topology/availability state.

Two protocols live here:

* link-state shortest-path forwarding tables (Dijkstra) used by ``compute``
  and every replay snapshot;
* synchronous distance-vector rounds used by ``converge`` and ``replay-dv``
  (with or without route hold-down).

Both are pure functions of the supplied adjacency map and ordered node
tuple.  Neighbors are always iterated in sorted name order and equal-metric
choices use a strict comparison, so ties deterministically select the
smaller next-hop name regardless of dict insertion order or hash seed.

Every distance-vector path shares one per-round route-selection kernel
(:func:`_select_route` plus :func:`_online_router_row`): candidate
filtering, the optional infinity cutoff, the smaller-next-hop tie break and
the down-router unreachable row live exactly once.  The three modes differ
only in what surrounds that scan -- their own round zero (cold start vs
event inheritance vs hold-down closure) and, for hold-down, the suppression
timers carried beside the vectors -- and in when a round counts as stable.
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


# ---------------------------------------------------------------------------
# Shared distance-vector building blocks
# ---------------------------------------------------------------------------
#
# The per-round semantics every distance-vector mode has in common:
#
# * a route is selected from the available direct neighbors' previous-round
#   advertisements: a ``null`` advertisement offers no candidate, and the
#   candidate metric is the direct link metric plus the advertised metric;
# * an optional infinity threshold discards every candidate at or above it;
# * the smallest total metric wins, and equal metrics keep the smaller
#   next-hop name (neighbors are scanned in sorted name order and the first
#   best candidate is kept);
# * a router's own entry is always ``null``/``0``;
# * an offline router advertises an entirely ``null``/``null`` row.
#
# ``converge`` runs this scan with no threshold and no down routers; the
# plain replay-dv mode adds the threshold and down-router rows; hold-down
# adds its suppression rules before falling back to this very scan.


def _unreachable_row(ordered_nodes):
    """A fresh row advertising every destination (including self) as down."""
    return {
        destination: {"nextHop": None, "metric": None}
        for destination in ordered_nodes
    }


def _select_route(router, destination, previous, adjacency, infinity_metric):
    """Select one route via the single shared best-neighbor scan.

    Returns the published ``{"nextHop", "metric"}`` entry: the available
    neighbor offering the smallest total metric, or ``null``/``null`` when no
    neighbor advertises a usable route.  ``infinity_metric`` is ``None`` for
    the threshold-free ``converge`` scan; otherwise a candidate at or above
    the threshold counts as unreachable.  Neighbors are scanned in sorted
    name order, so on a metric tie the first (smaller-named) next hop is the
    one retained.
    """
    best_metric = None
    best_hop = None
    for neighbor in sorted(adjacency[router]):
        advertised = previous[neighbor][destination]["metric"]
        if advertised is None:
            continue
        candidate = adjacency[router][neighbor] + advertised
        if infinity_metric is not None and candidate >= infinity_metric:
            continue
        # First best candidate keeps the smaller next-hop name on ties.
        if best_metric is None or candidate < best_metric:
            best_metric = candidate
            best_hop = neighbor
    if best_metric is None:
        return {"nextHop": None, "metric": None}
    return {"nextHop": best_hop, "metric": best_metric}


def _online_router_row(
    router, ordered_nodes, previous, adjacency, infinity_metric
):
    """One online router's synchronously updated row via the shared scan.

    The router's own entry is ``null``/``0``; every other destination comes
    from :func:`_select_route`.
    """
    row = {}
    for destination in ordered_nodes:
        if destination == router:
            row[destination] = {"nextHop": None, "metric": 0}
        else:
            row[destination] = _select_route(
                router, destination, previous, adjacency, infinity_metric
            )
    return row


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


# ---------------------------------------------------------------------------
# Event round zero: plain inheritance of the previous converged vectors
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Route hold-down (optional replay-dv suppression)
# ---------------------------------------------------------------------------
#
# A hold-down keeps a destination unreachable for a finite number of full
# synchronous update rounds after the router's own finite route to it was
# invalidated, so a possibly unstable alternative cannot be adopted at once.
# The suppression state is a ``{router: {destination: remaining_rounds}}``
# map carried *beside* the advertised vectors (never inside them): a held
# route still advertises exactly ``null``/``null``, so the event format, the
# infinity threshold, synchronous updates and tie breaks are all unchanged.
#
# An internal timer holds a *positive* remaining count while it is live:
#
# * Round 0 (immediately after an event) starts every invalidated route's
#   timer at ``hold_down`` and publishes that count.  Invalidation is found
#   as a monotone closure: a retained finite route whose selected next hop is
#   gone, is down, or itself advertises the destination unreachable at round 0
#   is itself invalidated -- and that conclusion propagates transitively.
# * Each later synchronous round suppresses while the carried count is
#   positive, then hands the next round one less.  A route invalidated later
#   (its selected hop's advertisement turns unusable) starts a fresh window at
#   that round.  More unreachability arriving inside a live window simply
#   keeps counting down; it never extends the window.
# * A destination that is a current direct neighbor is always adopted at its
#   direct metric, at any round, and its timer is cleared immediately.
# * Once the count reaches zero the *next* round rejoins ordinary selection.
#
# Down routers hold no timers; their whole row stays ``null``/``null``.


def _empty_holddowns(ordered_nodes):
    """A fresh ``{router: {destination: remaining}}`` map, one row per router."""
    return {router: {} for router in ordered_nodes}


def holddowns_public(holddowns, ordered_nodes):
    """A fresh sorted copy of the suppression map for published output."""
    return {
        router: {
            destination: holddowns[router][destination]
            for destination in sorted(holddowns[router])
        }
        for router in ordered_nodes
    }


def _holddown_round_zero_vectors(
    previous, ordered_nodes, active_adjacency, down_nodes
):
    """Round-zero forwarding vectors under hold-down.

    These agree with :func:`distance_vector_failure_round_zero` and then
    transitively null any retained finite route whose selected next hop now
    advertises the destination as unreachable.  Direct neighbors, self routes
    and down-router rows are exactly as in the legacy round zero.  The map is
    monotone (finite routes only become null), so iteration reaches a unique
    least fixed point independent of scan order.
    """
    current = distance_vector_failure_round_zero(
        previous, ordered_nodes, active_adjacency, down_nodes
    )
    while True:
        updated = {}
        for router in ordered_nodes:
            if router in down_nodes:
                updated[router] = _unreachable_row(ordered_nodes)
                continue
            row = {}
            for destination in ordered_nodes:
                if destination == router:
                    row[destination] = {"nextHop": None, "metric": 0}
                elif destination in active_adjacency[router]:
                    row[destination] = {
                        "nextHop": destination,
                        "metric": active_adjacency[router][destination],
                    }
                else:
                    entry = current[router][destination]
                    hop = entry["nextHop"]
                    if hop is None or hop not in active_adjacency[router]:
                        row[destination] = {"nextHop": None, "metric": None}
                    elif current[hop][destination]["metric"] is None:
                        # The still-present selected hop no longer offers a
                        # way to the destination: null it, possibly enabling
                        # further nulls on the next closure pass.
                        row[destination] = {"nextHop": None, "metric": None}
                    else:
                        row[destination] = dict(entry)
            updated[router] = row
        if updated == current:
            return current
        current = updated


def distance_vector_holddown_round_zero(
    previous, ordered_nodes, active_adjacency, down_nodes, hold_down
):
    """Round 0 after an event with hold-down enabled.

    Returns ``(vectors, holddowns)`` where every online router's previously
    finite, non-direct route that round zero renders unreachable starts a
    ``hold_down``-round timer; everything else carries no timer.
    """
    vectors = _holddown_round_zero_vectors(
        previous, ordered_nodes, active_adjacency, down_nodes
    )
    holddowns = _empty_holddowns(ordered_nodes)
    if hold_down <= 0:
        return vectors, holddowns
    for router in ordered_nodes:
        if router in down_nodes:
            continue
        timed = holddowns[router]
        for destination in ordered_nodes:
            if destination == router or destination in active_adjacency[router]:
                continue
            prior = previous[router][destination]
            if (
                prior["nextHop"] is not None
                and vectors[router][destination]["metric"] is None
            ):
                timed[destination] = hold_down
    return vectors, holddowns


def _selected_hop_usable(
    router, destination, hop, previous, adjacency, infinity_metric
):
    """Whether the route currently selected via ``hop`` stays finite.

    The same usability test the shared scan applies to that hop's own
    previous-round advertisement: a present neighbor advertising
    unreachable, or a total metric at/above the infinity threshold, no longer
    carries the route.
    """
    if hop not in adjacency[router]:
        return False
    advertised = previous[hop][destination]["metric"]
    if advertised is None:
        return False
    return adjacency[router][hop] + advertised < infinity_metric


# ---------------------------------------------------------------------------
# The synchronous rounds themselves: one shared scan, mode-specific rows
# ---------------------------------------------------------------------------


def distance_vector_round(previous, ordered_nodes, adjacency):
    """Synchronously update every router from the previous round's ads.

    The threshold-free ``converge`` scan: every router is online and every
    finite candidate is eligible.
    """
    return {
        router: _online_router_row(
            router, ordered_nodes, previous, adjacency, None
        )
        for router in ordered_nodes
    }


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
        else:
            current[router] = _online_router_row(
                router,
                ordered_nodes,
                previous,
                adjacency,
                infinity_metric,
            )
    return current


def distance_vector_holddown_round(
    previous,
    ordered_nodes,
    adjacency,
    down_nodes,
    infinity_metric,
    holddowns,
    hold_down,
):
    """One synchronous round with hold-down suppression.

    Returns ``(vectors, holddowns)`` for the *next* round.  For an online
    router and a destination that is not a current direct neighbor:

    * a live timer (positive carried count) forces ``null``/``null`` and
      refuses every learned finite advertisement this round; the next round
      receives one less, and the timer disappears at zero -- repeated
      unreachability never resets it;
    * a selected finite route whose hop has just become unusable starts a
      fresh window and is reported unreachable this round;
    * otherwise the shared synchronous best-neighbor scan runs, with the
      infinity threshold and smaller-next-hop tie rule unchanged.

    A destination that is currently directly reachable is always adopted at
    its direct metric and clears any timer at once.
    """
    current = {}
    next_holddowns = _empty_holddowns(ordered_nodes)
    for router in ordered_nodes:
        if router in down_nodes:
            current[router] = _unreachable_row(ordered_nodes)
            continue
        row = {}
        for destination in ordered_nodes:
            if destination == router:
                row[destination] = {"nextHop": None, "metric": 0}
                continue
            if destination in adjacency[router]:
                # Direct reachability overrides any live hold-down and clears
                # its timer.
                row[destination] = {
                    "nextHop": destination,
                    "metric": adjacency[router][destination],
                }
                continue

            remaining = holddowns[router].get(destination)
            selected_hop = previous[router][destination]["nextHop"]
            invalidated = (
                remaining is None
                and selected_hop is not None
                and not _selected_hop_usable(
                    router,
                    destination,
                    selected_hop,
                    previous,
                    adjacency,
                    infinity_metric,
                )
            )

            if invalidated:
                # The finite route in use failed this round; the window opens.
                row[destination] = {"nextHop": None, "metric": None}
                next_holddowns[router][destination] = hold_down
                continue

            if remaining is not None:
                # Inside the suppression window: no learned finite route may
                # restore the destination; count the consumed round down.
                row[destination] = {"nextHop": None, "metric": None}
                if remaining - 1 > 0:
                    next_holddowns[router][destination] = remaining - 1
                continue

            # No window: the ordinary shared best-route selection.
            row[destination] = _select_route(
                router,
                destination,
                previous,
                adjacency,
                infinity_metric,
            )
        current[router] = row
    return current, next_holddowns


# ---------------------------------------------------------------------------
# Convergence loops: record round 0 and only the rounds that still change
# ---------------------------------------------------------------------------


def _converge(round_zero, carried_zero, step, is_stable, publish):
    """Drive synchronous rounds from round 0 to a fixed point.

    Every distance-vector mode records round 0 as given, then repeatedly asks
    ``step(vectors, carried)`` for ``(next_vectors, next_carried)``.  A round
    is the fixed point when ``is_stable(next_vectors, vectors,
    next_carried, carried)`` holds; otherwise it becomes the current state
    and ``publish(number, vectors, carried)`` builds the recorded entry.
    Only genuinely changing rounds are ever published.  Returns
    ``(rounds, final_vectors)``.
    """
    rounds = [publish(0, round_zero, carried_zero)]
    vectors = round_zero
    carried = carried_zero
    while True:
        updated, next_carried = step(vectors, carried)
        if is_stable(updated, vectors, next_carried, carried):
            break
        vectors = updated
        carried = next_carried
        rounds.append(publish(len(rounds), vectors, carried))
    return rounds, vectors


def _plain_round_entry(number, vectors, _carried):
    """The published shape of a hold-down-free round."""
    return {"round": number, "routers": vectors}


def _vectors_stable(updated, previous, _next_carried, _carried):
    """Fixed point for the modes whose state is just the forwarding vectors."""
    return updated == previous


def distance_vector_convergence(ordered_nodes, adjacency):
    """Return every changed synchronous round from round 0 to a fixed point."""
    round_zero = initial_distance_vectors(ordered_nodes, adjacency)

    def step(vectors, _carried):
        return distance_vector_round(vectors, ordered_nodes, adjacency), None

    rounds, _vectors = _converge(
        round_zero,
        None,
        step,
        _vectors_stable,
        _plain_round_entry,
    )
    return rounds


def distance_vector_failure_convergence(
    round_zero, ordered_nodes, active_adjacency, down_nodes, infinity_metric
):
    """Record round 0 and every later changed round up to a fixed point.

    After a cost increase a stale route's metric rises by at least the link
    cost along its dependency cycle until it reaches the threshold and is
    flushed as unreachable (the count-to-infinity process), so the scan
    always terminates; a surviving finite alternative wins earlier.
    """

    def step(vectors, _carried):
        return (
            distance_vector_round_infinity(
                vectors,
                ordered_nodes,
                active_adjacency,
                down_nodes,
                infinity_metric,
            ),
            None,
        )

    rounds, _vectors = _converge(
        round_zero,
        None,
        step,
        _vectors_stable,
        _plain_round_entry,
    )
    return rounds


def distance_vector_holddown_convergence(
    round_zero,
    holddowns_zero,
    ordered_nodes,
    active_adjacency,
    down_nodes,
    infinity_metric,
    hold_down,
):
    """Record rounds under hold-down until vectors and timers are both stable.

    Round 0 is recorded as given; each later round is one synchronous
    hold-down update.  A round is recorded while either the forwarding
    vectors or the published hold-down map still change, so
    ``convergenceRound`` points at the last round that is not yet a fixed
    point under both.  Returns ``(rounds, final_vectors)``.
    """

    def step(vectors, holddowns):
        return distance_vector_holddown_round(
            vectors,
            ordered_nodes,
            active_adjacency,
            down_nodes,
            infinity_metric,
            holddowns,
            hold_down,
        )

    def is_stable(updated, previous, next_holddowns, holddowns):
        return (
            updated == previous
            and holddowns_public(next_holddowns, ordered_nodes)
            == holddowns_public(holddowns, ordered_nodes)
        )

    def publish(number, vectors, holddowns):
        return {
            "round": number,
            "routers": vectors,
            "holdDowns": holddowns_public(holddowns, ordered_nodes),
        }

    rounds, vectors = _converge(
        round_zero,
        holddowns_zero,
        step,
        is_stable,
        publish,
    )
    return rounds, vectors
