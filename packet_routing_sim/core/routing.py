"""Routing computations over an explicit topology/availability state.

Two protocols live here:

* link-state shortest-path forwarding tables (Dijkstra) used by ``compute``
  and every replay snapshot;
* synchronous distance-vector rounds used by ``converge`` and by the plain
  and hold-down modes of ``replay-dv``.

Both are pure functions of the supplied adjacency map and ordered node
tuple.  Neighbors are always iterated in sorted name order and equal-metric
choices use a strict comparison, so ties deterministically select the
smaller next-hop name regardless of dict insertion order or hash seed.

Every distance-vector path shares one selection kernel
(:func:`_best_neighbor_route`) and one pair of round recorders
(:func:`_record_vector_rounds` / :func:`_record_holddown_rounds`), so the
candidate filtering, the ``infinityMetric`` cutoff, the smaller-next-hop
tie break and "only rounds that actually change are recorded" are defined
exactly once.  The modes keep only what is genuinely their own:

* ``converge`` seeds round 0 from :func:`initial_distance_vectors` (self
  and direct neighbors only) and runs without an infinity threshold;
* plain ``replay-dv`` inherits the previous converged vectors through
  :func:`distance_vector_failure_round_zero` and clamps candidates at
  ``infinityMetric``;
* hold-down ``replay-dv`` adds its transitive round-zero invalidation, the
  per-destination suppression countdown and immediate direct-recovery
  clearing around the very same selection kernel.

Optional receiver-specific poison reverse (:func:`_advertised_metric`)
sits *outside* the published local tables: the stored next hop and metric
are unchanged, only the advertisement handed to a given neighbor is
rewritten.
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
# Shared distance-vector per-round selection semantics
# ---------------------------------------------------------------------------
#
# One function decides the best route for one online router/destination from
# the previous round's neighbor advertisements.  Every mode -- the threshold
# free ``converge`` rounds, the infinity-clamped replay-dv rounds and the
# hold-down rounds -- reaches its routing decision through this kernel, so
# the following rules exist in exactly one place:
#
# * only a router's currently available direct neighbors are candidates;
# * a neighbor advertising ``null`` metric offers no route;
# * a candidate whose total metric is at or above ``infinityMetric`` is
#   unreachable (an omitted threshold never filters);
# * the smallest total metric wins, and because neighbors are scanned in
#   sorted name order with a strict comparison, an equal metric keeps the
#   smaller next-hop name;
# * no usable candidate yields ``null``/``null``.
#
# With poison reverse enabled each neighbor is shown a receiver-specific
# advertisement (:func:`_advertised_metric`) instead of the sender's stored
# metric: a destination the sender currently reaches through that very
# neighbor is advertised to it as unreachable.  The kernel itself is
# otherwise unchanged, so the cutoff and tie rules above apply to those
# advertisements too.
#
# A down router never enters this kernel: its whole row is built as
# unreachable by the caller.


def _unreachable_entry():
    """A fresh ``null``/``null`` routing entry (never shared between rows)."""
    return {"nextHop": None, "metric": None}


def _unreachable_row(ordered_nodes):
    """A fresh row advertising every destination (including self) as down."""
    return {destination: _unreachable_entry() for destination in ordered_nodes}


def _advertised_metric(sender, neighbor, destination, previous, poison_reverse):
    """The metric ``neighbor`` is shown for ``destination`` this exchange.

    Without poison reverse every neighbor reads the sender's stored metric.
    With it, a destination whose stored next hop is that neighbor is
    advertised only to that neighbor as unreachable; every other neighbor
    still receives the stored metric.  The stored table is never modified.
    """
    entry = previous[sender][destination]
    if poison_reverse and entry["nextHop"] == neighbor:
        return None
    return entry["metric"]


def _best_neighbor_route(
    router,
    destination,
    previous,
    adjacency,
    infinity_metric,
    poison_reverse=False,
):
    """Select ``(next_hop, metric)`` for one online router/destination.

    Scans every currently available neighbor in sorted name order and reads
    only that neighbor's previous-round advertisement -- a receiver-specific
    value when poison reverse is enabled, see :func:`_advertised_metric` --
    so the update is synchronous and ties deterministically keep the smaller
    neighbor name.  ``infinity_metric`` of ``None`` disables the threshold;
    otherwise a candidate at or above it is treated as unreachable.  Returns
    ``(None, None)`` when no finite, below-infinity candidate exists.
    """
    best_metric = None
    best_hop = None
    for neighbor in sorted(adjacency[router]):
        advertised = _advertised_metric(
            neighbor, router, destination, previous, poison_reverse
        )
        if advertised is None:
            continue
        candidate = adjacency[router][neighbor] + advertised
        if infinity_metric is not None and candidate >= infinity_metric:
            continue
        # Neighbors are iterated in name order, so the first best candidate
        # keeps the smaller next-hop name on ties.
        if best_metric is None or candidate < best_metric:
            best_metric = candidate
            best_hop = neighbor
    if best_metric is None:
        return None, None
    return best_hop, best_metric


def _distance_vector_round(
    previous,
    ordered_nodes,
    adjacency,
    down_nodes,
    infinity_metric,
    poison_reverse=False,
):
    """Synchronously update every router via the shared selection kernel.

    Each online router recomputes every destination from its neighbors'
    previous-round advertisements (receiver-specific under poison reverse)
    with the infinity cutoff and tie rule of :func:`_best_neighbor_route`.
    A down router keeps an entirely unreachable row.
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
            hop, metric = _best_neighbor_route(
                router,
                destination,
                previous,
                adjacency,
                infinity_metric,
                poison_reverse,
            )
            table[destination] = {"nextHop": hop, "metric": metric}
        current[router] = table
    return current


def distance_vector_round(previous, ordered_nodes, adjacency):
    """Synchronously update every router from the previous round's ads.

    The static ``converge`` path: no routers are down and no infinity
    threshold applies.
    """
    return _distance_vector_round(
        previous, ordered_nodes, adjacency, frozenset(), None
    )


def distance_vector_round_infinity(
    previous,
    ordered_nodes,
    adjacency,
    down_nodes,
    infinity_metric,
    poison_reverse=False,
):
    """Synchronously update from prior ads with an infinity threshold.

    Same scan/tie rules as :func:`distance_vector_round`.  Without poison
    reverse every available neighbor's previous-round advertisement is
    read; with it each neighbor reads only its own receiver-specific
    advertisement from :func:`_advertised_metric`, so a route learned
    through a neighbor is advertised back to that neighbor as unreachable.
    A candidate metric at or above ``infinity_metric`` counts as
    unreachable.  Down routers keep an entirely unreachable row.
    """
    return _distance_vector_round(
        previous,
        ordered_nodes,
        adjacency,
        down_nodes,
        infinity_metric,
        poison_reverse,
    )


def _record_vector_rounds(round_zero, advance):
    """Record round 0 and every later changed round up to a fixed point.

    ``advance(vectors)`` is one synchronous update built on the shared
    selection kernel.  Round 0 is always recorded; a later round is appended
    only while the forwarding vectors actually differ from the previous
    round, so the scan stops exactly at the first fixed point.  Every round
    dictionary and every vector table is built fresh by ``advance``.
    """
    rounds = [{"round": 0, "routers": round_zero}]
    vectors = round_zero
    while True:
        updated = advance(vectors)
        if updated == vectors:
            break
        vectors = updated
        rounds.append({"round": len(rounds), "routers": vectors})
    return rounds


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
                table[destination] = _unreachable_entry()
        vectors[router] = table
    return vectors


def distance_vector_convergence(ordered_nodes, adjacency):
    """Return every changed synchronous round from round 0 to a fixed point."""
    return _record_vector_rounds(
        initial_distance_vectors(ordered_nodes, adjacency),
        lambda vectors: distance_vector_round(
            vectors, ordered_nodes, adjacency
        ),
    )


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
                    table[destination] = _unreachable_entry()
                else:
                    # A copy, so retained advertisements never alias caller
                    # data or later rounds.
                    table[destination] = dict(entry)
        vectors[router] = table
    return vectors


def distance_vector_failure_convergence(
    round_zero,
    ordered_nodes,
    active_adjacency,
    down_nodes,
    infinity_metric,
    poison_reverse=False,
):
    """Record round 0 and every later changed round up to a fixed point.

    After a cost increase a stale route's metric rises by at least the link
    cost along its dependency cycle until it reaches the threshold and is
    flushed as unreachable (the count-to-infinity process), so the scan
    always terminates; a surviving finite alternative wins earlier.  With
    poison reverse enabled the receiver-specific advertisements break the
    shortest-path cycles immediately, typically stopping that process at
    the first exchange after the event's round zero.
    """
    return _record_vector_rounds(
        round_zero,
        lambda vectors: distance_vector_round_infinity(
            vectors,
            ordered_nodes,
            active_adjacency,
            down_nodes,
            infinity_metric,
            poison_reverse,
        ),
    )


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
# Down routers hold no timers; their whole row stays ``null``/``null``.  When
# no timer is live and no route is freshly invalidated, a hold-down round is
# exactly the shared :func:`_best_neighbor_route` selection.
#
# Poison reverse is orthogonal to all of this: round zero keeps the same
# immediate invalidation above and the receiver-specific advertisements only
# take effect from the following synchronous exchange onward, shaping both
# the hop-usability check and the ordinary best-neighbor scan in exactly the
# same way as the plain rounds.


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
                        row[destination] = _unreachable_entry()
                    elif current[hop][destination]["metric"] is None:
                        # The still-present selected hop no longer offers a
                        # way to the destination: null it, possibly enabling
                        # further nulls on the next closure pass.
                        row[destination] = _unreachable_entry()
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
    router,
    destination,
    hop,
    previous,
    adjacency,
    infinity_metric,
    poison_reverse=False,
):
    """Whether the route currently selected via ``hop`` stays finite.

    The same usability test the shared selection kernel applies to that
    hop's own previous-round advertisement (receiver-specific under poison
    reverse): a present neighbor advertising unreachable, or a total metric
    at/above the infinity threshold, no longer carries the route.
    """
    if hop not in adjacency[router]:
        return False
    advertised = _advertised_metric(
        hop, router, destination, previous, poison_reverse
    )
    if advertised is None:
        return False
    return adjacency[router][hop] + advertised < infinity_metric


def distance_vector_holddown_round(
    previous,
    ordered_nodes,
    adjacency,
    down_nodes,
    infinity_metric,
    holddowns,
    hold_down,
    poison_reverse=False,
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
    * otherwise the ordinary synchronous best-neighbor scan runs through
      the shared selection kernel, with the infinity threshold and
      smaller-next-hop tie rule unchanged.

    Poison reverse, when enabled, shapes only the previous round's
    receiver-specific advertisements read by the two checks above; the
    timer, countdown and direct-neighbor semantics are unchanged.  A
    destination that is currently directly reachable is always adopted at
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
                    poison_reverse,
                )
            )

            if invalidated:
                # The finite route in use failed this round; the window opens.
                row[destination] = _unreachable_entry()
                next_holddowns[router][destination] = hold_down
                continue

            if remaining is not None:
                # Inside the suppression window: no learned finite route may
                # restore the destination; count the consumed round down.
                row[destination] = _unreachable_entry()
                if remaining - 1 > 0:
                    next_holddowns[router][destination] = remaining - 1
                continue

            # No window: the shared synchronous best-route selection.
            hop, metric = _best_neighbor_route(
                router,
                destination,
                previous,
                adjacency,
                infinity_metric,
                poison_reverse,
            )
            row[destination] = {"nextHop": hop, "metric": metric}
        current[router] = row
    return current, next_holddowns


def _record_holddown_rounds(
    round_zero, holddowns_zero, ordered_nodes, advance
):
    """Record hold-down rounds until both vectors and timers are stable.

    Round 0 is recorded as given; each later round is one synchronous
    hold-down update from ``advance(vectors, holddowns)``.  A round is
    recorded while either the forwarding vectors or the published hold-down
    map still change, so convergence points at the last round that is not yet
    a fixed point under both.  Returns ``(rounds, final_vectors)``.
    """
    rounds = [
        {
            "round": 0,
            "routers": round_zero,
            "holdDowns": holddowns_public(holddowns_zero, ordered_nodes),
        }
    ]
    vectors = round_zero
    holddowns = holddowns_zero
    while True:
        updated, next_holddowns = advance(vectors, holddowns)
        next_public = holddowns_public(next_holddowns, ordered_nodes)
        if updated == vectors and next_public == rounds[-1]["holdDowns"]:
            break
        vectors = updated
        holddowns = next_holddowns
        rounds.append(
            {
                "round": len(rounds),
                "routers": vectors,
                "holdDowns": next_public,
            }
        )
    return rounds, vectors


def distance_vector_holddown_convergence(
    round_zero,
    holddowns_zero,
    ordered_nodes,
    active_adjacency,
    down_nodes,
    infinity_metric,
    hold_down,
    poison_reverse=False,
):
    """Record rounds under hold-down until vectors and timers are both stable.

    Round 0 is recorded as given; each later round is one synchronous
    hold-down update.  A round is recorded while either the forwarding
    vectors or the published hold-down map still change, so
    ``convergenceRound`` points at the last round that is not yet a fixed
    point under both.  Poison reverse, when enabled, supplies the
    receiver-specific prior advertisements to every update.  Returns
    ``(rounds, final_vectors)``.
    """
    return _record_holddown_rounds(
        round_zero,
        holddowns_zero,
        ordered_nodes,
        lambda vectors, timers: distance_vector_holddown_round(
            vectors,
            ordered_nodes,
            active_adjacency,
            down_nodes,
            infinity_metric,
            timers,
            hold_down,
            poison_reverse,
        ),
    )
