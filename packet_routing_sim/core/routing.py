"""Routing computations over an explicit topology/availability state.

Three protocol views live here:

* link-state shortest-path forwarding tables (Dijkstra) used by ``compute``
  and every replay snapshot;
* synchronous distance-vector rounds used by ``converge`` and ``replay-dv``;
* synchronous link-state neighbor discovery, LSA flooding and per-LSDB SPF
  rounds used by ``replay-ls``.

All are pure functions of the supplied adjacency map and ordered node tuple.
Neighbors are always iterated in sorted name order and equal-metric choices
use a strict comparison, so ties deterministically select the smaller
next-hop name regardless of dict insertion order or hash seed.
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
# Link-state neighbor discovery, LSA flooding and per-LSDB SPF (replay-ls)
# ---------------------------------------------------------------------------
#
# An LSA is the immutable tuple ``(origin, sequence, neighbors)`` where
# ``neighbors`` is a tuple of ``(name, metric)`` pairs sorted by neighbor
# name.  A link-state database maps an origin name to the highest-sequence
# LSA known from that origin.  LSAs never carry their origin twice: the
# database key is the origin and the published view unfolds it back into an
# object.  Only the tuples below cross between rounds; every published round
# (databases and routers) is freshly built plain data.


def _local_lsa(router, sequence, active_adjacency):
    """The router's own LSA: its currently usable neighbors, name-sorted."""
    neighbors = tuple(
        (neighbor, active_adjacency[router][neighbor])
        for neighbor in sorted(active_adjacency[router])
    )
    return (router, sequence, neighbors)


def _lsa_view(lsa):
    """Publish one immutable LSA as fresh ``{seq, neighbors}`` JSON data."""
    _origin, sequence, neighbors = lsa
    return {
        "seq": sequence,
        "neighbors": [
            {"router": neighbor, "metric": metric} for neighbor, metric in neighbors
        ],
    }


def link_state_flood_round(previous, ordered_nodes, active_adjacency, down_nodes):
    """One synchronous flooding round over the current usable links.

    Every online router starts with a copy of its start-of-round database,
    then reads each currently available neighbor's start-of-round database
    and accepts an LSA only when it holds no copy from that origin or the
    advertisement carries a strictly higher sequence.  Down routers neither
    send nor receive, so their database row is left untouched.  Acceptance
    is order-independent: for a given origin the unique winner is the
    greatest sequence present at the start of the round.
    """
    current = {router: dict(previous[router]) for router in ordered_nodes}
    for receiver in ordered_nodes:
        if receiver in down_nodes:
            continue
        database = current[receiver]
        for neighbor in sorted(active_adjacency[receiver]):
            for origin in sorted(previous[neighbor]):
                advertised = previous[neighbor][origin]
                known = database.get(origin)
                if known is None or advertised[1] > known[1]:
                    database[origin] = advertised
    return current


def _flood_to_fixed(round_zero, ordered_nodes, active_adjacency, down_nodes):
    """Flood from a round-0 database map until no database changes."""
    databases = round_zero
    sequence = [databases]
    while True:
        updated = link_state_flood_round(
            databases, ordered_nodes, active_adjacency, down_nodes
        )
        if updated == databases:
            return sequence, databases
        databases = updated
        sequence.append(databases)


def link_state_graph(database):
    """Build the undirected graph one router sees in its own current LSDB.

    A link participates only when the latest LSAs known for *both* endpoints
    declare each other and agree on the metric; a one-way, absent or
    metric-disagreeing declaration is ignored.  The result is a fresh
    adjacency map keyed by the origins present in this database, which lets
    the ordinary Dijkstra routine run over exactly the links this router
    currently believes in.
    """
    declared = {
        origin: dict(neighbors)
        for origin, _sequence, neighbors in database.values()
    }
    adjacency = {origin: {} for origin in sorted(declared)}
    for source in sorted(declared):
        for target, metric in sorted(declared[source].items()):
            if target not in declared:
                continue
            remote_metric = declared[target].get(source)
            if remote_metric is None or remote_metric != metric:
                continue
            adjacency[source][target] = metric
            adjacency[target][source] = metric
    return adjacency


def link_state_round_view(databases, ordered_nodes, down_nodes):
    """Publish one round: every database and every router's own SPF result.

    Each online router independently runs SPF over the graph derived from
    its own current database, so a router that has not yet received a newer
    LSA keeps routing on its stale view.  A down router's whole ``routers``
    row is ``None``; its frozen database is still shown for auditability.
    """
    databases_view = {}
    routers_view = {}
    for router in ordered_nodes:
        database = databases[router]
        databases_view[router] = {
            origin: _lsa_view(database[origin]) for origin in sorted(database)
        }
        if router in down_nodes:
            routers_view[router] = None
        else:
            routers_view[router] = forwarding_table(
                router, ordered_nodes, link_state_graph(database)
            )
    return {"databases": databases_view, "routers": routers_view}


def _round_views(database_sequence, ordered_nodes, down_nodes):
    """Number a flooded database sequence from round 0 and publish it."""
    return [
        {"round": number,
         **link_state_round_view(databases, ordered_nodes, down_nodes)}
        for number, databases in enumerate(database_sequence)
    ]


def link_state_baseline_rounds(ordered_nodes, active_adjacency):
    """Neighbor discovery and flooding from the fault-free state.

    Round 0: every online (here: every) router discovers its usable direct
    neighbors, originates a sequence-1 LSA and installs only its own.  Later
    rounds flood the known latest LSAs over usable links until every
    database is stable.  Return ``(rounds, databases, sequence_numbers)``
    where the latter two carry the converged internal state into the first
    post-event phase.
    """
    databases = {}
    sequence_numbers = {}
    for router in ordered_nodes:
        databases[router] = {router: _local_lsa(router, 1, active_adjacency)}
        sequence_numbers[router] = 1

    database_sequence, databases = _flood_to_fixed(
        databases, ordered_nodes, active_adjacency, frozenset()
    )
    rounds = _round_views(database_sequence, ordered_nodes, frozenset())
    return rounds, databases, sequence_numbers


def link_state_event_rounds(
    previous_databases,
    sequence_numbers,
    ordered_nodes,
    previous_adjacency,
    active_adjacency,
    down_nodes,
    recovered_nodes,
):
    """Round 0 after one event, then flooding to a new fixed point.

    Every database starts as a copy of the previous phase's final one:

    * a recovered router restarts with an empty database and originates its
      next LSA using the sequence following its historical number;
    * any other online endpoint whose direct neighbor set changed
      originates the next sequence locally and installs it at round 0;
    * unchanged online routers keep their whole view, including LSAs of
      origins that have not (yet) advertised a replacement;
    * a down router keeps its frozen database and does not participate.
    """
    recovered = set(recovered_nodes)
    databases = {router: dict(previous_databases[router]) for router in ordered_nodes}
    numbers = dict(sequence_numbers)

    def originate(router):
        numbers[router] = numbers.get(router, 0) + 1
        databases[router][router] = _local_lsa(
            router, numbers[router], active_adjacency
        )

    for router in recovered:
        databases[router] = {}
        originate(router)
    for router in ordered_nodes:
        if router in down_nodes or router in recovered:
            continue
        if set(previous_adjacency[router]) != set(active_adjacency[router]):
            originate(router)

    database_sequence, databases = _flood_to_fixed(
        databases, ordered_nodes, active_adjacency, down_nodes
    )
    rounds = _round_views(database_sequence, ordered_nodes, down_nodes)
    return rounds, databases, numbers
