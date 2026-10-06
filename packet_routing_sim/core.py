"""Pure simulation core for the packet routing simulator.

This module owns topology validation, protocol computation and failure-state
advancement. It never reads or writes files, never touches standard streams,
command line arguments or the process environment, and never relies on dict
insertion order: every published mapping is built by iterating names in sorted
order. Callers hand in decoded JSON values (plain ``dict``/``list``/``str``/
``int``) and receive plain Python data back.

Validation failures are reported with distinguishable, deterministic results:

* :class:`InvalidTopology` -- a topology document is structurally or
  semantically illegal;
* :class:`InvalidScenario` -- a scenario document has a bad shape or refers to
  nodes/links the topology does not declare;
* :class:`IllegalTransition` -- a scenario event is well formed but cannot be
  applied from the current network state (repeated disable, restoring an
  enabled object). :class:`IllegalTransition` is an :class:`InvalidScenario`,
  so command layers may report both as "invalid scenario".

The public entries are :func:`compute`, :func:`converge` and
:func:`simulate_replay`. They hold no module-level mutable state: repeated
calls are independent, inputs are not modified, and mutating a returned value
cannot affect any later call.
"""
import copy
import heapq

__all__ = [
    "InvalidTopology",
    "InvalidScenario",
    "IllegalTransition",
    "Topology",
    "NetworkState",
    "validate_topology",
    "validate_scenario",
    "apply_event",
    "forwarding_table",
    "routers_snapshot",
    "initial_distance_vectors",
    "distance_vector_round",
    "compute",
    "converge",
    "simulate_replay",
]


class InvalidTopology(ValueError):
    """The topology document fails structural or semantic validation."""


class InvalidScenario(ValueError):
    """The scenario document fails structural or reference validation."""


class IllegalTransition(InvalidScenario):
    """An event contradicts the explicit node/link availability state."""


def _canonical_pair(source, target):
    """Canonical undirected key for a link endpoint pair."""
    return (source, target) if source < target else (target, source)


class Topology:
    """A validated undirected topology.

    ``nodes`` is the list of declared node names in declaration order (kept
    only so equality can mirror the input); ``ordered`` holds the same names
    stably sorted; ``adjacency`` maps every node to ``{neighbor: metric}``
    dicts; ``link_pairs`` is the set of canonical endpoint pairs.

    Instances are immutable from the caller's point of view: the constructor
    copies its arguments and the stored containers are never exposed in a way
    that later protocol code mutates.
    """

    __slots__ = ("nodes", "ordered", "adjacency", "link_pairs")

    def __init__(self, nodes, adjacency, link_pairs):
        self.nodes = list(nodes)
        self.ordered = sorted(nodes)
        self.adjacency = {
            node: dict(neighbors) for node, neighbors in adjacency.items()
        }
        self.link_pairs = set(link_pairs)

    def __eq__(self, other):
        if not isinstance(other, Topology):
            return NotImplemented
        return (
            self.nodes == other.nodes
            and self.adjacency == other.adjacency
            and self.link_pairs == other.link_pairs
        )


def validate_topology(topology):
    """Validate a decoded topology document.

    Returns a :class:`Topology` or raises :class:`InvalidTopology`.
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

    adjacency = {node: {} for node in nodes}
    pairs = set()
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
            raise InvalidTopology("self links are not allowed")
        if isinstance(metric, bool) or not isinstance(metric, int) or metric <= 0:
            raise InvalidTopology("metric must be a positive integer")
        pair = _canonical_pair(source, target)
        if pair in pairs:
            raise InvalidTopology("duplicate link between the same two nodes")
        pairs.add(pair)
        adjacency[source][target] = metric
        adjacency[target][source] = metric

    return Topology(nodes, adjacency, pairs)


class NetworkState:
    """Explicit node and link availability for a validated topology.

    Availability is tracked independently: ``down_nodes`` names stopped
    routers and ``down_links`` names links explicitly disabled by link-down
    events (canonical endpoint pairs). A link-down recorded while an endpoint
    is stopped therefore stays in force after that node recovers.

    ``down_nodes`` and ``down_links`` are advanced in place by
    :func:`apply_event`; use :meth:`clone` to branch an independent copy.
    Snapshot generation only reads the state and the topology, never mutates
    either.
    """

    __slots__ = ("topology", "down_nodes", "down_links")

    def __init__(self, topology, down_nodes=(), down_links=()):
        self.topology = topology
        self.down_nodes = set(down_nodes)
        self.down_links = set(down_links)

    def clone(self):
        return NetworkState(
            self.topology, self.down_nodes, self.down_links
        )


def _event_pair(event):
    """Canonical endpoint pair for a link event, or None if malformed."""
    source, target = event.get("from"), event.get("to")
    if not isinstance(source, str) or not isinstance(target, str):
        return None
    return _canonical_pair(source, target)


def validate_scenario(scenario, topology):
    """Validate a decoded scenario document against a validated topology.

    Returns the list of event dicts in their original order, or raises
    :class:`InvalidScenario` for bad structure/references and
    :class:`IllegalTransition` for impossible state transitions.
    """
    if not isinstance(scenario, dict):
        raise InvalidScenario("scenario must be an object")
    events_raw = scenario.get("events")
    if not isinstance(events_raw, list):
        raise InvalidScenario("events must be an array")

    events = []
    down_nodes = set()
    down_links = set()
    last_time = 0
    for event in events_raw:
        if not isinstance(event, dict):
            raise InvalidScenario("event must be an object")
        time = event.get("time")
        action = event.get("action")
        if isinstance(time, bool) or not isinstance(time, int):
            raise InvalidScenario("event time must be an integer")
        if time <= 0 or time <= last_time:
            raise InvalidScenario("event times must be strictly increasing")
        if not isinstance(action, str):
            raise InvalidScenario("event action must be a string")
        if action in ("node-down", "node-up"):
            node = event.get("node")
            if not isinstance(node, str) or node not in topology.adjacency:
                raise InvalidScenario("event references an undeclared node")
            if action == "node-down":
                if node in down_nodes:
                    raise IllegalTransition("node is already down")
                down_nodes.add(node)
            else:
                if node not in down_nodes:
                    raise IllegalTransition("node is already up")
                down_nodes.discard(node)
        elif action in ("link-down", "link-up"):
            pair = _event_pair(event)
            if pair is None or pair not in topology.link_pairs:
                raise InvalidScenario("event references an undeclared link")
            if action == "link-down":
                if pair in down_links:
                    raise IllegalTransition("link is already down")
                down_links.add(pair)
            else:
                if pair not in down_links:
                    raise IllegalTransition("link is already up")
                down_links.discard(pair)
        else:
            raise InvalidScenario("unknown event action")
        events.append(event)
        last_time = time
    return events


def apply_event(state, event):
    """Advance ``state`` in place by one already-shaped event dict.

    Only the transition itself is applied here; routing tables are not
    recomputed. Raises :class:`InvalidScenario` for an event that cannot name
    a topology object and :class:`IllegalTransition` for an impossible
    transition. Structural shape is assumed to have been checked by
    :func:`validate_scenario`, but is re-checked defensively so direct callers
    cannot corrupt the state with a malformed dict.
    """
    action = event.get("action") if isinstance(event, dict) else None
    topology = state.topology
    if action in ("node-down", "node-up"):
        node = event.get("node")
        if not isinstance(node, str) or node not in topology.adjacency:
            raise InvalidScenario("event references an undeclared node")
        if action == "node-down":
            if node in state.down_nodes:
                raise IllegalTransition("node is already down")
            state.down_nodes.add(node)
        else:
            if node not in state.down_nodes:
                raise IllegalTransition("node is already up")
            state.down_nodes.discard(node)
    elif action in ("link-down", "link-up"):
        pair = _event_pair(event)
        if pair is None or pair not in topology.link_pairs:
            raise InvalidScenario("event references an undeclared link")
        if action == "link-down":
            if pair in state.down_links:
                raise IllegalTransition("link is already down")
            state.down_links.add(pair)
        else:
            if pair not in state.down_links:
                raise IllegalTransition("link is already up")
            state.down_links.discard(pair)
    else:
        raise InvalidScenario("unknown event action")


def forwarding_table(source, ordered_nodes, adjacency):
    """Dijkstra from ``source``; ties on total metric pick the smaller first hop.

    ``adjacency`` needs to cover only the nodes that may carry traffic (down
    nodes omitted); destinations absent from it are reported unreachable.
    """
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


def _active_adjacency(state):
    """Adjacency containing only links usable in the current state.

    A link is usable only while neither endpoint is down and no link-down
    (not yet matched by a link-up) disabled it.
    """
    topology = state.topology
    active = {
        node: {} for node in topology.ordered if node not in state.down_nodes
    }
    for source in active:
        for target, metric in topology.adjacency[source].items():
            if target not in active:
                continue
            if _canonical_pair(source, target) in state.down_links:
                continue
            active[source][target] = metric
    return active


def routers_snapshot(state):
    """Build the full link-state routing snapshot for an explicit state.

    Every declared node appears as a router and destination. A down router's
    whole row is ``null``/``null``; an unreachable destination (including a
    down destination) is reported the same way. The returned structure is a
    fresh value: callers may mutate it freely.
    """
    topology = state.topology
    ordered = topology.ordered
    active = _active_adjacency(state)
    routers = {}
    for source in ordered:
        if source in state.down_nodes:
            routers[source] = {
                destination: {"nextHop": None, "metric": None}
                for destination in ordered
            }
        else:
            routers[source] = forwarding_table(source, ordered, active)
    return routers


def compute(topology):
    """Static shortest-path forwarding tables for a decoded topology.

    Returns ``{"routers": {router: {destination: {"nextHop", "metric"}}}}``
    or raises :class:`InvalidTopology`.
    """
    validated = validate_topology(topology)
    ordered = validated.ordered
    routers = {
        source: forwarding_table(source, ordered, validated.adjacency)
        for source in ordered
    }
    return {"routers": routers}


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
    """Synchronously update every router from the previous round's advertisements."""
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


def converge(topology):
    """Distance-vector convergence rounds for a decoded topology.

    Returns ``{"protocol", "convergenceRound", "rounds"}`` where round 0 holds
    only self plus direct links and only genuinely changed rounds are
    recorded. Raises :class:`InvalidTopology`.
    """
    validated = validate_topology(topology)
    ordered = validated.ordered
    vectors = initial_distance_vectors(ordered, validated.adjacency)
    rounds = [{"round": 0, "routers": vectors}]
    while True:
        updated = distance_vector_round(vectors, ordered, validated.adjacency)
        if updated == vectors:
            break
        vectors = updated
        rounds.append({"round": len(rounds), "routers": vectors})

    return {
        "protocol": "distance-vector",
        "convergenceRound": rounds[-1]["round"],
        "rounds": rounds,
    }


def simulate_replay(topology, scenario):
    """Replay node/link failures and recoveries over a decoded topology.

    Returns ``{"protocol": "link-state", "timeline": [...]}``: the first
    timeline entry is the fault-free baseline with ``event: None`` and each
    later entry holds the event's ``time``, the original ``event`` dict and a
    routing snapshot freshly generated from the resulting
    :class:`NetworkState`. Raises :class:`InvalidTopology` or
    :class:`InvalidScenario` (including :class:`IllegalTransition`).
    """
    validated = validate_topology(topology)
    events = validate_scenario(scenario, validated)

    state = NetworkState(validated)
    timeline = [{"event": None, "routers": routers_snapshot(state)}]
    for event in events:
        # Validation above already accepted exactly this transition; apply it
        # through the same single state-advancement path a direct caller uses.
        apply_event(state, event)
        # Copy the event so a caller mutating a timeline entry cannot change
        # the input scenario or any other result.
        carried = copy.deepcopy(event)
        timeline.append(
            {
                "time": carried["time"],
                "event": carried,
                "routers": routers_snapshot(state),
            }
        )
    return {"protocol": "link-state", "timeline": timeline}
