# packet-routing-sim

Network routing protocol simulator.

Pure-Python, no runtime dependencies.

## Usage

```bash
python3 -m packet_routing_sim version
python3 -m packet_routing_sim compute TOPOLOGY.json
python3 -m packet_routing_sim converge TOPOLOGY.json
python3 -m packet_routing_sim replay TOPOLOGY.json SCENARIO.json
python3 -m packet_routing_sim replay-dv TOPOLOGY.json SCENARIO.json
python3 -m packet_routing_sim replay-ls TOPOLOGY.json SCENARIO.json
python3 -m packet_routing_sim help
```

`compute` reads a static undirected link-state snapshot (a JSON object with
`nodes` and `links`) and writes per-router shortest-path forwarding tables
(Dijkstra; equal-metric ties break to the smaller next-hop name) as JSON to
standard output.

`converge` reads the same topology and shows distance-vector convergence:
round 0 holds each router's view of itself and its direct neighbors only, and
each subsequent round updates every router synchronously from its neighbors'
previous-round advertisements. Output records the initial snapshot and every
changed snapshot through `convergenceRound`, with unreachable destinations
reported as `null` next hop and metric.

`replay` reads a topology plus a scenario (a JSON object with an `events`
array) and replays link/node failures and recoveries in time order. Each
event carries a strictly increasing positive integer `time` and an `action`
(`node-down`, `node-up`, `link-down`, `link-up`); node events name a `node`,
link events name `from`/`to` endpoints of a declared link. `node-down` also
disables the node's incident links, while `link-down` state is tracked
independently, so a recovered node re-enables only links that were not
explicitly disabled and whose other endpoint is up. Output is
`{"protocol": "link-state", "timeline": [...]}`: the first timeline entry is
the fault-free baseline (`event: null`), and each later entry holds the
event's `time`, the `event` itself, and the recomputed link-state forwarding
tables. All declared nodes always appear as routers and destinations; a down
router's whole row is `null`/`null`, as is any entry for a down or
unreachable destination.

`replay-dv` reads the same topology and scenario format but replays the
failures with synchronous distance-vector updates, showing the
count-to-infinity process round by round. The scenario must additionally
carry an `infinityMetric`: a non-boolean positive integer strictly greater
than every declared link metric (otherwise the command reports
`invalid scenario` and exits with status 2). Output is
`{"protocol": "distance-vector", "infinityMetric": ..., "timeline": [...]}`.
As with `replay`, the first timeline entry is the fault-free baseline with
`event: null`; each later entry echoes the event's `time` and `event`
verbatim. Every timeline entry contains a `rounds` list (round 0 plus each
later round in which some route changes) and a `convergenceRound` pointer to
the stable round.

Round 0 after an event keeps each online router's previously advertised
routes, except routes through a direct neighbor that just failed (or over a
link that just went down) immediately become unreachable; a down router's
whole row is unreachable; a recovering router starts knowing only itself and
its currently available direct neighbors; and both endpoints of a recovered
link regain the direct route at once. Subsequent rounds read only the
previous round's advertisements from currently available neighbors, with no
split horizon or poison reverse. A candidate metric at or above
`infinityMetric` is reported as `null`/`null`, and ties on finite metric
break to the smaller next-hop name. The recorded rounds make the rising
metrics, alternating advertisements and final unreachability fully
auditable.

`replay-ls` reads the same topology and scenario format but replays the
link-state protocol itself, exposing neighbor discovery, LSA flooding and
SPF round by round. Output is
`{"protocol": "link-state", "timeline": [...]}`; as with `replay`, the first
timeline entry is the fault-free baseline (`event: null`) and each later
entry echoes the event's `time` and `event` verbatim. Every timeline entry
contains a `rounds` list and a `convergenceRound` pointer to the stable
round. Each round holds `round` (starting at 0), `databases` and `routers`.

In the baseline round 0 every online router discovers its available direct
neighbors, originates a local LSA with sequence number 1 whose neighbors are
sorted by name, and installs only that LSA into its own link-state database.
In each following round every online router floods the newest LSAs it knows
over each currently available link, and a receiver accepts an LSA from a
given originator only when its sequence number is strictly higher than the
one already held; rounds are recorded until no database changes. After an
event the previous entry's converged databases are inherited; the online
endpoints whose adjacency changed increment their local sequence and install
their new LSA in round 0, while all other routers keep their old view. A
recovering node continues numbering from its own historical sequence; a down
node originates, sends or receives nothing and keeps no database.

Each router computes its forwarding table independently from its own
current-round database. A link participates in SPF only while both
endpoints' newest held LSAs still declare each other (the single topology
metric being the agreed metric); consequently a remote router that has not
yet received a newer LSA can briefly keep a stale route, while a down
router's whole row is `null`. Equal-cost paths still break to the smaller
next-hop name.
