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

`replay-dv` runs the distance-vector protocol through the same topology,
events and state-transition rules. The scenario additionally carries an
`infinityMetric`, which must be a non-boolean positive integer strictly
larger than every link metric; otherwise the command reports
`invalid scenario` and exits with status 2 (topology errors still take
precedence). Output is
`{"protocol": "distance-vector", "infinityMetric": ..., "timeline": [...]}`.
The baseline item has `event: null` and converges from a round 0 in which
each router knows only itself and its usable direct neighbors. Each later
item echoes `time` and `event`; its round 0 retains online routers' prior
advertisements but immediately marks routes via failed direct neighbors
unreachable, a recovered router cold-starts knowing only itself and its
usable direct neighbors, and both ends immediately regain direct routes
when a link recovers. Subsequent rounds read only the previous round's
available-neighbor advertisements (no split horizon or poison reverse); a
candidate metric at or above `infinityMetric` becomes `null` next hop and
metric, and tied finite metrics pick the smaller next-hop name. Each item
records round 0 plus every later changed round, with `convergenceRound`
pointing at the stable round, so increasing metrics, alternating
advertisements and final unreachability are auditable round by round.
