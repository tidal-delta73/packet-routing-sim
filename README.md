# packet-routing-sim

Network routing protocol simulator.

Pure-Python, no runtime dependencies.

## Usage

```bash
python3 -m packet_routing_sim version
python3 -m packet_routing_sim compute TOPOLOGY.json
python3 -m packet_routing_sim converge TOPOLOGY.json
python3 -m packet_routing_sim replay TOPOLOGY.json SCENARIO.json
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

`replay` replays a failure scenario over the same topology. The scenario is a
JSON object with an `events` array; each event has a strictly increasing
positive integer `time` and an `action` of `node-down`, `node-up`,
`link-down`, or `link-up`. Node events name an existing `node`; link events
name an existing link by `from` and `to`. Output is a link-state document
whose `timeline` starts with the failure-free baseline at time 0
(`event: null`) and then records the recomputed forwarding tables after every
event. Down routers keep a full row of `null` entries (self included);
active routers report down or unreachable destinations as `null` next hop and
metric. Link-down state is tracked independently of node state, so a node-up
re-enables only original links that are not link-disabled and whose endpoints
are both active. Malformed scenarios, unreadable scenario files, and argument
errors exit with status 2 and write nothing to standard output.
