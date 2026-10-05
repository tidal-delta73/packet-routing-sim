# packet-routing-sim

Network routing protocol simulator.

Pure-Python, no runtime dependencies.

## Usage

```bash
python3 -m packet_routing_sim version
python3 -m packet_routing_sim compute TOPOLOGY.json
python3 -m packet_routing_sim converge TOPOLOGY.json
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
