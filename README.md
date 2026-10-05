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

`converge` reads the same topology and shows how a distance-vector protocol
converges round by round: round 0 contains only each router's own entry and
its direct links, and every later round updates all routers synchronously
from the previous round's neighbor advertisements until the tables stop
changing. The output reports `protocol`, `convergenceRound` and the retained
`rounds` snapshots.
