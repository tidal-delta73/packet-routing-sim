# packet-routing-sim

Network routing protocol simulator.

Pure-Python, no runtime dependencies.

## Usage

```bash
python3 -m packet_routing_sim version
python3 -m packet_routing_sim compute TOPOLOGY.json
python3 -m packet_routing_sim help
```

`compute` reads a static undirected topology snapshot (nodes and weighted
links) and writes a deterministic shortest-path forwarding table (Dijkstra,
smallest next-hop name on equal-cost ties) as JSON to standard output.
