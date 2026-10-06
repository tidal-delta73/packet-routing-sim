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

## Tests

```bash
python3 -m unittest discover -s tests
```

The suite exercises only the public CLI: differential checks that `converge`'s
final snapshot matches `compute` exactly, hand-computed tables for fixed
topologies, convergence-trajectory properties (round 0 contents, synchronous
derivation, first stable round), byte-identical output for equivalent inputs
and across `PYTHONHASHSEED` values, and the documented error behavior. It
uses only the standard library.
