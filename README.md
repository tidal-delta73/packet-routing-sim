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
split horizon or poison reverse unless the optional `poisonReverse` mode is
enabled. A candidate metric at or above `infinityMetric` is reported as
`null`/`null`, and ties on finite metric break to the smaller next-hop name.
The recorded rounds make the rising metrics, alternating advertisements and
final unreachability fully auditable.

The scenario may additionally carry `holdDownRounds`: a non-boolean
non-negative integer selecting optional route hold-down after the baseline.
When it is omitted, input, output and behavior are unchanged; `0` also
disables suppression explicitly (and likewise produces the legacy output).
Any other type or a negative value reports `invalid scenario` with exit
status 2 (topology errors still take precedence). When enabled, the root
object echoes `holdDownRounds`, and every round of every timeline entry
gains a `holdDowns` object: for each declared router it maps a destination
to the positive number of full update rounds still left on that route's
hold-down timer, and is `{}` when no timer is live.

At round 0 after a failure event (and in later rounds), an online router's
previously finite route whose selected next hop is no longer an available
direct neighbor, or whose selected next hop now advertises the destination
as unreachable, is immediately reported as `null`/`null` and starts a timer
for `holdDownRounds` complete update rounds (the conclusion propagates
transitively, so a route via such a hop is held as well). While a timer is
live, finite advertisements from other neighbors cannot restore that
destination; a destination that becomes a directly connected neighbor is
adopted immediately at its direct metric and clears the timer. Repeated
unreachable advertisements never extend a running timer; after the
remainder reaches zero the destination rejoins normal selection in the next
round. A router's own route and a down router's whole row keep their existing
semantics, and down routers carry no timers. Rounds are recorded until both
the forwarding tables and the `holdDowns` maps stop changing, and
`convergenceRound` points at the last recorded round. Synchronous updates,
the event format, `infinityMetric`, the infinity threshold and the
smaller-next-hop tie rule are unchanged. `replay`, `replay-ls`, `compute`
and `converge` ignore `holdDownRounds`.

The scenario may additionally carry `poisonReverse`: a JSON boolean
selecting receiver-specific poison-reverse advertisements. Omission or
`false` leaves input, output and behavior byte-for-byte unchanged (the
result gains no such field); a non-boolean value reports `invalid scenario`
with exit status 2, again after topology errors. When `true`, the root
object echoes `poisonReverse: true`; the topology, events, timeline
entries and forwarding-entry formats are otherwise unchanged.

Each router's local table keeps storing the actually selected next hop and
metric. Only the advertisements of the synchronous exchanges after round 0
are generated per receiver: when the sender's current next hop to a
destination is exactly the receiving neighbor, it advertises that
destination unreachable to that neighbor alone, while every other neighbor
still receives the stored metric. The receiver chooses candidates only
from the advertisements addressed to it in the previous round, adds the
current direct-link metric, still treats a candidate at or above
`infinityMetric` as unreachable, and still breaks finite ties to the
smaller next-hop name. The fault-free baseline begins at the same
self/direct round 0, and the round 0 immediately after an event keeps its
existing immediate invalidation, node-recovery initialization and
link-recovery direct-route rules; poison reverse first takes effect at the
following synchronous exchange, so a loop that previously counted to
infinity can be suppressed immediately. Only rounds that change are still
recorded and `convergenceRound` still points at the stable round. The
stable tables are the same shortest-path answers as without poison reverse.

`poisonReverse` combines independently with `holdDownRounds`: the
receiver-specific advertisements of the previous round are obtained first,
then the existing invalidation decision, timer countdown, immediate timer
clearing on a direct-link recovery and refusal of finite alternatives while
suppressed all apply unchanged; the per-round `routers` and `holdDowns`
structures are unchanged and convergence still requires both to be stable.
`replay`, `replay-ls`, `compute` and `converge` ignore `poisonReverse`.

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
