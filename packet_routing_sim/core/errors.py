"""Deterministic, distinguishable validation failures for the simulation core.

The core never reads files or streams; every malformed input is reported by
raising one of these exceptions so callers (e.g. the command layer) can map
it to their own messages and exit codes.

Hierarchy:

* ``InvalidTopology``       - topology structure or values are not legal
* ``InvalidScenario``       - scenario structure or values are not legal
* ``InvalidStateTransition`` - a structurally valid event requests an illegal
                               state change; it is a kind of invalid scenario
                               but remains separately catchable
"""


class SimulationError(ValueError):
    """Base class for all core validation failures."""

    def __init__(self, reason=None):
        self.reason = reason or self.default_reason
        super().__init__(self.reason)

    default_reason = "simulation error"


class InvalidTopology(SimulationError):
    """The decoded topology value is not a legal topology."""

    default_reason = "invalid topology"


class InvalidScenario(SimulationError):
    """The decoded scenario value is not a legal scenario."""

    default_reason = "invalid scenario"


class InvalidStateTransition(InvalidScenario):
    """An event is well-formed but cannot be applied from the current state."""

    default_reason = "invalid state transition"
