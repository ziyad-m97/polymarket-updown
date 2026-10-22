"""Pure functions of the snapshots observed so far in a cycle."""

from collections.abc import Sequence
from itertools import pairwise

from updown.domain import Snapshot


def lead_changes(snapshots: Sequence[Snapshot]) -> int:
    """How many times the favourite switched sides across consecutive snapshots."""
    return sum(a.favorite is not b.favorite for a, b in pairwise(snapshots))
