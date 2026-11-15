"""Scenarios (initial conditions) and exogenous random draws indexed by episode/time."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .geometry import GridMap


@dataclass
class Scenario:
    npc_cells: list
    player_cell: int
    phase: int            # initial waypoint index (ignored by the Q player)
    player_seed: int      # seed for the player's per-time exogenous draws
    scenario_id: str = ""
    meta: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return Scenario(**d)


def exo_draws(player_seed: int, horizon: int):
    """Per-time uniform draws for the player; indexed by time so that changed NPC
    actions never shift the player's randomness stream."""
    rng = np.random.default_rng(np.random.SeedSequence([int(player_seed), 7]))
    return rng.random(horizon + 1), rng.random(horizon + 1)


def sample_scenario(gmap: GridMap, rng: np.random.Generator, n_npcs: int, min_dist: int = 4,
                    dist_range: tuple[int, int] | None = None, scenario_id: str = "") -> Scenario:
    """Distinct uniform cells by rejection sampling. Standard: every NPC-player graph
    distance >= min_dist. Curriculum: every distance within [lo, hi]."""
    free = gmap.free_cells
    lo, hi = dist_range if dist_range else (min_dist, 10 ** 6)
    for _ in range(100000):
        p = int(rng.choice(free))
        if dist_range:  # sample NPCs directly from the allowed ring (still uniform over allowed cells)
            ring = free[(gmap.dist[p, free] >= lo) & (gmap.dist[p, free] <= hi)]
            if len(ring) < n_npcs:
                continue
            npcs = [int(c) for c in rng.choice(ring, size=n_npcs, replace=False)]
            ok = True
        else:
            npcs = [int(c) for c in rng.choice(free, size=n_npcs, replace=False)]
            ok = all(gmap.dist[p, c] >= lo for c in npcs) and p not in npcs
        if ok:
            phase = int(rng.integers(0, 6))
            pseed = int(rng.integers(0, 2 ** 31 - 1))
            return Scenario(npcs, p, phase, pseed, scenario_id)
    raise RuntimeError("could not sample scenario")
