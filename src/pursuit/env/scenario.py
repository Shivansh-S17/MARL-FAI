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


ROOM_EDGES = (6, 13)   # vertical walls of the corridor map separate three rooms


def room_of(c: int) -> int:
    """0/1/2 for the three rooms; -1 for a doorway column cell."""
    x = c % 20
    if x < ROOM_EDGES[0]:
        return 0
    if ROOM_EDGES[0] < x < ROOM_EDGES[1]:
        return 1
    if x > ROOM_EDGES[1]:
        return 2
    return -1


def sample_split_scenario(gmap: GridMap, rng: np.random.Generator, n_npcs: int, radius: int,
                          scenario_id: str = "") -> Scenario:
    """Informative split-spawn sampler (plan 6.1): one observer NPC starts at graph distance exactly 3
    with line of sight to the target; the other NPCs start at distance 8..12 without sight of the target
    and together occupy at least two rooms. Uniform rejection sampling over the constraints; target cells
    admitting no configuration are skipped. Observer identity and the waypoint phase are not disclosed."""
    vis = gmap.visibility(radius)
    free = gmap.free_cells
    for _ in range(100000):
        p = int(rng.choice(free))
        obs_c = free[(gmap.dist[p, free] == 3) & vis[free, p]]
        ring = free[(gmap.dist[p, free] >= 8) & (gmap.dist[p, free] <= 12) & ~vis[free, p]]
        if len(obs_c) == 0 or len(ring) < n_npcs - 1:
            continue
        observer = int(rng.integers(n_npcs))
        oc = int(rng.choice(obs_c))
        others = [int(c) for c in rng.choice(ring, size=n_npcs - 1, replace=False)]
        rooms = {room_of(c) for c in others} - {-1}
        if len(rooms) < min(2, n_npcs - 1) or oc in others:
            continue
        npcs, it = [], iter(others)
        for i in range(n_npcs):
            npcs.append(oc if i == observer else next(it))
        phase = int(rng.integers(0, 6))
        pseed = int(rng.integers(0, 2 ** 31 - 1))
        return Scenario(npcs, p, phase, pseed, scenario_id, meta=dict(sampler="split", observer=observer))
    raise RuntimeError("could not sample split scenario")


def sample_for_cfg(cfg, gmap: GridMap, rng: np.random.Generator, dist_range=None, scenario_id: str = "") -> Scenario:
    """Task-level dispatcher. A curriculum dist_range relaxes any split constraint (disclosed in the protocol)."""
    if cfg.env.get("spawn", "standard") == "split" and not dist_range:
        return sample_split_scenario(gmap, rng, cfg.env.n_npcs, cfg.env.radius, scenario_id)
    return sample_scenario(gmap, rng, cfg.env.n_npcs, cfg.env.min_start_dist, dist_range, scenario_id)
