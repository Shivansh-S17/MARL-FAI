"""Simultaneous-move pursuit simulator core (no learning logic, no controller logic)."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from ..opponents.players import PlayerObs
from .channel import Channel
from .geometry import GridMap
from .scenario import Scenario, exo_draws


@dataclass
class LocalObs:
    """Everything the simulator reveals to one NPC at decision time t."""
    t: int
    own_cell: int
    vis: np.ndarray            # bool[400] currently visible cells (from own cell)
    player_cell: int           # -1 if not visible
    teammates: tuple           # ((id, cell), ...) visible teammates


@dataclass
class StepResult:
    reward: float
    base_reward: float
    shaped_reward: float
    done: bool
    captured: bool
    broadcasts: int
    moves: int


class PursuitCore:
    def __init__(self, gmap: GridMap, n_npcs=4, horizon=200, radius=5, delay=0,
                 budget=Fraction(1, 8), reward=None, player_radius=5):
        self.map, self.N, self.H, self.r, self.delay = gmap, n_npcs, horizon, radius, delay
        self.budget = Fraction(budget)
        self.vis = gmap.visibility(radius)
        self.pvis = self.vis if player_radius == radius else gmap.visibility(player_radius)  # player sees radius 5 regardless of NPC radius
        self.channel = Channel(n_npcs, self.budget, delay)
        rw = dict(capture=1.0, step=0.10, comm=0.01, move=0.02, shaping=False, shaping_coef=0.10)
        rw.update(reward or {})
        self.rw = rw
        self.player = None
        self.t = 0
        self.done = True

    # ------------------------------------------------------------------ setup
    def set_player(self, player):
        self.player = player

    def reset(self, scenario: Scenario):
        self.scenario = scenario
        self.npc = list(scenario.npc_cells)
        self.pcell = int(scenario.player_cell)
        self.t = 0
        self.done = False
        self.captured = False
        self.channel.reset()
        self.u_eps, self.u_act = exo_draws(scenario.player_seed, self.H)
        self.player.reset(scenario)
        self.total_moves = 0
        self.total_broadcasts = 0

    # ----------------------------------------------------------- observations
    def observe(self, i: int) -> LocalObs:
        c = self.npc[i]
        vis = self.vis[c]
        tm = tuple((j, self.npc[j]) for j in range(self.N) if j != i and vis[self.npc[j]])
        return LocalObs(self.t, c, vis, self.pcell if vis[self.pcell] else -1, tm)

    def player_obs(self) -> PlayerObs:
        v = self.pvis[self.pcell]
        return PlayerObs(self.pcell, self.t, tuple((j, c) for j, c in enumerate(self.npc) if v[c]))

    def potential(self, npc=None, pcell=None) -> float:
        npc = self.npc if npc is None else npc
        pcell = self.pcell if pcell is None else pcell
        return -min(int(self.map.dist[c, pcell]) for c in npc) / self.map.diameter

    def deliver(self):
        return self.channel.deliver(self.t)

    # ---------------------------------------------------------------- dynamics
    def step(self, moves, payloads) -> StepResult:
        """moves[i] in 0..4 for each NPC; payloads[i] is bytes if NPC i broadcasts else None.
        Packets are charged even if the capture happens on this transition."""
        assert not self.done
        t, gm = self.t, self.map
        phi0 = self.potential() if self.rw["shaping"] else 0.0
        nb = 0
        for i, p in enumerate(payloads):
            if p is not None:
                self.channel.send(i, t, p)
                nb += 1
        new_npc = []
        for i, m in enumerate(moves):
            n = int(gm.nxt[self.npc[i], m])
            assert n >= 0, f"illegal move {m} for NPC {i} at cell {self.npc[i]}"
            new_npc.append(n)
        # player decides from its pre-transition observation
        pobs = self.player_obs()
        pa = self.player.act(pobs, self.u_eps[t], self.u_act[t])
        new_p = int(gm.nxt[self.pcell, pa])
        assert new_p >= 0
        captured = False
        for i in range(self.N):
            if new_npc[i] == new_p or (new_npc[i] == self.pcell and self.npc[i] == new_p and new_npc[i] != self.npc[i]):
                captured = True
        nmoves = sum(1 for m in moves if m != 0)
        self.npc, self.pcell = new_npc, new_p
        self.t += 1
        self.captured = captured
        self.done = captured or self.t >= self.H
        H, N, rw = self.H, self.N, self.rw
        base = (rw["capture"] if captured else 0.0) - rw["step"] / H - rw["comm"] * nb / (N * H) \
            - rw["move"] * nmoves / (N * H)
        shaped = base
        if rw["shaping"]:
            phi1 = 0.0 if self.done else self.potential()
            shaped = base + rw["shaping_coef"] * (phi1 - phi0)
        self.total_moves += nmoves
        self.total_broadcasts += nb
        # player update from its own post-transition observation
        if getattr(self.player, "learning", False):
            self.player.learn(captured, self.done, None if self.done else self.player_obs())
        return StepResult(shaped if rw["shaping"] else base, base, shaped, self.done, captured, nb, nmoves)

    def terminal_local_obs(self, i: int) -> LocalObs:
        """Post-terminal view (used only to record a directly visible endpoint label)."""
        return self.observe(i)
