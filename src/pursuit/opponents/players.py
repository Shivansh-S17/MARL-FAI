"""Players: controlled waypoint player and genuinely-updating tabular Q-learning player."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from ..env.geometry import ACTIONS, SIZE, GridMap, cell, xy

WAYPOINTS = [(3, 3), (10, 3), (16, 3), (16, 16), (10, 16), (3, 16)]
WAYPOINT_CELLS = [cell(x, y) for x, y in WAYPOINTS]


@dataclass
class PlayerObs:
    pos: int
    t: int
    npcs: tuple  # ((npc_id, cell), ...) visible to the player (radius, line of sight)


def _random_legal(gmap: GridMap, pos: int, u: float) -> int:
    legal = np.flatnonzero(gmap.legal[pos])
    return int(legal[min(int(u * len(legal)), len(legal) - 1)])


class WaypointPlayer:
    kind = "waypoint"
    learning = False

    def __init__(self, gmap: GridMap, xi: float = 0.0):
        self.gmap, self.xi = gmap, xi
        self.phase = 0

    def reset(self, scenario):
        self.phase = scenario.phase % len(WAYPOINT_CELLS)

    def act(self, pobs: PlayerObs, u_eps: float, u_act: float) -> int:
        if pobs.pos == WAYPOINT_CELLS[self.phase]:
            self.phase = (self.phase + 1) % len(WAYPOINT_CELLS)
        intended = int(self.gmap.best_action[pobs.pos, WAYPOINT_CELLS[self.phase]])
        if u_eps < self.xi:
            return _random_legal(self.gmap, pobs.pos, u_act)
        return intended

    def learn(self, *a, **k):
        pass

    def public_state(self):
        return dict(kind="waypoint", phase=self.phase, xi=self.xi)


N_Q_STATES = 16 * 9 * 3 * 5


class QPlayer:
    kind = "qlearning"

    def __init__(self, gmap: GridMap, horizon: int = 200, epsilon=0.10, alpha=0.10, gamma=0.95,
                 Q: np.ndarray | None = None, learning: bool = False):
        self.gmap, self.H = gmap, horizon
        self.epsilon, self.alpha, self.gamma = epsilon, alpha, gamma
        self.Q = np.zeros((N_Q_STATES, 5)) if Q is None else Q.copy()
        self.learning = learning
        self.prev_action = 0
        self._last = None
        self.n_updates = 0
        self.visited = set()

    # -- state aggregation -------------------------------------------------
    def key(self, pobs: PlayerObs, prev_action: int | None = None) -> int:
        pa = self.prev_action if prev_action is None else prev_action
        region = int(self.gmap.region[pobs.pos])
        if pobs.npcs:
            px, py = xy(pobs.pos)
            best = None
            for nid, c in pobs.npcs:  # nearest by Manhattan, ties by ID
                nx, ny = xy(c)
                d = abs(nx - px) + abs(ny - py)
                if best is None or (d, nid) < (best[0], best[1]):
                    best = (d, nid, nx - px, ny - py)
            d, _, dx, dy = best
            direction = (int(np.sign(dx)) + 1) * 3 + (int(np.sign(dy)) + 1)
            db = 0 if d <= 2 else 1
        else:
            direction, db = 4, 2  # 'none'
        return ((region * 9 + direction) * 3 + db) * 5 + pa

    def reset(self, scenario):
        self.prev_action = 0
        self._last = None

    def greedy(self, key: int, pos: int) -> int:
        q = np.where(self.gmap.legal[pos], self.Q[key], -np.inf)
        return int(np.argmax(q))  # first max -> fixed-order ties

    def act(self, pobs: PlayerObs, u_eps: float, u_act: float) -> int:
        key = self.key(pobs)
        a = _random_legal(self.gmap, pobs.pos, u_act) if u_eps < self.epsilon else self.greedy(key, pobs.pos)
        self._last = (key, a)
        self.visited.add(key)
        self.prev_action = a
        return a

    def learn(self, captured: bool, done: bool, pobs_next: PlayerObs | None):
        if not self.learning or self._last is None:
            return
        key, a = self._last
        r = -1.0 if captured else 1.0 / self.H
        if done:
            target = r
        else:
            key2 = self.key(pobs_next)
            q2 = np.where(self.gmap.legal[pobs_next.pos], self.Q[key2], -np.inf).max()
            target = r + self.gamma * q2
        self.Q[key, a] += self.alpha * (target - self.Q[key, a])
        self.n_updates += 1

    def clone(self, learning: bool | None = None) -> "QPlayer":
        c = QPlayer(self.gmap, self.H, self.epsilon, self.alpha, self.gamma, self.Q,
                    self.learning if learning is None else learning)
        return c

    def q_hash(self) -> str:
        return hashlib.sha256(self.Q.tobytes()).hexdigest()[:16]

    def public_state(self):
        return dict(kind="qlearning", learning=self.learning, n_updates=self.n_updates, prev_action=self.prev_action)
