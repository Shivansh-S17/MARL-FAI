"""Per-NPC permitted-information state: target memory, packet slots, role hint, history.

This is the ONLY place controllers, the observation adapter, the packet builder and the
predictor labeller read from.  It is fed exclusively by `LocalObs` (own cell, visible cells,
visible agents) and delivered packets.  Nothing here can read hidden simulator state.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from ..env.channel import F_DEST, F_DISP, F_ROLE, F_SENDER, F_TARGET, Packet, allowed_sends
from ..env.core import LocalObs
from ..env.geometry import ACTIONS, NC, GridMap, xy

MAX_SLOTS = 6
ROLE_CHASE, ROLE_INTERCEPT, ROLE_BLOCK, ROLE_SEARCH = 0, 1, 2, 3
FRESH = 10


def displacement_action(c0: int, c1: int) -> int:
    """Single-step displacement between consecutive sightings, or -1 if not one move."""
    x0, y0 = xy(c0)
    x1, y1 = xy(c1)
    d = (x1 - x0, y1 - y0)
    for a, v in enumerate(ACTIONS):
        if v == d:
            return a
    return -1


class LocalInfo:
    def __init__(self, agent_id: int, gmap: GridMap, n_npcs: int, horizon: int, budget: Fraction,
                 radius: int, use_packets: bool = True, belief=None, labeller=None):
        self.id, self.map, self.N, self.H = agent_id, gmap, n_npcs, horizon
        self.budget, self.radius, self.use_packets = budget, radius, use_packets
        self.belief, self.labeller = belief, labeller
        self.reset()

    def reset(self):
        self.t = -1
        self.own_cell = -1
        self.vis = None
        self.last_seen = -np.ones(NC, dtype=np.int32)
        self.mem_cell, self.mem_step, self.mem_disp, self.mem_disp_valid, self.mem_direct = -1, -1, 0, False, False
        self.slots = [None] * MAX_SLOTS  # latest packet per sender
        self.prev_move, self.prev_send = 0, 0
        self.sent = 0
        self.last_send_step = -10 ** 6
        self.last_sent_key = None
        self.role, self.dest = ROLE_SEARCH, -1
        self.history: list[tuple] = []  # per step: (own_cell, direct_cell, teammates)
        self.player_visible = False
        self.direct_cell = -1
        self.teammates = ()
        self.new_labels = 0
        self.last_delivered_raw = []
        if self.belief is not None:
            self.belief.reset()

    # ----------------------------------------------------------------- update
    def update(self, obs: LocalObs, delivered: list[bytes]):
        t = obs.t
        self.t, self.own_cell, self.vis = t, obs.own_cell, obs.vis
        self.last_seen[obs.vis] = t
        self.direct_cell = obs.player_cell
        self.player_visible = obs.player_cell >= 0
        self.teammates = obs.teammates
        self.last_delivered_raw = list(delivered)
        self.history.append((obs.own_cell, obs.player_cell, obs.teammates))
        candidates = []  # (step, source_direct, cell, disp, disp_valid)
        if self.use_packets:
            for raw in delivered:
                p = Packet.unpack(raw)
                if p.sender == self.id or p.sender >= MAX_SLOTS:
                    continue
                cur = self.slots[p.sender]
                if cur is None or p.send_step >= cur["send_step"]:
                    self.slots[p.sender] = dict(
                        send_step=p.send_step, arrival=t,
                        sender_cell=p.sender_cell if p.flags & F_SENDER else -1,
                        target_cell=p.target_cell if p.flags & F_TARGET else -1,
                        target_step=p.target_step if p.flags & F_TARGET else -1,
                        disp=p.disp, disp_valid=bool(p.flags & F_DISP),
                        role=p.role if p.flags & F_ROLE else -1,
                        dest=p.dest if p.flags & F_DEST else -1)
                if p.flags & F_TARGET:
                    candidates.append((p.target_step, False, p.target_cell, p.disp, bool(p.flags & F_DISP)))
        if obs.player_cell >= 0:
            candidates.append((t, True, obs.player_cell, 0, False))
        candidates.sort(key=lambda c: (c[0], c[1]))
        replay_from = None
        for step, direct, c, disp, dvalid in candidates:
            if step > self.mem_step or (step == self.mem_step and direct and not self.mem_direct):
                if self.mem_cell >= 0 and self.mem_step == step - 1:
                    d = displacement_action(self.mem_cell, c)
                    new_disp, new_valid = (max(d, 0), d >= 0)
                else:
                    new_disp, new_valid = (disp, dvalid) if not direct else (0, False)
                self.mem_cell, self.mem_step, self.mem_disp, self.mem_disp_valid = c, step, new_disp, new_valid
                self.mem_direct = direct
                replay_from = None if direct else step
        # predictor labels from three consecutive DIRECT sightings (frames t-2, t-1, t)
        self.new_labels = 0
        if self.labeller is not None and obs.player_cell >= 0 and len(self.history) >= 3:
            h = self.history
            if h[-1][1] >= 0 and h[-2][1] >= 0 and h[-3][1] >= 0:
                d_prev = displacement_action(h[-3][1], h[-2][1])
                d_next = displacement_action(h[-2][1], h[-1][1])
                if d_prev >= 0 and d_next >= 0:
                    self.labeller(self.id, h[-2][0], h[-2][1], d_prev, h[-2][2], d_next)
                    self.new_labels = 1
        if self.belief is not None:
            self.belief.update(self, replay_from)
        self._compute_role()

    def observe_terminal(self, obs: LocalObs):
        """Final post-transition frame: only used for a directly visible endpoint label."""
        if self.labeller is not None and obs.player_cell >= 0 and len(self.history) >= 2:
            h = self.history
            if h[-1][1] >= 0 and h[-2][1] >= 0:
                d_prev = displacement_action(h[-2][1], h[-1][1]) if len(h) >= 2 else -1
                d_next = displacement_action(h[-1][1], obs.player_cell)
                if d_prev >= 0 and d_next >= 0:
                    self.labeller(self.id, h[-1][0], h[-1][1], d_prev, h[-1][2], d_next)
                    self.new_labels += 1

    # ------------------------------------------------------------------- roles
    def target_age(self) -> int:
        return self.t - self.mem_step if self.mem_cell >= 0 else -1

    def search_dest(self) -> int:
        q = self.map.quadrants[self.id % 4]
        return int(q[np.argmin(self.last_seen[q])])

    def extrapolate(self, steps: int = 3) -> int:
        c = self.mem_cell
        if not self.mem_disp_valid:
            return c
        a = self.mem_disp
        for _ in range(steps):
            n = int(self.map.nxt[c, a])
            if n < 0:
                break
            c = n
        return c

    def _compute_role(self):
        gm = self.map
        cands = []
        if self.mem_cell >= 0:
            ext = self.extrapolate(3)
            cands.append((ROLE_CHASE, self.mem_cell))
            cands.append((ROLE_INTERCEPT, ext))
            if gm.doorways:
                cands.append((ROLE_BLOCK, int(gm.nearest_door[ext])))
        cands.append((ROLE_SEARCH, self.search_dest()))
        fresh = [s["dest"] for j, s in enumerate(self.slots)
                 if s is not None and j != self.id and s["dest"] >= 0 and self.t - s["send_step"] <= FRESH]
        best = None
        for role, c in cands:
            score = -float(gm.dist[self.own_cell, c]) / gm.diameter
            score -= 0.5 * sum(1 for d in fresh if gm.dist[d, c] <= 2)
            key = (-score, role, c)
            if best is None or key < best[0]:
                best = (key, role, c)
        self.role, self.dest = best[1], best[2]

    # ------------------------------------------------------------ channel side
    def credit(self) -> int:
        return allowed_sends(self.budget, self.t) - self.sent

    def can_send(self) -> bool:
        return self.credit() > 0

    def build_packet(self) -> bytes:
        fl = F_SENDER | F_DEST | F_ROLE
        p = Packet(self.id, 0, self.own_cell, 0, 0, 0, self.role, self.dest, self.t)
        if self.mem_cell >= 0:
            fl |= F_TARGET
            p.target_cell, p.target_step = self.mem_cell, self.mem_step
            if self.mem_disp_valid:
                fl |= F_DISP
                p.disp = self.mem_disp
        p.flags = fl
        return p.pack()

    def commit(self, move: int, send: bool):
        self.prev_move, self.prev_send = move, int(send)
        if send:
            self.sent += 1
            self.last_send_step = self.t
            self.last_sent_key = (self.role, self.dest)
