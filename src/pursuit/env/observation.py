"""Single adapter that builds every learned actor observation from a LocalInfo.

See docs/observation_schema.md for field-by-field documentation.
"""
from __future__ import annotations

import numpy as np

from ..controllers.local import MAX_SLOTS, LocalInfo
from .geometry import NC, SIZE

N = SIZE - 1
SLOT_DIM = 21
# vector layout
OFF_ID = 0                      # 6   NPC ID one-hot
OFF_TIME = 6                    # 1   remaining time (H - t)/H
OFF_BUDGET = 7                  # 1   B
OFF_CREDIT = 8                  # 1   available send credit, min(credit,8)/8
OFF_PREV = 9                    # 6   previous movement one-hot (5) + previous send (1)
OFF_SIGHT = 15                  # 10  valid, x, y, age/H, prev disp one-hot(5), disp valid
OFF_ROLE = 25                   # 7   role one-hot(4), dest valid, dest x, dest y
OFF_PKT = 32                    # 6 x 21 latest packet per teammate slot
OFF_PRED = OFF_PKT + MAX_SLOTS * SLOT_DIM   # 7 explicit prediction features
VEC_DIM = OFF_PRED + 7


def _slot_vec(s, t, H):
    v = np.zeros(SLOT_DIM, dtype=np.float32)
    if s is None:
        return v
    v[0] = 1.0
    if s["sender_cell"] >= 0:
        v[1], v[2] = (s["sender_cell"] % SIZE) / N, (s["sender_cell"] // SIZE) / N
    if s["target_cell"] >= 0:
        v[3] = 1.0
        v[4], v[5] = (s["target_cell"] % SIZE) / N, (s["target_cell"] // SIZE) / N
        v[6] = (t - s["target_step"]) / H
    if s["disp_valid"]:
        v[7 + s["disp"]] = 1.0
        v[12] = 1.0
    if s["role"] >= 0:
        v[13 + s["role"]] = 1.0
    if s["dest"] >= 0:
        v[17] = 1.0
        v[18], v[19] = (s["dest"] % SIZE) / N, (s["dest"] // SIZE) / N
    v[20] = (t - s["send_step"]) / H
    return v


def build_obs(info: LocalInfo, budget: float, features7=None):
    t, H = info.t, info.H
    maps = np.zeros((5, SIZE, SIZE), dtype=np.uint8)
    maps[0] = info.map.walls
    maps[1] = info.vis.reshape(SIZE, SIZE)
    maps[2].flat[info.own_cell] = 1
    for _, c in info.teammates:
        maps[3].flat[c] = 1
    if info.direct_cell >= 0:
        maps[4].flat[info.direct_cell] = 1
    v = np.zeros(VEC_DIM, dtype=np.float32)
    v[OFF_ID + info.id] = 1.0
    v[OFF_TIME] = (H - t) / H
    v[OFF_BUDGET] = budget
    v[OFF_CREDIT] = min(info.credit(), 8) / 8.0
    v[OFF_PREV + info.prev_move] = 1.0
    v[OFF_PREV + 5] = info.prev_send
    if info.mem_cell >= 0:
        o = OFF_SIGHT
        v[o] = 1.0
        v[o + 1], v[o + 2] = (info.mem_cell % SIZE) / N, (info.mem_cell // SIZE) / N
        v[o + 3] = (t - info.mem_step) / H
        if info.mem_disp_valid:
            v[o + 4 + info.mem_disp] = 1.0
            v[o + 9] = 1.0
    v[OFF_ROLE + info.role] = 1.0
    if info.dest >= 0:
        v[OFF_ROLE + 4] = 1.0
        v[OFF_ROLE + 5], v[OFF_ROLE + 6] = (info.dest % SIZE) / N, (info.dest // SIZE) / N
    for j, s in enumerate(info.slots):
        if j != info.id:
            v[OFF_PKT + j * SLOT_DIM: OFF_PKT + (j + 1) * SLOT_DIM] = _slot_vec(s, t, H)
    if features7 is not None:
        v[OFF_PRED:OFF_PRED + 7] = features7
    return maps, v


def action_mask(info: LocalInfo) -> np.ndarray:
    """Joint 10-way mask: index = move*2 + send."""
    legal = info.map.legal[info.own_cell]
    can = info.can_send()
    m = np.zeros(10, dtype=bool)
    for a in range(5):
        if legal[a]:
            m[2 * a] = True
            m[2 * a + 1] = can
    return m


# ------------------------------------------------------------- critic state
CRITIC_VEC = 1 + 1 + 6 + 6 + 1 + 6 + 5 + 1 + 1
CRITIC_DIM = 3 * NC + CRITIC_VEC


def build_critic(core, player, budget: float) -> np.ndarray:
    """Centralised training state: full positions, map, channel credit/usage, time and the
    fixed training-opponent controller state. Never used at execution."""
    grid = np.zeros((3, SIZE, SIZE), dtype=np.float32)
    grid[0] = core.map.walls
    for c in core.npc:
        grid[1].flat[c] += 1.0
    grid[2].flat[core.pcell] = 1.0
    v = np.zeros(CRITIC_VEC, dtype=np.float32)
    v[0] = (core.H - core.t) / core.H
    v[1] = budget
    tt = min(core.t, core.H - 1)
    for i in range(core.N):
        v[2 + i] = min(core.channel.sent[i], 40) / 40.0
        v[8 + i] = min(max(core.channel.credit(i, tt), 0), 8) / 8.0
    v[14] = len(core.channel.pending) / 4.0
    if player.kind == "waypoint":
        v[15 + player.phase] = 1.0
        v[27] = player.xi
    else:
        v[21 + player.prev_action] = 1.0
        v[26] = 1.0
    return np.concatenate([grid.reshape(-1), v])
