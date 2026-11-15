"""Trace recording (evaluation-recorder-only data) and loading for the replay viewer.

Traces hold full-state ground truth for visualisation/analysis. They are never fed back to controllers.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..env.channel import Packet, allowed_sends


class TraceRecorder:
    def __init__(self, record_belief: bool = False):
        self.record_belief = record_belief
        self.rows = []
        self.meta = {}

    # called after team.begin_step(), before the transition: state at decision time t
    def pre_step(self, core, team, moves, sends):
        N = core.N
        row = dict(
            t=core.t,
            npc=list(core.npc), player=core.pcell,
            moves=[int(m) for m in moves], sends=[bool(s) for s in sends],
            roles=[inf.role for inf in team.infos], dests=[inf.dest for inf in team.infos],
            vis=np.stack([np.packbits(inf.vis) for inf in team.infos]),
            direct=[inf.direct_cell for inf in team.infos],
            mem_cell=[inf.mem_cell for inf in team.infos], mem_step=[inf.mem_step for inf in team.infos],
            teammates=[[(j, c) for j, c in inf.teammates] for inf in team.infos],
            delivered=[[(p.sender, p.send_step, p.target_cell if p.flags & 2 else -1)
                        for p in (Packet.unpack(r) for r in inf.last_delivered_raw)] for inf in team.infos],
            sent_cum=list(core.channel.sent),
            moves_cum=core.total_moves,
            allowed=allowed_sends(core.budget, core.t),
            player_state=core.player.public_state(),
        )
        if self.record_belief:
            row["belief"] = np.stack([inf.belief.marginal_full().astype(np.float16) if inf.belief is not None
                                      else np.zeros(400, np.float16) for inf in team.infos])
        self.rows.append(row)

    def post_step(self, core, team, res):
        self.rows[-1]["captured_after"] = bool(res.captured)
        self.rows[-1]["done_after"] = bool(res.done)
        self.rows[-1]["npc_after"] = list(core.npc)
        self.rows[-1]["player_after"] = core.pcell
        self.rows[-1]["sent_cum_after"] = list(core.channel.sent)

    def finalize(self, path, meta: dict):
        r = self.rows
        T = len(r)
        arr = dict(
            npc=np.array([x["npc"] for x in r], np.int16), player=np.array([x["player"] for x in r], np.int16),
            npc_after=np.array([x["npc_after"] for x in r], np.int16),
            player_after=np.array([x["player_after"] for x in r], np.int16),
            moves=np.array([x["moves"] for x in r], np.int8), sends=np.array([x["sends"] for x in r], bool),
            roles=np.array([x["roles"] for x in r], np.int8), dests=np.array([x["dests"] for x in r], np.int16),
            vis=np.stack([x["vis"] for x in r]), direct=np.array([x["direct"] for x in r], np.int16),
            mem_cell=np.array([x["mem_cell"] for x in r], np.int16),
            mem_step=np.array([x["mem_step"] for x in r], np.int16),
            sent_cum=np.array([x["sent_cum_after"] for x in r], np.int16),
            moves_cum=np.array([x["moves_cum"] for x in r], np.int32),
            allowed=np.array([x["allowed"] for x in r], np.int32),
        )
        if self.record_belief:
            arr["belief"] = np.stack([x["belief"] for x in r])
        meta = dict(meta)
        meta.update(T=T, captured=bool(r[-1]["captured_after"]),
                    teammates=[x["teammates"] for x in r], delivered=[x["delivered"] for x in r],
                    player_state=[x["player_state"] for x in r], has_belief=self.record_belief)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, meta=json.dumps(meta), **arr)
        return path


class Trace:
    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.path = str(path)
        self.meta = json.loads(str(z["meta"]))
        self.a = {k: z[k] for k in z.files if k != "meta"}
        self.T = self.meta["T"]
        self.N = self.a["npc"].shape[1]

    def vis(self, t, i):
        return np.unpackbits(self.a["vis"][t, i])[:400].astype(bool)

    def positions(self, t):
        """Positions at decision time t (t == T gives the terminal frame)."""
        if t < self.T:
            return self.a["npc"][t], int(self.a["player"][t])
        return self.a["npc_after"][-1], int(self.a["player_after"][-1])
