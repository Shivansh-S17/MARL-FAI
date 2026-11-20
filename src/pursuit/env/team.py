"""Team: couples the simulator core with per-NPC LocalInfo, and runs episodes."""
from __future__ import annotations

import numpy as np

from ..controllers.local import LocalInfo
from ..opponents.players import QPlayer, WaypointPlayer
from .core import PursuitCore
from .geometry import load_map


def make_core(cfg, player=None, budget=None) -> PursuitCore:
    gm = load_map(cfg.env.map)
    from ..config import parse_budget
    b = parse_budget(cfg.env.budget if budget is None else budget)
    core = PursuitCore(gm, cfg.env.n_npcs, cfg.env.horizon, cfg.env.radius, cfg.env.delay, b, dict(cfg.reward),
                      player_radius=cfg.env.get('player_radius', 5))
    core.set_player(player if player is not None else make_player(cfg, gm))
    return core


def make_player(cfg, gm):
    if cfg.player.type == "waypoint":
        return WaypointPlayer(gm, cfg.player.xi)
    return QPlayer(gm, cfg.env.horizon, cfg.q.epsilon, cfg.q.alpha, cfg.q.gamma)


class Team:
    def __init__(self, core: PursuitCore, use_packets=True, beliefs=None, labeller=None):
        self.core = core
        self.infos = [LocalInfo(i, core.map, core.N, core.H, core.budget, core.r, use_packets,
                                None if beliefs is None else beliefs[i], labeller) for i in range(core.N)]

    def reset(self):
        for inf in self.infos:
            inf.reset()

    def begin_step(self):
        """Deliver packets, push permitted observations into each NPC's local state."""
        core = self.core
        deliveries = core.deliver()
        for i, inf in enumerate(self.infos):
            inf.update(core.observe(i), deliveries[i])

    def send_mask(self):
        return [inf.can_send() for inf in self.infos]

    def commit_and_step(self, moves, sends):
        payloads = [self.infos[i].build_packet() if sends[i] else None for i in range(self.core.N)]
        for i, inf in enumerate(self.infos):
            inf.commit(int(moves[i]), bool(sends[i]))
        res = self.core.step(moves, payloads)
        if res.done:
            for i, inf in enumerate(self.infos):
                inf.observe_terminal(self.core.observe(i))
        return res


def run_episode(core: PursuitCore, team: Team, scenario, controller, recorder=None):
    """Run one full episode with a rule-based controller object having act(info)->(move,send)."""
    core.reset(scenario)
    team.reset()
    res = None
    while not core.done:
        team.begin_step()
        acts = [controller.act(inf) for inf in team.infos]
        moves = [a[0] for a in acts]
        sends = [bool(a[1]) and team.infos[i].can_send() for i, a in enumerate(acts)]
        if recorder is not None:
            recorder.pre_step(core, team, moves, sends)
        res = team.commit_and_step(moves, sends)
        if recorder is not None:
            recorder.post_step(core, team, res)
    return dict(captured=core.captured, length=core.t, packets=core.channel.total_sent,
                moves=core.total_moves, sent=list(core.channel.sent))
