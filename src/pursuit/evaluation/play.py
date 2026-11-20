"""Evaluation episodes for rule-based and learned pursuers (frozen / online protocols)."""
from __future__ import annotations

import numpy as np
import torch

from ..controllers.baselines import IPController, RCController
from ..env.observation import action_mask, build_obs
from ..env.scenario import Scenario, sample_for_cfg, sample_scenario
from ..env.team import Team, make_core
from ..models.belief import Belief
from ..models.networks import Actor
from ..models.predictor import PredictorBank


class RuleController:
    def __init__(self, kind):
        self.c = IPController() if kind == "IP" else RCController()
        self.name = kind

    def begin_episode(self):
        pass

    def act_team(self, team):
        out = [self.c.act(inf) for inf in team.infos]
        return [o[0] for o in out], [bool(o[1]) and inf.can_send() for o, inf in zip(out, team.infos)]


class LearnedController:
    """Greedy (arg-max, fixed-order ties) decentralised execution of the shared GRU actor."""

    def __init__(self, actor: Actor, budget: float, n: int = 4):
        self.actor, self.budget, self.n = actor, budget, n
        self.name = "learned"
        self.begin_episode()

    def begin_episode(self):
        self.h = torch.zeros(self.n, self.actor.hidden)

    @torch.no_grad()
    def act_team(self, team):
        maps, vec, mask = [], [], []
        for inf in team.infos:
            f7 = inf.belief.features() if inf.belief is not None else None
            m, v = build_obs(inf, self.budget, f7)
            maps.append(m); vec.append(v); mask.append(action_mask(inf))
        logits, self.h = self.actor.step(torch.from_numpy(np.stack(maps)), torch.from_numpy(np.stack(vec)),
                                         self.h, torch.from_numpy(np.stack(mask)))
        a = logits.numpy().argmax(-1)  # first max wins -> fixed-order ties
        return (a // 2).tolist(), (a % 2).astype(bool).tolist()


class EvalPredictorHook:
    """Wraps a PredictorBank for evaluation: records prediction quality on every permitted label,
    optionally lets labels enter the buffers (online) or discards them (frozen)."""

    def __init__(self, bank: PredictorBank, online: bool):
        self.bank, self.online = bank, online
        self.reset_stats()

    def reset_stats(self):
        self.n, self.ll, self.acc = 0, 0.0, 0.0

    def __call__(self, agent_id, own_cell, target_cell, prev_disp, teammates, label):
        feat = self.bank.fb.single(own_cell, target_cell, prev_disp, teammates)
        legal = self.bank.gm.legal[target_cell]
        p = self.bank.preds[agent_id].probs(feat[None], legal[None])[0]
        self.n += 1
        self.ll += -float(np.log(max(p[label], 1e-9)))
        self.acc += float(p.argmax() == label)
        if self.online:
            self.bank.preds[agent_id].add(feat, legal, label)


def build_team(cfg, core, bank: PredictorBank | None, hook=None, use_packets=True) -> Team:
    beliefs = None
    if bank is not None:
        beliefs = [Belief(core.map, core.vis, bank.preds[i], bank.fb) for i in range(core.N)]
    return Team(core, use_packets, beliefs, hook)


def play(core, team, scenario: Scenario, ctrl, recorder=None, bank=None, online=False, online_every=20):
    core.reset(scenario)
    team.reset()
    ctrl.begin_episode()
    while not core.done:
        team.begin_step()
        moves, sends = ctrl.act_team(team)
        if recorder is not None:
            recorder.pre_step(core, team, moves, sends)
        res = team.commit_and_step(moves, sends)
        if recorder is not None:
            recorder.post_step(core, team, res)
        if online and bank is not None and (core.t % online_every == 0 or res.done):
            bank.update_all()
    return dict(scenario_id=scenario.scenario_id, captured=int(core.captured), length=core.t,
                capped_duration=core.t, packets=core.channel.total_sent, bits=core.channel.total_sent * 128,
                moves=core.total_moves, npc_steps=core.t * core.N)


def make_scenarios(cfg, seed_start: int, n: int, prefix="s"):
    from ..env.geometry import load_map
    gm = load_map(cfg.env.map)
    out = []
    for k in range(n):
        s = seed_start + k
        rng = np.random.default_rng(np.random.SeedSequence([s, 3]))
        out.append(sample_for_cfg(cfg, gm, rng, None, f"{prefix}{s}"))
    return out
