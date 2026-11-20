"""PettingZoo Parallel-API wrapper around the same core/team used everywhere else."""
from __future__ import annotations

import functools

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from ..config import parse_budget
from .observation import VEC_DIM, action_mask, build_obs
from .scenario import sample_for_cfg
from .team import Team, make_core


class PursuitParallelEnv(ParallelEnv):
    metadata = {"name": "pursuit_v0"}

    def __init__(self, cfg):
        self.cfg = cfg
        self.core = make_core(cfg)
        self.team = Team(self.core)
        self.possible_agents = [f"npc_{i}" for i in range(cfg.env.n_npcs)]
        self.agents = []
        self._rng = np.random.default_rng(0)
        self._b = float(parse_budget(cfg.env.budget))

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Dict(dict(map=spaces.Box(0, 1, (5, 20, 20), np.uint8),
                                vec=spaces.Box(-10.0, 10.0, (VEC_DIM,), np.float32),
                                action_mask=spaces.Box(0, 1, (10,), np.int8)))

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(10)  # move*2 + send

    def _obs(self):
        self.team.begin_step()
        out = {}
        for i, a in enumerate(self.possible_agents):
            m, v = build_obs(self.team.infos[i], self._b, None)
            out[a] = dict(map=m, vec=v, action_mask=action_mask(self.team.infos[i]).astype(np.int8))
        return out

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        sc = sample_for_cfg(self.cfg, self.core.map, self._rng)
        self.core.reset(sc)
        self.team.reset()
        self.agents = list(self.possible_agents)
        return self._obs(), {a: {} for a in self.agents}

    def step(self, actions):
        moves = [int(actions[a]) // 2 for a in self.possible_agents]
        sends = [bool(int(actions[a]) % 2) and self.team.infos[i].can_send() for i, a in enumerate(self.possible_agents)]
        res = self.team.commit_and_step(moves, sends)
        rew = {a: float(res.reward) for a in self.possible_agents}
        term = {a: bool(res.captured) for a in self.possible_agents}
        trunc = {a: bool(res.done and not res.captured) for a in self.possible_agents}
        if res.done:
            obs = {a: dict(map=np.zeros((5, 20, 20), np.uint8), vec=np.zeros(VEC_DIM, np.float32),
                           action_mask=np.ones(10, np.int8)) for a in self.possible_agents}
            self.agents = []
        else:
            obs = self._obs()
        return obs, rew, term, trunc, {a: {} for a in self.possible_agents}
