"""Train the tabular Q-learning player bank against rule-based coordination at B=1."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..controllers.baselines import RCController
from ..env.geometry import load_map
from ..env.scenario import sample_scenario
from ..env.team import Team, make_core
from .players import QPlayer

TRAIN_BANK = [7000, 7001, 7002]
HELDOUT_BANK = [7100, 7101, 7102, 7103, 7104]


def train_q_player(cfg, seed: int, steps: int = 200_000, out_dir=None):
    gm = load_map(cfg.env.map)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 5]))
    q = QPlayer(gm, cfg.env.horizon, cfg.q.epsilon, cfg.q.alpha, cfg.q.gamma, learning=True)
    core = make_core(cfg, player=q, budget="1")
    team = Team(core)
    rc = RCController()
    total, curve, win = 0, [], []
    snaps = {}
    t0 = time.time()
    next_mark = 10_000
    prev_greedy = None
    changes = []
    while total < steps:
        sc = sample_scenario(gm, rng, cfg.env.n_npcs, cfg.env.min_start_dist)
        core.reset(sc); team.reset()
        while not core.done:
            team.begin_step()
            acts = [rc.act(inf) for inf in team.infos]
            team.commit_and_step([a[0] for a in acts], [bool(a[1]) and inf.can_send() for a, inf in zip(acts, team.infos)])
            total += 1
        win.append((core.captured, core.t))
        if total >= next_mark:
            caps = [w[0] for w in win]; lens = [w[1] for w in win]
            greedy = q.Q.argmax(1)
            ch = None if prev_greedy is None else int((greedy != prev_greedy).sum())
            prev_greedy = greedy
            curve.append(dict(steps=total, capture_rate=float(np.mean(caps)), mean_survival=float(np.mean(lens)),
                              episodes=len(win), q_updates=q.n_updates, greedy_changes_vs_prev=ch,
                              states_visited=len(q.visited)))
            win = []
            next_mark += 10_000
    meta = dict(seed=seed, steps=total, q_updates=q.n_updates, elapsed_s=time.time() - t0, curve=curve,
                visited=sorted(int(k) for k in q.visited), q_hash=q.q_hash())
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / f"q_{seed}.npz", Q=q.Q, visited=np.array(meta["visited"]))
        (out / f"q_{seed}.json").write_text(json.dumps({k: v for k, v in meta.items() if k != "visited"}, indent=1))
    return q, meta


def load_q(cfg, seed: int, bank_dir, learning=False):
    z = np.load(Path(bank_dir) / f"q_{seed}.npz")
    return QPlayer(load_map(cfg.env.map), cfg.env.horizon, cfg.q.epsilon, cfg.q.alpha, cfg.q.gamma, z["Q"], learning)
