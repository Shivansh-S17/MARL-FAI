from fractions import Fraction

import numpy as np
import pytest
import torch

from conftest import mk, scen
from pursuit.config import load_config
from pursuit.controllers.local import LocalInfo
from pursuit.env.core import LocalObs
from pursuit.env.geometry import cell, load_map
from pursuit.models.belief import Belief
from pursuit.models.predictor import PredictorBank
from pursuit.training.trainer import Trainer, gae


def test_gae_hand_computed():
    rew = np.array([[1.0], [0.0], [2.0]], np.float32)
    val = np.array([[0.5], [0.4], [0.3]], np.float32)
    term = np.array([[False], [False], [True]])
    adv, ret = gae(rew, val, term, np.array([9.0], np.float32), 1.0, 0.95)
    d2 = 2.0 - 0.3                       # terminal: no bootstrap (value 9 ignored)
    d1 = 0.0 + 0.3 - 0.4
    d0 = 1.0 + 0.4 - 0.5
    a2 = d2; a1 = d1 + 0.95 * a2; a0 = d0 + 0.95 * a1
    np.testing.assert_allclose(adv[:, 0], [a0, a1, a2], rtol=1e-5)
    np.testing.assert_allclose(ret[:, 0], adv[:, 0] + val[:, 0], rtol=1e-5)
    # rollout boundary inside an episode bootstraps
    adv2, _ = gae(rew[:1], val[:1], np.array([[False]]), np.array([3.0], np.float32), 1.0, 0.95)
    np.testing.assert_allclose(adv2[0, 0], 1.0 + 3.0 - 0.5, rtol=1e-5)


def test_shaping_telescopes():
    cfg = load_config(overrides=dict(reward=dict(shaping=True), env=dict(budget="0")))
    from pursuit.env.team import make_core, Team
    core = make_core(cfg); team = Team(core)
    core.reset(scen([(2, 2), (3, 15), (16, 3), (15, 15)], (9, 9)))
    phi0 = core.potential()
    diff = 0.0
    rng = np.random.default_rng(0)
    while not core.done:
        mv = [int(rng.choice(np.flatnonzero(core.map.legal[c]))) for c in core.npc]
        r = core.step(mv, [None] * 4)
        diff += r.shaped_reward - r.base_reward
    assert diff == pytest.approx(0.10 * (0.0 - phi0), abs=1e-9)   # start-state constant


def test_ppo_ratio_one_and_gradients_and_masks():
    cfg = load_config(overrides=dict(ppo=dict(n_envs=2, rollout_len=32)))
    tr = Trainer(cfg, "marl_om", 3, dict(type="waypoint", xi=0.0), Fraction(1, 2), device="cpu")
    B, lv = tr.collect(32)
    # masked probabilities: chosen actions are always legal under stored masks
    m = B["mask"].reshape(-1, 10); a = B["act"].reshape(-1)
    assert m[np.arange(len(a)), a].all()
    # recompute log-probs through the sequence path: ratio == 1 before any update
    L = 16
    for e in range(2):
        for n in range(4):
            sl = slice(0, L)
            maps = torch.from_numpy(B["maps"][sl, e, n])[None]; vec = torch.from_numpy(B["vec"][sl, e, n])[None]
            mask = torch.from_numpy(B["mask"][sl, e, n])[None]
            h0 = torch.from_numpy(B["h"][0, e, n])[None]
            first = torch.from_numpy(B["first"][sl, e])[None]
            logits = tr.actor_act.sequence(maps, vec, h0, first, mask)
            lp = torch.distributions.Categorical(logits=logits).log_prob(torch.from_numpy(B["act"][sl, e, n])[None])
            np.testing.assert_allclose(lp[0].detach().numpy(), B["logp"][sl, e, n], atol=1e-4)
    info = tr.update(B, lv)
    assert all(np.isfinite(v) for v in info.values())
    for p in list(tr.actor.parameters()) + list(tr.critic.parameters()):
        assert torch.isfinite(p).all()
    assert info["gnorm"] > 0


def test_recurrent_reset_correctness():
    from pursuit.models.networks import Actor
    torch.manual_seed(0)
    act = Actor()
    T = 6
    maps = torch.randint(0, 2, (1, T, 5, 20, 20)).float(); vec = torch.randn(1, T, 165 + 0 if False else act.fc.in_features - 64)
    mask = torch.ones(1, T, 10, dtype=torch.bool)
    first = torch.zeros(1, T, dtype=torch.bool); first[0, 3] = True
    lg = act.sequence(maps, vec, torch.randn(1, 64), first, mask)
    # stepwise with explicit reset at t=3
    h = torch.zeros(1, 64)
    h0 = None
    lg2 = []
    # sequence() started from a random h0, so only steps >=3 are comparable to a zero-initialised step run
    h = torch.zeros(1, 64)
    for t in range(3, T):
        l, h = act.step(maps[:, t], vec[:, t], h, mask[:, t])
        lg2.append(l)
    np.testing.assert_allclose(lg[0, 3:].detach().numpy(), torch.cat(lg2).detach().numpy(), atol=1e-5)


def test_curriculum_and_budget_count():
    cfg = load_config(overrides=dict(ppo=dict(n_envs=2, rollout_len=8)))
    tr = Trainer(cfg, "marl", 1, dict(type="waypoint", xi=0.0), Fraction(1), device="cpu")
    tr.train(8 * 2 * 3 + 4, verbose=False)   # partial final batch must not exceed budget
    assert tr.steps == 8 * 2 * 3 + 4


# ------------------------------------------------------------- belief / predictor
def mk_belief(seed=0):
    cfg = load_config()
    g = load_map("corridor")
    bank = PredictorBank(g, cfg, seed)
    vis = g.visibility(5)
    return cfg, g, bank, vis


def test_belief_normalised_legal_and_visible_empty():
    cfg, g, bank, vis = mk_belief()
    info = LocalInfo(0, g, 4, 200, Fraction(1), 5, True, Belief(g, vis, bank.preds[0], bank.fb))
    own = cell(8, 8)
    for t in range(6):
        info.update(LocalObs(t, own, vis[own], -1, ()), [])
        b = info.belief.b
        assert b.sum() == pytest.approx(1.0)
        assert (b[:, :] >= 0).all()
        full = info.belief.marginal_full()
        assert (full[g.wall_flat] == 0).all()            # legal support
        assert (full[vis[own]] == 0).all()               # visible empty cells carry no mass
    info.update(LocalObs(6, own, vis[own], cell(9, 8), ()), [])
    assert info.belief.marginal_full()[cell(9, 8)] == pytest.approx(1.0)


def test_belief_delayed_sighting_replay():
    cfg, g, bank, vis = mk_belief()
    # train predictor a little so propagation is non-trivial
    rng = np.random.default_rng(0)
    for _ in range(100):
        c = int(rng.choice(g.free_cells))
        bank.preds[0].add(bank.fb.single(cell(8, 8), c, int(rng.integers(5)), ()), g.legal[c], int(rng.choice(np.flatnonzero(g.legal[c]))))
    bank.preds[0].update()
    bel = Belief(g, vis, bank.preds[0], bank.fb)
    info = LocalInfo(0, g, 4, 200, Fraction(1), 5, True, bel)
    own = cell(8, 8)
    from pursuit.env.channel import Packet
    pk = Packet(1, 31, cell(2, 2), cell(15, 15), 2, 0, 0, cell(15, 15), 3).pack()
    for t in range(2):
        info.update(LocalObs(t, own, vis[own], -1, ()), [])
    # packet at t=5 carrying a sighting from t=2 (delayed)
    for t in range(2, 5):
        info.update(LocalObs(t, own, vis[own], -1, ()), [])
    info.update(LocalObs(5, own, vis[own], -1, ()), [pk])
    # expected: point mass at step 2, propagate 3 steps with visible-empty conditioning
    b = bel.condition(np.zeros((bel.F, 5)), own, cell(15, 15), 0, True)  # packet flags carry a valid disp
    for k in range(3, 6):
        b = bel.propagate(b, info.history[k - 1])
        b = bel.condition(b, own, -1, 0, False)
    np.testing.assert_allclose(bel.b, b, atol=1e-12)
    assert info.target_age() == 3


def test_frozen_vs_online_predictor_buffers():
    from pursuit.evaluation.play import EvalPredictorHook
    cfg, g, bank, vis = mk_belief()
    f0 = bank.fingerprint()
    own = cell(8, 8)
    for online in (False, True):
        hook = EvalPredictorHook(bank, online)
        hook(0, own, cell(9, 8), 2, (), 2)
        fp = bank.fingerprint()
        assert (fp != f0) if online else (fp == f0)
        if online:
            bank.update_all()   # needs >=16 labels; no change yet is allowed
            for _ in range(20):
                hook(0, own, cell(9, 8), 2, (), 2)
            bank.update_all()
            assert bank.fingerprint()[0] != f0[0]
            assert bank.fingerprint()[1] == f0[1]   # other NPC IDs untouched: no label pooling


def test_q_player_update_and_masking():
    from pursuit.opponents.players import PlayerObs, QPlayer
    g = load_map("corridor")
    q = QPlayer(g, 200, learning=True)
    pos = cell(1, 1)
    po = PlayerObs(pos, 0, ())
    a = q.act(po, 0.5, 0.5)           # greedy (u_eps >= epsilon)
    assert g.legal[pos, a]
    k = q._last[0]
    q.learn(False, False, PlayerObs(cell(2, 1), 1, ()))
    assert q.Q[k, a] == pytest.approx(0.10 * (1 / 200))
    q.act(PlayerObs(cell(2, 1), 1, ()), 0.5, 0.5)
    q.learn(True, True, None)
    assert q.n_updates == 2
    frozen = q.clone(learning=False)
    before = frozen.Q.copy()
    frozen.act(po, 0.5, 0.5); frozen.learn(True, True, None)
    assert (frozen.Q == before).all()
