"""Recurrent MAPPO trainer (centralised critic, shared GRU actor, decentralised execution)."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from ..env.geometry import load_map
from ..env.observation import CRITIC_DIM, OFF_PRED, VEC_DIM, action_mask, build_critic, build_obs
from ..env.scenario import sample_for_cfg, sample_scenario
from ..env.team import Team, make_core, make_player
from ..models.belief import Belief
from ..models.networks import make_networks
from ..models.predictor import PredictorBank
from ..opponents.players import QPlayer, WaypointPlayer


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def gae(rew, val, term, last_val, gamma, lam):
    """rew,val,term: [T,E]; last_val: [E]. Terminal (capture / horizon) => zero bootstrap;
    a rollout boundary inside an episode bootstraps from V(s_T)."""
    T = rew.shape[0]
    adv = np.zeros_like(rew)
    nxt_v, last = last_val, np.zeros_like(last_val)
    for t in reversed(range(T)):
        nt = 1.0 - term[t].astype(np.float32)
        delta = rew[t] + gamma * nxt_v * nt - val[t]
        last = delta + gamma * lam * nt * last
        adv[t] = last
        nxt_v = val[t]
    return adv, adv + val


def curriculum_range(cfg, steps):
    for lo_s, hi_s, dlo, dhi in cfg.train.curriculum:
        if lo_s <= steps < hi_s:
            return (dlo, dhi)
    return None


class Trainer:
    """method: 'marl' | 'marl_om'; opponent: dict(type='waypoint', xi=..) or dict(type='qbank')."""

    def __init__(self, cfg, method: str, seed: int, opponent: dict, budget, qbank=None, device="cpu",
                 shuffle_features=False, out_dir=None):
        self.cfg, self.method, self.seed, self.opponent = cfg, method, seed, opponent
        self.budget, self.qbank = budget, qbank
        self.device = torch.device(device)
        self.shuffle_features = shuffle_features
        self.out_dir = Path(out_dir) if out_dir else None
        pc = cfg.ppo
        self.E, self.T, self.N = pc.n_envs, pc.rollout_len, cfg.env.n_npcs
        self.gm = load_map(cfg.env.map)
        self.actor, self.critic = make_networks(seed)  # identical for MARL / MARL-OM at equal seed
        import copy
        self.actor_act, self.critic_act = copy.deepcopy(self.actor), copy.deepcopy(self.critic)  # CPU acting copies
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.opt = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=pc.lr)
        self.om = method == "marl_om"
        self.bank = PredictorBank(self.gm, cfg, seed) if self.om else None  # separate RNG
        self.gen = torch.Generator().manual_seed(int(seed) * 7919 + 1)
        ss = np.random.SeedSequence([int(seed), 11])
        kids = ss.spawn(self.E * 2)
        self.place_rng = [np.random.default_rng(kids[2 * e]) for e in range(self.E)]
        self.qsel_rng = [np.random.default_rng(kids[2 * e + 1]) for e in range(self.E)]
        self.steps = 0          # environment transitions (one simultaneous step in one env)
        self.rollouts = 0
        self.log = []
        self.episodes = []
        self.t_start = time.time()
        self.budget_f = float(budget)
        self.envs = []
        for e in range(self.E):
            core = make_core(cfg, player=self._new_player(e), budget=budget)
            beliefs = None
            if self.om:
                beliefs = [Belief(self.gm, core.vis, self.bank.preds[i], self.bank.fb) for i in range(self.N)]
            team = Team(core, True, beliefs, self.bank.label if self.om else None)
            self.envs.append((core, team))
        self.h = torch.zeros(self.E, self.N, 64)
        self.cur = None
        self.ep_ret = np.zeros(self.E)
        self.ep_shaped = np.zeros(self.E)
        self.stats = dict(captures=0, episodes=0, labels=0)
        for e in range(self.E):
            self._reset_env(e)
        self._observe_all()

    # --------------------------------------------------------------- envs
    def _new_player(self, e):
        if self.opponent["type"] == "waypoint":
            return WaypointPlayer(self.gm, self.opponent["xi"])
        q = self.qbank[int(self.qsel_rng[e].integers(len(self.qbank)))]
        return q.clone(learning=False)

    def _reset_env(self, e):
        core, team = self.envs[e]
        if self.opponent["type"] == "qbank":
            core.set_player(self._new_player(e))
        sc = sample_for_cfg(self.cfg, self.gm, self.place_rng[e], curriculum_range(self.cfg, self.steps))
        core.reset(sc)
        team.reset()
        self.ep_ret[e] = 0.0
        self.ep_shaped[e] = 0.0

    def _observe_all(self):
        maps = np.zeros((self.E, self.N, 5, 20, 20), np.uint8)
        vec = np.zeros((self.E, self.N, VEC_DIM), np.float32)
        mask = np.zeros((self.E, self.N, 10), bool)
        cs = np.zeros((self.E, CRITIC_DIM), np.float32)
        for e, (core, team) in enumerate(self.envs):
            team.begin_step()
            for i, inf in enumerate(team.infos):
                f7 = inf.belief.features() if inf.belief is not None else None
                maps[e, i], vec[e, i] = build_obs(inf, self.budget_f, f7)
                mask[e, i] = action_mask(inf)
            cs[e] = build_critic(core, core.player, self.budget_f)
        if self.shuffle_features:  # diagnostic control: break the link between feature and agent state
            flat = vec.reshape(-1, VEC_DIM)
            perm = self.gen_perm(flat.shape[0])
            flat[:, OFF_PRED:OFF_PRED + 7] = flat[perm, OFF_PRED:OFF_PRED + 7]
        self.cur = dict(maps=maps, vec=vec, mask=mask, cs=cs)

    def gen_perm(self, n):
        return torch.randperm(n, generator=self.gen).numpy()

    # ------------------------------------------------------------ rollout
    @torch.no_grad()
    def collect(self, T):
        E, N, dev = self.E, self.N, torch.device("cpu")
        self._sync_acting()
        B = dict(maps=np.zeros((T, E, N, 5, 20, 20), np.uint8), vec=np.zeros((T, E, N, VEC_DIM), np.float32),
                 mask=np.zeros((T, E, N, 10), bool), cs=np.zeros((T, E, CRITIC_DIM), np.float32),
                 act=np.zeros((T, E, N), np.int64), logp=np.zeros((T, E, N), np.float32),
                 val=np.zeros((T, E), np.float32), rew=np.zeros((T, E), np.float32),
                 term=np.zeros((T, E), bool), first=np.zeros((T, E), bool),
                 h=np.zeros((T, E, N, 64), np.float32), base=np.zeros((T, E), np.float32))
        first = np.zeros(E, bool)
        caps0 = self.stats["captures"]
        for t in range(T):
            cur = self.cur
            for k in ("maps", "vec", "mask", "cs"):
                B[k][t] = cur[k]
            B["first"][t] = first
            B["h"][t] = self.h.cpu().numpy()
            maps = torch.from_numpy(cur["maps"]).to(dev).view(E * N, 5, 20, 20)
            vec = torch.from_numpy(cur["vec"]).to(dev).view(E * N, -1)
            mask = torch.from_numpy(cur["mask"]).to(dev).view(E * N, 10)
            logits, h = self.actor_act.step(maps, vec, self.h.view(E * N, -1), mask)
            dist = Categorical(logits=logits)
            a = torch.multinomial(dist.probs.cpu(), 1, generator=self.gen).squeeze(-1).to(dev)
            logp = dist.log_prob(a)
            B["val"][t] = self.critic_act(torch.from_numpy(cur["cs"]).to(dev)).cpu().numpy()
            B["act"][t] = a.view(E, N).cpu().numpy()
            B["logp"][t] = logp.view(E, N).cpu().numpy()
            self.h = h.view(E, N, -1)
            acts = B["act"][t]
            first[:] = False
            for e, (core, team) in enumerate(self.envs):
                moves = (acts[e] // 2).tolist()
                sends = (acts[e] % 2).astype(bool).tolist()
                res = team.commit_and_step(moves, sends)
                B["rew"][t, e] = res.reward
                B["base"][t, e] = res.base_reward
                self.ep_ret[e] += res.base_reward
                self.ep_shaped[e] += res.reward
                self.steps += 1
                if res.done:
                    B["term"][t, e] = True
                    self.stats["episodes"] += 1
                    self.stats["captures"] += int(res.captured)
                    self.episodes.append(dict(step=self.steps, captured=bool(res.captured), length=core.t,
                                              base_return=float(self.ep_ret[e]),
                                              shaped_return=float(self.ep_shaped[e]),
                                              packets=core.channel.total_sent))
                    self._reset_env(e)
                    self.h[e] = 0.0
                    first[e] = True
            self._observe_all()
        # bootstrap value of the state after the last transition
        last_val = self.critic_act(torch.from_numpy(self.cur["cs"]).to(dev)).cpu().numpy()
        return B, last_val

    def _sync_acting(self):
        self.actor_act.load_state_dict({k: v.cpu() for k, v in self.actor.state_dict().items()})
        self.critic_act.load_state_dict({k: v.cpu() for k, v in self.critic.state_dict().items()})

    # -------------------------------------------------------------- update
    def update(self, B, last_val):
        pc = self.cfg.ppo
        T = B["rew"].shape[0]
        adv, ret = gae(B["rew"], B["val"], B["term"], last_val, pc.gamma, pc.gae_lambda)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)  # all T*E team samples are valid
        dev, L = self.device, pc.seq_len
        tens = {k: torch.from_numpy(B[k]).to(dev) for k in ("maps", "vec", "mask", "cs", "act", "logp", "first", "h")}
        adv_t, ret_t = torch.from_numpy(adv).to(dev), torch.from_numpy(ret).to(dev)
        n_chunks = (T + L - 1) // L
        groups = [(e, c) for e in range(self.E) for c in range(n_chunks)]
        mb = max(1, len(groups) // pc.minibatches)
        info = dict(kl=[], ent=[], vloss=[], ploss=[], gnorm=[], clipfrac=[])
        rng = np.random.default_rng(self.seed * 31 + self.rollouts)
        for _ in range(pc.epochs):
            order = rng.permutation(len(groups))
            for s in range(0, len(groups), mb):
                sel = [groups[i] for i in order[s:s + mb]]
                self._ppo_minibatch(sel, tens, adv_t, ret_t, T, info)
        self.rollouts += 1
        out = {k: float(np.mean(v)) for k, v in info.items()}
        v_old = B["val"].reshape(-1)
        out["explained_var"] = float(1 - np.var(ret.reshape(-1) - v_old) / (np.var(ret.reshape(-1)) + 1e-8))
        return out

    def _ppo_minibatch(self, sel, tens, adv_t, ret_t, T, info):
        pc, dev, L, N = self.cfg.ppo, self.device, self.cfg.ppo.seq_len, self.N
        idx = torch.full((len(sel), L), -1, dtype=torch.long)
        envs = torch.tensor([e for e, _ in sel])
        for g, (e, c) in enumerate(sel):
            n = min(L, T - c * L)
            idx[g, :n] = torch.arange(c * L, c * L + n)
        valid = (idx >= 0).to(dev)                   # [G,L]
        ic = idx.clamp(min=0)
        G = len(sel)
        gi = envs.unsqueeze(1).expand(G, L)

        def gather(x):  # x [T,E,...] -> [G,L,...]
            return x[ic.to(dev), gi.to(dev)]
        maps, vec, mask = gather(tens["maps"]), gather(tens["vec"]), gather(tens["mask"])
        act, oldlp, h_all = gather(tens["act"]), gather(tens["logp"]), gather(tens["h"])
        first = gather(tens["first"])
        cs = gather(tens["cs"])
        adv, ret = gather(adv_t), gather(ret_t)
        h0 = h_all[:, 0]                                                       # [G,N,H]
        BN = G * N
        tr = lambda x: x.transpose(1, 2).reshape(BN, L, *x.shape[3:])         # [G,L,N,..] -> [G*N,L,..]
        logits = self.actor.sequence(tr(maps), tr(vec), h0.reshape(BN, -1),
                                     first.unsqueeze(2).expand(G, L, N).transpose(1, 2).reshape(BN, L), tr(mask))
        dist = Categorical(logits=logits)
        newlp = dist.log_prob(tr(act))
        oldlp = tr(oldlp)
        vm = valid.unsqueeze(1).expand(G, N, L).reshape(BN, L).float()
        A = adv.unsqueeze(2).expand(G, L, N).transpose(1, 2).reshape(BN, L)
        ratio = (newlp - oldlp).exp()
        s1, s2 = ratio * A, ratio.clamp(1 - pc.clip, 1 + pc.clip) * A
        ploss = -(torch.min(s1, s2) * vm).sum() / vm.sum()
        ent = (dist.entropy() * vm).sum() / vm.sum()
        v = self.critic(cs)
        vf = valid.float()
        vloss = (((v - ret) ** 2) * vf).sum() / vf.sum()
        loss = ploss + pc.value_coef * vloss - pc.entropy_coef * ent
        self.opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()),
                                            pc.max_grad_norm)
        self.opt.step()
        with torch.no_grad():
            info["kl"].append(((oldlp - newlp) * vm).sum().item() / vm.sum().item())
            info["ent"].append(ent.item()); info["vloss"].append(vloss.item()); info["ploss"].append(ploss.item())
            info["gnorm"].append(float(gn))
            info["clipfrac"].append((((ratio - 1).abs() > pc.clip).float() * vm).sum().item() / vm.sum().item())

    # ---------------------------------------------------------------- run
    def train(self, total_steps, log_every=10, ckpt_cb=None, eval_cb=None, eval_at=(), ckpt_every=None, verbose=True):
        per_roll = self.E
        next_eval = sorted(eval_at)
        while self.steps < total_steps:
            T = min(self.T, (total_steps - self.steps) // per_roll)
            if T <= 0:
                break
            t0 = time.time()
            B, lv = self.collect(T)
            t1 = time.time()
            info = self.update(B, lv)
            t2 = time.time()
            pl = {}
            if self.om:
                t3 = time.time()
                losses = self.bank.update_all()
                pl = dict(pred_loss=float(np.nanmean([l for l in losses if l is not None] or [np.nan])),
                          labels=int(sum(self.bank.label_log.values())), pred_time=time.time() - t3)
            rec = dict(steps=self.steps, rollout=self.rollouts, collect_s=t1 - t0, update_s=t2 - t1,
                       elapsed=time.time() - self.t_start, captures=self.stats["captures"],
                       episodes=self.stats["episodes"], sps=T * per_roll / (time.time() - t0), **info, **pl)
            self.log.append(rec)
            if verbose and self.rollouts % log_every == 0:
                print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
            if eval_cb and next_eval and self.steps >= next_eval[0]:
                while next_eval and self.steps >= next_eval[0]:
                    next_eval.pop(0)
                eval_cb(self)
            if ckpt_cb and ckpt_every and (self.steps // ckpt_every) != ((self.steps - T * per_roll) // ckpt_every):
                ckpt_cb(self)
        return self.log

    # --------------------------------------------------------- checkpoint
    def state(self):
        return dict(actor=self.actor.state_dict(), critic=self.critic.state_dict(), opt=self.opt.state_dict(),
                    steps=self.steps, rollouts=self.rollouts, stats=self.stats, method=self.method,
                    seed=self.seed, opponent=self.opponent, budget=str(self.budget),
                    bank=self.bank.state() if self.om else None, gen=self.gen.get_state(),
                    place=[r.bit_generator.state for r in self.place_rng],
                    qsel=[r.bit_generator.state for r in self.qsel_rng],
                    log=self.log, episodes=self.episodes,
                    resume_note="environments resume at a fresh episode boundary; placement RNG streams restored")

    def save(self, path):
        torch.save(self.state(), path)

    def load(self, path):
        st = torch.load(path, weights_only=False)
        self.actor.load_state_dict(st["actor"]); self.critic.load_state_dict(st["critic"])
        self.opt.load_state_dict(st["opt"])
        self.steps, self.rollouts, self.stats = st["steps"], st["rollouts"], st["stats"]
        if self.om:
            self.bank.load(st["bank"])
        self.gen.set_state(st["gen"])
        for r, s in zip(self.place_rng, st["place"]):
            r.bit_generator.state = s
        for r, s in zip(self.qsel_rng, st["qsel"]):
            r.bit_generator.state = s
        self.log, self.episodes = st["log"], st["episodes"]
        self.h.zero_()
        for e in range(self.E):
            self._reset_env(e)
        self._observe_all()
