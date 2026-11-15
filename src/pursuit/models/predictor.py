"""Explicit local opponent predictor: next-displacement distribution of the player.

One independent predictor (network, optimiser, FIFO buffer) per NPC ID.
Feature layout (42): prev-disp one-hot(5) + validity(1) + target region one-hot(16)
+ own position relative to target(2) + 6 teammate slots x (rel x, rel y, visible)(18).
"""
from __future__ import annotations

import collections
import copy
import hashlib

import numpy as np
import torch
import torch.nn as nn

from ..env.geometry import NC, SIZE, GridMap

FEAT_DIM = 42
N_SLOTS = 6


class FeatureBuilder:
    def __init__(self, gm: GridMap):
        self.gm = gm
        self.region_oh = np.zeros((NC, 16), dtype=np.float32)
        self.region_oh[np.arange(NC), gm.region] = 1.0
        self.xs, self.ys = gm.xs.astype(np.float32), gm.ys.astype(np.float32)

    def cell_part(self, cells, own_cell, teammates):
        """Features depending on the (hypothetical) target cell(s): [n, 36]."""
        cells = np.asarray(cells)
        cx, cy = self.xs[cells], self.ys[cells]
        out = np.zeros((len(cells), 36), dtype=np.float32)
        out[:, :16] = self.region_oh[cells]
        out[:, 16] = (self.xs[own_cell] - cx) / (SIZE - 1)
        out[:, 17] = (self.ys[own_cell] - cy) / (SIZE - 1)
        for j, tc in teammates:
            o = 18 + 3 * j
            out[:, o] = (self.xs[tc] - cx) / (SIZE - 1)
            out[:, o + 1] = (self.ys[tc] - cy) / (SIZE - 1)
            out[:, o + 2] = 1.0
        return out

    @staticmethod
    def disp_part(prev_disp, valid=1.0):
        v = np.zeros(6, dtype=np.float32)
        v[prev_disp] = 1.0
        v[5] = valid
        return v

    def single(self, own_cell, target_cell, prev_disp, teammates):
        return np.concatenate([self.disp_part(prev_disp), self.cell_part([target_cell], own_cell, teammates)[0]])


class LocalPredictor:
    def __init__(self, gm: GridMap, cfg, seed: int):
        g = torch.Generator().manual_seed(int(seed))
        h = cfg.predictor.hidden
        self.cfg, self.gm = cfg, gm
        self.net = nn.Sequential(nn.Linear(FEAT_DIM, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 5))
        with torch.no_grad():  # init from the dedicated predictor RNG (never touches actor/critic init)
            for m in self.net:
                if isinstance(m, nn.Linear):
                    bound = 1.0 / np.sqrt(m.in_features)
                    m.weight.copy_((torch.rand(m.weight.shape, generator=g) * 2 - 1) * bound)
                    m.bias.copy_((torch.rand(m.bias.shape, generator=g) * 2 - 1) * bound)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.predictor.lr)
        self.buf = collections.deque(maxlen=cfg.predictor.buffer)
        self.rng = np.random.default_rng(int(seed) + 1)
        self.ready = False
        self.n_labels = 0
        self.n_updates = 0
        self.last_loss = float("nan")
        self.sync()

    def sync(self):
        self.W = [(m.weight.detach().numpy().T.copy(), m.bias.detach().numpy().copy())
                  for m in self.net if isinstance(m, nn.Linear)]

    # ------------------------------------------------------------ inference
    def _logits(self, x):
        (w1, b1), (w2, b2), (w3, b3) = self.W
        return np.maximum(np.maximum(x @ w1 + b1, 0) @ w2 + b2, 0) @ w3 + b3

    def probs(self, x, legal):
        """x [n,42], legal [n,5] -> masked probabilities (uniform over legal before training)."""
        if not self.ready:
            p = legal.astype(np.float64)
        else:
            z = np.where(legal, self._logits(x), -1e9)
            z = z - z.max(-1, keepdims=True)
            p = np.exp(z) * legal
        return p / p.sum(-1, keepdims=True)

    def belief_probs(self, fb: FeatureBuilder, free_cells, legal_free, own_cell, teammates, rows=None):
        """P(next disp | free cell, prev disp) for flattened rows r = cell_idx*5 + prev_disp
        (all rows if None): [R,5].  Rows are restricted to the belief support for speed."""
        if rows is None:
            rows = np.arange(len(free_cells) * 5)
        ci, di = rows // 5, rows % 5
        lg = legal_free[ci]
        if not self.ready:
            p = lg.astype(np.float32)
            return p / p.sum(-1, keepdims=True)
        (w1, b1), (w2, b2), (w3, b3) = self.W
        xc = fb.cell_part(np.asarray(free_cells)[ci], own_cell, teammates)
        xd = np.stack([fb.disp_part(d) for d in range(5)])
        h1 = np.maximum(xc @ w1[6:] + xd[di] @ w1[:6] + b1, 0)
        z = np.maximum(h1 @ w2 + b2, 0) @ w3 + b3
        z = np.where(lg, z, np.float32(-1e9))
        z = z - z.max(-1, keepdims=True)
        p = np.exp(z) * lg
        return p / p.sum(-1, keepdims=True)

    # ------------------------------------------------------------- training
    def add(self, feat, legal, label):
        self.buf.append((feat.astype(np.float32), legal.copy(), int(label)))
        self.n_labels += 1

    def update(self):
        c = self.cfg.predictor
        if len(self.buf) < c.min_labels:
            return None
        losses = []
        for _ in range(c.updates_per_rollout):
            n = min(c.batch, len(self.buf))
            idx = self.rng.choice(len(self.buf), n, replace=False)
            x = torch.from_numpy(np.stack([self.buf[i][0] for i in idx]))
            m = torch.from_numpy(np.stack([self.buf[i][1] for i in idx]))
            y = torch.tensor([self.buf[i][2] for i in idx])
            logits = self.net(x).masked_fill(~m, -1e9)
            loss = nn.functional.cross_entropy(logits, y)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            losses.append(loss.item())
        self.ready = True
        self.n_updates += 1
        self.last_loss = float(np.mean(losses))
        self.sync()
        return self.last_loss

    def evaluate(self, x, legal, y):
        p = self.probs(x, legal)
        ll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1)).mean()
        return float(ll), float((p.argmax(-1) == y).mean())

    # ---------------------------------------------------- state / freezing
    def state(self):
        return dict(net=copy.deepcopy(self.net.state_dict()), opt=copy.deepcopy(self.opt.state_dict()),
                    buf=list(self.buf), ready=self.ready, n_labels=self.n_labels, n_updates=self.n_updates,
                    rng=copy.deepcopy(self.rng.bit_generator.state))

    def load(self, st):
        self.net.load_state_dict(st["net"])
        self.opt.load_state_dict(st["opt"])
        self.buf = collections.deque(st["buf"], maxlen=self.cfg.predictor.buffer)
        self.ready, self.n_labels, self.n_updates = st["ready"], st["n_labels"], st["n_updates"]
        self.rng.bit_generator.state = st["rng"]
        self.sync()

    def fingerprint(self):
        h = hashlib.sha256()
        for p in self.net.parameters():
            h.update(p.detach().numpy().tobytes())
        for f, l, y in self.buf:
            h.update(f.tobytes())
            h.update(bytes([y]))
        return h.hexdigest()[:16]


class PredictorBank:
    """Independent predictor per NPC ID (labels are never pooled across IDs)."""

    def __init__(self, gm, cfg, seed):
        self.gm, self.cfg = gm, cfg
        self.fb = FeatureBuilder(gm)
        self.preds = [LocalPredictor(gm, cfg, seed * 100 + i) for i in range(N_SLOTS)]
        self.label_log = collections.defaultdict(int)
        self.frozen = False

    def label(self, agent_id, own_cell, target_cell, prev_disp, teammates, label):
        """Labeller hook called by LocalInfo for three-frame direct sightings."""
        if self.frozen:
            return
        feat = self.fb.single(own_cell, target_cell, prev_disp, teammates)
        self.preds[agent_id].add(feat, self.gm.legal[target_cell], label)
        self.label_log[agent_id] += 1

    def update_all(self):
        return [p.update() for p in self.preds]

    def state(self):
        return [p.state() for p in self.preds]

    def load(self, st):
        for p, s in zip(self.preds, st):
            p.load(s)

    def fingerprint(self):
        return [p.fingerprint() for p in self.preds]
