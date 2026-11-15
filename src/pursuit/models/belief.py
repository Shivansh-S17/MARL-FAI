"""Local belief over (free cell, previous displacement) of the player, one per NPC."""
from __future__ import annotations

import numpy as np

from ..env.geometry import NC, SIZE
from .predictor import FeatureBuilder, LocalPredictor


class Belief:
    def __init__(self, gm, vis: np.ndarray, predictor: LocalPredictor, fb: FeatureBuilder):
        self.gm, self.vis, self.pred, self.fb = gm, vis, predictor, fb
        self.F = gm.n_free
        self.fc = gm.free_cells
        self.x = gm.xs[self.fc].astype(np.float64)
        self.y = gm.ys[self.fc].astype(np.float64)
        self.logF = np.log(self.F)
        self.reset()

    def reset(self):
        self.b = np.full((self.F, 5), 1.0 / (5 * self.F))

    # ----------------------------------------------------------------- core
    def propagate(self, b, frame, eps=1e-9):
        own, _, tms = frame
        flat = b.reshape(-1)
        rows = np.flatnonzero(flat > eps)
        P = self.pred.belief_probs(self.fb, self.fc, self.gm.legal_free, own, tms, rows)  # [R,5]
        w = flat[rows][:, None] * P
        cells = rows // 5
        new = np.zeros_like(b)
        nf = self.gm.nxt_free
        for a in range(5):
            tgt = nf[cells, a]
            ok = tgt >= 0
            new[:, a] = np.bincount(tgt[ok], weights=w[ok, a], minlength=self.F)
        return new / new.sum()

    def condition(self, b, own, direct, disp, disp_valid):
        if direct >= 0:
            b = np.zeros_like(b)
            i = self.gm.free_index[direct]
            if disp_valid:
                b[i, disp] = 1.0
            else:
                b[i, :] = 0.2
            return b
        unseen = ~self.vis[own][self.fc]
        b = b * unseen[:, None]
        s = b.sum()
        if s < 1e-12:  # evidence eliminated all mass: reset uniform over feasible unseen cells
            b = np.zeros_like(b)
            b[unseen, :] = 1.0
            s = b.sum()
        return b / s

    def update(self, info, replay_from):
        t = info.t
        own, direct, _ = info.history[-1]
        if replay_from is not None and replay_from < t:
            self._replay(info, replay_from)
            return
        if t > 0:
            self.b = self.propagate(self.b, info.history[-2])
        self.b = self.condition(self.b, own, direct, info.mem_disp, info.mem_disp_valid and info.mem_step == t)

    def _replay(self, info, s):
        """Incorporate a delayed sighting at its own timestamp, then roll forward with the
        locally stored observation history (visible-empty evidence) to the present."""
        b = self.condition(np.zeros((self.F, 5)), info.history[s][0], info.mem_cell, info.mem_disp,
                           info.mem_disp_valid)
        for k in range(s + 1, info.t + 1):
            b = self.propagate(b, info.history[k - 1])
            own, direct, _ = info.history[k]
            b = self.condition(b, own, direct, 0, False)
        self.b = b

    # ------------------------------------------------------------ summaries
    def marginal(self):
        return self.b.sum(1)

    def marginal_full(self):
        m = np.zeros(NC)
        m[self.fc] = self.marginal()
        return m

    def features(self):
        """7 actor features: mean x,y, var x,y, cov xy, normalised entropy, max prob."""
        m = self.marginal()
        m = m / m.sum()
        mx, my = (m * self.x).sum(), (m * self.y).sum()
        vx, vy = (m * (self.x - mx) ** 2).sum(), (m * (self.y - my) ** 2).sum()
        cxy = (m * (self.x - mx) * (self.y - my)).sum()
        pos = m[m > 0]
        ent = -(pos * np.log(pos)).sum() / self.logF
        n = SIZE - 1
        return np.array([mx / n, my / n, vx / n ** 2, vy / n ** 2, cxy / n ** 2, ent, m.max()], dtype=np.float32)
