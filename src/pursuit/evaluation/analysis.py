"""Statistics (hierarchical bootstrap, randomisation test) and figures. Rebuilds everything from saved CSVs."""
from __future__ import annotations

import itertools
import json
from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import ROOT

RES = ROOT / "artifacts" / "results"
FIG = ROOT / "artifacts" / "figures"
BUDGETS = ["0", "1/8", "1/2", "1"]
BVAL = [Fraction(b) for b in BUDGETS]
BOOT_SEED, NBOOT = 40000, 5000
MCOL = {"marl": "#0072B2", "marl_om": "#D55E00", "IP": "#666666", "RC": "#009E73"}


# --------------------------------------------------------------------------- metrics
def crossings(p0, p1):
    """p0,p1: [..., 4] capture by budget (ascending). Returns S(a), min-B for each, NaN if undefined."""
    a = 0.9 * p0[..., 3]
    def minb(p):
        ok = p >= a[..., None] - 1e-12
        first = np.argmax(ok, axis=-1)
        has = ok.any(-1)
        val = np.array([float(b) for b in BVAL])[first]
        return np.where(has, val, np.nan)
    m0, m1 = minb(p0), minb(p1)
    return m0 - m1, m0, m1


def metrics_from_capture(p0, p1):
    """p0,p1 [..., 4]. Returns dict of arrays: G [...,4], F [...,3] (Blo = 0,1/8,1/2), S, comp."""
    G = p1 - p0
    den = p0[..., 3:4] - p0[..., :3]
    with np.errstate(divide="ignore", invalid="ignore"):
        F = np.where(den > 0, G[..., :3] / den, np.nan)
    S, m0, m1 = crossings(p0, p1)
    comp = p1[..., 1] - p0[..., 3]
    return dict(G=G, F=F, S=S, minB0=m0, minB1=m1, comp=comp)


def bootstrap(C0, C1, strata=None, nboot=NBOOT, seed=BOOT_SEED):
    """C0,C1: [4(B), S(seeds), E(episodes)] binary capture. Resample seed groups, then matched scenario IDs
    within group (within strata when given). Returns point and percentile CIs."""
    rng = np.random.default_rng(seed)
    nb, S, E = C0.shape
    point = metrics_from_capture(C0.mean((1, 2)), C1.mean((1, 2)))
    sidx = rng.integers(0, S, size=(nboot, S))
    if strata is None:
        eidx = rng.integers(0, E, size=(nboot, S, E))
    else:
        eidx = np.empty((nboot, S, E), dtype=int)
        for st in np.unique(strata):
            cols = np.flatnonzero(strata == st)
            eidx[:, :, cols] = cols[rng.integers(0, len(cols), size=(nboot, S, len(cols)))]
    # gather -> [4, R, S, E]
    g0 = C0[:, sidx[:, :, None], eidx]
    g1 = C1[:, sidx[:, :, None], eidx]
    p0, p1 = np.moveaxis(g0.mean((2, 3)), 0, -1), np.moveaxis(g1.mean((2, 3)), 0, -1)  # [R,4]
    # NOTE g.mean over (S,E) = mean over seeds of mean over scenarios (equal group sizes)
    m = metrics_from_capture(p0, p1)
    out = {}
    for k, v in m.items():
        lo, hi = np.nanpercentile(v, [2.5, 97.5], axis=0) if np.isfinite(v).any() else (np.nan, np.nan)
        out[k] = dict(point=point[k], lo=lo, hi=hi, undefined_frac=np.isnan(v).mean(axis=0))
    out["comp_lb95"] = float(np.nanpercentile(m["comp"], 5))
    out["p0"] = dict(point=C0.mean((1, 2)), lo=np.percentile(p0, 2.5, axis=0), hi=np.percentile(p0, 97.5, axis=0))
    out["p1"] = dict(point=C1.mean((1, 2)), lo=np.percentile(p1, 2.5, axis=0), hi=np.percentile(p1, 97.5, axis=0))
    return out


def sign_flip_p(d):
    """Two-sided paired randomisation p-value by full sign-flip enumeration of seed-level differences."""
    d = np.asarray(d, float)
    obs = abs(d.mean())
    flips = np.array(list(itertools.product([1, -1], repeat=len(d))))
    stat = np.abs((flips * d).mean(1))
    return float((stat >= obs - 1e-12).mean())


def holm(ps, alpha=0.05):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj


# --------------------------------------------------------------------------- loading
def load_frozen_tensor(df, player):
    """DataFrame for one player/xi -> (C0, C1, extras) with C[B, seed, episode] aligned by scenario_id."""
    d = df[df.player == player]
    out = {}
    for method in ("marl", "marl_om"):
        dm = d[d.method == method]
        arrs = []
        seeds = sorted(dm.seed.unique())
        for b in BUDGETS:
            row = []
            for s in seeds:
                x = dm[(dm.B.astype(str) == b) & (dm.seed == s)].sort_values(["block", "episode_in_block"] if "block" in dm else "scenario_id")
                row.append(x.captured.to_numpy())
            arrs.append(row)
        out[method] = (np.array(arrs, dtype=float) if all(len(r) == len(arrs[0][0]) for a in arrs for r in a) else None, seeds)
    return out


def complete_cells(df, player, metric="captured"):
    """Keep only seeds for which both methods and all budgets are present."""
    d = df[df.player == player]
    seeds = None
    for method in ("marl", "marl_om"):
        for b in BUDGETS:
            s = set(d[(d.method == method) & (d.B.astype(str) == b)].seed.unique())
            seeds = s if seeds is None else seeds & s
    return sorted(seeds or [])


def tensor(df, player, seeds, method, metric="captured"):
    d = df[(df.player == player) & (df.method == method)]
    sortk = ["block", "episode_in_block"] if "block" in d.columns and d.block.notna().any() else ["scenario_id"]
    return np.array([[d[(d.B.astype(str) == b) & (d.seed == s)].sort_values(sortk)[metric].to_numpy(float)
                      for s in seeds] for b in BUDGETS])


def load_all(pattern):
    frames = []
    for sub in ("main", "secondary"):
        for f in sorted((RES / sub).glob(f"*/{pattern}")) if (RES / sub).exists() else []:
            frames.append(pd.read_csv(f))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------------- analysis
def analyse_frozen(df, label_players, out_tables, tag="waypoint"):
    rows, hyp = [], []
    results = {}
    for player in label_players:
        seeds = complete_cells(df, player)
        if len(seeds) < 2:
            rows.append(dict(player=player, note=f"incomplete: complete seeds={seeds}"))
            continue
        C0, C1 = tensor(df, player, seeds, "marl"), tensor(df, player, seeds, "marl_om")
        strata = (np.arange(C0.shape[2]) // 20) if player == "qbank" else None
        res = bootstrap(C0, C1, strata)
        results[player] = (res, seeds, C0, C1)
        for bi, b in enumerate(BUDGETS):
            rows.append(dict(player=player, B=b, n_seeds=len(seeds), p_marl=res["p0"]["point"][bi],
                             p_marl_lo=res["p0"]["lo"][bi], p_marl_hi=res["p0"]["hi"][bi],
                             p_om=res["p1"]["point"][bi], p_om_lo=res["p1"]["lo"][bi], p_om_hi=res["p1"]["hi"][bi],
                             G=res["G"]["point"][bi], G_lo=res["G"]["lo"][bi], G_hi=res["G"]["hi"][bi],
                             Fgap=res["F"]["point"][bi] if bi < 3 else np.nan,
                             Fgap_lo=res["F"]["lo"][bi] if bi < 3 else np.nan,
                             Fgap_hi=res["F"]["hi"][bi] if bi < 3 else np.nan,
                             Fgap_undefined_frac=res["F"]["undefined_frac"][bi] if bi < 3 else np.nan))
        rows.append(dict(player=player, B="summary", S_point=res["S"]["point"], S_lo=res["S"]["lo"],
                         S_hi=res["S"]["hi"], S_undefined_frac=res["S"]["undefined_frac"],
                         comp_lb95=res["comp_lb95"], comp_point=res["comp"]["point"]))
        d = C1[1].mean(-1) - C0[1].mean(-1)          # seed-level G(1/8)
        hyp.append(dict(player=player, G_1o8=float(d.mean()), p_raw=sign_flip_p(d), seed_diffs=list(map(float, d))))
    if hyp:
        adj = holm([h["p_raw"] for h in hyp])
        for h, a in zip(hyp, adj):
            h["p_holm"] = float(a)
            h["min_possible_p"] = 2.0 / 2 ** len(h["seed_diffs"])
    pd.DataFrame(rows).to_csv(out_tables / f"frozen_summary_{tag}.csv", index=False)
    (out_tables / f"primary_hypotheses_{tag}.json").write_text(json.dumps(hyp, indent=1))
    return results


def plot_capture_vs_B(results, base_df, out, tag="waypoint"):
    n = len(results)
    if n == 0:
        return
    fig, axs = plt.subplots(1, n, figsize=(4.2 * n, 3.8), sharey=True, squeeze=False)
    x = np.arange(4)
    for ax, (player, (res, seeds, C0, C1)) in zip(axs[0], results.items()):
        for key, C, col, off in (("p0", C0, MCOL["marl"], -0.07), ("p1", C1, MCOL["marl_om"], 0.07)):
            ax.fill_between(x + off, res[key]["lo"], res[key]["hi"], color=col, alpha=0.18)
            ax.plot(x + off, res[key]["point"], "-o", color=col, label="MARL" if key == "p0" else "MARL-OM (frozen)")
            ax.plot(np.repeat(x + off, C.shape[1]).reshape(4, -1) .T, C.mean(-1).T, ".", color=col, alpha=0.5, ms=5)
        if base_df is not None and len(base_df):
            for rule, ls in (("RC", "--"), ("IP", ":")):
                b = base_df[(base_df.method == rule) & (base_df.player == player)]
                if len(b):
                    if rule == "IP":
                        ax.axhline(b.captured.mean(), color=MCOL[rule], ls=ls, lw=1, label="IP")
                    else:
                        y = [b[b.B.astype(str) == bb].captured.mean() for bb in BUDGETS]
                        ax.plot(x, y, ls, color=MCOL[rule], marker="s", ms=4, lw=1, label="RC")
        ax.set_xticks(x); ax.set_xticklabels(BUDGETS)
        ax.set_xlabel("communication budget B (packets / NPC / step)")
        ax.set_title(player, fontsize=10)
        ax.set_ylim(-0.02, 1.03)
    axs[0, 0].set_ylabel("capture rate")
    axs[0, 0].legend(fontsize=7, loc="lower right")
    fig.suptitle("Capture vs communication budget (dots: individual training seeds; bands: 95% bootstrap)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / f"capture_vs_budget_{tag}.png", dpi=150); fig.savefig(out / f"capture_vs_budget_{tag}.pdf")
    plt.close(fig)


def plot_packets(df, out):
    if df.empty:
        return
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    for method, col in (("marl", MCOL["marl"]), ("marl_om", MCOL["marl_om"])):
        d = df[df.method == method].groupby(["player", "B", "seed"]).agg(pk=("packets", "mean"), cap=("captured", "mean"),
                                                                       dur=("capped_duration", "mean")).reset_index()
        axs[0].scatter(d.pk, d.cap, s=12, color=col, alpha=0.6, label=method)
        axs[1].scatter(d.pk, d.dur, s=12, color=col, alpha=0.6)
    axs[0].set_xlabel("actual packets per episode"); axs[0].set_ylabel("capture rate")
    axs[1].set_xlabel("actual packets per episode"); axs[1].set_ylabel("capped duration (steps)")
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "packets_vs_capture.png", dpi=150); fig.savefig(out / "packets_vs_capture.pdf")
    plt.close(fig)


def analyse_adaptive(df, out_tables, out_fig):
    if df.empty:
        return
    g = df.groupby(["method", "protocol", "seed", "episode_in_block"]).agg(
        survival=("length", "mean"), capture=("captured", "mean"), qnorm=("q_change_norm", "mean"),
        greedy=("greedy_changes", "mean"), plogloss=("pred_logloss", "mean") if "pred_logloss" in df else ("length", "mean")).reset_index()
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.4))
    styles = {("marl", "off_frozen"): ("MARL, Q off", MCOL["marl"], ":"), ("marl_om", "off_frozen"): ("OM frozen, Q off", MCOL["marl_om"], ":"),
              ("marl", "on_frozen"): ("MARL, Q on", MCOL["marl"], "-"), ("marl_om", "on_frozen"): ("OM frozen, Q on", MCOL["marl_om"], "-"),
              ("marl_om", "on_online"): ("OM online, Q on", "#CC79A7", "-")}
    for (m, p), (lab, col, ls) in styles.items():
        s = g[(g.method == m) & (g.protocol == p)]
        if s.empty:
            continue
        for ax, key in zip(axs, ("capture", "survival", "qnorm", "greedy")):
            y = s.groupby("episode_in_block")[key].mean()
            ax.plot(y.index + 1, y.values, ls, color=col, label=lab)
    for ax, t in zip(axs, ("capture rate", "player survival (steps)", "Q change norm", "greedy-action changes on probe set")):
        ax.set_xlabel("episode index within block"); ax.set_title(t, fontsize=9)
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_fig / "adaptive_curves.png", dpi=150); fig.savefig(out_fig / "adaptive_curves.pdf")
    plt.close(fig)
    # first-five versus last-five (exploratory)
    rows = []
    for (m, p), s in df.groupby(["method", "protocol"]):
        for tag, lo, hi in (("first5", 0, 5), ("last5", 15, 20)):
            x = s[(s.episode_in_block >= lo) & (s.episode_in_block < hi)]
            per_seed = x.groupby("seed").agg(capture=("captured", "mean"), survival=("length", "mean"))
            rows.append(dict(method=m, protocol=p, window=tag, capture=per_seed.capture.mean(),
                             survival=per_seed.survival.mean(), n_seeds=len(per_seed)))
    pd.DataFrame(rows).to_csv(out_tables / "adaptive_first_last.csv", index=False)


def run_analysis(a, cfg):
    out_t = Path(a.output) if a.output else RES / "tables"
    out_t.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    base_p = RES / "eval" / "baselines.csv"
    base = pd.read_csv(base_p) if base_p.exists() else None
    fw = load_all("eval_frozen_waypoint.csv")
    if not fw.empty:
        res = analyse_frozen(fw, sorted(fw.player.unique()), out_t)
        bwp = base[base.protocol == "frozen_waypoint"] if base is not None else None
        plot_capture_vs_B(res, bwp, FIG)
        plot_packets(fw, FIG)
    pc = load_all("eval_player_cells.csv")
    if not pc.empty:
        off = pc[pc.protocol == "off_frozen"].copy()
        res = analyse_frozen(off.assign(player="qbank"), ["qbank"], out_t, "player") if not off.empty else {}
        bq = base[base.protocol == "off_frozen"].assign(player="qbank") if base is not None else None
        plot_capture_vs_B(res, bq, FIG, "player")
        analyse_adaptive(pc, out_t, FIG)
    print("analysis written to", out_t, "and", FIG)
