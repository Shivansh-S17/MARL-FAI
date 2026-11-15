"""Pilot learning gate (plan Section 6) and pilot figures, rebuilt from saved run logs only."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..config import ROOT
from .pilot import PILOT_CONDITIONS

PILOT = ROOT / "artifacts" / "results" / "pilot"


def load(tag, cond, method, seed):
    d = PILOT / f"pilot_{cond}_{method}_s{seed}_{tag}"
    if not (d / "val.json").exists():
        return None
    val = json.loads((d / "val.json").read_text())
    tl = json.loads((d / "train_log.json").read_text()) if (d / "train_log.json").exists() else []
    st = json.loads((d / "status.json").read_text()).get("status") if (d / "status.json").exists() else None
    return dict(val=val, log=tl, status=st)


def gate(tag="base", seeds=(9000, 9001), budgets=(300_000,), prev=200_000):
    """Return (per-condition table, overall pass) for candidate budget(s)."""
    out = {}
    for final in budgets:
        ref = {300_000: 200_000, 600_000: 300_000, 1_200_000: 600_000}.get(final, prev)
        rows, allpass = [], True
        for cond in PILOT_CONDITIONS:
            for method in ("marl", "marl_om"):
                fin, imp, chg, caps, finite = [], [], [], [], True
                missing = False
                for s in seeds:
                    r = load(tag, cond, method, s)
                    if r is None:
                        missing = True
                        continue
                    v = {x["steps"]: x for x in r["val"]}
                    if final not in v or ref not in v or 0 not in v:
                        missing = True
                        continue
                    fin.append(v[final]["capture"]); imp.append(v[final]["capture"] - v[0]["capture"])
                    chg.append(v[final]["capture"] - v[ref]["capture"]); caps.append(v[final]["train_captures"])
                    finite &= all(np.isfinite([l.get("kl", 0), l.get("vloss", 0), l.get("ent", 0)]).all() for l in r["log"])
                if missing or not fin:
                    rows.append(dict(cond=cond, method=method, status="incomplete"))
                    allpass = False
                    continue
                ok = dict(finite=finite, captures=min(caps) >= 100, level=float(np.median(fin)) >= 0.20,
                          improve=float(np.median(imp)) >= 0.10, stable=abs(float(np.median(chg))) <= 0.03)
                passed = all(ok.values())
                allpass &= passed
                rows.append(dict(cond=cond, method=method, final=float(np.median(fin)), improvement=float(np.median(imp)),
                                 change_vs_prev=float(np.median(chg)), min_train_captures=int(min(caps)),
                                 checks=ok, passed=passed))
        out[final] = dict(reference=ref, rows=rows, passed=allpass)
    return out


def figures(tag="base", seeds=(9000, 9001), out=None):
    out = Path(out or ROOT / "artifacts" / "figures")
    out.mkdir(parents=True, exist_ok=True)
    conds = list(PILOT_CONDITIONS)
    fig, axs = plt.subplots(3, len(conds), figsize=(4 * len(conds), 9), sharex="col")
    col = {"marl": "#0072B2", "marl_om": "#D55E00"}
    for j, cond in enumerate(conds):
        for method in ("marl", "marl_om"):
            for s in seeds:
                r = load(tag, cond, method, s)
                if not r or not r["val"]:
                    continue
                x = [v["steps"] for v in r["val"]]
                ls = "-" if s == seeds[0] else "--"
                axs[0, j].plot(x, [v["capture"] for v in r["val"]], ls, marker="o", color=col[method], label=f"{method} s{s}")
                axs[1, j].plot(x, [v["capped_duration"] for v in r["val"]], ls, marker="o", color=col[method])
                if r["log"]:
                    axs[2, j].plot([l["steps"] for l in r["log"]], [l["captures"] for l in r["log"]], ls, color=col[method])
        axs[0, j].set_title(cond, fontsize=10)
        axs[0, j].axhline(0.2, color="gray", lw=0.7, ls=":")
        axs[0, j].set_ylim(0, 1.02)
    axs[0, 0].set_ylabel("validation capture rate (100 eps)")
    axs[1, 0].set_ylabel("capped duration (steps)")
    axs[2, 0].set_ylabel("cumulative training captures")
    for ax in axs[2]:
        ax.set_xlabel("environment transitions")
    axs[0, 0].legend(fontsize=7)
    fig.suptitle(f"Pilot learning curves ({tag} reward)")
    fig.tight_layout()
    fig.savefig(out / f"pilot_curves_{tag}.png", dpi=140)
    fig.savefig(out / f"pilot_curves_{tag}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "base"
    budgets = tuple(int(x) for x in sys.argv[2].split(",")) if len(sys.argv) > 2 else (300_000,)
    g = gate(tag, budgets=budgets)
    print(json.dumps(g, indent=1, default=str))
    figures(tag)
