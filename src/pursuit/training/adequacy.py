"""Revised training-adequacy report (plan 6.2): per condition/method final-checkpoint learning checks,
with checkpoint changes reported (400k->600k, 800k->1.2M) rather than gated by a plateau tolerance."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from ..config import ROOT
from .pilot import PILOT_CONDITIONS

REQUIRED = ("wp0_B1", "wp75_B1o8", "wp25_B0")


def report(tag="base", seeds=(9000, 9001), final=1_200_000, root=None):
    root = Path(root or ROOT / "artifacts" / "results" / "pilot")
    rows = []
    for cond in PILOT_CONDITIONS:
        for method in ("marl", "marl_om"):
            per = []
            for s in seeds:
                d = root / f"pilot_{cond}_{method}_s{s}_{tag}"
                if not (d / "val.json").exists():
                    continue
                v = {x["steps"]: x for x in json.loads((d / "val.json").read_text())}
                log = json.loads((d / "train_log.json").read_text()) if (d / "train_log.json").exists() else []
                fin = v.get(final)
                if fin is None:
                    per.append(dict(seed=s, missing=True))
                    continue
                g = lambda k: v[k]["capture"] if k in v else None
                per.append(dict(seed=s, final=fin["capture"], untrained=v[0]["capture"], c400=g(400_000), c600=g(600_000),
                                c800=g(800_000), train_captures=fin["train_captures"],
                                finite=all(np.isfinite([l.get("kl", 0), l.get("vloss", 0), l.get("ent", 0)]).all() for l in log)))
            ok = [p for p in per if not p.get("missing")]
            if not ok:
                rows.append(dict(cond=cond, method=method, status="no data"))
                continue
            fm = float(np.median([p["final"] for p in ok]))
            imp = float(np.median([p["final"] - p["untrained"] for p in ok]))
            ch1 = [p["c600"] - p["c400"] for p in ok if p["c400"] is not None and p["c600"] is not None]
            ch2 = [p["final"] - p["c800"] for p in ok if p["c800"] is not None]
            checks = dict(finite=all(p["finite"] for p in ok), captures=min(p["train_captures"] for p in ok) >= 100,
                          level=fm >= 0.20, improve=imp >= 0.10,
                          no_collapse=(float(np.median(ch2)) if ch2 else 0.0) >= -0.10)
            rows.append(dict(cond=cond, method=method, final_median=fm, improvement_median=imp,
                             change_400k_600k=float(np.median(ch1)) if ch1 else None,
                             change_800k_1200k=float(np.median(ch2)) if ch2 else None,
                             per_seed=ok, checks=checks, passed=all(checks.values()),
                             required=cond in REQUIRED,
                             trend="budget-limited, not shown converged" if ch2 and np.median(ch2) > 0.0 else "flat/declining"))
    return rows


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "base"
    seeds = tuple(int(x) for x in sys.argv[2].split(",")) if len(sys.argv) > 2 else (9000, 9001)
    final = int(sys.argv[3]) if len(sys.argv) > 3 else 1_200_000
    rows = report(tag, seeds, final)
    print(json.dumps(rows, indent=1, default=str))
