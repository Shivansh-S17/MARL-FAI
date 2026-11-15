"""Pilot run specifications (Section 6 of the plan)."""
from __future__ import annotations

from ..config import ROOT

PILOT_CONDITIONS = {
    "wp0_B1": dict(opponent=dict(type="waypoint", xi=0.0), budget="1"),
    "wp75_B1o8": dict(opponent=dict(type="waypoint", xi=0.75), budget="1/8"),
    "wp25_B0": dict(opponent=dict(type="waypoint", xi=0.25), budget="0"),
    "qbank_B1o8": dict(opponent=dict(type="qbank"), budget="1/8"),
}


def pilot_specs(a):
    seeds = [int(s) for s in a.pilot_seeds.split(",")]
    total = a.total_steps
    ev = [s for s in (50_000, 100_000, 200_000, 300_000, 400_000, 600_000, 800_000, 1_200_000) if s <= total]
    root = str(ROOT / "artifacts" / "results" / "pilot")
    tag = "shaped" if a.shaping else "base"
    specs = []
    for cname, c in PILOT_CONDITIONS.items():
        for seed in seeds:
            for method in ("marl", "marl_om"):
                specs.append(dict(run_id=f"pilot_{cname}_{method}_s{seed}_{tag}", method=method, seed=seed,
                                  total_steps=total, eval_steps=ev, shaping=a.shaping, root=root,
                                  device=a.device or "cuda", val_n=100, **c))
    return specs
