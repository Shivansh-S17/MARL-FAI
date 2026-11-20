"""Main-run manifest (Section 7.1) and spec selection for `cli train`."""
from __future__ import annotations

import fnmatch
import json
import sys

from ..config import ROOT

MANIFEST = ROOT / "configs" / "final_manifest.json"
BUDGETS = ["0", "1/8", "1/2", "1"]
BTAG = {"0": "B0", "1/8": "B1o8", "1/2": "B1o2", "1": "B1"}
OPPONENTS = {
    "wp0": dict(type="waypoint", xi=0.0),
    "wp25": dict(type="waypoint", xi=0.25),
    "wp75": dict(type="waypoint", xi=0.75),
}   # qbank is not a training condition (plan revision, Decision A); the Q player is evaluation-only
TASK_ID = "C0"
SEEDS = [1000, 1001, 1002, 1003, 1004]


def build_manifest(total_steps: int, shaping: bool, device="cuda"):
    root = str(ROOT / "artifacts" / "results" / "main")
    runs = []
    for opp, o in OPPONENTS.items():
        for b in BUDGETS:
            for method in ("marl", "marl_om"):
                for s in SEEDS:
                    runs.append(dict(run_id=f"main_{opp}_{method}_{BTAG[b]}_s{s}", group="main", method=method,
                                     seed=s, opponent=o, budget=b, total_steps=total_steps, shaping=shaping,
                                     eval_steps=[x for x in (400_000, 600_000, 800_000, total_steps) if x <= total_steps],
                                     root=root, device=device, val_n=100))
    return dict(description="Main training matrix (3 waypoint levels x 4 budgets x 2 methods x 5 seeds); frozen after the pilot decision.",
                task_id=TASK_ID, total_steps=total_steps, shaping=shaping, runs=runs,
                scenario_seeds=dict(frozen_waypoint=100000, frozen_learned_player=200000, diagnostic=300000,
                                    validation=20000, bootstrap=40000))


def secondary_runs(total_steps, shaping, factor=None):
    root = str(ROOT / "artifacts" / "results" / "secondary")
    runs = []
    for opp in ("wp0", "wp75"):   # shuffled-feature control (priority 1): OM architecture, shuffled features
        for s in SEEDS:
            runs.append(dict(run_id=f"ctrl_shuffle_{opp}_B1o8_s{s}", group="shuffle_control", method="marl_om", seed=s,
                             opponent=OPPONENTS[opp], budget="1/8", total_steps=total_steps, shaping=shaping,
                             eval_steps=[x for x in (400_000, 600_000, 800_000, total_steps) if x <= total_steps],
                             root=root, shuffle_features=True, val_n=100))
    return runs


def load_manifest():
    return json.loads(MANIFEST.read_text())


def select_specs(a):
    m = load_manifest()
    runs = m["runs"] + secondary_runs(m["total_steps"], m["shaping"])
    if a.run_ids:
        pats = a.run_ids.split(",")
        runs = [r for r in runs if any(fnmatch.fnmatch(r["run_id"], p) for p in pats)]
    for r in runs:  # portable roots: never trust an absolute path written on another machine
        r["root"] = str(ROOT / "artifacts" / "results" / ("main" if r["group"] == "main" else "secondary"))
    if a.device:
        for r in runs:
            r["device"] = a.device
    return runs


if __name__ == "__main__":
    steps = int(sys.argv[1]); shaping = "--shaping" in sys.argv
    MANIFEST.write_text(json.dumps(build_manifest(steps, shaping), indent=1))
    print("wrote", MANIFEST, len(json.loads(MANIFEST.read_text())["runs"]), "runs")
