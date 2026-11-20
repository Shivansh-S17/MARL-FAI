"""Single CLI entry point: python -m pursuit.cli COMMAND [--config F] [--output D] ..."""
from __future__ import annotations

import argparse
import json
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .config import ROOT, load_config

RESULTS = ROOT / "artifacts" / "results"


def cmd_check(a, cfg):
    from .env.geometry import load_map
    from .env.team import Team, make_core, run_episode
    from .env.scenario import sample_scenario
    from .controllers.baselines import RCController
    import numpy as np
    for m in ("corridor", "arena"):
        g = load_map(m)
        assert g.connected, f"{m} disconnected"
        print(f"map {m}: free={g.n_free} diameter={g.diameter} doorways={len(g.doorways)} connected")
    core = make_core(cfg)
    team = Team(core)
    rng = np.random.default_rng(0)
    r = run_episode(core, team, sample_scenario(core.map, rng, cfg.env.n_npcs), RCController())
    print("smoke episode:", r)
    try:
        from pettingzoo.test import parallel_api_test
        from .env.pz import PursuitParallelEnv
        parallel_api_test(PursuitParallelEnv(cfg), num_cycles=50)
        print("PettingZoo parallel API test passed")
    except ImportError as e:
        print("PettingZoo wrapper skipped:", e)
    print("check OK")


def cmd_render_map(a, cfg):
    from .rendering.figures import render_map_figures
    out = Path(a.output or ROOT / "artifacts" / "figures")
    render_map_figures(out)
    print("saved map figures to", out)


def _bank_dir(a):
    return Path(a.output) if a.output else ROOT / "artifacts" / "checkpoints" / "player_bank"


def cmd_train_player(a, cfg):
    from .opponents.train_player import HELDOUT_BANK, TRAIN_BANK, train_q_player
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else TRAIN_BANK + HELDOUT_BANK
    for s in seeds:
        out = _bank_dir(a)
        if (out / f"q_{s}.npz").exists():
            print("exists, skipping", s)
            continue
        _, meta = train_q_player(cfg, s, a.steps, out)
        print(s, "steps", meta["steps"], "updates", meta["q_updates"], "last capture",
              meta["curve"][-1]["capture_rate"], "survival", meta["curve"][-1]["mean_survival"])


def _run(spec):
    from .training.run import run_training
    return str(run_training(spec, Path(spec["root"])))


def _assign_devices(specs, a):
    """--gpus 0,1 spreads runs round-robin over those GPUs; --threads sets torch threads per run."""
    gpus = [g for g in (a.gpus or "").split(",") if g != ""]
    for k, s in enumerate(specs):
        if gpus and (a.device or "cuda").startswith("cuda"):
            s["device"] = f"cuda:{gpus[k % len(gpus)]}"
        if a.threads:
            s["threads"] = a.threads
    return specs


def cmd_pilot(a, cfg):
    from .training.pilot import pilot_specs
    import os
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"
    specs = _assign_devices(pilot_specs(a), a)
    print(f"{len(specs)} pilot runs; workers={a.workers}")
    if a.dry_run:
        for s in specs:
            print(s["run_id"])
        return
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context('spawn')) as ex:
        for r in ex.map(_run, specs):
            print("done", r)


def cmd_train(a, cfg):
    from .training.manifest import select_specs
    import os
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"
    specs = _assign_devices(select_specs(a), a)
    print(f"{len(specs)} runs")
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context('spawn')) as ex:
        for r in ex.map(_run, specs):
            print("done", r)


def cmd_evaluate(a, cfg):
    from .evaluation.protocols import run_protocol
    run_protocol(a, cfg)


def cmd_analyze(a, cfg):
    from .evaluation.analysis import run_analysis
    run_analysis(a, cfg)


def cmd_demo(a, cfg):
    from .rendering.viewer import run_viewer
    run_viewer(a)


def cmd_scenes(a, cfg):
    """Build comparison scenes 2-4 from saved main-run checkpoints (scene 1 is an illustrative RC scene)."""
    from .config import ROOT
    from .evaluation import scenes
    root = ROOT / "artifacts" / "results" / "main"
    out = a.output or scenes.REPLAYS
    scenes.build_scene1(out=out)
    for fn in (scenes.build_scene2, scenes.build_scene3, scenes.build_scene4):
        print(fn.__name__, fn(root, out=out))


def cmd_export_demo(a, cfg):
    from .rendering.export import export_demo
    export_demo(a)


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m pursuit.cli")
    p.add_argument("command", choices=["check", "render-map", "train-player", "pilot", "train", "evaluate",
                                       "analyze", "demo", "export-demo", "scenes"])
    p.add_argument("--config")
    p.add_argument("--output")
    p.add_argument("--run-ids", default=None, help="comma separated manifest run IDs")
    p.add_argument("--seeds", default=None)
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--total-steps", type=int, default=300_000)
    p.add_argument("--shaping", action="store_true")
    p.add_argument("--pilot-seeds", default="9000,9001")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--protocol", default=None)
    p.add_argument("--replay", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--gpus", default=None, help="comma list of GPU indices for round-robin run placement")
    p.add_argument("--threads", type=int, default=None, help="torch threads per training process")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    globals()["cmd_" + a.command.replace("-", "_")](a, cfg)


if __name__ == "__main__":
    main()
