"""One training run: spec -> immutable run directory with logs, validation curve, checkpoints."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
import traceback
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from ..config import ROOT, Cfg, load_config, parse_budget
from ..env.geometry import load_map
from ..env.team import make_core
from ..evaluation.play import EvalPredictorHook, LearnedController, build_team, make_scenarios, play
from ..opponents.players import QPlayer, WaypointPlayer
from ..opponents.train_player import TRAIN_BANK, load_q
from .trainer import Trainer

PLAYER_BANK_DIR = ROOT / "artifacts" / "checkpoints" / "player_bank"


def code_hash() -> str:
    h = hashlib.sha256()
    for p in sorted((ROOT / "src" / "pursuit").rglob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def file_hash(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def validate(trainer: Trainer, n=100, seed_start=20000, qbank_players=None):
    """Greedy validation on fixed standard-placement scenarios; does not touch trainer state."""
    cfg = trainer.cfg
    trainer._sync_acting()
    scen = make_scenarios(cfg, seed_start, n, "val")
    gm = trainer.gm
    opp = trainer.opponent
    player = WaypointPlayer(gm, opp.get("xi", 0.0)) if opp["type"] == "waypoint" else qbank_players[0]
    core = make_core(cfg, player=player, budget=trainer.budget)
    hook = None
    if trainer.om:
        hook = EvalPredictorHook(trainer.bank, online=False)
    team = build_team(cfg, core, trainer.bank if trainer.om else None, hook)
    ctrl = LearnedController(trainer.actor_act, float(trainer.budget), cfg.env.n_npcs)
    rows = []
    for k, sc in enumerate(scen):
        if opp["type"] == "qbank":
            core.set_player(trainer.qbank[k % len(trainer.qbank)].clone(learning=False))
        rows.append(play(core, team, sc, ctrl))
    out = dict(steps=trainer.steps, capture=float(np.mean([r["captured"] for r in rows])),
               capped_duration=float(np.mean([r["capped_duration"] for r in rows])),
               packets=float(np.mean([r["packets"] for r in rows])),
               base_return=float("nan"), n=n)
    if hook is not None and hook.n:
        out.update(pred_logloss=hook.ll / hook.n, pred_acc=hook.acc / hook.n, pred_labels=hook.n)
    return out


def untrained_val(spec, cfg):
    return None


def run_training(spec: dict, root: Path | None = None, resume=True):
    """spec keys: run_id, method, seed, opponent{type,xi}, budget, total_steps, shaping, eval_steps,
    device, shuffle_features, cfg_overrides."""
    root = Path(root or ROOT / "artifacts" / "results")
    rd = root / spec["run_id"]
    status_p = rd / "status.json"
    if status_p.exists() and json.loads(status_p.read_text()).get("status") == "complete":
        try:
            have = json.loads((rd / "meta.json").read_text()).get("actual_steps", 0)
        except Exception:
            have = 10 ** 12
        if have >= spec["total_steps"] - 8:   # a final run is never re-opened; a pilot may be extended
            print("run complete, skipping:", spec["run_id"])
            return rd
    rd.mkdir(parents=True, exist_ok=True)
    ck = rd / "checkpoint.pt"
    ov = dict(train=dict(total_steps=spec["total_steps"]), reward=dict(shaping=bool(spec.get("shaping", False))))
    ov = {**ov, **spec.get("cfg_overrides", {})}
    cfg = load_config(overrides=ov)
    torch.set_num_threads(spec.get("threads", 2))
    status_p.write_text(json.dumps(dict(status="running", start=time.time())))
    try:
        qbank = None
        if spec["opponent"]["type"] == "qbank":
            qbank = [load_q(cfg, s, PLAYER_BANK_DIR) for s in TRAIN_BANK]
        tr = Trainer(cfg, spec["method"], spec["seed"], spec["opponent"], parse_budget(spec["budget"]), qbank,
                     spec.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
                     spec.get("shuffle_features", False), rd)
        resumed = False
        if resume and ck.exists():
            tr.load(ck)
            resumed = True
        val_path = rd / "val.json"
        vals = json.loads(val_path.read_text()) if val_path.exists() else []
        evals = sorted(set(spec.get("eval_steps", [])))
        done_ev = {v["steps"] for v in vals}

        def do_eval(tr_, target=None):
            v = validate(tr_, spec.get("val_n", 100), 20000, qbank)
            v["train_captures"] = tr_.stats["captures"]
            v["elapsed"] = time.time() - tr_.t_start
            v["target_step"] = target
            vals.append(v)
            val_path.write_text(json.dumps(vals, indent=1))
            print(spec["run_id"], "VAL", json.dumps(v), flush=True)

        if not vals:
            do_eval(tr, 0)  # untrained policy
        pending = [s for s in evals if s > 0 and s not in done_ev and s > tr.steps - 1]
        nxt = list(pending)

        def eval_cb(tr_):
            do_eval(tr_)

        def ckpt_cb(tr_):
            tr_.save(ck)

        total = spec["total_steps"]
        # train in segments so that validation happens at the requested budgets
        for target in sorted(set([s for s in evals if s <= total] + [total])):
            if target <= tr.steps:
                continue
            tr.train(target, ckpt_cb=ckpt_cb, ckpt_every=cfg.train.checkpoint_every, verbose=True, log_every=20)
            if target in evals:
                do_eval(tr, target)
        tr.save(ck)
        meta = dict(spec=spec, cfg=cfg.to_dict(), code_hash=code_hash(), python=sys.version.split()[0],
                    torch=torch.__version__, numpy=np.__version__, platform=platform.platform(),
                    device=str(tr.device), actual_steps=tr.steps, elapsed_s=time.time() - tr.t_start,
                    checkpoint_hash=file_hash(ck), resumed=resumed,
                    deterministic_note="CUDA kernels not forced deterministic; CPU acting path is seeded",
                    captures=tr.stats["captures"], episodes=tr.stats["episodes"],
                    labels={str(k): v for k, v in (tr.bank.label_log.items() if tr.om else [])})
        (rd / "meta.json").write_text(json.dumps(meta, indent=1))
        (rd / "train_log.json").write_text(json.dumps(tr.log))
        (rd / "episodes.json").write_text(json.dumps(tr.episodes))
        status_p.write_text(json.dumps(dict(status="complete", end=time.time())))
    except Exception:
        status_p.write_text(json.dumps(dict(status="failed", error=traceback.format_exc())))
        (rd / "error.log").write_text(traceback.format_exc())
        raise
    return rd
