"""Frozen / adaptive evaluation protocols (Sections 7.2-7.4)."""
from __future__ import annotations

import fnmatch
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..config import ROOT, load_config, parse_budget
from ..env.geometry import load_map
from ..env.team import make_core
from ..models.networks import Actor
from ..models.predictor import PredictorBank
from ..opponents.players import QPlayer, WaypointPlayer
from ..opponents.train_player import HELDOUT_BANK, load_q
from ..rendering.trace import TraceRecorder
from .play import (EvalPredictorHook, LearnedController, RuleController, build_team, make_scenarios, play)

PLAYER_BANK_DIR = ROOT / "artifacts" / "checkpoints" / "player_bank"
EVAL_DIR = ROOT / "artifacts" / "results" / "eval"
FROZEN_WP_SEED, BLOCK_SEED, DIAG_SEED = 100000, 200000, 300000
N_BLOCK_EPISODES = 20


def is_transfer_run(spec):
    """Q-player transfer/adaptation uses the preselected waypoint xi=0.75-trained pursuers (plan 7.3)."""
    o = spec["opponent"]
    return o["type"] == "waypoint" and abs(o.get("xi", -1) - 0.75) < 1e-9 and not spec.get("shuffle_features", False)


def probe_set(bank_dir=PLAYER_BANK_DIR):
    keys = set()
    for s in HELDOUT_BANK:
        keys |= set(int(k) for k in np.load(Path(bank_dir) / f"q_{s}.npz")["visited"])
    return np.array(sorted(keys))


def load_run(run_dir, cfg=None):
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text())
    cfg = cfg or load_config(overrides=meta["cfg"])
    st = torch.load(run_dir / "checkpoint.pt", weights_only=False, map_location="cpu")
    actor = Actor()
    actor.load_state_dict({k: v.cpu() for k, v in st["actor"].items()})
    actor.eval()
    return cfg, meta["spec"], actor, st.get("bank")


def fresh_bank(cfg, spec, bank_state):
    if bank_state is None:
        return None
    b = PredictorBank(load_map(cfg.env.map), cfg, spec["seed"])
    b.load(bank_state)
    return b


def run_block(cfg, spec, actor, bank_state, player, scenarios, pred_mode="none", q0=None, probe=None,
              recorder_ids=None, trace_dir=None, tag="", rule=None, meta_extra=None, use_packets=True):
    """One block of consecutive episodes. The player object (and predictor bank) persist across the
    episodes of the block; the caller creates fresh clones per block."""
    budget = parse_budget(spec["budget"] if spec else cfg.env.budget)
    core = make_core(cfg, player=player, budget=budget)
    bank = fresh_bank(cfg, spec, bank_state) if spec and spec["method"] == "marl_om" else None
    hook = EvalPredictorHook(bank, online=(pred_mode == "online")) if bank is not None else None
    team = build_team(cfg, core, bank, hook, use_packets)
    if rule is not None:
        ctrl = RuleController(rule)
    else:
        ctrl = LearnedController(actor, float(budget), cfg.env.n_npcs)
    rows = []
    for e, sc in enumerate(scenarios):
        if hook is not None:
            hook.reset_stats()
        rec = TraceRecorder(record_belief=bank is not None) if recorder_ids and sc.scenario_id in recorder_ids else None
        r = play(core, team, sc, ctrl, rec, bank, online=(pred_mode == "online"))
        r["episode_in_block"] = e
        if isinstance(player, QPlayer):
            r["q_updates"] = player.n_updates
            if q0 is not None:
                r["q_change_norm"] = float(np.linalg.norm(player.Q - q0.Q))
                if probe is not None:
                    r["greedy_changes"] = int((player.Q[probe].argmax(1) != q0.Q[probe].argmax(1)).sum())
        if hook is not None and hook.n:
            r.update(pred_logloss=hook.ll / hook.n, pred_acc=hook.acc / hook.n, pred_labels=hook.n)
        elif hook is not None:
            r.update(pred_logloss=float("nan"), pred_acc=float("nan"), pred_labels=0)
        if rec is not None:
            meta = dict(map=cfg.env.map, B=str(budget), run_id=spec["run_id"] if spec else rule, protocol=tag,
                        scenario=sc.to_dict(), episode=e + 1, **(meta_extra or {}))
            p = Path(trace_dir) / f"{tag}_{sc.scenario_id}.npz"
            rec.finalize(p, meta)
            r["trace"] = str(p)
        rows.append(r)
    return rows, player, bank


# -------------------------------------------------------------------- 7.2 frozen waypoint
def eval_frozen_waypoint(run_dir, n=100, out_dir=EVAL_DIR / "frozen_waypoint"):
    cfg, spec, actor, bs = load_run(run_dir)
    xi = spec["opponent"]["xi"]
    scen = make_scenarios(cfg, FROZEN_WP_SEED, n, "f")
    player = WaypointPlayer(load_map(cfg.env.map), xi)
    t0 = time.time()
    rows, _, bank = run_block(cfg, spec, actor, bs, player, scen, pred_mode="frozen")
    fp = bank.fingerprint() if bank else None
    for r in rows:
        r.update(run_id=spec["run_id"], method=spec["method"], seed=spec["seed"], B=spec["budget"],
                 player=f"wp{int(xi * 100)}", protocol="frozen_waypoint")
    return rows, dict(elapsed=time.time() - t0, predictor_fp_after=fp)


def eval_baseline_waypoint(rule, xi, B, n=100):
    cfg = load_config(overrides=dict(env=dict(budget=B)))
    scen = make_scenarios(cfg, FROZEN_WP_SEED, n, "f")
    player = WaypointPlayer(load_map(cfg.env.map), xi)
    rows, _, _ = run_block(cfg, None, None, None, player, scen, rule=rule)
    for r in rows:
        r.update(run_id=f"{rule}_wp{int(xi * 100)}_B{B}", method=rule, seed=-1, B=B, player=f"wp{int(xi * 100)}",
                 protocol="frozen_waypoint")
    return rows


# --------------------------------------------------------------- 7.3 / 7.4 learned player
def block_scenarios(cfg, block):
    return make_scenarios(cfg, BLOCK_SEED + 100 * block, N_BLOCK_EPISODES, f"b{block}_")


def eval_player_cells(run_dir, cells=("off_frozen", "on_frozen", "on_online"), bank_dir=PLAYER_BANK_DIR,
                      record_ids=None, trace_dir=None, diag=True):
    """cells: player-update/predictor-update combination. MARL: predictor 'none'."""
    cfg, spec, actor, bs = load_run(run_dir)
    probe = probe_set(bank_dir)
    om = spec["method"] == "marl_om"
    out, diag_out, finals = [], [], {}
    for cell_ in cells:
        qon = cell_.startswith("on")
        pm = "none" if not om else ("online" if cell_.endswith("online") else "frozen")
        if cell_ == "on_online" and not om:
            continue
        for k, qseed in enumerate(HELDOUT_BANK):
            q0 = load_q(cfg, qseed, bank_dir, learning=False)
            q = q0.clone(learning=qon)
            rows, q, _ = run_block(cfg, spec, actor, bs, q, block_scenarios(cfg, k), pred_mode=pm, q0=q0,
                                   probe=probe, recorder_ids=record_ids, trace_dir=trace_dir,
                                   tag=f"{spec['method']}_{cell_}", meta_extra=dict(
                                       method="MARL-OM" if om else "MARL",
                                       player_label=f"Q-learning player (checkpoint {qseed}), updates {'ON' if qon else 'OFF'}",
                                       note=f"predictor {pm}" if om else ""))
            for r in rows:
                r.update(run_id=spec["run_id"], method=spec["method"], seed=spec["seed"], B=spec["budget"],
                         protocol=cell_, block=k, q_ckpt=qseed, player="qbank", train_opponent="wp75")
            out += rows
            if cell_ == "on_frozen" and not om and diag:
                finals[k] = q  # final Q of each MARL adaptive block for the matched-opponent diagnostic
    return out, finals


def eval_matched_diagnostic(run_dir, final_qs, frozen_q_source=None):
    """Evaluate a pursuer on 20 fresh diagnostic episodes per block against a FROZEN final Q-table."""
    cfg, spec, actor, bs = load_run(run_dir)
    om = spec["method"] == "marl_om"
    rows = []
    for k, q in final_qs.items():
        scen = make_scenarios(cfg, DIAG_SEED + 100 * k, N_BLOCK_EPISODES, f"d{k}_")
        qq = q.clone(learning=False)
        r, _, _ = run_block(cfg, spec, actor, bs, qq, scen, pred_mode="frozen" if om else "none")
        for x in r:
            x.update(run_id=spec["run_id"], method=spec["method"], seed=spec["seed"], B=spec["budget"],
                     protocol="matched_diagnostic", block=k, player="qbank")
        rows += r
    return rows


def eval_baseline_player(rule, B, qon: bool, bank_dir=PLAYER_BANK_DIR):
    cfg = load_config(overrides=dict(env=dict(budget=B)))
    probe = probe_set(bank_dir)
    out = []
    for k, qseed in enumerate(HELDOUT_BANK):
        q0 = load_q(cfg, qseed, bank_dir, learning=False)
        rows, _, _ = run_block(cfg, None, None, None, q0.clone(learning=qon), block_scenarios(cfg, k), q0=q0,
                               probe=probe, rule=rule)
        for r in rows:
            r.update(run_id=f"{rule}_qbank_B{B}", method=rule, seed=-1, B=B, protocol="on_frozen" if qon else "off_frozen",
                     block=k, q_ckpt=qseed, player="qbank")
        out += rows
    return out


# ------------------------------------------------------------------------- CLI glue
def _run_dirs(a):
    pats = a.run_ids.split(",") if a.run_ids else ["*"]
    dirs = []
    for sub in ("main", "secondary"):
        base = ROOT / "artifacts" / "results" / sub
        if base.exists():
            dirs += [d for d in sorted(base.iterdir()) if (d / "status.json").exists()
                     and json.loads((d / "status.json").read_text()).get("status") == "complete"
                     and any(fnmatch.fnmatch(d.name, p) for p in pats)]
    return dirs


def _eval_reliance_dir(d):
    """Reliance diagnostic: same scenarios with delivered packets used versus ignored by the receivers."""
    cfg, spec, actor, bs = load_run(d)
    scen = make_scenarios(cfg, FROZEN_WP_SEED, 100, "f")
    rows = []
    for mode, use in (("packets_used", True), ("packets_ignored", False)):
        player = WaypointPlayer(load_map(cfg.env.map), spec["opponent"]["xi"])
        r, _, _ = run_block(cfg, spec, actor, bs, player, scen, pred_mode="frozen", use_packets=use)
        for x in r:
            rows.append(dict(run_id=spec["run_id"], method=spec["method"], seed=spec["seed"], B=str(spec["budget"]),
                             player=f"wp{int(round(spec['opponent']['xi'] * 100))}", mode=mode,
                             scenario_id=x["scenario_id"], captured=x["captured"], length=x["length"], packets=x["packets"]))
    pd.DataFrame(rows).to_csv(Path(d) / "eval_reliance.csv", index=False)
    return spec["run_id"]


def run_protocol(a, cfg):
    out = Path(a.output) if a.output else EVAL_DIR
    out.mkdir(parents=True, exist_ok=True)
    proto = a.protocol
    if proto == "baselines":
        rows = []
        for xi in (0.0, 0.25, 0.75):
            rows += eval_baseline_waypoint("IP", xi, "0")
            for B in ("0", "1/8", "1/2", "1"):
                rows += eval_baseline_waypoint("RC", xi, B)
        for B in ("0", "1/8", "1/2", "1"):
            for rule in (("IP", "RC") if B != "0" else ("IP", "RC")):
                for qon in (False, True):
                    rows += eval_baseline_player(rule, B, qon)
        pd.DataFrame(rows).to_csv(out / "baselines.csv", index=False)
        print("baselines rows:", len(rows))
        return
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    workers = max(1, int(getattr(a, "workers", 1) or 1))
    dirs = [str(d) for d in _run_dirs(a)]

    def pool_map(fn, items):
        if workers == 1:
            yield from map(fn, items)
        else:
            with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
                yield from ex.map(fn, items)

    if proto == "frozen_waypoint":
        for name in pool_map(_eval_frozen_dir, dirs):
            print("evaluated", name, flush=True)
    elif proto == "reliance":
        sel = []
        for d in dirs:
            sp = json.loads((Path(d) / "meta.json").read_text())["spec"]
            if str(sp["budget"]) == "1" and sp["opponent"]["type"] == "waypoint" and abs(sp["opponent"]["xi"] - 0.75) < 1e-9                     and not sp.get("shuffle_features", False):
                sel.append(d)
        for name in pool_map(_eval_reliance_dir, sel):
            print("reliance", name, flush=True)
    elif proto == "adaptive":
        for name in pool_map(_eval_adaptive_dir, dirs):
            print("evaluated", name, flush=True)
        # matched-opponent diagnostic: needs the MARL block Qs for both methods
        jobs = []
        for d in map(Path, dirs):
            spec = json.loads((d / "meta.json").read_text())["spec"]
            if is_transfer_run(spec) and spec["method"] == "marl" and (d / "final_q_blocks.npz").exists():
                jobs.append(str(d))
        for name in pool_map(_eval_diag_dir, jobs):
            print("diagnostic", name, flush=True)
    else:
        raise SystemExit(f"unknown protocol {proto}")


def _setup_worker():
    import torch
    torch.set_num_threads(1)


def _eval_frozen_dir(d):
    _setup_worker()
    d = Path(d)
    if json.loads((d / "meta.json").read_text())["spec"]["opponent"]["type"] != "waypoint":
        return d.name + " (skipped)"
    tgt = d / "eval_frozen_waypoint.csv"
    if tgt.exists():
        return d.name + " (exists)"
    rows, info = eval_frozen_waypoint(d)
    pd.DataFrame(rows).to_csv(tgt, index=False)
    return d.name


def _eval_adaptive_dir(d):
    _setup_worker()
    d = Path(d)
    if not is_transfer_run(json.loads((d / "meta.json").read_text())["spec"]):
        return d.name + " (skipped)"
    tgt = d / "eval_player_cells.csv"
    if tgt.exists():
        return d.name + " (exists)"
    rows, finals = eval_player_cells(d)
    pd.DataFrame(rows).to_csv(tgt, index=False)
    if finals:
        np.savez(d / "final_q_blocks.npz", **{str(k): q.Q for k, q in finals.items()})
    return d.name


def _eval_diag_dir(d):
    _setup_worker()
    d = Path(d)
    twin = d.parent / d.name.replace("_marl_", "_marl_om_")
    z = np.load(d / "final_q_blocks.npz")
    cfg2 = load_config()
    finals = {int(k): QPlayer(load_map(cfg2.env.map), cfg2.env.horizon, Q=z[k]) for k in z.files}
    rows = eval_matched_diagnostic(d, finals)
    if twin.exists() and (twin / "status.json").exists():
        rows += eval_matched_diagnostic(twin, finals)
    pd.DataFrame(rows).to_csv(d / "eval_matched_diagnostic.csv", index=False)
    return d.name
