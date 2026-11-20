"""Comparison scenes built from actual recorded evaluation traces (Section 9.3)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import ROOT, load_config, parse_budget
from ..env.geometry import load_map
from ..opponents.players import WaypointPlayer
from ..rendering.trace import TraceRecorder
from .play import RuleController, make_scenarios, play, build_team
from ..env.team import make_core

REPLAYS = ROOT / "evaluator_pack" / "replays"


def record_rule(cfg, scenario, rule, B, player, path, meta_extra=None):
    core = make_core(cfg, player=player, budget=B)
    team = build_team(cfg, core, None)
    rec = TraceRecorder()
    r = play(core, team, scenario, RuleController(rule), rec)
    meta = dict(map=cfg.env.map, B=str(parse_budget(B)), method=rule, run_id=rule, scenario=scenario.to_dict(),
                protocol="frozen (rule-based)", episode=1, **(meta_extra or {}))
    rec.finalize(path, meta)
    return r


def first_local_only_scenario(cfg, B="1/2", n=200, seed0=100000):
    """First scenario where, at some step, the player is visible to exactly one NPC."""
    gm = load_map(cfg.env.map)
    for sc in make_scenarios(cfg, seed0, n, "f"):
        core = make_core(cfg, player=WaypointPlayer(gm, 0.25), budget=B)
        team = build_team(cfg, core, None)
        rec = TraceRecorder()
        play(core, team, sc, RuleController("RC"), rec)
        if any(sum(1 for d in row["direct"] if d >= 0) == 1 for row in rec.rows):
            return sc
    return None


def build_scene1(cfg=None, out=REPLAYS):
    """Local-information scene: deterministic illustration with RC (clearly labelled)."""
    cfg = cfg or load_config()
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    sc = first_local_only_scenario(cfg)
    gm = load_map(cfg.env.map)
    p = WaypointPlayer(gm, 0.25)
    r = record_rule(cfg, sc, "RC", "1/2", p, out / "scene1_local_info.npz",
                    dict(player_label="Scripted waypoint player (xi = 0.25)",
                         note="Press V then 1-4: what each NPC knows", caption="Illustrative: local information"))
    (out / "scene1_local_info.json").write_text(json.dumps(dict(
        title="Local information", kind="illustrative", traces=["scene1_local_info.npz"],
        caption="Illustrative deterministic example (RC, B=1/2)",
        card=["Each NPC sees only radius 5 with line of sight.",
              "Press V and 1-4 in the viewer to see one NPC's view.",
              "Dashed arrows: packets sent. Solid arrows: packets delivered."])))
    return r


# ------------------------------------------------------------------ scenes 2-4 (recorded evaluation traces)
BTAG = {"0": "B0", "1/8": "B1o8", "1/2": "B1o2", "1": "B1"}
SCENE_SEED = 1000   # first training seed; fixed in advance, not chosen by outcome


def _run_dir(group_root, opp, method, B, seed=SCENE_SEED):
    return Path(group_root) / f"main_{opp}_{method}_{BTAG[B]}_s{seed}"


def _record_cell(run_dir, scenario, player, out, tag, pred_mode="frozen", extra=None):
    """Record one learned-pursuer episode on `scenario` via the same block runner used for the experiments."""
    from .protocols import load_run, run_block
    cfg, spec, actor, bs = load_run(run_dir)
    rows, _, _ = run_block(cfg, spec, actor, bs, player, [scenario],
                           pred_mode=pred_mode if spec["method"] == "marl_om" else "none",
                           recorder_ids={scenario.scenario_id}, trace_dir=out, tag=tag,
                           meta_extra=dict(method="MARL-OM" if spec["method"] == "marl_om" else "MARL",
                                           **(extra or {})))
    return Path(rows[0]["trace"]).name, rows[0]


def _write_scene(out, name, title, traces, titles, caption, card, kind="experimental"):
    (Path(out) / f"{name}.json").write_text(json.dumps(
        dict(title=title, kind=kind, traces=traces, titles=titles, caption=caption, card=card), indent=1))


def build_scene2(runs_root, out=REPLAYS, B="1/8", xi=0.75, opp="wp75"):
    """IP, RC, MARL, MARL-OM from the same first final-test scenario and player-randomness seed."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    from .protocols import FROZEN_WP_SEED, load_run, run_block
    d_marl, d_om = _run_dir(runs_root, opp, "marl", B), _run_dir(runs_root, opp, "marl_om", B)
    cfg, _, _, _ = load_run(d_marl)
    sc = make_scenarios(cfg, FROZEN_WP_SEED, 1, "f")[0]     # first final-test scenario; not selected by outcome
    gm = load_map(cfg.env.map)
    names, results = [], {}
    for rule in ("IP", "RC"):
        cfg_b = load_config(overrides=dict(env=dict(budget=B)))
        _, _, _ = run_block(cfg_b, None, None, None, WaypointPlayer(gm, xi), [sc], rule=rule,
                            recorder_ids={sc.scenario_id}, trace_dir=out, tag=f"s2_{rule}",
                            meta_extra=dict(method=rule, player_label=f"Scripted waypoint player (xi = {xi})"))
        names.append(f"s2_{rule}_{sc.scenario_id}.npz")
    for d in (d_marl, d_om):
        n, _ = _record_cell(d, sc, WaypointPlayer(gm, xi), out, "s2_" + ("om" if "_om_" in d.name else "marl"),
                            extra=dict(player_label=f"Scripted waypoint player (xi = {xi})"))
        names.append(n)
    _write_scene(out, "scene2_methods", "Four methods, same scenario", names,
                 ["IP", "RC", f"MARL (B={B})", f"MARL-OM (B={B})"],
                 f"First final-test scenario (seed {FROZEN_WP_SEED}); player paths can differ after the start",
                 ["Same start and same player-randomness seed for all four methods.",
                  "The player reacts to each team, so later paths can differ.",
                  f"Pursuers trained against waypoint xi = {xi}; budget B = {B}."])
    return names


def build_scene3(runs_root, out=REPLAYS, low="1/8", high="1", xi=0.75, opp="wp75"):
    """Low-budget MARL versus low-budget MARL-OM versus high-budget MARL on the same scenario."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    from .protocols import FROZEN_WP_SEED, load_run
    cfg, _, _, _ = load_run(_run_dir(runs_root, opp, "marl", low))
    sc = make_scenarios(cfg, FROZEN_WP_SEED, 1, "f")[0]
    gm = load_map(cfg.env.map)
    names = []
    for method, B, tag in (("marl", low, "lowMARL"), ("marl_om", low, "lowOM"), ("marl", high, "highMARL")):
        n, _ = _record_cell(_run_dir(runs_root, opp, method, B), sc, WaypointPlayer(gm, xi), out, "s3_" + tag,
                            extra=dict(player_label=f"Scripted waypoint player (xi = {xi})"))
        names.append(n)
    _write_scene(out, "scene3_budget", "Budget versus opponent model", names,
                 [f"MARL B={low}", f"MARL-OM B={low}", f"MARL B={high}"],
                 f"First final-test scenario (seed {FROZEN_WP_SEED}); not selected by outcome",
                 ["Low-budget MARL, low-budget MARL-OM and high-budget MARL.",
                  "Same start and player-randomness seed; later paths can differ."])
    return names


def build_scene4(runs_root, out=REPLAYS, B="1/8", opp="wp75", block=0):
    """Q-learning player early versus late in a block, with frozen and online predictor labels."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    from .protocols import BLOCK_SEED, N_BLOCK_EPISODES, eval_player_cells
    d_om = _run_dir(runs_root, opp, "marl_om", B)
    first, last = f"b{block}_{BLOCK_SEED + 100 * block}", f"b{block}_{BLOCK_SEED + 100 * block + N_BLOCK_EPISODES - 1}"
    eval_player_cells(d_om, cells=("on_frozen", "on_online"), record_ids={first, last}, trace_dir=out, diag=False)
    names = [f"marl_om_on_online_{first}.npz", f"marl_om_on_online_{last}.npz", f"marl_om_on_frozen_{last}.npz"]
    keep = {n for n in names}
    for f in out.glob("marl_om_on_*_b*.npz"):       # drop traces of other blocks recorded by the same call
        if f.name not in keep:
            f.unlink()
    _write_scene(out, "scene4_learning_player", "A learning player, early and late", names,
                 ["episode 1, predictor online", f"episode {N_BLOCK_EPISODES}, predictor online",
                  f"episode {N_BLOCK_EPISODES}, predictor frozen"],
                 "Q-player transfer from waypoint xi = 0.75 training; the player updates its Q-table during the block",
                 ["Out-of-distribution transfer: pursuers were trained against the waypoint player.",
                  "The Q-learning player updates after every transition within the block.",
                  "Compare early and late episodes; the frozen-predictor panel is the control."])
    return names
