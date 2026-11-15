import json
import os

import numpy as np

from conftest import scen
from pursuit.config import load_config
from pursuit.env.geometry import load_map
from pursuit.env.team import make_core
from pursuit.evaluation.play import RuleController, build_team, make_scenarios, play
from pursuit.opponents.players import QPlayer, WaypointPlayer
from pursuit.rendering.trace import Trace, TraceRecorder


def run(rule, B, record, seed_idx=0, xi=0.25):
    cfg = load_config(overrides=dict(env=dict(budget=B)))
    gm = load_map("corridor")
    core = make_core(cfg, player=WaypointPlayer(gm, xi), budget=B)
    team = build_team(cfg, core, None)
    sc = make_scenarios(cfg, 100000, 5)[seed_idx]
    rec = TraceRecorder() if record else None
    r = play(core, team, sc, RuleController(rule), rec)
    return r, rec, cfg


def test_recording_does_not_change_outcome():
    for rule in ("IP", "RC"):
        r1, _, _ = run(rule, "1/2", False)
        r2, _, _ = run(rule, "1/2", True)
        assert r1 == r2


def test_identical_seeds_reproduce_traces(tmp_path):
    _, rec1, _ = run("RC", "1/2", True)
    _, rec2, _ = run("RC", "1/2", True)
    p1 = rec1.finalize(tmp_path / "a.npz", dict(map="corridor", B="1/2", method="RC"))
    p2 = rec2.finalize(tmp_path / "b.npz", dict(map="corridor", B="1/2", method="RC"))
    a, b = Trace(p1), Trace(p2)
    for k in a.a:
        assert (a.a[k] == b.a[k]).all()


def test_metrics_recomputed_from_trace_match(tmp_path):
    r, rec, _ = run("RC", "1/2", True)
    p = rec.finalize(tmp_path / "a.npz", dict(map="corridor", B="1/2", method="RC"))
    tr = Trace(p)
    assert tr.T == r["length"] and tr.meta["captured"] == bool(r["captured"])
    assert int(tr.a["sent_cum"][-1].sum()) == r["packets"]
    assert int(tr.a["moves_cum"][-1]) + int((tr.a["moves"][-1] != 0).sum()) == r["moves"]


def test_headless_render_and_scrub_determinism(tmp_path):
    from pursuit.rendering.renderer import Renderer
    _, rec, _ = run("RC", "1/2", True)
    p = rec.finalize(tmp_path / "a.npz", dict(map="corridor", B="1/2", method="RC", player_label="x",
                                              protocol="p", episode=1))
    tr = Trace(p)
    rd = Renderer(headless=True)
    seq = []
    for t in [0, 3, 1, 3, tr.T, 2, 3]:
        rd.frame([tr], t, "full", 0, True, False)
        seq.append((t, rd.to_image().tobytes()))
    imgs = {t: b for t, b in seq}
    assert [b for t, b in seq if t == 3][0] == [b for t, b in seq if t == 3][1] == [b for t, b in seq if t == 3][2]
    assert len(seq[0][1]) == 960 * 540 * 3


def test_npc_view_does_not_reveal_hidden_player(tmp_path):
    """In NPC view the true player position is drawn only when directly visible to the selected NPC."""
    from pursuit.rendering.renderer import PLAYER_COL, Renderer
    _, rec, _ = run("IP", "0", True)
    p = rec.finalize(tmp_path / "a.npz", dict(map="corridor", B="0", method="IP", player_label="x",
                                              protocol="p", episode=1))
    tr = Trace(p)
    rd = Renderer(headless=True)
    for t in range(tr.T):
        for sel in range(4):
            if tr.a["direct"][t, sel] >= 0 or tr.a["mem_cell"][t, sel] >= 0:
                continue  # visible or remembered: allowed
            rd.frame([tr], t, "npc", sel, True, False)
            import pygame
            surf = pygame.surfarray.array3d(rd.surf)[:540, :540]   # map region only, [x,y,3]
            hits = (np.abs(surf.astype(int) - np.array(PLAYER_COL)).sum(-1) < 6).sum()
            assert hits == 0, "player-coloured pixels drawn although not known to this NPC"
