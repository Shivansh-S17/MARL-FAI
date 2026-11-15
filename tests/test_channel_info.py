from fractions import Fraction

import numpy as np
import pytest

from conftest import mk, scen
from pursuit.controllers.local import LocalInfo
from pursuit.env.channel import PACKET_BYTES, Channel, Packet, allowed_sends
from pursuit.env.core import LocalObs
from pursuit.env.geometry import cell, load_map
from pursuit.env.observation import OFF_PRED, build_obs
from pursuit.env.team import Team
from pursuit.env.scenario import sample_scenario


@pytest.mark.parametrize("B", ["0", "1/8", "1/2", "1"])
def test_every_prefix_channel_limit(B):
    b = Fraction(B)
    ch = Channel(1, b, 0)
    sent_steps = []
    for t in range(200):
        if ch.can_send(0, t):  # always send when feasible (maximal sender)
            ch.send(0, t, bytes(16))
            sent_steps.append(t)
        # prefix limit sent(T) <= floor(B*T) with T = t+1 decisions made
        assert ch.sent[0] <= (b.numerator * (t + 1)) // b.denominator
    assert ch.sent[0] == (b.numerator * 200) // b.denominator  # allowance is fully usable
    if B == "0":
        assert ch.sent[0] == 0
    if B == "1":
        assert ch.sent[0] == 200
    if B == "1/8":
        assert sent_steps[0] == 7  # credit starts at zero


def test_packet_is_16_bytes_roundtrip():
    p = Packet(3, 31, 399, 123, 77, 2, 1, 55, 78)
    raw = p.pack()
    assert len(raw) == PACKET_BYTES == 16
    q = Packet.unpack(raw)
    assert q == p


@pytest.mark.parametrize("delta", [0, 2])
def test_delayed_arrival_and_timestamps(delta):
    ch = Channel(3, Fraction(1), delta)
    ch.send(0, 5, bytes(16))
    for t in range(5, 5 + delta + 1):
        assert all(len(v) == 0 for v in ch.deliver(t).values())
    out = ch.deliver(6 + delta)
    assert len(out[1]) == 1 and len(out[2]) == 1 and len(out[0]) == 0  # broadcast, one transmission charged
    assert ch.total_sent == 1


def test_charged_on_capture_and_reset():
    cfg, core, team = mk("1", n_npcs=1)
    core.reset(scen([(8, 5)], (10, 5)))
    team.reset()

    class West:
        kind = "x"; learning = False
        def reset(self, s): pass
        def act(self, p, a, b): return 4
    core.set_player(West())
    core.reset(scen([(8, 5)], (10, 5)))
    team.begin_step()
    res = team.commit_and_step([2], [True])
    assert res.captured and core.channel.total_sent == 1 and res.broadcasts == 1
    core.reset(scen([(8, 5)], (10, 5)))
    assert core.channel.total_sent == 0 and core.channel.pending == []


def _info(i=0, budget="1"):
    g = load_map("corridor")
    return LocalInfo(i, g, 4, 200, Fraction(budget), 5), g


def obs_at(g, t, own, player=-1, tm=()):
    vis = g.visibility(5)[own]
    return LocalObs(t, own, vis, player, tuple(tm))


def test_target_age_uses_observation_time_not_arrival():
    a, g = _info(0)
    b, _ = _info(1)
    b.update(obs_at(g, 3, cell(8, 8), cell(9, 8)), [])      # b sees player at t=3
    pkt = b.build_packet()
    a.update(obs_at(g, 10, cell(2, 2)), [pkt])               # arrives at t=10
    assert a.mem_step == 3 and a.mem_cell == cell(9, 8) and a.target_age() == 7


def test_displacement_only_from_consecutive_sightings():
    a, g = _info(0)
    a.update(obs_at(g, 0, cell(8, 8), cell(9, 8)), [])
    a.update(obs_at(g, 1, cell(8, 8), cell(10, 8)), [])      # consecutive -> east
    assert a.mem_disp_valid and a.mem_disp == 2
    a.update(obs_at(g, 2, cell(2, 2)), [])
    a.update(obs_at(g, 3, cell(8, 8), cell(10, 10)), [])     # gap in sightings -> no label
    assert not a.mem_disp_valid


def test_three_frame_labels_only_direct():
    labels = []
    a, g = _info(0)
    a.labeller = lambda *args: labels.append(args)
    own = cell(8, 8)
    a.update(obs_at(g, 0, own, cell(9, 8)), [])
    a.update(obs_at(g, 1, own, cell(10, 8)), [])
    assert labels == []
    a.update(obs_at(g, 2, own, cell(10, 9)), [])             # frames 0,1,2 -> label for context at t=1
    assert len(labels) == 1
    aid, ownc, tgt, prev, tms, lab = labels[0]
    assert (tgt, prev, lab) == (cell(10, 8), 2, 3)
    a.update(obs_at(g, 3, cell(2, 2)), [])                    # unseen
    a.update(obs_at(g, 4, own, cell(10, 10)), [])
    a.update(obs_at(g, 5, own, cell(10, 11)), [])
    assert len(labels) == 1                                   # no label across the gap
    # packet sightings never create labels
    b, _ = _info(1); b.labeller = lambda *x: labels.append(x)
    for t in range(3):
        b.update(obs_at(g, t, cell(2, 2)), [Packet(2, 31, 5, cell(9, 8), t, 0, 0, 5, t).pack()])
    assert len(labels) == 1


def test_hidden_state_invariance():
    """Local histories identical, hidden remote positions differ -> identical actor inputs/roles/packets."""
    cfg, c1, t1 = mk("0")
    _, c2, t2 = mk("0")
    sa = scen([(2, 2), (17, 17), (16, 16), (17, 15)], (15, 12))
    sb = scen([(2, 2), (11, 17), (17, 3), (9, 16)], (17, 16), seed=99)
    c1.reset(sa); c2.reset(sb); t1.reset(); t2.reset()
    compared = 0
    for t in range(12):
        t1.begin_step(); t2.begin_step()
        i1, i2 = t1.infos[0], t2.infos[0]
        if i1.player_visible or i2.player_visible or i1.teammates or i2.teammates:
            break  # histories would legitimately differ once something becomes visible
        compared += 1
        m1, v1 = build_obs(i1, 0.0, None); m2, v2 = build_obs(i2, 0.0, None)
        assert (m1 == m2).all() and (v1 == v2).all()
        assert (i1.role, i1.dest) == (i2.role, i2.dest) and i1.build_packet() == i2.build_packet()
        mv = int(c1.map.best_action[c1.npc[0], cell(10, 2)])
        t1.commit_and_step([mv, 0, 0, 0], [False] * 4); t2.commit_and_step([mv, 0, 0, 0], [False] * 4)
        if c1.done or c2.done:
            break
    assert compared >= 4
    # critic state is allowed to differ
    from pursuit.env.observation import build_critic
    assert (build_critic(c1, c1.player, 0.0) != build_critic(c2, c2.player, 0.0)).any()


def test_actor_input_has_no_player_type_fields():
    from pursuit.env import observation as ob
    src = open(ob.__file__).read()
    start = src.index("def build_obs")
    end = src.index("def action_mask")
    body = src[start:end]
    for forbidden in ("phase", "xi", ".Q", "checkpoint", "player.", "core.pcell", "core.npc"):
        assert forbidden not in body


def test_role_hint_does_not_use_predictor():
    import inspect
    from pursuit.controllers import local
    src = inspect.getsource(local.LocalInfo._compute_role)
    assert "belief" not in src and "pred" not in src
