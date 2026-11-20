from pursuit.evaluation.calibration import audit, deterministic_counterexample


def test_b0_is_silent_and_receiver_stays_uninformed():
    c = deterministic_counterexample("0")
    assert c["npc0_sees"] and not any(c["sends_t0"])
    assert c["npc1_received"] == 0 and c["npc1_mem_after"] == -1


def test_b1_packet_updates_memory_and_role_hint():
    a = audit()
    assert a["pass_B0_silent"] and a["pass_B1_delivers"] and a["pass_role_hint_changes"]
