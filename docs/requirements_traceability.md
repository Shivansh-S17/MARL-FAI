# Requirements traceability

Status key: **done** (implemented and checked), **running** (in progress), **pending** (not yet produced). Updated as work completes.

## Course requirements and reviewer feedback

| Requirement | Where it is met | Status |
| --- | --- | --- |
| Precise computational question | `finalReport/report.tex` (introduction); `evaluator_pack/START_HERE.md` | done |
| Explicit agents, actions, information, objectives | `docs/environment_contract.md`, `docs/observation_schema.md` | done |
| Meaningful implementation beyond a solver | `src/pursuit/` (simulator, channel, IP, RC, recurrent MAPPO, local predictor and belief) | done |
| Baselines | IP, RC (`controllers/baselines.py`), evaluated on the same scenarios as learned methods | done |
| Experiments varying constraints | Budget B in {0, 1/8, 1/2, 1} x waypoint randomness (`configs/final_manifest.json`, 120 runs) | running |
| Quantitative evaluation | Capture, capped duration, actual packets, bootstrap intervals (`evaluation/analysis.py`) | done; results in tables |
| Analysis of results and limitations | `finalReport/report.tex` | done |
| Reviewer: player repeats a scripted route | Tabular Q-learning player with frozen and adapting modes (`opponents/players.py`, `docs/design_decisions.md`) | done; transfer and adaptive evaluation run (40 runs) |
| Reviewer: 300k steps may not suffice | Pilot to 1.2M, revised adequacy check, 1.2M main budget (`docs/pilot_report.md`) | done |
| Reviewer: a map would help | `artifacts/figures/map_information.png/.pdf`, replay viewer, scenes, GIF | map done; viewer done; scenes 1-4 built from final checkpoints, 90 s GIF exported |

## Plan sections

| Plan item | Evidence | Status |
| --- | --- | --- |
| 3.1 map, motion, capture, termination | `env/core.py`, `env/geometry.py`; `tests/test_env.py` | done |
| 3.2 local visibility, one actor adapter, no hidden state | `env/observation.py`, `controllers/local.py`; `tests/test_channel_info.py` | done |
| 3.3 budget rule, 16-byte packet, delay, charging | `env/channel.py`; `tests/test_channel_info.py` | done |
| 3.4 rewards and shaping | `env/core.py`; `tests/test_learning.py` | done |
| 4.1 waypoint player | `opponents/players.py` | done |
| 4.2 Q player, eight banks, separate player radius | `opponents/train_player.py`; `env.player_radius`; `tests/test_task_variants.py` | done |
| 5.1 target memory and role hints | `controllers/local.py` | done |
| 5.4 recurrent MAPPO | `models/networks.py`, `training/trainer.py`; `tests/test_learning.py` | done |
| 5.5 local predictor and belief | `models/predictor.py`, `models/belief.py` | done |
| 6.1 message-use audit | `evaluation/calibration.py`; `tests/test_message_audit.py` | done; reliance diagnostic: protocol `reliance` (results in `eval_reliance.csv` per run) |
| 6.1 bounded calibration C0-C3 | `artifacts/results/calibration_tasks/rc_candidates.csv`; `docs/pilot_report.md` section 4 | done (C0 retained) |
| 6.2 revised adequacy | `training/adequacy.py`; `artifacts/results/pilot_adequacy_base.json` (server) | done |
| 6.3 resources | `docs/pilot_report.md` section 6 | done |
| 7.1 main matrix | `configs/final_manifest.json`; `artifacts/results/main/` (server) | running |
| 7.2 frozen waypoint evaluation | `evaluation/protocols.py` | done; run on all main runs |
| 7.3 Q-player transfer | `evaluation/protocols.py` (`is_transfer_run`) | done; run on all main runs |
| 7.4 adaptive and matched diagnostic | `evaluation/protocols.py` | done; run on all main runs |
| 7.5 shuffled-feature control | `training/manifest.py` (`secondary_runs`) | defined; runs after the main matrix |
| 8 statistics | `evaluation/analysis.py`, `docs/experiment_protocol.md` | done; results in tables |
| 9.1-9.2 renderer, trace format, viewer | `rendering/`; `tests/test_rendering.py` | done |
| 9.3 comparison scenes 1-4 | `evaluation/scenes.py`, CLI `scenes` | done; built from main-run checkpoints |
| 9.4 GIF, slides, evaluator pack | `rendering/export.py`, `evaluator_pack/` | done; contents assembled |
| 10 tests | `tests/` (41 pass) | done |
| 11 report and figures | `finalReport/` | done (report.pdf, slides.pdf) |
| 12 phases | Phases 0-7 done; Phase 8 (delivery checks) in progress | see `docs/pilot_report.md` |
