# Pilot report

Task version for all results below: **C0** (corridor map, N=4, NPC radius 5, H=200, standard spawn, delta=0, base reward).
Original pilot runs used 2 training seeds (9000, 9001) per method, with 100 validation episodes per checkpoint.

## 1. What was run

| Stage | Runs | Steps | Notes |
| --- | --- | --- | --- |
| Base pilot | 16 | 300k, extended to 600k and 1.2M | 4 conditions x 2 methods x 2 seeds. |
| Shaped-reward pilot | 16 | 600k | Fresh seeds 9010, 9011. |
| Q-player bank | 8 | 200k each | Seeds 7000-7002 (training/validation), 7100-7104 (held-out). |

Server throughput (measured): MARL about 700 s and MARL-OM about 1700-2400 s per 300k-step run with 12 concurrent processes on two GPUs (acting on CPU, PPO updates on GPU). Each run directory is about 12 MB.

## 2. Original gate (retained, not rewritten)

| Step | Reference checkpoint | Conditions passing | Failing |
| --- | --- | --- | --- |
| 300k base | 200k | below the gate | wp75 stability, qbank |
| 600k base | 300k | 5 of 8 | wp75 MARL-OM (change 0.05), qbank x2 |
| 1.2M base | 600k | 5 of 8 | wp75 MARL-OM (change 0.05), qbank x2 |
| 600k shaped | 300k | 4 of 8 | wp0 MARL, wp75 MARL-OM, qbank x2 |

The reference checkpoints above are the ones `pilot_analysis.py` used. The plan names 400k/600k and 800k/1.2M; the revised report below uses those.
An earlier summary said 6 of 8 and 5 of 8; the recount is 5 of 8 and 4 of 8.
Qbank (Q-player training) did not learn under any variant (capture 0.01-0.19). It is no longer a training condition (Decision A) and these attempts remain recorded here.

## 3. Revised training-adequacy check (Decision D)

Source: `artifacts/results/pilot_adequacy_base.json`, produced by `python -m pursuit.training.adequacy base 9000,9001 1200000`.
Median over 2 seeds; checks: finite losses, at least 100 training captures, final capture at least 0.20, improvement over untrained at least 0.10, median change 800k to 1.2M not below -0.10.

| Condition | Method | Final capture | Improvement | Change 400k to 600k | Change 800k to 1.2M | Passed |
| --- | --- | --- | --- | --- | --- | --- |
| wp0, B=1 | MARL | 1.00 | 0.48 | +0.005 | 0.00 | yes |
| wp0, B=1 | MARL-OM | 1.00 | 0.48 | 0.00 | 0.00 | yes |
| wp75, B=1/8 | MARL | 0.76 | 0.27 | +0.06 | -0.01 | yes |
| wp75, B=1/8 | MARL-OM | 0.88 | 0.41 | +0.025 | +0.06 | yes (still improving: budget-limited, not shown converged) |
| wp25, B=0 | MARL | 1.00 | 0.55 | +0.005 | 0.00 | yes |
| wp25, B=0 | MARL-OM | 1.00 | 0.57 | 0.00 | 0.00 | yes |

All six required conditions pass. The 800k and 1.2M checkpoints cannot be re-evaluated on 300 fresh scenarios: only the latest checkpoint of each run was kept, so those comparisons rest on the saved 100-episode validation points.

## 4. Message-use audit and task calibration (Decision B)

Audit (`python -m pursuit.evaluation.calibration`, tests in `tests/test_message_audit.py`):
- B=0: no packet is sent or received and the receiver's target memory stays empty.
- B=1: a sighting packet is delivered, the receiver stores the correct cell and timestamp, and its role hint changes from search to block in the deterministic case.

Bug found during this audit and fixed: the player's visibility used the NPC radius table, so changing the NPC radius also changed the player's view. The player now has a separate radius-5 table (`env.player_radius`). Earlier radius 2 and 3 exploratory results are not used.

Calibration (RC, 300 matched episodes per xi, validation seeds from 50000; `artifacts/results/calibration_tasks/rc_candidates.csv`):

| Candidate | xi | Capture B=0 | Capture B=1 | Gain (95% CI) | Capped duration B=0 / B=1 | Promising |
| --- | --- | --- | --- | --- | --- | --- |
| C0 | 0 | 1.000 | 1.000 | 0.000 (0.000, 0.000) | 23.3 / 22.2 | no |
| C0 | 0.75 | 1.000 | 1.000 | 0.000 (0.000, 0.000) | 21.4 / 19.3 | no |
| C1 | 0 | 0.997 | 0.987 | -0.010 | 22.8 / 25.3 | no |
| C1 | 0.75 | 0.973 | 0.990 | +0.017 (0.000, 0.037) | 23.5 / 22.0 | no |
| C2 | 0 | 1.000 | 0.980 | -0.020 | 16.9 / 18.9 | no |
| C2 | 0.75 | 0.987 | 0.993 | +0.007 | 8.0 / 7.5 | no |
| C3 | 0 | 1.000 | 0.987 | -0.013 | 16.9 / 19.4 | no |
| C3 | 0.75 | 0.997 | 0.997 | 0.000 | 7.3 / 8.1 | no |

No candidate meets the criteria (capture at B=1 between 0.20 and 0.95 with a gain of at least 0.10 and a positive lower bound). Following the plan, the search stops after C3 and **C0 is the main task**. The study is framed as a test of when communication and opponent prediction add value in a setting where a rule-based team already captures nearly every episode.

Not yet done: the reliance diagnostic on a B=1 trained MARL checkpoint (packets suppressed on matched scenarios), which needs a main-matrix checkpoint.

## 5. Decisions recorded

- A: qbank removed from training and from the gate; the Q player is used for labelled out-of-distribution transfer and adaptation evaluation of waypoint xi=0.75 pursuers.
- B: audit, then bounded calibration C0-C3 on validation seeds; C0 retained as a null-result protocol.
- C: 120 runs, 5 seeds (1000-1004), no reduced variant.
- D: the 0.03 plateau rule is retired; learning and correctness checks retained; 1.2M base-reward transitions as the common budget.
These are post-pilot protocol changes, not preregistered.

## 6. Resource estimate for the main matrix (corrected)

From the measured pilot speeds (MARL about 2800 s and MARL-OM about 8000 s per 1.2M-step run at 12 concurrent processes), 60 + 60 runs need about 650,000 process-seconds, i.e. roughly 15 hours at 12 workers. The earlier 2-3 hour estimate used the 300k budget and is superseded. Storage: about 12 MB per run, about 1.5 GB for 120 runs, inside the 5 GB allowance (1.4 GB already used by the environment and earlier runs). Launched 12 workers across two GPUs; aggregate throughput is recorded in each run's metadata.
