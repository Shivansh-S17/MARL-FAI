# Pilot decision brief

Internal working note for the team. It records what the plan asked for, what we ran, where we stand, and the decisions we need to make before the main experiments start.

## 1. Where we stand against the plan

| Phase | Status |
| --- | --- |
| 0. Contracts and setup | Done. Traceability document still to write. |
| 1. Environment and first picture | Done. |
| 2. Baselines, Q player, viewer | Done. Only comparison scene 1 of 4 is built. |
| 3. MARL and MARL-OM | Done. 37 tests pass, plus extra checks (PPO/GAE, determinism, kill-and-resume, extension). |
| 4. Pilot | Run through the plan's whole ladder. Gate not yet passed (section 2). |
| 5. Protocol freeze | Blocked on the decisions in section 4. |
| 6. Main experiments | Not started. About 2-3 hours of server compute for 120 runs. |
| 7. Analysis and visual pack | Analysis code written. Scenes 2-4, GIF, evaluator pack still to do. |
| 8. Final delivery | Not started: LaTeX report, Beamer slides, reproduction check. |

Roughly 40-45% of the total work is complete. The build is finished and verified. The experiments and deliverables remain.

## 2. What the pilot ladder did

The plan says: train at 300k steps, extend to 600k and then 1.2M if the gate fails, then try the shaped-reward ladder with fresh seeds. Gate per condition: at least 100 training captures, median validation capture at least 0.20, improvement of at least 0.10 over the untrained policy, and a change of at most 0.03 from the previous checkpoint.

All 16 runs per ladder step used 2 seeds, MARL and MARL-OM, 4 conditions, on the GPU server.

| Step | Conditions passing | Failing |
| --- | --- | --- |
| 300k, base reward (reference 200k) | below the gate | wp75 stability, qbank |
| 600k, base reward (reference 300k) | 5 of 8 | wp75 MARL-OM (change 0.05), qbank both |
| 1.2M, base reward (reference 600k) | 5 of 8 | wp75 MARL-OM (change 0.05), qbank both |
| 600k, shaped reward, seeds 9010-9011 (reference 300k) | 4 of 8 | wp0 MARL (change -0.04), wp75 MARL-OM (change 0.19), qbank both |

Final capture rates (median over 2 seeds):

| Condition | MARL, 1.2M base | MARL-OM, 1.2M base | MARL, 600k shaped | MARL-OM, 600k shaped |
| --- | --- | --- | --- | --- |
| wp0 | 1.00 | 1.00 | 0.96 | 1.00 |
| wp25 | 1.00 | 1.00 | 1.00 | 1.00 |
| wp75 | 0.76 | 0.88 | 0.78 | 0.96 |
| qbank | 0.01 | 0.08 | 0.14 | 0.09 |

Findings:

- Both learners reach full capture against the stationary and mildly random waypoint players (wp0, wp25) and learn strongly against wp75.
- The wp75 MARL-OM miss is a stability criterion only (and the gate's reference checkpoints were 300k and 600k, not the plan's original 400k and 600k / 800k and 1.2M). Its capture is 0.83-0.96 and still rising slowly.
- Against the Q-learning player (qbank) neither method learns in any variant: 300k, 600k, 1.2M, or shaped. The rule baselines reach 0.74-0.98 there, so the task is solvable.
- Shaping did not improve the gate result.

## 3. Open issue: communication does not matter yet

Calibration sweeps over radius (2, 3, 5), team size and horizon found that capture with no communication (B=0) matches capture with full communication (B=1) for the IP and RC rule baselines and for learned MARL. The plan's main contrast asks whether opponent modelling can replace communication. In this task that contrast would be about zero, because communication has no measurable value to replace.

You chose earlier to make the environment harder so communication matters. That change is not yet designed or implemented. The plan requires a documented revised protocol and a fresh pilot for any task change.

## 4. Decisions we need to make

### Decision A: how to handle the qbank condition

1. **Evaluation-only opponent.** Train learned teams against the waypoint players, which pass the gate. Use the Q player only at evaluation. Quickest. Needs a one-paragraph protocol change.
2. **Make the Q player easier to learn against.** For example slower speed or a fixed training bank. Needs calibration and a fresh pilot (several hours).
3. **Fold it into the task redesign (Decision B).** Retrain the Q bank in the new setting and pilot it again.

### Decision B: redesign for a communication bottleneck

Options for the redesign, all needing calibration with rule baselines first:

1. Visibility-limited information sharing, so a pursuer cannot see what teammates see unless a packet is sent.
2. A smaller radius with a larger map region.
3. Keep the current task and report the contrast as measured.

Any choice here resets the baselines and the Q bank, and needs a revised protocol, a fresh pilot, then the freeze.

### Decision C: main matrix size

- 120 runs, 5 seeds (recommended): about 2-3 hours on the server, roughly 1.6 GB of the 5 GB allowance.
- 3-seed variant: faster, with less statistical power.

### Decision D: stability criterion for wp0 and wp75

The remaining failures are stability only. Options: accept the plan gate as written and continue on the passing conditions, or record a revised stability tolerance in the protocol with the measured change values.

## 5. Recommendation

1. Do Decision A option 1 now, so the main matrix is not blocked by the one condition that does not learn.
2. Run cheap rule-baseline sweeps for the Decision B redesign next, then a short learned pilot. Use the result to decide whether to keep the base task.
3. Use the 1.2M base reward for the main matrix (the base reward passed more conditions than shaping).
4. Choose 120 runs with 5 seeds.
5. Write the revised experiment protocol, then freeze it and generate the final manifest.

## 6. Remaining work after the decisions

- Pilot report with measured throughput.
- Requirements traceability.
- Frozen protocol and final manifest.
- Main matrix on both GPUs (about 24 workers), then frozen, adaptive and online evaluation.
- Bootstrap analysis, comparison scenes 2-4, demo GIF and keyframes.
- Evaluator pack, LaTeX report, Beamer slides, clean-directory reproduction check.

## Correction note

An earlier version of this brief and of the chat summaries said 6 of 8 (base) and 5 of 8 (shaped) conditions passed. Recounting the gate rows gives 5 of 8 and 4 of 8. The shaped ladder ran to 600k only, not 1.2M.
