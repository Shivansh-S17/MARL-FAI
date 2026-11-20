# Experiment protocol (frozen before the main matrix)

Task ID: **C0** (corridor, N=4, NPC radius 5, player radius 5, H=200, delta=0, standard spawn with NPC-player graph distance at least 4, base reward). Manifest: `configs/final_manifest.json` (120 runs). Task versions are never pooled.

## Training
- 3 waypoint levels (xi = 0, 0.25, 0.75, trained separately) x B in {0, 1/8, 1/2, 1} x {MARL, MARL-OM} x seeds 1000-1004 = 120 runs.
- 1,200,000 base-reward environment transitions per run; final checkpoint is the scientific checkpoint; no best-checkpoint selection.
- Curriculum: first 30k transitions start distance 1-3, next 60k distance 4-8, then the standard spawn distribution.
- Validation points (100 episodes, seeds from 20000) at 400k, 600k, 800k and 1.2M. Intermediate weights are not kept, so these are the only comparisons across checkpoints.
- Paired seeds: MARL and MARL-OM share seed IDs, initial actor and critic tensors, curriculum and environment streams.

## Evaluation
- Frozen waypoint: 100 fresh episodes per checkpoint, scenario seeds from 100000, greedy actions; IP and RC on the same scenarios.
- Q-player transfer (exploratory, out-of-distribution): xi=0.75-trained pursuers at every B, all five seeds, five held-out Q checkpoints (7100-7104), 20 episodes each, Q updates off, epsilon 0.10. Labelled as transfer from waypoint xi=0.75 training.
- Adaptive (exploratory): the same pursuers, 5 blocks of 20 consecutive episodes, five protocol cells (Q off/on, predictor none/frozen/online); the first two reuse the transfer episodes. Matched-opponent diagnostic: 20 fresh episodes against each block's final frozen Q-table.
- Q banks (seeds 7000-7002 and 7100-7104, 200k transitions each, RC at B=1) were trained under C0 with player radius 5 and remain valid.

## Statistics
- 5,000 bootstrap repetitions, analysis seed 40000. Resample paired seed groups, then matched scenarios; recompute data-dependent crossings and gaps each time; report undefined rates.
- Confirmatory family: G(1/8) = p1(1/8) - p0(1) style gains as defined in the plan, separately for xi = 0, 0.25, 0.75; two-sided paired sign-flip randomization, Holm at 0.05. Five seeds give coarse p-values.
- Q-player transfer, adaptive and online comparisons are exploratory and are never pooled with frozen waypoint results.
- Near-ceiling results are labelled as such: a zero budget difference at ceiling is not communication savings.

## Invalid-run handling
Runs with status other than complete stay listed with their status and logs. No seed is removed after results are seen. A systemic bug invalidates and reruns the affected paired cells under a new run ID.

## Scope statement
RC reaches near-full capture at B=0 in this task (calibration, `docs/pilot_report.md`), so the study measures when communication and opponent prediction add value in a setting with already strong baselines. No positive communication or opponent-modeling result is guaranteed.

## Resources
Server runs only inside `/home/gowtham.reddy/EEG/tam`, no existing file edited or deleted, 5 GB allowance (about 12 MB per run, about 1.5 GB for 120 runs plus evaluation traces), modest concurrency (12 processes, one process per run).
