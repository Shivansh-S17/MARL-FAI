# Experiment protocol (to be frozen before final test analysis)

Status: DRAFT until the pilot decision is recorded in `docs/pilot_report.md`; then the training budget and reward
variant below are filled in and this file is not changed after final tests are evaluated.

## Frozen after pilot
- Common training budget (transitions / run): TBD from pilot. Reward variant: TBD (base or shaped, identical for both methods).

## Statistical units and contrasts
- Unit: paired trained-policy seed group (5 seeds, 1000-1004); within a group, matched scenario IDs.
  Learned-player evaluation preserves the five held-out player-checkpoint strata (20 episodes each).
- Frozen waypoint scenarios: seeds 100000-100099 (same for every method/seed/B; IP/RC reuse them).
- Frozen learned-player and adaptive blocks: seeds 200000 + 100*block + episode; matched diagnostic: 300000 + 100*block.
- Validation (pilot only): seeds from 20000, never used in final tests. Bootstrap: 5000 repetitions, seed 40000.
- Metrics: p0(B)=MARL capture, p1(B)=frozen MARL-OM capture, G(B)=p1-p0, Fgap(Blo,1)=G(Blo)/(p0(1)-p0(Blo)) for
  Blo in {0,1/8,1/2}, a=0.9*p0(1), S(a)=minB reaching a (MARL) minus minB (MARL-OM). Undefined values are counted, not
  discarded; negative gains and gap fractions >1 are kept; no interpolation or monotonic smoothing.
- Primary confirmatory family: G(1/8) for xi=0, xi=0.25, xi=0.75 (when xi=0.25 is in main scope) and frozen learned
  player. Two-sided sign-flip randomisation over five seed differences, Holm adjustment, alpha=0.05. With five seeds the
  smallest attainable two-sided p is 0.0625, so adjusted significance is not attainable; reported as a limitation.
- Low-budget compensation: one-sided 95% lower bound of p1(1/8)-p0(1) > -0.05; otherwise inconclusive (not equivalence).
- Adaptive blocks: resample paired seeds and whole blocks (20 ordered episodes); online-vs-frozen and early-vs-late are
  exploratory and never pooled with frozen primary results.
- Invalid runs: any run with status != complete is listed as failed/incomplete in tables; no seed is dropped after
  seeing results. Implementation faults invalidate and re-run affected paired cells under new run IDs.

## Figures
capture vs B (per player), actual packets vs capture/duration, adaptive curves, prediction quality, pilot curves,
labelled trajectory example. All rebuilt by `python -m pursuit.cli analyze` from saved CSV/JSON.
