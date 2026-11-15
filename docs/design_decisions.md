# Design decisions after the idea review

These are final-project developments, not edits to the submitted idea review (`projectIdea/` is untouched).

## 1. A genuinely learning player
The submitted idea used a scripted route repeater. The reviewer pointed out it never learns. The scripted waypoint
player remains as the controlled baseline (xi = 0, 0.25, 0.75), and a tabular Q-learning player is added
(`opponents/players.py`). It observes NPCs only inside its own radius-5 line of sight, has the same speed and actions,
and receives no packets or controller state. Q updates happen after every transition when learning is enabled.
Eight initial checkpoints are trained against RC at B=1 (3 training bank seeds 7000-7002, 5 held-out seeds 7100-7104,
200k transitions each). Frozen mode (Q updates off) and adapting mode (fresh clone per block, Q updates on) are both
evaluated. A weak learned player is a measured limitation, not hidden.

## 2. Evidence-based training budget
Because captures can be rare, 300k steps was not assumed sufficient. A pilot (seeds 9000/9001, four conditions, both
methods, validation at 0/50k/100k/200k/300k) applies an operational gate. See `docs/pilot_report.md`. The selected
budget and reward variant are frozen for every main run.

## 3. Separate online-prediction evaluation
Opponent-model predictors can be frozen or updated online during evaluation. Frozen and online results are separate
protocol cells and never pooled. Online-versus-frozen and early-versus-late contrasts are exploratory.

## 4. Visual demonstration
A shared renderer (pygame display / headless PNG / GIF) and a trace format that records only evaluation-recorder data
provide an interactive viewer and portable recordings built from the real simulator, not a separate one.

## 5. Implementation choices that fill gaps in the plan (documented, not silent)
- Search "visited" = a cell was inside this NPC's visibility mask at some step (including its own cell).
- Role-hint tie-break "coordinate order" = lowest cell index (y*20+x).
- In the arena map (no doorways) the block candidate is omitted.
- Predictor belief rows with mass < 1e-9 are skipped during propagation (speed); the belief is renormalised.
- The shuffled-feature control shuffles the seven feature vectors across (agent, environment) slots at each step before
  the actor sees them, so stored log-probabilities remain consistent with the data PPO trains on.
- Probe-set greedy-action changes use the unmasked arg-max of a Q row (legality depends on the cell, which the
  aggregated state does not determine).
- Q-player pretraining stops at the first episode boundary after 200,000 transitions; the actual step counts
  (200,003-200,187) are stored in `artifacts/checkpoints/player_bank/*.json`.
- Rollouts are collected with a CPU copy of the actor; PPO updates run on the GPU. Training environments run in one
  process in lockstep (8 environments, 128 steps per rollout; last rollout shortened so the budget is not exceeded).
- Checkpoint resume restores model/optimiser/predictor/RNG state but restarts environments at a fresh episode
  boundary; the stream therefore differs from an uninterrupted run (stated, not claimed exact).
