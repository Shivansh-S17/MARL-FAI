# Opponent learning vs limited communication in cooperative pursuit

Four pursuer methods (IP, RC, MARL, MARL-OM), a scripted and a Q-learning player, budgeted 16-byte communication,
recurrent MAPPO with an explicit local opponent predictor, and an interactive replay viewer.

## Install and smoke-test (Windows PowerShell, Python 3.12)
```powershell
python -m venv .venv --system-site-packages   # reuse an installed CUDA torch wheel; or install torch yourself
.\.venv\Scripts\python -m pip install -r requirements-lock.txt
.\.venv\Scripts\python -m pip install -e . --no-deps
.\.venv\Scripts\python -m pursuit.cli check
.\.venv\Scripts\python -m pytest -q
```
Smoke runs are pipeline checks, not reproductions of the scientific results.

## Commands (`python -m pursuit.cli COMMAND`)
| command | purpose |
| --- | --- |
| check | validate maps, run a smoke episode and the PettingZoo API test |
| render-map | map + information diagram (PNG/PDF) |
| train-player | train the Q-learning player bank (seeds 7000-7002, 7100-7104) |
| pilot | validation pilot (`--total-steps`, `--shaping`, `--pilot-seeds`) |
| train | main/secondary manifest runs (`--run-ids pattern,...`, resumable) |
| evaluate | `--protocol baselines|frozen_waypoint|adaptive` |
| analyze | tables, bootstrap intervals, figures from saved CSVs |
| demo | interactive viewer (`--replay DIR_OR_FILE`) |
| export-demo | GIF + keyframes from recorded replays |

Manifest: `python -m pursuit.training.manifest STEPS [--shaping] [--reduced]` writes `configs/final_manifest.json`.
Completed runs are immutable (`status.json`); a changed implementation/config needs a new run ID.

## Layout
`src/pursuit/{env,opponents,controllers,models,training,evaluation,rendering}`, `docs/`, `tests/`,
`artifacts/{results,checkpoints,figures,logs}`, `evaluator_pack/`. See `docs/` for the contracts, protocol and traceability.
