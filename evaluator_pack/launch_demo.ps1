# Opens the replay viewer with the bundled replays. Run explicitly; it installs nothing, trains nothing, needs no internet.
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$proj = Split-Path -Parent $here
$py = Join-Path $proj ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
$env:PYTHONPATH = (Join-Path $proj "src")
& $py -m pursuit.cli demo --replay (Join-Path $here "replays")
