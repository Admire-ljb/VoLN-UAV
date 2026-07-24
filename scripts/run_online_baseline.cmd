@echo off
setlocal
cd /d %~dp0\..
if "%PYTHON%"=="" set PYTHON=python
if "%CONFIG%"=="" set CONFIG=configs\eval_airsim_dataset_release.yaml
if "%EPISODES_FILE%"=="" set EPISODES_FILE=episodes.jsonl
if "%BASELINE%"=="" set BASELINE=random
if "%TRIALS%"=="" set TRIALS=10
if "%DEVICE%"=="" set DEVICE=cuda
if "%EVAL_MODE%"=="" set EVAL_MODE=normal
if "%SCENE%"=="" set SCENE=BrushifyUrban
set "EVAL_MODE_ARGS=--control-mode move_to_position"
set "REFERENCE_BOOTSTRAP_ARGS="
if /I "%BASELINE%"=="reference" set "REFERENCE_BOOTSTRAP_ARGS=--reference-bootstrap-steps 2"
if /I "%EVAL_MODE%"=="normal" (
  if /I "%BASELINE%"=="reference" set "EVAL_MODE_ARGS=--control-mode move_on_path"
  goto run
)
if /I "%EVAL_MODE%"=="fast" (
  set "EVAL_MODE_ARGS=--control-mode teleport --fast-reset --settle-sec 0.0 --max-teleport-step-m 10.0"
  goto run
)
if /I "%EVAL_MODE%"=="exact" (
  set "EVAL_MODE_ARGS=--control-mode teleport --fast-reset --settle-sec 0.0 --max-teleport-step-m 100.0 --max-teleport-vertical-step-m 100.0"
  goto run
)
echo Unsupported EVAL_MODE "%EVAL_MODE%". Use normal, exact, or fast.
exit /b 2

:run
if "%PYTHONPATH%"=="" (
  set "PYTHONPATH=src"
) else (
  set "PYTHONPATH=src;%PYTHONPATH%"
)
"%PYTHON%" -m voln_uav.cli.eval_online_baselines --config "%CONFIG%" --episodes-file "%EPISODES_FILE%" --baseline %BASELINE% --trials %TRIALS% --scene "%SCENE%" %REFERENCE_BOOTSTRAP_ARGS% %EVAL_MODE_ARGS% %*
