@echo off
setlocal
cd /d %~dp0\..
if "%PYTHON%"=="" set PYTHON=C:\Users\ljb\.conda\envs\dinov3\python.exe
if "%CONFIG%"=="" set CONFIG=configs\eval_airsim_dataset_release.yaml
if "%BASELINE%"=="" set BASELINE=reference
if "%TRIALS%"=="" set TRIALS=10
if "%DEVICE%"=="" set DEVICE=cuda
if "%EVAL_MODE%"=="" set EVAL_MODE=normal
set "EVAL_MODE_ARGS=--control-mode move_to_position"
if /I "%EVAL_MODE%"=="normal" goto run
if /I "%EVAL_MODE%"=="fast" (
  set "EVAL_MODE_ARGS=--control-mode teleport --fast-reset --settle-sec 0.0 --max-teleport-step-m 10.0"
  goto run
)
echo Unsupported EVAL_MODE "%EVAL_MODE%". Use normal or fast.
exit /b 2

:run
set PYTHONPATH=src;D:\VoLN_dataset\.pydeps
"%PYTHON%" -m voln_uav.cli.eval_online_baselines --config "%CONFIG%" --baseline %BASELINE% --trials %TRIALS% %EVAL_MODE_ARGS% %*
