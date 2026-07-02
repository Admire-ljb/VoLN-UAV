@echo off
setlocal
cd /d %~dp0\..
if "%PYTHON%"=="" set PYTHON=C:\Users\ljb\.conda\envs\dinov3\python.exe
if "%CONFIG%"=="" set CONFIG=configs\eval_airsim_dataset_release.yaml
if "%BASELINE%"=="" set BASELINE=reference
if "%TRIALS%"=="" set TRIALS=10
if "%DEVICE%"=="" set DEVICE=cuda
set PYTHONPATH=src;D:\VoLN_dataset\.pydeps
"%PYTHON%" -m voln_uav.cli.eval_online_baselines --config "%CONFIG%" --baseline %BASELINE% --trials %TRIALS% %*
