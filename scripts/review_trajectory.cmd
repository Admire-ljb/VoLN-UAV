@echo off
setlocal
cd /d %~dp0\..
if "%PYTHON%"=="" set PYTHON=C:\Users\ljb\.conda\envs\dinov3\python.exe
if "%CONFIG%"=="" set CONFIG=configs\eval_airsim_dataset_release.yaml
if "%EPISODE_INDEX%"=="" set EPISODE_INDEX=0
if "%LOOPS%"=="" set LOOPS=1
if "%DELAY_SEC%"=="" set DELAY_SEC=0.7
if "%STRIDE%"=="" set STRIDE=1
set PYTHONPATH=src;D:\VoLN_dataset\.pydeps
"%PYTHON%" -m voln_uav.cli.review_trajectory --config "%CONFIG%" --episode-index %EPISODE_INDEX% --loops %LOOPS% --delay-sec %DELAY_SEC% --stride %STRIDE%
