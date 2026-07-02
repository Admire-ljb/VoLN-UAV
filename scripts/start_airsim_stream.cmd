@echo off
setlocal
cd /d %~dp0\..
if "%PYTHON%"=="" set PYTHON=C:\Users\ljb\.conda\envs\dinov3\python.exe
if "%CONFIG%"=="" set CONFIG=configs\eval_airsim_dataset_release.yaml
if "%HTTP_PORT%"=="" set HTTP_PORT=8765
if "%EPISODE_INDEX%"=="" set EPISODE_INDEX=0
if "%FPS%"=="" set FPS=8
set PYTHONPATH=src;D:\VoLN_dataset\.pydeps
"%PYTHON%" -m voln_uav.cli.stream_airsim --config "%CONFIG%" --episode-index %EPISODE_INDEX% --host 127.0.0.1 --http-port %HTTP_PORT% --fps %FPS%