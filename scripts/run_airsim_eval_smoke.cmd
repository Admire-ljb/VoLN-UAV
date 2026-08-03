@echo off
setlocal
cd /d %~dp0\..
if "%PYTHON%"=="" set PYTHON=python
if "%CONFIG%"=="" set CONFIG=configs\debug\eval_airsim_smoke.yaml
if "%DEVICE%"=="" set DEVICE=cuda
if "%PYTHONPATH%"=="" (
  set "PYTHONPATH=src"
) else (
  set "PYTHONPATH=src;%PYTHONPATH%"
)
"%PYTHON%" -m voln_uav.cli.eval_airsim --config "%CONFIG%" --device %DEVICE%
