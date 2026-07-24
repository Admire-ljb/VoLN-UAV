@echo off
setlocal
cd /d %~dp0\..
if "%PYTHON%"=="" set PYTHON=python
if "%PYTHONPATH%"=="" (
  set "PYTHONPATH=src"
) else (
  set "PYTHONPATH=src;%PYTHONPATH%"
)
"%PYTHON%" -m voln_uav.cli.launch_airsim %*
