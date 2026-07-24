@echo off
setlocal
cd /d %~dp0\..
if "%PYTHON%"=="" set PYTHON=python
if "%CONFIG%"=="" set CONFIG=configs\train_adapter_dataset_release.yaml
if "%DEVICE%"=="" set DEVICE=cuda
if "%PYTHONPATH%"=="" (
  set "PYTHONPATH=src"
) else (
  set "PYTHONPATH=src;%PYTHONPATH%"
)
"%PYTHON%" -m voln_uav.cli.train_adapter --config "%CONFIG%" --device %DEVICE%
