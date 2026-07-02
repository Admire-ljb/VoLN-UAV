@echo off
setlocal
cd /d %~dp0\..
if "%PYTHON%"=="" set PYTHON=C:\Users\ljb\.conda\envs\dinov3\python.exe
if "%CONFIG%"=="" set CONFIG=configs\train_adapter_dataset_release.yaml
if "%DEVICE%"=="" set DEVICE=cuda
set PYTHONPATH=src;D:\VoLN_dataset\.pydeps
"%PYTHON%" -m voln_uav.cli.train_adapter --config "%CONFIG%" --device %DEVICE%
