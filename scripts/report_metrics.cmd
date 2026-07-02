@echo off
setlocal
cd /d %~dp0\..
if "%PYTHON%"=="" set PYTHON=C:\Users\ljb\.conda\envs\dinov3\python.exe
if "%RUN_DIR%"=="" set RUN_DIR=D:\VoLN_dataset\VoLN-UAV-runs\eval_offline_dataset_release
if "%~1"=="" goto collect_args
set FIRST=%~1
if "%FIRST:~0,1%"=="-" goto collect_args
set RUN_DIR=%~1
shift
:collect_args
set EXTRA_ARGS=
:collect_loop
if "%~1"=="" goto run
set EXTRA_ARGS=%EXTRA_ARGS% %1
shift
goto collect_loop
:run
set PYTHONPATH=src;D:\VoLN_dataset\.pydeps
"%PYTHON%" -m voln_uav.cli.report_metrics --run-dir "%RUN_DIR%" %EXTRA_ARGS%