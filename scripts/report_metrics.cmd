@echo off
setlocal
cd /d %~dp0\..
if "%PYTHON%"=="" set PYTHON=python
if "%RUN_DIR%"=="" set RUN_DIR=runs\eval_offline_dataset_release
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
if "%PYTHONPATH%"=="" (
  set "PYTHONPATH=src"
) else (
  set "PYTHONPATH=src;%PYTHONPATH%"
)
"%PYTHON%" -m voln_uav.cli.report_metrics --run-dir "%RUN_DIR%" %EXTRA_ARGS%
