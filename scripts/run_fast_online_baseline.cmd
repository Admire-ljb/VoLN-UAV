@echo off
setlocal
cd /d %~dp0\..

if "%BASELINE%"=="" set BASELINE=random
if "%TRIALS%"=="" set TRIALS=10
if "%EPISODE_INDEX%"=="" set EPISODE_INDEX=0
if "%EPISODE_STRIDE%"=="" set EPISODE_STRIDE=1
if "%REFERENCE_STRIDE%"=="" set REFERENCE_STRIDE=1
if "%SETTLE_SEC%"=="" set SETTLE_SEC=0.0
if "%RANDOM_STEPS%"=="" set RANDOM_STEPS=80
if "%WORK_DIR%"=="" set WORK_DIR=runs\%BASELINE%_test_%TRIALS%_fast

set RANDOM_ARGS=
if /I "%BASELINE%"=="random" set RANDOM_ARGS=--random-steps %RANDOM_STEPS%

call "%~dp0run_online_baseline.cmd" ^
  --episode-index %EPISODE_INDEX% ^
  --episode-stride %EPISODE_STRIDE% ^
  --reference-stride %REFERENCE_STRIDE% ^
  %RANDOM_ARGS% ^
  --control-mode teleport ^
  --fast-reset ^
  --settle-sec %SETTLE_SEC% ^
  --work-dir "%WORK_DIR%" ^
  %*
