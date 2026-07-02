@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$procs = Get-CimInstance Win32_Process -Filter \"name = 'python.exe'\" | Where-Object { $_.CommandLine -like '*voln_uav.cli.stream_airsim*' }; foreach ($p in $procs) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }; $cmds = Get-CimInstance Win32_Process -Filter \"name = 'cmd.exe'\" | Where-Object { $_.CommandLine -like '*run_stream.cmd*' -or $_.CommandLine -like '*start_airsim_stream.cmd*' -or $_.CommandLine -like '*stream_airsim*' }; foreach ($p in $cmds) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }"
