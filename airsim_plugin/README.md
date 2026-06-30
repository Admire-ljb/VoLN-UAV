# AirVoLNSimulatorServerTool

This script is the VoLN-UAV counterpart of TravelUAV's AirSim server launcher.

## Purpose

- keep a TravelUAV-style simulator entrypoint
- map scene ids to UE/AirSim executables
- reserve a port and optionally launch the executable
- allow quick replacement of the lightweight replay env with real OpenUAV/AirSim scenes

## Example

```bash
python airsim_plugin/AirVoLNSimulatorServerTool.py \
  --root_path /path/to/your/envs \
  --scene BrushifyUrban \
  --port 41451 \
  --dry_run
```

To override the default scene-to-executable mapping, provide a JSON file through `--mapping_json`.
