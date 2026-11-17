#!/usr/bin/env bash
# Play every selected LeRobot v3 episode; --mode both also runs EE playback.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
isaac_python="${ISAACLAB_PYTHON:-$HOME/IsaacLab/.venv/bin/python}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
exec "$isaac_python" "$repo_root/playback.py" --mode joint "$@"
