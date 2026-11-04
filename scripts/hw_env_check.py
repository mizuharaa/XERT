"""Read-only readiness check for the real SO-101: modules, CLI tools, calibration dir, serial ports.

Never opens a serial port or a camera, and never imports lerobot.robots / lerobot.motors (the
import-graph test in tests/test_hw_env_check.py pins that). Enumerating ports is a registry read.
"""
import importlib.util
import os
import shutil
import sys
from pathlib import Path

MODULES = ("lerobot", "scservo_sdk", "serial", "deepdiff", "cv2", "mediapipe", "mujoco")
TOOLS = ("lerobot-find-port", "lerobot-calibrate", "lerobot-record", "lerobot-teleoperate", "lerobot-find-cameras")


def report(comports=None):
    """Return (lines, ready). comports is injectable for tests; default enumerates via pyserial."""
    lines, ready = [], True
    for name in MODULES:
        ok = importlib.util.find_spec(name) is not None
        ready &= ok
        lines.append(f"module  {name:<14} {'ok' if ok else 'MISSING'}")
    scripts = Path(sys.executable).parent
    for tool in TOOLS:
        ok = shutil.which(tool, path=str(scripts)) is not None
        ready &= ok
        lines.append(f"tool    {tool:<22} {'ok' if ok else 'MISSING'}")
    calib = Path(os.environ.get("HF_LEROBOT_CALIBRATION", Path.home() / ".cache/huggingface/lerobot/calibration"))
    followers = sorted((calib / "robots/so_follower").glob("*.json")) if calib.exists() else []
    lines.append(f"calib   {calib} {'exists' if calib.exists() else 'not yet created (lerobot-calibrate makes it)'}")
    lines.append(f"calib   follower files: {[p.name for p in followers] or 'none yet'}")
    if comports is None:
        import serial.tools.list_ports
        comports = [(p.device, p.description) for p in serial.tools.list_ports.comports()]
    lines.append(f"ports   {comports or 'none (plug the arm in; the STS3215 bus shows as a USB serial port)'}")
    lines.append("")
    lines.append("Next, in order (see docs/handoff/F1-map-from-calibration.md):")
    lines.append("  lerobot-find-port                      # identify the follower and leader ports")
    lines.append("  lerobot-calibrate --robot.type=so101_follower --robot.port=<PORT> --robot.id=<id>")
    lines.append("  lerobot-calibrate --teleop.type=so101_leader --teleop.port=<PORT> --teleop.id=<id>")
    lines.append("  lerobot-record ... --robot.cameras='{scene: ..., wrist: ...}'   # two still poses, then demos")
    lines.append("  python -m xert map-from-calibration --calibration <robots/so_follower/<id>.json> --poses poses.json --output <fresh>")
    return lines, ready


def main():
    lines, ready = report()
    print("\n".join(lines))
    print("\nenvironment:", "READY (software side; hardware still needs the arm)" if ready else "NOT READY")
    return 0 if ready else 1


if __name__ == "__main__":
    sys.exit(main())
