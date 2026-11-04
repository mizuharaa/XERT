"""The readiness check reads the environment only: no ports opened, no hardware modules imported."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_report_lists_modules_tools_and_injected_ports():
    sys.path.insert(0, str(ROOT))
    from scripts.hw_env_check import MODULES, TOOLS, report
    lines, _ready = report(comports=[("COM7", "USB Serial Device")])
    text = "\n".join(lines)
    assert all(name in text for name in MODULES) and all(tool in text for tool in TOOLS)
    assert "COM7" in text and "map-from-calibration" in text


def test_check_never_imports_hardware_modules():
    code = ("import sys; sys.path.insert(0, %r); from scripts.hw_env_check import report; report(comports=[]); "
            "bad = sorted(m for m in sys.modules if m.startswith(('lerobot.robots', 'lerobot.motors'))); "
            "print('BAD', bad)" % str(ROOT))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    assert "BAD []" in out.stdout, out.stdout
