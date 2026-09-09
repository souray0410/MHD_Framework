import runpy
from pathlib import Path

def test_quick_start_executes():
    root = Path(__file__).resolve().parents[2]
    runpy.run_path(str(root / "examples/basic.py"), run_name="__main__")
