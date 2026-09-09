"""Toolbox maintenance. Does not submit jobs or access research data."""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
def check():
    for p in (ROOT / "src").rglob("*.py"):
        ast.parse(p.read_text(), filename=str(p))
    for name in ("__init__.py", "core.py", "utils.py"):
        if not (ROOT / "src/mhd_framework" / name).is_file():
            raise ValueError("Missing public module")
    if any((ROOT / "src/mhd_framework" / name).is_dir() for name in ("v4", "v5")):
        raise ValueError("One implementation per installed release")
    return {"structure": True, "api": "V4"}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "test"))
    args = parser.parse_args()
    print(json.dumps(check()))
    if args.command == "test":
        subprocess.run([sys.executable, "-m", "pytest"], cwd=ROOT, check=True)
if __name__ == "__main__":
    main()
