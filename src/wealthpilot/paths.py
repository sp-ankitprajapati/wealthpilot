"""
Single source of truth for where data files live, relative to this
package's location on disk — so moving the repo doesn't break every
module that reads eval_cases.json with its own guessed relative path.
"""

from pathlib import Path

# src/wealthpilot/paths.py -> parent=wealthpilot, .parent=src, .parent=repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
if not (DATA_DIR / "eval_cases.json").is_file():
    DATA_DIR = Path(__file__).resolve().parent / "data"
EVAL_CASES_PATH = DATA_DIR / "eval_cases.json"
