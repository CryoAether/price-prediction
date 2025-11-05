import sys
from pathlib import Path

# Ensure the src/ layout package is importable in all test contexts
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
