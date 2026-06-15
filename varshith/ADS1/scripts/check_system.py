import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.diagnostics import run_all_checks


def main() -> None:
    print(json.dumps(run_all_checks(), indent=2, default=str))


if __name__ == "__main__":
    main()
