"""Run the independent model smoke test with python -m pymind.independent."""

import json

from pymind.independent.factory import run_smoke_test


if __name__ == "__main__":
    print(json.dumps(run_smoke_test(), indent=2))
