"""Copy the pacing and heat models from their sibling repos into src/tools/vendor/.

Each repo ships `<name>.py` at its root and `model/<name>.json`, plus a
`model/MODEL_READY` file holding the commit SHA of the final model. This copies
the module and its JSON, keeping the same relative layout, and records the
source repo and commit in `src/tools/vendor/SOURCE`.

    python -m scripts.vendor_models
"""

from __future__ import annotations

import shutil
import subprocess
from datetime import date

from src.tools.models import SIBLINGS, VENDOR

REPOS = {"pace_model": "lyhjeremy/marathon-pacing-decay",
         "heat_model": "lyhjeremy/marathon-heat-tax"}


def main() -> None:
    (VENDOR / "model").mkdir(parents=True, exist_ok=True)
    lines = [f"Vendored {date.today().isoformat()} by scripts/vendor_models.py."]
    for name, repo_dir in SIBLINGS.items():
        ready = repo_dir / "model" / "MODEL_READY"
        if not ready.exists():
            raise SystemExit(f"{ready} missing: the {name} model is not final yet")
        sha = ready.read_text().strip().split()[0]
        head = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
        shutil.copy2(repo_dir / f"{name}.py", VENDOR / f"{name}.py")
        copied = [f"{name}.py"]
        for js in sorted((repo_dir / "model").glob("*.json")):
            shutil.copy2(js, VENDOR / "model" / js.name)
            copied.append(f"model/{js.name}")
        lines.append(f"{name}: github.com/{REPOS[name]} MODEL_READY commit {sha} "
                     f"(checkout HEAD {head[:12]}); files: {', '.join(copied)}")
    (VENDOR / "SOURCE").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
