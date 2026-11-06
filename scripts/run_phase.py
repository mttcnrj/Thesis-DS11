"""Run a pipeline phase as a script (Section 2.3: long jobs run as scripts, not notebook cells).

    python scripts/run_phase.py --phase 1            # Notebook 01: parse, QC, coverage, eval hours (G1 G2 G3 G5 G6)
    python scripts/run_phase.py --phase 1 --tag      # also tag `phase1-frozen` when G6 passes
    python scripts/run_phase.py --phase 2            # Notebook 02: legend G4, G20, link analysis, H1 (G7), alert levels (G8)
    python scripts/run_phase.py --phase 3            # Notebook 03: tide (G9), features, train/val windows, scalers
    python scripts/run_phase.py --phase 4            # Notebook 04: Granger selection (G10, H2) -> configs/selected_features.yaml
    python scripts/run_phase.py --phase 5 [--stage S] [--tag]   # Notebook 05 (laptop): G11-G17, G19
    PASIG_PROFILE=rehearsal python scripts/run_phase.py --phase 5   # tiny CPU dry run, outputs under runs/rehearsal/
    python scripts/run_phase.py --phase 6 [--stage S] [--rerun-reason "..."]   # Notebook 06: G18, opens 2024 ONCE
    PASIG_PROFILE=rehearsal python scripts/run_phase.py --phase 6   # dry run on the 2023 stand-in (2024 never read)
    python scripts/run_phase.py --phase 7 [--stage S]   # Notebook 07: walk-forward, ablations, case study, Aviles 2024
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pasig.config import root  # noqa: E402
from pasig.gates import require_gates  # noqa: E402

REQUIRED_BEFORE = {1: [], 2: ["G6"], 3: ["G6", "G7", "G8"], 4: ["G6", "G9"], 5: ["G0", "G6", "G8", "G9", "G10"], 6: ["G17"], 7: ["G18"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, required=True)
    ap.add_argument("--no-tests", action="store_true", help="skip the phase tests (G6 will not pass)")
    ap.add_argument("--tag", action="store_true", help="create the phase git tag when its freeze gate passes")
    ap.add_argument("--stage", help="phase 5: compute, variants, grid, finals, baselines, thresholds, checks, freeze; "
                                    "phase 6: unseal, predict, evaluate; phase 7: walkforward, ablations, casestudy, aviles")
    ap.add_argument("--rerun-reason", help="phase 6 only: required to open the test season a second time (logged)")
    a = ap.parse_args()
    if not (a.phase in (5, 7) and a.stage):
        require_gates(*REQUIRED_BEFORE.get(a.phase, []))
    if a.phase == 1:
        from pasig import phase1
        out = phase1.run(run_tests=not a.no_tests)
        print(out["table_iii6"].set_index("split").T.to_string())
        print("eligible inputs:", out["coverage"]["eligible"] or "none")
        if out.get("g6") and a.tag:
            subprocess.check_call(["git", "tag", "-f", "phase1-frozen"], cwd=root())
            print("tagged phase1-frozen")
        return 0 if out.get("g6", a.no_tests) else 1
    if a.phase == 2:
        from pasig import phase2
        out = phase2.run()
        print("H1:", out["h1"]["verdict"], "| alert levels:", out["alert"]["high_water_m"], out["alert"]["flood_risk_m"])
        code = 0 if a.no_tests else subprocess.call([sys.executable, "-m", "pytest", "-q", "tests/test_link.py"], cwd=root())
        return code
    if a.phase == 3:
        from pasig import phase3
        out = phase3.run()
        print(out["summary"].to_string(index=False))
        print("G9:", {k: v["added"] for k, v in out["tide"]["evidence"].items()})
        tests = ["tests/test_windows.py", "tests/test_leakage.py"]
        return 0 if a.no_tests else subprocess.call([sys.executable, "-m", "pytest", "-q", *tests], cwd=root())
    if a.phase == 4:
        from pasig import phase4
        out = phase4.run()
        print(out["table"].to_string(index=False))
        print("G10:", out["outcome"], "| selected:", out["selected"], "| H2:", out["h2"])
        tests = ["tests/test_granger.py", "tests/test_leakage.py"]
        return 0 if a.no_tests else subprocess.call([sys.executable, "-m", "pytest", "-q", *tests], cwd=root())
    if a.phase == 5:
        from pasig import phase5
        if a.stage:
            phase5.run_stage(a.stage, tag=a.tag)
            return 0
        code = phase5.run(tag=a.tag)
        print(phase5.summary())
        return code
    if a.phase == 6:
        import os

        from pasig import phase6
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")      # inference only; CPU keeps the scores reproducible
        out = phase6.run([a.stage] if a.stage else None, rerun_reason=a.rerun_reason)
        if "evaluate" in out:
            print("H3:", out["evaluate"]["H3"]["verdict"], "| H4:", out["evaluate"]["H4"]["verdict"],
                  "| strongest simple baseline:", out["evaluate"]["strongest_simple_baseline"])
        tests = ["tests/test_metrics.py", "tests/test_events.py", "tests/test_dm.py", "tests/test_unseal.py",
                 "tests/test_same_hours.py"]
        return subprocess.call([sys.executable, "-m", "pytest", "-q", *tests], cwd=root())
    if a.phase == 7:
        from pasig import phase7
        if a.stage:
            phase7.run_stage(a.stage)
            return 0
        code = phase7.run()
        if code == 0:
            tests = ["tests/test_same_hours.py", "tests/test_walkforward.py"]
            code = subprocess.call([sys.executable, "-m", "pytest", "-q", *tests], cwd=root())
        return code
    raise SystemExit(f"phase {a.phase} is not implemented yet")


if __name__ == "__main__":
    raise SystemExit(main())
