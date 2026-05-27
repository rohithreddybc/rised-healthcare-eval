"""Reproduce every numerical result and figure in the RISED paper.

Usage:
    python -m rised.reproduce_all

This entry point runs, in sequence:
  1. Synthetic-cohort baseline RISED evaluation
  2. Three external-validation cohorts (UCI Heart, Diabetes 130, NHIS 2024)
  3. Multi-model robustness comparison (XGBoost / LR / RF)
  4. Fairlearn comparison on the same cohort
  5. Two cross-domain demos (UCI Adult Income, German Credit) referenced
     in Section 5 of the paper

All scripts use random_state=42 and n_bootstrap=1000. Console output is
the primary deliverable; figures are written to figures/ by
generate_figures.py, which is invoked separately.

The script is deliberately written as a thin orchestrator that imports
and calls each example's `main()` rather than re-implementing the
evaluations, so that the published examples remain the single source
of truth for the numerical results in the paper.
"""

from __future__ import annotations

import importlib
import sys
import time
import traceback
from pathlib import Path

EXAMPLES = [
    ("examples.external_validation_uci_heart",
     "UCI Heart Disease external validation"),
    ("examples.external_validation_diabetes130",
     "UCI Diabetes 130-US Hospitals external validation"),
    ("examples.external_validation_nhis2024",
     "NCHS NHIS 2024 Sample Adult external validation"),
    ("examples.multi_model_robustness",
     "Multi-model robustness check (XGBoost / LR / RF)"),
    ("examples.fairlearn_comparison",
     "Fairlearn comparison on the synthetic cohort"),
    ("examples.adult_income_demo",
     "Cross-domain demo: UCI Adult Income"),
    ("examples.folktables_acs_income_demo",
     "Cross-domain demo: Folktables ACS-Income (Ding et al. 2021)"),
    ("examples.german_credit_demo",
     "Cross-domain demo: Statlog German Credit"),
]


def _add_repo_root_to_path() -> None:
    """Ensure `examples.*` is importable when this module is invoked from
    inside the rised package."""
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def main() -> int:
    _add_repo_root_to_path()
    print("=" * 72)
    print("RISED: reproducing all paper results")
    print("=" * 72)

    failed = []
    for module_name, description in EXAMPLES:
        print(f"\n--- {description} ---")
        t0 = time.time()
        try:
            module = importlib.import_module(module_name)
            module.main()
            print(f"    [OK] {time.time() - t0:.1f}s")
        except Exception as exc:  # noqa: BLE001 — surface everything
            print(f"    [FAIL] {exc}")
            traceback.print_exc()
            failed.append((module_name, exc))

    print("\n" + "=" * 72)
    if failed:
        print(f"Done with {len(failed)} failures:")
        for name, exc in failed:
            print(f"  - {name}: {exc}")
        return 1
    print("All evaluations completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
