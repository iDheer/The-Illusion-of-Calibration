"""
run_full_pipeline.py - Run-3: Full Netra-Adapt Pipeline

Run-3 changes:
- All result paths point to /workspace/results_run3/
- Banner updated to reflect Run-3 (Grayscale + Balanced)

Usage:
    python run_full_pipeline.py

Output:
    logs/run_YYYY-MM-DD_HH-MM-SS/
    ├── experiment_log.txt
    ├── metadata.json
    ├── EXPERIMENT_SUMMARY.md
    ├── 01_source_training/
    ├── 02_oracle_training/
    ├── 03_adaptation/
    ├── 04_evaluation/
    └── 05_advanced_analysis/
"""

import sys
import subprocess
import time
from datetime import datetime
from training_logger import get_logger, reset_logger


def run_script(script_name, description):
    """Run a Python script and track execution time."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*80}\n")

    start_time = time.time()
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,
            text=True
        )
        elapsed = time.time() - start_time
        print(f"\n✅ {description} completed in {elapsed/60:.1f} minutes")
        return True, elapsed
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n❌ {description} failed after {elapsed/60:.1f} minutes")
        print(f"   Error: {e}")
        return False, elapsed


def main():
    print("\n" + "="*80)
    print("   NETRA-ADAPT RUN-3: FULL EXPERIMENTAL PIPELINE")
    print("   Cross-Ethnic Glaucoma Screening — Grayscale + Balanced Training")
    print("="*80)
    print(f"\nExperiment Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    reset_logger()
    exp_logger = get_logger()
    print(f"\n📊 Logging to: {exp_logger.run_dir}")

    total_start = time.time()
    results = {}

    # Step 1: Prepare Data
    success, elapsed = run_script("prepare_data.py", "Data Preparation (with balance reporting)")
    results["prepare_data"] = {"success": success, "time": elapsed}
    if not success:
        print("\n❌ Pipeline aborted: data preparation failed")
        return

    # Step 2: Train Source Model (AIROGS — balanced + grayscale)
    success, elapsed = run_script("train_source.py", "Phase A: Source Training (AIROGS) — balanced")
    results["train_source"] = {"success": success, "time": elapsed}
    if not success:
        print("\n❌ Pipeline aborted: source training failed")
        return

    # Step 3: Train Oracle Model (Chákṣu — balanced + grayscale)
    success, elapsed = run_script("train_oracle.py", "Phase B: Oracle Training (Chákṣu) — balanced")
    results["train_oracle"] = {"success": success, "time": elapsed}

    # Step 4: Adapt (MixEnt — grayscale-aware)
    success, elapsed = run_script("adapt_target.py", "Phase C: MixEnt-Adapt SFDA — grayscale")
    results["adapt_target"] = {"success": success, "time": elapsed}
    if not success:
        print("\n❌ Pipeline aborted: adaptation failed")
        return

    # Step 5: Evaluate
    success, elapsed = run_script("evaluate.py", "Phase D: Model Evaluation")
    results["evaluate"] = {"success": success, "time": elapsed}

    # Summary
    total_time = time.time() - total_start
    print("\n" + "="*80)
    print("   PIPELINE COMPLETE — RUN-3 SUMMARY")
    print("="*80)
    for step, info in results.items():
        status = "✅" if info["success"] else "❌"
        print(f"  {status} {step:<20} {info['time']/60:.1f} min")
    print(f"\n  Total time: {total_time/60:.1f} minutes")
    print(f"  Results: /workspace/results_run3/evaluation/results_table.csv")
    print("="*80)

    try:
        exp_logger.generate_summary_report()
    except Exception as e:
        print(f"[WARN] Could not generate summary report: {e}")


if __name__ == "__main__":
    main()
