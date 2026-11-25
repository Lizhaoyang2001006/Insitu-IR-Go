#!/usr/bin/env python3
"""
End-to-End Reproducible Pipeline Runner (Step 0 -> Step 1A -> Step 2 -> Step 3 -> Step 4).

Executes the entire In-Situ FTIR processing workflow from raw CSV/XLSX spectra to
publication-ready figures and Origin matrices in one command:
  python3 run_pipeline.py
"""
import argparse
import os
import subprocess
import sys


def run_step(cmd):
    print(f"\n>>> Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run full In-Situ FTIR data processing pipeline (Step 0 -> Step 4).")
    parser.add_argument("--input-dir", default="./raw_data", help="Raw FTIR CSV/XLSX directory (default: ./raw_data).")
    parser.add_argument("--out-root", default="./output", help="Root output directory (default: ./output).")
    parser.add_argument("--corr-gate", type=float, default=0.50, help="Step 2 Pearson |r| gate on 1700-1800 cm^-1.")
    parser.add_argument("--max-hold-min", type=float, default=10.6, help="Aligned 120 °C isothermal hold window (min).")
    parser.add_argument("--metric", choices=["decon", "r1590", "r1600", "all"], default="all", help="Step 4 metric(s).")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))
    scripts_dir = os.path.join(repo_root, "scripts")
    s1_dir = os.path.join(args.out_root, "step1A")
    s23_dir = os.path.join(args.out_root, "step2_3")
    s4_dir = os.path.join(args.out_root, "step4")

    py = sys.executable
    run_step([
        py, os.path.join(scripts_dir, "step1A_baseline_truncate.py"),
        "--input-dir", args.input_dir,
        "--out-dir", s1_dir,
    ])
    run_step([
        py, os.path.join(scripts_dir, "step2_step3_wvclean_and_ratios.py"),
        "--in-dir", s1_dir,
        "--out-dir", s23_dir,
        "--corr-gate", str(args.corr_gate),
        "--max-hold-min", str(args.max_hold_min),
    ])
    run_step([
        py, os.path.join(scripts_dir, "step4_isothermal_fig4_template.py"),
        "--in-dir", s23_dir,
        "--out-dir", s4_dir,
        "--max-hold-min", str(args.max_hold_min),
        "--metric", args.metric,
    ])
    print("\n" + "=" * 88)
    print(f"[ALL DONE] Full In-Situ FTIR pipeline completed successfully -> {args.out_root}")
    print("=" * 88)


if __name__ == "__main__":
    main()
