#!/usr/bin/env python3
"""Comprehensive result compilation for Evo-4DGS paper revision experiments.

Generates:
  Exp 1: Static vs Time-Adaptive ordering comparison table
  Exp 5: Quality-FPS trade-off table (per-scene + average)
  Exp 8: Full-budget degradation analysis per scene
"""

import csv
import os
import json
from collections import defaultdict

RESULTS_DIR = "/home/leo/evo-4dgs/base_4dgs/results/mgs"
SCENES = ["bouncingballs", "hellwarrior", "hook", "jumpingjacks",
          "mutant", "standup", "trex"]
NERFDS_SCENES = ["plate", "press"]
ALL_SCENES = SCENES + NERFDS_SCENES

BUDGETS = [0.1, 0.25, 0.5, 0.75, 1.0]


def read_csv(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "budget": float(row["budget"]),
                "PSNR": float(row["PSNR"]),
                "SSIM": float(row["SSIM"]),
                "FPS": float(row.get("FPS", 0)),
            })
    return rows


def get_budget_value(rows, budget, key="PSNR"):
    for r in rows:
        if abs(r["budget"] - budget) < 0.001:
            return r[key]
    return None


# ============================================================================
# EXP 1: Static vs Time-Adaptive Ordering
# ============================================================================
print("=" * 80)
print("EXP 1: Static Ordering Credibility — All Scenes, Multi-Budget")
print("=" * 80)

static_comparison = []
for scene in ALL_SCENES:
    prefix = f"nerfds_{scene}" if scene in NERFDS_SCENES else scene
    adaptive_rows = read_csv(os.path.join(RESULTS_DIR, f"{prefix}_ours.csv"))
    static_rows = read_csv(os.path.join(RESULTS_DIR, f"{prefix}_static.csv"))

    if not adaptive_rows or not static_rows:
        # Try alternate names
        if scene == "bouncingballs":
            adaptive_rows = read_csv(os.path.join(RESULTS_DIR, "bouncingballs_ours.csv"))
            static_rows = read_csv(os.path.join(RESULTS_DIR, "bouncingballs_static.csv"))
        print(f"  {scene}: SKIP (missing data)")
        continue

    row_data = {"scene": scene}
    for budget in BUDGETS:
        a_psnr = get_budget_value(adaptive_rows, budget, "PSNR")
        s_psnr = get_budget_value(static_rows, budget, "PSNR")
        if a_psnr and s_psnr:
            delta = a_psnr - s_psnr
            row_data[f"b{budget}"] = f"{a_psnr:.2f}/{s_psnr:.2f} (Δ={delta:+.3f})"
        else:
            row_data[f"b{budget}"] = "N/A"

    # FPS comparison at full budget
    a_fps = get_budget_value(adaptive_rows, 1.0, "FPS")
    s_fps = get_budget_value(static_rows, 1.0, "FPS")
    row_data["fps"] = f"{s_fps:.0f} vs {a_fps:.0f} (static/adaptive)" if a_fps and s_fps else "N/A"

    print(f"\n  {scene}:")
    for budget in BUDGETS:
        print(f"    budget={budget}: {row_data.get(f'b{budget}', 'N/A')}")
    print(f"    FPS: {row_data['fps']}")

# ============================================================================
# EXP 5: Quality-FPS Trade-off
# ============================================================================
print("\n" + "=" * 80)
print("EXP 5: Quality–FPS Trade-off — Deployment Analysis")
print("=" * 80)

# Collect per-scene data
fps_table = defaultdict(dict)
for scene in ALL_SCENES:
    prefix = f"nerfds_{scene}" if scene in NERFDS_SCENES else scene
    rows = read_csv(os.path.join(RESULTS_DIR, f"{prefix}_ours.csv"))
    if not rows and scene == "bouncingballs":
        rows = read_csv(os.path.join(RESULTS_DIR, "bouncingballs_ours.csv"))
    if not rows:
        print(f"  {scene}: No data")
        continue
    for budget in BUDGETS:
        val = get_budget_value(rows, budget)
        if val:
            fps_table[scene][budget] = val

# Print per-scene trade-off table
headers = ["Scene"] + [f"B={b}" for b in BUDGETS] + ["B=1.0 FPS", "10% vs 100% speedup"]
print(f"\n{'Scene':<16}", end="")
for b in BUDGETS:
    label = f"B={b}"
    print(f"{label:>10}", end="")
print(f"{'100%FPS':>8}  {'Speedup':>8}")
print("-" * 80)

for scene in ALL_SCENES:
    if scene not in fps_table:
        continue
    print(f"{scene:<16}", end="")
    base_fps = fps_table[scene].get(1.0)
    for b in BUDGETS:
        psnr = fps_table[scene].get(b)
        if psnr:
            print(f"{psnr:>10.2f}", end="")
        else:
            print(f"{'N/A':>10}", end="")
    fps_10 = None
    prefix = f"nerfds_{scene}" if scene in NERFDS_SCENES else scene
    rows_fps = read_csv(os.path.join(RESULTS_DIR, f"{prefix}_ours.csv"))
    if not rows_fps and scene == "bouncingballs":
        rows_fps = read_csv(os.path.join(RESULTS_DIR, "bouncingballs_ours.csv"))
    for r in rows_fps:
        if abs(r["budget"] - 1.0) < 0.001:
            fps_100 = r.get("FPS", 0)
        if abs(r["budget"] - 0.1) < 0.001:
            fps_10 = r.get("FPS", 0)
    speedup = fps_10 / fps_100 if fps_10 and fps_100 else 0
    print(f"{fps_100:>8.0f}  {speedup:>8.2f}x")

# Average table
print(f"\nAverage across {len(SCENES)} D-NeRF scenes:")
print(f"{'Budget':<10} {'Avg PSNR':>10} {'Avg FPS':>10} {'Speedup':>10}")
print("-" * 45)

avg_data = defaultdict(list)
fps_data = defaultdict(list)
for scene in SCENES:
    prefix = scene
    rows = read_csv(os.path.join(RESULTS_DIR, f"{prefix}_ours.csv"))
    if not rows:
        continue
    for r in rows:
        avg_data[r["budget"]].append(r["PSNR"])
        fps_data[r["budget"]].append(r.get("FPS", 0))

base_fps_avg = sum(fps_data.get(1.0, [0])) / max(1, len(fps_data.get(1.0, [0])))
for budget in sorted(avg_data.keys()):
    avg_psnr = sum(avg_data[budget]) / len(avg_data[budget])
    avg_fps = sum(fps_data[budget]) / len(fps_data[budget])
    speedup = avg_fps / base_fps_avg if base_fps_avg else 0
    print(f"{budget:<10} {avg_psnr:>10.2f} {avg_fps:>10.1f} {speedup:>10.2f}x")

# ============================================================================
# EXP 8: Full-budget degradation analysis
# ============================================================================
print("\n" + "=" * 80)
print("EXP 8: Full-Budget Degradation — Per-Scene Analysis")
print("=" * 80)

print(f"\n{'Scene':<16} {'Baseline':>10} {'Ours 100%':>10} {'ΔPSNR':>10} {'Motion Type':<22}")
print("-" * 72)

degradation_data = []
for scene in SCENES:
    # Read baseline (standard 4DGS) from existing eval data
    baseline_rows = read_csv(os.path.join(RESULTS_DIR, f"{scene}_psnr_vs_budget.csv"))
    ours_rows = read_csv(os.path.join(RESULTS_DIR, f"{scene}_ours.csv"))

    base_psnr = get_budget_value(baseline_rows, 1.0, "PSNR") if baseline_rows else None
    ours_psnr = get_budget_value(ours_rows, 1.0, "PSNR") if ours_rows else None

    # Fallback to known baseline values from SUMMARY.md
    known_baselines = {
        "bouncingballs": 40.72, "hellwarrior": 28.68, "hook": 33.04,
        "jumpingjacks": 35.54, "lego": 29.25, "mutant": 38.02,
        "standup": 38.62, "trex": 34.52
    }
    motion_types = {
        "bouncingballs": "rigid, simple motion",
        "hellwarrior": "articulated, complex",
        "hook": "articulated, rotation",
        "jumpingjacks": "human articulation",
        "mutant": "non-rigid deformation",
        "standup": "human articulation",
        "trex": "articulated motion",
    }

    if not base_psnr:
        base_psnr = known_baselines.get(scene)
    if ours_psnr is None:
        ours_psnr = 0

    delta = ours_psnr - base_psnr if base_psnr else 0
    motion = motion_types.get(scene, "unknown")

    print(f"{scene:<16} {base_psnr:>10.2f} {ours_psnr:>10.2f} {delta:>+10.2f} {motion:<22}")

# ============================================================================
# EXP 1: Kendall τ estimation (pairwise comparison)
# ============================================================================
print("\n" + "=" * 80)
print("EXP 1: Pairwise Static vs Adaptive Ordering Comparison")
print("=" * 80)

print(f"\n{'Scene':<16} {'0.1 PSNR':>12} {'0.25 PSNR':>12} {'1.0 PSNR':>12} {'Match?':>10}")
print(f"{'':16} {'Adapt/Static':>12} {'Adapt/Static':>12} {'Adapt/Static':>12}")
print("-" * 70)

for scene in ALL_SCENES:
    prefix = f"nerfds_{scene}" if scene in NERFDS_SCENES else scene
    adaptive = read_csv(os.path.join(RESULTS_DIR, f"{prefix}_ours.csv"))
    static = read_csv(os.path.join(RESULTS_DIR, f"{prefix}_static.csv"))
    if not adaptive or not static:
        continue

    match_count = 0
    total = 0
    line = f"{scene:<16}"
    for budget in [0.1, 0.25, 1.0]:
        a = get_budget_value(adaptive, budget, "PSNR")
        s = get_budget_value(static, budget, "PSNR")
        if a and s:
            line += f" {a:>5.2f}/{s:<5.2f}"
            if abs(a - s) < 0.01:
                match_count += 1
            total += 1
        else:
            line += f" {'N/A':>12}"
    match_str = f"{match_count}/{total} identical" if total else "N/A"
    print(f"{line}  {match_str:>10}")

print("\n" + "=" * 80)
print("COMPILATION COMPLETE")
print("=" * 80)
