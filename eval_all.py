#!/usr/bin/env python3
"""Batch evaluation script for Evo-4DGS experiments.

Usage:
    python eval_all.py --method baseline    # evaluate baseline checkpoints
    python eval_all.py --method ode         # evaluate ODE checkpoints
    python eval_all.py --method all         # evaluate both

Auto-discovers model directories under output/dnerf/ matching the pattern
{scene}_{method}.  Runs render.py (if needed) then metrics.py per scene,
collecting PSNR/SSIM/LPIPS into results/{method}.json.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

WORK_DIR = Path(__file__).resolve().parent
OUTPUT_BASE = WORK_DIR / "output" / "dnerf"
RESULTS_DIR = WORK_DIR / "results"


def discover_scenes(method: str) -> List[Path]:
    """Find all output directories for a given method under output/dnerf/.

    Matches directories named like bouncingballs_baseline, hellwarrior_ode, etc.
    Returns sorted list of absolute paths.
    """
    if not OUTPUT_BASE.exists():
        return []

    suffix = f"_{method}"
    dirs = sorted(
        p for p in OUTPUT_BASE.iterdir()
        if p.is_dir() and p.name.endswith(suffix)
    )
    return dirs


def render_needed(model_path: Path) -> bool:
    """Return True if test-set renders are missing or empty."""
    test_dir = model_path / "test"
    if not test_dir.exists():
        return True
    # Check for any ours_* / renders / *.png structure
    for ours_dir in test_dir.iterdir():
        if ours_dir.is_dir() and ours_dir.name.startswith("ours_"):
            renders = ours_dir / "renders"
            if renders.exists() and list(renders.glob("*.png")):
                return False
    return True


def run_cmd(cmd: str, cwd: Optional[Path] = None) -> str:
    """Run a shell command and return combined stdout+stderr."""
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True,
        cwd=str(cwd) if cwd else None, timeout=600,
    )
    return result.stdout + result.stderr


def parse_metrics(stdout: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Extract PSNR, SSIM, LPIPS-vgg from metrics.py stdout.

    Matches lines like:
        "Scene:  output/dnerf/scene_baseline PSNR :  30.1234567"
        "Scene:  output/dnerf/scene_baseline SSIM :  0.9501234"
        "Scene:  output/dnerf/scene_baseline LPIPS-vgg:  0.0501234"
    """
    psnr = ssim = lpips = None
    m = re.search(r"PSNR\s*:\s*([\d.]+)", stdout)
    if m:
        psnr = float(m.group(1))
    m = re.search(r"SSIM\s*:\s*([\d.]+)", stdout)
    if m:
        ssim = float(m.group(1))
    m = re.search(r"LPIPS-vgg\s*:\s*([\d.]+)", stdout)
    if m:
        lpips = float(m.group(1))
    return psnr, ssim, lpips


def evaluate_method(method: str, render_flags: str = "") -> Dict[str, Dict[str, float]]:
    """Evaluate all scenes for a given method.  Returns results dict."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_paths = discover_scenes(method)
    if not model_paths:
        print(f"[SKIP] No output dirs found for --method {method}")
        return {}

    results: Dict[str, Dict[str, float]] = {}

    for model_path in model_paths:
        # Scene name = dirname without the _method suffix
        scene = model_path.name[: -len(f"_{method}")]
        print(f"\n--- [{scene}] ---")

        # Render test views if needed
        if render_needed(model_path):
            print(f"  [RENDER] Running render.py for {model_path.name}")
            render_cmd = (
                f"{sys.executable} render.py {render_flags} --model_path {model_path} --skip_train"
            )
            run_cmd(render_cmd, WORK_DIR)
        else:
            print(f"  [SKIP] Renders already exist")

        # Run metrics
        print(f"  [METRICS] Running metrics.py")
        metrics_cmd = (
            f"{sys.executable} metrics.py --model_paths {model_path}"
        )
        stdout = run_cmd(metrics_cmd, WORK_DIR)

        psnr, ssim, lpips = parse_metrics(stdout)
        if psnr is not None:
            results[scene] = {
                "psnr": round(psnr, 2),
                "ssim": round(ssim, 4),
                "lpips": round(lpips, 4),
            }
            print(f"  PSNR={psnr:.2f}  SSIM={ssim:.4f}  LPIPS={lpips:.4f}")
        else:
            print(f"  [WARN] Could not parse metrics for {scene}")

    # Write JSON
    outfile = RESULTS_DIR / f"{method}.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} scenes to {outfile}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch evaluation for Evo-4DGS experiments"
    )
    parser.add_argument(
        "--method",
        default="all",
        choices=["baseline", "ode", "hybrid", "all"],
        help="Which experiment group to evaluate (default: all)",
    )
    args = parser.parse_args()

    if args.method in ("baseline", "all"):
        evaluate_method("baseline")
    if args.method in ("ode", "all"):
        evaluate_method("ode")
    if args.method in ("hybrid", "all"):
        evaluate_method("hybrid", "--hybrid")


if __name__ == "__main__":
    main()
