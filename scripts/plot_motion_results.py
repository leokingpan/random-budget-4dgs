#!/usr/bin/env python3
"""Visualization script for Experiment 1 classification results.

Reads results JSON and generates 4 plots:
  1. PR curve
  2. F1 curve vs threshold
  3. Score distribution histogram (static vs dynamic)
  4. Motion score vs reference deformation magnitude scatter

Usage:
    python scripts/plot_motion_results.py results/bouncingballs_classification.json
"""

import matplotlib
matplotlib.use('Agg')

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def load_data(input_path):
    """Load JSON data with graceful error handling."""
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found")
        sys.exit(0)

    with open(input_path, 'r') as f:
        return json.load(f)


def plot_pr_curve(data, out_dir):
    """Plot precision-recall curve."""
    thresholds_data = data['thresholds']
    precision = np.array(thresholds_data['precision'])
    recall = np.array(thresholds_data['recall'])
    auroc = data['summary']['auroc']

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, 'b-', linewidth=1.5, label=f'AP = {auroc:.4f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve (bouncingballs)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left')

    out_path = os.path.join(out_dir, 'bouncingballs_pr_curve.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_f1_curve(data, out_dir):
    """Plot F1 score vs motion score threshold."""
    thresholds_data = data['thresholds']
    thresholds = np.array(thresholds_data['thresholds'])
    f1 = np.array(thresholds_data['f1'])
    best_threshold = data['summary']['best_threshold']
    best_f1_val = data['summary']['best_f1']

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(thresholds, f1, 'b-', linewidth=1.5)
    ax.axvline(x=best_threshold, color='r', linestyle='--', linewidth=1.2,
               label=f'Best F1 = {best_f1_val:.4f} @ threshold = {best_threshold:.2f}')
    ax.set_xlabel('Motion Score Threshold')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score vs Motion Score Threshold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left')

    out_path = os.path.join(out_dir, 'bouncingballs_f1_curve.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_score_distribution(data, out_dir):
    """Plot histogram of motion scores split into static vs dynamic."""
    per_gaussian = data['per_gaussian']
    best_threshold = data['summary']['best_threshold']

    motion_scores = np.array([g['motion_score'] for g in per_gaussian])
    static_mask = motion_scores <= best_threshold
    dynamic_mask = motion_scores > best_threshold

    fig, ax = plt.subplots(figsize=(8, 6))
    bins = np.linspace(0, motion_scores.max(), 80)
    ax.hist(motion_scores[static_mask], bins=bins, alpha=0.5, color='blue',
            label=f'Static (≤ {best_threshold:.1f}), n={static_mask.sum()}')
    ax.hist(motion_scores[dynamic_mask], bins=bins, alpha=0.5, color='red',
            label=f'Dynamic (> {best_threshold:.1f}), n={dynamic_mask.sum()}')
    ax.set_xlabel('Motion Score')
    ax.set_ylabel('Count')
    ax.set_title('Motion Score Distribution (Static vs Dynamic)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(out_dir, 'bouncingballs_score_distribution.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_scatter(data, out_dir):
    """Plot motion score vs reference deformation magnitude as hexbin."""
    per_gaussian = data['per_gaussian']
    motion_scores = np.array([g['motion_score'] for g in per_gaussian])
    ref_deformation = np.array([g['reference_deformation_magnitude'] for g in per_gaussian])

    fig, ax = plt.subplots(figsize=(8, 6))
    hb = ax.hexbin(motion_scores, ref_deformation, gridsize=60,
                   cmap='viridis', mincnt=1)
    fig.colorbar(hb, ax=ax, label='Count')
    ax.set_xlabel('Motion Score')
    ax.set_ylabel('Reference Deformation Magnitude')
    ax.set_title('Motion Score vs Reference Deformation Magnitude')
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(out_dir, 'bouncingballs_scatter.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def print_summary(data):
    """Print text summary of classification results to stdout."""
    s = data['summary']
    print("\n=== Classification Results Summary ===")
    print(f"  AUROC:            {s['auroc']:.6f}")
    print(f"  Best F1:          {s['best_f1']:.6f}")
    print(f"  Best Threshold:   {s['best_threshold']:.4f}")
    print(f"  Best Precision:   {s['best_precision']:.6f}")
    print(f"  Best Recall:      {s['best_recall']:.6f}")
    print(f"  N Gaussians:      {s['n_gaussians']}")
    print(f"  N Frames:         {s['n_frames']}")
    print("======================================\n")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <path_to_classification_json>")
        sys.exit(0)

    input_path = sys.argv[1]
    data = load_data(input_path)

    out_dir = os.path.join(os.path.dirname(input_path) or '.', 'figures')
    os.makedirs(out_dir, exist_ok=True)

    plot_pr_curve(data, out_dir)
    plot_f1_curve(data, out_dir)
    plot_score_distribution(data, out_dir)
    plot_scatter(data, out_dir)
    print_summary(data)

    print("All plots generated successfully.")


if __name__ == '__main__':
    main()
