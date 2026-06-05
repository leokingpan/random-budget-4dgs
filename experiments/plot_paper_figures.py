#!/usr/bin/env python3
"""Generate all paper figures using matplotlib."""
import csv, os, sys, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

BASE = '/home/leo/evo-4dgs/base_4dgs'
OUTDIR = os.path.join(BASE, 'paper', 'figures')
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

def load_csv(path):
    rows = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            rows[float(row['budget'])] = {k: float(v) if k != 'budget' else float(v) for k, v in row.items()}
    return rows

# ============================================================
# Figure 1: Budget-Quality Curves (4 scenes, 2x2 layout)
# ============================================================
print("Generating Figure 1: Budget-quality curves...")
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

scenes = [
    ('bouncingballs', 'bouncingballs_ours.csv', 'bouncingballs_posthoc.csv'),
    ('standup', 'standup_psnr_ssim_lpips.csv', 'standup_posthoc.csv'),
    ('mutant', 'mutant_full_metrics.csv', 'mutant_posthoc.csv'),
    ('jumpingjacks', 'jumpingjacks_ours.csv', 'jumpingjacks_posthoc.csv'),
]

for ax, (name, ours_f, post_f) in zip(axes.flat, scenes):
    ours = load_csv(os.path.join(BASE, 'results/mgs', ours_f))
    post = load_csv(os.path.join(BASE, 'results/mgs', post_f))

    budgets = sorted(ours.keys())
    ours_psnr = [ours[b]['PSNR'] for b in budgets]
    post_psnr = [post[b]['PSNR'] for b in budgets]

    ax.plot(budgets, ours_psnr, 'o-', color='#2196F3', linewidth=2, markersize=6, label='MGS (Ours)')
    ax.plot(budgets, post_psnr, 's--', color='#F44336', linewidth=2, markersize=6, label='Post-hoc')

    ax.set_title(name, fontweight='bold')
    ax.set_xlabel('Budget Ratio')
    ax.set_ylabel('PSNR (dB)')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_xlim(0, 1.05)

axes[0, 0].legend(loc='lower right', framealpha=0.9)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'fig1_budget_curves.pdf'))
fig.savefig(os.path.join(OUTDIR, 'fig1_budget_curves.png'))
plt.close()
print("  -> fig1_budget_curves.pdf/png")


# ============================================================
# Figure 2: Full Budget Curves (all 7 D-NeRF scenes + 2 real, compact)
# ============================================================
print("Generating Figure 2: All-scene budget curves...")
fig, axes = plt.subplots(3, 3, figsize=(14, 12))

all_scenes = [
    ('bouncingballs', 'bouncingballs_ours.csv', 'bouncingballs_posthoc.csv'),
    ('standup', 'standup_psnr_ssim_lpips.csv', 'standup_posthoc.csv'),
    ('mutant', 'mutant_full_metrics.csv', 'mutant_posthoc.csv'),
    ('trex', 'trex_ours.csv', 'trex_posthoc.csv'),
    ('jumpingjacks', 'jumpingjacks_ours.csv', 'jumpingjacks_posthoc.csv'),
    ('hook', 'hook_ours.csv', 'hook_posthoc.csv'),
    ('hellwarrior', 'hellwarrior_ours_v2.csv', 'hellwarrior_posthoc_v2.csv'),
    ('plate (real)', 'nerfds_plate_fulltable.csv', 'nerfds_plate_psnr_vs_budget.csv'),
    ('press (real)', 'nerfds_press_ours.csv', 'nerfds_press_posthoc.csv'),
]

for ax, (name, ours_f, post_f) in zip(axes.flat, all_scenes):
    ours = load_csv(os.path.join(BASE, 'results/mgs', ours_f))
    post = load_csv(os.path.join(BASE, 'results/mgs', post_f))

    budgets = sorted(ours.keys())
    ours_psnr = [ours[b]['PSNR'] for b in budgets]
    post_psnr = [post[b]['PSNR'] for b in budgets]

    ax.plot(budgets, ours_psnr, 'o-', color='#2196F3', linewidth=1.8, markersize=4, label='Ours')
    ax.plot(budgets, post_psnr, 's--', color='#F44336', linewidth=1.5, markersize=4, label='Post-hoc')

    ax.set_title(name, fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 1.05)

# Unify y-label for left column
for i in range(3):
    axes[i, 0].set_ylabel('PSNR (dB)', fontsize=10)

axes[0, 0].legend(loc='lower right', fontsize=8, framealpha=0.9)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'fig2_all_budget_curves.pdf'))
fig.savefig(os.path.join(OUTDIR, 'fig2_all_budget_curves.png'))
plt.close()
print("  -> fig2_all_budget_curves.pdf/png")


# ============================================================
# Figure 3: Naive MGS-4D Comparison (Grouped Bar Chart)
# ============================================================
print("Generating Figure 3: Naive MGS-4D comparison...")
fig, ax = plt.subplots(figsize=(8, 5))

scenes_naive = ['bouncingballs', 'jumpingjacks', 'press (real)']
posthoc_avg = [35.06, 32.62, 19.76]
naive_avg = [38.05, 32.86, 23.86]
ours_avg = [38.15, 33.17, 22.27]

x = np.arange(len(scenes_naive))
width = 0.25

bars1 = ax.bar(x - width, posthoc_avg, width, label='Post-hoc', color='#F44336', alpha=0.85)
bars2 = ax.bar(x, naive_avg, width, label='Naive MGS-4D', color='#FF9800', alpha=0.85)
bars3 = ax.bar(x + width, ours_avg, width, label='MGS (Ours)', color='#2196F3', alpha=0.85)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.3, f'{h:.1f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(scenes_naive)
ax.set_ylabel('Budget-Avg PSNR (dB)')
ax.set_title('Naive MGS-4D vs Ours (Budget-Avg)', fontweight='bold')
ax.legend(loc='upper left', framealpha=0.9)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
ax.set_ylim(0, max(ours_avg + naive_avg) * 1.15)

fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'fig3_naive_mgs_comparison.pdf'))
fig.savefig(os.path.join(OUTDIR, 'fig3_naive_mgs_comparison.png'))
plt.close()
print("  -> fig3_naive_mgs_comparison.pdf/png")


# ============================================================
# Figure 4: Per-budget vs MGS (Budget-Avg scatter/bar)
# ============================================================
print("Generating Figure 4: Per-budget comparison...")
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

pbudget_data = {
    'bouncingballs': {
        'Post-hoc': 35.06, 'PB-10%': 38.94, 'PB-25%': 37.57, 'PB-50%': 35.67, 'Naive MGS': 38.05, 'MGS': 38.15
    },
    'jumpingjacks': {
        'Post-hoc': 32.62, 'PB-10%': 33.65, 'PB-25%': 32.67, 'PB-50%': 31.41, 'Naive MGS': 32.86, 'MGS': 33.17
    },
    'trex': {
        'Post-hoc': 30.03, 'PB-10%': 32.76, 'PB-25%': 32.04, 'MGS': 32.11
    },
    'press (real)': {
        'Post-hoc': 19.76, 'PB-10%': 23.28, 'PB-25%': 24.45, 'Naive MGS': 23.86, 'MGS': 22.27
    },
}

colors_map = {'Post-hoc': '#F44336', 'PB-10%': '#4CAF50', 'PB-25%': '#FF9800',
              'PB-50%': '#9C27B0', 'Naive MGS': '#795548', 'MGS': '#2196F3'}

for ax, (scene, data) in zip(axes.flat, pbudget_data.items()):
    labels = list(data.keys())
    values = list(data.values())
    colors = [colors_map.get(l, '#999') for l in labels]
    
    bars = ax.bar(labels, values, color=colors, alpha=0.85)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{v:.1f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    ax.set_title(scene, fontweight='bold')
    ax.set_ylabel('Budget-Avg PSNR (dB)')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.tick_params(axis='x', rotation=30, labelsize=8)

fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'fig4_perbudget_comparison.pdf'))
fig.savefig(os.path.join(OUTDIR, 'fig4_perbudget_comparison.png'))
plt.close()
print("  -> fig4_perbudget_comparison.pdf/png")


# ============================================================
# Figure 5: Temporal Stability (frame-by-frame PSNR curves)
# ============================================================
print("Generating Figure 5: Temporal stability...")
import subprocess, re

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

temporal_scenes = ['bouncingballs', 'standup', 'mutant', 'jumpingjacks']

for ax, scene in zip(axes.flat, temporal_scenes):
    ckpt = os.path.join(BASE, f'output/dnerf/{scene}_mgs')
    py = os.path.join(BASE, '../../miniconda3/envs/Gaussians4D/bin/python')
    
    # Get Ours per-frame PSNR
    out = subprocess.run([py, '-u', os.path.join(BASE, 'render_mgs.py'),
        '--checkpoint', ckpt, '--budgets', '0.25', '--frames', '20',
        '--output', '/dev/null'],
        capture_output=True, text=True, cwd=BASE, timeout=300)
    ours_psnr = [float(m.group(1)) for m in re.finditer(r'PSNR=([\d.]+)', out.stdout + out.stderr)]
    
    # Get Post-hoc per-frame PSNR
    out2 = subprocess.run([py, '-u', os.path.join(BASE, 'render_mgs.py'),
        '--checkpoint', ckpt, '--budgets', '0.25', '--frames', '20',
        '--ordering', 'posthoc_opacity', '--output', '/dev/null'],
        capture_output=True, text=True, cwd=BASE, timeout=300)
    posthoc_psnr = [float(m.group(1)) for m in re.finditer(r'PSNR=([\d.]+)', out2.stdout + out2.stderr)]
    
    frames = range(1, len(ours_psnr) + 1)
    
    ax.plot(frames, ours_psnr, 'o-', color='#2196F3', linewidth=1.5, markersize=5, label=f'Ours (μ={np.mean(ours_psnr):.1f})')
    ax.plot(frames, posthoc_psnr, 's--', color='#F44336', linewidth=1.5, markersize=5, label=f'Post-hoc (μ={np.mean(posthoc_psnr):.1f})')
    
    # Add mean lines
    ax.axhline(y=np.mean(ours_psnr), color='#2196F3', linestyle=':', alpha=0.5, linewidth=1)
    ax.axhline(y=np.mean(posthoc_psnr), color='#F44336', linestyle=':', alpha=0.5, linewidth=1)
    
    ax.set_title(scene, fontweight='bold')
    ax.set_xlabel('Frame')
    ax.set_ylabel('PSNR (dB)')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)

fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'fig5_temporal_stability.pdf'))
fig.savefig(os.path.join(OUTDIR, 'fig5_temporal_stability.png'))
plt.close()
print("  -> fig5_temporal_stability.pdf/png")


# ============================================================
# Summary
# ============================================================
print(f"\n{'='*50}")
print(f"All figures saved to: {OUTDIR}/")
print(f"{'='*50}")
for f in sorted(os.listdir(OUTDIR)):
    size = os.path.getsize(os.path.join(OUTDIR, f))
    print(f"  {f}  ({size/1024:.1f} KB)")
print(f"\nCopy to paper/evidence/:")
print(f"  cp {OUTDIR}/*.pdf {OUTDIR}/*.png {BASE}/paper/evidence/")
