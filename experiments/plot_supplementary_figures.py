#!/usr/bin/env python3
"""Generate supplementary paper figures: ablations, overhead, errors, analysis."""
import csv, os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = '/home/leo/evo-4dgs/base_4dgs'
OUTDIR = os.path.join(BASE, 'paper', 'figures')
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'legend.fontsize': 10, 'figure.dpi': 150, 'savefig.dpi': 150,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
})

def load_csv(path):
    rows = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            rows[float(row['budget'])] = {k: float(v) if k != 'budget' else float(v) for k, v in row.items()}
    return rows

# ============================================================
# Figure 6: Full-Loss Ablation (fw comparison bar chart)
# ============================================================
print("Fig 6: Full-Loss Ablation...")
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

variants = ['Prefix-only\n(fw=0)', 'Equal\n(fw=1)', 'Strong\n(fw=2)']
psnr_25 = [39.17, 38.82, 38.87]
psnr_100 = [40.41, 40.37, 40.08]
lpips_25 = [0.0113, 0.0130, 0.0120]
colors = ['#2196F3', '#FF9800', '#9C27B0']

ax1 = axes[0]
x = np.arange(len(variants))
w = 0.35
b1 = ax1.bar(x - w/2, psnr_25, w, label='25% Budget', color='#2196F3', alpha=0.85)
b2 = ax1.bar(x + w/2, psnr_100, w, label='100% Budget', color='#4CAF50', alpha=0.85)
for bars in [b1, b2]:
    for bar in bars:
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()-0.5, f'{bar.get_height():.2f}',
                ha='center', va='top', fontsize=8, color='white', fontweight='bold')
ax1.set_xticks(x); ax1.set_xticklabels(variants, fontsize=9)
ax1.set_ylabel('PSNR (dB)'); ax1.set_title('PSNR @ 25% & 100% Budget', fontweight='bold')
ax1.legend(); ax1.grid(True, alpha=0.3, axis='y', linestyle='--')

ax2 = axes[1]
b3 = ax2.bar(variants, lpips_25, color=colors, alpha=0.85, width=0.5)
for bar, v in zip(b3, lpips_25):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0003, f'{v:.4f}',
            ha='center', fontsize=9, fontweight='bold')
ax2.set_ylabel('LPIPS (25% Budget)'); ax2.set_title('LPIPS @ 25% Budget', fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

fig.suptitle('Full-Loss Weight Ablation (bouncingballs)', fontweight='bold', fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'fig6_ablation_full_loss.pdf'))
fig.savefig(os.path.join(OUTDIR, 'fig6_ablation_full_loss.png'))
plt.close()

# ============================================================
# Figure 7: Importance Score Ablation (4 scenes, 10% budget)
# ============================================================
print("Fig 7: Importance Score Ablation...")
fig, ax = plt.subplots(figsize=(9, 5))

scenes = ['bouncingballs', 'trex', 'standup', 'mutant']
ours = [32.09, 30.99, 31.26, 29.65]
opacity_only = [24.74, 27.15, 28.07, 25.43]
random_vals = [24.83, 21.42, 23.78, 20.06]

x = np.arange(len(scenes))
w = 0.25
ax.bar(x - w, ours, w, label='Ours (opacity×area)', color='#2196F3', alpha=0.85)
ax.bar(x, opacity_only, w, label='Opacity-only', color='#FF9800', alpha=0.85)
ax.bar(x + w, random_vals, w, label='Random', color='#F44336', alpha=0.85)

# Annotate bars
for i in range(len(scenes)):
    for j, vals in enumerate([ours, opacity_only, random_vals]):
        ax.text(x[i] + (j-1)*w, vals[i] + 0.5, f'{vals[i]:.1f}',
                ha='center', fontsize=7, fontweight='bold')

ax.set_xticks(x); ax.set_xticklabels(scenes)
ax.set_ylabel('PSNR (dB)'); ax.set_title('Importance Score Ablation @ 10% Budget', fontweight='bold')
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'fig7_ablation_importance.pdf'))
fig.savefig(os.path.join(OUTDIR, 'fig7_ablation_importance.png'))
plt.close()

# ============================================================
# Figure 8: Training Overhead Comparison
# ============================================================
print("Fig 8: Training Overhead...")
fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

# Time per iter
axes[0].bar(['Baseline', 'MGS (Ours)'], [19.8, 21.5], color=['#9E9E9E', '#2196F3'], alpha=0.85, width=0.5)
axes[0].set_ylabel('ms / iteration'); axes[0].set_title('Iteration Time', fontweight='bold')
for i, v in enumerate([19.8, 21.5]):
    axes[0].text(i, v+0.3, f'{v:.1f} ms', ha='center', fontweight='bold')
axes[0].set_ylim(0, 25); axes[0].grid(True, alpha=0.3, axis='y')

# Total 30K
axes[1].bar(['Baseline', 'MGS (Ours)'], [9.9, 10.7], color=['#9E9E9E', '#2196F3'], alpha=0.85, width=0.5)
axes[1].set_ylabel('minutes'); axes[1].set_title('30K Iterations Total', fontweight='bold')
for i, v in enumerate([9.9, 10.7]):
    axes[1].text(i, v+0.2, f'{v:.1f} min', ha='center', fontweight='bold')
axes[1].set_ylim(0, 13); axes[1].grid(True, alpha=0.3, axis='y')

# GPU Memory
axes[2].bar(['Baseline', 'MGS (Ours)'], [802, 801], color=['#9E9E9E', '#2196F3'], alpha=0.85, width=0.5)
axes[2].set_ylabel('MiB'); axes[2].set_title('Peak GPU Memory', fontweight='bold')
for i, v in enumerate([802, 801]):
    axes[2].text(i, v+5, f'{v:.0f} MiB', ha='center', fontweight='bold')
axes[2].set_ylim(0, 900); axes[2].grid(True, alpha=0.3, axis='y')

fig.suptitle('Training Overhead (bouncingballs)', fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'fig8_training_overhead.pdf'))
fig.savefig(os.path.join(OUTDIR, 'fig8_training_overhead.png'))
plt.close()

# ============================================================
# Figure 9: Per-budget Multi-Budget Heatmap
# ============================================================
print("Fig 9: Per-budget Heatmap...")
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

heatmap_data = {
    'bouncingballs': {
        'Post-hoc': [24.35, 35.14, 40.37, 40.37],
        'PB 10%': [37.98, 38.36, 39.64, 39.78],
        'PB 25%': [30.80, 39.54, 39.96, 40.00],
        'PB 50%': [27.99, 34.17, 40.23, 40.28],
        'MGS': [33.18, 38.82, 40.24, 40.37],
    },
    'jumpingjacks': {
        'Post-hoc': [28.81, 32.71, 34.48, 34.49],
        'PB 10%': [33.29, 33.54, 33.82, 33.96],
        'PB 25%': [28.80, 33.82, 34.01, 34.05],
        'PB 50%': [25.97, 31.20, 34.21, 34.26],
        'MGS': [30.13, 33.67, 34.40, 34.49],
    },
    'press': {
        'Post-hoc': [12.16, 20.33, 23.21, 23.35],
        'PB 10%': [23.04, 23.25, 23.39, 23.45],
        'PB 25%': [23.65, 24.73, 24.72, 24.70],
        'Naive MGS': [23.65, 23.94, 23.94, 23.92],
        'MGS': [20.20, 22.34, 23.17, 23.35],
    },
}

for ax, (scene, methods) in zip(axes, heatmap_data.items()):
    method_names = list(methods.keys())
    budgets = ['10%', '25%', '50%', '100%']
    data = np.array([methods[m] for m in method_names])
    
    im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=10, vmax=42)
    
    for i in range(len(method_names)):
        for j in range(len(budgets)):
            text_color = 'white' if data[i, j] < 25 else 'black'
            ax.text(j, i, f'{data[i,j]:.1f}', ha='center', va='center',
                   fontsize=8, fontweight='bold', color=text_color)
    
    ax.set_xticks(range(len(budgets))); ax.set_xticklabels(budgets, fontsize=10)
    ax.set_yticks(range(len(method_names))); ax.set_yticklabels(method_names, fontsize=10)
    ax.set_title(scene, fontweight='bold', fontsize=12)

fig.colorbar(im, ax=axes, label='PSNR (dB)', shrink=0.6, aspect=30)
fig.suptitle('Per-Budget Comparison Heatmap', fontweight='bold', fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'fig9_heatmap.pdf'))
fig.savefig(os.path.join(OUTDIR, 'fig9_heatmap.png'))
plt.close()

# ============================================================
# Figure 10: Combined Overview (Big Summary Figure)
# ============================================================
print("Fig 10: Overview figure...")
fig = plt.figure(figsize=(14, 10))

# Top-left: 25% budget gain per scene
ax1 = fig.add_subplot(2, 3, 1)
scene_names = ['b-balls', 'standup', 'mutant', 'trex', 'j-jacks', 'hook', 'hellw.', 'plate', 'press']
gains = [3.7, 2.4, 3.2, 1.1, 1.0, 1.8, 2.0, 1.3, 2.0]
colors_bar = ['#4CAF50' if g > 2 else '#FF9800' if g > 1 else '#F44336' for g in gains]
bars = ax1.barh(scene_names, gains, color=colors_bar, alpha=0.85)
for bar, g in zip(bars, gains):
    ax1.text(bar.get_width()+0.05, bar.get_y()+bar.get_height()/2, f'+{g:.1f}',
            va='center', fontsize=9, fontweight='bold')
ax1.set_xlabel('Δ PSNR (dB)'); ax1.set_title('25% Budget Gain vs Post-hoc', fontweight='bold')
ax1.axvline(x=2.0, color='gray', linestyle='--', alpha=0.5, label='Avg +2.0')
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3, axis='x')

# Top-middle: Budget-Avg comparison
ax2 = fig.add_subplot(2, 3, 2)
method_labels = ['Post-hoc', 'Naive\nMGS-4D', 'PB\n10%', 'PB\n25%', 'PB\n50%', 'MGS\n(Ours)']
bb_vals = [35.06, 38.05, 38.94, 37.57, 35.67, 38.15]
colors_m = ['#F44336', '#FF9800', '#4CAF50', '#4CAF50', '#4CAF50', '#2196F3']
bars2 = ax2.bar(method_labels, bb_vals, color=colors_m, alpha=0.85)
for bar, v in zip(bars2, bb_vals):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f'{v:.1f}',
            ha='center', fontsize=8, fontweight='bold')
ax2.set_ylabel('Budget-Avg PSNR (dB)'); ax2.set_title('bouncingballs Budget-Avg', fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# Top-right: Naive MGS comparison
ax3 = fig.add_subplot(2, 3, 3)
naive_scenes = ['b-balls', 'j-jacks', 'press']
naive_ours = [38.15, 33.17, 22.27]
naive_base = [38.05, 32.86, 23.86]
x3 = np.arange(len(naive_scenes)); w3 = 0.35
ax3.bar(x3-w3/2, naive_base, w3, label='Naive MGS-4D', color='#FF9800', alpha=0.85)
ax3.bar(x3+w3/2, naive_ours, w3, label='MGS (Ours)', color='#2196F3', alpha=0.85)
for i in range(3):
    for j, vals in enumerate([naive_base, naive_ours]):
        ax3.text(x3[i]+(j-0.5)*w3, vals[i]+0.3, f'{vals[i]:.1f}', ha='center', fontsize=8, fontweight='bold')
ax3.set_xticks(x3); ax3.set_xticklabels(naive_scenes)
ax3.set_ylabel('Budget-Avg PSNR'); ax3.set_title('Naive MGS-4D vs Ours', fontweight='bold')
ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3, axis='y')

# Bottom row: 3 budget curves (spanning)
for idx, (scene, ours_f, post_f) in enumerate([
    ('bouncingballs', 'bouncingballs_ours.csv', 'bouncingballs_posthoc.csv'),
    ('jumpingjacks', 'jumpingjacks_ours.csv', 'jumpingjacks_posthoc.csv'),
    ('press (real)', 'nerfds_press_ours.csv', 'nerfds_press_posthoc.csv'),
]):
    ax = fig.add_subplot(2, 3, 4+idx)
    ours = load_csv(os.path.join(BASE, 'results/mgs', ours_f))
    post = load_csv(os.path.join(BASE, 'results/mgs', post_f))
    buds = sorted(ours.keys())
    ax.plot(buds, [ours[b]['PSNR'] for b in buds], 'o-', color='#2196F3', lw=2, ms=5, label='MGS')
    ax.plot(buds, [post[b]['PSNR'] for b in buds], 's--', color='#F44336', lw=1.5, ms=5, label='Post-hoc')
    ax.set_title(scene, fontweight='bold')
    ax.set_xlabel('Budget'); ax.set_ylabel('PSNR (dB)')
    ax.grid(True, alpha=0.3, linestyle='--'); ax.legend(fontsize=8)
    ax.set_xlim(0, 1.05)

fig.suptitle('MGS × 4DGS — Experimental Overview', fontweight='bold', fontsize=15)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'fig10_overview.pdf'))
fig.savefig(os.path.join(OUTDIR, 'fig10_overview.png'))
plt.close()

# ============================================================
print(f"\nDone! {len(os.listdir(OUTDIR))} files in {OUTDIR}/")
for f in sorted(os.listdir(OUTDIR)):
    print(f"  {f}")
