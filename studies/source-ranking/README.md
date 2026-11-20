# Choosing a primitive ranking for low-budget 4D Gaussian Splatting

[Manuscript (PDF)](manuscript/main.pdf) · [LaTeX source](manuscript/main.tex) · [Absolute PSNR data](data/absolute_psnr.csv)

## Reproduce the Cut seed-zero table entries

Run from this directory with Python and NumPy.

```sh
python replay/check.py
```

The check reconstructs O (opacity), C (influence within the more opaque half) and U (unrestricted influence) at 10% and 15%, verifies six SHA-256 membership hashes, and recomputes six mean PSNR values from the supplied frame measurements. Input scores, canonical opacity order and frame data are bundled. This check runs on CPU. Rendering new images requires a trained model, its camera loader and the compatible native GPU renderer.

## Score a loaded model

The CUDA kernel in [scoring/moments_v3.cu](scoring/moments_v3.cu) reads the geometry, binning and image buffers from the study's depth-enabled differential Gaussian rasterizer. Its internal layout is ABI-dependent. The buffer adapter, training-view average, deterministic candidate ranking and four-pass comparator are in [scoring/selection.py](scoring/selection.py). The archived rasterizer returns seven outputs with geometry, binning and image buffers last. Use the matching implementation rather than passing another rasterizer's opaque buffers. The extension requires PyTorch with CUDA, a compatible compiler and nvcc. See the kernel's [license](scoring/LICENSE.md).

For a loaded dynamic Gaussian model, its list of calibration camera/time observations, and a CUDA RGB background, build the extension with `load_scoring_extension()`, then call `mean_scores(model, cameras, background, extension)`. For U use `rank_candidates(scores)`. For C pass the full descending opacity order and fraction 0.5. Select the first round(N × retained fraction) canonical IDs and sort them before using the native subset renderer. `four_pass_order` updates scores after retaining 60%, 35% and 20% of the original N and returns the final ranking for the shared 10% and 15% endpoints.

## Data and protocols

- `data/absolute_psnr.csv` contains both source families for eight matched scenes, seed zero, full point counts, full PSNR, exact retained counts and absolute O/C/U PSNR. Each family uses its own native full point count.
- `data/cut_seed_results.csv` gives each seed's PSNR and the original warm preparation timings. Pool variants share the full-source scoring measurement. Each seed was measured once per procedure in the original comparison. `repeat_timing.csv` contains a separate five-repeat, alternating-order check on each seed for C50 versus four passes; both endpoints are materialized.
- `data/deletion_pairs.csv` contains the same 96 point–view pairs in both contexts, including zero perturbations. The full and 15% comparisons use the same canonical IDs.
- `data/calibration_ids.json`, `calibration.csv` and `source_manifest.json` record the sampled observations, sensitivity results and primary Cut membership fingerprints.

The highest-PSNR candidate width on Cut is a descriptive comparison. Training settings and point counts also differ between the two source families. The paired scene study does not isolate a causal effect of prefix supervision.

The release kernel and adapter were rebuilt with PyTorch 2.5.1+cu121 and tested on an RTX 3090. For Cut seed zero the fresh C, U and four-pass runs exactly reproduced all six archived membership hashes for 10% and 15% retention. They require the complete trained checkpoint and the original calibration camera tensors. The CPU check above can be run from this repository alone.

## Ordinary 4DGS training repeats

The ordinary-source repeat study covers Broom, Chicken and Flame Steak, chosen to span a large, intermediate and near-zero seed-zero filtering effect at 10% retention. It reuses the completed ordinary Broom seed 1 and trains Chicken and Flame Steak seed 1 from initialization. Each pair uses the same scene configuration, prepared data, train-only calibration observations and held-out frames. Training runs for 3,000 coarse and 30,000 fine updates with prefix supervision disabled. Seed 0 and seed 1 are distinct training runs. Both targets use 32 calibration observations and the same 50% opacity eligibility rule for C.

Run `python3 data/ordinary_repeats/verify.py` from this directory (Python standard library). It checks all 42 source/arm/budget means and the point counts, verifies matched frame identities and prints U minus C per seed. The files in `data/ordinary_repeats/` include:

- `summary.csv` with full-source and O/C/U PSNR at 10% and 15%, counts, test-frame counts and sorted membership fingerprints.
- Six per-frame CSVs with the held-out source index and PSNR for every render.
- `sources.json` with checkpoint-component, dataset-manifest and test-order fingerprints plus the calibration indices.
- `training_configuration.json` with scene-specific training overrides; the model and dataset locations are supplied locally.
- `broom_frame.json` recording the middle test-frame choice for the ordinary seed-zero 10% example.

Both CPU checkers operate on saved scores or saved measurements. Rendering new images requires the trained checkpoint, the matching scene/camera loader and prepared images, and compatible GPU deformation and CUDA rasterization code. This archive does not include the six trained models. The manuscript distinguishes the eight-scene seed-zero comparison from these three within-scene repeats.
