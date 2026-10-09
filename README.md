# TAVL-Syn

Conditional Rectified Flow for paired medical image synthesis, with **Transport-Aware Velocity Steering (TAVS)** and **Damped Self-Correction (DSC)**.

This release provides the **breast MRI pre-contrast to first post-contrast (Pre -> Post1)** pipeline. The model operates on 2D axial patches; overlapping patch predictions are assembled into full-resolution volumes. It is not a 3D generative model. Patient images, labels, checkpoints, and experimental outputs are not included.

## Method

For a paired source image `c` and target `x1`, training samples Gaussian noise `x0` and a time `t`:

```text
xt = (1 - t) * x0 + t * x1
u  = x1 - x0
L_RF = mean((v_theta(t, xt, c) - u)^2)
```

**TAVS** uses source features and current transport features to predict spatial routing weights over learnable local operators. A spatial strength map and time-dependent channel gates modulate the residual update. At a 256 x 256 input size, TAVS is applied at the 16 x 16 bottleneck and the 64 / 128 / 256 decoder stages. The operators act on features, not fixed physical displacement directions.

**DSC** supervises the same velocity estimator at a detached, model-induced rollout state:

```text
s       = t + eta * (1 - t)
x_roll  = xt + (s - t) * stop_gradient(v_theta(t, xt, c))
d       = x_roll - [xt + (s - t) * u]
R       = (1 - s) * [v_theta(s, x_roll, c) - u] + beta * d
L_DSC   = sum(R^2) / max(sum(broadcast(1 - s)^2), 1e-8)
L_total = L_RF + lambda_sc * ramp(epoch) * L_DSC
```

The denominator is summed over the same tensor dimensions as the residual. DSC is evaluated periodically on eligible samples. Its detached construction prevents gradients through the initial rollout velocity. Main and DSC losses accumulate gradients before one optimizer update. DSC adds training computation but no inference branch or additional sampling evaluations. The corrective target is a finite-horizon local constraint, not a guarantee of global trajectory contraction.

## Installation

Use Python 3.10 or later and a PyTorch installation appropriate for your hardware. CUDA is recommended for the full model. Install PyTorch using its official installer for your CUDA platform, then install the remaining dependencies:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run commands from the repository root. No external source checkout, pretrained encoder download, or hard-coded workstation path is required. CPU execution is available with `--device cpu`.

Verified environment: Python 3.11, PyTorch 2.11.0 (CUDA 12.8 build), NumPy 2.4.4, SciPy 1.17.1, NiBabel 5.4.2, Pillow 12.2.0, scikit-image 0.26.0, and tqdm 4.67.3. The dependency ranges also permit other versions, but those combinations have not all been tested.

## Data Preparation

Obtain the dataset through its authorized distribution channel. This repository does not redistribute MAMA-MIA or provide patient identifiers. Keep all acquisitions from the same patient in one split; fix the split before training or tuning.

The input to the preparation script is a CSV with these columns:

```csv
patient_id,split,pre,post1
EXAMPLE_TRAIN,train,images/EXAMPLE_TRAIN_pre.nii.gz,images/EXAMPLE_TRAIN_post1.nii.gz
EXAMPLE_VAL,val,images/EXAMPLE_VAL_pre.nii.gz,images/EXAMPLE_VAL_post1.nii.gz
EXAMPLE_TEST,test,images/EXAMPLE_TEST_pre.nii.gz,images/EXAMPLE_TEST_post1.nii.gz
```

Paths are absolute or relative to the CSV. `examples/pairs.csv` contains placeholders, not downloadable data. Supply **separate 3D, aligned Pre and first post-contrast images in raw intensity space**, on the same **1 mm grid**. Perform any necessary registration and resampling upstream. The script only canonicalizes axis orientation to RAS, validates geometry, and packages the arrays; it does not register or resample images, select the contrast phase, or generate a split.

```bash
python prepare_data.py --manifest /path/to/pairs.csv --output data/breast
```

The resulting layout is:

```text
data/breast/
  dataset.json
  metadata/splits.csv
  tasks/generation_3d.csv
  tasks/generation_2d.csv
  train/<patient_id>/{pre.npy,post1.npy,metadata.json}
  val/<patient_id>/{pre.npy,post1.npy,metadata.json}
  test/<patient_id>/{pre.npy,post1.npy,metadata.json}
```

- NPY arrays: contiguous float32 **ZXY**, retaining raw intensities.
- Internal volume views and NIfTI exports: **XYZ**; original canonical geometry is retained.
- Per-patient normalization: `c = median(nonzero Pre)`, `s = 1.4826 * median(abs(nonzero Pre - c))`; both phases use `(raw - c) / s`. Invalid or zero source MAD is rejected.
- No target-derived training normalization, intensity clipping, or tumor-mask sampling is used.
- Training cycles through source-nonempty slices and source-nonempty grid crops with 50% overlap. The patient-block sampler shuffles patients and processes each patient's patches consecutively for efficient volume access; a batch is not guaranteed to contain distinct patients.

Existing datasets matching this schema can be used directly with `--data_root`. The legacy paired 4D NIfTI index format is also readable. A `TAVL_DATA_ROOT` environment variable may supply the default path. Keep the generated index CSV files unchanged when relocating a prepared dataset: their content fingerprint is recorded in checkpoints.

## Training

Train the full model:

```bash
python train.py --data_root data/breast --output_dir runs/tavl_syn --max_epoch 100
```

This example explicitly requests 100 epochs. The parser retains a 200-epoch default; always specify the budget used for the experiment. Training uses Adam and plain velocity MSE plus DSC. There is no pretrained backbone requirement.

| Option | Default | Meaning |
|---|---:|---|
| `img_size` | 256 | Square 2D patch size |
| `model_chn` | 128 | U-Net base channels |
| `mr_base_ch` | 32 | Source encoder base channels |
| `tavs_num_dirs` | 8 | Number of local steering operators |
| `batch_size` | 4 | Patches per training batch |
| `patches_per_patient` | 16 | Patches per patient per epoch |
| `lr` | 1e-4 | Adam learning rate |
| `grad_clip` | 1.0 | Gradient norm clipping |
| `sc_loss_weight` | 0.10 | DSC weight after ramp-up |
| `sc_warmup_epochs` | 20 | Initial epochs without DSC |
| `sc_ramp_epochs` | 20 | Linear DSC ramp duration |
| `sc_update_interval` | 4 | Apply DSC every fourth batch |
| `sc_min_step_ratio` / `sc_max_step_ratio` | 0.10 / 0.30 | Rollout fraction of remaining time |
| `sc_max_start_time` | 0.8 | Largest eligible starting time |
| `sc_correction_fraction` | 0.25 | Correction fraction beta |
| `seed` | 20260727 | Training seed |
| `num_workers` | 0 | DataLoader workers |
| `fp16` | off | Opt-in CUDA mixed precision |

The linear ramp is zero through epoch 20, 0.05 at epoch 21, and one at epoch 40. Set `--sc_loss_weight 0` to disable DSC without changing TAVS. Run `python train.py --help` for all options.

### Checkpoints and Resuming

The output directory contains `run_config.json`, `train_log.csv`, and `checkpoints/`:

- `model_latest.pth`: latest completed epoch, model, optimizer, AMP scaler, history, and RNG states.
- `model_final.pth`: model at the requested final epoch.
- `model_epoch_<n>.pth`: optional snapshots enabled with `--save_interval`.
- `model_best.pth`: only written if validation is enabled and its monitored MAE improves.

```bash
python train.py --data_root data/breast --output_dir runs/tavl_syn --max_epoch 100 --continue_train
```

The same directory automatically resumes its latest checkpoint when present. `--resume /path/to/model_latest.pth` selects an explicit checkpoint. All checked training settings must match. An interrupted epoch is repeated. `--fresh_start` requires an empty output directory; it never deletes an existing run.

### Validation and Debugging

Validation is disabled by default, and the main example uses the final epoch. Enable whole-volume monitoring explicitly:

```bash
python train.py --data_root data/breast --output_dir runs/monitored --max_epoch 100 --val_every 5 --val_limit 4
```

This monitors **raw-intensity patient-mean MAE** on the first four validation patients; it is a runtime check, not the normalized metric reported by `evaluate.py`. Set `--val_limit` to the full validation count for full-set monitoring. Select hyperparameters and inference steps using validation data, not test outcomes. Do not treat the limited-monitoring `model_best.pth` as a full-validation optimum.

```bash
python train.py --data_root data/breast --dry_run --train_limit 1
```

Dry-run performs one forward/loss check without optimizer updates or training checkpoints. For a small development run, use a separate output directory, smaller `--model_chn 32`, and `--train_limit 1`. Such weights are not the full model.

## Inference

```bash
python infer.py --data_root data/breast --model_path runs/tavl_syn/checkpoints/model_final.pth --split test --ode_steps 5 --output_dir outputs
```

The architecture is read from the checkpoint. The final inference configuration uses **5 explicit Euler steps (NFE = 5 per patch)**; `k=0` gives uniform time steps. Five steps are used by default even when `--ode_steps` is omitted, independently of the checkpoint's `sample_step`. The training-time validation and preview setting `--sample_step` remains 20 and does not control final inference. Explicitly pass `--ode_steps` to evaluate a different sampling budget.

Inference loads **Pre pixels only**, not Post1 pixels or tumor masks. The paired-dataset index still requires target files and metadata to exist; this release is a paired research pipeline, not a standalone unpaired deployment API. A patient/slice-seeded Gaussian field is shared across overlapping patches, which are combined using Gaussian blending weights. Source-empty slices remain raw zero; pixels within nonempty slices are not erased using a source mask.

Each run creates a non-overwriting output folder containing:

- `nifti/`: float32 generated Post1 volumes with canonical XYZ geometry.
- `npy_volumes/`: the same raw predictions in ZXY order.
- `png/`: display-only Pre/generated slices and representative previews.
- `generated_manifest.csv`, `run_config.json`, `completion.json`: local output metadata.

PNG windows use source-only percentiles; this display clipping is never applied to saved floating-point predictions or metrics. `--save_npy` additionally saves individual XY slices. Generated logs and manifests contain local paths and patient identifiers and must not be committed.

## Evaluation

```bash
python evaluate.py --data_root data/breast --predictions outputs/epoch_100_test/nifti --split test
```

Use the actual run directory printed by inference. The evaluator requires every patient in the requested split and verifies prediction shape and affine. It prints PSNR, SSIM, and MAE; `--patient_csv outputs/patient_metrics.csv` optionally saves patient-level values.

The evaluation protocol is separate from training normalization:

1. Use each real Post1 volume's min/max as a common mapping for that target and its prediction. Predictions are **not clipped** or independently normalized.
2. Evaluate complete axial XY slices, excluding only slices whose Pre image is entirely zero. No spatial foreground/tumor mask is used.
3. Compute PSNR with data range 1, SSIM with an 11 x 11 Gaussian window (sigma 1.5, population covariance), and MAE in the same normalized scale. SSIM excludes the five-pixel filter border.
4. Average metrics over valid slices within each patient, then report the mean and sample standard deviation across patients. MAE is dimensionless here, not raw MRI intensity or HU.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests cover TAVS routing/scales, timestep batching, detached DSC gradients, exact corrective targets, RNG restoration, noise consistency, synthetic-data preparation, training/resume, source-only inference, volume export, and evaluation. They do not require patient data or formal training. One full-resolution forward test uses CUDA when available and otherwise skips.

## Repository Layout

```text
train.py / engine.py       Training entry point and RF/DSC objectives
infer.py / common.py       Sliding-window Euler sampling and image export
prepare_data.py           Package aligned NIfTI pairs and patient-level indexes
evaluate.py               Whole-slice, patient-balanced evaluation
models/                   U-Net, TAVS, and linear Rectified Flow
datasets/                 Patient indexes, normalization, patch coverage
tests/                    Algorithm and pipeline regression tests
examples/pairs.csv        Synthetic placeholder manifest
```

## License and Research Use

The existing GNU GPL version 2 license text is retained in `LICENSE`; this cleanup does not relicense inherited code. See `NOTICE` for the scope of this source release. Dataset access and redistribution are governed by the dataset provider's terms, separately from the code. Use only trusted checkpoints: full training checkpoints contain pickled optimizer and RNG state and are loaded with `weights_only=False`.

This is research software, not a clinically validated diagnostic system. A prepared dataset, matching configuration, and trained checkpoint are required to reproduce an experiment. No pretrained weights, patient-level results, or unsupported performance claims are bundled.
