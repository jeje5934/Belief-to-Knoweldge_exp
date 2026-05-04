# onlyextrinsic — Turbo-style BP + Source-Extrinsic Denoiser for LDPC

## Project Goal

Improve 5G LDPC decoding of **Fashion-MNIST images** over an AWGN channel by
augmenting standard Belief-Propagation (BP) with a learned EDM score-denoiser.
The denoiser provides **source extrinsic information** (image-domain prior
knowledge) that is fed back into BP in a turbo-style loop.

This folder is a **self-contained, standalone** experimental branch.
It does not depend on `nocompression` or any sibling directory at runtime.

---

## How onlyextrinsic differs from nocompression

| Aspect | nocompression | onlyextrinsic |
|--------|--------------|---------------|
| Feedback signal | Single `denoiser_weight × extrinsic` added to channel LLR | Two independent knobs: `α × src_ext` + `β × bp_ext` |
| Extrinsic definition | `src_post − channel` (always subtracts channel) | Turbo-style: `bp_ext = BP_post − payload_intr`, `src_ext = src_post − BP_post` |
| BP a priori baseline | Always channel LLR | `payload_intr` (includes previous extrinsic terms) |
| gamma parameter | Present (blends posterior vs extrinsic) | **Removed** — `src_ext` is always pure extrinsic |
| Best known result (Eb/N0 = 0.8 dB) | BLER ≈ 0.057 | BLER ≈ 0.030 |

Key insight: by subtracting `payload_intr` (the actual a priori fed to BP)
instead of `channel`, BP extrinsic is computed correctly in the turbo sense.
This allows a small β ≈ 0.1 to further improve decoding.

---

## Decoding / Update Equations

At each comma in the BP schedule (except after the last chunk):

```
bp_ext   = BP_post  − payload_intr        # BP extrinsic (posterior − a priori)
src_ext  = src_post − BP_post             # source extrinsic (posterior − denoiser input)

new_input = channel + β · bp_ext + α · src_ext
```

Where:
- `channel` = `payload0` = channel intrinsic LLR (frozen, never modified)
- `payload_intr` = the a priori LLR that was actually input to BP for this chunk
- `BP_post` = BP posterior LLR after the chunk's iterations
- `src_post` = denoiser posterior logits (denoiser takes `BP_post` as input)
- `bp_ext` = information BP discovered beyond its a priori input
- `src_ext` = information the denoiser discovered beyond BP posterior

### Definition: BP Extrinsic

`bp_ext = BP_post − payload_intr`

This is the standard turbo extrinsic: posterior minus a priori.
In the **first** chunk, `payload_intr == channel`, so this reduces to
`BP_post − channel` (identical to the nocompression variant).
From the second chunk onward, `payload_intr` already contains previous
extrinsic terms, so the subtraction correctly removes self-feedback.

### Definition: Source Extrinsic

`src_ext = src_post − BP_post`

The denoiser receives `BP_post` as input, produces posterior logits `src_post`,
and the extrinsic is `src_post − BP_post` (posterior minus a priori, turbo style).
Since the denoiser is non-linear, this is the standard **practical approximation**
of true source extrinsic.

### Special Cases

| α | β | Effect |
|---|---|--------|
| 0 | 0 | `new_input = channel` (baseline BP, reset to channel each chunk) |
| 0.1 | 0 | `new_input = channel + 0.1·src_ext` (denoiser-only correction) |
| 0 | 1 | `new_input = channel + bp_ext` (turbo BP feedback, no denoiser) |
| **0.1** | **0.1** | **Best known setting** |

---

## Folder / File Overview

```
onlyextrinsic/
├── README.md                  ← this file
├── checkpoints/
│   └── denoiser.pt            ← pretrained EDM denoiser weights (~27 MB)
├── score_denoiser/
│   ├── __init__.py            ← exports EDMPrecond, SongUNet
│   └── networks.py            ← EDM UNet backbone (from NVlabs/edm)
├── source_prior.py            ← SourcePriorDenoiser: LLR↔image + EDM denoise + extrinsic
├── denoiser.py                ← SoftDenoiser: TF layer wrapping the PyTorch SourcePriorDenoiser
├── decoder.py                 ← LDPC5GDecoder_soft: BP + turbo extrinsic denoiser loop
├── train_denoiser.py          ← trains the EDM denoiser on Fashion-MNIST (denoising score matching)
├── experiment.py              ← α/β sweep: heatmap of BLER over (α, β) grid
├── plot_comparison.py         ← Eb/N0 sweep: baseline BP vs BP+denoiser curves
├── visualize_progression.py   ← per-image step-by-step decoding visualization (3-row plot)
└── results/                   ← output directory for all generated plots
    ├── alpha_beta_sweep_ebno*.png
    ├── comparison.png
    └── progression_idx*_ebno*.png
```

### File Roles

| File | Role |
|------|------|
| `decoder.py` | Core algorithm. Subclasses `LDPC5GDecoder` from Sionna. Implements the multi-chunk BP schedule with turbo extrinsic feedback. The `call()` method runs BP chunks, computes `bp_ext` and `src_ext`, and updates `payload_intr`. |
| `denoiser.py` | TF↔PyTorch bridge. Wraps `SourcePriorDenoiser` (PyTorch) as a Keras layer callable from TF. Handles numpy conversion. Returns pure source extrinsic. |
| `source_prior.py` | PyTorch module. Converts LLR→soft image→EDM denoise→posterior logits→extrinsic. Contains `llr_to_soft_field()` and `soft_field_to_posterior_logits()`. |
| `score_denoiser/networks.py` | EDM-preconditioned SongUNet. Extracted from NVlabs/edm. ~7M parameters with default config (model_channels=64). |
| `train_denoiser.py` | Trains the denoiser using EDM denoising score matching loss on Fashion-MNIST train set (60K images). Saves to `checkpoints/denoiser.pt`. |
| `experiment.py` | Runs a 2D grid sweep over α and β values. Produces a BLER heatmap. |
| `plot_comparison.py` | Sweeps Eb/N0 and compares baseline BP vs BP+denoiser at fixed (α, β). Produces BLER and BER curves. |
| `visualize_progression.py` | Decodes a single image step-by-step, showing baseline BP row, BP+denoiser row, and corrected intrinsic row. Useful for debugging and understanding the algorithm. |

---

## Checkpoint

- **File**: `checkpoints/denoiser.pt`
- **Size**: ~27 MB
- **Format**: PyTorch `state_dict` of `SourcePriorDenoiser` (includes `net.*` keys for EDMPrecond and buffer keys for `bit_weights`, `bit_masks`, `pixel_values`)
- **Training data**: Fashion-MNIST train set (60K images, 28×28 grayscale)
- **Training loss**: EDM denoising score matching: `E[w(σ) · ||D(x+σn; σ) − x||²]`
- **Architecture**: SongUNet, model_channels=64, channel_mult=(1,2,2), num_blocks=2, attn_resolutions=(7,), sigma_data=0.5

To retrain from scratch:
```bash
cd onlyextrinsic
python train_denoiser.py --epochs 5 --batch 128 --lr 2e-4
```
Output: `checkpoints/denoiser.pt`

---

## Exact Run Commands

All commands assume `cwd = onlyextrinsic/`.

### Training the denoiser
```bash
python train_denoiser.py
python train_denoiser.py --epochs 20 --lr 2e-4
```

### Main experiment — α/β sweep (heatmap)
```bash
CUDA_VISIBLE_DEVICES=0 python experiment.py
CUDA_VISIBLE_DEVICES=0 python experiment.py --ebno 0.8 --sigma 0.3
CUDA_VISIBLE_DEVICES=0 python experiment.py --ebno 0.7
```
Output: `results/alpha_beta_sweep_ebno{X}.png`

### Eb/N0 comparison sweep (baseline vs denoiser)
```bash
CUDA_VISIBLE_DEVICES=0 python plot_comparison.py
CUDA_VISIBLE_DEVICES=0 python plot_comparison.py --alpha 0.1 --beta 0.1
CUDA_VISIBLE_DEVICES=0 python plot_comparison.py --alpha 0.1 --beta 0.0 --sigma 0.3
```
Output: `results/comparison.png`

### Per-image visualization
```bash
CUDA_VISIBLE_DEVICES=0 python visualize_progression.py --ebno 0.6 --alpha 0.1 --beta 0.1
CUDA_VISIBLE_DEVICES=0 python visualize_progression.py --img_idx 3 --ebno 0.55 --alpha 0.1 --beta 0
```
Output: `results/progression_idx{N}_ebno{X}.png`

---

## Output Locations

All outputs are saved to `results/` (created automatically).

| Script | Output file pattern |
|--------|-------------------|
| `experiment.py` | `results/alpha_beta_sweep_ebno{X}.png` |
| `plot_comparison.py` | `results/comparison.png` |
| `visualize_progression.py` | `results/progression_idx{N}_ebno{X}.png` |
| `train_denoiser.py` | `checkpoints/denoiser.pt` |

---

## Dependencies

- Python 3.8+
- TensorFlow 2.x (with Keras)
- PyTorch (CPU or CUDA)
- Sionna (for LDPC encoder/decoder, mapper, AWGN channel)
- NumPy, Matplotlib
- torchvision (for Fashion-MNIST download and loading)

The pipeline uses **TensorFlow for the communication chain** (Sionna LDPC encoder/decoder, mapper, AWGN) and **PyTorch for the denoiser** (EDM UNet). The bridge between them is `denoiser.py`, which converts via NumPy.

---

## Best Known Results

Eb/N0 = 0.8 dB, σ = 0.3, schedule = `[10, 10, 10]`, Fashion-MNIST test set,
BATCH=200, ROUNDS=5 → 1000 codewords:

| Setting | BLER | ΔACK vs baseline |
|---------|-----:|---:|
| Baseline BP (no denoiser) | 0.535 | — |
| α=0.05, β=0.05 | 0.072 | +463 |
| α=0.1, β=0 | 0.057 | +478 |
| α=0.1, β=0.05 | 0.035 | +500 |
| **α=0.1, β=0.1** | **0.030** | **+505** |

Eb/N0 sweep (best α=0.1, β=0.1):

| Eb/N0 (dB) | Baseline BLER | Denoiser BLER |
|---:|---:|---:|
| 0.4 | 1.000 | 0.946 |
| 0.6 | 0.985 | 0.407 |
| 0.7 | 0.869 | 0.142 |
| 0.8 | 0.518 | 0.030 |
| 0.9 | 0.167 | 0.003 |
| 1.0 | 0.032 | 0.000 |

Waterfall shift: ~0.3 dB to the left. At Eb/N0 = 0.8 dB: BLER 17× improvement, BER 100× improvement.

---

## Known Assumptions / Approximations

1. **Denoiser extrinsic is approximate.** `src_ext = src_post − BP_post` is the
   standard practical approximation. The denoiser is non-linear, so this is not
   an exact turbo extrinsic in the information-theoretic sense.

2. **BP message state persists across chunks.** `msg_v2c` is passed between BP
   chunks. This means BP does not fully restart; it warm-starts from the previous
   message state with updated a priori input.

3. **Channel LLR is frozen.** `payload0` (the original channel reception) is
   always used as the base of the update equation, never modified.

4. **Denoiser sigma is fixed.** The `sigma` parameter passed to the EDM denoiser
   is a constant (default 0.3) and does not adapt to the actual noise level or
   BP iteration.

5. **Fashion-MNIST only.** The denoiser is trained on 28×28 grayscale Fashion-MNIST.
   Applying to other image types requires retraining.

6. **BPSK modulation.** All experiments use PAM-1 (BPSK), `num_bits_per_symbol=1`.

7. **Coderate ≈ 0.5.** K_PAYLOAD=6272, N_CODEWORD=12600, with CRC24A.

---

## Known Issues / Caveats

1. **TF↔PyTorch bridge is CPU-bound.** The denoiser conversion goes through
   NumPy (`tensor.numpy()` → `torch.from_numpy()`), which forces CPU sync.
   This is a performance bottleneck but does not affect correctness.

2. **No end-to-end training.** The denoiser is pretrained independently.
   Joint training of denoiser + BP is not implemented.

3. **Small evaluation sample.** Default is 1000 codewords per Eb/N0 point
   (BATCH=200, ROUNDS=5). For publication-quality results, increase ROUNDS.

4. **`sigma` tuning is manual.** The denoiser sigma=0.3 was found empirically.
   A sweep over sigma jointly with α/β is not automated.

5. **CRC false-positive rate.** CRC24A has a small but nonzero probability of
   undetected errors. Not accounted for in BLER counting.

---

## Recommended Next Steps

1. **Adaptive sigma scheduling.** Use a different sigma per BP chunk (decreasing
   as BP converges), or tie sigma to the estimated BER.

2. **End-to-end fine-tuning.** Make the TF↔PyTorch bridge differentiable and
   jointly optimize the denoiser weights with the turbo loop.

3. **Scale to larger images / codecs.** Test on CIFAR-10 (32×32×3) or
   higher-resolution images with different LDPC rates.

4. **Three-way extrinsic.** If BP itself has identifiable sub-components,
   finer extrinsic decomposition may be beneficial.

5. **Publication-quality evaluation.** Increase sample size (ROUNDS ≥ 50),
   add confidence intervals, sweep sigma jointly.

6. **Profile and optimize the TF↔PyTorch bridge.** Consider a pure-PyTorch
   LDPC decoder to eliminate the bridge overhead.
