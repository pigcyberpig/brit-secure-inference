# STAR: Secure Transformer Inference Benchmark Code

This repository contains the cleaned experiment code for our
work on **secure (2-party) Transformer inference** built on top of
[CrypTen](https://github.com/facebookresearch/CrypTen) and
[SHAFT](https://github.com/andeskyl/SHAFT).

It implements and benchmarks improved **softmax** and **LayerNorm** protocols
(`scaled_k2` softmax + `MLFormer` inverse-sqrt) against the SHAFT baseline
(`ode_clip_i16` softmax + Newton-Raphson sqrt), across BERT-base / BERT-large /
GPT-2, multiple padded sequence lengths, and several `tc`/`netem` network
profiles.

> The accompanying paper is under review and is **not** included in this
> repository. Generated data, model weights, checkpoints, logs, and paper assets
> are intentionally excluded.

---

## 1. Environment

Verified on Ubuntu with NVIDIA GPUs (CUDA 11.8). All 2PC / CrypTen experiments
use a dedicated conda environment named **`shaft`** (Python 3.8,
`transformers==4.20.0.dev0`, `torch==2.0.1`).

> **GPU policy:** only GPU 0 is used. The runner scripts set
> `CUDA_VISIBLE_DEVICES=0`; do not enable multi-GPU.

---

## 2. Installing Dependencies

This repository does **not** vendor CrypTen. It depends on **SHAFT's modified
CrypTen**, which must be installed first. The steps below mirror the
[SHAFT README](https://github.com/andeskyl/SHAFT).

### 2.1 Create the conda environment

```bash
conda create -n shaft python=3.8 -y
conda activate shaft
```

### 2.2 Install PyTorch (CUDA 11.8)

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118
pip install wheel==0.40.0
```

### 2.3 Install SHAFT (provides the modified CrypTen)

```bash
git clone https://github.com/andeskyl/SHAFT "$SHAFT_ROOT"
cd "$SHAFT_ROOT"
pip install .
```

### 2.4 Install the pinned transformers

Our runner `scripts/runner/run_glue_private_local.py` calls
`check_min_version("4.20.0.dev0")`, so the dev build is required:

```bash
git clone https://github.com/huggingface/transformers
cd transformers
git checkout v4.20.0   # tag closest to the 4.20.0.dev0 dev snapshot we use
pip install .
```

### 2.5 Install this repository's extra dependencies

```bash
conda run -n shaft pip install -r requirements.txt
```

---

## 3. Configure Paths

All external paths are read from environment variables — set them once per
shell:

```bash
# Required: where you cloned & installed SHAFT (its modified CrypTen lives here)
export SHAFT_ROOT=/path/to/SHAFT

# Required: directory holding bert-base-cased-sst2/, bert-large-uncased/, glue/
export DATA_ROOT=/path/to/text-classification

# Optional: local GPT-2 weights (otherwise HuggingFace auto-downloads)
export GPT2_MODEL=/path/to/gpt2    # or just "gpt2"
```

`scripts/paths.py` centralizes this; every entry script reads from it.

---

## 4. Running Experiments

The main commands are listed below. More notes are in
[`EXPERIMENTS.md`](EXPERIMENTS.md).

```bash
cd /path/to/this/repo

# (1) Single-length, single-config real 2PC run (main benchmark)
conda run -n shaft python scripts/runner/run_len128_single_matrix.py \
    --softmax-config scaled_k2_i8 --sqrt-method MLFormer --max-length 128 \
    --max-samples 1 --backend gpu \
    --output-dir artifacts/benchmark/len128_single_softmax_layernorm_20260528/e2e/both_optimized

# (2) Aggregate a length's metrics
conda run -n shaft python scripts/shared/summarize_len128_matrix.py \
    --run-dir artifacts/benchmark/len128_single_softmax_layernorm_20260528/ --max-length 128

# (3) Throttled network replay — set the throttle first (needs sudo):
#       sudo bash scripts/throttle_lo.sh blb-lan
conda run -n shaft python scripts/runner/run_blb_network_replay.py \
    --network-profile blb_lan --backend gpu --case-name both_optimized --length 128

# (4) GLUE single-party accuracy (full validation set)
conda run -n shaft python scripts/runner/run_glue_private_local.py \
    --task_name sst2 --softmax_config scaled_k2_i8 --sqrt_method MLFormer \
    --num_data -1 --max_length 128 --pad_to_max_length --output_dir eval_private/sst2_ours/
```

### Network throttling

`tc`/`netem` throttling requires root. Use the helper script:

```bash
sudo bash scripts/throttle_lo.sh blb-lan      # set profile
sudo bash scripts/throttle_lo.sh del          # clear
bash  scripts/throttle_lo.sh show             # inspect active qdisc (no sudo)
```

Profiles: `lan3g03` (3 Gbps / 0.3 ms), `blb-lan` (1 G / 0.3 ms),
`blb-wan1` (400 M / 4 ms), `blb-wan2` (100 M / 4 ms), `blb-wan3` (100 M / 80 ms).
Throttle is global and GPU 0 is exclusive, so cases must run **serially**.

---

## 5. Directory Layout

```
scripts/
  paths.py            centralized SHAFT_ROOT / DATA_ROOT config
  runner/             formal experiment runners (e2e 2PC, length matrix, network replay)
  softmax/            softmax single-operator benchmark
  shared/             LayerNorm benchmark, aggregators, network profiles
  layernorm/          MLFormer inverse-sqrt / LayerNorm implementation
  throttle_lo.sh      tc/netem throttle helper
EXPERIMENTS.md        concise reproduction notes
```

`artifacts/`, `data/`, `models/`, `checkpoints/`, paper assets, and diagnostic
scripts are intentionally **not** part of this public release.

---

## 6. Citation

If you use this code, please cite: (paper reference to be added upon publication)

```bibtex
@inproceedings{star2026,
  title     = {<!-- paper title -->},
  author    = {pigcyberpig},
  booktitle = {<!-- venue -->},
  year      = {2026}
}
```

---

## 7. Acknowledgements

This work builds directly on **[SHAFT](https://github.com/andeskyl/SHAFT)**
(Kei & Chow, NDSS 2025) and Facebook Research's
**[CrypTen](https://github.com/facebookresearch/CrypTen)**. Baseline comparisons
draw on **BLB** and **BumbleBee**. We gratefully acknowledge their open-source
contributions.

---

## 8. License

MIT — see [`LICENSE`](LICENSE). Note that SHAFT/CrypTen and any third-party code
you install via the steps above remain under their own respective licenses.
