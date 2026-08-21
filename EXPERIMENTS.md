# Reproduction Notes

This repository is a code-only public release. It does not include generated
benchmark data, model weights, checkpoints, logs, paper drafts, or external
reference PDFs.

## Kept Entry Points

| Purpose | Script |
|---|---|
| Single BERT/SST-2 private inference run | `scripts/runner/run_glue_private_local.py` |
| One length/config 2PC benchmark | `scripts/runner/run_len128_single_matrix.py` |
| Build BLB-style network replay manifest | `scripts/runner/prepare_blb_network_replay.py` |
| Execute a prepared network replay manifest | `scripts/runner/run_blb_network_replay.py` |
| GPT-2 generation wrapper for replay manifests | `scripts/runner/wrap_gpt2_generation.py` |
| Softmax single-operator benchmark | `scripts/softmax/benchmark_len128_masked_softmax.py` |
| LayerNorm single-operator benchmark | `scripts/shared/benchmark_layernorm_masked.py` |
| MLFormer inverse-sqrt / LayerNorm protocol | `scripts/layernorm/inv_sqrt_mlformer.py` |
| Summarize length-matrix outputs | `scripts/shared/summarize_len128_matrix.py` |
| Summarize length scaling outputs | `scripts/shared/summarize_length_scaling.py` |

## Environment

Use the `shaft` conda environment described in `README.md`. External paths are
configured with environment variables:

```bash
export SHAFT_ROOT=/path/to/SHAFT
export DATA_ROOT=/path/to/text-classification
export GPT2_MODEL=/path/to/gpt2   # optional
```

GPU runs must use GPU0 only. The runner scripts set `CUDA_VISIBLE_DEVICES=0`
for GPU mode.

## Common Commands

```bash
# Single length/config 2PC benchmark
conda run -n shaft python scripts/runner/run_len128_single_matrix.py \
    --softmax-config scaled_k2_i8 --sqrt-method MLFormer \
    --max-length 128 --max-samples 1 --backend gpu \
    --output-dir artifacts/benchmark/len128/e2e/both_optimized

# Summarize one length directory
conda run -n shaft python scripts/shared/summarize_len128_matrix.py \
    --run-dir artifacts/benchmark/len128 --max-length 128

# Softmax single-operator benchmark
conda run -n shaft python scripts/softmax/benchmark_len128_masked_softmax.py \
    --max-length 128 --layers 0 5 11 --repeats 5 \
    --json-output artifacts/benchmark/len128/operators/softmax_len128.json

# LayerNorm single-operator benchmark
conda run -n shaft python scripts/shared/benchmark_layernorm_masked.py \
    --max-length 128 --layers 0 5 11 --repeats 5 \
    --json-output artifacts/benchmark/len128/operators/layernorm_len128.json
```

Network throttling uses `tc`/`netem` and requires root:

```bash
sudo bash scripts/throttle_lo.sh blb-lan
sudo bash scripts/throttle_lo.sh del
```

Generated outputs under `artifacts/` are ignored by Git.
