# Scripts Workspace

## Main Entry

- `runner/run_glue_private_local.py`: canonical private inference runner

## Subdirectories

- `softmax/`: softmax single-operator benchmark
- `shared/`: shared benchmark utilities, summaries, and network profiles
- `runner/`: formal experiment runners
- `layernorm/`: MLFormer inverse-sqrt / LayerNorm protocol code

## Usage Note

The old root-level compatibility script links have been removed. Invoke scripts from their real paths under `scripts/`.
