# EXPERIMENTS.md — Agent 运行指南（004 仓库）

> 本文件是给 agent 的"实验索引 + 复跑手册"。目标：下次被问到任何实验时，**先读本文件**，可直接定位代码位置并给出运行命令。
> 维护规则：每新增/改动实验，请同步更新本文件对应小节。
> 配套文档：`AGENTS.md`（GPU 策略）。仓库目录约定与整理备忘见本文 §6、§7。

---

## 0. 不可违反的约束（先读）

- **GPU 策略**（`AGENTS.md`）：只用 GPU0。`CUDA_VISIBLE_DEVICES=0`。禁止 GPU1 / 多卡 / `rank % device_count`。脚本默认已设 `CUDA_VISIBLE_DEVICES=0`，无需手动加。
- **Conda 环境**：本仓库所有 2PC/CrypTen 实验用 **`conda run -n shaft`**（Python 3.8, transformers `4.20.0.dev0`, torch 2.0.1）。`run_glue_private_local.py` 内有 `check_min_version("4.20.0.dev0")`，必须用旧 transformers 栈，**不要用 torchenv**。
- **CrypTen 来源**：脚本 `sys.path.insert(0, SHAFT_ROOT)`，用的是 `$SHAFT_ROOT`（即按 README 安装的 SHAFT 改版 crypten，不是 conda 自带的原版 CrypTen）。
- **网络节流**（`tc/netem`）需 root。本机无免密 sudo，节流命令请让用户用 `! sudo ...` 执行，或通过 `scripts/throttle_lo.sh`。
- **i8 vs i10**：Table 3 时延仍用 `scaled_k2_i8`（不换）；Table 3 通信量已手动改为 `scaled_k2_i10` 重测（见 §2.I 的 `e2e_*_k2i10_20260630`）。GLUE 精度（Table 4）：SST-2/QNLI 用 k2i10、CoLA 用 k2i8、**STS-B 用 k3i8**（注意：k3i10 实验在 STS-B 上 pearson=-0.025 灾难性失败，已弃用，见 §7）。

---

## 1. 仓库总览（顶层结构）

| 路径 | 作用 |
|---|---|
| `scripts/runner/` | **正式实验 runner**（端到端 2PC、长度矩阵、网络重放）|
| `scripts/softmax/` | softmax 单算子 benchmark / sweep / 对比 |
| `scripts/shared/` | 共享工具：layernorm 单算子、汇总脚本、网络成本模型、节流配置 |
| `scripts/layernorm/` `scripts/glue_research/` | 研究侧脚本（LayerNorm 泄露/攻击、MLFormer sqrt、GLUE 精度评估）。进论文 |
| `scripts/legacy/` | 历史/诊断脚本（softmax STS-B 调参、test_*、trace_*）。**非主线入口** |
| `artifacts/experiment/` | 当前可信端到端实验结果（GLUE single-party 准确率等）|
| `artifacts/benchmark/` | 当前可信 benchmark（长度矩阵、单算子、网络重放）|
| `artifacts/analysis/` | 安全/泄露分析结果（LayerNorm mask 泄露、token 重建攻击）|
| `paper/` | 论文主稿 `BRiT.docx`、图、参考文献 |

> 外部代码库（不在本仓库，被脚本引用，路径由环境变量配置，见 README §3）：
> - SHAFT/CrypTen：`$SHAFT_ROOT`（按 README §2.3 克隆并 `pip install .` 安装的 SHAFT）
> - 模型/数据：`$DATA_ROOT`（含 `bert-base-cased-sst2`、`glue/sst2/validation.parquet` 等）
> - BLB 仓库与二进制（BLB 复现用，可选）：单独克隆 BLB（USENIX Sec 2025），二进制 `SCI/build/bin/ckks_bert_large_main`
> - OpenBumbleBee（可选，NDSS 2025 复现）：单独克隆 OpenBumbleBee（conda env `bumblebee`/`new_spu`）

---

## 2. 实验清单（每个实验：作用 / 代码 / 运行 / 产物）

> 下表是"目录 → 实验类型"的总索引。每个实验的**精确运行命令**见对应小节。

### 2.A 端到端 2PC 长度矩阵（softmax+layernorm）— **主线 benchmark**

比较 **SHAFT baseline**（`ode_clip_i16` softmax + NR sqrt）vs **our method**（`scaled_k2_i8` softmax + MLFormer sqrt），在 padded SST-2 长度 32/64/128/256、3 种网络下的 comm/rounds/time。

| 长度 | 产物目录 | 状态 |
|---|---|---|
| 32 | `artifacts/benchmark/len32_single_softmax_layernorm_20260528/` | ✅ |
| 64 | `artifacts/benchmark/len64_single_softmax_layernorm_20260528/` | ✅ |
| 128 | `artifacts/benchmark/len128_single_softmax_layernorm_20260528/` | ✅ |
| 256 | `artifacts/benchmark/len256_single_softmax_layernorm_20260615/` | ✅ |
| 汇总 | `artifacts/benchmark/length_scaling_softmax_layernorm_20260528/` | ✅（`length_scaling.md/csv`, `layernorm_perop_scaling.md`）|

**代码**：`scripts/runner/run_len128_single_matrix.py`（虽然名字叫 len128，但用 `--max-length` 参数化）。
**运行（单配置单长度，一个真实 2PC 样本，~55s/run@len256）**：
```bash
cd /path/to/this/repo   # 仓库根（即 QUEST_ROOT）
conda run -n shaft python scripts/runner/run_len128_single_matrix.py \
    --softmax-config scaled_k2_i8 --sqrt-method MLFormer \
    --max-length 128 --max-samples 1 --backend gpu \
    --output-dir artifacts/benchmark/len128_single_softmax_layernorm_20260528/e2e/both_optimized
```
- `--softmax-config`：`ode_clip_i16`(SHAFT) | `scaled_k2_i8` | `scaled_k2_i12` | `scaled_k2_i16`
- `--sqrt-method`：`NR`(SHAFT) | `MLFormer`(ours)
- `--backend gpu|cpu`：gpu 设 `CUDA_VISIBLE_DEVICES=0`，cpu 设为空串。
- 每个长度要跑 4 个主配置：`shaft_original`(ode_clip_i16/NR)、`softmax_only`(scaled_k2_i8/NR)、`layernorm_only`(ode_clip_i16/MLFormer)、`both_optimized`(scaled_k2_i8/MLFormer)；附录加 `i16_both`(scaled_k2_i16/MLFormer)。
- 输出：`<output-dir>/summary_<cfg>.json` + `all_results.json`。

**汇总（生成 metrics.json/md/csv + summary.md）**：
```bash
conda run -n shaft python scripts/shared/summarize_len128_matrix.py \
    --run-dir artifacts/benchmark/len128_single_softmax_layernorm_20260528/ --max-length 128
```
- run-name→(softmax,sqrt) 映射硬编码在脚本 `RUNS`/`OPTIONAL_RUNS` 里。
**跨长度汇总表**：`conda run -n shaft python scripts/shared/summarize_length_scaling.py` → 写 `artifacts/benchmark/length_scaling_softmax_layernorm_20260528/length_scaling.{md,csv}`（glob 各长度 `metrics.json`）。

### 2.B 单算子 per-op 附录（softmax / layernorm）

与 2.A 的 e2e 表不同，这是**单次算子调用**的 comm/rounds（论文 Table 3/4 用）。位于每个长度目录的 `operators/`。

**Softmax 单算子**：`scripts/softmax/benchmark_len128_masked_softmax.py`（参数化，非只 len128）
```bash
conda run -n shaft python scripts/softmax/benchmark_len128_masked_softmax.py \
    --max-length 128 --layers 0 5 11 --repeats 5 \
    --json-output artifacts/benchmark/len128_single_softmax_layernorm_20260528/operators/softmax_len128.json
```
**LayerNorm 单算子**：`scripts/shared/benchmark_layernorm_masked.py`（同款 CLI；cases NR=SHAFT / MLFormer=ours）
```bash
conda run -n shaft python scripts/shared/benchmark_layernorm_masked.py \
    --max-length 128 --layers 0 5 11 --repeats 5 \
    --json-output artifacts/benchmark/len128_single_softmax_layernorm_20260528/operators/layernorm_len128.json
```
- 注意：两个脚本现在都用 GPU（`resolve_device()`→cuda:0）；CPU 单 repeat 数据噪声大、已废弃。
- len256 单算子汇总：`scripts/shared/summarize_len256_single_op.py`；LayerNorm 跨长度：`scripts/shared/summarize_layernorm_length_scaling.py`。
- 论文 Table 3/4 数字（`paper/BRiT.docx`）= 这些单算子数，**不是** e2e 聚合数。

### 2.C GLUE 单方准确率（full-dataset）

真实 2PC 跑全验证集太慢（CoLA 单配置 ~1h），故准确率用 **single-party 动态路径**测。代码：`scripts/runner/run_glue_private_local.py`（也是 2.A runner 的底层 main）。
```bash
conda run -n shaft python scripts/runner/run_glue_private_local.py \
    --task_name sst2 --model_name_or_path "$DATA_ROOT/bert-base-cased-sst2" \
    --validation_file "$DATA_ROOT/glue/sst2/validation.parquet" \
    --max_length 128 --len_data 128 --pad_to_max_length --per_device_eval_batch_size 1 \
    --num_data -1 --softmax_config scaled_k2_i8 --sqrt_method MLFormer \
    --output_dir eval_private/sst2_ours/
```
- `--num_data -1` = 全集；`--acc` 评估准确率；`--comp` 只估计算时间。
- 产物：`artifacts/experiment/<task>_singleparty_bs1_dynamic_*`（cola/sst2/stsb/qnli 均有）。汇总见 `artifacts/experiment/end_to_end_metrics_20260603.md`。

### 2.D BLB/BumbleBee 网络重放（真实 tc/netem）— **当前重点**

把 2.A 的真实 2PC 跑在 loopback `tc/netem` 节流下（替代旧的 SHAFT 公式外推）。目录：`artifacts/benchmark/blb_network_replay_20260617/`。

**网络档**（`scripts/throttle_lo.sh`，单向 delay = RTT/2）：
| profile | 带宽 | RTT | throttle 参数 |
|---|---|---|---|
| `lan_3g_0p3ms` | 3 Gbps | 0.3 ms | `lan3g03` |
| `blb_lan` | 1 Gbps | 0.3 ms | `blb-lan` |
| `blb_wan1` | 400 Mbps | 4 ms | `blb-wan1` |
| `blb_wan2` | 100 Mbps | 4 ms | `blb-wan2` |
| `blb_wan3` | 100 Mbps | 80 ms | `blb-wan3` |

**运行（节流需 sudo，让用户执行节流；runner 本身不需 sudo）**：
1. 节流：用户执行 `! sudo bash scripts/throttle_lo.sh blb-lan`（或 `... del` 清除）。
2. 跑一个 case：
```bash
conda run -n shaft python scripts/runner/run_blb_network_replay.py \
    --network-profile blb_lan --backend gpu --case-name shaft_original --length 128 --suite bert_len_matrix
```
- `--network-profile all|blb_lan|...`，`--backend gpu|cpu|all`，`--case-name`（可多次），`--length`，`--suite all|bert_len_matrix|gpt2_generation`，`--dry-run`，`--rerun-completed`。
- Manifest 由 `scripts/runner/prepare_blb_network_replay.py` 生成（默认写 `artifacts/benchmark/blb_network_replay_20260617/manifest.json`，共 270 个 case）。
- **执行约束**：tc/netem 是全局的、GPU0 独占 → case 必须**串行**跑。

**当前覆盖（2026-06-17 实际产物，详见目录 README）**：
- GPU：len32 + len128 × 5 网络档，每档 4 case（`both_optimized`/`layernorm_only`/`shaft_original`/`softmax_only`），约 39 case 目录完成。**len64/len256/gpt2 尚未跑**。
- CPU：len128 × 5 档（仅 shaft_original + both_optimized）+ len32 少量，约 11 case。
- 权威汇总表：`combined_len128_with_blb_results_20260617.md`（SHAFT/Both vs 旧 BLB 理论/实测对比）、`len128_shaft_vs_both_minutes_20260617.md`。

### 2.E BLB / BumbleBee 原论文复现（在 005 仓库）

> 这些在 **005 仓库**和外部 env，**不在 004**。如需重跑，去 005。

- **BLB**（USENIX Sec 2025）：单独克隆 BLB 仓库（不在本仓库内），二进制 `SCI/build/bin/ckks_bert_large_main`（CKKS+MPC, 2PC, loopback 32 线程）。节流脚本 `blb-main/throttle.sh`。
- **OpenBumbleBee**（NDSS 2025）：单独克隆 OpenBumbleBee（SPU+JAX/Flax），conda env `bumblebee`/`new_spu`，bazel 构建。

### 2.F 安全/泄露分析（LayerNorm mask 泄露）

研究"MaskInv-LN 协议泄露 `W=X·Z²` 是否泄露 token / 标签"。**分析类**，不是性能 benchmark。代码在 `scripts/layernorm/`：
- 泄露分析：`layernorm_mask_leakage.py`、`analyze_layernorm_mask_config_leakage.py`
- 攻击：`layernorm_mask_attribute_attack.py`、`layernorm_token_reconstruction_attack.py`、`layernorm_token_presence_attack.py`、`layernorm_token_recovery_three_settings.py`、`layernorm_paper_inspired_attack.py`
- 产物：`artifacts/analysis/layernorm_mask_leakage_20260527/`（AUC 0.74 标签泄露、token 重建弱）、`artifacts/analysis/layernorm_privacy_attack_20260602/`（top-100 token 重建 AUC 0.04，semihonest share 无增益）。
- 多为 public PyTorch forward + 离线 Monte Carlo 模拟，GPU0。

### 2.G GELU 优化研究（已合并/清理）

Even-Poly / Abs-Poly / A2B-once / fused / batched / poly 拟合等多种 GELU 实现的 benchmark。研究侧，**非主线**。原 `claude/gelu/` 脚本已清理（空目录已删）；仅保留产物 `artifacts/benchmark/gelu_private_forward_20260526/`（K6 GELU 比 K8 省 2.91s、精度相同）。

### 2.H Supernet 微基准→端到端 campaign（5 网络 × GPU/CPU）— 2026-06-20

跨 microbench（单算子 softmax/layernorm）→ 端到端（bert-large len128、gpt2 len64），在 5 档真实 `tc/netem` 网络 × GPU/CPU 下对比 ours（`scaled_k2_i8`+`MLFormer`）vs SHAFT（`ode_clip_i16`+`NR`）。目录：`artifacts/benchmark/supernet_bench_20260619/`。bert-base 端到端**复用** `blb_network_replay_20260617/`，不重跑。

**4 个脚本：**
1. Manifest：`conda run -n shaft python scripts/runner/prepare_blb_network_replay.py --suite supernet --manifest-out artifacts/benchmark/supernet_bench_20260619/manifest.json`（55 case）。
2. 串行驱动（**sudo-free，用户手动节流**）：用户 `! sudo bash scripts/throttle_lo.sh <arg>` 设档 → `conda run -n shaft python scripts/runner/run_supernet_bench.py --network-profile <P> --confirm-throttle <arg> --yes`。按 profile 分组、串行跑该 profile 全部 11 case，`replay_status.json` rc==0 自动跳过。
3. 汇总：`conda run -n shaft python scripts/shared/summarize_supernet_bench.py` → `aggregate/{microbench,bert_large,gpt2,bert_base_reference}.csv` + `all_results.json` + 不变量检查。
4. 画图：`conda run -n base python paper/figures/build_supernet_figures.py`（**用 base，shaft 无 matplotlib**）→ 4 图 × 2 配色。

**关键 gotcha**：单算子脚本 `benchmark_len128_masked_softmax.py` / `benchmark_layernorm_masked.py` 调 `crypten.get_communication_stats()`，只在 SHAFT 版 crypten 里有；必须 `PYTHONPATH=$SHAFT_ROOT`（manifest microbench case 已在 env 设好）。两脚本 2026-06-20 新增 `comm_time_s`/`wall_time_s` 字段（原本只存公式外推时间）。GPT-2 runner 不写 JSON → `scripts/runner/wrap_gpt2_generation.py` 解析 stdout。bert-large 用 `bert-large-uncased`（未微调，head 随机初始化，只读 `private_forward`/`running_time_s` 不读精度）。

**Scope（用户 2026-06-20 决定）**：microbench 全 case GPU+CPU；bert-large `shaft_original` 仅 GPU，`both_optimized` GPU+CPU；gpt2 shaft+both GPU+CPU。

### 2.I scaled_k2_i10 重跑（Table 3 通信量来源）— 2026-06-30

把 §2.A/§2.H 的 BRiT（both_optimized）配置从 `scaled_k2_i8` 换成 `scaled_k2_i10`，重测三个模型的 LAN/WAN2 × CPU/GPU。**Table 3 的通信量列已采用这批 i10 数据；时延列仍用 i8（差异 <1%，不换）。** 三个目录各带一份 `comparison_i8_vs_i10.md` 记录 i8 vs i10 逐项对比。

| 模型 | 产物目录 | i10 vs i8 通信量 | 备注 |
|---|---|---|---|
| GPT-2 (len64) | `artifacts/benchmark/e2e_gpt2_k2i10_20260630/` | 5.08 vs 5.01 GB | rounds 1004→1052 |
| BERT-base (len128) | `artifacts/benchmark/e2e_bertbase_k2i10_20260630/` | 8.26 vs 7.96 GB | rounds 1032→1080 |
| BERT-large (len128) | `artifacts/benchmark/e2e_bertlarge_k2i10_20260630/` | 22.67 vs 21.87 GB | rounds 2028→2124 |

**运行**（与 §2.A 同一 runner，仅 `--softmax-config` 改 `scaled_k2_i10`，输出到新目录）：
```bash
conda run -n shaft python scripts/runner/run_len128_single_matrix.py \
    --backend gpu --softmax-config scaled_k2_i10 --sqrt-method MLFormer \
    --max-length 128 --max-samples 1 --output-dir <new-dir>
```
GPT-2 用 `scripts/runner/wrap_gpt2_generation.py`（或 `run_supernet_bench.py` 驱动）。

**注意**：`artifacts/benchmark/e2e_k2i10_20260626/` 是早期 i10 实验，WAN 档**误用了 blb-wan3 (80ms)**，与 Table 3 的 WAN2 (4ms) 不可比；仅其 `bert_large/gpu/lan`（LAN，blb_lan）被 §2.I 的 BERT-large 对比表复用，其余已弃。

---

---

## 3. 快速复跑速查（最常用 4 条）

```bash
cd /path/to/this/repo   # 仓库根（即 QUEST_ROOT）

# (1) 单长度单配置真实 2PC（主线）
conda run -n shaft python scripts/runner/run_len128_single_matrix.py \
    --softmax-config scaled_k2_i8 --sqrt-method MLFormer --max-length 128 \
    --max-samples 1 --backend gpu \
    --output-dir artifacts/benchmark/len128_single_softmax_layernorm_20260528/e2e/both_optimized

# (2) 汇总某长度 metrics
conda run -n shaft python scripts/shared/summarize_len128_matrix.py \
    --run-dir artifacts/benchmark/len128_single_softmax_layernorm_20260528/ --max-length 128

# (3) 网络重放一个 case（先让用户 ! sudo bash scripts/throttle_lo.sh blb-lan 节流）
conda run -n shaft python scripts/runner/run_blb_network_replay.py \
    --network-profile blb_lan --backend gpu --case-name both_optimized --length 128

# (4) GLUE 单方准确率（全验证集）
conda run -n shaft python scripts/runner/run_glue_private_local.py \
    --task_name sst2 --softmax_config scaled_k2_i8 --sqrt_method MLFormer \
    --num_data -1 --max_length 128 --pad_to_max_length --output_dir eval_private/sst2_ours/
```

---

## 4. 结论性数字（一眼看到主要结果）

- **长度矩阵（e2e，comm 下降 / round 下降）**：len32 7.6%/33.8% → len256 43.7%/31.1%。softmax 单算子 comm 降 ~71%，layernorm rounds 26→5（-80.8%，跨长度稳定）。见 `length_scaling.md`。
- **GLUE 准确率（ours vs SHAFT baseline，接近明文）**：SST-2 0.9197 vs 0.9209；QNLI 0.9023 vs 0.9044；CoLA 0.8265 vs 0.8236（ours 更高）；STS-B 0.8910 vs -0.004（k=3 修复灾难尾部）。见 `end_to_end_metrics_20260603.md`。
- **网络重放（len128，Both GPU vs SHAFT GPU）**：1G/0.3ms 1.38 vs 1.80 min；100M/80ms 12.76 vs 17.88 min（-29%）。在 100M 档已优于旧 BLB 实测（18.5/24.2 min）。见 `combined_len128_with_blb_results_20260617.md`。
- **BLB/BumbleBee 复现**：本机 BLB ~30s HE/block ≈ 论文 28s（可比）；BumbleBee 128-token 7.67min（结论在 005 仓库，见 §2.E）。

---

## 5. 排错速查

- **`check_min_version` / transformers 报错** → 没用 `shaft` env。所有 runner 必须 `conda run -n shaft`。
- **GPU OOM**（gpt2 len128 both）→ 已知，manifest 已排除该 case，别强跑。
- **`length_scaling.csv` 与 `metrics.json` 数字不一致** → 历史上 csv 曾 stale（指向已删的 `artifacts/experiment/` 数据）。重跑 `summarize_length_scaling.py` 即同步。权威源是各长度 `metrics.json/md`。
- **网络重放"未执行"提示** → 见 `blb_network_replay_20260617/README.md` 的真实覆盖表（README 已更新为实际产物）。
- **路径 `artifacts/experiment/len128_single...` 找不到** → 已搬到 `artifacts/benchmark/`。

---

## 6. 仓库目录约定

- `scripts/` — 所有论文相关代码的唯一入口，namespace package（靠 `QUEST_ROOT` 在 sys.path）。按主题分：
  - `runner/` `shared/` `softmax/` + `throttle_lo.sh`：正式实验入口（Table 3/4 性能与精度）。
  - `layernorm/`：LayerNorm 泄露/攻击、MLFormer sqrt、安全分析、public-mask batching（进论文）。
  - `glue_research/`：GLUE 精度评估（Table 4）。
  - `legacy/`：历史/诊断脚本（`test_*`/`trace_*`/`diagnose_*`），非主线入口。
  > 原 `claude/` 目录已于 2026-06-30 合并入 `scripts/`；`artifacts/experiment/public_mask_batching_20260605/` 里的 2 个脚本也已挪到 `scripts/layernorm/`（产物留在 artifacts 原位）。
- `paper/figures/` — 论文图生成脚本（`build_*.py` 等 10 个）+ 生成的 PDF/JSON。图脚本只读 `artifacts/`、不依赖 `scripts/`，故留在 `paper/figures/` 与产出的图放一起。
- `artifacts/` — 实验产物，分 `experiment/`（GLUE 精度）、`benchmark/`（性能 e2e/单算子）、`analysis/`（安全/泄露分析）、`scout/`（跨论文调研）。旧产物隔离区 `legacy_pending_review_20260605/` 已不存在（早期约定，未实际建立）。
- `paper/` — 主稿 `BRiT.docx`、参考文献 `cas-refs.bib`、`figures/`、调研 md。
- `papers/` — 外部参考文献 PDF。

> **代码路径约定**：所有脚本以仓库根为 `QUEST_ROOT`（由 `scripts/paths.py` 从文件位置自动推导）；从仓库根运行（`conda run -n shaft python scripts/...`）。外部依赖（SHAFT/数据/模型）路径通过 `$SHAFT_ROOT` / `$DATA_ROOT` 环境变量配置，见 README §3。

---

## 7. 整理备忘（有用 vs 待清理）

> 2026-06-30 盘点。论文在用的不要动；其余可按需归档或清理。

**🟢 论文在用**
- 主稿：`paper/BRiT.docx` + `cas-refs.bib` + `paper/figures/`。
- Table 3 时延源（i8）：`artifacts/benchmark/supernet_bench_20260619`、`blb_network_replay_20260617`。
- Table 3 通信量源（i10）：`artifacts/benchmark/e2e_{gpt2,bertbase,bertlarge}_k2i10_20260630`。
- Table 4 精度：`artifacts/experiment/{sst2,qnli,cola,stsb}_singleparty_*` + `*_ode_nr_*`(SHAFT 对照) + `plaintext_glue_baselines_20260602`。
- 支撑：`paper/{softmax_layernorm_lan_wan_comparison,bolt_bumblebee_native_vs_crypten,baseline_reproducibility_analysis}.md`、`artifacts/benchmark/invsqrt_error_curve_20260629`。

**🟡 调研/写作参考**（BRiT.docx 正文未直接引用，是素材）
- `paper/{softmax_layernorm_survey,gelu_protocol_survey}.md`、`artifacts/scout/cross_paper_comparison_report_20260616.md`、`writing_prompts.md`（与论文无关，可移出）。

**🔴 旧产物**（2026-06-30 已核实并处理）
- **已删除**（原移入 `artifacts/legacy_20260630/`，已于同日整体删除）：
  - `experiment/stsb_*_k3i10_20260627/`：STS-B 的 k3i10 实验，pearson=-0.025 灾难失败（Table 4 用 k3i8）。
  - `experiment/sst2_*_k3_20260604/`：SST-2 的 k3 替代配置（Table 4 用 k2i10）。
  - `benchmark/sst2_scaled_tradeoff_*`：早期精度 tradeoff 日志。
  - `paper/archive_pending_review_20260605/`：6/2~4 旧草稿，已被 BRiT.docx 取代。
- **暂留主路径**（用户 2026-06-30 决定不清理）：
  - `benchmark/e2e_k2i10_20260626/`：早期 i10，LAN-GPU 仍被 §2.I 复用。
  - `experiment/public_mask_batching_20260605/`：mask batching 探索。
  - `analysis/stsb_*_diagnosis_20260603.json`（2 个）：STS-B 调试 trace。
- 重复 `paper/survey.pdf` 已不存在（仅 `papers/survey.pdf` 一份）。

> **⚠️ 不要删的"看似旧实则进论文"**（已核实被图脚本/汇总引用）：
> - `len32/len64_single_softmax_layernorm_20260528`：被 `summarize_length_scaling.py` glob，喂 length_scaling 图。
> - `stsb_singleparty_bs1_dynamic_k3_20260603`（k3i8）+ `stsb_..._ode_20260603`：STS-B 图脚本 `build_curated_results_figures.py` 的数据源。
> - `bolt_native_20260625`、`softmax_bolt_bumblebee_20260625`：被 `paper/bolt_bumblebee_native_vs_crypten.md` 等支撑文档引用。

