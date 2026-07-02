---
name: inv-sqrt-mlformer-optimization
description: MLFormer-style inv_sqrt 协议优化 CrypTen LayerNorm，rounds -38%，已集成到 CrypTen 源码
metadata:
  type: project
---

# MLFormer-style inv_sqrt 优化 CrypTen LayerNorm

## 问题

CrypTen 默认 LayerNorm 使用 NR (Newton-Raphson) 方法计算 `1/√x`，需要 exp 初始化 + 5 次 NR 迭代，Beaver 乘法多达 10 次，是 LayerNorm 的主要通信瓶颈。

## 方案

借鉴已有 `_reciprocal_mlformer_protocol` 的随机掩码思路，推广到逆平方根：

```
目标: Y = 1/√X
步骤:
  1) 受信方生成正随机 Z，明文算 Z²，分别分享 [Z] 和 [Z²]
  2) [W] = [X+eps] * [Z²]       — 1 Beaver
  3) Reveal W，明文算 1/√W
  4) [Y] = [Z] * (1/√W)         — 明文×密文，无通信
验证: Z / √(X·Z²) = 1/√X  (要求 Z>0)
```

## 实验结果

### 单进程 benchmark (shape=(1,128,768), hidden=768)

| 指标 | Original (NR) | MLFormer |
|------|--------------|----------|
| Beaver 乘法 | 12 | **3** (减少 75%) |
| 耗时 | 50ms | **11ms** (4.5x 加速) |
| Max Error vs PyTorch | 0.998 | **0.006** |
| Cosine Similarity | 0.999997 | **1.000000** |

### 真实 2PC 单样本对比 (分布式多进程, ode_clip_i16, len=128)

| 指标 | NR | MLFormer | 变化 |
|------|-----|----------|------|
| Comm rounds | 1,375 | **850** | **-38%** |
| Comm bytes | 8.35 GB | 8.35 GB | 不变 |
| Wall time | 43.3s | **40.9s** | **-5.5%** |

WAN 估算：每样本省 525 rounds × 4ms = 2.1s，436 样本省 ~15 分钟。

### 精度 (x in [0.01, 100])

MLFormer 在极端小值 (x=0.01) 下精度优于 NR：0.08% vs 5.98%。
在正常范围 (x ∈ [0.1, 50]) 两者误差均 < 0.03%。

### 稳定性

实际 BERT LayerNorm 的 variance 范围约 [0.01, 10]，不会低于 0.001。
在此范围内 MLFormer 完全稳定，无需额外处理。

## 实现

**已直接修改 CrypTen 源码**（两个副本）：
- `_inv_sqrt_mlformer()` 函数加入 `approximations.py`
- `inv_sqrt()` 增加 `method == "MLFormer"` 分支
- `default.yaml` 配置 `sqrt_method` 支持 `"MLFormer"` 选项
- 修改位置（在 SHAFT 仓库内）：
  - `$SHAFT_ROOT/crypten/common/functions/approximations.py`
  - 以及 CrypTen 的对应文件（如 conda/site-packages 内的 `crypten/common/functions/approximations.py`）

## 踩坑记录

1. **monkey-patch `approximations.inv_sqrt` 无效**：CrypTen import 时通过 `setattr(CrypTensor, 'inv_sqrt', ...)` 绑定函数到类，之后 patch module 属性不会更新已绑定的类方法。
2. **`cfg.temp_override` 不穿透 multiprocessing spawn**：子进程获得独立的 cfg 实例，读默认配置 NR。解决方案：用环境变量 `CRYPTEN_SQRT_METHOD` 传递，子进程内读环境变量再 override。
3. **LayerNorm eps 被移除**：源码注释 `#先把eps给去了`，原始 inv_sqrt 不加 eps。MLFormer 版本内置 eps=1e-5，在正常 variance 范围内影响可忽略。

## 文件

- `claude/run_sst2_layernorm_mlf.py` — env var 方式全量推理 launcher
- `claude/inv_sqrt_mlformer.py` — 单进程 benchmark（standalone / LayerNorm / stability）
- `claude/test_single_sample.py` — 单样本 2PC 对比工具
