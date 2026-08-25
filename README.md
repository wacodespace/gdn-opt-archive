# GDN Opt Archive

Static HTML archive of **Gated DeltaNet (GDN)** optimizations in [SGLang](https://github.com/sgl-project/sglang).

It covers Qwen3 linear-attention on CUDA and HIP. Open [`html/index.html`](html/index.html) and start from the timeline.

SGLang GDN（Gated DeltaNet）优化归档：每条只写背景、收益和 trade-off。从 [`html/index.html`](html/index.html) 读。

## How to read

Each page keeps three sections only:

1. **Background** — why this change existed
2. **Gains** — what it actually bought
3. **Trade-off** — the cost: extra paths, correctness debt, or a later fallback

This is a reading archive, not a dump of kernel source.

## Layout

```
html/index.html     timeline, four threads, recurring trade-offs
html/*.html         one page per optimization
html/style.css
README.md
```

The GDN forward chain most pages sit on:

`in_proj (qkv/z/b/a GEMM)` → `causal_conv1d (+ split qkv)` → `gating` → delta-rule recurrence → gated RMSNorm → `out_proj`

## Four threads

1. **Drop redundant copies / host sync** — Python loops, defensive `clone` / `contiguous`, and `.item()` / `torch.any` bubbles
2. **Fuse operators to cut launches** — gating, MoE gate, fused projection GEMM; dual-stream is the alternative to fusion
3. **Hand-written decode kernels** — CuTe DSL on CUDA, inline ASM on HIP, plus BV / autotune
4. **State layout and consistency** — stride-agnostic conv, VK/KV dual layout, then fixing slot-layout drift

CUDA/HIP splits, fused-proj stride bugs, and shape-gated fallbacks (`T >= 64`, `head_dim == 128`) show up across more than one page. The index lists those as recurring trade-off themes.

## Not in this repo

The generator script and the markdown notes that produce these pages live next to the SGLang tree. They are intentionally left out of this archive.
