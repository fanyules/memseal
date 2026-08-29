# Upstream memory-admission baseline audit

Audit date: 2026-08-29

This audit fixes the baseline boundary before MemSeal produces scientific
output. It uses official vLLM and vLLM-Ascend sources only.

## Version boundary

- CUDA current release: vLLM
  [`v0.28.0`](https://github.com/vllm-project/vllm/releases/tag/v0.28.0),
  with audited main commit
  [`cacc429f`](https://github.com/vllm-project/vllm/commit/cacc429f62c3738c9c95093e9bd410e96103221a).
- Ascend current compatible release: vLLM-Ascend
  [`v0.23.0`](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.23.0),
  with audited main commit
  [`28bf3a2e`](https://github.com/vllm-project/vllm-ascend/commit/28bf3a2e8e8db396ee5dcfa6812626eee2d394ab).

The CUDA and CANN lanes need not use equal package versions. Each must use its
own current supported stack.

The present A100 host has driver 570.169 / CUDA 12.8 and an isolated vLLM
0.23.0 environment. The vLLM 0.28.0 PyPI metadata pins PyTorch 2.13.0 and its
published GPU dependency set targets CUDA 13. The formal CUDA lane therefore
requires a separate qualified environment or source build; it must not mutate
the working 0.23.0 environment. This staging cost is deferred until MS-Q0
passes.

## Existing mechanisms that are not MemSeal contributions

Current vLLM already measures a single `profile_run`, separates allocator peak
from aggregate non-PyTorch growth, optionally estimates CUDA graph memory, and
reduces common KV blocks to the minimum supported across ranks. It also offers
explicit `--kv-cache-memory` and an opt-in persisted startup plan. See the
official [`GPUWorker`](https://github.com/vllm-project/vllm/blob/cacc429f62c3738c9c95093e9bd410e96103221a/vllm/v1/worker/gpu_worker.py),
[`kv_cache_utils.py`](https://github.com/vllm-project/vllm/blob/cacc429f62c3738c9c95093e9bd410e96103221a/vllm/v1/core/kv_cache_utils.py),
and startup-plan [PR #47388](https://github.com/vllm-project/vllm/pull/47388).

Therefore per-rank minimum KV, explicit KV bytes, profiler persistence, and
post-capture graph byte reporting are baselines, not novel components. The
defensible MemSeal difference is a legal execution-trace envelope, a
pre-execution canary reducer, and a ready/revoke admission contract.

## CUDA fixes and residual candidates

- CUDA graph estimation was added in
  [PR #30515](https://github.com/vllm-project/vllm/pull/30515) and enabled by
  default in [PR #38284](https://github.com/vllm-project/vllm/pull/38284).
- The current dense Model Runner V2 still has a dummy
  `profile_cudagraph_memory()` returning zero; open
  [Issue #49224](https://github.com/vllm-project/vllm/issues/49224) documents
  capture-time OOM. `VLLM_USE_V2_MODEL_RUNNER=0` is therefore a required strong
  corrected baseline, not an ablation favoring MemSeal.
- Open [Issue #50780](https://github.com/vllm-project/vllm/issues/50780)
  documents the opposite failure: a hybrid-GDN profiling transient charged as
  graph residency, with substantial KV-capacity loss.
- Open [Issue #54122](https://github.com/vllm-project/vllm/issues/54122)
  reports cold compilation state changing the estimated KV capacity. Cold and
  warm compiler-cache lanes must be stratified.

## CANN fixes and residual candidates

- vLLM-Ascend [PR #8289](https://github.com/vllm-project/vllm-ascend/pull/8289)
  added actual graph-byte reporting, a manual KV fast path, and a recommendation
  for the next run.
- Pre-KV ACL graph estimation in
  [PR #9865](https://github.com/vllm-project/vllm-ascend/pull/9865) was reverted
  by [PR #11562](https://github.com/vllm-project/vllm-ascend/pull/11562) after an
  MTP/lmhead-TP/AIV-HCCL hang. It must not be mechanically cherry-picked as a
  corrected baseline.
- Open [Issue #14300](https://github.com/vllm-project/vllm-ascend/issues/14300)
  identifies a roughly 98 MiB per-rank ATB pool allocated after profiling and
  outside the PyTorch allocator. Its focused fix
  [PR #14302](https://github.com/vllm-project/vllm-ascend/pull/14302) is the
  targeted patched-stack baseline once rebased and qualified.

## Required formal baselines

The Oracle Census, if MS-Q0 passes, must compare the same trace, graph coverage,
precision, TP topology, scheduler limits, and feature settings under:

1. current release auto admission;
2. the strongest targeted corrected path above;
3. a Dense-calibrated fixed margin transferred without retuning to held-out
   Hybrid cells;
4. one frozen maximum-request profile;
5. the complete finite execution-trace oracle;
6. the dominance-reduced oracle.

An oracle-tuned scalar margin is reported only as a diagnostic upper bound. If
it is allowed to inspect every path in the same deployment, it is algebraically
equivalent to the scalar oracle envelope and cannot be used to claim a capacity
advantage for MemSeal.
