# MS-Q0 TP4 correctness and observability qualification

Status: **PROSPECTIVE — no MS-Q0 output may be generated before freeze**

Date: 2026-08-29

## Purpose

MS-Q0 is the only engineering precondition before the MemSeal Oracle Census.
It asks whether the current two machines can execute the same dense TP4 model
with stable rank dispatch and expose a common minimum set of per-rank memory
signals. It is not scientific evidence for a memory gap.

The previous KneeTP Qwen3-14B TP4 result found non-deterministic 910B output on
the older CANN/HDK stack. Because the 910B host has since moved to the official
vLLM-Ascend 0.23.0, CANN 9.1.0, HDK 25.5.1 combination, MS-Q0 performs one
prospective requalification rather than inheriting either the old failure or a
TP1 success.

## Frozen matrix

| Platform | Model | TP | Plan | Fresh processes |
| --- | --- | ---: | --- | ---: |
| 4x A100 PCIe | Qwen3-14B BF16 | 4 | eager | 1 |
| 4x A100 PCIe | Qwen3-14B BF16 | 4 | runtime default compiled | 1 |
| 4x Ascend 910B | Qwen3-14B BF16 | 4 | eager | 1 |
| 4x Ascend 910B | Qwen3-14B BF16 | 4 | runtime default compiled | 1 |

The A100 lane intentionally uses the already qualified local vLLM 0.23.0
environment only for this engineering check. A pass does not admit it as the
formal current-release baseline. MS-G0D remains blocked until vLLM 0.28.0 is
staged separately and its legacy-runner comparison is frozen. The 910B lane
uses the official matched 0.23.0 plugin stack; CUDA and CANN version numbers do
not need to match.

Both hosts must use the model configuration and weight-index hashes in
`configs/ms_q0.json`. Devices 0-3 are exclusive. Each process uses automatic
KV sizing at `gpu_memory_utilization=0.75`, so this qualification exercises the
runtime profiler without creating an OOM stress test.

## Workload and checks

Each process runs three identical waves. A wave contains eight byte-identical
128-token prompts, greedy decoding, eight output tokens, and no EOS stop. The
instrumented wave records actual FULL/PIECEWISE/eager dispatch on every rank.

Required checks are:

1. all 24 requests finish and each plan is deterministic across its three waves;
2. eager and compiled token IDs match exactly within each platform;
3. all four TP ranks report identical dispatch rows;
4. eager reports only non-graph execution and compiled reports at least one
   real graph replay;
5. all four ranks report synchronized device free/total memory, allocator
   current allocated/reserved bytes, and path-local allocator peaks;
6. resolved KV bytes/tokens and effective graph configuration are retained;
7. no device OOM, host OOM, collective error, timeout, ownership error, or
   incomplete shutdown occurs.

Allocator snapshots and private allocation-history APIs are capability probes,
not pass requirements. Driver-used memory is the aggregate truth; allocator,
graph, collective, and ATB observations are nested attribution signals and
must never be summed as disjoint categories.

## Decision

- All checks pass: `pass_unlock_stack_staging`. This only permits staging the
  formal current/fixed baselines and freezing MS-G0D.
- Output determinism or rank-dispatch identity fails: stop the cross-runtime
  MemSeal claim. Do not reinterpret a TP1 run as a substitute.
- The common required memory fields cannot be observed on either runtime: stop
  for insufficient observability.
- A model asset or engine fails before the checks run: classify the Gate as
  blocked by stack/asset qualification; it is not a positive or negative
  MemSeal mechanism result.

No Hybrid model is downloaded or qualified during MS-Q0. That investment is
allowed only after this dense vertical slice passes.
