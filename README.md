# MemSeal

MemSeal investigates path-complete memory admission for compiled LLM serving.
The proposed unit of admission is the maximum per-rank non-KV memory envelope
over legal execution traces, including persistent state accumulated by trace
order, rather than one worst-shape profiling call.

MS-Q0 passed Qwen3-14B BF16 TP4 correctness, graph/eager dispatch, and common
per-rank memory observability on A100 and Ascend 910B. The subsequent frozen
two-process path census found no large cross-runtime signal, so MemSeal stopped
before current-stack staging, a Hybrid model, or the formal Oracle Gate.

- Qualification protocol: `docs/MS_Q0_PROTOCOL.md`
- Qualification repair: `docs/MS_Q0_R1_PROTOCOL.md`
- Frozen configuration: `configs/ms_q0.json`
- Upstream baseline audit: `docs/UPSTREAM_BASELINE_AUDIT.md`
- Results: `results/ms_q0/`
- Quick path-census result: `results/ms_g0d_quick/`

The scientific Oracle Census was not started. Its stronger baselines would
have required current vLLM on CUDA, the legacy CUDA runner where stronger, and
the official Ascend stack with only the targeted ATB accounting fix.

The first MS-Q0 launch is retained as `blocked_stack_or_asset`: the observer
initialized NPU state before vLLM forked TP workers, while the A100 shell lost
its compiler PATH. Neither engine reached ready. MS-Q0-R1 is a prospective,
source-frozen repair of only those two invocation boundaries.
