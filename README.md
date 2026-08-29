# MemSeal

MemSeal investigates path-complete memory admission for compiled LLM serving.
The proposed unit of admission is the maximum per-rank non-KV memory envelope
over legal execution traces, including persistent state accumulated by trace
order, rather than one worst-shape profiling call.

The project is currently at **MS-Q0**, an engineering qualification only. It
checks Qwen3-14B BF16 TP4 correctness, graph/eager dispatch, and common
per-rank memory observability on A100 and Ascend 910B. MS-Q0 cannot establish a
memory gap or support a paper claim.

- Qualification protocol: `docs/MS_Q0_PROTOCOL.md`
- Qualification repair: `docs/MS_Q0_R1_PROTOCOL.md`
- Frozen configuration: `configs/ms_q0.json`
- Upstream baseline audit: `docs/UPSTREAM_BASELINE_AUDIT.md`
- Results: `results/ms_q0/`

The scientific Oracle Census is not allowed to start until MS-Q0 passes and
the stronger baselines are staged: current vLLM on CUDA, the legacy CUDA model
runner where it is stronger, and the official Ascend stack with only the
targeted ATB accounting fix. The reverted ACL-graph pre-estimation patch is not
an admissible baseline.

The first MS-Q0 launch is retained as `blocked_stack_or_asset`: the observer
initialized NPU state before vLLM forked TP workers, while the A100 shell lost
its compiler PATH. Neither engine reached ready. MS-Q0-R1 is a prospective,
source-frozen repair of only those two invocation boundaries.
