# MS-Q0 blocked launch note

Status: **BLOCKED — no engine reached ready; no MemSeal hypothesis was tested**

Date: 2026-08-29

The frozen MS-Q0 launch at commit `d199292` attempted only the eager cell on
each platform. Both failed during engine initialization, before any workload or
memory-path observation:

- On 910B, the runner called `torch.npu.device_count()` and device-name queries
  during its parent-process runtime audit. That initialized NPU state before
  vLLM forked TP workers. Every worker then failed with `Cannot re-initialize
  NPU in forked subprocess`.
- On A100, the nested shell invocation prepended the ninja environment using a
  host-expanded `$PATH`; the resulting remote PATH omitted `/usr/bin`. Triton
  then failed to find a C compiler during the profile run.

The frozen adjudicator returned `blocked_stack_or_asset`. Compiled cells were
not run after both first cells failed at the same engineering boundary. This is
neither a pass nor a negative MemSeal mechanism result.

The raw JSON, full logs, freeze, and blocked decision are retained in
`results/ms_q0/MS_Q0_BLOCKED_RAW.tar.gz`. MS-Q0-R1 may change only:

1. defer accelerator device-name validation until post-init worker RPC, using
   environment/filesystem metadata before worker spawn; and
2. use an explicit complete A100 PATH containing ninja, the selected Python
   environment, and system compiler directories.

The model, TP degree, engine parameters, workload, plans, memory fields, and
decision rules remain unchanged.
