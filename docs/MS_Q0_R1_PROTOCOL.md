# MS-Q0-R1 prospective qualification repair

Status: **PROSPECTIVE — no R1 output may be generated before its own freeze**

Date: 2026-08-29

MS-Q0-R1 is not a reinterpretation of the blocked MS-Q0 launch. It reruns the
same four-cell matrix and unchanged `configs/ms_q0.json` after two minimal
control-plane repairs:

1. The parent process must not call `torch.cuda` or `torch.npu` device APIs
   before `LLM(...)` creates TP workers. Pre-init qualification uses only the
   frozen visibility environment and version files. Four device names and rank
   IDs are validated from synchronized post-init worker snapshots.
2. The A100 invocation uses this literal PATH, without shell-variable
   expansion:

   ```text
   /root/miniconda3/envs/qwen36-shard/bin:
   /root/miniconda3/envs/rimlink-vllm023/bin:
   /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
   ```

All scientific and engineering semantics remain those in
`docs/MS_Q0_PROTOCOL.md`: Qwen3-14B BF16, TP4, devices 0-3, eager and runtime
default compiled, three waves of eight identical prompts, identical token and
dispatch requirements, the same memory fields, and the same four verdicts.

R1 outputs are written only under `results/ms_q0_r1/`. The old raw files and
decision remain immutable under `results/ms_q0/`. A successful R1 still unlocks
only current-stack staging; it cannot pass the MemSeal Oracle Census.
