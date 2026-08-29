# MS-G0D-Q quick current-stack path census

This is a two-process triage, not the MemSeal scientific Gate. A100 and 910B
each run one Qwen3-14B BF16 TP4 compiled process with an explicit 8 GiB KV slab
and the same ordered trace:

`baseline decode -> sampler first -> sampler repeat -> max prefill -> wide decode -> mixed`.

Every path records synchronized per-rank driver used memory, allocator current
and peak bytes, resolved dispatch, and persistent state before/after the path.
The order is fixed before output so lazy state cannot be reordered after seeing
results.

Continue to the expensive current/patched-stack Oracle Census only if the same
path has at least a 256 MiB signal on both platforms and at least 512 MiB on one
platform, using either persistent driver delta or path-local allocator peak but
never summing nested counters. Otherwise stop MemSeal before staging vLLM
0.28.0 or a Hybrid model.
