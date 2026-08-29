# MS-Q0-R2 cleanup-metadata repair

R1 completed all four cells and all workload checks, but vLLM 0.23 exposes no
`LLMEngine.shutdown()` method. R2 keeps the model, matrix, workload, sources,
and decision thresholds unchanged except that a successful fresh-process exit
is recorded as the cleanup mode when no explicit engine method exists.

R2 has its own pre-output freeze and writes only under `results/ms_q0_r2/`.
It does not overwrite the MS-Q0 or R1 attempts.
