# MS-G0D-Q results

Reserved for the two-process current-stack triage. These results cannot satisfy
the formal MemSeal Gate.

`MS_G0D_QUICK_FREEZE.json` binds both compiled TP4 processes, the six-path
order, sources, and the 256/512 MiB continuation rule before output.

Decision: **STOP — `stop_no_large_cross_runtime_path_signal`**. Both runs were
valid, but no path reached the 256 MiB floor on both platforms, and none reached
the 512 MiB strong threshold.

| Ordered path | A100 signal | 910B signal | Qualifies |
| --- | ---: | ---: | --- |
| baseline decode | 52.5 MiB | 150.3 MiB | no |
| sampler first | 210.0 MiB | 232.5 MiB | no |
| sampler repeat | 210.0 MiB | 232.5 MiB | no |
| max prefill | 210.0 MiB | 226.0 MiB | no |
| wide decode | 210.0 MiB | 226.0 MiB | no |
| mixed | 105.8 MiB | 121.8 MiB | no |

The signal is the larger of per-rank persistent driver growth and path-local
allocator peak increment; nested counters and ranks are never summed. The
largest first-use persistent growth was the sampler path: approximately
148 MiB/rank on A100 and 182 MiB/rank on 910B. This is real but too small to
justify staging vLLM 0.28, a Hybrid asset, or the formal Oracle Census.
