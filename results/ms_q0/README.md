# MS-Q0 results

This directory is reserved for the frozen qualification, four fresh-process
JSON results, retained logs, and one adjudication artifact. MS-Q0 is an
engineering precondition and cannot be cited as evidence for a MemSeal memory
gap.

`MS_Q0_FREEZE.json` fixes the configuration, protocol, source hashes, four-cell
matrix, and decision labels before any accelerator output.

The first launch is closed as `blocked_stack_or_asset`: neither eager engine
reached ready, and the compiled cells were not run. `MS_Q0_BLOCKED_RAW.tar.gz`
retains both JSON files, full logs, the freeze, and the blocked decision. Its
SHA-256 is recorded separately. The prospective repair lives under
`results/ms_q0_r1/` and cannot overwrite these files.
