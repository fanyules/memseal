# MS-Q0-R1 results

This directory is reserved for the independently frozen repair run. It must
not overwrite or replace any artifact under `results/ms_q0/`.

`MS_Q0_R1_FREEZE.json` binds the unchanged configuration and decision rules to
the R1 protocol and repaired source at pre-output commit `835c798`.

All four cells ran successfully and passed their local checks, but the runner
marked cleanup false because vLLM 0.23 has no explicit engine shutdown method.
The raw R1 archive is retained; R2 changes only that metadata contract.
