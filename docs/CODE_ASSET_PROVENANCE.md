# Code asset provenance

MemSeal may reuse small, audited utilities from the user's existing local
research repositories:

- GraphLease `scripts/run_g0.py`: cross-runtime worker resource snapshots,
  resolved KV/graph configuration, dispatch tracing, and failure-preserving
  process artifacts.
- KneeTP `scripts/run_offline_probe.py`: per-rank peak-counter reset/read and
  graph/eager TP dispatch checks.
- GraphLease `scripts/run_gb_q0.py`: official 910B stack validation.

Any reused function is adapted under the same user-owned workspace and noted
in source comments. MemSeal does not copy old decisions or results. GraphTE's
private allocator-history hook and SemBridge's coarse driver observer remain
optional references and are not treated as path-complete instrumentation.
