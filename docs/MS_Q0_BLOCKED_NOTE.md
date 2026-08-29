# MS-Q0 blocked launch note

The first launch never reached ready: the observer initialized NPU state before
TP-worker fork, while the A100 remote PATH omitted the compiler. It was retained
as `blocked_stack_or_asset`; MS-Q0-R1 repaired only those invocation boundaries.
