# Phase 1 fixture

`harbor_trial/result.json` is a synthetic subset of Harbor's finalized result
shape. It has a stable attempt ID and input 100/cache 40/output 10, demonstrating
the implemented cost collector without running an agent. It is not a benchmark
result or invoice; missing USD cost remains unknown.

Device tests keep their short command replies inside their four test cases.
There is no shared HTTP server, stream fixture, or dataset fixture in this phase.
The owner supplies recorded rows when replay is implemented in Phase 3.
