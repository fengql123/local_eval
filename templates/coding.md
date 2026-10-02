# Implement serving for {{ task.model }}

This task measures {{ task.workload }} on {{ machine.category }}. Harbor completed
the reference search and its gate before this coding step. Read `/workspace/run.md` for the
candidate interface, and inspect the recorded public inputs at
`/workspace/data/public`. The original source checkpoint is at `/models/source`;
baseline checkpoint locations and native model tags are listed in
`/workspace/run.md`. Identical source/baseline checkpoints share the same files.

Build and tune the serving implementation in `/workspace`. You may choose any
implementation, dependencies, server settings, or tuning approach, including
quantizing the source checkpoint. Write derived weights and other tuning outputs
inside `/workspace`; the supplied original checkpoints remain read-only. Preserve the
fixed evaluator request and scoring policy described in the benchmark documentation.
Leave a runnable candidate launch script when you finish; Harbor then invokes the
verifier in this same sandbox.
