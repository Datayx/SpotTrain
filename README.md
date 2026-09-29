# SpotTrain

Reliable recovery for interrupted PyTorch training jobs.

AWS Spot Instances can reduce compute cost, but they may be reclaimed with
little notice. If a training job cannot recover correctly, the team loses
compute time and may silently resume from an inconsistent state. SpotTrain is a
Python library for capturing the complete state of a PyTorch training run and
continuing from the next unprocessed batch after an interruption.

> **Project status:** early development. Local checkpoint recovery is
> implemented. AWS Spot interruption handling and S3 storage are planned.

## The problem

Saving model weights is not enough to resume training correctly. A recoverable
job may also depend on:

- Optimizer and learning-rate scheduler state
- Python and PyTorch random-number generators
- CUDA random-number generator state
- Current epoch, global step, and next batch
- Data ordering and sampler position
- Experiment configuration and dataset version

If any of these are missing, a resumed job can repeat or skip examples, change
its stochastic behavior, or follow a different optimization path while still
appearing to work.

SpotTrain starts with one testable invariant:

> Under deterministic conditions, an interrupted-and-resumed run must produce
> the same losses and final parameters as an uninterrupted run.

## Current milestone

The first implementation provides:

- Atomic local checkpoint writes
- Model and optimizer restoration
- Optional scheduler restoration
- Python, CPU, and CUDA random-state restoration
- Exact next-batch continuation metadata
- A recovery-equivalence integration test

Checkpoint files are written to a temporary file, flushed to durable storage,
and atomically renamed. A process failure cannot expose a partially written
checkpoint as the latest valid state.

The equivalence test uses a model with dropout. This makes random-state recovery
observable: restoring only model and optimizer weights is insufficient for the
test to pass.

## Example API

```python
from spottrain import CheckpointManager, TrainingSnapshot

checkpoint = CheckpointManager("checkpoints/run.pt")

checkpoint.save(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    snapshot=TrainingSnapshot(
        epoch=current_epoch,
        next_batch=batch_index + 1,
        global_step=global_step,
    ),
)

snapshot, metadata = checkpoint.load(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
)
```

## Recovery model

```text
Training process
      |
      | save complete state
      v
Temporary checkpoint
      |
      | flush + atomic rename
      v
Valid checkpoint
      |
      | process interruption
      v
New training process
      |
      | restore state
      v
Next unprocessed batch
```

## Run locally

```bash
git clone <repository-url>
cd spottrain
python -m venv .venv
```

Activate the environment:

```bash
# Linux or macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the package and run the tests:

```bash
python -m pip install -e ".[dev]"
pytest
```

## Planned AWS architecture

```text
PyTorch training job
        |
        | EC2 Spot interruption notice
        v
Checkpoint coordinator
        |
        +----> S3: versioned checkpoint artifacts
        |
        +----> DynamoDB: run and recovery state
        |
        +----> CloudWatch: recovery and cost metrics
        |
        v
EventBridge restart workflow
        |
        v
Resumed training job
```

## Roadmap

1. Capture scheduler and resumable sampler state through a higher-level API.
2. Add checkpoint checksums and version compatibility checks.
3. Handle `SIGTERM` and EC2 Spot interruption notices.
4. Upload versioned checkpoints to S3 without replacing the last valid state.
5. Store run leases and recovery state in DynamoDB.
6. Provision the AWS path with CDK.
7. Benchmark checkpoint overhead, recovery time, and avoided recomputation.
8. Test the package with external PyTorch users.

## Engineering questions

This project is intended to explore questions that a basic `torch.save` wrapper
does not answer:

- What state is required for a semantically correct resume?
- How can checkpoint publication remain atomic when storage is remote?
- How should stale workers be prevented from overwriting newer state?
- When is checkpointing more expensive than the recomputation it avoids?
- How should incompatible code, dataset, or model versions be detected?
- Which recovery guarantees are possible with multi-worker data loading?

## Scope

SpotTrain is not an experiment tracker or a managed training platform. Its scope
is interruption detection, consistent checkpoint publication, and measurable
training recovery. It may integrate with existing tracking systems later rather
than replace them.

## License

MIT
