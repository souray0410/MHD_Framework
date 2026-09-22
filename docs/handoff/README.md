# Handoff

Current source and release notes are in README.md and docs/development.md.

## Current status publication

Machine-readable current evidence is [status.json](status.json). Source review and live runtime verification are distinct; this publication does not change scientific jobs or certify unfinished experiments. Update this record after material evidence review.
# 2026-09-22 candidate: incomplete accumulation windows

The V5 release candidate tracks accumulation-window position separately from the
total completed microstep count. A forced tail update now closes the window, so
the next epoch starts a fresh window with correct loss weighting. Two three-batch
epochs with accumulation=2 match native PyTorch SGD with momentum at
rtol=1e-5/atol=1e-6. The complete local PyTorch 2.8 CPU suite passed 131 tests;
18 GPU/unsupported precision cases were explicitly skipped.

This candidate is not the formal release. Pending-window checkpoint recovery,
PP mixed-precision metadata, architecture porting, distributed/GPU acceptance,
consumer conversion and release verification remain open. Message remains Tensor.
