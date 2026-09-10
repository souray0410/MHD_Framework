# Model companion and reproducibility contract

Architecture implementations and trained task results are separate versioned
artifacts. Every published trained result should provide the following companion,
without making the framework core depend on a dataset or training workflow.

| Component | Required information |
|---|---|
| Definition | Architecture ID, dimensional variant, configuration, exact framework/model commit, task head, input/aggregation adapters, node interface |
| Data pipeline | Executable inventory, labels, pairing, exclusions, splits and preprocessing; exact source/config; dataset access instructions and versioned manifests |
| Recipe | Runnable training/evaluation entrypoints; initialization, loss, seeds, optimizer, batch/accumulation, precision, selection and stop rules |
| Environment | Recreate/install lock plus actually observed software, CUDA build, driver, cuDNN, GPU and distributed settings |
| Weights | Complete selected task model, integrity digest, license/access conditions, exact load entrypoint and parent/run identity |
| Evidence | Expected input/output contract, executed replay checks and tolerances, metrics/seed variation and limitations |

The model README links to an explicitly versioned companion for each released task
recipe. Architecture configuration is not a complete training recipe. A trained
result is not advertised as publicly reproducible while its required companion is
private or unavailable. Missing companion or weights must be marked unavailable;
never substitute an untrained example or invented score.

Keep reusable model source once; independently version data preparation, training
recipes and execution artifacts. Dataset-specific workflows may live in a separate
repository/package. Public companions must remain usable without unrelated research
project code. Core/model imports do not download weights, access datasets or start
training. Restricted data and participant manifests remain in authorized storage.

## What users can reproduce

- **Architecture checks:** construct the named model and run the documented
  synthetic forward/gradient/update comparisons with its pinned implementation.
- **Selected-weight replay:** load complete verified task weights with matching
  preprocessing, labels and aggregation, then check predictions/metrics within
  declared numerical tolerances on data the user is authorized to access.
- **Training repetition:** execute the released data and training recipes in a
  compatible environment, register a new run, and compare documented seed results.
  Cross-hardware or nondeterministic retraining is not guaranteed bitwise identical.

Each trained artifact keeps one timestamped execution identity and upstream data/
source/parent digests. An unchanged resume preserves that identity and appends an
observed environment record; a new recipe, seed or deliberate repeat gets a new run.
Cache/model downloads are checked by content identity, not newest modification time.

## Public and private companions

Both distributions use the same versioned configuration, processing/training
entrypoints, strict loader and acceptance rules. Visibility changes disclosure and
resource resolution, not scientific behavior or default parameters. Do not maintain
a simplified public trainer that cannot reproduce the private result.

Public companions include permitted source/recipes/locks/weights and declare any
restricted assets and access procedure. Authorized users can supply those assets
to the same pipeline. Withheld assets are explicit; there is no silent substitute.
Preserve original recipe/artifact identities through export; keep participant
records, credentials and unpublished unrelated project code outside the release.
Publishing requires a reviewed export, not merely changing a visibility flag.

## Current release status

The model catalog currently registers architecture definitions and numerical
acceptance only; it does not yet register released task-trained weights or complete
public task-reproduction companions. See each model's validation scope and the
catalog for actual availability. No new downstream accuracy claim is made here.
