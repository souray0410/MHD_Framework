# Models registry

The catalog separates architecture definitions from trained artifacts. Current builders cover ResNet, DenseNet and RETFound-MAE; architecture acceptance is tracked separately; an empty weights list means no models artifact has been accepted. Existing synthetic tests or research-application checkpoints do not automatically certify all listed architectures.

See [models design](../docs/models.md). `artifact.schema.json` describes a complete trained-artifact record, not a promise that weights are public or interchangeable. Schema validation is structural only; scientific acceptance requires receipts, verified hashes and strict replay. Restricted paths and participant records do not belong in the public catalog.

Validate the catalog: `python models/validate.py`. Identity regression checks: `python -m unittest discover -s models -p "test_*.py"`. These commands use no GPU or research data.
