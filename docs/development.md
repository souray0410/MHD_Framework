# Development

Install editable with `pip install -e '.[dev]'` in Python 3.11. Run `python scripts/manage.py test`, then `python -m build`; verify the wheel from outside the checkout. Use a separate environment for each installed release. Commit pins, not a floating branch, identify reproducible application dependencies.

CPU tests use synthetic data. GPU/distributed validation must be explicitly invoked and reported with the actual environment. Historical cross-version migration tests and old full-object checkpoint tools remain on the archive. Current release tests check the installed API against explicit tensor formulas and native model references.

Full extended-architecture acceptance is opt-in (`MHD_EXTENDED_MODELS=1`) in a
separate adequately sized environment with `.[models_extended]`. CPU CI installs
PyTorch 2.8 and `.[models]`; skipped extended/GPU cases are not acceptance.
GPU suites additionally set `MHD_TEST_CUDA=1`, `MHD_MODEL_TEST_DEVICE=cuda:0` and
`MHD_MODELS_TEST_DEVICE=cuda:0`. On two admitted devices, run
`torchrun --standalone --nproc_per_node=2 -m tests.integration.parallel_acceptance
--mode MODE --precision PRECISION --output DIRECTORY` for ddp/fsdp2/tp/gpipe/1f1b
and fp32/bf16/fp16. Receipts include the hardware, tolerances and checkpoint
position; record source commit/digests and environment alongside them.
