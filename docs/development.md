# Development

Install editable with `pip install -e '.[dev]'` in Python 3.11. Run `python scripts/manage.py test`, then `python -m build`; verify the wheel from outside the checkout. Use a separate environment for each installed release. Commit pins, not a floating branch, identify reproducible application dependencies.

CPU tests use synthetic data. GPU/distributed validation must be explicitly invoked and reported with the actual environment. Historical cross-version migration tests and old full-object checkpoint tools remain on the archive. Current release tests check the installed API against explicit tensor formulas and native model references.
