# Release policy

V4 is the frozen stable API. V5 is a development preview; main tracks ongoing V5 development. Public names and installable package layout are consistent, while semantic changes are documented per release.

GitHub release names/tags are V4 and V5, and package version metadata is 4 and 5. V5 is explicitly marked as a GitHub prerelease. A plain numeric package version is not a Python prerelease marker: do not assume pip will infer development status from the GitHub badge. These initial distributions are provided through GitHub, not a combined PyPI index.

Published release tags and assets are immutable. Fixes require a new unique version rather than overwriting a published artifact; choose subsequent release identifiers before publication. Prepublication setup refs are finalized before the initial release and must not be used as production dependencies.

Releases require current unit tests, executable examples, wheel and source builds, and import checks outside the checkout. CPU acceptance is not a substitute for GPU/distributed validation. Historical source and old serialized-object environments are described in history.md.
