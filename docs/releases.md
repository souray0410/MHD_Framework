# Release policy

V4 is the frozen stable API. V5 is a development preview; main tracks ongoing V5 development. Public names and installable package layout are consistent, while semantic changes are documented per release.

GitHub release names/tags are V4 and V5, and package version metadata is 4 and 5. V5 is explicitly marked as a GitHub prerelease. A plain numeric package version is not a Python prerelease marker: do not assume pip will infer development status from the GitHub badge. These initial distributions are provided through GitHub, not a combined PyPI index.

Stable release tags and assets are immutable. The user-authorized exception is replacing the existing V5 preview with the accepted formal version 5: first archive its tag target, assets and SHA256; preserve V4. Only after framework and registered consumer candidate acceptance may V5 be updated. Download the published assets again, verify digests and install/run examples outside the repository before declaring release completion. Future fixes require a new unique version. Prepublication setup refs are finalized before the initial release and must not be used as production dependencies.

Releases require current unit tests, executable examples, wheel and source builds, and import checks outside the checkout. CPU acceptance is not a substitute for GPU/distributed validation. Historical source and old serialized-object environments are described in history.md.
