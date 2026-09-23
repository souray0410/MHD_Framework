# Release policy

V4 is the frozen research API. V5 is the formal Tensor API release; main tracks ongoing development. Public names and installable package layout are consistent, while semantic changes are documented per release.

GitHub release names/tags are V4 and V5, and package version metadata is 4 and 5. The former V5 development preview is archived before replacement by the formal release. These distributions are provided through GitHub, not a combined PyPI index.

Stable release tags and assets are immutable. The one user-authorized exception is replacing the archived V5 preview with this accepted formal version 5; V4 remains unchanged. The preview tag target, assets and SHA256 are preserved in the private PHD release archive. Re-download the published assets, verify digests and install/run examples outside the repository before declaring publication complete. Future fixes require a new unique version. Applications bind exact accepted commits rather than floating branches.

Releases require current unit tests, executable examples, wheel and source builds, and import checks outside the checkout. CPU acceptance is not a substitute for GPU/distributed validation. Historical source and old serialized-object environments are described in history.md.
