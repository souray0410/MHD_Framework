# Validation scope

Packaging preserves the V4 core byte-for-byte and changes only the relative core import in utils.py. Two historical unit-test files used removed criteria_node / gradient_accumulation interfaces; they now exercise the existing frozen criteria callable and current Mermaid labels, matching the previously reviewed tests in commit 568701b. The framework implementation is not changed to accommodate obsolete tests.

Tests use synthetic tensors. Cross-version V3 migration tooling/tests remain on the historical archive; they are not part of this release's advertised API. Application state_dict, output and gradient migration are validated separately.
