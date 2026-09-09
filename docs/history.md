# History and reproduction

[Pre-toolbox archive](https://github.com/souray0410/MHD_Project/tree/archive/2026_09_09_10_30_34_before_toolbox) preserves the original V1–V5 trees, experiments, migration utilities and tests. Use the exact source recorded by an old experiment to load its full-object pickles.

From this transition onward each branch/release contains one implementation under src/mhd_framework: main develops V5; release/v4 packages frozen V4. The package source version, dependency commit and application environment are explicit. Identical package imports do not assert identical semantics across major versions.

The reorganization itself preserves tensor code. Relative core imports are rebased and exports/package metadata added; source_migration.json records original and packaged digests. V3 migration tools and cross-version comparison tests remain in the archive rather than pulling old frameworks into the new package.
