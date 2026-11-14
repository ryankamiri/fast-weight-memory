# Bundled benchmark helpers

These small, unmodified upstream files keep benchmark semantics reproducible without requiring sibling checkouts or runtime source downloads. They are not entire benchmark repositories or datasets.

- BABILong prompts and metrics: https://github.com/booydar/babilong/tree/7a6efee29f5cac03c3c410e6799c80fd2ffe3610
- RULER reference metrics (used for parity tests): https://github.com/NVIDIA/RULER/tree/db43f92ff634bbce1c33d1fb15858aec4bba509a

Each subdirectory includes its upstream Apache-2.0 license. `sources.json` records original file paths, immutable revisions, and SHA-256 checksums. Preserve original files and attribution when updating these snapshots. RULER generation data and BABILong examples are supplied as ordinary dataset inputs, not imported from an external checkout.
