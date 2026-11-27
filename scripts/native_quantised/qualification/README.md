# NativeQuantised qualification tooling — UNVERIFIED import

Canonical source import: `factory@192.168.178.79:/home/factory/pr73-production/tools-v9/`.
All 21 files are byte-preserved. `tools-SHA256SUMS-v9` SHA256 is
`72f815965e96bdc483936126b2176f55369422f110827223ff61a9a47997d46a`.
The manifest authenticates the historical source, not current qualification.

This import is NOT activated in CMake, CI or any qualification workflow.
Historical shell launchers retain their original absolute paths and must not be
run as installed repository commands. Review any invocation separately.
No production/economy acceptance follows from importing these tools.

Verify source integrity from this directory with `shasum -a 256 -c tools-SHA256SUMS-v9`.
The import commit preserves the original implementation for negative controls;
subsequent proposed changes must be reviewed before qualification use. Do not
edit or replace the remote originals. Owner owns push and activation.
