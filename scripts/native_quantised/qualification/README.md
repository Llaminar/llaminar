# NativeQuantised qualification tooling provenance

The dormant 21-file tools-v9 import has been removed; it was never activated in
CMake or CI. Historical commit `ce7806d5e79bf5a756ed719a426c77761d92e2c5`
preserves the byte-identical original import and its original manifest SHA256
`72f815965e96bdc483936126b2176f55369422f110827223ff61a9a47997d46a`.
That digest identifies historical source only, not the current repository tools.
Remote diagnostic evidence and originals are unchanged.

The single source-owned analysis authority remains
`tests/v2/cmake/CPUNativeTileRegisterEvidence.py`; its regression registrations
live in `tests/v2/cmake/V2CPUArchitectureTests.cmake`. Dated proposal history is
under `docs/v2/projects/2026-11/2026-11-27-tlsle-static/`.
No qualification or economy acceptance follows from the import or its removal.
