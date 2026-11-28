# Pinned compact sources for the tile-stack role map

`CPUNativeTileStackRoles.json` holds one role-map entry per admitted tree. Each
entry's `compact` member must be exactly the mechanical entry
(`CPUNativeTileStackTable.compact_entry`) of the
`CPUNativeQuantisedMultiScaleKernels.h` version named by its `file_sha256`.

`test_cpu_native_tile_stack_table.py` re-derives every entry from an
authenticated source. That source is either this worktree's header or one of
the historical versions pinned here. Installed test runners have no git, so
the historical versions are committed as byte-exact copies:

- `<sha256>.txt` is the file content. Its name is its own SHA256.
- `manifest.json` binds each SHA256 to the source path, to the git blob ID of
  that content, and to the commit that admitted it.

The test recomputes the SHA256 and the git blob ID from the bytes. It rejects
a fixture without a manifest pin, a pin without a fixture, and any fixture
that no role-map entry uses.

When the worktree header changes and a role-map entry still names the old
version, add that version here:

```sh
git show <commit>:src/v2/kernels/cpu/gemm/CPUNativeQuantisedMultiScaleKernels.h
```

Write that output to a file named by its SHA256, then add the matching pin to
`manifest.json`. Never edit a fixture's content.
