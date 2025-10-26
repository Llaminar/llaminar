# Git Hooks for Llaminar

This directory contains Git hook templates for the Llaminar project.

## Pre-Commit Hook

The pre-commit hook ensures code quality by building and testing before commits.

### What It Does

The hook performs 4 steps before allowing a commit:

1. **[1/4] Build V2 Debug** - Ensures Debug build compiles without errors
2. **[2/4] Build V2 Release** - Ensures Release build compiles without errors  
3. **[3/4] Run Unit Tests** - Validates all unit tests pass (using Debug build)
4. **[4/4] Run Integration Tests** - Validates all integration tests pass (using Debug build)

### Why This Approach?

**Builds First, Then Tests**: This catches two classes of issues:
- **Build failures**: CMake configuration errors, missing files, syntax errors
- **Test failures**: Runtime bugs, logic errors, broken assumptions

**Both Debug and Release**: 
- **Debug**: Used for testing (includes assertions, debug symbols)
- **Release**: Ensures optimized code compiles (catches optimization-specific issues)

**Unit + Integration**: 
- **Unit tests**: Fast, isolated component validation (`^V2_Unit_`)
- **Integration tests**: End-to-end validation including MPI, model loading, etc. (`^V2_Integration_`)

### Installation

```bash
# Copy the hook to .git/hooks/
cp .githooks/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

### Usage

The hook runs automatically before every commit:

```bash
git commit -m "Your commit message"
```

**If all checks pass**: Commit proceeds normally  
**If any check fails**: Commit is blocked with error details

### Override

If you need to commit despite failing tests (e.g., work in progress):

```bash
git commit -m "WIP: debugging" --no-verify
```

⚠️ **Warning**: Only use `--no-verify` for WIP commits. All PR merges must pass the hook.

### Performance

Typical runtime on first run:
- Debug build: ~30-60s
- Release build: ~30-60s  
- Unit tests: ~10-40s
- Integration tests: ~5-15s
- **Total: ~2-3 minutes**

Subsequent runs (incremental builds):
- **Total: ~30-60s** (only changed files rebuild)

### Troubleshooting

**Hook fails with build errors:**
```bash
# Run manually to see full error output
cmake --build build_v2 --parallel
cmake --build build_v2_release --parallel
```

**Hook fails with test errors:**
```bash
# Run tests manually to see detailed output
cd build_v2
ctest -R "^V2_Unit_" --output-on-failure --verbose
ctest -R "^V2_Integration_" --output-on-failure --verbose
```

### CI/CD Integration

The same test suite runs in CI/CD pipelines:
- GitHub Actions uses identical `ctest` commands
- Ensures local validation matches CI validation
- Reduces "works on my machine" issues

---

**Last Updated**: October 26, 2025  
**Maintainer**: GitHub Copilot / David Sanftenberg

cmake --build build_v2_release --parallel
```

**Hook fails with test errors:**
```bash
# Run tests manually to see detailed output
cd build_v2
ctest -R "^V2_Unit_" --output-on-failure --verbose
ctest -R "^V2_Integration_" --output-on-failure --verbose
```

The hook automatically detects which build directory to use:
1. `build_v2_coverage` (preferred for coverage builds)
2. `build_v2` (standard V2 builds)
3. `build` (V1 builds)

If no build directory is found, the hook will fail with instructions.

## Future Hooks

Additional hooks can be added here:
- `pre-push` - Run integration tests before pushing
- `commit-msg` - Enforce commit message format
- `post-merge` - Rebuild after pulling changes

## Notes

Git hooks are **not** automatically installed when cloning a repository (for security reasons). Each developer must manually install them after cloning.

Consider adding this to your onboarding documentation:

```bash
# After cloning the repo
cd llaminar
cp .githooks/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```
