# Git Hooks for Llaminar

This directory contains Git hook templates for the Llaminar project.

## Pre-commit Hook

The pre-commit hook runs all unit tests before allowing a commit.

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

**If tests pass**: Commit proceeds normally  
**If tests fail**: Commit is blocked with a summary of failures

### Override

If you need to commit despite failing tests (e.g., work in progress):

```bash
git commit -m "WIP: debugging" --no-verify
```

⚠️ **Warning**: Only use `--no-verify` when necessary. All commits to main branches should have passing tests.

### Customization

To add more test executables, edit the `TEST_EXECUTABLES` array in the hook:

```bash
TEST_EXECUTABLES=(
    "$BUILD_DIR/tests/v2/v2_test_model_loader"
    "$BUILD_DIR/tests/v2/v2_test_pipeline"      # Add new tests here
    "$BUILD_DIR/tests/v2/v2_test_kernels"
)
```

### Build Directory Detection

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
