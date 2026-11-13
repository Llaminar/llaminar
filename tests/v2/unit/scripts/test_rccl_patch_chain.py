#!/usr/bin/env python3
"""Device-free actual-Git replay of atomic canonical RCCL patch-chain upgrades."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location(
    "rccl_patch_chain", ROOT / "scripts/docker/apply-rccl-patch-chain.py")
CHAIN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CHAIN
SPEC.loader.exec_module(CHAIN)


class RCCLPatchChainTests(unittest.TestCase):
    """Preserve every real source byte on rejection and mtimes on repeated no-op."""

    def setUp(self):
        """Generate overlapping patches using Git itself, including a space in paths."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.source = self.root / "source tree"
        self.author = self.root / "patch-author"
        self.author.mkdir()
        CHAIN.git(self.author, ["init", "--quiet"])
        self.base = {"src/engine state.cpp": "// base owner\nint value = 0;\n",
                     "src/worker.cpp": "// worker owner\n" + "// untouched tail\n" * 20,
                     "src/storage.cpp": "// original storage\n"}
        self.write(self.author, self.base)
        CHAIN.git(self.author, ["add", "--all"])
        before = CHAIN.git(self.author, ["write-tree"]).stdout.decode().strip()
        states = [dict(self.base)]
        for change in (
            {"src/engine state.cpp": "// captured owner\nint value = 1;\n"},
            {"src/worker.cpp": "// worker owner\nint created_thread = 1;\n" + "// untouched tail\n" * 20},
            {"src/engine state.cpp": "// protocol owner\nint value = 2;\n"},
            {"src/engine state.cpp": "// small AllGather owner\nint value = 3;\n"},
        ):
            state = dict(states[-1]); state.update(change); states.append(state)
        self.patches = []
        for index, state in enumerate(states[1:]):
            self.write(self.author, state)
            CHAIN.git(self.author, ["add", "--all"])
            patch = self.root / (str(index) + ".patch")
            patch.write_bytes(CHAIN.git(self.author, ["diff", "--cached", "--binary", "--full-index", before]).stdout)
            self.patches.append(patch)
            before = CHAIN.git(self.author, ["write-tree"]).stdout.decode().strip()
        self.upgrade = self.patches[1]
        self.current = states[-1]
        self.previous = dict(self.current); self.previous["src/worker.cpp"] = self.base["src/worker.cpp"]
        self.source.mkdir()
        self.write(self.source, self.base)

    @staticmethod
    def write(directory, state):
        """Populate only fixture-owned files; no real SDK source is involved."""
        for name, data in state.items():
            path = directory / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(data)

    def snapshot(self):
        """Observe bytes, modes and mtimes, including unrelated files."""
        return {str(path.relative_to(self.source)): (path.read_bytes(), path.stat().st_mode, path.stat().st_mtime_ns)
                for path in self.source.rglob("*") if path.is_file()}

    def apply(self):
        """Use the production transaction with its exact ordered chain."""
        return CHAIN.apply_chain(self.source, self.patches, (self.upgrade,))

    def second_upgrade(self):
        """Author one later independent repair while retaining actual overlapping patches."""
        before = CHAIN.git(self.author, ["write-tree"]).stdout.decode().strip()
        current = dict(self.current)
        current["src/storage.cpp"] = "// original storage\nint exclusive_creation = 1;\n"
        self.write(self.author, current)
        CHAIN.git(self.author, ["add", "--all"])
        patch = self.root / "later-storage.patch"
        patch.write_bytes(CHAIN.git(self.author, ["diff", "--cached", "--binary", "--full-index", before]).stdout)
        patches = [self.patches[0], patch, *self.patches[1:]]
        return patches, (patch, self.upgrade), current

    def test_published_and_intermediate_predecessors_both_upgrade(self):
        """A private intermediate build cannot displace the last published generation."""
        patches, upgrades, current = self.second_upgrade()
        for previous in (self.previous, self.current):
            with self.subTest(worker=previous["src/worker.cpp"]):
                self.write(self.source, previous)
                self.assertTrue(CHAIN.apply_chain(self.source, patches, upgrades))
                self.assertEqual({name: (self.source / name).read_text() for name in current}, current)
                after = self.snapshot()
                self.assertFalse(CHAIN.apply_chain(self.source, patches, upgrades))
                self.assertEqual(self.snapshot(), after)

    def test_nonhistorical_mix_of_new_repairs_rejects_without_mutation(self):
        """The newer repair without its predecessor is not an authored source generation."""
        patches, upgrades, current = self.second_upgrade()
        mixed = dict(current)
        mixed["src/worker.cpp"] = self.base["src/worker.cpp"]
        self.write(self.source, mixed)
        before = self.snapshot()
        with self.assertRaises(CHAIN.PatchChainError):
            CHAIN.apply_chain(self.source, patches, upgrades)
        self.assertEqual(self.snapshot(), before)

    def test_invalid_predecessor_declarations_reject_without_mutation(self):
        """No empty, duplicated or foreign history can broaden source acceptance."""
        foreign = self.root / "foreign.patch"
        foreign.write_bytes(self.upgrade.read_bytes())
        for upgrades in ((), (self.upgrade, self.upgrade), (foreign,)):
            with self.subTest(upgrades=upgrades):
                before = self.snapshot()
                with self.assertRaises(CHAIN.PatchChainError):
                    CHAIN.apply_chain(self.source, self.patches, upgrades)
                self.assertEqual(self.snapshot(), before)

    def test_fresh_complete_chain_and_repeat_preserve_mtimes(self):
        """Only one validated combined transition writes real source; repeat is inert."""
        self.assertTrue(self.apply())
        self.assertEqual({name: (self.source / name).read_text() for name in self.current}, self.current)
        after = self.snapshot()
        self.assertFalse(self.apply())
        self.assertEqual(self.snapshot(), after)

    def test_exact_previous_chain_upgrades_despite_reverse_context_overlap(self):
        """Later Doxygen changes cannot disguise an already-applied earlier patch."""
        self.write(self.source, self.previous)
        check = CHAIN.git(self.source, ["apply", "--reverse", "--check", str(self.patches[0])], check=False)
        self.assertNotEqual(check.returncode, 0)
        self.assertTrue(self.apply())
        self.assertEqual((self.source / "src/worker.cpp").read_text(), self.current["src/worker.cpp"])
        self.assertFalse(self.apply())

    def test_current_chain_preserves_unrelated_files_and_nonconflicting_edits(self):
        """The scratch index contains only patch paths and never resets local work."""
        self.write(self.source, self.current)
        unrelated = self.source / "user-notes.txt"; unrelated.write_text("keep this\n")
        path = self.source / "src/worker.cpp"
        path.write_text(path.read_text() + "\n// independent local edit\n")
        before = self.snapshot()
        self.assertFalse(self.apply())
        self.assertEqual(self.snapshot(), before)

    def test_unknown_hunk_edit_rejects_without_any_source_mutation(self):
        """No per-patch real writes occur before full reverse/forward validation."""
        self.write(self.source, self.previous)
        (self.source / "src/engine state.cpp").write_text("// foreign owner\nint value = 99;\n")
        before = self.snapshot()
        with self.assertRaises(CHAIN.PatchChainError): self.apply()
        self.assertEqual(self.snapshot(), before)

    def test_partial_chain_rejects_without_mutation(self):
        """A partially completed installer state is not silently advertised as supported."""
        CHAIN.git(self.source, ["apply", str(self.patches[0])])
        before = self.snapshot()
        with self.assertRaises(CHAIN.PatchChainError): self.apply()
        self.assertEqual(self.snapshot(), before)

    def test_missing_late_patch_cannot_leave_earlier_mutations(self):
        """An invalid last repair fails before the first real source write."""
        self.patches[-1].write_text("not a patch\n")
        before = self.snapshot()
        with self.assertRaises(CHAIN.PatchChainError): self.apply()
        self.assertEqual(self.snapshot(), before)

    def test_git_inventory_handles_spaces_and_rejects_duplicate_patch_set(self):
        """Paths come from Git's parser rather than ad hoc diff-header tokenization."""
        self.assertEqual(CHAIN.patch_paths(self.patches[0]), ["src/engine state.cpp"])
        with self.assertRaises(CHAIN.PatchChainError):
            CHAIN.apply_chain(self.source, [*self.patches, self.patches[0]], (self.upgrade,))

    def test_partial_hunk_rejects_without_mutation(self):
        """Neither a forward nor reverse patch may guess a partially edited hunk."""
        (self.source / "src/engine state.cpp").write_text("// captured owner\nint value = 0;\n")
        before = self.snapshot()
        with self.assertRaises(CHAIN.PatchChainError): self.apply()
        self.assertEqual(self.snapshot(), before)

    def test_source_nested_in_another_repository_is_not_silently_ignored(self):
        """Git's caller prefix must not filter the requested source's patch paths."""
        CHAIN.git(self.root, ["init", "--quiet"])
        self.assertTrue(self.apply())
        self.assertEqual((self.source / "src/worker.cpp").read_text(), self.current["src/worker.cpp"])
        self.assertFalse(self.apply())

    def test_inventory_of_repeated_file_sections_copies_one_actual_path(self):
        """Git may parse more than one complete section for a real affected path."""
        combined = self.root / "combined.patch"
        combined.write_bytes(self.patches[0].read_bytes() + self.patches[2].read_bytes())
        self.assertEqual(CHAIN.patch_paths(combined), ["src/engine state.cpp"])

    def test_symlink_affected_path_rejects_without_touching_target(self):
        """A selected source path cannot redirect the transaction outside its source root."""
        target = self.root / "outside.cpp"; target.write_text(self.base["src/engine state.cpp"])
        path = self.source / "src/engine state.cpp"; path.unlink(); path.symlink_to(target)
        before = target.read_bytes()
        with self.assertRaises(CHAIN.PatchChainError): self.apply()
        self.assertEqual(target.read_bytes(), before)
        self.assertTrue(path.is_symlink())

    def test_concurrent_affected_edit_is_preserved_and_rejected(self):
        """Recheck actual affected bytes before final validation/application."""
        original_git = CHAIN.git
        changed = False
        def alter_during_sandbox(directory, arguments, **kwargs):
            nonlocal changed
            result = original_git(directory, arguments, **kwargs)
            if not changed and arguments[:2] == ["diff", "--cached"]:
                changed = True
                (self.source / "src/worker.cpp").write_text("// concurrent user work\n")
            return result
        with mock.patch.object(CHAIN, "git", side_effect=alter_during_sandbox):
            with self.assertRaises(CHAIN.PatchChainError): self.apply()
        self.assertEqual((self.source / "src/worker.cpp").read_text(), "// concurrent user work\n")
        self.assertEqual((self.source / "src/engine state.cpp").read_text(), self.base["src/engine state.cpp"])

    def test_inherited_git_index_cannot_modify_an_external_index(self):
        """All Git operations isolate index/config environment supplied by callers."""
        sentinel = self.root / "outside-index"; sentinel.write_bytes(b"keep-index")
        with mock.patch.dict("os.environ", {"GIT_INDEX_FILE": str(sentinel)}): self.assertTrue(self.apply())
        self.assertEqual(sentinel.read_bytes(), b"keep-index")


if __name__ == "__main__":
    unittest.main()
