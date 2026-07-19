"""Contract tests for the one-command NativeVNNI training pipeline."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
PIPELINE = REPOSITORY_ROOT / "scripts" / "train_native_vnni_dispatch.sh"
REFRESH = REPOSITORY_ROOT / "scripts" / "refresh_native_vnni_dispatch_tables.sh"


def test_dry_run_builds_both_scorers_and_seals_one_backend(tmp_path: Path) -> None:
    """One invocation owns build, collection, fitting, install, and sealing."""

    result = subprocess.run(
        (
            str(PIPELINE),
            "--backend",
            "cuda",
            "--corpus-root",
            str(tmp_path / "corpora"),
            "--workspace-root",
            str(tmp_path / "work"),
            "--skip-scorer-tests",
            "--install",
            "--dry-run",
            "--",
            "--cuda-measurement-lanes",
            "2",
        ),
        cwd=REPOSITORY_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "v2_native_vnni_leaf_primary_scorer_cuda" in result.stdout
    assert "v2_native_vnni_leaf_primary_scorer_rocm" in result.stdout
    assert "v2_perf_cuda_native_vnni_decode_trainer" in result.stdout
    assert "refresh_native_vnni_dispatch_tables.sh" in result.stdout
    assert "--install" in result.stdout
    assert "native_vnni_dispatch.corpus_bundle" in result.stdout
    assert " seal " in result.stdout
    assert "--refresh-argument=--cuda-measurement-lanes" in result.stdout


def test_profiler_reuse_is_exclusive_to_fit_only_transactions() -> None:
    """Profiler sidecars cannot bypass collection outside --skip-sweep."""

    result = subprocess.run(
        (
            str(REFRESH),
            "--backend",
            "cuda",
            "--profile",
            "all",
            "--reuse-profiler-evidence",
            "--dry-run",
        ),
        cwd=REPOSITORY_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode != 0
    assert "--reuse-profiler-evidence requires --skip-sweep" in result.stderr


def test_turnkey_cpu_prefill_discovers_bundle_replay_recipe() -> None:
    """The public one-command driver must not require manual lineage flags."""

    source = PIPELINE.read_text(encoding="utf-8")
    assert "cpu_prefill_replay_recipe.v1.json" in source
    assert "--cpu-prefill-fit-replay-recipe" in source


@pytest.mark.parametrize(
    "private_arguments",
    (("--shapes", "PrivateBackendOverlay"), ("--shapes=PrivateBackendOverlay",)),
)
def test_turnkey_pipeline_rejects_backend_private_shape_overlays(
    tmp_path: Path,
    private_arguments: tuple[str, ...],
) -> None:
    """Production shape additions must update the one shared inventory."""

    result = subprocess.run(
        (
            str(PIPELINE),
            "--backend",
            "rocm",
            "--corpus-root",
            str(tmp_path / "corpora"),
            "--workspace-root",
            str(tmp_path / "work"),
            "--dry-run",
            "--",
            *private_arguments,
        ),
        cwd=REPOSITORY_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode != 0
    assert "add shapes to the shared inventory" in result.stderr


def test_existing_corpus_selects_verified_fit_only_replay(tmp_path: Path) -> None:
    """A matching generation is refit without collection or profiler launches."""

    common = (
        str(PIPELINE),
        "--backend",
        "cpu",
        "--corpus-root",
        str(tmp_path / "corpora"),
        "--workspace-root",
        str(tmp_path / "work"),
        "--skip-build",
        "--skip-scorer-tests",
        "--no-lfs-pull",
        "--dry-run",
    )
    discovery = subprocess.run(
        common,
        cwd=REPOSITORY_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert discovery.returncode == 0, discovery.stderr
    prefix = "Published NativeVNNI Git LFS corpus: "
    publication = next(
        line.removeprefix(prefix)
        for line in discovery.stdout.splitlines()
        if line.startswith(prefix)
    )
    corpus = Path(publication)
    corpus.mkdir(parents=True)
    (corpus / "corpus.manifest.json").write_text("{}\n", encoding="utf-8")

    replay = subprocess.run(
        common,
        cwd=REPOSITORY_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert replay.returncode == 0, replay.stderr
    assert "corpus_bundle verify" in replay.stdout
    assert "--skip-sweep" in replay.stdout
    assert "--reuse-profiler-evidence" in replay.stdout
    assert " corpus_bundle seal " not in replay.stdout
