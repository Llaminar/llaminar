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


def test_turnkey_cpu_prefill_authenticates_its_measurement_plan(
    tmp_path: Path,
) -> None:
    """CPU prefill plan changes must select a fresh resumable workspace."""

    result = subprocess.run(
        (
            str(PIPELINE),
            "--backend",
            "cpu-prefill",
            "--corpus-root",
            str(tmp_path / "corpora"),
            "--workspace-root",
            str(tmp_path / "work"),
            "--skip-build",
            "--skip-scorer-tests",
            "--dry-run",
        ),
        cwd=REPOSITORY_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--inventory-source" in result.stdout
    assert "prefill_matrix.py" in result.stdout
    assert "native_vnni_cpu_prefill_split_v13.json" in result.stdout


def test_turnkey_rejects_cpu_prefill_install() -> None:
    """Offline prefill research cannot replace production heuristics."""

    result = subprocess.run(
        (
            str(PIPELINE),
            "--backend",
            "cpu-prefill",
            "--install",
            "--dry-run",
        ),
        cwd=REPOSITORY_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 2
    assert "ordinary prefill is heuristic-only" in result.stderr


def test_all_backend_transaction_excludes_prefill_install() -> None:
    """The production bundle owns M=1/grouped policies only."""

    source = PIPELINE.read_text(encoding="utf-8")
    loop = next(
        line.strip()
        for line in source.splitlines()
        if line.strip().startswith("for selected_backend in ")
    )
    assert loop == "for selected_backend in cpu cuda rocm; do"
    assert "for selected_backend in cpu cpu-prefill cuda rocm; do" not in source


def test_fit_threshold_does_not_select_a_new_measurement_corpus(
    tmp_path: Path,
) -> None:
    """Promotion policy may be changed without recollecting kernel evidence."""

    publications = []
    for passing_percent in ("0", "95"):
        result = subprocess.run(
            (
                str(PIPELINE),
                "--backend",
                "cpu-prefill",
                "--corpus-root",
                str(tmp_path / "corpora"),
                "--workspace-root",
                str(tmp_path / "work"),
                "--skip-build",
                "--skip-scorer-tests",
                "--minimum-passing-domain-percent",
                passing_percent,
                "--dry-run",
            ),
            cwd=REPOSITORY_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert (
            f"--minimum-passing-domain-percent {passing_percent}"
            in result.stdout
        )
        publications.append(next(
            line.removeprefix("Published NativeVNNI Git LFS corpus: ")
            for line in result.stdout.splitlines()
            if line.startswith("Published NativeVNNI Git LFS corpus: ")
        ))

    assert publications[0] == publications[1]


def test_cpu_checkpoint_controls_resume_the_same_turnkey_workspace(
    tmp_path: Path,
) -> None:
    """Bounded and resumed collection must not fork corpus identity."""

    publications = []
    workspaces = []
    argument_sets = (
        (),
        ("--cpu-batch-limit", "10"),
        ("--cpu-batch-limit=1", "--resume-cpu-partials"),
    )
    for forwarded_arguments in argument_sets:
        result = subprocess.run(
            (
                str(PIPELINE),
                "--backend",
                "cpu-prefill",
                "--corpus-root",
                str(tmp_path / "corpora"),
                "--workspace-root",
                str(tmp_path / "work"),
                "--skip-build",
                "--skip-scorer-tests",
                "--dry-run",
                "--",
                *forwarded_arguments,
            ),
            cwd=REPOSITORY_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        publications.append(next(
            line.removeprefix("Published NativeVNNI Git LFS corpus: ")
            for line in result.stdout.splitlines()
            if line.startswith("Published NativeVNNI Git LFS corpus: ")
        ))
        mkdir_line = next(
            line for line in result.stdout.splitlines()
            if line.startswith("dry-run: mkdir -p ")
            and "collect-cpu-prefill" in line
        )
        workspaces.append(mkdir_line)

    assert len(set(publications)) == 1
    assert len(set(workspaces)) == 1


def test_cpu_decode_checkpoint_and_grouped_continuation_share_workspace(
    tmp_path: Path,
) -> None:
    """The two documented CPU phases must address one staging transaction."""

    workspaces = []
    for phase_control in (
        "--stop-after-cpu-decode",
        "--resume-after-cpu-decode",
    ):
        result = subprocess.run(
            (
                str(PIPELINE),
                "--backend",
                "cpu",
                "--corpus-root",
                str(tmp_path / "corpora"),
                "--workspace-root",
                str(tmp_path / "work"),
                "--skip-build",
                "--skip-scorer-tests",
                "--install",
                "--dry-run",
                "--",
                "--cpu-format-shards",
                phase_control,
            ),
            cwd=REPOSITORY_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        workspaces.append(next(
            line for line in result.stdout.splitlines()
            if line.startswith("dry-run: mkdir -p ")
            and "collect-cpu-" in line
        ))

    assert workspaces[0] == workspaces[1]


def test_cpu_burned_seal_replay_reopens_the_measurement_workspace(
    tmp_path: Path,
) -> None:
    """Adding failed-seal evidence must not trigger full corpus recollection."""

    workspaces = []
    argument_sets = (
        ("--stop-after-cpu-decode",),
        (
            "--stop-after-cpu-decode",
            "--cpu-decode-max-leaves",
            "32",
        ),
        (
            "--stop-after-cpu-decode",
            "--cpu-decode-burned-sealed-plan",
            "/evidence/generation-000/plan.json",
            "--cpu-decode-burned-sealed-paired-dir",
            "/evidence/generation-000/paired",
        ),
    )
    for forwarded_arguments in argument_sets:
        result = subprocess.run(
            (
                str(PIPELINE),
                "--backend",
                "cpu",
                "--corpus-root",
                str(tmp_path / "corpora"),
                "--workspace-root",
                str(tmp_path / "work"),
                "--skip-build",
                "--skip-scorer-tests",
                "--install",
                "--dry-run",
                "--",
                *forwarded_arguments,
            ),
            cwd=REPOSITORY_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        workspaces.append(next(
            line for line in result.stdout.splitlines()
            if line.startswith("dry-run: mkdir -p ")
            and "collect-cpu-" in line
        ))

    assert len(set(workspaces)) == 1


def test_turnkey_checks_collection_completeness_before_sealing() -> None:
    """A bounded refresh checkpoint cannot fall through into publication."""

    source = PIPELINE.read_text(encoding="utf-8")
    refresh_offset = source.index('refresh_command "${staging_dir}"')
    complete_offset = source.index("corpus_bundle collection-complete")
    seal_offset = source.index("corpus_bundle seal", complete_offset)

    assert refresh_offset < complete_offset < seal_offset
    assert "cpu_batch_limit > 0 || stop_after_cpu_decode > 0" in source


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
