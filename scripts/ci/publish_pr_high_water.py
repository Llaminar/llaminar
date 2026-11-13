#!/usr/bin/env python3
## @file publish_pr_high_water.py
# @brief Commit a merged PR's certified benchmark marks to develop with skip-ci.
"""Commit a merged PR's certified benchmark marks to develop with skip-ci.

This runs only *after* the PR has merged and the master image release was
published. Writing to the still-open PR head would invalidate its exact-ref
required checks; writing after merge preserves the proof and gives the next
source change an upward-only baseline. The commit contains only benchmark
evidence and its README presentation, never rebuilt inference code.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from production_artifacts import digest, ratchet, write_json
import publish_master_release as release
import run_published_image_suite as suite

ROOT = Path(__file__).resolve().parents[2]
HIGH_WATER = "benchmarks/production/high_water.json"


def git(*arguments: str, env: dict | None = None, input_bytes: bytes | None = None) -> str:
    """Use the repository's existing authenticated origin without rewriting it."""
    return subprocess.check_output(["git", *arguments], cwd=ROOT, env=env,
                                   input=input_bytes).decode().strip()


def git_file(revision: str, path: str) -> bytes:
    """Read tracked bytes exactly, including significant trailing whitespace."""
    return subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT)


def proposed_payloads(evidence: dict, baseline: dict, develop_revision: str, develop_baseline: dict) -> dict[str, bytes]:
    """Recompute one cross-ISA ratchet from the original source baseline."""
    if any(report["baseline_digest"] != digest(baseline)
           for report in evidence["benchmarks"].values()):
        raise ValueError("PR benchmarks used a different source high-water baseline")
    rows = [row for isa in suite.ISAS for row in evidence["benchmarks"][isa]["cells"]]
    _, comparisons = ratchet(baseline, rows)
    if not all(comparison["passed"] for comparison in comparisons):
        raise ValueError("red PR benchmark cannot advance high water")
    if develop_baseline['regression_threshold_pct'] != baseline['regression_threshold_pct']:
        raise ValueError('develop changed benchmark policy while the RC was being certified')
    # Later develop marks may exceed this frozen RC's admitted baseline. Keep
    # those upward-only values; the source-bound PR gate already adjudicated
    # this release against its own policy. Never overwrite a concurrent gain.
    proposed, _ = ratchet(develop_baseline, rows)
    result = evidence["result"]
    readme = suite.readme_with_results(
        git_file(develop_revision, "README.md").decode(), result)
    return {
        HIGH_WATER: (json.dumps(proposed, indent=2, sort_keys=True) + "\n").encode(),
        str(suite.REPORT_DIRECTORY / "results.json"):
            (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
        str(suite.REPORT_DIRECTORY / "benchmarks.svg"):
            (evidence["benchmark_directory"] / "benchmarks.svg").read_bytes(),
        "README.md": readme.encode(),
    }


def commit_payloads(parent: str, master_sha: str,
                    payloads: dict[str, bytes], number: int, certified_source: str) -> str:
    """Join the squash-merged master into develop with one evidence commit.

    A squash merge has the same tree as the certified develop head but is not
    its ancestor. Recording both parents makes the next strict master PR
    up-to-date without rewriting either branch or rebuilding an untested image.
    """
    if git("rev-parse", f"{certified_source}^{{tree}}") != git("rev-parse", f"{master_sha}^{{tree}}"):
        raise ValueError("merged master and certified develop trees differ")
    if subprocess.run(['git', 'merge-base', '--is-ancestor', certified_source, parent], cwd=ROOT).returncode != 0:
        raise ValueError('current develop does not descend from the frozen RC source')
    with tempfile.TemporaryDirectory(prefix="llaminar-pr-benchmark-index-") as directory:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(directory) / "index"),
               "GIT_AUTHOR_NAME": "Llaminar benchmarks",
               "GIT_COMMITTER_NAME": "Llaminar benchmarks",
               "GIT_AUTHOR_EMAIL": "41898282+github-actions[bot]@users.noreply.github.com",
               "GIT_COMMITTER_EMAIL": "41898282+github-actions[bot]@users.noreply.github.com"}
        git("read-tree", parent, env=env)
        for path, content in payloads.items():
            blob = git("hash-object", "-w", "--stdin", input_bytes=content)
            git("update-index", "--add", "--cacheinfo", "100644", blob, path, env=env)
        tree = git("write-tree", env=env)
        # Even an unchanged mark needs this ancestry join after a squash
        # merge; otherwise the next PR is permanently behind strict master.
        return git("commit-tree", tree, "-p", parent, "-p", master_sha,
                   "-m", f"benchmarks: PR #{number} certified high water [skip ci]", env=env)


def publish(repository: str, master_sha: str, output: Path,
            push_url: str) -> dict:
    """Reauthenticate master proof and advance only the exact merged head.

    The push transport is explicit: production supplies the dedicated release
    deploy-key SSH URL; local bare-repository tests supply their own remote.
    """
    if git("rev-parse", "HEAD") != master_sha:
        raise ValueError("high-water checkout is not the released master commit")
    master_tree = git("rev-parse", f"{master_sha}^{{tree}}")
    pr = release.merged_rc_pr(repository, master_sha)
    release.fetch_rc_head(pr)
    head, branch, number = pr['head']['sha'], pr['head']['ref'], int(pr['number'])
    run = release.certified_pr_run(repository, number, head, branch)
    proof = release.download_proof(repository, int(run['id']), number, head, output)
    evidence = release.validate_proof(proof['preflight'], proof['benchmarks'], repository, head, master_tree,
        coding_directory=proof['opencode'], accuracy_directory=proof['accuracy'],
        source_directory=proof['rc-source'], rc_branch=branch)
    if json.loads((proof['release-eligibility'] / 'release-candidate.json').read_text()) != release.candidate_certificate(evidence):
        raise ValueError('release eligibility certificate changed before high-water publication')
    evidence['benchmark_directory'] = proof['benchmarks']
    source = evidence['pair']['source']['revision']
    git('fetch', 'origin', 'refs/heads/develop')
    current = git('rev-parse', 'FETCH_HEAD')
    baseline = json.loads(git_file(source, HIGH_WATER))
    latest = json.loads(git_file(current, HIGH_WATER))
    payloads = proposed_payloads(evidence, baseline, current, latest)
    joined = subprocess.run(['git', 'merge-base', '--is-ancestor', master_sha, current], cwd=ROOT).returncode == 0
    if joined and all(git_file(current, path) == content for path, content in payloads.items()):
        return dict(schema=1, pr=number, master_sha=master_sha, develop_sha=current, source_sha=source, reused=True)
    committed = commit_payloads(current, master_sha, payloads, number, source)
    git('push', push_url, f'{committed}:refs/heads/develop')
    return dict(schema=1, pr=number, master_sha=master_sha, develop_sha=committed, source_sha=source, reused=False)


def main(argv: list[str] | None = None) -> int:
    """Leave a compact receipt; a failed fast-forward never rewrites develop."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--master-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--push-url", required=True,
                        help="Dedicated release deploy-key SSH remote")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repository):
        parser.error("--repository requires OWNER/REPO")
    if not re.fullmatch(r"[0-9a-f]{40}", args.master_sha):
        parser.error("--master-sha requires a full Git SHA")
    expected_push_url = f"git@github.com:{args.repository}.git"
    if args.push_url != expected_push_url:
        parser.error("--push-url must be the exact repository SSH URL")
    args.output = args.output.resolve()
    if args.output.exists():
        raise ValueError("high-water output already exists; preserve evidence and choose a new path")
    args.output.mkdir(parents=True)
    receipt = publish(args.repository, args.master_sha, args.output, args.push_url)
    write_json(args.output / "high-water-publication.json", receipt)
    print(f"[pr-high-water] develop={receipt['develop_sha']} reused={receipt['reused']}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
        print(f"[pr-high-water] ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
