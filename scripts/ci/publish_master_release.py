#!/usr/bin/env python3
## @file publish_master_release.py
# @brief Promote one PR-certified RC image pair after its merge to master.
"""Promote one PR-certified RC image pair after its merge to master.

The master merge may be a squash commit, so its commit ID need not equal the
image's admitted develop source ID. Its *tree* must match exactly. Preflight, numerical panels, coding and benchmarks
are revalidated from the successful PR run's compact artifacts before any
image alias or GitHub release is written. Mutable master tags move last;
date and full-master-SHA tags remain stable evidence of what was released.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

from production_artifacts import digest, write_json
import run_published_image_suite as suite
from run_develop_image_gate import remote_image_id
from wait_for_develop_image_gate import github_json
from opencode_evidence import validate_matrix as validate_coding_matrix
import release_candidate
from release_accuracy_evidence import admit_accuracy

ROOT = Path(__file__).resolve().parents[2]
DATE_TAG = re.compile(r"(?P<day>\d{4}-\d{2}-\d{2})\.(?P<sequence>[1-9]\d*)")


def git(*arguments: str) -> str:
    """Read the complete checked-out master history without changing it."""
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def merged_rc_pr(repository: str, master_sha: str) -> dict:
    """Find the merged RC PR by its exact master commit identity.

    GitHub's commit-to-PR index may omit a newly created squash commit even
    after the PR itself publishes ``merge_commit_sha``. The closed-PR listing
    is a second view of the same typed merge identity, not a weaker match on
    commit message text or branch names alone.
    """
    def matches(pr: dict) -> bool:
        return (bool(pr.get("merged_at"))
                and pr.get("merge_commit_sha") == master_sha
                and release_candidate.is_rc(pr.get("head", {}).get("ref", ""))
                and pr.get("base", {}).get("ref") == "master"
                and pr.get("head", {}).get("repo", {}).get("full_name", "").lower()
                == repository.lower())

    matched = [pr for pr in github_json(f"repos/{repository}/commits/{master_sha}/pulls")
               if matches(pr)]
    if not matched:
        # Only the exact merge SHA may admit a listing candidate. Stop once
        # the recent closed-PR inventory ends; never guess from a squash title.
        for page in range(1, 21):
            candidates = github_json(f"repos/{repository}/pulls", "state=closed",
                                     "base=master", "sort=updated", "direction=desc",
                                     "per_page=100", f"page={page}")
            matched.extend(pr for pr in candidates if matches(pr))
            if matched or len(candidates) < 100:
                break
    if len(matched) != 1:
        raise ValueError("master commit is not one merged same-repository rc/* PR")
    return matched[0]


def certified_pr_run(repository: str, number: int, rc_sha: str, rc_branch: str) -> dict:
    """Find the successful PR workflow whose run-name pins both PR and head."""
    title = f"master PR #{number} — {rc_sha}"
    for page in range(1, 21):
        response = github_json(
            f"repos/{repository}/actions/workflows/master-pr-certification.yml/runs",
            "event=pull_request", f"branch={rc_branch}", "per_page=100", f"page={page}")
        runs = response.get("workflow_runs", [])
        matching = [run for run in runs if run.get("display_title") == title
                    and run.get("event") == "pull_request"
                    and run.get("head_branch") == rc_branch
                    and run.get("status") == "completed"
                    and run.get("conclusion") == "success"]
        if matching:
            return matching[0]
        if len(runs) < 100:
            break
    raise ValueError("merged PR has no successful exact-RC preflight/accuracy/coding/benchmark gate")


def fetch_rc_head(pr: dict) -> None:
    """Fetch the immutable PR head even after GitHub auto-deletes its RC branch.

    Squash merges do not retain the head commit in master ancestry. GitHub's
    read-only pull ref preserves the original source; the exact PR metadata SHA
    must still match before any tree or lineage validation uses that object.
    """
    number = int(pr['number'])
    if number <= 0:
        raise ValueError('invalid merged RC pull-request identity')
    subprocess.run(['git', 'fetch', '--no-tags', 'origin', f'refs/pull/{number}/head'], cwd=ROOT, check=True)
    if git('rev-parse', 'FETCH_HEAD') != pr['head']['sha']:
        raise ValueError('saved RC pull ref differs from the merged PR head')


def download_proof(repository: str, run_id: int, number: int,
                   rc_sha: str, output: Path) -> dict[str, Path]:
    """Download all independently authenticated phases from one exact RC run."""
    result = {}
    for lane in ('rc-source', 'preflight', 'accuracy', 'opencode', 'benchmarks', 'release-eligibility'):
        destination = output / lane
        destination.mkdir()
        subprocess.run(['gh', 'run', 'download', str(run_id), '--repo', repository,
                        '--name', f'pr-{lane}-{number}-{rc_sha}', '--dir', str(destination)], check=True)
        result[lane] = destination
    return result


def validate_proof(preflight_directory: Path, benchmark_directory: Path,
                   repository: str, rc_sha: str, master_tree: str, *, coding_directory: Path,
                   accuracy_directory: Path, source_directory: Path, rc_branch: str) -> dict:
    """Replay frozen lineage and the preflight → accuracy → stress → benchmark chain."""
    source = json.loads((source_directory / 'source.json').read_text())
    develop_sha = release_candidate.validate(source, repository, rc_branch, rc_sha, master_tree)
    pair = json.loads((preflight_directory / 'images.json').read_text())
    suite.validate_pair(pair)
    if (pair.get('repository', '').lower() != repository.lower() or pair.get('branch') != 'develop'
            or pair['source']['revision'] != develop_sha or pair['source']['tree'] != master_tree
            or pair.get('workflow_revision') != develop_sha):
        raise ValueError('master tree or frozen RC source differs from certified image source')
    manifest, prerequisites = suite.admit_preflight(preflight_directory, pair)
    accuracy = admit_accuracy(accuracy_directory, preflight_directory, pair, manifest, prerequisites)
    coding = validate_coding_matrix(json.loads((coding_directory / 'opencode.json').read_text()),
                                    pair, manifest, prerequisites, accuracy)
    result, reports = suite.admit_benchmarks(benchmark_directory, pair, manifest, prerequisites, release_preflight=True)
    expected = {'accuracy_reports': {isa: digest(accuracy[isa]) for isa in suite.ISAS},
                'opencode_digest': digest(coding)}
    if result.get('release_predecessors') != expected:
        raise ValueError('release benchmarks did not consume the completed accuracy and stress proof')
    return {'rc_source': source, 'pair': pair, 'manifest': manifest, 'prerequisites': prerequisites,
            'preflight_receipt': json.loads((preflight_directory / 'preflight-receipt.json').read_text()),
            'accuracy': accuracy, 'benchmarks': reports, 'result': result, 'coding': coding,
            'accuracy_receipt': json.loads((accuracy_directory / 'accuracy-receipt.json').read_text())}


def candidate_certificate(evidence: dict) -> dict:
    """Bind all release phases to the same immutable RC and runtime image pair."""
    return dict(schema='llaminar.release-candidate.v1', complete=True, passed=True,
                rc_source=evidence['rc_source'], images_digest=digest(evidence['pair']),
                manifest_digest=digest(evidence['manifest']), preflight_receipt_digest=digest(evidence['preflight_receipt']),
                accuracy_receipt_digest=digest(evidence['accuracy_receipt']),
                prerequisites_reports={isa: digest(evidence['prerequisites'][isa]) for isa in suite.ISAS},
                accuracy_reports={isa: digest(evidence['accuracy'][isa]) for isa in suite.ISAS},
                opencode_digest=digest(evidence['coding']), benchmark_result_digest=digest(evidence['result']))


def release_tag(day: str, releases: list[dict], master_sha: str) -> tuple[str, str | None]:
    """Choose the next date sequence; reuse only this commit's own draft."""
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
        raise ValueError("release day must use UTC YYYY-MM-DD")
    dated = [release for release in releases if DATE_TAG.fullmatch(release.get("tag_name", ""))]
    for release in dated:
        if release.get("target_commitish") == master_sha:
            if not release.get("draft"):
                raise ValueError("this master commit already has a published release")
            return release["tag_name"], previous_release_tag(dated, release["tag_name"])
    sequence = max((int(match.group("sequence")) for release in dated
                    if (match := DATE_TAG.fullmatch(release["tag_name"]))
                    and match.group("day") == day), default=0) + 1
    return f"{day}.{sequence}", previous_release_tag(dated)


def previous_release_tag(releases: list[dict], exclude: str | None = None) -> str | None:
    """Use the latest published dated release, never an abandoned draft."""
    published = [release for release in releases
                 if not release.get("draft") and release.get("tag_name") != exclude]
    if not published:
        return None
    return max(published, key=lambda release: release.get("published_at") or "")["tag_name"]


def master_image_tags(repository: str, master_sha: str, date_tag: str,
                      isa: str) -> tuple[str, str, str]:
    """Return dated, full-master-SHA and mutable branch tags for one ISA."""
    if isa not in suite.ISAS or not re.fullmatch(r"[0-9a-f]{40}", master_sha):
        raise ValueError("invalid master image identity")
    image = f"ghcr.io/{repository.lower()}"
    suffix = "-avx2" if isa == "AVX2" else ""
    return (f"{image}:{date_tag}{suffix}",
            f"{image}:master{suffix}-{master_sha}",
            f"{image}:master{suffix}")


def commit_changes(previous_tag: str | None, master_sha: str) -> list[str]:
    """List every post-release commit subject; omit the unbounded first history."""
    if previous_tag is None:
        return []
    subprocess.run(["git", "merge-base", "--is-ancestor", previous_tag, master_sha],
                   cwd=ROOT, check=True)
    subjects = git("log", "--format=%h %s", f"{previous_tag}..{master_sha}")
    return subjects.splitlines() if subjects else []


def release_notes(tag: str, previous_tag: str | None, changes: list[str],
                  evidence: dict, repository: str, run_url: str, master_sha: str) -> str:
    """Keep the bootstrap release brief; later notes cover the exact Git range."""
    base = f"https://github.com/{repository}/releases/download/{tag}"
    lines = [f"# Llaminar {tag}", "", "## Changes", ""]
    if previous_tag is None:
        lines.append("Initial dated release of the RC snapshot proven on both shipping CPU ISAs.")
    elif changes:
        lines.extend(f"- {subject}" for subject in changes)
    else:
        lines.append("No source changes since the preceding release.")
    lines.extend(["", "## Certification evidence", "",
                  f"The [master PR certification run]({run_url}) passed complete production preflight, frozen-history PyTorch panels, OpenCode stress and benchmark gates.",
                  "The attached reports contain per-cell outcomes, throughput samples and exact image digests.", ""])
    for isa in suite.ISAS:
        measured = len(evidence["benchmarks"][isa]["cells"])
        lines.append(f"- {isa}: complete Unit/production preflight; {measured}/{measured} benchmark cells. "
                     f"[Preflight JSON]({base}/production-preflight-{isa.lower()}.json), "
                     f"[benchmark JSON]({base}/benchmark-{isa.lower()}.json).")
    for isa in suite.ISAS:
        lines.append(f"- {isa} numerical accuracy: [panel rows]({base}/accuracy-rows-{isa.lower()}.csv), "
                     f"[prompt-balanced prefill/decode summaries]({base}/accuracy-summary-{isa.lower()}.csv), "
                     f"[complete accuracy evidence]({base}/accuracy-evidence-{isa.lower()}.zip).")
    lines.append(f"- [Test run]({base}/test-run.json), [release eligibility]({base}/release-candidate.json), "
                 f"[frozen RC source]({base}/rc-source.json).")
    lines.extend([f"- [Benchmark numbers]({base}/benchmark-results.json) and "
                  f"[chart]({base}/benchmarks.svg).",
                  f"- {evidence['coding']['expected_jobs']} complete OpenCode app sessions: "
                  f"[coding proof]({base}/opencode.json). Model-authored app quality is reported separately "
                  "from engine/protocol correctness; tool errors must remain below 5%.",
                  "", "## Images", ""])
    for isa in suite.ISAS:
        dated, pinned, mutable = master_image_tags(repository, master_sha, tag, isa)
        digest_ref = evidence["pair"]["images"][isa]["registry_ref"]
        lines.append(f"- {isa}: `{dated}` · `{pinned}` · `{mutable}` · source `{digest_ref}`")
    lines.extend(["", f"Master source tree: `{evidence['pair']['source']['tree']}`.", ""])
    return "\n".join(lines)


def release_asset_sources(e2e: Path, benchmarks: Path, *, model_kld: bool = False) -> dict[str, Path]:
    """Name the public evidence once for release upload and documentation."""
    sources = {
        "image-pair.json": e2e / "images.json",
        "matrix.json": e2e / "manifest.json",
        "e2e-receipt.json": e2e / "e2e-receipt.json",
        "benchmark-results.json": benchmarks / "results.json",
        "benchmarks.svg": benchmarks / "benchmarks.svg",
    }
    for isa in suite.ISAS:
        sources[f"e2e-{isa.lower()}.json"] = e2e / isa.lower() / "e2e.json"
        sources[f"benchmark-{isa.lower()}.json"] = benchmarks / isa.lower() / "benchmarks.json"
        # Historical display-only bundles need no invented numerical evidence.
        # New releases require this pair through validate_proof before copying.
        if model_kld:
            sources[f"kld-{isa.lower()}.json"] = e2e / isa.lower() / 'kld/kld-canaries.json'
    return sources


def release_candidate_asset_names() -> list[str]:
    """Own the compact RC publication/download contract, including numerical CSVs."""
    names = ['image-pair.json','matrix.json','all-cells.json','preflight-receipt.json',
             'benchmark-results.json','benchmarks.svg','opencode.json','rc-source.json',
             'release-candidate.json','accuracy-receipt.json','test-run.json']
    for isa in suite.ISAS:
        suffix=isa.lower()
        names.extend(f'{prefix}-{suffix}.{extension}' for prefix,extension in (
            ('benchmark','json'),('production-preflight','json'),('accuracy-retirement','json'),
            ('accuracy-rows','csv'),('accuracy-summary','csv'),('accuracy','json'),('accuracy-evidence','zip')))
    return sorted(names)


def release_assets(preflight: Path, benchmarks: Path, coding: Path, accuracy: Path,
                   source: Path, certificate: Path, output: Path) -> list[Path]:
    """Publish panel CSVs, test runs and complete compact proof with every release."""
    import zipfile
    assets = output / 'assets'
    assets.mkdir()
    sources = {'image-pair.json': preflight / 'images.json', 'matrix.json': preflight / 'manifest.json',
               'all-cells.json': preflight / 'all-cells.json',
               'preflight-receipt.json': preflight / 'preflight-receipt.json',
               'benchmark-results.json': benchmarks / 'results.json', 'benchmarks.svg': benchmarks / 'benchmarks.svg'}
    for isa in suite.ISAS:
        sources[f'benchmark-{isa.lower()}.json'] = benchmarks / isa.lower() / 'benchmarks.json'
    sources.update({'opencode.json': coding / 'opencode.json', 'rc-source.json': source / 'source.json',
                    'release-candidate.json': certificate / 'release-candidate.json',
                    'accuracy-receipt.json': accuracy / 'accuracy-receipt.json'})
    for isa in suite.ISAS:
        sources[f'production-preflight-{isa.lower()}.json'] = preflight / 'prerequisites' / isa.lower() / 'prerequisites/prerequisites.json'
        sources[f'accuracy-retirement-{isa.lower()}.json'] = accuracy / isa.lower() / 'outer-cleanup.json'
        panel = accuracy / isa.lower() / 'panel'
        for name in ('accuracy-rows', 'accuracy-summary'):
            sources[f'{name}-{isa.lower()}.csv'] = panel / (name + '.csv')
        sources[f'accuracy-{isa.lower()}.json'] = panel / 'panel.json'
        with zipfile.ZipFile(assets / f'accuracy-evidence-{isa.lower()}.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(panel.rglob('*')):
                if path.is_file() and not path.is_symlink() and path.suffix in ('.csv', '.json', '.log', '.txt'):
                    archive.write(path, path.relative_to(panel))
    for name, path in sources.items():
        shutil.copyfile(path, assets / name)
    return sorted(assets.iterdir())



def promote_tag(alias: str, image: dict, *, immutable: bool) -> None:
    """Alias exact registry bytes and reject a conflicting immutable ref."""
    existing = remote_image_id(alias)
    if existing == image["id"]:
        return
    if immutable and existing is not None:
        raise ValueError(f"immutable release image tag already has different bytes: {alias}")
    subprocess.run(["docker", "buildx", "imagetools", "create",
                    "--prefer-index=false", "--tag", alias,
                    image["registry_ref"]], check=True)
    if remote_image_id(alias) != image["id"]:
        raise ValueError(f"registry tag did not resolve to certified image bytes: {alias}")


def publish(repository: str, master_sha: str, tag: str, notes: Path,
            assets: list[Path], evidence: dict, draft_exists: bool) -> None:
    """Create a draft, promote immutable refs, attach proof, then move master."""
    if draft_exists:
        subprocess.run(["gh", "release", "edit", tag, "--repo", repository,
                        "--notes-file", str(notes)], check=True)
    else:
        subprocess.run(["gh", "release", "create", tag, "--repo", repository,
                        "--target", master_sha, "--title", f"Llaminar {tag}",
                        "--notes-file", str(notes), "--draft"], check=True)
    for isa in suite.ISAS:
        image = evidence["pair"]["images"][isa]
        dated, pinned, _ = master_image_tags(repository, master_sha, tag, isa)
        promote_tag(dated, image, immutable=True)
        promote_tag(pinned, image, immutable=True)
    subprocess.run(["gh", "release", "upload", tag, "--repo", repository,
                    "--clobber", *(str(asset) for asset in assets)], check=True)
    for isa in suite.ISAS:
        image = evidence["pair"]["images"][isa]
        _, _, mutable = master_image_tags(repository, master_sha, tag, isa)
        promote_tag(mutable, image, immutable=False)
    subprocess.run(["gh", "release", "edit", tag, "--repo", repository,
                    "--draft=false", "--latest"], check=True)


def main(argv: list[str] | None = None) -> int:
    """Resolve the merged PR, authenticate its proof and publish one release."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--master-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true",
                        help="validate and render without writing registry or GitHub release state")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repository):
        parser.error("--repository requires OWNER/REPO")
    if not re.fullmatch(r"[0-9a-f]{40}", args.master_sha):
        parser.error("--master-sha requires a full Git SHA")
    args.output = args.output.resolve()
    if args.output.exists():
        raise ValueError("release output already exists; preserve evidence and choose a new path")
    args.output.mkdir(parents=True)
    if git("rev-parse", "HEAD") != args.master_sha:
        raise ValueError("release checkout is not the pushed master commit")
    master_tree = git("rev-parse", f"{args.master_sha}^{{tree}}")
    pr = merged_rc_pr(args.repository, args.master_sha)
    fetch_rc_head(pr)
    head_sha = pr["head"]["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", head_sha):
        raise ValueError("merged RC PR has no full head revision")
    run = certified_pr_run(args.repository, int(pr['number']), head_sha, pr['head']['ref'])
    proof = download_proof(args.repository, int(run['id']), int(pr['number']), head_sha, args.output)
    evidence = validate_proof(proof['preflight'], proof['benchmarks'], args.repository, head_sha, master_tree,
                              coding_directory=proof['opencode'], accuracy_directory=proof['accuracy'],
                              source_directory=proof['rc-source'], rc_branch=pr['head']['ref'])
    certificate = json.loads((proof['release-eligibility'] / 'release-candidate.json').read_text())
    if certificate != candidate_certificate(evidence):
        raise ValueError('merged RC differs from its successful release eligibility certificate')
    releases = github_json(f"repos/{args.repository}/releases", "per_page=100")
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    tag, previous = release_tag(date, releases, args.master_sha)
    changes = commit_changes(previous, args.master_sha)
    notes = args.output / "release-notes.md"
    notes.write_text(release_notes(tag, previous, changes, evidence,
                                   args.repository, run["html_url"], args.master_sha))
    assets = release_assets(proof['preflight'], proof['benchmarks'], proof['opencode'], proof['accuracy'],
                            proof['rc-source'], proof['release-eligibility'], args.output)
    receipt = {"schema": 1, "master_sha": args.master_sha, "master_tree": master_tree,
               "source_sha": evidence['pair']['source']['revision'], "rc_sha": head_sha,
               "rc_branch": pr['head']['ref'], "pr_number": pr["number"], "pr_run": run["id"],
               "tag": tag, "previous_tag": previous, "image_pair_digest": digest(evidence["pair"]),
               "benchmark_result_digest": digest(evidence["result"]),
               "coding_result_digest": digest(evidence['coding']),
               "accuracy_reports": certificate['accuracy_reports'],
               "certification_run_url": run['html_url'],
               "published": False}
    write_json(args.output / "master-release.json", receipt)
    test_run = args.output / 'assets/test-run.json'
    write_json(test_run, {k: v for k, v in receipt.items() if k != 'published'})
    assets.append(test_run)
    if {path.name for path in assets} != set(release_candidate_asset_names()):
        raise ValueError("release asset inventory differs from the RC publication contract")
    if not args.plan_only:
        draft_exists = any(release.get("tag_name") == tag and release.get("draft")
                           for release in releases)
        publish(args.repository, args.master_sha, tag, notes, assets, evidence, draft_exists)
        receipt["published"] = True
        write_json(args.output / "master-release.json", receipt)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
        print(f"[master-release] ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
