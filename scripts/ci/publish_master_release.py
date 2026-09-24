#!/usr/bin/env python3
"""Promote one PR-certified develop image pair after its merge to master.

The master merge may be a squash commit, so its commit ID need not equal the
image's develop source ID. Its *tree* must match exactly. E2E and benchmarks
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

ROOT = Path(__file__).resolve().parents[2]
DATE_TAG = re.compile(r"(?P<day>\d{4}-\d{2}-\d{2})\.(?P<sequence>[1-9]\d*)")


def git(*arguments: str) -> str:
    """Read the complete checked-out master history without changing it."""
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def merged_develop_pr(repository: str, master_sha: str) -> dict:
    """Find the one same-repository develop PR responsible for this master SHA."""
    candidates = github_json(f"repos/{repository}/commits/{master_sha}/pulls")
    matched = [pr for pr in candidates
               if pr.get("merged_at") and pr.get("merge_commit_sha") == master_sha
               and pr.get("head", {}).get("ref") == "develop"
               and pr.get("base", {}).get("ref") == "master"
               and pr.get("head", {}).get("repo", {}).get("full_name", "").lower() == repository.lower()]
    if len(matched) != 1:
        raise ValueError("master commit is not one merged same-repository develop PR")
    return matched[0]


def certified_pr_run(repository: str, number: int, develop_sha: str) -> dict:
    """Find the successful PR workflow whose run-name pins both PR and head."""
    title = f"master PR #{number} — {develop_sha}"
    for page in range(1, 21):
        response = github_json(
            f"repos/{repository}/actions/workflows/master-pr-certification.yml/runs",
            "event=pull_request", "branch=develop", "per_page=100", f"page={page}")
        runs = response.get("workflow_runs", [])
        matching = [run for run in runs if run.get("display_title") == title
                    and run.get("event") == "pull_request"
                    and run.get("head_branch") == "develop"
                    and run.get("status") == "completed"
                    and run.get("conclusion") == "success"]
        if matching:
            return matching[0]
        if len(runs) < 100:
            break
    raise ValueError("merged PR has no successful exact-head E2E/benchmark gate")


def download_proof(repository: str, run_id: int, number: int,
                   develop_sha: str, output: Path) -> tuple[Path, Path]:
    """Download only the two compact phase artifacts from the admitted run."""
    e2e = output / "e2e"
    benchmarks = output / "benchmarks"
    for lane, destination in (("e2e", e2e), ("benchmarks", benchmarks)):
        name = f"pr-{lane}-{number}-{develop_sha}"
        destination.mkdir()
        subprocess.run(["gh", "run", "download", str(run_id), "--repo", repository,
                        "--name", name, "--dir", str(destination)], check=True)
    return e2e, benchmarks


def validate_proof(e2e_directory: Path, benchmark_directory: Path,
                   repository: str, develop_sha: str, master_tree: str) -> dict:
    """Join both ISA E2E/benchmark reports to their exact image/source pair."""
    pair = json.loads((e2e_directory / "images.json").read_text())
    suite.validate_pair(pair)
    if (pair.get("repository", "").lower() != repository.lower()
            or pair.get("branch") != "develop"
            or pair["source"]["revision"] != develop_sha
            or pair["source"]["tree"] != master_tree
            or pair.get("workflow_revision") != develop_sha):
        raise ValueError("master tree or merged PR head differs from certified image source")
    manifest, e2e_reports = suite.admit_e2e(e2e_directory, pair)
    if json.loads((benchmark_directory / "images.json").read_text()) != pair:
        raise ValueError("benchmark pair differs from E2E pair")
    if json.loads((benchmark_directory / "manifest.json").read_text()) != manifest:
        raise ValueError("benchmark matrix differs from E2E matrix")
    result = json.loads((benchmark_directory / "results.json").read_text())
    baseline = json.loads((ROOT / "benchmarks/production/high_water.json").read_text())
    if (result.get("source") != pair["source"] or result.get("images") != pair["images"]
            or result.get("repository", "").lower() != repository.lower()
            or result.get("branch") != "develop"
            or result.get("scope") != "full-http-e2e-and-benchmarks"
            or result.get("passed") is not True
            or result.get("regression_threshold_pct") != baseline["regression_threshold_pct"]
            or result.get("full_image_certification") is not False
            or set(result.get("variants", {})) != set(suite.ISAS)):
        raise ValueError("compact benchmark result differs from proven image pair")
    reports = {}
    for isa in suite.ISAS:
        report = json.loads((benchmark_directory / isa.lower() / "benchmarks.json").read_text())
        suite.validate_benchmark(report, pair, manifest, e2e_reports[isa], isa)
        variant = result["variants"][isa]
        if (variant.get("report_digest") != digest(report)
                or variant.get("e2e_report_digest") != digest(e2e_reports[isa])
                or variant.get("comparisons") != report["comparisons"]
                or variant.get("cells") != [{key: value for key, value in row.items()
                                              if key != "artifacts"} for row in report["cells"]]):
            raise ValueError(f"{isa} compact benchmark numbers differ from complete report")
        reports[isa] = report
    return {"pair": pair, "manifest": manifest, "e2e": e2e_reports,
            "benchmarks": reports, "result": result}


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
        lines.append("Initial dated release of the develop build proven on both shipping CPU ISAs.")
    elif changes:
        lines.extend(f"- {subject}" for subject in changes)
    else:
        lines.append("No source changes since the preceding release.")
    lines.extend(["", "## Certification evidence", "",
                  f"The [master PR certification run]({run_url}) passed the full HTTP E2E and benchmark matrix.",
                  "The attached reports contain per-cell outcomes, throughput samples and exact image digests.", ""])
    for isa in suite.ISAS:
        count = len(evidence["e2e"][isa]["cells"])
        measured = len(evidence["benchmarks"][isa]["cells"])
        lines.append(f"- {isa}: {count}/{count} HTTP E2E cells; {measured}/{measured} benchmark cells. "
                     f"[E2E JSON]({base}/e2e-{isa.lower()}.json), "
                     f"[benchmark JSON]({base}/benchmark-{isa.lower()}.json).")
    lines.extend([f"- [Benchmark numbers]({base}/benchmark-results.json) and "
                  f"[chart]({base}/benchmarks.svg).", "", "## Images", ""])
    for isa in suite.ISAS:
        dated, pinned, mutable = master_image_tags(repository, master_sha, tag, isa)
        digest_ref = evidence["pair"]["images"][isa]["registry_ref"]
        lines.append(f"- {isa}: `{dated}` · `{pinned}` · `{mutable}` · source `{digest_ref}`")
    lines.extend(["", f"Master source tree: `{evidence['pair']['source']['tree']}`.", ""])
    return "\n".join(lines)


def release_assets(e2e: Path, benchmarks: Path, output: Path) -> list[Path]:
    """Copy only compact authenticated receipts, reports and numbers."""
    assets = output / "assets"
    assets.mkdir()
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
    for name, source in sources.items():
        shutil.copyfile(source, assets / name)
    return list(assets.iterdir())


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
    pr = merged_develop_pr(args.repository, args.master_sha)
    head_sha = pr["head"]["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", head_sha):
        raise ValueError("merged develop PR has no full head revision")
    run = certified_pr_run(args.repository, int(pr["number"]), head_sha)
    e2e, benchmarks = download_proof(args.repository, int(run["id"]),
                                     int(pr["number"]), head_sha, args.output)
    evidence = validate_proof(e2e, benchmarks, args.repository, head_sha, master_tree)
    releases = github_json(f"repos/{args.repository}/releases", "per_page=100")
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    tag, previous = release_tag(date, releases, args.master_sha)
    changes = commit_changes(previous, args.master_sha)
    notes = args.output / "release-notes.md"
    notes.write_text(release_notes(tag, previous, changes, evidence,
                                   args.repository, run["html_url"], args.master_sha))
    assets = release_assets(e2e, benchmarks, args.output)
    receipt = {"schema": 1, "master_sha": args.master_sha, "master_tree": master_tree,
               "develop_sha": head_sha, "pr_number": pr["number"], "pr_run": run["id"],
               "tag": tag, "previous_tag": previous, "image_pair_digest": digest(evidence["pair"]),
               "benchmark_result_digest": digest(evidence["result"]),
               "published": False}
    write_json(args.output / "master-release.json", receipt)
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
