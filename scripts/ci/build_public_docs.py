#!/usr/bin/env python3
"""Build public documentation from authored Markdown and published evidence.

GitHub releases own the archive. Each build reads every published dated release,
preserves its attached reports byte-for-byte, and projects their recorded results
into human-readable pages. The release publisher remains the certification
authority: this consumer checks receipt consistency, never reruns inference or
rejudges old measurements against today's high-water marks. Nothing is written
to the source tree, model cache, registry, or Git branches.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from html import escape
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import publish_master_release as release
from production_artifacts import digest, positive, write_json
from published_benchmark_chart import topology_label
import run_published_image_suite as suite
from wait_for_develop_image_gate import github_json

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PublishedEvidence:
    """One published release and its unchanged, mutually consistent reports."""

    metadata: dict
    assets: Path
    pair: dict
    manifest: dict
    e2e: dict
    benchmarks: dict
    result: dict

    @property
    def tag(self) -> str:
        """Return the validated path-safe dated release identity."""
        return self.metadata["tag_name"]


def published_releases(repository: str) -> tuple[list[dict], str | None]:
    """Page through the entire archive; GitHub's latest pointer owns the alias."""
    published = []
    page = 1
    while True:
        rows = github_json(f"repos/{repository}/releases", "per_page=100", f"page={page}")
        published.extend(row for row in rows
                         if not row["draft"] and not row["prerelease"]
                         and release.DATE_TAG.fullmatch(row["tag_name"]))
        if len(rows) < 100:
            break
        page += 1
    tags = [row["tag_name"] for row in published]
    if len(tags) != len(set(tags)):
        raise ValueError("duplicate published release tags")
    if not published:
        return [], None
    latest = github_json(f"repos/{repository}/releases/latest")["tag_name"]
    if latest not in tags:
        raise ValueError("latest release is not in the published dated archive")
    # Compare numeric suffixes, so the tenth release sorts after the second.
    published.sort(key=lambda row: (row["tag_name"].split(".")[0],
                                    int(row["tag_name"].split(".")[1])), reverse=True)
    return published, latest


def download_assets(repository: str, metadata: dict, destination: Path) -> None:
    """Download the publisher's exact named public assets, without wildcards."""
    names = release.release_asset_sources(Path("e2e"), Path("benchmarks"))
    attached = [asset["name"] for asset in metadata["assets"]]
    if any(attached.count(name) != 1 for name in names):
        raise ValueError(f"{metadata['tag_name']}: missing or duplicate certificate assets")
    destination.mkdir(parents=True)
    command = ["gh", "release", "download", metadata["tag_name"], "--repo", repository,
               "--dir", str(destination)]
    for name in names:
        command.extend(["--pattern", name])
    subprocess.run(command, check=True)


def read_json(path: Path) -> dict:
    """Read one required certificate object, rejecting non-object payloads."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path.name}")
    return value


def load_evidence(repository: str, metadata: dict, assets: Path) -> PublishedEvidence:
    """Check the archived receipt and complete image/matrix/report relationships.

    The release has already passed the publishing gate. Rechecking its original
    digests prevents a truncated or mixed download from being displayed as a
    certificate. Historical comparisons stay bound to their recorded baseline,
    not the current checkout's upward-only high-water file.
    """
    tag = metadata["tag_name"]
    if (not release.DATE_TAG.fullmatch(tag) or metadata.get("draft") is not False
            or metadata.get("prerelease") is not False or not metadata.get("published_at")):
        raise ValueError("documentation requires a published dated release")
    datetime.fromisoformat(metadata["published_at"].replace("Z", "+00:00"))
    pair = read_json(assets / "image-pair.json")
    suite.validate_pair(pair)
    if pair["repository"].lower() != repository.lower() or pair["branch"] != "develop":
        raise ValueError("release image pair belongs to a different repository or branch")
    manifest = read_json(assets / "matrix.json")
    cells = suite.validate_manifest(manifest, pair["source"]["revision"])
    expected = {row["case"]: row for row in cells}
    receipt = read_json(assets / "e2e-receipt.json")
    if (receipt.get("schema") != 1 or receipt.get("complete") is not True
            or receipt.get("images_digest") != digest(pair)
            or receipt.get("manifest_digest") != digest(manifest)
            or set(receipt.get("reports", {})) != set(suite.ISAS)):
        raise ValueError("published E2E receipt is incomplete or mismatched")
    result = read_json(assets / "benchmark-results.json")
    if (result.get("source") != pair["source"] or result.get("images") != pair["images"]
            or result.get("repository", "").lower() != repository.lower()
            or result.get("branch") != pair["branch"] or result.get("passed") is not True
            or result.get("scope") != "full-http-e2e-and-benchmarks"
            or result.get("full_image_certification") is not False
            or set(result.get("variants", {})) != set(suite.ISAS)):
        raise ValueError("published benchmark result has a different image pair or scope")
    e2e, benchmarks = {}, {}
    for isa in suite.ISAS:
        e2e[isa] = read_json(assets / f"e2e-{isa.lower()}.json")
        suite.validate_image_e2e(e2e[isa], manifest, pair["images"][isa]["id"])
        if digest(e2e[isa]) != receipt["reports"][isa]:
            raise ValueError(f"{isa}: E2E certificate differs from its receipt")
        report = read_json(assets / f"benchmark-{isa.lower()}.json")
        rows = report.get("cells", [])
        if (report.get("complete") is not True or report.get("passed") is not True
                or report.get("diagnostic") is not False
                or report.get("image") != pair["images"][isa]["id"]
                or report.get("source_revision") != pair["source"]["revision"]
                or report.get("manifest_digest") != digest(manifest)
                or report.get("e2e_report_digest") != digest(e2e[isa])
                or len(rows) != len(expected) or {row["case"] for row in rows} != set(expected)):
            raise ValueError(f"{isa}: incomplete or mismatched benchmark certificate")
        for row in rows:
            config = expected[row["case"]]["configuration"]
            if row["identity"].get("cpu_isa") != isa or row["identity"].get("configuration") != {
                    **config, "model": Path(config["model"]).name}:
                raise ValueError(f"{isa}: benchmark configuration differs from release matrix")
            for phase in ("prefill", "decode"):
                positive(row["tokens_per_second"][phase])
        variant = result["variants"][isa]
        if (variant.get("report_digest") != digest(report)
                or variant.get("e2e_report_digest") != digest(e2e[isa])
                or variant.get("comparisons") != report.get("comparisons")
                or variant.get("cells") != [{key: value for key, value in row.items()
                                             if key != "artifacts"} for row in rows]):
            raise ValueError(f"{isa}: compact benchmark differs from the attached certificate")
        benchmarks[isa] = report
    # Assets are copied unchanged. Check that the browser-facing SVG is an
    # actual chart before admitting the whole site for deployment.
    chart = ET.parse(assets / "benchmarks.svg").getroot()
    if chart.tag != "{http://www.w3.org/2000/svg}svg":
        raise ValueError("release chart is not SVG")
    return PublishedEvidence(metadata, assets, pair, manifest, e2e, benchmarks, result)


def table_text(value: object) -> str:
    """Escape report labels for Markdown tables without interpreting markup."""
    return escape(str(value), quote=False).replace("|", "&#124;").replace("\n", " ")


def release_navigation(prefix: str) -> list[dict[str, str]]:
    """Keep dated and latest views on the same four public pages."""
    return [{name: f"{prefix}/{file}.md"} for name, file in (
        ("Overview", "index"), ("Release notes", "release-notes"),
        ("E2E certificate", "e2e"), ("Benchmark certificate", "benchmarks"))]


def render_release(evidence: PublishedEvidence, destination: Path) -> None:
    """Project all cells and both ISAs; preserve original report downloads."""
    destination.mkdir(parents=True)
    shutil.copytree(evidence.assets, destination / "assets")
    tag = evidence.tag
    repository = evidence.pair["repository"]
    source = evidence.pair["source"]["revision"]
    count = len(evidence.manifest["cells"])
    base = f"https://github.com/{repository}"
    summary = [f"# Llaminar {tag}", "",
        f"Published {evidence.metadata['published_at'][:10]}. "
        f"[GitHub release]({base}/releases/tag/{tag}).", "",
        "## Certified results", "",
        f"All **{count} configurations** passed the full HTTP end-to-end suite and "
        "benchmarks on **both AVX512 and AVX2** images.", "",
        "- [Release notes](release-notes.md)",
        "- [E2E test certificate](e2e.md)",
        "- [Benchmark certificate and chart](benchmarks.md)", "",
        "The scope of these certificates is the published HTTP E2E and benchmark "
        "matrix. The pages display the original reports; building this site does "
        "not run tests or issue new certification.", "",
        "## Exact images", "", f"Tested source: [{source[:12]}]({base}/commit/{source}).", "",
        "| CPU build | Immutable image |", "|---|---|"]
    for isa in suite.ISAS:
        summary.append(f"| {isa} | `{evidence.pair['images'][isa]['registry_ref']}` |")
    summary += ["", "[Image identities](assets/image-pair.json) · "
                "[Complete test matrix](assets/matrix.json)", ""]
    (destination / "index.md").write_text("\n".join(summary))
    # Escape raw HTML in commit-derived release notes. Markdown headings and
    # ordinary links are retained; source text cannot insert HTML elements.
    notes = escape(evidence.metadata.get("body") or "No release notes were supplied.", quote=False)
    (destination / "release-notes.md").write_text(notes + "\n")
    e2e_lines = [f"# E2E certificate · {tag}", "",
        "Every configuration below passed the published HTTP server suite on both "
        "CPU builds. Download the original certificates for individual checks and diagnostics.", "",
        "[AVX512 certificate](assets/e2e-avx512.json) · "
        "[AVX2 certificate](assets/e2e-avx2.json) · "
        "[Pair receipt](assets/e2e-receipt.json)", "",
        "| Model | Configuration | AVX512 | AVX2 |", "|---|---|---|---|"]
    benchmark_lines = [f"# Benchmark certificate · {tag}", "",
        "Both images passed their complete HTTP E2E suites before benchmarking. "
        "Prefill measures reading the prompt; decode measures generating the answer. "
        "Rates are tokens per second, with higher values indicating faster inference.", "",
        "![Certified prefill and decode results](assets/benchmarks.svg)", "",
        "Each chart row uses its own AVX512 result as the reference. Compare "
        "printed token rates across configurations, rather than bar lengths.", "",
        "[AVX512 certificate](assets/benchmark-avx512.json) · "
        "[AVX2 certificate](assets/benchmark-avx2.json) · "
        "[Combined results](assets/benchmark-results.json) · [SVG chart](assets/benchmarks.svg)", "",
        "| Model / configuration | AVX512 prefill | AVX512 decode | AVX2 prefill | AVX2 decode |",
        "|---|---:|---:|---:|---:|"]
    measured = {isa: {row["case"]: row for row in evidence.benchmarks[isa]["cells"]}
                for isa in suite.ISAS}
    for cell in evidence.manifest["cells"]:
        config = cell["configuration"]
        model = table_text(Path(config["model"]).name.removesuffix(".gguf"))
        topology = table_text(topology_label(config))
        e2e_lines.append(f"| {model} | {topology} | Passed | Passed |")
        rates = [f"{measured[isa][cell['case']]['tokens_per_second'][phase]:,.1f}"
                 for isa in suite.ISAS for phase in ("prefill", "decode")]
        benchmark_lines.append(f"| {model} / {topology} | " + " | ".join(rates) + " |")
    (destination / "e2e.md").write_text("\n".join(e2e_lines) + "\n")
    (destination / "benchmarks.md").write_text("\n".join(benchmark_lines) + "\n")


def prepare_docs(repository: str, output: Path, source: Path = ROOT / "docs/public") -> list[dict]:
    """Stage authored pages plus the full archive, then select the latest view."""
    if (source / "releases").exists():
        raise ValueError("docs/public/releases is generated; keep authored pages outside it")
    content = output / "docs"
    shutil.copytree(source, content)
    metadata, latest = published_releases(repository)
    archive = content / "releases"
    archive.mkdir()
    listing = ["# Releases", "", "Each dated release retains its original notes, "
        "HTTP E2E certificates, benchmark certificates, and image identities.", ""]
    navigation = [{"Archive": "releases/index.md"}]
    if latest:
        listing += [f"[Latest release: {latest}](latest/index.md)", "",
                    "| Release | Published | E2E configurations per CPU build |",
                    "|---|---|---:|"]
        navigation.insert(0, {"Latest": release_navigation("releases/latest")})
    else:
        listing.append("No dated releases have been published yet.")
    for record in metadata:
        tag = record["tag_name"]
        assets = output / "downloads" / tag
        download_assets(repository, record, assets)
        evidence = load_evidence(repository, record, assets)
        render_release(evidence, archive / tag)
        navigation.append({tag: release_navigation(f"releases/{tag}")})
        listing.append(f"| [{tag}]({tag}/index.md) | {record['published_at'][:10]} | "
                       f"{len(evidence.manifest['cells'])} |")
        if tag == latest:
            # Copy only the already-admitted release. An unrelated benchmark
            # publication can never advance the stable README image URL.
            shutil.copytree(archive / tag, archive / "latest")
        print(f"[public-docs] {tag}: both ISA certificates admitted", flush=True)
    (archive / "index.md").write_text("\n".join(listing) + "\n")
    write_json(output / "publication.json", {
        "schema": 1, "repository": repository, "latest": latest,
        "releases": [{"tag": row["tag_name"], "id": row["id"],
                      "published_at": row["published_at"]} for row in metadata]})
    return navigation


def main(argv: list[str] | None = None) -> int:
    """Build a fresh Pages artifact locally or in the lightweight docs job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default="Llaminar/llaminar")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--site-url", default="https://llaminar.github.io/llaminar/")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repository):
        parser.error("--repository requires OWNER/REPO")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    nav = prepare_docs(args.repository, output)
    # Import site tooling only during an actual build. The canonical model-free
    # tests need no MkDocs install, credentials, network, or release downloads.
    from mkdocs.config import load_config
    from mkdocs.commands.build import build
    config = load_config(str(ROOT / "mkdocs.public.yml"),
                         docs_dir=str(output / "docs"), site_dir=str(output / "site"),
                         site_url=args.site_url.rstrip("/") + "/", strict=True)
    config["nav"].append({"Releases": nav})
    build(config)
    print(f"[public-docs] site: {output / 'site'}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError, ET.ParseError, subprocess.SubprocessError) as error:
        print(f"[public-docs] ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
