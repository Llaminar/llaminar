#!/usr/bin/env python3
"""Render exact published-image measurements as a dependency-free README SVG.

No values are synthesized or fitted. Each cell/phase uses its own AVX512 median
as the 100% reference, and both ISA lanes share that local linear scale. Model
size sections make the matrix readable without comparing unrelated bar lengths.
The JSON retains full policy and samples; display labels never select tests.
"""
from __future__ import annotations

from collections import Counter
from html import escape
from itertools import groupby
from pathlib import Path
import re
import textwrap

from production_artifacts import positive


def model_section(configuration: dict) -> tuple[float, str]:
    """Group display rows by the total parameter count in the declared GGUF name.

    Match a standalone size token, not an MoE active-count token such as A3B or
    the shard number. Unknown names remain visible in a final section. This is
    presentation only: neither the canonical cell identity nor policy changes.
    """
    model = Path(configuration["model"]).name
    size = re.search(r"(?:^|[-_])(\d+(?:\.\d+)?)B(?=$|[-_.])", model, re.IGNORECASE)
    if size:
        count = float(size[1])
        return count, f"{count:g}B models"
    return float("inf"), "Other models"


def topology_label(configuration: dict) -> str:
    """Summarize exported participant addresses, never select a test topology."""
    planning = configuration["e2e"].get("planning")
    if planning is not None:
        counts = planning.get("device_counts", {})
        strategies = {"single": "single", "tp": "TP", "pp": "PP", "expert-overlay": "ExpertOverlay"}
        if (planning.get("mode") != "auto" or planning.get("strategy") not in strategies
                or not counts or not set(counts) <= {"cpu", "cuda", "rocm"}
                or any(type(count) is not int or count <= 0 for count in counts.values())):
            raise ValueError("canonical automatic chart metadata is incomplete")
        # Cardinalities are constraints, not authored stage/tier order. A plus
        # sign avoids claiming that the first vendor owns the continuation.
        labels = {"cpu": "CPU", "cuda": "CUDA", "rocm": "ROCm"}
        return " + ".join(f"{count}×{labels[kind]}" for kind, count in sorted(counts.items())) + \
            f" · auto {strategies[planning['strategy']]}"
    argv = configuration["e2e"]["server_args"]
    values = {}
    for index, argument in enumerate(argv[:-1]):
        if argument.startswith("--"):
            values.setdefault(argument, []).append(argv[index + 1])

    def participants(text: str) -> str:
        counts = Counter(match.upper() for match in re.findall(r"(?:^|:|,)(cpu|cuda|rocm):\d+", text))
        return " + ".join(f"{count}×{'ROCm' if kind == 'ROCM' else kind}"
                          for kind, count in counts.items())

    domains = {}
    for value in values.get("--moe-routed-expert-domain", []) + values.get("--expert-tier", []):
        name, addresses = value.split(";", 1)[0].split("=", 1)
        domains[name] = participants(addresses)
    if domains:
        # Exported tier priorities give direction to the continuation/capacity
        # layout. The display does not assign hot/cold roles from vendor names.
        tiers = []
        for value in values.get("--moe-routed-expert-tier", []):
            name = value.split(";", 1)[0].split("@", 1)[1]
            priority = re.search(r"(?:^|;)priority=(\d+)(?:;|$)", value)
            if name not in domains or priority is None:
                raise ValueError("canonical tier omitted its domain or priority")
            tiers.append((int(priority[1]), domains[name]))
        ordered = [label for _, label in sorted(tiers)] if tiers else list(domains.values())
        return " → ".join(ordered) + " · ExpertOverlay"
    for flag in ("--tp-devices", "--device-map", "--device", "-d"):
        if flag in values:
            label = participants(",".join(values[flag]))
            if label:
                return label + (" · TP" if flag == "--tp-devices" else "")
    if "--define-domain" in values:
        return " → ".join(participants(value.split("=", 1)[1].split(";", 1)[0])
                          for value in values["--define-domain"]) + " · PP"
    # Auto cells declare constraints, not concrete endpoints in their argv.
    # Retain the complete canonical identity instead of guessing placement.
    return configuration["id"]


def render_chart(result: dict) -> str:
    """Emit an accessible standalone SVG with escaped metadata and exact rates."""
    variants = result["variants"]
    if set(variants) != {"AVX512", "AVX2"}:
        raise ValueError("benchmark chart requires both shipping ISAs")
    lanes = {isa: {row["case"]: row for row in variant["cells"]}
             for isa, variant in variants.items()}
    if (not lanes["AVX512"] or set(lanes["AVX512"]) != set(lanes["AVX2"])
            or any(len(lanes[isa]) != len(variants[isa]["cells"]) for isa in variants)):
        raise ValueError("benchmark chart has empty, duplicate or mismatched cells")
    def section(case: str) -> tuple[float, str]:
        """Read the display grouping from the same reference row as its rates."""
        return model_section(lanes["AVX512"][case]["identity"]["configuration"])

    cases = sorted(lanes["AVX512"], key=lambda case: (
        section(case), lanes["AVX512"][case]["identity"]["configuration"]["model"], case))
    groups = [(key, list(rows)) for key, rows in groupby(cases, key=section)]
    height = 224 + len(cases) * 116 + len(groups) * 44
    revision = result["source"]["revision"]
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="{height}" viewBox="0 0 1280 {height}" role="img" aria-labelledby="title description">',
           '<title id="title">Llaminar published-image benchmarks</title>',
           f'<desc id="description">Both ISA images passed full HTTP E2E. Source {escape(revision)}. Grouped by model size. Exact prefill and decode medians in tokens per second. Each cell and phase is normalized to its own AVX512 result at 100%; bar lengths do not compare different cells.</desc>',
           '<style>text{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif;fill:#17243b}.muted{fill:#52647b}.small{font-size:13px}.value{font-size:15px;font-weight:650}.label{font-size:17px;font-weight:650}</style>',
           f'<rect width="1280" height="{height}" rx="18" fill="#f8fafc"/>',
           '<path d="M18 0H1262Q1280 0 1280 18V126H0V18Q0 0 18 0" fill="#122238"/>',
           '<text x="28" y="39" style="fill:white;font-size:27px;font-weight:750">Llaminar · Published-image benchmarks</text>',
           f'<text x="28" y="66" style="fill:#cbd5e1;font-size:14px">Image source {escape(revision)} · {escape(result["recorded_at_utc"][:10])}</text>',
           '<text x="28" y="94" style="fill:#cbd5e1;font-size:14px">Full HTTP E2E passed · warmed production inference · median of measured requests</text>',
           '<rect x="1000" y="80" width="15" height="15" rx="3" fill="#078c86"/><text x="1022" y="93" style="fill:white;font-size:13px">AVX512</text>',
           '<rect x="1110" y="80" width="15" height="15" rx="3" fill="#5663da"/><text x="1132" y="93" style="fill:white;font-size:13px">AVX2</text>',
           '<text x="28" y="151" class="small muted">AVX512 = 100% within each cell and phase · Compare ISA lanes within a row, not bar lengths between rows.</text>',
           '<text x="28" y="182" class="small muted">MODEL / TOPOLOGY / CONTEXT CAPACITY / ACTUAL PROMPT</text>',
           '<text x="630" y="182" class="small muted">PREFILL · TOK/S</text>',
           '<text x="955" y="182" class="small muted">DECODE · TOK/S</text>']
    group_starts = {rows[0]: (label, len(rows)) for (_, label), rows in groups}
    y = 202
    for index, case in enumerate(cases):
        if case in group_starts:
            label, count = group_starts[case]
            svg += [f'<g class="model-section"><title>{escape(label)}</title>',
                    f'<rect x="12" y="{y}" width="1256" height="36" rx="8" fill="#dce7f3"/>',
                    f'<text x="28" y="{y + 24}" class="label">{escape(label)}</text>',
                    f'<text x="1228" y="{y + 24}" text-anchor="end" class="small muted">{count} {"cell" if count == 1 else "cells"}</text></g>']
            y += 44
        reference = lanes["AVX512"][case]
        config = reference["identity"]["configuration"]
        other = lanes["AVX2"][case]
        if (other["identity"]["configuration"] != config
                or other["prefill_tokens"] != reference["prefill_tokens"]):
            raise ValueError("ISA chart lanes have mismatched workloads/configurations")
        model = Path(config["model"]).name.removesuffix(".gguf")
        svg += [f'<g class="benchmark-cell" data-case="{escape(case)}"><title>{escape(case)}</title>',
                f'<rect x="12" y="{y}" width="1256" height="108" rx="8" fill="{("#ffffff" if index % 2 == 0 else "#edf2f8")}"/>']
        svg.append(f'<text x="28" y="{y + 24}" class="label">{escape(model)}</text>')
        topology = topology_label(config)
        for line, content in enumerate(textwrap.wrap(topology, width=65)[:2]):
            svg.append(f'<text x="28" y="{y + 47 + line * 17}" class="small muted">{escape(content)}</text>')
        context = config["e2e"]["context_length"]
        prompt = reference["prefill_tokens"]
        svg.append(f'<text x="28" y="{y + 88}" class="small muted">Context {context:,} · Prompt {prompt:,} · Decode {variants["AVX512"]["workload"]["decode_tokens"]:,}</text>')
        for phase, x in (("prefill", 630), ("decode", 955)):
            rates = {isa: positive(lanes[isa][case]["tokens_per_second"][phase])
                     for isa in ("AVX512", "AVX2")}
            baseline = rates["AVX512"]
            # Only this pair sets its axis. Extend the visible range if AVX2
            # wins; never clip it, force it below AVX512, or use another cell's
            # speed. The marker and percentages retain AVX512 as the reference.
            local_maximum = max(rates.values())
            reference_x = x + baseline / local_maximum * 205
            svg.append(f'<g data-phase="{phase}" data-reference="AVX512">')
            for offset, isa, color in ((18, "AVX512", "#078c86"), (58, "AVX2", "#5663da")):
                rate = rates[isa]
                width = rate / local_maximum * 205
                percent = rate / baseline * 100
                svg += [f'<rect x="{x}" y="{y + offset}" width="205" height="12" rx="4" fill="#dde5ef"/>',
                        f'<rect data-isa="{isa}" x="{x}" y="{y + offset}" width="{width:.2f}" height="12" rx="4" fill="{color}"><title>{isa}: {rate:,.1f} tok/s · {percent:.1f}% of AVX512</title></rect>',
                        f'<text x="{x + 215}" y="{y + offset + 12}" class="value">{rate:,.1f}</text>',
                        f'<text x="{x + 215}" y="{y + offset + 28}" class="small muted">{percent:.1f}%</text>']
            svg.append(f'<line class="reference" x1="{reference_x:.2f}" x2="{reference_x:.2f}" y1="{y + 13}" y2="{y + 74}" stroke="#52647b" stroke-dasharray="2 3"><title>AVX512 reference: 100%</title></line></g>')
        svg.append('</g>')
        y += 116
    svg += [f'<text x="28" y="{height - 7}" class="small muted">Phase-specific evidence, not full-image certification. See accompanying JSON for exact policies, image digests, hardware and samples.</text>', '</svg>\n']
    return "\n".join(svg)
