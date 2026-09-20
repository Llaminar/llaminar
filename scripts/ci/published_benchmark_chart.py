#!/usr/bin/env python3
"""Render exact published-image measurements as a dependency-free README SVG.

No values are synthesized or fitted. Both ISA lanes share phase-specific linear
scales and show their exact medians. The JSON retains full policy and individual
samples; the chart labels topology from canonical CLI metadata for readability.
"""
from __future__ import annotations

from collections import Counter
from html import escape
from pathlib import Path
import re
import textwrap

from production_artifacts import positive


def topology_label(configuration: dict) -> str:
    """Summarize exported participant addresses, never select a test topology."""
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
    cases = sorted(lanes["AVX512"], key=lambda case: (
        lanes["AVX512"][case]["identity"]["configuration"]["model"], case))
    maxima = {phase: max(positive(row["tokens_per_second"][phase])
                        for lane in lanes.values() for row in lane.values())
              for phase in ("prefill", "decode")}
    height = 198 + len(cases) * 116
    revision = result["source"]["revision"]
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="{height}" viewBox="0 0 1280 {height}" role="img" aria-labelledby="title description">',
           '<title id="title">Llaminar published-image benchmarks</title>',
           f'<desc id="description">Both ISA images passed full HTTP E2E. Source {escape(revision)}. Prefill and decode medians in tokens per second; independent linear scales.</desc>',
           '<style>text{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif;fill:#17243b}.muted{fill:#52647b}.small{font-size:13px}.value{font-size:15px;font-weight:650}.label{font-size:17px;font-weight:650}</style>',
           f'<rect width="1280" height="{height}" rx="18" fill="#f8fafc"/>',
           '<path d="M18 0H1262Q1280 0 1280 18V126H0V18Q0 0 18 0" fill="#122238"/>',
           '<text x="28" y="39" style="fill:white;font-size:27px;font-weight:750">Llaminar · Published-image benchmarks</text>',
           f'<text x="28" y="66" style="fill:#cbd5e1;font-size:14px">Image source {escape(revision)} · {escape(result["recorded_at_utc"][:10])}</text>',
           '<text x="28" y="94" style="fill:#cbd5e1;font-size:14px">Full HTTP E2E passed · warmed production inference · median of measured requests</text>',
           '<rect x="1000" y="80" width="15" height="15" rx="3" fill="#078c86"/><text x="1022" y="93" style="fill:white;font-size:13px">AVX512</text>',
           '<rect x="1110" y="80" width="15" height="15" rx="3" fill="#5663da"/><text x="1132" y="93" style="fill:white;font-size:13px">AVX2</text>',
           '<text x="28" y="156" class="small muted">MODEL / TOPOLOGY / CONTEXT CAPACITY / ACTUAL PROMPT</text>',
           '<text x="630" y="156" class="small muted">PREFILL · TOK/S</text>',
           '<text x="955" y="156" class="small muted">DECODE · TOK/S</text>']
    for index, case in enumerate(cases):
        y = 176 + index * 116
        reference = lanes["AVX512"][case]
        config = reference["identity"]["configuration"]
        other = lanes["AVX2"][case]
        if (other["identity"]["configuration"] != config
                or other["prefill_tokens"] != reference["prefill_tokens"]):
            raise ValueError("ISA chart lanes have mismatched workloads/configurations")
        model = Path(config["model"]).name.removesuffix(".gguf")
        svg += [f'<g><title>{escape(case)}</title>',
                f'<rect x="12" y="{y}" width="1256" height="108" rx="8" fill="{("#ffffff" if index % 2 == 0 else "#edf2f8")}"/>']
        svg.append(f'<text x="28" y="{y + 24}" class="label">{escape(model)}</text>')
        topology = topology_label(config)
        for line, content in enumerate(textwrap.wrap(topology, width=65)[:2]):
            svg.append(f'<text x="28" y="{y + 47 + line * 17}" class="small muted">{escape(content)}</text>')
        context = config["e2e"]["context_length"]
        prompt = reference["prefill_tokens"]
        svg.append(f'<text x="28" y="{y + 88}" class="small muted">Context {context:,} · Prompt {prompt:,} · Decode {variants["AVX512"]["workload"]["decode_tokens"]:,}</text>')
        for phase, x in (("prefill", 630), ("decode", 955)):
            for offset, isa, color in ((18, "AVX512", "#078c86"), (58, "AVX2", "#5663da")):
                rate = positive(lanes[isa][case]["tokens_per_second"][phase])
                width = rate / maxima[phase] * 205
                svg += [f'<rect x="{x}" y="{y + offset}" width="205" height="12" rx="4" fill="#dde5ef"/>',
                        f'<rect x="{x}" y="{y + offset}" width="{width:.2f}" height="12" rx="4" fill="{color}"/>',
                        f'<text x="{x + 215}" y="{y + offset + 12}" class="value">{rate:,.1f}</text>']
        svg.append('</g>')
    svg += [f'<text x="28" y="{height - 7}" class="small muted">Phase-specific evidence, not full-image certification. See accompanying JSON for exact policies, image digests, hardware and samples.</text>', '</svg>\n']
    return "\n".join(svg)
