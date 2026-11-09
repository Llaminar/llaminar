"""Classify the Mac Docker VM (NUMA-less linuxkit kernel, 4K pages, no published cache sizes)
Unit+preflight failures at a27c7ed4f. Kernel-truth launch: SYS_NICE + seccomp=unconfined + 16 GiB shm."""
import csv, re, sys
from collections import Counter
sys.path.insert(0, '/tmp/armqual-b1/15'); from extract import blocks
R = '/tmp/armqual-b1'
def failset(p):
    s = open(p).read(); i = s.index("The following tests FAILED:")
    return set(re.findall(r"\d+ - (V2_\S+) \(", s[i:]))
wsl = failset(f'{R}/raw/wsl/unit.log') | failset(f'{R}/raw/wsl/preflight.log')
standalone = open(f'{R}/mac/timeouts-standalone-a27c7ed4f.log').read()
passed_alone = set(re.findall(r"Test +#\d+: (V2_\S+) \.+ +Passed", standalone))
RULES = [
 ('X1 BUILD-BREAK (unguarded AVX2 .inc, cc502b95a)', r'^$', 'BUILD BREAK (reported to Main)'),
 ('N1 NUMA-less kernel: NUMA binding/policy required', r'libnuma reports NUMA unavailable|NUMA policy APIs are unavailable|NUMA APIs are unavailable|CPU root first-touch placement failed|cannot resolve its pinned NUMA participant|\(node\) >= \(0\), actual: -1|CPU shorthand requires observed NUMA affinity|Test__NUMAAllocator|NUMAPolicy', 'PRODUCT GAP (known: WP J single-memory-domain endpoint)'),
 ('N2 NUMA-less kernel: multi-rank inventory/rank-local backend requires NUMA affinity', r'Multi-rank hardware discovery requires exact CPU NUMA affinity|Rank-local CPU backend initialization requires exact NUMA affinity', 'PRODUCT GAP (known: WP J; shares G5/G7 detection path)'),
 ('N3 cache sizes unpublished: streaming measurement refused', r'Streaming CPU measurement requires observed aggregate cache coverage', 'PRODUCT GAP (known: WP J measured cache-geometry source)'),
]
rows = []
for kind in ('unit', 'preflight'):
    for name, v in blocks(f'{R}/mac/{kind}-a27c7ed4f.log').items():
        if 'Passed' in v['status']: continue
        text = '\n'.join(v['lines'])
        base = name in wsl or re.sub(r'_neon(_dotprod)?$', '_scalar', name) in wsl
        hits = [c for c, p, _ in RULES[1:] if re.search(p, text)]
        if re.search(r'could not access\s+or execute an executable|cannot prepare exact child spawn: No such file or directory', text) and 'fused_completion' in open(f'{R}/mac/build-a27c7ed4f.log').read():
            cls, verdict = RULES[0][0], RULES[0][2]
        elif 'Timeout' in v['status'] and name in passed_alone:
            cls, verdict = 'V1 slow VM: load timeout, passes standalone', 'ENVIRONMENT (VM)'
        elif 'Timeout' in v['status'] and 'MTP' in name and 'BenchmarkMatrix' in name or name == 'V2_Integration_MTPBenchmarkPolicyEvidence':
            cls, verdict = 'V2 slow VM: suite 34 s vs 30 s budget (idea 01-4)', 'ENVIRONMENT (VM budget)'
        elif base:
            cls, verdict = 'B0 base (fails on x86 WSL @39c24d8b4 by name)', 'BASE'
        elif hits:
            cls, verdict = hits[0], dict((c, vd) for c, _, vd in RULES)[hits[0]]
        else:
            cls, verdict = 'U UNCLASSIFIED', '?'
        rows.append(dict(test=name, suite=kind, status=v['status'].strip('*'), cls=cls, verdict=verdict,
                         secondary=';'.join(h.split()[0] for h in hits if h != cls)))
with open(f'{R}/15/classification-mac-a27c7ed4f.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows), 'rows; unit', sum(r['suite'] == 'unit' for r in rows), 'preflight', sum(r['suite'] == 'preflight' for r in rows))
for (c, v), k in sorted(Counter((r['cls'], r['verdict']) for r in rows).items()): print(f'{k:4} {c} [{v}]')
for r in rows:
    if r['cls'].startswith('U'): print('UNCLASSIFIED', r['test'], r['status'])
