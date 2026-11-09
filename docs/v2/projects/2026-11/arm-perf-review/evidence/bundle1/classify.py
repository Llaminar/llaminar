"""Classify DGX (Grace, 64K pages, memory-only NUMA nodes, default-seccomp container) and WSL x86
failures at 39c24d8b4 from raw ctest logs. Every row cites the log signature that decided it."""
import json, re, sys, csv
sys.path.insert(0, '/tmp/armqual-b1/15'); from extract import blocks
R = '/tmp/armqual-b1/raw'
def failset(p):
    s = open(p).read(); i = s.index("The following tests FAILED:")
    return set(re.findall(r"\d+ - (V2_\S+) \(", s[i:]))
wsl = failset(f'{R}/wsl/unit.log') | failset(f'{R}/wsl/preflight.log')
dgx = {**{k: ('unit', v) for k, v in blocks(f'{R}/dgx/unit_test3.log').items()},
       **{k: ('preflight', v) for k, v in blocks(f'{R}/dgx/preflight.log').items()}}
dgx_fail = {k: v for k, v in dgx.items() if 'Passed' not in v[1]['status']}
# Ordered rules: (class, regex on block text, verdict, note). First match is the primary cause.
RULES = [
 ('P1 64K page: NUMAAllocator 4K page constant', r'Could not revoke \d+ bytes before first touch.*Invalid argument', 'PRODUCT GAP', 'NUMAAllocator.cpp PAGE_SIZE=4096; madvise EINVAL'),
 ('S1 container seccomp: NUMA syscalls EPERM', r'(mbind\(.*errno=1|set_mempolicy\(MPOL_BIND.*errno=1|get_mempolicy: Operation not permitted|Could not observe first-touch thread memory policy: Operation not permitted|memory not bound|failed to bind memory)', 'ENVIRONMENT', 'Docker default seccomp without CAP_SYS_NICE; masks page/memory-only effects behind it'),
 ('T1 64K page: test mmap window 4096', r'page-aligned and nonze', 'TEST ASSUMPTION', 'tests pass forStorage(...,4096,...); production default 64 KiB'),
 ('T2 64K page: recorded 4K byte constant', r'(4491052224|8392704u|recordedFragmentBytes|kSliceBytes \* kSliceCount|expected_final\) > \(expected_native\)|backing_region_count\) > \(1u\)|\(expected\) < \(31u\)|Native source cap cannot hold)', 'TEST ASSUMPTION', 'production rounds by sysconf page; test literal derived on 4K host'),
 ('I1 inventory: CPU device per NUMA node dir', r'CPU memory capacity should be positive', 'PRODUCT GAP (candidate)', 'DeviceRegistry registers every nodeN incl. CPU-less/zero-memory nodes'),
 ('E1 container git: origin probe', r"'oneDNN origin authentication failed' not found", 'ENVIRONMENT', 'devcontainer git ownership/probe; not ARM/page related'),
]
def classify(name, text):
    hits = [r for r in RULES if re.search(r[1], text)]
    return hits
rows = []
for n in sorted(set(dgx_fail) | wsl):
    on_dgx, on_wsl = n in dgx_fail, n in wsl
    base_equiv = on_wsl or (n.endswith('_neon') and n[:-5] + '_scalar' in wsl)
    text = '\n'.join(dgx_fail[n][1]['lines']) if on_dgx else ''
    hits = classify(n, text) if on_dgx else []
    if base_equiv:
        cls, verdict, note = 'B0 base defect (fails on x86 at same SHA)', 'BASE', 'not host-attributable'
    elif hits:
        cls, verdict, note = hits[0][0], hits[0][2], hits[0][3]
    elif on_wsl:
        cls, verdict, note = 'W x86-only', 'n/a', ''
    else:
        cls, verdict, note = 'U UNCLASSIFIED', '?', ''
    rows.append(dict(test=n, suite=dgx_fail[n][0] if on_dgx else 'wsl', dgx=on_dgx, wsl=on_wsl, cls=cls,
                     secondary=';'.join(h[0].split()[0] for h in hits[1:]), verdict=verdict, note=note))
with open('/tmp/armqual-b1/15/classification-39c24d8b4.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
from collections import Counter
print(len(rows), 'rows; dgx', sum(r['dgx'] for r in rows), 'wsl', sum(r['wsl'] for r in rows))
for (c, v), k in sorted(Counter((r['cls'], r['verdict']) for r in rows).items()): print(f'{k:4} {c} [{v}]')
for r in rows:
    if r['cls'].startswith('U'): print('UNCLASSIFIED', r['test'])

# ---- Unmasked rerun (SYS_NICE + seccomp=unconfined, 64 MiB shm), same build/SHA ----
UNMASK_RULES = [
 ('G1 NUMAAllocator 4K page constant', r'Could not revoke \d+ bytes before first touch.*Invalid argument', 'PRODUCT GAP'),
 ('G2 CPUBackend 4K-aligned mbind', r'\[CPUBackend\] mbind\(.*errno=22', 'PRODUCT GAP'),
 ('G3 CPU device per CPU-less/zero node', r'CPU memory capacity should be positive', 'PRODUCT GAP'),
 ('G6 expert estimate Heap vs anon-page materialization', r'live claim exceeds admitted routed_expert_weights', 'PRODUCT GAP'),
 ('G7 NUMA detection fallback -> inventory', r'Multi-rank hardware discovery requires exact CPU NUMA affinity', 'PRODUCT GAP'),
 ('M1 test iterates CPU-less nodes', r'Topology exposes no CPU on requested NUMA node', 'TEST ASSUMPTION'),
 ('T1 test mmap window 4096', r'page-aligned and nonze', 'TEST ASSUMPTION'),
 ('T2 4K-host byte literal', RULES[3][1], 'TEST ASSUMPTION'),
]
ub = blocks(f'{R}/dgx/unmasking/unmask_rerun.log')
for r in rows:
    if r['test'] not in ub: r['unmasked'] = ''; r['unmasked_classes'] = ''; continue
    v = ub[r['test']]
    if 'Passed' in v['status']:
        r['unmasked'] = 'PASS'; r['unmasked_classes'] = 'S1-only (seccomp mask)'; continue
    t = '\n'.join(v['lines'])
    hits = [c for c, p, _ in UNMASK_RULES if re.search(p, t)]
    r['unmasked'] = 'FAIL'
    r['unmasked_classes'] = 'B0 base' if r['verdict'] == 'BASE' else (';'.join(hits) or 'UNCLASSIFIED')
with open('/tmp/armqual-b1/15/classification-39c24d8b4.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
um = [r for r in rows if r['unmasked']]
print('\nunmasked rows', len(um), Counter(r['unmasked'] for r in um))
cc = Counter()
for r in um:
    if r['unmasked'] == 'FAIL':
        for c in r['unmasked_classes'].split(';'): cc[c] += 1
for c, k in sorted(cc.items()): print(f'{k:4} {c}')
for r in um:
    if r['unmasked'] == 'FAIL': print('  ', r['test'], '->', r['unmasked_classes'])
