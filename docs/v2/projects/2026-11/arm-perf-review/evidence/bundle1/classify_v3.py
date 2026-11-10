"""Bundle-1 idea 15 HEURISTIC diagnostic classification, v3 (reviewer #144).

Tags are diagnostic text matches, not established causes. Logger lines go to stderr and are not
positioned inside a gtest case in ctest's merged output, so case-bound evidence sees only assertion
and exception text; whole_output_tags records the unbound (weaker) heuristic for every row. Where the output contains gtest
cases, evidence is bound to the FAILED cases only ([ RUN ] X .. [  FAILED  ] X); otherwise the
whole ctest output is scanned and the row says so (evidence_scope).

For every failing test of each ARM run this records three INDEPENDENT fields:
- x86_same_name: the same test name also failed on WSL x86 at 39c24d8b4. This is a baseline
  OBSERVATION only. The WSL logs were captured without --output-on-failure, so the x86 cause
  is unobservable and this overlap is never used as a cause.
- tags: every cause rule whose evidence line occurs in the ARM failure output. Tags overlap,
  and are matched over the whole ctest output of that test (including passing gtest siblings).
- primary: the first matching tag in the priority order below, or UNRESOLVED.
Runs: dgx-canonical (39c24d8b4, default seccomp), dgx-unmasked (52 tests, SYS_NICE +
seccomp=unconfined), mac (a27c7ed4f, kernel-truth launch)."""
import csv, re, sys
from collections import Counter
sys.path.insert(0, '/tmp/armqual-b1/15'); from extract import blocks
R = '/tmp/armqual-b1'
RULES = [  # (tag, regex, verdict) in priority order
 ('X1 build break: unguarded AVX2 .inc (cc502b95a)', r'could not access\s+or execute an executable|cannot prepare exact child spawn: No such file or directory', 'BUILD BREAK'),
 ('S1 container seccomp: NUMA syscall EPERM', r'mbind\(.*errno=1 |set_mempolicy\(MPOL_BIND.*errno=1 |get_mempolicy: Operation not permitted|thread memory policy: Operation not permitted|failed to bind memory', 'ENVIRONMENT (launch)'),
 ('G1 NUMAAllocator 4K page constant', r'Could not revoke \d+ bytes before first touch.*Invalid argument', 'PRODUCT GAP'),
 ('G2 CPUBackend 4K-aligned mbind', r'\[CPUBackend\] mbind\(.*errno=22', 'PRODUCT GAP'),
 ('G3 CPU device per CPU-less/zero node', r'CPU memory capacity should be positive', 'PRODUCT GAP'),
 ('G6 expert estimate Heap vs page-mapped', r'live claim exceeds admitted routed_expert_weights', 'PRODUCT GAP'),
 ('N2 multi-rank/rank-local requires NUMA affinity (G5/G7 path)', r'Multi-rank hardware discovery requires exact CPU NUMA affinity|Rank-local CPU backend initialization requires exact NUMA affinity|CPU shorthand requires observed NUMA affinity', 'PRODUCT GAP'),
 ('N1 NUMA binding/policy required on NUMA-less kernel', r'libnuma reports NUMA unavailable|NUMA policy APIs are unavailable|NUMA APIs are unavailable|get_mempolicy: Function not implemented|cannot resolve its pinned NUMA participant|\(node\) >= \(0\), actual: -1', 'PRODUCT GAP (WP J)'),
 ('N3 unpublished cache sizes: streaming measurement refused', r'Streaming CPU measurement requires observed aggregate cache coverage', 'PRODUCT GAP (WP J)'),
 ('M1 test targets a CPU-less node', r'Topology exposes no CPU on requested NUMA node', 'TEST ASSUMPTION'),
 ('T1 test mmap window 4096', r'page-aligned and nonze', 'TEST ASSUMPTION'),
 ('T2 4K-host byte literal/cap', r'4491052224|8392704u|recordedFragmentBytes|kSliceBytes \* kSliceCount|expected_final\) > \(expected_native\)|backing_region_count\) > \(1u\)|\(expected\) < \(31u\)|Native source cap cannot hold', 'TEST ASSUMPTION'),
 ('E1 container git: oneDNN origin probe', r"'oneDNN origin authentication failed' not found", 'ENVIRONMENT'),
 ('E2 docker CLI absent in container', r"No such file or directory: 'docker'", 'ENVIRONMENT'),
 ('E3 nested CMake configure fails in test sandbox', r'Configuring incomplete, errors occurred', 'UNRESOLVED-ENV? (cause of configure error not extracted)'),
 ('F1 ROCm trainer fixtures absent from tree', r"rocm/fixtures", 'FIXTURE MISSING'),
 ('V1 timeout under parallel gate', r'^$', 'ENVIRONMENT (VM)'),
]
VERDICT = {t: v for t, _, v in RULES}
def failset(p):
    s = open(p).read(); i = s.index("The following tests FAILED:")
    return set(re.findall(r"\d+ - (V2_\S+) \(", s[i:]))
x86 = failset(f'{R}/raw/wsl/unit.log') | failset(f'{R}/raw/wsl/preflight.log')
standalone = open(f'{R}/mac/timeouts-standalone-a27c7ed4f.log').read()
passed_alone = set(re.findall(r"Test +#\d+: (V2_\S+) \.+ +Passed", standalone))
RUNS = {'dgx-canonical': [f'{R}/raw/dgx/unit_test3.log', f'{R}/raw/dgx/preflight.log'],
        'dgx-unmasked': [f'{R}/raw/dgx/unmasking/unmask_rerun.log'],
        'mac': [f'{R}/mac/unit-a27c7ed4f.log', f'{R}/mac/preflight-a27c7ed4f.log']}
rows = []
for run, paths in RUNS.items():
    for path in paths:
        for name, v in blocks(path).items():
            if 'Passed' in v['status']:
                if run == 'dgx-unmasked':
                    rows.append(dict(run=run, test=name, status='Passed', x86_same_name=name in x86, evidence_scope='n/a', tags='', whole_output_tags='', primary='PASS under unmasked launch', verdict='ENVIRONMENT (launch)'))
                continue
            failed = set(re.findall(r'\[  FAILED  \] (\S+) \(', '\n'.join(v['lines'])))
            seg, cur = [], None
            for line in v['lines']:
                m = re.search(r'\[ RUN      \] (\S+)', line)
                if m: cur = m.group(1) if m.group(1) in failed else None; continue
                if re.search(r'\[ +(OK|FAILED) +\] ', line): cur = None; continue
                if cur: seg.append(line)
            scope = 'failed-gtest-cases' if failed else 'whole-output'
            text = '\n'.join(seg) if failed else '\n'.join(v['lines'])
            tags = [t for t, p, _ in RULES if p != '^$' and re.search(p, text)]
            whole = '\n'.join(v['lines'])
            whole_tags = [t.split()[0] for t, p, _ in RULES if p != '^$' and re.search(p, whole)]
            if 'Timeout' in v['status']:
                if run == 'mac' and name in passed_alone: tags.insert(0, 'V1 timeout under parallel gate')
                elif run == 'mac' and name in ('V2_Unit_MTPIterationBenchmarkMatrix', 'V2_Integration_MTPBenchmarkPolicyEvidence'):
                    tags.insert(0, 'V2 slow VM: 34 s suite vs 30 s budget (01-4)'); VERDICT[tags[0]] = 'ENVIRONMENT (VM budget)'
            primary = tags[0] if tags else 'NO-TAG'
            rows.append(dict(run=run, test=name, status=v['status'].strip('*'), x86_same_name=name in x86, evidence_scope=scope,
                             tags=';'.join(t.split()[0] for t in tags), whole_output_tags=';'.join(whole_tags), primary=primary, verdict=VERDICT.get(primary, '?')))
with open(f'{R}/15/classification-v3.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for run in RUNS:
    rr = [r for r in rows if r['run'] == run]
    print(f'===== {run}: {len(rr)} rows ({sum(r["status"]=="Passed" for r in rr)} passed), x86_same_name={sum(r["x86_same_name"] for r in rr)}')
    for p, k in sorted(Counter(r['primary'] for r in rr).items(), key=lambda x: -x[1]):
        same = sum(r['x86_same_name'] for r in rr if r['primary'] == p)
        scoped = sum(r['evidence_scope'] == 'failed-gtest-cases' for r in rr if r['primary'] == p)
        print(f'  {k:4} (x86-same-name {same:2}, failed-case-bound {scoped:2}) {p}')
    tagc = Counter(t for r in rr for t in r['tags'].split(';') if t)
    print('  tag occurrences (overlapping):', dict(sorted(tagc.items())))
    print('  UNRESOLVED:', [r['test'] for r in rr if r['primary'] == 'NO-TAG'])
