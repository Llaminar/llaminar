"""Bundle-1 idea 15 HEURISTIC diagnostic categorization, v4 (reviewer #144/#146).

Nothing in this file establishes a cause. For every failing test of each ARM run it records:
- x86_same_name: the same name also failed on WSL x86 at 39c24d8b4 (observation only; the WSL logs
  carry no failure output).
- evidence_scope: 'failed-gtest-text-segments' when the ctest output contains completed
  "[  FAILED  ] X (" markers; rule text is then selected only from the lines between "[ RUN ] X" and
  that marker for the failed cases. This is a text-selection rule: line position does not prove which
  case produced a line (stderr logger lines are not ordered with gtest stdout). Otherwise
  'whole-output-no-failure-marker' (Python/MPI/non-gtest, aborted or timed-out gtests): the whole
  output is used and the evidence is weaker.
- tags: rule matches within that scope (overlapping); whole_output_tags: matches over the whole output.
- heuristic_category: first scoped tag in priority order, or NO-TAG.
- cause_status: always 'not-established'. related_source_gap names the section-4 source analysis a
  G-tag refers to; it does not attribute this row's failure to that gap.
"""
import csv, re, sys
from collections import Counter
sys.path.insert(0, '/tmp/armqual-b1/15'); from extract import blocks
R = '/tmp/armqual-b1'
RULES = [  # (tag, regex) in priority order; labels describe matched text only
 ('X1 test executable missing', r'could not access\s+or execute an executable|cannot prepare exact child spawn: No such file or directory'),
 ('S1 EPERM from NUMA syscalls / MPI memory bind', r'mbind\(.*errno=1 |set_mempolicy\(MPOL_BIND.*errno=1 |get_mempolicy: Operation not permitted|thread memory policy: Operation not permitted|failed to bind memory'),
 ('G1 "Could not revoke ... Invalid argument"', r'Could not revoke \d+ bytes before first touch.*Invalid argument'),
 ('G2 CPUBackend mbind errno=22', r'\[CPUBackend\] mbind\(.*errno=22'),
 ('G3 "CPU memory capacity should be positive"', r'CPU memory capacity should be positive'),
 ('G6 "live claim exceeds admitted routed_expert_weights"', r'live claim exceeds admitted routed_expert_weights'),
 ('N2 "requires exact/observed NUMA affinity"', r'Multi-rank hardware discovery requires exact CPU NUMA affinity|Rank-local CPU backend initialization requires exact NUMA affinity|CPU shorthand requires observed NUMA affinity'),
 ('N1 NUMA unavailable text', r'libnuma reports NUMA unavailable|NUMA policy APIs are unavailable|NUMA APIs are unavailable|get_mempolicy: Function not implemented|cannot resolve its pinned NUMA participant|\(node\) >= \(0\), actual: -1'),
 ('N3 "requires observed aggregate cache coverage"', r'Streaming CPU measurement requires observed aggregate cache coverage'),
 ('M1 "Topology exposes no CPU on requested NUMA node"', r'Topology exposes no CPU on requested NUMA node'),
 ('T1 "mmap window must be page-aligned"', r'page-aligned and nonze'),
 ('T2 4K-host literal in failing assertion', r'4491052224|8392704u|recordedFragmentBytes|kSliceBytes \* kSliceCount|expected_final\) > \(expected_native\)|backing_region_count\) > \(1u\)|\(expected\) < \(31u\)|Native source cap cannot hold'),
 ('E1 oneDNN origin-probe text', r"'oneDNN origin authentication failed' not found"),
 ('E2 "No such file or directory: \'docker\'"', r"No such file or directory: 'docker'"),
 ('E3 "Configuring incomplete"', r'Configuring incomplete, errors occurred'),
 ('F1 rocm/fixtures path missing', r"rocm/fixtures"),
]
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
                    rows.append(dict(run=run, test=name, status='Passed', x86_same_name=name in x86, evidence_scope='n/a', tags='', whole_output_tags='', heuristic_category='PASS under changed launch', cause_status='not-established', related_source_gap=''))
                continue
            failed = set(re.findall(r'\[  FAILED  \] (\S+) \(', '\n'.join(v['lines'])))
            seg, cur = [], None
            for line in v['lines']:
                m = re.search(r'\[ RUN      \] (\S+)', line)
                if m: cur = m.group(1) if m.group(1) in failed else None; continue
                if re.search(r'\[ +(OK|FAILED) +\] ', line): cur = None; continue
                if cur: seg.append(line)
            scope = 'failed-gtest-text-segments' if failed else 'whole-output-no-failure-marker'
            text = '\n'.join(seg) if failed else '\n'.join(v['lines'])
            tags = [t for t, p in RULES if p != '^$' and re.search(p, text)]
            whole = '\n'.join(v['lines'])
            whole_tags = [t.split()[0] for t, p in RULES if p != '^$' and re.search(p, whole)]
            if 'Timeout' in v['status']:
                if run == 'mac' and name in passed_alone: tags.insert(0, 'V1 timeout in gate; passed once standalone')
                elif run == 'mac' and name in ('V2_Unit_MTPIterationBenchmarkMatrix', 'V2_Integration_MTPBenchmarkPolicyEvidence'):
                    tags.insert(0, 'V2 timeout in gate; 34 s standalone vs 30 s budget')
            category = tags[0] if tags else 'NO-TAG'
            rows.append(dict(run=run, test=name, status=v['status'].strip('*'), x86_same_name=name in x86, evidence_scope=scope,
                             tags=';'.join(t.split()[0] for t in tags), whole_output_tags=';'.join(whole_tags), heuristic_category=category, cause_status='not-established',
                             related_source_gap=category.split()[0] if category[:2] in ('G1', 'G2', 'G3', 'G6') else ''))
with open(f'{R}/15/classification-v4.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for run in RUNS:
    rr = [r for r in rows if r['run'] == run]
    print(f'===== {run}: {len(rr)} rows ({sum(r["status"]=="Passed" for r in rr)} passed), x86_same_name={sum(r["x86_same_name"] for r in rr)}')
    for c, k in sorted(Counter(r['heuristic_category'] for r in rr).items(), key=lambda x: -x[1]):
        sub = [r for r in rr if r['heuristic_category'] == c]
        seg = sum(r['evidence_scope'] == 'failed-gtest-text-segments' for r in sub)
        print(f'  {k:4} | x86-same-name {sum(r["x86_same_name"] for r in sub):2} | gtest-segment {seg:2} | whole-output {k - seg if c != "PASS under changed launch" else 0:2} | {c}')
    ids = sorted({t for r in rr for t in (r['tags'] + ';' + r['whole_output_tags']).split(';') if t})
    for t in ids:
        scoped = [r for r in rr if t in r['tags'].split(';')]
        seg = sum(r['evidence_scope'] == 'failed-gtest-text-segments' for r in scoped)
        whole = sum(t in r['whole_output_tags'].split(';') for r in rr)
        print(f'    tag {t}: scoped {len(scoped)} (gtest-segment {seg}, no-marker {len(scoped)-seg}) | whole-output {whole}')
    print('  NO-TAG:', [r['test'] for r in rr if r['heuristic_category'] == 'NO-TAG'])
