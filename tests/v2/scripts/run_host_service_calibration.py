#!/usr/bin/env python3
"""Validate explicit unqualified CPU worker diagnostics, never derive limits."""
import argparse
import json
import subprocess
from pathlib import Path

def _digest(value):
    """Require a complete nonzero source/program identity, never a placeholder."""
    return (isinstance(value, str) and len(value) == 64 and
            any(c != '0' for c in value) and
            all(c in '0123456789abcdef' for c in value))


def _geometry(row):
    """Authenticate one immutable role's declared stack, guard and exact root."""
    fields = ('exact_stack_bytes', 'virtual_guard_bytes', 'exact_allocation_bytes')
    values = tuple(row.get(field) for field in fields)
    if (any(type(value) is not int for value in values) or
            values[0] <= 0 or values[1] < 0 or values[2] <= values[0]):
        raise ValueError('Incomplete or invalid exact CPU storage geometry')
    if not _digest(row.get('identity')) or not _digest(row.get('resource_class')):
        raise ValueError('Incomplete original CPU program identity')
    return (row['identity'], row['resource_class'], *values)


def _trace_identity(rows, expected):
    """Reject missing lifetimes or mixed creator/diagnostic source generations."""
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError('Incomplete original CPU diagnostic records')
    if [(row.get('phase'), row.get('lifetime')) for row in rows] != expected:
        raise ValueError('Incomplete original CPU phase/lifetime sequence')
    if any(type(row.get('lifetime')) is not int for row in rows):
        raise ValueError('CPU lifetime must be an exact integer')
    sources = set()
    for row in rows:
        source = (row.get('creator_source'), row.get('diagnostic_source'))
        if source[0] != 'mapped-pthread-v1' or not _digest(source[1]):
            raise ValueError('Unsupported CPU creator family or incomplete test-build provenance')
        sources.add(source)
        if 'exact_claimed_bytes' in row and (type(row['exact_claimed_bytes']) is not int or row['exact_claimed_bytes'] < 0):
            raise ValueError('Exact CPU claim observation must be a nonnegative integer')
    if len(sources) != 1:
        raise ValueError('Changed CPU creator/diagnostic source identity')


def validate(rows, role, mode):
    """Check original program identity, exact phase sequence and mapped-root release."""
    if role not in ('immutable_lookup_io', 'executor_parent_ticket') or mode not in ('key', 'diagnostic-lifecycle'):
        raise ValueError('Unknown CPU diagnostic role or mode')
    expected = [('original_metadata_complete', 0)]
    if mode == 'key':
        expected.append(('key_only_complete', 0))
    else:
        group = ('before_exact_claim', 'original_claim_complete', 'original_worker_ready',
                 'original_worker_used', 'exact_join_unmap_complete')
        expected += [(phase, lifetime) for lifetime in range(1, 22) for phase in group]
        expected.append(('original_process_use_closed', 21))
    _trace_identity(rows, expected)
    if len({_geometry(row) for row in rows}) != 1:
        raise ValueError('Changed original CPU program identity or storage geometry')
    for row in rows:
        if (row.get('schema')!='retained-host-service-diagnostic-v1' or row.get('role')!=role or row.get('opaque_limit_installed') is not False or row.get('whole_process_bytes_complete') is not False or row.get('evidence_role')!='unqualified_actual_program'):
            raise ValueError('Changed diagnostic qualification contract')
        if row['phase'] in ('before_exact_claim', 'exact_join_unmap_complete', 'original_process_use_closed') and row.get('exact_claimed_bytes') != 0:
            raise ValueError('Exact CPU mapping remains outside its admitted lifetime')
        if row['phase'] in ('original_claim_complete', 'original_worker_ready', 'original_worker_used') and row.get('exact_claimed_bytes') != row['exact_allocation_bytes']:
            raise ValueError('Actual CPU root differs from declaration')
    return rows

def validate_population(rows, io, executor, claims_only=False):
    """Validate exact population, all-live checkpoints and zero-native controls without deriving bounds."""
    if type(io) is not int or type(executor) is not int or io<0 or executor<0 or io>16 or executor>16 or not 0<io+executor<=16:raise ValueError('Invalid explicit CPU population')
    group=['before_exact_claim','original_claims_complete']
    if not claims_only:group+=['all_original_workers_ready','all_original_tls_live','all_original_tasks_complete']
    group+=['exact_join_unmap_complete']
    expected = [('original_metadata_complete', 0)]
    expected += [(phase, lifetime) for lifetime in range(1, 22) for phase in group]
    expected.append(('original_process_use_closed', 21))
    _trace_identity(rows, expected)
    signatures=set()
    for row in rows:
        if type(row.get('io_workers')) is not int or type(row.get('executor_workers')) is not int:
            raise ValueError('CPU population observations must be exact integers')
        if (row.get('schema')!='retained-host-service-population-v1' or row.get('io_workers')!=io or row.get('executor_workers')!=executor or row.get('opaque_limit_installed') is not False or row.get('whole_process_bytes_complete') is not False or row.get('actual_consumers_qualified') is not False or row.get('work')!=('zero_native_claim_control' if claims_only else 'all_original_tls_blocks')):
            raise ValueError('Changed CPU population/qualification contract')
        classes=row.get('classes',[])
        if not isinstance(classes, list) or any(not isinstance(c, dict) for c in classes):
            raise ValueError('Incomplete original role identities')
        expected_roles=({'immutable_lookup_io'} if io else set())|({'executor_parent_ticket'} if executor else set())
        if {c.get('role') for c in classes}!=expected_roles or len(classes)!=len(expected_roles):raise ValueError('Incomplete original role identities')
        signatures.add(tuple((c['role'], *_geometry(c)) for c in classes))
        exact=sum(c['exact_allocation_bytes']*(io if c['role']=='immutable_lookup_io' else executor) for c in classes)
        if row['phase'] in ('original_claims_complete','all_original_workers_ready','all_original_tls_live','all_original_tasks_complete') and row.get('exact_claimed_bytes')!=exact:raise ValueError('Population exact CPU declaration differs from actual claims')
        if row['phase'] in ('before_exact_claim','exact_join_unmap_complete','original_process_use_closed') and row.get('exact_claimed_bytes')!=0:raise ValueError('Population exact roots did not retire')
    if len(signatures)!=1:raise ValueError('Original population class identity or storage geometry changed')
    return rows

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',required=True)
    parser.add_argument('--mode',choices=['key','diagnostic-lifecycle','diagnostic-population','diagnostic-claims'],required=True)
    parser.add_argument('--role',choices=['io','executor-ticket'])
    parser.add_argument('--io-workers',type=int)
    parser.add_argument('--executor-workers',type=int)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    command=[args.binary,'--mode='+args.mode]
    population=args.mode in ('diagnostic-population','diagnostic-claims')
    if population:
        if args.role or args.io_workers is None or args.executor_workers is None:parser.error('Population modes require both counts and no role')
        command+=['--io-workers='+str(args.io_workers),'--executor-workers='+str(args.executor_workers)]
    else:
        if not args.role or args.io_workers is not None or args.executor_workers is not None:parser.error('Key/single lifecycle require role and no counts')
        command+=['--role='+args.role]
    child=subprocess.run(command,text=True,capture_output=True,check=True)
    rows=[json.loads(line) for line in child.stdout.splitlines() if line.startswith('{')]
    if population:validate_population(rows,args.io_workers,args.executor_workers,args.mode=='diagnostic-claims')
    else:validate(rows,'immutable_lookup_io' if args.role=='io' else 'executor_parent_ticket',args.mode)
    args.output.write_text(json.dumps({'qualified':False,'whole_process_bytes_complete':False,'rows':rows,'stderr':child.stderr},indent=2)+'\n')
if __name__=='__main__':main()
