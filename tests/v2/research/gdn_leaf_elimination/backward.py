#!/usr/bin/env python3
"""Compare three CPU research variants without confusing algebra with bytes.

Leaf-only retains rounded internal states. Backward propagates each query to
the root. Combined retains an upper prefix and contracts suffixes of at most
two updates. All three resolve accepted state in the original rounded order.
Timing includes two verifications, both acceptances and terminal publication;
topology preparation, allocation, normalization and gates are outside timing.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
from exact import check_guard, digest, guarded, inputs
from harness import Arithmetic, Comparison, LCG, Precision
from shared import Native

MODES = ('leaves', 'backward', 'combined')


def tree_widths(widths):
    """Build a breadth-first, parent-before-child tree with fixed level widths."""
    parents, previous = [], [-1]
    for width in widths:
        current = []
        for parent in previous:
            for _ in range(width):
                current.append(len(parents))
                parents.append(parent)
        previous = current
    return tuple(parents)


SHAPES = {'singleton': (-1,), 'star4': (-1,)*4, 'star16': (-1,)*16,
          'star39': (-1,)*39, 'fork4x4': tree_widths((4,4)),
          'ternary3': tree_widths((3,3,3)), 'binary4': tree_widths((2,2,2,2)),
          'chain16': tuple(range(-1,15)),
          'irregular': (-1,0,0,-1,1,4,2,3,7)}


def backward_reference(arithmetic, checkpoint, tree, prefix_states=()):
    """Apply literal backward contraction, optionally stopping at retained prefixes.

    The caller supplies actual prefix states; the exact-rational test constructs
    them through the independent dense recurrence. This function never computes
    a candidate correction or intermediate dense state.
    """
    outputs = []
    for i, node in enumerate(tree.nodes):
        x, y, current = node.q.copy(), arithmetic.zeros(len(node.v)), i
        while current >= 0 and current not in prefix_states:
            update = tree.nodes[current]
            c = update.beta * arithmetic.dot(update.k, x)
            y = y + c * update.v
            x = update.decay * (x - c * update.k)
            current = update.parent
        base = checkpoint if current < 0 else prefix_states[current]
        outputs.append(arithmetic.action(base, x) + y)
    return outputs


class Experiment:
    """Own immutable inputs and guarded persistent buffers for every candidate."""
    def __init__(self, native, parents, dk, dv, regime, seed, first=None, last=None):
        """Authenticate both rounds against actual production M=1 before timing."""
        self.native, self.parents = native, tuple(parents)
        self.n, self.dk, self.dv = len(parents), dk, dv
        self.indices = np.array(parents, dtype=np.int32)
        self.depth = []
        for parent in parents:
            self.depth.append(1 if parent < 0 else self.depth[parent]+1)
        self.cutoff = max(1, max(self.depth)-2)
        leaves = sorted(set(range(self.n))-set(parents))
        self.first = leaves[0] if first is None else first
        self.last = self.n-1 if last is None else last
        rng, arithmetic = LCG(seed), Arithmetic(Precision.FP32)
        self.root = rng.vector(arithmetic, dk*dv, 128).reshape(dk,dv)
        self.raw_q, self.k, self.v, self.gates = inputs(rng, (-1,)*(2*self.n), dk,dv,regime)
        prepared = [native.prepare(q,g) for q,g in zip(self.raw_q,self.gates)]
        self.q = np.stack([item[0] for item in prepared])
        self.decay = np.array([item[1][0] for item in prepared],np.float32)
        self.beta = np.array([item[1][1] for item in prepared],np.float32)
        self.buffers = [guarded(shape) for shape in (
            (2*self.n,dv),(self.n,dk,dv),(dk,dv),(dk,dv),(self.n,dk),(self.n,dv))]
        self.output,self.states,self.resolved,self.terminal,self.x,self.y = [p[0] for p in self.buffers]
        self.elapsed = ctypes.c_double()
        self.oracle()

    def oracle(self):
        """Save serial readouts and both accepted states with identical histories."""
        self.expected = np.empty_like(self.output)
        self.legacy = np.empty_like(self.output)
        root = self.root
        self.expected_states = []
        for round_index, accepted in enumerate((self.first,self.last)):
            states = {-1:root}
            for i,parent in enumerate(self.parents):
                j = round_index*self.n+i
                states[i],self.expected[j],self.legacy[j] = self.native.serial(
                    states[parent],self.raw_q[j],self.k[j],self.v[j],self.gates[j])
            root = states[accepted]
            self.expected_states.append(root.copy())

    def retained(self, mode):
        """Return exactly the state slots this mode is allowed to publish."""
        internal = set(self.parents)-{-1}
        return internal if mode==0 else set() if mode==1 else {i for i in internal if self.depth[i]<=self.cutoff}

    def invoke(self, mode, iterations=1, accumulation=1):
        """Call the native transaction; malformed or failed execution is fatal."""
        arrays = (self.root,self.q,self.k,self.v,self.decay,self.beta,self.output,
                  self.states,self.resolved,self.terminal,self.x,self.y)
        self.native.call('gdn_backward_lifecycle',mode,accumulation,iterations,self.n,self.dk,self.dv,
            self.cutoff,self.first,self.last,self.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            *(self.native.ptr(a) for a in arrays),ctypes.byref(self.elapsed))
        return self.elapsed.value

    def validate(self, mode, accumulation=1):
        """Require state bytes and guards; retain output differences as failures."""
        outputs, states, legacy = Comparison(), Comparison(), Comparison()
        outputs.observe(self.expected,self.output,'serial readouts')
        legacy.observe(self.legacy,self.output,'old serial readouts')
        for expected, actual in zip(self.expected_states,(self.resolved,self.terminal)):
            states.observe(expected,actual,'accepted state')
        for _,guard in self.buffers:
            check_guard(guard)
        retained = self.retained(mode)
        for i in set(range(self.n))-retained:
            if np.any(self.states[i].view(np.uint32)!=0x7FC12345):
                raise AssertionError('unretained state was published')
        if states.mismatches or (mode==0 and accumulation==1 and outputs.mismatches):
            raise AssertionError(f'control/state failure: {vars(outputs)}, {vars(states)}')
        return {'serial_outputs':vars(outputs),'accepted_states':vars(states),
                'legacy_outputs':vars(legacy),'byte_equivalent':outputs.mismatches==0,
                'retained_internal_states':len(retained)}

    def check(self, mode, accumulation=1):
        """Poison state banks before one fresh complete two-round execution."""
        self.states.view(np.uint32).fill(0x7FC12345)
        self.invoke(mode,accumulation=accumulation)
        return self.validate(mode,accumulation)


def bind(library):
    """Load the new bridge and its explicitly selected core, with exact ABI types."""
    native = Native(library)
    function = native.library.gdn_backward_lifecycle
    function.argtypes = [ctypes.c_int]*9+[ctypes.POINTER(ctypes.c_int)]+[native.pointer]*12+[ctypes.POINTER(ctypes.c_double)]
    function.restype = ctypes.c_int
    return native


def witness(native):
    """An erased 2^-25 update disproves even nested-FMA two-hop equivalence."""
    case = Experiment(native,(-1,0,1),1,8,'ordinary',1)
    case.root.fill(1)
    case.raw_q.fill(1); case.q.fill(1); case.k.fill(0); case.v.fill(0)
    case.k[1].fill(1); case.v[1].fill(2**-25)
    # exp(-0*softplus(0))=1 and sigmoid(80)=1 exactly in FP32.
    case.gates[:] = np.array([0,80,0,0],np.float32)
    case.decay.fill(1); case.beta.fill(1)
    case.oracle()
    records = {}
    for mode,name in enumerate(MODES):
        record = case.check(mode)
        record['depth3_serial'] = float(case.expected[2,0])
        record['depth3_output'] = float(case.output[2,0])
        records[name] = record
    if records['leaves']['depth3_output']!=0 or any(records[name]['depth3_output']!=2**-25 for name in MODES[1:]):
        raise AssertionError('minimal rounding witness changed')
    return records


def work(case, mode):
    """Count logical matrix actions and path terms including accepted resolution."""
    retained = case.retained(mode)
    def suffix(i):
        length = 0
        while i>=0 and i not in retained:
            length+=1; i=case.parents[i]
        return length
    return {'query_actions':2*case.n, 'key_actions':2*len(retained)+suffix(case.first)+suffix(case.last),
            'backward_updates':2*sum(suffix(i) for i in range(case.n) if i not in retained),
            'prefix_states':len(retained), 'accepted_missing_updates':[suffix(case.first),suffix(case.last)]}


def measure(case, samples, iterations):
    """Rotate all six mode orders and retain paired, unprofiled native timings."""
    checks = {name:case.check(mode) for mode,name in enumerate(MODES)}
    for mode in range(3):
        start=time.perf_counter()
        while time.perf_counter()-start<0.1:
            case.invoke(mode,iterations)
    timings = {name:[] for name in MODES}
    for sample in range(samples):
        for mode in ((0,1,2),(1,2,0),(2,0,1),(2,1,0),(1,0,2),(0,2,1))[sample%6]:
            case.states.view(np.uint32).fill(0x7FC12345)
            case.invoke(mode)
            timings[MODES[mode]].append(case.invoke(mode,iterations))
            current=case.validate(mode)
            if current!=checks[MODES[mode]]:
                raise AssertionError('replay changed output comparison or state proof')
    medians={name:statistics.median(values)/1000 for name,values in timings.items()}
    return {'checks':checks,'median_us':medians,'samples_ns':timings,
            'speedup_over_leaves':{name:medians['leaves']/medians[name] for name in MODES},
            'work':{name:work(case,mode) for mode,name in enumerate(MODES)}}


def main():
    """Run a byte diagnostic by default; rejected arithmetic remains rejected."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--measure',action='store_true')
    parser.add_argument('--samples',type=int,default=24)
    parser.add_argument('--iterations',type=int,default=32)
    args=parser.parse_args()
    if args.samples<6 or args.iterations<1: parser.error('need six samples and positive iterations')
    library=args.library.resolve(strict=True)
    manifest=json.loads(library.with_name('native-build.json').read_text())
    if args.measure and (manifest['core_build_type']!='Release' or 'CMAKE_BUILD_TYPE:STRING=Release\n' not in (library.parent/'CMakeCache.txt').read_text()):
        parser.error('timing requires Release core and bridge')
    native=bind(library)
    cases=[]
    geometries=((128,128),) if args.measure else ((7,16),(64,64),(128,128))
    regimes=('ordinary',) if args.measure else ('ordinary','endpoint','tiny')
    for dk,dv in geometries:
        for shape,parents in SHAPES.items():
            for regime in regimes:
                case=Experiment(native,parents,dk,dv,regime,20261024)
                row={'shape':shape,'rows':len(parents),'depth':max(case.depth),'dk':dk,'dv':dv,'regime':regime,'cutoff':case.cutoff}
                if args.measure: row.update(measure(case,args.samples,args.iterations))
                else:
                    row['checks']={name:case.check(mode) for mode,name in enumerate(MODES)}
                    row['literal_backward']=case.check(1,accumulation=0)
                cases.append(row)
                print(f'{shape}/{dk}/{regime}: '+str(row.get('median_us',{name:check['serial_outputs']['mismatches'] for name,check in row['checks'].items()})),flush=True)
    # Every acceptance endpoint, including total rejection, gets a separate
    # two-round continuation oracle in a small, genuinely branching geometry.
    endpoints=[]
    if not args.measure:
        for endpoint in range(-1,len(SHAPES['irregular'])):
            case=Experiment(native,SHAPES['irregular'],7,16,'ordinary',20261025,first=endpoint,last=endpoint)
            endpoints.append({name:case.check(mode) for mode,name in enumerate(MODES)})
    result={'schema':'gdn-backward-comparison-v1','scope':'CPU synthetic research; failed byte checks never certify production',
            'build':manifest,'core_sha256':digest(Path(manifest['core_library'])),'bridge_sha256':digest(library),
            'leaf_bridge_sha256':digest(library.with_name('libv2_research_gdn_leaf_native.so')),
            'driver_sha256':digest(__file__),'measured_isa':'AVX2','core_runtime_isa':os.environ.get('LLAMINAR_ISA_LEVEL','auto'),
            'affinity':sorted(os.sched_getaffinity(0)),'samples':args.samples,'iterations':args.iterations,
            'cases':cases,'every_acceptance_endpoint':endpoints,'rounding_witness':witness(native)}
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
