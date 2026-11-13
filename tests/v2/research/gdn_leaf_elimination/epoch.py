#!/usr/bin/env python3
"""Qualify a new common serial/grouped arithmetic with fixed checkpoint periods.

State includes its raw pending updates. Neither speculative acceptance nor a
serialization/reset boundary may force a flush. The native scalar interpreter
and grouped backward/shared-prefix algorithms have different execution bodies.
This is research state, not yet the production cache/snapshot ABI.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import ctypes
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
from backward import SHAPES
from exact import guarded,check_guard,digest,inputs
from harness import Arithmetic,Comparison,LCG,Precision,physical_workers
from shared import Native as CoreNative
from shared import State,RawUpdate
from backward import bind as bind_leaf

MODES=('serial_snapshots','backward','combined')


class Native:
    """Bind fixed-epoch kernels and separately authenticate input preparation."""
    def __init__(self,library):
        self.library=ctypes.CDLL(str(Path(library).resolve(strict=True)))
        self.core=CoreNative(Path(library).with_name('libv2_research_gdn_leaf_native.so'))
        self.leaf=bind_leaf(Path(library).with_name('libv2_research_gdn_backward_native.so'))
        self.pointer=ctypes.POINTER(ctypes.c_float)
        self.ptr=self.core.ptr
        signatures={
            'gdn_epoch_state_size':[ctypes.c_int]*3,
            'gdn_epoch_serial':[ctypes.c_int]*3+[self.pointer]*3+[ctypes.c_float]*2+[self.pointer]*2,
            'gdn_epoch_lifecycle':[ctypes.c_int]*8+[ctypes.POINTER(ctypes.c_int)]+[self.pointer]*9+[ctypes.POINTER(ctypes.c_double)],
        }
        for name,args in signatures.items():
            function=getattr(self.library,name);function.argtypes=args;function.restype=ctypes.c_int

    def call(self,name,*args):
        """Reject any malformed native execution rather than changing policy."""
        result=getattr(self.library,name)(*args)
        if result!=1: raise AssertionError(f'{name}: {result}')

    def serial(self,period,dk,dv,state,q,k,v,decay,beta):
        """Independent ordinary M=1 arithmetic on a copied complete state."""
        result=state.copy();out=np.empty(dv,np.float32)
        self.call('gdn_epoch_serial',period,dk,dv,*(self.ptr(a) for a in (q,k,v)),
                  decay,beta,self.ptr(result),self.ptr(out))
        return result,out


class Experiment:
    """Prepared branch-correct inputs and complete canonical-state oracles."""
    def __init__(self,native,period,phase,parents,dk,dv,seed=20261024,first=None,last=None):
        self.native,self.period,self.dk,self.dv=native,period,dk,dv
        self.parents=tuple(parents);self.n=len(parents)
        self.parent_ids=np.array(parents,np.int32)
        self.size=native.library.gdn_epoch_state_size(period,dk,dv)
        if self.size<0 or not 0<=phase<period: raise ValueError('state geometry/phase')
        rng,arithmetic=LCG(seed),Arithmetic(Precision.FP32)
        self.root=np.zeros(self.size,np.float32)
        self.root[1:1+dk*dv]=rng.vector(arithmetic,dk*dv,128)
        rawq,k,v,gates=inputs(rng,(-1,)*(phase+2*self.n),dk,dv,'ordinary')
        prepared=[native.core.prepare(q,g) for q,g in zip(rawq,gates)]
        scaled=np.stack([x[0] for x in prepared]);decay=np.array([x[1][0] for x in prepared],np.float32);beta=np.array([x[1][1] for x in prepared],np.float32)
        for h in range(phase):
            self.root,_=native.serial(period,dk,dv,self.root,scaled[h],k[h],v[h],decay[h],beta[h])
        self.q,self.k,self.v,self.decay,self.beta=[a[phase:].copy() for a in (scaled,k,v,decay,beta)]
        self.raw_q,self.raw_gates=rawq[phase:].copy(),gates[phase:].copy()
        leaves=sorted(set(range(self.n))-set(parents))
        self.first=leaves[0] if first is None else first
        self.last=self.n-1 if last is None else last
        self.buffers=[guarded(shape) for shape in ((2*self.n,dv),(self.size,),(self.size,))]
        self.output,self.resolved,self.terminal=[x[0] for x in self.buffers]
        self.elapsed=ctypes.c_double()
        self.oracle()

    def oracle(self):
        """Branch serial histories, preserving raw logs at every accepted boundary."""
        self.expected=np.empty_like(self.output);self.expected_states=[]
        root=self.root
        for round_index,accepted in enumerate((self.first,self.last)):
            states={-1:root}
            for i,p in enumerate(self.parents):
                j=round_index*self.n+i
                states[i],self.expected[j]=self.native.serial(self.period,self.dk,self.dv,states[p],
                    self.q[j],self.k[j],self.v[j],self.decay[j],self.beta[j])
            root=states[accepted]
            self.expected_states.append(root.copy())

    def invoke(self,mode,iterations=1):
        """Execute the complete two-round native lifecycle without Python timing."""
        arrays=(self.root,self.q,self.k,self.v,self.decay,self.beta,self.output,self.resolved,self.terminal)
        self.native.call('gdn_epoch_lifecycle',mode,self.period,iterations,self.n,self.dk,self.dv,self.first,self.last,
            self.parent_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),*(self.native.ptr(a) for a in arrays),ctypes.byref(self.elapsed))
        return self.elapsed.value

    def validate(self):
        """Require all output and serialized-state bytes, including inactive zeros."""
        metric=Comparison()
        metric.observe(self.expected,self.output,'canonical outputs')
        for expected,actual in zip(self.expected_states,(self.resolved,self.terminal)):
            metric.observe(expected,actual,'complete serialized state')
        for _,guard in self.buffers: check_guard(guard)
        if metric.mismatches: raise AssertionError(vars(metric))
        return vars(metric)

    def materialize(self,state):
        """Diagnostic dense observation; it never replaces canonical live state."""
        dense=state[1:1+self.dk*self.dv].copy().reshape(self.dk,self.dv)
        stride=self.dk+self.dv+2
        for h in range(int(state[0])):
            record=state[1+self.dk*self.dv+h*stride:1+self.dk*self.dv+(h+1)*stride]
            dense=self.native.core.resolve(State(dense,RawUpdate(record[:self.dk].copy(),
                record[self.dk:self.dk+self.dv].copy(),record[-2],record[-1])))
        return dense

    def prepare_leaf_control(self):
        """Bind the existing qualified leaf algorithm to exactly the same operands."""
        self.leaf_root=self.materialize(self.root)
        self.leaf_buffers=[guarded(shape) for shape in ((2*self.n,self.dv),
            (self.n,self.dk,self.dv),(self.dk,self.dv),(self.dk,self.dv),(self.n,self.dk),(self.n,self.dv))]
        self.leaf_output,self.leaf_states,self.leaf_resolved,self.leaf_terminal,self.leaf_x,self.leaf_y=[x[0] for x in self.leaf_buffers]
        root=self.leaf_root;self.leaf_expected=np.empty_like(self.leaf_output)
        self.leaf_expected_states=[]
        for round_index,accepted in enumerate((self.first,self.last)):
            states={-1:root}
            for i,p in enumerate(self.parents):
                j=round_index*self.n+i
                states[i],self.leaf_expected[j],_=self.native.core.serial(states[p],self.raw_q[j],self.k[j],self.v[j],self.raw_gates[j])
            root=states[accepted];self.leaf_expected_states.append(root.copy())
            if root.tobytes()!=self.materialize(self.expected_states[round_index]).tobytes():
                raise AssertionError('canonical history changes materialized state')

    def invoke_leaf(self,iterations=1):
        """Time the existing one-hop leaf lifecycle, including both resolutions."""
        arrays=(self.leaf_root,self.q,self.k,self.v,self.decay,self.beta,self.leaf_output,self.leaf_states,
                self.leaf_resolved,self.leaf_terminal,self.leaf_x,self.leaf_y)
        self.native.leaf.call('gdn_backward_lifecycle',0,1,iterations,self.n,self.dk,self.dv,0,self.first,self.last,
            self.parent_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),*(self.native.ptr(a) for a in arrays),ctypes.byref(self.elapsed))
        return self.elapsed.value

    def validate_leaf(self):
        """Admit the baseline only against actual production M=1 and state bytes."""
        metric=Comparison();metric.observe(self.leaf_expected,self.leaf_output,'qualified leaf outputs')
        for expected,actual in zip(self.leaf_expected_states,(self.leaf_resolved,self.leaf_terminal)):
            metric.observe(expected,actual,'qualified leaf accepted state')
        for _,guard in self.leaf_buffers:check_guard(guard)
        if metric.mismatches:raise AssertionError(vars(metric))


def check_case(spec):
    """A physical-core worker owns one isolated native oracle and topology."""
    library,period,phase,shape,dk,dv=spec
    native=Native(library);case=Experiment(native,period,phase,SHAPES[shape],dk,dv)
    checks={}
    for mode,name in enumerate(MODES):
        case.invoke(mode);checks[name]=case.validate()
    return {'period':period,'phase':phase,'shape':shape,'dk':dk,'dv':dv,'checks':checks}


def grouping(native,period,dk=7,dv=16):
    """Repartition the same history, round-trip snapshots, and reset/reuse state."""
    rng=LCG(9102);length=65
    rawq,k,v,gates=inputs(rng,(-1,)*length,dk,dv,'ordinary')
    prepared=[native.core.prepare(q,g) for q,g in zip(rawq,gates)]
    q=np.stack([x[0] for x in prepared]);decay=np.array([x[1][0] for x in prepared],np.float32);beta=np.array([x[1][1] for x in prepared],np.float32)
    initial=np.zeros(native.library.gdn_epoch_state_size(period,dk,dv),np.float32)
    expected_states=[initial];expected=[]
    for i in range(length):
        state,out=native.serial(period,dk,dv,expected_states[-1],q[i],k[i],v[i],decay[i],beta[i])
        expected_states.append(state);expected.append(out)
    metric=Comparison()
    for mode in (1,2):
        for pattern in ((1,),(2,),(3,),(7,),(16,),(31,),(1,16,3,2,31)):
            state=initial.copy();position=0;batch=0
            while position<length:
                count=min(pattern[batch%len(pattern)],length-position)
                case=Experiment(native,period,0,tuple(range(-1,count-1)),dk,dv)
                case.root=np.frombuffer(state.tobytes(),dtype=np.float32).copy()
                for name,array in [('q',q),('k',k),('v',v),('decay',decay),('beta',beta)]:
                    segment=array[position:position+count]
                    setattr(case,name,np.concatenate((segment,segment)))
                case.first=case.last=count-1
                case.invoke(mode)
                position+=count
                metric.observe(np.array(expected[position-count:position]),case.output[:count],{'pattern':pattern,'mode':mode})
                metric.observe(expected_states[position],case.resolved,'checkpoint/history round trip')
                state=case.resolved.copy();batch+=1
            # Reset reconstructs the identical initial authority, not a dense
            # materialization of the terminal history.
            metric.observe(initial,np.zeros_like(state),'reset')
    if metric.mismatches: raise AssertionError(vars(metric))
    return vars(metric)


def measure(case,samples,iterations):
    """Match the qualified leaf baseline and rotate all six three-mode orders."""
    case.prepare_leaf_control()
    def invoke(mode,iterations=1):
        return case.invoke_leaf(iterations) if mode==0 else case.invoke(mode,iterations)
    def validate(mode):
        case.validate_leaf() if mode==0 else case.validate()
    for mode in (0,1,2):
        invoke(mode);validate(mode);start=time.perf_counter()
        while time.perf_counter()-start<0.1:invoke(mode,iterations)
    values={0:[],1:[],2:[]}
    for sample in range(samples):
        for mode in ((0,1,2),(1,2,0),(2,0,1),(2,1,0),(1,0,2),(0,2,1))[sample%6]:
            invoke(mode)
            values[mode].append(invoke(mode,iterations))
            validate(mode)
    names=('qualified_leaves','backward','combined')
    return {names[mode]:{'median_us':statistics.median(times)/1000,'samples_ns':times} for mode,times in values.items()}


def main():
    """Fail the entire experiment on any common-contract byte difference."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--periods',type=int,nargs='+',default=[1,2,4,8,16])
    parser.add_argument('--workers',type=int,default=physical_workers())
    parser.add_argument('--measure',action='store_true')
    parser.add_argument('--samples',type=int,default=12)
    parser.add_argument('--iterations',type=int,default=16)
    args=parser.parse_args()
    if not all(1<=p<=32 for p in args.periods) or min(args.workers,args.samples,args.iterations)<1:parser.error('invalid positive limits')
    library=args.library.resolve(strict=True);native=Native(library)
    manifest=json.loads(library.with_name('native-build.json').read_text())
    if args.measure and (manifest['core_build_type']!='Release' or 'CMAKE_BUILD_TYPE:STRING=Release\n' not in (library.parent/'CMakeCache.txt').read_text()):parser.error('require Release bridge and core')
    if args.measure:
        cases=[]
        for period in args.periods:
            for phase in range(period):
                for shape in ('star16','ternary3','binary4','chain16'):
                    case=Experiment(native,period,phase,SHAPES[shape],128,128)
                    row={'period':period,'phase':phase,'shape':shape,'timing':measure(case,args.samples,args.iterations)}
                    cases.append(row)
                    print(period,phase,shape,{k:v['median_us'] for k,v in row['timing'].items()},flush=True)
        partitions=[]
    else:
        specs=[(str(library),period,phase,shape,dk,dv) for period in args.periods for phase in range(period)
               for shape in SHAPES for dk,dv in ((7,16),(64,64),(128,128))]
        with ProcessPoolExecutor(max_workers=args.workers) as pool: cases=list(pool.map(check_case,specs))
        partitions=[{'period':period,'comparison':grouping(native,period)} for period in args.periods]
        print('PASS',len(cases),'tree/phase/geometries and',len(partitions),'repartitioning sweeps',flush=True)
    result={'schema':'gdn-canonical-epoch-v1','contract':'fixed-position checkpoint; backward nested-FMA output; rounded checkpoint flush',
            'scope':'research canonical state; production model/cache ABI not changed',
            'byte_equivalence_observed':True,'periods':args.periods,'cases':cases,'grouping':partitions,
            'build':manifest,'bridge_sha256':digest(library),'driver_sha256':digest(__file__),
            'baseline_contract':'adjusted-parent-fma-v1; each algorithm matches its own serial definition',
            'baseline_bridge_sha256':digest(library.with_name('libv2_research_gdn_backward_native.so')),
            'core_sha256':digest(Path(manifest['core_library'])),
            'affinity':sorted(os.sched_getaffinity(0)),'samples':args.samples,'iterations':args.iterations}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
