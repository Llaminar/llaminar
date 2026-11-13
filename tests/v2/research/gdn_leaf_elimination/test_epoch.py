#!/usr/bin/env python3
"""Adversarial canonical-state, acceptance and forced-flush negative controls.

The supplied native library is a research implementation. Tests distinguish
valid period changes from forbidden acceptance-dependent rounding boundaries.
Every acceptance endpoint and every admitted phase has an independent serial
state oracle. No passing diagnostic claims production model qualification.
"""
import argparse
import ctypes
from pathlib import Path
import unittest
import numpy as np
from backward import SHAPES
from epoch import Native,Experiment


class EpochContract(unittest.TestCase):
    """Exercise state ownership independently of numerical tolerances."""
    native=None

    def test_every_acceptance_and_rejection(self):
        """No acceptance decision changes canonical checkpoint placement."""
        for period in (1,2,4,8,16):
            for phase in range(period):
                for endpoint in range(-1,len(SHAPES['irregular'])):
                    case=Experiment(self.native,period,phase,SHAPES['irregular'],7,16,
                                    first=endpoint,last=endpoint)
                    before=case.root.tobytes()
                    for mode in (1,2):
                        case.invoke(mode);case.validate()
                        self.assertEqual(before,case.root.tobytes())

    def test_all_admitted_periods_and_phase_boundaries(self):
        """Period admission is total, including periods absent from timing."""
        for period in range(1,33):
            for phase in range(period):
                case=Experiment(self.native,period,phase,SHAPES['irregular'],1,8)
                for mode in (1,2):case.invoke(mode);case.validate()

    def test_premature_flush_is_observably_wrong(self):
        """A dense-only cache snapshot must fail the new arithmetic contract."""
        period,dk,dv=4,1,8
        state=np.zeros(self.native.library.gdn_epoch_state_size(period,dk,dv),np.float32)
        state[1:9]=1
        q=k=np.ones(1,np.float32);v=np.full(8,2**-25,np.float32)
        accepted,_=self.native.serial(period,dk,dv,state,q,k,v,1,1)
        restored=np.frombuffer(accepted.tobytes(),dtype=np.float32).copy()
        good,out=self.native.serial(period,dk,dv,restored,q,np.zeros_like(k),np.zeros_like(v),1,1)
        self.assertTrue(np.all(out==np.float32(2**-25)))
        # Materializing the accepted tiny erase gives zero. Treating that dense
        # matrix as a new checkpoint prematurely changes the following output.
        bad=np.zeros_like(state)
        _,wrong=self.native.serial(period,dk,dv,bad,q,np.zeros_like(k),np.zeros_like(v),1,1)
        self.assertTrue(np.all(wrong==0))
        self.assertNotEqual(out.tobytes(),wrong.tobytes())
        self.assertEqual(int(good[0]),2)

    def test_invalid_geometry_and_phase_fail_closed(self):
        """Reject overflow-sized geometry and malformed live counts before access."""
        for args in ((0,1,8),(33,1,8),(4,0,8),(4,2**31-1,2**31-1),(4,1,7)):
            self.assertEqual(self.native.library.gdn_epoch_state_size(*args),-1)
        q=np.ones(1,np.float32);v=np.ones(8,np.float32);output=np.full(8,123,np.float32)
        for count in (-1,0.5,4,float('nan')):
            state=np.zeros(self.native.library.gdn_epoch_state_size(4,1,8),np.float32);state[0]=count
            code=self.native.library.gdn_epoch_serial(4,1,8,*(self.native.ptr(a) for a in (q,q,v)),
                ctypes.c_float(1),ctypes.c_float(1),self.native.ptr(state),self.native.ptr(output))
            self.assertEqual(code,-1);self.assertTrue(np.all(output==123))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--library',type=Path,required=True)
    args,rest=parser.parse_known_args();EpochContract.native=Native(args.library)
    unittest.main(argv=['test_epoch.py',*rest])
