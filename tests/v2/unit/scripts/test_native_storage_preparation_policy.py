#!/usr/bin/env python3
"""Adversarial policy replay for typed native preparation and exact root permits."""
import unittest
from check_native_storage_preparation_policy import scan_source
class NativePreparationPolicy(unittest.TestCase):
    def test_exact_authority_owners(self):
        for path in ("src/v2/planning/PhysicalMemoryAuthority.cpp","src/v2/collective/NativeStoragePreparationPmaBinding.h","src/v2/collective/NativeStoragePmaBinding.h"):
            self.assertEqual(scan_source(path,"auto token=epoch.claimPreparedNative(declaration);"),[])
    def test_ordinary_engine_cannot_open_or_claim(self):
        for source in ("PreparingNativeStorageTag bypass;","auto method=&Epoch::claimPreparation;","PhysicalMemoryNativeSetupEpoch hidden;","auto s=std::make_shared<NativeStoragePreparationPmaBinding>(resources,rank);","NativeStoragePreparationReplayAccess bypass;"):
            self.assertTrue(scan_source("src/v2/execution/Forbidden.cpp",source))
    def test_neighbor_does_not_inherit_exception(self):
        self.assertTrue(scan_source("src/v2/collective/Other.cpp","auto result=epoch.publishNativeAdmission(bundle);"))
    def test_session_can_hold_binding_but_not_raw_epoch(self):
        self.assertEqual(scan_source("src/v2/collective/LocalTPContext.h","std::shared_ptr<NativeStoragePmaBinding> binding;"),[])
        self.assertTrue(scan_source("src/v2/collective/LocalTPContext.h","auto epoch=beginPreparationEpoch(rank);"))
    def test_comments_and_raw_literals_do_not_grant_or_deny_capability(self):
        source='// NativeStoragePreparationPmaBinding\n/* claimPreparation */ auto s=R"x(PreparingNativeStorageTag)x";'
        self.assertEqual(scan_source("src/v2/models/Example.cpp",source),[])
    def test_planner_may_consume_immutable_geometry_without_opening_authority(self):
        self.assertEqual(scan_source("src/v2/planning/MemoryPlanner.cpp","void add(const NativeMandatoryStorageDeclarations& geometry){plan.add(geometry.mandatoryBOM());}"),[])
if __name__=="__main__":unittest.main()
