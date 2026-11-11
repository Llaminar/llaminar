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
    def test_original_worker_owners_retain_bindings_without_raw_authority(self):
        for name in ("NativeContextPreparation", "PreparedGPUWorkerThread"):
            for suffix in ("h", "cpp"):
                path=f"src/v2/backends/{name}.{suffix}"
                with self.subTest(path=path):
                    self.assertEqual(scan_source(path,"std::shared_ptr<NativeStoragePreparationPmaBinding> source;"),[])
                    self.assertTrue(scan_source(path,"auto epoch=beginPreparationEpoch(rank);"))
        self.assertTrue(scan_source("src/v2/backends/OtherWorker.cpp","NativeStoragePreparationPmaBinding source;"))
    def test_process_owners_only_retain_original_source(self):
        from check_native_storage_preparation_policy import PROCESS_RETENTION_OWNERS
        for path in PROCESS_RETENTION_OWNERS:
            with self.subTest(path=path):
                for source in ("class NativeStoragePreparationPmaBinding;",
                               "std::shared_ptr<NativeStoragePreparationPmaBinding> source;",
                               "void retain(const std::shared_ptr<NativeStoragePreparationPmaBinding>& source);"):
                    self.assertEqual(scan_source(path,source),[])
                for source in ("auto source=std::make_shared<NativeStoragePreparationPmaBinding>(resources,rank);",
                               "auto source=new NativeStoragePreparationPmaBinding(resources,rank);",
                               "NativeStoragePreparationPmaBinding* source;",
                               "using Binding=NativeStoragePreparationPmaBinding;",
                               "std::shared_ptr<NativeStoragePmaBinding> adopted;",
                               "PhysicalMemoryNativeSetupEpoch epoch;",
                               "auto epoch=beginPreparationEpoch(rank);"):
                    self.assertTrue(scan_source(path,source))
        self.assertTrue(scan_source("src/v2/backends/OtherProcessLifetime.cpp",
                                   "std::shared_ptr<NativeStoragePreparationPmaBinding> source;"))
    def test_new_native_replay_access_cannot_escape_into_production(self):
        from check_native_storage_preparation_policy import REPLAY_DECLARATIONS
        for name, owners in REPLAY_DECLARATIONS.items():
            for path in owners:
                with self.subTest(name=name,path=path):
                    self.assertEqual(scan_source(path,f"namespace test {{ struct {name}; }} friend struct test::{name};"),[])
                    for body in (f"test::{name}::create();",f"struct {name} {{ void bypass(); }};",f"using Access=test::{name};"):
                        self.assertTrue(scan_source(path,body))
            self.assertTrue(scan_source("src/v2/execution/Other.cpp",f"friend struct test::{name};"))
    def test_original_event_observation_is_declaration_only_in_exact_native_owners(self):
        """The diagnostic bridge cannot become a production constructor/caller escape."""
        from check_native_storage_preparation_policy import REPLAY_DECLARATIONS
        name="NativeContextSourceLifecycleDiagnosticAccess"
        self.assertEqual(REPLAY_DECLARATIONS[name], {"src/v2/backends/cuda/NvidiaDeviceContext.h", "src/v2/backends/rocm/AMDDeviceContext.h"})
        for path in REPLAY_DECLARATIONS[name]:
            self.assertEqual(scan_source(path,f"namespace test {{struct {name};}} friend struct test::{name};"),[])
            for body in (f"struct {name} {{ void construct(); }};", f"test::{name}::orderingEvents(permit);", f"using Access=test::{name};"):
                self.assertTrue(scan_source(path,body))
        self.assertTrue(scan_source("src/v2/backends/Other.cpp",f"friend struct test::{name};"))
    def test_calibration_slot_friend_remains_declaration_only(self):
        """The original physical holder admits a test friend, never a production bridge call."""
        path="src/v2/backends/RetainedParticipantGraphExecutable.h"
        name="NativeContextPreparationReplayAccess"
        self.assertEqual(scan_source(path,
            f"class PreparedServiceCalibrationSlot {{ private: friend struct test::{name}; }};"),[])
        for source in (f"test::{name}::create();",f"struct {name} {{ void bypass(); }};",
                       f"using Access=test::{name};", "auto epoch=beginPreparationEpoch(rank);"):
            self.assertTrue(scan_source(path,source))
        self.assertTrue(scan_source("src/v2/backends/OtherGraph.h",f"friend struct test::{name};"))
    def test_native_replay_literals_and_comments_are_not_calls(self):
        source='// NativeContextPreparationReplayAccess::create()\nconst char* s=R"x(NativeContextBirthDiagnosticAccess::worker())x";'
        self.assertEqual(scan_source("src/v2/execution/Other.cpp",source),[])
    def test_host_service_bridges_are_exact_declarations_only(self):
        from check_native_storage_preparation_policy import REPLAY_DECLARATIONS
        for name in ("RetainedHostServiceReplayAccess", "RetainedHostServiceCalibrationAccess"):
            for path in REPLAY_DECLARATIONS[name]:
                self.assertEqual(scan_source(path,f"namespace test {{struct {name};}} friend struct test::{name};"),[])
                for body in (f"struct {name} {{ void construct(); }};", f"test::{name}::program(role);", f"using Access=test::{name};"):
                    self.assertTrue(scan_source(path,body))
            self.assertTrue(scan_source("src/v2/execution/Other.cpp",f"friend struct test::{name};"))
if __name__=="__main__":unittest.main()
