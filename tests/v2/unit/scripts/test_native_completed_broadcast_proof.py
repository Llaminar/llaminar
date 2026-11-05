#!/usr/bin/env python3
"""Device-free negative controls for the installed native dependency verifier.

Synthetic reports test validation policy only; actual source-body certificates
are produced by the dependency builder and consumed by separate preflight tests.
No SDK, runtime library loading, model or GPU discovery occurs here.
"""
from pathlib import Path
import copy
import importlib.util
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location("native_completed_contract",
    ROOT / "scripts/docker/native-completed-broadcast/proof.py")
PROOF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROOF)


class NativeCompletedProofTests(unittest.TestCase):
    """Reject disconnected native/old-body outcomes without consulting hardware."""

    def setUp(self):
        """Construct a policy-only report bound to temporary bytes and current inputs."""
        self.temporary = tempfile.TemporaryDirectory(prefix="native-completed-policy-")
        self.addCleanup(self.temporary.cleanup)
        self.library = Path(self.temporary.name) / "policy-fixture.so"
        self.library.write_bytes(b"validation-only-not-a-runtime")
        self.backend = "rccl"
        self.receipt = "targets-policy-completed-" + PROOF.digest(PROOF.patch(self.backend)) + "-capture-policy"
        successes = dict(prefix="work_size=160 payload_offset=120\n" + PROOF.PREFIX_OK,
                         slice=PROOF.SLICE_OK, completion=PROOF.COMPLETION_OK, cancellation=PROOF.CANCELLATION_OK,
                         signal=PROOF.SIGNAL_OK, retirement=PROOF.RETIREMENT_OK,
                         proxy="actual SHM send/receive progress PASS: 20 epochs, 0/partial/full, exact memcpy bytes and all capacity steps\n",
                         work="actual parameter/FIFO native loader PASS, including independently wrapping geometry/tag words, abort-stale count zero, nonabort malformed count traps\n")
        observations = {key: [dict(backend=self.backend, returncode=0, output=value)]
                        for key, value in successes.items()}
        observations["original"] = dict(compile_exit=0, runs=[dict(returncode=87, stdout="",
            stderr="ORIGINAL_MISSING_COMPLETION\n") for _ in range(20)])
        observations["barrier"] = [
            dict(backend=self.backend,label="candidate",returncode=0,
                output="PASS actual native barrier:20 delayed peers,20 missing-peer aborts,unchanged gate/arrival,deferred plans,checked launch failure\n"),
            dict(backend=self.backend,label="original",expected_original_blocked=True,
                watchdog_seconds=1,output="ORIGINAL_BARRIER_ENTERED\n")]
        observations["protocol"] = dict(backend=self.backend,compile_exit=0,returncode=0,
            protocol_version=3,runs=20,source_sha256="2"*64)
        observations["plan"] = dict(backend=self.backend, returncode=0,
            output="PASS native plan:60admissions,40unsupported,20legacy\n", original_compile_exit=0,
            original=[dict(returncode=88, stdout="", stderr="ORIGINAL_GENERIC_FALLTHROUGH\n")
                      for _ in range(20)])
        observations["addressed"]=[dict(backend=self.backend,original=False,exit=0,stdout=PROOF.ADDRESSED_OK,stderr=""),
            dict(backend=self.backend,original=False,variant="capacity",exit=90,stdout="",stderr="UNNECESSARY_CAPACITY_METADATA_STEPS\n"),
            dict(backend=self.backend,original=False,variant="boundary",exit=91,stdout="",stderr="ZERO_SLICE_POINTER_LEAVES_SOURCE_BANK\n"),
            dict(backend=self.backend,original=True,exit=87,stdout="",stderr="ORIGINAL_UNADDRESSED_PAYLOAD_OR_MISSING_COMPLETION\n")]
        observations["p2p_launch"]=dict(compile_exit=0,returncode=0,output=
            "PASS 20 reset/replays, actual wave32/64, unroll/trace, high-channel and source/geometry/connection cases per replay plus pending multi-plan retirement, 5083 assertions\n")
        resources=dict(scope="sealed-addressed-native-p2p",native_executed=False,
            whole_library_zero_spill_qualified=False,allocation_evidence_complete=True,
            library_sha256=PROOF.digest(self.library),targets=["gfx906","gfx1100"],
            producer_sha256=PROOF.digest(PROOF.DOCKER/"llvm-lto-linker/artifact.py"),
            spill_guard_sha256=PROOF.digest(PROOF.DOCKER.parent/"build/gpu_spill_guard.py"),
            final_resource_evidence_sha256="4"*64,entries=[
                dict(architecture=target,function="_Z30ncclDevKernel_LocalP2p_"+str(unroll)+"_"+trace+"24ncclDevKernelArgsStorageILm4096EE",
                    private_bytes=0,vgpr_spills=0,frame_slots=0,out_of_line_calls=0,dynamic_stack=False,
                    allocation_instances=1,sgpr_lane_spills=4,code_sha256="5"*64)
                for target in ("gfx906","gfx1100") for unroll in (1,2,4) for trace in ("Plain","Trace")])
        resources["entries"] += [dict(architecture=target,function="_Z36ncclDevKernel_P2pProgressObservationPKv",
            private_bytes=0,vgpr_spills=0,frame_slots=0,out_of_line_calls=0,dynamic_stack=False,
            allocation_instances=1,sgpr_lane_spills=0,code_sha256="5"*64) for target in resources["targets"]]
        observations["progress"]=dict(backend=self.backend,compile_exit=0,returncode=0,source_sha256="2"*64,
            output="PASS original native connector observer:20 reset/replays,all actual channels/connections,immutable steps,140 fatal provenance cases\n")
        observations["p2p_role"]=dict(exit_code=0,stdout="PASS actual role predicate: 198 Native/Sealed valid/fatal cases\n",
            source_sha256="2"*64,snippet_sha256="6"*64)
        self.report = dict(rccl_owned_resources=resources,schema=1, certificate=True, backend=self.backend, scope=PROOF.SCOPE, protocol_version=3,
            native_runtime_qualified=False, sources=PROOF.source_identity(self.backend),
            library_sha256=PROOF.digest(self.library), build_receipt=self.receipt,
            exports=list(PROOF.EXPORTS), architecture="gfx906,gfx1100",
            compiler=dict(path="/policy/compiler", version="policy-only", sha256="1" * 64),
            native_sources={name: "2" * 64 for name in PROOF.source_files(self.backend)},
            original_ring_sha256="3" * 64, observations=observations)

    def verify(self, report):
        """Execute the actual passive verifier on one policy fixture."""
        PROOF.verify_report(report, self.backend, self.library, self.receipt)

    def test_complete_policy_fixture(self):
        """A complete report passes without loading the bytes as a native library."""
        self.verify(self.report)

    def test_live_step_and_original_progress_observations_are_mandatory(self):
        """Reject missing causal controls and an observer disconnected from original native types."""
        changes=[]
        for index in (1,2):
            report=copy.deepcopy(self.report);report["observations"]["addressed"][index]["exit"]=0;changes.append(report)
        report=copy.deepcopy(self.report);report["observations"].pop("progress");changes.append(report)
        report=copy.deepcopy(self.report);report["observations"]["progress"]["source_sha256"]="7"*64;changes.append(report)
        report=copy.deepcopy(self.report);report["rccl_owned_resources"]["entries"].pop();changes.append(report)
        for report in changes:
            with self.assertRaises(RuntimeError):self.verify(report)

    def test_rccl_owned_resources_require_complete_call_free_entry_coverage(self):
        """Reject missing targets/entries, memory spills, dynamic frames and unanalyzed callees."""
        mutations=[]
        for field,value in (("private_bytes",1),("vgpr_spills",1),("frame_slots",1),
                            ("out_of_line_calls",1),("dynamic_stack",True),("allocation_instances",0)):
            changed=copy.deepcopy(self.report)
            changed["rccl_owned_resources"]["entries"][0][field]=value
            mutations.append(changed)
        changed=copy.deepcopy(self.report);changed.pop("rccl_owned_resources");mutations.append(changed)
        changed=copy.deepcopy(self.report);changed["rccl_owned_resources"]["entries"].pop();mutations.append(changed)
        changed=copy.deepcopy(self.report);changed["rccl_owned_resources"]["entries"][-1]=copy.deepcopy(changed["rccl_owned_resources"]["entries"][0]);mutations.append(changed)
        changed=copy.deepcopy(self.report);changed["rccl_owned_resources"]["targets"]=["gfx906"];mutations.append(changed)
        for changed in mutations:
            with self.assertRaises(RuntimeError):
                self.verify(changed)

    def test_missing_partial_or_unrelated_replays_reject(self):
        """Zero work, partial negatives and unrelated failures cannot certify an ABI."""
        mutations = []
        for key in self.report["observations"]:
            changed = copy.deepcopy(self.report)
            del changed["observations"][key]
            mutations.append(changed)
        changed = copy.deepcopy(self.report)
        changed["observations"]["original"]["runs"].pop()
        mutations.append(changed)
        changed = copy.deepcopy(self.report)
        changed["observations"]["original"]["compile_exit"] = 1
        mutations.append(changed)
        changed = copy.deepcopy(self.report)
        changed["observations"]["plan"]["original"][0]["stderr"] = "segmentation fault\n"
        mutations.append(changed)
        changed = copy.deepcopy(self.report)
        changed["observations"]["completion"][0]["output"] = "PASS zero cases\n"
        mutations.append(changed)
        for changed in mutations:
            with self.subTest(changed=changed["observations"].keys()):
                with self.assertRaises(RuntimeError):
                    self.verify(changed)

    def test_stale_source_library_receipt_or_scope_reject(self):
        """The CPU proof is neither transferable to another DSO nor runtime evidence."""
        for key, value in (("sources", {}), ("library_sha256", "0" * 64),
                           ("build_receipt", "old-ABI"), ("native_runtime_qualified", True),
                           ("exports", []), ("original_ring_sha256", "2" * 64)):
            changed = copy.deepcopy(self.report)
            changed[key] = value
            with self.subTest(field=key):
                with self.assertRaises(RuntimeError):
                    self.verify(changed)

    def test_host_barrier_and_protocol_cannot_be_omitted(self):
        """Origin-only/old reports cannot attest host abort or semantic admission."""
        changes=[]
        for version in (None,0,1,2,4,3.0):
            changed=copy.deepcopy(self.report)
            if version is None:changed.pop("protocol_version")
            else:changed["protocol_version"]=version
            changes.append(changed)
        for key in ("barrier","protocol"):
            changed=copy.deepcopy(self.report);changed["observations"].pop(key);changes.append(changed)
        changed=copy.deepcopy(self.report);changed["observations"]["barrier"][1]["expected_original_blocked"]=False;changes.append(changed)
        changed=copy.deepcopy(self.report);changed["observations"]["barrier"][0]["returncode"]=1;changes.append(changed)
        changed=copy.deepcopy(self.report);changed["observations"]["protocol"]["protocol_version"]=1;changes.append(changed)
        changed=copy.deepcopy(self.report);changed["observations"]["protocol"]["runs"]=0;changes.append(changed)
        changed=copy.deepcopy(self.report);changed["observations"]["protocol"]["source_sha256"]="4"*64;changes.append(changed)
        for changed in changes:
            with self.subTest(changed=changed):
                with self.assertRaises(RuntimeError):self.verify(changed)

    def test_cuda_resource_inventory_rejects_duplicate_targets_and_missing_entries(self):
        """Entry counts alone cannot hide an omitted mode/architecture behind duplicates."""
        names = ["ncclDevKernel_Bytes_Broadcast_RING_" + protocol + "_" + str(mode)
                 for protocol in ("LL", "LL128", "SIMPLE") for mode in range(3)]
        resources = dict(scope="generated-native-byte-entries", native_executed=False,
                         targets=["sm_80", "sm_86"], entries=[])
        for target in resources["targets"]:
            for name in names:
                resources["entries"].append(dict(architecture=target,
                    function="_Z" + str(len(name)) + name + "24ncclDevKernelArgsStorageILm4096EE",
                    stack_bytes=0, spill_stores=0, spill_loads=0))
        resources["p2p"]=dict(scope="sealed-addressed-native-p2p",native_executed=False,targets=resources["targets"],
            entries=[dict(architecture=target,function="ncclDevKernel_LocalP2p",stack_bytes=0,spill_stores=0,spill_loads=0,registers=168)
                     for target in resources["targets"]],
            spill_controls=[dict(architecture=target,actual_spill_rejected=True) for target in resources["targets"]])
        resources["progress"]=[dict(architecture=target,function="ncclDevKernel_P2pProgressObservation",
            stack_bytes=0,spill_stores=0,spill_loads=0) for target in resources["targets"]]
        PROOF.verify_cuda_resources(resources)
        changed = copy.deepcopy(resources)
        changed["entries"][-1] = copy.deepcopy(changed["entries"][0])
        with self.assertRaises(RuntimeError):
            PROOF.verify_cuda_resources(changed)
        changed = copy.deepcopy(resources)
        changed["entries"].pop()
        with self.assertRaises(RuntimeError):
            PROOF.verify_cuda_resources(changed)
        changed = copy.deepcopy(resources)
        changed["targets"] = ["sm_80", "sm_80"]
        for row in changed["entries"]:
            row["architecture"] = "sm_80"
        with self.assertRaises(RuntimeError):
            PROOF.verify_cuda_resources(changed)

    def test_addressed_native_resources_and_original_payload_control_are_mandatory(self):
        """An absent/foreign payload witness or aliased edge claim cannot satisfy the existing proof owner."""
        for change in ("missing","wrong_payload","wrong_backend","no_original"):
            report=copy.deepcopy(self.report)
            if change=="missing":report["observations"].pop("addressed")
            elif change=="wrong_payload":report["observations"]["addressed"][0]["stdout"]=""
            elif change=="wrong_backend":report["observations"]["addressed"][0]["backend"]="nccl"
            else:report["observations"]["addressed"][1]["exit"]=0
            with self.subTest(change=change):
                with self.assertRaises(RuntimeError):self.verify(report)

    def test_native_first_error_and_zero_payload_obligations_are_not_optional(self):
        """Every installed report must retain the exact completion and physical-copy trace."""
        for key in ("completion", "cancellation", "proxy", "work", "signal", "retirement", "plan"):
            changed = copy.deepcopy(self.report)
            if key == "plan":
                changed["observations"][key]["original"] = []
            else:
                changed["observations"][key] = []
            with self.subTest(family=key):
                with self.assertRaises(RuntimeError):
                    self.verify(changed)


if __name__ == "__main__":
    unittest.main()
