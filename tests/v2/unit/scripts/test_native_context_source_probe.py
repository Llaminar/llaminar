#!/usr/bin/env python3
"""Device-free actual protocol replay; recorded values never certify a native class."""
import copy
import hashlib
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/"scripts/diagnostics"))
import run_native_context_source_probe as probe


def records(role="key_only",backend="cuda",lifetime="ContextGeneration"):
    """Recorded metadata exercises original observation validators without SDK execution."""
    device=backend+":0";providers=[]
    for symbol,digest in zip(("cublasCreate_v2","cublasLtCreate") if backend=="cuda" else ("hipblasCreate","hipblasLtCreate"),("c"*64,"d"*64)):
        path="/recorded/native/"+symbol+".so";providers.append(dict(symbol=symbol,path=path,sha256=digest,expected_path=path,matches_configured_provider=True,provider_is_executable=False))
    key=dict(schema="llaminar.native_context_birth_key.v1",device=device,population="additional_context",source_program_version=1,
        visible_device_count=2,compute_units=80,native_wave_width=32,physical_uuid="1"*32,architecture="recorded",runtime_version="1",driver_version="1",
        source_sha256="e"*64,backend_sha256=hashlib.sha256(("c"*64+":"+"d"*64).encode()).hexdigest(),runtime_sha256="f"*64,driver_sha256="1"*64,
        runtime_policy_sha256="2"*64,hsa_sha256="3"*64 if backend=="rocm" else "",visibility_identity="4"*64,pthread_defaults_identity="5"*64,
        worker_stack_bytes=8192,worker_guard_bytes=4096,context_object_bytes=100,context_object_alignment=64,
        primary_streams=1,primary_blas_handles=1,secondary_blas_handles=1,dependency_events=32)
    base=dict(device=device,projection_role="uninstrumented_production_birth_key",resource_source_sha256="a"*64,diagnostic_source_sha256="b"*64,compiler="recorded")
    closure=dict(base,birth_key=key,native_blas_providers=providers,creator_symbols_interposed=False,existing_pool_worker=False)
    pma=dict(phase="Preparing",exact_preparation_bytes=12288,opaque_committed_bytes=12288,host_opaque_bytes=4096,device_opaque_bytes=8192)
    stages=[("before_backend_lookup",{}),("startup_metadata_complete",{})]
    if role=="key_only":stages.append(("production_key_complete",dict(source_created=False,catalog_invoked=False,pool_worker_created=False)))
    else:
        admitted=dict(pma,phase="Admitted");remaining=dict(admitted,exact_preparation_bytes=0)
        if lifetime=="ContextGeneration":remaining.update(opaque_committed_bytes=0,host_opaque_bytes=0,device_opaque_bytes=0)
        stages.extend([("preworker_observations_ready",dict(device_total_bytes=1<<30,device_free_bytes=1<<29,cpu_total_bytes=1<<30,cpu_free_bytes=1<<29)),
            ("source_prepared",dict(pma=pma,contract_host_upper=4096,contract_device_upper=8192,physical_lifetime=lifetime)),
            ("original_worker_ready",dict(generation=1,exact_original_worker=True,original_key_alias_verified=True,pma=pma)),
            ("same_pma_admitted",dict(same_authority_controlblock=True,same_certificate=True,pma=admitted)),
            ("original_event_family_used",dict(replay=0,ordering_events=32,same_retained_worker=True,exact_primary_stream=True)),
            ("original_event_family_used",dict(replay=1,ordering_events=32,same_retained_worker=True,exact_primary_stream=True)),
            ("original_permit_released",{}),
            ("original_source_retired",dict(generation=1,reset_invoked=True,runtime_post_reset_state="quiescent",retired_context_generation=1,
                retired_runtime_generation=1,successor_runtime_generation=2,canonical_allocations_before_runtime_reset=0,canonical_bytes_before_runtime_reset=0,physical_lifetime=lifetime,pma=remaining))])
    return [dict(schema=probe.SCHEMA,pid=123,sequence=i,stage=name,closure=copy.deepcopy(base if i==0 else closure),
        geometry=dict(role=role,selected_workers=1,replays=0 if role=="key_only" else 2),production_admission=role!="key_only" and i>=5,
        numerical_inference_executed=False,host=dict(maxrss_kib=1),extra=copy.deepcopy(extra)) for i,(name,extra) in enumerate(stages)]

class OriginalSourceProbeReplay(unittest.TestCase):
    def test_key_only_requires_actual_uninstrumented_provider_and_complete_class(self):
        for backend in ("cuda","rocm"):
            trace=records(backend=backend);probe.validate_records(trace,backend+":0")
            for field in set(trace[1]["closure"]["birth_key"])-{"source_sha256"}:
                bad=copy.deepcopy(trace)
                for row in bad[1:]:row["closure"]["birth_key"].pop(field)
                with self.subTest(backend=backend,field=field),self.assertRaises(ValueError):probe.validate_records(bad,backend+":0")
        for mutate in (lambda c:c.update(creator_symbols_interposed=True),lambda c:c["native_blas_providers"][0].update(provider_is_executable=True),
                lambda c:c["native_blas_providers"][0].update(matches_configured_provider=False),lambda c:c["native_blas_providers"][0].update(path="/diagnostic/interposer"),
                lambda c:c["native_blas_providers"][0].update(sha256="9"*64),lambda c:c.update(projection_role="raw_interposed_measurement")):
            trace=records()
            for row in trace[1:]:mutate(row["closure"])
            with self.assertRaises(ValueError):probe.validate_records(trace,"cuda:0")
    def test_raw_interposed_receipt_cannot_qualify_production_key(self):
        trace=records()
        for row in trace:row["schema"]="llaminar.native_context_birth_calibration.v1"
        with self.assertRaises(ValueError):probe.validate_records(trace,"cuda:0")
        trace=records();trace[-1]["extra"].update(catalog_invoked=True)
        with self.assertRaises(ValueError):probe.validate_records(trace,"cuda:0")
        trace=records();trace[-1]["production_admission"]=True
        with self.assertRaises(ValueError):probe.validate_records(trace,"cuda:0")
    def test_actual_source_order_same_pma_and_both_closed_lifetimes(self):
        for backend in ("cuda","rocm"):
            for lifetime in ("ContextGeneration","ProcessRuntime"):
                with self.subTest(backend=backend,lifetime=lifetime):probe.validate_records(records("source_lifecycle",backend,lifetime),backend+":0")
        for mutate in (lambda r:r.pop(-2),lambda r:r[4]["extra"].update(exact_original_worker=False),lambda r:r[4]["extra"].update(original_key_alias_verified=False),lambda r:r[4]["extra"].pop("original_key_alias_verified"),lambda r:r[5]["extra"].update(same_authority_controlblock=False),
                lambda r:r[6]["extra"].update(exact_primary_stream=False),lambda r:r[-1]["extra"].update(retired_context_generation=2),
                lambda r:r[-1]["extra"].update(canonical_allocations_before_runtime_reset=1),lambda r:r[-1]["extra"]["pma"].update(opaque_committed_bytes=0)):
            trace=records("source_lifecycle",lifetime="ProcessRuntime");mutate(trace)
            with self.assertRaises(ValueError):probe.validate_records(trace,"cuda:0")
    def test_changed_source_and_incomplete_key_reject_before_acknowledgment(self):
        trace=records();probe.validate_checkpoint(trace[0],"cuda:0","a"*64);probe.validate_checkpoint(trace[1],"cuda:0","a"*64)
        with self.assertRaises(ValueError):probe.validate_checkpoint(trace[0],"cuda:0","f"*64)
        trace[1]["closure"]["birth_key"].pop("runtime_policy_sha256")
        with self.assertRaises(ValueError):probe.validate_checkpoint(trace[1],"cuda:0","a"*64)
    def test_application_source_is_optional_evidence_but_snapshot_remains_exact(self):
        for evidence in (None,"","unrelated implementation revision"):
            trace=records()
            for row in trace:
                if evidence is None:row["closure"].get("birth_key",{}).pop("source_sha256",None)
                elif "birth_key" in row["closure"]:row["closure"]["birth_key"]["source_sha256"]=evidence
            probe.validate_records(trace,"cuda:0")
            with self.assertRaises(ValueError):probe.validate_checkpoint(trace[0],"cuda:0","f"*64)
if __name__=="__main__":unittest.main()
