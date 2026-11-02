#!/usr/bin/env python3
"""Device-free replay of raw original creator observation and owned-child protocol."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/diagnostics"))
import run_native_context_birth_calibration as driver


def records(failure=0):
    """Recorded fixture metadata exercises validators, never a production certificate."""
    closure = dict(device="CUDA:0", physical_uuid="recorded", architecture="recorded", runtime_version=1,
                   driver_api_version=1, startup_visible_devices=2, resource_source_sha256="a"*64,
                   calibration_source_sha256="b"*64)
    geometry = dict(primary_streams=1, primary_blas_handles=1, secondary_blas_handles=1, ordering_events=32,
                    selected_workers=1, lifetimes=1, replays=2, fail_creator=failure, pool_admission=False,
                    opaque_family_excludes_exact_stack_object=True)
    stages = [("before_backend_lookup", {}), ("startup_metadata_complete", {}),
              ("exact_stack_object_root_ready", dict(lifetime=0, stack_bytes=8192, guard_bytes=4096,
               object_bytes=100, object_alignment=64, exact_pma_bytes=12288))]
    for index in range(1, (failure or 35)+1):
        role = ("primary_stream", "primary_blas", "secondary_blas")[index-1] if index <= 3 else "ordering_event"
        extra = dict(lifetime=0, creator=index, role=role, caller="/recorded/libllaminar2_core.so",
                     injected=index == failure, success=False, handle=0)
        stages.append(("before_original_creator", extra))
        stages.append(("after_original_creator", dict(extra, success=index != failure, handle=index if index != failure else 0,native_status=2 if index==failure else 0)))
    if failure:
        for index in range(1,failure):
            role=("primary_stream","primary_blas","secondary_blas")[index-1] if index<=3 else "ordering_event"
            stages.append(("after_original_destroy",dict(lifetime=0,creator=index,role=role,handle=index,destroy_success=True,destroy_status=0)))
        stages.append(("original_constructor_failed",dict(lifetime=0,reason="recorded original constructor failure after SDK status")))
        stages.append(("worker_birth_failed_cleanup_complete", dict(lifetime=0, creator=failure)))
    else:
        stages += [("selected_worker_ready", dict(lifetime=0, pool_generation=0, raw_direct_owner=True)),
                   ("native_first_use_complete", dict(lifetime=0, replay=0, ordering_events_used=32)),
                   ("native_reuse_complete", dict(lifetime=0, replay=1, ordering_events_used=32))]
    if not failure:
        for index in range(1,36):
            role=("primary_stream","primary_blas","secondary_blas")[index-1] if index<=3 else "ordering_event"
            stages.append(("after_original_destroy",dict(lifetime=0,creator=index,role=role,handle=index,destroy_success=True,destroy_status=0)))
    handles=failure-1 if failure else 35
    stages += [("original_worker_destroyed", dict(lifetime=0, failed=False,created_handles=handles,destroyed_handles=handles,rollback_complete=True)),
               ("exact_stack_object_root_released", dict(lifetime=0)),
               ("exclusive_context_retirement_complete", dict(reset_invoked=True, retired_context_generation=0,
                raw_direct_owner=True, retired_runtime_generation=1, successor_runtime_generation=2,
                runtime_post_reset_state="quiescent", canonical_allocations_before_runtime_reset=0,
                canonical_bytes_before_runtime_reset=0, injected_failure=bool(failure), diagnostic_failed=False))]
    return [dict(schema=driver.SCHEMA, pid=123, sequence=i, stage=name, closure=copy.deepcopy(closure),
                 geometry=copy.deepcopy(geometry), qualified=False, production_admission=False,
                 certified_limit=None, host=dict(maxrss_kib=1), extra=extra) for i, (name, extra) in enumerate(stages)]


def repeated_records(lifetimes=20, reset_between=True):
    """Recorded reset/rebirth protocol; never executes or certifies a GPU family."""
    original=records();rows=copy.deepcopy(original[:2])
    for lifetime in range(lifetimes):
        body=copy.deepcopy(original[2:-1])
        for row in body:row["extra"]["lifetime"]=lifetime
        rows.extend(body)
        if reset_between or lifetime+1==lifetimes:
            terminal=copy.deepcopy(original[-1]);terminal["extra"].update(lifetime=lifetime,
                retired_runtime_generation=lifetime+1 if reset_between else 1,
                successor_runtime_generation=lifetime+2 if reset_between else 2)
            rows.append(terminal)
    for index,row in enumerate(rows):
        row["sequence"]=index;row["geometry"].update(lifetimes=lifetimes,reset_between_lifetimes=reset_between)
    return rows


class OriginalBirthReceipt(unittest.TestCase):
    def test_twenty_actual_reset_boundaries_and_original_warm_default(self):
        """Protocol distinguishes20physical resets from20lifetimes sharing one reset."""
        for reset_between in (False,True):
            trace=repeated_records(reset_between=reset_between)
            driver.validate_records(trace,"cuda:0",0)
            self.assertEqual(sum(row["stage"]=="exclusive_context_retirement_complete" for row in trace),20 if reset_between else 1)
        driver.validate_checkpoint(repeated_records()[0],"cuda:0","a"*64,True)
        with self.assertRaisesRegex(ValueError,"reset policy"):
            driver.validate_checkpoint(records()[0],"cuda:0","a"*64,True)

    def test_reset_cycle_rejects_missing_stale_reordered_or_intervening_observation(self):
        """Every reset retains its exact generation and precedes next requested birth."""
        for mutation in ("missing","duplicate","generation","lifetime","not_quiescent","after_birth","intervening","wrong_policy","chain"):
            trace=repeated_records(3);reset=next(row for row in trace if row["stage"]=="exclusive_context_retirement_complete")
            index=trace.index(reset)
            if mutation=="missing":trace.pop(index)
            elif mutation=="duplicate":trace.insert(index,copy.deepcopy(reset))
            elif mutation=="generation":reset["extra"]["successor_runtime_generation"]=99
            elif mutation=="lifetime":reset["extra"]["lifetime"]=1
            elif mutation=="not_quiescent":reset["extra"]["runtime_post_reset_state"]="active"
            elif mutation=="after_birth":trace[index],trace[index+1]=trace[index+1],trace[index]
            elif mutation=="intervening":
                row=copy.deepcopy(reset);row["stage"]="post_reset_sdk_probe";trace.insert(index+1,row)
            elif mutation=="chain":
                second=[row for row in trace if row["stage"]=="exclusive_context_retirement_complete"][1]
                second["extra"].update(retired_runtime_generation=7,successor_runtime_generation=8)
            else:
                for row in trace:row["geometry"]["reset_between_lifetimes"]=False
            for number,row in enumerate(trace):row["sequence"]=number
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):driver.validate_records(trace,"cuda:0",0)

    def test_driver_forwards_only_explicit_reset_policy(self):
        """The default command stays warm; a requested reset cannot be dropped silently."""
        with tempfile.TemporaryDirectory() as folder:
            for selected in (False,True):
                args=argparse.Namespace(binary=Path("/recorded/native-birth"),device="cuda:0",output=Path(folder),replays=2,
                    reset_between_lifetimes=selected,creator_source={"source_sha256":"a"*64})
                with mock.patch.object(driver.common,"run_one",return_value={}) as runner:
                    driver.run_one(args,"command-only",20)
                    command=runner.call_args.kwargs["child_command"]
                    self.assertEqual("--reset-between-lifetimes" in command,selected)
                    checkpoint=records()[0]
                    if selected:checkpoint["geometry"]["reset_between_lifetimes"]=True
                    runner.call_args.kwargs["checkpoint_validator"](checkpoint)
                    checkpoint["geometry"]["reset_between_lifetimes"]=not selected
                    with self.assertRaisesRegex(ValueError,"reset policy"):
                        runner.call_args.kwargs["checkpoint_validator"](checkpoint)

    def test_owned_child_observes_each_reset_before_next_birth(self):
        """Real owned subprocess handshakes expose every recorded reset and reap cleanly."""
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);child=root/"reset-child.py";trace=repeated_records(2)
            child.write_text("import json,os,sys\nrows="+repr(trace)+"\nfor row in rows:\n row['pid']=os.getpid()\n print(json.dumps(row),flush=True)\n assert sys.stdin.readline().strip()=='continue'\n")
            args=argparse.Namespace(binary=Path(sys.executable),device="cuda:0",output=root,external_observer=None,identity_cache={},sample_ms=5)
            result=driver.common.run_one(args,"reset-recorded",2,child_command=[sys.executable,str(child)],expected_schema=driver.SCHEMA,record_validator=driver.validate_records,
                checkpoint_validator=lambda row:driver.validate_checkpoint(row,"cuda:0","a"*64,True))
            self.assertEqual(result["exit_code"],0);self.assertTrue(result["cleanup"]["reaped"])
            observed=[json.loads(path.read_text())["child"] for path in sorted((root/"reset-recorded").glob("checkpoint-*.json"))]
            self.assertEqual(sum(row["stage"]=="exclusive_context_retirement_complete" for row in observed),2)
            self.assertFalse(result["sampler_alive"])

    def test_complete_raw_family_and_every_partial_creator_prefix(self):
        """All35original prefix failures retain constructor rollback/reset, including late events."""
        driver.validate_records(records(), "cuda:0", 0)
        for index in range(1, 36):
            driver.validate_records(records(index), "cuda:0", index)

    def test_missing_created_family_or_reset_never_certifies(self):
        """Incomplete, duplicated, stale and manufactured native observations fail precisely."""
        for mutate in (
            lambda rows: rows.pop(4),
            lambda rows: rows[-1]["extra"].update(reset_invoked=False),
            lambda rows: rows[-1]["extra"].update(retired_context_generation=1),
            lambda rows: rows[-1]["extra"].update(canonical_allocations_before_runtime_reset=1),
            lambda rows: rows[4]["extra"].update(caller=""),
            lambda rows: rows[4]["extra"].update(success=None),
            lambda rows: rows[3]["geometry"].update(ordering_events=31),
            lambda rows: rows[1]["closure"].update(resource_source_sha256="foreign"),
            lambda rows: rows[3].update(qualified=True),
            lambda rows: rows[3].update(production_admission=True),
            lambda rows: next(row for row in rows if row["stage"]=="after_original_destroy")["extra"].update(destroy_success=False),
            lambda rows: next(row for row in rows if row["stage"]=="original_worker_destroyed")["extra"].update(rollback_complete=False),
        ):
            with self.subTest(mutation=mutate):
                trace=records();mutate(trace)
                with self.assertRaises(ValueError):driver.validate_records(trace,"cuda:0",0)

    def test_injected_birth_requires_unique_nonempty_original_constructor_reason(self):
        """Cleanup cannot replace the original SDK/constructor failure diagnostic."""
        for malformed in (None,"","   ",123):
            trace=records(19)
            next(row for row in trace if row["stage"]=="original_constructor_failed")["extra"]["reason"]=malformed
            with self.subTest(reason=malformed), self.assertRaisesRegex(ValueError,"primary reason"):
                driver.validate_records(trace,"cuda:0",19)
        for duplicate in (False,True):
            trace=records(35);cause=next(row for row in trace if row["stage"]=="original_constructor_failed")
            if duplicate:trace.insert(trace.index(cause),copy.deepcopy(cause))
            else:trace.remove(cause)
            for index,row in enumerate(trace):row["sequence"]=index
            with self.subTest(duplicate=duplicate), self.assertRaisesRegex(ValueError,"primary reason"):
                driver.validate_records(trace,"cuda:0",35)

    def test_mapped_elf_requires_content_identity_without_optional_build_id(self):
        """The actual project/SDK DSOs can have no GNU build-ID note."""
        original={"sha256":"a"*64,"size":530732576,"elf_build_ids":[]}
        driver.validate_caller_identity(original)
        driver.validate_caller_identity({**original,"elf_build_ids":["f"*40]})
        for bad in ({}, {**original,"sha256":""}, {**original,"sha256":"g"*64},
                    {**original,"size":0}, {**original,"size":True},
                    {**original,"elf_build_ids":None}):
            with self.subTest(identity=bad), self.assertRaises(ValueError):
                driver.validate_caller_identity(bad)

    def test_failed_prefix_cannot_publish_ready_worker(self):
        trace=records(35)
        target=next(row for row in trace if row["stage"]=="worker_birth_failed_cleanup_complete")
        target["stage"]="selected_worker_ready"
        with self.assertRaises(ValueError):driver.validate_records(trace,"cuda:0",35)

    def test_original_source_snapshot_and_first_checkpoint_are_authenticated(self):
        """Immutable source copies must match the linked constructor before SDK acknowledgment."""
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/"original.cpp";source.write_text("recorded original constructor\n")
            digest=driver.common.digest(source);relative="backends/original.cpp"
            manifest=root/"sources.json";manifest.write_text(json.dumps(dict(schema="llaminar.native_context_birth_source_snapshot.v1",
                inputs=[dict(path=relative,sha256=digest,snapshot=source.name)],
                source_sha256=driver.common.hashlib.sha256((relative+":"+digest+";").encode()).hexdigest())))
            expected=driver.validate_source_snapshot(manifest)
            trace=records();trace[0]["closure"]["resource_source_sha256"]=expected["source_sha256"]
            driver.validate_checkpoint(trace[0],"cuda:0",expected["source_sha256"])
            with self.assertRaises(ValueError):driver.validate_checkpoint(trace[0],"cuda:0","f"*64)
            source.write_text("changed creator body\n")
            with self.assertRaises(ValueError):driver.validate_source_snapshot(manifest)

    def test_wrong_constructor_source_is_reaped_before_first_api_acknowledgment(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);marker=root/"api-was-acknowledged";child=root/"recorded-child.py"
            trace=records(1)
            child.write_text("import json,os,sys\nrow="+repr(trace[0])+"\nrow['pid']=os.getpid()\nprint(json.dumps(row),flush=True)\nassert sys.stdin.readline().strip()=='continue'\nopen("+repr(str(marker))+",'w').write('forbidden')\n")
            args=argparse.Namespace(binary=Path(sys.executable),device="cuda:0",output=root,
                external_observer=None,identity_cache={},sample_ms=5)
            with self.assertRaisesRegex(RuntimeError,"diagnostic failed"):
                driver.common.run_one(args,"wrong-source",1,child_command=[sys.executable,str(child)],
                    expected_schema=driver.SCHEMA,record_validator=driver.validate_records,
                    checkpoint_validator=lambda record: driver.validate_checkpoint(record,"cuda:0","f"*64))
            self.assertFalse(marker.exists())
            result=json.loads((root/"wrong-source/process-result.json").read_text())
            self.assertTrue(result["cleanup"]["reaped"])

    def test_existing_owned_child_observer_accepts_explicit_other_schema_and_validator(self):
        """Actual pipe/handshake/session retirement is the existing driver implementation."""
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);child=root/"recorded-child.py"
            trace=records(1)
            child.write_text("import json,os,sys\nrows="+repr(trace)+"\nfor row in rows:\n row['pid']=os.getpid()\n print(json.dumps(row),flush=True)\n assert sys.stdin.readline().strip()=='continue'\n")
            args=argparse.Namespace(binary=Path(sys.executable),device="cuda:0",output=root,
                external_observer=None,identity_cache={},sample_ms=5)
            result=driver.common.run_one(args,"recorded",1,1,child_command=[sys.executable,str(child)],
                expected_schema=driver.SCHEMA,record_validator=driver.validate_records)
            self.assertEqual(result["exit_code"],0)
            self.assertTrue(result["cleanup"]["reaped"])
            self.assertFalse(result["sampler_alive"])
            self.assertFalse(result["stdout_reader_thread_created"])
            self.assertFalse(result["qualified"])


def lookup_records(program,prefix=None):
    """Recorded actual-hook shape only; not an SDK observation or bound."""
    rows=records();count=driver.LOOKUP_PROGRAM_COUNTS[program];used=count if prefix is None else prefix
    offset=next(i for i,r in enumerate(rows) if r["stage"]=="after_original_destroy")
    geometry=dict(rows[0]["geometry"],lookup_event_program=program,lookup_event_count=count,lookup_event_prefix=prefix)
    def row(stage,**extra):return dict(rows[0],stage=stage,extra=dict(lifetime=0,**extra))
    aux=[row("lookup_program_baseline",original_ordering_events=32,program=program,simultaneous_events=count,requested_prefix=used)]
    for i in range(used):
        data=dict(creator=36+i,role="ordering_event",caller="/recorded/libllaminar2_core.so",handle=100+i,injected=False,success=True,native_status=0)
        aux += [row("before_lookup_event_creator",**data),row("after_lookup_event_creator",**data),row("lookup_event_created",event=i,live_events=i+1)]
    aux += [row("lookup_program_prefix_ready",live_events=used,partial_prefix=used<count)]
    aux += [row("lookup_program_use_complete",replay=i,ordering_events_used=used,rerecord=i!=0) for i in range(2)]
    aux += [row("after_lookup_event_destroy",creator=36+i,role="ordering_event",handle=100+i,destroy_success=True,destroy_status=0) for i in reversed(range(used))]
    aux += [row("lookup_program_handles_destroyed",destroyed_events=used,physical_release_claimed=False)]
    rows[offset:offset]=aux
    for i,r in enumerate(rows):
        r["sequence"]=i;r["geometry"]=copy.deepcopy(geometry);r["closure"]=dict(r["closure"],lookup_program_family="native-auxiliary-operations",lookup_receipt_role="raw_existing_ordering_program_no_certificate")
        if r["stage"]=="original_worker_destroyed":r["extra"].update(created_handles=35+used,destroyed_handles=35+used)
    return rows


class LookupOrderingReceipt(unittest.TestCase):
    def test_every_closed_program_and_successful_partial_prefix(self):
        for program,count in driver.LOOKUP_PROGRAM_COUNTS.items():
            for prefix in (None,*range(count+1)):
                with self.subTest(program=program,prefix=prefix):driver.validate_records(lookup_records(program,prefix),"cuda:0",0)

    def test_named_operation_family_required_without_application_source_qualification(self):
        trace=lookup_records("local-positive")
        for row in trace:row["closure"]["lookup_resource_source_sha256"]="unrelated implementation revision"
        driver.validate_records(trace,"cuda:0",0)
        for invalid in (None,"","other-family","c"*64):
            bad=copy.deepcopy(trace)
            for row in bad:
                if invalid is None:row["closure"].pop("lookup_program_family")
                else:row["closure"]["lookup_program_family"]=invalid
            with self.assertRaises(ValueError):driver.validate_records(bad,"cuda:0",0)

    def test_missing_duplicate_reordered_or_false_release_receipts_reject(self):
        mutations=[lambda r:r.pop(next(i for i,v in enumerate(r) if v["stage"]=="after_lookup_event_destroy")),
                   lambda r:next(v for v in r if v["stage"]=="after_lookup_event_creator")["extra"].update(creator=3),
                   lambda r:next(v for v in r if v["stage"]=="lookup_program_use_complete")["extra"].update(ordering_events_used=99),
                   lambda r:next(v for v in r if v["stage"]=="lookup_program_handles_destroyed")["extra"].update(physical_release_claimed=True),
                   lambda r:next(v for v in r if v["stage"]=="original_worker_destroyed")["extra"].update(created_handles=35)]
        for mutate in mutations:
            trace=lookup_records("local-positive");mutate(trace)
            for i,r in enumerate(trace):r["sequence"]=i
            with self.assertRaises(ValueError):driver.validate_records(trace,"cuda:0",0)


if __name__ == "__main__":
    unittest.main()
