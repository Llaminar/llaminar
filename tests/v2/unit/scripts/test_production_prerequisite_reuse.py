#!/usr/bin/env python3
"""Prove explicit prerequisite reuse rebuilds first and rejects stale evidence.

The entrypoint shares the existing complete-inventory validator. These tests
exercise its command boundary without running compilers, CTest or accelerators.
The validator's build/timestamp/inventory negative controls remain in the
canonical production-parity campaign suite.
"""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/ci"))
import run_production_prerequisites as entrypoint


class PrerequisiteReuseTests(unittest.TestCase):
    """A receipt request must never become an implicit alternate test mode."""

    def test_rebuild_precedes_unchanged_evidence_validation(self):
        """Source edits reach Ninja before the canonical receipt can be admitted."""
        with tempfile.TemporaryDirectory() as temporary:
            build = Path(temporary).resolve()
            report = build / "prior report.json"
            order = []
            def build_current(command):
                order.append("build")
                self.assertEqual(command, ["cmake", "--build", str(build), "--parallel",
                    "--target", "v2_unit_gate", "v2_production_test_preflight_gate"])
                return subprocess.CompletedProcess(command, 0)
            def validate(actual_build, actual_report):
                order.append("validate")
                self.assertEqual((actual_build, actual_report), (build, report))
                return entrypoint.PreflightResult(0, 0.0, ("unit", "preflight"), None)
            with patch.object(entrypoint.subprocess, "run", side_effect=build_current), \
                 patch.object(entrypoint, "validate_source_freshness", side_effect=lambda *_: order.append("source")), \
                 patch.object(entrypoint, "reuse_unchanged_production_test_preflight", side_effect=validate), \
                 patch.object(entrypoint, "run_production_test_preflight") as full:
                self.assertEqual(entrypoint.main(["--build-dir", str(build), "--reuse-report", str(report)]), 0)
                self.assertEqual(order, ["build", "source", "validate"])
                full.assert_not_called()

    def test_failed_build_cannot_reuse_evidence(self):
        """An old report cannot certify sources whose current rebuild failed."""
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(entrypoint.subprocess, "run", return_value=subprocess.CompletedProcess([], 17)), \
             patch.object(entrypoint, "reuse_unchanged_production_test_preflight") as reuse:
            self.assertEqual(entrypoint.main(["--build-dir", temporary, "--reuse-report", "prior.json"]), 17)
            reuse.assert_not_called()

    def test_stale_receipt_fails_without_running_an_alternate_gate(self):
        """Explicit stale evidence is rejected even after a successful rebuild."""
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(entrypoint.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)), \
             patch.object(entrypoint, "validate_source_freshness"), \
             patch.object(entrypoint, "reuse_unchanged_production_test_preflight", side_effect=ValueError("stale build")), \
             patch.object(entrypoint, "run_production_test_preflight") as full:
            with self.assertRaises(SystemExit) as stopped:
                entrypoint.main(["--build-dir", temporary, "--reuse-report", "prior.json"])
            self.assertEqual(stopped.exception.code, 2)
            full.assert_not_called()

    def test_new_evidence_still_runs_the_complete_authority(self):
        """Normal qualification retains the existing complete Unit/preflight transaction."""
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(entrypoint, "run_production_test_preflight", return_value=entrypoint.PreflightResult(0, 1.0, (), None)) as full, \
             patch.object(entrypoint, "reuse_unchanged_production_test_preflight") as reuse:
            build = Path(temporary).resolve()
            output = build / "new evidence"
            self.assertEqual(entrypoint.main(["--build-dir", str(build), "--output", str(output)]), 0)
            full.assert_called_once_with(build, None, installed_build_receipt=None, artifact_directory=output)
            reuse.assert_not_called()

    def test_interpreted_source_edits_and_deletions_reject_reuse(self):
        """Real metadata detects changed scripts even when Ninja has no work."""
        for mutation in ("unchanged", "edit", "backdated_edit", "delete", "add"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                source = root / "scripts/ci/example.py"
                source.parent.mkdir(parents=True)
                source.write_text("original\n")
                build = root / "build"
                build.mkdir()
                (build / "CMakeCache.txt").write_text(
                    f"CMAKE_HOME_DIRECTORY:INTERNAL={root / 'src/v2'}\n")
                report = build / "prerequisites.json"
                report.touch()
                completed = time.time_ns()
                report.write_text(json.dumps({"preflight_completed_ns": completed}))
                # Keep receipt/build output outside Git's source inventory.
                names = b"scripts/ci/example.py\0"
                if mutation in ("edit", "backdated_edit"):
                    source.write_text("changed\n")
                    if mutation == "backdated_edit":
                        os.utime(source, ns=(1, 1))
                elif mutation == "delete":
                    source.unlink()
                    # Directory timestamps may coalesce unlink with the setup
                    # writes in one filesystem clock tick. Establish the exact
                    # post-receipt boundary this negative control requires;
                    # elapsed wall time is not a portable filesystem clock.
                    os.utime(source.parent, ns=(completed + 1, completed + 1))
                elif mutation == "add":
                    (source.parent / "new.py").write_text("new\n")
                    names += b"scripts/ci/new.py\0"
                with patch.object(entrypoint, "__file__", str(root / "scripts/ci/run_production_prerequisites.py")), \
                     patch.object(entrypoint.subprocess, "check_output", return_value=names):
                    if mutation == "unchanged":
                        entrypoint.validate_source_freshness(build, report)
                    else:
                        with self.assertRaisesRegex(ValueError, "source changed"):
                            entrypoint.validate_source_freshness(build, report)

    def test_receipt_cannot_admit_a_build_from_another_checkout(self):
        """A selectable build must retain the source ownership of this hook."""
        with tempfile.TemporaryDirectory() as temporary:
            build = Path(temporary)
            (build / "CMakeCache.txt").write_text("CMAKE_HOME_DIRECTORY:INTERNAL=/unrelated/src/v2\n")
            with self.assertRaisesRegex(ValueError, "this checkout"):
                entrypoint.validate_source_freshness(build, build / "missing.json")


class FocusedPanelPrerequisiteTests(unittest.TestCase):
    """Toy host metadata only; no real build/native/approval is fabricated."""

    def fixture(self, root):
        """Create bounded non-executable inputs for actual receipt validators."""
        from dataclasses import asdict
        build=root/'build';build.mkdir();evidence=root/'evidence';evidence.mkdir()
        script=root/'scripts/ci/check.py';script.parent.mkdir(parents=True);script.write_text('fixture-only\n')
        (build/'CMakeCache.txt').write_text(f'CMAKE_HOME_DIRECTORY:INTERNAL={root / "src/v2"}\n')
        for name in ['build.ninja','.ninja_log','CTestTestfile.cmake']:(build/name).write_text('fixture-only\n')
        names=('V2_Integration_FixtureDeviceFree',)
        records=({'name':names[0],'command':['/usr/bin/python3',str(script)],'properties':{
            'LABELS':['Integration','ProductionTestPreflight','DeviceFree'],'TIMEOUT':30}},)
        (evidence/'focused-tests.log').write_text('fixture-only CTest output\n')
        (evidence/'focused-tests.xml').write_text('<testsuite><testcase name="'+names[0]+'" status="run"><system-out>Ran 1 test</system-out></testcase></testsuite>')
        paths=entrypoint._focused_artifact_paths(build,records)|{evidence/'focused-tests.log',evidence/'focused-tests.xml'}
        report=entrypoint.FocusedPanelPrerequisiteReport(str(build),time.time_ns(),0,0.25,names,1,records,
            {str(path):entrypoint._focused_sha(path) for path in paths},'DIAGNOSTIC_FOCUSED_CHECKS_PASSED',True)
        path=evidence/'focused-prerequisites.json';path.write_text(json.dumps(asdict(report)))
        return build,path,records,script

    def checked(self, root, build, report, records):
        """Mock only discovery/inventory; run all real freshness/artifact checks."""
        with patch.object(entrypoint,'__file__',str(root/'scripts/ci/run_production_prerequisites.py')), \
             patch.object(entrypoint,'_focused_registrations',return_value=records), \
             patch.object(entrypoint.subprocess,'check_output',return_value=b'scripts/ci/check.py\0'):
            return entrypoint.validate_focused_panel_prerequisites(build,report)

    def test_reusable_receipt_is_explicit_diagnostic_without_running_any_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);build,report,records,_=self.fixture(root)
            with patch.object(entrypoint,'run_production_test_preflight') as full, \
                 patch.object(entrypoint,'run_focused_panel_prerequisites') as focused:
                first=self.checked(root,build,report,records);second=self.checked(root,build,report,records)
            self.assertEqual(first['focused_completed_ns'],second['focused_completed_ns'])
            self.assertFalse(first['full_preflight']);self.assertFalse(first['certification_eligible'])
            self.assertNotIn('passed',first)
            self.assertEqual(first['native_backend_regressions'],'not_covered_by_DeviceFree_selection')
            full.assert_not_called();focused.assert_not_called()
            with self.assertRaisesRegex(ValueError,'canonical prerequisite'):
                entrypoint.reuse_unchanged_production_test_preflight(build,report)

    def test_complete_or_certifiable_flags_cannot_be_smuggled_into_focused_receipt(self):
        mutations={'schema':1,'mode':'model-free-prerequisites','status':'GREEN','complete':False,
            'diagnostic_only':False,'full_preflight':True,'certification_eligible':True,'passed':True,
            'focused_return_code':False,'focused_test_count':False,'focused_completed_ns':True,
            'focused_elapsed_seconds':float('nan'),'native_backend_regressions':'covered'}
        for field,value in mutations.items():
            with self.subTest(field=field),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);build,report,records,_=self.fixture(root)
                data=json.loads(report.read_text());data[field]=value;report.write_text(json.dumps(data))
                with self.assertRaises((ValueError,TypeError)):self.checked(root,build,report,records)

    def test_changed_source_build_definitions_and_evidence_reject_reuse(self):
        for mutation in ['source','ninja','junit','missing_log','registration','foreign_build']:
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);build,report,records,script=self.fixture(root)
                if mutation=='source':script.write_text('changed source\n')
                elif mutation=='ninja':(build/'.ninja_log').write_text('changed actual build\n')
                elif mutation=='junit':(report.parent/'focused-tests.xml').write_text('<testsuite/>')
                elif mutation=='missing_log':(report.parent/'focused-tests.log').unlink()
                elif mutation=='registration':records=({**records[0],'command':['/other/interpreter',str(script)]},)
                else:
                    data=json.loads(report.read_text());data['focused_build_directory']='/another/build';report.write_text(json.dumps(data))
                with self.assertRaises((ValueError,OSError)):self.checked(root,build,report,records)

    def test_no_skipped_missing_failed_or_zero_body_check_is_admitted(self):
        name='V2_Integration_FixtureDeviceFree'
        cases=['<testsuite/>','<testsuite><testcase name="foreign"/></testsuite>',
            '<testsuite><testcase name="'+name+'"><skipped/></testcase></testsuite>',
            '<testsuite><testcase name="'+name+'"><failure/></testcase></testsuite>',
            '<testsuite><testcase name="'+name+'"><system-out>[ SKIPPED ] fixture</system-out></testcase></testsuite>',
            '<testsuite><testcase name="'+name+'"><system-out>Ran 0 tests</system-out></testcase></testsuite>']
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'junit.xml'
            for document in cases:
                path.write_text(document)
                with self.subTest(document=document),self.assertRaises(ValueError):entrypoint._focused_junit(path,(name,))

    def test_only_exact_registered_devicefree_nonmodel_checks_can_run(self):
        from copy import deepcopy
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);build,_,records,_=self.fixture(root)
            original=records[0]
            for label in ['FullModel','Campaign','FullDeviceInventory']:
                bad=deepcopy(original);bad['properties']['LABELS'].append(label)
                result=subprocess.CompletedProcess([],0,json.dumps({'tests':[bad]}),'')
                with patch.object(entrypoint.subprocess,'run',return_value=result),self.subTest(label=label),self.assertRaises(ValueError):
                    entrypoint._focused_registrations(build,(bad['name'],))
            for mutation in ['unknown','duplicate','missing_command','fixture','timeout','not_devicefree']:
                bad=deepcopy(original);names=(bad['name'],)
                if mutation=='unknown':bad['name']='V2_Integration_Other'
                elif mutation=='duplicate':names=names+names
                elif mutation=='missing_command':bad.pop('command')
                elif mutation=='fixture':bad['properties']['FIXTURES_REQUIRED']=['model']
                elif mutation=='timeout':bad['properties']['TIMEOUT']=121
                else:bad['properties']['LABELS'].remove('DeviceFree')
                result=subprocess.CompletedProcess([],0,json.dumps({'tests':[bad]}),'')
                with patch.object(entrypoint.subprocess,'run',return_value=result),self.subTest(mutation=mutation),self.assertRaises(ValueError):
                    entrypoint._focused_registrations(build,names)

    def test_canonical_process_guardian_keeps_devicefree_eligibility_and_interpreter_pin(self):
        """The ordinary Integration session owner is required even for host checks."""
        from copy import deepcopy
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);build,_,records,script=self.fixture(root)
            record=deepcopy(records[0]);record['properties']['LABELS'].append('ProcessOwned')
            helper=build/'llaminar_mpi_session';helper.write_text('toy guardian binary\n');helper.chmod(0o755)
            interpreter=root/'python3.12';interpreter.write_text('toy interpreter binary\n');interpreter.chmod(0o755)
            record['command']=[str(helper),'--session-base',str(build/'mpi-sessions/check'),'--',str(interpreter),str(script)]
            result=subprocess.CompletedProcess([],0,json.dumps({'tests':[record]}),'')
            with patch.object(entrypoint.subprocess,'run',return_value=result):
                selected=entrypoint._focused_registrations(build,(record['name'],))
            self.assertEqual(selected,(record,))
            paths=entrypoint._focused_artifact_paths(build,selected)
            self.assertTrue({helper,interpreter,script}.issubset(paths))

    def test_actual_ctest_discovery_accepts_exact_selection_without_executing_checks(self):
        """Replay the actual CTest regex parser; its dialect rejects Python groups."""
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);build,_,records,_=self.fixture(root)
            name=records[0]['name']
            (build/'CTestTestfile.cmake').write_text('add_test('+name+' "/usr/bin/python3" "--version")\n'
                'set_tests_properties('+name+' PROPERTIES LABELS "DeviceFree;ProductionTestPreflight;ProcessOwned" TIMEOUT "15")\n')
            selected=entrypoint._focused_registrations(build,(name,))
            self.assertEqual([record['name']for record in selected],[name])
            self.assertIn('ProcessOwned',selected[0]['properties']['LABELS'])

    def test_failed_owning_build_never_runs_ctest_or_complete_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);build,_,records,_=self.fixture(root);output=root/'fresh'
            with patch.object(entrypoint,'_focused_registrations',return_value=records), \
                 patch('run_production_parity_campaigns._run_process',return_value=17) as run, \
                 patch.object(entrypoint,'run_production_test_preflight') as full:
                self.assertEqual(entrypoint.run_focused_panel_prerequisites(build,[records[0]['name']],output),17)
            self.assertEqual(run.call_count,1);command=run.call_args.args[0]
            self.assertNotIn('v2_unit_gate',command);self.assertNotIn('v2_production_test_preflight_gate',command)
            report=json.loads((output/'focused-prerequisites.json').read_text())
            self.assertEqual(report['status'],'DIAGNOSTIC_FOCUSED_BUILD_FAILED')
            self.assertFalse(report['complete']);full.assert_not_called()

    def test_successful_producer_preserves_exact_ctest_evidence_and_reuses_without_rerun(self):
        """Exercise the original producer body using explicitly toy host CTest output."""
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);build,_,records,_=self.fixture(root);output=root/'fresh'
            def process(command,*_args,**_kwargs):
                if command[0]=='ctest':
                    (output/'focused-tests.log').write_text('toy host CTest output\n')
                    (output/'focused-tests.xml').write_text('<testsuite><testcase name="'+records[0]['name']+
                        '" status="run"><system-out>Ran 1 test</system-out></testcase></testsuite>')
                return 0
            with patch.object(entrypoint,'_focused_registrations',return_value=records), \
                 patch('run_production_parity_campaigns._run_process',side_effect=process) as run, \
                 patch.object(entrypoint,'run_production_test_preflight') as full:
                self.assertEqual(entrypoint.run_focused_panel_prerequisites(build,[records[0]['name']],output),0)
            self.assertEqual(run.call_count,2);full.assert_not_called()
            receipt=output/'focused-prerequisites.json'
            first=self.checked(root,build,receipt,records);second=self.checked(root,build,receipt,records)
            self.assertEqual(first,second);self.assertEqual(first['status'],'DIAGNOSTIC_FOCUSED_CHECKS_PASSED')

    def test_focus_option_cannot_implicitly_change_the_default_complete_cli(self):
        with tempfile.TemporaryDirectory() as temporary:
            for arguments in [['--focused-test','V2_Integration_Check'],['--diagnostic-focused-panel']]:
                with self.subTest(arguments=arguments),patch.object(entrypoint,'run_production_test_preflight') as full, \
                     patch.object(entrypoint,'run_focused_panel_prerequisites') as focused,self.assertRaises(SystemExit):
                    entrypoint.main(['--build-dir',temporary,'--output',str(Path(temporary)/'new'),*arguments])
                full.assert_not_called();focused.assert_not_called()


if __name__ == "__main__":
    unittest.main()
