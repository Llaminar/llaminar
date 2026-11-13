#!/usr/bin/env python3
"""Device-free proof verifier guards; real native parser executes only in builder."""
import copy, importlib.util, tempfile, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location('selector_proof',
    ROOT / 'tests/v2/integration/build/test_rccl_selector_taxonomy.py')
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)

class SelectorProofTests(unittest.TestCase):
    """Reject stale, missing and vacuous dependency parser evidence."""

    def setUp(self):
        """Create only tiny metadata fixture artifacts, no real dependency or accelerator."""
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.library = Path(self.tmp.name) / 'lib.so'
        self.library.write_bytes(b'fixture')
        self.report = dict(schema=1,
            sources=P.identities(),
            library_sha256=P.digest(self.library),
            native_sources={'/fixture/' + n: 'a' * 64 for n in (*P.NATIVE, 'src/include/device.h', 'src/include/nccl_common.h')},
            rccl_receipt='p2p-' + P.digest(ROOT / P.p2p_selection.PATCH) + '-fixture-selector-' + P.identities()[P.PATCH],
            p2p_selection=dict(
                native_sources={'/fixture/' + n: 'a' * 64 for n in P.p2p_selection.NATIVE},
                native_compile_inputs=[dict(source='/fixture/' + n, flags=['-DFIXTURE']) for n in P.p2p_selection.NATIVE],
                observations=[dict(variant=v, policy=name, stderr='',
                    returncode=int(v == 'original' and P.p2p_selection.FAILURES.get(name, 0) != 0),
                    measured=dict(checks=P.p2p_selection.CHECKS,
                        failures=P.p2p_selection.FAILURES.get(name, 0) if v == 'original' else 0))
                    for v in ('native', 'original') for name, _, _ in P.p2p_selection.POLICIES]),
            compiler=dict(path='/fixture/clang', stat=[1, 2, 3, 4, 5]),
            compiler_version='fixture',
            architecture='gfx906',
            sdk='/fixture/sdk',
            native_compile_inputs=[dict(source='/fixture/' + n, flags=['-DFIXTURE']) for n in P.NATIVE],
            observations=[
                dict(
                    variant=v,
                    returncode=int(v == 'original'),
                    stderr=(
                        '' if v == 'native'
                        else 'Original selector rejected unknown name'
                    ),
                    measured=dict(
                        checks=P.CHECKS,
                        failures=int(v == 'original'),
                        logging_warning_calls=4 if v == 'original' else 0,
                        logging_other_calls=0,
                    ),
                )
                for v in ['native', 'original']
            ],
        )

    def test_p2p_selector_requires_complete_actual_policy_matrix(self):
        """Missing defaults, level boundaries, exact negative counts or source pins fail closed."""
        for index in range(len(self.report['p2p_selection']['observations'])):
            r = copy.deepcopy(self.report)
            r['p2p_selection']['observations'].pop(index)
            with self.assertRaises(RuntimeError):
                P.verify(r, self.library)
        for field in ('native_sources', 'native_compile_inputs', 'observations'):
            r = copy.deepcopy(self.report)
            r['p2p_selection'][field] = [] if field != 'native_sources' else {}
            with self.assertRaises(RuntimeError):
                P.verify(r, self.library)
        for key, value in (('returncode', 0), ('measured', {'checks': P.p2p_selection.CHECKS, 'failures': 0})):
            r = copy.deepcopy(self.report)
            row = next(x for x in r['p2p_selection']['observations']
                       if x['variant'] == 'original' and x['policy'] == 'disable')
            row[key] = value
            with self.assertRaises(RuntimeError):
                P.verify(r, self.library)
        for receipt in (None, 'p2p-' + '0' * 64 + '-fixture-selector-' + P.identities()[P.PATCH]):
            r = copy.deepcopy(self.report)
            r['rccl_receipt'] = receipt
            with self.assertRaises(RuntimeError):
                P.verify(r, self.library)

    def test_complete_passes(self):
        """Both complete opposite-polarity outcomes are necessary."""
        P.verify(self.report, self.library)

    def test_source_or_library_changes_reject(self):
        """Proof cannot survive replaced runtime or source."""
        for key in P.INPUTS:
            r = copy.deepcopy(self.report)
            r['sources'][key] = 'bad'
            with self.assertRaises(RuntimeError):
                P.verify(r, self.library)
        self.library.write_bytes(b'changed')
        with self.assertRaises(RuntimeError):
            P.verify(self.report, self.library)

    def test_missing_or_vacuous_outcomes_reject(self):
        """Old-code success and signals do not establish a detecting regression."""
        for variant in [0, 1]:
            for code, checks, failures in [(0, 0, 0), (-11, P.CHECKS, 1), (0, P.CHECKS, True)]:
                r = copy.deepcopy(self.report)
                r['observations'][variant].update(returncode=code,
                    measured=dict(checks=checks, failures=failures))
                with self.assertRaises(RuntimeError):
                    P.verify(r, self.library)
        r = copy.deepcopy(self.report)
        r['observations'].pop()
        with self.assertRaises(RuntimeError):
            P.verify(r, self.library)

    def test_installer_and_receipt_bind_repair(self):
        """Packaging must copy/apply/hash the patch before any native build is accepted."""
        patch = 'rccl-collective-selector-taxonomy.patch'
        self.assertIn(patch, (ROOT / 'Dockerfile').read_text())
        self.assertIn(patch, (ROOT / 'scripts/docker/apply-rccl-capture-patch.sh').read_text())
        # The whole ordered pin remains mandatory: adding shared-file and thread repairs must
        # not disconnect selector, protocol or small-AllGather source identities.
        self.assertIn(
            '${RCCL_GIT_REF}-smi-${rccl_smi_patch_sha}-completed-${rccl_completed_patch_sha}-capture-${rccl_capture_patch_sha}'
            '-rows-${rccl_rows_patch_sha}-storage-${rccl_storage_patch_sha}'
            '-shm-${rccl_shm_patch_sha}-selector-${rccl_selector_patch_sha}-thread-${rccl_thread_patch_sha}'
            '-protocol-${rccl_protocol_patch_sha}-smallag-${rccl_small_allgather_patch_sha}',
            (ROOT / 'Dockerfile').read_text())
        self.assertIn(
            '${RCCL_COMMIT}-smi-${RCCL_SMI_PATCH_HASH}-completed-${RCCL_COMPLETED_PATCH_HASH}-capture-${RCCL_CAPTURE_PATCH_HASH}'
            '-rows-${RCCL_ROWS_PATCH_HASH}-storage-${RCCL_HOST_STORAGE_PATCH_HASH}'
            '-shm-${RCCL_SHM_PATCH_HASH}-selector-${RCCL_SELECTOR_PATCH_HASH}-thread-${RCCL_THREAD_PATCH_HASH}'
            '-protocol-${RCCL_PROTOCOL_PATCH_HASH}-smallag-${RCCL_SMALL_ALLGATHER_PATCH_HASH}',
            (ROOT / 'src/v2/CMakeLists.txt').read_text())
        self.assertIn(patch, (ROOT / 'src/v2/CMakeLists.txt').read_text())
        # Compiler provenance precedes, and never truncates, the strict repair suffix.
        self.assertIn(
            'targets-${rccl_target_hash}-llvm-lto-${rccl_lto_receipt_sha}'
            '-producer-${rccl_lto_producer_sha}-${RCCL_GIT_REF}-smi-${rccl_smi_patch_sha}-completed-',
            (ROOT / 'Dockerfile').read_text())
        self.assertIn(
            'string(PREPEND RCCL_BUILD_ID "targets-${RCCL_TARGET_POLICY_HASH}'
            '-llvm-lto-${RCCL_LTO_RECEIPT_HASH}-producer-${RCCL_LTO_PRODUCER_HASH}-")',
            (ROOT / 'src/v2/CMakeLists.txt').read_text())

    def test_missing_each_native_pin_and_closure_rejects(self):
        """All five real source/enum files and compiler/receipt metadata are mandatory."""
        for name in self.report['native_sources']:
            r = copy.deepcopy(self.report)
            r['native_sources'].pop(name)
            with self.assertRaises(RuntimeError):
                P.verify(r, self.library)
        for key in ['compiler_version',
            'compiler',
            'architecture',
            'sdk',
            'rccl_receipt',
            'native_compile_inputs']:
            r = copy.deepcopy(self.report)
            r.pop(key)
            with self.assertRaises(RuntimeError):
                P.verify(r, self.library)

    def test_duplicate_outcomes_and_boolean_counts_reject(self):
        """Bool counts/return codes and duplicated outcomes cannot mimic full coverage."""
        r = copy.deepcopy(self.report)
        r['observations'].append(r['observations'][0])
        with self.assertRaises(RuntimeError):
            P.verify(r, self.library)
        for key in ['checks', 'failures']:
            r = copy.deepcopy(self.report)
            r['observations'][0]['measured'][key] = False
            with self.assertRaises(RuntimeError):
                P.verify(r, self.library)
        r = copy.deepcopy(self.report)
        r['observations'][0]['returncode'] = False
        with self.assertRaises(RuntimeError):
            P.verify(r, self.library)

    def test_positive_stderr_rejects_negative_diagnostics_remain_valid(self):
        """Native warnings are anomalous; expected old-parser rejection text is retained."""
        P.verify(self.report, self.library)
        r = copy.deepcopy(self.report)
        r['observations'][0]['stderr'] = 'unexpected native diagnostic'
        with self.assertRaises(RuntimeError):
            P.verify(r, self.library)

    def test_logging_seam_must_be_observed_and_native_quiet(self):
        """Old parser must hit the host-only logger exactly; disconnected/native logs reject."""
        for variant, count in [(0, 1), (1, 0), (1, True), (1, 3)]:
            r = copy.deepcopy(self.report)
            r['observations'][variant]['measured']['logging_warning_calls'] = count
            with self.assertRaises(RuntimeError):
                P.verify(r, self.library)
        r = copy.deepcopy(self.report)
        r['observations'][1]['measured']['logging_other_calls'] = 1
        with self.assertRaises(RuntimeError):
            P.verify(r, self.library)
if __name__ == '__main__':
    unittest.main()
