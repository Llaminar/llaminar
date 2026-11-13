"""Replay malformed/complete diagnostic receipts independently of native hardware."""
import importlib.util
import copy
from pathlib import Path
import unittest
spec = importlib.util.spec_from_file_location('host_driver', Path(__file__).parents[1]/'run_host_service_calibration.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)

class HostServiceProtocol(unittest.TestCase):
    def row(self, phase, lifetime=0):
        return dict(schema='retained-host-service-diagnostic-v1', phase=phase, role='immutable_lookup_io',
                    lifetime=lifetime, identity='a'*64, resource_class='b'*64,
                    creator_source='mapped-pthread-v1', diagnostic_source='f'*64,
                    exact_stack_bytes=4096, virtual_guard_bytes=4096,
                    exact_allocation_bytes=8192, exact_claimed_bytes=0,
                    evidence_role='unqualified_actual_program',
                    whole_process_bytes_complete=False, opaque_limit_installed=False)
    def lifecycle(self):
        rows=[self.row('original_metadata_complete')]
        for lifetime in range(1,22):
            for phase in ('before_exact_claim','original_claim_complete','original_worker_ready',
                          'original_worker_used','exact_join_unmap_complete'):
                row=self.row(phase,lifetime)
                if phase in ('original_claim_complete','original_worker_ready','original_worker_used'):
                    row['exact_claimed_bytes']=8192
                rows.append(row)
        rows.append(self.row('original_process_use_closed',21))
        return rows
    def test_key(self):
        rows=[self.row('original_metadata_complete'),self.row('key_only_complete')]
        driver.validate(rows,'immutable_lookup_io','key')
        with self.assertRaises(ValueError):driver.validate(rows[:1],'immutable_lookup_io','key')
    def test_no_qualification_or_identity_substitution(self):
        for field,value in [('opaque_limit_installed',True),('whole_process_bytes_complete',True),('resource_class','foreign')]:
            rows=[self.row('original_metadata_complete'),self.row('key_only_complete')];rows[-1][field]=value
            with self.assertRaises(ValueError):driver.validate(rows,'immutable_lookup_io','key')
    def population(self,io=8,executor=8,claims=False):
        classes=[]
        if io:classes.append(dict(role='immutable_lookup_io',identity='a'*64,resource_class='b'*64,exact_allocation_bytes=8192,exact_stack_bytes=4096,virtual_guard_bytes=4096))
        if executor:classes.append(dict(role='executor_parent_ticket',identity='c'*64,resource_class='d'*64,exact_allocation_bytes=8192,exact_stack_bytes=4096,virtual_guard_bytes=4096))
        group=['before_exact_claim','original_claims_complete']
        if not claims:group+=['all_original_workers_ready','all_original_tls_live','all_original_tasks_complete']
        group+=['exact_join_unmap_complete']
        phases=['original_metadata_complete']+group*21+['original_process_use_closed']
        live={'original_claims_complete','all_original_workers_ready','all_original_tls_live','all_original_tasks_complete'}
        return [dict(schema='retained-host-service-population-v1',phase=phase,io_workers=io,executor_workers=executor,classes=copy.deepcopy(classes),
            lifetime=0 if index==0 else min(21,1+(index-1)//len(group)),
            creator_source='mapped-pthread-v1',diagnostic_source='f'*64,
            opaque_limit_installed=False,whole_process_bytes_complete=False,actual_consumers_qualified=False,
            work='zero_native_claim_control' if claims else 'all_original_tls_blocks',exact_claimed_bytes=(io+executor)*8192 if phase in live else 0) for index,phase in enumerate(phases)]
    def test_concurrent_population_and_zero_native_control(self):
        driver.validate_population(self.population(),8,8)
        driver.validate_population(self.population(claims=True),8,8,True)
        driver.validate_population(self.population(16,0),16,0)
        driver.validate_population(self.population(0,16),0,16)
    def test_population_missing_live_phase_and_retained_exact_mapping_reject(self):
        rows=self.population();rows.pop(4)
        with self.assertRaises(ValueError):driver.validate_population(rows,8,8)
        rows=self.population();rows[-1]['exact_claimed_bytes']=8192
        with self.assertRaises(ValueError):driver.validate_population(rows,8,8)
    def test_claim_control_cannot_claim_consumers_or_native_work(self):
        rows=self.population(claims=True);rows[1]['work']='all_original_tls_blocks'
        with self.assertRaises(ValueError):driver.validate_population(rows,8,8,True)
        rows=self.population();rows[1]['actual_consumers_qualified']=True
        with self.assertRaises(ValueError):driver.validate_population(rows,8,8)

    def test_single_lifecycle_claim_and_release_contract(self):
        driver.validate(self.lifecycle(),'immutable_lookup_io','diagnostic-lifecycle')
        for phase in ('before_exact_claim','original_claim_complete','exact_join_unmap_complete'):
            with self.subTest(phase=phase):
                rows=self.lifecycle()
                row=next(row for row in rows if row['phase']==phase)
                row['exact_claimed_bytes']=1
                with self.assertRaises(ValueError):driver.validate(rows,'immutable_lookup_io','diagnostic-lifecycle')

    def test_lifetime_relabel_and_missing_lifetimes_reject(self):
        for population in (False,True):
            for invalid in (None,0,2,True):
                with self.subTest(population=population,invalid=invalid):
                    rows=self.population() if population else self.lifecycle()
                    if invalid is None:rows[1].pop('lifetime')
                    else:rows[1]['lifetime']=invalid
                    with self.assertRaises(ValueError):
                        if population:driver.validate_population(rows,8,8)
                        else:driver.validate(rows,'immutable_lookup_io','diagnostic-lifecycle')

    def test_creator_family_and_test_provenance_are_complete_and_immutable(self):
        for population in (False,True):
            for field in ('creator_source','diagnostic_source'):
                for invalid in (None,'','0'*64,'A'*64,'9'*64):
                    with self.subTest(population=population,field=field,invalid=invalid):
                        rows=self.population() if population else self.lifecycle()
                        if invalid is None:rows[1].pop(field)
                        else:rows[1][field]=invalid
                        with self.assertRaises(ValueError):
                            if population:driver.validate_population(rows,8,8)
                            else:driver.validate(rows,'immutable_lookup_io','diagnostic-lifecycle')

    def test_new_build_provenance_does_not_change_unchanged_resource_classes(self):
        for population in (False, True):
            rows = self.population() if population else self.lifecycle()
            for row in rows:
                row['diagnostic_source'] = '9'*64
            if population:
                driver.validate_population(rows, 8, 8)
            else:
                driver.validate(rows, 'immutable_lookup_io', 'diagnostic-lifecycle')

    def test_stack_and_guard_geometry_cannot_change_between_phases(self):
        for population in (False,True):
            for field in ('exact_stack_bytes','virtual_guard_bytes','exact_allocation_bytes'):
                with self.subTest(population=population,field=field):
                    rows=self.population() if population else self.lifecycle()
                    row=rows[1]['classes'][0] if population else rows[1]
                    row[field]+=1
                    with self.assertRaises(ValueError):
                        if population:driver.validate_population(rows,8,8)
                        else:driver.validate(rows,'immutable_lookup_io','diagnostic-lifecycle')

    def test_incomplete_or_impossible_geometry_rejects_even_when_consistent(self):
        for field,invalid in [('identity','placeholder'),('resource_class','0'*64),
                              ('exact_stack_bytes',0),('virtual_guard_bytes',-1),
                              ('exact_allocation_bytes',4096)]:
            with self.subTest(field=field):
                rows=self.population()
                for row in rows:row['classes'][0][field]=invalid
                with self.assertRaises(ValueError):driver.validate_population(rows,8,8)

    def test_unknown_single_mode_and_boolean_population_reject(self):
        with self.assertRaises(ValueError):driver.validate(self.lifecycle(),'immutable_lookup_io','unknown')
        with self.assertRaises(ValueError):driver.validate_population(self.population(1,0),True,0)
if __name__=='__main__':unittest.main()
