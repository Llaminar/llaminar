#!/usr/bin/env python3
"""Device-free bypass controls for CPU preparation and workspace source policy."""
from __future__ import annotations
import unittest
from pathlib import Path
import re
import tempfile
from check_cpu_prepared_allocation_policy import scan_source, executable_source, scan

PATH = "src/v2/execution/example.cpp"
OWNED = "CPUWeightStoragePlacement::accounted(memory,device,owner,CPUWeightStoragePlacement::local())"

def call(api: str, placement: str | None) -> str:
    """Return the exact arity of one supported preparation seam."""
    args = {"prepareGemmHandleLocal": "tensor,device,Kind::AUTO,precision",
            "prepareExpertGemmLocal": "tensor,device,Kind::AUTO",
            "createGemmSlicedLocal": "tensor,0,32"}[api]
    return f"KernelFactory::{api}({args}{','+placement if placement else ''});"

class CPUPreparedAllocationPolicyTests(unittest.TestCase):
    """Every injected allocation must pass its own explicit ownership check."""
    def test_inline_and_immutable_aliases_are_accounted(self):
        for api in ("prepareGemmHandleLocal","prepareExpertGemmLocal","createGemmSlicedLocal"):
            with self.subTest(api=api):
                self.assertEqual(scan_source(PATH,"void owner(){"+call(api,OWNED)+"}"),[])
                source="void owner(){const auto p="+OWNED+";const auto alias=p;"+call(api,"alias")+"}"
                self.assertEqual(scan_source(PATH,source),[])

    def test_omitted_and_unaccounted_placement_fail_each_seam(self):
        for api in ("prepareGemmHandleLocal","prepareExpertGemmLocal","createGemmSlicedLocal"):
            for placement in (None,"CPUWeightStoragePlacement::local()","CPUWeightStoragePlacement::onNode(1)"):
                with self.subTest(api=api,placement=placement):
                    self.assertTrue(scan_source(PATH,"void owner(){"+call(api,placement)+"}"))

    def test_fallback_beside_accounted_call_is_not_hidden(self):
        for api in ("prepareGemmHandleLocal","prepareExpertGemmLocal","createGemmSlicedLocal"):
            source="void owner(){if(ready){"+call(api,OWNED)+"}else{"+call(api,"CPUWeightStoragePlacement::local()")+"}}"
            failures=scan_source(PATH,source)
            self.assertEqual(len(failures),1)
            self.assertIn("unaccounted",failures[0].message)

    def test_conditional_constructor_and_mutated_alias_fail(self):
        for expression in ("ready?"+OWNED+":CPUWeightStoragePlacement::local()",
                           "CPUWeightStoragePlacement::local()"):
            source="void owner(){const auto p="+expression+";"+call("prepareExpertGemmLocal","p")+"}"
            self.assertTrue(scan_source(PATH,source))
        source="void owner(){auto p="+OWNED+";if(fallback)p=CPUWeightStoragePlacement::local();"+call("prepareExpertGemmLocal","p")+"}"
        self.assertTrue(scan_source(PATH,source))

    def test_unadmitted_allocator_references_fail(self):
        for expression in ("placement.allocateUnadmitted<float>(32);",
                           "auto bypass=&CPUStoragePlacement::allocateUnadmitted<float>;"):
            self.assertTrue(scan_source(PATH,"void owner(){"+expression+"}"))
        # Even the infrastructure file cannot acquire a new executable caller.
        source="void owner(){placement.allocateUnadmitted<float>(32);}"
        self.assertTrue(scan_source("src/v2/memory/CPUStorageRoot.h",source))

    def test_legacy_wrappers_fail_in_each_production_role(self):
        for directory in ("execution","loaders","planning","models"):
            for expression in ("tensor->createGemm();","KernelFactory::createGemm(tensor);",
                               "auto bypass=&KernelFactory::createGemm;"):
                self.assertTrue(scan_source(f"src/v2/{directory}/example.cpp","void owner(){"+expression+"}"))

    def test_optional_initial_context_cannot_receive_unaccounted_placement(self):
        source="void owner(){ctx.initial_cpu_storage_placement="+OWNED+";}"
        self.assertEqual(scan_source(PATH,source),[])
        source="void owner(){ctx.initial_cpu_storage_placement=CPUWeightStoragePlacement::local();}"
        self.assertTrue(scan_source(PATH,source))

    def test_comments_regular_raw_and_character_literals_are_ignored(self):
        source=r'''void owner(){
            // prepareExpertGemmLocal(tensor,device);
            /* placement.allocateUnadmitted<float>(32); */
            auto ordinary="KernelFactory::prepareGemmHandleLocal(tensor,device)";
            auto raw=R"tag(quote " createGemm(); allocateUnadmitted(); )tag";
            auto character='\'';
        }'''
        self.assertEqual(scan_source(PATH,source),[])
        self.assertNotIn("createGemm",executable_source(source))

    def test_digit_separator_does_not_hide_later_code(self):
        source="void owner(){auto rows=1'024;"+call("prepareExpertGemmLocal",None)+"}"
        self.assertEqual(len(scan_source(PATH,source)),1)

    def test_mutable_store_normalization_is_exact_and_call_local(self):
        path="src/v2/loaders/PreparedWeightStore.cpp"
        prefix='''void PreparedWeightStore::prepareGemm(){
            auto placement=CPUWeightStoragePlacement::local();
            if(device.is_cpu()){
                if(const auto authority=physicalMemoryAuthority())
                    placement=CPUWeightStoragePlacement::accounted(authority,device,owner,placement);
                else throw std::logic_error("missing");
            }
        '''
        self.assertEqual(scan_source(path,prefix+call("prepareGemmHandleLocal","placement")+"}"),[])
        self.assertTrue(scan_source(path,prefix+call("prepareGemmHandleLocal","CPUWeightStoragePlacement::local()")+"}"))
        source=prefix+"if(fallback)placement=CPUWeightStoragePlacement::local();"+call("prepareGemmHandleLocal","placement")+"}"
        self.assertTrue(scan_source(path,source))
        # The exact same normalization is not permission for another function.
        source=(prefix+call("prepareGemmHandleLocal","placement")+"}").replace("PreparedWeightStore::prepareGemm","other")
        self.assertTrue(scan_source(path,source))

    def test_factory_forwarding_is_bounded_and_cannot_rebind_placement(self):
        path="src/v2/kernels/KernelFactory.cpp"
        source="void KernelFactory::prepareExpertGemmLocal(CPUWeightStoragePlacement placement){"+call("prepareExpertGemmLocal","placement")+"}"
        self.assertEqual(scan_source(path,source),[])
        duplicate=source.replace("}",call("prepareExpertGemmLocal","placement")+"}")
        self.assertTrue(scan_source(path,duplicate))
        rebound=source.replace("{","{placement=CPUWeightStoragePlacement::local();",1)
        self.assertTrue(scan_source(path,rebound))

    def test_real_store_call_mutation_cannot_hide_behind_normalization(self):
        root=Path(__file__).resolve().parents[4]
        path="src/v2/loaders/PreparedWeightStore.cpp"
        source=(root/path).read_text(encoding="utf-8")
        self.assertEqual(scan_source(path,source),[])
        needle="binding.floating_operand_precision, placement"
        self.assertEqual(source.count(needle),1)
        mutated=source.replace(needle,"binding.floating_operand_precision, CPUWeightStoragePlacement::local()")
        self.assertTrue(scan_source(path,mutated))

    def test_real_service_valid_call_does_not_hide_new_fallback(self):
        root=Path(__file__).resolve().parents[4]
        path="src/v2/execution/moe/MoEExpertWeightService.cpp"
        source=(root/path).read_text(encoding="utf-8")
        self.assertEqual(scan_source(path,source),[])
        pattern=r"gate_engine\s*=\s*KernelFactory::prepareExpertGemmLocal\([^;]+;"
        match=re.search(pattern,source)
        self.assertIsNotNone(match)
        fallback=("if(ctx.prepared_store){"+match.group()+"}else{gate_engine="
            "KernelFactory::prepareExpertGemmLocal(ctx.expert_gate_views[e],ctx.device_id);}")
        mutated=source[:match.start()]+fallback+source[match.end():]
        self.assertTrue(scan_source(path,mutated))

    def test_real_factory_forwarding_exception_cannot_be_rebound(self):
        root=Path(__file__).resolve().parents[4]
        path="src/v2/kernels/KernelFactory.cpp"
        source=(root/path).read_text(encoding="utf-8")
        self.assertEqual(scan_source(path,source),[])
        needle="tensor.get(), target_device, resolved_kind, placement"
        self.assertEqual(source.count(needle),1)
        mutated=source.replace(needle,"tensor.get(), target_device, resolved_kind, CPUWeightStoragePlacement::local()")
        self.assertTrue(scan_source(path,mutated))

    def test_exact_unadmitted_facade_cannot_gain_another_branch(self):
        root=Path(__file__).resolve().parents[4]
        path="src/v2/memory/CPUWeightStoragePlacement.h"
        source=(root/path).read_text(encoding="utf-8")
        self.assertEqual(scan_source(path,source),[])
        needle="return placement_.allocateUnadmitted<T>(count, CPUStorageInitialization::Uninitialized);"
        self.assertEqual(source.count(needle),1)
        mutated=source.replace(needle,"if(authority_) { placement_.allocateUnadmitted<T>(count); }"+needle)
        self.assertTrue(scan_source(path,mutated))

    def test_indirect_preparation_reference_fails_closed(self):
        source="void owner(){auto bypass=&KernelFactory::prepareExpertGemmLocal;}"
        self.assertTrue(scan_source(PATH,source))

    def test_real_initial_override_setter_rejects_unaccounted_mutation(self):
        root=Path(__file__).resolve().parents[4]
        path="src/v2/loaders/WeightManager.cpp"
        source=(root/path).read_text(encoding="utf-8")
        self.assertEqual(scan_source(path,source),[])
        pattern=r"ctx\.initial_cpu_storage_placement\s*=\s*CPUWeightStoragePlacement::accounted\([^;]+;"
        match=re.search(pattern,source)
        self.assertIsNotNone(match)
        mutated=(source[:match.start()]+"ctx.initial_cpu_storage_placement=CPUWeightStoragePlacement::local();"+
                 source[match.end():])
        self.assertTrue(scan_source(path,mutated))

    def test_direct_unaccounted_weight_policy_allocation_fails(self):
        for expression in ("CPUWeightStoragePlacement::local().allocate<float>(32);",
            "const auto p=CPUWeightStoragePlacement::onNode(0);p.allocate<float>(32);",
            "const auto p=CPUWeightStoragePlacement::local();const auto alias=p;alias.allocate<float>(32);"):
            self.assertTrue(scan_source(PATH,"void owner(){"+expression+"}"))
        source="void owner(){const auto p="+OWNED+";p.allocate<float>(32);}"
        self.assertEqual(scan_source(PATH,source),[])

    def test_real_store_normalization_cannot_be_conditionally_disabled(self):
        root=Path(__file__).resolve().parents[4]
        path="src/v2/loaders/PreparedWeightStore.cpp"
        source=(root/path).read_text(encoding="utf-8")
        anchor=source.index("auto placement = CPUWeightStoragePlacement::local();")
        for needle,replacement in (
            ("if (const auto authority = physicalMemoryAuthority())",
             "if (false) if (const auto authority = physicalMemoryAuthority())"),
            ("if (const auto authority = physicalMemoryAuthority())","if (false)"),
            ("if (device.is_cpu())","if (false)")):
            with self.subTest(needle=needle,replacement=replacement):
                index=source.index(needle,anchor)
                mutated=source[:index]+replacement+source[index+len(needle):]
                self.assertTrue(scan_source(path,mutated))

    def test_real_service_normalization_requires_its_exact_conditions(self):
        root=Path(__file__).resolve().parents[4]
        path="src/v2/execution/moe/MoEExpertWeightService.cpp"
        source=(root/path).read_text(encoding="utf-8")
        needle="if (!ctx.initial_cpu_storage_placement && ctx.prepared_store)"
        self.assertEqual(source.count(needle),1)
        for replacement in ("if (false)","if (false) "+needle,
                            "if (ctx.prepared_store)"):
            self.assertTrue(scan_source(path,source.replace(needle,replacement)))

    def test_real_initial_override_rejects_emplace_and_unsupported_aliases(self):
        root=Path(__file__).resolve().parents[4]
        path="src/v2/loaders/WeightManager.cpp"
        source=(root/path).read_text(encoding="utf-8")
        match=re.search(r"ctx\.initial_cpu_storage_placement\s*=\s*CPUWeightStoragePlacement::accounted\([^;]+;",source)
        self.assertIsNotNone(match)
        for replacement in (
            "ctx.initial_cpu_storage_placement.emplace(CPUWeightStoragePlacement::local());",
            "ctx.initial_cpu_storage_placement.emplace(CPUWeightStoragePlacement::onNode(0));",
            "ctx.initial_cpu_storage_placement.reset();",
            "auto& alias=ctx.initial_cpu_storage_placement;alias=CPUWeightStoragePlacement::local();"):
            with self.subTest(replacement=replacement):
                mutated=source[:match.start()]+replacement+source[match.end():]
                self.assertTrue(scan_source(path,mutated))

    def test_unbound_manager_construction_forms_fail_in_production_roles(self):
        forms=("DeviceWorkspaceManager workspace(device,bytes);",
               "DeviceWorkspaceManager workspace{device,bytes};",
               "auto workspace=DeviceWorkspaceManager(device,bytes);",
               "auto workspace=DeviceWorkspaceManager{device,bytes};",
               "auto workspace=std::make_unique<DeviceWorkspaceManager>(device,bytes);",
               "auto workspace=std::make_shared<llaminar2::DeviceWorkspaceManager>(device,bytes);",
               "auto workspace=new DeviceWorkspaceManager(device,bytes);",
               "DeviceWorkspaceManager good(device,bytes,memory),bad(device,bytes);")
        for role in ("execution","loaders","planning","models"):
            for form in forms:
                with self.subTest(role=role,form=form):
                    self.assertTrue(scan_source(f"src/v2/{role}/example.cpp","void owner(){"+form+"}"))

    def test_explicit_manager_and_allocator_authority_forms_remain_valid(self):
        forms=("DeviceWorkspaceManager workspace(device,bytes,memory);",
               "DeviceWorkspaceManager workspace{device,bytes,memory,owner};",
               "auto workspace=std::make_shared<DeviceWorkspaceManager>(device,bytes,memory);",
               "auto workspace=std::make_unique<llaminar2::DeviceWorkspaceManager>(device,bytes,memory,owner);",
               "WorkspaceAllocator workspace(memory);",
               "auto workspace=std::make_shared<WorkspaceAllocator>(memory);")
        for form in forms:
            self.assertEqual(scan_source(PATH,"void owner(){"+form+"}"),[])

    def test_default_allocator_construction_forms_fail(self):
        for form in ("WorkspaceAllocator workspace;","WorkspaceAllocator workspace{};",
                     "auto workspace=WorkspaceAllocator();","auto workspace=WorkspaceAllocator{};","WorkspaceAllocator workspace={};",
                     "auto workspace=std::make_shared<WorkspaceAllocator>();",
                     "auto workspace=std::make_unique<WorkspaceAllocator>();",
                     "auto workspace=new WorkspaceAllocator;",
                     "auto workspace=new ::llaminar2::WorkspaceAllocator;",
                     "WorkspaceAllocator good(memory),bad;"):
            with self.subTest(form=form):
                self.assertTrue(scan_source(PATH,"void owner(){"+form+"}"))

    def test_actual_unbound_manager_fixture_boundary_is_exact(self):
        root=Path(__file__).resolve().parents[4]
        path="src/v2/execution/local_execution/device/WorkspaceAllocator.cpp"
        source=(root/path).read_text(encoding="utf-8")
        start=source.index("WorkspaceAllocator::createDeviceWorkspaceManager(")
        end=source.index("\n    }",start)+6
        excerpt="std::unique_ptr<DeviceWorkspaceManager> "+source[start:end]
        self.assertEqual(scan_source(path,excerpt),[])
        self.assertTrue(scan_source(path,excerpt.replace("if (physical_memory_authority_)","if (false)")))
        self.assertTrue(scan_source(path,excerpt.replace("return std::make_unique<DeviceWorkspaceManager>(",
            "if (false) return std::make_unique<DeviceWorkspaceManager>(",1)))
        changed=excerpt[:excerpt.rfind("}")]+"std::make_shared<DeviceWorkspaceManager>(device,budget_bytes);}"
        self.assertTrue(scan_source(path,changed))

    def test_actual_default_allocator_selectors_are_named_exact_and_count_bound(self):
        root=Path(__file__).resolve().parents[4]
        cases=(
            ("src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp",
             "DeviceGraphOrchestrator::ensureDeviceWorkspaceAllocated",r"workspace_allocator_\s*=\s*physical_memory_authority_\s*\?[^;]+;",
             "physical_memory_authority_"),
            ("src/v2/execution/local_execution/device/ReusableExecutionWorkspace.h",
             "acquireImpl",r"slot\.allocator\s*=\s*physical_memory_authority\s*\?[^;]+;",
             "physical_memory_authority"))
        for path,caller,pattern,condition in cases:
            # Read the real selector, but tokenize only its authentic named
            # owner excerpt; repeatedly parsing the giant DGO is unnecessary.
            source=(root/path).read_text(encoding="utf-8")
            match=re.search(pattern,source)
            self.assertIsNotNone(match)
            excerpt="void "+caller+"(){"+match.group()+"}"
            with self.subTest(path=path):
                self.assertEqual(scan_source(path,excerpt),[])
                self.assertTrue(scan_source(path,excerpt.replace(condition,"false",1)))
                self.assertTrue(scan_source(path,excerpt[:-1]+"std::make_shared<WorkspaceAllocator>();}"))
                self.assertTrue(scan_source(path,excerpt.replace(caller,"unreviewedOwner")))

    def test_workspace_authority_call_cannot_hide_unbound_fallback(self):
        source=("void owner(){if(ready)std::make_unique<DeviceWorkspaceManager>(device,bytes,memory);"
                "else std::make_unique<DeviceWorkspaceManager>(device,bytes);}")
        self.assertEqual(len(scan_source(PATH,source)),1)
        source="void owner(){std::make_unique<DeviceWorkspaceManager>(device,budget<A,B>());}"
        self.assertTrue(scan_source(PATH,source))

    def test_missing_empty_and_comment_only_source_trees_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.assertTrue(scan(root))
            source=root/"src/v2"
            source.mkdir(parents=True)
            self.assertTrue(scan(root))
            unit=source/"empty.cpp"
            unit.write_text("// no executable source\n",encoding="utf-8")
            self.assertTrue(scan(root))
            unit.write_text("void ordinary() {}",encoding="utf-8")
            self.assertEqual(scan(root),[])

    def test_full_scan_prefilter_cannot_hide_workspace_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=root/PATH
            source.parent.mkdir(parents=True)
            for form in ("auto bad=new WorkspaceAllocator;",
                         "auto bad=new :: llaminar2 :: WorkspaceAllocator;",
                         "auto bad=WorkspaceAllocator{};",
                         "WorkspaceAllocator bad;",
                         "DeviceWorkspaceManager bad(device,bytes);",
                         "auto bad=std::make_shared<DeviceWorkspaceManager>(device,bytes);",
                         "auto bad=std::make_unique<WorkspaceAllocator>();"):
                with self.subTest(form=form):
                    source.write_text("void owner(){"+form+"}",encoding="utf-8")
                    self.assertTrue(scan(root))

    def test_workspace_type_declarations_and_borrows_are_not_allocations(self):
        source="""class WorkspaceAllocator;
            class DeviceWorkspaceManager;
            using Alias=WorkspaceAllocator;
            WorkspaceAllocator* allocator_pointer;
            WorkspaceAllocator& borrowed_allocator=existing;
            DeviceWorkspaceManager* manager_pointer;
            DeviceWorkspaceManager& borrowed_manager=manager;
            std::shared_ptr<WorkspaceAllocator> retained;
            WorkspaceAllocator factory();
            void consume(WorkspaceAllocator&,DeviceWorkspaceManager*);
            void owner(){DeviceWorkspaceManager valid(device,bytes,memory);}
        """
        self.assertEqual(scan_source(PATH,source),[])

    def test_named_workspace_arrays_require_individual_accounted_owners(self):
        for name in ("WorkspaceAllocator","DeviceWorkspaceManager"):
            forms=(f"{name} owners[2];",
                   f"{name} owners[]{{{{device,bytes,memory}},{{device,bytes,memory}}}};",
                   f"auto owners=new ::llaminar2::{name}[2];",
                   f"auto owners=std::make_unique<llaminar2::{name}[]>(2);",
                   f"auto owners=std::make_shared<{name}[]>(2);",
                   f"auto owners=std::make_shared<{name}[2]>();")
            for form in forms:
                with self.subTest(form=form):
                    findings=scan_source(PATH,"void owner(){"+form+"}")
                    self.assertTrue(findings)
                    self.assertTrue(any("arrays are unsupported" in finding.message for finding in findings))

    def test_full_scan_prefilter_covers_named_arrays_and_comment_separators(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=root/PATH
            source.parent.mkdir(parents=True)
            for name in ("WorkspaceAllocator","DeviceWorkspaceManager"):
                forms=(f"{name} /* named owner */ owners[2];",
                       f"auto owners=new /* owner */ ::llaminar2::{name}[2];",
                       f"auto owners=std::make_unique /* owner */ <{name}[]>(2);",
                       f"auto owners=std::make_shared<{name}[]>(2);")
                for form in forms:
                    with self.subTest(form=form):
                        source.write_text("void owner(){"+form+"}",encoding="utf-8")
                        findings=scan(root)
                        self.assertTrue(any("arrays are unsupported" in finding.message for finding in findings))

    def test_pointer_arrays_and_array_owner_type_mentions_remain_borrows(self):
        source="""class WorkspaceAllocator;
            class DeviceWorkspaceManager;
            WorkspaceAllocator* allocator_borrows[2];
            DeviceWorkspaceManager* manager_borrows[2];
            std::unique_ptr<WorkspaceAllocator[]> retained_allocators;
            std::shared_ptr<DeviceWorkspaceManager[]> retained_managers;
        """
        self.assertEqual(scan_source(PATH,source),[])

if __name__ == "__main__": unittest.main()
