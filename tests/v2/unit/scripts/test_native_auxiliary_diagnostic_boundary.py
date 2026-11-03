#!/usr/bin/env python3
"""Authenticate typed raw diagnostic exclusion from ordinary native factories.

Recognize the closed CPU owner's private snapshot constructor as the shared
implementation of installed and diagnostic origins. Its private declaration and
explicit public-factory checks keep recognition from permitting external callers
or turning an ordinary factory into a diagnostic path.
"""
import argparse
from pathlib import Path
import re
import unittest
import shutil
import tempfile


def strip_closed_cpu_diagnostic_tokens(path, text, source):
    """Recognize only exact CPU owner declarations and named private-owner bodies."""
    owner=source / "src/v2/execution/local_execution/graph/RetainedHostServiceProgram"
    spans=[]
    if path==owner.with_suffix(".h"):
        declarations=(
            r"enum class RetainedHostServiceEvidence\s*\{Installed,Recorded,(UnqualifiedDiagnostic),InstalledWireProjection\}",
            r"enum class MetadataAuthority\s*\{Installed,Recorded,(UnqualifiedDiagnostic),InstalledWire\}",
            r"struct (UnqualifiedDiagnostic) final \{\};",
            r"using Original=std::variant<PhysicalMemoryOwnerReservation,(UnqualifiedDiagnostic)>;",
        )
        for declaration in declarations:
            spans += [m.span(1) for m in re.finditer(declaration,text)]
    elif path==owner.with_suffix(".cpp"):
        bodies=[]
        for function in ("RetainedHostServiceProgram::snapshot",
                         "RetainedHostServiceProgram::unqualifiedDiagnostic", "RetainedHostServiceProgram::recordedDiagnostic",
                         "RetainedHostServiceProgram::bom", "RetainedHostServiceProgram::evidenceKind", "RetainedHostServiceUseToken::claim"):
            start=text.find(function+"(")
            if start<0:continue
            opening=text.find("{",start);depth=0
            for end in range(opening,len(text)):
                depth += (text[end]=="{")-(text[end]=="}")
                if depth==0:bodies.append((opening,end+1));break
        qualified=r"(?:RetainedHostServiceProgram::)?MetadataAuthority::(UnqualifiedDiagnostic)|RetainedHostServiceEvidence::(UnqualifiedDiagnostic)|RetainedHostServiceProcessObligation::(UnqualifiedDiagnostic)"
        for match in re.finditer(qualified,text):
            if any(start<=match.start() and match.end()<=end for start,end in bodies):
                spans += [match.span(group) for group in range(1,4) if match.group(group)]
    for start,end in sorted(spans,reverse=True):text=text[:start]+" "*(end-start)+text[end:]
    return text

def strip_closed_remote_diagnostic_tokens(path, text, source):
    """Only the same closed remote owner can select or compare its explicit raw origin."""
    if path != source / "src/v2/backends/RetainedParticipantGraphExecutable.cpp":return text
    spans=[]
    for function in ("RetainedParticipantGraphExecutable::createDiagnosticRemote", "RetainedParticipantGraphExecutable::createBound"):
        start=text.find(function+"(")
        if start<0:continue
        opening=text.find("{",start);depth=0
        for end in range(opening,len(text)):
            depth += (text[end]=="{")-(text[end]=="}")
            if depth==0:
                for match in re.finditer(r"GraphHostQualification::(UnqualifiedDiagnostic)",text[opening:end+1]):
                    spans.append((opening+match.start(1),opening+match.end(1)))
                break
    for start,end in sorted(spans,reverse=True):text=text[:start]+" "*(end-start)+text[end:]
    return text

def validate(source):
    errors = []
    native = source / "src/v2/backends"
    family = (native / "ReadAheadNativeResourceFamily.h").read_text()
    parent = (native / "ReadAheadNativeParentTerminal.h").read_text()
    for name, text in (("family", family), ("parent", parent)):
        if "std::variant<" not in text or "UnqualifiedDiagnostic" not in text:
            errors.append(name + " lacks explicit qualified/diagnostic ownership")
        if "friend struct test::PooledNativeProgramDiagnosticAccess;" not in text:
            errors.append(name + " lacks its named closed diagnostic friend")
        if re.search(r"UnqualifiedDiagnostic\s*\([^;]*\)\s*:\s*", text, re.S):
            errors.append(name + " defines diagnostic construction in a production header")
    for filename, factory, gate in (
        ("ReadAheadNativeResourceFamily.cpp", "ReadAheadNativeResourceFamily::materialize(", "requireInstalledCertificate(device)"),
        ("ReadAheadNativeParentTerminal.cpp", "ReadAheadNativeParentTerminal::materialize(", "requireInstalled(device)"),
        ("RetainedParticipantGraphExecutable.cpp", "RetainedParticipantGraphExecutable::create(\n    const PreparedGPUContextUsePermit& original,void* stream)", "GeneralHostOwnership::OriginalAllowance"),
    ):
        text = (native / filename).read_text()
        start = text.index(factory)
        opening = text.index("{", start)
        end = text.index("\n}", opening)
        if gate not in text[opening:end]:
            errors.append(filename + (" ordinary factory lost original General host ownership" if filename=="RetainedParticipantGraphExecutable.cpp" else " ordinary factory lost installed qualification"))
        if "LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC" in text or "UnqualifiedDiagnostic{}" in text:
            errors.append(filename + " selects a diagnostic state in production")
    cpu_header=source / "src/v2/execution/local_execution/graph/RetainedHostServiceProgram.h"
    if cpu_header.exists():
        cpu=cpu_header.read_text()
        for owner,markers in (("RetainedHostServiceProgram",("enum class MetadataAuthority","snapshot(","unqualifiedDiagnostic(","recordedDiagnostic(")),
                              ("RetainedHostServiceProcessObligation",("struct UnqualifiedDiagnostic", "using Original="))):
            start=cpu.find("class "+owner+" final")
            if start<0:
                errors.append("CPU diagnostic owner lost its exact closed class");continue
            for marker in markers:
                position=cpu.find(marker,start)
                prefix=cpu[start:position] if position>=0 else ""
                if position<0 or prefix.rfind("private:")<=prefix.rfind("public:"):
                    errors.append("CPU diagnostic construction became public or lost its closed owner")
    fixture = (source / "tests/v2/fixtures/backends/PooledNativeProgramDiagnosticAccess.h").read_text()
    if '#ifndef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC' not in fixture or '#error ' not in fixture:
        errors.append("raw native bridge lacks explicit diagnostic-target guard")
    if "ReadAheadNativeResourceCertificate(" in fixture or "materializeCertified(" in fixture or "reserveNewAllocations(" in fixture:
        errors.append("raw native bridge fabricates certification or commitment")
    allowed_headers={native / name for name in (
        "ReadAheadNativeResourceFamily.h","ReadAheadNativeParentTerminal.h","RetainedParticipantGraphExecutable.h")}
    friends=("PooledNativeProgramDiagnosticAccess","RetainedParticipantGraphExecutableFixtureAccess","RetainedParticipantGraphExecutableReplayAccess")
    for path in (source / "src/v2").rglob("*"):
        if not path.is_file() or path.suffix not in (".h",".hpp",".cpp",".cu",".cuh",".hip"):
            continue
        text=path.read_text()
        if "LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC" in text:
            errors.append(str(path)+": diagnostic target macro appears in production")
        if path not in allowed_headers and any(friend in text for friend in friends):
            errors.append(str(path)+": raw diagnostic friend appears outside its original owner declarations")
        if path not in allowed_headers:
            stripped=strip_closed_remote_diagnostic_tokens(path,strip_closed_cpu_diagnostic_tokens(path,text,source),source).replace("NativeAuxiliaryOwnershipKind::UnqualifiedDiagnostic", "")
            if re.search(r"\bUnqualifiedDiagnostic\b",stripped):
                errors.append(str(path)+": raw diagnostic state selected outside its original closed headers")
        if path not in {native / "RetainedParticipantGraphExecutable.h",native / "NativeAuxiliaryResourceCatalog.h"} and "PooledRemoteGraphHostReplayAccess" in text:
            errors.append(str(path)+": remote host replay friend appears outside its original closed owners")
        if path not in {native / "RetainedParticipantGraphExecutable.h",native / "RetainedParticipantGraphExecutable.cpp"} and ("createDeviceBound(" in text or "createDiagnosticRemote(" in text):
            errors.append(str(path)+": raw graph device-bound constructor invoked from production")
    return errors


class DiagnosticBoundary(unittest.TestCase):
    def test_original_typed_boundary(self):
        self.assertEqual(validate(SOURCE), [])

    def test_remote_installed_factory_and_private_raw_origin(self):
        native=SOURCE / "src/v2/backends"
        text=(native / "RetainedParticipantGraphExecutable.cpp").read_text()
        signature="RetainedParticipantGraphExecutable::create(\n    const PreparedGPUContextUsePermit& original,void* stream,const PooledRemoteGraphProgram& program)"
        start=text.index(signature);end=text.index("\n}",start)
        self.assertIn("NativeAuxiliaryResourceCatalog::require(",text[start:end])
        self.assertNotIn("createDiagnosticRemote(",text[start:end])
        header=(native / "RetainedParticipantGraphExecutable.h").read_text()
        position=header.index("createDiagnosticRemote(")
        self.assertGreater(header[:position].rfind("private:"),header[:position].rfind("public:"))

    def test_negative_remote_creator_or_raw_origin_from_unrelated_body(self):
        errors=self.mutation("OtherRemote.cpp","void rogue(){RetainedParticipantGraphExecutable::createDiagnosticRemote(); GraphHostQualification::UnqualifiedDiagnostic;}\n")
        self.assertTrue(any("raw graph device-bound" in e for e in errors),errors)
        self.assertTrue(any("raw diagnostic state" in e for e in errors),errors)

    def test_no_public_constructor_or_runtime_flag(self):
        for name in ("ReadAheadNativeResourceFamily.h", "ReadAheadNativeParentTerminal.h"):
            text = (SOURCE / "src/v2/backends" / name).read_text()
            before = text[:text.index("struct UnqualifiedDiagnostic")]
            self.assertIn("private:", before)
            self.assertNotIn("getenv", text)

    def test_wire_has_no_diagnostic_selection(self):
        wire = SOURCE / "src/v2/planning/ImmutableRowTableAdmissionWire.cpp"
        if wire.exists():
            text = wire.read_text()
            self.assertNotIn("UnqualifiedDiagnostic", text)
            self.assertNotIn("POOLED_NATIVE_PROGRAM_DIAGNOSTIC", text)

    def mutation(self,relative,body):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            backend=root / "src/v2/backends";backend.mkdir(parents=True)
            for name in ("ReadAheadNativeResourceFamily.h","ReadAheadNativeResourceFamily.cpp",
                         "ReadAheadNativeParentTerminal.h","ReadAheadNativeParentTerminal.cpp",
                         "RetainedParticipantGraphExecutable.h","RetainedParticipantGraphExecutable.cpp"):
                shutil.copyfile(SOURCE / "src/v2/backends" / name,backend / name)
            fixture=root / "tests/v2/fixtures/backends";fixture.mkdir(parents=True)
            shutil.copyfile(SOURCE / "tests/v2/fixtures/backends/PooledNativeProgramDiagnosticAccess.h",
                            fixture / "PooledNativeProgramDiagnosticAccess.h")
            path=root / "src/v2/execution" / relative
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(body)
            return validate(root)

    def test_general_original_factory_cannot_drop_host_allowance(self):
        body=(SOURCE / "src/v2/backends/RetainedParticipantGraphExecutable.cpp").read_text()
        body=body.replace("std::nullopt,GeneralHostOwnership::OriginalAllowance", "std::nullopt,GeneralHostOwnership::DeviceOnly")
        errors=self.mutation("../backends/RetainedParticipantGraphExecutable.cpp",body)
        self.assertTrue(any("original General host ownership" in error for error in errors),errors)

    def test_negative_raw_friend_from_unrelated_production_cpp(self):
        errors=self.mutation("Unrelated.cpp","void rogue(){test::PooledNativeProgramDiagnosticAccess::side();}\n")
        self.assertTrue(any("raw diagnostic friend" in error for error in errors),errors)

    def test_remote_diagnostic_friend_cannot_authorize_source_graph_lifecycle(self):
        """Neither ordinary source tokens nor auxiliary branches grant raw replay access."""
        for name in ("IRetainedGraphParticipants.h", "IGraphCaptureAuxiliaryBranch.h"):
            with self.subTest(header=name):
                relative = "local_execution/graph/" + name
                body = (SOURCE / "src/v2/execution" / relative).read_text()
                body += "\nnamespace llaminar2 { namespace test { struct PooledRemoteGraphHostReplayAccess; } }\n"
                errors = self.mutation(relative, body)
                self.assertTrue(any("remote host replay friend" in error for error in errors), errors)

    def test_negative_macro_from_unrelated_production_header(self):
        errors=self.mutation("Unrelated.h","#define LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC 1\n")
        self.assertTrue(any("diagnostic target macro" in error for error in errors),errors)

    def test_negative_raw_graph_or_tag_from_unrelated_production_cpp(self):
        errors=self.mutation("Other.cpp","void rogue(){UnqualifiedDiagnostic raw; RetainedParticipantGraphExecutable::createDeviceBound();}\n")
        self.assertTrue(any("raw diagnostic state" in error for error in errors),errors)
        self.assertTrue(any("raw graph device-bound" in error for error in errors),errors)

    def test_cpu_closed_enum_is_not_a_generic_owner_path_exception(self):
        for relative,body in (
            ("local_execution/graph/RetainedHostServiceProgram.h", "UnqualifiedDiagnostic rogue;\n"),
            ("local_execution/graph/RetainedHostServiceProgram.cpp", "void rogue(){auto state=MetadataAuthority::UnqualifiedDiagnostic;}\n"),
            ("Other.cpp", "auto state=RetainedHostServiceEvidence::UnqualifiedDiagnostic;\n"),
        ):
            errors=self.mutation(relative,body)
            self.assertTrue(any("raw diagnostic state" in e for e in errors),errors)

    def test_cpu_public_original_factory_cannot_select_diagnostic(self):
        body="auto RetainedHostServiceProgram::original(Role role){return MetadataAuthority::UnqualifiedDiagnostic;}\n"
        errors=self.mutation("local_execution/graph/RetainedHostServiceProgram.cpp",body)
        self.assertTrue(any("raw diagnostic state" in e for e in errors),errors)

    def test_cpu_diagnostic_origin_cannot_be_moved_to_public(self):
        owner=SOURCE / "src/v2/execution/local_execution/graph/RetainedHostServiceProgram.h"
        body=owner.read_text().replace("    enum class MetadataAuthority", "public:\n    enum class MetadataAuthority",1)
        errors=self.mutation("local_execution/graph/RetainedHostServiceProgram.h",body)
        self.assertTrue(any("CPU diagnostic construction became public" in e for e in errors),errors)

    def test_cpu_shared_snapshot_constructor_cannot_be_moved_to_public(self):
        """The named implementation exception is valid only behind its private API."""
        owner=SOURCE / "src/v2/execution/local_execution/graph/RetainedHostServiceProgram.h"
        declaration="    static RetainedHostServiceProgram snapshot("
        body=owner.read_text()
        self.assertEqual(body.count(declaration),1)
        body=body.replace(declaration,"public:\n"+declaration,1)
        errors=self.mutation("local_execution/graph/RetainedHostServiceProgram.h",body)
        self.assertTrue(any("CPU diagnostic construction became public" in e for e in errors),errors)

    def test_cpu_exact_closed_declarations_and_private_body_are_recognized(self):
        owner=SOURCE / "src/v2/execution/local_execution/graph/RetainedHostServiceProgram"
        for suffix in (".h",".cpp"):
            text=owner.with_suffix(suffix).read_text()
            stripped=strip_closed_cpu_diagnostic_tokens(owner.with_suffix(suffix),text,SOURCE)
            self.assertNotRegex(stripped,r"\bUnqualifiedDiagnostic\b")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    SOURCE = args.source
    unittest.main(argv=[__file__, *remaining])
