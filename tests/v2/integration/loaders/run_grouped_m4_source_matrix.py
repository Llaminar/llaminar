"""Plan counted verifier source windows and summarize matched captured event timings."""
import argparse
import csv
import os
from fnmatch import fnmatchcase
import hashlib
import json
from pathlib import Path
import re
from statistics import median
import subprocess

NATIVE_SUITES = {4: "OriginalSources/GroupedM4Source", 6: "OriginalSources/GroupedM6Family"}
PREFIX = ("M4_SOURCE_INPUT", "M4_SOURCE_CAPTURE", "M4_SOURCE_SAMPLE",
          "M4_SOURCE_PROGRAM", "M4_SOURCE_DESCRIPTORS")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cases(listing, rows):
    declared = {name:set() for name in ("Static","Deferred","DISABLED_Events")}
    current = False
    for line in listing.splitlines():
        if not line.strip():
            continue
        if not line.startswith(" "):
            current = line.strip() == NATIVE_SUITES[rows] + "."
            continue
        if not current:
            continue
        identity = line.strip().split(maxsplit=1)[0]
        method,sep,name = identity.partition("/")
        if sep and method in declared:
            declared[method].add(name)
    all_cases = declared["Static"]
    require(all_cases and all(covered==all_cases for covered in declared.values()),
            "Compile the complete Static, Deferred and Events fixture inventory")
    if rows == 4:
        require({name for name in all_cases if name.endswith("_source_boundary")} == {"Q8_1_source_boundary"},
                "The original tensor source admission boundary changed")
    else:
        # The four-row fixture owns the source declarations. Reject an incomplete
        # six-row listing even if its three observation modes all omit the same family.
        expected = {name for name in cases(listing, 4) if re.fullmatch(r"G\d{2}_.*", name)
                    and not name.endswith("_wide_direct_tiles")}
        require(expected and all_cases == expected, "Six-row family coverage differs from the source catalog")
    return tuple(sorted(all_cases))


def parse(log, tag):
    return [dict(pair.split("=",1) for pair in line.split(",")[1:])
            for line in log.splitlines() if line.startswith(tag + ",")]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def command(test, fixture, launch):
    args = [str(fixture), f"--gtest_filter={test}"]
    if ".DISABLED_" in test:
        args += ["--gtest_also_run_disabled_tests"]
    return [*launch, *args]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "run"))
    parser.add_argument("--rows", type=int, choices=(4, 6), default=4,
                        help="Counted physical verifier rows for source families or an original GGUF shard")
    parser.add_argument("--case", action="append")
    parser.add_argument("--case-glob", action="append", help="Select from the compiled case names")
    parser.add_argument("--original-window", action="store_true",
                        help="Use one explicit original QFN source TP shard from the fixture environment")
    parser.add_argument("--stage", action="append", choices=("static", "deferred", "events"),
                        help="Run chosen observations. Defaults to all three.")
    parser.add_argument("--arm", choices=("both", "control", "candidate"), default="both",
                        help="Use one arm for byte qualification without event economics.")
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--listing-launcher",type=Path,
        help="Device-free launcher taking the fixture path and GoogleTest listing arguments")
    parser.add_argument("--listing",type=Path,help="Recorded --gtest_list_tests output from the same fixture")
    parser.add_argument("--core", type=Path)
    parser.add_argument("--launcher", nargs="+", default=[])
    parser.add_argument("--selector", help="Exact independent production environment flag")
    parser.add_argument("--symbol", help="Positive selected candidate native marker")
    parser.add_argument("--program", choices=("all_formats","fixed"), default="all_formats",
                        help="Expected measured graph program. Fixed remains selected in both switch arms.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.fixture.is_file(), "Supply the compiled native test fixture")
    list_prefix = [str(args.listing_launcher.resolve())] if args.listing_launcher else []
    listing = (args.listing.read_text() if args.listing else subprocess.check_output(
        [*list_prefix,str(args.fixture.resolve()),"--gtest_list_tests","--gtest_filter=" + ":".join(suite + ".*" for suite in NATIVE_SUITES.values())],
        text=True,stderr=subprocess.STDOUT))
    available = cases(listing, args.rows)
    explicit = list(args.case or [])
    for pattern in args.case_glob or []:
        matches = [case for case in available if fnmatchcase(case, pattern)]
        require(matches, f"Compiled source case pattern matched nothing: {pattern}")
        explicit.extend(matches)
    original_source = {}
    if args.original_window:
        require(not explicit, "Compare one original QFN source window per process configuration")
        keys = ("LLAMINAR_TEST_CASTAGNA_GGUF", "LLAMINAR_TEST_QFN_SOURCE_LAYER",
                "LLAMINAR_TEST_QFN_SOURCE_PARTICIPANT", "LLAMINAR_TEST_QFN_SOURCE_ROCM_ORDINALS")
        original_source = {key: os.getenv(key,"") for key in keys}
        require(all(original_source.values()), "Set the exact original GGUF and TP shard coordinates")
        original_source["LLAMINAR_TEST_QFN_WINDOW_ROWS"] = str(args.rows)
        original_source["LLAMINAR_GPU_EXPERT_BALANCE_NATIVE_TILES"] = os.getenv(
            "LLAMINAR_GPU_EXPERT_BALANCE_NATIVE_TILES","0")
    selected = ("OriginalQFN",) if original_source else tuple(explicit or available)
    require(len(selected)==len(set(selected)) and (original_source or set(selected)<=set(available)),
            "Invalid or repeated source case")
    if args.phase=="run":
        require(args.listing is None, "Native runs enumerate their own compiled fixture")
        require((explicit or original_source) and args.launcher and args.core and args.selector and args.symbol,
                "Run requires explicit cases, assigned isolation launcher, same-ELF core and selected production symbol")
    schedule = []
    method_names = {"static":"Static", "deferred":"Deferred", "events":"DISABLED_Events"}
    methods = tuple(method_names[stage] for stage in dict.fromkeys(args.stage or method_names))
    require(args.arm=="both" or "DISABLED_Events" not in methods,
            "Event economy requires matched control and candidate arms")
    require("Q8_1_source_boundary" not in selected or "Static" in methods,
            "Q8_1 GGUF admission is a static negative only")
    for method in methods:
        ordinary = tuple(c for c in selected if c!="Q8_1_source_boundary")
        groups = [(ordinary, ("0","1","1","0") if method=="DISABLED_Events" else ("0","1"))]
        if method=="Static":
            groups.append((tuple(c for c in selected if c=="Q8_1_source_boundary"), ("0",)))
        for scope,switches in groups:
            if not scope:
                continue
            all_formats = scope!=("Q8_1_source_boundary",) and args.selector
            selected_arms = switches if all_formats else ("0",)
            if all_formats and args.arm != "both":
                selected_arms = ("1" if args.arm=="candidate" else "0",)
            for slot,switch in enumerate(selected_arms):
                expected_presence = "1" if all_formats and args.program=="fixed" else switch
                filters = ":".join(f"{NATIVE_SUITES[args.rows]}.{method}/{case}" for case in scope)
                launch_env = {"LLAMINAR_MOE_ROUTER_ARITHMETIC": "fp16"}
                if original_source:
                    filters = "GroupedM4SourceWindow.DISABLED_OriginalQFN"
                    launch_env |= original_source | {
                        "LLAMINAR_TEST_GROUPED_M4_DEMAND":"deferred" if method=="Deferred" else "static",
                        "LLAMINAR_TEST_QFN_WINDOW_EVENTS":"1" if method=="DISABLED_Events" else "0"}
                if all_formats:
                    launch_env |= {"LLAMINAR_QFN_ROCM_GROUPED_FFN_FRONTIER": "1",
                                   "LLAMINAR_QFN_ROCM_GROUPED_FFN_DIRECT_BANK": "1",
                                   "LLAMINAR_QFN_ROCM_GROUPED_FFN_EQUAL_FRONTIER": "1" if args.program=="fixed" else "0",
                                   "LLAMINAR_QFN_ROCM_GROUPED_FFN_AFFINE_SUMS": "0",
                                   "LLAMINAR_PERF_STATS_SUMMARY": "1",
                                   "LLAMINAR_PERF_STATS_FILTER": "kernel.rocm_grouped_ffn_frontier_calls"}
                    launch_env[args.selector] = switch
                    launch_env["LLAMINAR_TEST_GROUPED_M4_REQUIRED_SYMBOL"] = args.symbol or ""
                    launch_env["LLAMINAR_TEST_GROUPED_M4_SYMBOL_PRESENT"] = expected_presence
                    launch_env["LLAMINAR_TEST_GROUPED_M4_PROGRAM"] = args.program
                schedule.append(dict(cases=scope, method=method, slot=slot, switch=switch,
                                     expected_presence=expected_presence, env=launch_env,
                                     command=command(filters,args.fixture,args.launcher)))
    args.output.mkdir()
    (args.output/"compiled-cases.txt").write_text(listing)
    (args.output/"plan.json").write_text(json.dumps(schedule,indent=2)+"\n")
    if args.phase=="plan":
        return
    identities = {path.resolve(): sha(path) for path in (args.fixture,args.core)}
    observations = {}
    for index, observation in enumerate(schedule):
        sink = args.output/f"{index:02d}-{observation['method']}-{observation['slot']}.log"
        with sink.open("x") as log:
            subprocess.run(observation["command"], env=os.environ | observation["env"], stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        text = sink.read_text()
        count = len(observation["cases"])
        verdict = "test" if count==1 else "tests"
        require(f"[  PASSED  ] {count} {verdict}." in text and "[  SKIPPED ]" not in text, f"No real replay in {sink}")
        require(all(sha(path)==expected and f"{path}: OK" in text for path,expected in identities.items()) and
                f"libllaminar2_core.so => {args.core.resolve()}" in text,
                "Launcher did not authenticate the exact mapped fixture and core ELFs")
        for case in observation["cases"]:
            if case=="Q8_1_source_boundary":
                observations[case] = {"byte_verdict":"source_rejected"}
                continue
            def records(tag):
                return [row for row in parse(text,tag) if row["case"]==case]
            input_row, = records(PREFIX[0])
            if original_source:
                require(input_row["origin"]=="original_qfn_gguf", "Missing original model weight preparation")
                input_row |= {"layer":original_source["LLAMINAR_TEST_QFN_SOURCE_LAYER"],
                              "source_participant":original_source["LLAMINAR_TEST_QFN_SOURCE_PARTICIPANT"]}
            witness, = records(PREFIX[1])
            descriptors, = records(PREFIX[4])
            require(witness["m"]==str(args.rows) and witness["selected"]==observation["expected_presence"],f"Wrong capture program in {sink}")
            programs = records(PREFIX[3])
            require(len(programs)==int(observation["expected_presence"]) and
                    all(p["program"]==args.program for p in programs),f"Wrong grouped program count in {sink}")
            input_row["rows"] = args.rows
            row = observations.setdefault(case, dict(input=input_row, descriptors=descriptors, windows=[]))
            require(row["input"]==input_row and row["descriptors"]==descriptors,
                    f"Source columns or execution format changed across arms for {case}")
            samples = records(PREFIX[2])
            if observation["method"]=="DISABLED_Events":
                require(len(samples)==12 and all(sample["operations"]=="32" for sample in samples),
                        f"Invalid native event observations: {sink}")
            row["windows"].append(dict(stage_mode=observation["method"], switch=observation["switch"],
                byte_verdict="pass", capture=witness, program=programs, samples=samples, log=str(sink)))
    csv_rows = []
    for case,row in observations.items():
        if "windows" not in row:
            csv_rows.append(dict(case=case,byte_verdict=row["byte_verdict"]))
            continue
        by_arm = {arm:[median(float(sample["us_per_op"]) for sample in w["samples"]) for w in row["windows"]
                      if w["switch"]==arm and w["samples"]] for arm in ("0","1")}
        row["economy"] = {}
        if by_arm["0"] and by_arm["1"]:
            control,treatment=(median(by_arm[arm]) for arm in ("0","1"))
            row["economy"] = dict(control_us=control,treatment_us=treatment,treatment_over_control=treatment/control)
        for window in row["windows"]:
            csv_rows.append((row["input"]|{"exec_"+k:v for k,v in row["descriptors"].items() if k!="case"}|
                {k:window[k] for k in ("stage_mode","switch","byte_verdict","log")}|row["economy"]|
                {"samples_us":json.dumps([float(sample["us_per_op"]) for sample in window["samples"]]),
                 "decoder_family":window["program"][0]["decoder_family"] if window["program"] else ""}))
    fields = sorted({k for row in csv_rows for k in row})
    with (args.output/"result.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(csv_rows)
    (args.output/"result.json").write_text(json.dumps(dict(passed=True,files={str(k):v for k,v in identities.items()},
                                                           original_source=original_source,cases=observations),indent=2)+"\n")


if __name__=="__main__":
    main()
