"""Validate native retained-closed event receipts and summarize the paired A/B."""

import argparse
import json
from math import isfinite
from pathlib import Path
import re
from statistics import median


def rows(log, name):
    """Read the named comma-delimited fixture records in emission order."""
    return [
        {key: value for key, _, value in (field.partition("=") for field in line.split(","))}
        for line in re.findall(rf"{name},([^\r\n]+)", log)
    ]


def require(condition, message):
    """Reject partial or mismatched receipts without emitting benchmark metrics."""
    if not condition:
        raise ValueError(message)


def summarize(log):
    """Require the real eligible captures, staged byte replay and balanced samples."""
    clean_log = re.sub(r"\x1b\[[0-9;]*m", "", log)
    require("[  PASSED  ] 1 test." in clean_log, "missing successful native comparison")
    device, = rows(clean_log, "FULL_PRODUCTION_CASTAGNA_DEVICE")
    require(
        device["source"] == "real_gguf" and device["timing"] == "1"
        and device["native_isa"].split(":")[0] == "gfx906" and device["pci"]
        and device["full_production_mode"] == "1",
        "expected one real-weight gfx906 timing owner",
    )
    geometry = {"H": "2560", "F": "256", "S": "192", "E": "512", "topK": "10"}
    require(all(device[key] == value for key, value in geometry.items()), "unexpected participant geometry")
    source, = rows(clean_log, "CASTAGNA_FULL_AB_SOURCE")
    window = {"real_gguf": "1", "generated": "0", "layer": "1", "tp": "4", "participant": "3",
              "routed_first": "384", "routed_count": "256", "shared_first": "448", "shared_count": "192"}
    require(all(source.get(key) == value for key, value in window.items()), "unexpected GGUF weight slice")
    scopes = rows(clean_log, "CASTAGNA_FULL_AB_SCOPE")
    require([scope["arm"] for scope in scopes] == ["0", "1", "2"], "expected staged, retained and raw arms")
    for scope in scopes[1:]:
        require(
            scope["kernels"] == "1" and scope["real_gguf"] == "1"
            and scope["scope"] == "post+HC+router+localFFN"
            and scope["middle_injection"] == scope["bounded_centered"] == "0"
            and scope["seeded_boundary_input"] == "1"
            and {"no_collective", "no_early"} <= scope.keys(),
            "expected equal local one-node measurement boundaries",
        )
    require(scopes[0]["kernels"] == "24", "expected the original 24-node staged oracle")
    require(scopes[1]["full_production"] == "1" and scopes[2]["adapted_weight_readers"] == "0",
            "expected retained production and unadapted Castagna")
    replay, = rows(clean_log, "FULL_PRODUCTION_HC_REPLAY")
    expected_replay = {"histories": "20", "repeated": "1", "reset": "1", "empty_repeated": "2",
                       "active_after_empty": "1", "public_byte_oracle": "staged"}
    require(all(replay.get(key) == value for key, value in expected_replay.items()),
            "missing complete staged byte replay")
    routes = rows(clean_log, "CASTAGNA_FULL_AB_ROUTES")
    require(
        [int(row["history"]) for row in routes] == list(range(20))
        and all(row["selected_set_intersection"] == row["top_k"] == "10" for row in routes),
        "route work differs or is missing in the 20 histories",
    )
    samples = rows(clean_log, "CASTAGNA_FULL_AB_SAMPLE")
    require(len(samples) == 32 * 4, "expected 32 paired quartets")
    owner, castagna, deltas, orders = [], [], [], []
    for index in range(32):
        quartet = samples[index * 4:(index + 1) * 4]
        require(
            all(row["pair"] == "12" and int(row["round"]) == index and row["operations"] == "32"
                and int(row["slot"]) == slot for slot, row in enumerate(quartet))
            and len({row["record"] for row in quartet}) == 1,
            f"invalid matching in quartet {index}",
        )
        order = "".join(row["arm"] for row in quartet)
        require(order in ("1221", "2112"), f"invalid ABBA/BAAB order in quartet {index}")
        orders.append(order)
        pair = ([float(row["us_per_op"]) for row in quartet if row["arm"] == arm]
                for arm in ("1", "2"))
        owner_times, castagna_times = pair
        require(all(isfinite(value) and value > 0 for value in owner_times + castagna_times),
                f"invalid event measurement in quartet {index}")
        owner.extend(owner_times)
        castagna.extend(castagna_times)
        deltas.append(median(owner_times) - median(castagna_times))
    require(orders.count("1221") == orders.count("2112") == 16, "paired order is unbalanced")
    return {
        "pci": device["pci"],
        "retained_median_us": median(owner),
        "castagna_median_us": median(castagna),
        "paired_retained_minus_castagna_median_us": median(deltas),
        "rounds": len(deltas),
        "operations_per_sample": 32,
    }


def validate_checkpoint(sessions):
    """All three session medians must meet the MI50 target and beat the paired fork."""
    require(len(sessions) == 3, "the MI50 checkpoint requires three fresh CTest sessions")
    require(len({session["pci"] for session in sessions}) == 1, "sessions use different GPUs")
    for index, session in enumerate(sessions, start=1):
        require(session["retained_median_us"] <= 70.0, f"session {index} exceeds 70 microseconds")
        require(session["paired_retained_minus_castagna_median_us"] < 0.0,
                f"session {index} did not beat paired Castagna")


def main():
    """Summarize one native log, or enforce the reference-card milestone across three."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path, help="verbose successful CTest log(s)")
    parser.add_argument("--mi50-checkpoint", action="store_true",
                        help="require three runs on one MI50, each at most 70 us and faster than Castagna")
    args = parser.parse_args()
    try:
        if not args.mi50_checkpoint:
            require(len(args.logs) == 1, "supply one log for a session summary")
            print(json.dumps(summarize(args.logs[0].read_text()), indent=2))
            return
        require(len(args.logs) == 3, "supply a separate log for each of three CTest invocations")
        logs = [path.read_text() for path in args.logs]
        require(len(set(logs)) == 3, "repeated native log in the checkpoint")
        sessions = [summarize(log) for log in logs]
        print(json.dumps({"sessions": sessions}, indent=2))
        validate_checkpoint(sessions)
    except (OSError, KeyError, ValueError) as error:
        parser.exit(1, f"Comparison failed: {error}\n")


if __name__ == "__main__":
    main()
