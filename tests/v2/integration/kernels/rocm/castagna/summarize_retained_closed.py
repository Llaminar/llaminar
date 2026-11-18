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


def validate_source_window(device, source, layer, degree, participant, encodings):
    """Authenticate recorded widths, source offsets and independently requested GGUF formats."""
    require(layer >= 0 and 0 <= participant < degree, "invalid requested checkpoint coordinate")
    def window(grain):
        # The original fork rounds the cumulative per-member boundary down.
        start = (640 * participant // degree) // grain * grain
        end = (640 * (participant + 1) // degree) // grain * grain
        return start, end - start

    routed, shared = window(128), window(64)
    geometry = {"H": "2560", "F": str(routed[1]), "S": str(shared[1]), "E": "512", "topK": "10"}
    require(all(device[key] == value for key, value in geometry.items()), "unexpected participant geometry")
    required = {"real_gguf": "1", "generated": "0", "layer": str(layer), "tp": str(degree), "participant": str(participant),
                "routed_first": str(routed[0]), "routed_count": str(routed[1]),
                "shared_first": str(shared[0]), "shared_count": str(shared[1])}
    if encodings is not None:
        required.update(gate=encodings[0], down=encodings[1])
    require(all(source.get(key) == value for key, value in required.items()), "unexpected GGUF weights or slice")


def validate_route_work(replay, timing, all_histories):
    """Prove the fixed timing schedule did equal work; return every divergent replay history."""
    require([int(row["history"]) for row in replay] == list(range(20)),
            "missing or reordered replay histories")
    divergent = []
    for index, row in enumerate(replay):
        count = int(row["selected_set_intersection"])
        require(row["top_k"] == "10" and 0 <= count <= 10, "invalid replay route extent")
        if count != 10:
            divergent.append(index)
    # Five warmups and 32 measured quartets cycle the four fixed inputs.
    require(len(timing) == 37 and all(
        int(row["record"]) == (index + 5) % 4 and row["intersection"] == row["top_k"] == "10"
        and int(row["record"]) not in divergent for index, row in enumerate(timing)),
        "a scheduled timing record has missing or unequal expert work")
    require(not all_histories or not divergent, "replay route work differs in the checkpoint")
    return divergent


def summarize(log, *, layer=1, degree=4, participant=3, encodings=None, all_histories=True):
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
    source, = rows(clean_log, "CASTAGNA_FULL_AB_SOURCE")
    validate_source_window(device, source, layer, degree, participant, encodings)
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
    divergent = validate_route_work(rows(clean_log, "CASTAGNA_FULL_AB_ROUTES"),
                                    rows(clean_log, "CASTAGNA_FULL_AB_TIMING_ROUTES"), all_histories)
    samples = rows(clean_log, "CASTAGNA_FULL_AB_SAMPLE")
    require(len(samples) == 32 * 4, "expected 32 paired quartets")
    owner, castagna, deltas, orders = [], [], [], []
    for index in range(32):
        quartet = samples[index * 4:(index + 1) * 4]
        require(
            all(row["pair"] == "12" and int(row["round"]) == index and row["operations"] == "32"
                and int(row["slot"]) == slot for slot, row in enumerate(quartet))
            and all(int(row["record"]) == (index + 10) % 4 for row in quartet),
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
        "divergent_replay_histories": divergent,
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
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--logical-slot", dest="participant", type=int, default=3, help="source split slot independent of the physical rotation")
    parser.add_argument("--encodings", nargs=2, metavar=("GATE", "DOWN"), help="require the exact GGUF format pair")
    parser.add_argument("--timed-records", action="store_true",
                        help="compare the four fixed timing records and report any differences across the 20 replay histories")
    args = parser.parse_args()
    try:
        if not args.mi50_checkpoint:
            require(len(args.logs) == 1, "supply one log for a session summary")
            print(json.dumps(summarize(args.logs[0].read_text(), layer=args.layer, degree=args.degree, participant=args.participant,
                                       encodings=args.encodings, all_histories=not args.timed_records), indent=2))
            return
        require(not args.timed_records and (args.layer, args.degree, args.participant) == (1, 4, 3) and args.encodings in (None, ["Q4_K", "Q5_1"]),
                "the 70-us checkpoint has a fixed source format and coordinate")
        require(len(args.logs) == 3, "supply a separate log for each of three CTest invocations")
        logs = [path.read_text() for path in args.logs]
        require(len(set(logs)) == 3, "repeated native log in the checkpoint")
        sessions = [summarize(log, encodings=args.encodings) for log in logs]
        print(json.dumps({"sessions": sessions}, indent=2))
        validate_checkpoint(sessions)
    except (OSError, KeyError, ValueError) as error:
        parser.exit(1, f"Comparison failed: {error}\n")


if __name__ == "__main__":
    main()
