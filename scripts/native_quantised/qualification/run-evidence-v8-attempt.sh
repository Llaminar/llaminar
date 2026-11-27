#!/bin/bash
# run-evidence-v6b.sh <tree> <kind> <isa-lower>: tools-v8/run-evidence-v8.sh (unchanged analysis) wrapped with a
# reboot-safe attempt record: <artifacts>/attempt.json {boot_id_start, boot_id_end, started, ended, exit,
# complete}. complete=true only when both boot ids are equal, the run exited 0 and SHA256SUMS exists; a run cut by a
# reboot leaves boot_id_end absent (or different) and is never consumable.
set -uo pipefail
W=$HOME/pr73-production; tree=$1; kind=$2; lisa=$3; a=$W/artifacts-v8/$tree-$kind-$lisa
boot() { cat /proc/sys/kernel/random/boot_id; }
start_boot=$(boot); started=$(date -u +%FT%TZ)
echo "{\"tree\":\"$tree\",\"kind\":\"$kind\",\"isa\":\"$lisa\",\"boot_id_start\":\"$start_boot\",\"started\":\"$started\",\"tool\":\"tools-v8/run-evidence-v8.sh\"}" >> $W/attempts/ledger.jsonl
$W/tools-v8/run-evidence-v8.sh $tree $kind $lisa; rc=$?
end_boot=$(boot); ended=$(date -u +%FT%TZ)
complete=false; [ "$rc" = 0 ] && [ "$start_boot" = "$end_boot" ] && [ -f $a/SHA256SUMS ] && complete=true
test -d $a && echo "{\"boot_id_start\":\"$start_boot\",\"boot_id_end\":\"$end_boot\",\"started\":\"$started\",\"ended\":\"$ended\",\"exit\":$rc,\"complete\":$complete,\"sha256sums_sha256\":\"$(sha256sum $a/SHA256SUMS 2>/dev/null | cut -c1-64)\"}" > $a/attempt.json
echo "{\"tree\":\"$tree\",\"kind\":\"$kind\",\"isa\":\"$lisa\",\"boot_id_end\":\"$end_boot\",\"ended\":\"$ended\",\"exit\":$rc,\"complete\":$complete}" >> $W/attempts/ledger.jsonl
exit $rc
