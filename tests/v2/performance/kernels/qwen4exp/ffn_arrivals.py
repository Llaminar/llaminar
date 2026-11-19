#!/usr/bin/env python3
"""Measure when physical TP participants finish each FFN layer in a fixed-depth QFN verifier.

Read a retired ROCProfiler benchmark window and the original per-PCI source
inventory. Use device dispatch timestamps on rocprof's common clock. Match a
completed verifier round on every participant before comparing layer ends.
The final peer-push end before each FFN provides a second, local elapsed-time
comparison. Wider shards can rotate by layer or tie in width.
"""
import argparse
import json
import re
import sqlite3
from pathlib import Path
from statistics import median


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def participant_rounds(db, tables, agent, rounds, routed_widths):
    """Collect one ordered FFN producer per layer between successive verifier terminals."""
    kernel, graph, symbol = (tables[key] for key in ('kernel_dispatch', 'graph_launch', 'info_kernel_symbol'))
    # Graph IDs can be reused between warmup and measurement. The last prefill
    # launch fences warmup out even when both requests used the same executable.
    prefill = db.execute(f'''
        SELECT k.graph_exec_id, k.start FROM {kernel} k JOIN {symbol} s ON s.id=k.kernel_id
        WHERE k.agent_id=? AND instr(s.display_name,'preparePrefillChunkViewKernel')>0
        ORDER BY k.start DESC LIMIT 1''', (agent,)).fetchone()
    require(prefill and prefill[0], 'Missing captured prefill boundary')
    end, count = db.execute(f'SELECT MAX(end), COUNT(*) FROM {kernel} WHERE agent_id=? AND graph_exec_id=? AND start>=?',
                            (agent, *prefill)).fetchone()
    def graph_size(executable):
        sizes = db.execute(f'SELECT DISTINCT kernel_dispatch_count FROM {graph} WHERE agent_id=? AND graph_exec_id=?',
                           (agent, executable)).fetchall()
        require(len(sizes) == 1 and sizes[0][0] > 1, 'Ambiguous captured graph geometry')
        return sizes[0][0]
    require(count == graph_size(prefill[0]), 'Incomplete final prefill')
    terminals = db.execute(f'''
        SELECT graph_exec_id,COUNT(*) FROM {kernel} k JOIN {symbol} s ON s.id=k.kernel_id
        WHERE agent_id=? AND start>=? AND instr(s.display_name,'summarize_greedy_speculative_verify')>0
        GROUP BY graph_exec_id''', (agent, end)).fetchall()
    require(len(terminals) == 1 and terminals[0][0] and terminals[0][1] == rounds,
            'Verifier terminals differ from the measured request')
    target = terminals[0][0]
    count, = db.execute(f'SELECT COUNT(*) FROM {kernel} WHERE agent_id=? AND start>=? AND graph_exec_id=?',
                        (agent, end, target)).fetchone()
    require(count == rounds * graph_size(target), 'Partial verifier graph')
    events = db.execute(f'''
        SELECT k.start,k.end,s.display_name FROM {kernel} k JOIN {symbol} s ON s.id=k.kernel_id
        WHERE k.agent_id=? AND k.start>=? AND k.graph_exec_id=? AND
        (instr(s.display_name,'peerPushFP32Kernel')>0 OR instr(s.display_name,'groupedFFNTokenDown')>0 OR
         instr(s.display_name,'groupedFFNFormatProgramDown')>0 OR
         instr(s.display_name,'rocm_moe_shared_expert_gate_add_kernel')>0 OR
         instr(s.display_name,'summarize_greedy_speculative_verify')>0)
        ORDER BY k.start''', (agent, end, target))
    history, layers, push_end = [], [], None
    for start, finish, name in events:
        if 'peerPushFP32Kernel' in name:
            push_end = finish
        elif 'summarize_greedy_speculative_verify' in name:
            require(len(layers) == len(routed_widths), 'Missing or extra FFN producer within a verifier round')
            history.append(layers)
            layers, push_end = [], None
        else:
            require(push_end is not None and push_end <= start <= finish, 'Missing ordered peer-to-FFN boundary')
            if 'groupedFFNFormatProgramDown' in name:
                # The last template argument is zero for a runtime width. A
                # fixed-width reader independently identifies the actual shard.
                match = re.search(r'groupedFFNFormatProgramDown<[^<>]*,\s*(\d+)>', name)
                require(match and len(layers) < len(routed_widths),
                        'Unknown grouped down signature or extra FFN producer')
                width = int(match[1])
                require(width == 0 or width == routed_widths[len(layers)],
                        'Compiled FFN shard width differs from original participant inventory')
            layers.append((finish, finish - push_end))
            push_end = None
    require(len(history) == rounds and not layers, 'Incomplete verifier rounds')
    return history


def measure(window, inventory):
    """Join the end of each participant's FFN using original layer and verifier round."""
    result = read(window/'data/run/result.json')
    require(result['passed'] and result['complete'] and result['instrumented'] and result['profile'] == 'llaminar'
            and result['mtp'].startswith('fixed') and result['custody']['physical_retirement']
            and read(window/'driver-report.json')['passed'] and (window/'status').read_text().strip() == '0'
            and (window/'physical-before.txt').read_bytes() == (window/'physical-after.txt').read_bytes(),
            'Requires a retired fixed-depth native Llaminar profile')
    (observations,) = result['observations'].values()
    (observation,) = observations
    rounds = observation['mtp']['verifier_runs']
    require(rounds > 0, 'No measured verification work')
    participants = read(window/'data/admission.json')['pci_to_runtime_ordinal']
    require(set(inventory) == set(participants) and len(participants) >= 2, 'Original source inventory differs from admission')
    count = len(next(iter(inventory.values()))['layers'])
    for owner in inventory.values():
        rows = owner['layers']
        require(len(rows) == count and count and all(v['layer'] == n and v['routed'] > 0 for n, v in enumerate(rows)),
                'Source inventory must cover every original layer in order')
    (database,) = (window/'data/run/profiler').rglob('*_results.db')
    history = {}
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as db:
        names = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        tables = {}
        for prefix in ('info_agent', 'kernel_dispatch', 'graph_launch', 'info_kernel_symbol'):
            (name,) = [n for n in names if n.startswith('rocpd_'+prefix+'_')]
            tables[prefix] = '"' + name.replace('"', '""') + '"'
        for agent, raw in db.execute(f"SELECT id,extdata FROM {tables['info_agent']} WHERE type='GPU'"):
            data = json.loads(raw)
            address = data['location_id']
            pci = f"{data['domain']:04x}:{address >> 8:02x}:{(address >> 3) & 31:02x}.{address & 7}"
            if pci in participants:
                require(pci not in history, 'Duplicate physical GPU in trace')
                history[pci] = participant_rounds(db, tables, agent, rounds,
                                                  [layer['routed'] for layer in inventory[pci]['layers']])
    require(history.keys() == participants.keys(), 'Native trace omitted a compute participant')
    layers = []
    for layer in range(count):
        widths = {p: inventory[p]['layers'][layer]['routed'] for p in participants}
        wide = [p for p in participants if widths[p] == max(widths.values())]
        narrow = [p for p in participants if p not in wide]
        gaps, elapsed_gaps, spreads = [], [], []
        ties = 0
        for r in range(rounds):
            ends = [history[p][r][layer][0] for p in participants]
            spreads.append((max(ends) - min(ends)) / 1000)
            ties += ends.count(max(ends)) > 1
            if narrow:
                for metric, samples in enumerate((gaps, elapsed_gaps)):
                    diff = max(history[p][r][layer][metric] for p in wide) - max(history[p][r][layer][metric] for p in narrow)
                    samples.append(diff / 1000)
        layers.append(dict(layer=layer, routed_columns=widths, rounds=rounds, end_ties=ties,
                           median_first_to_last_us=median(spreads),
                           widest_finishes_strictly_last=sum(v > 0 for v in gaps) if gaps else None,
                           median_widest_minus_narrower_us=median(gaps) if gaps else None,
                           median_since_previous_peer_difference_us=median(elapsed_gaps) if gaps else None))
    return dict(prompt=next(iter(result['observations'])), rounds=rounds, layers=layers)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('window', type=Path)
    parser.add_argument('source_inventory', type=Path)
    args = parser.parse_args()
    print(json.dumps(measure(args.window, read(args.source_inventory)), indent=2))
