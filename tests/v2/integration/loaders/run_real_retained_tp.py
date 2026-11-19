"""Resolve the original same-NUMA gfx906 roster for a real-source CTest."""
import argparse
from collections import defaultdict
import os
from pathlib import Path


def source_roster():
    """Enumerate original KFD devices without starting a GPU runtime."""
    nodes = Path('/sys/class/kfd/kfd/topology/nodes')
    devices = defaultdict(list)
    ordinal = 0
    for node in sorted(nodes.iterdir(), key=lambda path: int(path.name)):
        fields = dict(line.split() for line in (node / 'properties').read_text().splitlines())
        minor = int(fields.get('drm_render_minor', '0'))
        if minor < 128:
            continue
        sysfs = (Path('/sys/class/drm') / f'renderD{minor}' / 'device').resolve()
        if fields['gfx_target_version'] == '90006':
            numa = int((sysfs / 'numa_node').read_text())
            if numa >= 0:
                devices[numa].append((ordinal, sysfs.name))
        ordinal += 1
    if not devices:
        raise SystemExit('No original same-NUMA gfx906 source roster')
    return max(devices.items(), key=lambda entry: len(entry[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model')
    parser.add_argument('mode', choices=('audit', 'window', 'movement'))
    parser.add_argument('executable')
    parser.add_argument('--width', type=int, choices=(1, 2, 4, 8), default=8)
    args = parser.parse_args()
    resolved = os.environ.get('LLAMINAR_TEST_QFN_ORIGINAL_ROCM_ROSTER')
    if resolved:
        candidates = resolved.split(',')
        if len(set(candidates)) != len(candidates) or any(not value.isdecimal() for value in candidates):
            raise SystemExit('Original ROCm roster requires unique, nonnegative ordinals')
        ordinals = candidates[:args.width]
        print(f'QFN_TP_SOURCE_ROSTER,width={args.width},host_resolved_order={ordinals}', flush=True)
    else:
        numa, candidates = source_roster()
        roster = candidates[:args.width]
        ordinals = [str(ordinal) for ordinal, _ in roster]
        print(f'QFN_TP_SOURCE_ROSTER,numa={numa},width={args.width},order={roster}', flush=True)
    if len(ordinals) != args.width:
        raise SystemExit(f'Original source roster has {len(ordinals)} members, needs {args.width}')
    if args.mode == 'audit':
        if args.width != 8:
            raise SystemExit('The source audit requires the complete TP8 roster')
        os.execv(args.executable, [args.executable, args.model, *ordinals])
    if args.mode == 'movement':
        if args.width != 4 or not os.environ.get('LLAMINAR_TEST_RETAINED_HC_FFN_MOVEMENT_PCIS'):
            raise SystemExit('TP4 motion requires explicit hot, four warm and capacity PCI owners')
        os.environ.update(
            LLAMINAR_TEST_CASTAGNA_GGUF=args.model,
            LLAMINAR_TEST_QFN_SOURCE_ROCM_ORDINALS=','.join(ordinals),
            LLAMINAR_TEST_QFN_SOURCE_LAYER='0',
            LLAMINAR_TEST_QFN_SOURCE_PARTICIPANT='0',
        )
        os.execv(args.executable, [
            args.executable, '--gtest_also_run_disabled_tests',
            '--gtest_filter=FullNativeHCFFN.DISABLED_ROCmRetainedProductionTP4Epoch',
        ])
    if not os.environ.get('LLAMINAR_TEST_RETAINED_HC_FFN_PCI'):
        raise SystemExit('Set LLAMINAR_TEST_RETAINED_HC_FFN_PCI to the dedicated native replay card')
    os.environ.update(
        LLAMINAR_TEST_CASTAGNA_GGUF=args.model,
        LLAMINAR_TEST_QFN_SOURCE_ROCM_ORDINALS=','.join(ordinals),
    )
    os.environ.setdefault('LLAMINAR_TEST_QFN_SOURCE_LAYER', '1')
    os.environ.setdefault('LLAMINAR_TEST_QFN_SOURCE_PARTICIPANT', '0')
    os.execv(args.executable, [
        args.executable, '--gtest_also_run_disabled_tests',
        '--gtest_filter=FullNativeHCFFN.DISABLED_ROCmRetainedProductionTPWindow',
    ])


if __name__ == '__main__':
    main()
