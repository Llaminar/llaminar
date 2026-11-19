"""Device-free replay of the OpenCode collective loader and rank contract."""
from copy import deepcopy
import hashlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from collective_runtime_binding import IMAGE_LIBRARIES, bind_mappings, observe, replay_collective_binding, require_native_owners


class CollectiveRuntimeBindingTest(unittest.TestCase):
    """Exercise native map identities and rank ownership independently."""

    def setUp(self):
        self.images = {'rocm': {'identity': (8, 1, 404)}, 'cuda': {'identity': (8, 1, 405)}}
        self.rccl = '1000-2000 r-xp 00000000 08:01 404 /usr/local/lib/librccl.so.1.0'
        self.nccl = '2000-3000 r-xp 00000000 08:01 405 /usr/local/lib/libllaminar_nccl.so.2.0'

    def test_maps_bind_both_shipped_inodes(self):
        self.assertEqual(set(bind_mappings(self.rccl + '\n' + self.nccl, self.images)), {'cuda', 'rocm'})

    def test_collective_plugins_are_distinct_from_the_primary_libraries(self):
        """Network, profiling and tuning plugins can coexist with either collective."""
        plugins = [
            f'3000-4000 r-xp 00000000 08:01 {index} /usr/lib/{name}'
            for index, name in enumerate(('libnccl-net.so', 'librccl-net.so',
                                         'libnccl-tuner.so', 'librccl-profiler.so'), start=500)
        ]
        primary = self.rccl + '\n' + self.nccl
        self.assertEqual(bind_mappings('\n'.join([primary, *plugins]), self.images),
                         bind_mappings(primary, self.images))
        self.assertEqual(bind_mappings('\n'.join(plugins), self.images), {})

    def test_serving_ranks_resolve_image_aliases_against_the_mapped_files(self):
        """Observe both SONAME links and catch a library replaced after the ranks mapped it."""
        with TemporaryDirectory() as directory:
            proc = Path(directory)
            image_root = proc / 'image'
            libs = image_root / 'usr/local/lib'
            libs.mkdir(parents=True)
            maps = []
            for backend, path in IMAGE_LIBRARIES.items():
                alias = Path(path).name
                target = libs / (alias + '.0')
                target.write_bytes(backend.encode())
                (libs / alias).symlink_to(target.name)
                stat = target.stat()
                device = f'{os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x}'
                maps.append(f'1000-2000 r-xp 00000000 {device} {stat.st_ino} /usr/local/lib/{target.name}')
            ranks = [dict(pid=88, start_ticks=90), dict(pid=89, start_ticks=91)]
            for rank in ranks:
                process = proc / str(rank['pid'])
                process.mkdir()
                (process / 'root').symlink_to(image_root, target_is_directory=True)
                (process / 'comm').write_text('llaminar2\n')
                (process / 'stat').write_text(f"{rank['pid']} (llaminar2) " +
                    ' '.join(['0'] * 19 + [str(rank['start_ticks'])]))
                (process / 'maps').write_text('\n'.join(maps))
            for observed in observe(ranks, proc=proc)['processes']:
                for backend in IMAGE_LIBRARIES:
                    shipped = observed['shipped'][backend]
                    self.assertEqual(shipped['sha256'], hashlib.sha256(backend.encode()).hexdigest())
                    self.assertEqual(observed['mappings'][backend][0]['inode'], shipped['identity'][2])
            replacement = libs / 'replacement'
            replacement.write_bytes(b'rocm')
            replacement.replace((image_root / IMAGE_LIBRARIES['rocm'].lstrip('/')).resolve())
            with self.assertRaisesRegex(ValueError, 'Unexpected rocm collective mapping'):
                observe(ranks, proc=proc)

    def test_reject_same_soname_shadow_and_platform_nccl(self):
        for shadow in (self.rccl.replace('404 /usr/local', '501 /tmp/stale'),
                       self.nccl.replace('405 /usr/local/lib/libllaminar_nccl', '502 /opt/lib/libnccl')):
            with self.subTest(shadow=shadow), self.assertRaisesRegex(ValueError, 'Unexpected'):
                bind_mappings(self.rccl + '\n' + self.nccl + '\n' + shadow, self.images)

    def test_reject_deleted_or_non_executable_files(self):
        with self.assertRaisesRegex(ValueError, 'Unexpected'):
            bind_mappings(self.rccl + ' (deleted)', self.images)
        with self.assertRaisesRegex(ValueError, 'No executable'):
            bind_mappings(self.rccl.replace('r-xp', 'r--p'), self.images)

    def test_require_selected_backends_on_actual_owner(self):
        rocm = SimpleNamespace(rank=0, pid=88, start_ticks=90, devices=[SimpleNamespace(device='ROCm:0')])
        cuda = SimpleNamespace(rank=1, pid=89, start_ticks=91, devices=[SimpleNamespace(device='CUDA:0')])
        cpu = SimpleNamespace(rank=2, pid=90, start_ticks=92, devices=[])
        records = [dict(pid=rank.pid, start_ticks=rank.start_ticks, mappings=bind_mappings(maps, self.images))
                   for rank, maps in ((rocm, self.rccl), (cuda, self.nccl), (cpu, ''))]
        calls = [
            dict(rank=0, value=1, name='calls', device='ROCm:0', domain='tp_raw_allgather_runtime',
                 tags=dict(backend='rccl', degree='4')),
            dict(rank=1, value=1, name='calls', device='CUDA:0', domain='tp_allreduce_runtime',
                 tags=dict(backend='nccl', degree='2')),
        ]
        used = require_native_owners({'processes': records}, [rocm, cuda, cpu], calls)
        self.assertEqual({entry['backend'] for entry in used}, {'rocm', 'cuda'})
        records[1]['mappings'] = records[0]['mappings']
        with self.assertRaisesRegex(ValueError, 'cuda'):
            require_native_owners({'processes': records}, [rocm, cuda, cpu], calls)
        records[1]['start_ticks'] += 1
        with self.assertRaisesRegex(ValueError, 'lifetimes'):
            require_native_owners({'processes': records}, [rocm, cuda, cpu], calls)

    def test_mixed_native_receipt_replays_saved_image_ranks_and_maps(self):
        image = 'sha256:' + 'a' * 64
        files = {backend: {**self.images[backend], 'path': path, 'bytes': 4096,
                           'sha256': hashlib.sha256(backend.encode()).hexdigest()}
                 for backend, path in IMAGE_LIBRARIES.items()}
        ranks = [SimpleNamespace(rank=index, pid=88 + index, start_ticks=90 + index,
                                 devices=[SimpleNamespace(device=device)] if device else [])
                 for index, device in enumerate(('ROCm:0', 'CUDA:0', None))]
        processes = [dict(pid=rank.pid, start_ticks=rank.start_ticks, shipped=deepcopy(files),
                          mappings=bind_mappings(maps, files))
                     for rank, maps in zip(ranks, (self.rccl, self.nccl, ''))]
        used = [
            dict(rank=0, backend='rocm', records=['tp_raw_allgather_runtime']),
            dict(rank=1, backend='cuda', records=['tp_allreduce_runtime']),
        ]
        receipt = dict(schema='llaminar.collective_runtime_binding.v1', image=image, passed=True,
                       processes=processes, native_communicators=used)
        calls = [
            dict(rank=0, value=1, name='calls', device='ROCm:0', domain='tp_raw_allgather_runtime',
                 tags=dict(backend='rccl', degree='4')),
            dict(rank=1, value=1, name='calls', device='CUDA:0', domain='tp_allreduce_runtime',
                 tags=dict(backend='nccl', degree='2')),
        ]
        self.assertEqual(replay_collective_binding(receipt, image, ranks, calls), used)

        mutations = [
            ('missing processes', lambda r: r.pop('processes')),
            ('stale image', lambda r: r.update(image='sha256:' + 'b' * 64)),
            ('duplicate process', lambda r: r['processes'].append(r['processes'][0])),
            ('stale ticks', lambda r: r['processes'][0].update(start_ticks=999)),
            ('DSO shadow', lambda r: r['processes'][0]['mappings']['rocm'][0].update(inode=501)),
            ('foreign backend file', lambda r: r['processes'][0]['mappings']['rocm'][0]
                .update(path='/usr/local/lib/libllaminar_nccl.so.2')),
            ('no executable', lambda r: r['processes'][0]['mappings']['rocm'][0].update(permissions='r--p')),
            ('foreign fingerprint', lambda r: r['processes'][2]['shipped']['rocm'].update(sha256='b' * 64)),
            ('contradictory call list', lambda r: r['native_communicators'].pop()),
        ]
        for label, mutate in mutations:
            changed = deepcopy(receipt)
            mutate(changed)
            with self.subTest(label=label), self.assertRaises(ValueError):
                replay_collective_binding(changed, image, ranks, calls)
        calls[1]['device'] = 'CUDA:1'
        with self.assertRaisesRegex(ValueError, 'outside the owning rank'):
            replay_collective_binding(receipt, image, ranks, calls)

    def test_isolated_compute_and_cpu_with_unrelated_preload(self):
        with TemporaryDirectory() as directory:
            proc = Path(directory)
            for pid in (88, 89):
                process = proc / str(pid)
                (process / 'root').mkdir(parents=True)
                (process / 'stat').write_text(f'{pid} (llaminar2) ' + ' '.join(['0'] * 19 + ['90']))
                (process / 'comm').write_text('llaminar2\n')
                (process / 'environ').write_bytes(b'LD_PRELOAD=/opt/libunrelated.so\0')
                (process / 'maps').write_text('1000-2000 r-xp 00000000 08:01 501 /opt/libunrelated.so')
            isolated = SimpleNamespace(rank=0, pid=88, start_ticks=90, devices=[SimpleNamespace(device='CUDA:0')])
            cpu = SimpleNamespace(rank=1, pid=89, start_ticks=90, devices=[])
            receipt = observe([{'pid': rank.pid, 'start_ticks': rank.start_ticks} for rank in (isolated, cpu)], proc=proc)
        noop = dict(rank=0, value=1, name='calls', device='CUDA:0', domain='tp_allreduce_runtime',
                    tags=dict(backend='nccl', degree='1'))
        self.assertEqual(require_native_owners(receipt, [isolated, cpu], [noop]), [])
        # Separate singleton pipeline stages can share a rank; residency alone requires no collective.
        isolated.devices.append(SimpleNamespace(device='CUDA:1'))
        self.assertEqual(require_native_owners(receipt, [isolated, cpu], []), [])
        image = 'sha256:' + 'a' * 64
        receipt.update(image=image, passed=True, native_communicators=[])
        self.assertEqual(replay_collective_binding(receipt, image, [isolated, cpu], [noop]), [])


if __name__ == '__main__':
    unittest.main()
