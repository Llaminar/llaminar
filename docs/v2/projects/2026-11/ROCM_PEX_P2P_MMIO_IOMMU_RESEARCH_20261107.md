# ROCm PCIe P2P research: MMIO placement, IOMMU routing and recovery

Research checkpoint: 2026-11-07. This is an investigation record, not a tuning
dashboard or a production qualification certificate.

Status: the earlier translated-DMA configuration fails the small native read
with ATS on and off. The current combined mitigation **passes all 20 directed
AMD read links**: BIOS MMIO High Base 4T (1024G unchanged), identity DMA
(`iommu=pt`), kernel `6.17.0-1000-generic`, P2P enabled, boot
`2ef86b85-de24-4e81-8796-d5a21af56178`. The original unchanged read binary
verifies 16,384 words per direction with clean driver windows and retirement.
This is not a BIOS-alone causal experiment. The first Llaminar captured peer
test subsequently found a byte mismatch with a clean driver window; production
transport, model and performance qualification remain open. The normal default
remains ABI 999/P2P-off; the successful identity-DMA entry was one-shot and is
consumed.

This investigation is separate from QFN arithmetic, ngram caching, RCCL/MPI
and the large mapped-RAM scatterlist correction. Earlier failure and recovery
evidence below is preserved.

## Tested software and firmware configuration

The running stack was checked after reboot. Installed packages alone do not
establish which kernel or modules executed these tests.

| Component | Tested identity |
|---|---|
| Running Linux kernel | `6.17.0-1000-generic` |
| Custom kernel package | `6.17.0-1000.42+llaminar2` |
| Ubuntu source base | `6.17.0-42.42`, tag `Ubuntu-hwe-6.17-6.17.0-42.42`, commit `7cb9b24dc95cd7faa58265648bd74bd17b03f55d` |
| Kernel build ID | `bc3a31fc96512be21f0f3ed43ee8727b0de6f1e1` |
| Previous custom kernel | `6.17.0-999-generic`, package `6.17.0-999.42+llaminar1`; retained as P2P-off recovery |
| Earlier recovery kernels | `6.14.0-37-generic` and `6.14.0-36-generic` |
| Loaded AMD driver | `7.1.9.31600000`, DKMS source `7.1.9-2407772.24.04`, srcversion `BFEAE8D9287FE23B9907691` |
| AMD installed package / firmware | `1:7.1.9.31600000-2407772.24.04+local.dce1` / `1:31.60.0.0.31600000-2407772.24.04` |
| Loaded NVIDIA open driver | `595.99.02`, srcversion `919309EF6954372EFADD77E` |
| NVIDIA package version | `595.99.02-0ubuntu0.24.04.1`; `nvidia-driver-595-open` and `nvidia-dkms-595-open` |
| Loaded PEX resource module | `pex-window` / `pex_window` `0.1.2`, srcversion `7D4D55A110866DB75810F63` |
| Native runtime | Container ROCm **10.0.0** SDK at `/opt/rocm-sdk`; host ROCm/CUDA SDKs removed, host driver/runtime integration retained |
| Native allocation policy | `HSA_USERPTR_FOR_PAGED_MEM=0`, `HSA_USE_SVM=0`; `HSA_OVERRIDE_GFX_VERSION` unset |

The AMD driver also retains a local legacy-DCE build correction, separate from
the Linux patches below. Do not replace it with a stock driver solely because
the displayed upstream version matches. Driver and package provenance is also
recorded in the [host activation note](QFN_HOST_KERNEL_DRIVER_ACTIVATION_20261107.md).

### Patches in the tested kernel

ABI 1000 builds on the ABI 999 candidate. Both preserve the Ubuntu generic
configuration, including Rust, BTF and DWARF, apart from the custom version
signature. No P2P raw-BAR substitution or DMA-mask widening was installed.

| Change | Source and effect | Boundary of the evidence |
|---|---|---|
| Intel queued-invalidation recovery | Upstream [`fe89277c9ceb0d6af0aa665bcf24a41d8b1b79cd`](https://github.com/torvalds/linux/commit/fe89277c9ceb0d6af0aa665bcf24a41d8b1b79cd), `drivers/iommu/intel/dmar.c`: mark every fetched in-use entry aborted after timeout, including even-indexed wait descriptors | Present in ABI 999 and 1000. Actual-function CPU replay covers 524,309 patched cases; it repairs abort bookkeeping, not the initial device fault or every possible retry loop |
| Scatterlist overflow detection | Upstream [`1c8f6308e557f32412e9d4678b3f029f47f6d9aa`](https://github.com/torvalds/linux/commit/1c8f6308e557f32412e9d4678b3f029f47f6d9aa), `drivers/iommu/dma-iommu.c`: reject lengths/padding that overflow the scatterlist field | Added in ABI 1000 after the separate large mapped-RAM failure; not a demonstrated fix for peer-VRAM routing |
| Bounded DRM scatterlist input segments | Local patch to `drivers/gpu/drm/drm_prime.c`: `max_segment = min_t(size_t, max_segment, UINT_MAX / 2)` before building the page-array scatterlist | Added in ABI 1000. Splits large contiguous runs while retaining total allocation size and stricter device limits; actual-function CPU composition covers 16 geometries |
| Rust header package closure | Local patch to `debian/rules.d/2-binary-arch.mk`: pass `$(baserustpkg)` instead of `$(indeppkg)` to `link-lib-rust` | Packaging correction retained from ABI 999; links headers to the actual Rust library package, not an inference arithmetic or DMA change |
| Package/ABI identity | Custom ABI 999 then 1000, package suffixes `+llaminar1` then `+llaminar2` | Keeps candidates distinct from stock Ubuntu packages and preserves recovery entries |

The retained patch hunks are committed beside this note, so reconstructing the
changes does not depend on temporary investigation directories. The AMD patch's
two file headers are normalized from temporary absolute paths to `a/` and `b/`;
its code hunk is unchanged.

1. [Intel QI recovery](patches/rocm-pex-p2p-20261107/0001-intel-iommu-qi-recovery.patch).
2. [IOMMU scatterlist overflow guards](patches/rocm-pex-p2p-20261107/0002-iommu-sg-overflow.patch).
3. [Local DRM segment bound](patches/rocm-pex-p2p-20261107/0003-drm-prime-segment-bound.patch).
4. [Rust header packaging correction](patches/rocm-pex-p2p-20261107/0004-rust-header-package-link.patch).
5. [AMD legacy-DCE build correction](patches/rocm-pex-p2p-20261107/0005-amdgpu-legacy-dce.patch),
   applied to the separate AMD DKMS source, not the Linux source tree. It adds
   `export_macro_mk CONFIG_DRM_AMD_DC_DCE` in `amd/dkms/dkms-config.sh`, preserving
   the legacy DCE configuration default in the external build.

The first four patches target the Ubuntu base above. The fifth targets AMD
`7.1.9-2407772.24.04`. These are records of the tested local changes, not a
general installer; package versioning, the stock configuration and module
rebuilds remain part of reconstruction. The custom kernel and derived AMD
package are locally built artifacts, not vendor-signed release packages.

For this documentation checkpoint, all five committed patches were reapplied
without fuzz to isolated copies of their original source files. The resulting
kernel/packaging files are byte-identical to the ABI 1000 candidate source;
the AMD file is byte-identical to the local package source. This validates the
recorded patches, not a new kernel build or an additional runtime qualification.

The ABI 1000 installation receipt is
`/tmp/qfn-p2p-root-r906/kernel-install-r2/report.json`; it records the six
installed kernel/header/Rust packages and 15 validated external-module
products. Patch/build/package provenance remains under the r889, r894 and r905
evidence roots listed below. Those local evidence paths are historical
artifacts, not repository dependencies or guaranteed permanent storage.

### BIOS and effective Linux settings

The user reports AMI Aptio (2025), PCI Bus Driver A5.01.18. The only deliberate
BIOS change in this experiment was **MMIO High Base: 16T → 4T**. Its unit labels
are firmware labels; all address arithmetic in this note uses binary bytes.

| Setting | Successful experiment |
|---|---|
| MMIO High Base / Granularity | **4T / 1024G**; granularity unchanged |
| Above 4G Decoding, SR-IOV, ARI | Enabled |
| VT-d / interrupt remapping | Enabled; kernel confirms IRQ remapping in x2APIC mode |
| BIOS ACS Control | Disabled; existing targeted Linux ACS-redirection configuration retained |
| BIOS PassThrough DMA / ATS | Enabled; these labels alone do not establish the effective Linux DMA domain |
| Linux DMA-domain selection | `intel_iommu=on iommu=pt`; sysfs reports actual `identity` domains for all five AMD GPUs |
| AMD P2P parameter | `amdgpu.pcie_p2p=Y` |
| PCI resource setup | Existing `pci=realloc,assign-busses` and `pex_window` configuration retained |
| PEX configuration | `apply=1 bar_gib=32 slot_gib=64 empty_min_width=8` |

The successful boot does **not** use `pci=noats` or `iommu=off`. Identity DMA
removes I/O address translation for these domains while IRQ remapping remains
enabled. This distinction is also described in AMD's
[IOMMU mode documentation](https://rocm.docs.amd.com/en/docs-6.3.1/conceptual/iommu.html).
The observed configuration is qualified only to the scope below, not a generic
recommendation for every Radeon, root complex or virtualized workload.

Final post-PEX VRAM BAR0 ranges on the successful boot are:

| GPU | Inclusive physical range | Size |
|---|---|---|
| 7900 XTX `49:00.0` | `0x64000000000–0x647ffffffff` | 32 GiB aperture |
| MI50 `8c:00.0` | `0x90000000000–0x907ffffffff` | 32 GiB aperture |
| MI50 `91:00.0` | `0x91000000000–0x917ffffffff` | 32 GiB aperture |
| MI50 `97:00.0` | `0x93000000000–0x937ffffffff` | 32 GiB aperture |
| MI50 `9a:00.0` | `0x94000000000–0x947ffffffff` | 32 GiB aperture |

An aperture size is not physical VRAM capacity. All these ranges now fit below
`0x100000000000` (16 TiB, the exclusive 44-bit limit). Highest observed System
RAM address is `0xc03fffffff`, so the new windows remain above installed RAM.

### Complete user-supplied BIOS record

The following preserves the user's transcription of five BIOS screenshots.
Values in these tables are the **original supplied record**, before the later
confirmed change to **MMIO High Base = 4T**. No other BIOS change was reported.
This is not a full firmware dump; unavailable dropdown choices are not inferred.

- BIOS: **AMI Aptio Setup Utility (2025)**.
- PCI Bus Driver Version: **A5.01.18**.

**PCIe / PCI / PnP Configuration**

| Setting | Original supplied value |
|---|---|
| Above 4G Decoding | Enabled |
| SR-IOV Support | Enabled |
| ARI Support | Enabled |
| MMIO High Base | **16T**; subsequently changed to **4T** |
| MMIO High Granularity Size | **1024G** |
| Maximum Read Request | Auto |
| MMCFG Base | 3G |
| VGA Priority | Onboard |

Available **MMIO High Base** options: `1T`, `4T`, `16T`, `24T`, `40T`, `56T`.
Available **MMIO High Granularity Size** options: `1G`, `4G`, `16G`, `64G`,
`256G`, `1024G`. Only these two complete dropdown lists were supplied.

**PCI Devices Option ROM Settings**

| Device / slot | Original supplied value |
|---|---|
| Onboard NVMe 1 OPROM | EFI |
| Onboard NVMe 2 OPROM | EFI |
| CPU1 Slot 1 PCI-E x8 OPROM | EFI |
| CPU1 Slot 2 PCI-E x16 OPROM | EFI |
| CPU1 Slot 3 PCI-E x8 OPROM | EFI |
| CPU2 Slot 4 PCI-E x16 OPROM | EFI |
| CPU2 Slot 5 PCI-E x16 OPROM | EFI |
| CPU2 Slot 6 PCI-E x16 OPROM | EFI |
| Onboard Video OPROM | UEFI |

**Intel VT for Directed I/O (VT-d)**

| Setting | Original supplied value |
|---|---|
| Intel VT-d | Enable |
| ACS Control | Disable |
| Interrupt Remapping | Enable |
| PassThrough DMA | Enable |
| ATS | Enable |
| Posted Interrupt | Enable |
| Posted Interrupt Throttle | Enable |
| Coherency Support (Non-Isoch) | Enable |

**IOAT Configuration**

| Setting | Original supplied value |
|---|---|
| Disable TPH | No |
| Prioritize TPH | Disable |
| Relaxed Ordering | Enable |

Thus TPH is not disabled, but TPH prioritization is disabled. The BIOS permits
MMIO High Base up to 56T and granularity up to 1024G; those available maxima do
not establish that such settings satisfy every device's addressing limit.
The later 4T confirmation and actual post-reboot resources above are the
authority for this experiment's successful address placement.

### `pex-window` and Resizable BAR

`pex-window` is an existing locally authored DKMS module, not part of the AMD
or NVIDIA driver packages. DKMS package `pex-window/0.1.2` builds module
`pex_window` from `/usr/src/pex-window-0.1.2`, installs under `updates/dkms`, and
has `AUTOINSTALL=yes`. Its retained source is
`/tmp/qfn-nvidia-pex617-qualification-r891/original/pex-window-0.1.2/`;
`pex_window.c`, `plan.h` and `transaction.h` own the implementation. No upstream
repository/revision was authenticated for this local module. The original
source required no compatibility modification for the tested 6.17 kernels;
the installed ABI 1000 product has the version/srcversion recorded above.

The module discovers the PEX880xx switch trees (`1000:c010`) and plans their
complete 64-bit prefetchable resource hierarchy. It assigns root/downstream
bridge windows and GPU BARs together, preserving bus numbers and unrelated
MMIO. ReBAR resizing is one part of that transaction: an existing sufficiently
large VRAM BAR is retained; enlargement requires the endpoint's actual ReBAR
capability and supported-size mask. It does not force an unsupported size or
shrink an already larger BAR. A 32 GiB aperture on a 24 GiB card does not add
physical VRAM.

The installed configuration is:

```text
options pex_window apply=1 bar_gib=32 slot_gib=64 empty_min_width=8
softdep amdgpu pre: pex_window
softdep nvidia pre: pex_window
softdep nouveau pre: pex_window
softdep snd_hda_intel pre: pex_window
```

| Parameter | Meaning in version 0.1.2 |
|---|---|
| `apply=1` | Apply the verified transaction; the module's default `apply=0` only plans |
| `bar_gib=32` | Request at least a 32 GiB VRAM BAR, subject to the actual capability; retain a larger existing BAR |
| `slot_gib=64` | Reserve 64 GiB of MMIO address space per occupied or qualifying empty slot; this is not a VRAM allocation |
| `empty_min_width=8` | Reserve empty downstream ports whose maximum link width is at least x8; zero disables that empty-port reservation |
| `roots` unset | Discover all eligible PEX trees; an explicit root-port list is also supported |
| `fail_step=0` | No transaction fault injection |

The two inspected trees each reserve five 64 GiB slots, including the configured
empty-slot capacity. Sizes must be powers of two and a slot must accommodate its
BAR. This reserves space within a host aperture; it does not create another
aperture or extend the root complex's physical decoder.

Early loading matters. `/etc/initramfs-tools/hooks/pex-window` includes the
module, and `/etc/initramfs-tools/scripts/init-top/00-pex-window` loads it before
udev coldplug and ordinary module loading. The soft dependencies above also
cover later/manual endpoint probes. The early shell script logs load failure;
it is not itself a fatal gate guaranteeing that no GPU driver can start afterward.
The diagnostic AMD loader and Docker readiness gate are described separately in
the [recovery note](ROCM_PEX_P2P_DIAGNOSTIC_RECOVERY_20261107.md).

Application requires detached endpoint drivers. The module serializes PCI
rescan/removal, disables memory decoding and bus mastering during replacement,
updates the Linux resource hierarchy and hardware registers, and checks writes.
An unsuccessful application attempts to restore the original state. Successful
resources belong to the PCI devices, so **unloading the module does not undo the
layout**. Its explicit rollback operation requires retained transaction state,
detached endpoints and unchanged resource ownership; it is not safe to use as a
live-inference toggle. Reboot/recovery and firmware rollback remain separate.

The crucial address limitation is that planning retains each root's existing
ACPI/firmware host aperture. Version 0.1.2 has no base-address override and cannot
move a BAR outside that aperture. With BIOS MMIO High Base at 16T, every high
host window began at or above the MI50's 44-bit ceiling. Changing slot sizes or
rerunning the module could not provide an in-range direct address. The BIOS 4T
setting supplies lower host windows; the same module can then allocate within
them. It does not change DMA masks, IOMMU domains, ATS/ACS policy or P2P routing.

Read the **final post-PEX** sysfs resources when checking this configuration.
Early PCI enumeration addresses precede the transaction and can differ from the
addresses later imported by the GPU driver. Preserve the boot ID along with
the BAR ranges, actual IOMMU domain type and loaded module identity.

## Experiment summary

| MMIO base | Effective DMA domain | ATS control | Native remote read |
|---|---|---|---|
| 16T | Translated `DMA-FQ` | Enabled | 97→9a faults; reset fails, later VT-d/NIC disruption and user-observed watchdog reboot |
| 16T | Translated `DMA-FQ` | Disabled with `pci=noats`, actual endpoint state checked | 97→9a still faults and reset fails; no VT-d/NIC cascade during the observed 313.881-second window, but TTM cleanup stalls |
| 4T | `identity` via `iommu=pt` | No `pci=noats` override | All 20 directed AMD links pass, with full 64 KiB verification and clean driver/process retirement |

There was no 4T/translated-DMA remote-read experiment and no 16T/identity-DMA
remote-read experiment. Accordingly, the result establishes a working combined
mitigation for bounded native reads, not the independent effect of either change.
Passing all directed pairs also does not prove those pairs ran concurrently.

## Reproduction and recovery

The controlled baseline used kernel `6.17.0-1000-generic`, AMD
`7.1.9.31600000`, the existing PEX resource module and unchanged IOMMU/ACS
settings. Delayed AMD loading followed persistent journal flush and verification
of the active IPMI watchdog. GRUB consumed the one-shot entry on both boots.

With P2P off, individual mapped-RAM allocation/alias/free processes passed for
4 KiB, 5,391,777,792 bytes (the earlier failing extent) and 6,197,084,160 bytes.
All three retired with clean canonical driver observations. These checks do not
establish the original physical scatterlist geometry or qualify full QFN model
admission. A native peer-capability query returned unsupported as expected.

With P2P on, the same kernel booted successfully and reported peer capability.
The two visible devices were physical MI50s `97:00.0` and `9a:00.0`, the nearest
switch pair. The reader at `9a:00.0` enabled access to the owner at `97:00.0`.
Two 64 KiB native device allocations, owner initialization, and owner-stream
completion succeeded. A reader kernel then directly loaded the owner's pointer.
The host allocation policy matched Llaminar startup:
`HSA_USERPTR_FOR_PAGED_MEM=0`, `HSA_USE_SVM=0`; no ISA override was used.

The first remote load triggered a fatal AMD RAS event and a GPU reset. The native
process aborted and retired; no reverse-direction test or collective followed.
The host subsequently restarted. The user reports a kernel lockup followed by
hardware-watchdog reset. The new boot is kernel 999 with P2P off, proving that
the one-shot selection did not repeat the failing boot. It does not identify the
reset source by itself.

The [recovery procedure](ROCM_PEX_P2P_DIAGNOSTIC_RECOVERY_20261107.md) records the
visible menu, retained kernels, persistent logging and exact recovery commands.

## Observed failure chain

Times below are monotonic seconds in failed boot
`3e7241d5-a22d-49ed-b0db-a6bc20890979`.

| Time | Evidence |
|---|---|
| 151.838716 | Owner initialization stream completed successfully |
| 151.846850 | Reader remote-load kernel submitted |
| 151.851289 | Host entered reader-stream completion wait |
| 151.852781 | Reader reports `ERREVENT_ATHUB_INTERRUPT` |
| 151.856692 | RAS-triggered GPU reset begins |
| 168.693661 | Pre-ASIC reset phase fails; SMU/firmware errors precede it |
| 172.436184 | Hardware reset step reports success and attempts resume |
| 176.175814 | Overall GPU reset/resume ends with `-22` |
| 176.200020 | Firmware reports a PCIe error for reader `9a:00.0` |
| 177.010678 onward | Intel VT-d invalidation timeouts recur |
| 184.410786 | Network adapter TX watchdog reports stalled progress |
| 199.447053 | Last retained kernel diagnostic; final message is incomplete |

The ATHUB label comes from a generic RAS handler and is not a decoded GPUVM fault
address. Source inspection separates the initial event from later SMU/reset
failures. The intermediate reset-success message does not mean recovery worked.

The PCIe status `0x8000` is Completer Abort. The captured TLP decodes as a
one-DWORD 32-bit memory write with requester ID zero, targeting `0xf007f000`.
That address lies in the reader's register BAR5 at offset `0x7f000`, rather than
its VRAM window. This is a later reported/latched transaction; it cannot be
substituted for the original GPU remote-read request.

There are 29 retained VT-d timeout reports, about 0.8 seconds apart. Most later
queue heads identify interrupt-entry-cache invalidation. `ITE cleared` reports
clearing a fault status bit, not successful queue recovery. The installed QI
patch repairs abort marking but leaves repeated `-EAGAIN` retries unbounded.
Some interrupt-remapping callers hold their outer IRQ lock with interrupts
disabled across that submission, providing a concrete system-stall mechanism
to investigate. No retained CPU stack proves which caller stopped progressing.

BMC record `0397` reports a bus uncorrectable error at 14:50:18 UTC. The current
watchdog has its SMS/OS expiration flag set, but no newly timestamped Watchdog2
reset event was found in the inspected SEL tail. ERST/pstore is empty. These
limits do not contradict the user's observed lockup; they prevent claiming a
specific final reset cause from the saved logs alone.

## Mapping and address-layout suspect

Exact vendor source maps peer VRAM using `dma_map_resource`, places its returned
DMA address in the scatterlist, and consumes those addresses in the imported
GPU mapping. Under this host's translated DMA-FQ domains, that API allocates an
IOVA. Earlier P2PDMA distance admission is used as a boolean and does not retain
the local-bus versus host-bridge routing class in this VRAM mapping path.

This is a concrete hypothesis, not proof that an IOVA was routed incorrectly.
An address outside a peer BAR can travel upstream; ACS redirection being clear
does not by itself prove that the transaction bypassed the IOMMU. Actual
physical/DMA addresses and, if needed, final PTE/ATS state remain necessary.

The addressability constraint is also concrete. Vega20's driver selects a
44-bit DMA mask (below 16 TiB), while its MC/PTE address width is 48 bits. The
failed configuration's post-PEX MI50 VRAM BARs start at 21 TiB and above,
requiring 45 bits. The modern peer predicate admits IOMMU remapping for
DMA/DMA-FQ domains. Therefore replacing
IOVAs with raw BAR addresses is not a justified fix: those addresses exceed the
declared outbound DMA contract.

Before the BIOS change, the host advertised eight high PCI windows, all starting at 16 TiB or above.
The MI50 root's low window was only 88 MiB. `pex_window` retains the actual parent
aperture and has no base override; its slot/BAR sizing cannot create a valid
32 GiB window below 16 TiB. The exact-board official manual documents lower
MMIOHBase choices, but any firmware change needs separate review and validation.
GRUB rollback cannot undo a persistent BIOS setting.

Initial PCI enumeration and final PEX placement differ within a boot. For the
failed boot, initial `15c/15d` high addresses must not replace the post-PEX
`153/154` addresses established by later AMD registration and the same-boot
sysfs snapshot. All still exceed the 44-bit boundary.

## Mapping-only result

The one-shot boot `28cb50dd-642c-40d8-a443-7b4ea3d9f889` completed with
kernel 1000 and P2P on. The normal P2P-off GRUB default remained intact and the
one-shot selection was consumed. Existing hard/soft-lockup all-CPU backtrace
controls were enabled by the diagnostic loader; no custom watchdog was added.

The separate native mapped-only process passed in 0.718 seconds and retired
with a clean canonical driver observation. It issued no remote GPU load or
peer-copy API. A private tracefs instance captured 90,830 raw bytes in 59 durable
chunks, with no drops, overruns, cap truncation or cleanup errors. All three
observed DMA resource maps have matching unmaps; the owned instance was removed.

The actual peer buffer joins as follows:

| Address domain | Observed value |
|---|---|
| Native owner GPU VA | `0x736080400000` |
| Underlying BO extent | 2 MiB for a 64 KiB HIP allocation |
| Post-PEX owner 97 BAR0 base | `0x153000000000` |
| Owner-local VRAM offset | `0x7fec00000` |
| Physical peer resource | `0x1537fec00000` |
| DMA address returned for importer 9a | `0xf2200000` |
| Planned peer PTE target for that VA | `0xf2200000` |

The local and peer VM-update records differ only by SYSTEM and SNOOPED in
their common flags (`0x4e1` versus `0x4e7`). The trace reports planned updates,
before backend-specific final adjustments; it is not a hardware PTE or TLP
readback. VM contexts distinguish the records but are not BDFs by themselves.
The physical address, importer, exact VA range and target addresses supply the
join. Native PID 1697 and trace VM-owner PID 11007 belong to different PID
namespaces and must not be compared numerically.

This evidence weakens a direct high-BAR truncation explanation: the observed
peer mapping uses an in-range IOVA, not the raw 45-bit BAR. It does not prove
the actual hardware access, endpoint ATS translation, IOMMU execution or root
complex return route. It is a new boot/process and cannot assign its IOVA to
the earlier failed transaction. A subsequent passive PCI snapshot shows ATS
enabled on both MI50s and all captured ACS redirect/translation-block controls
off; those capability bits do not identify an individual request's address type.

## ATS-off comparison and local compute controls

Boot `f5f1a94f-3914-4c1b-a10f-6970cf0e0695` used the same kernel and drivers,
with the supported one-shot `pci=noats` control. Actual PCI reads showed ATS
Enable- on both endpoints. The inspected ACS controls, complete post-PEX
resource arrays and DMA-FQ domain types matched the prior boot. Although the
parameter can affect ACS translation-block defaults in general, those captured
controls did not change here.

Metadata and mapping-only probes passed with clean driver windows. The fresh
mapping-only allocation again used physical `0x1537fec00000`, importer IOVA
`0xf2200000` and planned peer-PTE flags `0x4e7`. The original, unchanged
64 KiB remote-read binary then reproduced the same ATHUB/RAS failure on 9a.
The process aborted and retired in 5.873 seconds; no repeat or GPU workload
followed it.

Unlike the first failed attempt, the actual failing process was traced. Its
owner VA `0x7d3d6c200000` maps locally to `0x7fec00000`; reader 9a's resource
mapping returns `0xf2200000` for physical `0x1537fec00000`, and the planned peer
PTE uses that same IOVA. The trace is complete and loss-free. It contains no
matching peer-target invalidation or unmap before capture ended: successful
trace cleanup and retired userspace do not certify kernel/GPU resource cleanup.

The kernel reports RAS at 277.539921, failed overall reset at 301.125587, and
the same later BAR5 Completer-Abort write as before. These times share the
kernel-journal clock. Native/ftrace monotonic timestamps have a measured offset
from that clock; do not subtract the two domains to assert millisecond onset
ordering. A final kmsg marker brackets one endpoint only and does not establish
a constant offset for earlier records.

The secondary behavior differs substantially in this observation. Through the
durable kernel marker at 591.420692, **313.881 seconds after RAS**, the ATS-off
run has no DMAR invalidation timeout and no NIC watchdog. In an equal early
35.538-second post-RAS window, ATS-on had 13 invalidation timeouts and a NIC
watchdog; ATS-off had none. An explicit SysRq-l captured responses from all
112 CPUs. However, six TTM delayed-delete tasks subsequently waited more than
122 seconds on DMA fences. ATS-off therefore did not fix P2P, failed GPU
recovery or cleanup; it only avoided the observed host-wide cascade in this
single comparison.

After requesting a clean P2P-off reboot, the journal closes at 15:51:14 UTC and
the next boot's first record is at 16:05:17. Service shutdown reached the reboot
target in about 12 seconds and a 10-minute shutdown hardware watchdog was
configured. The interval after logger closure is not explained by these logs;
there is no fresh watchdog SEL record or final restart message proving whether
late kernel shutdown, reset or firmware caused the delay.

On the recovered P2P-off boot, a separate local-only control executes the
original kernel body, 64-by-256 geometry, pattern and full 16,384-word oracle
against each GPU's own memory. Both 9a and 97 pass, with successful cleanup,
clean driver windows and process retirement. This rules out a general failure
to execute that kernel on the corrected driver stack. No peer API exists in
the local control, and it is not a P2P qualification substitute.

## Successful lower-MMIO and identity-DMA comparison

The user changed BIOS MMIO High Base to 4T, leaving 1024G unchanged. On the
current boot the original read succeeds for all 12 directed MI50 links and all
eight MI50/XTX links. Each process checks the complete varying 64 KiB payload,
finishes cleanup and retires, with no driver errors in its observed window.
These include cross-switch/root paths; they are bounded access tests, not
sustained concurrency, collective or model coverage.

The actual successful 97→9a read has native namespace PID 2159, host TGID 11311
and owner VA `0x729cc4200000`. Do not substitute the adjacent earlier mapped-only
process's VA or VM contexts for this access witness. Owner BAR0 starts at
`0x93000000000`; the exported physical resource is `0x937fec00000`. The importer
DMA address and planned peer destination both equal that physical address,
within the declared 44-bit DMA contract. Local destination remains
`0x7fec00000`; local/peer flags remain `0x4e1`/`0x4e7`.

Native/ftrace monotonic ordering is map at 252.428505, peer update at
252.428508, kernel submission at 252.453851, successful reader completion at
252.462839 and full verification at 252.479526. Target invalidations at
252.549102/252.549112 precede matching resource unmap at 252.549171. Unlike the
complete failed trace, this trace supplies a target resource-retirement witness.
The public VM event is still a planned update, not hardware PTE readback or an
observed PCIe transaction.

The combined lower-BAR and supported identity-DMA route is a working mitigation
for these reads. BIOS lowering and DMA-domain changes were made together, so
neither is independently causal. The observations narrow the address/routing
hypothesis but do not establish the original faulty TLP, root-complex behavior
or final PTE attributes. ATS-off failure shows enabled ATS was not necessary.
No raw-BAR bypass, DMA-mask widening or blanket IOMMU-disable fix is justified.
IRQ remapping remains enabled. The firmware change persists across GRUB rollback;
the consumed one-shot did not promote this configuration to the normal default.

Next qualification is the captured-transfer mismatch below, then genuine
collective and model execution, followed separately by performance. Repeating
the known failing translated route unchanged is not needed for this checkpoint.

Independent review found no causal direction, bounds, pointer or lifetime defect
in the failing read. It did find a latent asynchronous-error cleanup lifetime
issue in the private harness; the new mapped-only version keeps its host source
buffer alive through resource cleanup and records both HSA policy variables.

## Captured production transport: separate outstanding mismatch

The first existing Llaminar production-path regression was run on the same
successful native-read boot, with original physical MI50s 97 and 9a explicitly
selected as ROCm ordinals 0 and 1:

```text
V2_Integration_CapturedPeerPayload_ROCm2_OriginalInventory
  CapturedTransferPeerAdmission.ROCmOriginalInventory
  coverage=complete
  ROCm:0 -> ROCm:1 replay=0 bytes=1 offset=1 first_bad_byte=1
```

CTest failed normally after 7.69 seconds; the complete driver bracket took
8.079 seconds and was clean. Both device contexts and the owned MPI process
retired. This is an output mismatch, with no observed repeat of ATHUB, GPU reset
or the host lockup. It does not revoke the direct-read results, and the direct
reads do not qualify this more demanding protocol.

The failing path is the actual captured PeerDevice channel round trip with five
warmups. It is not RCCL: the native-collective option is false. The first live
byte fails, while the preceding inactive guard byte passes. The subsequent
counted-transfer replay was not reached. Read-only review found consistent
source/binding offsets and an explicit one-byte copy path, but did not locate a
definite fixture defect. The current diagnostic omits actual/expected byte
values and intermediate payload observations. Capture, ordering, visibility
and kernel behavior remain open possibilities; no cause is assigned yet.

Preserved evidence is
`/tmp/qfn-p2p-identity-r911/captured-peer-admission/`: exact invocation and
physical visibility, CTest output, JUnit, summary and canonical driver journal.
The tested source is the r709 projection at
`f48c856105f6d1495a4c342ed2cc0c62b4d11273` plus its existing scoped admission-test
changes; build r710. No Llaminar production code was changed in this host
experiment. Keep the regression red until its cause is fixed and replayed;
do not declare P2P production-ready or promote the one-shot boot from this note.

After that focused repair, qualification should cover the retained native RCCL
parent/reentry and four-device captured epoch/gather tests, then the original
QFN admission and numerical workload. RCCL transport-selection logs are needed
to distinguish actual P2P from another native transport; a collective pass
alone does not prove its route. Measure performance separately with quiet,
matched workloads. Durable breadcrumb/fsync wall times are not link benchmarks.

## Evidence locations

- `/tmp/qfn-p2p-identity-r911/{mi50-directed-pairs,xtx-directed-pairs}/results.json`:
  12+8 successful directed links and per-process driver/retirement receipts.
- `/tmp/qfn-p2p-identity-r911/independent-read-correlation/REVIEW.md` and
  `read-trace.txt`: actual successful-process address and lifetime join.
- `/tmp/qfn-p2p-identity-r911/captured-peer-admission/`: outstanding captured
  byte mismatch, clean driver observation and complete owned test retirement.
- `/tmp/qfn-kernel617-qualification-r889/` and
  `/tmp/qfn-kernel617-sg-qualification-r905/`: source, patch, package and build
  identities for ABI 999 and 1000 respectively.
- `/tmp/qfn-current-amd-driver-r894/`: local AMD package and DCE patch provenance.
- `/tmp/qfn-current-nvidia-driver-r893/`: exact NVIDIA package closure.
- `/tmp/qfn-nvidia-pex617-qualification-r891/`: preserved original PEX 0.1.2
  source, stock-6.17 build evidence and source-preservation receipt.

- `/tmp/qfn-p2p-root-r906/`: pre-boot checks, native allocation baselines,
  one-shot boot receipts and the original failed native probe observation.
- `/var/tmp/llaminar-p2p-r906/failures/3e7241d5-a22d-49ed-b0db-a6bc20890979-r2/`:
  host-durable full failed-boot kernel/journal and BMC records.
- `/tmp/qfn-rocm-p2p-failure-r907/failed-boot/`: local kernel-log copy and report.
- `/tmp/qfn-rocm-p2p-failure-r907/{source-mapping-review,reset-review,probe-review}/`:
  exact source/API audits and their limits.
- `/tmp/qfn-rocm-p2p-failure-r907/{aperture-review,bios-aperture-research}/`:
  actual host resource constraints and primary board documentation.
- `/tmp/qfn-rocm-p2p-failure-r907/{mapped-only-probe,dma-resource-trace}/`:
  native diagnostic, actual complete mapping/PTE trace, startup/cleanup smoke
  checks and schema validation on the running kernel.
- `/tmp/qfn-rocm-p2p-failure-r907/{mapped-correlation,pte-review,pte-schema-review}/`:
  independent address joins, exact source interpretation and trace limits.
- `/tmp/qfn-rocm-p2p-failure-r907/mapped-pci-snapshot/report.json`:
  same-boot ATS/ACS, IOMMU domain and resource snapshot.
- `/tmp/qfn-rocm-p2p-noats-r908/`: ATS-off boot, unchanged resource/ACS
  comparison, clean mapping-only control, actual failed-read trace and kernel
  recovery/CPU-stack evidence. `reset-comparison/` owns the matched windows.
- `/tmp/qfn-p2p-safe-compute-r909/`: recovered P2P-off boot and both successful
  same-device compute controls; source/build identity is retained in the
  r908 `local-compute-probe/` directory.

The first attempted historical journal export used a hyphenated UUID, which
`journalctl -b` rejected. That failed read is retained. The corrected export
uses the 32-hex boot ID, checks command success and contains 751,118 bytes of
kernel evidence. Empty output from the failed read is not absence-of-error proof.
