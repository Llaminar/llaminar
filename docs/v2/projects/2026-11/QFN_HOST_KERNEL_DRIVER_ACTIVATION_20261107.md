# QFN host kernel and driver activation

This is a host-maintenance qualification record, not a release certificate or
tuning dashboard. The user requested current GPU drivers, removal of host
ROCm/CUDA SDKs, and an updated 6.x HWE kernel with the Intel IOMMU recovery fix.
The user explicitly approved the reboot. Container compiler SDKs are unchanged.

## Initial activation

The host rebooted on 2026-11-07. Boot ID changed from
`c3caf5ee-e879-46a1-a6ac-619051c016ed` to
`39889ba3-4ea8-4465-a9fc-d2e700fd8090`. Post-boot observations authenticate the
loaded modules, rather than treating installed packages as runtime evidence.

| Component | Activated version / identity |
|---|---|
| Kernel | `6.17.0-999-generic`, package `6.17.0-999.42+llaminar1` |
| AMD | `7.1.9.31600000`, srcversion `BFEAE8D9287FE23B9907691` |
| NVIDIA open | `595.99.02`, srcversion `919309EF6954372EFADD77E` |
| PEX window | `0.1.2`, srcversion `7D4D55A110866DB75810F63` |

The kernel starts from published Ubuntu 6.17.0-42.42 HWE source, commit
`7cb9b24dc95cd7faa58265648bd74bd17b03f55d`, with upstream Intel IOMMU fix
[`fe89277c9ceb`](https://github.com/torvalds/linux/commit/fe89277c9ceb0d6af0aa665bcf24a41d8b1b79cd).
The stock generic configuration is retained apart from the version signature;
Rust, BTF and DWARF were not removed. A separate packaging correction repairs
the HWE generic-header link to its actual Rust library package. ABI 999 keeps
the custom kernel distinct from Ubuntu's stock packages. This is a 6.x HWE
selection, not a claim to be the latest overall rolling HWE release.

The AMD local DCE correction and package holds remain. All three external
module families are installed for both the prior 6.14.0-37 kernel and the new
kernel. Previous 6.14 kernels and their GRUB entries remain available. Kernel
command-line arguments, PEX configuration and the existing GRUB default policy
were preserved; only the obsolete 6.17 package pin was removed.

The host's 88 ROCm user-space packages and 39 remaining CUDA SDK/profiler
packages were removed through explicit package lists, without autoremove.
GPU driver runtime libraries and NVIDIA container integration remain installed.
Package rollback snapshots and configuration backups are retained on the host.

## Post-boot checks

Native host checks confirm clean dpkg audit and active Docker/containerd.
All four MI50s, the 7900 XTX and both RTX 3090s are bound to their expected
drivers. Every GPU retains a 32 GiB VRAM BAR window. The original early-init PEX
setup reports successful application before GPU initialization. Container
`rocminfo` enumerates the five actual AMD devices with their correct gfx906 or
gfx1100 identities; `nvidia-smi` enumerates both 3090s on driver 595.99.02.
The inherited `HSA_OVERRIDE_GFX_VERSION` is removed for qualification.

Twelve focused CTest registrations pass with no skips and clean driver windows:

- Six CUDA/ROCm workspace-declaration, retained ticket-ownership and MoE ticket
  segment checks: 16.34 seconds including their observation wrapper.
- Six CUDA/ROCm actual Current-to-Future source-ordering and consumed-Future
  zero/small-local-cache checks: 12.35 seconds including the wrapper.

These checks establish native execution and focused lifetime/read-ahead
behavior after the reboot. The larger original ROCm4 model
admission/recreation run subsequently **failed**. That failure required the
separate kernel correction below before further model-admission qualification.

## Large mapped-buffer failure

At 13:36:52 UTC the original ROCm4 RAM/Off case failed a
5,391,777,792-byte `hipHostMalloc(Mapped)` on ROCm:1 during serving-family
preparation. The host still had more than 500 GB available RAM. Kernel logs
first report a GPUVM mapping failure, followed by repeated Intel IOMMU
already-populated PTE warnings. The owned MPI guardian was terminated and
CTest retired with failure; no further GPU tests were launched.

The warning volume wrapped the dmesg buffer, so its canonical driver observer
correctly rejected that window. The host journal retains the earlier sequence
in `POSTBOOT-MODEL-FAILURE.kernel.log`; the first DMA mapping error is at boot
monotonic 681.336158 seconds, and the first recorded PTE warning at
682.657925 seconds. A captured dmesg tail alone is not the complete incident.

The application allocation method matches the clean feature checkpoint. The
large request is geometric backing for the persistent mapped ticket arena;
its logical M512 ticket payload is approximately 10.5 MB. The failing backing
is smaller than the previously observed successful 6,197,084,160-byte backing.
Declaration-only allocation removal changed the growth sequence, so exact
physical-allocation-history equality is not claimed.

The exact 6.14 and 6.17 source both contain a plausible matching defect:
segment-boundary padding can overflow a 32-bit scatterlist length. A short
positive mapping then releases its IOVA without unmapping the installed
portion. The [upstream discussion](https://www.mail-archive.com/intel-gfx@lists.freedesktop.org/msg381888.html)
describes these constraints. An actual-function CPU regression reproduces the
old behavior for a valid hypothetical layout of the failed allocation size;
the upstream overflow checks reject it before mapping. This proves the code
defect, **not** that the failed runtime used that physical segment layout.

A separately built kernel candidate retains the timeout-recovery patch, adds
upstream overflow guards, and applies a local DRM page-array partition bound.
The local bound preserves total buffer size and smaller device limits while
keeping each input segment below half the unsigned length range. It adds SG
entries for very large contiguous runs; no zero-cost performance claim is made.
The actual DRM/SG-builder/guarded-IOMMU CPU composition passes original/candidate
tests for 16 geometries, including the failed extent, prior larger extent,
fragmentation, smaller limits, chaining and allocation-failure cleanup.
Physical pages and hardware operations are explicitly mocked.

## Corrected kernel activation and focused native evidence

The correction is now built and installed as `6.17.0-1000-generic`, package
`6.17.0-1000.42+llaminar2`. It retains the QI-recovery and Rust packaging fixes,
adds upstream guard commit `1c8f6308e557f32412e9d4678b3f029f47f6d9aa`, and limits
DRM input segments as described above. All 15 AMD/NVIDIA/PEX module products
were validated against this ABI. Complete header, Rust and initramfs closure
passed; ABI 999 and earlier 6.14 recovery kernels remain installed.

On the corrected P2P-off boot, separate native mapped-allocation/alias/free
processes passed for 4 KiB, 5,391,777,792 bytes and 6,197,084,160 bytes, with
clean driver windows and observed process retirement. These runs authenticate
the requested extents and original HSA allocation policy; they do not establish
identical physical scatterlists to the earlier failure or qualify the original
complete ROCm4 source-admission run, which remains pending.

The recovered P2P-off boot `57abad18-10ea-4f56-a3a2-8c182a106481` also passed
full 64 KiB local-memory verification independently on MI50s 9a and 97, with
clean driver windows, successful cleanup and retirement. These controls execute
the original load/store kernel, not just allocation or host transfers.

The current boot is `2ef86b85-de24-4e81-8796-d5a21af56178`, kernel 1000 with P2P
on and identity DMA (`iommu=pt`). The user changed BIOS MMIO High Base to 4T,
leaving 1024G unchanged. All 20 directed AMD reads pass (12 MI50 links plus eight
MI50/XTX links), each verifying all 16,384 varying words with clean driver
windows, cleanup and process retirement. The traced 97→9a access uses identical
physical/DMA/peer planned addresses `0x937fec00000` and has target invalidations
followed by matching resource unmap.

This establishes a bounded working mitigation combining lower MMIO and identity
DMA, not BIOS-alone causality or the exact original hardware failure. Earlier
translated-DMA peer reads failed with ATS on and off; that is separate from the
large RAM scatterlist defect. Collectives, full model execution and performance
on the new P2P configuration remain unqualified. See the
[P2P research note](ROCM_PEX_P2P_MMIO_IOMMU_RESEARCH_20261107.md) for exact
versions, committed patch files and both failure and success witnesses. The
first captured peer-channel test subsequently reports a one-byte mismatch with
a clean driver window and owned-process retirement; production P2P remains
unqualified. The initial identity-DMA trial was a consumed one-shot. The user
then authorized promotion of the unchanged `llaminar-p2p-trial-identity` entry to
`GRUB_DEFAULT`: ordinary reboot now selects kernel 1000/P2P-on/`iommu=pt`.
Verification is `/tmp/qfn-p2p-default-r915/applied.json`; no reboot was performed
by that change. Menu visibility remains 10 seconds; ABI 999/P2P-off and drivers-
disabled entries remain available for manual recovery. Default promotion does
not certify captured transport or native model inference.

Meanwhile, the scoped ROCm4 RAM/Off Current and KnownNext numerical declarations
build successfully, and seven focused device-free registrations pass with no
skips. Their actual model numerical panels have not run on this host state.

## Boundaries and retained evidence

The IOMMU patch repairs timeout recovery for even-indexed completion entries.
An offline regression exercising the actual original and patched function
reproduces the original failure and passes 524,309 patched cases. This does not
prove the cause of the initial device-TLB timeout, eliminate AMD IH2 overflow,
or qualify indefinite host uptime. The separate
[IH2 investigation](QFN_IH2_OVERFLOW_INVESTIGATION_20261107.md) retains those
distinctions and the independently qualified Llaminar declaration-allocation fix.

Old instruction-offset observers must be reauthenticated against the newly
loaded module before use. A matching driver version alone is insufficient.
No production throughput comparison is inferred from these focused checks.
The complete preflight gate remains outside this iteration's requested scope.

Durable host records are under `/var/tmp/llaminar-driver-update-r895/`:
`REBOOT-HANDOFF.json`, `REBOOT-SCHEDULED.json`, `POSTBOOT-IDENTITY.json`,
installation logs, configuration backups and the explicitly labeled repacked
rollback archives. Workspace receipts are under:

- `/tmp/qfn-kernel617-qualification-r889/`: kernel build and package evidence.
- `/tmp/qfn-iommu-recovery-regression-r896/`: actual-function recovery regression.
- `/tmp/qfn-host-driver-update-r895/postboot-identity/`: complete post-boot report.
- `/tmp/qfn-postboot-native-r899/` and `/tmp/qfn-postboot-readahead-r900/`:
  focused CTest results and complete driver windows.
- `/tmp/qfn-postboot-model-r901/`: failed model run, complete host journal and
  exact owned-session stop receipt.
- `/tmp/qfn-iommu-sg-mapping-r902/`: source comparison, upstream guard patch,
  actual-function CPU regression and local DRM candidate.
- `/tmp/qfn-drm-sg-builder-regression-r905/`: actual source-body composition
  with explicit CPU boundary doubles; no native mapping claim.
- `/tmp/qfn-rocm4-numerical-build-r903/`: coherent test-owner build and seven
  focused device-free passes.
- `/tmp/qfn-kernel617-sg-qualification-r905/`: isolated ABI 1000 candidate build,
  completed package/header validation and retained compiler-object cache.
- `/tmp/qfn-p2p-root-r906/kernel-install-r2/` and `mapped-*-off/`:
  actual ABI 1000 installation and focused mapped-buffer controls.
- `/tmp/qfn-p2p-safe-compute-r909/`: corrected P2P-off recovery identity and
  both local-memory compute controls.

- `/tmp/qfn-p2p-identity-r911/{mi50-directed-pairs,xtx-directed-pairs}/results.json`:
  all 20 native read directions and clean lifetime/driver observations.
- `/tmp/qfn-p2p-identity-r911/independent-read-correlation/REVIEW.md`:
  successful actual-process mapping/access/unmap comparison and causal limits.
