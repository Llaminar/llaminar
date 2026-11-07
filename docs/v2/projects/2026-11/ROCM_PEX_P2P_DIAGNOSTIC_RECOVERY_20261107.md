# ROCm PEX P2P diagnostic and recovery

This is a host investigation and recovery record, not a tuning dashboard or a
passing inference receipt. The user reports that enabling PCIe P2P previously
caused AMD-driver/kernel lockups, including during boot.

## Current status

The current boot is `2ef86b85-de24-4e81-8796-d5a21af56178`, kernel
`6.17.0-1000-generic`, with P2P enabled and identity DMA (`iommu=pt`). The user
changed BIOS MMIO High Base to **4T** and left the **1024G** setting unchanged.
All 20 directed AMD remote-read links pass: 12 MI50-to-MI50 directions and eight
MI50/XTX directions. Every process verifies all 16,384 words (64 KiB), cleans up,
retires and has a clean driver window. The traced nearest-pair read additionally
has matching target VM invalidations and DMA-resource unmap.

This qualifies the bounded native read under the combined lower-BAR and identity-
DMA configuration, not BIOS alone or an exact hardware root cause. Collectives,
full model execution and performance remain unqualified. The first captured
PeerDevice round trip reports a one-byte mismatch with clean driver observations
and process retirement; it is a separate outstanding qualification failure.
The earlier translated
DMA configuration still has retained failures with ATS on and off; see the
[research note](ROCM_PEX_P2P_MMIO_IOMMU_RESEARCH_20261107.md).

The successful identity-DMA entry was a consumed **one-shot**. The normal default
remains ABI 999/P2P-off; no permanent promotion has occurred.

Recovery configuration is installed and GRUB syntax validation passed:

- Normal default: **Llaminar recovery: Linux 6.17.0-999, AMD P2P OFF**.
- Additional entry: **Llaminar recovery: Linux 6.17.0-999, GPU drivers DISABLED**.
- GRUB displays the menu for 10 seconds; a failed ordinary boot gets 15 seconds.
- Persistent journaling synchronizes every second, with a 4 GiB limit and an
  8 GiB free-space reserve. Existing journals occupied approximately 2.2 GiB.
- The actual persistent panic-log backend is ACPI **ERST**. It remains selected;
  EFI pstore was already ignored because ERST owns that role.
- Four diagnostic entries exist: ABI 1000 P2P-off, P2P-on, P2P-on with
  `pci=noats`, and P2P-on with identity DMA (`llaminar-p2p-trial-identity`).
  They use the delayed-load service. Each executed one-shot has
  been consumed; no diagnostic entry became the normal default.

These are known-bootable recovery choices, not certification that ABI 999 can
complete every model admission. Its separate large-DMA mapping failure is
documented in [the activation record](QFN_HOST_KERNEL_DRIVER_ACTIVATION_20261107.md).

## Recovery from a trial

The trial uses GRUB's `next_entry`, which is cleared before starting Linux.
Rehearsal and subsequent actual boots confirmed consumption and preservation
of the P2P-off normal default. `grub-editenv /boot/grub/grubenv list` may show
`next_entry=` with an empty value; it need not produce entirely empty output.
The environment resides on ordinary NVMe ext4 without a GRUB abstraction layer.

If automatic recovery does not occur, reset the host through its normal power
or BMC controls. In the visible GRUB menu choose the P2P-off entry. If GPU-driver
initialization itself still prevents boot, select **GPU drivers DISABLED**.
That entry deliberately has no `/dev/kfd`; GPU-dependent containers may fail to
start, while the host console and networking remain the recovery boundary.

From a working host shell, cancel a pending one-shot selection with:

```sh
sudo grub-editenv /boot/grub/grubenv unset next_entry
```

The retained Ubuntu advanced menu also contains the earlier 6.14 kernels. Do
not delete old images or the new known-bootable recovery entry during this
investigation. Configuration preimages are in
`/var/tmp/llaminar-p2p-r906/recovery-preimages/`; the manifest records which files
were previously absent. Restoring GRUB input files requires regeneration and
syntax validation of the resulting configuration.

## Diagnostic boot design

The normal boot initializes AMD at approximately 11.2–15.3 seconds, before PID 1
arms the IPMI watchdog at approximately 16.9 seconds. Merely relying on the
existing 60-second hardware watchdog therefore leaves an early-boot gap.

The installed diagnostic entries suppress automatic AMD alias loading with
`modprobe.blacklist=amdgpu`. A host-owned service then explicitly loads AMD after
persistent journal flush and verification that PID 1 owns `/dev/watchdog`.
It preserves the existing `pex_window` soft dependency and its resource layout.
The service runs before Docker because this workspace maps `/dev/kfd` and
`/dev/dri`. The test must not depend on the workspace container to load AMD.
The trial also excludes `gpu-manager` as an independent automatic loader.

PID 1 owns an independent 90-second running-job deadline during driver
initialization. A blocked `modprobe` cannot cancel it. This covers the case where PID 1 continues
feeding the hardware watchdog while a driver task is stuck. Panic/oops and
NMI/soft-lockup controls and their existing all-CPU backtrace options apply only
to the diagnostic boot. Warnings are not
converted into panics. Narrow topology and KFD memory-map debug sites supplement
ordinary driver logs; no device-wide tracing or broad driver debug flood is
required for the first attempt. Docker requires successful driver initialization
and a matching durable ready record on a diagnostic boot; it cannot start merely
because a failed driver job stopped waiting. On ordinary boots the service's
condition is false and its Docker gate performs no work.

The native systemd rehearsal used a harmless sleep, a one-second job deadline
and `JobTimeoutAction=none`. It returned the expected timeout after 1.127 seconds
while the service was still starting. The real service's actual loaded 90-second
deadline and recovery actions were checked. Its ordinary-boot condition skipped
successfully. No panic or reboot was induced by this rehearsal.

ERST stores panic snapshots, not a continuous console: this kernel does not
enable `CONFIG_PSTORE_CONSOLE`. A total hardware lockup or power loss may leave
only the last journal writes, and watchdog recovery is not proof of a captured
panic. BMC/local-console observation remains valuable. No destructive test panic
has been induced to certify firmware persistence.

## P2P test scope

All four MI50s share the `85:00/86:00` upstream tree and IOMMU group 36. Full PCI
configuration reads show ACS request/completion redirection cleared on their
observed paths. The nearest pair is `97:00.0` and `9a:00.0`, under `93:00.0`.
The first experiment retained that ACS policy and translated DMA-FQ domains;
the successful experiment later selected identity DMA explicitly.

The later `pci=noats` control retains translated DMA-FQ domains. Actual PCI
snapshots proved ATS disabled and the inspected ACS controls and post-PEX
resources unchanged. It did not repair the peer read. Do not repeat either
failed configuration unchanged or turn the diagnostic entry into a default.
The user's subsequent BIOS 4T change is persistent and is **not** reversed by the
GRUB one-shot recovery mechanism. The 1024G setting was unchanged; preserve the
firmware preimage independently of boot selection.

The 7900 XTX at `49:00.0` resides under the other root, with different ACS
controls. `amdgpu.pcie_p2p` is global: enabling it can affect all AMD KFD topology
nodes even when a userspace test restricts ROCm visibility to two MI50s. This
distinction matters when interpreting failures in driver or runtime discovery.

Native discovery, peer enablement and direct remote-pointer reads now pass in
all 20 directed AMD links on the identity-DMA boot. A generic copy alone would
not establish that result. Per-call durable breadcrumbs and full output oracles
are retained. RCCL and production graph tests still follow this native-access
checkpoint; they have not been qualified by these reads. The initial captured
peer-channel test is currently red on byte accuracy, as recorded in the research
note. No permanent boot promotion has been made.

## Evidence

- `/tmp/qfn-p2p-identity-r911/{mi50-directed-pairs,xtx-directed-pairs}/results.json`:
  all 20 directed native reads, clean windows and process retirement.
- `/tmp/qfn-p2p-identity-r911/independent-read-correlation/REVIEW.md`:
  actual successful access, mapping/PTE and target-unmap comparison.

- `/tmp/qfn-p2p-root-r906/recovery-install/`: installed files and command receipts.
- `/tmp/qfn-p2p-root-r906/kernel-install-r2/`: six packages, 15 module identities,
  complete header/initramfs closure and preserved recovery/default validation.
- `/tmp/qfn-p2p-root-r906/trial-install/` and `deadline-rehearsal/`: actual unit,
  GRUB and non-rebooting job-timeout checks.
- `/var/tmp/llaminar-p2p-r906/RECOVERY-PREPARED.json`: host-durable receipt.
- `/tmp/qfn-p2p-recovery-audit-r906/`: original GRUB, initramfs, watchdog, journal
  and firmware-store audit; no GPU execution.
- `/tmp/qfn-rocm-p2p-audit-r906/`: actual PCI/ACS/KFD inventory, driver/source
  findings, targeted dynamic-debug selectors and private native probe.

The primary [GRUB one-shot selection documentation](https://www.gnu.org/software/grub/manual/grub/html_node/next_005fentry.html)
describes the required environment-block write. The
[Linux PCI P2PDMA documentation](https://docs.kernel.org/driver-api/pci/p2pdma.html)
explains why topology and routing must be checked rather than inferred from a
driver capability flag.
