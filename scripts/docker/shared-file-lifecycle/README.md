# Native shared-file partial creation

The compiler builder proves the complete selected NCCL/RCCL `shmutils.cc` with
real libc file/mapping operations. It reconstructs its literal original by
reversing only the new cleanup patch. Both variants return success only after
checking all 20 replays of the expected observations. GPU entrypoints abort.

The shared fixture covers fresh/explicit creator names, sizing/mapping failure,
failed-open metadata allocation, preexisting caller files/symlinks, descriptor
zero, failed pre-mapping attachment, live neighboring data, positive multi-peer
refcounts and source-first mapped attachment retirement. Original-only lost
names/allocations are captured and cleaned by their exact recorded identity.
The diagnostic cleanup does not imply production cleanup succeeded.

NCCL's existing installer runs `proof.py compile --backend nccl` with its actual
source build, SDK, compiler, architecture contract, DSO and receipt. RCCL's CMake
proof target runs `compile --backend rccl` using its selected compile database,
SDK/compiler/architecture, DSO and receipt; compiler-produced native **and probe**
dependencies supply its depfile. The existing canonical receipt includes the new
patch; unpatched builds cannot emit this proof. Neither script builds kernels.

Installed `verify --backend {nccl,rccl} --report REPORT --library LIBRARY` reads
only small policy/report/DSO bytes. NCCL additionally passes `--receipt RECEIPT`.
It never invokes the recorded compiler or SDK, loads the DSO or queries a device.
The explicit family Unit/preflight registrations require complete evidence.
`NativeSharedFileProofPackaging` is always device-free and tests malformed,
missing, stale, incomplete and interrupted evidence and package binding.

For private host development use `source-only` with `--evidence-dir DIRECTORY`,
plus NCCL's `--source BUILD_SOURCE --sdk SDK`, or RCCL's
`--rccl-build BUILD --sdk SDK --compiler CLANGXX --architecture gfx906`.
`--candidate-source FILE` is allowed only in this diagnostic mode. Its JSON is
always `certificate=false`; `--report`, `--library` and `--depfile` are forbidden.

This proof covers **host named-file partial creation only**. GPU registration
failure, failed cleanup retry/owner retention, CuMem/uncached IPC, full PMA root
admission, opaque resource ownership and complete communicator readiness retain
separate qualification requirements. In particular, no host-only development
run certifies an existing unpatched DSO.
