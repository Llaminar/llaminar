/** @file NativeAuxiliaryQualifiedLimits.h
 * @brief Conservative supported-backend allowances for typed lookup ordering families.
 *
 * Original environment classes each passed 33 native lifetimes and 660 uses.
 * Clean loader-policy classes add 21 lifetimes and 420 uses on identical native
 * providers/geometry. C0/mixed/partial controls retain their original evidence.
 * Representative checks repeat the original 12-event creator on both backends.
 * Evidence describes representative native runs; supported backend and typed
 * resource family select allowances without exact software/hardware fingerprints.
 * Source review preserves the same native calls and ProcessRuntime retention;
 * Remote owner graph rows add an inclusive conservative 1 GiB per started 1024
 * declared nodes. The 69-node/448-byte observations are evidence, never support
 * ceilings; planned instance multiplicity contributes additively to original PMA.
 * These explicit conservative ceilings are opaque retained commitments, not
 * exact allocation sizes or automatically fitted peaks. They remain charged
 * through ProcessRuntime. The read-ahead side/parent families each passed 40
 * native lifetimes, 800 captured replays and three partial-creator controls on
 * CUDA and ROCm. Their allowances deliberately exceed observed heap/queue
 * growth; ordinary graph storage remains independently charged. These family
 * allowances do not depend on application, runtime or hardware fingerprints.
 * Unmeasured programs cannot borrow another family's allowance.
 */
#pragma once
#include "NativeAuxiliaryResourceCatalog.h"
#include <array>
namespace llaminar2::detail {
/** @brief Supported backend/resource families own the stated units through their original physical lifetime. */
inline constexpr std::array<NativeAuxiliaryQualifiedClass,18> kNativeAuxiliaryQualifiedClasses{{
    NativeAuxiliaryQualifiedClass{
        .backend = DeviceType::CUDA,
        .program = NativeAuxiliaryResourceProgram::ordering(NativeOrderingEventProgram::pooledOwner(false)),
        .hostBytes = 1048576,
        .deviceBytes = 2097152,
        .evidence = "b82fa882e6bb6b7a5175df363bd1f591175c8754000afbbd2c94ac943d7fd7ce",
        .lifetime = NativePhysicalLifetime::ProcessRuntime,
    },
    NativeAuxiliaryQualifiedClass{
        .backend = DeviceType::CUDA,
        .program = NativeAuxiliaryResourceProgram::ordering(NativeOrderingEventProgram::pooledOwner(true)),
        .hostBytes = 2097152,
        .deviceBytes = 2097152,
        .evidence = "b82fa882e6bb6b7a5175df363bd1f591175c8754000afbbd2c94ac943d7fd7ce",
        .lifetime = NativePhysicalLifetime::ProcessRuntime,
    },
    NativeAuxiliaryQualifiedClass{
        .backend = DeviceType::CUDA,
        .program = NativeAuxiliaryResourceProgram::ordering(NativeOrderingEventProgram::pooledCoordinator(false)),
        .hostBytes = 1048576,
        .deviceBytes = 2097152,
        .evidence = "b82fa882e6bb6b7a5175df363bd1f591175c8754000afbbd2c94ac943d7fd7ce",
        .lifetime = NativePhysicalLifetime::ProcessRuntime,
    },
    NativeAuxiliaryQualifiedClass{
        .backend = DeviceType::CUDA,
        .program = NativeAuxiliaryResourceProgram::ordering(NativeOrderingEventProgram::pooledCoordinator(true)),
        .hostBytes = 2097152,
        .deviceBytes = 2097152,
        .evidence = "b82fa882e6bb6b7a5175df363bd1f591175c8754000afbbd2c94ac943d7fd7ce",
        .lifetime = NativePhysicalLifetime::ProcessRuntime,
    },
    NativeAuxiliaryQualifiedClass{
        .backend = DeviceType::CUDA,
        .program = NativeAuxiliaryResourceProgram::ordering(NativeOrderingEventProgram::pooledSource()),
        .hostBytes = 1048576,
        .deviceBytes = 2097152,
        .evidence = "b82fa882e6bb6b7a5175df363bd1f591175c8754000afbbd2c94ac943d7fd7ce",
        .lifetime = NativePhysicalLifetime::ProcessRuntime,
    },
    NativeAuxiliaryQualifiedClass{
        .backend = DeviceType::CUDA,
        .program = NativeAuxiliaryResourceProgram::ordering(NativeOrderingEventProgram::lookupSubmissionRetirement()),
        .hostBytes = 524288,
        .deviceBytes = 2097152,
        .evidence = "b82fa882e6bb6b7a5175df363bd1f591175c8754000afbbd2c94ac943d7fd7ce",
        .lifetime = NativePhysicalLifetime::ProcessRuntime,
    },
    NativeAuxiliaryQualifiedClass{
        .backend = DeviceType::ROCm,
        .program = NativeAuxiliaryResourceProgram::ordering(NativeOrderingEventProgram::pooledOwner(false)),
        .hostBytes = 1048576,
        .deviceBytes = 2097152,
        .evidence = "b82fa882e6bb6b7a5175df363bd1f591175c8754000afbbd2c94ac943d7fd7ce",
        .lifetime = NativePhysicalLifetime::ProcessRuntime,
    },
    NativeAuxiliaryQualifiedClass{
        .backend = DeviceType::ROCm,
        .program = NativeAuxiliaryResourceProgram::ordering(NativeOrderingEventProgram::pooledOwner(true)),
        .hostBytes = 2097152,
        .deviceBytes = 2097152,
        .evidence = "b82fa882e6bb6b7a5175df363bd1f591175c8754000afbbd2c94ac943d7fd7ce",
        .lifetime = NativePhysicalLifetime::ProcessRuntime,
    },
    NativeAuxiliaryQualifiedClass{
        .backend = DeviceType::ROCm,
        .program = NativeAuxiliaryResourceProgram::ordering(NativeOrderingEventProgram::pooledCoordinator(false)),
        .hostBytes = 1048576,
        .deviceBytes = 2097152,
        .evidence = "b82fa882e6bb6b7a5175df363bd1f591175c8754000afbbd2c94ac943d7fd7ce",
        .lifetime = NativePhysicalLifetime::ProcessRuntime,
    },
    NativeAuxiliaryQualifiedClass{
        .backend = DeviceType::ROCm,
        .program = NativeAuxiliaryResourceProgram::ordering(NativeOrderingEventProgram::pooledCoordinator(true)),
        .hostBytes = 2097152,
        .deviceBytes = 2097152,
        .evidence = "b82fa882e6bb6b7a5175df363bd1f591175c8754000afbbd2c94ac943d7fd7ce",
        .lifetime = NativePhysicalLifetime::ProcessRuntime,
    },
    NativeAuxiliaryQualifiedClass{
        .backend = DeviceType::ROCm,
        .program = NativeAuxiliaryResourceProgram::ordering(NativeOrderingEventProgram::pooledSource()),
        .hostBytes = 1048576,
        .deviceBytes = 2097152,
        .evidence = "b82fa882e6bb6b7a5175df363bd1f591175c8754000afbbd2c94ac943d7fd7ce",
        .lifetime = NativePhysicalLifetime::ProcessRuntime,
    },
    NativeAuxiliaryQualifiedClass{
        .backend = DeviceType::ROCm,
        .program = NativeAuxiliaryResourceProgram::ordering(NativeOrderingEventProgram::lookupSubmissionRetirement()),
        .hostBytes = 524288,
        .deviceBytes = 2097152,
        .evidence = "b82fa882e6bb6b7a5175df363bd1f591175c8754000afbbd2c94ac943d7fd7ce",
        .lifetime = NativePhysicalLifetime::ProcessRuntime,
    },
    NativeAuxiliaryQualifiedClass{
        .backend = DeviceType::CUDA,
        .program = NativeAuxiliaryQualifiedClass::reviewedRemoteProgram({0xf2,0x02,0x3a,0x13,0x18,0xd2,0x26,0x72,0x86,0x1b,0xd6,0xa4,0xf9,0x58,0xc7,0x2c,0x17,0x37,0x9e,0x11,0x64,0xfd,0x92,0xb0,0x2a,0x5d,0x48,0x8c,0xa7,0xd4,0xfc,0x21}, {69,68,GPUFlatGraphLimits::bit(GPUFlatGraphNodeKind::Kernel)}),
        .hostBytes = std::size_t{1}<<30,
        .deviceBytes = 0,
        .evidence = "c0d2b756bc9c597cf592a3627fe455099f3a5a64695e3378072fb891477b7b5e",
        .lifetime = NativePhysicalLifetime::ProcessRuntime,
        .remoteFamily = NativeAuxiliaryRemoteFamilyScope{PooledRemoteGraphProgramClass::familyVersion(),
            {69,68,GPUFlatGraphLimits::bit(GPUFlatGraphNodeKind::Kernel)}, {3,448,8}},
    },
    NativeAuxiliaryQualifiedClass{
        .backend = DeviceType::ROCm,
        .program = NativeAuxiliaryQualifiedClass::reviewedRemoteProgram({0xf2,0x02,0x3a,0x13,0x18,0xd2,0x26,0x72,0x86,0x1b,0xd6,0xa4,0xf9,0x58,0xc7,0x2c,0x17,0x37,0x9e,0x11,0x64,0xfd,0x92,0xb0,0x2a,0x5d,0x48,0x8c,0xa7,0xd4,0xfc,0x21}, {69,68,GPUFlatGraphLimits::bit(GPUFlatGraphNodeKind::Kernel)}),
        .hostBytes = std::size_t{1}<<30,
        .deviceBytes = 0,
        .evidence = "c0d2b756bc9c597cf592a3627fe455099f3a5a64695e3378072fb891477b7b5e",
        .lifetime = NativePhysicalLifetime::ProcessRuntime,
        .remoteFamily = NativeAuxiliaryRemoteFamilyScope{PooledRemoteGraphProgramClass::familyVersion(),
            {69,68,GPUFlatGraphLimits::bit(GPUFlatGraphNodeKind::Kernel)}, {3,448,8}},
    },
    // One side queue and two ordering events, with headroom for native pool
    // granularity and lazy creation. Both backends retain the same portable
    // units; these are conservative commitments, not fitted allocation peaks.
    NativeAuxiliaryQualifiedClass{
        .backend = DeviceType::CUDA,
        .program = NativeAuxiliaryResourceProgram::readAheadSide(),
        .hostBytes = 4u * 1024u * 1024u,
        .deviceBytes = 16u * 1024u * 1024u,
        .evidence = "220fa9fcb0cd4b0fc31d0b25c0215941eba3bafe15b55213ce4681336d438279",
        .lifetime = NativePhysicalLifetime::ProcessRuntime,
    },
    NativeAuxiliaryQualifiedClass{
        .backend = DeviceType::ROCm,
        .program = NativeAuxiliaryResourceProgram::readAheadSide(),
        .hostBytes = 4u * 1024u * 1024u,
        .deviceBytes = 16u * 1024u * 1024u,
        .evidence = "220fa9fcb0cd4b0fc31d0b25c0215941eba3bafe15b55213ce4681336d438279",
        .lifetime = NativePhysicalLifetime::ProcessRuntime,
    },
    // The independent parent terminal uses the established one-event units.
    // Destroying its handle does not prove that the runtime pool released it.
    NativeAuxiliaryQualifiedClass{
        .backend = DeviceType::CUDA,
        .program = NativeAuxiliaryResourceProgram::ordering(NativeOrderingEventProgram::readAheadParentTerminal()),
        .hostBytes = 512u * 1024u,
        .deviceBytes = 2u * 1024u * 1024u,
        .evidence = "220fa9fcb0cd4b0fc31d0b25c0215941eba3bafe15b55213ce4681336d438279",
        .lifetime = NativePhysicalLifetime::ProcessRuntime,
    },
    NativeAuxiliaryQualifiedClass{
        .backend = DeviceType::ROCm,
        .program = NativeAuxiliaryResourceProgram::ordering(NativeOrderingEventProgram::readAheadParentTerminal()),
        .hostBytes = 512u * 1024u,
        .deviceBytes = 2u * 1024u * 1024u,
        .evidence = "220fa9fcb0cd4b0fc31d0b25c0215941eba3bafe15b55213ce4681336d438279",
        .lifetime = NativePhysicalLifetime::ProcessRuntime,
    },
}};
}
