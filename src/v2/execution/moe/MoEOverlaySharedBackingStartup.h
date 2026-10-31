/**
 * @file MoEOverlaySharedBackingStartup.h
 * @brief Required startup-only creator publication for owned shared mmap files.
 *
 * These fixed records authenticate the original file independently of its
 * contents before any participant maps it. The existing channel/fabric remains
 * the lifetime owner; no graph controller, byte ledger or process manager lives
 * here. Ordinary transactions exchange no additional MPI messages.
 */
#pragma once
#include "utils/MPISessionOwner.h"
#include <cstdint>
#include <span>
#include <string>
namespace llaminar2 {
    class IMPIContext;
    /** @brief Distinct real consumers sharing one construction protocol. */
    enum class MoEOverlaySharedBackingKind : std::uint64_t { ActivationRows=1, DeviceControllerFabric=2 };
    /** @brief Immutable existing namespace/epoch facts naming one startup transaction. */
    struct MoEOverlaySharedBackingEpoch final {
        MoEOverlaySharedBackingKind kind;
        std::uint64_t node_namespace;
        std::uint64_t generation;
        std::uint64_t fingerprint_low;
        std::uint64_t fingerprint_high;
    };
    /** @brief Reserved construction tag; ordinary rank-batch payload tags remain distinct. */
    inline constexpr int kMoEOverlaySharedBackingStartupTag=32742;
    namespace detail {
        inline constexpr std::uint64_t kMoEOverlaySharedBackingMagicV1=0x4c4c53484d535431ull;
        /** @brief Canonical startup states replayed through the actual production decoder. */
        enum class MoEOverlaySharedBackingPhaseV1 : std::uint64_t
        { Created=1,Failed=2,Accepted=3,Rejected=4,Commit=5,Abort=6 };
        /** @brief Fixed source declaration independent of the file's own bytes. */
        struct MoEOverlaySharedBackingDeclarationV1 final {
            std::uint64_t magic=kMoEOverlaySharedBackingMagicV1,version=1,phase=0,kind=0,node=0,generation=0,low=0,high=0;
            std::uint64_t source=0,target=0,participants=0;
            MPISessionOwner::SharedIdentity root{},file{};
            std::uint64_t bytes=0,name_bytes=0;
        };
        /** @brief ACK/terminal facts; root and live name were authenticated by the retained declaration. */
        struct MoEOverlaySharedBackingReplyV1 final {
            std::uint64_t magic=kMoEOverlaySharedBackingMagicV1,phase=0,kind=0,node=0,generation=0,low=0,high=0;
            MPISessionOwner::SharedIdentity file{};
            std::uint64_t bytes=0;
        };
    }
    /**
     * @brief Create/open only after Created/Failed, exact ACK and Commit/Abort complete.
     * @param context Existing selected startup namespace; requests use its actual MPI interface.
     * @param epoch Complete immutable channel/fabric epoch, never a mutable execution decision.
     * @param creator Original designated source/leader in context rank coordinates.
     * @param participants Sorted exact live consumers, including creator and caller.
     * @param basename Exact one-component file name; only its live bytes cross MPI.
     * @param bytes Exact caller-admitted mapping extent, preserving its existing layout/NUMA plan.
     * @return Original typed descriptor; caller may mmap only after successful return.
     * @throws After truthful Abort on a known file/root/extent rejection. Irrecoverable MPI
     * failure, malformed/stale records and the common30s deadline use the existing fatal MPI edge.
     *
     * Production activation preflight visits canonical channels with a group-status
     * reduction after each construction; the device fabric is one declared setup.
     * That existing order isolates this reserved startup tag from other phases.
     */
    [[nodiscard]] MPISessionOwner::SharedFile establishMoEOverlaySharedBacking(
        const IMPIContext &context, const MoEOverlaySharedBackingEpoch &epoch,
        int creator, std::span<const int> participants, const std::string &basename,
        std::uint64_t bytes);
}
