/**
 * @file NativeStoragePmaBinding.h
 * @brief Native SDK callbacks into the canonical PMA epoch and physical leases.
 *
 * Native owner records retain unique opaque tokens; imported descriptors borrow
 * their existing root instead of claiming again. The adapter owns no capacity
 * balance or pointer map. Driver/library context present before the original
 * observation remains distinct from newly created SDK roots claimed here.
 */
#pragma once
#include "NativeStorageAuthorityABI.h"
#include "planning/PhysicalMemoryAuthority.h"
#include <algorithm>
#include <limits>
#include <vector>
#include <memory>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    /**
     * @brief Dedicated SDK owner capacity authored in the immutable source BOM.
     * Scratch/fence storage has its own reservation and cannot be passed here as
     * spare capacity. The native immutable preparation result sizes this reserve.
     */
    struct NativeStorageDeclaredReservation final
    {
        DeviceId device;                            ///< Exact original native physical allocator.
        PhysicalMemoryOwnerReservation reservation; ///< Existing source-owned SDK reservation.
    };

    /**
     * @brief Session-owned retirement gate for the next source admission.
     *
     * The application's existing native/session owner keeps this gate through SDK
     * teardown. A new independent source authority is forbidden until every actual
     * native root and alias from this session has physically retired.
     */
    class NativeStorageRetirementGate final
    {
    public:
        /** @brief Reject source re-admission before final native physical retirement. */
        void requireRetired() const
        {
            epoch_.requireRetired();
        }
        /** @return Whether both setup and its final root/alias lifetime have ended. */
        [[nodiscard]] bool retired() const
        {
            return epoch_.retired();
        }

    private:
        friend class NativeStoragePmaBinding;
        friend class NativeStoragePreparationPmaBinding;
        /** @brief Retain the exact canonical epoch without a replacement authority. */
        explicit NativeStorageRetirementGate(PhysicalMemoryNativeSetupEpoch epoch)
            : epoch_(std::move(epoch))
        {
        }
        PhysicalMemoryNativeSetupEpoch epoch_;
    };

    /**
     * @brief One native session's immutable C ABI binding to its existing PMA.
     *
     * The native session retains this context through callback-producing setup and
     * host recording. Native root tokens retain the canonical state independently;
     * release remains valid after sealing or context/source-object destruction.
     * Sealing permits replay and prohibits all later root growth.
     */
    class NativeStoragePmaBinding final
    {
    public:
        NativeStoragePmaBinding(const NativeStoragePmaBinding &) = delete;
        NativeStoragePmaBinding &operator=(const NativeStoragePmaBinding &) = delete;
        /**
         * @brief Close growth for an abandoned context without publishing execution.
         * Actual native tokens retain physical cleanup authority independently.
         */
        ~NativeStoragePmaBinding()
        {
            if (armed_)
                try
                {
                    epoch_.abandonSetup();
                }
                catch (...)
                {
                    std::terminate();
                }
        }

        /** @return Exact versioned callback binding; request epochs are authenticated
         * independently. */
        [[nodiscard]] llaminarNativeStorageAuthorityV1 authority() noexcept
        {
            return {1,
                    sizeof(llaminarNativeStorageAuthorityV1),
                    this,
                    claim,
                    materialize,
                    release,
                    epoch_.identity(),
                    0};
        }
        /**
         * @brief Permit exact CPU0 metadata during one session's cold recordings.
         * Repeated calls in the recording phase are idempotent, including parallel
         * recordings. A sealed execution session cannot return to recording.
         */
        void beginHostRecording()
        {
            epoch_.beginHostRecording();
        }
        /** @brief Seal all SDK growth before publishing an executable or first replay. */
        void sealForExecution()
        {
            epoch_.sealForExecution();
        }
        /** @brief Close failed setup without publishing execution; actual roots retain cleanup
         * authority. */
        void cancelSetup()
        {
            epoch_.abandonSetup();
        }
        /** @return Gate retained by the existing native/session owner through teardown. */
        [[nodiscard]] NativeStorageRetirementGate retirementGate() const
        {
            return NativeStorageRetirementGate(epoch_);
        }

    private:
        friend class NativeStoragePreparationPmaBinding;
        friend class NativeMandatoryStorageDeclarations;
        /** @brief Unpublished final binding; preparation epoch remains unchanged until commit arms
         * this context. */
        NativeStoragePmaBinding(PhysicalMemoryNativeSetupEpoch epoch,
                                std::vector<NativeStorageDeclaredReservation> declared)
            : epoch_(std::move(epoch)), declared_(std::move(declared)), armed_(false)
        {
        }
        /** @brief Unique actual-root token; only the native root's field owns this pointer. */
        struct Token final
        {
            static constexpr std::uint64_t live_magic = 0x4c4c4d504d415631ULL;
            std::uint64_t magic = live_magic;
            PhysicalMemoryNativeSetupLease lease;
            const llaminarNativeRootRequestV1 request;
            /** @brief Preserve exact validated request and its sole canonical claim. */
            Token(PhysicalMemoryNativeSetupLease value, const llaminarNativeRootRequestV1 &root)
                : lease(std::move(value)), request(root)
            {
            }
        };
        /** @brief Check allocator geometry without native queries or capacity arithmetic. */
        static bool validGeometry(const llaminarNativeRootRequestV1 &r) noexcept
        {
            if (r.abi_version != 1 || r.record_bytes != sizeof(r) || r.reserved_zero ||
                !r.owner_identity || !r.shared_anchor_identity || !r.requested_bytes ||
                !r.allocation_granularity ||
                r.requested_bytes > std::numeric_limits<std::size_t>::max() ||
                r.physical_bytes > std::numeric_limits<std::size_t>::max() ||
                r.allocation_granularity > std::numeric_limits<std::size_t>::max() ||
                r.requested_bytes >
                    std::numeric_limits<std::uint64_t>::max() - (r.allocation_granularity - 1))
                return false;
            return r.physical_bytes ==
                   ((r.requested_bytes + r.allocation_granularity - 1) / r.allocation_granularity) *
                       r.allocation_granularity;
        }
        /** @brief Bind common native declaration fields without copying geometry formulas. */
        static PhysicalMemoryNativeRootDeclaration coreDeclaration(
            const llaminarNativeRootRequestV1 &r)
        {
            return {r.owner_identity,
                    r.shared_anchor_identity,
                    std::size_t(r.requested_bytes),
                    std::size_t(r.physical_bytes),
                    std::size_t(r.allocation_granularity),
                    r.allocator_flags,
                    resource(r),
                    r.host_numa,
                    r.owner_family,
                    r.lifetime,
                    r.allocator_kind,
                    static_cast<PhysicalMemoryNativeRootPhase>(r.phase)};
        }
        /** @brief Resolve only the selected exact allocator kind and physical resource. */
        static DeviceId resource(const llaminarNativeRootRequestV1 &r)
        {
            if (r.host_numa < -1 || r.resource_ordinal < 0)
                throw std::invalid_argument("Invalid native resource placement");
            switch (r.resource_kind)
            {
            case llaminarNativeCUDADeviceV1:
                if (r.allocator_kind != llaminarNativeCUDADeviceVmmV1 || r.host_numa != -1)
                    throw std::invalid_argument("CUDA native root allocator/placement mismatch");
                return DeviceId::cuda(r.resource_ordinal);
            case llaminarNativeROCmDeviceV1:
                if ((r.allocator_kind != llaminarNativeROCmDeviceVmmV1 &&
                     r.allocator_kind != llaminarNativeROCmUncachedDeviceVmmV1) ||
                    r.host_numa != -1)
                    throw std::invalid_argument("ROCm native root allocator/placement mismatch");
                return DeviceId::rocm(r.resource_ordinal);
            case llaminarNativeHostCPU0V1:
                if (r.resource_ordinal != 0 ||
                    (r.allocator_kind != llaminarNativeCUDAHostVmmV1 &&
                     r.allocator_kind != llaminarNativeROCmUncachedHostVmmV1 &&
                     r.allocator_kind != llaminarNativeRegisteredHostFileV1 &&
                     r.allocator_kind != llaminarNativeHostAnonymousPagesV1))
                    throw std::invalid_argument(
                        "Native host root must name CPU0 and its exact allocator");
                return DeviceId::cpu();
            default:
                throw std::out_of_range("Unsupported selected native physical resource");
            }
        }
        /**
         * @brief Atomically preclaim one validated actual root before physical Create.
         * The SDK publishes no alias until materialize succeeds. All exceptions map
         * to fixed statuses; the native owner supplies the precise allocation site.
         */
        static int32_t claim(void *context, const llaminarNativeRootRequestV1 *request,
                             void **output) noexcept
        {
            if (!context || !request || !output || *output || !validGeometry(*request))
                return llaminarNativeStorageInvalidV1;
            auto &binding = *static_cast<NativeStoragePmaBinding *>(context);
            if (!binding.armed_)
                return llaminarNativeStorageInvalidV1;
            if (request->setup_epoch != binding.epoch_.identity())
                return llaminarNativeStorageInvalidV1;
            if (request->owner_family < llaminarNativeCommMetadataV1 ||
                request->owner_family > llaminarNativeDriverContextV1 ||
                request->lifetime < llaminarNativeCommLifetimeV1 ||
                request->lifetime > llaminarNativeProcessLifetimeV1)
                return llaminarNativeStorageUnsupportedV1;
            try
            {
                const auto device = resource(*request);
                const bool recording = request->phase == llaminarNativeGraphRecordingHostMetadataV1;
                if (recording)
                {
                    if (device != DeviceId::cpu() ||
                        request->allocator_kind != llaminarNativeHostAnonymousPagesV1 ||
                        (request->owner_family != llaminarNativeCommMetadataV1 &&
                         request->owner_family != llaminarNativeSharedPeersV1 &&
                         request->owner_family != llaminarNativeCapturedWorkV1))
                        return llaminarNativeStorageInvalidV1;
                }
                else if (request->phase != llaminarNativeColdProbeV1 &&
                         request->phase != llaminarNativeRetainedSetupV1)
                    return llaminarNativeStorageInvalidV1;
                auto lease = binding.epoch_.claimPreparedNative(coreDeclaration(*request));
                auto token = std::make_unique<Token>(std::move(lease), *request);
                *output = token.release();
                return llaminarNativeStorageSuccessV1;
            }
            catch (const PhysicalMemoryCapacityExhausted &)
            {
                return llaminarNativeStorageCapacityV1;
            }
            catch (const std::out_of_range &)
            {
                return llaminarNativeStorageUnsupportedV1;
            }
            catch (const std::logic_error &)
            {
                return llaminarNativeStorageInvalidV1;
            }
            catch (...)
            {
                return llaminarNativeStorageInternalV1;
            }
        }
        /** @brief Publish one created root through only its unique move-only PMA lease. */
        static int32_t materialize(void *opaque) noexcept
        {
            if (!opaque)
                return llaminarNativeStorageInvalidV1;
            auto &token = *static_cast<Token *>(opaque);
            if (token.magic != Token::live_magic)
                return llaminarNativeStorageInvalidV1;
            try
            {
                token.lease.materialize();
                return llaminarNativeStorageSuccessV1;
            }
            catch (const std::logic_error &)
            {
                return llaminarNativeStorageInvalidV1;
            }
            catch (...)
            {
                return llaminarNativeStorageInternalV1;
            }
        }
        /**
         * @brief Release after physical cleanup, clearing the actual native root field.
         * A repeated call on that field rejects null before dereferencing a token.
         * Native alias descriptors never own or copy this token pointer.
         */
        static int32_t release(void **field) noexcept
        {
            if (!field || !*field)
                return llaminarNativeStorageInvalidV1;
            auto *token = static_cast<Token *>(*field);
            if (token->magic != Token::live_magic)
                return llaminarNativeStorageInvalidV1;
            token->magic = 0;
            delete token;
            *field = nullptr;
            return llaminarNativeStorageSuccessV1;
        }
        PhysicalMemoryNativeSetupEpoch epoch_;
        std::vector<NativeStorageDeclaredReservation> declared_;
        bool armed_ = true;
    };
} // namespace llaminar2
