/**
 * @file MoEOverlayDynamicPolicyWorkspaceOwner.h
 * @brief PMA-owned persistent allocation for the serialized device policy author.
 *
 * The same immutable layout supplies both the admission BOM and materialization
 * lease. Construction publishes only the setup header on the exact consumer
 * stream. The caller retires all captured users before destroying this owner;
 * request reset never allocates, rebinds, uploads or clears its topology.
 */
#pragma once
#include "MoEOverlayDynamicPolicyWorkspace.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "backends/BackendManager.h"
#include "backends/IBackend.h"
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>

namespace llaminar2
{
    /** @brief Sole allocation/lifetime owner; all policy modes borrow the same serialized bytes. */
    class MoEOverlayDynamicPolicyWorkspaceOwner final
    {
    public:
        /** @brief Exact ExecutionWorkspace BOM contribution, before backend materialization.
         * @param shape Frozen controller geometry. @return Admitted byte demand.
         * @throws std::invalid_argument For an invalid or unrepresentable extent. */
        static std::size_t allocationBytes(MoEOverlayDynamicPolicyShape shape)
        {
            const auto bytes = MoEOverlayDynamicPolicyWorkspaceHeader::bytesFor(shape);
            if (!bytes || bytes > SIZE_MAX)
                throw std::invalid_argument("Dynamic policy workspace has no representable physical extent");
            return static_cast<std::size_t>(bytes);
        }

        /**
         * @brief Claim and materialize exactly the previously admitted workspace.
         * @param memory Original rank PMA, never a separate sampled capacity ledger.
         * @param device Exact authority device, or CPU for device-free storage replay.
         * @param shape Frozen original controller geometry.
         * @param stream Exact non-null GPU setup/consumer stream; CPU uses no stream.
         * @param generation Positive immutable captured allocation generation.
         * @throws std::invalid_argument For missing authority, stream or geometry.
         * @throws std::runtime_error For allocation or ordered header publication failure.
         */
        MoEOverlayDynamicPolicyWorkspaceOwner(std::shared_ptr<PhysicalMemoryAuthority> memory,
            DeviceId device, MoEOverlayDynamicPolicyShape shape, void *stream, std::uint64_t generation = 1u)
            : memory_(std::move(memory)), device_(device),
              header_(MoEOverlayDynamicPolicyWorkspaceHeader::checked(shape, generation))
        {
            const auto bytes = allocationBytes(shape);
            if (!memory_ || !device_.is_valid() || !memory_->contains(device_) ||
                (device_.is_gpu() && !stream))
                throw std::invalid_argument("Dynamic policy workspace requires its original PMA and exact consumer stream");
            lease_ = std::make_unique<PhysicalMemoryAllocationLease>(
                memory_->claimNewAllocation(device_, PhysicalMemoryOwner::ExecutionWorkspace, bytes));
            try
            {
                if (device_.is_cpu())
                {
                    binding_.data = ::operator new(bytes, std::align_val_t{alignof(MoEOverlayDynamicPolicyWorkspaceHeader)});
                    std::memcpy(binding_.data, &header_, sizeof(header_));
                }
                else
                {
                    backend_ = getBackendFor(device_);
                    if (!backend_) throw std::runtime_error("Dynamic policy workspace has no original backend");
                    binding_.data = backend_->allocate(bytes, device_.toKernelDeviceIndex());
                    if (!binding_.data || !backend_->hostToDeviceOnStream(binding_.data, &header_,
                            sizeof(header_), device_.toKernelDeviceIndex(), stream))
                        throw std::runtime_error("Dynamic policy workspace could not publish its immutable header");
                }
                binding_.bytes = bytes;
                binding_.generation = generation;
                binding_.shape = shape;
            }
            catch (...)
            {
                release();
                throw;
            }
        }

        /** @brief Retire physical bytes before releasing their original PMA lease. */
        ~MoEOverlayDynamicPolicyWorkspaceOwner() { release(); }
        MoEOverlayDynamicPolicyWorkspaceOwner(const MoEOverlayDynamicPolicyWorkspaceOwner &) = delete;
        MoEOverlayDynamicPolicyWorkspaceOwner &operator=(const MoEOverlayDynamicPolicyWorkspaceOwner &) = delete;

        /** @return Immutable pointer/extent/generation identity borrowed by retained graphs. */
        MoEOverlayDynamicPolicyWorkspaceBinding binding() const noexcept { return binding_; }
    private:
        /** @brief Release only after the caller's captured users have retired; safe on setup rollback. */
        void release() noexcept
        {
            if (binding_.data)
            {
                if (device_.is_cpu()) ::operator delete(binding_.data, std::align_val_t{alignof(MoEOverlayDynamicPolicyWorkspaceHeader)});
                else if (backend_) backend_->free(binding_.data, device_.toKernelDeviceIndex());
                binding_ = {};
            }
            lease_.reset();
        }
        std::shared_ptr<PhysicalMemoryAuthority> memory_;
        DeviceId device_;
        /** Stable source bytes for asynchronous setup upload; immutable geometry only. */
        const MoEOverlayDynamicPolicyWorkspaceHeader header_;
        IBackend *backend_ = nullptr;
        std::unique_ptr<PhysicalMemoryAllocationLease> lease_;
        MoEOverlayDynamicPolicyWorkspaceBinding binding_;
    };
}
