/**
 * @file CPUPackedWeights.h
 * @brief Concrete IPackedWeights implementation for CPU NativeVNNI format.
 *
 * Wraps CPUNativeVNNIPackedWeights with the IPackedWeights transfer interface.
 * Supports NUMA-local cloning for cross-socket expert migration.
 *
 * ## Clone Semantics
 *
 * clone() deep-copies all eager packed data buffers (native_interleaved,
 * payload, int8_flat). The new buffers are allocated on the calling thread's
 * NUMA node via first-touch policy, ensuring NUMA locality for the destination
 * socket.
 * Destination clones allocate each final section through its typed placement
 * authority. Moves retain allocation leases through detach, transfer and attach.
 */

#pragma once

#include "kernels/IPackedWeights.h"
#include "CPUNativeVNNIWeightPacker.h"
#include "utils/Logger.h"

#include <cstring>
#include <utility>

namespace llaminar2::cpu::native_vnni
{

    /**
     * @brief IPackedWeights implementation wrapping CPUNativeVNNIPackedWeights.
     *
     * Owns a CPUNativeVNNIPackedWeights by value. Supports:
     * - Move construction from existing packed data (zero-copy handoff)
     * - Copy construction (NUMA-local deep copy)
     * - Clone for cross-socket transfer
     */
    class CPUPackedWeights : public IPackedWeights
    {
    public:
        /// Move-construct from existing packed data (zero-copy).
        explicit CPUPackedWeights(CPUNativeVNNIPackedWeights &&packed)
            : packed_(std::move(packed))
        {
            packed_.clearWorkspace(); // workspace pointers are not transferable
        }

        /// Copy-construct from existing packed data (NUMA-local deep copy).
        /// The new buffers will be allocated on the calling thread's NUMA node.
        explicit CPUPackedWeights(const CPUNativeVNNIPackedWeights &packed)
            : packed_(packed)
        {
            packed_.clearWorkspace();
        }

        ~CPUPackedWeights() override = default;

        // ── IPackedWeights interface ─────────────────────

        PackedWeightsFormat format() const override { return PackedWeightsFormat::CPU_NATIVE_VNNI; }
        DeviceType deviceType() const override { return DeviceType::CPU; }
        int N() const override { return packed_.N; }
        int K() const override { return packed_.K; }

        size_t sizeBytes() const override
        {
            return packed_.native_interleaved.size() + packed_.payload.size() + packed_.int8_flat.size();
        }

        std::unique_ptr<IPackedWeights> clone() const override
        {
            // Copy constructor allocates new buffers via AlignedVector/std::vector
            // copy ctors. First-touch NUMA policy places pages on calling thread's node.
            auto cloned = std::make_unique<CPUPackedWeights>(packed_);
            return cloned;
        }

        /**
         * @brief Copy published packed bytes into the destination's exact allocations.
         * @param destination Caller-admitted CPU staging or final-storage owner.
         * @return Complete independent packed owner; no execution workspace is copied.
         */
        std::unique_ptr<IPackedWeights> cloneTo(CPUWeightStoragePlacement destination) const override
        {
            return copyTo(packed_, destination);
        }

        /**
         * @brief Copy a borrowed packed descriptor through an explicit destination.
         * @param source Published immutable packed storage.
         * @param destination Admitted placement for each independently owned buffer.
         * @return New independently owned packed storage.
         */
        static std::unique_ptr<IPackedWeights> copyTo(const CPUNativeVNNIPackedWeights &source,
            CPUWeightStoragePlacement destination)
        {
            CPUNativeVNNIPackedWeights copy;
            copy.N = source.N;
            copy.K = source.K;
            copy.N_padded = source.N_padded;
            copy.blocks_per_row = source.blocks_per_row;
            copy.payload_bytes = source.payload_bytes;
            copy.codebook_id = source.codebook_id;
            copy.is_asymmetric = source.is_asymmetric;
            copy.is_superblock = source.is_superblock;
            copy.encoding = source.encoding;
            copy.numerical_policy = source.numerical_policy;
            copy.data_stride = source.data_stride;
            copy.interleaved_block_stride = source.interleaved_block_stride;
            const auto clone_storage = [&]<typename T>(const AlignedVector<T> &section) {
                auto storage = destination.allocate<T>(section.size());
                if (!section.empty())
                    std::memcpy(storage.data(), section.data(), section.size() * sizeof(T));
                return storage;
            };
            copy.native_interleaved = clone_storage(source.native_interleaved);
            copy.payload = clone_storage(source.payload);
            copy.int8_flat = clone_storage(source.int8_flat);
            return std::make_unique<CPUPackedWeights>(std::move(copy));
        }

        std::unique_ptr<IPackedWeights> convertTo(DeviceType target) const override
        {
            if (target == DeviceType::CPU)
                return clone(); // Same device type → clone

            // Phase 2: CPU → CUDA, CPU → ROCm conversions
            LOG_WARN("[CPUPackedWeights] Cross-device conversion CPU → "
                     << static_cast<int>(target)
                     << " not implemented (phase 2)");
            return nullptr;
        }

        // ── Accessors ────────────────────────────────────

        /// Access the underlying packed weights (read-only).
        const CPUNativeVNNIPackedWeights &packed() const { return packed_; }

        /// Access the underlying packed weights (mutable, for attach).
        CPUNativeVNNIPackedWeights &mutablePacked() { return packed_; }

        /// Move out the underlying packed data (invalidates this object).
        CPUNativeVNNIPackedWeights takePacked() { return std::move(packed_); }

    private:
        CPUNativeVNNIPackedWeights packed_;
    };

    /**
     * @brief Holds both packed weights and native block data for full kernel reconstruction.
     *
     * When transferring weights for deferred-packing kernels, we need both the
     * packed metadata (N, K, codebook, etc.) AND the native block data that
     * the kernel uses for on-demand repacking.
     */
    class CPUPackedWeightsWithNativeBlocks : public CPUPackedWeights
    {
    public:
        CPUPackedWeightsWithNativeBlocks(
            CPUNativeVNNIPackedWeights &&packed,
            std::vector<uint8_t> &&native_blocks,
            size_t native_block_size)
            : CPUPackedWeights(std::move(packed)),
              native_blocks_(native_blocks),
              native_block_size_(native_block_size)
        {
        }

        /**
         * @brief Adopt final admitted native source storage without copying it.
         * @param packed Prepared metadata and interleaved storage.
         * @param native_blocks Independently owned admitted source bytes.
         * @param native_block_size Source format block extent.
         */
        CPUPackedWeightsWithNativeBlocks(CPUNativeVNNIPackedWeights &&packed,
            AlignedVector<uint8_t> &&native_blocks, size_t native_block_size)
            : CPUPackedWeights(std::move(packed)), native_blocks_(std::move(native_blocks)),
              native_block_size_(native_block_size) {}

        /** @copydoc CPUPackedWeights::cloneTo */
        std::unique_ptr<IPackedWeights> cloneTo(CPUWeightStoragePlacement destination) const override
        {
            auto base = CPUPackedWeights::cloneTo(destination);
            auto storage = destination.allocate<uint8_t>(native_blocks_.size());
            if (!native_blocks_.empty())
                std::memcpy(storage.data(), native_blocks_.data(), native_blocks_.size());
            return std::make_unique<CPUPackedWeightsWithNativeBlocks>(
                static_cast<CPUPackedWeights *>(base.get())->takePacked(),
                std::move(storage), native_block_size_);
        }

        /// Deep copy including native blocks (NUMA-local).
        std::unique_ptr<IPackedWeights> clone() const override
        {
            auto cloned = std::make_unique<CPUPackedWeightsWithNativeBlocks>(
                CPUNativeVNNIPackedWeights(packed()), // copy packed
                AlignedVector<uint8_t>(native_blocks_), // copy native blocks
                native_block_size_);
            return cloned;
        }

        size_t sizeBytes() const override
        {
            return CPUPackedWeights::sizeBytes() + native_blocks_.size();
        }

        const AlignedVector<uint8_t> &nativeBlocks() const { return native_blocks_; }
        size_t nativeBlockSize() const { return native_block_size_; }

    private:
        AlignedVector<uint8_t> native_blocks_;
        size_t native_block_size_ = 0;
    };

} // namespace llaminar2::cpu::native_vnni
