/**
 * @file CPUNativeQuantisedPublishedBytes.h
 * @brief Owner of one prepared CPU weight stream and the currency of its metadata seal.
 *
 * Readers get const access only. Every mutation goes through this type:
 * replacing or resizing the storage, or opening a rewrite of the existing
 * bytes. Each mutation advances the publication generation. A metadata seal
 * records the generation it was taken at, so after any mutation the bytes
 * cannot be admitted again until a producer reseals them. That makes slot
 * reuse explicit: a slot rewritten at the same address is a new generation.
 *
 * Rewrites hand out the mutable span once per generation. Infrastructure that
 * receives bytes asynchronously (MPI, TransferEngine, migration writers) keeps
 * that span until its completion and reseals after it. A copy of sealed
 * bytes is content-identical and inherits a rebound seal; a moved owner keeps
 * its root, generation and seal.
 */
#pragma once

#include "CPUNativeQuantisedWeightView.h"
#include "tensors/AlignedVector.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <span>
#include <utility>

namespace llaminar2::cpu::native_quantised
{
    /**
     * @brief Reject execution over bytes whose metadata is not currently sealed.
     * @param reason Precise lifecycle diagnostic.
     * @details A missing seal is a producer lifecycle defect, not recoverable
     * input. It is detected at binding, before any worker team exists, and is
     * fatal so no caller can retry a different path.
     */
    [[noreturn]] inline void rejectUnsealedNativeQuantisedPublication(const char *reason) noexcept
    {
        // One stdio call, so concurrent diagnostics cannot interleave.
        std::fprintf(stderr, "ERROR: native CPU weight publication is not admitted: %s\n", reason);
        std::abort();
    }

    /** @brief Canonical interleaved bytes plus the generation and seal that admit them. */
    class CPUNativeQuantisedPublishedBytes final
    {
    public:
        using value_type = std::uint8_t;
        using const_iterator = const std::uint8_t *;

        CPUNativeQuantisedPublishedBytes() = default;

        /**
         * @brief Deep-copy the bytes; a current seal follows the identical copy.
         * @param other Source publication.
         */
        CPUNativeQuantisedPublishedBytes(const CPUNativeQuantisedPublishedBytes &other) : bytes_(other.bytes_)
        { inheritSeal(other); }

        /** @brief Take storage, generation and seal; the source becomes empty and unsealed. */
        CPUNativeQuantisedPublishedBytes(CPUNativeQuantisedPublishedBytes &&other) noexcept
            : bytes_(std::move(other.bytes_)), seal_(std::move(other.seal_)),
              generation_(other.generation_), sealed_generation_(other.sealed_generation_)
        { other.retire(); }

        /** @brief Deep-copy assignment; a current seal follows the identical copy. */
        CPUNativeQuantisedPublishedBytes &operator=(const CPUNativeQuantisedPublishedBytes &other)
        {
            if (this != &other)
            {
                bytes_ = other.bytes_;
                ++generation_;
                inheritSeal(other);
            }
            return *this;
        }

        /** @brief Move assignment; the destination's previous publication is retired. */
        CPUNativeQuantisedPublishedBytes &operator=(CPUNativeQuantisedPublishedBytes &&other) noexcept
        {
            if (this != &other)
            {
                bytes_ = std::move(other.bytes_);
                seal_ = std::move(other.seal_);
                generation_ = other.generation_;
                sealed_generation_ = other.sealed_generation_;
                other.retire();
            }
            return *this;
        }

        /** @return Immutable byte root. */
        [[nodiscard]] const std::uint8_t *data() const noexcept { return bytes_.data(); }
        /** @return Live byte count. */
        [[nodiscard]] std::size_t size() const noexcept { return bytes_.size(); }
        /** @return Whether no bytes are retained. */
        [[nodiscard]] bool empty() const noexcept { return bytes_.empty(); }
        /** @param index Byte offset. @return Immutable byte. */
        [[nodiscard]] const std::uint8_t &operator[](std::size_t index) const noexcept { return bytes_[index]; }
        /** @return First immutable byte. */
        [[nodiscard]] const_iterator begin() const noexcept { return bytes_.data(); }
        /** @return One past the last immutable byte. */
        [[nodiscard]] const_iterator end() const noexcept { return bytes_.data() + bytes_.size(); }
        /** @return Physical allocation bytes, for capacity accounting. */
        [[nodiscard]] std::size_t allocationBytes() const noexcept { return bytes_.allocationBytes(); }
        /** @return Physical storage kind, for capacity accounting. */
        [[nodiscard]] auto storageKind() const noexcept { return bytes_.storageKind(); }
        /** @return Monotonic publication generation of this owner. */
        [[nodiscard]] std::uint64_t generation() const noexcept { return generation_; }

        /** @brief Install new storage; its contents are a new, unsealed generation. */
        void replace(AlignedVector<std::uint8_t> &&storage) noexcept
        {
            bytes_ = std::move(storage);
            ++generation_;
        }
        /** @brief Resize with value-initialized new bytes; a new generation. @param count Byte count. */
        void resize(std::size_t count)
        {
            ++generation_;
            bytes_.resize(count);
        }
        /** @brief Resize filling new bytes; a new generation. @param count Byte count. @param value Fill byte. */
        void resize(std::size_t count, std::uint8_t value)
        {
            ++generation_;
            bytes_.resize(count, value);
        }
        /** @brief Resize without initialization; a new generation. @param count Byte count. */
        void resize_uninitialized(std::size_t count)
        {
            ++generation_;
            bytes_.resize_uninitialized(count);
        }
        /** @brief Aligned uninitialized resize; a new generation. @param count Byte count. @param alignment Byte alignment. */
        void resize_uninitialized_aligned(std::size_t count, std::size_t alignment)
        {
            ++generation_;
            bytes_.resize_uninitialized_aligned(count, alignment);
        }
        /** @brief Reserve page-sized direct-receive capacity; a new generation. @param count Element capacity. */
        void reserve(std::size_t count)
        {
            ++generation_;
            bytes_.reserve(count);
        }
        /** @brief Release the storage; an empty, unsealed generation. */
        void release() noexcept
        {
            AlignedVector<std::uint8_t> empty;
            bytes_.swap(empty);
            ++generation_;
        }

        /**
         * @brief Open a rewrite of the existing bytes.
         * @return Mutable span of the current storage; the seal is no longer current.
         * @details The caller writes, waits for any asynchronous writer, then a
         * producer reseals. Retaining the span across generations is legal only
         * for slot infrastructure that reopens the rewrite before every reuse.
         */
        [[nodiscard]] std::span<std::uint8_t> beginRewrite() noexcept
        {
            ++generation_;
            return {bytes_.data(), bytes_.size()};
        }

        /**
         * @brief Write into the current, still unsealed generation.
         * @return Mutable span of the current storage; no new generation is opened.
         * @details For a producer finishing the generation it owns (for example
         * normalizing padding after the live writes, before sealing). Writing a
         * sealed generation is a lifecycle defect and fatal.
         */
        [[nodiscard]] std::span<std::uint8_t> unsealedGenerationBytes() noexcept
        {
            if (seal_ && sealed_generation_ == generation_)
                rejectUnsealedNativeQuantisedPublication("a producer tried to write a sealed generation");
            return {bytes_.data(), bytes_.size()};
        }

        /**
         * @brief Mutable root of the current storage without opening a generation.
         * @return Stable slot destination identity.
         * @details Overlay slot infrastructure records this once at adoption; it
         * may write through it only after beginRewrite() for the acquiring epoch.
         */
        [[nodiscard]] std::span<std::uint8_t> retainedSlotRoot() noexcept
        { return {bytes_.data(), bytes_.size()}; }

        /**
         * @brief Seal the current generation after its last write.
         * @param descriptor Interpretation readers will apply.
         * @throws std::invalid_argument For invalid structure or reserved metadata bits;
         *         the publication then stays unsealed.
         */
        void seal(const CPUNativeQuantisedWeightDescriptor &descriptor)
        {
            seal_.reset();
            seal_ = CPUNativeQuantisedMetadataSeal::validate(descriptor, {bytes_.data(), bytes_.size()});
            sealed_generation_ = generation_;
        }

        /**
         * @param descriptor Interpretation a reader would apply.
         * @return Whether the current generation is sealed for @p descriptor.
         */
        [[nodiscard]] bool sealedFor(const CPUNativeQuantisedWeightDescriptor &descriptor) const noexcept
        {
            return seal_ && sealed_generation_ == generation_ &&
                seal_->admits(descriptor, {bytes_.data(), bytes_.size()});
        }

        /**
         * @brief Borrow the sealed publication for execution.
         * @param descriptor Interpretation the reader applies.
         * @return Metadata-admitted view in O(1).
         * @throws std::invalid_argument For a structural mismatch with the sealed bytes.
         * @details An unsealed, stale or differently interpreted publication is fatal.
         */
        [[nodiscard]] CPUNativeQuantisedWeightView admittedView(const CPUNativeQuantisedWeightDescriptor &descriptor) const
        {
            if (!seal_)
                rejectUnsealedNativeQuantisedPublication("its metadata was never sealed");
            if (sealed_generation_ != generation_)
                rejectUnsealedNativeQuantisedPublication("its bytes changed after the metadata seal (stale generation)");
            const std::span<const std::uint8_t> bytes{bytes_.data(), bytes_.size()};
            if (!seal_->admits(descriptor, bytes))
                rejectUnsealedNativeQuantisedPublication("the seal covers other bytes or another interpretation");
            return CPUNativeQuantisedWeightView::fromPublished(descriptor, bytes, *seal_);
        }

    private:
        /** @brief Adopt a deep copy's seal only when the source seal was current. */
        void inheritSeal(const CPUNativeQuantisedPublishedBytes &other)
        {
            if (other.seal_ && other.sealed_generation_ == other.generation_)
            {
                seal_ = other.seal_->rebound({bytes_.data(), bytes_.size()});
                sealed_generation_ = generation_;
            }
            else
                seal_.reset();
        }

        /** @brief Leave a moved-from owner empty and unsealed. */
        void retire() noexcept
        {
            bytes_ = AlignedVector<std::uint8_t>{};
            seal_.reset();
            ++generation_;
        }

        AlignedVector<std::uint8_t> bytes_;               ///< Canonical prepared stream.
        std::optional<CPUNativeQuantisedMetadataSeal> seal_;   ///< Last metadata seal, possibly stale.
        std::uint64_t generation_ = 0;                    ///< Advanced by every mutation.
        std::uint64_t sealed_generation_ = 0;             ///< Generation the seal was taken at.
    };
} // namespace llaminar2::cpu::native_quantised
