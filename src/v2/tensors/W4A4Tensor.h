/**
 * @file W4A4Tensor.h
 * @brief Native W4A4/H tensor identity over one admitted immutable weight lease.
 *
 * This is an operand with an online activation policy, not a weight-only codec.
 * Floating conversion, requantization and mutable aliases are rejected. Prepared
 * engines and complete-output-row views retain the same WeightStorage owner;
 * retiring this source handle cannot revoke another consumer's physical bytes.
 * GPU preparation uses its native load pipeline, never generic tensor upload.
 */
#pragma once

#include "TensorClasses.h"
#include "W4A4WeightStorage.h"
#include <optional>

namespace llaminar2
{
    /** @brief Immutable complete-K native matrix; its owning lease carries the numerical policy. */
    class W4A4Tensor final : public TensorBase
    {
    public:
        /**
         * @brief Publish a tensor over already admitted native storage without copying.
         * @param storage Exact immutable matrix lease; shape/type/allocator derive from it.
         */
        explicit W4A4Tensor(w4a4::WeightStorage storage);
        /** @brief Retire this source handle after any infrastructure-owned host transfer lifetime. */
        ~W4A4Tensor() override;

        /** @return Immutable logical [N,K], preserved after this source handle retires. */
        const std::vector<size_t> &shape() const override { return shape_; }
        /** @return W4A4 or W4A4H, derived from the admitted source transform. */
        TensorType native_type() const override { return type_; }
        /** @return Original CPU allocator, not the execution worker's NUMA placement. */
        DeviceId home_device() const override { return allocator_; }
        /** @return Logical complete-row storage; this does not relabel the packed byte ABI. */
        TensorLayout layout() const override { return TensorLayout::ROW_MAJOR_2D; }
        /** @brief Validate an unchanged logical layout. @param layout Must remain ROW_MAJOR_2D. */
        void setLayout(TensorLayout layout) override;
        /** @return Exact logical native byte extent, including after source retirement. */
        size_t size_bytes() const override { return contract_.sourceBytes(); }
        /** @return True when created by the complete-output-row tensor-view API. */
        bool is_view() const override { return view_kind_ == ViewKind::CompleteRows; }
        /** @return Whether live bytes still derive from a retained mapping. */
        bool is_mmap_data() const override { return storage_ && storage_->isMapped(); }
        /** @return True exactly when this tensor no longer retains its source lease. */
        bool is_raw_data_released() const override { return !storage_; }
        /** @return A copied lifetime authority, or throw if this source handle already retired. */
        [[nodiscard]] w4a4::WeightStorage retainWeightStorage() const;
        /** @return Immutable native blocks while this handle is live; never an FP32 weight mirror. */
        [[nodiscard]] const w4a4::SourceBlock *typed_data() const;
        /** @brief Retire only this source lease; already prepared consumers remain valid. */
        void release_raw_data() override;
        /** @brief Retire the same source lease; no floating cache or shadow weights exist. */
        void release_host_weight_data() override;
        /** @return A factory-created native engine retaining this exact numerical/storage authority. */
        std::unique_ptr<ITensorGemm> createGemm() override;
        /**
         * @brief Create independently retained complete rows without materializing bytes.
         * @param new_shape Positive [N,K] with the unchanged complete K.
         * @param offset Logical element offset, aligned to the original complete row.
         * @return New tensor whose lease outlives either source tensor's retirement.
         * @throws std::invalid_argument For a reshape, K fragment or unaligned offset.
         * @throws std::out_of_range For a row interval outside the original matrix.
         */
        std::shared_ptr<TensorBase> create_view(const std::vector<size_t> &new_shape, size_t offset = 0) override;

        /** @brief Reject floating materialization of an online-A4 operand. */
        const float *data() const override;
        /** @brief Reject mutable floating storage. */
        float *mutable_data() override;
        /** @brief Reject mutable raw aliases into sealed native bytes. */
        void *raw_mutable_data() override;
        /** @brief Reject mutable execution aliases into sealed native bytes. */
        void *active_mutable_data_ptr() override;
        /** @brief Reject mutable floating aliases into sealed native bytes. */
        float *mutable_fp32_data() override;
        /** @brief Reject writes. @param src Unused; a published native operand is immutable. */
        bool copyFrom(const TensorBase *src) override;
        /** @brief Reject generic transfer; native weight preparation owns device representations. */
        TensorBase *transferStorageOwner() override;
        /** @brief Reject generic transfer; native weight preparation owns device representations. */
        const TensorBase *transferStorageOwner() const override;
        /** @brief Reject generic upload. @param device Unused target. @param stream Unused stream. */
        bool ensureOnDevice(DeviceId device, void *stream = nullptr) override;
        /** @brief Reject generic allocation. @param device Unused target. @param stream Unused stream. */
        bool allocateOnDevice(DeviceId device, void *stream = nullptr) override;
        /** @brief Verify the existing CPU source, without movement. @param stream Unused CPU argument. */
        bool ensureOnHost(void *stream = nullptr) override;
        /** @brief Reject floating conversion. @param dst Destination remains untouched. */
        void to_fp32(float *dst) const override;
        /** @brief Reject floating conversion. @param dst Destination remains untouched. */
        void to_bf16(uint16_t *dst) const override;
        /** @brief Reject floating conversion. @param dst Destination remains untouched. */
        void to_fp16(uint16_t *dst) const override;
        /**
         * @brief Reject alternate quantization before writing any output.
         * @param dst Unused codes destination. @param scales Unused scale destination.
         * @param block_size Unused alternative block geometry.
         */
        void to_int8_blocked(int8_t *dst, float *scales, size_t block_size = 32) const override;
        /**
         * @brief Reject alternate quantization before writing any output.
         * @param dst Unused codes. @param columns Unused column scales. @param rows Unused row scales.
         */
        bool to_int8_perchannel(int8_t *dst, float *columns, float *rows = nullptr) const override;
        /** @brief Reject floating row conversion. @param row Unused row. @param dst Untouched output. */
        void to_fp32_row(size_t row, float *dst) const override;
        /**
         * @brief Reject floating span conversion before writing any output.
         * @param offset Unused offset. @param count Unused length. @param dst Untouched output.
         */
        void to_fp32_span(size_t offset, size_t count, float *dst) const override;

    protected:
        /** @return Exact native bytes, independent of how many handles share the physical owner. */
        size_t byte_size() const override { return contract_.sourceBytes(); }
        /** @brief Reject a writable borrow of immutable native source bytes. */
        void *raw_host_data_ptr() override;
        /** @return Read-only native bytes, or null after this source handle retires. */
        const void *raw_host_data_ptr() const override;

    private:
        enum class ViewKind { Matrix, CompleteRows };
        /** @brief Bind already admitted storage and its typed tensor-view provenance. */
        W4A4Tensor(w4a4::WeightStorage storage, ViewKind kind);
        /** @brief Diagnose unsupported arithmetic conversion without inventing another representation. */
        [[noreturn]] static void rejectConversion();
        /** @brief Diagnose forbidden writable aliases before any source mutation. */
        [[noreturn]] static void rejectMutation();
        /** @brief Diagnose a transfer bypass of the native weight preparation authority. */
        [[noreturn]] static void rejectTransfer();

        const w4a4::MatrixContract contract_; ///< Pointer-free policy survives source retirement.
        const std::vector<size_t> shape_; ///< Geometry derives from that same sealed policy.
        const TensorType type_; ///< Exact online-transform identity, never a VNNI codebook alias.
        const DeviceId allocator_; ///< Original physical allocation endpoint.
        const ViewKind view_kind_; ///< Tensor view provenance, not another allocation ledger.
        std::optional<w4a4::WeightStorage> storage_; ///< This handle's explicit live/retired lifecycle.
    };
} // namespace llaminar2
