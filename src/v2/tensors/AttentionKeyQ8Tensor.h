/**
 * @file AttentionKeyQ8Tensor.h
 * @brief Persistent CPU attention-key storage with one serialized request basis.
 *
 * A physical key plane contains [FP32 anchor][AQ8 blocks]. The anchor is one
 * head per local KV head, not one per cached token. The ring cache owns when
 * that basis may be initialized; attention only borrows immutable native views.
 * There is no dequantized cache, implicit FP32 materialization, or separate
 * memory ledger. Production allocation is admitted by the enclosing KV cache's
 * PhysicalMemoryAuthority lease before this storage is constructed.
 */
#pragma once

#include "TensorClasses.h"

#include <span>

namespace llaminar2
{
enum class ActivationPrecision;
/** @brief Native CPU key plane; its geometry and physical layout never change. */
class AttentionKeyQ8Tensor final : public TensorBase
{
public:
    /**
     * @brief Allocate one complete basis and fixed-capacity compressed key plane.
     * @param positions Physical token capacity (positive).
     * @param heads Local KV heads, after tensor-parallel sharding (positive).
     * @param head_dim Supported complete head width: 64, 128, or 256.
     * @param layout Explicit position-major or head-major KV layout.
     * @param device CPU storage owner, including its NUMA participant identity.
     * @throws std::invalid_argument for an unsupported device or geometry.
     * @throws std::overflow_error before allocation if any extent overflows.
     */
    AttentionKeyQ8Tensor(std::size_t positions, std::size_t heads, int head_dim,
                         TensorLayout layout, DeviceId device = DeviceId::cpu());

    /** @brief Join outstanding transfers before releasing the underlying bytes. */
    ~AttentionKeyQ8Tensor() override;

    /** @brief Two-dimensional physical tensor shape; not the visible ring length. */
    const std::vector<std::size_t> &shape() const override { return shape_; }
    /** @brief Physical key codec, independent of the public Q8/TQ value policy. */
    TensorType native_type() const override { return TensorType::AQ8; }
    /** @brief CPU participant that owns this persistent plane. */
    DeviceId home_device() const override { return device_; }
    /** @brief Complete immutable addressing contract, never inferred from shape. */
    TensorLayout layout() const override { return physical_layout_; }
    /** @brief Reject relabeling already allocated bytes as another layout. */
    void setLayout(TensorLayout layout) override;

    /** @brief Reject implicit whole-cache FP32 materialization. */
    const float *data() const override;
    /** @brief Reject writes through a nonexistent FP32 shadow. */
    float *mutable_data() override;
    /** @brief Copy the entire native plane, including basis, without allocation. */
    bool copyFrom(const TensorBase *source) override;

    /** @brief Explicit diagnostic conversion into caller-owned storage. */
    void to_fp32(float *destination) const override;
    /** @brief Decode exactly one physical tensor row into caller-owned storage. */
    void to_fp32_row(std::size_t row, float *destination) const override;
    /** @brief Decode a bounded physical span using at most one head of scratch. */
    void to_fp32_span(std::size_t offset, std::size_t count, float *destination) const override;
    /** @brief Attention keys are not BF16 activation-conversion operands. */
    void to_bf16(uint16_t *) const override;
    /** @brief Attention keys are not FP16 activation-conversion operands. */
    void to_fp16(uint16_t *) const override;
    /** @brief Reject use of request-relative keys as ordinary INT8 activations. */
    void to_int8_blocked(int8_t *, float *, std::size_t) const override;
    /** @brief This native attention operand does not expose INT8 activation packing. */
    bool to_int8_perchannel(int8_t *, float *, float *) const override { return false; }
    /** @brief AQ8 is an attention-key operand, not a GEMM weight format. */
    std::unique_ptr<ITensorGemm> createGemm() override;
    /** @brief Reject untyped slices that lose the request basis or head geometry. */
    std::shared_ptr<TensorBase> create_view(const std::vector<std::size_t> &, std::size_t) override;

    /** @brief Number of physical slots in each KV head. */
    std::size_t positions() const { return positions_; }
    /** @brief Local, not model-global, number of KV heads. */
    std::size_t heads() const { return heads_; }
    /** @brief Number of coordinates encoded by each physical block. */
    int head_dim() const { return head_dim_; }
    /** @brief Bytes in one scale-plus-code block. */
    std::size_t block_bytes() const { return block_bytes_; }
    /** @brief Serialized prefix preceding compressed token blocks. */
    std::size_t anchor_bytes() const { return heads_ * head_dim_ * sizeof(float); }
    /** @brief Read-only complete request basis, ordered by local KV head. */
    std::span<const float> anchor() const;
    /** @brief Compressed token payload, excluding the serialized basis. */
    const uint8_t *blocks() const;
    /** @brief Validate indices and resolve an explicit physical head/token pair. */
    std::size_t block_index(std::size_t position, std::size_t head) const;

protected:
    /** @brief Exact complete serialized extent used by the transfer contract. */
    std::size_t byte_size() const override { return storage_.size(); }
    /** @brief Infrastructure raw access includes both anchor and compressed bytes. */
    void *raw_host_data_ptr() override { return storage_.data(); }
    /** @brief Const infrastructure raw access has the same complete extent. */
    const void *raw_host_data_ptr() const override { return storage_.data(); }

private:
    /// Only the ring owner initializes a basis or publishes newly encoded keys.
    template <ActivationPrecision, ActivationPrecision> friend class CPURingKVCache;

    /** @brief Ring-owned basis initialization; allowed only while the entry is empty. */
    std::span<float> mutable_anchor();
    /** @brief Ring-owned compressed destination, excluding the request basis. */
    uint8_t *mutable_blocks();

    std::size_t positions_;             ///< Immutable physical token capacity.
    std::size_t heads_;                 ///< Immutable local-head count.
    int head_dim_;                     ///< Immutable codec geometry.
    std::size_t block_bytes_;           ///< Typed scale-plus-code extent.
    TensorLayout physical_layout_;     ///< Explicit physical addressing order.
    DeviceId device_;                  ///< CPU storage placement identity.
    std::vector<std::size_t> shape_;    ///< Generic tensor diagnostic geometry.
    AlignedVector<uint8_t> storage_;    ///< Sole physical owner, including anchor.
};
} // namespace llaminar2
