/**
 * @file PreparedCPUWeightTensor.h
 * @brief Store-committed CPU matrix metadata retaining its actual NativeVNNI engine.
 *
 * Raw source windows retire after preparation. This object preserves their
 * logical matrix/source identity and the independently owned final engine,
 * without inventing a raw tensor buffer or an unpackable source interface.
 * PreparedWeightStore alone constructs it after successful writer completion.
 */
#pragma once
#include "tensors/TensorClasses.h"
#include <memory>
#include <vector>
namespace llaminar2 {
class PreparedWeightStore;
/** @brief Immutable logical identity of one actual committed CPU packed matrix. */
class PreparedCPUWeightTensor final : public TensorBase {
public:
    /** @brief Retire metadata only; the shared engine releases its own admitted root last. */
    ~PreparedCPUWeightTensor() override;
    /** @return Final logical [N,K] matrix, not a raw storage extent. */
    const std::vector<size_t>& shape() const override { return shape_; }
    /** @return Published execution-native scalar/codebook; original source provenance stays in the Store binding. */
    TensorType native_type() const override { return execution_type_; }
    /** @return The actual CPU storage participant; NUMA remains engine-owned. */
    DeviceId home_device() const override { return allocator_; }
    /** @return No raw tensor layout is represented by this prepared-only identity. */
    TensorLayout layout() const override { return TensorLayout::UNKNOWN; }
    /** @brief Reject relabeling engine storage as an ordinary raw tensor layout. */
    void setLayout(TensorLayout) override;
    /** @return Zero raw bytes; actual packed bytes belong exclusively to the retained engine. */
    size_t size_bytes() const override { return 0; }
    /** @return No raw host source exists after successful preparation. */
    const void* raw_data() const override { return nullptr; }
    /** @return No mutable raw source can be republished. */
    void* raw_mutable_data() override { return nullptr; }
    /** @return No raw device or host payload is represented by this metadata. */
    const void* active_data_ptr() const override { return nullptr; }
    /** @return No mutable active payload is represented by this metadata. */
    void* active_mutable_data_ptr() override { return nullptr; }
    /** @return No GPU tensor allocation exists. */
    const void* gpu_data_ptr() const override { return nullptr; }
    /** @return No GPU tensor allocation exists. */
    void* gpu_data_ptr() override { return nullptr; }
    /** @return True: source bytes are absent; final prepared storage is a separate representation. */
    bool is_raw_data_released() const override { return true; }
    /** @return Actual engine storage validity, rather than a metadata-owned readiness flag. */
    bool hasPreparedDeviceState() const override;
    /** @brief Idempotent raw retirement cannot revoke the independently owned engine. */
    void release_raw_data() override {}
    /** @brief No raw host representation exists; retain the actual prepared engine. */
    void release_host_weight_data() override {}
    /** @brief Reject an implicit FP32 source reconstruction. */
    const float* data() const override;
    /** @brief Reject mutable FP32 reconstruction. */
    float* mutable_data() override;
    /** @brief Reject writes to an immutable prepared-only identity. */
    bool copyFrom(const TensorBase*) override;
    /** @brief Reject repacking; callers borrow the committed engine from PreparedWeightStore. */
    std::unique_ptr<ITensorGemm> createGemm() override;
    /** @brief Reject views that would invent raw storage or another engine binding. */
    std::shared_ptr<TensorBase> create_view(const std::vector<size_t>&,size_t=0) override;
    /** @brief Reject untyped raw movement; packed storage has its own original owner. */
    TensorBase* transferStorageOwner() override;
    /** @brief Reject untyped raw movement; packed storage has its own original owner. */
    const TensorBase* transferStorageOwner() const override;
    /** @brief Reject upload of nonexistent raw source bytes. */
    bool ensureOnDevice(DeviceId,void* =nullptr) override;
    /** @brief Reject host materialization of nonexistent raw source bytes. */
    bool ensureOnHost(void* =nullptr) override;
    /** @brief Reject implicit source reconstruction into caller storage. */
    void to_fp32(float*) const override;
    /** @brief Reject implicit source reconstruction into caller storage. */
    void to_bf16(uint16_t*) const override;
    /** @brief Reject implicit source reconstruction into caller storage. */
    void to_fp16(uint16_t*) const override;
    /** @brief Reject requantization of a committed prepared representation. */
    void to_int8_blocked(int8_t*,float*,size_t=32) const override;
    /** @brief Reject requantization of a committed prepared representation. */
    bool to_int8_perchannel(int8_t*,float*,float* =nullptr) const override;
    /** @brief Reject row reconstruction from prepared execution storage. */
    void to_fp32_row(size_t,float*) const override;
    /** @brief Reject span reconstruction from prepared execution storage. */
    void to_fp32_span(size_t,size_t,float*) const override;
protected:
    /** @return No raw allocation is owned by this identity. */
    size_t byte_size() const override { return 0; }
    /** @return No raw host allocation is owned by this identity. */
    void* raw_host_data_ptr() override { return nullptr; }
    /** @return No raw host allocation is owned by this identity. */
    const void* raw_host_data_ptr() const override { return nullptr; }
private:
    friend class PreparedWeightStore;
    /**
     * @brief Validate the actual Store-committed CPU engine and retain its original root.
     * @param engine Actual writer result, never a recorded pointer/readiness substitute.
     * @param final_shape Complete final [N,K], matching the engine's own dimensions.
     * @param execution_native_type Published execution-native codebook/scalar after any authorized floating conversion.
     * @param original_allocator Exact CPU allocation identity retained by the Store binding/PMA.
     * @return Metadata retaining the same final engine without another allocation of weights.
     * @throws std::invalid_argument For missing/foreign/unowned storage or mismatched geometry/provenance.
     */
    static std::shared_ptr<PreparedCPUWeightTensor> fromCommitted(std::shared_ptr<ITensorGemm> engine,
        std::vector<size_t> final_shape,TensorType execution_native_type,DeviceId original_allocator);
    /** @brief Adopt only a validated committed engine through the Store-only factory. */
    PreparedCPUWeightTensor(std::shared_ptr<ITensorGemm>,std::vector<size_t>,TensorType,DeviceId);
    /** @brief Every raw/conversion/repacking attempt fails through the same precise diagnostic. */
    [[noreturn]] static void rejectRawOperation();
    std::shared_ptr<ITensorGemm> engine_; ///< The actual engine retains its admitted final packed root.
    const std::vector<size_t> shape_; ///< Immutable final matrix coordinates.
    const DeviceId allocator_; ///< Exact original CPU identity; never inferred from NUMA or an engine ordinal.
    const TensorType execution_type_; ///< Published tensor format; original source provenance remains binding-owned.
};
} // namespace llaminar2
