/**
 * @file BoundedGPUExpertSource.h
 * @brief Immutable native source coordinates read directly into the admitted GPU load ring.
 *
 * Graph bindings need stable matrix identities before preparation, not CPU copies
 * of selected experts or shared column covers. This metadata-only tensor retains the original sealed
 * loader and native rectangles. WeightJobSource fills the existing PMA-owned
 * pinned slots; no payload, scratch arena, mmap or second memory ledger is created.
 * The model owns the loader for longer than its bindings and prepared engines.
 */
#pragma once

#include "loaders/IModelLoader.h"
#include "loaders/gpu_pipeline/DeviceLoadPipeline.h"
#include "tensors/NativeWeightSourceRange.h"
#include "tensors/TensorClasses.h"
#include "tensors/TensorSlice.h"
#include "loaders/gpu_pipeline/NativeQuantisedRepackColumns.h"
#include <algorithm>
#include <limits>
#include <utility>

namespace llaminar2 {
class WeightManager;
class ModelLoader;

/** @brief Source-only expert parent or native column-cover matrix; prepared bytes remain pool-owned. */
class BoundedGPUExpertSource final : public TensorBase {
    /** @brief One original selection, shared by all of its logical expert matrices. */
    class Reader final : public WeightJobSource {
    public:
        /** @brief Validate complete source rectangles without reading or allocating weight bytes. */
        Reader(IModelLoader &loader, std::string name, std::vector<NativeWeightSourceProjection> matrices)
            : WeightJobSource(extent(matrices)), loader_(loader), name_(std::move(name)), matrices_(std::move(matrices)) {}
        /** @return Complete immutable original-coordinate matrix selection. */
        const auto &matrices() const noexcept { return matrices_; }
    private:
        /** @return Checked selected native extent; every matrix has the same original shape and cut. */
        static size_t extent(const std::vector<NativeWeightSourceProjection> &matrices)
        {
            if (matrices.empty()) throw std::invalid_argument("Bounded GPU expert source is empty");
            const auto &first = matrices.front();
            const size_t bytes = first.range.nativeBytes();
            if (matrices.size() > std::numeric_limits<size_t>::max() / bytes)
                throw std::overflow_error("Bounded GPU expert source extent overflows");
            for (size_t i = 0; i < matrices.size(); ++i) {
                const auto &matrix = matrices[i];
                if (matrix.matrix_count != 1 || matrix.assembly || matrix.original != first.original ||
                    matrix.range.rows != first.range.rows || matrix.range.columns != first.range.columns ||
                    matrix.range.first_row != first.range.first_row || matrix.range.first_column != first.range.first_column ||
                    matrix.range.native_row_bytes != first.range.native_row_bytes ||
                    matrix.range.selected_expert_ordinal != i ||
                    (i && matrix.range.original_expert <= matrices[i-1].range.original_expert))
                    throw std::invalid_argument("Bounded GPU expert source changed its original ordered rectangles");
                forEachNativeWeightSourceWindow(matrix.original, matrix.range, bytes, [](const auto &) {});
            }
            return bytes * matrices.size();
        }
        /** @brief Fill complete native rows in the existing slot, including chunks crossing expert boundaries. */
        void readRange(size_t offset, std::span<std::byte> destination) const override
        {
            const size_t matrix_bytes = matrices_.front().range.nativeBytes();
            const size_t row_bytes = matrices_.front().range.native_row_bytes;
            if (offset % row_bytes || destination.size() % row_bytes)
                throw std::invalid_argument("Bounded GPU source read must contain complete native rows");
            while (!destination.empty()) {
                const auto &source = matrices_.at(offset / matrix_bytes);
                const size_t within = offset % matrix_bytes;
                const size_t bytes = std::min(destination.size(), matrix_bytes - within);
                auto window = source.range;
                window.first_row += within / row_bytes;
                window.rows = bytes / row_bytes;
                loader_.readNativeWeightSourceWindow(name_, window, destination.first(bytes));
                destination = destination.subspan(bytes);
                offset += bytes;
            }
        }
        IModelLoader &loader_; ///< Model-owned sealed source; file reader serializes its own buffered I/O.
        const std::string name_;
        const std::vector<NativeWeightSourceProjection> matrices_;
    };
public:
    /** @return Exact logical K32 selection inside the original encoded source cover, when present. */
    const std::optional<NativeQuantisedRepackColumns> &columns() const noexcept { return columns_; }
    /** @return Same bounded reader for all matrices in one selected parent. */
    std::shared_ptr<const WeightJobSource> source() const { return reader_; }
    /** @return Logical offset in the reader, never an address or a physical allocation extent. */
    size_t sourceOffset() const noexcept { return offset_; }
    /** @return Source bytes needed by this logical matrix/parent; physical tensor bytes remain zero. */
    size_t sourceBytes() const { return shape_.size() == 3 ? reader_->sizeBytes() : reader_->matrices().front().range.nativeBytes(); }
    /** @return Original source facts for this exact expert matrix. */
    const NativeWeightSourceProjection &matrixSource() const
    {
        if (shape_.size() != 2) throw std::logic_error("Original GPU matrix source requires a projected matrix");
        return reader_->matrices().at(offset_ / sourceBytes());
    }
    /** @return Source encoding metadata only; no native unpacking is advertised. */
    const NativeQuantisedFormatInfo *format() const { return native_quantised_formats::forQuantType(tensorTypeName(native_type())); }
    /** @return Original logical parent [K,N,E] or matrix [N,K] shape. */
    const std::vector<size_t> &shape() const override { return shape_; }
    /** @return Unchanged original scalar/codebook. */
    TensorType native_type() const override { return reader_->matrices().front().original.original_type; }
    /** @return Original declared execution target, without claiming device storage. */
    DeviceId home_device() const override { return device_; }
    /** @return No raw storage layout is represented by source coordinates. */
    TensorLayout layout() const override { return TensorLayout::UNKNOWN; }
    /** @brief Reject mutable source reinterpretation. */
    void setLayout(TensorLayout) override { reject(); }
    /** @return Zero physical payload; only the existing load ring receives source bytes. */
    size_t size_bytes() const override { return 0; }
    /** @return No whole-source host pointer exists. */
    const void *raw_data() const override { return nullptr; }
    /** @return No mutable whole-source storage exists. */
    void *raw_mutable_data() override { return nullptr; }
    /** @return This identity has no active raw representation. */
    const void *active_data_ptr() const override { return nullptr; }
    /** @return This identity has no mutable active representation. */
    void *active_mutable_data_ptr() override { return nullptr; }
    /** @return GPU bytes belong to prepared engines, never to this source descriptor. */
    const void *gpu_data_ptr() const override { return nullptr; }
    /** @return No mutable GPU source buffer exists. */
    void *gpu_data_ptr() override { return nullptr; }
    /** @brief Reject raw upload outside the admitted load ring. */
    bool ensureOnDevice(DeviceId, void * = nullptr) override { reject(); }
    /** @brief Reject unbounded host reconstruction. */
    bool ensureOnHost(void * = nullptr) override { reject(); }
    /** @brief Reject moving nonexistent raw storage through an ordinary tensor transfer. */
    TensorBase *transferStorageOwner() override { reject(); }
    /** @brief Reject moving nonexistent raw storage through an ordinary tensor transfer. */
    const TensorBase *transferStorageOwner() const override { reject(); }
    /** @return No raw source is retained by this logical identity. */
    bool is_raw_data_released() const override { return true; }
    /** @brief No payload exists to retire; the prepared pool has its own lifetime. */
    void release_raw_data() override {}
    /** @brief No host copy exists to retire. */
    void release_host_weight_data() override {}
    /** @brief Reject eager source reconstruction. */
    const float *data() const override { reject(); }
    /** @brief Reject mutable source reconstruction. */
    float *mutable_data() override { reject(); }
    /** @brief Reject replacement of immutable source coordinates. */
    bool copyFrom(const TensorBase *) override { reject(); }
    /** @brief Prepared kernels must come from the original GPU pool. */
    std::unique_ptr<ITensorGemm> createGemm() override { reject(); }
    /** @brief Project exactly one original expert identity without materializing its bytes. */
    std::shared_ptr<TensorBase> create_view(const std::vector<size_t> &shape, size_t elements = 0) override
    {
        const auto &range = reader_->matrices().front().range;
        const size_t matrix_elements = range.rows * range.columns;
        if (shape_.size() != 3 || shape != std::vector<size_t>{range.rows, range.columns} ||
            elements % matrix_elements || elements / matrix_elements >= reader_->matrices().size())
            throw std::invalid_argument("Bounded GPU view must name one complete original expert matrix");
        return std::shared_ptr<TensorBase>(new BoundedGPUExpertSource(reader_, shape, device_,
            elements / matrix_elements * range.nativeBytes()));
    }
    /** @brief All conversions must consume bounded source windows instead. */
    void to_fp32(float *) const override { reject(); }
    /** @brief Reject unbounded conversion. */
    void to_fp16(uint16_t *) const override { reject(); }
    /** @brief Reject unbounded conversion. */
    void to_bf16(uint16_t *) const override { reject(); }
    /** @brief Reject requantization of original native source weights. */
    void to_int8_blocked(int8_t *, float *, size_t = 32) const override { reject(); }
    /** @brief Reject whole-source per-channel conversion. */
    bool to_int8_perchannel(int8_t *, float *, float * = nullptr) const override { reject(); }
    /** @brief Reject implicit row reconstruction outside the admitted loader. */
    void to_fp32_row(size_t, float *) const override { reject(); }
    /** @brief Reject implicit span reconstruction outside the admitted loader. */
    void to_fp32_span(size_t, size_t, float *) const override { reject(); }
protected:
    /** @return No physical tensor buffer. */
    size_t byte_size() const override { return 0; }
    /** @return No host allocation. */
    void *raw_host_data_ptr() override { return nullptr; }
    /** @return No host allocation. */
    const void *raw_host_data_ptr() const override { return nullptr; }
private:
    friend class WeightManager;
    friend class ModelLoader;
    /**
     * @brief Declare one logical GPU K32 rectangle inside its minimal original encoded cover.
     * @param loader Original sealed reader, retained by the model throughout preparation.
     * @param name Original matrix identity, never a synthetic packed tensor.
     * @param original Actual on-disk [K,N] geometry and source format.
     * @param first Original logical K start, aligned to the native execution block32.
     * @param count Actual final K extent; encoded source groups may be larger.
     * @param device Original GPU target; no device allocation occurs here.
     */
    static std::shared_ptr<TensorBase> declareColumns(IModelLoader &loader, const std::string &name,
        NativeWeightSourceGeometry original, size_t first, size_t count, DeviceId device)
    {
        if (!device.is_gpu() || original.original_shape.size() != 2 || !count ||
            first % 32 || count % 32 || first > original.original_shape[0] ||
            count > original.original_shape[0] - first || !original.native_group_elements ||
            original.native_group_elements % 32 || !original.native_group_bytes ||
            original.original_shape[0] % original.native_group_elements)
            throw std::invalid_argument("GPU native column source requires a bounded original K32 rectangle");
        if (original.original_shape[0] > size_t(std::numeric_limits<int>::max()) ||
            original.original_shape[1] > size_t(std::numeric_limits<int>::max()) ||
            first > size_t(std::numeric_limits<int>::max()) || count > size_t(std::numeric_limits<int>::max()) ||
            original.native_group_elements > size_t(std::numeric_limits<int>::max()))
            throw std::overflow_error("GPU native column source exceeds native kernel geometry");
        const auto cover = NativeQuantisedColumnSourceCover::resolve(int(original.original_shape[0]),
            int(first), int(count), int(original.native_group_elements));
        NativeWeightSourceProjection matrix{original,
            {.rows = original.original_shape[1], .first_column = size_t(cover.first_source_column),
             .columns = size_t(cover.selected.sourceColumns()),
             .native_row_bytes = size_t(cover.selected.sourceColumns()) / original.native_group_elements * original.native_group_bytes}};
        auto reader = std::make_shared<const Reader>(loader, name, std::vector{matrix});
        return std::shared_ptr<TensorBase>(new BoundedGPUExpertSource(reader,
            {original.original_shape[1], count}, device, 0,
            cover.selected));
    }
    /** @brief Construct the original parent's passive metadata from the canonical native projection resolver. */
    static std::shared_ptr<TensorBase> declare(IModelLoader &loader, const std::string &name,
        std::vector<NativeWeightSourceProjection> matrices, DeviceId device)
    {
        if (!device.is_gpu()) throw std::invalid_argument("Bounded GPU source requires its original GPU target");
        auto reader = std::make_shared<const Reader>(loader, name, std::move(matrices));
        const auto &range = reader->matrices().front().range;
        return std::shared_ptr<TensorBase>(new BoundedGPUExpertSource(reader,
            {range.columns, range.rows, reader->matrices().size()}, device, 0));
    }
    /** @brief Retain metadata and an original reader, never physical source storage. */
    BoundedGPUExpertSource(std::shared_ptr<const Reader> reader, std::vector<size_t> shape, DeviceId device, size_t offset,
        std::optional<NativeQuantisedRepackColumns> columns = {})
        : reader_(std::move(reader)), shape_(std::move(shape)), device_(device), offset_(offset), columns_(std::move(columns)) {}
    /** @brief Prevent hidden whole-source reconstruction through ordinary Tensor APIs. */
    [[noreturn]] static void reject() { throw std::logic_error("GPU expert source has coordinates only; use the bounded weight-load ring"); }
    std::shared_ptr<const Reader> reader_;
    const std::vector<size_t> shape_;
    const DeviceId device_;
    const size_t offset_;
    const std::optional<NativeQuantisedRepackColumns> columns_; ///< Logical output selection; source range remains encoded-cover coordinates.
};

/**
 * @brief Resolve a source-only owner through wrappers that have no unapplied selection.
 * @param tensor Actual preparation operand, including original ownership wrappers.
 * @return Bounded owner, or null for ordinary raw tensors whose existing slice path remains authoritative.
 * @throws std::logic_error When a bounded owner is wrapped in an unapplied row/column slice.
 */
inline const BoundedGPUExpertSource *boundedGPUWeightSource(const TensorBase &tensor)
{
    const TensorBase *source = &tensor;
    bool unapplied_slice = false;
    while (const auto *slice = dynamic_cast<const TensorSlice *>(source))
    {
        // A presliced wrapper records original ownership only. An unapplied
        // interval still needs a real source projection and must not vanish
        // when the load ring resolves its physical source-only owner.
        unapplied_slice = unapplied_slice || (!slice->is_full() && !slice->metadata().inner_is_presliced);
        source = slice->inner();
    }
    const auto *bounded = dynamic_cast<const BoundedGPUExpertSource *>(source);
    if (bounded && unapplied_slice)
        throw std::logic_error("Bounded GPU source cannot discard an unapplied TensorSlice interval; declare its original source rectangle first");
    return bounded;
}

/**
 * @brief Source-superblock phase of a GPU weight operand's first prepared K block.
 * @return firstBlock() % 8 of a bounded column selection; zero for whole-K sources.
 */
inline std::uint32_t gpuWeightSourceSuperblockPhase(const TensorBase &tensor)
{
    const auto *bounded = boundedGPUWeightSource(tensor);
    return bounded && bounded->columns() ? static_cast<std::uint32_t>(bounded->columns()->firstBlock() % 8) : 0u;
}

/** @return Canonical source format from either real raw storage or bounded metadata, without advertising unpacking. */
inline const NativeQuantisedFormatInfo *gpuWeightSourceFormat(const TensorBase &tensor)
{
    if (const auto *source = boundedGPUWeightSource(tensor)) return source->format();
    const auto *unpackable = dynamic_cast<const IINT8Unpackable *>(&tensor);
    return unpackable ? unpackable->vnniFormatInfo() : nullptr;
}
} // namespace llaminar2
