/**
 * @file ImmutableResidentLookupKernels.h
 * @brief Direct capture-safe lookup into an immutable source-native VRAM table.
 *
 * The caller retains the sole PMA-backed table allocation and seals its runtime
 * generation before capture. This interface owns code/geometry only: it never
 * allocates, copies, converts or re-quantizes table payloads. RAM/mmap service
 * protocols are separate first-class paths and are not used by this operation.
 */
#pragma once
#include "ImmutableLookupDecode.h"
#include "Qwen4ExpKernels.h"
#include "../../backends/DeviceId.h"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>

namespace llaminar2::qwen4exp
{
/** @brief Immutable checked identity of the original table's GPU allocation. */
class ImmutableResidentTableView final
{
public:
    /**
     * @brief Bind exact original source rows, without acquiring or changing storage.
     * @param device Exact GPU owning the resident table allocation.
     * @param format Original GGUF source scalar/codebook; no derived representation.
     * @param nativeData Persistent immutable device address retained by the caller.
     * @param rowCount Exact number of source rows, including every hash partition.
     * @param rowElements Logical FP32 values decoded from each original row.
     * @param rowBytes Original source row stride from the canonical native layout.
     * @param nativeBytes Exact live source allocation extent, rowCount*rowBytes.
     * @param storageGeneration Nonzero immutable owner/runtime generation for capture identity.
     * @return Checked non-owning table view; the source owner outlives every borrowed graph.
     * @throws std::invalid_argument For missing ownership, unsupported format or changed geometry.
     * @throws std::overflow_error For unrepresentable table bytes or address extent.
     */
    [[nodiscard]] static ImmutableResidentTableView bind(DeviceId device,TensorType format,
        const std::byte *nativeData,std::uint64_t rowCount,int rowElements,
        std::size_t rowBytes,std::size_t nativeBytes,std::uint64_t storageGeneration)
    {
        if(!device.is_valid()||!device.is_gpu()||!nativeData||!rowCount||!storageGeneration)
            throw std::invalid_argument("Resident lookup requires exact GPU/source/generation ownership");
        lookup_detail::validateGeometry(format,rowElements,rowBytes);
        const auto alignment=format==TensorType::FP32||format==TensorType::Q8_K?alignof(float):alignof(std::uint16_t);
        if(reinterpret_cast<std::uintptr_t>(nativeData)%alignment)
            throw std::invalid_argument("Resident lookup source address is misaligned for original metadata");
        if(NativeEmbeddingLayout::supports(format) &&
            static_cast<std::size_t>(rowElements)%NativeEmbeddingLayout::forSource(format,1,rowElements).block_elements)
            throw std::invalid_argument("Resident lookup requires complete original quantization blocks per row");
        if(rowCount>std::numeric_limits<std::size_t>::max()/rowBytes)
            throw std::overflow_error("Resident lookup native table bytes overflow");
        if(nativeBytes!=static_cast<std::size_t>(rowCount)*rowBytes)
            throw std::invalid_argument("Resident lookup allocation differs from exact original table bytes");
        if(reinterpret_cast<std::uintptr_t>(nativeData)>
            std::numeric_limits<std::uintptr_t>::max()-nativeBytes)
            throw std::overflow_error("Resident lookup native address extent overflows");
        return ImmutableResidentTableView(device,format,nativeData,rowCount,rowElements,rowBytes,nativeBytes,storageGeneration);
    }
    /** @return Exact GPU allocation owner, never a transfer destination selector. */
    [[nodiscard]] DeviceId device() const noexcept { return device_; }
    /** @return Unchanged original source scalar/codebook. */
    [[nodiscard]] TensorType format() const noexcept { return format_; }
    /** @return Immutable source-native device pointer. */
    [[nodiscard]] const std::byte *nativeData() const noexcept { return data_; }
    /** @return Exact original table row count. */
    [[nodiscard]] std::uint64_t rowCount() const noexcept { return rows_; }
    /** @return Logical FP32 values decoded from one source row. */
    [[nodiscard]] int rowElements() const noexcept { return width_; }
    /** @return Exact packed source row stride, including native metadata. */
    [[nodiscard]] std::size_t rowBytes() const noexcept { return stride_; }
    /** @return Exact source-native physical extent. */
    [[nodiscard]] std::size_t nativeBytes() const noexcept { return bytes_; }
    /** @return Frozen storage/runtime generation; include it in graph/cache identity. */
    [[nodiscard]] std::uint64_t storageGeneration() const noexcept { return generation_; }
    /** @return Whether every source pointer, device, format, geometry and generation agrees. */
    bool operator==(const ImmutableResidentTableView &) const = default;
private:
    /** @brief Only bind() mints a completely checked allocation identity. */
    ImmutableResidentTableView(DeviceId device,TensorType format,const std::byte *data,std::uint64_t rows,
        int width,std::size_t stride,std::size_t bytes,std::uint64_t generation)
        :device_(device),format_(format),data_(data),rows_(rows),width_(width),stride_(stride),bytes_(bytes),generation_(generation) {}
    DeviceId device_;
    TensorType format_;
    const std::byte *data_;
    std::uint64_t rows_;
    int width_;
    std::size_t stride_,bytes_;
    std::uint64_t generation_;
};

/** @brief Persistent device operands; liveRows is GPU-owned and never read by launch code. */
struct ImmutableResidentGatherArgs
{
    RowExtent tokenRows; ///< Positive admitted token capacity, optional native live count in[0,capacity].
    const std::int64_t *ids=nullptr; ///< Original hash IDs [capacity,16], preserving signed64 mathematical identity.
    std::size_t idCapacity=0; ///< Physical INT64 capacity, at least16*tokenRows.rows.
    float *output=nullptr; ///< FP32 [capacity,16,rowElements]; inactive rows remain byte-untouched.
    std::size_t outputElements=0; ///< Physical FP32 capacity, never the execution extent.
    int *healthy=nullptr; ///< Canonical generation health, cleared after any invalid live count/ID.
    int *error=nullptr; ///< Canonical generation error; first failure retained as ImmutableStorageFailure.
};

namespace resident_lookup_detail
{
/** @brief One setup-authenticated half-open pointer range; no physical memory accounting. */
struct Range { std::uintptr_t begin,end; };
/** @brief Validate one bounded persistent operand without touching its device bytes. */
inline Range range(const void *pointer,std::size_t bytes,std::size_t alignment)
{
    const auto address=reinterpret_cast<std::uintptr_t>(pointer);
    if(!pointer||address%alignment||address>std::numeric_limits<std::uintptr_t>::max()-bytes)
        throw std::invalid_argument("Resident lookup operand is absent, misaligned or has an overflowing extent");
    return {address,address+bytes};
}
/** @brief Reject overlapping write/control/source authorities before graph capture. */
inline void disjoint(Range a,Range b)
{
    if(a.begin<b.end&&b.begin<a.end)
        throw std::invalid_argument("Resident lookup operands have overlapping physical authorities");
}
/**
 * @brief Authenticate exact admitted capacities, explicit stream and nonoverlap.
 * @param table Immutable original source allocation identity.
 * @param a Persistent device operands and admitted token geometry.
 * @param stream Exact non-default backend stream; never queried or substituted.
 * @throws std::invalid_argument For capacity, pointer, stream or ownership mismatch.
 */
inline void validate(const ImmutableResidentTableView &table,const ImmutableResidentGatherArgs &a,void *stream)
{
    if(!stream||reinterpret_cast<std::uintptr_t>(stream)<=2||a.tokenRows.rows<=0)
        throw std::invalid_argument("Resident lookup requires positive admitted rows and exact non-default stream");
    const auto tokens=static_cast<std::size_t>(a.tokenRows.rows);
    if(tokens>std::numeric_limits<std::size_t>::max()/kNgramHeads)
        throw std::invalid_argument("Resident lookup token/ID geometry overflows");
    const auto ids=tokens*kNgramHeads;
    if(ids>std::numeric_limits<std::size_t>::max()/sizeof(std::int64_t)||
       ids>std::numeric_limits<std::size_t>::max()/static_cast<unsigned>(table.rowElements())/sizeof(float)||
       a.idCapacity<ids||a.outputElements<ids*static_cast<unsigned>(table.rowElements()))
        throw std::invalid_argument("Resident lookup physical operands do not cover admitted logical rows");
    const auto source=range(table.nativeData(),table.nativeBytes(),1);
    const auto keys=range(a.ids,ids*sizeof(std::int64_t),alignof(std::int64_t));
    const auto output=range(a.output,ids*static_cast<unsigned>(table.rowElements())*sizeof(float),alignof(float));
    const auto healthy=range(a.healthy,sizeof(int),alignof(int));
    const auto error=range(a.error,sizeof(int),alignof(int));
    const Range ranges[]{source,keys,output,healthy,error};
    for(std::size_t i=0;i<5;++i)for(std::size_t j=0;j<i;++j)disjoint(ranges[i],ranges[j]);
    if(a.tokenRows.liveRows)
    {
        const auto count=range(a.tokenRows.liveRows,sizeof(int),alignof(int));
        for(const auto &other:ranges)disjoint(count,other);
    }
}
} // namespace resident_lookup_detail

/** @brief Stateless code/geometry owner for direct native VRAM lookup; no service or scratch. */
class IImmutableResidentLookupKernels
{
public:
    /** @brief Retire setup-only codebook bindings after every borrowed graph has retired. */
    virtual ~IImmutableResidentLookupKernels()=default;
    /** @return Complete immutable allocation identity, required in stage/cache signature. */
    [[nodiscard]] virtual const ImmutableResidentTableView &table() const noexcept=0;
    /**
     * @brief Validate live IDs then gather/decode original rows on one ordered native stream.
     * @param args Persistent sources/output/control banks; no host device-count mirror.
     * @param stream Exact non-default consumer stream retained by the executor.
     * @details Empty live rows perform no payload reads/writes. Invalid live counts/IDs
     * clear canonical health and preserve the first error, with output untouched.
     * The parent owns forward liveness: non-admitted health0/error0 is valid
     * prefill, while a nonzero canonical first error suppresses every payload write.
     * Each value uses the same native decoder as CPU and RAM/mmap, independent of
     * grouped/serial row geometry. No protocol, allocation, transfer or sync occurs.
     */
    virtual void gather(const ImmutableResidentGatherArgs &args,void *stream) const=0;
};
/** @brief Bind exact CUDA source/codebook pointers before capture.
 * @param table Checked immutable source identity; caller retains its physical owner.
 * @return Stateless CUDA bundle with the same complete table identity.
 * @throws std::runtime_error For wrong backend or failed source-codebook preparation. */
std::unique_ptr<IImmutableResidentLookupKernels> createCUDAImmutableResidentLookupKernels(ImmutableResidentTableView table);
/** @brief Bind exact ROCm source/codebook pointers before capture.
 * @param table Checked immutable source identity; caller retains its physical owner.
 * @return Stateless HIP bundle with the same complete table identity.
 * @throws std::runtime_error For wrong backend or failed source-codebook preparation. */
std::unique_ptr<IImmutableResidentLookupKernels> createROCmImmutableResidentLookupKernels(ImmutableResidentTableView table);
} // namespace llaminar2::qwen4exp
