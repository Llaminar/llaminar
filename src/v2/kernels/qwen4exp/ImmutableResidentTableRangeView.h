/**
 * @file ImmutableResidentTableRangeView.h
 * @brief Explicit original-global-row view of one immutable native device range.
 *
 * A partial allocation cannot convert to ImmutableResidentTableView. Native
 * pack kernels must authenticate the original global IDs and subtract the
 * explicit global base before addressing this local allocation. Full source
 * file identity and physical lifetime remain with ResidentImmutableRowTable.
 */
#pragma once
#include "ImmutableResidentLookupKernels.h"
namespace llaminar2::qwen4exp {
/** @brief Checked nonowning range identity; its real source owner outlives every capture. */
class ImmutableResidentTableRangeView final {
public:
    /**
     * @brief Bind exact original global geometry and immutable local storage.
     * @param device Exact native allocation owner.
     * @param format Unchanged original source scalar/codebook.
     * @param nativeData Persistent local native allocation base.
     * @param totalSourceRows Full original source row count, never local capacity.
     * @param firstGlobalRow Original row represented by the first local byte.
     * @param rowCount Exact positive local owned rows.
     * @param rowElements Logical values decoded from one original row.
     * @param rowBytes Exact original native stride.
     * @param nativeBytes Actual local allocation, exactly rowCount*rowBytes.
     * @param storageGeneration Nonzero actual immutable allocation generation.
     * @return Explicit range view, without acquiring, copying or reinterpreting ownership.
     * @throws std::exception For invalid global bounds or the existing native storage/codec checks.
     */
    static ImmutableResidentTableRangeView bind(DeviceId device,TensorType format,const std::byte *nativeData,
        std::uint64_t totalSourceRows,std::uint64_t firstGlobalRow,std::uint64_t rowCount,int rowElements,
        std::size_t rowBytes,std::size_t nativeBytes,std::uint64_t storageGeneration)
    {
        if(!totalSourceRows || firstGlobalRow>=totalSourceRows || !rowCount || rowCount>totalSourceRows-firstGlobalRow ||
            !rowBytes || totalSourceRows>std::numeric_limits<std::size_t>::max()/rowBytes)
            throw std::invalid_argument("Resident native range exceeds its complete original source");
        // Reuse the existing allocation/codec validator only; no whole-table
        // view escapes. Global source extent remains explicit in this type.
        (void)ImmutableResidentTableView::bind(device,format,nativeData,rowCount,rowElements,rowBytes,nativeBytes,storageGeneration);
        return ImmutableResidentTableRangeView(device,format,nativeData,totalSourceRows,firstGlobalRow,rowCount,
            rowElements,rowBytes,nativeBytes,storageGeneration);
    }
    /** @return Exact GPU allocation owner. */
    DeviceId device() const noexcept {return device_;}
    /** @return Original scalar/codebook. */
    TensorType format() const noexcept {return format_;}
    /** @return Local immutable native bytes, not a globally indexed pointer. */
    const std::byte *nativeData() const noexcept {return data_;}
    /** @return Full original source rows. */
    std::uint64_t sourceRowCount() const noexcept {return source_rows_;}
    /** @return Original global row at local offset0. */
    std::uint64_t globalRowBegin() const noexcept {return first_;}
    /** @return Actual local native rows. */
    std::uint64_t rowCount() const noexcept {return rows_;}
    /** @return Logical source values per row. */
    int rowElements() const noexcept {return width_;}
    /** @return Original native stride. */
    std::size_t rowBytes() const noexcept {return stride_;}
    /** @return Exact local physical native extent. */
    std::size_t nativeBytes() const noexcept {return bytes_;}
    /** @return Immutable physical/runtime allocation identity. */
    std::uint64_t storageGeneration() const noexcept {return generation_;}
    /** @return Complete embedded capture identity, excluding changing execution data. */
    bool operator==(const ImmutableResidentTableRangeView &) const = default;
private:
    /** @brief Only bind() publishes a complete checked original/local storage identity. */
    ImmutableResidentTableRangeView(DeviceId d,TensorType f,const std::byte *p,std::uint64_t total,std::uint64_t first,
        std::uint64_t rows,int width,std::size_t stride,std::size_t bytes,std::uint64_t generation)
        :device_(d),format_(f),data_(p),source_rows_(total),first_(first),rows_(rows),width_(width),stride_(stride),bytes_(bytes),generation_(generation) {}
    DeviceId device_;TensorType format_;const std::byte *data_;
    std::uint64_t source_rows_,first_,rows_;int width_;std::size_t stride_,bytes_;std::uint64_t generation_;
};
} // namespace llaminar2::qwen4exp
