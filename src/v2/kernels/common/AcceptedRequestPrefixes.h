/**
 * @file AcceptedRequestPrefixes.h
 * @brief Typed execution-owned accepted state prefixes in independent verifier requests.
 *
 * Count and flat-row decisions are distinct encodings. Flat decisions carry
 * explicit first-row coordinates, including compact CPU metadata and padded
 * GPU transactions. Decisions and origins may use independent native strides.
 * Nothing reads device words during setup, mirrors counts, or changes ownership.
 */
#pragma once
#include "DeviceRequestRowRanges.h"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#if defined(__CUDACC__) || defined(__HIPCC__)
#define AR_HD __host__ __device__
#else
#define AR_HD
#endif
namespace llaminar2
{
/** @brief Explicit verifier first-row coordinates; unrelated to candidate storage capacity. */
class RequestRowOrigins final
{
public:
    /** @brief Bind an explicit affine coordinate system, including padded GPU requests. */
    static RequestRowOrigins fixedStride(int firstRow,int rowStride)
    {
        if(firstRow<0||rowStride<=0)throw std::invalid_argument("Verifier row origins require nonnegative first and positive stride");
        return {nullptr,1,firstRow,rowStride};
    }
    /** @brief Borrow request first-row words, including compact CPU query-cursor metadata. */
    static RequestRowOrigins nativeWords(const std::int32_t *words,int wordStride=1)
    {
        if(!words||wordStride<=0||reinterpret_cast<std::uintptr_t>(words)%4)
            throw std::invalid_argument("Verifier native origins require aligned words and positive independent stride");
        return {words,wordStride,0,0};
    }
    /** @brief Prove static address/coordinate extent without reading execution-owned words. */
    bool validForRequests(int requests) const noexcept
    {
        if(requests<=0)return false;
        if(words_)return static_cast<std::uint64_t>(requests-1)*wordStride_<=
            (std::numeric_limits<std::uintptr_t>::max()-reinterpret_cast<std::uintptr_t>(words_))/4;
        return static_cast<std::int64_t>(first_)+static_cast<std::int64_t>(requests-1)*requestStride_<=std::numeric_limits<int>::max();
    }
    /** @return Borrowed origins or null for the explicit affine policy. */
    AR_HD const std::int32_t *words() const noexcept{return words_;}
    /** @return Native word stride, independent of decision-word stride. */
    AR_HD int wordStride() const noexcept{return wordStride_;}
    /** @brief Read one coordinate only on the execution owner after its producer edge. */
    AR_HD int resolveNative(int request) const noexcept
    {return words_?words_[static_cast<std::size_t>(request)*wordStride_]:first_+request*requestStride_;}
private:
    /** @brief Construct only through checked explicit origin policies. */
    RequestRowOrigins(const std::int32_t *words,int wordStride,int first,int requestStride)
        :words_(words),wordStride_(wordStride),first_(first),requestStride_(requestStride){}
    const std::int32_t *words_;int wordStride_,first_,requestStride_;
};
/** @brief Checked decision encoding and canonical request-major candidate geometry. */
class AcceptedRequestPrefixes final
{
public:
    /** @brief Decision encoding; negative flat rows are the existing no-advance sentinel family. */
    enum class Encoding : unsigned char { Fixed,StateCounts,FlatRestoreRows };
    /** @brief Freeze an explicit equal accepted prefix for every request, without native metadata. */
    static AcceptedRequestPrefixes fixed(DeviceRequestRowRanges rows,int count)
    {
        if(count<0||count>rows.rowsPerRequest())throw std::invalid_argument("Fixed acceptance exceeds one request's candidate stride");
        return {rows,Encoding::Fixed,nullptr,1,count,RequestRowOrigins::fixedStride(0,1)};
    }
    /** @brief Borrow accepted state counts, including verifier input zero and response clipping. */
    static AcceptedRequestPrefixes counted(DeviceRequestRowRanges rows,const std::int32_t *counts,int decisionWordStride=1)
    {return bind(rows,Encoding::StateCounts,counts,decisionWordStride,RequestRowOrigins::fixedStride(0,1));}
    /** @brief Borrow flat restore rows with explicit first-row coordinates and independent strides. */
    static AcceptedRequestPrefixes fromFlatRows(DeviceRequestRowRanges rows,const std::int32_t *flatRows,
        RequestRowOrigins origins,int decisionWordStride=1)
    {return bind(rows,Encoding::FlatRestoreRows,flatRows,decisionWordStride,origins);}
    /** @return Canonical physical request geometry and live-count authority. */
    AR_HD DeviceRequestRowRanges candidates() const noexcept{return rows_;}
    /** @return Frozen decision interpretation, never inferred from pointer presence. */
    AR_HD Encoding encoding() const noexcept{return encoding_;}
    /** @return Borrowed native decisions, null only for the explicit fixed policy. */
    AR_HD const std::int32_t *decisionWords() const noexcept{return decisions_;}
    /** @return Native decision stride, independent of origin metadata stride. */
    AR_HD int decisionWordStride() const noexcept{return decisionStride_;}
    /** @return Explicit origin policy; meaningful for flat restore rows only. */
    AR_HD RequestRowOrigins origins() const noexcept{return origins_;}
    /** @brief Validate already-observed words without reading any native pointer.
     * @return Exact local accepted state count or -1 for corruption; negative flat decisions preserve prestate.
     */
    AR_HD int resolveFor(int request,int live,int published,int firstRow) const noexcept
    {
        if(rows_.activeRowsFor(request,live)<0)return -1;
        std::int64_t count=encoding_==Encoding::Fixed?fixedCount_:published;
        if(encoding_==Encoding::FlatRestoreRows) {
            if(published<0)return 0;
            if(firstRow<0||published<firstRow)return -1;
            count=static_cast<std::int64_t>(published)-firstRow+1;
        }
        return count>=0&&count<=live?static_cast<int>(count):-1;
    }
    /** @brief Read native lengths/decisions only on their owning CPU or GPU after publication.
     * @return Exact accepted count or -1; callers validate all requests before CPU state mutation.
     */
    AR_HD int resolveNative(int request) const noexcept
    {
        if(request<0||request>=rows_.requests())return -1;
        const int live=rows_.lengthOwner()?rows_.lengthOwner()[request]:rows_.rowsPerRequest();
        const int decision=decisions_?decisions_[static_cast<std::size_t>(request)*decisionStride_]:fixedCount_;
        const int first=encoding_==Encoding::FlatRestoreRows&&decision>=0?origins_.resolveNative(request):0;
        return resolveFor(request,live,decision,first);
    }
private:
    /** @brief Validate static native-word extents without reading a CPU/GPU publication. */
    static AcceptedRequestPrefixes bind(DeviceRequestRowRanges rows,Encoding encoding,const std::int32_t *decisions,
        int decisionStride,RequestRowOrigins origins)
    {
        if(!decisions||decisionStride<=0||reinterpret_cast<std::uintptr_t>(decisions)%4||!origins.validForRequests(rows.requests())||
           static_cast<std::uint64_t>(rows.requests()-1)*decisionStride>
               (std::numeric_limits<std::uintptr_t>::max()-reinterpret_cast<std::uintptr_t>(decisions))/4)
            throw std::invalid_argument("Accepted request decisions require exact aligned native words, origins and independent positive strides");
        return {rows,encoding,decisions,decisionStride,0,origins};
    }
    /** @brief Construct only a fully checked immutable interpretation. */
    AcceptedRequestPrefixes(DeviceRequestRowRanges rows,Encoding encoding,const std::int32_t *decisions,
        int decisionStride,int fixedCount,RequestRowOrigins origins)
        :rows_(rows),encoding_(encoding),decisions_(decisions),decisionStride_(decisionStride),fixedCount_(fixedCount),origins_(origins){}
    DeviceRequestRowRanges rows_;Encoding encoding_;const std::int32_t *decisions_;
    int decisionStride_,fixedCount_;RequestRowOrigins origins_;
};
static_assert(std::is_trivially_copyable_v<RequestRowOrigins>);
static_assert(std::is_trivially_copyable_v<AcceptedRequestPrefixes>);
}
#undef AR_HD
