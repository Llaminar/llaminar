/**
 * @file Qwen4ExpPrefillNextTokenWindow.h
 * @brief Borrowed next-chunk token view derived from the original published current chunk.
 *
 * No cursor, count bank or token copy is introduced. Hash and Probe resolve the
 * same original request/position/count publication on their execution device.
 * The KV cursor is deliberately absent: later model layers may advance it.
 */
#pragma once
#include <cstdint>
#include <cstddef>
#include <limits>
#if defined(__CUDACC__) || defined(__HIPCC__)
#define QFN_WINDOW_HD __host__ __device__
#else
#define QFN_WINDOW_HD
#endif
namespace llaminar2::qwen4exp {
/** @brief Exact bounded live source view; invalid windows never become empty successes. */
struct Qwen4ExpPrefillResolvedWindow {
    std::int32_t first=0,rows=0;
    bool valid=false;
};
/** @brief Captured original request inputs; values behind pointers remain device-owned. */
struct Qwen4ExpPrefillNextTokenWindow {
    const std::int32_t* request_tokens=nullptr; ///< Complete original request token bank.
    const std::int32_t* request_positions=nullptr; ///< Original canonical contiguous positions.
    const std::int32_t* request_total=nullptr; ///< Live logical request length.
    const std::int32_t* current_positions=nullptr; ///< Published current chunk positions.
    const std::int32_t* current_rows=nullptr; ///< Published current logical count, never overwritten here.
    std::int32_t request_capacity=0,current_bucket=0,next_bucket=0;
    std::uint32_t version=0; ///< Zero is the unchanged contiguous primitive; one is this closed view.
    /** @return True only for the default absent alternative, without reading live inputs. */
    QFN_WINDOW_HD bool absent()const noexcept {
        return !request_tokens&&!request_positions&&!request_total&&!current_positions&&!current_rows&&
            !request_capacity&&!current_bucket&&!next_bucket&&!version;
    }
    /** @return Complete aligned immutable declaration; never reads a device count. */
    QFN_WINDOW_HD bool declared()const noexcept {
        if(version!=1||!request_tokens||!request_positions||!request_total||!current_positions||!current_rows||
            request_capacity<=0||current_bucket<=0||next_bucket<=0||current_bucket>request_capacity||
            next_bucket>request_capacity||next_bucket>INT32_MAX/16)return false;
        const auto aligned=[](const void* p){return reinterpret_cast<std::uintptr_t>(p)%alignof(std::int32_t)==0;};
        const auto bounded=[&](const void* p,std::uint32_t count){return reinterpret_cast<std::uintptr_t>(p)<=
            std::numeric_limits<std::uintptr_t>::max()-std::uintptr_t(count)*sizeof(std::int32_t);};
        return aligned(request_tokens)&&aligned(request_positions)&&aligned(request_total)&&
            aligned(current_positions)&&aligned(current_rows)&&bounded(request_tokens,request_capacity)&&
            bounded(request_positions,request_capacity)&&bounded(current_positions,1)&&
            bounded(request_total,1)&&bounded(current_rows,1);
    }
    /**
     * @brief Resolve on the execution owner after the original current-chunk publication.
     * @return Invalid for malformed values; a valid exhausted request has zero rows.
     * @note Widened arithmetic prevents signed overflow before any request indexing.
     */
    QFN_WINDOW_HD Qwen4ExpPrefillResolvedWindow resolve()const noexcept {
        if(!declared())return {};
        const std::int64_t total=*request_total,first=request_positions[0],current=current_positions[0],count=*current_rows;
        const std::int64_t start=current-first,end=start+count;
        if(first<0||current<first||total<=0||total>request_capacity||count<=0||count>current_bucket||
            start<0||start>=total||end>total)return {};
        const auto remaining=total-end;
        return {static_cast<std::int32_t>(end),static_cast<std::int32_t>(remaining<next_bucket?remaining:next_bucket),true};
    }
    /** @return Checked sixteen-head query count, or -1 for malformed original publication. */
    QFN_WINDOW_HD int originalIds()const noexcept {const auto w=resolve();return w.valid?w.rows*16:-1;}
};
} // namespace llaminar2::qwen4exp
#undef QFN_WINDOW_HD
