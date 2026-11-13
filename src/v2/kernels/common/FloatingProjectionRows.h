/**
 * @file FloatingProjectionRows.h
 * @brief Original-coordinate work selection for floating projection kernels.
 *
 * Work selection is independent of arithmetic policy. Original publications
 * are resolved before any activation, partial or output access. Request holes
 * and causal suffixes retain their physical matrix strides. An inherited
 * verifier count must agree with the original live word, never replace it.
 */
#pragma once
#include "ProjectionRowWork.h"
#include <algorithm>
#include <stdexcept>
#include <utility>
#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_FLOAT_ROWS_HD __host__ __device__
#else
#define LLAMINAR_FLOAT_ROWS_HD
#endif
namespace llaminar2 {
/** @brief One compact prefix bound retained across a native arithmetic body. */
struct FloatingProjectionPrefixTile final {
    int rows; ///< Authored rows relative to the fixed original tile origin.
    /** @return Whether an original tile offset is authored. */
    LLAMINAR_FLOAT_ROWS_HD constexpr bool contains(int offset) const noexcept { return offset < rows; }
    /** @return True when no tensor access or arithmetic belongs to this tile. */
    LLAMINAR_FLOAT_ROWS_HD constexpr bool empty() const noexcept { return rows == 0; }
};
/** @brief One compact request/interval mask retained after publication resolution. */
struct FloatingProjectionMaskTile final {
    unsigned mask; ///< One bit per original tile offset, without compaction.
    /** @return Whether an original tile offset is authored. */
    LLAMINAR_FLOAT_ROWS_HD constexpr bool contains(int offset) const noexcept { return (mask & (1u << offset)) != 0; }
    /** @return True when no tensor access or arithmetic belongs to this tile. */
    LLAMINAR_FLOAT_ROWS_HD constexpr bool empty() const noexcept { return mask == 0; }
};
/**
 * @brief Compact native argument for the established single-prefix contract.
 * @details The full original work descriptor is absent from legacy kernels.
 * Prefix slices and row repetitions retain exactly DeviceRowRange semantics.
 */
class FloatingProjectionPrefixRows final {
public:
    static constexpr bool masksTileArithmetic=false; ///< Preserve legacy neutral-tail arithmetic exactly.
    /** @brief Borrow checked prefix metadata without reading its publication. */
    explicit constexpr FloatingProjectionPrefixRows(DeviceRowRange rows) : rows_(rows) {}
    /** @return Immutable original matrix row capacity. */
    LLAMINAR_FLOAT_ROWS_HD constexpr int physicalRows() const noexcept { return rows_.physicalRows(); }
    /** @brief Pure original-row admission used by device-free geometry regressions. */
    LLAMINAR_FLOAT_ROWS_HD constexpr int rowAdmissionFor(int row,int published,int = 0) const noexcept {
        if(row<0 || row>=physicalRows())return 0;
        const int active=rows_.activeRowsFor(published);
        return active<0 ? -1 : int(row<active);
    }
#if defined(__CUDACC__) || defined(__HIPCC__)
    /** @brief Resolve the original prefix once before CTA-uniform row arithmetic. */
    __device__ inline __attribute__((always_inline)) bool rowIsActiveUniform(int row) const {
        return row>=0 && row<rows_.activeRows();
    }
    /** @brief Resolve one prefix bound before shared-tile accumulator lifetimes.
     * @tparam Width Fixed existing row tile, at most eight original rows.
     * @param first Original physical tile origin. @param physical Original matrix M.
     * @return Compact active-row bound; neither output nor scratch is reindexed.
     */
    template<int Width>
    __device__ inline __attribute__((always_inline)) FloatingProjectionPrefixTile nativeTile(int first,int physical) const {
        static_assert(Width>0 && Width<=8);
        const int active=rows_.activeRows();
        const int remaining=active>first ? active-first : 0;
        const int available=physical>first ? physical-first : 0;
        return {remaining<Width ? remaining : (available<Width ? available : Width)};
    }
#endif
private:
    DeviceRowRange rows_; ///< Existing compact prefix ABI, never a second count owner.
};
/** @brief Compact native argument for independently published request prefixes. */
class FloatingProjectionRequestRows final {
public:
    static constexpr bool masksTileArithmetic=true; ///< Request holes are unauthored original rows.
    /** @brief Borrow original request geometry and its one length owner. */
    explicit constexpr FloatingProjectionRequestRows(DeviceRequestRowRanges rows) : rows_(rows) {}
    /** @return Immutable original matrix row capacity. */
    LLAMINAR_FLOAT_ROWS_HD constexpr int physicalRows() const noexcept { return rows_.physicalRows(); }
    /** @brief Pure admission of one row using its original request's publication. */
    LLAMINAR_FLOAT_ROWS_HD constexpr int rowAdmissionFor(int row,int published,int = 0) const noexcept {
        if(row<0 || row>=physicalRows())return 0;
        const int request=row/rows_.rowsPerRequest();
        const int active=rows_.activeRowsFor(request,published);
        return active<0 ? -1 : int(row%rows_.rowsPerRequest()<active);
    }
#if defined(__CUDACC__) || defined(__HIPCC__)
    /** @brief Resolve a CTA-uniform request before activation or weight access. */
    __device__ inline __attribute__((always_inline)) bool rowIsActiveUniform(int row) const {
        if(row<0 || row>=physicalRows())return false;
        const int request=row/rows_.rowsPerRequest();
        return row%rows_.rowsPerRequest()<rows_.activeRows(request);
    }
    /** @brief Resolve each intersected request once before shared-tile arithmetic.
     * @tparam Width Fixed existing row tile, at most eight original rows.
     * @param first Original physical tile origin. @param physical Original matrix M.
     * @return One original-offset bitmask; requests retain their original strides.
     */
    template<int Width>
    __device__ inline __attribute__((always_inline)) FloatingProjectionMaskTile nativeTile(int first,int physical) const {
        static_assert(Width>0 && Width<=8);
        unsigned mask=0; int previous=-1,active=0;
        #pragma unroll
        for(int offset=0;offset<Width;++offset)if(offset<physical-first){
            const int row=first+offset,request=row/rows_.rowsPerRequest();
            if(request!=previous){active=rows_.activeRows(request);previous=request;}
            if(row%rows_.rowsPerRequest()<active)mask|=1u<<offset;
        }
        return {mask};
    }
#endif
private:
    DeviceRequestRowRanges rows_; ///< Original request-prefix argument only.
};
static_assert(sizeof(FloatingProjectionPrefixRows)==sizeof(DeviceRowRange));
static_assert(sizeof(FloatingProjectionRequestRows)==sizeof(DeviceRequestRowRanges));
static_assert(std::is_trivially_copyable_v<FloatingProjectionPrefixRows>);
static_assert(std::is_trivially_copyable_v<FloatingProjectionRequestRows>);
/** @brief Immutable floating-kernel selection with one publication authority. */
class FloatingProjectionRows final {
public:
    static constexpr bool masksTileArithmetic=true; ///< Original interval holes own no arithmetic.
    /**
     * @brief Bind original work and any inherited verifier publication.
     * @param physical Exact matrix row capacity, never a mutable live count.
     * @param prefix Optional single-request verifier geometry.
     * @param requests Optional independent-request verifier geometry.
     * @return Complete borrowed work descriptor; no live value is read here.
     * @throws std::invalid_argument For conflicting selectors or geometry.
     */
    static FloatingProjectionRows bind(int physical,const DeviceRowRange *prefix,
        const DeviceRequestRowRanges *requests) {
        if(physical<=0 || (prefix && requests) ||
            (prefix && prefix->physicalRows()!=physical) ||
            (requests && requests->physicalRows()!=physical))
            throw std::invalid_argument("Floating projection requires one exact prefix/request selection");
        const auto *scoped=currentProjectionRowWork(physical);
        auto work=scoped ? *scoped : requests ? ProjectionRowWork::fromRequests(*requests) :
            ProjectionRowWork::fromPrefix(prefix ? *prefix : DeviceRowRange::fullyActive(physical));
        if(scoped && prefix)work=work.withInheritedPrefix(*prefix);
        if(scoped && requests)work=work.withInheritedRequests(*requests);
        return FloatingProjectionRows(work);
    }
    /**
     * @brief Select one native argument type from immutable binding policy.
     * @param physical Exact original matrix capacity. @param prefix Optional prefix authority.
     * @param requests Optional independent-request authority. @param launch Typed host launch callback.
     * @return The selected callback result; mutable publication contents never select a kernel.
     * @details Legacy launches retain compact arguments. Only an explicit work scope
     * instantiates the full interval descriptor; all types share the same numeric body.
     */
    template<class Launch>
    static decltype(auto) dispatchNative(int physical,const DeviceRowRange *prefix,
        const DeviceRequestRowRanges *requests,Launch&& launch) {
        if(physical<=0 || (prefix && requests) ||
            (prefix && prefix->physicalRows()!=physical) ||
            (requests && requests->physicalRows()!=physical))
            throw std::invalid_argument("Floating projection requires one exact prefix/request selection");
        if(currentProjectionRowWork(physical))
            return std::forward<Launch>(launch)(bind(physical,prefix,requests));
        if(requests)
            return std::forward<Launch>(launch)(FloatingProjectionRequestRows(*requests));
        return std::forward<Launch>(launch)(FloatingProjectionPrefixRows(
            prefix ? *prefix : DeviceRowRange::fullyActive(physical)));
    }
    /** @brief Hash immutable request selection metadata, never its published values. */
    static std::uint64_t requestRecordingIdentity(DeviceRequestRowRanges rows) noexcept {
        std::uint64_t result=1469598103934665603ULL;
        const auto mix=[&](std::uint64_t word){result^=word;result*=1099511628211ULL;};
        mix(rows.requests());mix(rows.rowsPerRequest());
        mix(reinterpret_cast<std::uintptr_t>(rows.lengthOwner()));
        return result ? result : 1;
    }
    /** @return Complete immutable publication/geometry identity, excluding mutable live values. */
    LLAMINAR_FLOAT_ROWS_HD ProjectionRowWork::CaptureIdentity captureIdentity() const noexcept {
        return work_.captureIdentity();
    }
    /** @return Immutable complete matrix row capacity. */
    LLAMINAR_FLOAT_ROWS_HD int physicalRows() const noexcept {return work_.physicalRows();}
    /** @brief Pure admission against original live and frontier words. */
    LLAMINAR_FLOAT_ROWS_HD int rowAdmissionFor(int row,int published,int frontier=0) const noexcept {
        const int request=work_.requestForRow(row);
        if(request<0)return 0;
        const auto span=work_.resolveFor(request,published,frontier);
        return span.valid() ? int(span.contains(row)) : -1;
    }
    /** @brief Validate every CPU publication before projection writes. */
    void validateHostPublications() const {
        for(int r=0;r<work_.requests();++r)(void)work_.resolveHost(r);
    }
    /** @brief Resolve one original CPU row without inspecting other rows' values. */
    bool hostRowIsActive(int row) const {
        const int request=work_.requestForRow(row);
        return request>=0 && work_.resolveHost(request).contains(row);
    }
    /** @return Fixed request-aligned tile count, independent of mutable live counts. */
    int hostTiles(int width) const {
        if(width<=0)throw std::invalid_argument("Floating projection tile width must be positive");
        return work_.requests()*((work_.rowsPerRequest()-1)/width+1);
    }
    /**
     * @brief Intersect a fixed tile with its original request's authored interval.
     * @param tile Fixed request-major tile coordinate.
     * @param width Positive physical tile capacity.
     * @return Authored intersection; an empty result retains the original clipped tile origin.
     * @throws std::out_of_range For a tile outside admitted geometry.
     * @details Both endpoints are resolved together. Advancing only the count
     * would process stale prefix rows when the causal boundary crosses a tile.
     */
    ProjectionRowSpan hostTileSpan(int tile,int width) const {
        if(tile<0 || tile>=hostTiles(width))throw std::out_of_range("Floating projection tile exceeds request geometry");
        const int per=(work_.rowsPerRequest()-1)/width+1;
        const int request=tile/per;
        const auto authored=work_.resolveHost(request);
        const auto identity=work_.captureIdentity();
        const std::int64_t requestBase=static_cast<std::int64_t>(request)*work_.rowsPerRequest();
        const auto originalFirst=requestBase+static_cast<std::int64_t>(tile%per)*width;
        const auto originalEnd=std::min(originalFirst+width,requestBase+work_.rowsPerRequest());
        const int first=static_cast<int>(std::clamp<std::int64_t>(originalFirst-identity.first_physical_row,0,physicalRows()));
        const int end=static_cast<int>(std::clamp<std::int64_t>(originalEnd-identity.first_physical_row,0,physicalRows()));
        const int selectedFirst=std::max(first,authored.first);
        const int selectedEnd=std::min(end,authored.end);
        // An empty tile retains its original physical origin. The prefix API
        // exposes that coordinate even when another request is empty or ends
        // earlier; only a nonempty causal suffix advances the first row.
        if(selectedFirst>=selectedEnd)return {first,first};
        return {selectedFirst,selectedEnd};
    }
    /** @return First authored row, or the original fixed-tile origin when empty. */
    int hostTileFirst(int tile,int width) const {return hostTileSpan(tile,width).first;}
    /** @return Exact authored row count of one fixed CPU tile. */
    int hostTileRows(int tile,int width) const {return hostTileSpan(tile,width).size();}
#if defined(__CUDACC__) || defined(__HIPCC__)
    /** @brief Resolve a CTA-uniform work row before projection accumulator lifetimes. */
    __device__ inline __attribute__((always_inline)) bool rowIsActiveUniform(int row) const {
        const int active=work_.rowIsActive(row);
#if defined(__HIP_DEVICE_COMPILE__)
        return __builtin_amdgcn_readfirstlane(active)!=0;
#else
        return active!=0;
#endif
    }
    /** @brief Resolve one compact original-offset mask before shared-tile arithmetic.
     * @tparam Width Fixed existing row tile, at most eight original rows.
     * @param first Original physical tile origin. @param physical Original matrix M.
     * @return Authored original offsets, resolving each intersected request once.
     * @pre All participating lanes use the same tile and request coordinates.
     */
    template<int Width>
    __device__ inline __attribute__((always_inline)) FloatingProjectionMaskTile nativeTile(int first,int physical) const {
        static_assert(Width>0 && Width<=8);
        unsigned mask=0; int previous=-1; ProjectionRowSpan span{0,0};
        #pragma unroll
        for(int offset=0;offset<Width;++offset)if(offset<physical-first){
            const int row=first+offset,request=work_.requestForRow(row);
            if(request!=previous){span=work_.resolveDevice(request);previous=request;}
            if(span.contains(row))mask|=1u<<offset;
        }
#if defined(__HIP_DEVICE_COMPILE__)
        mask=__builtin_amdgcn_readfirstlane(mask);
#endif
        return {mask};
    }
    /** @brief Admit native rows before access; lane-varying requests never broadcast counts. */
    __device__ inline __attribute__((always_inline)) bool rowIsActive(int row) const {return work_.rowIsActive(row);}
#endif
private:
    /** @brief Only checked binding constructs a kernel argument. */
    explicit FloatingProjectionRows(ProjectionRowWork work):work_(work) {}
    ProjectionRowWork work_; ///< Original publication and immutable geometry.
};
static_assert(std::is_trivially_copyable_v<FloatingProjectionRows>);
}
#undef LLAMINAR_FLOAT_ROWS_HD
