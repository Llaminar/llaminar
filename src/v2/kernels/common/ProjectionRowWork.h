/**
 * @file ProjectionRowWork.h
 * @brief Original-coordinate projection work over existing request publications.
 *
 * A work view borrows live/frontier words; it never publishes a count, allocates
 * storage, or changes arithmetic policy. Strided headers require cold owner/
 * offset proof. Indirect headers borrow an immutable workspace publication of
 * addresses only, retained exclusively for the graph lifetime. All tensor and
 * split-K strides remain physical capacity. Owners authenticate device, source
 * subview, generation and exact producer edges before constructing native views.
 */
#pragma once
#include "kernels/common/DeviceRowRange.h"
#include "kernels/common/DeviceRequestRowRanges.h"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_PROJECTION_WORK_HD __host__ __device__
#else
#define LLAMINAR_PROJECTION_WORK_HD
#endif

namespace llaminar2 {
/** @brief Immutable addresses into one original request publication, never copied live values. */
struct ProjectionRequestPublication final {
    const std::int32_t *frontier=nullptr; ///< Original logical source rows preceding this transaction.
    const std::int32_t *live=nullptr; ///< Same publication's original logical live count.
};
/** @brief Half-open original physical row extent relative to the current matrix slice. */
struct ProjectionRowSpan final {
    int first=0,end=0;
    /** @return Whether metadata passed validation rather than being clamped from corruption. */
    LLAMINAR_PROJECTION_WORK_HD constexpr bool valid() const noexcept {return first>=0&&end>=first;}
    /** @return Authored row count, or -1 for an invalid publication. */
    LLAMINAR_PROJECTION_WORK_HD constexpr int size() const noexcept {return valid()?end-first:-1;}
    /** @return Whether an original slice-relative row belongs to this interval. */
    LLAMINAR_PROJECTION_WORK_HD constexpr bool contains(int row) const noexcept {return valid()&&row>=first&&row<end;}
};
/** @brief One local observation of existing publication words and their validated authored extent. */
struct ProjectionRowObservation final {
    std::int32_t frontier=0,live=0; ///< Original words read once; never published as execution authority.
    ProjectionRowSpan span;
};
/** @brief Immutable generic request-major row admission, independent of projection arithmetic. */
class ProjectionRowWork final {
public:
    /** @brief Preserve original prefix, slice and repeat semantics without reading its count. */
    [[nodiscard]] static ProjectionRowWork fromPrefix(DeviceRowRange rows) {
        const auto id=rows.captureIdentity();
        return ProjectionRowWork(id.count?Kind::Prefix:Kind::Full,1,id.count_capacity,
            id.first_physical_row,id.physical_rows,id.repeats_per_row,id.count,nullptr,0,nullptr,0,0,0);
    }
    /** @brief Preserve independent request prefixes and original request-major row bases. */
    [[nodiscard]] static ProjectionRowWork fromRequests(DeviceRequestRowRanges rows) {
        return ProjectionRowWork(rows.lengthOwner()?Kind::Requests:Kind::Full,rows.requests(),rows.rowsPerRequest(),
            0,rows.physicalRows(),1,rows.lengthOwner(),nullptr,1,nullptr,0,0,0);
    }
    /**
     * @brief Borrow cold-proven strided original frontier/live publications.
     * @param requests Number of independent canonical requests.
     * @param rowsPerRequest Immutable original logical row capacity per request.
     * @param frontierBase First request's original frontier word.
     * @param liveBase First request's original live word, from the same publication.
     * @param strideWords Exact positive request stride in native INT32 words.
     * @param sourceCapacity Bound on each original frontier plus live count.
     * @param firstRequiredLogicalRow Generic first required logical source row.
     * @return Checked borrowed native descriptor; no live value is read.
     * @throws std::invalid_argument For missing/alignment/geometry/policy errors.
     * @details Only the owner may prove strides, using tensor identity and exact
     * offsets. Address arithmetic across unrelated allocations is not that proof.
     */
    [[nodiscard]] static ProjectionRowWork causalStrided(int requests,int rowsPerRequest,
        const std::int32_t *frontierBase,const std::int32_t *liveBase,int strideWords,
        int sourceCapacity,int firstRequiredLogicalRow) {
        validateShape(requests,rowsPerRequest);
        if(!aligned(frontierBase)||!aligned(liveBase)||strideWords<=0||
            sourceCapacity<rowsPerRequest||firstRequiredLogicalRow<0||
            std::size_t(requests-1)>std::numeric_limits<std::size_t>::max()/std::size_t(strideWords)/sizeof(std::int32_t))
            throw std::invalid_argument("Projection work requires exact aligned strided publications and legal bounds");
        return ProjectionRowWork(Kind::StridedCausal,requests,rowsPerRequest,0,requests*rowsPerRequest,1,
            liveBase,frontierBase,strideWords,nullptr,0,sourceCapacity,firstRequiredLogicalRow);
    }
    /**
     * @brief Borrow an admitted immutable address table for arbitrary request owners.
     * @param requests Independent canonical request count.
     * @param rowsPerRequest Original logical request row capacity.
     * @param table Native immutable table with exactly one address pair per request.
     * @param sourceCapacity Bound on each original frontier plus live count.
     * @param firstRequiredLogicalRow Generic first required logical source row.
     * @param tablePublicationGeneration Exact retained publication identity, nonzero.
     * @return Native view; table entries are deliberately not read by this factory.
     * @throws std::invalid_argument For incomplete/invalid immutable metadata.
     * @details An ExecutionWorkspace BOM charge, exclusive retained publication
     * lease and exact per-entry identity are prerequisites. The table contains
     * addresses only and must never be placed in reusable transaction scratch.
     */
    [[nodiscard]] static ProjectionRowWork causalIndirect(int requests,int rowsPerRequest,
        const ProjectionRequestPublication *table,int sourceCapacity,int firstRequiredLogicalRow,
        std::uint64_t tablePublicationGeneration) {
        validateShape(requests,rowsPerRequest);
        if(!table||reinterpret_cast<std::uintptr_t>(table)%alignof(ProjectionRequestPublication)||
            !tablePublicationGeneration||sourceCapacity<rowsPerRequest||firstRequiredLogicalRow<0)
            throw std::invalid_argument("Projection work requires an admitted immutable publication table");
        return ProjectionRowWork(Kind::IndirectCausal,requests,rowsPerRequest,0,requests*rowsPerRequest,1,
            nullptr,nullptr,0,table,tablePublicationGeneration,sourceCapacity,firstRequiredLogicalRow);
    }
    /** @brief Preserve geometry and require an existing inherited count to agree at execution.
     * @param rows Existing arithmetic scope's original single-request prefix metadata.
     * @return View whose native resolver rejects unequal publications.
     * @throws std::invalid_argument For mismatched geometry or conflicting check sources.
     * @details An absent inherited count declares geometry only. A counted
     * inherited publication never silently replaces the primary original live word.
     */
    [[nodiscard]] ProjectionRowWork withInheritedPrefix(DeviceRowRange rows) const {
        const auto id=rows.captureIdentity();
        // A count-free prefix declares the existing physical launch geometry,
        // not a competing logical request topology. Its slice and expansion
        // still must match: native arguments may not silently rebase a slice.
        if(!id.count) {
            if(first_!=id.first_physical_row||physical_!=id.physical_rows||repeat_!=id.repeats_per_row)
                throw std::invalid_argument("Projection work and count-free prefix disagree on physical geometry");
            return *this;
        }
        if(requests_!=1||logical_rows_!=id.count_capacity||first_!=id.first_physical_row||
            physical_!=id.physical_rows||repeat_!=id.repeats_per_row)
            throw std::invalid_argument("Projection work and inherited prefix disagree on original coordinates");
        return withCountCheck(id.count,0);
    }
    /** @brief Require existing independent inherited request counts to agree at execution.
     * @param rows Existing arithmetic scope's original request-prefix metadata.
     * @return View preserving each primary publication and checking counted inherited values.
     * @throws std::invalid_argument For mismatched geometry or conflicting check sources.
     */
    [[nodiscard]] ProjectionRowWork withInheritedRequests(DeviceRequestRowRanges rows) const {
        if(requests_!=rows.requests()||logical_rows_!=rows.rowsPerRequest()||repeat_!=1||first_!=0||physical_!=rows.physicalRows())
            throw std::invalid_argument("Projection work and inherited requests disagree on original coordinates");
        return withCountCheck(rows.lengthOwner(),1);
    }
    /** @brief Form a fixed nonempty physical slice without changing original request coordinates. */
    [[nodiscard]] ProjectionRowWork slice(int first,int rows) const {
        if(first<0||rows<=0||first>physical_||rows>physical_-first)
            throw std::out_of_range("Projection work slice exceeds admitted physical rows");
        auto result=*this;result.first_+=first;result.physical_=rows;return result;
    }
    /** @brief Expand each original logical row while retaining its sole original count.
     * @param factor Positive immutable expansion, for example native query heads.
     * @return Expanded geometry; logical frontier and threshold units do not change.
     * @throws std::invalid_argument For a nonpositive factor.
     * @throws std::overflow_error For unrepresentable physical capacity.
     */
    [[nodiscard]] ProjectionRowWork repeatEachRow(int factor) const {
        if(factor<=0)throw std::invalid_argument("Projection work expansion must be positive");
        if(factor>std::numeric_limits<int>::max()/(requests_*logical_rows_*repeat_))
            throw std::overflow_error("Projection work expansion exceeds native physical rows");
        auto result=*this;result.first_*=factor;result.physical_*=factor;result.repeat_*=factor;return result;
    }
    /** @return Matrix slice width used by all native tensor and workspace strides. */
    [[nodiscard]] LLAMINAR_PROJECTION_WORK_HD constexpr int physicalRows() const noexcept {return physical_;}
    /** @return Original number of independent requests, including requests outside a slice. */
    [[nodiscard]] LLAMINAR_PROJECTION_WORK_HD constexpr int requests() const noexcept {return requests_;}
    /** @return Original physical request stride after immutable row expansion. */
    [[nodiscard]] LLAMINAR_PROJECTION_WORK_HD constexpr int rowsPerRequest() const noexcept {return logical_rows_*repeat_;}
    /** @return Logical request capacity before head expansion. */
    [[nodiscard]] LLAMINAR_PROJECTION_WORK_HD constexpr int logicalRowsPerRequest() const noexcept {return logical_rows_;}
    /** @return Original flattened physical row, or -1 outside this matrix slice. */
    [[nodiscard]] LLAMINAR_PROJECTION_WORK_HD constexpr int originalRow(int row) const noexcept {
        return row>=0&&row<physical_ ? first_+row : -1;
    }
    /** @return Original request owning a slice-relative row, or -1 outside this matrix slice. */
    [[nodiscard]] LLAMINAR_PROJECTION_WORK_HD constexpr int requestForRow(int row) const noexcept {
        const int original=originalRow(row);return original<0 ? -1 : original/rowsPerRequest();
    }
    /** @brief Resolve supplied publication words without accessing any execution pointer.
     * @param request Original request coordinate, never a compacted request ID.
     * @param published Original unexpanded logical live count.
     * @param frontier Original logical frontier, ignored for noncausal prefix work.
     * @return Original slice-relative physical span, or {-1,-1} for corrupt metadata.
     * @details Invalid values fail before intersection. A request outside the
     * slice resolves to an empty interval at the corresponding slice boundary.
     */
    [[nodiscard]] LLAMINAR_PROJECTION_WORK_HD constexpr ProjectionRowSpan resolveFor(
        int request,int published,int frontier=0) const noexcept {
        if(request<0||request>=requests_||published<0||published>logical_rows_)return {-1,-1};
        std::int64_t logicalFirst=0;
        if(causal()) {
            if(frontier<0||static_cast<std::int64_t>(frontier)+published>source_capacity_)return {-1,-1};
            const auto missing=static_cast<std::int64_t>(threshold_)-frontier;
            logicalFirst=missing<=0 ? 0 : missing>=published ? published : missing;
        }
        const auto base=static_cast<std::int64_t>(request)*logical_rows_*repeat_;
        const auto rawEnd=base+static_cast<std::int64_t>(published)*repeat_-first_;
        const int end=rawEnd<=0 ? 0 : rawEnd>=physical_ ? physical_ : static_cast<int>(rawEnd);
        const auto rawFirst=base+logicalFirst*repeat_-first_;
        const int first=rawFirst<=0 ? 0 : rawFirst>=end ? end : static_cast<int>(rawFirst);
        return {first,end};
    }
    /** @brief Observe only host-owned CPU publications exactly once; GPU factories never call this.
     * @param request Original request coordinate.
     * @return Original words and validated authored span in original slice-relative coordinates.
     * @throws std::invalid_argument For invalid owners/counts or inconsistent inherited values.
     */
    [[nodiscard]] ProjectionRowObservation observeHost(int request=0) const {
        if(request<0||request>=requests_)throw std::invalid_argument("Projection work request exceeds admitted geometry");
        const auto publication=publicationFor(request);
        if((kind_!=Kind::Full&&!aligned(publication.live))||(causal()&&!aligned(publication.frontier)))
            throw std::invalid_argument("Projection work publication contains an invalid original owner");
        const int published=publication.live?*publication.live:logical_rows_;
        const auto *inherited=check_counts_?check_counts_+static_cast<std::size_t>(request)*check_stride_:nullptr;
        if(inherited&&inherited!=publication.live&&published!=*inherited)
            throw std::invalid_argument("Projection work inherited live count disagrees with original publication");
        const int frontier=publication.frontier?*publication.frontier:0;
        const auto span=resolveFor(request,published,frontier);
        if(!span.valid())throw std::invalid_argument("Projection work native publication exceeds admitted bounds");
        return {frontier,published,span};
    }
    /** @brief Resolve the validated authored span from one CPU-owned publication observation.
     * @param request Original request coordinate.
     * @return Original slice-relative authored span.
     * @throws std::invalid_argument For invalid or inconsistent original metadata.
     */
    [[nodiscard]] ProjectionRowSpan resolveHost(int request=0) const {
        return observeHost(request).span;
    }
    /** @brief Complete native capture identity; table owner separately seals every entry and generation. */
    struct CaptureIdentity final {
        std::uint32_t kind,schema_version=1;
        int requests,logical_rows,first_physical_row,physical_rows,repeats_per_row;
        const std::int32_t *counts,*frontiers;
        int publication_stride_words;
        const ProjectionRequestPublication *table;
        std::uint64_t table_publication_generation;
        int source_capacity,first_required_logical_row;
        const std::int32_t *inherited_counts;
        int inherited_count_stride;
        bool operator==(const CaptureIdentity &) const=default;
    };
    /** @return Immutable pointer/geometry/policy tuple; mutable word contents are excluded. */
    [[nodiscard]] LLAMINAR_PROJECTION_WORK_HD constexpr CaptureIdentity captureIdentity() const noexcept {
        return {static_cast<std::uint32_t>(kind_),1,requests_,logical_rows_,first_,physical_,repeat_,
            counts_,frontiers_,stride_,table_,table_generation_,source_capacity_,threshold_,check_counts_,check_stride_};
    }
#if defined(__CUDACC__) || defined(__HIPCC__)
    /** @brief Observe existing device publications exactly once after their exact producer edge.
     * @param request Original request coordinate; callers must not broadcast across different requests.
     * @return Original words and authored original-row span; bad metadata terminates on device.
     */
    [[nodiscard]] __device__ inline __attribute__((always_inline)) ProjectionRowObservation observeDevice(int request=0) const {
        requireDevice(request>=0&&request<requests_);
        const auto publication=publicationFor(request);
        requireDevice((kind_==Kind::Full||aligned(publication.live))&&(!causal()||aligned(publication.frontier)));
        const int published=publication.live?*publication.live:logical_rows_;
        const auto *inherited=check_counts_?check_counts_+static_cast<std::size_t>(request)*check_stride_:nullptr;
        requireDevice(!inherited||inherited==publication.live||published==*inherited);
        const int frontier=publication.frontier?*publication.frontier:0;
        const auto span=resolveFor(request,published,frontier);
        requireDevice(span.valid());return {frontier,published,span};
    }
    /** @brief Resolve the authored span from one device-owned publication observation.
     * @param request Original request coordinate, including lane-varying requests.
     * @return Authored original-row span; bad metadata terminates on its device.
     */
    [[nodiscard]] __device__ inline __attribute__((always_inline)) ProjectionRowSpan resolveDevice(int request=0) const {
        return observeDevice(request).span;
    }
    /** @brief Admit an original slice-relative row before activation/weight/scratch accesses. */
    [[nodiscard]] __device__ inline __attribute__((always_inline)) bool rowIsActive(int row) const {
        const int request=requestForRow(row);return request>=0&&resolveDevice(request).contains(row);
    }
#endif
private:
    /** @brief Closed native publication layout; absence is explicit full geometry. */
    enum class Kind : std::uint32_t {Full,Prefix,Requests,StridedCausal,IndirectCausal};
    /** @brief Keep all geometry representable before native descriptor construction. */
    static void validateShape(int requests,int rows) {
        if(requests<=0||rows<=0||requests>std::numeric_limits<int>::max()/rows)
            throw std::invalid_argument("Projection work requires positive representable request geometry");
    }
    /** @return Whether a nonnull INT32 owner has native alignment; never dereference it. */
    [[nodiscard]] LLAMINAR_PROJECTION_WORK_HD static bool aligned(const std::int32_t *pointer) noexcept {
        return pointer&&reinterpret_cast<std::uintptr_t>(pointer)%alignof(std::int32_t)==0;
    }
    /** @return Whether work requires the original logical frontier. */
    [[nodiscard]] LLAMINAR_PROJECTION_WORK_HD constexpr bool causal() const noexcept {
        return kind_==Kind::StridedCausal||kind_==Kind::IndirectCausal;
    }
    /** @brief Bind only immutable consistency metadata, never copy a count value. */
    [[nodiscard]] ProjectionRowWork withCountCheck(const std::int32_t *counts,int stride) const {
        if(!counts)return *this;
        if(!aligned(counts)||(check_counts_&&(check_counts_!=counts||check_stride_!=stride)))
            throw std::invalid_argument("Projection work has conflicting inherited count sources");
        auto result=*this;result.check_counts_=counts;result.check_stride_=stride;return result;
    }
    /** @brief Resolve addresses only; indirect entries are immutable admitted metadata. */
    [[nodiscard]] LLAMINAR_PROJECTION_WORK_HD ProjectionRequestPublication publicationFor(int request) const noexcept {
        if(kind_==Kind::IndirectCausal)return table_[request];
        const auto offset=static_cast<std::size_t>(request)*stride_;
        return {frontiers_?frontiers_+offset:nullptr,counts_?counts_+offset:nullptr};
    }
#if defined(__CUDACC__) || defined(__HIPCC__)
    /** @brief Fail hard on the device owning corrupt execution metadata. */
    __device__ inline __attribute__((always_inline)) static void requireDevice(bool condition) {
        if(!condition) {
#if defined(__CUDA_ARCH__)
            __trap();
#else
            __builtin_trap();
#endif
        }
    }
#endif
    /** @brief Only checked factories construct a native descriptor. */
    constexpr ProjectionRowWork(Kind kind,int requests,int logicalRows,int first,int physical,int repeat,
        const std::int32_t *counts,const std::int32_t *frontiers,int stride,
        const ProjectionRequestPublication *table,std::uint64_t tableGeneration,int capacity,int threshold) noexcept
        :kind_(kind),requests_(requests),logical_rows_(logicalRows),first_(first),physical_(physical),repeat_(repeat),
        counts_(counts),frontiers_(frontiers),stride_(stride),table_(table),table_generation_(tableGeneration),
        source_capacity_(capacity),threshold_(threshold) {}
    Kind kind_;
    int requests_,logical_rows_,first_,physical_,repeat_;
    const std::int32_t *counts_,*frontiers_;
    int stride_;
    const ProjectionRequestPublication *table_;
    std::uint64_t table_generation_;
    int source_capacity_,threshold_;
    const std::int32_t *check_counts_=nullptr;
    int check_stride_=0;
};
static_assert(std::is_trivially_copyable_v<ProjectionRowWork>);
static_assert(std::is_trivially_copyable_v<ProjectionRequestPublication>);
static_assert(std::is_trivially_copyable_v<ProjectionRowSpan>);
static_assert(std::is_trivially_copyable_v<ProjectionRowObservation>);
/**
 * @brief Recorder-local projection metadata scope, independent of every arithmetic scope.
 *
 * Native wrappers copy the descriptor by value before the scope ends. The
 * thread-local object contains addresses and immutable policy only; it neither
 * reads live words nor represents host-owned GPU execution state. No inference
 * capability is advertised by creating this metadata scope.
 */
class ProjectionRowWorkScope final {
public:
    /** @brief Publish immutable launch metadata for this recorder without reading counts. */
    explicit ProjectionRowWorkScope(ProjectionRowWork work):previous_(current_) {current_=work;}
    /** @brief Restore enclosing metadata on normal return and exception unwinding. */
    ~ProjectionRowWorkScope() {current_=previous_;}
    ProjectionRowWorkScope(const ProjectionRowWorkScope &)=delete;
    ProjectionRowWorkScope &operator=(const ProjectionRowWorkScope &)=delete;
    ProjectionRowWorkScope(ProjectionRowWorkScope &&)=delete;
    ProjectionRowWorkScope &operator=(ProjectionRowWorkScope &&)=delete;
    /** @brief Borrow current recorder metadata without consulting execution-owned values.
     * @param physicalRows Native matrix/scratch stride of the imminent launch.
     * @return Current work descriptor, or nullptr for the original full/prefix path.
     * @throws std::logic_error If metadata is accidentally reused for another matrix.
     */
    [[nodiscard]] static const ProjectionRowWork *workFor(int physicalRows) {
        if(physicalRows<=0||(current_&&current_->physicalRows()!=physicalRows))
            throw std::logic_error("Projection work recording scope disagrees with physical matrix rows");
        return current_?&*current_:nullptr;
    }
private:
    std::optional<ProjectionRowWork> previous_; ///< Enclosing immutable recorder metadata.
    inline static thread_local std::optional<ProjectionRowWork> current_;
};
/** @brief Lightweight host native-wrapper getter; GPU TUs need no tensor interface dependency.
 * @param physicalRows Exact captured matrix/scratch row capacity.
 * @return Borrowed immutable recorder metadata, never a host copy of live values.
 * @throws std::logic_error For a physical-geometry mismatch.
 */
[[nodiscard]] inline const ProjectionRowWork *currentProjectionRowWork(int physicalRows) {
    return ProjectionRowWorkScope::workFor(physicalRows);
}
} // namespace llaminar2
#undef LLAMINAR_PROJECTION_WORK_HD
