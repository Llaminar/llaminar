/**
 * @file RecordedHeadCohort.h
 * @brief Closed recorded-input compute cohort and bounded diagnostic round admission.
 *
 * This performance-only contract preserves the original local request parser.
 * A complete concurrent participant round can claim actual_tp for partitioned
 * projection compute, never for vocabulary publication or whole inference.
 * Metadata validation and host readiness contain no backend or tensor access.
 */
#pragma once
#include "RecordedHeadRequest.h"
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <future>
#include <mutex>
#include <span>

namespace llaminar2::test::recorded_head
{
/**
 * @brief One immutable equal-slice cohort sharing the same recorded source/trajectory.
 *
 * Device order is explicit membership. Logical ranks are vector positions, not
 * device ordinals. All member source, arithmetic and trajectory fields derive
 * from the one already validated template, so independent requests cannot mix.
 */
class CohortRequest final
{
public:
    /**
     * @brief Parse one closed cohort before backend initialization or payload access.
     * @param j Schema, scope, boundary, local member template and ordered devices only.
     * @return Immutable complete request; aggregate output uses the template output.
     * @throws std::invalid_argument For incomplete, duplicated or unsupported membership.
     * @note The inner scope remains local_shape until real cohort execution passes.
     */
    [[nodiscard]] static CohortRequest fromJson(const nlohmann::json &j)
    {
        require(j.is_object() && j.size()==5 && j.at("schema")=="llaminar.recorded_head_cohort.v1" &&
            j.at("scope")=="actual_tp" && j.at("operation_boundary")=="partitioned_projection_compute",
            "Recorded cohort requires its exact concurrent-compute schema/boundary");
        auto source=parse(j.at("member_template"));
        require(source.participant==0 && source.first==0,"Recorded cohort template must name the first logical slice");
        const auto &devices=j.at("device_ordinals");
        require(devices.is_array() && devices.size()==std::size_t(source.degree),"Recorded cohort membership differs from slice degree");
        std::vector<Request> members;std::vector<int> seen;
        for(std::size_t rank=0;rank<devices.size();++rank)
        {
            const int ordinal=integer(devices.at(rank));
            require(ordinal>=0 && std::find(seen.begin(),seen.end(),ordinal)==seen.end(),"Recorded cohort has an invalid or duplicate physical device");
            seen.push_back(ordinal);
            auto member=source.document;member["device_ordinal"]=ordinal;
            member["slice_participant"]=rank;member["first_column"]=rank*std::size_t(source.n);
            members.push_back(parse(member));
        }
        require(!members.empty() && members.front().ordinal==source.ordinal,"Recorded cohort template device must equal its first member");
        return CohortRequest(std::move(source),std::move(members),j);
    }
    /** @return Original exact qualified row/source/policy and aggregate destination. */
    [[nodiscard]] const Request &source() const noexcept {return source_;}
    /** @return Complete immutable logical-rank ordered participant requests. */
    [[nodiscard]] const std::vector<Request> &members() const noexcept {return members_;}
    /** @return Original authored cohort document retained in the final evidence. */
    [[nodiscard]] const nlohmann::json &document() const noexcept {return document_;}
private:
    /** @brief Construction is restricted to successful total validation. */
    CohortRequest(Request source,std::vector<Request> members,nlohmann::json document)
        :source_(std::move(source)),members_(std::move(members)),document_(std::move(document)){}
    const Request source_;
    const std::vector<Request> members_;
    const nlohmann::json document_;
};

/**
 * @brief One performance-local ready/release gate, with standard thirty-second admission.
 *
 * All original GPU worker callbacks are queued before the controller releases
 * them. Admission failure aborts already waiting callbacks; it cannot strand a
 * participant barrier. This coordinates diagnostic submission, not GPU state.
 */
class CohortRoundGate final
{
public:
    /** @param participants Exact supported cohort degree, never a capacity. */
    explicit CohortRoundGate(int participants):participants_(participants),
        deadline_(std::chrono::steady_clock::now()+std::chrono::seconds(30))
    {require(participants==1||participants==2||participants==4,"Invalid recorded cohort gate degree");}
    /**
     * @brief Publish one successful binding admission and await controller release.
     * @param rank Original immutable logical participant index.
     * @throws std::runtime_error For aborted/timed-out rounds; no GPU work is submitted here.
     */
    void arriveAndWait(int rank)
    {
        std::unique_lock lock(mutex_);
        if(rank<0 || rank>=participants_ || arrived_[std::size_t(rank)])
        {aborted_=true;changed_.notify_all();throw std::invalid_argument("Duplicate or foreign cohort admission rank");}
        arrived_[std::size_t(rank)]=true;++ready_;changed_.notify_all();
        if(!changed_.wait_until(lock,deadline_,[&]{return released_||aborted_;}))
        {aborted_=true;changed_.notify_all();}
        if(aborted_)throw std::runtime_error("Recorded cohort round admission aborted or exceeded30s");
    }
    /**
     * @brief Release every admitted callback only after all original members are ready.
     * @throws std::runtime_error For missing/failed admission, preserving all queued owners.
     */
    void releaseWhenReady()
    {
        std::unique_lock lock(mutex_);
        if(!changed_.wait_until(lock,deadline_,[&]{return ready_==participants_||aborted_;}))
        {aborted_=true;changed_.notify_all();}
        if(aborted_)throw std::runtime_error("Recorded cohort cannot release an incomplete round");
        if(released_)throw std::logic_error("Recorded cohort gate was already released");
        released_ns_=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
        released_=true;changed_.notify_all();
    }
    /** @return Exact common monotonic host release timestamp, never a GPU clock. */
    [[nodiscard]] std::int64_t releaseHostNanoseconds() const
    {
        std::lock_guard lock(mutex_);
        if(!released_)throw std::logic_error("Recorded cohort gate has not released");
        return released_ns_;
    }
    /** @brief Wake all admitted callbacks when setup/queue/launch fails; idempotent. */
    void abort() noexcept
    {std::lock_guard lock(mutex_);aborted_=true;changed_.notify_all();}
private:
    const int participants_;
    const std::chrono::steady_clock::time_point deadline_;
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    std::array<bool,4> arrived_{};
    int ready_=0;
    std::int64_t released_ns_=0;
    bool released_=false,aborted_=false;
};

/**
 * @brief Preserve source/trajectory/precision identity while removing only varying physical slice axes.
 * @param request Exact original member request, not a synthetic operation name.
 * @return Canonical JSON identity used verbatim by the existing scaling reporter's workload field.
 * @note No model, activation, logits or runtime payload is read or hashed.
 */
[[nodiscard]] inline nlohmann::json cohortWorkloadIdentity(const Request &request)
{
    auto identity=request.document;
    for(const char *axis:{"output","device_ordinal","slice_degree","slice_participant","first_column","local_n"})identity.erase(axis);
    return identity;
}

/**
 * @brief Observe a worker's immediate admission failure before waiting on missing callbacks.
 * @param work Newly returned actual submitAsync future, before controller release.
 * @throws For rejected, invalid or deferred work; successful callbacks cannot
 *         already complete while their round gate has not been released.
 * @note Existing CUDA/HIP workers return a ready exceptional future when stopped.
 */
inline void requireQueuedCohortAdmission(std::future<void> &work)
{
    require(work.valid(),"Recorded cohort worker returned no future");
    const auto state=work.wait_for(std::chrono::milliseconds(0));
    if(state==std::future_status::ready)
    {work.get();throw std::logic_error("Recorded cohort callback completed before release");}
    require(state!=std::future_status::deferred,"Recorded cohort requires its actual device worker queue");
}

/** @brief One actual participant event interval; rank/round are explicit identities. */
struct CohortRankInterval {int round=0,rank=0;double latency_us=0;};
/**
 * @brief Validate a complete paired round before deriving its slowest participant.
 * @param intervals Every original rank exactly once, with finite positive device intervals.
 * @param degree Exact equal-slice cohort degree.
 * @param round Original nonnegative measurement-round index.
 * @return Maximum rank interval; median reduction remains the existing scaling reporter.
 */
[[nodiscard]] inline double cohortMaximum(std::span<const CohortRankInterval> intervals,int degree,int round)
{
    require((degree==1||degree==2||degree==4) && round>=0 && intervals.size()==std::size_t(degree),
        "Recorded cohort round is incomplete");
    std::array<bool,4> seen{};double maximum=0;
    for(const auto &value:intervals)
    {
        require(value.round==round && value.rank>=0 && value.rank<degree && !seen[std::size_t(value.rank)] &&
            std::isfinite(value.latency_us) && value.latency_us>0,"Recorded cohort round has foreign, duplicate or invalid intervals");
        seen[std::size_t(value.rank)]=true;maximum=std::max(maximum,value.latency_us);
    }
    return maximum;
}
} // namespace llaminar2::test::recorded_head
