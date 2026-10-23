/**
 * @file ROCmNativeM1Reduction.h
 * @brief Exact-once ordinary M1 reduction tickets and live projection tile geometry.
 *
 * Producers retain their original decoder, launch geometry and source arithmetic
 * policy. A successful producer issues a noncopyable ticket on its exact stream.
 * Submission consumes tickets explicitly; destruction never submits GPU work.
 */
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <exception>
#include <cstdio>
#include "kernels/common/NativeVNNIGroupedDecodePolicy.h"

namespace llaminar2::rocm
{
/** @brief One original reducer's physical column/partition geometry. */
struct NativeM1ReductionShape
{
    int columns=0, partitions=0, tile_columns=0, columns_per_thread=0;
};
/** @brief Bounded copied kernel ABI; every interval names live original tiles. */
struct NativeM1ReductionGrid
{
    std::array<const float *,4> partials{};
    std::array<float *,4> outputs{};
    std::array<int,4> columns{},partitions{},offsets{};
    int count=0,tiles=0;
};
/**
 * @brief Validate exact reducer geometry and concatenate live tile intervals.
 * @param shapes One to four original reducers with one identical compiled geometry.
 * @return Pointer-free live-grid ABI, ready for producer-issued pointer publication.
 * @throws std::invalid_argument For incompatible or unrepresentable geometry.
 */
inline NativeM1ReductionGrid nativeM1ReductionGrid(std::span<const NativeM1ReductionShape> shapes)
{
    if(shapes.empty() || shapes.size()>4)throw std::invalid_argument("M1 reducer group requires one to four original reducers");
    NativeM1ReductionGrid result;
    const auto geometry=shapes.front();
    if(!((geometry.tile_columns==64 && geometry.columns_per_thread==1) ||
         ((geometry.tile_columns==128 || geometry.tile_columns==256) &&
          (geometry.columns_per_thread==1 || geometry.columns_per_thread==2))))
        throw std::invalid_argument("M1 reducer geometry has no compiled original specialization");
    for(const auto &s:shapes)
    {
        if(s.columns<=0 || s.partitions<=1 || s.partitions>NativeVNNIGroupedDecodePolicy::maximum_k_partitions ||
           s.tile_columns!=geometry.tile_columns || s.columns_per_thread!=geometry.columns_per_thread)
            throw std::invalid_argument("M1 reducer group changes an original geometry or partition count");
        const int tiles=1+(s.columns-1)/s.tile_columns;
        if(tiles>std::numeric_limits<int>::max()-result.tiles)
            throw std::invalid_argument("M1 reducer live tile sum overflows");
        const auto i=static_cast<std::size_t>(result.count++);
        result.offsets[i]=result.tiles;result.columns[i]=s.columns;result.partitions[i]=s.partitions;
        result.tiles+=tiles;
    }
    return result;
}
struct NativeM1ReductionIssuer;
/** @brief A successful original producer's exact-once deferred publication authority. */
class NativeM1ReductionTicket
{
public:
    /** @brief Construct an unissued ticket; only the native producer can issue it. */
    NativeM1ReductionTicket()=default;
    /** @brief Reject normal abandonment; exceptional transaction retirement aborts without enqueue. */
    ~NativeM1ReductionTicket() noexcept
    {
        if(state_!=State::Pending)return;
        if(std::uncaught_exceptions()>0){state_=State::Aborted;return;}
        std::fputs("Fatal: native M1 producer reduction ticket abandoned before explicit submission\n",stderr);
        std::terminate();
    }
    /** @return Actual successfully issued original producer geometry, never inferred policy. */
    [[nodiscard]] const NativeM1ReductionShape &producerShape() const
    {
        if(state_!=State::Pending && state_!=State::Direct)
            throw std::invalid_argument("M1 producer geometry requires a live issued ticket");
        return shape_;
    }
    NativeM1ReductionTicket(const NativeM1ReductionTicket &)=delete;
    NativeM1ReductionTicket &operator=(const NativeM1ReductionTicket &)=delete;
    NativeM1ReductionTicket(NativeM1ReductionTicket &&)=delete;
    NativeM1ReductionTicket &operator=(NativeM1ReductionTicket &&)=delete;
private:
    friend struct NativeM1ReductionIssuer;
    friend bool submitNativeM1Reductions(std::span<NativeM1ReductionTicket>,int,void *);
    enum class State : std::uint8_t { Empty,Direct,Pending,Consumed,Aborted };
    State state_=State::Empty;
    NativeM1ReductionShape shape_;
    const float *partials_=nullptr;
    float *output_=nullptr;
    int device_=-1;
    void *stream_=nullptr;
};
/**
 * @brief Enqueue the exact original serial producer and issue its reduction ticket.
 * @param ticket Unissued ticket receiving only successful native producer publication.
 * @param A Existing original quantized activation.
 * @param payload Existing native prepared payload.
 * @param scales Existing native primary scales.
 * @param mins Existing optional source-owned secondary metadata.
 * @param emins Existing optional source-owned extended metadata.
 * @param output Original output owner; KB1 publishes directly.
 * @param scale_a Original optional per-row scale.
 * @param partials Distinct admitted persistent partial slot, never shared by pending producers.
 * @param n Original physical output columns.
 * @param k Original reduction columns.
 * @param execution Physical decoder codebook.
 * @param policy Original source arithmetic-policy codebook.
 * @param device Exact original ROCm device.
 * @param stream Exact non-null producer stream.
 * @param block_scales Original optional blockwise activation scales.
 * @return Whether original submission succeeded; failure never retries another route.
 */
bool enqueueNativeM1Producer(NativeM1ReductionTicket &ticket,const std::int8_t *A,
    const std::uint8_t *payload,const void *scales,const void *mins,const void *emins,
    float *output,const float *scale_a,float *partials,int n,int k,
    std::uint8_t execution,std::uint8_t policy,int device,void *stream,const float *block_scales);
/**
 * @brief Consume one wave's tickets with exact geometry buckets and ordered RN folds.
 * @param tickets At most four successfully issued tickets; KB1 needs no reducer.
 * @param device Producer device, unchanged across this wave.
 * @param stream Exact producer stream, also owning every reducer and output publication.
 * @return Whether all native submissions succeeded; consumed tickets cannot be replayed.
 * @throws std::invalid_argument For absent, consumed, foreign or aliasing tickets.
 */
bool submitNativeM1Reductions(std::span<NativeM1ReductionTicket> tickets,int device,void *stream);
} // namespace llaminar2::rocm
