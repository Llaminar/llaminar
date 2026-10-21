/**
 * @file AdaptiveACKReadyCancellationNative.cuh
 * @brief Backend-shared diagnostic cancellation and exact original graph authentication.
 *
 * Thin CUDA/HIP wrappers supply native graph operations. Captured argument
 * metadata is inspected at setup; GPU values remain device-owned. A distinct
 * frontier error prevents an unrelated failed response from satisfying the test.
 */
#pragma once
#include "AdaptiveACKReadyCancellation.h"
#include "kernels/common/SamplingMath.h"
#include "kernels/qwen4exp/AdaptiveImmutableRowCacheState.h"
#include <array>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace llaminar2::test::adaptive_native::ACK_CANCEL_NAMESPACE
{
using namespace qwen4exp;
using namespace qwen4exp::adaptive_lookup;

/** @brief Fail setup before graph mutation if the exact native contract is absent. */
inline void require(bool valid, const char *message)
{
    if (!valid)
        throw std::runtime_error(message);
}
/**
 * @brief Read stable CPU-authored mapped metadata with the production fresh-load idiom.
 * @param word Actual mapped scalar, already release-published by the source operation.
 * @return Fresh value after the original validate/ACK/fill graph frontier.
 * @note Stability after ACK prevents concurrent mutation; a noncached load also
 * prevents a prior retained replay's mapped cache line from satisfying this check.
 */
__device__ __forceinline__ std::uint64_t freshMapped(const std::uint64_t *word)
{
#if defined(__HIPCC__)
    return __builtin_nontemporal_load(word);
#else
    return __ldcv(word);
#endif
}
/** @brief Preserve the first controller error using the production cancellation equation. */
__device__ __forceinline__ void cancel(AdaptiveLookupDeviceArgs a, bool frontier)
{
    (void)sampling_math::fail_device_generation_control(
        a.control, frontier ? sampling_math::DeviceGenerationError::InvalidController
                            : sampling_math::DeviceGenerationError::ImmutableStorageFailure);
}
/**
 * @brief Cancel only the target9 frame after actual ACK retirement and native fill.
 * @param a Original production operands copied from authenticated captured arguments.
 * @note Row7/empty warm and reset frames pass unchanged through this same node.
 * The only authored mutation is the original canonical controller cancellation.
 */
__global__ void cancelAfterValidatedACKBeforeReady(AdaptiveLookupDeviceArgs a)
{
    if (blockIdx.x || threadIdx.x)
        return;
    const auto &f = *reinterpret_cast<const FrameControl *>(a.storage.frame + a.plan.frame.control);
    if (f.unique_rows > f.original_ids || f.original_ids > a.plan.maximum_ids)
    {
        cancel(a, false);
        return;
    }
    const auto *unique = reinterpret_cast<const UniqueRecord *>(a.storage.frame + a.plan.frame.unique);
    const UniqueRecord *target = nullptr;
    for (std::uint64_t i = 0; i < f.unique_rows; ++i)
        if (unique[i].row == 9)
            target = unique + i;
    if (!target)
        return;
    const auto &state = *a.storage.protocol;
    bool valid = f.phase == Phase::Validated && f.status == Status::Success && f.source_rows &&
                 f.source_rows <= f.unique_rows && f.observed_ids == f.original_ids && f.policy_started &&
                 !*a.error && *a.healthy == 1 && state.admissionGeneration &&
                 state.generation == state.admissionGeneration && state.issuedSequence &&
                 state.issuedSequence == state.consumedSequence && !state.status && !state.reserved &&
                 freshMapped(&a.channel.base.acknowledgement->consumedSequence) == state.consumedSequence &&
                 freshMapped(&a.channel.base.completion->publication) == state.issuedSequence &&
                 freshMapped(&a.channel.base.completion->generation) == state.admissionGeneration &&
                 freshMapped(&a.channel.base.completion->idCount) == f.source_rows &&
                 freshMapped(&a.channel.base.completion->nativeBytes) == f.source_rows * a.plan.row_bytes &&
                 !(target->flags & InitialHit) && (target->flags & Probed) &&
                 target->response_coordinate < f.source_rows;
    if (!valid)
    {
        cancel(a, false);
        return;
    }
    const auto *acks = reinterpret_cast<const ImmutableRowInvalidationAcknowledgement *>(
        a.storage.frame + a.plan.frame.acknowledgements);
    const bool reserved = target->flags & ReservedFill;
    const auto token = reserved ? target->expected_slot_generation : 0;
    valid = acks[target->response_coordinate].destination_grant == token;
    if (a.plan.slots)
    {
        valid = valid && a.plan.row_width == RowKeyWidth::UInt32 && reserved &&
                target->slot < a.plan.slots && token && (target->flags & 4U);
        if (valid)
        {
            const auto &slot = reinterpret_cast<const Slot32 *>(a.storage.cache + a.plan.cache.slots)[target->slot];
            valid = slot.row == 9 && slot.generation == token &&
                    (slot.flags & StateMask) == static_cast<unsigned>(SlotState::Reserved) &&
                    (slot.flags & FramePin);
            const auto *filled = a.storage.cache + a.plan.cache.payload + target->slot * a.plan.slot_stride;
            const auto *privateNative = a.storage.frame + a.plan.frame.native_response +
                                        target->response_coordinate * a.plan.row_bytes;
            for (std::size_t byte = 0; valid && byte < a.plan.row_bytes; ++byte)
                valid = filled[byte] == privateNative[byte];
        }
    }
    else
        valid = valid && !reserved && target->slot == None && !token;
    cancel(a, valid);
}

/** @brief Authenticate all source aliases, not just equal table shape. */
inline bool sameSource(ImmutableSubordinateLookupDeviceView a, ImmutableSubordinateLookupDeviceView b)
{
    return a.base.request == b.base.request && a.base.completion == b.base.completion &&
           a.base.acknowledgement == b.base.acknowledgement && a.base.ids == b.base.ids &&
           a.base.nativeRows == b.base.nativeRows && a.base.table == b.base.table &&
           a.base.maximumIds == b.base.maximumIds && a.base.rowBytes == b.base.rowBytes &&
           a.intents == b.intents && a.acknowledgements == b.acknowledgements;
}
/** @brief Compare complete immutable operands without reading any GPU value or hashing payloads. */
inline bool sameArguments(const AdaptiveLookupDeviceArgs &a, const AdaptiveLookupDeviceArgs &b)
{
    return a.plan == b.plan && a.storage.cache == b.storage.cache && a.storage.frame == b.storage.frame &&
           a.storage.protocol == b.storage.protocol &&
           a.query.token_rows.captureIdentity() == b.query.token_rows.captureIdentity() &&
           a.query.ids == b.query.ids && a.query.id_capacity == b.query.id_capacity &&
           a.query.output == b.query.output && a.query.output_elements == b.query.output_elements &&
           sameSource(a.channel, b.channel) && a.healthy == b.healthy && a.error == b.error &&
           a.control == b.control;
}
/**
 * @brief Find the exact four native functions from runtime-owned inspection records.
 * @param graph Original production capture, not yet instantiated.
 * @return Identities in validateResponse, acknowledge, fill, ready order.
 * @throws std::exception If a symbol is unresolved, nested, absent, duplicated or aliases another function.
 */
inline std::array<std::uintptr_t, 4> identities(IGPUGraphCapture &graph)
{
    std::vector<GPUGraphKernelNodeInfo> records;
    std::string error;
    require(graph.inspectKernelNodes(records, &error), "Cancellation native kernel inspection failed");
    std::array<std::uintptr_t, 4> result{};
    const std::array<const char *, 4> names{"validateResponse", "acknowledge", "fill", "ready"};
    for (const auto &record : records)
    {
        require(record.name_resolved && record.nesting_depth == 0 && record.function_identity,
                "Cancellation requires resolved flat original native kernel identities");
        require(record.name.find("adaptive_lookup") != std::string::npos,
                "Cancellation capture contains a foreign native namespace");
        for (std::size_t i = 0; i < names.size(); ++i)
            if (record.name.find(names[i]) != std::string::npos)
            {
                require(!result[i], "Cancellation native target is ambiguous");
                result[i] = record.function_identity;
            }
    }
    for (std::size_t i = 0; i < result.size(); ++i)
    {
        require(result[i], "Cancellation original native target is absent");
        for (std::size_t j = 0; j < i; ++j)
            require(result[i] != result[j], "Cancellation native function identities alias");
    }
    return result;
}
/**
 * @brief Add one native predecessor to Ready, preserving every original node and edge.
 * @param capture Original uninstantiated backend capture.
 * @param expected Exact retained production arguments to authenticate at all four nodes.
 * @param stream Exact capture stream; null/default handles are rejected before native API calls.
 * @throws std::exception Before mutation on malformed topology/arguments, or fatally to the caller on native edit failure.
 */
inline void splice(IGPUGraphCapture &capture, const AdaptiveLookupDeviceArgs &expected, void *stream)
{
    require(stream && reinterpret_cast<std::uintptr_t>(stream) > 2,
            "Cancellation requires an exact non-default native stream");
    require(expected.plan.device.ACK_DEVICE_PREDICATE() && expected.plan.row_width == RowKeyWidth::UInt32 &&
                (expected.plan.slots == 0 || expected.plan.slots == 2),
            "Cancellation fixture supports only its actual C0/C2 row32 backend");
    auto *owner = dynamic_cast<ACK_CAPTURE *>(&capture);
    require(owner && owner->deviceOrdinal() == expected.plan.device.ordinal,
            "Cancellation capture belongs to a foreign backend/device");
    const auto ids = identities(capture);
    const auto graph = owner->graph();
    std::size_t count = 0;
    ACK_CHECK(ACK_RT(GraphGetNodes)(graph, nullptr, &count));
    std::vector<ACK_NODE> all(count);
    ACK_CHECK(ACK_RT(GraphGetNodes)(graph, all.data(), &count));
    std::array<ACK_NODE, 4> selected{};
    for (const auto node : all)
    {
        ACK_NODE_TYPE type{};
        ACK_CHECK(ACK_RT(GraphNodeGetType)(node, &type));
        require(type == ACK_RT(GraphNodeTypeKernel), "Cancellation capture must contain only flat native kernels");
        auto params = originalParameters(node);
        const auto identity = originalIdentity(params);
        for (std::size_t i = 0; i < ids.size(); ++i)
            if (identity == ids[i])
            {
                require(!selected[i] && params.kernelParams && params.kernelParams[0],
                        "Cancellation captured argument metadata is missing or ambiguous");
                auto actual = expected;
                static_assert(std::is_trivially_copyable_v<AdaptiveLookupDeviceArgs>);
                std::memcpy(&actual, params.kernelParams[0], sizeof(actual));
                require(sameArguments(actual, expected), "Cancellation node changed original launch operands");
                selected[i] = node;
            }
    }
    for (const auto node : selected)
        require(node, "Cancellation inspector identity did not match an actual native graph node");
    for (std::size_t i = 1; i < selected.size(); ++i)
    {
        std::size_t n = 0;
        ACK_CHECK(ACK_DEPENDENCIES(selected[i], nullptr, &n));
        std::vector<ACK_NODE> before(n);
        ACK_CHECK(ACK_DEPENDENCIES(selected[i], before.data(), &n));
        require(n == 1 && before.front() == selected[i - 1],
                "Cancellation original validate/ack/fill/ready dependency chain changed");
    }
    auto arguments = expected;
    void *kernelArguments[]{&arguments};
    ACK_KERNEL_PARAMS addition{};
    addition.func = reinterpret_cast<void *>(cancelAfterValidatedACKBeforeReady);
    addition.gridDim = dim3(1);
    addition.blockDim = dim3(1);
    addition.kernelParams = kernelArguments;
    ACK_NODE cancellation{};
    ACK_CHECK(ACK_RT(GraphAddKernelNode)(&cancellation, graph, &selected[2], 1, &addition));
    ACK_CHECK(ACK_ADD_DEPENDENCIES(graph, &cancellation, &selected[3], 1));
    // The original fill→Ready edge remains. Ready now also waits on the one
    // diagnostic node, whose only predecessor is the original fill frontier.
}
} // namespace llaminar2::test::adaptive_native::ACK_CANCEL_NAMESPACE
