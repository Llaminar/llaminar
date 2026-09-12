/**
 * @file Test__GPUEffectiveKVSnapshot.cpp
 * @brief Cold captured attention diagnostics across CUDA/ROCm cache formats.
 *
 * The snapshot arena is bound before the first append or attention execution.
 * Real production stages must retain that complete output manifest through
 * capture and request reset. No eager attention warmup may discover missing
 * descriptors. Backend-specialized graph integration binaries include this
 * translation unit in their existing ProductionParityPreflight registrations.
 */

#include <gtest/gtest.h>
#include "backends/GPUDeviceContextPool.h"
#include "backends/IGPUGraphCapture.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/compute_stages/stages/AttentionComputeStage.h"
#include "execution/compute_stages/stages/KVCacheAppendStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "kernels/KernelFactory.h"
#include "kernels/HybridKVCacheConfig.h"
#include "kernels/cpu/turboquant/TurboQuantContext.h"
#include "backends/BackendManager.h"
#include "tensors/GpuTensorView.h"
#include "kernels/attention/AttentionWorkspaceContract.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "planning/MemoryPlanner.h"
#include "planning/ModelMemoryProfile.h"
#include "execution/prefix_cache/DeviceHotPrefixStorageBackend.h"
#include "execution/prefix_cache/PrefixPayloadLayout.h"
#include "utils/DebugEnv.h"
#include "../../../utils/TestTensorFactory.h"
#include "../../kernels/KVCacheTestWorkspace.h"

#include <cstring>
#include <algorithm>
#include <array>
#include <map>
#include <tuple>
#include <type_traits>
#include <cmath>

using namespace llaminar2;
using namespace llaminar2::test;

namespace
{
/** @return The explicitly linked test backend, never a substitute device. */
std::optional<DeviceId> snapshotTestDevice()
{
#if defined(GPU_CONTEXT_TEST_BACKEND_CUDA)
    ensureNvidiaFactoryRegistered();
    if (GPUDeviceContextPool::instance().hasNvidiaSupport())
        return DeviceId::cuda(0);
#elif defined(GPU_CONTEXT_TEST_BACKEND_ROCM)
    ensureAMDFactoryRegistered();
    if (GPUDeviceContextPool::instance().hasAMDSupport())
        return DeviceId::rocm(0);
#endif
    return std::nullopt;
}

/**
 * @brief Price the concrete hybrid and attention-only cache, not its dtype label.
 *
 * Hybrid GPU Q8 caches use linear Q8_1 keys, while attention-only Q8 caches
 * use anchored AQ8 keys. Both enter the same physical-memory authority. This
 * model-free production-factory sweep compares real serialized layouts with
 * the admitted prefix slab before allocating it, including sharded heads and
 * every floating-point hybrid format on each linked backend.
 */
TEST(GPUPrefixCacheAccounting, NativeLayoutMatchesAdmittedSlab)
{
    const auto available = snapshotTestDevice();
    if (!available)
        GTEST_SKIP() << "Linked GPU backend unavailable";
    const DeviceId device = *available;
    auto &worker = GPUDeviceContextPool::instance().getContext(device);

    for (const bool hybrid_model : {false, true})
    for (const auto precision : {ActivationPrecision::FP16, ActivationPrecision::BF16,
                                 ActivationPrecision::FP32, ActivationPrecision::Q8_1,
                                 ActivationPrecision::TQ4, ActivationPrecision::TQ8})
    for (const int width : {64, 128, 256})
    for (const int local_heads : {1, 2})
    {
        const bool turboquant = precision == ActivationPrecision::TQ4 ||
                                precision == ActivationPrecision::TQ8;
        // Hybrid TQ has no installed factory; its admission rejection has a
        // separate device-free test. Do not substitute another storage codec.
        if (hybrid_model && turboquant)
            continue;
        SCOPED_TRACE(::testing::Message() << "hybrid=" << hybrid_model
            << " precision=" << activationPrecisionToString(precision)
            << " width=" << width << " heads=" << local_heads);
        ModelMemoryProfile profile;
        profile.architecture = hybrid_model ? "qwen35" : "qwen2";
        profile.n_layers = 4;
        profile.n_heads = profile.n_kv_heads = 2;
        profile.head_dim = width;
        profile.d_model = 2 * width;
        profile.d_ff = 4 * profile.d_model;
        profile.vocab_size = 128;
        profile.max_seq_len = 64;
        HybridKVCacheConfig hybrid;
        if (hybrid_model)
        {
            profile.full_attention_interval = 2;
            profile.gdn_conv_kernel_size = hybrid.gdn_conv_kernel_size = 3;
            profile.gdn_state_size = hybrid.gdn_state_size = 64;
            profile.gdn_inner_size = hybrid.gdn_inner_size = 128;
            profile.gdn_group_count = hybrid.gdn_group_count = 2;
            profile.gdn_time_step_rank = hybrid.gdn_time_step_rank = 2;
            hybrid.n_heads = 2;
            hybrid.local_n_heads = local_heads;
            hybrid.layer_types = {"gdn", "full_attention", "gdn", "full_attention"};
        }
        DevicePlanConfig cfg;
        cfg.world_rank = 0;
        cfg.device = device;
        cfg.device_total_bytes = cfg.device_free_bytes = 1u << 30;
        cfg.device_compute_units = 1;
        cfg.batch_size = 1;
        cfg.max_seq_len = 64;
        cfg.activation_seq_len = 16;
        cfg.total_shards = 2 / local_heads;
        cfg.local_kv_heads = local_heads;
        cfg.kv_precision = activationPrecisionToString(precision);
        cfg.prefix_cache = PrefixCacheRuntimeConfig{};
        cfg.associated_host_memory = PhysicalMemoryResource{
            .world_rank = 0, .device = DeviceId::cpu(),
            .total_bytes = 1u << 30, .admission_available_bytes = 1u << 30};
        cfg.prefix_cache.device_budget_bytes = 3u << 20;
        cfg.prefix_cache.ram_budget_bytes = 8u << 20;
        cfg.prefix_cache.disk_budget_bytes = 0;
        const auto plan = MemoryPlanner::plan(profile, {cfg});
        auto authority = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(plan.physicalPlan()), 0);
        llaminar::v2::kernels::KVCacheConfig cache_config;
        cache_config.device = device;
        cache_config.precision = precision;
        cache_config.num_layers = 4;
        cache_config.batch_size = 1;
        cache_config.max_seq_len = 64;
        cache_config.n_kv_heads = 2;
        cache_config.local_n_kv_heads = local_heads;
        cache_config.head_dim = width;
        cache_config.hybrid_config = hybrid_model ? &hybrid : nullptr;
        std::unique_ptr<TurboQuantContext> rotations;
        if (turboquant)
            rotations = std::make_unique<TurboQuantContext>(width, 42);
        cache_config.turboquant_ctx = rotations.get();
        cache_config.physical_memory_authority = authority;
        auto cache = llaminar::v2::kernels::KernelFactory::createKVCache(cache_config);
        ASSERT_NE(cache, nullptr);
        const auto layout = buildDensePrefixPayloadLayout(*cache, device,
            cfg.prefix_cache.block_size, 0, profile.vocab_size * sizeof(float));
        const size_t actual_slab_bytes = prefixCacheWholeBlockReservationBytes(
            cfg.prefix_cache.device_budget_bytes, layout.totalBytes());
        const auto entry = std::find_if(plan.devices.begin(), plan.devices.end(),
            [&](const auto &candidate) { return candidate.device() == device; });
        ASSERT_NE(entry, plan.devices.end());
        EXPECT_EQ(entry->kv_cache_bytes(), cache_config.estimateBytes());
        EXPECT_EQ(entry->prefix_cache_device_hot_bytes(), actual_slab_bytes)
            << "The native serialized block is " << layout.totalBytes() << " bytes";
        if (entry->prefix_cache_device_hot_bytes() == actual_slab_bytes)
        {
            std::string error;
            auto slab = DeviceHotPrefixStorageBackend::create(device,
                cfg.prefix_cache.device_budget_bytes, layout.totalBytes(), authority, &error);
            ASSERT_NE(slab, nullptr) << error;
            EXPECT_EQ(slab->reservedBytes(), actual_slab_bytes);
        }
        // Construction owns asynchronous state initialization; drain its exact
        // stream at this test-only destruction boundary before the next case.
        ASSERT_TRUE(worker.synchronizeStreamChecked(worker.defaultStream()));
    }
}

/** @brief Restore typed diagnostic configuration even when an assertion exits. */
class EffectiveKVSnapshotScope
{
public:
    /** @brief Enable complete K/V diagnostics before constructing any stage. */
    EffectiveKVSnapshotScope() : saved_(debugEnv().attention)
    {
        mutableDebugEnv().attention.debug_effective_kv_snapshot = true;
        mutableDebugEnv().attention.debug_effective_kv_snapshot_layer.reset();
    }
    /** @brief Do not leak diagnostic state into the next preflight case. */
    ~EffectiveKVSnapshotScope() { mutableDebugEnv().attention = saved_; }
    EffectiveKVSnapshotScope(const EffectiveKVSnapshotScope &) = delete;
    EffectiveKVSnapshotScope &operator=(const EffectiveKVSnapshotScope &) = delete;
private:
    std::remove_cvref_t<decltype(debugEnv().attention)> saved_;
};

/** @brief Native dtype and head width, independent of model activation dtype. */
using NativeAttentionCase = std::tuple<TensorType, int>;
/** @brief Small model-free native-cache arithmetic matrix on either real GPU. */
class GPUNativeKVAttentionMath : public ::testing::TestWithParam<NativeAttentionCase> {};

/** @brief Bind the production workspace BOM to exact test geometry, not maxima. */
class NativeAttentionWorkspace
{
public:
    /** @brief Borrow one attention consumer and own its exact persistent arena. */
    NativeAttentionWorkspace(ITensorAttention &kernel, DeviceId device,
                             int rows, int heads, int width, int capacity)
        : consumer_(dynamic_cast<IWorkspaceConsumer &>(kernel)),
          requirements_(attention_workspace::requirements({
              .compact_query_rows = rows, .request_count = 2,
              .local_query_heads = heads, .local_kv_heads = 1,
              .head_dim = width, .context_rows = capacity,
              .include_device_params = true, .include_fp32_kv_conversion = false})),
          arena_(device, requirements_.total_bytes_with_alignment())
    {
        if (!arena_.allocate(requirements_))
            throw std::runtime_error("Native attention test workspace allocation failed");
        consumer_.bindWorkspace(&arena_);
    }
    /** @brief Retire borrowed addresses before releasing their arena. */
    ~NativeAttentionWorkspace() { consumer_.unbindWorkspace(); }
private:
    IWorkspaceConsumer &consumer_;
    WorkspaceRequirements requirements_;
    DeviceWorkspaceManager arena_;
};

/**
 * @brief Construct a native cache plus an exact expanded FP32 CPU oracle.
 * @param type Physical floating-point cache dtype.
 * @param rows Complete physical cache rows, including request strides.
 * @param width Logical elements per row.
 * @param phase Distinguishes keys from values and avoids symmetric operands.
 * @return Owned native bytes and their exact FP32 values, before device upload.
 */
std::pair<std::unique_ptr<ITensor>, std::vector<float>> nativeAttentionFixture(
    TensorType type, int rows, int width, int phase)
{
    std::vector<float> values(static_cast<size_t>(rows) * width);
    for (size_t i = 0; i < values.size(); ++i)
        values[i] = std::sin(float(i * 7 + phase) * 0.031f) * 0.7f;
    const std::vector<size_t> shape{static_cast<size_t>(rows), static_cast<size_t>(width)};
    std::unique_ptr<ITensor> tensor;
    if (type == TensorType::FP32)
        tensor = TestTensorFactory::createFP32(shape);
    else if (type == TensorType::BF16)
        tensor = TestTensorFactory::createBF16(shape);
    else
        tensor = TestTensorFactory::createFP16(shape);
    if (type == TensorType::FP32)
        std::memcpy(tensor->mutable_data(), values.data(), values.size() * sizeof(float));
    else
    {
        auto *bits = static_cast<uint16_t *>(tensor->raw_mutable_data());
        for (size_t i = 0; i < values.size(); ++i)
        {
            bits[i] = type == TensorType::BF16 ? simd::fp32_to_bf16(values[i]) : fp32_to_fp16(values[i]);
            values[i] = type == TensorType::BF16 ? simd::bf16_to_fp32(bits[i]) : fp16_to_fp32(bits[i]);
        }
    }
    return {std::move(tensor), std::move(values)};
}

/**
 * @brief Prove native prefill and every grouped M against the actual scalar path.
 *
 * Requests use unequal cache lengths and distinctive fixed-stride banks. A
 * graph has a constant number of nodes regardless of M; it must never conceal
 * a serial row replay. Serial rows are an isolated test oracle only. FP64
 * attention independently checks the scalar arithmetic as well as byte equality.
 */
TEST_P(GPUNativeKVAttentionMath, CapturedNativeRowsMatchSerialAndIndependentMath)
{
    const auto available = snapshotTestDevice();
    if (!available)
        GTEST_SKIP() << "Linked GPU backend unavailable";
    const DeviceId device = *available;
    auto &worker = GPUDeviceContextPool::instance().getContext(device);
    auto *backend = getBackendFor(device);
    ASSERT_NE(backend, nullptr);
    void *const stream = worker.defaultStream();
    const auto [type, width] = GetParam();
    constexpr int capacity = 320;
    constexpr int heads = 2; // GQA makes the query and cache strides distinct.
    auto [keys, key_values] = nativeAttentionFixture(type, 2 * capacity, width, 1);
    auto [values, value_values] = nativeAttentionFixture(type, 2 * capacity, width, 19);
    auto query = TestTensorFactory::createFP32Random({16, size_t(heads * width)}, -0.4f, 0.4f, 51);
    auto grouped = TestTensorFactory::createFP32Zeros({16, size_t(heads * width)});
    auto serial = TestTensorFactory::createFP32Zeros({16, size_t(heads * width)});
    auto counts = TestTensorFactory::createINT32({2});
    for (ITensor *tensor : {keys.get(), values.get(), static_cast<ITensor *>(query.get()),
                           static_cast<ITensor *>(grouped.get()), static_cast<ITensor *>(serial.get()),
                           static_cast<ITensor *>(counts.get())})
    {
        auto *owned = dynamic_cast<TensorBase *>(tensor);
        ASSERT_NE(owned, nullptr);
        ASSERT_TRUE(owned->ensureOnDevice(device, stream));
    }
    auto make_attention = [&] {
        auto kernel = llaminar::v2::kernels::KernelFactory::createAttention(
            query.get(), device.type, device.ordinal);
        kernel->setGPUStream(stream);
        return kernel;
    };
    auto grouped_kernel = make_attention();
    auto serial_kernel = make_attention();
    ASSERT_NE(grouped_kernel, nullptr);
    ASSERT_NE(serial_kernel, nullptr);
    auto *grouped_consumer = dynamic_cast<IWorkspaceConsumer *>(grouped_kernel.get());
    auto *serial_consumer = dynamic_cast<IWorkspaceConsumer *>(serial_kernel.get());
    ASSERT_NE(grouped_consumer, nullptr);
    ASSERT_NE(serial_consumer, nullptr);
    // The common binding owns all partial rows through the admitted M16 envelope.
    NativeAttentionWorkspace grouped_workspace(*grouped_kernel, device, 16, heads, width, capacity);
    NativeAttentionWorkspace serial_workspace(*serial_kernel, device, 1, heads, width, capacity);
    ASSERT_TRUE(worker.synchronizeStreamChecked(stream));

    for (int requests : {1, 2})
        for (int history : {17, 257})
            for (int rows = 1; rows <= 16 / requests; ++rows)
            {
                SCOPED_TRACE(::testing::Message() << "requests=" << requests
                    << " history=" << history << " rows=" << rows);
                const int total_rows = requests * rows;
                const int lengths[2] = {history + rows, history + rows + 11};
                ASSERT_TRUE(backend->hostToDevice(counts->gpu_data_ptr(), lengths,
                    sizeof(lengths), device.ordinal, stream));
                std::unique_ptr<IGPUGraphCapture> capture;
                worker.submitAndWait([&] { capture = worker.createGraphCapture(); });
                ASSERT_TRUE(capture->beginCapture());
                bool success = false;
                if (requests == 2)
                    success = grouped_kernel->compute_device_request_batch_decode_equivalent(
                        query.get(), keys.get(), values.get(),
                        static_cast<const int *>(counts->gpu_data_ptr()), grouped.get(),
                        requests, rows, capacity, heads, 1, width, true);
                else
                {
                    success = grouped_kernel->prepareDynamicAttnParamsFromDeviceSequenceState(
                        static_cast<const int *>(counts->gpu_data_ptr()),
                        rows, rows, stream, capacity);
                    if (success && rows > 1)
                        success = grouped_kernel->compute_verifier_rows_decode_equivalent(
                            query.get(), keys.get(), values.get(), grouped.get(),
                            rows, lengths[0], heads, 1, width, true);
                    else if (success)
                        success = grouped_kernel->compute_tensor(
                            query.get(), keys.get(), values.get(), grouped.get(),
                            1, 1, lengths[0], heads, 1, width, true);
                }
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(success);
                EXPECT_LE(capture->nodeCount(), 4u) << "Grouped attention must not replay rows";
                ASSERT_TRUE(capture->instantiate());
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(worker.synchronizeStreamChecked(stream));

                // Every scalar launch sees the identical native bank and the
                // exact prefix its grouped row was permitted to consume.
                for (int row = 0; row < total_rows; ++row)
                {
                    const int request = row / rows;
                    const int visible = lengths[request] - rows + row % rows + 1;
                    const size_t element_bytes = type == TensorType::FP32 ? 4 : 2;
                    GpuTensorView key(static_cast<uint8_t *>(keys->gpu_data_ptr()) +
                        request * capacity * width * element_bytes, capacity, width, type, device);
                    GpuTensorView value(static_cast<uint8_t *>(values->gpu_data_ptr()) +
                        request * capacity * width * element_bytes, capacity, width, type, device);
                    GpuTensorView q(static_cast<float *>(query->gpu_data_ptr()) +
                        row * heads * width, 1, heads * width, TensorType::FP32, device);
                    GpuTensorView o(static_cast<float *>(serial->gpu_data_ptr()) +
                        row * heads * width, 1, heads * width, TensorType::FP32, device);
                    ASSERT_TRUE(serial_kernel->prepareDynamicAttnParams(
                        visible, visible - 1, 1, stream, capacity));
                    ASSERT_TRUE(serial_kernel->compute_tensor(
                        &q, &key, &value, &o, 1, 1, visible, heads, 1, width, true));
                }
                std::vector<float> actual(total_rows * heads * width), expected(actual.size());
                ASSERT_TRUE(backend->deviceToHost(actual.data(), grouped->gpu_data_ptr(),
                    actual.size() * sizeof(float), device.ordinal, stream));
                ASSERT_TRUE(backend->deviceToHost(expected.data(), serial->gpu_data_ptr(),
                    expected.size() * sizeof(float), device.ordinal, stream));
                ASSERT_EQ(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)), 0);
                for (int row = 0; row < total_rows; ++row)
                    for (int head = 0; head < heads; ++head)
                    {
                        const int request = row / rows;
                        const int visible = lengths[request] - rows + row % rows + 1;
                        std::vector<double> scores(visible);
                        double maximum = -INFINITY;
                        for (int k = 0; k < visible; ++k)
                        {
                            double score = 0;
                            for (int d = 0; d < width; ++d)
                            {
                                float q = query->typed_data()[(row * heads + head) * width + d];
                                // ROCm's native FP16 dot2 path expands Q into
                                // half LDS before arithmetic. Account for that
                                // declared operand rounding, not a looser output
                                // tolerance. BF16/FP32 and CUDA use FP32 Q here.
                                if (device.is_rocm() && type == TensorType::FP16)
                                    q = fp16_to_fp32(fp32_to_fp16(q));
                                score += double(q) * key_values[(request * capacity + k) * width + d];
                            }
                            scores[k] = score / std::sqrt(double(width));
                            maximum = std::max(maximum, scores[k]);
                        }
                        double sum = 0;
                        for (double &score : scores) { score = std::exp(score - maximum); sum += score; }
                        for (int d = 0; d < width; ++d)
                        {
                            double result = 0;
                            for (int k = 0; k < visible; ++k)
                                result += scores[k] * value_values[(request * capacity + k) * width + d];
                            ASSERT_NEAR(actual[(row * heads + head) * width + d], result / sum, 2e-6);
                        }
                    }
                // The same executable must reproduce identical bytes, without
                // relying on partial-buffer initialization from the serial oracle.
                ASSERT_TRUE(capture->launch());
                std::vector<float> replay(actual.size());
                ASSERT_TRUE(backend->deviceToHost(replay.data(), grouped->gpu_data_ptr(),
                    replay.size() * sizeof(float), device.ordinal, stream));
                ASSERT_EQ(std::memcmp(actual.data(), replay.data(), actual.size() * sizeof(float)), 0);
            }
}

/**
 * @brief Native prefill must equal the same cache values expanded to FP32.
 *
 * The FP32 oracle preserves the native rounding of each stored element. Both
 * calls execute captured production attention; equality proves native tile
 * loads, dtype dispatch and context-partition producers without accepting a
 * persistent converted cache as the implementation under test.
 */
TEST_P(GPUNativeKVAttentionMath, CapturedPrefillKeepsNativeStorageArithmetic)
{
    const auto available = snapshotTestDevice();
    if (!available)
        GTEST_SKIP() << "Linked GPU backend unavailable";
    const DeviceId device = *available;
    auto &worker = GPUDeviceContextPool::instance().getContext(device);
    auto *backend = getBackendFor(device);
    void *const stream = worker.defaultStream();
    const auto [type, width] = GetParam();
    constexpr int capacity = 320;
    constexpr int heads = 2;
    auto [keys, key_values] = nativeAttentionFixture(type, capacity, width, 7);
    auto [values, value_values] = nativeAttentionFixture(type, capacity, width, 23);
    auto expanded_k = TestTensorFactory::createFP32({capacity, size_t(width)});
    auto expanded_v = TestTensorFactory::createFP32({capacity, size_t(width)});
    std::copy(key_values.begin(), key_values.end(), expanded_k->mutable_data());
    std::copy(value_values.begin(), value_values.end(), expanded_v->mutable_data());
    auto query = TestTensorFactory::createFP32Random({32, size_t(heads * width)}, -0.4f, 0.4f, 51);
    auto native_output = TestTensorFactory::createFP32Zeros({32, size_t(heads * width)});
    auto expanded_output = TestTensorFactory::createFP32Zeros({32, size_t(heads * width)});
    for (ITensor *tensor : {keys.get(), values.get(), static_cast<ITensor *>(expanded_k.get()),
                           static_cast<ITensor *>(expanded_v.get()), static_cast<ITensor *>(query.get()),
                           static_cast<ITensor *>(native_output.get()), static_cast<ITensor *>(expanded_output.get())})
    {
        auto *owned = dynamic_cast<TensorBase *>(tensor);
        ASSERT_NE(owned, nullptr);
        ASSERT_TRUE(owned->ensureOnDevice(device, stream));
    }
    auto native_kernel = llaminar::v2::kernels::KernelFactory::createAttention(query.get(), device.type, device.ordinal);
    auto expanded_kernel = llaminar::v2::kernels::KernelFactory::createAttention(query.get(), device.type, device.ordinal);
    ASSERT_NE(native_kernel, nullptr);
    ASSERT_NE(expanded_kernel, nullptr);
    native_kernel->setGPUStream(stream);
    expanded_kernel->setGPUStream(stream);
    NativeAttentionWorkspace native_workspace(*native_kernel, device, 32, heads, width, capacity);
    NativeAttentionWorkspace expanded_workspace(*expanded_kernel, device, 32, heads, width, capacity);
    ASSERT_TRUE(worker.synchronizeStreamChecked(stream));
    for (int rows : {3, 32})
        for (auto axis : {attention::AttentionPrefillParallelAxis::QuerySequence,
                          attention::AttentionPrefillParallelAxis::KeyValueContext})
        {
            SCOPED_TRACE(::testing::Message() << "rows=" << rows << " axis=" << int(axis));
            attention::AttentionExecutionPolicy policy;
            policy.prefill_parallel_axis = axis;
            std::unique_ptr<IGPUGraphCapture> capture;
            worker.submitAndWait([&] { capture = worker.createGraphCapture(); });
            ASSERT_TRUE(capture->beginCapture());
            const bool native_prepared = native_kernel->prepareDynamicAttnParams(
                capacity, capacity - rows, 1, stream, capacity);
            const bool native_ok = native_prepared && native_kernel->compute_tensor(
                query.get(), keys.get(), values.get(), native_output.get(), 1, rows,
                capacity, heads, 1, width, true, -1, nullptr, nullptr, nullptr, device.ordinal,
                0, heads, 1, 0, policy);
            const bool expanded_prepared = expanded_kernel->prepareDynamicAttnParams(
                capacity, capacity - rows, 1, stream, capacity);
            const bool expanded_ok = expanded_prepared && expanded_kernel->compute_tensor(
                query.get(), expanded_k.get(), expanded_v.get(), expanded_output.get(), 1, rows,
                capacity, heads, 1, width, true, -1, nullptr, nullptr, nullptr, device.ordinal,
                0, heads, 1, 0, policy);
            ASSERT_TRUE(capture->endCapture());
            ASSERT_TRUE(native_ok);
            ASSERT_TRUE(expanded_ok);
            ASSERT_TRUE(capture->instantiate());
            ASSERT_TRUE(capture->launch());
            std::vector<float> actual(rows * heads * width), expected(actual.size());
            ASSERT_TRUE(backend->deviceToHost(actual.data(), native_output->gpu_data_ptr(),
                actual.size() * sizeof(float), device.ordinal, stream));
            ASSERT_TRUE(backend->deviceToHost(expected.data(), expanded_output->gpu_data_ptr(),
                expected.size() * sizeof(float), device.ordinal, stream));
            ASSERT_EQ(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)), 0);
        }
}

INSTANTIATE_TEST_SUITE_P(AllNativeFormats, GPUNativeKVAttentionMath,
    ::testing::Combine(
        ::testing::Values(TensorType::FP16, TensorType::BF16, TensorType::FP32),
        ::testing::Values(64, 128, 256)),
    ([](const ::testing::TestParamInfo<NativeAttentionCase> &info) {
        const auto [type, width] = info.param;
        return std::string(type == TensorType::FP16 ? "FP16" : type == TensorType::BF16 ? "BF16" : "FP32") +
            "_Head" + std::to_string(width);
    }));

using SnapshotCase = std::tuple<ActivationPrecision, int,
    attention::AttentionKeyCacheEncoding>;

/** @brief One production cache format, request geometry and read encoding. */
class GPUEffectiveKVSnapshotTest : public ::testing::TestWithParam<SnapshotCase>
{
};

/**
 * @brief Bind complete cold storage, capture real stages, then reset and replay.
 *
 * The descriptor check occurs before any cache read. Replaying the same graph
 * after an ordered request reset checks both the topology contract and the
 * diagnostic payload. Distinct request values expose a missing request bank.
 */
TEST_P(GPUEffectiveKVSnapshotTest, ColdCaptureAndRequestResetKeepCompleteManifest)
{
    const auto available = snapshotTestDevice();
    if (!available)
        GTEST_SKIP() << "Linked GPU backend unavailable";
    const DeviceId device = *available;
    auto &worker = GPUDeviceContextPool::instance().getContext(device);
    auto ctx = IDeviceContext::create(device, 1);
    EffectiveKVSnapshotScope diagnostics;
    const auto [precision, requests, encoding] = GetParam();
    constexpr int rows = 3;
    constexpr int capacity = 16;
    constexpr int width = 64;
    TurboQuantContext rotations(width, 42);
    llaminar::v2::kernels::KVCacheConfig cache_config;
    cache_config.device = device;
    cache_config.precision = precision;
    cache_config.num_layers = 1;
    cache_config.batch_size = requests;
    cache_config.max_seq_len = capacity;
    cache_config.n_kv_heads = 1;
    cache_config.head_dim = width;
    cache_config.turboquant_ctx = &rotations;
    auto cache = llaminar::v2::kernels::KernelFactory::createKVCache(cache_config);
    ASSERT_NE(cache, nullptr);
    auto *consumer = dynamic_cast<IWorkspaceConsumer *>(cache.get());
    ASSERT_NE(consumer, nullptr);
    using ReadKind = IKVCache::DeviceReadStorageKind;
    const IKVCache::DeviceReadStorageRequest gather_request{
        0, 0, requests, ReadKind::NativeRequestMajor};
    EXPECT_FALSE(cache->describeDeviceReadStorage(gather_request))
        << "Unbound gather workspace must not advertise readable storage";
    KVCacheTestWorkspaceBinding cache_workspace(*consumer, device);
    const auto prepared = cache->describeDeviceReadStorage(gather_request);
    ASSERT_TRUE(prepared);
    EXPECT_EQ(prepared->rows, requests * capacity);
    EXPECT_EQ(prepared->columns, width);
    EXPECT_EQ(prepared->device, device);
    for (const auto invalid : {
             IKVCache::DeviceReadStorageRequest{-1, 0, 1, ReadKind::NativeRequestMajor},
             IKVCache::DeviceReadStorageRequest{1, 0, 1, ReadKind::NativeRequestMajor},
             IKVCache::DeviceReadStorageRequest{0, -1, 1, ReadKind::NativeRequestMajor},
             IKVCache::DeviceReadStorageRequest{0, 0, 0, ReadKind::NativeRequestMajor},
             IKVCache::DeviceReadStorageRequest{0, 0, requests + 1, ReadKind::NativeRequestMajor}})
        EXPECT_FALSE(cache->describeDeviceReadStorage(invalid));

    const std::vector<size_t> shape{static_cast<size_t>(requests * rows), width};
    auto query = TestTensorFactory::createFP32(shape);
    auto key = TestTensorFactory::createFP32(shape);
    auto value = TestTensorFactory::createFP32(shape);
    auto output = TestTensorFactory::createFP32Zeros(shape);
    for (size_t i = 0; i < query->numel(); ++i)
    {
        query->mutable_data()[i] = float(int(i % 13) - 6) * 0.0625f;
        key->mutable_data()[i] = float(int(i % 19) - 9) * 0.03125f;
        value->mutable_data()[i] = 0.25f + float(i / (rows * width)) +
                                    float(i % 7) * 0.0625f;
    }
    void *const stream = worker.defaultStream();
    ASSERT_NE(stream, nullptr);
    for (auto *tensor : {query.get(), key.get(), value.get(), output.get()})
        ASSERT_TRUE(tensor->ensureOnDevice(device, stream));
    ASSERT_TRUE(worker.synchronizeStreamChecked(stream));

    BufferArena arena;
    ASSERT_TRUE(arena.bindExternalBuffer(BufferId::Q_PROJ, query.get()));
    ASSERT_TRUE(arena.bindExternalBuffer(BufferId::K_PROJ, key.get()));
    ASSERT_TRUE(arena.bindExternalBuffer(BufferId::V_PROJ, value.get()));
    ASSERT_TRUE(arena.bindExternalBuffer(BufferId::ATTN_OUTPUT, output.get()));
    KVCacheAppendStage::Params append_params;
    append_params.device_id = device;
    append_params.K = key.get();
    append_params.V = value.get();
    append_params.kv_cache = cache.get();
    append_params.batch_size = requests;
    append_params.num_tokens = requests * rows;
    append_params.seq_len = rows;
    append_params.head_dim = width;
    append_params.turboquant_ctx = &rotations;
    append_params.k_buffer_id = BufferId::K_PROJ;
    append_params.v_buffer_id = BufferId::V_PROJ;
    AttentionComputeStage::Params attention_params;
    attention_params.device_id = device;
    attention_params.Q = query.get();
    attention_params.K = key.get();
    attention_params.V = value.get();
    attention_params.output = output.get();
    attention_params.batch_size = requests;
    attention_params.seq_len = rows;
    attention_params.kv_len = rows;
    attention_params.n_heads = attention_params.n_kv_heads = 1;
    attention_params.head_dim = width;
    attention_params.kv_cache = cache.get();
    attention_params.layer_idx = 0;
    attention_params.turboquant_ctx = &rotations;
    attention_params.execution_policy.key_cache.encoding = encoding;
    attention_params.q_buffer_id = BufferId::Q_PROJ;
    attention_params.output_buffer_id = BufferId::ATTN_OUTPUT;
    ComputeGraph graph;
    auto attention_stage = std::make_unique<AttentionComputeStage>(attention_params);
    auto *attention = attention_stage.get();
    graph.addNode("layer0_append", std::make_unique<KVCacheAppendStage>(append_params), device);
    graph.addNode("layer0_attention", std::move(attention_stage), device);
    graph.addDependency("layer0_attention", "layer0_append");
    KVCacheTestWorkspaceBinding attention_workspace(*attention, device, rows);

    std::map<std::string, std::vector<uint8_t>> published;
    GraphExecutorConfig config;
    config.snapshot_callback = [&](const std::string &name, const StageDumpInfo &dump) {
        if (name != "layer0_attention")
            return;
        for (const auto &tensor : dump.outputs)
        {
            ASSERT_NE(tensor.data, nullptr);
            const auto *bytes = static_cast<const uint8_t *>(tensor.data);
            published[tensor.name] = std::vector<uint8_t>(bytes,
                bytes + computeByteSizeForDtype(tensor.dtype, tensor.rows, tensor.cols));
        }
    };
    // Admit snapshots through the same planner and live ledger as a runner.
    // Reference-style projection/output bytes do not include full-context K/V
    // banks. The typed diagnostic inventory must supply that separate bound.
    ModelMemoryProfile profile;
    profile.architecture = "qwen2";
    profile.n_layers = profile.n_heads = profile.n_kv_heads = 1;
    profile.d_model = profile.head_dim = width;
    profile.d_ff = width * 4;
    profile.vocab_size = 128;
    profile.max_seq_len = capacity;
    DevicePlanConfig memory_config;
    memory_config.world_rank = 0;
    memory_config.device = device;
    memory_config.device_total_bytes = memory_config.device_free_bytes = 1u << 30;
    memory_config.device_compute_units = 1;
    memory_config.batch_size = requests;
    memory_config.max_seq_len = capacity;
    memory_config.activation_seq_len = rows;
    memory_config.kv_precision = "fp32";
    memory_config.graph_snapshot_memory = {
        .per_accelerator_bytes = 3u * requests * rows * width * sizeof(float),
        .effective_kv = GraphSnapshotMemoryCapacity::EffectiveKV{
            .layer = 0, .retained_arena_count = 1}};
    const auto memory_plan = MemoryPlanner::plan(profile, {memory_config});
    const auto snapshot_bytes = memory_plan.devices.front().graph_snapshot_bytes();
    PhysicalMemoryAuthority authority(std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(
        memory_plan.physicalPlan()), 0);
    auto snapshot_reservation = std::make_shared<PhysicalMemoryOwnerReservation>(
        authority.reserveNewAllocations(device, PhysicalMemoryOwner::GraphSnapshotArena, snapshot_bytes));
    config.require_snapshot_memory_authority = true;
    DeviceGraphExecutor executor(config);
    executor.setGraphSnapshotMemoryReservation(snapshot_reservation);
    executor.setArena(&arena);
    // Declare capture last: it must retire before any borrowed cache/workspace.
    std::unique_ptr<IGPUGraphCapture> capture;
    worker.submitAndWait([&] { capture = worker.createGraphCapture(); });
    ASSERT_NE(capture, nullptr);
    ASSERT_EQ(capture->executionStream(), stream);
    ASSERT_TRUE(executor.prepareSnapshotsForGraphCapture(
        graph, ctx.get(), stream, "effective_kv_cold"));
    const auto before = attention->getDumpInfo();
    ASSERT_EQ(before.outputs.size(), 3u + 2u * requests);
    const auto *key_identity = before.outputs[1].tensor;
    const auto *value_identity = before.outputs[2].tensor;
    const std::array<ITensor *, 1> outputs{output.get()};
    ASSERT_TRUE(executor.executeWithGraphCapture(graph, ctx.get(), capture.get(), outputs));
    ASSERT_TRUE(capture->hasExecutable());
    ASSERT_GT(capture->nodeCount(), 0u);
    ASSERT_TRUE(executor.publishSnapshotsAfterGraphExecution(graph, stream, "effective_kv_cold"));
    ASSERT_EQ(published.size(), 3u + 2u * requests);
    const auto first = published;
    for (int request = 0; request < requests; ++request)
    {
        const auto &count_bytes = published.at("device_kv_count_request_" + std::to_string(request));
        ASSERT_EQ(count_bytes.size(), sizeof(int32_t));
        int32_t count = 0;
        std::memcpy(&count, count_bytes.data(), sizeof(count));
        EXPECT_EQ(count, rows);
    }
    EXPECT_TRUE(std::any_of(first.at("output").begin(), first.at("output").end(),
                           [](uint8_t byte) { return byte != 0; }));

    // Another graph can borrow a different request geometry from this cache.
    // That replaces the cache's transient tensor wrapper, not the stage-owned
    // immutable descriptor or its underlying workspace allocation.
    ITensor *other_key = nullptr;
    ITensor *other_value = nullptr;
    ASSERT_TRUE(cache->get_kv_batched_device_view(
        0, 0, 1, &other_key, &other_value, stream));

    ASSERT_TRUE(cache->resetRequestState({IKVCache::StateResetBoundary::RequestBoundary,
                                        stream, "effective-KV captured reset regression"}));
    attention->resetSessionStatePreservingCapturedReplay();
    // Reset clears the stage's borrowed stream binding, not the graph stream.
    // A new request's metadata inspection receives that exact stream again.
    attention->setGPUStream(stream);
    const auto after = attention->getDumpInfo();
    ASSERT_EQ(after.outputs.size(), before.outputs.size());
    EXPECT_EQ(after.outputs[1].tensor, key_identity);
    EXPECT_EQ(after.outputs[2].tensor, value_identity);
    ASSERT_TRUE(capture->launch());
    ASSERT_TRUE(executor.publishSnapshotsAfterGraphExecution(graph, stream, "effective_kv_reset"));
    EXPECT_EQ(published, first) << "Reset/replay must retain byte-identical diagnostic payloads";
    ASSERT_TRUE(worker.synchronizeStreamChecked(stream));
    capture->reset();
    attention->resetSessionState();
    attention->setGPUStream(stream);
    const auto retired = attention->getDumpInfo();
    ASSERT_EQ(retired.outputs.size(), before.outputs.size());
    EXPECT_EQ(retired.outputs[1].tensor, key_identity);
    EXPECT_EQ(retired.outputs[2].tensor, value_identity);
}

INSTANTIATE_TEST_SUITE_P(AllCacheFormats, GPUEffectiveKVSnapshotTest,
    ::testing::Combine(
        ::testing::Values(ActivationPrecision::FP16, ActivationPrecision::BF16,
            ActivationPrecision::FP32, ActivationPrecision::Q8_1,
            ActivationPrecision::TQ4, ActivationPrecision::TQ8),
        ::testing::Values(1, 2),
        ::testing::Values(attention::AttentionKeyCacheEncoding::PostRotary,
            attention::AttentionKeyCacheEncoding::PreRotaryDeviceTransform)),
    ([](const ::testing::TestParamInfo<SnapshotCase> &info) {
        const auto [precision, requests, encoding] = info.param;
        return std::string(activationPrecisionToString(precision)) + "_Requests" +
            std::to_string(requests) +
            (encoding == attention::AttentionKeyCacheEncoding::PostRotary
                ? "_Native" : "_Converted");
    }));

/**
 * @brief Hybrid descriptions map global layers to FA slots exactly once.
 *
 * A PP offset and interleaved recurrent layers make accidental use of either
 * the global index or the compact FA index observable. The query must reject
 * GDN layers, while every valid descriptor names the real backend read buffer.
 */
TEST(GPUEffectiveKVSnapshotStorage, HybridOffsetLayersDescribeActualReadDestination)
{
    const auto available = snapshotTestDevice();
    if (!available)
        GTEST_SKIP() << "Linked GPU backend unavailable";
    const DeviceId device = *available;
    auto &worker = GPUDeviceContextPool::instance().getContext(device);
    void *const stream = worker.defaultStream();
    ASSERT_NE(stream, nullptr);
    HybridKVCacheConfig hybrid;
    hybrid.layer_types = {"gdn", "full_attention", "gdn", "full_attention"};
    hybrid.first_layer_index = 8;
    hybrid.gdn_conv_kernel_size = 3;
    hybrid.gdn_state_size = hybrid.gdn_inner_size = 128;
    hybrid.gdn_group_count = hybrid.gdn_time_step_rank = hybrid.n_heads = 1;
    for (auto precision : {ActivationPrecision::FP16, ActivationPrecision::BF16,
                           ActivationPrecision::FP32, ActivationPrecision::Q8_1})
    {
        SCOPED_TRACE(activationPrecisionToString(precision));
        llaminar::v2::kernels::KVCacheConfig config;
        config.device = device;
        config.precision = precision;
        config.num_layers = 4;
        config.first_layer_index = hybrid.first_layer_index;
        config.batch_size = 2;
        config.max_seq_len = 16;
        config.n_kv_heads = 1;
        config.head_dim = 64;
        config.hybrid_config = &hybrid;
        PhysicalMemoryPlanBuilder plan;
        const PhysicalMemoryResource resource{
            .world_rank = 0,
            .device = device,
            .total_bytes = 1u << 30,
            .admission_available_bytes = 1u << 30,
        };
        // This direct cache fixture declares its two physical allocation owners,
        // just as rank admission does before constructing a hybrid cache.
        plan.add(resource, PhysicalMemoryOwner::RecurrentLiveState, 64u << 20);
        plan.add(resource, PhysicalMemoryOwner::KVCache, 64u << 20);
        config.physical_memory_authority = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
        auto cache = llaminar::v2::kernels::KernelFactory::createKVCache(config);
        ASSERT_NE(cache, nullptr);
        auto *consumer = dynamic_cast<IWorkspaceConsumer *>(cache.get());
        ASSERT_NE(consumer, nullptr);
        KVCacheTestWorkspaceBinding workspace(*consumer, device);
        using Kind = IKVCache::DeviceReadStorageKind;
        for (int invalid_layer : {-1, 8, 10, 12})
            EXPECT_FALSE(cache->describeDeviceReadStorage(
                {invalid_layer, 0, 2, Kind::NativeRequestMajor}));
        std::vector<void *> ring_keys;
        for (int layer : {9, 11})
        {
            SCOPED_TRACE(layer);
            const auto storage = cache->describeDeviceReadStorage(
                {layer, 0, 2, Kind::NativeRequestMajor});
            ASSERT_TRUE(storage);
            ITensor *key = nullptr;
            ITensor *value = nullptr;
            ASSERT_TRUE(cache->get_kv_batched_device_view(layer, 0, 2, &key, &value, stream));
            ASSERT_NE(key, nullptr);
            ASSERT_NE(value, nullptr);
            EXPECT_EQ(storage->key, key->gpu_data_ptr());
            EXPECT_EQ(storage->value, value->gpu_data_ptr());
            EXPECT_EQ(storage->type, key->native_type());
            EXPECT_EQ(storage->rows, 32u);
            EXPECT_EQ(storage->columns, 64u);
            if (precision != ActivationPrecision::Q8_1)
            {
                const auto ring = cache->describeDeviceReadStorage({layer, 1, 1, Kind::NativeRing});
                ASSERT_TRUE(ring);
                ASSERT_TRUE(cache->get_kv_device_ring_view(layer, 1, &key, &value,
                    nullptr, nullptr, nullptr, stream));
                EXPECT_EQ(ring->key, key->gpu_data_ptr());
                EXPECT_EQ(ring->value, value->gpu_data_ptr());
                ring_keys.push_back(ring->key);
            }
        }
        if (ring_keys.size() == 2)
            EXPECT_NE(ring_keys[0], ring_keys[1]) << "Distinct FA layers must not alias";
        ASSERT_TRUE(worker.synchronizeStreamChecked(stream));
    }
}
} // namespace
