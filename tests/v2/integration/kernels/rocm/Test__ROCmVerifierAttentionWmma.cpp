/**
 * @file Test__ROCmVerifierAttentionWmma.cpp
 * @brief Captured byte and lifecycle proofs for lane-exact grouped ROCm attention.
 *
 * Tests enter the public production parameter writer, verifier attention and
 * original split reducer on the owning GPU worker. Independent serial M=1
 * execution owns the numerical contract. Retained replay changes device row
 * metadata and inputs without topology changes, and observes poisoned partial
 * banks before another launch can repair them. These native gfx1100 tests are
 * explicitly registered in the focused feature and production-preflight suites.
 */
#include <gtest/gtest.h>
#ifdef HAVE_ROCM
#include "backends/rocm/ROCmHardwareMetadata.h"
#include "backends/rocm/HIPGraphCapture.h"
#include "backends/GPUDeviceContextPool.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "kernels/rocm/attention/ROCmFlashAttentionKernelT.h"
#include "kernels/rocm/attention/ROCmVerifierAttentionWmma.h"
#include "tensors/Tensors.h"
#include "tensors/FP16Utils.h"
#include "transfer/TransferEngine.h"
#include "utils/MPIContext.h"
#include "utils/DebugEnv.h"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <exception>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <memory>
#include <random>
#include <vector>

using namespace llaminar2;
namespace
{
/** @return The admitted native gfx1100 endpoint, or -1 when unavailable. */
int verifierWmmaDevice()
{
    initializeROCmHardwareMetadata();
    for (int ordinal = 0; ordinal < int(rocmHardwareMetadata().size()); ++ordinal)
    {
        const auto &hardware = rocmHardwareMetadata().at(ordinal);
        if (hardware.isa_target == "gfx1100" || hardware.isa_target.starts_with("gfx1100:")) return ordinal;
    }
    return -1;
}

/** @brief Preserve exact process control presence, including absent versus empty. */
class DecodeControls final
{
    static constexpr std::array<const char *, 2> names{
        "LLAMINAR_ROCM_FA_DECODE_NUM_SPLITS", "LLAMINAR_ROCM_FA_DECODE_TPB"};
    std::array<std::optional<std::string>, 2> original;
public:
    /** @brief Capture both controls before any mutation. */
    DecodeControls()
    {
        for (size_t i = 0; i < names.size(); ++i)
            if (const char *value = std::getenv(names[i])) original[i] = value;
    }
    /** @brief Restore exact original control presence before another fixture runs. */
    ~DecodeControls()
    {
        for (size_t i = 0; i < names.size(); ++i)
            if ((original[i] ? ::setenv(names[i], original[i]->c_str(), 1) : ::unsetenv(names[i])) != 0) std::terminate();
        mutableDebugEnv().rocm.reload();
    }
    /** @brief Publish a fresh diagnostic recording policy.
     * @param splits Null really unsets the variable; empty retains legacy presence.
     * @param threads Null selects the ordinary default, otherwise an explicit TPB. */
    void select(const char *splits, const char *threads)
    {
        for (size_t i = 0; i < names.size(); ++i)
        {
            const char *value = i ? threads : splits;
            if ((value ? ::setenv(names[i], value, 1) : ::unsetenv(names[i])) != 0)
                throw std::runtime_error("Unable to select attention decode controls");
        }
        mutableDebugEnv().rocm.reload();
    }
};

/**
 * @brief Retain the actual parameter writer, grouped producer and split reducer.
 *
 * Both grouped engines consume the same device publications. Every physical
 * row also has an independent M=1 witness over byte-reordered native K/V,
 * including the one-key sentinel rows emitted by production for inactive work.
 * Poisoned partial banks expose missing writes before any subsequent launch
 * can repair them. All staging and observation storage outlives the terminal
 * drain on assertion exits; engine scopes live on the recording GPU worker.
 * @param ordinal Admitted native endpoint. @param rows Retained physical queries.
 * @param capacity Physical KV rows. @param replicated Second TP query-head shard.
 * @param specials Include exceptional native operands. @param heads Local query heads.
 * @param kv_heads Visible cache heads; distinct input words exercise head addressing.
 * @param cancellation_threads Optional arithmetic witness: zero means default, otherwise explicit TPB.
 */
void runVerifierWmmaRingReplay(
    int ordinal, int rows, int capacity, bool replicated, bool specials, int heads = 8, int kv_heads = 1,
    std::optional<int> cancellation_threads = std::nullopt)
{
    using Engine = rocm::VerifierAttentionEngine;
    constexpr int width = 256;
    MPIContext mpi_ctx(0, 1, MPI_COMM_WORLD);
    std::mt19937 rng(42);
    const auto randomFP32Scaled = [&](size_t count, float scale) {
        std::uniform_real_distribution<float> distribution(-0.5f * scale, 0.5f * scale);
        std::vector<float> result(count);
        for (auto &value : result) value = distribution(rng);
        return result;
    };
    const int head_start = replicated ? 8 : 0, repetition = replicated ? 16 : 0;
    const auto device = DeviceId::rocm(ordinal);
    auto &worker = GPUDeviceContextPool::instance().getContext(device);
    worker.submitAndWait([&]
    {
        SCOPED_TRACE(::testing::Message() << "M=" << rows << " capacity=" << capacity
            << " replicated=" << replicated << " specials=" << specials);
        const auto stream = static_cast<hipStream_t>(worker.defaultStream());
        auto &transfer = TransferEngine::instance();
        const size_t q_columns = heads * width, kv_columns = kv_heads * width;
        auto query_values = randomFP32Scaled(size_t(rows) * q_columns, 0.25f);
        const auto key_values = randomFP32Scaled(size_t(capacity) * kv_columns, 0.25f);
        const auto value_values = randomFP32Scaled(key_values.size(), 0.25f);
        std::vector<uint16_t> physical_k(key_values.size()), physical_v(key_values.size());
        std::transform(key_values.begin(), key_values.end(), physical_k.begin(), fp32_to_fp16);
        std::transform(value_values.begin(), value_values.end(), physical_v.begin(), fp32_to_fp16);
        if (specials)
        {
            // Payloads, infinities, signed zero and subnormals are native input
            // bytes. They must survive the same arithmetic as serial decode.
            constexpr std::array<uint16_t, 8> values{0x7e01, 0xfe35, 0x7c00, 0xfc00,
                                                    0x0001, 0x8000, 0x7bff, 0xfbff};
            for (int row = 0; row < capacity; ++row)
            {
                physical_k[size_t(row) * kv_columns + (row * 17) % width] = values[row % values.size()];
                physical_v[size_t(row) * kv_columns + (row * 13) % width] = values[(row + 3) % values.size()];
            }
        }
        if (cancellation_threads)
        {
            ASSERT_EQ(capacity, 8);
            std::fill(physical_k.begin(), physical_k.end(), uint16_t{0});
            std::fill(physical_v.begin(), physical_v.end(), uint16_t{0});
            const std::array<float, 8> cancellation{65504.0f, 0x1p-10f, -65504.0f, 0.0f,
                                                  -65504.0f, 0.0f, 65504.0f, 0.0f};
            for (int row = 0; row < capacity; ++row) physical_v[size_t(row) * kv_columns] = fp32_to_fp16(cancellation[row]);
        }
        FP16Tensor ring_k({size_t(capacity), kv_columns}, physical_k);
        FP16Tensor ring_v({size_t(capacity), kv_columns}, physical_v);
        FP16Tensor linear_k({size_t(capacity), kv_columns}, physical_k);
        FP16Tensor linear_v({size_t(capacity), kv_columns}, physical_v);
        FP32Tensor query({size_t(rows), q_columns}), ordered_output({size_t(rows), q_columns});
        FP32Tensor native_output({size_t(rows), q_columns}), serial_query({size_t{1}, q_columns});
        FP32Tensor serial_output({size_t{1}, q_columns});
        INT32Tensor cached({size_t{1}}, std::vector<int32_t>{capacity});
        INT32Tensor head({size_t{1}}, std::vector<int32_t>{0});
        INT32Tensor active({size_t{1}}, std::vector<int32_t>{rows});
        std::copy(query_values.begin(), query_values.end(), query.mutable_data());
        std::array<rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32>, 3> kernels{
            rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32>(ordinal),
            rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32>(ordinal),
            rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32>(ordinal)};
        const auto requirements = kernels[0].getWorkspaceRequirements(rows, heads, width);
        std::array<std::unique_ptr<DeviceWorkspaceManager>, 3> workspaces;
        for (int path = 0; path < 3; ++path)
        {
            workspaces[path] = std::make_unique<DeviceWorkspaceManager>(device, requirements.total_bytes_with_alignment());
            ASSERT_TRUE(workspaces[path]->allocate(requirements));
            kernels[path].bindWorkspace(workspaces[path].get());
            kernels[path].setGPUStream(stream);
        }
        const std::array<const char *, 3> partial_names{
            rocm::AttentionWorkspaceBuffers::PARTIAL_OUTPUT,
            rocm::AttentionWorkspaceBuffers::PARTIAL_M, rocm::AttentionWorkspaceBuffers::PARTIAL_L};
        std::array<std::vector<uint32_t>, 2> observed;
        std::array<std::array<std::vector<uint32_t>, 3>, 2> partials;
        for (int path = 0; path < 2; ++path)
        {
            observed[path].resize(query_values.size());
            for (int bank = 0; bank < 3; ++bank)
                partials[path][bank].resize(workspaces[path]->getBufferSize(partial_names[bank]) / sizeof(uint32_t));
        }
        std::vector<uint32_t> expected(q_columns);
        std::vector<uint16_t> logical_k(physical_k.size()), logical_v(physical_v.size());
        int length = capacity, live_rows = rows, ring_head = 0;
        std::array<std::unique_ptr<HIPGraphCapture>, 2> graphs;
        auto drain = [&](void *) {
            if (hipStreamSynchronize(stream) != hipSuccess) std::terminate();
        };
        std::unique_ptr<void, decltype(drain)> completion(&worker, drain);
        for (TensorBase *tensor : std::array<TensorBase *, 12>{&ring_k, &ring_v, &linear_k, &linear_v,
                &query, &ordered_output, &native_output, &serial_query, &serial_output, &cached, &head, &active})
            ASSERT_TRUE(transfer.uploadFull(tensor, device, stream).success);
        const std::array<FP32Tensor *, 2> outputs{&ordered_output, &native_output};
        const auto enqueue = [&](int path)
        {
            return kernels[path].prepareDynamicAttnParamsFromDeviceSequenceState(
                static_cast<const int *>(cached.gpu_data_ptr()), rows, rows, stream,
                capacity, static_cast<const int *>(active.gpu_data_ptr()), {},
                static_cast<const int *>(head.gpu_data_ptr()), capacity) &&
                kernels[path].compute_verifier_rows_decode_equivalent(&query, &ring_k, &ring_v,
                    outputs[path], rows, capacity, heads, kv_heads, width, true, -1,
                    &mpi_ctx, ordinal, head_start, repetition);
        };
        for (int path = 0; path < 2; ++path)
        {
            rocm::ScopedVerifierAttentionEngine engine(path ? Engine::WmmaLaneExact : Engine::OrderedDot2);
            ASSERT_TRUE(enqueue(path));
            ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
            graphs[path] = std::make_unique<HIPGraphCapture>(stream, ordinal);
            ScopedBackendGraphCapture capture(worker, *graphs[path], "lane-exact verifier attention replay");
            ASSERT_TRUE(capture.begin());
            const bool submitted = enqueue(path);
            capture.finish();
            ASSERT_TRUE(submitted);
            std::vector<GPUGraphKernelNodeInfo> nodes;
            ASSERT_TRUE(graphs[path]->inspectKernelNodes(nodes));
            ASSERT_EQ(nodes.size(), 3u);
            const char *producer_name = path ? "flash_decode_verifier_wmma_fp16_hd256_kernel" : "flash_decoding_mi50_kernel";
            const auto producer = std::find_if(nodes.begin(), nodes.end(), [&](const auto &node) {
                return node.name.find(producer_name) != std::string::npos;
            });
            ASSERT_NE(producer, nodes.end());
            EXPECT_EQ(producer->grid_x, unsigned(heads));
            EXPECT_GT(producer->grid_y, 0u);
            EXPECT_EQ(producer->block_x, path ? 1024u : 256u);
            EXPECT_EQ(producer->grid_z, path ? (rows + 3u) / 4u : unsigned(rows));
            EXPECT_EQ(producer->local_memory_bytes_per_thread, 0u);
            EXPECT_GT(producer->max_active_blocks_per_sm, 0u);
            const auto reducer = std::find_if(nodes.begin(), nodes.end(), [](const auto &node) {
                return node.name.find("flash_decoding_reduce_mi50_kernel") != std::string::npos;
            });
            ASSERT_NE(reducer, nodes.end());
            EXPECT_EQ(reducer->grid_x, unsigned(heads));
            EXPECT_EQ(reducer->grid_y, unsigned(rows));
            EXPECT_EQ(reducer->grid_z, 2u);
            // Authenticate the actual native node's retained bank/parameter
            // pointers, rather than accepting a matching name elsewhere.
            size_t count = 0;
            ASSERT_EQ(hipGraphGetNodes(graphs[path]->graph(), nullptr, &count), hipSuccess);
            std::vector<hipGraphNode_t> native_nodes(count);
            ASSERT_EQ(hipGraphGetNodes(graphs[path]->graph(), native_nodes.data(), &count), hipSuccess);
            bool bound = false, reducer_bound = false;
            for (const auto node : native_nodes)
            {
                hipKernelNodeParams params{};
                ASSERT_EQ(hipGraphKernelNodeGetParams(node, &params), hipSuccess);
                if (reinterpret_cast<uintptr_t>(params.func) == reducer->function_identity)
                {
                    reducer_bound = true;
                    for (int bank = 0; bank < 3; ++bank)
                        EXPECT_EQ(*static_cast<void **>(params.kernelParams[bank]), workspaces[path]->getBuffer(partial_names[bank]));
                    EXPECT_EQ(*static_cast<void **>(params.kernelParams[3]), outputs[path]->gpu_data_ptr());
                    continue;
                }
                if (reinterpret_cast<uintptr_t>(params.func) != producer->function_identity) continue;
                bound = true;
                EXPECT_EQ(*static_cast<void **>(params.kernelParams[0]), query.gpu_data_ptr());
                EXPECT_EQ(*static_cast<void **>(params.kernelParams[1]), ring_k.gpu_data_ptr());
                EXPECT_EQ(*static_cast<void **>(params.kernelParams[2]), ring_v.gpu_data_ptr());
                EXPECT_EQ(*static_cast<int *>(params.kernelParams[10]), int(producer->grid_y));
                if (cancellation_threads)
                {
                    EXPECT_FALSE(*static_cast<bool *>(params.kernelParams[path ? 13 : 16]));
                    EXPECT_EQ(producer->grid_y, 1u);
                    EXPECT_EQ(*static_cast<bool *>(params.kernelParams[path ? 15 : 18]), *cancellation_threads != 0);
                    if (*cancellation_threads)
                        EXPECT_EQ(*static_cast<int *>(params.kernelParams[path ? 16 : 19]), *cancellation_threads);
                }
                for (int bank = 0; bank < 3; ++bank)
                    EXPECT_EQ(*static_cast<void **>(params.kernelParams[3 + bank]), workspaces[path]->getBuffer(partial_names[bank]));
                EXPECT_EQ(*static_cast<void **>(params.kernelParams[path ? 6 : 12]),
                    workspaces[path]->getBuffer(rocm::AttentionWorkspaceBuffers::DEVICE_PARAMS));
            }
            ASSERT_TRUE(bound);
            ASSERT_TRUE(reducer_bound);
            ASSERT_TRUE(graphs[path]->instantiate());
        }
        const std::array<int, 6> lengths{capacity, std::min(capacity, 37), 1,
                                       std::min(capacity, 513), capacity, capacity};
        const std::array<int, 6> live{rows, std::min(rows, 3), 0, 1, rows - 1, rows};
        for (size_t replay = 0; replay < lengths.size(); ++replay)
        {
            SCOPED_TRACE(::testing::Message() << "replay=" << replay);
            length = lengths[replay];
            live_rows = live[replay];
            const int origin = cancellation_threads ? 0 : (capacity - 1 + int(replay) * 7) % capacity;
            ring_head = (origin + length) % capacity;
            for (size_t i = 0; i < query_values.size(); ++i)
                query_values[i] = std::sin(float((i + replay * 71) % 32749) * 0.031f) * 0.23f;
            if (cancellation_threads) std::fill(query_values.begin(), query_values.end(), 0.0f);
            if (specials)
                for (int row = 0; row < rows; ++row)
                    query_values[size_t(row) * q_columns + (row * 7 + replay) % q_columns] =
                        std::bit_cast<float>(uint32_t(0x7fc00100 + row));
            ASSERT_EQ(hipMemcpyAsync(query.gpu_data_ptr(), query_values.data(), query_values.size() * sizeof(float), hipMemcpyHostToDevice, stream), hipSuccess);
            ASSERT_EQ(hipMemcpyAsync(cached.gpu_data_ptr(), &length, sizeof(length), hipMemcpyHostToDevice, stream), hipSuccess);
            ASSERT_EQ(hipMemcpyAsync(head.gpu_data_ptr(), &ring_head, sizeof(ring_head), hipMemcpyHostToDevice, stream), hipSuccess);
            ASSERT_EQ(hipMemcpyAsync(active.gpu_data_ptr(), &live_rows, sizeof(live_rows), hipMemcpyHostToDevice, stream), hipSuccess);
            for (int path = 0; path < 2; ++path)
            {
                ASSERT_EQ(hipMemsetAsync(outputs[path]->gpu_data_ptr(), 0xa5, query_values.size() * sizeof(float), stream), hipSuccess);
                for (int bank = 0; bank < 3; ++bank)
                    ASSERT_EQ(hipMemsetAsync(workspaces[path]->getBuffer(partial_names[bank]), 0xa5,
                        workspaces[path]->getBufferSize(partial_names[bank]), stream), hipSuccess);
                ASSERT_TRUE(graphs[path]->launch());
                ASSERT_EQ(hipMemcpyAsync(observed[path].data(), outputs[path]->gpu_data_ptr(), observed[path].size() * sizeof(uint32_t), hipMemcpyDeviceToHost, stream), hipSuccess);
                for (int bank = 0; bank < 3; ++bank)
                    ASSERT_EQ(hipMemcpyAsync(partials[path][bank].data(), workspaces[path]->getBuffer(partial_names[bank]),
                        partials[path][bank].size() * sizeof(uint32_t), hipMemcpyDeviceToHost, stream), hipSuccess);
                ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
            }
            const auto compare_words = [&](const auto &expected, const auto &actual, const char *name)
            {
                size_t first = expected.size(), differences = 0;
                for (size_t word = 0; word < expected.size(); ++word)
                    if (expected[word] != actual[word]) { first = std::min(first, word); ++differences; }
                if (!differences) return true;
                ADD_FAILURE() << name << " differing words=" << differences << " first=" << first
                    << " ordered=0x" << std::hex << expected[first] << " native=0x" << actual[first];
                return false;
            };
            bool exact = true;
            for (int bank = 0; bank < 3; ++bank)
                exact = compare_words(partials[0][bank], partials[1][bank], partial_names[bank]) && exact;
            exact = compare_words(observed[0], observed[1], "final output") && exact;
            ASSERT_TRUE(exact);
            if (cancellation_threads && length == 8 && live_rows == rows)
            {
                // Q=K=0 makes every probability exactly one. Cancellation
                // removes the small term in one/three waves; two/four retain it.
                const int threads = *cancellation_threads ? *cancellation_threads : 256;
                const float expected = threads == 64 || threads == 192 ? 0.0f : 0x1p-13f;
                ASSERT_EQ(observed[1][size_t(rows - 1) * q_columns], std::bit_cast<uint32_t>(expected));
            }
            for (int row = 0; row < capacity; ++row)
            {
                const size_t source = size_t((origin + row) % capacity) * kv_columns;
                std::copy_n(physical_k.data() + source, kv_columns, logical_k.data() + size_t(row) * kv_columns);
                std::copy_n(physical_v.data() + source, kv_columns, logical_v.data() + size_t(row) * kv_columns);
            }
            ASSERT_EQ(hipMemcpyAsync(linear_k.gpu_data_ptr(), logical_k.data(), logical_k.size() * sizeof(uint16_t), hipMemcpyHostToDevice, stream), hipSuccess);
            ASSERT_EQ(hipMemcpyAsync(linear_v.gpu_data_ptr(), logical_v.data(), logical_v.size() * sizeof(uint16_t), hipMemcpyHostToDevice, stream), hipSuccess);
            for (int row = 0; row < rows; ++row)
            {
                // Production clamps zero live rows to one sentinel query and
                // publishes one visible key for each remaining physical row.
                const int effective_live = std::max(1, live_rows);
                const int visible = row < effective_live ? std::max(1, length - effective_live + 1 + row) : 1;
                ASSERT_EQ(hipMemcpyAsync(serial_query.gpu_data_ptr(), query_values.data() + size_t(row) * q_columns,
                    q_columns * sizeof(float), hipMemcpyHostToDevice, stream), hipSuccess);
                ASSERT_TRUE(kernels[2].prepareDynamicAttnParams(visible, visible - 1, 1, stream, capacity));
                ASSERT_TRUE(kernels[2].compute_tensor(&serial_query, &linear_k, &linear_v, &serial_output,
                    1, 1, visible, heads, kv_heads, width, true, -1, nullptr, nullptr, &mpi_ctx, ordinal, head_start, heads, kv_heads, repetition));
                ASSERT_EQ(hipMemcpyAsync(expected.data(), serial_output.gpu_data_ptr(), q_columns * sizeof(uint32_t), hipMemcpyDeviceToHost, stream), hipSuccess);
                ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
                ASSERT_TRUE(std::equal(expected.begin(), expected.end(), observed[1].begin() + size_t(row) * q_columns))
                    << "serial row=" << row << " visible=" << visible << " origin=" << origin;
            }
        }
        for (auto &kernel : kernels) kernel.unbindWorkspace();
    });
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
}

/** @brief Cover every supported MTP width, tile tails and unequal split/ring metadata. */
TEST(ROCmVerifierAttentionWmma, DirectRingDirtyReplay)
{
    const int ordinal = verifierWmmaDevice();
    if (ordinal < 0) GTEST_SKIP() << "Native gfx1100 is not visible";
    for (int rows = 2; rows <= 16; ++rows)
        for (const int capacity : {64, 8193})
            ASSERT_NO_FATAL_FAILURE(runVerifierWmmaRingReplay(ordinal, rows, capacity, rows % 2 == 0, false));
    ASSERT_NO_FATAL_FAILURE(runVerifierWmmaRingReplay(ordinal, 5, 64, false, false, 16, 2));
}

/** @brief Preserve lane-specific exceptional arithmetic through the actual reducer. */
TEST(ROCmVerifierAttentionWmma, SpecialValuesDirtyReplay)
{
    const int ordinal = verifierWmmaDevice();
    if (ordinal < 0) GTEST_SKIP() << "Native gfx1100 is not visible";
    for (const int rows : {2, 4, 9, 16})
        ASSERT_NO_FATAL_FAILURE(runVerifierWmmaRingReplay(ordinal, rows, 64, rows == 9, true));
}

/** @brief A TPB-only override must select its real arithmetic without a split variable. */
TEST(ROCmVerifierAttentionWmma, ThreadOverrideOwnsArithmeticWithoutSplitOverride)
{
    const int ordinal = verifierWmmaDevice();
    if (ordinal < 0) GTEST_SKIP() << "Native gfx1100 is not visible";
    DecodeControls controls;
    controls.select(nullptr, nullptr);
    ASSERT_NO_FATAL_FAILURE(runVerifierWmmaRingReplay(ordinal, 2, 8, false, false, 8, 1, 0));
    controls.select(nullptr, "64");
    ASSERT_NO_FATAL_FAILURE(runVerifierWmmaRingReplay(ordinal, 2, 8, false, false, 8, 1, 64));
    controls.select("", "64");
    ASSERT_NO_FATAL_FAILURE(runVerifierWmmaRingReplay(ordinal, 2, 8, false, false, 8, 1, 64));
    for (const int threads : {128, 192, 256})
    {
        const auto value = std::to_string(threads);
        controls.select(nullptr, value.c_str());
        ASSERT_NO_FATAL_FAILURE(runVerifierWmmaRingReplay(ordinal, 2, 8, false, false, 8, 1, threads));
    }
}

/**
 * @brief Preserve serial NaN payload selection in every original wave-count mode.
 *
 * A one-wave output exposes online probability/value FMA operand ordering;
 * later multiwave scales can mask that difference. The actual native candidate
 * must retain the serial operand roles in addition to mathematical operation
 * order. Dirty ring replay and independent serial observations exercise both.
 */
TEST(ROCmVerifierAttentionWmma, AllWaveSpecialValuesDirtyReplay)
{
    const int ordinal = verifierWmmaDevice();
    if (ordinal < 0) GTEST_SKIP() << "Native gfx1100 is not visible";
    DecodeControls controls;
    for (const int threads : {64, 128, 192, 256})
    {
        SCOPED_TRACE(::testing::Message() << "original threads=" << threads);
        const auto value = std::to_string(threads);
        controls.select(nullptr, value.c_str());
        for (const int rows : {2, 4, 9, 16})
            ASSERT_NO_FATAL_FAILURE(runVerifierWmmaRingReplay(ordinal, rows, 64, rows == 9, true));
    }
}

} // namespace
#endif // HAVE_ROCM
