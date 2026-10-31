/**
 * @file Test__CUDAAttentionCentroidFirstCapture.cpp
 * @brief First-use captured TQ attention authenticates every immutable centroid.
 *
 * Each visible device executes the actual fused TQ8-key/TQ4-value ABI without
 * eager attention warmup. Independent heads expose all256 key indices, while
 * value coordinates cover all16 packed indices. PhysicalMemoryAuthority admits
 * the exact fixture backing before allocation. Retained dirty replays are byte
 * exact and agree with the canonical CPU centroid arithmetic. A production
 * exclusive model-retirement transaction then proves first capture in the next
 * native runtime generation, without a raw runtime reset.
 */
#include "backends/BackendManager.h"
#include "backends/cuda/CUDAHardwareMetadata.h"
#include "kernels/attention/AttentionDeviceParams.h"
#include "kernels/cpu/attention/TQFusedAttentionPrimitives.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "transfer/TransferEngine.h"
#include "tensors/BlockStructures.h"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

/** @brief Exact production fused split-decode entrypoint; no test implementation substitutes it. */
extern "C" int cudaFlashAttn_decode_tqkv(
    const float *, float *, float *, float *, float *, const void *, const void *,
    const float *, const float *, int, int, int, int, int, int, int, int, int, int,
    const llaminar2::attention::AttentionDeviceParams *, void *, int, int, int);

namespace llaminar2::test
{
    namespace
    {
        /** @brief Reject setup/capture/replay failures at their exact native boundary. */
        void checked(cudaError_t status)
        {
            if (status != cudaSuccess)
                throw std::runtime_error(std::string("CUDA centroid first-capture: ") + cudaGetErrorString(status));
        }

        /** @brief One PMA-claimed, backend-tracked physical test root. */
        class Root final
        {
        public:
            /** @brief Claim before native allocation. @param memory Sole admitted fixture authority.
             * @param device Exact CUDA owner. @param bytes Exact backing extent. */
            Root(const std::shared_ptr<PhysicalMemoryAuthority> &memory, DeviceId device, std::size_t bytes)
                : device_(device), backend_(getBackendFor(device)),
                  lease_(memory->claimNewAllocation(device, PhysicalMemoryOwner::ExecutionWorkspace, bytes)), bytes_(bytes)
            {
                if (!backend_) throw std::runtime_error("Centroid fixture has no CUDA backend");
                data_ = backend_->allocate(bytes, device.ordinal);
                if (!data_) throw std::runtime_error("Centroid fixture admitted allocation failed");
            }
            Root(const Root &) = delete;
            Root &operator=(const Root &) = delete;
            /** @brief Free backing before the retained PMA claim retires. */
            ~Root() { if (data_) backend_->free(data_, device_.ordinal); }
            /** @return Borrowed native backing. */
            void *data() const noexcept { return data_; }
            /** @return Exact immutable root extent. */
            std::size_t bytes() const noexcept { return bytes_; }
            /** @return Typed borrowed native view without a second allocation. */
            template <typename T> T *as() const noexcept { return static_cast<T *>(data_); }
        private:
            DeviceId device_;
            IBackend *backend_;
            PhysicalMemoryAllocationLease lease_;
            std::size_t bytes_;
            void *data_ = nullptr;
        };

        /** @brief Exact-owner stream and retained graph, destroyed before fixture data. */
        class Capture final
        {
        public:
            /** @brief Create one explicit non-default producer before any recording. */
            explicit Capture(int ordinal) : ordinal_(ordinal)
            {
                checked(cudaSetDevice(ordinal_));
                checked(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
            }
            Capture(const Capture &) = delete;
            Capture &operator=(const Capture &) = delete;
            /** @brief Retire capture/executable/stream on normal and exceptional unwind. */
            ~Capture()
            {
                if (cudaSetDevice(ordinal_) != cudaSuccess) std::terminate();
                if (capturing_) (void)cudaStreamEndCapture(stream_, &graph_);
                if (execution_ && cudaGraphExecDestroy(execution_) != cudaSuccess) std::terminate();
                if (graph_ && cudaGraphDestroy(graph_) != cudaSuccess) std::terminate();
                if (cudaStreamSynchronize(stream_) != cudaSuccess || cudaStreamDestroy(stream_) != cudaSuccess)
                    std::terminate();
            }
            /** @return Exact owned producer stream. */
            cudaStream_t stream() const noexcept { return stream_; }
            /** @brief Record the first production attention invocation and retain its executable. */
            template <typename Launch> void record(Launch launch)
            {
                checked(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
                capturing_ = true;
                const int status = launch();
                const auto ended = cudaStreamEndCapture(stream_, &graph_);
                capturing_ = false;
                checked(ended);
                if (status != 0) throw std::runtime_error("First TQ attention invocation failed inside capture");
                checked(cudaGraphInstantiate(&execution_, graph_, nullptr, nullptr, 0));
            }
            /** @brief Submit the same native executable, without reentering host attention dispatch. */
            void launch() { checked(cudaGraphLaunch(execution_, stream_)); }
        private:
            int ordinal_;
            cudaStream_t stream_ = nullptr;
            cudaGraph_t graph_ = nullptr;
            cudaGraphExec_t execution_ = nullptr;
            bool capturing_ = false;
        };

        /** @brief Build canonical raw blocks and CPU reference, then run first-use captured replay.
         * @tparam D Actual supported64/128/256 source geometry.
         * @param ordinal Exact visible CUDA device. @param retire Whether to retire this real BOM afterward. */
        template <int D> void exercise(int ordinal, bool retire)
        {
            constexpr int heads = 256, rows = 2, splits = 1, guards = 16;
            constexpr std::size_t elements = heads * D;
            const DeviceId device = DeviceId::cuda(ordinal);
            checked(cudaSetDevice(ordinal));
            auto *backend = getBackendFor(device);
            ASSERT_NE(backend, nullptr);
            const auto generation = backend->deviceRuntimeGeneration(ordinal);
            ASSERT_NE(generation, 0u);
            const auto *physical = &cudaHardwareMetadata().at(ordinal);
            const std::vector<std::size_t> sizes{
                elements * sizeof(float), (elements + guards) * sizeof(float), elements * sizeof(float),
                heads * sizeof(float), heads * sizeof(float), rows * heads * sizeof(TQ8Block<D>),
                rows * heads * sizeof(TQ4Block<D>), heads * D * D * sizeof(float)};
            std::size_t total = 0;
            for (auto bytes : sizes) total += bytes;
            std::size_t available = 0, native_total = 0;
            checked(cudaMemGetInfo(&available, &native_total));
            PhysicalMemoryPlanBuilder builder;
            builder.add(PhysicalMemoryResource{.world_rank = 0, .device = device,
                .total_bytes = physical->total_memory, .admission_available_bytes = available},
                PhysicalMemoryOwner::ExecutionWorkspace, total);
            const auto memory = std::make_shared<PhysicalMemoryAuthority>(
                std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(builder.build()), 0);
            std::vector<std::unique_ptr<Root>> roots;
            for (auto bytes : sizes) roots.push_back(std::make_unique<Root>(memory, device, bytes));
            auto &q = *roots[0], &output = *roots[1], &partial = *roots[2], &maximum = *roots[3],
                 &denominator = *roots[4], &keys = *roots[5], &values = *roots[6], &rotation = *roots[7];
            {
                Capture capture(ordinal);
                const auto stream = capture.stream();
                std::vector<float> query(elements, 0.f), identity(heads * D * D, 0.f);
                std::vector<TQ8Block<D>> k(rows * heads);
                std::vector<TQ4Block<D>> v(rows * heads);
                std::vector<float> expected(elements, 0.f);
                const float inverse_root = 1.f / std::sqrt(static_cast<float>(D));
                for (int head = 0; head < heads; ++head)
                {
                    query[head * D] = 0.5f;
                    for (int d = 0; d < D; ++d) identity[(head * D + d) * D + d] = 1.f;
                    for (int row = 0; row < rows; ++row)
                    {
                        auto &key = k[row * heads + head];
                        key.norm = static_cast<float>(D);
                        key.reconstruction_norm = 0.25f; // Deliberately different; key scoring uses source norm.
                        std::fill(std::begin(key.indices), std::end(key.indices), row == 0 ? head : 127);
                        auto &value = v[row * heads + head];
                        value.norm = 0.25f; // Value reconstruction uses the other, fitted norm.
                        value.reconstruction_norm = std::sqrt(static_cast<float>(D));
                        for (int group = 0; group < D / 8; ++group)
                        {
                            std::uint8_t low[8]{};
                            for (int lane = 0; lane < 8; ++lane)
                            {
                                const auto index = static_cast<std::uint8_t>((head + group * 8 + lane + row * 8) % 16);
                                low[lane] = index & 7;
                                value.high_bits[group] |= static_cast<std::uint8_t>((index >> 3) << lane);
                            }
                            tq3_pack_8(low, value.mse_indices + group * 3);
                        }
                    }
                    const auto *rotated_q = query.data() + head * D;
                    const float score0 = tq8_dot_rotated_q_scalar(rotated_q,
                        reinterpret_cast<const std::uint8_t *>(&k[head]), D) / D;
                    const float score1 = tq8_dot_rotated_q_scalar(rotated_q,
                        reinterpret_cast<const std::uint8_t *>(&k[heads + head]), D) / D;
                    const float max_score = std::max(score0, score1);
                    const float weight0 = std::exp(score0 - max_score), weight1 = std::exp(score1 - max_score);
                    auto *result = expected.data() + head * D;
                    tq4_accum_weighted_scalar(result, reinterpret_cast<const std::uint8_t *>(&v[head]),
                        weight0 * inverse_root, D);
                    tq4_accum_weighted_scalar(result, reinterpret_cast<const std::uint8_t *>(&v[heads + head]),
                        weight1 * inverse_root, D);
                    for (int d = 0; d < D; ++d) result[d] /= weight0 + weight1;
                }
                checked(cudaMemcpyAsync(q.data(), query.data(), q.bytes(), cudaMemcpyHostToDevice, stream));
                checked(cudaMemcpyAsync(keys.data(), k.data(), keys.bytes(), cudaMemcpyHostToDevice, stream));
                checked(cudaMemcpyAsync(values.data(), v.data(), values.bytes(), cudaMemcpyHostToDevice, stream));
                checked(cudaMemcpyAsync(rotation.data(), identity.data(), rotation.bytes(), cudaMemcpyHostToDevice, stream));
                checked(cudaStreamSynchronize(stream)); // Setup publication only; no attention warmup.
                capture.record([&] {
                    return cudaFlashAttn_decode_tqkv(q.as<float>(), output.as<float>(), partial.as<float>(),
                        maximum.as<float>(), denominator.as<float>(), keys.data(), values.data(),
                        rotation.as<float>(), rotation.as<float>(), 1, rows, heads, heads, D, splits,
                        rows, 0, sizeof(TQ8Block<D>), sizeof(TQ4Block<D>), nullptr, stream, ordinal, 0, 0);
                });
                std::vector<float> first;
                std::vector<float> actual(elements + guards);
                for (int replay = 0; replay < 20; ++replay)
                {
                    checked(cudaMemsetAsync(output.data(), 0xa5, output.bytes(), stream));
                    checked(cudaMemsetAsync(partial.data(), 0xcd, partial.bytes(), stream));
                    checked(cudaMemsetAsync(maximum.data(), 0xcd, maximum.bytes(), stream));
                    checked(cudaMemsetAsync(denominator.data(), 0xcd, denominator.bytes(), stream));
                    capture.launch();
                    checked(cudaMemcpyAsync(actual.data(), output.data(), output.bytes(), cudaMemcpyDeviceToHost, stream));
                    checked(cudaStreamSynchronize(stream)); // One explicit terminal observation per replay.
                    if (replay == 0) first.assign(actual.begin(), actual.begin() + elements);
                    else ASSERT_EQ(std::memcmp(first.data(), actual.data(), elements * sizeof(float)), 0);
                    for (std::size_t i = 0; i < elements; ++i)
                    {
                        ASSERT_TRUE(std::isfinite(actual[i]));
                        // Two-key FP32 softmax permits only rounding/exp error;
                        // codebook reconstruction itself uses the canonical CPU primitives.
                        ASSERT_NEAR(actual[i], expected[i], 2e-6f) << "element=" << i << " replay=" << replay;
                    }
                    for (std::size_t i = elements; i < actual.size(); ++i)
                    {
                        std::uint32_t bits = 0;
                        std::memcpy(&bits, &actual[i], sizeof(bits));
                        ASSERT_EQ(bits, 0xa5a5a5a5u);
                    }
                }
            } // Graph and exact stream retire before any backing root.
            if (retire)
            {
                TransferEngine transfer;
                auto ticket = transfer.beginExclusiveModelRetirement(
                    ModelDeviceMemoryRetention{.device = device, .reusable_workspace_bytes = total});
                roots.clear();
                const auto receipt = transfer.completeExclusiveModelRetirement(std::move(ticket));
                EXPECT_TRUE(receipt.runtime_reset_invoked);
                EXPECT_EQ(receipt.retired_runtime_generation, generation);
                EXPECT_GT(receipt.successor_runtime_generation, generation);
                EXPECT_EQ(&cudaHardwareMetadata().at(ordinal), physical);
            }
            std::cout << "[CUDAAttentionCentroidFirstCapture] device=" << ordinal << " D=" << D
                      << " generation=" << generation << " heads=" << heads
                      << " key_centroids=256 value_centroids=16 captured_replays=20 retired=" << retire << '\n';
        }
    }

    /** @test First native capture reads all static codebooks on every visible device and successor generation. */
    TEST(CUDAAttentionCentroidFirstCapture, AllCentroidsEveryDeviceAndRetiredGeneration)
    {
        initializeCUDAHardwareMetadata();
        const auto count = cudaHardwareMetadata().size();
        if (count == 0) GTEST_SKIP() << "UNQUALIFIED: no visible CUDA device for native centroid replay";
        for (std::size_t ordinal = 0; ordinal < count; ++ordinal)
            for (int generation = 0; generation < 2; ++generation)
            {
                SCOPED_TRACE(ordinal);
                SCOPED_TRACE(generation);
                exercise<64>(static_cast<int>(ordinal), false);
                if (::testing::Test::HasFatalFailure()) return;
                exercise<128>(static_cast<int>(ordinal), false);
                if (::testing::Test::HasFatalFailure()) return;
                exercise<256>(static_cast<int>(ordinal), generation == 0);
                if (::testing::Test::HasFatalFailure()) return;
            }
    }
}
