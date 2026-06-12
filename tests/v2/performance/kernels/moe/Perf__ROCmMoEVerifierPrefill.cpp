#include <gtest/gtest.h>

#include "execution/moe/MoEWorkspaceRequirements.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "kernels/IMoEKernel.h"
#include "kernels/KernelFactory.h"
#include "tensors/Tensors.h"

#include "../../../utils/GpuPreparedGemmHarness.h"
#include "../../../utils/TestTensorFactory.h"

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

/**
 * @file Perf__ROCmMoEVerifierPrefill.cpp
 * @brief ROCm half of the MoE verifier-prefill speedometer.
 *
 * CUDA and HIP runtime headers cannot be included in the same translation unit
 * in heterogeneous builds because both define vector types such as `dim3`. This
 * file mirrors the CUDA harness with ROCm-only timing and graph-capture calls.
 */

namespace
{
    using KernelFactory = llaminar::v2::kernels::KernelFactory;

    struct CloseMetrics
    {
        double cosine = 0.0;
        double relative_l2 = 0.0;
        double max_abs = 0.0;
    };

    struct BenchResult
    {
        std::string backend;
        std::string case_name;
        int m = 0;
        int top_k = 0;
        int num_experts = 0;
        int d_model = 0;
        int intermediate = 0;
        double eager_ms = 0.0;
        double graph_ms = 0.0;
        double rowwise_ms = 0.0;
        CloseMetrics metrics;
    };

    class ScopedEnvOverride
    {
    public:
        ScopedEnvOverride(const char *name, const char *value)
            : name_(name)
        {
            const char *old = std::getenv(name);
            if (old)
            {
                had_old_ = true;
                old_ = old;
            }
            setenv(name, value, 1);
        }

        ~ScopedEnvOverride()
        {
            if (had_old_)
                setenv(name_.c_str(), old_.c_str(), 1);
            else
                unsetenv(name_.c_str());
        }

    private:
        std::string name_;
        bool had_old_ = false;
        std::string old_;
    };

    int envInt(const char *name, int fallback)
    {
        const char *value = std::getenv(name);
        if (!value || !*value)
            return fallback;
        char *end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (end == value || parsed <= 0)
            return fallback;
        return static_cast<int>(parsed);
    }

    std::shared_ptr<llaminar2::FP32Tensor> makeTensor(
        const std::vector<size_t> &shape,
        const std::vector<float> &values)
    {
        auto tensor = std::make_shared<llaminar2::FP32Tensor>(shape);
        std::copy(values.begin(), values.end(), tensor->mutable_data());
        return tensor;
    }

    std::shared_ptr<llaminar2::FP32Tensor> makeZeros(const std::vector<size_t> &shape)
    {
        auto tensor = std::make_shared<llaminar2::FP32Tensor>(shape);
        std::fill(tensor->mutable_data(), tensor->mutable_data() + tensor->numel(), 0.0f);
        return tensor;
    }

    std::vector<float> makeHiddenValues(int rows, int d_model)
    {
        std::vector<float> values(static_cast<size_t>(rows) * d_model);
        for (size_t i = 0; i < values.size(); ++i)
        {
            values[i] =
                0.013f * static_cast<float>(static_cast<int>(i % 29) - 14) +
                0.002f * static_cast<float>(static_cast<int>((i / 7) % 11) - 5);
        }
        return values;
    }

    std::vector<float> makeRoutingIndices(int rows, int top_k, int num_experts)
    {
        std::vector<float> values(static_cast<size_t>(rows) * top_k);
        for (int row = 0; row < rows; ++row)
        {
            for (int k = 0; k < top_k; ++k)
                values[static_cast<size_t>(row) * top_k + k] =
                    static_cast<float>((k + ((row & 1) ? 4 : 0)) % num_experts);
        }
        return values;
    }

    /**
     * @brief Build a route table that maximizes active expert slots.
     *
     * The normal route fixture intentionally reuses experts across rows because
     * that resembles many real prompts.  The combined routed+shared verifier
     * experiment needs the opposite pressure: top-8 routed experts plus one
     * shared expert represented as a top-9 table.  Unique routes make the active
     * slot count deterministic and comparable to the CUDA speedometer.
     */
    std::vector<float> makeUniqueRoutingIndices(int rows, int top_k, int num_experts)
    {
        std::vector<float> values(static_cast<size_t>(rows) * top_k);
        for (int row = 0; row < rows; ++row)
        {
            for (int k = 0; k < top_k; ++k)
            {
                values[static_cast<size_t>(row) * top_k + k] =
                    static_cast<float>((row * top_k + k) % num_experts);
            }
        }
        return values;
    }

    std::vector<float> makeRoutingWeights(int rows, int top_k)
    {
        std::vector<float> values(static_cast<size_t>(rows) * top_k);
        for (int row = 0; row < rows; ++row)
        {
            float sum = 0.0f;
            for (int k = 0; k < top_k; ++k)
            {
                const float weight = 0.05f + 0.01f * static_cast<float>((row + k) % top_k);
                values[static_cast<size_t>(row) * top_k + k] = weight;
                sum += weight;
            }
            for (int k = 0; k < top_k; ++k)
                values[static_cast<size_t>(row) * top_k + k] /= sum;
        }
        return values;
    }

    CloseMetrics compareVectors(
        const std::vector<float> &actual,
        const std::vector<float> &expected)
    {
        EXPECT_EQ(actual.size(), expected.size());
        CloseMetrics metrics;
        double dot = 0.0;
        double norm_actual = 0.0;
        double norm_expected = 0.0;
        double diff2 = 0.0;
        for (size_t i = 0; i < actual.size(); ++i)
        {
            EXPECT_TRUE(std::isfinite(actual[i])) << "actual[" << i << "]";
            EXPECT_TRUE(std::isfinite(expected[i])) << "expected[" << i << "]";
            const double a = actual[i];
            const double e = expected[i];
            const double diff = a - e;
            dot += a * e;
            norm_actual += a * a;
            norm_expected += e * e;
            diff2 += diff * diff;
            metrics.max_abs = std::max(metrics.max_abs, std::abs(diff));
        }
        metrics.cosine =
            dot / (std::sqrt(norm_actual) * std::sqrt(norm_expected) + 1.0e-30);
        metrics.relative_l2 =
            std::sqrt(diff2) / (std::sqrt(norm_expected) + 1.0e-30);
        return metrics;
    }

    void expectClose(const CloseMetrics &metrics)
    {
        EXPECT_GE(metrics.cosine, 0.990)
            << "relative_l2=" << metrics.relative_l2 << " max_abs=" << metrics.max_abs;
        EXPECT_LE(metrics.relative_l2, 0.08)
            << "cosine=" << metrics.cosine << " max_abs=" << metrics.max_abs;
    }

    void printResult(const BenchResult &result)
    {
        static bool printed_header = false;
        if (!printed_header)
        {
            std::cout
                << "backend,case,m,top_k,num_experts,d_model,intermediate,"
                   "eager_ms,graph_ms,rowwise_ms,cosine,relative_l2,max_abs\n";
            printed_header = true;
        }

        std::cout << std::fixed << std::setprecision(4)
                  << result.backend << ','
                  << result.case_name << ','
                  << result.m << ','
                  << result.top_k << ','
                  << result.num_experts << ','
                  << result.d_model << ','
                  << result.intermediate << ','
                  << result.eager_ms << ','
                  << result.graph_ms << ','
                  << result.rowwise_ms << ','
                  << std::setprecision(8) << result.metrics.cosine << ','
                  << result.metrics.relative_l2 << ','
                  << result.metrics.max_abs << '\n';
    }

    struct PreparedExpertTables
    {
        std::vector<std::unique_ptr<llaminar2::TensorBase>> weights;
        std::vector<llaminar2::test::GpuPreparedGemm> prepared;
        std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> gate_descs;
        std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> up_descs;
        std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> down_descs;
        int gateup_table_id = -1;
        int down_table_id = -1;
    };

    /**
     * @brief Combined routed+shared Qwen3.6 descriptor tables.
     *
     * The combined table models the vLLM-style verifier fast path.  The split
     * table IDs exist only for this benchmark's correctness oracle and measure
     * the older routed + shared + gate-add path.
     */
    struct PreparedCombinedSharedTables : PreparedExpertTables
    {
        int routed_gateup_table_id = -1;
        int routed_down_table_id = -1;
        int shared_gateup_table_id = -1;
        int shared_down_table_id = -1;
        int shared_slot = -1;
    };

    /**
     * @brief Prepare production-style native-VNNI expert descriptors.
     *
     * The returned owner must stay alive because every descriptor points into
     * VRAM owned by the corresponding prepared GEMM handle.
     */
    PreparedExpertTables prepareExpertTables(
        llaminar2::IMoEKernel *moe,
        llaminar2::DeviceId device,
        int num_experts,
        int d_model,
        int intermediate)
    {
        PreparedExpertTables tables;
        tables.weights.reserve(static_cast<size_t>(num_experts) * 3);
        tables.prepared.reserve(static_cast<size_t>(num_experts) * 3);
        tables.gate_descs.reserve(num_experts);
        tables.up_descs.reserve(num_experts);
        tables.down_descs.reserve(num_experts);

        auto add_desc = [&](int rows, int cols, int seed, const char *role, int expected_codebook)
        {
            std::unique_ptr<llaminar2::TensorBase> weight;
            if (expected_codebook == 13)
            {
                weight = llaminar2::test::TestTensorFactory::createIQ2_SRandom(
                    {static_cast<size_t>(rows), static_cast<size_t>(cols)},
                    static_cast<unsigned>(seed));
            }
            else
            {
                weight = llaminar2::test::TestTensorFactory::createIQ4_XSRandom(
                    {static_cast<size_t>(rows), static_cast<size_t>(cols)},
                    static_cast<unsigned>(seed));
            }

            auto *weight_ptr = weight.get();
            tables.weights.push_back(std::move(weight));
            tables.prepared.push_back(llaminar2::test::makeGpuPreparedGemm(
                weight_ptr,
                device,
                "perf.moe_verifier.rocm." + std::string(role) + "." + std::to_string(seed),
                llaminar2::ModelContextId{280000 + static_cast<uint64_t>(seed)}));

            llaminar2::DeviceNativeVNNIMatrixDesc desc{};
            EXPECT_TRUE(tables.prepared.back().kernel->exportNativeVNNIMatrixDesc(desc));
            EXPECT_EQ(desc.n, rows);
            EXPECT_EQ(desc.k, cols);
            EXPECT_EQ(desc.codebook_id, expected_codebook);
            return desc;
        };

        for (int expert = 0; expert < num_experts; ++expert)
        {
            tables.gate_descs.push_back(add_desc(intermediate, d_model, 4100 + expert, "gate", 13));
            tables.up_descs.push_back(add_desc(intermediate, d_model, 4200 + expert, "up", 13));
            tables.down_descs.push_back(add_desc(d_model, intermediate, 4300 + expert, "down", 4));
        }

        tables.gateup_table_id = moe->uploadGroupedExpertGateUpDescriptorTables(
            tables.gate_descs.data(), tables.up_descs.data(), num_experts, d_model, intermediate);
        EXPECT_GE(tables.gateup_table_id, 0);
        tables.down_table_id = moe->uploadGroupedExpertDownDescriptorTable(
            tables.down_descs.data(), num_experts, d_model, intermediate);
        EXPECT_GE(tables.down_table_id, 0);
        return tables;
    }

    PreparedCombinedSharedTables prepareQwen36CombinedSharedTables(
        llaminar2::IMoEKernel *moe,
        llaminar2::DeviceId device,
        int routed_experts,
        int d_model,
        int intermediate)
    {
        constexpr int shared_experts = 1;
        const int combined_experts = routed_experts + shared_experts;
        const int shared_slot = routed_experts;

        PreparedCombinedSharedTables tables;
        tables.shared_slot = shared_slot;
        tables.weights.reserve(static_cast<size_t>(combined_experts) * 3);
        tables.prepared.reserve(static_cast<size_t>(combined_experts) * 3);
        tables.gate_descs.reserve(combined_experts);
        tables.up_descs.reserve(combined_experts);
        tables.down_descs.reserve(combined_experts);

        auto add_desc = [&](int rows, int cols, int seed, const char *role, int expected_codebook)
        {
            std::unique_ptr<llaminar2::TensorBase> weight;
            if (expected_codebook == 13)
            {
                weight = llaminar2::test::TestTensorFactory::createIQ2_SRandom(
                    {static_cast<size_t>(rows), static_cast<size_t>(cols)},
                    static_cast<unsigned>(seed));
            }
            else
            {
                weight = llaminar2::test::TestTensorFactory::createIQ4_XSRandom(
                    {static_cast<size_t>(rows), static_cast<size_t>(cols)},
                    static_cast<unsigned>(seed));
            }

            auto *weight_ptr = weight.get();
            tables.weights.push_back(std::move(weight));
            tables.prepared.push_back(llaminar2::test::makeGpuPreparedGemm(
                weight_ptr,
                device,
                "perf.moe_verifier.rocm.combined_shared." +
                    std::string(role) + "." + std::to_string(seed),
                llaminar2::ModelContextId{380000 + static_cast<uint64_t>(seed)}));

            llaminar2::DeviceNativeVNNIMatrixDesc desc{};
            EXPECT_TRUE(tables.prepared.back().kernel->exportNativeVNNIMatrixDesc(desc));
            EXPECT_EQ(desc.n, rows);
            EXPECT_EQ(desc.k, cols);
            EXPECT_EQ(desc.codebook_id, expected_codebook);
            return desc;
        };

        for (int expert = 0; expert < combined_experts; ++expert)
        {
            const bool is_shared = expert == shared_slot;
            const int gateup_codebook = is_shared ? 4 : 13;
            const int down_codebook = is_shared ? 13 : 4;
            tables.gate_descs.push_back(add_desc(intermediate, d_model, 5100 + expert, "gate", gateup_codebook));
            tables.up_descs.push_back(add_desc(intermediate, d_model, 5200 + expert, "up", gateup_codebook));
            tables.down_descs.push_back(add_desc(d_model, intermediate, 5300 + expert, "down", down_codebook));
        }

        tables.gateup_table_id = moe->uploadGroupedExpertGateUpDescriptorTables(
            tables.gate_descs.data(), tables.up_descs.data(),
            combined_experts, d_model, intermediate);
        EXPECT_GE(tables.gateup_table_id, 0);
        tables.down_table_id = moe->uploadGroupedExpertDownDescriptorTable(
            tables.down_descs.data(), combined_experts, d_model, intermediate);
        EXPECT_GE(tables.down_table_id, 0);

        tables.routed_gateup_table_id = moe->uploadGroupedExpertGateUpDescriptorTables(
            tables.gate_descs.data(), tables.up_descs.data(),
            routed_experts, d_model, intermediate);
        EXPECT_GE(tables.routed_gateup_table_id, 0);
        tables.routed_down_table_id = moe->uploadGroupedExpertDownDescriptorTable(
            tables.down_descs.data(), routed_experts, d_model, intermediate);
        EXPECT_GE(tables.routed_down_table_id, 0);

        tables.shared_gateup_table_id = moe->uploadGroupedExpertGateUpDescriptorTables(
            tables.gate_descs.data() + shared_slot,
            tables.up_descs.data() + shared_slot,
            shared_experts, d_model, intermediate);
        EXPECT_GE(tables.shared_gateup_table_id, 0);
        tables.shared_down_table_id = moe->uploadGroupedExpertDownDescriptorTable(
            tables.down_descs.data() + shared_slot,
            shared_experts, d_model, intermediate);
        EXPECT_GE(tables.shared_down_table_id, 0);
        return tables;
    }

    std::vector<float> makeSharedGateValues(int d_model)
    {
        std::vector<float> values(static_cast<size_t>(d_model));
        for (int i = 0; i < d_model; ++i)
        {
            values[static_cast<size_t>(i)] =
                0.0005f * static_cast<float>((i % 31) - 15);
        }
        return values;
    }
}

#ifdef HAVE_ROCM
namespace
{
    bool hasROCmDevice()
    {
        int count = 0;
        return hipGetDeviceCount(&count) == hipSuccess && count > 0;
    }

    class HipGraphOwner
    {
    public:
        ~HipGraphOwner()
        {
            if (exec_)
                (void)hipGraphExecDestroy(exec_);
            if (graph_)
                (void)hipGraphDestroy(graph_);
        }

        hipGraph_t *graphPtr() { return &graph_; }
        hipGraphExec_t *execPtr() { return &exec_; }
        hipGraphExec_t execHandle() const { return exec_; }

    private:
        hipGraph_t graph_ = nullptr;
        hipGraphExec_t exec_ = nullptr;
    };

    double timeHipEvents(hipStream_t stream, int iterations, const std::function<bool()> &body)
    {
        hipEvent_t start = nullptr;
        hipEvent_t stop = nullptr;
        EXPECT_EQ(hipEventCreate(&start), hipSuccess);
        EXPECT_EQ(hipEventCreate(&stop), hipSuccess);
        EXPECT_EQ(hipEventRecord(start, stream), hipSuccess);
        for (int i = 0; i < iterations; ++i)
            EXPECT_TRUE(body());
        EXPECT_EQ(hipEventRecord(stop, stream), hipSuccess);
        EXPECT_EQ(hipEventSynchronize(stop), hipSuccess);
        float ms = 0.0f;
        EXPECT_EQ(hipEventElapsedTime(&ms, start, stop), hipSuccess);
        EXPECT_EQ(hipEventDestroy(start), hipSuccess);
        EXPECT_EQ(hipEventDestroy(stop), hipSuccess);
        return static_cast<double>(ms) / static_cast<double>(iterations);
    }

    std::vector<float> runRowwiseDecode(
        llaminar2::IMoEKernel *moe,
        hipStream_t stream,
        const std::vector<float> &hidden_values,
        const std::vector<float> &routing_indices,
        const std::vector<float> &routing_weights,
        int rows,
        int top_k,
        int d_model,
        int intermediate,
        int gateup_table,
        int down_table,
        double *avg_ms)
    {
        const auto device = llaminar2::DeviceId::rocm(0);
        std::vector<float> decoded;
        decoded.reserve(static_cast<size_t>(rows) * d_model);

        auto decode_once = [&]()
        {
            decoded.clear();
            for (int row = 0; row < rows; ++row)
            {
                const auto row_begin = hidden_values.begin() + static_cast<ptrdiff_t>(row) * d_model;
                std::vector<float> row_hidden_values(row_begin, row_begin + d_model);
                auto row_hidden = makeTensor({1, static_cast<size_t>(d_model)}, row_hidden_values);
                EXPECT_TRUE(row_hidden->ensureOnDevice(device, stream));

                std::vector<int> expert_ids(static_cast<size_t>(top_k));
                std::vector<float> expert_weights(static_cast<size_t>(top_k));
                for (int k = 0; k < top_k; ++k)
                {
                    const size_t slot = static_cast<size_t>(row) * top_k + k;
                    expert_ids[static_cast<size_t>(k)] = static_cast<int>(routing_indices[slot]);
                    expert_weights[static_cast<size_t>(k)] = routing_weights[slot];
                }

                std::vector<std::shared_ptr<llaminar2::FP32Tensor>> gate_owned;
                std::vector<std::shared_ptr<llaminar2::FP32Tensor>> up_owned;
                std::vector<llaminar2::ITensor *> gate_outputs(static_cast<size_t>(top_k));
                std::vector<llaminar2::ITensor *> up_outputs(static_cast<size_t>(top_k));
                gate_owned.reserve(top_k);
                up_owned.reserve(top_k);
                for (int k = 0; k < top_k; ++k)
                {
                    gate_owned.push_back(makeZeros({static_cast<size_t>(intermediate)}));
                    up_owned.push_back(makeZeros({static_cast<size_t>(intermediate)}));
                    EXPECT_TRUE(gate_owned.back()->ensureOnDevice(device, stream));
                    EXPECT_TRUE(up_owned.back()->ensureOnDevice(device, stream));
                    gate_outputs[static_cast<size_t>(k)] = gate_owned.back().get();
                    up_outputs[static_cast<size_t>(k)] = up_owned.back().get();
                }

                auto decode_output = makeZeros({static_cast<size_t>(d_model)});
                EXPECT_TRUE(decode_output->ensureOnDevice(device, stream));
                EXPECT_TRUE(moe->groupedExpertGateUpDecodeFromTable(
                    row_hidden.get(), expert_ids.data(), gateup_table, top_k,
                    gate_outputs.data(), up_outputs.data(), d_model, intermediate));
                EXPECT_TRUE(moe->groupedExpertDownDecodeFromTable(
                    gate_outputs.data(), up_outputs.data(), expert_ids.data(), expert_weights.data(),
                    down_table, top_k, decode_output.get(), d_model, intermediate));
                EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);
                decode_output->transitionTo(llaminar2::TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
                decoded.insert(
                    decoded.end(),
                    decode_output->data(),
                    decode_output->data() + decode_output->numel());
            }
            return true;
        };

        const int timing_iters = std::max(1, envInt("LLAMINAR_MOE_VERIFIER_PREFILL_ROWWISE_ITERS", 3));
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < timing_iters; ++i)
            EXPECT_TRUE(decode_once());
        const auto stop = std::chrono::steady_clock::now();
        *avg_ms = std::chrono::duration<double, std::milli>(stop - start).count() /
                  static_cast<double>(timing_iters);
        return decoded;
    }

    BenchResult runROCmCase(
        bool shared,
        int rows,
        int routed_top_k = 8,
        int routed_num_experts = 256,
        const char *case_name_override = nullptr,
        bool unique_routes = false)
    {
        /*
         * Keep this harness aligned with the Qwen3.6 MoE model shape.  The
         * actual benchmark matrix routes across 256 experts, so the focused
         * speedometer should pay the same descriptor-table and grouping setup
         * cost instead of testing only a small proxy table.
         */
        constexpr int shared_top_k = 1;
        constexpr int shared_num_experts = 1;
        constexpr int d_model = 2048;
        constexpr int intermediate = 512;
        const int top_k = shared ? shared_top_k : routed_top_k;
        const int num_experts = shared ? shared_num_experts : routed_num_experts;
        const int iterations = envInt("LLAMINAR_MOE_VERIFIER_PREFILL_ITERS", 30);
        const int warmups = envInt("LLAMINAR_MOE_VERIFIER_PREFILL_WARMUPS", 5);
        const auto device = llaminar2::DeviceId::rocm(0);

        EXPECT_EQ(hipSetDevice(0), hipSuccess);
        hipStream_t stream = nullptr;
        EXPECT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);

        auto *moe = KernelFactory::getOrCreateMoEKernel(device);
        EXPECT_NE(moe, nullptr);
        moe->setGPUStream(stream);
        auto *workspace_consumer = dynamic_cast<llaminar2::IWorkspaceConsumer *>(moe);
        EXPECT_NE(workspace_consumer, nullptr);
        auto reqs = llaminar2::MoEWorkspaceBuffers::rocmMoE(
            /*max_seq_len=*/4,
            d_model,
            intermediate,
            num_experts,
            top_k);
        auto workspace = std::make_unique<llaminar2::DeviceWorkspaceManager>(
            device,
            reqs.total_bytes_with_alignment() + 8 * 1024 * 1024);
        EXPECT_TRUE(workspace->allocate(reqs));
        workspace_consumer->bindWorkspace(workspace.get());

        auto tables = prepareExpertTables(moe, device, num_experts, d_model, intermediate);
        const auto hidden_values = makeHiddenValues(rows, d_model);
        const auto routing_indices = unique_routes
                                         ? makeUniqueRoutingIndices(rows, top_k, num_experts)
                                         : makeRoutingIndices(rows, top_k, num_experts);
        const auto routing_weights = makeRoutingWeights(rows, top_k);
        auto hidden = makeTensor({static_cast<size_t>(rows), static_cast<size_t>(d_model)}, hidden_values);
        auto route_indices_tensor = makeTensor({static_cast<size_t>(rows), static_cast<size_t>(top_k)}, routing_indices);
        auto route_weights_tensor = makeTensor({static_cast<size_t>(rows), static_cast<size_t>(top_k)}, routing_weights);
        auto grouped_output = makeZeros({static_cast<size_t>(rows), static_cast<size_t>(d_model)});
        EXPECT_TRUE(hidden->ensureOnDevice(device, stream));
        EXPECT_TRUE(route_indices_tensor->ensureOnDevice(device, stream));
        EXPECT_TRUE(route_weights_tensor->ensureOnDevice(device, stream));
        EXPECT_TRUE(grouped_output->ensureOnDevice(device, stream));

        auto run_grouped = [&]()
        {
            if (shared)
            {
                if (!moe->prepareSharedExpertPrefillGroup(rows))
                    return false;
            }
            else if (!moe->prepareExpertGroupsAsync(
                         route_indices_tensor.get(), route_weights_tensor.get(),
                         rows, num_experts, top_k))
            {
                return false;
            }
            return moe->executeGroupedPrefillPipeline(
                hidden.get(), grouped_output.get(),
                tables.gateup_table_id, tables.down_table_id,
                rows, d_model, intermediate, num_experts, top_k);
        };

        for (int i = 0; i < warmups; ++i)
            EXPECT_TRUE(run_grouped());
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);

        const double eager_ms = timeHipEvents(stream, iterations, run_grouped);
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);

        HipGraphOwner graph;
        EXPECT_EQ(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal), hipSuccess);
        const bool captured = run_grouped();
        const hipError_t end_status = hipStreamEndCapture(stream, graph.graphPtr());
        EXPECT_TRUE(captured);
        EXPECT_EQ(end_status, hipSuccess) << hipGetErrorString(end_status);
        EXPECT_NE(*graph.graphPtr(), nullptr);
        EXPECT_EQ(hipGraphInstantiate(graph.execPtr(), *graph.graphPtr(), nullptr, nullptr, 0), hipSuccess);
        for (int i = 0; i < warmups; ++i)
            EXPECT_EQ(hipGraphLaunch(graph.execHandle(), stream), hipSuccess);
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);
        const double graph_ms = timeHipEvents(
            stream,
            iterations,
            [&]()
            {
                return hipGraphLaunch(graph.execHandle(), stream) == hipSuccess;
            });
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);

        grouped_output->transitionTo(llaminar2::TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
        std::vector<float> grouped(
            grouped_output->data(),
            grouped_output->data() + grouped_output->numel());

        double rowwise_ms = 0.0;
        std::vector<float> rowwise = runRowwiseDecode(
            moe, stream, hidden_values, routing_indices, routing_weights,
            rows, top_k, d_model, intermediate,
            tables.gateup_table_id, tables.down_table_id, &rowwise_ms);
        CloseMetrics metrics = compareVectors(grouped, rowwise);

        EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);

        return BenchResult{
            "rocm",
            case_name_override ? case_name_override : (shared ? "shared" : "routed"),
            rows,
            top_k,
            num_experts,
            d_model,
            intermediate,
            eager_ms,
            graph_ms,
            rowwise_ms,
            metrics};
    }

    BenchResult runROCmCombinedSharedGateCase(int rows)
    {
        /*
         * Production-shaped combined verifier case.  This path exercises
         * prepareExpertGroupsWithSharedGateAsync(), including the device-side
         * shared gate and original-slot to grouped-slot map consumed by the
         * ordered scatter kernel.
         */
        constexpr int routed_top_k = 8;
        constexpr int routed_experts = 256;
        constexpr int combined_experts = routed_experts + 1;
        constexpr int combined_top_k = routed_top_k + 1;
        constexpr int d_model = 2048;
        constexpr int intermediate = 512;
        const int iterations = envInt("LLAMINAR_MOE_VERIFIER_PREFILL_ITERS", 30);
        const int warmups = envInt("LLAMINAR_MOE_VERIFIER_PREFILL_WARMUPS", 5);
        const auto device = llaminar2::DeviceId::rocm(0);

        EXPECT_EQ(hipSetDevice(0), hipSuccess);
        hipStream_t stream = nullptr;
        EXPECT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);

        auto *moe = KernelFactory::getOrCreateMoEKernel(device);
        EXPECT_NE(moe, nullptr);
        moe->setGPUStream(stream);
        auto *workspace_consumer = dynamic_cast<llaminar2::IWorkspaceConsumer *>(moe);
        EXPECT_NE(workspace_consumer, nullptr);
        auto reqs = llaminar2::MoEWorkspaceBuffers::rocmMoE(
            /*max_seq_len=*/4,
            d_model,
            intermediate,
            combined_experts,
            combined_top_k);
        auto workspace = std::make_unique<llaminar2::DeviceWorkspaceManager>(
            device,
            reqs.total_bytes_with_alignment() + 8 * 1024 * 1024);
        EXPECT_TRUE(workspace->allocate(reqs));
        workspace_consumer->bindWorkspace(workspace.get());

        auto tables = prepareQwen36CombinedSharedTables(
            moe, device, routed_experts, d_model, intermediate);
        const auto hidden_values = makeHiddenValues(rows, d_model);
        const auto shared_gate_values = makeSharedGateValues(d_model);
        const auto routing_indices =
            makeUniqueRoutingIndices(rows, routed_top_k, routed_experts);
        const auto routing_weights = makeRoutingWeights(rows, routed_top_k);

        auto hidden = makeTensor({static_cast<size_t>(rows), static_cast<size_t>(d_model)}, hidden_values);
        auto shared_gate = makeTensor({static_cast<size_t>(d_model)}, shared_gate_values);
        auto route_indices_tensor = makeTensor({static_cast<size_t>(rows), static_cast<size_t>(routed_top_k)}, routing_indices);
        auto route_weights_tensor = makeTensor({static_cast<size_t>(rows), static_cast<size_t>(routed_top_k)}, routing_weights);
        auto combined_output = makeZeros({static_cast<size_t>(rows), static_cast<size_t>(d_model)});
        auto routed_output = makeZeros({static_cast<size_t>(rows), static_cast<size_t>(d_model)});
        auto shared_output = makeZeros({static_cast<size_t>(rows), static_cast<size_t>(d_model)});
        auto split_output = makeZeros({static_cast<size_t>(rows), static_cast<size_t>(d_model)});
        EXPECT_TRUE(hidden->ensureOnDevice(device, stream));
        EXPECT_TRUE(shared_gate->ensureOnDevice(device, stream));
        EXPECT_TRUE(route_indices_tensor->ensureOnDevice(device, stream));
        EXPECT_TRUE(route_weights_tensor->ensureOnDevice(device, stream));
        EXPECT_TRUE(combined_output->ensureOnDevice(device, stream));
        EXPECT_TRUE(routed_output->ensureOnDevice(device, stream));
        EXPECT_TRUE(shared_output->ensureOnDevice(device, stream));
        EXPECT_TRUE(split_output->ensureOnDevice(device, stream));

        auto run_combined = [&]()
        {
            if (!moe->prepareExpertGroupsWithSharedGateAsync(
                    route_indices_tensor.get(), route_weights_tensor.get(),
                    hidden.get(), shared_gate.get(),
                    rows, d_model, routed_experts, routed_top_k))
            {
                return false;
            }
            return moe->executeGroupedPrefillPipeline(
                hidden.get(), combined_output.get(),
                tables.gateup_table_id, tables.down_table_id,
                rows, d_model, intermediate, combined_experts, combined_top_k);
        };

        auto run_split_reference = [&]()
        {
            if (!moe->prepareExpertGroupsAsync(
                    route_indices_tensor.get(), route_weights_tensor.get(),
                    rows, routed_experts, routed_top_k) ||
                !moe->executeGroupedPrefillPipeline(
                    hidden.get(), routed_output.get(),
                    tables.routed_gateup_table_id, tables.routed_down_table_id,
                    rows, d_model, intermediate, routed_experts, routed_top_k) ||
                !moe->prepareSharedExpertPrefillGroup(rows) ||
                !moe->executeGroupedPrefillPipeline(
                    hidden.get(), shared_output.get(),
                    tables.shared_gateup_table_id, tables.shared_down_table_id,
                    rows, d_model, intermediate, /*num_experts=*/1, /*top_k=*/1))
            {
                return false;
            }
            moe->sharedExpertGateAddFromTensors(
                hidden.get(), shared_gate.get(), shared_output.get(),
                routed_output.get(), split_output.get(), rows, d_model);
            return true;
        };

        for (int i = 0; i < warmups; ++i)
            EXPECT_TRUE(run_combined());
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);
        const double eager_ms = timeHipEvents(stream, iterations, run_combined);
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);

        HipGraphOwner graph;
        EXPECT_EQ(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal), hipSuccess);
        const bool captured = run_combined();
        const hipError_t end_status = hipStreamEndCapture(stream, graph.graphPtr());
        EXPECT_TRUE(captured);
        EXPECT_EQ(end_status, hipSuccess) << hipGetErrorString(end_status);
        EXPECT_NE(*graph.graphPtr(), nullptr);
        EXPECT_EQ(hipGraphInstantiate(graph.execPtr(), *graph.graphPtr(), nullptr, nullptr, 0), hipSuccess);
        for (int i = 0; i < warmups; ++i)
            EXPECT_EQ(hipGraphLaunch(graph.execHandle(), stream), hipSuccess);
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);
        const double graph_ms = timeHipEvents(
            stream,
            iterations,
            [&]()
            {
                return hipGraphLaunch(graph.execHandle(), stream) == hipSuccess;
            });
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);

        const double split_ms = timeHipEvents(stream, std::max(1, iterations / 3), run_split_reference);
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);

        combined_output->transitionTo(llaminar2::TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
        split_output->transitionTo(llaminar2::TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
        std::vector<float> combined(
            combined_output->data(),
            combined_output->data() + combined_output->numel());
        std::vector<float> split(
            split_output->data(),
            split_output->data() + split_output->numel());
        CloseMetrics metrics = compareVectors(combined, split);

        EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);

        return BenchResult{
            "rocm",
            "combined_shared_gate",
            rows,
            combined_top_k,
            combined_experts,
            d_model,
            intermediate,
            eager_ms,
            graph_ms,
            split_ms,
            metrics};
    }
}
#endif

TEST(Perf__MoEVerifierPrefill, ROCm_M1234_RoutedAndShared)
{
#ifndef HAVE_ROCM
    GTEST_SKIP() << "ROCm support not compiled";
#else
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    ScopedEnvOverride stats_env("LLAMINAR_PERF_STATS_JSON", "1");
    for (int rows : {1, 2, 3, 4})
    {
        auto routed = runROCmCase(/*shared=*/false, rows);
        expectClose(routed.metrics);
        printResult(routed);

        auto shared = runROCmCase(/*shared=*/true, rows);
        expectClose(shared.metrics);
        printResult(shared);
    }
#endif
}

TEST(Perf__MoEVerifierPrefill, ROCm_M4_CombinedRoutedSharedUpperBound)
{
#ifndef HAVE_ROCM
    GTEST_SKIP() << "ROCm support not compiled";
#else
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    /*
     * Keep this paired with CUDA_M4_CombinedRoutedSharedUpperBound.  It models
     * the production verifier trick of folding routed top-8 plus the shared
     * expert into a single top-9 grouped-prefill pipeline, using deterministic
     * unique routes so the active-slot count is stable across runs.
     */
    ScopedEnvOverride stats_env("LLAMINAR_PERF_STATS_JSON", "1");
    auto combined = runROCmCase(
        /*shared=*/false,
        /*rows=*/4,
        /*routed_top_k=*/9,
        /*routed_num_experts=*/257,
        /*case_name_override=*/"combined_top9_upper_bound",
        /*unique_routes=*/true);
    expectClose(combined.metrics);
    printResult(combined);
#endif
}

TEST(Perf__MoEVerifierPrefill, ROCm_M234_CombinedSharedGateProductionShape)
{
#ifndef HAVE_ROCM
    GTEST_SKIP() << "ROCm support not compiled";
#else
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    ScopedEnvOverride stats_env("LLAMINAR_PERF_STATS_JSON", "1");
    /*
     * Fixed MTP depths d1/d2/d3 verify draft_count + 1 target rows. Keep
     * the production shared-gate speedometer aligned with those real verifier
     * shapes instead of proving only the depth-3/M=4 case.
     */
    for (int rows : {2, 3, 4})
    {
        auto combined = runROCmCombinedSharedGateCase(rows);
        expectClose(combined.metrics);
        printResult(combined);
    }
#endif
}
