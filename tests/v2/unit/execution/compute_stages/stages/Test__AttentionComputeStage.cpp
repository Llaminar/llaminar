/**
 * @file Test__AttentionComputeStage.cpp
 * @brief AttentionComputeStage unit contracts and device-workspace integration proofs.
 * @author David Sanftenberg
 *
 * Tests the new AttentionComputeStage which uses KernelFactory for
 * type-safe attention kernel dispatch. CPU contracts run in Unit; the CUDA and
 * ROCm workspace cases use physical-device properties and are separately
 * registered in ProductionParityPreflight. No model weights are needed.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

#include "backends/DeviceId.h"
#include "execution/compute_stages/ComputeStages.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "v2/tensors/Tensors.h"
#include "v2/tensors/TensorFactory.h"
#include "v2/utils/DebugEnv.h"
#include "v2/utils/MPIContext.h"
#include "kernels/attention/AttentionWorkspaceContract.h"
#include <omp.h>

#ifdef HAVE_CUDA
#include "kernels/cuda/attention/CUDAFlashAttentionKernelT.h"
#endif

#ifdef HAVE_ROCM
#include "kernels/rocm/attention/ROCmFlashAttentionKernelT.h"
#endif

using namespace llaminar2;

namespace
{
    constexpr float TOLERANCE = 1e-4f;

    /**
     * @brief Temporarily override attention diagnostics and reload DebugEnv.
     *
     * DebugEnv caches parsed values, so changing the process environment alone
     * would make this unit depend on test order.  The helper restores every
     * original value and reloads the cache when it leaves scope.
     */
    class ScopedAttentionDebugEnv
    {
    public:
        explicit ScopedAttentionDebugEnv(
            std::initializer_list<std::pair<const char *, const char *>> values)
        {
            for (const auto &[name, value] : values)
            {
                Entry entry;
                entry.name = name;
                if (const char *old_value = std::getenv(name))
                {
                    entry.had_value = true;
                    entry.old_value = old_value;
                }
                entries_.push_back(std::move(entry));
                ::setenv(name, value, 1);
            }
            mutableDebugEnv().reload();
        }

        ~ScopedAttentionDebugEnv()
        {
            for (const auto &entry : entries_)
            {
                if (entry.had_value)
                    ::setenv(entry.name.c_str(), entry.old_value.c_str(), 1);
                else
                    ::unsetenv(entry.name.c_str());
            }
            mutableDebugEnv().reload();
        }

        ScopedAttentionDebugEnv(const ScopedAttentionDebugEnv &) = delete;
        ScopedAttentionDebugEnv &operator=(const ScopedAttentionDebugEnv &) = delete;

    private:
        struct Entry
        {
            std::string name;
            bool had_value = false;
            std::string old_value;
        };

        std::vector<Entry> entries_;
    };

    class Test__AttentionComputeStage : public ::testing::Test
    {
    protected:
        MPIContext mpi_ctx_{0, 1, MPI_COMM_WORLD};
        DeviceId device_id_ = DeviceId::cpu();

        void SetUp() override {}
        void TearDown() override {}
    };

    /**
     * @brief Cover every prefill bucket from one physical-device family BOM.
     * @param device Backend whose immutable physical properties are queried.
     * @param mpi_ctx Test-owned tensor allocation context.
     *
     * These are pre-binding descriptor queries, not kernel execution. Each
     * maximum-sized CPU operand supplies the real dtype and tensor capacity;
     * each stage declares the active rows it would capture. No large device
     * allocation or model is required to catch intermediate-M scratch peaks.
     */
    void expectPrefillWorkspaceFamilyCoverage(DeviceId device, MPIContext &mpi_ctx)
    {
        TensorFactory factory(mpi_ctx);
        std::vector<std::unique_ptr<ITensor>> queries;
        queries.push_back(factory.createFP32({4096, 24 * 256}, DeviceId::cpu()));
        queries.push_back(factory.createFP16({4096, 24 * 256}));
        queries.push_back(factory.createBF16({4096, 24 * 256}));
        AttentionComputeStage::Params params;
        params.batch_size = 1;
        params.seq_len = 4096;
        params.kv_len = 4096;
        params.n_heads = 24;
        params.n_kv_heads = 4;
        params.head_dim = 256;
        params.device_id = device;
        params.mpi_ctx = &mpi_ctx;
        params.execution_policy.prefill_parallel_axis =
            attention::AttentionPrefillParallelAxis::GeometrySelected;

        // These kernel dtypes share the workspace ABI even though production
        // model activation admission currently permits FP32 only.
        for (const auto &query : queries)
        {
            params.Q = query.get();
            for (const int context : {4096, 16384})
            {
                params.kv_len = context;
                size_t smallest_partial_bytes = 0;
                for (const int limit : {32, 512, 2048, 4096})
                {
                    SCOPED_TRACE(::testing::Message() << device.toString()
                        << " type=" << query->dtype_name()
                        << " context=" << context << " admitted_M=" << limit);
                    params.seq_len = limit;
                    AttentionComputeStage largest(params);
                    const auto family = largest.getWorkspaceRequirements(1, 24, 256);
                    const auto *partials = family.find("attn_partial_output");
                    ASSERT_NE(partials, nullptr);
                    if (limit == 32)
                        smallest_partial_bytes = partials->size_bytes;
                    if (limit == 4096)
                    {
                        // Identical KV capacity does not imply identical query
                        // capacity. A tiny admitted bucket must not reserve
                        // summaries for unadmitted, much wider query grids.
                        EXPECT_LT(smallest_partial_bytes, partials->size_bytes);
                    }
                    for (const int rows : {1, 16, 17, 32, 64, 128, 256, 512,
                                           1024, 1536, 2048, 3072, 4096})
                    {
                        if (rows > limit)
                            continue;
                        SCOPED_TRACE(::testing::Message() << " M=" << rows);
                        params.seq_len = rows;
                        AttentionComputeStage member(params);
                        const auto required = member.getWorkspaceRequirements(1, 24, 256);
                        for (const auto &buffer : required.buffers)
                        {
                            const auto *published = family.find(buffer.name);
                            ASSERT_NE(published, nullptr) << buffer.name;
                            EXPECT_GE(published->size_bytes, buffer.size_bytes) << buffer.name;
                        }
                    }
                }
            }
        }
    }

    /**
     * @brief Minimal KV cache stub for attention metadata contract tests.
     *
     * The lightweight fake keeps host-coherence and diagnostic regressions in
     * the unit suite without requiring CUDA/ROCm devices or allocating real KV
     * storage.
     */
    class FakeCaptureKVCache final : public IKVCache
    {
    public:
        int cached_tokens = 0;
        mutable int cached_token_queries = 0;
        int canonical_device_cached_tokens = 0;
        int canonical_device_ring_head = 0;
        ActivationPrecision k_precision_value = ActivationPrecision::FP16;
        ActivationPrecision v_precision_value = ActivationPrecision::FP16;
        std::optional<DeviceReadStorage> prepared_read_storage;

        /**
         * @brief Return opaque prepared addresses without reading any payload.
         * @param request Immutable query geometry; the fixture supplies one bank.
         * @return Declared descriptor, or no storage until the test prepares it.
         */
        std::optional<DeviceReadStorage> describeDeviceReadStorage(
            const DeviceReadStorageRequest &request) const override
        {
            (void)request;
            return prepared_read_storage;
        }

        ActivationPrecision k_precision() const override { return k_precision_value; }
        ActivationPrecision v_precision() const override { return v_precision_value; }
        int get_cached_tokens(int, int = 0) const override
        {
            ++cached_token_queries;
            return cached_tokens;
        }
        int max_seq_len() const override { return 4096; }
        int n_layers() const override { return 1; }

        bool get_kv(int, int, ITensor **out_k, ITensor **out_v,
                    int *out_kv_len = nullptr) override
        {
            if (out_k)
                *out_k = nullptr;
            if (out_v)
                *out_v = nullptr;
            if (out_kv_len)
                *out_kv_len = cached_tokens;
            return true;
        }

        bool get_kv(int, int, const ITensor **out_k, const ITensor **out_v,
                    int *out_kv_len = nullptr) const override
        {
            if (out_k)
                *out_k = nullptr;
            if (out_v)
                *out_v = nullptr;
            if (out_kv_len)
                *out_kv_len = cached_tokens;
            return true;
        }

        bool append(int, int, const ITensor *, const ITensor *, int) override
        {
            return false;
        }

        bool resetRequestState(const StateResetContext &) override
        {
            cached_tokens = 0;
            return true;
        }
        bool resetSequenceState(int, const StateResetContext &) override
        {
            cached_tokens = 0;
            return true;
        }
        bool resetLayerSequenceState(
            int,
            int,
            const StateResetContext &) override
        {
            cached_tokens = 0;
            return true;
        }
        bool resetLayerState(int, const StateResetContext &) override
        {
            cached_tokens = 0;
            return true;
        }

        /**
         * @brief Expose stable opaque addresses for device-metadata view tests.
         *
         * This CPU-only fake never dereferences these addresses through a GPU
         * backend. AttentionComputeStage merely wraps them in graph-stable
         * non-owning views while building diagnostic metadata. Providing both
         * pointers keeps the fake faithful to the production GPU cache contract:
         * effective-KV diagnostics may observe only the canonical count/head
         * pair, never a host coherence mirror.
         */
        const int *deviceCachedTokenCountPtr(int, int = 0) const override
        {
            return &canonical_device_cached_tokens;
        }

        const int *deviceRingHeadPtr(int, int = 0) const override
        {
            return &canonical_device_ring_head;
        }
    };

    /**
     * @brief Test that AttentionComputeStage can be constructed with valid params
     */
    TEST_F(Test__AttentionComputeStage, Construction)
    {
        TensorFactory factory(mpi_ctx_);

        // Minimal dimensions
        int seq_len = 4;
        int n_heads = 2;
        int n_kv_heads = 2;
        int head_dim = 8;
        size_t hidden_size = n_heads * head_dim;

        // Create tensors
        auto Q = factory.createFP32({static_cast<size_t>(seq_len), hidden_size}, device_id_);
        auto K = factory.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)}, device_id_);
        auto V = factory.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)}, device_id_);
        auto output = factory.createFP32({static_cast<size_t>(seq_len), hidden_size}, device_id_);

        // Construct stage
        AttentionComputeStage::Params params;
        params.Q = Q.get();
        params.K = K.get();
        params.V = V.get();
        params.output = output.get();
        params.batch_size = 1;
        params.seq_len = seq_len;
        params.kv_len = seq_len;
        params.n_heads = n_heads;
        params.n_kv_heads = n_kv_heads;
        params.head_dim = head_dim;
        params.causal = true;
        params.mpi_ctx = &mpi_ctx_;
        params.device_id = device_id_;

        auto stage = std::make_unique<AttentionComputeStage>(params);
        ASSERT_NE(stage, nullptr);
        EXPECT_EQ(stage->type(), ComputeStageType::ATTENTION);
    }

    /**
     * @brief Reject pre-RoPE cache bytes anywhere except the captured GPU reader.
     *
     * The policy replaces three independently mutable stage fields. These
     * constructor checks make it impossible for a manually assembled CPU stage,
     * cacheless GPU stage to claim that a later
     * operation will transform K.
     */
    TEST_F(Test__AttentionComputeStage, PreRotaryKeyCachePolicyRequiresCacheBackedGPUReader)
    {
        FakeCaptureKVCache cache;
        AttentionComputeStage::Params params;
        params.execution_policy.key_cache = {
            .encoding = attention::AttentionKeyCacheEncoding::
                PreRotaryDeviceTransform,
            .rope_theta = 10000.0f,
            .partial_rotary_factor = 1.0f,
        };
        params.kv_cache = &cache;

        params.device_id = DeviceId::cpu();
        EXPECT_THROW(
            { AttentionComputeStage stage(params); },
            std::invalid_argument);

        params.device_id = DeviceId::cuda(0);
        params.kv_cache = nullptr;
        EXPECT_THROW(
            { AttentionComputeStage stage(params); },
            std::invalid_argument);

        params.kv_cache = &cache;
        EXPECT_NO_THROW({ AttentionComputeStage stage(params); });

        params.execution_policy.key_cache.partial_rotary_factor = 0.0f;
        EXPECT_THROW(
            { AttentionComputeStage stage(params); },
            std::invalid_argument);
    }

    /**
     * @brief Test supportsBackend for CPU backends
     */
    TEST_F(Test__AttentionComputeStage, SupportsBackend_CPU)
    {
        TensorFactory factory(mpi_ctx_);

        int seq_len = 4;
        int n_heads = 2;
        int n_kv_heads = 2;
        int head_dim = 8;
        size_t hidden_size = n_heads * head_dim;

        auto Q = factory.createFP32({static_cast<size_t>(seq_len), hidden_size}, device_id_);
        auto K = factory.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)}, device_id_);
        auto V = factory.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)}, device_id_);
        auto output = factory.createFP32({static_cast<size_t>(seq_len), hidden_size}, device_id_);

        AttentionComputeStage::Params params;
        params.Q = Q.get();
        params.K = K.get();
        params.V = V.get();
        params.output = output.get();
        params.batch_size = 1;
        params.seq_len = seq_len;
        params.kv_len = seq_len;
        params.n_heads = n_heads;
        params.n_kv_heads = n_kv_heads;
        params.head_dim = head_dim;

        auto stage = std::make_unique<AttentionComputeStage>(params);

        // Should support CPU backends
        EXPECT_TRUE(stage->supportsBackend(ComputeBackendType::CPU));
        EXPECT_TRUE(stage->supportsBackend(ComputeBackendType::CPU));

        // Unknown backends should not be supported
        EXPECT_FALSE(stage->supportsBackend(static_cast<ComputeBackendType>(999)));
    }

    /**
     * @brief Dump metadata must not observe host KV state during graph capture.
     *
     * Snapshot nodes are assembled inside the same recording window as the
     * production attention launch. A host cached-token query there is both an
     * illegal coherence boundary and a stale-state risk. This test performs no
     * GPU work: the thread-local capture guard and counting cache fake prove
     * that capture-time metadata neither observes host state nor substitutes
     * the declarative projection tensor for an unresolved production cache
     * view. Complete immutable descriptors must be prepared before capture;
     * missing storage and address changes fail without consulting host state.
     */
    TEST_F(Test__AttentionComputeStage, GPUDumpInfoNeverQueriesHostKVState)
    {
        ScopedAttentionDebugEnv env({
            {"LLAMINAR_DEBUG_EFFECTIVE_KV_SNAPSHOT", "1"},
            {"LLAMINAR_DEBUG_EFFECTIVE_KV_SNAPSHOT_LAYER", "0"},
        });
        TensorFactory factory(mpi_ctx_);
        constexpr int seq_len = 8;
        constexpr int n_heads = 4;
        constexpr int n_kv_heads = 2;
        constexpr int head_dim = 16;
        const size_t q_cols = n_heads * head_dim;
        const size_t kv_cols = n_kv_heads * head_dim;

        auto Q = factory.createFP32({seq_len, q_cols}, device_id_);
        FakeCaptureKVCache kv_cache;
        kv_cache.cached_tokens = seq_len;
        const size_t physical_kv_rows = static_cast<size_t>(kv_cache.max_seq_len());
        auto K = factory.createFP32({physical_kv_rows, kv_cols}, device_id_);
        auto V = factory.createFP32({physical_kv_rows, kv_cols}, device_id_);
        auto output = factory.createFP32({seq_len, q_cols}, device_id_);

        AttentionComputeStage::Params params;
        params.Q = Q.get();
        params.K = K.get();
        params.V = V.get();
        params.output = output.get();
        params.batch_size = 1;
        params.seq_len = seq_len;
        params.kv_len = seq_len;
        params.n_heads = n_heads;
        params.n_kv_heads = n_kv_heads;
        params.head_dim = head_dim;
        params.kv_cache = &kv_cache;
        params.layer_idx = 0;
        // These host words are opaque identity sentinels, never GPU payloads.
        // Wrapping them as metadata must not allocate, copy or access a device.
        uint16_t opaque_key = 0;
        uint16_t opaque_value = 0;
        uint16_t changed_key = 0;
        for (DeviceId device : {DeviceId::cuda(0), DeviceId::rocm(0)})
        {
            SCOPED_TRACE(device.toString());
            params.device_id = device;
            kv_cache.prepared_read_storage.reset();
            AttentionComputeStage stage(params);
            EXPECT_THROW(stage.buildDumpInfoImpl(), std::runtime_error)
                << "A missing read descriptor must not silently omit outputs";
            kv_cache.prepared_read_storage = IKVCache::DeviceReadStorage{
                &opaque_key, &opaque_value, physical_kv_rows, kv_cols,
                TensorType::FP16, device};
            {
                GraphCaptureGuard capture;
                EXPECT_THROW(stage.buildDumpInfoImpl(), std::runtime_error)
                    << "Descriptor construction must precede graph recording";
            }

            const auto prepared = stage.buildDumpInfoImpl();
            ASSERT_EQ(prepared.outputs.size(), 5u);
            {
                GraphCaptureGuard capture;
                const StageDumpInfo captured = stage.buildDumpInfoImpl();
                ASSERT_EQ(captured.outputs.size(), prepared.outputs.size());
                const auto &effective_k = captured.outputs[1];
                const auto &effective_v = captured.outputs[2];
                EXPECT_STREQ(effective_k.name, "effective_k");
                EXPECT_STREQ(effective_v.name, "effective_v");
                EXPECT_STREQ(effective_k.dtype, "FP16");
                EXPECT_EQ(effective_k.rows, physical_kv_rows);
                EXPECT_EQ(effective_k.cols, kv_cols);
                ASSERT_NE(effective_k.tensor, nullptr);
                ASSERT_NE(effective_v.tensor, nullptr);
                EXPECT_NE(effective_k.tensor, K.get());
                EXPECT_NE(effective_v.tensor, V.get());
                EXPECT_EQ(effective_k.tensor->gpu_data_ptr(), &opaque_key);
                EXPECT_EQ(effective_v.tensor->gpu_data_ptr(), &opaque_value);
            }
            kv_cache.prepared_read_storage->key = &changed_key;
            EXPECT_THROW(stage.buildDumpInfoImpl(), std::runtime_error)
                << "An admitted pointer must not silently follow new storage";
            EXPECT_EQ(kv_cache.cached_token_queries, 0)
                << "Neither preparation nor recording may query host KV state";
        }
    }

    /**
     * @brief Test getDumpInfo returns valid scalar info
     */
    TEST_F(Test__AttentionComputeStage, GetDumpInfo)
    {
        TensorFactory factory(mpi_ctx_);

        int seq_len = 8;
        int n_heads = 4;
        int n_kv_heads = 2;
        int head_dim = 16;
        size_t hidden_size = n_heads * head_dim;

        auto Q = factory.createFP32({static_cast<size_t>(seq_len), hidden_size}, device_id_);
        auto K = factory.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)}, device_id_);
        auto V = factory.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)}, device_id_);
        auto output = factory.createFP32({static_cast<size_t>(seq_len), hidden_size}, device_id_);

        AttentionComputeStage::Params params;
        params.Q = Q.get();
        params.K = K.get();
        params.V = V.get();
        params.output = output.get();
        params.batch_size = 1;
        params.seq_len = seq_len;
        params.kv_len = seq_len;
        params.n_heads = n_heads;
        params.n_kv_heads = n_kv_heads;
        params.head_dim = head_dim;
        params.causal = true;
        params.window_size = -1;
        params.device_id = device_id_;

        auto stage = std::make_unique<AttentionComputeStage>(params);
        StageDumpInfo info = stage->getDumpInfo();

        // Check scalars captured key dimensions
        EXPECT_FALSE(info.scalars.empty());

        // Find specific scalars
        bool found_seq_len = false;
        bool found_n_heads = false;
        for (const auto &s : info.scalars)
        {
            if (std::string(s.name) == "seq_len")
            {
                found_seq_len = true;
                EXPECT_EQ(static_cast<int>(s.value), seq_len);
            }
            if (std::string(s.name) == "n_heads")
            {
                found_n_heads = true;
                EXPECT_EQ(static_cast<int>(s.value), n_heads);
            }
        }
        EXPECT_TRUE(found_seq_len);
        EXPECT_TRUE(found_n_heads);
    }

    /**
     * @brief Test estimatedFlops calculation
     */
    TEST_F(Test__AttentionComputeStage, EstimatedFlops)
    {
        TensorFactory factory(mpi_ctx_);

        int seq_len = 8;
        int kv_len = 16;
        int n_heads = 4;
        int n_kv_heads = 2;
        int head_dim = 16;
        size_t hidden_size = n_heads * head_dim;

        auto Q = factory.createFP32({static_cast<size_t>(seq_len), hidden_size}, device_id_);
        auto K = factory.createFP32({static_cast<size_t>(kv_len), static_cast<size_t>(n_kv_heads * head_dim)}, device_id_);
        auto V = factory.createFP32({static_cast<size_t>(kv_len), static_cast<size_t>(n_kv_heads * head_dim)}, device_id_);
        auto output = factory.createFP32({static_cast<size_t>(seq_len), hidden_size}, device_id_);

        AttentionComputeStage::Params params;
        params.Q = Q.get();
        params.K = K.get();
        params.V = V.get();
        params.output = output.get();
        params.batch_size = 1;
        params.seq_len = seq_len;
        params.kv_len = kv_len;
        params.n_heads = n_heads;
        params.n_kv_heads = n_kv_heads;
        params.head_dim = head_dim;

        auto stage = std::make_unique<AttentionComputeStage>(params);
        size_t flops = stage->estimatedFlops();

        // FLOPs should be:
        // Q@K^T: 2 * batch * n_heads * seq_len * kv_len * head_dim = 2*1*4*8*16*16 = 16384
        // softmax: ~4 * batch * n_heads * seq_len * kv_len = 4*1*4*8*16 = 2048
        // scores@V: 2 * batch * n_heads * seq_len * kv_len * head_dim = 16384
        // Total: 16384 + 2048 + 16384 = 34816
        size_t expected_qk = 2ULL * 1 * n_heads * seq_len * kv_len * head_dim;
        size_t expected_softmax = 4ULL * 1 * n_heads * seq_len * kv_len;
        size_t expected_sv = expected_qk;
        size_t expected_total = expected_qk + expected_softmax + expected_sv;

        EXPECT_EQ(flops, expected_total);
    }

    /**
     * @brief Test basic execute with FP32 tensors
     *
     * This test verifies that AttentionComputeStage::execute() can successfully
     * dispatch to the underlying attention kernel via KernelFactory.
     */
    TEST_F(Test__AttentionComputeStage, Execute_FP32_Basic)
    {
        TensorFactory factory(mpi_ctx_);

        // Small attention problem
        int seq_len = 2;
        int n_heads = 2;
        int n_kv_heads = 2;
        int head_dim = 4;
        size_t hidden_size = n_heads * head_dim;

        // Create tensors
        auto Q = factory.createFP32({static_cast<size_t>(seq_len), hidden_size}, device_id_);
        auto K = factory.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)}, device_id_);
        auto V = factory.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)}, device_id_);
        auto output = factory.createFP32({static_cast<size_t>(seq_len), hidden_size}, device_id_);

        // Initialize with simple values
        float *Q_data = Q->mutable_data();
        float *K_data = K->mutable_data();
        float *V_data = V->mutable_data();
        float *out_data = output->mutable_data();

        size_t total_q = seq_len * hidden_size;
        size_t total_kv = seq_len * n_kv_heads * head_dim;

        // Initialize Q, K, V with small values to avoid numerical issues
        for (size_t i = 0; i < total_q; ++i)
        {
            Q_data[i] = 0.1f * (i % 4);
        }
        for (size_t i = 0; i < total_kv; ++i)
        {
            K_data[i] = 0.1f * ((i + 1) % 4);
            V_data[i] = 0.1f * ((i + 2) % 4);
        }
        for (size_t i = 0; i < total_q; ++i)
        {
            out_data[i] = -999.0f; // Sentinel value
        }

        // Create workspace for attention scores
        auto workspace_scores = factory.createFP32(
            {static_cast<size_t>(n_heads * seq_len), static_cast<size_t>(seq_len)},
            device_id_);

        // Configure stage params
        AttentionComputeStage::Params params;
        params.Q = Q.get();
        params.K = K.get();
        params.V = V.get();
        params.output = output.get();
        params.batch_size = 1;
        params.seq_len = seq_len;
        params.kv_len = seq_len;
        params.n_heads = n_heads;
        params.n_kv_heads = n_kv_heads;
        params.head_dim = head_dim;
        params.causal = true;
        params.window_size = -1;
        params.workspace_scores = workspace_scores.get();
        params.mpi_ctx = &mpi_ctx_;
        params.device_id = device_id_;

        // Create and execute stage
        auto stage = std::make_unique<AttentionComputeStage>(params);

        // Execute with no device context (CPU execution)
        bool success = stage->execute(nullptr);
        EXPECT_TRUE(success) << "AttentionComputeStage::execute() should succeed";

        // Verify output was modified (not sentinel values)
        bool output_modified = false;
        for (size_t i = 0; i < total_q; ++i)
        {
            if (std::abs(out_data[i] - (-999.0f)) > 1e-6f)
            {
                output_modified = true;
                break;
            }
        }
        EXPECT_TRUE(output_modified) << "Output should be modified after execution";

        // Verify output is finite
        for (size_t i = 0; i < total_q; ++i)
        {
            EXPECT_TRUE(std::isfinite(out_data[i])) << "Output[" << i << "] = " << out_data[i] << " should be finite";
        }
    }

    /**
     * @brief Test execute with GQA (n_kv_heads < n_heads)
     */
    TEST_F(Test__AttentionComputeStage, Execute_FP32_GQA)
    {
        TensorFactory factory(mpi_ctx_);

        // GQA configuration: 4 query heads, 2 KV heads (ratio 2:1)
        int seq_len = 4;
        int n_heads = 4;
        int n_kv_heads = 2;
        int head_dim = 8;
        size_t hidden_size = n_heads * head_dim;

        auto Q = factory.createFP32({static_cast<size_t>(seq_len), hidden_size}, device_id_);
        auto K = factory.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)}, device_id_);
        auto V = factory.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)}, device_id_);
        auto output = factory.createFP32({static_cast<size_t>(seq_len), hidden_size}, device_id_);

        // Initialize with random-ish values
        float *Q_data = Q->mutable_data();
        float *K_data = K->mutable_data();
        float *V_data = V->mutable_data();

        for (size_t i = 0; i < Q->numel(); ++i)
            Q_data[i] = 0.05f * ((i * 7) % 11 - 5);
        for (size_t i = 0; i < K->numel(); ++i)
            K_data[i] = 0.05f * ((i * 13) % 11 - 5);
        for (size_t i = 0; i < V->numel(); ++i)
            V_data[i] = 0.05f * ((i * 17) % 11 - 5);

        auto workspace_scores = factory.createFP32(
            {static_cast<size_t>(n_heads * seq_len), static_cast<size_t>(seq_len)},
            device_id_);

        AttentionComputeStage::Params params;
        params.Q = Q.get();
        params.K = K.get();
        params.V = V.get();
        params.output = output.get();
        params.batch_size = 1;
        params.seq_len = seq_len;
        params.kv_len = seq_len;
        params.n_heads = n_heads;
        params.n_kv_heads = n_kv_heads;
        params.head_dim = head_dim;
        params.causal = true;
        params.workspace_scores = workspace_scores.get();
        params.mpi_ctx = &mpi_ctx_;
        params.device_id = device_id_;

        auto stage = std::make_unique<AttentionComputeStage>(params);
        bool success = stage->execute(nullptr);
        EXPECT_TRUE(success) << "GQA attention should succeed";

        // Verify output is valid
        float *out_data = output->mutable_data();
        for (size_t i = 0; i < output->numel(); ++i)
        {
            EXPECT_TRUE(std::isfinite(out_data[i])) << "GQA output[" << i << "] should be finite";
        }
    }

    TEST_F(Test__AttentionComputeStage, ContinuationCausalMaskMatchesFullPrefillForQwen35GQA)
    {
        TensorFactory factory(mpi_ctx_);

        constexpr int total_len = 9;
        constexpr int prefix_len = 4;
        constexpr int suffix_len = total_len - prefix_len;
        constexpr int n_heads = 8;
        constexpr int n_kv_heads = 2;
        constexpr int head_dim = 256;
        constexpr size_t q_cols = static_cast<size_t>(n_heads * head_dim);
        constexpr size_t kv_cols = static_cast<size_t>(n_kv_heads * head_dim);

        auto Q_full = factory.createFP32({total_len, q_cols}, device_id_);
        auto K_full = factory.createFP32({total_len, kv_cols}, device_id_);
        auto V_full = factory.createFP32({total_len, kv_cols}, device_id_);
        auto O_full = factory.createFP32({total_len, q_cols}, device_id_);
        auto Q_suffix = factory.createFP32({suffix_len, q_cols}, device_id_);
        auto O_suffix = factory.createFP32({suffix_len, q_cols}, device_id_);

        float *q_full = Q_full->mutable_data();
        float *k_full = K_full->mutable_data();
        float *v_full = V_full->mutable_data();
        for (size_t i = 0; i < Q_full->numel(); ++i)
            q_full[i] = std::sin(static_cast<float>(i % 251) * 0.013f) * 0.05f;
        for (size_t i = 0; i < K_full->numel(); ++i)
            k_full[i] = std::cos(static_cast<float>(i % 257) * 0.011f) * 0.05f;
        for (size_t i = 0; i < V_full->numel(); ++i)
            v_full[i] = std::sin(static_cast<float>(i % 263) * 0.017f) * 0.05f;

        std::copy(q_full + static_cast<size_t>(prefix_len) * q_cols,
                  q_full + static_cast<size_t>(total_len) * q_cols,
                  Q_suffix->mutable_data());

        AttentionComputeStage::Params full_params;
        full_params.Q = Q_full.get();
        full_params.K = K_full.get();
        full_params.V = V_full.get();
        full_params.output = O_full.get();
        full_params.batch_size = 1;
        full_params.seq_len = total_len;
        full_params.kv_len = total_len;
        full_params.n_heads = n_heads;
        full_params.n_kv_heads = n_kv_heads;
        full_params.head_dim = head_dim;
        full_params.causal = true;
        full_params.device_id = device_id_;
        full_params.mpi_ctx = &mpi_ctx_;

        AttentionComputeStage full_stage(full_params);
        ASSERT_TRUE(full_stage.execute(nullptr));

        AttentionComputeStage::Params suffix_params = full_params;
        suffix_params.Q = Q_suffix.get();
        suffix_params.output = O_suffix.get();
        suffix_params.seq_len = suffix_len;
        suffix_params.kv_len = total_len;
        suffix_params.position_offset = prefix_len;

        AttentionComputeStage suffix_stage(suffix_params);
        ASSERT_TRUE(suffix_stage.execute(nullptr));

        const float *full_out = O_full->data() + static_cast<size_t>(prefix_len) * q_cols;
        const float *suffix_out = O_suffix->data();
        float max_diff = 0.0f;
        for (size_t i = 0; i < static_cast<size_t>(suffix_len) * q_cols; ++i)
        {
            max_diff = std::max(max_diff, std::abs(full_out[i] - suffix_out[i]));
        }

        EXPECT_LT(max_diff, 2e-5f)
            << "Continuation causal position offset must match one-shot causal attention "
               "for suffix rows";
    }

    /**
     * @brief Test factory method createAttentionCompute
     */
    TEST_F(Test__AttentionComputeStage, FactoryMethod)
    {
        TensorFactory factory(mpi_ctx_);

        int seq_len = 4;
        int n_heads = 2;
        int n_kv_heads = 2;
        int head_dim = 8;
        size_t hidden_size = n_heads * head_dim;

        auto Q = factory.createFP32({static_cast<size_t>(seq_len), hidden_size}, device_id_);
        auto K = factory.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)}, device_id_);
        auto V = factory.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)}, device_id_);
        auto output = factory.createFP32({static_cast<size_t>(seq_len), hidden_size}, device_id_);

        AttentionComputeStage::Params params;
        params.Q = Q.get();
        params.K = K.get();
        params.V = V.get();
        params.output = output.get();
        params.batch_size = 1;
        params.seq_len = seq_len;
        params.kv_len = seq_len;
        params.n_heads = n_heads;
        params.n_kv_heads = n_kv_heads;
        params.head_dim = head_dim;

        // Use factory method
        auto stage = ComputeStageFactory::createAttentionCompute(params);
        ASSERT_NE(stage, nullptr);
        EXPECT_EQ(stage->type(), ComputeStageType::ATTENTION);
    }

    /**
     * @brief Local attention geometry must win over smaller and larger hints.
     *
     * A two-rank Qwen2 graph executes seven of fourteen Q heads per rank.
     * Taking max(global hint, local heads) inflated its runtime scratch beyond
     * the correctly admitted local BOM. Sweep TP degrees and worker counts:
     * CPU split cardinality depends on both, including non-divisible teams.
     * No kernel is launched and no GPU or model is required.
     */
    TEST_F(Test__AttentionComputeStage, CPUWorkspaceUsesExactParticipantGeometry)
    {
        /** Restore caller OpenMP policy even when an assertion exits early. */
        struct ThreadPolicyScope
        {
            int original = omp_get_max_threads();
            /** Restore the process test thread's inherited worker budget. */
            ~ThreadPolicyScope() { omp_set_num_threads(original); }
        } thread_policy;
        TensorFactory factory(mpi_ctx_);
        constexpr int global_heads = 14;
        constexpr int head_dim = 64;
        for (const int workers : {1, 3, 7, 28, 56})
        {
            omp_set_num_threads(workers);
            for (int tp = 1; tp <= 8; ++tp)
            {
                const int local_heads = (global_heads + tp - 1) / tp;
                auto Q = factory.createFP32(
                    {1u, static_cast<size_t>(local_heads * head_dim)}, device_id_);
                AttentionComputeStage::Params params;
                params.device_id = device_id_;
                params.Q = Q.get();
                params.n_heads = local_heads;
                params.n_kv_heads = 1;
                params.head_dim = head_dim;
                params.batch_size = 1;
                AttentionComputeStage stage(params);
                const auto expected = attention_workspace::cpuParallelRequirements({
                    .compact_query_rows = attention::kMaxGroupedVerifierAttentionRows,
                    .local_query_heads = local_heads,
                    .head_dim = head_dim,
                    .worker_count = workers,
                });
                for (const int rows : {1, 9, 16, 512})
                {
                    for (const int hinted_heads : {0, local_heads / 2, global_heads})
                    {
                        SCOPED_TRACE(::testing::Message()
                                     << "workers=" << workers << " tp=" << tp
                                     << " rows=" << rows << " hinted_heads=" << hinted_heads);
                        const auto actual = stage.getWorkspaceRequirements(
                            rows, hinted_heads, head_dim);
                        ASSERT_EQ(actual.buffers.size(), expected.buffers.size());
                        for (const auto &buffer : expected.buffers)
                        {
                            const auto *found = actual.find(buffer.name);
                            ASSERT_NE(found, nullptr);
                            EXPECT_EQ(found->size_bytes, buffer.size_bytes);
                        }
                        EXPECT_EQ(actual.total_bytes_with_alignment(),
                                  expected.total_bytes_with_alignment());
                    }
                }
                // Generic tensor-width hints are not attention head dimensions.
                EXPECT_EQ(stage.getWorkspaceRequirements(1, global_heads, global_heads * head_dim)
                              .total_bytes_with_alignment(),
                          expected.total_bytes_with_alignment());
            }
        }
    }

#ifdef HAVE_ROCM
    TEST_F(Test__AttentionComputeStage, ROCmWorkspaceRequirementsCoverStageShapeAndGroupedVerifierRows)
    {
        expectPrefillWorkspaceFamilyCoverage(DeviceId::rocm(0), mpi_ctx_);
        TensorFactory factory(mpi_ctx_);

        /*
         * Regressions for Qwen3.5 MoE long-context MTP: graph-level hints can
         * describe eight heads while this stage executes sixteen, and the same
         * one-request graph later executes up to four verifier rows.  Split
         * decode scratch must cover both maxima. The production descriptor
         * planner queries physical-device properties, so this is a model-free
         * Integration proof even though it does not launch attention kernels.
         */
        constexpr int seq_len = 1;
        constexpr int kv_len = 513;
        constexpr int stage_heads = 16;
        constexpr int hinted_heads = 8;
        constexpr int n_kv_heads = 2;
        constexpr int head_dim = 256;
        constexpr int default_decode_splits = 8;
        constexpr int grouped_verifier_rows = 4;

        auto Q = factory.createFP32(
            {static_cast<size_t>(seq_len), static_cast<size_t>(stage_heads * head_dim)},
            DeviceId::cpu());
        auto K = factory.createFP32(
            {static_cast<size_t>(kv_len), static_cast<size_t>(n_kv_heads * head_dim)},
            DeviceId::cpu());
        auto V = factory.createFP32(
            {static_cast<size_t>(kv_len), static_cast<size_t>(n_kv_heads * head_dim)},
            DeviceId::cpu());
        auto output = factory.createFP32(
            {static_cast<size_t>(seq_len), static_cast<size_t>(stage_heads * head_dim)},
            DeviceId::cpu());

        AttentionComputeStage::Params params;
        params.Q = Q.get();
        params.K = K.get();
        params.V = V.get();
        params.output = output.get();
        params.batch_size = 1;
        params.seq_len = seq_len;
        params.kv_len = kv_len;
        params.n_heads = stage_heads;
        params.n_kv_heads = n_kv_heads;
        params.head_dim = head_dim;
        params.causal = false;
        params.device_id = DeviceId::rocm(0);
        params.mpi_ctx = &mpi_ctx_;

        AttentionComputeStage stage(params);
        WorkspaceRequirements reqs = stage.getWorkspaceRequirements(
            /*m=*/seq_len,
            /*n=*/hinted_heads,
            /*k=*/head_dim);

        const auto *partial_output = reqs.find(rocm::AttentionWorkspaceBuffers::PARTIAL_OUTPUT);
        const auto *partial_m = reqs.find(rocm::AttentionWorkspaceBuffers::PARTIAL_M);
        const auto *partial_l = reqs.find(rocm::AttentionWorkspaceBuffers::PARTIAL_L);
        ASSERT_NE(partial_output, nullptr);
        ASSERT_NE(partial_m, nullptr);
        ASSERT_NE(partial_l, nullptr);

        const size_t expected_partial_output = static_cast<size_t>(grouped_verifier_rows) *
                                               static_cast<size_t>(stage_heads) *
                                               static_cast<size_t>(default_decode_splits) *
                                               static_cast<size_t>(head_dim) *
                                               sizeof(float);
        const size_t expected_partial_meta = static_cast<size_t>(grouped_verifier_rows) *
                                             static_cast<size_t>(stage_heads) *
                                             static_cast<size_t>(default_decode_splits) *
                                             sizeof(float);
        const size_t old_underallocated_partial_output = static_cast<size_t>(seq_len) *
                                                         static_cast<size_t>(hinted_heads) *
                                                         static_cast<size_t>(default_decode_splits) *
                                                         static_cast<size_t>(head_dim) *
                                                         sizeof(float);

        EXPECT_GE(partial_output->size_bytes, expected_partial_output);
        EXPECT_GE(partial_m->size_bytes, expected_partial_meta);
        EXPECT_GE(partial_l->size_bytes, expected_partial_meta);
        EXPECT_GT(partial_output->size_bytes, old_underallocated_partial_output)
            << "Attention workspace must cover both the stage's real head count "
               "and the maximum grouped verifier row span";

        // An unrelated global-model hint must not enlarge this local member.
        const auto oversized_hint = stage.getWorkspaceRequirements(
            seq_len, stage_heads * 8, head_dim * 8);
        ASSERT_EQ(oversized_hint.buffers.size(), reqs.buffers.size());
        for (const auto &buffer : reqs.buffers)
        {
            const auto *found = oversized_hint.find(buffer.name);
            ASSERT_NE(found, nullptr);
            EXPECT_EQ(found->size_bytes, buffer.size_bytes);
        }
    }
#endif

#ifdef HAVE_CUDA
    TEST_F(Test__AttentionComputeStage, CUDAWorkspaceRequirementsCoverStageShapeAndGroupedVerifierRows)
    {
        expectPrefillWorkspaceFamilyCoverage(DeviceId::cuda(0), mpi_ctx_);
        TensorFactory factory(mpi_ctx_);

        /*
         * Mirror the ROCm production-shape proof for CUDA.  CUDA uses a larger
         * conservative split bound, but it has the same M=2..4 verifier-row
         * lifetime and must not depend on an unrelated graph hint happening to
         * request a multi-row workspace first.
         */
        constexpr int seq_len = 1;
        constexpr int kv_len = 513;
        constexpr int stage_heads = 16;
        constexpr int hinted_heads = 8;
        constexpr int n_kv_heads = 2;
        constexpr int head_dim = 256;
        constexpr int max_decode_splits = 32;
        constexpr int grouped_verifier_rows = 4;

        auto Q = factory.createFP32(
            {static_cast<size_t>(seq_len), static_cast<size_t>(stage_heads * head_dim)},
            DeviceId::cpu());
        auto K = factory.createFP32(
            {static_cast<size_t>(kv_len), static_cast<size_t>(n_kv_heads * head_dim)},
            DeviceId::cpu());
        auto V = factory.createFP32(
            {static_cast<size_t>(kv_len), static_cast<size_t>(n_kv_heads * head_dim)},
            DeviceId::cpu());
        auto output = factory.createFP32(
            {static_cast<size_t>(seq_len), static_cast<size_t>(stage_heads * head_dim)},
            DeviceId::cpu());

        AttentionComputeStage::Params params;
        params.Q = Q.get();
        params.K = K.get();
        params.V = V.get();
        params.output = output.get();
        params.batch_size = 1;
        params.seq_len = seq_len;
        params.kv_len = kv_len;
        params.n_heads = stage_heads;
        params.n_kv_heads = n_kv_heads;
        params.head_dim = head_dim;
        params.causal = false;
        params.device_id = DeviceId::cuda(0);
        params.mpi_ctx = &mpi_ctx_;

        AttentionComputeStage stage(params);
        WorkspaceRequirements reqs = stage.getWorkspaceRequirements(
            /*m=*/seq_len,
            /*n=*/hinted_heads,
            /*k=*/head_dim);

        const auto *partial_output = reqs.find(cuda::AttentionWorkspaceBuffers::PARTIAL_OUTPUT);
        const auto *partial_m = reqs.find(cuda::AttentionWorkspaceBuffers::PARTIAL_M);
        const auto *partial_l = reqs.find(cuda::AttentionWorkspaceBuffers::PARTIAL_L);
        ASSERT_NE(partial_output, nullptr);
        ASSERT_NE(partial_m, nullptr);
        ASSERT_NE(partial_l, nullptr);

        const size_t expected_partial_output =
            static_cast<size_t>(grouped_verifier_rows) *
            static_cast<size_t>(stage_heads) *
            static_cast<size_t>(max_decode_splits) *
            static_cast<size_t>(head_dim) * sizeof(float);
        const size_t expected_partial_meta =
            static_cast<size_t>(grouped_verifier_rows) *
            static_cast<size_t>(stage_heads) *
            static_cast<size_t>(max_decode_splits) * sizeof(float);

        EXPECT_GE(partial_output->size_bytes, expected_partial_output);
        EXPECT_GE(partial_m->size_bytes, expected_partial_meta);
        EXPECT_GE(partial_l->size_bytes, expected_partial_meta);

        // Mirror the local-geometry ownership proof on the CUDA backend.
        const auto oversized_hint = stage.getWorkspaceRequirements(
            seq_len, stage_heads * 8, head_dim * 8);
        ASSERT_EQ(oversized_hint.buffers.size(), reqs.buffers.size());
        for (const auto &buffer : reqs.buffers)
        {
            const auto *found = oversized_hint.find(buffer.name);
            ASSERT_NE(found, nullptr);
            EXPECT_EQ(found->size_bytes, buffer.size_bytes);
        }
    }
#endif

} // namespace
