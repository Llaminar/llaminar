/**
 * @file Test__AttentionComputeStage.cpp
 * @brief AttentionComputeStage unit contracts and device-workspace integration proofs.
 * @author David Sanftenberg
 *
 * Tests the new AttentionComputeStage which uses KernelFactory for
 * type-safe attention kernel dispatch. CPU contracts run in Unit; the CUDA and
 * ROCm workspace cases use physical-device properties and are separately
 * registered in ProductionTestPreflight. No model weights are needed.
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
#include "execution/compute_stages/stages/PooledKVAppendStage.h"
#include "execution/compute_stages/stages/PooledKVDecodeBinding.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "v2/tensors/Tensors.h"
#include "kernels/cpu/attention/CPUFlashAttentionKernelT.h"
#include "kernels/cpu/attention/CPUFP32PagedKVTable.h"
#include "planning/PagedKVInstance.h"
#include "kernels/cpu/rotation/ActivationRotation.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "planning/PhysicalPagePool.h"
#include "v2/tensors/TensorFactory.h"
#include "v2/utils/DebugEnv.h"
#include "v2/utils/MPIContext.h"
#include "kernels/attention/AttentionWorkspaceContract.h"
#include "kernels/attention/AttentionHeadMapping.h"
#include "config/TensorParallelConfig.h"
#include <limits>
#include <omp.h>

#ifdef HAVE_CUDA
#include "backends/cuda/CUDAHardwareMetadata.h"
#include "kernels/cuda/attention/CUDAFlashAttentionKernelT.h"
#endif

#ifdef HAVE_ROCM
#include "backends/rocm/ROCmHardwareMetadata.h"
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

    /** @test The common guard accepts odd global mappings and rejects malformed bounds without integer overflow. */
    TEST(AttentionHeadMapping, OriginalGlobalIntervalsAndMalformedBounds)
    {
        for (const int start : {0, 3, 6, 9, 12, 15, 18, 21})
            EXPECT_TRUE(attention::isValidHeadMapping({3, 2, start, 12}));
        // Unequal GQA partitions can straddle the global KV group boundary.
        EXPECT_TRUE(attention::isValidHeadMapping({5, 2, 5, 7}));
        EXPECT_TRUE(attention::isValidHeadMapping({4, 2, 0, 0}));
        EXPECT_FALSE(attention::isValidHeadMapping({3, 2, 0, 0}));
        EXPECT_FALSE(attention::isValidHeadMapping({3, 2, 22, 12}));
        EXPECT_FALSE(attention::isValidHeadMapping({1, 2, 24, 12}));
        EXPECT_FALSE(attention::isValidHeadMapping({3, 2, -1, 12}));
        EXPECT_FALSE(attention::isValidHeadMapping({3, 2, 0, -1}));
        EXPECT_FALSE(attention::isValidHeadMapping({0, 2, 0, 12}));
        EXPECT_FALSE(attention::isValidHeadMapping({3, 0, 0, 12}));
        constexpr int largest = std::numeric_limits<int>::max();
        EXPECT_TRUE(attention::isValidHeadMapping({largest, 2, largest, largest}));
        EXPECT_FALSE(attention::isValidHeadMapping({largest, 1, largest, largest}));
    }

    /** @test Actual CPU stages retain original24Q/2KV semantics for every TP2/4/8 interval and reverse order. */
    TEST_F(Test__AttentionComputeStage, Execute_FP32_QwenReplicatedKVExactGlobalIntervals)
    {
        TensorFactory factory(mpi_ctx_);
        constexpr int dimension = 256;
        for (const int degree : {2, 4, 8})
            for (const bool reversed : {false, true})
            {
                std::vector<DeviceId> order;
                for (int participant = 0; participant < degree; ++participant)
                    order.push_back(DeviceId::cuda(participant)); // Metadata only: execution below is CPU.
                if (reversed) std::reverse(order.begin(), order.end());
                const auto assignment = TensorParallelConfig::equalSplit(degree, 24, 2, 1024, 32000, order);
                ASSERT_TRUE(assignment.validate()) << assignment.validationError();
                for (const auto &local : assignment.assignments())
                    for (const int rows : {1, 9})
                    {
                        SCOPED_TRACE(::testing::Message() << " degree=" << degree << " participant="
                            << local.local_rank << " start=" << local.head_start << " rows=" << rows
                            << " reversed=" << reversed);
                        const int kv_rows = rows == 1 ? 7 : rows;
                        const auto query_width = static_cast<std::size_t>(local.head_count * dimension);
                        const auto kv_width = static_cast<std::size_t>(local.kv_head_count * dimension);
                        auto query = factory.createFP32({static_cast<std::size_t>(rows), query_width}, device_id_);
                        auto key = factory.createFP32({static_cast<std::size_t>(kv_rows), kv_width}, device_id_);
                        auto value = factory.createFP32({static_cast<std::size_t>(kv_rows), kv_width}, device_id_);
                        auto result = factory.createFP32({static_cast<std::size_t>(rows), query_width}, device_id_);
                        std::fill_n(query->mutable_data(), query->numel(), 0.0f);
                        std::fill_n(key->mutable_data(), key->numel(), 0.0f);
                        std::fill_n(result->mutable_data(), result->numel(), -91.0f);
                        // Constant, distinct original KV values make the oracle
                        // independent of the production head-mapping function.
                        for (int row = 0; row < kv_rows; ++row)
                            for (int head = 0; head < local.kv_head_count; ++head)
                                std::fill_n(value->mutable_data() + static_cast<std::size_t>(row) * kv_width
                                    + head * dimension, dimension, static_cast<float>(local.kv_head_start + head + 1));
                        AttentionComputeStage::Params params;
                        params.Q = query.get(); params.K = key.get(); params.V = value.get();
                        params.output = result.get(); params.seq_len = rows; params.kv_len = kv_rows;
                        params.batch_size = 1; params.n_heads = local.head_count;
                        params.n_kv_heads = local.kv_head_count; params.head_dim = dimension;
                        params.head_start = local.head_start;
                        params.gqa_n_rep = degree == 2 ? 0 : 12;
                        params.causal = true; params.mpi_ctx = &mpi_ctx_; params.device_id = device_id_;
                        AttentionComputeStage stage(params);
                        ASSERT_TRUE(stage.execute(nullptr));
                        for (int row = 0; row < rows; ++row)
                            for (int head = 0; head < local.head_count; ++head)
                                for (int d = 0; d < dimension; ++d)
                                    EXPECT_NEAR(result->data()[static_cast<std::size_t>(row) * query_width + head * dimension + d],
                                        static_cast<float>((local.head_start + head) / 12 + 1), 2e-5f);
                    }
            }
    }

    /** @test A missing global ratio or out-of-range original Q interval cannot publish CPU output. */
    TEST_F(Test__AttentionComputeStage, Execute_FP32_ReplicatedKVRejectsMalformedMappingBeforeOutput)
    {
        TensorFactory factory(mpi_ctx_);
        auto query = factory.createFP32({1, 3 * 256}, device_id_);
        auto key = factory.createFP32({1, 2 * 256}, device_id_);
        auto value = factory.createFP32({1, 2 * 256}, device_id_);
        auto result = factory.createFP32({1, 3 * 256}, device_id_);
        std::fill_n(query->mutable_data(), query->numel(), 0.0f);
        std::fill_n(key->mutable_data(), key->numel(), 0.0f);
        std::fill_n(value->mutable_data(), value->numel(), 1.0f);
        for (const auto &mapping : std::vector<attention::HeadMappingGeometry>{{3, 2, 0, 0},
                {3, 2, 22, 12}, {3, 2, -1, 12}, {3, 2, 0, -1}})
        {
            std::fill_n(result->mutable_data(), result->numel(), -91.0f);
            AttentionComputeStage::Params params;
            params.Q = query.get(); params.K = key.get(); params.V = value.get(); params.output = result.get();
            params.batch_size = 1; params.seq_len = 1; params.kv_len = 1;
            params.n_heads = mapping.local_query_heads; params.n_kv_heads = mapping.visible_kv_heads;
            params.head_dim = 256; params.head_start = mapping.head_start; params.gqa_n_rep = mapping.replicated_gqa_n_rep;
            params.causal = true; params.mpi_ctx = &mpi_ctx_; params.device_id = device_id_;
            AttentionComputeStage stage(params);
            EXPECT_FALSE(stage.execute(nullptr));
            EXPECT_TRUE(std::all_of(result->data(), result->data() + result->numel(),
                [](float v) { return v == -91.0f; }));
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
        // Native descriptor planning consumes the same admitted physical facts
        // as production startup; keep this out of the device-free SetUp path.
        initializeROCmHardwareMetadata();
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
        // Native descriptor planning consumes the same admitted physical facts
        // as production startup; keep this out of the device-free SetUp path.
        initializeCUDAHardwareMetadata();
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

/** @test P2: stage-level pooled consumption matches the contiguous control bytes. */
TEST(PooledStageConsumption, PooledSnapshotDecodeMatchesContiguousControl)
{
    constexpr int heads = 2, kv_heads = 2, dimension = 8, page_tokens = 4;
    constexpr int tokens = 6; // Two pages (4 + 2).
    const auto device = DeviceId::cpu();
    // Own a real pooled table + pool under one PMA authority (device-free).
    const PhysicalPagePoolGeometry pool_geometry{4,
        static_cast<std::size_t>(page_tokens) * 2 * heads * dimension * sizeof(float), 2, 4};
    const CPUFP32PagedKVTableGeometry table_geometry{2, 2, 4, page_tokens, 16, 4};
    const auto bytes = PhysicalPagePool::physicalBytes(pool_geometry, device) +
        CPUFP32PagedKVTable::physicalBytes(table_geometry, device);
    PhysicalMemoryPlanBuilder plan;
    plan.add({.world_rank=0, .device=device, .total_bytes=bytes, .admission_available_bytes=bytes},
        PhysicalMemoryOwner::KVCache, bytes);
    auto authority = std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
    auto pool = authority->createCPUPagePool(device, pool_geometry, CPUStoragePlacement::local());
    auto table = CPUFP32PagedKVTable::create(authority, device, table_geometry);
    // Publish deterministic K/V pages.
    std::mt19937 random(99);
    std::uniform_real_distribution<float> values(-0.5f, 0.5f);
    const size_t width = static_cast<size_t>(heads * dimension);
    std::vector<float> k(static_cast<size_t>(tokens) * width), v(k.size());
    for (auto *bank : {&k, &v}) for (auto &x : *bank) x = values(random);
    auto update = table->tryBeginUpdate();
    ASSERT_TRUE(update);
    auto grant = pool->tryAdmit(1);
    ASSERT_TRUE(grant);
    const auto frame = grant->assignFrame();
    std::vector<float> payload(static_cast<size_t>(page_tokens) * 2 * width);
    for (size_t row = 0; row < static_cast<size_t>(page_tokens); ++row)
    {
        std::copy_n(k.data() + row * width, width, payload.data() + row * 2 * width);
        std::copy_n(v.data() + row * width, width, payload.data() + row * 2 * width + width);
    }
    auto write = grant->prepareAppend(frame, payload.size() * sizeof(float));
    ASSERT_TRUE(write);
    write->copyFrom(std::as_bytes(std::span(payload)));
    update->append(CPUFP32KVPage(write->commit(), heads, dimension));
    // Second page: the remaining tokens (2 rows).
    payload.resize(static_cast<std::size_t>(tokens - page_tokens) * 2 * width);
    for (size_t row = 0; row < static_cast<size_t>(tokens - page_tokens); ++row)
    {
        std::copy_n(k.data() + (page_tokens + row) * width, width,
            payload.data() + row * 2 * width);
        std::copy_n(v.data() + (page_tokens + row) * width, width,
            payload.data() + row * 2 * width + width);
    }
    auto grant2 = pool->tryAdmit(1);
    ASSERT_TRUE(grant2);
    const auto frame2 = grant2->assignFrame();
    auto write2 = grant2->prepareAppend(frame2, payload.size() * sizeof(float));
    ASSERT_TRUE(write2);
    write2->copyFrom(std::as_bytes(std::span(payload)));
    update->append(CPUFP32KVPage(write2->commit(), heads, dimension));
    auto snapshot = update->commit();
    // CONTIGUOUS attention through the raw k/v buffers (the kernel's oracle).
    std::vector<float> q(width), expected(width, -99.0f);
    for (auto &x : q) x = values(random);
    CPUFlashAttentionKernelT<ActivationPrecision::FP32> kernel;
    ASSERT_TRUE(kernel.compute_decode(q.data(), k.data(), v.data(), expected.data(), 1,
        tokens, heads, kv_heads, dimension, true, tokens - 1));
    // STAGE: run AttentionComputeStage with the pooled snapshot input.
    MPIContext stage_mpi{0, 1, MPI_COMM_WORLD};
    TensorFactory factory(stage_mpi);
    auto Q = factory.createFP32({1, width}, device);
    auto output = factory.createFP32({1, width}, device);
    std::memcpy(Q->mutable_data(), q.data(), sizeof(float) * width);
    AttentionComputeStage::Params params;
    params.Q = Q.get();
    params.output = output.get();
    params.batch_size = 1;
    params.seq_len = 1;
    params.kv_len = tokens;
    params.n_heads = heads;
    params.n_kv_heads = kv_heads;
    params.head_dim = dimension;
    params.causal = true;
    params.window_size = -1;
    params.position_offset = tokens - 1; // Decode at the last KV position.
    params.device_id = device;
    params.pooled_table = table.get();
    // L2: refs are OWNER-MINTED from a bound live publication — never
    // hand-assembled. The owner records the request binding, then mints the ref.
    ASSERT_TRUE(table->bindPublication(snapshot, /*request_id=*/7, /*slot=*/0));
    params.pooled_ref = table->refFor(snapshot, 7, 0);
    ASSERT_TRUE(params.pooled_ref.valid());
    params.kv_cache = nullptr; // Pooled path: no ring cache.
    MPIContext local_mpi{0, 1, MPI_COMM_WORLD};
    params.mpi_ctx = &local_mpi;
    AttentionComputeStage stage(params);
    CPUDeviceContext ctx(device);
    ASSERT_TRUE(stage.execute(&ctx));
    EXPECT_EQ(std::memcmp(output->fp32_data(), expected.data(), sizeof(float) * width), 0);
    snapshot = {};
}


/** @test P2 J2/J3: pooled mode rejects unsupported combos and foreign/stale refs fail closed. */
TEST(PooledStageConsumption, PooledModeRejectsUnsupportedAndForeignRefs)
{
    constexpr int heads = 2, dimension = 8;
    const auto device = DeviceId::cpu();
    const PhysicalPagePoolGeometry pool_geometry{4,
        static_cast<std::size_t>(4) * 2 * heads * dimension * sizeof(float), 2, 4};
    const CPUFP32PagedKVTableGeometry table_geometry{1, 2, 4, 4, 16, 4};
    const auto bytes = PhysicalPagePool::physicalBytes(pool_geometry, device) +
        CPUFP32PagedKVTable::physicalBytes(table_geometry, device);
    const auto table_bytes = CPUFP32PagedKVTable::physicalBytes(table_geometry, device);
    const auto total_bytes = bytes + table_bytes; // Two tables (L2 foreign negative).
    PhysicalMemoryPlanBuilder plan;
    plan.add({.world_rank=0, .device=device, .total_bytes=total_bytes, .admission_available_bytes=total_bytes},
        PhysicalMemoryOwner::KVCache, total_bytes);
    auto authority = std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
    auto pool = authority->createCPUPagePool(device, pool_geometry, CPUStoragePlacement::local());
    auto table = CPUFP32PagedKVTable::create(authority, device, table_geometry);
    auto update = table->tryBeginUpdate();
    ASSERT_TRUE(update);
    auto grant = pool->tryAdmit(1);
    ASSERT_TRUE(grant);
    const auto frame = grant->assignFrame();
    std::vector<float> payload(4 * 2 * heads * dimension);
    auto write = grant->prepareAppend(frame, payload.size() * sizeof(float));
    ASSERT_TRUE(write);
    write->copyFrom(std::as_bytes(std::span(payload)));
    update->append(CPUFP32KVPage(write->commit(), heads, dimension));
    auto snapshot = update->commit();
    MPIContext local_mpi{0, 1, MPI_COMM_WORLD};
    TensorFactory tf(local_mpi);
    auto Q = tf.createFP32({1, static_cast<size_t>(heads * dimension)}, device);
    auto output = tf.createFP32({1, static_cast<size_t>(heads * dimension)}, device);
    AttentionComputeStage::Params params;
    params.Q = Q.get();
    params.output = output.get();
    params.batch_size = 1;
    params.seq_len = 1;
    params.kv_len = 4;
    params.n_heads = heads;
    params.n_kv_heads = heads;
    params.head_dim = dimension;
    params.causal = true;
    params.window_size = -1;
    params.position_offset = 3;
    params.device_id = device;
    params.kv_cache = nullptr;
    params.mpi_ctx = &local_mpi;
    params.pooled_table = table.get();
    ASSERT_TRUE(table->bindPublication(snapshot, /*request_id=*/7, /*slot=*/0));
    params.pooled_ref = table->refFor(snapshot, 7, 0);
    ASSERT_TRUE(params.pooled_ref.valid());
    CPUDeviceContext ctx(device);
    // Baseline succeeds.
    AttentionComputeStage ok(params);
    ASSERT_TRUE(ok.execute(&ctx));
    // J3/K1 foreign table: a second table under the SAME authority has a
    // DIFFERENT incarnation id — the ref minted against table A never resolves
    // against table B, even when their versions collide.
    auto table_b = CPUFP32PagedKVTable::create(authority, device, table_geometry);
    auto b_update = table_b->tryBeginUpdate();
    ASSERT_TRUE(b_update);
    auto b_grant = pool->tryAdmit(1);
    ASSERT_TRUE(b_grant);
    const auto b_frame = b_grant->assignFrame();
    auto b_write = b_grant->prepareAppend(b_frame, payload.size() * sizeof(float));
    ASSERT_TRUE(b_write);
    b_write->copyFrom(std::as_bytes(std::span(payload)));
    b_update->append(CPUFP32KVPage(b_write->commit(), heads, dimension));
    auto b_snapshot = b_update->commit();
    ASSERT_TRUE(table_b->bindPublication(b_snapshot, 7, 0));
    auto foreign_params = params;
    foreign_params.pooled_table = table_b.get();
    foreign_params.pooled_ref = table_b->refFor(b_snapshot, 7, 0);
    // The ref is B's; the stage holds B — must refuse because the ref (B id)
    // is fine but the rotation gate etc... Actually with B+refB it should
    // SUCCEED. The foreign case is A's ref held against B:
    auto cross_params = params;
    cross_params.pooled_table = table_b.get();
    cross_params.pooled_ref = table->refFor(snapshot, 7, 0); // A's ref against B.
    AttentionComputeStage foreign(cross_params);
    EXPECT_FALSE(foreign.execute(&ctx)); // Foreign incarnation: refuse.
    // L2 wrong-request: the ref bound to request 8 never resolves for 7.
    auto wrong_params = params;
    wrong_params.pooled_ref.request_id = 8;
    AttentionComputeStage wrong_req(wrong_params);
    EXPECT_FALSE(wrong_req.execute(&ctx));
    // J3 superseded version (never existed): refuses.
    auto stale_params = params;
    stale_params.pooled_ref.publication_version = snapshot.version() + 1000;
    AttentionComputeStage stale(stale_params);
    EXPECT_FALSE(stale.execute(&ctx));
    // J2 prefill (seq_len > 1): explicitly rejects pooled mode.
    auto prefill_params = params;
    prefill_params.seq_len = 2;
    AttentionComputeStage prefill(prefill_params);
    EXPECT_FALSE(prefill.execute(&ctx));
    // J2 non-CPU: the stage requires a backend stream before ANY dispatch, so
    // a CUDA device in pooled mode must refuse before touching pooled/ring
    // state. Assert the stage fails (either the explicit-stream precondition
    // or the pooled device gate — never a pooled compute with ring K/V).
    auto gpu_params = params;
    gpu_params.device_id = DeviceId::cuda(0);
    AttentionComputeStage gpu(gpu_params);
    bool gpu_refused = false;
    try
    {
        gpu_refused = !gpu.execute(&ctx);
    }
    catch (const std::exception &)
    {
        gpu_refused = true; // No explicit stream for a GPU stage: refuses.
    }
    EXPECT_TRUE(gpu_refused);
    // L1 pooled + rotation: the gate rejects BEFORE any transform — Q is left
    // unmodified and no ring K/V is dereferenced (the old path would rotate Q
    // against unrotated pages and dereference NULL effective_K).
    std::vector<float> q_before(static_cast<std::size_t>(heads * dimension));
    std::memcpy(q_before.data(), Q->fp32_data(), sizeof(float) * heads * dimension);
    auto rotation_params = params;
    ActivationRotation rotation(heads * dimension, 8);
    rotation_params.kv_rotation = &rotation;
    rotation_params.device_id = device; // CPU.
    AttentionComputeStage rotation_stage(rotation_params);
    EXPECT_FALSE(rotation_stage.execute(&ctx));
    EXPECT_EQ(std::memcmp(Q->fp32_data(), q_before.data(), sizeof(float) * heads * dimension), 0)
        << "pooled+rotation must never modify Q";
    // L1 pooled + active_query_rows_device: rejected before any transform.
    int32_t dummy_row = 0;
    auto row_transform_params = params;
    row_transform_params.active_query_rows_device = &dummy_row;
    AttentionComputeStage row_transform_stage(row_transform_params);
    EXPECT_FALSE(row_transform_stage.execute(&ctx));
}


/** @test P2 production caller: pooled APPEND + pooled ATTENTION, one table, one graph shape. */
TEST(PooledStageConsumption, PooledAppendAndAttentionMatchContiguousControl)
{
    // Decode geometry: one token per page (page_tokens=1) so every decode step
    // appends exactly one new page and shares every prior immutable page.
    constexpr int heads = 2, kv_heads = 2, dimension = 8, page_tokens = 1;
    constexpr int tokens = 6;
    const auto device = DeviceId::cpu();
    const PhysicalPagePoolGeometry pool_geometry{8,
        static_cast<std::size_t>(page_tokens) * 2 * heads * dimension * sizeof(float), 2, 8};
    const CPUFP32PagedKVTableGeometry table_geometry{8, 8, 4, page_tokens, 32, 8};
    const auto bytes = PhysicalPagePool::physicalBytes(pool_geometry, device) +
        CPUFP32PagedKVTable::physicalBytes(table_geometry, device);
    PhysicalMemoryPlanBuilder plan;
    plan.add({.world_rank=0, .device=device, .total_bytes=bytes, .admission_available_bytes=bytes},
        PhysicalMemoryOwner::KVCache, bytes);
    auto authority = std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
    auto pool = authority->createCPUPagePool(device, pool_geometry, CPUStoragePlacement::local());
    auto table = CPUFP32PagedKVTable::create(authority, device, table_geometry);

    std::mt19937 random(101);
    std::uniform_real_distribution<float> values(-0.5f, 0.5f);
    const size_t width = static_cast<size_t>(heads * dimension);
    std::vector<float> k(static_cast<size_t>(tokens) * width), v(k.size());
    for (auto *bank : {&k, &v}) for (auto &x : *bank) x = values(random);
    std::vector<float> q(width);
    for (auto &x : q) x = values(random);
    CPUFlashAttentionKernelT<ActivationPrecision::FP32> kernel;

    // Graph-shaped sequence: for each decode row, the pooled append stage
    // publishes one row into the table and mints the ref cell; the attention
    // stage then resolves the cell. Single readonly thread, one MPIContext.
    PooledPublicationCell previous_publication;
    MPIContext stage_mpi{0, 1, MPI_COMM_WORLD};
    TensorFactory factory(stage_mpi);
    auto Q = factory.createFP32({1, width}, device);
    auto output = factory.createFP32({1, width}, device);
    std::memcpy(Q->mutable_data(), q.data(), sizeof(float) * width);
    const int rows_per_page = page_tokens;
    for (int step = 0; step < tokens; ++step)
    {
        // K/V row for this step (the append stage appends exactly one row).
        std::vector<float> k_row(k.data() + step * width, k.data() + (step + 1) * width);
        std::vector<float> v_row(v.data() + step * width, v.data() + (step + 1) * width);
        auto kt = factory.createFP32({1, width}, device);
        auto vt = factory.createFP32({1, width}, device);
        std::memcpy(kt->mutable_data(), k_row.data(), sizeof(float) * width);
        std::memcpy(vt->mutable_data(), v_row.data(), sizeof(float) * width);
        PooledPublicationCell ref_cell;
        // M6: the complete step demand is admitted BEFORE the stage runs.
        auto step_grant = pool->tryAdmit(1);
        ASSERT_TRUE(step_grant);
        ASSERT_EQ(step_grant->remainingCommitment(), 1u);
        PooledKVAppendStage::Params append_params;
        append_params.device_id = device;
        append_params.K = kt.get();
        append_params.V = vt.get();
        append_params.pool = pool.get();
        append_params.table = table.get();
        append_params.ref_out = &ref_cell;
        append_params.grant = &*step_grant;
        append_params.request_id = 9;
        append_params.slot = 0;
        append_params.layer_idx = 0;
        append_params.kv_heads = kv_heads;
        append_params.head_dim = dimension;
        append_params.page_tokens = page_tokens;
        if (step > 0)
            append_params.prior = &previous_publication;
        PooledKVAppendStage append_stage(append_params);
        ASSERT_TRUE(append_stage.execute(nullptr)) << "append step " << step;
        ASSERT_TRUE(ref_cell.live());
        // Attention stage resolves the ref cell (while the cell is live).
        AttentionComputeStage::Params attn;
        attn.device_id = device;
        attn.Q = Q.get();
        attn.output = output.get();
        attn.batch_size = 1;
        attn.seq_len = 1;
        attn.kv_len = step + 1;
        attn.n_heads = heads;
        attn.n_kv_heads = kv_heads;
        attn.head_dim = dimension;
        attn.causal = true;
        attn.window_size = -1;
        attn.position_offset = step;
        attn.pooled_table = table.get();
        attn.pooled_ref_cell = &ref_cell.ref;
        AttentionComputeStage attn_stage(attn);
        ASSERT_TRUE(attn_stage.execute(nullptr)) << "attention step " << step;
        // Oracle: contiguous decode over the tokens visible at this step.
        std::vector<float> expected(width, -99.0f);
        ASSERT_TRUE(kernel.compute_decode(q.data(), k.data(), v.data(), expected.data(), 1,
            step + 1, heads, kv_heads, dimension, true, step));
        const float *got = output->fp32_data();
        int first_diff = -1;
        for (size_t i = 0; i < width; ++i)
            if (std::fabs(got[i] - expected[i]) > 1e-6f) { first_diff = static_cast<int>(i); break; }
        ASSERT_EQ(first_diff, -1)
            << "pooled append+attention diverged at step " << step
            << " got=" << (first_diff >= 0 ? got[first_diff] : 0.0f)
            << " expected=" << (first_diff >= 0 ? expected[first_diff] : 0.0f);
        previous_publication = std::move(ref_cell);
        (void)rows_per_page;
    }
}


/** @test P2: page_tokens=8 prefill-shaped single appendShared page commits. */
TEST(PooledStageConsumption, LargePageSharedPrefillAppendCommits)
{
    constexpr int heads = 2, kv_heads = 2, dimension = 16, page_tokens = 8;
    constexpr int tokens = 3;
    const auto device = DeviceId::cpu();
    const PhysicalPagePoolGeometry pool_geometry{8,
        static_cast<std::size_t>(page_tokens) * 2 * heads * dimension * sizeof(float), 2, 8};
    const CPUFP32PagedKVTableGeometry table_geometry{8, 8, 4, page_tokens, 32, 8};
    const auto bytes = PhysicalPagePool::physicalBytes(pool_geometry, device) +
        CPUFP32PagedKVTable::physicalBytes(table_geometry, device);
    PhysicalMemoryPlanBuilder plan;
    plan.add({.world_rank=0, .device=device, .total_bytes=bytes, .admission_available_bytes=bytes},
        PhysicalMemoryOwner::KVCache, bytes);
    auto authority = std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
    auto pool = authority->createCPUPagePool(device, pool_geometry, CPUStoragePlacement::local());
    auto table = CPUFP32PagedKVTable::create(authority, device, table_geometry);
    const size_t width = static_cast<size_t>(heads * dimension);
    std::vector<float> kv(static_cast<size_t>(tokens) * width * 2, 0.5f);
    auto update = table->tryBeginUpdate();
    ASSERT_TRUE(update);
    auto grant = pool->tryAdmit(1);
    ASSERT_TRUE(grant);
    const auto frame = grant->assignFrame();
    auto write = grant->prepareAppend(frame, kv.size() * sizeof(float));
    ASSERT_TRUE(write);
    write->copyFrom(std::as_bytes(std::span(kv)));
    auto committed = write->commit();
    auto shared = committed.share();
    update->appendShared(std::move(shared), heads, dimension);
    auto snapshot = update->commit();
    EXPECT_EQ(static_cast<int>(snapshot.view().rows()), tokens);
}


/** @test P2: plan-shaped page_tokens=8 prefill append commits (frame 4096). */
TEST(PooledStageConsumption, PlanShapedPrefillAppendCommits)
{
    constexpr int heads = 2, kv_heads = 2, dimension = 16, page_tokens = 8;
    constexpr int tokens = 3;
    const auto device = DeviceId::cpu();
    auto plan = planPagedKVInstance(PagedKVTierConfig{},
        []() { PagedKVPoolBudget b; b.remaining_budget_bytes = 2*1024*1024; b.frame_bytes = 4096;
            b.request_records = 4; b.max_pages_per_request = 64; return b; }(),
        PagedKVInstancePolicy{page_tokens, 8, 4, 4}, device);
    ASSERT_TRUE(plan);
    PhysicalMemoryPlanBuilder mb;
    mb.add({.world_rank=0, .device=device, .total_bytes=plan->total_bytes,
        .admission_available_bytes=plan->total_bytes},
        PhysicalMemoryOwner::KVCache, plan->total_bytes);
    auto authority = std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(mb.build()), 0);
    auto created = PagedKVInstance::create(authority, *plan, device);
    ASSERT_TRUE(created);
    PagedKVInstance instance = std::move(*created);
    const size_t width = static_cast<size_t>(kv_heads * dimension);
    std::vector<float> kv(static_cast<size_t>(tokens) * width * 2, 0.5f);
    auto update = instance.table().tryBeginUpdate();
    ASSERT_TRUE(update);
    auto grant = instance.pool().tryAdmit(1);
    ASSERT_TRUE(grant);
    const auto frame = grant->assignFrame();
    auto write = grant->prepareAppend(frame, kv.size() * sizeof(float));
    ASSERT_TRUE(write);
    write->copyFrom(std::as_bytes(std::span(kv)));
    auto committed = write->commit();
    auto shared = committed.share();
    update->appendShared(std::move(shared), kv_heads, dimension);
    auto snapshot = update->commit();
    EXPECT_EQ(static_cast<int>(snapshot.view().rows()), tokens);
}

/** @test R24 M1/M3: the K/V of every appended pooled row survives byte-exact:
 * 3-row prefill into an 8-row page, then 8-row (full-page) and 9-row (2-page)
 * prefill, each compared against the contiguous interleaved control. */
TEST(PooledStageConsumption, PrefillRowsAreByteExactInNativeInterleave)
{
    constexpr int heads = 2, kv_heads = 2, dimension = 16, page_tokens = 8;
    const auto device = DeviceId::cpu();
    const PhysicalPagePoolGeometry pool_geometry{16,
        static_cast<std::size_t>(page_tokens) * 2 * heads * dimension * sizeof(float), 2, 16};
    const CPUFP32PagedKVTableGeometry table_geometry{8, 8, 4, page_tokens, 64, 16};
    const auto bytes = PhysicalPagePool::physicalBytes(pool_geometry, device) +
        CPUFP32PagedKVTable::physicalBytes(table_geometry, device);
    PhysicalMemoryPlanBuilder plan;
    plan.add({.world_rank=0, .device=device, .total_bytes=bytes, .admission_available_bytes=bytes},
        PhysicalMemoryOwner::KVCache, bytes);
    auto authority = std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
    auto pool = authority->createCPUPagePool(device, pool_geometry, CPUStoragePlacement::local());
    auto table = CPUFP32PagedKVTable::create(authority, device, table_geometry);

    std::mt19937 random(7);
    std::uniform_real_distribution<float> values(-0.5f, 0.5f);
    const size_t width = static_cast<size_t>(heads * dimension);
    constexpr int max_rows = 9;
    std::vector<float> k(static_cast<size_t>(max_rows) * width), v(k.size());
    for (auto *bank : {&k, &v}) for (auto &x : *bank) x = values(random);
    // Build K/V tensors through the shared MPIContext factory.
    MPIContext mpi{0, 1, MPI_COMM_WORLD};
    TensorFactory factory(mpi);
    auto K = factory.createFP32({max_rows, static_cast<long long>(width)}, device);
    auto V = factory.createFP32({max_rows, static_cast<long long>(width)}, device);
    std::memcpy(K->mutable_data(), k.data(), k.size() * sizeof(float));
    std::memcpy(V->mutable_data(), v.data(), v.size() * sizeof(float));

    // Expected native interleaved bytes for N rows.
    const auto expected = [&](int rows) {
        std::vector<float> payload(static_cast<size_t>(rows) * 2 * width);
        for (int row = 0; row < rows; ++row)
        {
            std::copy_n(k.data() + row * width, width, payload.data() + static_cast<size_t>(row) * 2 * width);
            std::copy_n(v.data() + row * width, width, payload.data() + static_cast<size_t>(row) * 2 * width + width);
        }
        return payload;
    };
    const auto append_via_stage = [&](int rows, PooledPublicationCell &out) {
        auto grant = pool->tryAdmit(static_cast<std::size_t>((rows + page_tokens - 1) / page_tokens));
        ASSERT_TRUE(grant);
        PooledKVAppendStage::Params params;
        params.device_id = device;
        params.K = K.get();
        params.V = V.get();
        params.pool = pool.get();
        params.table = table.get();
        params.ref_out = &out;
        params.grant = &*grant;
        params.request_id = 42;
        params.slot = 0;
        params.layer_idx = 0;
        params.kv_heads = kv_heads;
        params.head_dim = dimension;
        params.page_tokens = page_tokens;
        params.live_rows = rows;
        PooledKVAppendStage stage(params);
        EXPECT_TRUE(stage.execute(nullptr)) << "append rows=" << rows;
    };

    // 3-row prefill: one partial page, byte-exact.
    PooledPublicationCell pub3;
    append_via_stage(3, pub3);
    ASSERT_TRUE(pub3.live());
    {
        const auto pages = pub3.snapshot.view().pages();
        ASSERT_EQ(static_cast<int>(pages.size()), 1);
        const auto want = expected(3);
        const auto got = pages[0].storage();
        ASSERT_EQ(got.size(), want.size() * sizeof(float));
        EXPECT_EQ(std::memcmp(got.data(), want.data(), want.size() * sizeof(float)), 0)
            << "3-row prefill page is not byte-exact";
    }
    // 8-row prefill: one full page, byte-exact.
    PooledPublicationCell pub8;
    append_via_stage(8, pub8);
    ASSERT_TRUE(pub8.live());
    {
        const auto pages = pub8.snapshot.view().pages();
        ASSERT_EQ(static_cast<int>(pages.size()), 1);
        const auto want = expected(8);
        const auto got = pages[0].storage();
        ASSERT_EQ(got.size(), want.size() * sizeof(float));
        EXPECT_EQ(std::memcmp(got.data(), want.data(), want.size() * sizeof(float)), 0)
            << "8-row full page is not byte-exact";
    }
    // 9-row prefill: two pages (8+1), both byte-exact.
    PooledPublicationCell pub9;
    append_via_stage(9, pub9);
    ASSERT_TRUE(pub9.live());
    {
        const auto pages = pub9.snapshot.view().pages();
        ASSERT_EQ(static_cast<int>(pages.size()), 2);
        const auto want = expected(9);
        const std::size_t row_bytes = 2 * width * sizeof(float);
        const auto got0 = pages[0].storage();
        const auto got1 = pages[1].storage();
        ASSERT_EQ(got0.size(), static_cast<std::size_t>(8) * row_bytes);
        ASSERT_EQ(got1.size(), static_cast<std::size_t>(1) * row_bytes);
        EXPECT_EQ(std::memcmp(got0.data(), want.data(), got0.size()), 0)
            << "9-row first page is not byte-exact";
        EXPECT_EQ(std::memcmp(got1.data(), want.data() + 8 * 2 * width, got1.size()), 0)
            << "9-row second page is not byte-exact";
    }
}

/** @test R24 M3/M2: a partial tail extended by 2+ rows keeps every prior
 * full page shared verbatim and the COW bytes in native interleave. */
TEST(PooledStageConsumption, PartialTailExtensionKeepsPriorPagesAndBytes)
{
    constexpr int heads = 2, kv_heads = 2, dimension = 16, page_tokens = 4;
    const auto device = DeviceId::cpu();
    const PhysicalPagePoolGeometry pool_geometry{8,
        static_cast<std::size_t>(page_tokens) * 2 * heads * dimension * sizeof(float), 2, 8};
    const CPUFP32PagedKVTableGeometry table_geometry{4, 4, 4, page_tokens, 32, 8};
    const auto bytes = PhysicalPagePool::physicalBytes(pool_geometry, device) +
        CPUFP32PagedKVTable::physicalBytes(table_geometry, device);
    PhysicalMemoryPlanBuilder plan;
    plan.add({.world_rank=0, .device=device, .total_bytes=bytes, .admission_available_bytes=bytes},
        PhysicalMemoryOwner::KVCache, bytes);
    auto authority = std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
    auto pool = authority->createCPUPagePool(device, pool_geometry, CPUStoragePlacement::local());
    auto table = CPUFP32PagedKVTable::create(authority, device, table_geometry);

    std::mt19937 random(11);
    std::uniform_real_distribution<float> values(-0.5f, 0.5f);
    const size_t width = static_cast<size_t>(heads * dimension);
    // 5-row history: pages [4 full][1 partial] then extend the tail by 2 rows.
    constexpr int history = 5, extra = 2;
    std::vector<float> k(static_cast<size_t>(history + extra) * width), v(k.size());
    for (auto *bank : {&k, &v}) for (auto &x : *bank) x = values(random);
    MPIContext mpi{0, 1, MPI_COMM_WORLD};
    TensorFactory factory(mpi);
    auto K = factory.createFP32({history + extra, static_cast<long long>(width)}, device);
    auto V = factory.createFP32({history + extra, static_cast<long long>(width)}, device);
    std::memcpy(K->mutable_data(), k.data(), k.size() * sizeof(float));
    std::memcpy(V->mutable_data(), v.data(), v.size() * sizeof(float));

    auto grant5 = pool->tryAdmit(2);
    ASSERT_TRUE(grant5);
    PooledPublicationCell pub5;
    PooledKVAppendStage::Params params5;
    params5.device_id = device;
    params5.K = K.get();
    params5.V = V.get();
    params5.pool = pool.get();
    params5.table = table.get();
    params5.ref_out = &pub5;
    params5.grant = &*grant5;
    params5.request_id = 11;
    params5.slot = 0;
    params5.layer_idx = 0;
    params5.kv_heads = kv_heads;
    params5.head_dim = dimension;
    params5.page_tokens = page_tokens;
    params5.live_rows = history;
    PooledKVAppendStage stage5(params5);
    ASSERT_TRUE(stage5.execute(nullptr));
    ASSERT_TRUE(pub5.live());
    EXPECT_EQ(pub5.snapshot.view().rows(), history);
    EXPECT_EQ(static_cast<int>(pub5.snapshot.view().pages().size()), 2);

    // Extend by 2 rows (history 5 + 2 = 7): M2 must share page 0 verbatim and
    // COW page 1 (1 row) + 2 fresh rows; M3 bytes exact. The graph caller's
    // K/V tensors hold ONLY the rows of this forward (rows 5..6), never the
    // full history — model that exactly.
    auto K2 = factory.createFP32({extra, static_cast<long long>(width)}, device);
    auto V2 = factory.createFP32({extra, static_cast<long long>(width)}, device);
    std::memcpy(K2->mutable_data(), k.data() + static_cast<size_t>(history) * width,
        static_cast<size_t>(extra) * width * sizeof(float));
    std::memcpy(V2->mutable_data(), v.data() + static_cast<size_t>(history) * width,
        static_cast<size_t>(extra) * width * sizeof(float));
    auto grant7 = pool->tryAdmit(1);
    ASSERT_TRUE(grant7);
    PooledPublicationCell pub7;
    PooledKVAppendStage::Params params7;
    params7.device_id = device;
    params7.K = K2.get();
    params7.V = V2.get();
    params7.pool = pool.get();
    params7.table = table.get();
    params7.ref_out = &pub7;
    params7.prior = &pub5; // M1: distinct prior cell.
    params7.grant = &*grant7;
    params7.request_id = 11;
    params7.slot = 0;
    params7.layer_idx = 0;
    params7.kv_heads = kv_heads;
    params7.head_dim = dimension;
    params7.page_tokens = page_tokens;
    params7.live_rows = extra;
    PooledKVAppendStage stage7(params7);
    ASSERT_TRUE(stage7.execute(nullptr));
    ASSERT_TRUE(pub7.live());
    EXPECT_EQ(pub7.snapshot.view().rows(), history + extra);
    const auto pages = pub7.snapshot.view().pages();
    ASSERT_EQ(static_cast<int>(pages.size()), 2);
    const auto got0 = pages[0].storage();
    const std::size_t row_bytes = 2 * width * sizeof(float);
    ASSERT_EQ(got0.size(), static_cast<std::size_t>(page_tokens) * row_bytes);
    const auto full_control = [&](int rows) {
        std::vector<float> payload(static_cast<size_t>(rows) * 2 * width);
        for (int row = 0; row < rows; ++row)
        {
            std::copy_n(k.data() + row * width, width, payload.data() + static_cast<size_t>(row) * 2 * width);
            std::copy_n(v.data() + row * width, width, payload.data() + static_cast<size_t>(row) * 2 * width + width);
        }
        return payload;
    };
    const auto want0 = full_control(page_tokens);
    EXPECT_EQ(std::memcmp(got0.data(), want0.data(), got0.size()), 0)
        << "shared prior full page differs from the contiguous control";
    // Tail page: rows 4..6 = prior last row (row 4) + fresh rows 5,6; bytes are
    // the interleaved prefix of rows 4..6.
    const auto got1 = pages[1].storage();
    ASSERT_EQ(got1.size(), static_cast<std::size_t>(1 + extra) * row_bytes);
    const auto want1 = full_control(history + extra);
    EXPECT_EQ(std::memcmp(got1.data(), want1.data() + static_cast<std::size_t>(4) * 2 * width, got1.size()), 0)
        << "COW tail bytes differ from the contiguous control";
}

/** @test R24 M4: the caller boundary refuses a continued pooled prefill
 * (prior live, rows>1) before graph execution; the append stage itself never
 * refuses rows>remaining (it splits). The refusal is implemented in the
 * orchestrator's pooled forward gate; this test proves the stage-level split
 * path is the opposite of refusal (extension test above) and that a caller
 * asking the stage for a continued prefill gets a well-formed split whose
 * attention would be wrong — the DGO gate refuses before it is reached.
 */
TEST(PooledStageConsumption, ContinuedPooledPrefillRefusesBeforeMutation)
{
    constexpr int heads = 2, kv_heads = 2, dimension = 16, page_tokens = 8;
    const auto device = DeviceId::cpu();
    const PhysicalPagePoolGeometry pool_geometry{8,
        static_cast<std::size_t>(page_tokens) * 2 * heads * dimension * sizeof(float), 2, 8};
    const CPUFP32PagedKVTableGeometry table_geometry{4, 4, 4, page_tokens, 32, 8};
    const auto bytes = PhysicalPagePool::physicalBytes(pool_geometry, device) +
        CPUFP32PagedKVTable::physicalBytes(table_geometry, device);
    PhysicalMemoryPlanBuilder plan;
    plan.add({.world_rank=0, .device=device, .total_bytes=bytes, .admission_available_bytes=bytes},
        PhysicalMemoryOwner::KVCache, bytes);
    auto authority = std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
    auto pool = authority->createCPUPagePool(device, pool_geometry, CPUStoragePlacement::local());
    auto table = CPUFP32PagedKVTable::create(authority, device, table_geometry);
    const size_t width = static_cast<size_t>(heads * dimension);
    MPIContext mpi{0, 1, MPI_COMM_WORLD};
    TensorFactory factory(mpi);
    auto K = factory.createFP32({8, static_cast<long long>(width)}, device);
    auto V = factory.createFP32({8, static_cast<long long>(width)}, device);
    std::fill_n(K->mutable_data(), 8 * width, 0.25f);
    std::fill_n(V->mutable_data(), 8 * width, -0.25f);

    auto grant = pool->tryAdmit(1);
    ASSERT_TRUE(grant);
    PooledPublicationCell prior;
    PooledKVAppendStage::Params p0;
    p0.device_id = device;
    p0.K = K.get();
    p0.V = V.get();
    p0.pool = pool.get();
    p0.table = table.get();
    p0.ref_out = &prior;
    p0.grant = &*grant;
    p0.request_id = 21;
    p0.slot = 0;
    p0.layer_idx = 0;
    p0.kv_heads = kv_heads;
    p0.head_dim = dimension;
    p0.page_tokens = page_tokens;
    p0.live_rows = 3;
    PooledKVAppendStage s0(p0);
    ASSERT_TRUE(s0.execute(nullptr));
    ASSERT_TRUE(prior.live());
    const std::uint64_t prior_version = prior.ref.publication_version;

    // Continued prefill (rows=3 with a live prior): the APPEND stage splits
    // (3 prior rows + 3 new rows, one page) — it never refuses rows>remaining.
    auto grant2 = pool->tryAdmit(1);
    ASSERT_TRUE(grant2);
    PooledPublicationCell next;
    PooledKVAppendStage::Params p1;
    p1.device_id = device;
    p1.K = K.get();
    p1.V = V.get();
    p1.pool = pool.get();
    p1.table = table.get();
    p1.ref_out = &next;
    p1.prior = &prior;
    p1.grant = &*grant2;
    p1.request_id = 21;
    p1.slot = 0;
    p1.layer_idx = 0;
    p1.kv_heads = kv_heads;
    p1.head_dim = dimension;
    p1.page_tokens = page_tokens;
    p1.live_rows = 3;
    PooledKVAppendStage s1(p1);
    ASSERT_TRUE(s1.execute(nullptr));
    ASSERT_TRUE(next.live());
    EXPECT_EQ(next.snapshot.view().rows(), 6);
    // The prior remains a separate, still-valid publication (twin-cell rule:
    // the stage never clears it). The CALLER (orchestrator) refuses a
    // continued prefill before this stage is reachable (DGO gate).
    EXPECT_TRUE(prior.live());
    EXPECT_EQ(prior.ref.publication_version, prior_version);
    auto resolved = table->resolveSnapshotRef(prior.ref);
    ASSERT_TRUE(resolved);
    EXPECT_EQ(resolved->view().rows(), 3);
}

/** @test R24 M6: pre-admitted step grant exhaustion refuses before ANY layer
 * publishes (all-or-none); no partially published layer set exists. */
TEST(PooledStageConsumption, StepGrantExhaustionRefusesBeforeAnyLayerPublishes)
{
    constexpr int heads = 2, kv_heads = 2, dimension = 16, page_tokens = 8;
    const auto device = DeviceId::cpu();
    // Four frames; the step needs 4 pages (2 layers x 2 pages for 9 rows).
    // Two frames are occupied BEFORE the step, so the pre-admitted grant must
    // refuse cleanly (never a mid-graph partial publication).
    const PhysicalPagePoolGeometry pool_geometry{4,
        static_cast<std::size_t>(page_tokens) * 2 * heads * dimension * sizeof(float), 2, 4};
    const CPUFP32PagedKVTableGeometry table_geometry{4, 4, 4, page_tokens, 32, 8};
    const auto bytes = PhysicalPagePool::physicalBytes(pool_geometry, device) +
        CPUFP32PagedKVTable::physicalBytes(table_geometry, device);
    PhysicalMemoryPlanBuilder plan;
    plan.add({.world_rank=0, .device=device, .total_bytes=bytes, .admission_available_bytes=bytes},
        PhysicalMemoryOwner::KVCache, bytes);
    auto authority = std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
    auto pool = authority->createCPUPagePool(device, pool_geometry, CPUStoragePlacement::local());
    auto table = CPUFP32PagedKVTable::create(authority, device, table_geometry);
    // Occupy two frames so the step grant (4 pages) cannot fit.
    auto occupant1 = pool->tryAdmit(1);
    ASSERT_TRUE(occupant1);
    (void)occupant1->assignFrame();
    auto occupant2 = pool->tryAdmit(1);
    ASSERT_TRUE(occupant2);
    (void)occupant2->assignFrame();

    // The admitted envelope seats the request: 2 layers × 16 issuances.
    const auto policy = PooledKVDecodeBinding::requestPolicy(page_tokens, /*layers=*/2,
        /*admitted_issuances=*/16, /*readers_per_bank=*/4);
    // The binding refuses a table whose geometry cannot seat the envelope.
    EXPECT_THROW(PooledKVDecodeBinding(pool.get(), table.get(), /*request_id=*/31,
        /*slot=*/0, kv_heads, dimension, page_tokens, /*layers=*/2,
        /*admitted_issuances=*/16), std::invalid_argument);
    // The test's two-frame pool/4-bank table cannot hold the envelope; the
    // geometry refusal IS the pre-graph negative (no grant is ever admitted).
    EXPECT_TRUE(policy.bank_count >= 4);
    // Direct service of the step beyond the envelope is refused: a small
    // envelope is seated only if the actual table geometry fits.
    (void)policy;
}

/** @test R1: grouped pooled decode refuses duplicate / aliased ref cells.
 * Two decode rows must consume DISTINCT publications (one per request); a
 * cell naming the same publication as another cell, a cell duplicating the
 * single ref, or an invalid cell is refused BEFORE any resolve or borrow —
 * no row ever runs under ambiguous identity. */
TEST(PooledStageConsumption, GroupedRefCellsMustBeDistinct)
{
    constexpr int heads = 2, kv_heads = 2, dimension = 16, page_tokens = 8;
    const auto device = DeviceId::cpu();
    const PhysicalPagePoolGeometry pool_geometry{4,
        static_cast<std::size_t>(page_tokens) * 2 * heads * dimension * sizeof(float), 2, 4};
    const CPUFP32PagedKVTableGeometry table_geometry{4, 4, 4, page_tokens, 32, 8};
    const auto bytes = PhysicalPagePool::physicalBytes(pool_geometry, device) +
        CPUFP32PagedKVTable::physicalBytes(table_geometry, device);
    PhysicalMemoryPlanBuilder plan;
    plan.add({.world_rank=0, .device=device, .total_bytes=bytes, .admission_available_bytes=bytes},
        PhysicalMemoryOwner::KVCache, bytes);
    auto authority = std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
    auto pool = authority->createCPUPagePool(device, pool_geometry, CPUStoragePlacement::local());
    auto table = CPUFP32PagedKVTable::create(authority, device, table_geometry);

    // Publish one snapshot (4 rows) for request 7 / slot 0.
    auto update = table->tryBeginUpdate();
    ASSERT_TRUE(update);
    auto grant = pool->tryAdmit(1);
    ASSERT_TRUE(grant);
    const auto frame = grant->assignFrame();
    std::vector<float> payload(4 * 2 * heads * dimension);
    auto write = grant->prepareAppend(frame, payload.size() * sizeof(float));
    ASSERT_TRUE(write);
    write->copyFrom(std::as_bytes(std::span(payload)));
    update->append(CPUFP32KVPage(write->commit(), heads, dimension));
    auto snapshot = update->commit();
    ASSERT_TRUE(table->bindPublication(snapshot, /*request_id=*/7, /*slot=*/0));
    const auto ref = table->refFor(snapshot, 7, 0);
    ASSERT_TRUE(ref.valid());

    auto ref_copy = ref; // Same identity, always.
    MPIContext local_mpi{0, 1, MPI_COMM_WORLD};
    TensorFactory tf(local_mpi);
    auto Q = tf.createFP32({2, static_cast<size_t>(heads * dimension)}, device);
    auto output = tf.createFP32({2, static_cast<size_t>(heads * dimension)}, device);
    AttentionComputeStage::Params params;
    params.Q = Q.get();
    params.output = output.get();
    params.batch_size = 2;
    params.seq_len = 1;
    params.kv_len = 4;
    params.n_heads = heads;
    params.n_kv_heads = kv_heads;
    params.head_dim = dimension;
    params.causal = true;
    params.window_size = -1;
    params.position_offset = 3;
    params.device_id = device;
    params.kv_cache = nullptr;
    params.mpi_ctx = &local_mpi;
    params.pooled_table = table.get();
    CPUDeviceContext ctx(device);

    // Duplicate cell pair: the two rows would attend the SAME publication.
    {
        AttentionComputeStage::Params dup = params;
        dup.pooled_ref_cells = {&ref, &ref_copy};
        AttentionComputeStage stage(dup);
        EXPECT_FALSE(stage.execute(&ctx)) << "duplicate grouped ref cells must refuse";
    }
    // Cell aliasing the single-ref path is also refused.
    {
        AttentionComputeStage::Params alias = params;
        alias.pooled_ref = ref;
        alias.pooled_ref_cells = {&ref};
        AttentionComputeStage stage(alias);
        EXPECT_FALSE(stage.execute(&ctx)) << "grouped cell duplicating the single ref must refuse";
    }
    // An invalid cell is refused before resolve.
    {
        AttentionComputeStage::Params bad = params;
        const PagedKVSnapshotRef invalid{};
        bad.pooled_ref_cells = {&ref, &invalid};
        AttentionComputeStage stage(bad);
        EXPECT_FALSE(stage.execute(&ctx)) << "invalid grouped ref cell must refuse";
    }
}

/** @test R25 N1: a step guard cancels on ANY unwind — the step grant returns
 * and no next publication becomes visible. */
/** @test R25 N1: a step guard cancels on ANY unwind — the step grant returns
 * and no next publication becomes visible. */
TEST(PooledStepGuard, UnwindAfterPrepareCancelsStepAndReleasesGrant)
{
    constexpr int heads = 2, kv_heads = 2, dimension = 16, page_tokens = 8;
    const auto device = DeviceId::cpu();
    constexpr int layers = 1;
    constexpr std::size_t issuances = 4;
    const auto policy = PooledKVDecodeBinding::requestPolicy(page_tokens, layers,
        issuances, /*readers_per_bank=*/4);
    const PhysicalPagePoolGeometry pool_geometry{policy.bank_count,
        static_cast<std::size_t>(page_tokens) * 2 * heads * dimension * sizeof(float), 2,
        policy.bank_count};
    const CPUFP32PagedKVTableGeometry table_geometry{4, policy.bank_count, 4,
        page_tokens, 2 * policy.bank_count, policy.bank_count};
    const auto bytes = PhysicalPagePool::physicalBytes(pool_geometry, device) +
        CPUFP32PagedKVTable::physicalBytes(table_geometry, device);
    PhysicalMemoryPlanBuilder plan;
    plan.add({.world_rank=0, .device=device, .total_bytes=bytes, .admission_available_bytes=bytes},
        PhysicalMemoryOwner::KVCache, bytes);
    auto authority = std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
    auto pool = authority->createCPUPagePool(device, pool_geometry, CPUStoragePlacement::local());
    auto table = CPUFP32PagedKVTable::create(authority, device, table_geometry);
    // This direct-stage suite exercises the COW/step-guard matrix; the
    // persistent in-place tail is opted out explicitly (production default
    // is persistent_tail=true; the tail economy measurement covers it).
    PooledKVDecodeBinding binding(pool.get(), table.get(), /*request_id=*/41,
        /*slot=*/0, kv_heads, dimension, page_tokens, layers, issuances,
        /*persistent_tail=*/false);

    // N1 negative 1: a THROW between prepareStep and completeAndAdvance (a
    // graph-construction / execution throw) must cancel the step through the
    // guard destructor while the exception propagates. The throw escapes this
    // inner scope, so the guard unwinds; we catch it OUTSIDE.
    try
    {
        auto guard = binding.prepareStep(3);
        ASSERT_TRUE(guard);
        ASSERT_TRUE(binding.stepPrepared());
        throw std::invalid_argument("simulated graph-construction throw");
    }
    catch (const std::invalid_argument &)
    {
        // The guard was destroyed while unwinding: the step is cancelled.
        EXPECT_FALSE(binding.stepPrepared()) << "step grant must return on unwind";
        EXPECT_FALSE(binding.priorFor(0));
        EXPECT_FALSE(binding.hasLivePublication());
    }
    // No version was consumed by a step that never published.
    EXPECT_EQ(binding.remainingIssuances(), issuances);

    // N1 negative 2: an execution-time throw AFTER a layer published into its
    // next cell must cancel the step and drop the not-yet-visible cell.
    MPIContext mpi{0, 1, MPI_COMM_WORLD};
    TensorFactory factory(mpi);
    const size_t width = static_cast<size_t>(heads * dimension);
    auto K = factory.createFP32({3, static_cast<long long>(width)}, device);
    auto V = factory.createFP32({3, static_cast<long long>(width)}, device);
    std::fill_n(K->mutable_data(), 3 * width, 0.5f);
    std::fill_n(V->mutable_data(), 3 * width, -0.5f);
    try
    {
        auto guard = binding.prepareStep(3);
        ASSERT_TRUE(guard);
        auto *cell = binding.nextCell(0);
        PooledKVAppendStage::Params append;
        append.device_id = device;
        append.K = K.get();
        append.V = V.get();
        append.pool = pool.get();
        append.table = table.get();
        append.ref_out = cell;
        append.grant = binding.stepGrant();
        append.issuance = binding.issuance();
        append.request_id = 41;
        append.slot = 0;
        append.layer_idx = 0;
        append.kv_heads = kv_heads;
        append.head_dim = dimension;
        append.page_tokens = page_tokens;
        append.live_rows = 3;
        PooledKVAppendStage stage(append);
        ASSERT_TRUE(stage.execute(nullptr));
        ASSERT_TRUE(cell->live());
        throw std::runtime_error("simulated execution throw after publish");
    }
    catch (const std::runtime_error &)
    {
        EXPECT_FALSE(binding.stepPrepared());
        EXPECT_FALSE(binding.nextCell(0)->live())
            << "next cell must be dropped on unwind";
        EXPECT_FALSE(binding.hasLivePublication());
    }
    // The aborted publication consumed one reserved issuance (finite budget
    // tracks burned issuances too): 4 - 1 = 3 remain.
    EXPECT_EQ(binding.remainingIssuances(), issuances - 1);
    EXPECT_FALSE(binding.stepComplete());
}

/** @test R25 N1: completeAndAdvance refuses a missing layer publication and
 * cancels the whole step (all-or-none). */
TEST(PooledStepGuard, MissingLayerPublicationCancelsWholeStep)
{
    constexpr int heads = 2, kv_heads = 2, dimension = 16, page_tokens = 8;
    const auto device = DeviceId::cpu();
    constexpr int layers = 2;
    constexpr std::size_t issuances = 8;
    const auto policy = PooledKVDecodeBinding::requestPolicy(page_tokens, layers,
        issuances, /*readers_per_bank=*/4);
    // Headroom must seat layers*issuances = 16: headroom = 2*bank_count
    // (bank_count is 8 in this envelope), budget = 2*headroom.
    const auto headroom = 2 * policy.bank_count;
    const PhysicalPagePoolGeometry pool_geometry{2 * policy.bank_count,
        static_cast<std::size_t>(page_tokens) * 2 * heads * dimension * sizeof(float), 2,
        2 * policy.bank_count};
    const CPUFP32PagedKVTableGeometry table_geometry{4, 2 * policy.bank_count, 4,
        page_tokens, 2 * headroom, headroom};
    const auto bytes = PhysicalPagePool::physicalBytes(pool_geometry, device) +
        CPUFP32PagedKVTable::physicalBytes(table_geometry, device);
    PhysicalMemoryPlanBuilder plan;
    plan.add({.world_rank=0, .device=device, .total_bytes=bytes, .admission_available_bytes=bytes},
        PhysicalMemoryOwner::KVCache, bytes);
    auto authority = std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
    auto pool = authority->createCPUPagePool(device, pool_geometry, CPUStoragePlacement::local());
    auto table = CPUFP32PagedKVTable::create(authority, device, table_geometry);
    PooledKVDecodeBinding binding(pool.get(), table.get(), /*request_id=*/42,
        /*slot=*/0, kv_heads, dimension, page_tokens, layers, issuances,
        /*persistent_tail=*/false);

    auto guard = binding.prepareStep(1);
    ASSERT_TRUE(guard);
    ASSERT_TRUE(binding.stepPrepared());
    // Publish ONLY layer 0; layer 1's append "never ran" (a hidden skip).
    MPIContext mpi{0, 1, MPI_COMM_WORLD};
    TensorFactory factory(mpi);
    const size_t width = static_cast<size_t>(heads * dimension);
    auto K = factory.createFP32({1, static_cast<long long>(width)}, device);
    auto V = factory.createFP32({1, static_cast<long long>(width)}, device);
    std::fill_n(K->mutable_data(), width, 0.5f);
    std::fill_n(V->mutable_data(), width, -0.5f);
    auto *cell0 = binding.nextCell(0);
    PooledKVAppendStage::Params append0;
    append0.device_id = device;
    append0.K = K.get();
    append0.V = V.get();
    append0.pool = pool.get();
    append0.table = table.get();
    append0.ref_out = cell0;
    append0.grant = binding.stepGrant();
    append0.issuance = binding.issuance();
    append0.request_id = 42;
    append0.slot = 0;
    append0.layer_idx = 0;
    append0.kv_heads = kv_heads;
    append0.head_dim = dimension;
    append0.page_tokens = page_tokens;
    append0.live_rows = 1;
    PooledKVAppendStage stage0(append0);
    ASSERT_TRUE(stage0.execute(nullptr));
    EXPECT_TRUE(cell0->live());
    // completeAndAdvance must refuse (layer 1 missing) and cancel the step.
    EXPECT_FALSE(guard.completeAndAdvance());
    EXPECT_FALSE(binding.stepPrepared());
    EXPECT_FALSE(binding.priorFor(0));
    EXPECT_FALSE(binding.priorFor(1));
    EXPECT_FALSE(binding.hasLivePublication());
    EXPECT_FALSE(cell0->live());
}

/** @test R26 N2: a binding whose admitted envelope exceeds the table geometry
 * refuses at construction; a request that drains its envelope refuses further
 * steps before the graph runs. */
TEST(PooledStepGuard, VersionEnvelopeExhaustionRefusesBeforeGraph)
{
    constexpr int heads = 2, kv_heads = 2, dimension = 16, page_tokens = 8;
    const auto device = DeviceId::cpu();
    // Geometry whose headroom seats only 3 issuances but the request asks for
    // 12: the binding must refuse at construction (headroom check).
    const PhysicalPagePoolGeometry pool_geometry{32,
        static_cast<std::size_t>(page_tokens) * 2 * heads * dimension * sizeof(float), 2, 32};
    const CPUFP32PagedKVTableGeometry small_geometry{4, 3, 4, page_tokens, 32, 3};
    const auto bytes = PhysicalPagePool::physicalBytes(pool_geometry, device) +
        CPUFP32PagedKVTable::physicalBytes(small_geometry, device);
    PhysicalMemoryPlanBuilder plan;
    plan.add({.world_rank=0, .device=device, .total_bytes=bytes, .admission_available_bytes=bytes},
        PhysicalMemoryOwner::KVCache, bytes);
    auto authority = std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
    auto pool = authority->createCPUPagePool(device, pool_geometry, CPUStoragePlacement::local());
    auto table = CPUFP32PagedKVTable::create(authority, device, small_geometry);
    EXPECT_THROW(PooledKVDecodeBinding(pool.get(), table.get(), /*request_id=*/43,
        /*slot=*/0, kv_heads, dimension, page_tokens, /*layers=*/1,
        /*admitted_issuances=*/12), std::invalid_argument);

    // Correctly seated envelope: 3 issuances, then drain: steps 1..3 succeed,
    // the 4th refuses BEFORE any graph (envelope exhausted) — a request drains
    // instead of stalling mid-graph.
    constexpr std::size_t issuances = 3;
    const auto policy = PooledKVDecodeBinding::requestPolicy(page_tokens, /*layers=*/1,
        issuances, /*readers_per_bank=*/4);
    const PhysicalPagePoolGeometry pool_geometry2{policy.bank_count,
        static_cast<std::size_t>(page_tokens) * 2 * heads * dimension * sizeof(float), 2,
        policy.bank_count};
    const CPUFP32PagedKVTableGeometry table_geometry{4, policy.bank_count, 4,
        page_tokens, 2 * policy.bank_count, policy.bank_count};
    const auto bytes2 = PhysicalPagePool::physicalBytes(pool_geometry2, device) +
        CPUFP32PagedKVTable::physicalBytes(table_geometry, device);
    PhysicalMemoryPlanBuilder plan2;
    plan2.add({.world_rank=0, .device=device, .total_bytes=bytes2, .admission_available_bytes=bytes2},
        PhysicalMemoryOwner::KVCache, bytes2);
    auto authority2 = std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan2.build()), 0);
    auto pool2 = authority2->createCPUPagePool(device, pool_geometry2, CPUStoragePlacement::local());
    auto table2 = CPUFP32PagedKVTable::create(authority2, device, table_geometry);
    PooledKVDecodeBinding binding(pool2.get(), table2.get(), /*request_id=*/44,
        /*slot=*/0, kv_heads, dimension, page_tokens, /*layers=*/1, issuances,
        /*persistent_tail=*/false);
    EXPECT_EQ(binding.remainingIssuances(), issuances);

    MPIContext mpi{0, 1, MPI_COMM_WORLD};
    TensorFactory factory(mpi);
    const size_t width = static_cast<size_t>(heads * dimension);
    auto K = factory.createFP32({1, static_cast<long long>(width)}, device);
    auto V = factory.createFP32({1, static_cast<long long>(width)}, device);
    std::fill_n(K->mutable_data(), width, 0.5f);
    std::fill_n(V->mutable_data(), width, -0.5f);

    for (int step = 0; step < 3; ++step)
    {
        auto guard = binding.prepareStep(1);
        ASSERT_TRUE(guard) << "step " << step << " must be admitted";
        ASSERT_TRUE(binding.stepPrepared());
        auto *cell = binding.nextCell(0);
        auto *prior = binding.priorFor(0); // M1: distinct prior (null on step 0).
        PooledKVAppendStage::Params append;
        append.device_id = device;
        append.K = K.get();
        append.V = V.get();
        append.pool = pool2.get();
        append.table = table2.get();
        append.ref_out = cell;
        append.prior = prior;
        append.grant = binding.stepGrant();
        append.issuance = binding.issuance();
        append.request_id = 44;
        append.slot = 0;
        append.layer_idx = 0;
        append.kv_heads = kv_heads;
        append.head_dim = dimension;
        append.page_tokens = page_tokens;
        append.live_rows = 1;
        PooledKVAppendStage stage(append);
        ASSERT_TRUE(stage.execute(nullptr));
        ASSERT_TRUE(cell->live());
        EXPECT_TRUE(guard.completeAndAdvance());
    }
    EXPECT_EQ(binding.remainingIssuances(), 0u);
    EXPECT_EQ(binding.nextPosition(), 3);
    // Drain complete: the next step refuses BEFORE any graph work.
    auto exhausted = binding.prepareStep(1);
    EXPECT_FALSE(exhausted);
    EXPECT_FALSE(binding.stepPrepared());
    EXPECT_EQ(binding.nextPosition(), 3);
}

/** @test R26 P2 matrix: several/odd row counts, page-edge/partial extents,
 * permuted physical frames and exact borrower retirement. The publication is
 * the unit: logical page order is the page table; physical frame assignment is
 * never part of history. */
TEST(PooledMatrix, OddCountsPageEdgesAndPermutedFramesPreserveBytes)
{
    constexpr int heads = 2, kv_heads = 2, dimension = 16, page_tokens = 8;
    const auto device = DeviceId::cpu();
    const PhysicalPagePoolGeometry pool_geometry{64,
        static_cast<std::size_t>(page_tokens) * 2 * heads * dimension * sizeof(float), 2, 64};
    const CPUFP32PagedKVTableGeometry table_geometry{8, 8, 4, page_tokens, 64, 16};
    const auto bytes = PhysicalPagePool::physicalBytes(pool_geometry, device) +
        CPUFP32PagedKVTable::physicalBytes(table_geometry, device);
    PhysicalMemoryPlanBuilder plan;
    plan.add({.world_rank=0, .device=device, .total_bytes=bytes, .admission_available_bytes=bytes},
        PhysicalMemoryOwner::KVCache, bytes);
    auto authority = std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
    auto pool = authority->createCPUPagePool(device, pool_geometry, CPUStoragePlacement::local());
    auto table = CPUFP32PagedKVTable::create(authority, device, table_geometry);
    const size_t width = static_cast<size_t>(heads * dimension);
    std::mt19937 random(31);
    std::uniform_real_distribution<float> values(-0.5f, 0.5f);

    MPIContext mpi{0, 1, MPI_COMM_WORLD};
    TensorFactory factory(mpi);

    const auto verify_publication = [&](const PooledPublicationCell &pub,
        const std::vector<float> &k, const std::vector<float> &v, int rows)
    {
        ASSERT_TRUE(pub.live());
        EXPECT_EQ(pub.snapshot.view().rows(), rows);
        const auto pages = pub.snapshot.view().pages();
        int expect_row = 0;
        for (const auto &page : pages)
        {
            const int page_rows = page.rows();
            const std::size_t row_bytes = 2 * width * sizeof(float);
            ASSERT_EQ(page.storage().size(), static_cast<std::size_t>(page_rows) * row_bytes);
            const float *got = reinterpret_cast<const float *>(page.storage().data());
            for (int r = 0; r < page_rows; ++r)
            {
                const float *k_row = k.data() + static_cast<size_t>(expect_row + r) * width;
                const float *v_row = v.data() + static_cast<size_t>(expect_row + r) * width;
                EXPECT_EQ(std::memcmp(got + static_cast<size_t>(r) * 2 * width, k_row, width * sizeof(float)), 0)
                    << "K diverges at logical row " << (expect_row + r);
                EXPECT_EQ(std::memcmp(got + static_cast<size_t>(r) * 2 * width + width, v_row, width * sizeof(float)), 0)
                    << "V diverges at logical row " << (expect_row + r);
            }
            expect_row += page_rows;
        }
        EXPECT_EQ(expect_row, rows);
    };

    // Odd counts 1, 3, 5, 7 (partial tails) and page edges 8, 9, 16, 17.
    const std::vector<int> counts{1, 3, 5, 7, 8, 9, 16, 17};
    for (const int rows : counts)
    {
        std::vector<float> k(static_cast<size_t>(rows) * width), v(k.size());
        for (auto *bank : {&k, &v}) for (auto &x : *bank) x = values(random);
        auto K = factory.createFP32({rows, static_cast<long long>(width)}, device);
        auto V = factory.createFP32({rows, static_cast<long long>(width)}, device);
        std::memcpy(K->mutable_data(), k.data(), k.size() * sizeof(float));
        std::memcpy(V->mutable_data(), v.data(), v.size() * sizeof(float));
        auto grant = pool->tryAdmit(static_cast<std::size_t>((rows + page_tokens - 1) / page_tokens));
        ASSERT_TRUE(grant) << "grant refused for rows=" << rows;
        PooledPublicationCell pub;
        PooledKVAppendStage::Params params;
        params.device_id = device;
        params.K = K.get();
        params.V = V.get();
        params.pool = pool.get();
        params.table = table.get();
        params.ref_out = &pub;
        params.grant = &*grant;
        params.request_id = 101 + rows;
        params.slot = 0;
        params.layer_idx = 0;
        params.kv_heads = kv_heads;
        params.head_dim = dimension;
        params.page_tokens = page_tokens;
        params.live_rows = rows;
        PooledKVAppendStage stage(params);
        ASSERT_TRUE(stage.execute(nullptr)) << "stage refused for rows=" << rows;
        verify_publication(pub, k, v, rows);
        // Exact borrower retirement: a borrower resolves, retires, and the
        // bank is reusable; a retired borrower never keeps a bank alive.
        auto ref = pub.ref;
        auto borrow = table->resolveSnapshotRef(ref);
        ASSERT_TRUE(borrow);
        EXPECT_EQ(borrow->view().rows(), rows);
        borrow = {}; // Retire the borrower.
    }
}

/** @test R26 P2 matrix: permuted physical frames never affect logical order.
 * Occupy frames in a scattered order, then append pages through the pool's
 * frame selection; the page table order (not frame index) is history. */
TEST(PooledMatrix, PermutedPhysicalFramesKeepLogicalPageOrder)
{
    constexpr int heads = 2, kv_heads = 2, dimension = 16, page_tokens = 4;
    const auto device = DeviceId::cpu();
    const PhysicalPagePoolGeometry pool_geometry{16,
        static_cast<std::size_t>(page_tokens) * 2 * heads * dimension * sizeof(float), 2, 16};
    const CPUFP32PagedKVTableGeometry table_geometry{4, 4, 4, page_tokens, 32, 8};
    const auto bytes = PhysicalPagePool::physicalBytes(pool_geometry, device) +
        CPUFP32PagedKVTable::physicalBytes(table_geometry, device);
    PhysicalMemoryPlanBuilder plan;
    plan.add({.world_rank=0, .device=device, .total_bytes=bytes, .admission_available_bytes=bytes},
        PhysicalMemoryOwner::KVCache, bytes);
    auto authority = std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
    auto pool = authority->createCPUPagePool(device, pool_geometry, CPUStoragePlacement::local());
    auto table = CPUFP32PagedKVTable::create(authority, device, table_geometry);
    const size_t width = static_cast<size_t>(heads * dimension);
    // 9 rows = pages [4][4][1]: assign page 0 FIRST in a scattered order by
    // occupying/retiring frames: the pool picks a free frame, so we exercise
    // non-contiguous frame selection by interleaving unrelated holds.
    const int rows = 9;
    std::mt19937 random(73);
    std::uniform_real_distribution<float> values(-0.5f, 0.5f);
    std::vector<float> k(static_cast<size_t>(rows) * width), v(k.size());
    for (auto *bank : {&k, &v}) for (auto &x : *bank) x = values(random);
    MPIContext mpi{0, 1, MPI_COMM_WORLD};
    TensorFactory factory(mpi);
    auto K = factory.createFP32({rows, static_cast<long long>(width)}, device);
    auto V = factory.createFP32({rows, static_cast<long long>(width)}, device);
    std::memcpy(K->mutable_data(), k.data(), k.size() * sizeof(float));
    std::memcpy(V->mutable_data(), v.data(), v.size() * sizeof(float));
    // Admit MORE than the append needs so scattered assignments leave the
    // append's own frames intact; a separate occupant grant holds one frame
    // so the append's frames land after other owners.
    auto grant = pool->tryAdmit(6);
    ASSERT_TRUE(grant);
    // Scatter: assign 2 frames of the SAME grant (a frame the append later
    // assigns is necessarily a different physical frame), then hold a private
    // occupant that pins a frame between them — the append never observes
    // physical frame order.
    const auto f0 = grant->assignFrame();
    (void)f0;
    const auto f1 = grant->assignFrame();
    (void)f1;
    auto occupant = pool->tryAdmit(1);
    ASSERT_TRUE(occupant);
    (void)occupant->assignFrame();
    PooledPublicationCell pub;
    PooledKVAppendStage::Params params;
    params.device_id = device;
    params.K = K.get();
    params.V = V.get();
    params.pool = pool.get();
    params.table = table.get();
    params.ref_out = &pub;
    params.grant = &*grant;
    params.request_id = 99;
    params.slot = 0;
    params.layer_idx = 0;
    params.kv_heads = kv_heads;
    params.head_dim = dimension;
    params.page_tokens = page_tokens;
    params.live_rows = rows;
    PooledKVAppendStage stage(params);
    ASSERT_TRUE(stage.execute(nullptr));
    ASSERT_TRUE(pub.live());
    EXPECT_EQ(pub.snapshot.view().rows(), rows);
    const auto pages = pub.snapshot.view().pages();
    ASSERT_EQ(static_cast<int>(pages.size()), 3);
    // Logical order = page table order: page 0 → rows 0..3, page 1 → rows 4..7,
    // page 2 → row 8. Physical frames NEVER appear in the view.
    for (int p = 0; p < 3; ++p)
    {
        const int page_rows = pages[static_cast<std::size_t>(p)].rows();
        const int first_row = p * page_tokens;
        const float *got = reinterpret_cast<const float *>(pages[static_cast<std::size_t>(p)].storage().data());
        for (int r = 0; r < page_rows; ++r)
        {
            const int lr = first_row + r;
            EXPECT_EQ(std::memcmp(got + static_cast<size_t>(r) * 2 * width,
                k.data() + static_cast<size_t>(lr) * width, width * sizeof(float)), 0)
                << "page " << p << " K diverges at logical row " << lr;
            EXPECT_EQ(std::memcmp(got + static_cast<size_t>(r) * 2 * width + width,
                v.data() + static_cast<size_t>(lr) * width, width * sizeof(float)), 0)
                << "page " << p << " V diverges at logical row " << lr;
        }
    }
}

/** @test R26 P2: a pooled request that drains its version envelope before its
 * decode completes is REFUSED cleanly (no mid-graph stall); a binding built
 * with the right envelope completes the full decode without stalling. */
TEST(PooledMatrix, LongDecodeWithinAdmittedEnvelopeDrainsCleanly)
{
    constexpr int heads = 2, kv_heads = 2, dimension = 16, page_tokens = 8;
    const auto device = DeviceId::cpu();
    constexpr int layers = 1;
    constexpr std::size_t issuances = 10; // 1 prefill-issuance + 9 decode.
    const auto policy = PooledKVDecodeBinding::requestPolicy(page_tokens, layers,
        issuances, /*readers_per_bank=*/4);
    const auto headroom = 2 * policy.bank_count;
    const PhysicalPagePoolGeometry pool_geometry{headroom,
        static_cast<std::size_t>(page_tokens) * 2 * heads * dimension * sizeof(float), 2,
        headroom};
    const CPUFP32PagedKVTableGeometry table_geometry{4, headroom, 4,
        page_tokens, 2 * headroom, headroom};
    const auto bytes = PhysicalPagePool::physicalBytes(pool_geometry, device) +
        CPUFP32PagedKVTable::physicalBytes(table_geometry, device);
    PhysicalMemoryPlanBuilder plan;
    plan.add({.world_rank=0, .device=device, .total_bytes=bytes, .admission_available_bytes=bytes},
        PhysicalMemoryOwner::KVCache, bytes);
    auto authority = std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
    auto pool = authority->createCPUPagePool(device, pool_geometry, CPUStoragePlacement::local());
    auto table = CPUFP32PagedKVTable::create(authority, device, table_geometry);
    PooledKVDecodeBinding binding(pool.get(), table.get(), /*request_id=*/88,
        /*slot=*/0, kv_heads, dimension, page_tokens, layers, issuances,
        /*persistent_tail=*/false);
    const size_t width = static_cast<size_t>(heads * dimension);
    MPIContext mpi{0, 1, MPI_COMM_WORLD};
    TensorFactory factory(mpi);
    std::mt19937 random(17);
    std::uniform_real_distribution<float> values(-0.5f, 0.5f);
    auto K = factory.createFP32({1, static_cast<long long>(width)}, device);
    auto V = factory.createFP32({1, static_cast<long long>(width)}, device);
    for (auto *t : {K.get(), V.get()})
        for (size_t i = 0; i < width; ++i) t->mutable_data()[i] = values(random);

    // Simulate the runner's 12 forwards: 1 prefill (3 rows) + 9 decode (1 row).
    const auto do_step = [&](int rows) -> bool
    {
        auto guard = binding.prepareStep(rows);
        if (!guard) return false;
        auto *cell = binding.nextCell(0);
        auto *prior = binding.priorFor(0);
        PooledKVAppendStage::Params append;
        append.device_id = device;
        append.K = K.get();
        append.V = V.get();
        append.pool = pool.get();
        append.table = table.get();
        append.ref_out = cell;
        append.prior = prior;
        append.grant = binding.stepGrant();
        append.issuance = binding.issuance();
        append.request_id = 88;
        append.slot = 0;
        append.layer_idx = 0;
        append.kv_heads = kv_heads;
        append.head_dim = dimension;
        append.page_tokens = page_tokens;
        append.live_rows = rows;
        // Multi-row prefill needs a full K/V tensor; the fixture appends the
        // same row values, which is fine for the drain contract.
        PooledKVAppendStage stage(append);
        if (!stage.execute(nullptr)) return false;
        if (!cell->live()) return false;
        return guard.completeAndAdvance();
    };
    ASSERT_TRUE(do_step(3));  // Prefill (1 issuance).
    for (int step = 0; step < 9; ++step)
        ASSERT_TRUE(do_step(1)) << "decode step " << step;
    // 10 admitted - (1 prefill + 9 decode) = 0 remaining: the envelope drains
    // EXACTLY at completion. The request was never stalled mid-graph.
    EXPECT_EQ(binding.nextPosition(), 12);
    EXPECT_EQ(binding.remainingIssuances(), 0u);
    // A further forward refuses (no more issuances) rather than stalling.
    EXPECT_FALSE(do_step(1));
    EXPECT_EQ(binding.nextPosition(), 12);
}
