/**
 * @file CPUForwardSamplingTestFixture.h
 * @brief Shared tiny production-forward tensors and observable candidate protocol.
 *
 * These fixtures avoid model-file loading while exercising real graphs and
 * prepared weights. Tensor and store owners must outlive their runner.
 */
#pragma once
#include <gtest/gtest.h>
#include "backends/BackendManager.h"
#include "collective/IGlobalTPContext.h"
#include "execution/local_execution/orchestrators/DeviceGraphOrchestrator.h"
#include "loaders/PreparedWeightStore.h"
#include "mocks/MockMPIContext.h"
#include "models/qwen/QwenStandardGraph.h"
#include "utils/TestTensorFactory.h"
#include <cstring>
#include <memory>
#include <vector>
namespace llaminar2::test
{
    /** @brief Own reproducible, file-free, one-layer Qwen source tensors. */
    struct TinyQwenForwardFixture
    {
        /** @brief Layer tensor owners retained until runner teardown. */
        struct LayerTensors
        {
            std::unique_ptr<FP32Tensor> attn_norm;
            std::unique_ptr<FP32Tensor> wq;
            std::unique_ptr<FP32Tensor> wk;
            std::unique_ptr<FP32Tensor> wv;
            std::unique_ptr<FP32Tensor> wo;
            std::unique_ptr<FP32Tensor> ffn_norm;
            std::unique_ptr<FP32Tensor> gate_proj;
            std::unique_ptr<FP32Tensor> up_proj;
            std::unique_ptr<FP32Tensor> down_proj;
        };

        GraphConfig config;
        std::shared_ptr<MockMPIContext> mpi = std::make_shared<MockMPIContext>(0, 1);

        std::unique_ptr<FP32Tensor> embedding_table;
        std::unique_ptr<FP32Tensor> final_norm;
        std::unique_ptr<FP32Tensor> lm_head;
        std::vector<LayerTensors> layers;

        /** @brief Construct tiny weights; @param device Compute endpoint; @param kv_precision Test cache format. */
        TinyQwenForwardFixture(DeviceId device, KVCachePrecision kv_precision)
        {
            config.n_layers = 1;
            config.total_n_layers = 1;
            config.d_model = 64;
            config.n_heads = 4;
            config.n_kv_heads = 2;
            config.head_dim = 16;
            config.d_ff = 128;
            config.vocab_size = 128;
            config.rms_norm_eps = 1e-6f;
            config.rope_theta = 10000.0f;
            config.default_device = device;
            config.max_seq_len = 8;
            config.activation_precision = ActivationPrecision::FP32;
            config.kv_cache_precision = kv_precision;
            config.use_graph_buffer_management = true;
            config.mtp.enabled = true;
            config.mtp.draft_tokens = 1;
            /* This fixture is the ordinary sharded-TP oracle. */
            config.mtp.sidecar_dense_policy =
                MTPSidecarDensePolicy::TensorParallel;

            const size_t d = static_cast<size_t>(config.d_model);
            const size_t q_dim = static_cast<size_t>(config.n_heads * config.head_dim);
            const size_t kv_dim = static_cast<size_t>(config.n_kv_heads * config.head_dim);
            const size_t ff = static_cast<size_t>(config.d_ff);
            const size_t vocab = static_cast<size_t>(config.vocab_size);

            embedding_table = TestTensorFactory::createFP32Random({vocab, d}, -0.02f, 0.02f, 101);
            final_norm = TestTensorFactory::createFP32Ones({d});
            lm_head = TestTensorFactory::createFP32Random({vocab, d}, -0.02f, 0.02f, 102);

            layers.resize(static_cast<size_t>(config.n_layers));
            for (int i = 0; i < config.n_layers; ++i)
            {
                auto &layer = layers[static_cast<size_t>(i)];
                layer.attn_norm = TestTensorFactory::createFP32Ones({d});
                layer.wq = TestTensorFactory::createFP32Random({q_dim, d}, -0.02f, 0.02f, 110 + i);
                layer.wk = TestTensorFactory::createFP32Random({kv_dim, d}, -0.02f, 0.02f, 120 + i);
                layer.wv = TestTensorFactory::createFP32Random({kv_dim, d}, -0.02f, 0.02f, 130 + i);
                layer.wo = TestTensorFactory::createFP32Random({d, q_dim}, -0.02f, 0.02f, 140 + i);
                layer.ffn_norm = TestTensorFactory::createFP32Ones({d});
                layer.gate_proj = TestTensorFactory::createFP32Random({ff, d}, -0.02f, 0.02f, 150 + i);
                layer.up_proj = TestTensorFactory::createFP32Random({ff, d}, -0.02f, 0.02f, 160 + i);
                layer.down_proj = TestTensorFactory::createFP32Random({d, ff}, -0.02f, 0.02f, 170 + i);
            }
        }

        /** @return Borrowed graph bindings whose storage belongs to this fixture. */
        ModelWeights modelWeights()
        {
            ModelWeights weights;
            weights.embedding_table = embedding_table.get();
            weights.final_norm = final_norm.get();
            weights.lm_head = lm_head.get();
            weights.get_layer_weights = [this](int layer_idx)
            {
                const auto &src = layers.at(static_cast<size_t>(layer_idx));
                LayerWeights layer;
                layer.attn_norm = src.attn_norm.get();
                layer.wq = src.wq.get();
                layer.wk = src.wk.get();
                layer.wv = src.wv.get();
                layer.wo = src.wo.get();
                layer.ffn_norm = src.ffn_norm.get();
                return layer;
            };
            return weights;
        }
    };

    /**
     * @brief Prepare immutable graph weights through the production store.
     * @param orchestrator Runner owning the frozen bindings.
     * @param graph_builder Graph receiving the prepared references.
     * @param store Store retained throughout graph execution.
     * @param device Compute endpoint for these prepared weights.
     */
    inline void prepareDenseForwardWeights(
        const DeviceGraphOrchestrator &orchestrator,
        QwenStandardGraph &graph_builder,
        PreparedWeightStore &store,
        DeviceId device)
    {
        const FrozenModelWeightSet *frozen = orchestrator.frozenWeightSet();
        ASSERT_NE(frozen, nullptr);

        for (const auto &source_binding : frozen->bindings())
        {
            if (!source_binding.tensor ||
                source_binding.tensor->shape().size() != 2 ||
                source_binding.identity.role == WeightRole::Embedding)
            {
                continue;
            }

            WeightBinding binding = source_binding;
            binding.residency.home_device = device;
            binding.residency.resident_device = device;
            ASSERT_TRUE(binding.tensor->ensureOnDevice(device));
            store.prepareGemm(binding);
        }

        graph_builder.setPreparedWeightStore(&store);
    }

    /** @brief Count sampler collectives and supply an exact scripted peer record. */
    class ScriptedGlobalTPContext : public IGlobalTPContext
    {
    public:
        /** @brief Return the two-participant fixture degree. */
        int degree() const override { return 2; }
        /** @brief Return the scripted local index. */
        int myIndex() const override { return 0; }
        /** @brief Return the scripted protocol family. */
        CollectiveBackendType backend() const override { return CollectiveBackendType::MPI; }
        /** @brief Accept an unrelated tensor reduction. */
        bool allreduce(TensorBase * /*tensor*/) override { return true; }
        /** @brief Accept an unrelated tensor broadcast. */
        bool broadcast(TensorBase * /*tensor*/, int /*source_index*/ = 0) override { return true; }
        /** @brief Accept an unrelated tensor gather. */
        bool allgather(const TensorBase * /*local_shard*/, TensorBase * /*global_tensor*/) override { return true; }

        /** @brief Return no live communicator for scripted candidates. */
        MPI_Comm communicator() const override { return MPI_COMM_NULL; }
        /** @brief Return the fixture domain ID. */
        int domainId() const override { return 7; }
        /** @brief Return the fixture's two member IDs. */
        const std::vector<int> &worldRanks() const override { return world_ranks_; }
        /** @brief Return the local CPU endpoint. */
        GlobalDeviceAddress localDevice() const override { return GlobalDeviceAddress::cpu(0, "rank0"); }
        /** @brief No-op because candidate exchange is scripted. */
        void barrier() const override {}
        /** @brief Reject unsupported point-to-point sending. */
        bool send(const TensorBase * /*tensor*/, int /*dest_index*/) override { return false; }
        /** @brief Reject unsupported point-to-point receiving. */
        bool recv(TensorBase * /*tensor*/, int /*source_index*/) override { return false; }

        /** @brief Copy the peer record that the next fixed-size gather must match. */
        void setRemoteRecordBytes(const void *record, size_t byte_count)
        {
            remote_record_.resize(byte_count);
            std::memcpy(remote_record_.data(), record, byte_count);
        }

        /** @brief Count the call and join local/peer records; reject mismatched widths. */
        bool allgatherBytes(const void *send_data, void *recv_data, size_t byte_count) const override
        {
            ++allgather_bytes_calls_;
            if (!send_data || !recv_data || byte_count == 0 || remote_record_.size() != byte_count)
                return false;

            auto *out = static_cast<uint8_t *>(recv_data);
            std::memcpy(out, send_data, byte_count);
            std::memcpy(out + byte_count, remote_record_.data(), byte_count);
            return true;
        }
        /** @brief Reject variable-size traffic outside this fixture's protocol. */
        bool gatherVariableFloatRecordsToRoot(
            const float *, size_t, float *, size_t, size_t, int,
            size_t &, const std::string &) override
        {
            return false;
        }
        /** @brief Reject tensor broadcasts outside this fixture's protocol. */
        bool broadcastFloatElements(
            TensorBase *, size_t, int, const std::string &) override
        {
            return false;
        }

        /** @brief Return attempted sampling-collective count. */
        int allgatherBytesCalls() const { return allgather_bytes_calls_; }

    private:
        std::vector<int> world_ranks_ = {0, 1};
        std::vector<uint8_t> remote_record_;
        mutable int allgather_bytes_calls_ = 0;
    };

    /** @brief Fixed-width sampler candidate wire record. */
    struct GreedyCandidateRecord
    {
        float value = 0.0f;
        int32_t token = -1;
        int32_t valid = 0;
        int32_t reserved = 0;
    };

    static_assert(sizeof(GreedyCandidateRecord) == 16);

}
