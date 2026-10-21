/**
 * @file GDNQKVBundleTestHarness.h
 * @brief Four-distinct-source GDN stage regression over real prepared engines.
 *
 * Synthetic native Q8_0 QKV/Z and original F32 alpha/beta reproduce the source
 * formats of QFN (controls use its existing FP16 preparation). This component
 * uses the established test-only weight-pool allocation boundary; its single
 * named workspace is admitted exactly by PMA. It does not certify model
 * admission, semantic query provenance or a full generation policy. Serial
 * submissions are diagnostic byte oracles, never a production implementation.
 */
#pragma once
#include "GpuPreparedGemmHarness.h"
#include "TestTensorFactory.h"
#include "loaders/PreparedHalfWeights.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/compute_stages/stages/GDNProjectionStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "kernels/cpu/CPUInvocationWorkspace.h"
#include "kernels/common/Q8ResidualCorrection.h"
#include "kernels/common/Q8ResidualGroupedGeometry.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "transfer/TransferEngine.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <memory>
#include <map>
#include <tuple>
#include <vector>

namespace llaminar2::test
{
/** @brief Throw before an incomplete component contract can submit more work. */
inline void requireGDNBundle(bool ok, const char *message)
{
    if (!ok) throw std::runtime_error(message);
}

/** @brief One exact producer-stream fence protecting diagnostic host storage. */
struct GDNBundleCopyCompletion final
{
    IBackend *backend; DeviceId device; void *stream;
    /** @brief A partial submission must also retire before its host span. */
    ~GDNBundleCopyCompletion() noexcept
    {
        if (stream && !backend->synchronizeStream(stream, device.ordinal))
            std::terminate();
    }
};

/**
 * @brief Own one four-source stage family and its exact merged workspace.
 * All graphs share storage only because this fixture submits them serially on
 * one exact stream. The original source tensors, prepared store, loader pools,
 * count and outputs outlive graph retirement on every exceptional exit.
 */
class GDNQKVBundleFixture final
{
public:
    /**
     * @brief Prepare actual independent sources before any retained capture.
     * @param device Physical participant; CPU initialization never initializes GPUs.
     * @param context Actual CPU/CUDA/ROCm stage context.
     * @param worker Actual native graph owner; null only for CPU.
     * @param stream Exact non-default native stream; null only for CPU.
     * @param k Original input width in complete native32 blocks.
     * @param columns QKV/Z/alpha/beta native output widths.
     */
    GDNQKVBundleFixture(DeviceId device, IDeviceContext &context,
        IWorkerGPUContext *worker, void *stream, int k,
        std::array<int, 4> columns)
        : device_(device), context_(context), worker_(worker), stream_(stream),
          k_(k), columns_(columns), store_(ModelContextId{41971})
    {
        if (device_.is_cpu() && !hasCPUBackend()) initCPUBackend(-1);
        backend_ = getBackendFor(device_);
        requireGDNBundle(backend_ && (device_.is_cpu() ? !worker_ && !stream_ : worker_ && stream_),
            "Bundle requires its actual backend/worker/exact stream");
        originals_[0] = TestTensorFactory::createQ8_0Random(
            {size_t(columns_[0]), size_t(k_)}, 419710);
        originals_[1] = TestTensorFactory::createQ8_0Random(
            {size_t(columns_[1]), size_t(k_)}, 419711);
        for (size_t i : {2u, 3u})
            originals_[i] = TestTensorFactory::createFP32Random(
                {size_t(columns_[i]), size_t(k_)}, -.125f, .125f, 419710 + i);
        weights_ = originals_;
        for (size_t i : {2u, 3u})
            weights_[i] = materializePreparedHalfWeights(*originals_[i]);
        const std::array<std::string, 4> names{"blk.0.attn_qkv.weight",
            "blk.0.attn_gate.weight", "blk.0.ssm_alpha.weight", "blk.0.ssm_beta.weight"};
        for (size_t i = 0; i < 4; ++i)
        {
            const auto bytes = originals_[i]->size_bytes();
            const auto *raw = static_cast<const uint8_t *>(originals_[i]->raw_data());
            source_bytes_[i].assign(raw, raw + bytes);
            if (device_.is_cpu())
            {
                auto binding = makePreparedWeightTestBinding(
                    weights_[i].get(), device_, names[i], store_.modelId());
                if (i >= 2) binding.floating_operand_precision = FloatingActivationOperandPrecision::FP16RNE;
                refs_[i] = store_.prepareGemm(binding);
            }
            else if (i < 2)
                refs_[i] = registerGpuPreparedGemmInStore(weights_[i].get(), device_,
                    names[i], store_.modelId(), &store_, &loaders_);
            else
                refs_[i] = registerGpuPreparedFloatingPointGemmInStore(weights_[i].get(),
                    device_, names[i], store_, loaders_, FloatingActivationOperandPrecision::FP16RNE);
            kernels_[i] = store_.gemmKernel(refs_[i]);
            requireGDNBundle(kernels_[i] && store_.binding(refs_[i])->tensor == weights_[i].get(),
                "Actual source/store binding missing");
            outputs_[i] = TestTensorFactory::createFP32Zeros({15, size_t(columns_[i])});
            serial_outputs_[i] = TestTensorFactory::createFP32Zeros({1, size_t(columns_[i])});
            if (!device_.is_cpu())
            {
                requireGDNBundle(outputs_[i]->allocateOnDevice(device_, stream_) &&
                    serial_outputs_[i]->allocateOnDevice(device_, stream_), "Output materialization failed");
            }
        }
        for (size_t i = 0; i < 4; ++i)
        {
            requireGDNBundle(originals_[i]->native_type() == (i < 2 ? TensorType::Q8_0 : TensorType::FP32) &&
                weights_[i]->native_type() == (i < 2 ? TensorType::Q8_0 : TensorType::FP16),
                "Source format or default floating preparation changed");
            if (i < 2)
            {
                NativeVnniSourceIdentity source;
                requireGDNBundle(kernels_[i]->exportNativeVNNISourceIdentity(source) && source.present &&
                    !source.is_superblock && source.codebook_id == native_vnni_formats::Q8_0.codebook_id &&
                    source_bytes_[i].size() == size_t(columns_[i]) * (k_ / 32) * sizeof(Q8_0Block),
                    "Original GGUF type8 must retain native execution codebook19; no weight conversion");
            }
            else
                requireGDNBundle(kernels_[i]->floatingActivationOperandPrecision() == FloatingActivationOperandPrecision::FP16RNE,
                    "Alpha/beta must retain default prepared-half operand policy");
            for (size_t j = 0; j < i; ++j)
                requireGDNBundle(refs_[i].binding_id != refs_[j].binding_id && kernels_[i] != kernels_[j],
                    "Every distinct original projection must retain its own actual prepared handle");
        }
        requireGDNBundle(weights_[0].get() != weights_[1].get() &&
            weights_[2].get() != weights_[3].get() && refs_[0].binding_id != refs_[1].binding_id && refs_[2].binding_id != refs_[3].binding_id,
            "Four independent original sources are mandatory");
        input_ = TestTensorFactory::createFP32Random({15, size_t(k_)}, -.75f, .75f, 419715);
        original_input_.assign(input_->data(), input_->data() + 15 * size_t(k_));
        serial_input_ = TestTensorFactory::createFP32Zeros({1, size_t(k_)});
        count_ = std::make_unique<INT32Tensor>(std::vector<size_t>{1});
        *static_cast<int32_t *>(count_->raw_mutable_data()) = 15;
        publishInputs();
        if (!device_.is_cpu())
        {
            TransferEngine::publishHostWrite(serial_input_.get());
            TransferEngine::prepareDeviceInput(serial_input_.get(), device_, stream_);
            TransferEngine::requireDeviceInput(serial_input_.get(), device_, stream_);
        }
        for (int m : {1, 9, 15})
            for (bool grouped : {false, true})
                for (bool corrected : {false, true})
                    append(m, grouped, corrected, false);
        append(1, false, true, true);
        const auto bytes = requirements_.total_bytes_with_alignment();
        PhysicalMemoryPlanBuilder builder;
        builder.add({.world_rank = 0, .device = device_, .total_bytes = bytes,
            .admission_available_bytes = bytes}, PhysicalMemoryOwner::ExecutionWorkspace, bytes);
        memory_ = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()), 0);
        workspace_ = std::make_unique<DeviceWorkspaceManager>(device_, bytes, memory_);
        requireGDNBundle(workspace_->allocate(requirements_), "Exact merged named workspace failed");
        workspace_used_ = workspace_->used();
        for (auto &entry : entries_) entry.stage->bindWorkspace(workspace_.get());
    }

    /** @brief Fence before graphs, stages, workspace, pools or source roots retire. */
    ~GDNQKVBundleFixture() noexcept
    {
        if (stream_ && !backend_->synchronizeStream(stream_, device_.ordinal)) std::terminate();
        for (auto &entry : entries_) entry.graph.reset();
        for (auto &entry : entries_) entry.stage->unbindWorkspace();
        entries_.clear();
    }

    /** @brief Run retained large/small/count/tail controls and original-source guards. */
    void prove()
    {
        SCOPED_TRACE(::testing::Message() << device_.toString() << " K=" << k_);
        for (auto &entry : entries_)
        {
            capture(entry);
            ASSERT_FALSE(::testing::Test::HasFailure());
        }
        rejectMalformedBeforeWrites();
        proveWorkspaceHints();
        proveDefaultGrouping();
        ASSERT_FALSE(::testing::Test::HasFailure());
        // Every replay changes actual input tail bytes while keeping the live
        // prefix exact. No graph is recaptured after reset or count publication.
        for (int round = 0; round < 2; ++round)
            for (int m : {15, 1, 9, 15})
                for (bool grouped : {false, true})
                {
                    const auto live_counts = device_.is_cpu() && grouped
                        ? std::vector<int>{m} : std::vector<int>{m, 0, 1, std::min(9, m), 0, m};
                    for (int live : live_counts)
                    {
                        proveFrame(m, live, grouped, round);
                        ASSERT_FALSE(::testing::Test::HasFailure());
                    }
                }
        for (size_t i = 0; i < 4; ++i)
        {
            EXPECT_EQ(originals_[i]->size_bytes(), source_bytes_[i].size());
            EXPECT_EQ(std::memcmp(originals_[i]->raw_data(), source_bytes_[i].data(),
                source_bytes_[i].size()), 0) << "Original synthetic source changed: " << i;
            EXPECT_EQ(store_.binding(refs_[i])->tensor, weights_[i].get());
        }
        EXPECT_EQ(workspace_->used(), workspace_used_);
    }

private:
    /** @brief One immutable physical stage/capture; mutable counts are never cached. */
    struct Entry
    {
        int m; bool grouped; bool corrected; bool serial;
        std::unique_ptr<GDNProjectionStage> stage;
        std::unique_ptr<IGPUGraphCapture> graph;
        uint64_t signature = 0;
    };

    /** @brief Publish only setup/diagnostic input state on the exact producer stream. */
    void publishInputs()
    {
        if (device_.is_cpu()) return;
        TransferEngine::publishHostWrite(input_.get());
        TransferEngine::prepareDeviceInput(input_.get(), device_, stream_);
        TransferEngine::requireDeviceInput(input_.get(), device_, stream_);
        TransferEngine::publishHostWrite(count_.get());
        TransferEngine::prepareDeviceInput(count_.get(), device_, stream_);
        TransferEngine::requireDeviceInput(count_.get(), device_, stream_);
    }

    /** @return Original host/device count; never a copied or inferred authority. */
    const int32_t *countPointer() const
    {
        return static_cast<const int32_t *>(device_.is_cpu() ? count_->raw_data() : count_->gpu_data_ptr());
    }

    /** @brief Bind original sources, actual refs and independent query arithmetic once. */
    void append(int m, bool grouped, bool corrected, bool serial)
    {
        GDNProjectionStage::Params p;
        p.device_id = device_; p.m = m; p.k = k_;
        p.input = serial ? serial_input_.get() : input_.get();
        auto &out = serial ? serial_outputs_ : outputs_;
        p.w_qkv = weights_[0].get(); p.output_qkv = out[0].get(); p.n_qkv = columns_[0]; p.gemm_qkv = kernels_[0];
        p.w_z = weights_[1].get(); p.output_z = out[1].get(); p.n_z = columns_[1]; p.gemm_z = kernels_[1];
        p.w_a = weights_[2].get(); p.output_a = out[2].get(); p.n_a = columns_[2]; p.gemm_a = kernels_[2];
        p.w_b = weights_[3].get(); p.output_b = out[3].get(); p.n_b = columns_[3]; p.gemm_b = kernels_[3];
        p.prepared_store = &store_;
        p.prepared_ref_qkv = refs_[0]; p.prepared_ref_z = refs_[1];
        p.prepared_ref_a = refs_[2]; p.prepared_ref_b = refs_[3];
        p.force_decode_equivalent_verifier_prefill = grouped;
        const auto rows = serial || (device_.is_cpu() && grouped)
            ? DeviceRowRange::fullyActive(m) : DeviceRowRange::deviceCounted(m, countPointer());
        if (grouped && m > 1) p.verifier_row_range = rows;
        if (corrected)
        {
            p.qkv_operand_precision = QuantizedActivationOperandPrecision::TwoTermQ8;
            p.qkv_row_range = rows;
        }
        auto stage = std::make_unique<GDNProjectionStage>(p);
        requirements_.merge(stage->getWorkspaceRequirements(m, 0, k_));
        entries_.push_back({m, grouped, corrected, serial, std::move(stage), nullptr, 0});
    }

    /** @brief Capture the actual complete stage, with all original external inputs. */
    void capture(Entry &entry)
    {
        entry.signature = entry.stage->graphCaptureVariantSignature();
        if (device_.is_cpu()) return;
        requireGDNBundle(entry.stage->prepareGraphLaunch(&context_, stream_), "Stage capture preparation failed");
        auto *input = entry.serial ? serial_input_.get() : input_.get();
        auto &out = entry.serial ? serial_outputs_ : outputs_;
        TransferEngine::requireDeviceInput(input, device_, stream_);
        TransferEngine::requireDeviceInput(count_.get(), device_, stream_);
        std::vector<const TensorBase *> external{input};
        if (!entry.serial) external.push_back(count_.get());
        GraphCaptureDependencyLedger::StagePlan plan{.stage_identity = entry.stage.get(),
            .stage_name = "four_original_source_gdn_bundle", .external_inputs = std::move(external),
            .internal_inputs = {}, .outputs = {out[0].get(), out[1].get(), out[2].get(), out[3].get()}};
        GraphCaptureDependencyLedger ledger(device_, stream_, {std::move(plan)}, "QKV bundle component");
        entry.graph = worker_->createGraphCapture(stream_);
        requireGDNBundle(bool(entry.graph), "Actual native graph missing");
        ScopedBackendGraphCapture transaction(*worker_, *entry.graph, "QKV bundle component", &ledger);
        requireGDNBundle(transaction.begin(), "Bundle capture begin failed");
        bool success = false;
        {
            ScopedGraphCaptureStage scope(entry.stage.get());
            success = entry.stage->execute(&context_);
            if (success) scope.complete();
        }
        transaction.finish();
        requireGDNBundle(success && entry.graph->instantiate(), "Actual bundle capture failed");
        std::vector<GPUGraphKernelNodeInfo> nodes; std::string error;
        requireGDNBundle(entry.graph->inspectKernelNodes(nodes, &error) && !nodes.empty(), "Captured bundle has no native kernel evidence");
        size_t publishers = 0, paired_dots = 0;
        for (const auto &node : nodes)
        {
            requireGDNBundle(node.valid() && node.name_resolved, "Native launch geometry/name unresolved");
            if (node.name.find("q8_residual") == std::string::npos) continue;
            EXPECT_EQ(node.local_memory_bytes_per_thread, 0u);
            EXPECT_EQ(node.grid_y, 1u);
            EXPECT_EQ(node.grid_z, 1u);
            EXPECT_EQ(node.block_y, 1u);
            EXPECT_EQ(node.block_z, 1u);
            if (node.name.find("publish") != std::string::npos)
            {
                ++publishers;
                EXPECT_EQ(node.block_x, 256u);
                EXPECT_EQ(node.grid_x, (uint64_t(entry.m) * (k_ / 32) + 7) / 8);
            }
            if (node.name.find("project") != std::string::npos)
            {
                ++paired_dots;
                const auto geometry = q8ResidualGroupedGeometry(device_.is_cuda()
                    ? Q8ResidualGroupedBackend::CUDA : Q8ResidualGroupedBackend::ROCm, entry.m, columns_[0]);
                EXPECT_EQ(node.block_x, static_cast<unsigned>(geometry.threads));
                EXPECT_EQ(node.grid_x, geometry.blocks);
                EXPECT_EQ(node.name.find("projectGrouped") != std::string::npos,
                    geometry.strategy == Q8ResidualGroupedStrategy::CooperativeMatrix);
            }
        }
        if (entry.corrected)
        {
            EXPECT_EQ(publishers, 1u);
            EXPECT_EQ(paired_dots, 1u);
        }
        else
        {
            EXPECT_EQ(publishers, 0u);
            EXPECT_EQ(paired_dots, 0u);
        }
    }

    /** @brief Submit one retained graph or the actual CPU stage and retire diagnostics. */
    void launch(Entry &entry)
    {
        requireGDNBundle(entry.signature == entry.stage->graphCaptureVariantSignature(), "Capture identity changed");
        if (device_.is_cpu()) requireGDNBundle(entry.stage->execute(&context_), "Actual CPU bundle failed");
        else requireGDNBundle(entry.graph->launch(), "Retained bundle replay failed");
    }

    /** @brief Observe completed original bytes, with failure-safe host destination lifetime. */
    std::vector<uint8_t> observe(const void *pointer, size_t bytes)
    {
        std::vector<uint8_t> result(bytes);
        if (!bytes) return result;
        if (device_.is_cpu()) std::memcpy(result.data(), pointer, bytes);
        else
        {
            GDNBundleCopyCompletion completion{backend_, device_, stream_};
            requireGDNBundle(backend_->deviceToHostOnStream(result.data(), pointer, bytes,
                device_.ordinal, stream_), "Bundle terminal observation failed");
        }
        return result;
    }

    /** @return Actual destination on the explicitly selected participant. */
    void *address(FP32Tensor &tensor) const
    {
        return device_.is_cpu() ? tensor.mutable_data() : tensor.gpu_data_ptr();
    }

    /** @brief Poison every destination/bank outside captures to expose omitted writes. */
    void poison()
    {
        for (auto &out : outputs_) fill(address(*out), out->numel() * sizeof(float), 0xff);
        for (const auto &name : {device_.is_cpu() ? kCPUProjectionQ8 : kGPUProjectionPrimaryQ8,
            device_.is_cpu() ? kCPUProjectionResidualQ8 : kGPUProjectionResidualQ8})
            fill(workspace_->getBuffer(name), workspace_->getBufferSize(name), 0xa5);
        if (!device_.is_cpu())
            for (const auto &name : {kGPUProjectionPrimaryQ8Scales, kGPUProjectionResidualQ8Scales})
                fill(workspace_->getBuffer(name), workspace_->getBufferSize(name), 0xff);
    }

    /** @brief Exact byte fill on host or the retained native producer stream. */
    void fill(void *pointer, size_t bytes, int pattern)
    {
        requireGDNBundle(pointer && bytes, "Required named bank missing");
        if (device_.is_cpu()) std::memset(pointer, pattern, bytes);
        else requireGDNBundle(backend_->memset(pointer, pattern, bytes, device_.ordinal, stream_), "Diagnostic poison failed");
    }

    /** @return One immutable family entry, never inferred from a runtime label. */
    Entry &entry(int m, bool grouped, bool corrected, bool serial = false)
    {
        const auto found = std::find_if(entries_.begin(), entries_.end(), [&](const auto &e) {
            return e.m == m && e.grouped == grouped && e.corrected == corrected && e.serial == serial;
        });
        requireGDNBundle(found != entries_.end(), "Missing exact bundle family");
        return *found;
    }

    /** @brief Compare active prefixes and inactive physical capacity without numerical tolerance. */
    void proveFrame(int m, int live, bool grouped, int pattern)
    {
        SCOPED_TRACE(::testing::Message() << "M=" << m << " live=" << live << " grouped=" << grouped << " tail=" << pattern);
        auto *input = input_->mutable_data();
        std::copy(original_input_.begin(), original_input_.end(), input);
        for (size_t i = size_t(live) * k_; i < input_->numel(); ++i)
            input[i] = pattern ? float(int(i % 17) - 8) / 16.f : 0.f;
        *static_cast<int32_t *>(count_->raw_mutable_data()) = live;
        publishInputs();
        std::array<std::vector<uint8_t>, 4> baseline;
        std::vector<uint8_t> primary, scales;
        const auto primary_name = device_.is_cpu() ? kCPUProjectionQ8 : kGPUProjectionPrimaryQ8;
        const auto residual_name = device_.is_cpu() ? kCPUProjectionResidualQ8 : kGPUProjectionResidualQ8;
        const size_t bank_row = device_.is_cpu() ? size_t(k_ / 32) * sizeof(Q8_1Block) : size_t(k_);
        for (bool corrected : {false, true})
        {
            poison();
            auto &current = entry(m, grouped, corrected);
            launch(current);
            std::array<std::vector<uint8_t>, 4> got;
            for (size_t i = 0; i < 4; ++i)
                got[i] = observe(address(*outputs_[i]), outputs_[i]->numel() * sizeof(float));
            const int sibling_live = !device_.is_cpu() && grouped && m > 1 ? live : m;
            for (size_t i = 0; i < 4; ++i)
            {
                const int active = i == 0 && corrected ? live : sibling_live;
                bool finite = true;
                for (size_t element = 0; element < size_t(active) * columns_[i]; ++element)
                {
                    float value;
                    std::memcpy(&value, got[i].data() + element * sizeof(float), sizeof(float));
                    finite = std::isfinite(value) && finite;
                }
                EXPECT_TRUE(finite) << "Unwritten/nonfinite active projection " << i;
            }
            auto q0 = observe(workspace_->getBuffer(primary_name), size_t(live) * bank_row);
            auto s0 = device_.is_cpu() ? std::vector<uint8_t>{} : observe(workspace_->getBuffer(kGPUProjectionPrimaryQ8Scales), size_t(live) * (k_ / 32) * sizeof(float));
            if (!corrected)
            {
                baseline = got; primary = std::move(q0); scales = std::move(s0);
                continue;
            }
            EXPECT_EQ(primary, q0) << "Actual consumed primary codes/half scales changed";
            EXPECT_EQ(scales, s0) << "Actual native primary scales changed";
            for (size_t i : {1u, 2u, 3u})
                EXPECT_EQ(baseline[i], got[i]) << "Sibling policy/output bytes changed: " << i;
            const size_t qkv_live = size_t(live) * columns_[0] * sizeof(float);
            EXPECT_TRUE(std::all_of(got[0].begin() + qkv_live, got[0].end(), [](auto byte) { return byte == 0xff; }));
            const auto q1_tail = observe(static_cast<const uint8_t *>(workspace_->getBuffer(residual_name)) + size_t(live) * bank_row,
                size_t(15 - live) * bank_row);
            EXPECT_TRUE(std::all_of(q1_tail.begin(), q1_tail.end(), [](auto byte) { return byte == 0xa5; }));
            if (!device_.is_cpu())
            {
                const size_t scale_row = size_t(k_ / 32) * sizeof(float);
                const auto tail = observe(static_cast<const uint8_t *>(workspace_->getBuffer(kGPUProjectionResidualQ8Scales)) + size_t(live) * scale_row,
                    size_t(15 - live) * scale_row);
                EXPECT_TRUE(std::all_of(tail.begin(), tail.end(), [](auto byte) { return byte == 0xff; }));
            }
            const auto q1 = observe(workspace_->getBuffer(residual_name), size_t(live) * bank_row);
            const auto s1 = device_.is_cpu() ? std::vector<uint8_t>{} : observe(
                workspace_->getBuffer(kGPUProjectionResidualQ8Scales), size_t(live) * (k_ / 32) * sizeof(float));
            const auto key = std::tuple{m, grouped, live};
            const std::array planes{std::vector<uint8_t>(got[0].begin(), got[0].begin() + qkv_live), q0, s0, q1, s1};
            const auto [saved, inserted] = tail_records_.try_emplace(key, planes);
            if (!inserted)
            {
                EXPECT_EQ(saved->second, planes) << "Active QKV/output/operand bytes depend on finite padded tail";
            }
            if (live)
            {
                EXPECT_NE(std::memcmp(baseline[0].data(), got[0].data(), qkv_live), 0)
                    << "Selected correction was overwritten by the original sibling group";
            }
            // Same backend's genuine complete M1 stage is the byte oracle.
            // Ordinary siblings retain their existing batched arithmetic; only
            // grouped M>1 siblings promise serial-row identity.
            for (int row = 0; row < live; ++row)
            {
                std::memcpy(serial_input_->mutable_data(), input + size_t(row) * k_, size_t(k_) * sizeof(float));
                if (!device_.is_cpu())
                {
                    TransferEngine::publishHostWrite(serial_input_.get());
                    TransferEngine::prepareDeviceInput(serial_input_.get(), device_, stream_);
                    TransferEngine::requireDeviceInput(serial_input_.get(), device_, stream_);
                }
                auto &serial = entry(1, false, true, true);
                launch(serial);
                for (size_t i = 0; i < 4; ++i)
                {
                    if (i && !(grouped && m > 1)) continue;
                    const auto expected = observe(address(*serial_outputs_[i]), size_t(columns_[i]) * sizeof(float));
                    EXPECT_EQ(std::memcmp(got[i].data() + size_t(row) * columns_[i] * sizeof(float),
                        expected.data(), expected.size()), 0) << "Serial four-source row=" << row << " projection=" << i;
                }
            }
            current.stage->resetSessionStatePreservingCapturedReplay();
            EXPECT_EQ(current.signature, current.stage->graphCaptureVariantSignature());
            EXPECT_EQ(workspace_->used(), workspace_used_);
            ASSERT_FALSE(::testing::Test::HasFailure()) << "Retire failed component; do not replay invalid evidence";
        }
    }

    /** @brief Original SingleQ8 bundle matches each actual public projection entrypoint. */
    void proveDefaultGrouping()
    {
        std::copy(original_input_.begin(), original_input_.end(), input_->mutable_data());
        const std::array<const char *, 4> names{"qkv", "z", "alpha", "beta"};
        for (int m : {1, 9, 15})
            for (bool grouped : {false, true})
            {
                SCOPED_TRACE(::testing::Message() << "original SingleQ8 M=" << m << " grouped=" << grouped);
                *static_cast<int32_t *>(count_->raw_mutable_data()) = m;
                publishInputs();
                poison();
                launch(entry(m, grouped, false));
                std::array<std::vector<uint8_t>, 4> baseline;
                for (size_t i = 0; i < 4; ++i)
                    baseline[i] = observe(address(*outputs_[i]), outputs_[i]->numel() * sizeof(float));
                for (size_t i = 0; i < 4; ++i)
                {
                    fill(address(*outputs_[i]), outputs_[i]->numel() * sizeof(float), 0xff);
                    if (stream_) kernels_[i]->setGPUStream(stream_);
                    const std::vector<ITensorGemm::TensorProjectionDesc> projection{
                        {kernels_[i], outputs_[i].get(), columns_[i], nullptr, names[i]}};
                    const auto rows = device_.is_cpu() ? DeviceRowRange::fullyActive(m)
                        : DeviceRowRange::deviceCounted(m, countPointer());
                    auto verifier = grouped && m > 1
                        ? kernels_[i]->beginVerifierDecodeEquivalentScope(rows) : nullptr;
                    const bool success = grouped && m > 1
                        ? kernels_[i]->multiply_fused_verifier_rows_decode_equivalent(
                            input_.get(), projection, m, k_, nullptr, workspace_.get())
                        : kernels_[i]->multiply_fused_tensor(
                            input_.get(), projection, m, k_, nullptr, workspace_.get());
                    requireGDNBundle(success, "Original public single projection failed");
                    EXPECT_EQ(baseline[i], observe(address(*outputs_[i]), outputs_[i]->numel() * sizeof(float)))
                        << "Original SingleQ8 grouping changed projection " << i;
                }
                ASSERT_FALSE(::testing::Test::HasFailure());
            }
    }

    /** @brief Exact larger-bank hints must preserve this real stage's smaller K and full M. */
    void proveWorkspaceHints()
    {
        const auto &original = entry(15, false, true);
        EXPECT_NE(original.signature, 0u);
        EXPECT_EQ(entry(15, false, false).signature, 0u);
        auto params = original.stage->getParams();
        params.qkv_row_range = DeviceRowRange::fullyActive(15);
        GDNProjectionStage different_rows(params);
        EXPECT_NE(original.signature, different_rows.graphCaptureVariantSignature());
        auto alternate = TestTensorFactory::createFP32Zeros({15, size_t(columns_[3])});
        params = original.stage->getParams(); params.output_b = alternate.get();
        GDNProjectionStage different_destination(params);
        EXPECT_NE(original.signature, different_destination.graphCaptureVariantSignature());
        for (int m : {9, 15})
        {
            auto &selected = *entry(m, false, true).stage;
            const auto hinted = selected.getWorkspaceRequirements(1, 0, 2 * k_);
            const auto ordinary = entry(m, false, false).stage->getWorkspaceRequirements(m, 0, 2 * k_);
            const auto residual = kernels_[0]->q8ResidualCorrectionWorkspaceRequirements(m, columns_[0], k_);
            const auto covers = [&](const WorkspaceRequirements &expected)
            {
                for (const auto &bank : expected.buffers)
                {
                    const auto actual = std::find_if(hinted.buffers.begin(), hinted.buffers.end(),
                        [&](const auto &item) { return item.name == bank.name; });
                    ASSERT_NE(actual, hinted.buffers.end()) << bank.name;
                    EXPECT_GE(actual->size_bytes, bank.size_bytes) << bank.name;
                }
            };
            covers(ordinary);
            covers(residual);
            for (const auto &bank : residual.buffers)
            {
                if (bank.name != kCPUProjectionResidualQ8 && bank.name != kGPUProjectionResidualQ8 &&
                    bank.name != kGPUProjectionResidualQ8Scales) continue;
                const auto actual = std::find_if(hinted.buffers.begin(), hinted.buffers.end(),
                    [&](const auto &item) { return item.name == bank.name; });
                ASSERT_NE(actual, hinted.buffers.end());
                EXPECT_EQ(actual->size_bytes, bank.size_bytes) << "Correction must use original K, not wider shared hint";
            }
        }
    }

    /** @brief Reject malformed original sources/rows before any destination or bank write. */
    void rejectMalformedBeforeWrites()
    {
        const auto base = entry(15, true, true).stage->getParams();
        poison();
        std::array<std::vector<uint8_t>, 4> before;
        for (size_t i = 0; i < 4; ++i) before[i] = observe(address(*outputs_[i]), outputs_[i]->numel() * sizeof(float));
        const auto primary_name = device_.is_cpu() ? kCPUProjectionQ8 : kGPUProjectionPrimaryQ8;
        const auto residual_name = device_.is_cpu() ? kCPUProjectionResidualQ8 : kGPUProjectionResidualQ8;
        const auto primary_before = observe(workspace_->getBuffer(primary_name), workspace_->getBufferSize(primary_name));
        const auto residual_before = observe(workspace_->getBuffer(residual_name), workspace_->getBufferSize(residual_name));
        auto bad = base;
        bad.qkv_operand_precision = static_cast<QuantizedActivationOperandPrecision>(255);
        EXPECT_THROW({ GDNProjectionStage rejected(bad); }, std::invalid_argument);
        bad = base; bad.qkv_row_range.reset();
        EXPECT_THROW({ GDNProjectionStage rejected(bad); }, std::invalid_argument);
        bad = base; bad.qkv_row_range = DeviceRowRange::deviceCounted(5, countPointer()).repeatEachRow(3);
        EXPECT_THROW({ GDNProjectionStage rejected(bad); }, std::invalid_argument);
        bad = base; bad.qkv_row_range = DeviceRowRange::deviceCounted(30, countPointer()).slice(0, 15);
        EXPECT_THROW({ GDNProjectionStage rejected(bad); }, std::invalid_argument);
        bad = base; bad.qkv_row_range = DeviceRowRange::deviceCounted(16, countPointer()).slice(1, 15);
        EXPECT_THROW({ GDNProjectionStage rejected(bad); }, std::invalid_argument);
        bad = base; bad.verifier_row_range = DeviceRowRange::deviceCounted(15, countPointer()).slice(1, 14);
        EXPECT_THROW({ GDNProjectionStage rejected(bad); }, std::invalid_argument);
        if (!device_.is_cpu())
        {
            bad = base; bad.verifier_row_range.reset();
            EXPECT_THROW({ GDNProjectionStage rejected(bad); }, std::invalid_argument);
        }
        // A second original scalar is valid storage but cannot silently replace
        // this immutable bundle's common authority, even if its value matches.
        INT32Tensor foreign_count(std::vector<size_t>{1});
        bad = base; bad.qkv_row_range = DeviceRowRange::deviceCounted(15,
            static_cast<const int32_t *>(foreign_count.raw_data()));
        EXPECT_THROW({ GDNProjectionStage rejected(bad); }, std::invalid_argument);
        Q4_0Tensor wrong_qkv(std::vector<size_t>{size_t(columns_[0]), size_t(k_)},
            std::vector<uint8_t>(size_t(columns_[0]) * (k_ / 32) * sizeof(Q4_0Block)));
        bad = base; bad.w_qkv = &wrong_qkv;
        EXPECT_THROW({ GDNProjectionStage rejected(bad); }, std::invalid_argument);
        bad = base; bad.n_b += 1;
        EXPECT_THROW({ GDNProjectionStage rejected(bad); }, std::invalid_argument);
        auto short_beta = TestTensorFactory::createFP32Zeros({14, size_t(columns_[3])});
        bad = base; bad.output_b = short_beta.get();
        EXPECT_THROW({ GDNProjectionStage rejected(bad); }, std::invalid_argument);
        INT32Tensor integer_beta(std::vector<size_t>{15, size_t(columns_[3])});
        bad = base; bad.output_b = &integer_beta;
        EXPECT_THROW({ GDNProjectionStage rejected(bad); }, std::invalid_argument);
        auto wrong_beta = TestTensorFactory::createFP32Zeros({size_t(columns_[3]), size_t(k_ + 32)});
        bad = base; bad.w_b = wrong_beta.get();
        EXPECT_THROW({ GDNProjectionStage rejected(bad); }, std::invalid_argument);
        const auto rejects_binding = [&](GDNProjectionStage::Params params)
        {
            GDNProjectionStage rejected(params);
            std::string error;
            EXPECT_FALSE(rejected.validatePreparedWeights(&error));
            EXPECT_FALSE(error.empty());
            const auto attempt = [&]
            {
                struct Unbind
                {
                    GDNProjectionStage &stage;
                    /** @brief Revoke even a partially bound invalid stage. */
                    ~Unbind() { stage.unbindWorkspace(); }
                } unbind{rejected};
                rejected.bindWorkspace(workspace_.get());
                (void)rejected.execute(&context_);
            };
            EXPECT_THROW(attempt(), std::invalid_argument);
        };
        bad = base; bad.prepared_store = nullptr;
        rejects_binding(bad);
        bad = base; bad.prepared_ref_qkv = refs_[1];
        rejects_binding(bad);
        bad = base; bad.prepared_ref_b = refs_[2];
        rejects_binding(bad);
        bad = base; bad.gemm_qkv = kernels_[1];
        rejects_binding(bad);
        bad = base; bad.device_id = device_.is_cpu() ? DeviceId::cuda(0) : DeviceId::cpu();
        rejects_binding(bad);
        for (size_t i = 0; i < 4; ++i)
            EXPECT_EQ(before[i], observe(address(*outputs_[i]), outputs_[i]->numel() * sizeof(float)));
        EXPECT_EQ(primary_before, observe(workspace_->getBuffer(primary_name), primary_before.size()));
        EXPECT_EQ(residual_before, observe(workspace_->getBuffer(residual_name), residual_before.size()));
    }

    DeviceId device_; IDeviceContext &context_; IWorkerGPUContext *worker_; void *stream_;
    int k_; std::array<int, 4> columns_; IBackend *backend_ = nullptr;
    std::array<std::shared_ptr<TensorBase>, 4> originals_, weights_;
    std::array<std::vector<uint8_t>, 4> source_bytes_;
    std::vector<std::shared_ptr<LoadOrchestrator>> loaders_;
    PreparedWeightStore store_;
    std::array<PreparedWeightRef, 4> refs_;
    std::array<ITensorGemm *, 4> kernels_{};
    std::unique_ptr<FP32Tensor> input_, serial_input_;
    std::vector<float> original_input_;
    std::unique_ptr<INT32Tensor> count_;
    std::array<std::unique_ptr<FP32Tensor>, 4> outputs_, serial_outputs_;
    std::vector<Entry> entries_;
    WorkspaceRequirements requirements_;
    std::shared_ptr<PhysicalMemoryAuthority> memory_;
    std::unique_ptr<DeviceWorkspaceManager> workspace_;
    size_t workspace_used_ = 0;
    std::map<std::tuple<int, bool, int>, std::array<std::vector<uint8_t>, 5>> tail_records_;
    /** @brief Fence setup submissions even if this fixture constructor throws. */
    struct ConstructionFence
    {
        IBackend *&backend; DeviceId device; void *stream;
        /** @brief Runs before all earlier-declared roots on constructor failure. */
        ~ConstructionFence() noexcept
        {
            if (stream && backend && !backend->synchronizeStream(stream, device.ordinal)) std::terminate();
        }
    } construction_fence_{backend_, device_, stream_};
};
}
