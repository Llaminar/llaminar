/**
 * @file GDNRecurrenceStage.cpp
 * @brief Implementation of delta rule recurrence for GDN linear attention
 *
 * Pure glue: extracts raw pointers from tensors, validates them, and
 * delegates all computation to the ITensorGatedDeltaNet kernel.
 *
 * When Q, K, V identify one merged QKV buffer with row layout
 * `[Q | K | V]`, the stage passes that layout and its TP head mapping directly
 * through the mandatory backend contract. CPU recurrence consumes source-row
 * strides in place. GPU recurrence performs any required device-only layout
 * transform in caller-owned graph workspace on the exact stage stream.
 *
 * GPU path: The executor and TransferEngine prepare arena bindings on the
 * stage's explicit stream; this stage consumes device pointers only. Merged
 * Nonidentity merged M1 decode reads its original row directly. Ordinary
 * multi-row execution deinterleaves into admitted workspace. Grouped retained
 * verification reads the original merged rows through its immutable prepared
 * invocation and leases; it retains exact corrections for accepted publication.
 * No H2D/D2H copies occur in the hot path.
 *
 * CPU execution never materializes temporary Q/K/V matrices in the stage.
 *
 * Raw programs retain their kernel normalization/scaling. The explicit scalar
 * prepared contract instead consumes the original convolution producer's exact
 * compact Q/K activation; gates and every state/snapshot owner stay unchanged.
 */

#include "GDNRecurrenceStage.h"
#include "transfer/TransferEngine.h"
#include "backends/ExplicitGPUStream.h"
#include "kernels/common/CPURequestStateBankView.h"
#include "kernels/gdn/GDNDeinterleaveRows.h"
#include "GDNSpeculativeWorkspaceContract.h"
#include "../../../kernels/IKVCache.h"
#include "../../../execution/local_execution/device/DeviceWorkspaceManager.h"
#include "../../../execution/local_execution/device/WorkspaceDescriptor.h"
#include "../../../execution/local_execution/graph/GraphCaptureGuard.h"
#include "../../../tensors/Tensors.h"
#include "../../../tensors/TensorKernels.h"
#include "../../../utils/DebugEnv.h"
#include "../../../utils/Logger.h"
#include "../../../utils/PerfStatsCollector.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace llaminar2
{

    uint64_t GDNRecurrenceStage::graphCaptureVariantSignature() const
    {
        if (!params_.kernel)
            throw std::logic_error(
                "GDN recurrence capture identity requires its bound kernel state owner");
        const auto layout = params_.kernel->recurrenceStateLayout();
        if (!validGDNRecurrenceStateLayout(layout))
            throw std::invalid_argument(
                "GDN recurrence capture identity has an unknown persistent state layout");
        auto identity = (0x47444e4c00000000ULL | static_cast<uint64_t>(layout)) ^ gdnQKCaptureIdentity(params_.qk_contract);
        if (params_.gpu_retained_workspace)
        {
            const auto &g = params_.gpu_retained_workspace->geometry();
            for (auto word : {std::uint64_t(g.requests()), std::uint64_t(g.rowsPerRequest()), std::uint64_t(g.heads()),
                    std::uint64_t(g.keyWidth()), std::uint64_t(g.valueWidth()), std::uint64_t(g.encoding()),
                    std::uint64_t(g.retentionProgram().mode), std::uint64_t(g.retentionProgram().compact_min_live_rows),
                    std::uint64_t(params_.gpu_retained_workspace->terminalKind()), std::uint64_t{0x52455441494e4544}})
                identity ^= word + 0x9e3779b97f4a7c15ULL + (identity << 6) + (identity >> 2);
        }
        return identity;
    }

    std::array<const TensorBase *, 6> GDNRecurrenceStage::retainedTensors() const
    {
        return {dynamic_cast<const TensorBase *>(params_.Q), dynamic_cast<const TensorBase *>(params_.alpha),
            dynamic_cast<const TensorBase *>(params_.beta), dynamic_cast<const TensorBase *>(params_.A_log),
            dynamic_cast<const TensorBase *>(params_.dt_bias), dynamic_cast<const TensorBase *>(params_.output)};
    }

    gdn::MergedRecurrenceInput GDNRecurrenceStage::retainedSources() const
    {
        const auto &g = params_.gpu_retained_workspace->geometry();
        const auto tensors = retainedTensors();
        for (const auto *tensor : tensors)
            if (!tensor || tensor->native_type() != TensorType::FP32 || !tensor->gpu_data_ptr() ||
                tensor->current_device() != std::optional{params_.device_id})
                throw std::logic_error("Retained GDN requires original allocation-prebound FP32 operands on its device");
        const auto rows = params_.request_seq_lens_device
            ? DeviceRequestRowRanges::deviceCounted(g.requests(), g.rowsPerRequest(), params_.request_seq_lens_device)
            : DeviceRequestRowRanges::fullyActiveRequests(g.requests(), g.rowsPerRequest());
        return {.rows = rows, .key_heads = params_.n_k_heads > 0 ? params_.n_k_heads : params_.n_heads,
            .value_heads = g.heads(), .key_width = g.keyWidth(), .value_width = g.valueWidth(),
            .global_value_offset = params_.global_v_head_offset, .use_qk_l2norm = params_.use_qk_l2norm,
            .merged = static_cast<const float *>(tensors[0]->gpu_data_ptr()),
            .alpha = static_cast<const float *>(tensors[1]->gpu_data_ptr()),
            .beta_raw = static_cast<const float *>(tensors[2]->gpu_data_ptr()),
            .A_log = static_cast<const float *>(tensors[3]->gpu_data_ptr()),
            .dt_bias = static_cast<const float *>(tensors[4]->gpu_data_ptr()),
            .output = static_cast<float *>(const_cast<TensorBase *>(tensors[5])->gpu_data_ptr()),
            .initial_state = nullptr, .updated_state = nullptr, .snapshots = nullptr,
            .snapshot_stride_floats = 0, .max_snapshot_rows = 0,
            .recurrence_layout = params_.kernel->recurrenceStateLayout()};
    }

    void GDNRecurrenceStage::requireRetainedInvocation() const
    {
        if (!params_.gpu_retained_workspace || !retained_gpu_ || !retained_gpu_->invocation ||
            !bound_workspace_ || bound_workspace_ != retained_gpu_->workspace || !bound_workspace_->isAllocated() ||
            bound_workspace_->device() != device() || bound_workspace_->id() != retained_gpu_->workspace_owner_identity)
            throw std::logic_error("Retained GDN lost its exact prepared workspace owner");
        const auto &binding = retained_gpu_->invocation->binding();
        const auto &g = params_.gpu_retained_workspace->geometry();
        const auto &actual = binding.geometry;
        if (g.requests() != actual.requests() || g.rowsPerRequest() != actual.rowsPerRequest() ||
            g.heads() != actual.heads() || g.keyWidth() != actual.keyWidth() || g.valueWidth() != actual.valueWidth() ||
            g.encoding() != actual.encoding() || g.retentionProgram().mode != actual.retentionProgram().mode ||
            g.retentionProgram().compact_min_live_rows != actual.retentionProgram().compact_min_live_rows ||
            params_.gpu_retained_workspace->terminalKind() != binding.terminal_kind ||
            params_.seq_len != params_.gpu_retained_workspace->physicalRows() || params_.request_count != g.requests() ||
            params_.request_seq_len != g.rowsPerRequest() || !binding.capture_stream ||
            binding.workspace_owner_identity != retained_gpu_->workspace_owner_identity)
            throw std::logic_error("Retained GDN immutable geometry/output identity changed");
        const auto &lease = retained_gpu_->storage;
        if (!lease || lease->device() != device() || lease->data() != binding.storage.data() ||
            lease->sizeBytes() != binding.storage.size_bytes() ||
            bound_workspace_->getBuffer(speculativeStateSlotsBufferName()) != lease->data() ||
            bound_workspace_->getBufferSize(speculativeStateSlotsBufferName()) != lease->sizeBytes())
            throw std::logic_error("Retained GDN named storage changed after preparation");
        if (binding.terminal_kind == gdn::GDNDenseCompactTerminalKind::DenseDestination)
        {
            const auto &terminal = retained_gpu_->terminal;
            if (!terminal || terminal->device() != device() || terminal->data() != binding.terminal.data() ||
                terminal->sizeBytes() != binding.terminal.size_bytes() ||
                bound_workspace_->getBuffer(speculativeStateWorkBufferName()) != terminal->data() ||
                bound_workspace_->getBufferSize(speculativeStateWorkBufferName()) != terminal->sizeBytes())
                throw std::logic_error("Retained GDN terminal owner changed after preparation");
        }
        else if (!binding.terminal.empty() || binding.terminal.data() || retained_gpu_->terminal)
            throw std::logic_error("Retained-prefixes-only GDN cannot borrow a terminal destination");
        const auto tensors = retainedTensors();
        const auto now = retainedSources();
        const auto &frozen = retained_gpu_->invocation->captureInput().sources;
        const std::array<const void *, 6> pointers{now.merged, now.alpha, now.beta_raw, now.A_log, now.dt_bias, now.output};
        const std::array<const void *, 6> originals{frozen.merged, frozen.alpha, frozen.beta_raw, frozen.A_log, frozen.dt_bias, frozen.output};
        if (pointers != originals || now.rows.lengthOwner() != frozen.rows.lengthOwner() ||
            now.key_heads != frozen.key_heads || now.global_value_offset != frozen.global_value_offset ||
            now.use_qk_l2norm != frozen.use_qk_l2norm || now.recurrence_layout != frozen.recurrence_layout)
            throw std::logic_error("Retained GDN original source/length/axes identity changed");
        for (std::size_t i = 0; i < tensors.size(); ++i)
        {
            if (!retained_gpu_->operands[i] || tensors[i]->shape() != retained_gpu_->shapes[i])
                throw std::logic_error("Retained GDN tensor shape/physical borrower changed");
            TransferEngine{}.requireCapturedDeviceStorage(*retained_gpu_->operands[i], *tensors[i], device(), retained_gpu_->bytes[i]);
        }
    }

    bool GDNRecurrenceStage::prepareGraphLaunch(IDeviceContext *ctx, void *stream)
    {
        if (!params_.gpu_retained_workspace) return IComputeStage::prepareGraphLaunch(ctx, stream);
        (void)ExplicitGPUStream(stream);
        if (!ctx || ctx->deviceId() != device() || isGraphCaptureActive())
            throw std::logic_error("Retained GDN preparation requires its exact cold participant/stream boundary");
        setGPUStream(stream);
        if (retained_gpu_)
        {
            requireRetainedInvocation();
            if (retained_gpu_->invocation->binding().capture_stream != stream)
                throw std::logic_error("Retained GDN recording stream changed after preparation");
            // Repeated cold preparation authenticates backend-owned live-bank
            // and compiled scheduling identity too; no mutable shared binding.
            const auto observed = params_.kernel->prepareRetainedInvocation(
                retainedSources(), retained_gpu_->invocation->binding());
            if (!observed || !retained_gpu_->invocation->sameCaptureIdentity(*observed))
                throw std::logic_error("Retained GDN backend state/program identity changed");
            return true;
        }
        if (!bound_workspace_ || !bound_workspace_->isAllocated() || bound_workspace_->device() != device() ||
            !bound_workspace_->id() || !params_.kernel)
            throw std::logic_error("Retained GDN requires its original admitted workspace and canonical state owner");
        const auto &g = params_.gpu_retained_workspace->geometry();
        const auto sources = retainedSources();
        const auto tensors = retainedTensors();
        const auto merged_per_request = gdnDeinterleaveElementsPerRequest(sources.rows,
            sources.key_heads, sources.value_heads, sources.key_width, sources.value_width);
        if (merged_per_request <= 0) throw std::invalid_argument("Retained GDN merged source extent is invalid");
        RetainedGPUInvocation pending;
        pending.workspace = bound_workspace_;
        pending.workspace_owner_identity = bound_workspace_->id();
        // Admission above proves both source and expanded indexing; lease only
        // the ORIGINAL merged source width, never expanded Q/K scratch bytes.
        const auto source_width = 2 * std::size_t(sources.key_heads) * std::size_t(g.keyWidth()) +
            std::size_t(g.heads()) * std::size_t(g.valueWidth());
        pending.bytes = {gdn_workspace::detail::checkedMultiply(source_width,
                std::size_t(params_.seq_len) * sizeof(float), "retained merged input bytes"),
            g.gateFloats() * sizeof(float), g.gateFloats() * sizeof(float),
            std::size_t(g.heads()) * sizeof(float), std::size_t(g.heads()) * sizeof(float),
            g.outputFloats() * sizeof(float)};
        for (std::size_t i = 0; i < tensors.size(); ++i)
        {
            pending.shapes[i] = tensors[i]->shape();
            pending.operands[i] = TransferEngine{}.retainCapturedDeviceStorage(*tensors[i], device(), pending.bytes[i]);
        }
        const auto slots_bytes = bound_workspace_->getBufferSize(speculativeStateSlotsBufferName());
        if (slots_bytes < g.bankBytes() || slots_bytes % sizeof(float))
            throw std::logic_error("Retained GDN exact payload exceeds its admitted named storage");
        pending.storage = bound_workspace_->retainBuffer(speculativeStateSlotsBufferName(), 0, slots_bytes);
        std::span<float> terminal;
        if (params_.gpu_retained_workspace->terminalKind() == gdn::GDNDenseCompactTerminalKind::DenseDestination)
        {
            const auto bytes = bound_workspace_->getBufferSize(speculativeStateWorkBufferName());
            if (bytes < g.denseBankBytes() || bytes % sizeof(float))
                throw std::logic_error("Retained GDN required terminal exceeds its admitted named storage");
            pending.terminal = bound_workspace_->retainBuffer(speculativeStateWorkBufferName(), 0, bytes);
            terminal = {static_cast<float *>(pending.terminal->data()), bytes / sizeof(float)};
        }
        const gdn::GDNRetainedInvocationBinding binding{.geometry = g,
            .terminal_kind = params_.gpu_retained_workspace->terminalKind(),
            .storage = {static_cast<float *>(pending.storage->data()), slots_bytes / sizeof(float)},
            .terminal = terminal, .workspace_owner_identity = pending.workspace_owner_identity,
            .capture_stream = stream};
        pending.invocation = params_.kernel->prepareRetainedInvocation(sources, binding);
        if (!pending.invocation) throw std::logic_error("Retained GDN backend did not prepare its selected native program");
        retained_gpu_.emplace(std::move(pending));
        try { requireRetainedInvocation(); }
        catch (...) { retained_gpu_.reset(); throw; } // No partial preparation may advertise capture/publication readiness.
        // This is a successful cold install, not a replay/submission count.
        // Immutable inventory survives the benchmark's measured-window reset;
        // constructor policy selection alone cannot prove a native program exists.
        if (PerfStatsCollector::isDomainEnabled("gpu_graph_inventory"))
            PerfStatsCollector::addCounter("gpu_graph_inventory", "gdn_retained_prepared_programs", 1,
                "capture_setup", device().toString(),
                {{"program", g.retentionProgram().mode == gdn::GDNDenseRetentionMode::ExactCorrections
                    ? "exact_corrections" : g.retentionProgram().mode == gdn::GDNDenseRetentionMode::FullSnapshots
                    ? "full_snapshots" : "live_row_threshold"},
                 {"terminal", binding.terminal_kind == gdn::GDNDenseCompactTerminalKind::RetainedPrefixesOnly
                    ? "retained_prefixes_only" : "dense_destination"},
                 {"native_program", std::to_string(retained_gpu_->invocation->captureProgramIdentity())},
                 {"layer", std::to_string(params_.layer_idx)},
                 {"requests", std::to_string(g.requests())},
                 {"rows_per_request", std::to_string(g.rowsPerRequest())},
                 {"heads", std::to_string(g.heads())},
                 {"key_width", std::to_string(g.keyWidth())},
                 {"value_width", std::to_string(g.valueWidth())},
                 {"layout", std::string(gdnRecurrenceStateLayoutName(sources.recurrence_layout))}});
        return true;
    }

    bool GDNRecurrenceStage::executeRetainedInvocation()
    {
        requireRetainedInvocation();
        const auto execution = gpuExecution();
        if (execution.nativeStream() != retained_gpu_->invocation->binding().capture_stream)
            throw std::logic_error("Retained GDN execution escaped its prepared recording stream");
        if (!retained_gpu_->invocation->capture()) return false;
        if (PerfStatsCollector::isDomainEnabled("gdn_recurrence"))
            PerfStatsCollector::addCounter("gdn_recurrence", "retained_capture_submissions", 1,
                isGraphCaptureActive() ? "capture" : "execute", device().toString(),
                {{"program", params_.gpu_retained_workspace->geometry().retentionProgram().mode == gdn::GDNDenseRetentionMode::ExactCorrections
                    ? "exact_corrections" : params_.gpu_retained_workspace->geometry().retentionProgram().mode == gdn::GDNDenseRetentionMode::FullSnapshots
                    ? "full_snapshots" : "live_row_threshold"},
                 {"terminal", params_.gpu_retained_workspace->terminalKind() == gdn::GDNDenseCompactTerminalKind::RetainedPrefixesOnly
                    ? "retained_prefixes_only" : "dense_destination"},
                 {"layer", std::to_string(params_.layer_idx)}, {"rows", std::to_string(params_.seq_len)},
                 {"requests", std::to_string(params_.request_count)},
                 {"native_program", std::to_string(retained_gpu_->invocation->captureProgramIdentity())}});
        return !params_.sequence_state_cache || params_.sequence_state_cache->advanceRecurrentSequenceState(
            params_.layer_idx, params_.request_count, params_.request_seq_lens_device,
            params_.request_seq_len, execution.nativeStream());
    }

    bool GDNRecurrenceStage::publishRetainedInvocation(
        const gdn::GDNDenseAcceptedSelection &selection, void *stream)
    {
        requireRetainedInvocation();
        const auto execution = gpuExecutionForVerifierPublication(stream);
        if (!selection.matchesGeometry(params_.gpu_retained_workspace->geometry()))
            throw std::invalid_argument("Retained GDN publication selection has foreign physical geometry");
        if (!retained_gpu_->invocation->publish(selection, execution.nativeStream())) return false;
        if (PerfStatsCollector::isDomainEnabled("gdn_recurrence"))
            PerfStatsCollector::addCounter("gdn_recurrence", "retained_publication_submissions", 1,
                isGraphCaptureActive() ? "capture" : "execute", device().toString(),
                {{"layer", std::to_string(params_.layer_idx)}, {"requests", std::to_string(params_.request_count)}});
        return true;
    }

    void GDNRecurrenceStage::updateDynamicParams(int pos_offset, int seq_len)
    {
        (void)pos_offset;
        if (params_.gpu_retained_workspace && seq_len != params_.gpu_retained_workspace->geometry().rowsPerRequest())
            throw std::invalid_argument("Retained GDN cannot change physical rows behind its captured program");
        if(std::holds_alternative<GDNPreparedQKBinding>(params_.qk_contract)&&seq_len!=1)
            throw std::invalid_argument("Scalar prepared GDN recurrence cannot change captured row geometry");

        /*
         * The execution engine supplies a per-request sequence width, whereas
         * the grouped GDN kernel operates on one flattened [request, row]
         * tensor.  Keep both views synchronized so request boundaries remain
         * stable when the same graph object is prepared or replayed again.
         */
        params_.request_seq_len = seq_len;
        if (seq_len <= 0 ||
            params_.request_count <= 0 ||
            seq_len > std::numeric_limits<int>::max() / params_.request_count)
        {
            /*
             * Dynamic updates have a void interface.  Zero the flattened row
             * count on malformed geometry so execution fails closed instead
             * of indexing a multiplication result that overflowed `int`.
             */
            LOG_ERROR("[GDNRecurrenceStage] Invalid request-batched dynamic geometry: "
                      << "request_count=" << params_.request_count
                      << " request_seq_len=" << seq_len);
            params_.seq_len = 0;
            return;
        }
        params_.seq_len = params_.request_count * seq_len;
    }

    namespace
    {
        std::atomic<uint32_t> g_gdn_recurrence_workspace_slice_counter{0};
    } // namespace

    GDNRecurrenceStage::GDNRecurrenceStage(Params params)
        : IComputeStage(params.device_id),
          params_(std::move(params)),
          workspace_slice_id_(g_gdn_recurrence_workspace_slice_counter.fetch_add(1, std::memory_order_relaxed))
    {
        if (params_.gpu_retained_workspace)
        {
            const auto &g = params_.gpu_retained_workspace->geometry();
            if (!device().is_gpu() || !std::holds_alternative<GDNRawQKInput>(params_.qk_contract) ||
                params_.Q != params_.K || params_.K != params_.V ||
                params_.seq_len != params_.gpu_retained_workspace->physicalRows() || params_.request_count != g.requests() ||
                params_.request_seq_len != g.rowsPerRequest() || params_.n_heads != g.heads() ||
                params_.d_k != g.keyWidth() || params_.d_v != g.valueWidth() ||
                params_.speculative_state_slot_rows <= 0 || !params_.request_seq_lens_device)
                throw std::invalid_argument("Retained GDN stage differs from its exact grouped raw-input contract");
        }
        if(const auto *p=std::get_if<GDNPreparedQKBinding>(&params_.qk_contract))
        {
            const auto layout=p->layout();
            const auto *bank=dynamic_cast<const TensorBase*>(p->storage);
            const int keys=params_.n_k_heads>0?params_.n_k_heads:params_.n_heads;
            if(!device().is_rocm()||!params_.kernel||
                params_.kernel->recurrenceStateLayout()!=GDNRecurrenceStateLayout::ValueMajorKeyContiguous||
                params_.seq_len!=1||params_.request_count!=1||
                (params_.request_seq_len!=0&&params_.request_seq_len!=1)||!params_.Q||
                params_.Q!=params_.K||params_.K!=params_.V||
                p->key_heads!=keys||p->value_heads!=params_.n_heads||
                p->key_width!=params_.d_k||p->value_width!=params_.d_v||
                p->normalize_qk!=params_.use_qk_l2norm||
                !bank||bank->native_type()!=TensorType::FP32||bank->size_bytes()<layout.bytes||
                bank->home_device()!=device())
                throw std::invalid_argument("Scalar prepared GDN recurrence has an incomplete consumer binding");
        }
    }

    WorkspaceRequirements GDNRecurrenceStage::getWorkspaceRequirements(int m, int n, int k) const
    {
        (void)n;
        (void)k;

        WorkspaceRequirements reqs;
        if (params_.gpu_retained_workspace)
        {
            const auto footprint = params_.gpu_retained_workspace->footprint();
            reqs.buffers.push_back({speculativeStateSlotsBufferName(), footprint.slot_bytes, 256, true,
                WorkspaceExecutionRegime::Any, WorkspaceContentLifetime::SerialGraphFamily});
            if (footprint.work_bytes) reqs.buffers.push_back({speculativeStateWorkBufferName(), footprint.work_bytes, 256, true});
            return reqs;
        }
        if (params_.n_heads <= 0 || params_.d_k <= 0 || params_.d_v <= 0)
            return reqs;

        /*
         * `m` is dynamic rows per request, but merged-QKV deinterleave and
         * verifier snapshots cover the flattened request matrix. Preserve the
         * graph's total row count as the lower bound for every workspace.
         */
        const int max_seq_len = std::max(
            1,
            std::max(params_.seq_len, m > 0 ? m : 0));
        const int speculative_slot_rows = requestedSpeculativeStateSlotRows();
        if (speculative_slot_rows > 0)
        {
            // CPU retained storage follows the declared logical verifier ceiling,
            // never current/live rows or capacity inferred from alignment padding.
            const int rows=params_.device_id.is_cpu() ? speculative_slot_rows :
                std::min(speculative_slot_rows,max_seq_len);
            const auto footprint =
                gdn_workspace::recurrenceStateFootprint(
                    rows,
                    std::max(1, params_.request_count),
                    params_.n_heads,
                    params_.d_k,
                    params_.d_v,
                    params_.device_id.is_cpu() ? GDNStateStorage::CPUFixedEpoch : GDNStateStorage::DeviceDense);
            reqs.buffers.push_back({speculativeStateSlotsBufferName(),
                                    footprint.slot_bytes,
                                    256,
                                    true});
            if (params_.device_id.is_gpu())
            {
                reqs.buffers.push_back({speculativeStateWorkBufferName(),
                                        footprint.work_bytes,
                                        256,
                                        true});
            }
        }

        if (!params_.device_id.is_gpu())
            return reqs;

        // This immutable scalar input program never deinterleaves. Existing
        // speculative state banks above remain priced and owned exactly once.
        if(std::holds_alternative<GDNPreparedQKBinding>(params_.qk_contract))return reqs;
        const bool merged_qkv = params_.Q == params_.K && params_.K == params_.V;
        if (!merged_qkv)
            return reqs;

        const int n_k_heads = params_.n_k_heads > 0 ? params_.n_k_heads : params_.n_heads;
        const bool decode_zero_copy = max_seq_len == 1 &&
                                      n_k_heads == params_.n_heads &&
                                      params_.global_v_head_offset == 0;
        if (decode_zero_copy)
            return reqs;

        const size_t bytes = deinterleaveScratchFloats(max_seq_len) * sizeof(float);
        reqs.buffers.push_back({deinterleaveScratchBufferName(), bytes, 256, true});
        return reqs;
    }

    void GDNRecurrenceStage::bindWorkspace(DeviceWorkspaceManager *workspace)
    {
        retireCPUVerifierCandidates();
        if (params_.gpu_retained_workspace)
        {
            if (bound_workspace_ == workspace && retained_gpu_) { requireRetainedInvocation(); return; }
            retained_gpu_.reset();
            bound_workspace_ = workspace;
            return;
        }
        bound_workspace_ = workspace;
        bindKernelWorkspace();
    }

    void GDNRecurrenceStage::unbindWorkspace()
    {
        retireCPUVerifierCandidates();
        if (params_.gpu_retained_workspace) { retained_gpu_.reset(); bound_workspace_ = nullptr; return; }
        bound_workspace_ = nullptr;
        bindKernelWorkspace();
    }

    void GDNRecurrenceStage::bindKernelWorkspace()
    {
        if (params_.gpu_retained_workspace) return;
        if (!params_.kernel)
            return;

        float *scratch = nullptr;
        size_t scratch_floats = 0;
        const std::string scratch_name = deinterleaveScratchBufferName();
        if (bound_workspace_ && bound_workspace_->hasBuffer(scratch_name))
        {
            scratch = static_cast<float *>(bound_workspace_->getBuffer(scratch_name));
            scratch_floats = bound_workspace_->getBufferSize(scratch_name) / sizeof(float);
        }

        params_.kernel->bindDeinterleaveWorkspace(scratch, scratch_floats);

        const int capture_state_size = gdnRecurrenceStateFloats(params_.device_id.is_cpu() ? GDNStateStorage::CPUFixedEpoch : GDNStateStorage::DeviceDense, params_.n_heads, params_.d_k, params_.d_v);
        verifier_capture_state_size_bound_ = capture_state_size;

        const int speculative_slot_rows = requestedSpeculativeStateSlotRows();
        if (speculative_slot_rows <= 0)
        {
            // The recurrence kernel instance is shared through the hybrid KV
            // cache across graph shapes. Normal decode graphs must explicitly
            // clear verifier capture state that may have been bound by an
            // all-position speculative verifier graph; otherwise decode keeps
            // updating scratch state instead of the live recurrent state.
            clearKernelVerifierStateWorkspace();
            return;
        }

        verifier_capture_workspace_bound_ = false;
        speculative_state_work_bound_ = false;
        verifier_capture_rows_bound_ = 0;

        float *capture = nullptr;
        int capture_rows = 0;
        if (bound_workspace_ && bound_workspace_->hasBuffer(speculativeStateSlotsBufferName()))
        {
            const std::string capture_name = speculativeStateSlotsBufferName();
            capture = static_cast<float *>(bound_workspace_->getBuffer(capture_name));
            const size_t available_floats =
                bound_workspace_->getBufferSize(capture_name) / sizeof(float);
            if (params_.device_id.is_cpu()) {
                const CPUGDNStateLayout layout(params_.n_heads,params_.d_k,params_.d_v);
                const int requests=std::max(1,params_.request_count);
                const CPUSharedGDNLogicalCapacity capacity(speculative_slot_rows,requests);
                if (available_floats>=capacity.slotFloats(layout))
                    capture_rows=capacity.flatRows();
            }
            else if (capture_state_size > 0)
            {
                capture_rows = static_cast<int>(std::min<size_t>(
                    static_cast<size_t>(speculative_slot_rows),
                    available_floats / static_cast<size_t>(capture_state_size)));
            }
        }
        verifier_capture_workspace_bound_ =
            capture != nullptr && capture_rows > 0 && capture_state_size > 0;
        verifier_capture_rows_bound_ = capture_rows;
        params_.kernel->bindRecurrenceStateCaptureWorkspace(
            capture, capture_rows, capture_state_size, params_.kernel->recurrenceStateLayout());

        float *speculative_work = nullptr;
        int speculative_work_size = capture_state_size;
        if (bound_workspace_ && bound_workspace_->hasBuffer(speculativeStateWorkBufferName()))
        {
            const std::string work_name = speculativeStateWorkBufferName();
            const size_t available_floats =
                bound_workspace_->getBufferSize(work_name) / sizeof(float);
            const int required_work_slots =
                std::max(1, params_.request_count > 1 ? params_.request_count : 1);
            const size_t required_floats =
                static_cast<size_t>(required_work_slots) *
                static_cast<size_t>(std::max(0, capture_state_size));
            if (available_floats >= required_floats)
            {
                speculative_work = static_cast<float *>(bound_workspace_->getBuffer(work_name));
                speculative_work_size = static_cast<int>(std::min<size_t>(
                    available_floats,
                    static_cast<size_t>(std::numeric_limits<int>::max())));
            }
        }
        speculative_state_work_bound_ =
            speculative_work != nullptr || !params_.device_id.is_gpu();
        params_.kernel->bindRecurrenceSpeculativeWorkspace(
            speculative_work, speculative_work_size, params_.kernel->recurrenceStateLayout());
        if (params_.device_id.is_cpu())
            params_.kernel->bindCPUSharedStateCapture(capture,
                capture ? bound_workspace_->getBufferSize(speculativeStateSlotsBufferName())/sizeof(float) : 0,
                capture_rows);
    }

    void GDNRecurrenceStage::retireCPUVerifierCandidates()
    {
        if (!params_.device_id.is_cpu()) return;
        if (cpu_capture_generation_==std::numeric_limits<uint64_t>::max())
            throw std::overflow_error("CPU GDN stage transaction generation exhausted");
        ++cpu_capture_generation_;
        if (!params_.device_id.is_cpu() || !bound_workspace_ || verifier_capture_rows_bound_<=0 ||
            !bound_workspace_->hasBuffer(speculativeStateSlotsBufferName())) return;
        const int requests=std::max(1,params_.request_count);
        if (verifier_capture_rows_bound_%requests) return;
        const int width=verifier_capture_rows_bound_/requests;
        const CPUGDNStateLayout l(params_.n_heads,params_.d_k,params_.d_v);
        const size_t span=CPUSharedGDNSnapshots::headFloats(l,width);
        const size_t capacity=bound_workspace_->getBufferSize(speculativeStateSlotsBufferName())/sizeof(float);
        if (span>capacity/static_cast<size_t>(params_.n_heads)/requests) return;
        auto *slots=static_cast<float *>(bound_workspace_->getBuffer(speculativeStateSlotsBufferName()));
        if (!slots) return;
        for(int request=0;request<requests;++request)
            for(int head=0;head<params_.n_heads;++head)
                CPUSharedGDNSnapshots(l,slots+(static_cast<size_t>(request)*params_.n_heads+head)*span,width).retire();
    }

    void GDNRecurrenceStage::clearKernelVerifierStateWorkspace()
    {
        retireCPUVerifierCandidates();
        verifier_capture_workspace_bound_ = false;
        speculative_state_work_bound_ = false;
        verifier_capture_rows_bound_ = 0;
        verifier_capture_state_size_bound_ = gdnRecurrenceStateFloats(params_.device_id.is_cpu() ? GDNStateStorage::CPUFixedEpoch : GDNStateStorage::DeviceDense, params_.n_heads, params_.d_k, params_.d_v);

        if (!params_.kernel)
            return;

        params_.kernel->bindCPUSharedStateCapture(nullptr,0,0);
        params_.kernel->bindRecurrenceStateCaptureWorkspace(
            nullptr,
            0,
            verifier_capture_state_size_bound_,
            params_.kernel->recurrenceStateLayout());
        params_.kernel->bindRecurrenceSpeculativeWorkspace(
            nullptr, verifier_capture_state_size_bound_, params_.kernel->recurrenceStateLayout());
    }

    int GDNRecurrenceStage::effectivePrefillSeqLen() const
    {
        if (params_.seq_len <= 0)
            return 0;
        if (!prefill_replay_params_set_ || prefill_effective_seq_len_ <= 0)
            return params_.seq_len;
        return std::clamp(prefill_effective_seq_len_, 1, params_.seq_len);
    }

    bool GDNRecurrenceStage::shouldUseScalarRealLengthContract() const
    {
        /*
         * Scalar padded prefill and request batching have different state
         * owners.  A one-request graph must advance the primary recurrence
         * bank through chunkForwardWithEffectiveSeqLen(); only a genuine
         * multi-request graph may update the packed request-state bank.
         */
        return params_.seq_len > 1 && params_.request_count == 1 &&
               params_.request_seq_len == params_.seq_len &&
               params_.request_seq_lens_device != nullptr &&
               params_.kernel &&
               params_.kernel->supportsPaddedPrefillRealLength();
    }

    std::string GDNRecurrenceStage::workspaceStableId() const
    {
        const std::string role_prefix =
            params_.workspace_namespace.empty()
                ? std::string{}
                : params_.workspace_namespace + "_";
        if (params_.layer_idx >= 0)
        {
            const std::string layer_id = "layer" + std::to_string(params_.layer_idx);
            /*
             * Main GDN graphs historically name their buffers by layer only
             * (`..._layer0`).  Role namespaces are for sidecar/verifier graph
             * variants such as `MTP0_layer0`; when the namespace is already
             * the same layer identity, keep the stable main-graph name.
             */
            if (params_.workspace_namespace == layer_id)
                return layer_id;
            return role_prefix + layer_id;
        }
        return role_prefix + "slice" + std::to_string(workspace_slice_id_);
    }

    std::string GDNRecurrenceStage::deinterleaveScratchBufferName() const
    {
        /*
         * Every GDN layer in one graph role executes in dependency order and
         * can economically reuse one large deinterleave allocation. Different
         * graph roles may replay on independent streams, so their mutable Q/K/V
         * scratch must have distinct keys. Including the layer here would avoid
         * the race but multiply long-context scratch by the number of GDN
         * layers, which is unnecessary and prohibitively expensive.
         */
        if (params_.workspace_namespace.empty())
            return WS_DEINTERLEAVE_SCRATCH;
        return std::string(WS_DEINTERLEAVE_SCRATCH) + "_" +
               params_.workspace_namespace;
    }

    std::string GDNRecurrenceStage::speculativeStateSlotsBufferName() const
    {
        return std::string(params_.gpu_retained_workspace ? WS_RETAINED_STATE_SLOTS : WS_SPECULATIVE_STATE_SLOTS) + "_" + workspaceStableId();
    }

    std::string GDNRecurrenceStage::speculativeStateWorkBufferName() const
    {
        /*
         * Only GPU recurrence needs a mutable state bank. GPU layer dependencies
         * serialize its use within a graph role; independent roles retain names.
         * CPU recurrence builds retained checkpoints directly and has no work bank.
         */
        if (params_.workspace_namespace.empty())
            return WS_SPECULATIVE_STATE_WORK;
        return std::string(WS_SPECULATIVE_STATE_WORK) + "_" +
               params_.workspace_namespace;
    }

    int GDNRecurrenceStage::requestedSpeculativeStateSlotRows() const
    {
        if (params_.gpu_retained_workspace) return params_.gpu_retained_workspace->physicalRows();
        return std::max(
            params_.speculative_state_slot_rows,
            params_.verifier_state_capture_rows);
    }

    bool GDNRecurrenceStage::hasVerifierStateCapture() const
    {
        // Producer discovery is declarative, exactly as for legacy capture.
        // Actual capture/publication separately require the immutable prepared handle.
        if (params_.gpu_retained_workspace) return params_.kernel && verifierStateCaptureWorkspaceRequired();
        return params_.kernel &&
               verifierStateCaptureWorkspaceRequired() &&
               (params_.device_id.is_gpu() || params_.recurrence_state != nullptr);
    }

    bool GDNRecurrenceStage::verifierStateCaptureWorkspaceRequired() const
    {
        return requestedSpeculativeStateSlotRows() > 0 &&
               params_.n_heads > 0 &&
               params_.d_k > 0 &&
               params_.d_v > 0;
    }

    bool GDNRecurrenceStage::ensureVerifierStateCaptureWorkspaceBound() const
    {
        if (params_.gpu_retained_workspace) { requireRetainedInvocation(); return true; }
        if (!verifierStateCaptureWorkspaceRequired())
            return true;
        if (verifier_capture_workspace_bound_ && speculative_state_work_bound_)
            return true;

        LOG_ERROR("[GDNRecurrenceStage] Missing required verifier state capture workspace '"
                  << speculativeStateSlotsBufferName()
                  << "' or speculative state workspace '"
                  << speculativeStateWorkBufferName()
                  << "' (requested_rows=" << requestedSpeculativeStateSlotRows()
                  << ", bound_rows=" << verifier_capture_rows_bound_
                  << ", speculative_work_bound=" << speculative_state_work_bound_
                  << ", state_size=" << verifier_capture_state_size_bound_
                  << ", layer=" << params_.layer_idx
                  << ", device=" << params_.device_id.toString() << ")");
        return false;
    }

    float *GDNRecurrenceStage::cpuVerifierStateCaptureWorkspace() const
    {
        if (params_.device_id.is_gpu() ||
            verifier_capture_rows_bound_ <= 0 ||
            verifier_capture_state_size_bound_ <= 0)
        {
            return nullptr;
        }

        if (bound_workspace_ &&
            bound_workspace_->hasBuffer(speculativeStateSlotsBufferName()))
        {
            return static_cast<float *>(
                bound_workspace_->getBuffer(speculativeStateSlotsBufferName()));
        }

        return nullptr;
    }

    bool GDNRecurrenceStage::restoreCPUVerifierStateCaptureRowDirect(int row)
    {
        const CPUVerifierStateRestorePlan plan =
            planCPUVerifierStateRestoreRow(row);
        if (!plan.ready())
            return false;

        /*
         * CPU GDN kernels are shared backend objects. Rebinding that shared
         * object during parallel MTP publication is racy, so CPU publication
         * copies directly from this stage's manager-owned workspace binding.
         * GPU stages still delegate to the backend restore kernel because their
         * slots live on device and must stay ordered on the explicit stream.
         */
        if (plan.shared_gdn_prefix)
        {
            const auto &accepted=*plan.shared_gdn_prefix;
            const auto &t=accepted.transaction;
            const CPUGDNStateLayout layout(t.heads,t.dk,t.dv);
            const size_t span=CPUSharedGDNSnapshots::headFloats(layout,t.width);
            for(int head=0;head<t.heads;++head)
                CPUSharedGDNSnapshots(layout,t.slots+static_cast<size_t>(head)*span,t.width)
                    .materialize(accepted.row,static_cast<float *>(plan.destination)+static_cast<size_t>(head)*layout.headFloats());
            PerfStatsCollector::addCounter("kernel","cpu_gdn_shared_accepted_materializations",1,"publication","cpu");
            PerfStatsCollector::addCounter("kernel","cpu_gdn_shared_accepted_materialization_bytes",
                static_cast<double>(plan.bytes),"publication","cpu");
        }
        else
        {
            std::memcpy(plan.destination,plan.source,plan.bytes);
        }
        return true;
    }

    CPUVerifierStateRestorePlan
    GDNRecurrenceStage::planCPUVerifierStateRestoreRow(int row)
    {
        if (params_.device_id.is_gpu())
            return {};
        if (!hasVerifierStateCapture() ||
            !ensureVerifierStateCaptureWorkspaceBound())
        {
            return {
                .status = CPUVerifierStateRestorePlanStatus::Invalid,
            };
        }
        if (row < 0)
        {
            return {
                .status = CPUVerifierStateRestorePlanStatus::NoOp,
            };
        }
        if (row >= verifier_capture_rows_bound_ ||
            verifier_capture_state_size_bound_ <= 0)
        {
            return {
                .status = CPUVerifierStateRestorePlanStatus::Invalid,
            };
        }

        const float *capture = cpuVerifierStateCaptureWorkspace();
        if (!capture || !params_.recurrence_state)
        {
            return {
                .status = CPUVerifierStateRestorePlanStatus::Invalid,
            };
        }

        const CPUGDNStateLayout layout(params_.n_heads,params_.d_k,params_.d_v);
        const int requests=std::max(1,params_.request_count);
        if (requests!=1) return {.status=CPUVerifierStateRestorePlanStatus::Invalid};
        const int width=verifier_capture_rows_bound_;
        const size_t span=CPUSharedGDNSnapshots::headFloats(layout,width);
        const size_t slots=bound_workspace_->getBufferSize(speculativeStateSlotsBufferName())/sizeof(float);
        if (span>slots/static_cast<size_t>(params_.n_heads))
            return {.status=CPUVerifierStateRestorePlanStatus::Invalid};
        const size_t state_bytes=static_cast<size_t>(layout.floats())*sizeof(float);
        const size_t slot_bytes=bound_workspace_->getBufferSize(speculativeStateSlotsBufferName());
        if (CPUSharedGDNSnapshots::buffersOverlap(params_.recurrence_state,state_bytes,capture,slot_bytes))
            return {.status=CPUVerifierStateRestorePlanStatus::Invalid};
        for(int head=0;head<params_.n_heads;++head)
            if (!CPUSharedGDNSnapshots(layout,const_cast<float *>(capture)+static_cast<size_t>(head)*span,width).validRow(row))
                return {.status=CPUVerifierStateRestorePlanStatus::Invalid};
        CPUSharedGDNTransaction transaction{
            .slots=const_cast<float *>(capture),.slot_floats=slots,
            .width=width,.requests=1,
            .live_width=params_.request_seq_len>0 ? params_.request_seq_len : params_.seq_len,
            .heads=params_.n_heads,.dk=params_.d_k,.dv=params_.d_v,
            .generation_authority=&cpu_capture_generation_,.generation=cpu_capture_generation_};
        return {.status=CPUVerifierStateRestorePlanStatus::Ready,
            .destination=params_.recurrence_state,.source=capture,
            .bytes=static_cast<size_t>(layout.floats())*sizeof(float),
            .kind=CPUVerifierStateRestorePlanKind::SharedGDNPrefixScatter,
            .shared_gdn_prefix=CPUSharedGDNAcceptedPrefix{transaction,row}};
    }

    bool GDNRecurrenceStage::planCPUVerifierStateRestoreRows(const int *rows,int requests,
        std::vector<CPUVerifierStateRestorePlan> &plans)
    {
        if(params_.device_id.is_gpu() || !rows || requests<=0 || requests!=params_.request_count ||
            !hasVerifierStateCapture() || !ensureVerifierStateCaptureWorkspaceBound())return false;
        if(requests==1)return IComputeStage::planCPUVerifierStateRestoreRows(rows,requests,plans);
        auto *owner=dynamic_cast<ICPURequestStateBankView *>(params_.kernel);
        if(!owner || !params_.recurrence_state)return false;
        const auto bank=owner->cpuRequestStateBankView();
        const auto *capture=cpuVerifierStateCaptureWorkspace();
        const int stride=params_.request_seq_len;
        const CPUGDNStateLayout layout(params_.n_heads,params_.d_k,params_.d_v);
        if(!capture || stride<=0 || !bank.contains(requests,bank.stateElements) ||
            bank.stateElements!=layout.floats() ||
            verifier_capture_state_size_bound_!=bank.stateElements ||
            verifier_capture_rows_bound_%requests ||
            (params_.request_seq_lens_host && params_.request_seq_lens_host->size()!=std::size_t(requests)))return false;
        const int width=verifier_capture_rows_bound_/requests;
        const size_t head_span=CPUSharedGDNSnapshots::headFloats(layout,width);
        const size_t slots=bound_workspace_->getBufferSize(speculativeStateSlotsBufferName())/sizeof(float);
        if(stride>width || head_span>slots/static_cast<size_t>(requests)/params_.n_heads)return false;
        const size_t request_span=head_span*static_cast<size_t>(params_.n_heads);
        // The current owner supplies addresses only. Each request borrows its
        // own immutable checkpoint/log slice: flattened candidate rows are not
        // full state snapshots. The publisher validates every slice and all
        // destination spans before scattering any accepted state.
        for(int request=0;request<requests;++request) {
            const int row=rows[request];
            const int live=params_.request_seq_lens_host ? (*params_.request_seq_lens_host)[request] : stride;
            const auto first=std::int64_t(request)*stride;
            if(live<0 || live>stride || (row>=0 && (row<first || row>=first+live || row>=verifier_capture_rows_bound_)))return false;
            if(row<0)continue;
            const int local_row=static_cast<int>(row-first);
            auto *selected=const_cast<float *>(capture)+static_cast<size_t>(request)*request_span;
            for(int head=0;head<params_.n_heads;++head)
                if(!CPUSharedGDNSnapshots(layout,selected+static_cast<size_t>(head)*head_span,width).validRow(local_row))
                    return false;
            CPUSharedGDNTransaction transaction{
                .slots=selected,.slot_floats=request_span,.width=width,.requests=1,
                .live_width=live,.heads=params_.n_heads,.dk=params_.d_k,.dv=params_.d_v,
                .generation_authority=&cpu_capture_generation_,.generation=cpu_capture_generation_};
            const auto bytes=std::size_t(bank.stateElements)*sizeof(float);
            auto *destination=bank.storage.data()+std::size_t(request)*bank.stateElements;
            plans.push_back({.status=CPUVerifierStateRestorePlanStatus::Ready,
                .destination=destination,.source=selected,.bytes=bytes,
                .kind=CPUVerifierStateRestorePlanKind::SharedGDNPrefixScatter,
                .shared_gdn_prefix=CPUSharedGDNAcceptedPrefix{transaction,local_row}});
            if(request==0 && params_.recurrence_state!=destination)
                plans.push_back({.status=CPUVerifierStateRestorePlanStatus::Ready,
                    .destination=params_.recurrence_state,.source=selected,.bytes=bytes,
                    .kind=CPUVerifierStateRestorePlanKind::SharedGDNPrefixScatter,
                    .shared_gdn_prefix=CPUSharedGDNAcceptedPrefix{transaction,local_row}});
        }
        return true;
    }

    bool GDNRecurrenceStage::restoreVerifierStateCaptureRow(int row, void *stream)
    {
        if (params_.gpu_retained_workspace)
            throw std::logic_error("Retained GPU verifier publication requires its existing ordered device selection");
        if (!hasVerifierStateCapture())
            return false;
        if (!params_.device_id.is_gpu())
            return restoreCPUVerifierStateCaptureRowDirect(row);
        /*
         * The GDN kernel object can be shared across graph instances for the
         * same layer.  Normal decode graphs clear speculative verifier slots,
         * and another verifier stage can bind its own workspace before this
         * publication call runs.  Rebind here so restoring an accepted verifier
         * row is owned by this stage, not by whatever graph touched the shared
         * kernel most recently.
         */
        bindKernelWorkspace();
        if (!ensureVerifierStateCaptureWorkspaceBound())
            return false;
        // GPU publication restores the kernel-owned live bank on the verifier
        // stream. GPU graph params deliberately contain no host state pointer.
        return params_.kernel->restoreVerifierStateCaptureRow(
            nullptr,
            row,
            stream ? stream : gpuStream());
    }

    bool GDNRecurrenceStage::restoreVerifierStateCaptureRows(
        const int *host_row_indices,
        int request_count,
        void *stream)
    {
        if (!hasVerifierStateCapture() || params_.device_id.is_gpu() ||
            !host_row_indices || request_count <= 0 || !params_.kernel)
        {
            return false;
        }
        if (!ensureVerifierStateCaptureWorkspaceBound() || !bound_workspace_ ||
            request_count!=std::max(1,params_.request_count) || verifier_capture_rows_bound_%request_count) return false;
        CPUSharedGDNTransaction transaction{
            .slots=cpuVerifierStateCaptureWorkspace(),
            .slot_floats=bound_workspace_->getBufferSize(speculativeStateSlotsBufferName())/sizeof(float),
            .width=verifier_capture_rows_bound_/request_count,.requests=request_count,
            .live_width=params_.request_seq_len>0 ? params_.request_seq_len : params_.seq_len,
            .heads=params_.n_heads,.dk=params_.d_k,.dv=params_.d_v,
            .generation_authority=&cpu_capture_generation_,.generation=cpu_capture_generation_};
        return params_.kernel->restoreCPUSharedStateRows(transaction,params_.recurrence_state,host_row_indices);
    }

    bool GDNRecurrenceStage::restoreVerifierStateCaptureRowFromDeviceIndex(
        std::span<const int> device_row_index_view,
        void *stream)
    {
        if (!deviceVerifierStateMetadataCovers(device_row_index_view, 1, 1)) return false;
        if (params_.gpu_retained_workspace)
        {
            if (params_.gpu_retained_workspace->geometry().requests() != 1)
                throw std::invalid_argument("Scalar retained GDN publication requires its scalar request geometry");
            return publishRetainedInvocation(gdn::GDNDenseAcceptedSelection::ownRequestFlattenedRows(
                params_.gpu_retained_workspace->geometry(), device_row_index_view.data(), device_row_index_view.size()), stream);
        }
        const int *device_row_index = device_row_index_view.data();
        if (!hasVerifierStateCapture() || !device_row_index || !stream)
            return false;
        // Device-resident publication has the same ownership requirement as
        // host-row publication: the stage must make its verifier slots current
        // before launching the backend restore kernel on the explicit stream.
        bindKernelWorkspace();
        if (!ensureVerifierStateCaptureWorkspaceBound())
            return false;
        if (DebugEnv::isTruthyEnv("LLAMINAR_MTP_PUBLICATION_DIAGNOSTICS"))
        {
            LOG_INFO("[MTPPublicationDiagnostics] phase=gdn_restore_device_row"
                     << " layer=" << params_.layer_idx
                     << " device=" << params_.device_id.toString()
                     << " capture_rows=" << verifier_capture_rows_bound_
                     << " state_size=" << verifier_capture_state_size_bound_
                     << " stream=" << stream
                     << " row_ptr=" << static_cast<const void *>(device_row_index));
        }
        // The accepted row index and destination live bank are both resident.
        return params_.kernel->restoreVerifierStateCaptureRowFromDeviceIndex(
            nullptr,
            device_row_index,
            stream);
    }

    bool GDNRecurrenceStage::restoreVerifierStateCaptureRowsFromDeviceIndices(
        std::span<const int> device_row_indices_view,
        int request_count,
        int row_index_stride,
        void *stream)
    {
        if (!deviceVerifierStateMetadataCovers(device_row_indices_view, request_count, row_index_stride)) return false;
        if (params_.gpu_retained_workspace)
        {
            if (request_count != params_.gpu_retained_workspace->geometry().requests())
                throw std::invalid_argument("Retained GDN publication request count changed");
            return publishRetainedInvocation(gdn::GDNDenseAcceptedSelection::ownRequestFlattenedRows(
                params_.gpu_retained_workspace->geometry(), device_row_indices_view.data(), device_row_indices_view.size(), row_index_stride), stream);
        }
        const int *device_row_indices = device_row_indices_view.data();
        if (!hasVerifierStateCapture() ||
            !device_row_indices ||
            request_count <= 0 ||
            row_index_stride <= 0 ||
            !stream)
        {
            return false;
        }
        if (!params_.device_id.is_gpu())
        {
            /*
             * CPU request banks consume host row indices through
             * restoreVerifierStateCaptureRows(). Device metadata remains a
             * GPU-only publication contract.
             */
            LOG_ERROR("[GDNRecurrenceStage] Device-indexed batch restore is GPU-only; CPU publication requires host rows");
            return false;
        }

        // Re-establish this verifier graph's capture/workspace binding before
        // the backend consumes row indices on the explicit stream.
        bindKernelWorkspace();
        if (!ensureVerifierStateCaptureWorkspaceBound())
            return false;
        if (DebugEnv::isTruthyEnv("LLAMINAR_MTP_PUBLICATION_DIAGNOSTICS"))
        {
            LOG_INFO("[MTPPublicationDiagnostics] phase=gdn_restore_device_rows"
                     << " layer=" << params_.layer_idx
                     << " device=" << params_.device_id.toString()
                     << " request_count=" << request_count
                     << " row_index_stride=" << row_index_stride
                     << " capture_rows=" << verifier_capture_rows_bound_
                     << " state_size=" << verifier_capture_state_size_bound_
                     << " stream=" << stream
                     << " row_ptr=" << static_cast<const void *>(device_row_indices));
        }

        return params_.kernel->restoreVerifierStateCaptureRowsFromDeviceIndices(
            nullptr,
            0,
            device_row_indices,
            request_count,
            row_index_stride,
            stream);
    }

    bool GDNRecurrenceStage::restoreVerifierStateCaptureRequestTerminalRows(
        const int *device_request_seq_lens,
        int request_count,
        int request_row_width,
        void *stream)
    {
        if (params_.gpu_retained_workspace)
            throw std::logic_error("Retained grouped GPU verifier publication requires an owner-backed device selection, not raw terminal lengths");
        if (!hasVerifierStateCapture() ||
            !params_.device_id.is_gpu() ||
            !device_request_seq_lens ||
            request_count <= 0 ||
            request_row_width <= 0 ||
            !stream)
        {
            return false;
        }

        bindKernelWorkspace();
        if (!ensureVerifierStateCaptureWorkspaceBound())
            return false;
        return params_.kernel->restoreVerifierStateCaptureRequestTerminalRows(
            /*dst_states=*/nullptr,
            device_request_seq_lens,
            request_count,
            request_row_width,
            stream);
    }

    bool GDNRecurrenceStage::requestBatchedTerminalStateCommittedDuringExecution(
        int request_count,
        int request_row_width) const
    {
        if (params_.gpu_retained_workspace) return false;
        if (!params_.kernel ||
            request_count <= 1 ||
            request_row_width <= 0 ||
            params_.request_count != request_count ||
            params_.request_seq_len != request_row_width)
        {
            return false;
        }
        const int state_size =
            gdnRecurrenceStateFloats(params_.device_id.is_cpu() ? GDNStateStorage::CPUFixedEpoch : GDNStateStorage::DeviceDense, params_.n_heads, params_.d_k, params_.d_v);
        return params_.kernel->supportsRequestLiveStateBank(
                   request_count,
                   state_size) &&
               verifier_capture_rows_bound_ < request_count * request_row_width;
    }

    void GDNRecurrenceStage::clearVerifierStateCaptureBindingAfterPublication()
    {
        if (params_.gpu_retained_workspace) { requireRetainedInvocation(); return; }
        if (!params_.device_id.is_gpu())
            return;

        /*
         * Device-indexed MTP publication has copied the accepted capture row
         * into the backend-owned live GPU recurrence state.  Leave that live
         * state resident, but detach the verifier capture/work buffers from the
         * shared kernel object so the next ordinary decode row updates live
         * state rather than speculative scratch.
         */
        clearKernelVerifierStateWorkspace();
    }

    size_t GDNRecurrenceStage::deinterleaveScratchFloats(int seq_len) const
    {
        if (seq_len <= 0 || params_.n_heads <= 0 || params_.d_k <= 0 || params_.d_v <= 0)
            return 0;

        const size_t row_floats = static_cast<size_t>(params_.n_heads) *
                                  static_cast<size_t>((2 * params_.d_k) + params_.d_v);
        return static_cast<size_t>(seq_len) * row_floats;
    }

    bool GDNRecurrenceStage::ensureGpuDeinterleaveWorkspaceBound(int seq_len) const
    {
        const size_t required_bytes = deinterleaveScratchFloats(seq_len) * sizeof(float);
        const std::string scratch_name = deinterleaveScratchBufferName();
        if (required_bytes == 0)
        {
            LOG_ERROR("[GDNRecurrenceStage] Invalid merged-QKV deinterleave scratch shape"
                      << " seq_len=" << seq_len
                      << " n_heads=" << params_.n_heads
                      << " d_k=" << params_.d_k
                      << " d_v=" << params_.d_v);
            return false;
        }

        if (!bound_workspace_ ||
            !bound_workspace_->hasBuffer(scratch_name) ||
            bound_workspace_->getBufferSize(scratch_name) < required_bytes)
        {
            LOG_ERROR("[GDNRecurrenceStage] Missing required graph workspace buffer '"
                      << scratch_name << "' for merged-QKV GPU deinterleave"
                      << " (requested=" << required_bytes << " bytes"
                      << ", available="
                      << (bound_workspace_ && bound_workspace_->hasBuffer(scratch_name)
                              ? bound_workspace_->getBufferSize(scratch_name)
                              : 0)
                      << " bytes, seq_len=" << seq_len
                      << ", n_heads=" << params_.n_heads
                      << ", d_k=" << params_.d_k
                      << ", d_v=" << params_.d_v
                      << ", device=" << params_.device_id.toString() << ")");
            return false;
        }

        return true;
    }

    void GDNRecurrenceStage::updatePrefillReplayParams(const PrefillReplayParams &replay)
    {
        prefill_replay_params_set_ = true;
        prefill_bucket_seq_len_ = replay.bucket_seq_len > 0 ? replay.bucket_seq_len : params_.seq_len;
        const int real_seq_len = replay.real_seq_len > 0 ? replay.real_seq_len : params_.seq_len;
        prefill_effective_seq_len_ = std::clamp(real_seq_len, 1, std::max(1, params_.seq_len));
    }

    bool GDNRecurrenceStage::supportsPaddedPrefillRealLengthContract() const
    {
        if (params_.device_id.is_gpu())
        {
            if (!params_.kernel || !params_.request_seq_lens_device)
                return false;
            if (params_.request_count <= 1)
                return params_.kernel->supportsPaddedPrefillRealLength();

            const int state_size =
                gdnRecurrenceStateFloats(params_.device_id.is_cpu() ? GDNStateStorage::CPUFixedEpoch : GDNStateStorage::DeviceDense, params_.n_heads, params_.d_k, params_.d_v);
            return params_.kernel->supportsRequestLiveStateBank(
                params_.request_count,
                state_size);
        }
        return params_.kernel && params_.kernel->supportsPaddedPrefillRealLength();
    }

    bool GDNRecurrenceStage::supportsLazyPrefillGraphCapturePreflight() const
    {
        if (params_.seq_len == 1)
            return isGraphCapturable();

        if (params_.seq_len <= 1 || !params_.device_id.is_gpu() || !params_.kernel)
            return false;

        if (params_.device_id.is_cuda())
        {
    #ifdef HAVE_CUDA
            return true;
    #else
            return false;
#endif
        }

        if (params_.device_id.is_rocm())
        {
    #ifdef HAVE_ROCM
            return true;
    #else
            return false;
    #endif
        }

        return false;
    }

    bool GDNRecurrenceStage::supportsPaddedPrefillGraphCapturePreflight() const
    {
        return supportsLazyPrefillGraphCapturePreflight() &&
               supportsPaddedPrefillRealLengthContract();
    }

    // =========================================================================
    // Main execute
    // =========================================================================

    bool GDNRecurrenceStage::execute(IDeviceContext *ctx)
    {
        retireCPUVerifierCandidates();
        if (!ensureContext(ctx, "GDNRecurrenceStage"))
            return false;

        if (!ensureRequiredPointers("GDNRecurrenceStage",
                                    {{"Q", params_.Q},
                                     {"K", params_.K},
                                     {"V", params_.V},
                                     {"alpha", params_.alpha},
                                     {"beta", params_.beta},
                                     {"A_log", params_.A_log},
                                     {"dt_bias", params_.dt_bias},
                                     {"output", params_.output}}))
            return false;

        if (!params_.kernel)
        {
            LOG_ERROR("[GDNRecurrenceStage] kernel (ITensorGatedDeltaNet) not set");
            return false;
        }

        const PerfStatsCollector::Tags detail_tags{
            {"layer", std::to_string(params_.layer_idx)},
            {"seq_len", std::to_string(params_.seq_len)},
            {"request_count", std::to_string(params_.request_count)},
            {"capture_rows", std::to_string(requestedSpeculativeStateSlotRows())},
        };

        if (params_.gpu_retained_workspace) return executeRetainedInvocation();

        // Bind stage stream to kernel before execution.
        bindStageStream(params_.kernel);
        {
            PerfStatsCollector::ScopedTimer timer(
                "gdn_recurrence_cpu_detail",
                "bind_kernel_workspace",
                "execute",
                params_.device_id.toString(),
                detail_tags);
            bindKernelWorkspace();
        }
        if (!ensureVerifierStateCaptureWorkspaceBound())
            return false;

        auto *q_base = requireTensorBasePtr(params_.Q, "Q");
        auto *k_base = requireTensorBasePtr(params_.K, "K");
        auto *v_base = requireTensorBasePtr(params_.V, "V");
        auto *alpha_base = requireTensorBasePtr(params_.alpha, "alpha");
        auto *beta_base = requireTensorBasePtr(params_.beta, "beta");
        auto *alog_base = requireTensorBasePtr(params_.A_log, "A_log");
        auto *dtbias_base = requireTensorBasePtr(params_.dt_bias, "dt_bias");
        auto *out_base = requireTensorBasePtr(params_.output, "output");

        if (!q_base || !k_base || !v_base || !alpha_base || !beta_base ||
            !alog_base || !dtbias_base || !out_base)
            return false;

        /*
         * A raw recurrence pointer is part of the CPU execution contract only.
         * CUDA and ROCm implementations select and, before capture, allocate
         * their own persistent device state inside ITensorGatedDeltaNet. Their
         * graph params therefore carry a null pointer by design. Rejecting that
         * null here would prevent the backend owner from resolving its resident
         * state and would reintroduce the obsolete host-mirror dependency.
         */
        if (!device().is_gpu() && !params_.recurrence_state)
        {
            LOG_ERROR("[GDNRecurrenceStage] CPU recurrence_state is null");
            return false;
        }

        // =====================================================================
        // GPU path: keep data on-device, pass device pointers to kernel
        // =====================================================================
        if (device().is_gpu())
        {
            if(std::holds_alternative<GDNPreparedQKBinding>(params_.qk_contract)&&
                (params_.seq_len!=1||params_.request_count!=1||
                (params_.request_seq_len!=0&&params_.request_seq_len!=1)||
                 params_.kernel->recurrenceStateLayout()!=GDNRecurrenceStateLayout::ValueMajorKeyContiguous))
                throw std::invalid_argument("Prepared GDN consumer no longer matches its immutable scalar program");
            // Coherence is handled by the executor via bufferContract():
            //   - Arena inputs (QKV, alpha, beta) via prepareForRead
            //   - Model weights (A_log, dt_bias) via contract.weight_tensors
            //   - Output (ATTN_OUTPUT) via prepareForWrite + markWritten

            // Get device pointers
            const float *d_alpha = static_cast<const float *>(alpha_base->gpu_data_ptr());
            const float *d_beta = static_cast<const float *>(beta_base->gpu_data_ptr());
            const float *d_alog = static_cast<const float *>(alog_base->gpu_data_ptr());
            const float *d_dtbias = static_cast<const float *>(dtbias_base->gpu_data_ptr());
            float *d_output = static_cast<float *>(const_cast<TensorBase *>(out_base)->gpu_data_ptr());

            // Resolve Q, K, V device pointers — handle merged QKV case
            const float *d_q = nullptr;
            const float *d_k = nullptr;
            const float *d_v = nullptr;

            // Check if merged: when Q, K, V all point to the same tensor
            const bool merged_qkv = (params_.Q == params_.K && params_.K == params_.V);
            const int nkh = (params_.n_k_heads > 0) ? params_.n_k_heads : params_.n_heads;
            // This is immutable input geometry, not a recovery/capability branch.
            const bool prepared_qk=std::holds_alternative<GDNPreparedQKBinding>(params_.qk_contract);
            const bool direct_merged_m1 =
                merged_qkv && params_.seq_len == 1 &&
                (prepared_qk || nkh != params_.n_heads || params_.global_v_head_offset != 0);

            if (merged_qkv && !direct_merged_m1)
            {
                // Merged QKV buffer on device
                const float *d_merged = static_cast<const float *>(q_base->gpu_data_ptr());

                const int q_src_dim = nkh * params_.d_k;
                const int k_src_dim = nkh * params_.d_k;
                const int v_dim = params_.n_heads * params_.d_v;

                if (params_.seq_len == 1 && nkh == params_.n_heads && params_.global_v_head_offset == 0)
                {
                    // Decode + identity + offset=0: Q, K, V are contiguous sub-regions
                    // in a single row — just use offset pointers (zero-copy)
                    d_q = d_merged;
                    d_k = d_merged + q_src_dim;
                    d_v = d_merged + q_src_dim + k_src_dim;
                }
                else
                {
                    // Use GPU deinterleave kernel for all other cases
                    if (!ensureGpuDeinterleaveWorkspaceBound(params_.seq_len))
                        return false;

                    float *dq_mut = nullptr, *dk_mut = nullptr, *dv_mut = nullptr;
                    // Deinterleave borrows the same lengths as recurrence.
                    // The request stride remains physical even when an earlier
                    // request is empty or a retained verifier envelope shrinks.
                    const auto deinterleave_rows = params_.request_seq_lens_device && params_.seq_len > 1
                        ? DeviceRequestRowRanges::deviceCounted(
                              params_.request_count,
                              params_.request_count > 1 ? params_.request_seq_len : params_.seq_len,
                              params_.request_seq_lens_device)
                        : DeviceRequestRowRanges::fullyActive(params_.seq_len);
                    if (deinterleave_rows.physicalRows() != params_.seq_len)
                    {
                        LOG_ERROR("[GDNRecurrenceStage] Deinterleave request strides do not match the captured row matrix");
                        return false;
                    }
                    if (!params_.kernel->deinterleave_qkv_device(
                            d_merged, dq_mut, dk_mut, dv_mut,
                            deinterleave_rows, nkh, params_.n_heads,
                            params_.d_k, params_.d_v, params_.global_v_head_offset))
                    {
                        LOG_ERROR("[GDNRecurrenceStage] GPU deinterleave_qkv_device failed");
                        return false;
                    }
                    d_q = dq_mut;
                    d_k = dk_mut;
                    d_v = dv_mut;
                }

                LOG_TRACE("[GDNRecurrenceStage] GPU merged QKV: "
                          << params_.seq_len << "x" << (q_src_dim + k_src_dim + v_dim)
                          << " nkh=" << nkh << " n_heads=" << params_.n_heads
                          << " offset=" << params_.global_v_head_offset
                          << (params_.seq_len == 1 ? " (decode offset)" : " (kernel deinterleave)"));
            }
            else if (!direct_merged_m1)
            {
                // Separate Q, K, V tensors — already on device via executor coherence
                d_q = static_cast<const float *>(q_base->gpu_data_ptr());
                d_k = static_cast<const float *>(k_base->gpu_data_ptr());
                d_v = static_cast<const float *>(v_base->gpu_data_ptr());
            }

            bool ok;
            if (direct_merged_m1)
            {
                if (gdnDeinterleaveElementsPerRequest(
                        DeviceRequestRowRanges::fullyActive(1),
                        nkh, params_.n_heads, params_.d_k, params_.d_v) <= 0)
                {
                    LOG_ERROR("[GDNRecurrenceStage] Invalid merged decode geometry");
                    return false;
                }
                const auto stride = 2 * std::int64_t(nkh) * params_.d_k +
                    std::int64_t(params_.n_heads) * params_.d_v;
                if (stride <= 0 || stride > std::numeric_limits<int>::max())
                {
                    LOG_ERROR("[GDNRecurrenceStage] Merged decode row stride exceeds native indexing");
                    return false;
                }
                if(const auto *p=std::get_if<GDNPreparedQKBinding>(&params_.qk_contract))
                {
                    auto execution=gpuExecution();execution.requirePreparedInput(p->storage);
                    const auto *bank=dynamic_cast<const TensorBase*>(p->storage);
                    ok=params_.kernel->chunkForwardPreparedQK({nkh,params_.n_heads,params_.d_k,params_.d_v,
                        params_.global_v_head_offset,params_.use_qk_l2norm,
                        static_cast<const float*>(q_base->gpu_data_ptr()),static_cast<const float*>(bank->gpu_data_ptr()),
                        d_alpha,d_beta,d_alog,d_dtbias,d_output});
                }
                else ok = params_.kernel->chunkForwardMergedQKV(
                    static_cast<const float *>(q_base->gpu_data_ptr()),
                    static_cast<int>(stride), d_alpha, d_beta, d_alog, d_dtbias,
                    d_output, params_.recurrence_state, 1, nkh, params_.n_heads,
                    params_.d_k, params_.d_v, params_.global_v_head_offset,
                    params_.chunk_size, params_.use_qk_l2norm);
            }
            else if (params_.seq_len == 1)
            {
                LOG_TRACE("[GDNRecurrenceStage] GPU launch pointers layer=" << params_.layer_idx
                                                                            << " Q=" << static_cast<const void *>(d_q)
                                                                            << " K=" << static_cast<const void *>(d_k)
                                                                            << " V=" << static_cast<const void *>(d_v)
                                                                            << " alpha=" << static_cast<const void *>(d_alpha)
                                                                            << " beta=" << static_cast<const void *>(d_beta)
                                                                            << " A_log=" << static_cast<const void *>(d_alog)
                                                                            << " dt_bias=" << static_cast<const void *>(d_dtbias)
                                                                            << " output=" << static_cast<void *>(d_output)
                                                                            << " state=" << static_cast<void *>(params_.recurrence_state)
                                                                            << " heads=" << params_.n_heads
                                                                            << " d_k=" << params_.d_k
                                                                            << " d_v=" << params_.d_v
                                                                            << " seq=" << params_.seq_len);
                ok = params_.kernel->recurrent_step(
                    d_q, d_k, d_v,
                    d_alpha, d_beta,
                    d_alog, d_dtbias,
                    d_output, params_.recurrence_state,
                    params_.n_heads, params_.d_k, params_.d_v,
                    params_.use_qk_l2norm);
            }
            else
            {
                const bool request_batched =
                    params_.request_count > 1 &&
                    params_.request_seq_len > 0 &&
                    params_.seq_len == params_.request_count * params_.request_seq_len;
                if (params_.request_count > 1 && !request_batched)
                {
                    LOG_ERROR("[GDNRecurrenceStage] Request-batched GPU recurrence requires flattened "
                              "seq_len == request_count * request_seq_len"
                              << " (seq_len=" << params_.seq_len
                              << ", request_count=" << params_.request_count
                              << ", request_seq_len=" << params_.request_seq_len << ")");
                    return false;
                }
                const int effective_seq_len = effectivePrefillSeqLen();
                const bool padded_effective_len =
                    !request_batched &&
                    prefill_replay_params_set_ &&
                    effective_seq_len < params_.seq_len;
                const bool use_scalar_real_length_contract =
                    shouldUseScalarRealLengthContract();
                if (padded_effective_len && !use_scalar_real_length_contract)
                {
                    LOG_ERROR("[GDNRecurrenceStage] Padded prefill requires a backend real-length contract");
                    return false;
                }

                LOG_TRACE("[GDNRecurrenceStage] GPU launch pointers layer=" << params_.layer_idx
                                                                            << " Q=" << static_cast<const void *>(d_q)
                                                                            << " K=" << static_cast<const void *>(d_k)
                                                                            << " V=" << static_cast<const void *>(d_v)
                                                                            << " alpha=" << static_cast<const void *>(d_alpha)
                                                                            << " beta=" << static_cast<const void *>(d_beta)
                                                                            << " A_log=" << static_cast<const void *>(d_alog)
                                                                            << " dt_bias=" << static_cast<const void *>(d_dtbias)
                                                                            << " output=" << static_cast<void *>(d_output)
                                                                            << " state=" << static_cast<void *>(params_.recurrence_state)
                                                                            << " heads=" << params_.n_heads
                                                                            << " d_k=" << params_.d_k
                                                                            << " d_v=" << params_.d_v
                                                                            << " seq=" << params_.seq_len
                                                                            << " effective_seq=" << effective_seq_len);
                if (request_batched)
                {
                    if (!params_.request_seq_lens_device)
                    {
                        LOG_ERROR("[GDNRecurrenceStage] GPU recurrence with dynamic real lengths requires device-owned request lengths");
                        return false;
                    }
                    const int state_size = gdnRecurrenceStateFloats(params_.device_id.is_cpu() ? GDNStateStorage::CPUFixedEpoch : GDNStateStorage::DeviceDense, params_.n_heads, params_.d_k, params_.d_v);
                    if (!params_.kernel->supportsRequestLiveStateBank(params_.request_count, state_size))
                    {
                        LOG_ERROR("[GDNRecurrenceStage] Backend lacks device-owned live recurrence-state bank for "
                                  << params_.request_count << " requests");
                        return false;
                    }
                    ok = params_.kernel->chunkForwardBatchedRequestsWithDeviceSeqLens(
                        d_q, d_k, d_v,
                        d_alpha, d_beta,
                        d_alog, d_dtbias,
                        d_output, params_.recurrence_state,
                        params_.seq_len, params_.request_count, params_.request_seq_len,
                        params_.n_heads, params_.d_k, params_.d_v,
                        params_.chunk_size, params_.use_qk_l2norm,
                        params_.request_seq_lens_device);
                }
                else if (use_scalar_real_length_contract)
                {
                    /*
                     * Request zero's resident length scalar is graph-stable
                     * metadata, but the recurrence owner remains the primary
                     * scalar bank.  The effective-length kernel reads that
                     * scalar on device and commits the terminal real row
                     * directly into the state consumed by decode.
                     */
                    ok = params_.kernel->chunkForwardWithEffectiveSeqLen(
                        d_q, d_k, d_v,
                        d_alpha, d_beta,
                        d_alog, d_dtbias,
                        d_output, params_.recurrence_state,
                        params_.seq_len, params_.n_heads, params_.d_k, params_.d_v,
                        params_.chunk_size, params_.use_qk_l2norm,
                        params_.request_seq_lens_device);
                }
                else
                {
                    ok = params_.kernel->chunk_forward(
                        d_q, d_k, d_v,
                        d_alpha, d_beta,
                        d_alog, d_dtbias,
                        d_output, params_.recurrence_state,
                        params_.seq_len, params_.n_heads, params_.d_k, params_.d_v,
                        params_.chunk_size, params_.use_qk_l2norm);
                }
            }

            if (!ok)
            {
                LOG_ERROR("[GDNRecurrenceStage] GPU kernel failed");
                return false;
            }

            if (PerfStatsCollector::isDomainEnabled("gdn_recurrence"))
            {
                // This observes successful recording/submission, not replayed
                // tokens. No device read or extra native operation is needed.
                const char *layout = direct_merged_m1 ? "direct_merged_m1" :
                    !merged_qkv ? "planar" :
                    params_.seq_len == 1 ? "identity_row_alias" : "deinterleaved";
                PerfStatsCollector::addCounter(
                    "gdn_recurrence", "input_layout_submission", 1.0,
                    isGraphCaptureActive() ? "capture" : "execute",
                    device().toString(),
                    {{"layout", layout},
                     {"qk_preparation",prepared_qk?"producer_prepared":"raw"},
                     {"state_layout", std::string(gdnRecurrenceStateLayoutName(
                         params_.kernel->recurrenceStateLayout()))},
                     {"layer", std::to_string(params_.layer_idx)},
                     {"rows", std::to_string(params_.seq_len)},
                     {"key_heads", std::to_string(nkh)},
                     {"value_heads", std::to_string(params_.n_heads)}});
            }

            // A pipeline slice can contain no attention append at all. Its
            // cache elects the terminal recurrent layer to advance the same
            // canonical metadata used by checkpoint/restore. Padded graphs
            // consume resident lengths, never the padded physical row count.
            if (params_.sequence_state_cache &&
                !params_.sequence_state_cache->advanceRecurrentSequenceState(
                    params_.layer_idx, params_.request_count,
                    params_.seq_len == 1 ? nullptr : params_.request_seq_lens_device,
                    params_.request_seq_len > 0 ? params_.request_seq_len : params_.seq_len,
                    gpuStream()))
            {
                LOG_ERROR("[GDNRecurrenceStage] Recurrent sequence frontier publication failed");
                return false;
            }

            LOG_TRACE("[GDNRecurrenceStage] GPU layer=" << params_.layer_idx
                                                        << " seq_len=" << params_.seq_len
                                                        << " effective_seq_len=" << effectivePrefillSeqLen()
                                                        << " n_heads=" << params_.n_heads
                                                        << " d_k=" << params_.d_k
                                                        << " d_v=" << params_.d_v
                                                        << (params_.seq_len == 1 ? " (decode)" : " (prefill)"));

            return true;
        }

        // =====================================================================
        // CPU path: host-owned tensors, with merged rows consumed in place.
        // =====================================================================
        const float *q_data = q_base->data();
        const float *k_data = k_base->data();
        const float *v_data = v_base->data();
        const float *alpha_data = alpha_base->data();
        const float *beta_data = beta_base->data();
        const float *alog_data = alog_base->data();
        const float *dtbias_data = dtbias_base->data();
        float *output_data = out_base->mutable_data();

        // When Q, K, V identify one merged QKV buffer, preserve that layout.
        // Merged layout: [seq_len, q_dim + k_dim + v_dim] per row.
        // q_dim = k_dim = n_k_heads * d_k, v_dim = n_heads * d_v (n_heads = n_v_heads)
        //
        // Three modes depending on n_k_heads vs n_heads (n_v_heads_local):
        //   1) n_k < n_v_local: Expansion — modular GQA repeat (single-device, repeat_factor > 1)
        //   2) n_k == n_v_local: Identity deinterleave (TP where n_k is replicated and equals n_v_local)
        //   3) n_k > n_v_local: Selection — pick the correct K-head subset for each V-head (high-degree TP)
        //
        // The mandatory merged-layout backend contract owns these mappings.
        const bool merged_qkv = (q_data == k_data && k_data == v_data);

        // Effective key head count for QKV split (may be full count if Q/K replicated for TP)
        const int nkh = (params_.n_k_heads > 0) ? params_.n_k_heads : params_.n_heads;
        const bool request_batched =
            params_.request_count > 1 &&
            params_.request_seq_len > 0 &&
            params_.seq_len == params_.request_count * params_.request_seq_len;
        if (params_.request_count > 1 && !request_batched)
        {
            LOG_ERROR("[GDNRecurrenceStage] Request-batched CPU recurrence requires flattened "
                      "seq_len == request_count * request_seq_len");
            return false;
        }
        if (request_batched &&
            (!params_.request_seq_lens_host ||
             static_cast<int>(params_.request_seq_lens_host->size()) <
                 params_.request_count))
        {
            LOG_ERROR("[GDNRecurrenceStage] Request-batched CPU recurrence requires host request lengths");
            return false;
        }
        if (request_batched)
        {
            const int state_size =
                gdnRecurrenceStateFloats(params_.device_id.is_cpu() ? GDNStateStorage::CPUFixedEpoch : GDNStateStorage::DeviceDense, params_.n_heads, params_.d_k, params_.d_v);
            if (!params_.kernel->supportsRequestLiveStateBank(
                    params_.request_count, state_size))
            {
                LOG_ERROR("[GDNRecurrenceStage] CPU backend lacks request-owned live recurrence state");
                return false;
            }
        }
        bool ok = false;

        if (merged_qkv)
        {
            // Dimensions in the merged QKV buffer
            const int q_src_dim = nkh * params_.d_k;
            const int k_src_dim = nkh * params_.d_k;
            const int v_dim = params_.n_heads * params_.d_v;
            const int qkv_stride = q_src_dim + k_src_dim + v_dim;

            if (request_batched)
            {
                PerfStatsCollector::ScopedTimer timer(
                    "gdn_recurrence_cpu_detail",
                    "request_batched_merged_qkv",
                    "execute",
                    params_.device_id.toString(),
                    detail_tags);
                ok = params_.kernel->chunkForwardBatchedMergedQKVWithHostSeqLens(
                    q_data,
                    qkv_stride,
                    alpha_data,
                    beta_data,
                    alog_data,
                    dtbias_data,
                    output_data,
                    params_.recurrence_state,
                    params_.seq_len,
                    params_.request_count,
                    params_.request_seq_len,
                    nkh,
                    params_.n_heads,
                    params_.d_k,
                    params_.d_v,
                    params_.global_v_head_offset,
                    params_.use_qk_l2norm,
                    params_.request_seq_lens_host->data());
                if (!ok)
                {
                    LOG_ERROR("[GDNRecurrenceStage] CPU grouped merged-QKV request kernel failed");
                    return false;
                }
                return true;
            }

            /*
             * All-position verifier rows are bounded by the graph's declared
             * runtime capacity and need post-row state snapshots rather than a
             * generic prefill layout. Let CPU kernels that understand the merged
             * layout read Q/K/V slices directly so we do not spend more time
             * copying QKV than advancing the recurrence.
             */
            const bool direct_merged_verifier =
                params_.seq_len > 1 &&
                requestedSpeculativeStateSlotRows() > 0 &&
                verifier_capture_workspace_bound_ &&
                verifier_capture_state_size_bound_ > 0;
            if (direct_merged_verifier)
            {
                float *const verifier_state_slots =
                    cpuVerifierStateCaptureWorkspace();
                if (!verifier_state_slots || verifier_capture_rows_bound_ <= 0)
                {
                    LOG_ERROR("[GDNRecurrenceStage] CPU merged-QKV verifier requires manager-owned state slots");
                    return false;
                }

                const int effective_seq_len = effectivePrefillSeqLen();
                const int kernel_seq_len = std::min(effective_seq_len, params_.seq_len);
                {
                    PerfStatsCollector::ScopedTimer timer(
                        "gdn_recurrence_cpu_detail",
                        "direct_merged_verifier",
                        "execute",
                        params_.device_id.toString(),
                        detail_tags);
                    ok = params_.kernel->chunkForwardMergedQKVWithStateSnapshots(
                        q_data,
                        qkv_stride,
                        alpha_data,
                        beta_data,
                        alog_data,
                        dtbias_data,
                        output_data,
                        params_.recurrence_state,
                        kernel_seq_len,
                        nkh,
                        params_.n_heads,
                        params_.d_k,
                        params_.d_v,
                        params_.global_v_head_offset,
                        params_.chunk_size,
                        params_.use_qk_l2norm,
                        verifier_state_slots,
                        verifier_capture_state_size_bound_,
                        verifier_capture_rows_bound_);
                }
                if (!ok)
                {
                    LOG_ERROR("[GDNRecurrenceStage] CPU merged-QKV verifier kernel failed");
                    return false;
                }
                if (kernel_seq_len < params_.seq_len)
                {
                    const size_t first_pad =
                        static_cast<size_t>(kernel_seq_len) *
                        static_cast<size_t>(params_.n_heads * params_.d_v);
                    const size_t pad_count =
                        static_cast<size_t>(params_.seq_len - kernel_seq_len) *
                        static_cast<size_t>(params_.n_heads * params_.d_v);
                    std::memset(output_data + first_pad, 0, pad_count * sizeof(float));
                }
                return true;
            }

            const int effective_seq_len = effectivePrefillSeqLen();
            const int kernel_seq_len = params_.seq_len > 1
                                           ? effective_seq_len
                                           : params_.seq_len;
            const bool padded_effective_len =
                prefill_replay_params_set_ && effective_seq_len < params_.seq_len;
            if (padded_effective_len && !supportsPaddedPrefillRealLengthContract())
            {
                LOG_ERROR("[GDNRecurrenceStage] Padded CPU merged-QKV prefill "
                          "requires a real-length contract");
                return false;
            }

            {
                PerfStatsCollector::ScopedTimer timer(
                    "gdn_recurrence_cpu_detail",
                    params_.seq_len == 1
                        ? "direct_merged_decode"
                        : "direct_merged_prefill",
                    "execute",
                    params_.device_id.toString(),
                    detail_tags);
                ok = params_.kernel->chunkForwardMergedQKV(
                    q_data,
                    qkv_stride,
                    alpha_data,
                    beta_data,
                    alog_data,
                    dtbias_data,
                    output_data,
                    params_.recurrence_state,
                    kernel_seq_len,
                    nkh,
                    params_.n_heads,
                    params_.d_k,
                    params_.d_v,
                    params_.global_v_head_offset,
                    params_.chunk_size,
                    params_.use_qk_l2norm);
            }
            if (!ok)
            {
                LOG_ERROR("[GDNRecurrenceStage] Mandatory CPU merged-QKV kernel failed");
                return false;
            }
            if (kernel_seq_len < params_.seq_len)
            {
                const size_t first_pad =
                    static_cast<size_t>(kernel_seq_len) *
                    static_cast<size_t>(params_.n_heads * params_.d_v);
                const size_t pad_count =
                    static_cast<size_t>(params_.seq_len - kernel_seq_len) *
                    static_cast<size_t>(params_.n_heads * params_.d_v);
                std::memset(
                    output_data + first_pad,
                    0,
                    pad_count * sizeof(float));
            }
            return true;
        }

        if (request_batched)
        {
            PerfStatsCollector::ScopedTimer timer(
                "gdn_recurrence_cpu_detail",
                "request_batched_separate_qkv",
                "execute",
                params_.device_id.toString(),
                detail_tags);
            ok = params_.kernel->chunkForwardBatchedRequestsWithHostSeqLens(
                q_data, k_data, v_data,
                alpha_data, beta_data,
                alog_data, dtbias_data,
                output_data, params_.recurrence_state,
                params_.seq_len, params_.request_count,
                params_.request_seq_len,
                params_.n_heads, params_.d_k, params_.d_v,
                params_.chunk_size, params_.use_qk_l2norm,
                params_.request_seq_lens_host->data());
        }
        else if (params_.seq_len == 1)
        {
            PerfStatsCollector::ScopedTimer timer(
                "gdn_recurrence_cpu_detail",
                "recurrent_step",
                "execute",
                params_.device_id.toString(),
                detail_tags);
            ok = params_.kernel->recurrent_step(
                    q_data, k_data, v_data,
                    alpha_data, beta_data,
                    alog_data, dtbias_data,
                    output_data, params_.recurrence_state,
                    params_.n_heads, params_.d_k, params_.d_v,
                    params_.use_qk_l2norm);
        }
        else
        {
            const int effective_seq_len = effectivePrefillSeqLen();
            const int kernel_seq_len = params_.seq_len > 1 ? effective_seq_len : params_.seq_len;
            const bool padded_effective_len =
                prefill_replay_params_set_ && effective_seq_len < params_.seq_len;
            if (padded_effective_len && !supportsPaddedPrefillRealLengthContract())
            {
                LOG_ERROR("[GDNRecurrenceStage] Padded CPU prefill requires a real-length contract");
                return false;
            }

            {
                PerfStatsCollector::ScopedTimer timer(
                    "gdn_recurrence_cpu_detail",
                    "chunk_forward",
                    "execute",
                    params_.device_id.toString(),
                    detail_tags);
                ok = params_.kernel->chunk_forward(
                    q_data, k_data, v_data,
                    alpha_data, beta_data,
                    alog_data, dtbias_data,
                    output_data, params_.recurrence_state,
                    kernel_seq_len, params_.n_heads, params_.d_k, params_.d_v,
                    params_.chunk_size, params_.use_qk_l2norm);
            }

            if (ok && kernel_seq_len < params_.seq_len)
            {
                const size_t first_pad = static_cast<size_t>(kernel_seq_len) * static_cast<size_t>(params_.n_heads * params_.d_v);
                const size_t pad_count = static_cast<size_t>(params_.seq_len - kernel_seq_len) * static_cast<size_t>(params_.n_heads * params_.d_v);
                std::memset(output_data + first_pad, 0, pad_count * sizeof(float));
            }
        }

        if (!ok)
        {
            LOG_ERROR("[GDNRecurrenceStage] Kernel failed");
            return false;
        }

        LOG_TRACE("[GDNRecurrenceStage] layer=" << params_.layer_idx
                                                << " seq_len=" << params_.seq_len
                                                << " effective_seq_len=" << effectivePrefillSeqLen()
                                                << " n_heads=" << params_.n_heads
                                                << " d_k=" << params_.d_k
                                                << " d_v=" << params_.d_v
                                                << (params_.seq_len == 1 ? " (decode)" : " (prefill)"));

        return ok;
    }

    // =========================================================================
    // Estimation and metadata
    // =========================================================================

    size_t GDNRecurrenceStage::estimatedFlops() const
    {
        const size_t S = static_cast<size_t>(params_.seq_len);
        const size_t H = static_cast<size_t>(params_.n_heads);
        const size_t dk = static_cast<size_t>(params_.d_k);
        const size_t dv = static_cast<size_t>(params_.d_v);
        // Per timestep per head: decay(dk*dv) + kv_mem(dk*dv) + delta(dv) + rank1(dk*dv) + output(dk*dv)
        // ≈ 4*dk*dv + dv per step per head
        return S * H * (4 * dk * dv + dv);
    }

    size_t GDNRecurrenceStage::estimatedMemoryBytes() const
    {
        const size_t S = static_cast<size_t>(params_.seq_len);
        const size_t H = static_cast<size_t>(params_.n_heads);
        const size_t dk = static_cast<size_t>(params_.d_k);
        const size_t dv = static_cast<size_t>(params_.d_v);
        // State: H*dk*dv, Q: S*H*dk, K: S*H*dk, V: S*H*dv, output: S*H*dv
        return (H * dk * dv + S * H * (2 * dk + 2 * dv)) * sizeof(float);
    }

    bool GDNRecurrenceStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU:
            return true;
#ifdef HAVE_CUDA
        case ComputeBackendType::GPU_CUDA:
            return true;
#endif
#ifdef HAVE_ROCM
        case ComputeBackendType::GPU_ROCM:
            return true;
#endif
        default:
            return false;
        }
    }

    bool GDNRecurrenceStage::isGraphCapturable() const
    {
        if (params_.gpu_retained_workspace) return retained_gpu_ && retained_gpu_->invocation;
        // Decode has stable pointers, but an absent implementation cannot
        // advertise capture merely because its device identifier is a GPU.
        if (params_.seq_len == 1)
            return params_.device_id.is_cuda()
                ? supportsBackend(ComputeBackendType::GPU_CUDA)
                : params_.device_id.is_rocm() && supportsBackend(ComputeBackendType::GPU_ROCM);

        // Prefill: capturable only on compiled GPU backends when GPU state is pre-allocated.
        // chunk_forward() will skip its lazy allocateState() if state is ready.
        // Snapshot capture is rejected by PrefillGraphCache when a snapshot callback is active;
        // do not use the Integration-build macro here or focused graph tests cannot exercise
        // the release capture path.
        if (!params_.device_id.is_gpu() || !params_.kernel)
            return false;

        if (params_.device_id.is_cuda())
        {
#ifndef HAVE_CUDA
            return false;
#endif
        }
        else if (params_.device_id.is_rocm())
        {
#ifndef HAVE_ROCM
            return false;
#endif
        }
        else
        {
            return false;
        }

        const int required_state_size = gdnRecurrenceStateFloats(params_.device_id.is_cpu() ? GDNStateStorage::CPUFixedEpoch : GDNStateStorage::DeviceDense, params_.n_heads, params_.d_k, params_.d_v);
        return params_.kernel->isGPUStateReady(required_state_size);
    }

    StageDumpInfo GDNRecurrenceStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;

        // Use actual seq_len dimensions, not the buffer capacity
        const size_t rows = static_cast<size_t>(params_.seq_len);
        const size_t cols = static_cast<size_t>(params_.n_heads) * params_.d_v;
        const size_t alpha_beta_cols = static_cast<size_t>(params_.n_heads);
        if (params_.output)
            info.addOutput("output", params_.output, rows, cols);
        if (params_.alpha)
            info.addOutput("alpha", params_.alpha, rows, alpha_beta_cols);
        if (params_.beta)
            info.addOutput("beta", params_.beta, rows, alpha_beta_cols);

        return info;
    }

    StageBufferRequirements GDNRecurrenceStage::getBufferRequirements() const
    {
        return {};
    }

    StageBufferContract GDNRecurrenceStage::bufferContract() const
    {
        StageBufferContract contract;
        // Arena-managed activation inputs
        if (params_.qkv_buffer_id)
            contract.addInput(*params_.qkv_buffer_id);
        if (params_.alpha_buffer_id)
            contract.addInput(*params_.alpha_buffer_id);
        if (params_.beta_buffer_id)
            contract.addInput(*params_.beta_buffer_id);
        // Arena-managed output
        if (params_.output_buffer_id)
            contract.addOutput(*params_.output_buffer_id);
        if(const auto *p=std::get_if<GDNPreparedQKBinding>(&params_.qk_contract))
            contract.addInput(p->buffer_id);
        // Model weights (not arena-managed)
        if (params_.A_log)
            contract.addWeight(const_cast<ITensor *>(params_.A_log));
        if (params_.dt_bias)
            contract.addWeight(const_cast<ITensor *>(params_.dt_bias));
        return contract;
    }

} // namespace llaminar2
