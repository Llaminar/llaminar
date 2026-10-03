/**
 * @file TPLocalReduceOverlap.cpp
 * @brief One setup-owned event pair for overlapping captured LocalTP collectives.
 *
 * Recording and live execution are deliberately distinct. The small host enum
 * verifies paired submission/join while recording; graph replay runs the
 * complete device event DAG without touching host state. Only the join publishes
 * the reduced tensor, on the canonical graph stream after its completion wait.
 */
#include "TPLocalReduceOverlap.h"
#include "CapturedAllGatherStage.h"
#include "TPColumnReduceScatterStage.h"
#include "../../../backends/BackendManager.h"
#include "../../../backends/GPUDeviceContextPool.h"
#include "../../../backends/IBackend.h"
#include "../../local_execution/graph/ComputeGraph.h"
#include "../../local_execution/graph/GraphCaptureGuard.h"
#include "../../../tensors/TensorClasses.h"
#include "../../../transfer/TransferEngine.h"
#include "../../../utils/PerfStatsCollector.h"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
#include <variant>

namespace llaminar2
{
    namespace
    {
        /** @brief Setup-owned events; no resource is created by execute(). */
        struct ReduceEvents final
        {
            IBackend *backend;
            int ordinal;
            void *ready = nullptr;
            void *complete = nullptr;
            void *stream = nullptr;

            /** @brief Retain the backend identity without initializing events. */
            explicit ReduceEvents(DeviceId device)
                : backend(getBackendFor(device)), ordinal(device.gpu_ordinal())
            {
                if (!backend)
                    throw std::runtime_error("LocalTP reduce overlap has no backend");
            }

            /** @brief Release events after their graph-owning stages retire. */
            ~ReduceEvents()
            {
                if (complete) backend->destroyEvent(complete, ordinal);
                if (ready) backend->destroyEvent(ready, ordinal);
            }
        };

        /** @brief Paired recording authority shared only by the two graph nodes. */
        class ReduceTransaction final
        {
            using Gather = std::unique_ptr<CapturedAllGatherStage>;
        public:
            /** @brief Construction is device-free; setup materializes resources. */
            explicit ReduceTransaction(TPLocalRootedCollectiveStage::Params params)
                : operation_(std::in_place_type<TPLocalRootedCollectiveStage>, std::move(params)) {}

            /** @brief Freeze a lossless allgather in the same event lifecycle. */
            explicit ReduceTransaction(Gather operation)
                : operation_(std::in_place_type<Gather>, std::move(operation)) {}

            /** @brief Retain the existing sum's precision and arithmetic authority. */
            explicit ReduceTransaction(TPAllreduceStage::Params params)
                : operation_(std::in_place_type<TPAllreduceStage>, std::move(params)) {}

            /** @brief Column ownership uses the same single paired event lifecycle. */
            explicit ReduceTransaction(TPColumnReduceScatterStage::Params params)
                : operation_(std::in_place_type<TPColumnReduceScatterStage>, std::move(params)) {}

            /** @return Canonical stage metadata; execution is split only at its event edges. */
            const IComputeStage &operation() const
            {
                return std::visit([](const auto &op) -> const IComputeStage & {
                    if constexpr (std::is_same_v<std::decay_t<decltype(op)>, Gather>) return *op;
                    else return op;
                }, operation_);
            }

            /** @return Exact participant-local execution endpoint. */
            DeviceId device() const
            { return operation().device(); }

            /** @return Arena bank whose producer frontier is forked. */
            ITensor *input() const
            {
                return std::visit([](const auto &op) -> ITensor * {
                    if constexpr (std::is_same_v<std::decay_t<decltype(op)>, Gather>)
                        return op->exchangeBuffers().input;
                    else return op.params().tensor;
                }, operation_);
            }

            /** @return Bank published by the join, or none for a rooted contributor. */
            ITensor *output() const
            {
                return std::visit([](const auto &op) -> ITensor * {
                    if constexpr (std::is_same_v<std::decay_t<decltype(op)>, Gather>)
                        return op->exchangeBuffers().output;
                    else if constexpr (std::is_same_v<std::decay_t<decltype(op)>, TPAllreduceStage> ||
                                       std::is_same_v<std::decay_t<decltype(op)>, TPColumnReduceScatterStage>)
                        return op.params().tensor;
                    else return op.tensorRole() == TPLocalRootedCollectiveTensorRole::ReduceRootInOut
                        ? op.params().tensor : nullptr;
                }, operation_);
            }

            /** @return Immutable input arena role, validated by the concrete operation. */
            BufferId inputId() const
            {
                return std::visit([](const auto &op) {
                    if constexpr (std::is_same_v<std::decay_t<decltype(op)>, Gather>)
                        return op->exchangeBuffers().input_id;
                    else return *op.params().tensor_buffer_id;
                }, operation_);
            }

            /** @return Stable collective identity retained through capture and replay. */
            std::string stageName() const
            {
                return std::visit([](const auto &op) -> std::string {
                    if constexpr (std::is_same_v<std::decay_t<decltype(op)>, Gather>) return op->name();
                    else return op.params().stage_name;
                }, operation_);
            }

            /** @brief Prepare the retained operation and bounded event lane before capture.
             * @param ctx Exact participant context for the original operation.
             * @param stream Canonical graph stream; transport uses its acquired fork later. */
            void prepare(IDeviceContext *ctx, void *stream)
            {
                if (events_) return;
                if (isGraphCaptureActive())
                    throw std::logic_error("LocalTP reduce overlap resources must precede capture");
                const bool prepared = std::visit([&](auto &op) {
                    if constexpr (std::is_same_v<std::decay_t<decltype(op)>, Gather>)
                        return op->prepareGraphLaunch(ctx, stream);
                    else return op.prepareGraphLaunch(ctx, stream);
                }, operation_);
                if (!prepared) throw std::runtime_error("LocalTP overlap operation preparation failed");
                const auto endpoint = device();
                auto events = std::make_unique<ReduceEvents>(endpoint);
                events->stream = GPUDeviceContextPool::instance().getContext(endpoint)
                    .getOrCreateAuxiliaryStream("tp_native_reduce_overlap");
                events->ready = events->backend->createEvent(events->ordinal);
                events->complete = events->backend->createEvent(events->ordinal);
                if (!events->stream || !events->ready || !events->complete)
                    throw std::runtime_error("LocalTP reduce overlap event preparation failed");
                events_ = std::move(events);
            }

            /** @brief Enqueue a producer fork and native reduction, never a host wait. */
            bool submit(const StageGPUExecution &execution)
            {
                if (!events_ || phase_ == Phase::Submitted)
                    throw std::logic_error("LocalTP reduce overlap requires prepared, paired submission");
                if (const auto *gather = std::get_if<Gather>(&operation_))
                    (*gather)->validateEnqueue(execution);
                if (const auto *scatter = std::get_if<TPColumnReduceScatterStage>(&operation_))
                    scatter->validateEnqueue(execution);
                void *const main_stream = execution.nativeStream();
                const auto fork = TransferEngine::instance().recordDeviceInputFork(
                    input(), device(), main_stream, events_->ready);
                const auto acquired = TransferEngine::instance().acquireDeviceInputFork(
                    fork, events_->stream);
                if (!input()->gpu_data_ptr() || (output() && !output()->gpu_data_ptr()))
                    throw std::logic_error("LocalTP reduce overlap requires resident input storage");
                // A failed submission is terminal, not permission to reuse a half-recorded pair.
                phase_ = Phase::Submitted;
                recording_stream_ = main_stream;
                const bool submitted = std::visit([&](const auto &op) {
                    if constexpr (std::is_same_v<std::decay_t<decltype(op)>, Gather>)
                        return op->enqueueAcquiredInput(acquired);
                    else if constexpr (std::is_same_v<std::decay_t<decltype(op)>, TPAllreduceStage> ||
                                       std::is_same_v<std::decay_t<decltype(op)>, TPColumnReduceScatterStage>)
                        return op.enqueueAcquiredInput(acquired);
                    else
                    {
                        const auto &params = op.params();
                        void *const buffer = params.tensor->gpu_data_ptr();
                        return params.tp_ctx->reduceRawOnStream(
                            buffer, buffer, params.count, params.dtype,
                            CollectiveOp::ALLREDUCE_SUM, params.root_device_index,
                            params.participant_device_index, acquired.consumerStream(), params.stage_name);
                    }
                }, operation_);
                if (!submitted || !GPUDeviceContextPool::instance().getContext(device())
                         .recordEventChecked(events_->complete, acquired.consumerStream()))
                    throw std::runtime_error("LocalTP overlapped native reduction submission failed");
                return true;
            }

            /** @brief Join exactly this submission before publishing its root output. */
            bool join(const StageGPUExecution &execution)
            {
                if (!events_ || phase_ != Phase::Submitted ||
                    recording_stream_ != execution.nativeStream())
                    throw std::logic_error("LocalTP reduce overlap join has no matching stream submission");
                if (!GPUDeviceContextPool::instance().getContext(device())
                         .waitEventChecked(events_->complete, execution.nativeStream()))
                    throw std::runtime_error("LocalTP reduce overlap completion wait failed");
                if (output()) execution.publish(output());
                if (const auto *scatter = std::get_if<TPColumnReduceScatterStage>(&operation_))
                    execution.publish(scatter->params().packing);
                phase_ = Phase::Joined;
                PerfStatsCollector::addCounter(
                    "tp_rooted_collective_overlap", "recorded_joins", 1.0, {},
                    device().toString(),
                    {{"stage", stageName()}, {"transport", operation().type() == ComputeStageType::DEVICE_COUNTED_ALLGATHER
                        ? "device_counted_exchange" : "native_collective"},
                     {"accounting", "graph_template_or_eager_launch"}});
                return true;
            }

        private:
            /** Canonical operation owns validation and arena roles, never another graph. */
            std::variant<TPLocalRootedCollectiveStage, Gather, TPAllreduceStage,
                TPColumnReduceScatterStage> operation_;
            /** Recording lifecycle only; replay has no host-owned progress state. */
            enum class Phase { Ready, Submitted, Joined };
            Phase phase_ = Phase::Ready;
            void *recording_stream_ = nullptr;
            std::unique_ptr<ReduceEvents> events_;
        };

        /** @brief A visible graph edge, never an independent lifecycle authority. */
        enum class ReduceEdge { Submit, Join };

        /** @brief Native collective fork/join node generated only as a matched pair. */
        class ReduceEdgeStage final : public IComputeStage
        {
        public:
            /** @brief Bind one immutable role to its graph-retained transaction. */
            ReduceEdgeStage(std::shared_ptr<ReduceTransaction> transaction, ReduceEdge edge)
                : IComputeStage(transaction->device()),
                  transaction_(std::move(transaction)), edge_(edge) {}
            /** @brief Submit the reduction or acquire its exact completion event. */
            bool execute(IDeviceContext *ctx) override
            {
                if (!ctx || ctx->deviceId() != transaction_->device())
                    throw std::invalid_argument("LocalTP reduce overlap requires its exact device context");
                return edge_ == ReduceEdge::Submit
                    ? transaction_->submit(gpuExecution())
                    : transaction_->join(gpuExecution());
            }
            /** @return Explicit collective classification for capture planning. */
            ComputeStageType type() const override { return transaction_->operation().type(); }
            /** @return Diagnostic role, independent of model-specific node names. */
            std::string name() const override
            { return transaction_->stageName() + (edge_ == ReduceEdge::Submit ? "_submit" : ""); }
            /** @return Both edges belong to the native collective transaction. */
            bool requiresAllreduce() const override { return true; }
            /** @return Native backend support inherited from the reduction contract. */
            bool supportsBackend(ComputeBackendType backend) const override
            { return transaction_->operation().supportsBackend(backend); }
            /** @return Whether the immutable native reduction is capturable. */
            bool isGraphCapturable() const override
            { return transaction_->operation().isGraphCapturable(); }
            /** @return The retained operation's cold setup contract, not current readiness. */
            bool supportsGraphCaptureAfterLaunchPreparation() const override
            { return transaction_->operation().supportsGraphCaptureAfterLaunchPreparation(); }
            /** @return Exact buckets retain the concrete operation's preparation eligibility. */
            bool supportsLazyPrefillGraphCapturePreflight() const override
            { return transaction_->operation().supportsLazyPrefillGraphCapturePreflight(); }
            /** @return Padding semantics remain owned by the concrete collective. */
            bool supportsPaddedPrefillGraphCapturePreflight() const override
            { return transaction_->operation().supportsPaddedPrefillGraphCapturePreflight(); }
            /** @return Forward the operation's device-owned real-row masking contract. */
            bool supportsPaddedPrefillRealLengthContract() const override
            { return transaction_->operation().supportsPaddedPrefillRealLengthContract(); }
            /** @brief Materialize events outside capture and bind the exact main stream. */
            bool prepareGraphLaunch(IDeviceContext *ctx, void *stream) override
            {
                transaction_->prepare(ctx, stream);
                return IComputeStage::prepareGraphLaunch(ctx, stream);
            }
            /** @return Replay uses captured event edges without host preparation. */
            GraphLaunchPreparationPolicy graphLaunchPreparationPolicy() const override
            { return GraphLaunchPreparationPolicy::CaptureOnly; }
            /** @return Submission consumes the partial; completion owns its publication. */
            StageBufferContract bufferContract() const override
            {
                if (edge_ == ReduceEdge::Join) return transaction_->operation().bufferContract();
                return StageBufferContract::build().addInput(transaction_->inputId());
            }
            /** @return Dump output only after the exact completion wait, never at the fork. */
            StageDumpInfo buildDumpInfoImpl() const override
            { return edge_ == ReduceEdge::Join ? transaction_->operation().getDumpInfoSnapshot() : StageDumpInfo{}; }
            /** @return No output may become authoritative before the join. */
            CoherencePolicy coherencePolicy() const override
            { return edge_ == ReduceEdge::Join ? CoherencePolicy::OUTPUT : CoherencePolicy::NONE; }

        private:
            std::shared_ptr<ReduceTransaction> transaction_;
            ReduceEdge edge_;
        };

        /** @brief Reject an edge which would make producer -> compute cyclic. */
        bool dependsOn(const ComputeGraph &graph, const std::string &node,
                       const std::string &dependency, std::unordered_set<std::string> &visited)
        {
            if (node == dependency) return true;
            if (!visited.insert(node).second) return false;
            const auto *current = graph.getNode(node);
            if (!current) throw std::invalid_argument("LocalTP overlap references a missing graph dependency");
            return std::any_of(current->dependencies.begin(), current->dependencies.end(),
                [&](const auto &parent) { return dependsOn(graph, parent, dependency, visited); });
        }
    }

    void addTPLocalReduceOverlap(
        ComputeGraph &graph, TPLocalRootedCollectiveStage::Params params,
        const TPLocalReduceOverlapWindow &window)
    {
        const auto *producer = graph.getNode(window.producer);
        const auto *compute = graph.getNode(window.independent_compute);
        if (!params.device_id.is_gpu() || !params.tp_ctx || params.tp_ctx->degree() <= 1 ||
            !params.tensor || !params.tensor_buffer_id || params.count == 0 ||
            params.participant_device_index < 0 || params.participant_device_index >= params.tp_ctx->degree() ||
            params.root_device_index < 0 || params.root_device_index >= params.tp_ctx->degree() ||
            params.operation != TPLocalRootedCollectiveOperation::ReduceSum ||
            !params.sideband_workspace_bindings.empty() || params.mapped_dense_publication_exchange ||
            params.stage_name.empty() || graph.getNode(params.stage_name) ||
            graph.getNode(params.stage_name + "_submit"))
            throw std::invalid_argument("LocalTP reduce overlap requires a named native ReduceSum without sidebands");
        const auto native_backend = params.device_id.is_cuda()
            ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL;
        if (params.tp_ctx->backend() != native_backend)
            throw std::invalid_argument("LocalTP reduce overlap requires the device's native homogeneous collective backend");
        if (!producer || !compute || producer == compute || !producer->stage || !compute->stage)
            throw std::invalid_argument("LocalTP reduce overlap requires two existing, distinct window stages");
        if (producer->device != params.device_id || compute->device != params.device_id ||
            compute->stage->isCollectiveStage())
            throw std::invalid_argument("LocalTP reduce overlap window must contain same-device, non-collective compute");
        // Model lowering precedes arena/kernel binding. Use the typed setup
        // eligibility contract here; the executor still proves exact capture
        // readiness after preparation, before recording either operation.
        if (!producer->stage->supportsLazyPrefillGraphCapturePreflight() ||
            !compute->stage->supportsLazyPrefillGraphCapturePreflight())
            throw std::invalid_argument("LocalTP reduce overlap window cannot prepare native capture");
        if (producer->graph_capture_wave || compute->graph_capture_wave ||
            producer->heterogeneous_ticket_unit_contract || compute->heterogeneous_ticket_unit_contract ||
            requiresHeterogeneousTicketSegmentation(graph.nativeCaptureEnvelope()))
            throw std::invalid_argument("LocalTP reduce overlap cannot cross a capture-wave or heterogeneous ticket boundary");

        const auto contract = compute->stage->bufferContract();
        const auto aliases = [&](const auto &bindings)
        {
            return std::any_of(bindings.begin(), bindings.end(),
                [&](const auto &binding) { return binding.id == *params.tensor_buffer_id; });
        };
        std::unordered_set<std::string> visited;
        if (aliases(contract.inputs) || aliases(contract.outputs) || aliases(contract.inouts) ||
            dependsOn(graph, window.producer, window.independent_compute, visited))
            throw std::invalid_argument("LocalTP reduce overlap compute is not independent of its partial");

        // A matching use at both ends keeps the partial live through the whole
        // window in the arena plan. No caller can install a submit without join.
        const auto device = params.device_id;
        const auto join_name = params.stage_name;
        const auto submit_name = join_name + "_submit";
        auto transaction = std::make_shared<ReduceTransaction>(std::move(params));
        graph.addNode(submit_name, std::make_unique<ReduceEdgeStage>(transaction, ReduceEdge::Submit), device);
        graph.addNode(join_name, std::make_unique<ReduceEdgeStage>(transaction, ReduceEdge::Join), device);
        graph.addDependency(submit_name, window.producer);
        graph.addDependency(window.independent_compute, submit_name);
        graph.addDependency(join_name, window.independent_compute);
    }

    void overlapTPLocalAllGather(ComputeGraph &graph, const std::string &gather_node,
                                const std::string &independent_compute)
    {
        auto *gather = graph.getNode(gather_node);
        const auto *compute = graph.getNode(independent_compute);
        auto *operation = gather ? dynamic_cast<CapturedAllGatherStage *>(gather->stage.get()) : nullptr;
        if (!operation || !compute || !compute->stage || gather == compute || gather->dependencies.size() != 1 ||
            graph.getNode(gather_node + "_submit"))
            throw std::invalid_argument("Allgather overlap requires one existing captured gather, producer and independent compute");
        const auto buffers = operation->exchangeBuffers();
        const auto device = operation->device();
        if (!device.is_gpu() || gather->device != device || operation->name() != gather_node ||
            !buffers.input || !buffers.output || buffers.input == buffers.output || buffers.input_id == buffers.output_id)
            throw std::invalid_argument("Allgather overlap requires disjoint same-device captured arena banks");
        const auto producer_name = gather->dependencies.front();
        const auto *producer = graph.getNode(producer_name);
        if (!producer || !producer->stage || producer == compute ||
            producer->device != device || compute->device != device ||
            compute->stage->isCollectiveStage() ||
            !producer->stage->supportsLazyPrefillGraphCapturePreflight() ||
            !compute->stage->supportsLazyPrefillGraphCapturePreflight())
            throw std::invalid_argument("Allgather overlap requires capturable same-device independent compute");
        for (const auto *node : {producer, compute, static_cast<const ComputeNode *>(gather)})
            if (node->graph_capture_wave || node->heterogeneous_ticket_unit_contract ||
                requiresHeterogeneousTicketSegmentation(graph.nativeCaptureEnvelope()))
                throw std::invalid_argument("Allgather overlap cannot cross native capture boundaries");
        const auto contract = compute->stage->bufferContract();
        const auto aliases = [&](const auto &bindings) {
            return std::any_of(bindings.begin(), bindings.end(), [&](const auto &binding) {
                return binding.id == buffers.input_id || binding.id == buffers.output_id;
            });
        };
        std::unordered_set<std::string> visited;
        if (aliases(contract.inputs) || aliases(contract.outputs) || aliases(contract.inouts) ||
            dependsOn(graph, producer_name, independent_compute, visited))
            throw std::invalid_argument("Allgather overlap compute aliases a transport bank or precedes its producer");
        visited.clear();
        if (dependsOn(graph, independent_compute, gather_node, visited))
            throw std::invalid_argument("Allgather overlap compute already consumes collective completion");

        // Reuse the original operation's exact geometry and join identity.
        // Only an event edge moves: the existing receive bank stays live until
        // the same downstream consumers acquire the joined publication.
        // Transfer the original stage itself: transport, prepared binding and
        // cold-capture policy retain one authority, rather than cloning params
        // into a second native-only implementation of the exchange.
        std::unique_ptr<CapturedAllGatherStage> owned_operation(
            static_cast<CapturedAllGatherStage *>(gather->stage.release()));
        auto transaction = std::make_shared<ReduceTransaction>(std::move(owned_operation));
        const auto submit_name = gather_node + "_submit";
        gather->stage = std::make_unique<ReduceEdgeStage>(transaction, ReduceEdge::Join);
        graph.addNode(submit_name, std::make_unique<ReduceEdgeStage>(transaction, ReduceEdge::Submit), device);
        graph.addDependency(submit_name, producer_name);
        graph.addDependency(independent_compute, submit_name);
        graph.addDependency(gather_node, independent_compute);
    }

    /**
     * @brief Share the sum-overlap graph checks between full and column-owned results.
     *
     * Only the operation's immutable validation and touched-bank projection
     * differ. Both variants install the identical submit/compute/join lifecycle.
     */
    template<class SumStage>
    static void overlapNativeSum(ComputeGraph &graph, const std::string &allreduce_node,
                                  const std::string &independent_compute)
    {
        auto *reduce = graph.getNode(allreduce_node);
        const auto *compute = graph.getNode(independent_compute);
        const auto *operation = reduce ? dynamic_cast<const SumStage *>(reduce->stage.get()) : nullptr;
        if (!operation || !compute || !compute->stage || reduce == compute ||
            reduce->dependencies.size() != 1 || graph.getNode(allreduce_node + "_submit"))
            throw std::invalid_argument("Allreduce overlap requires one native reduction, producer and independent compute");
        const auto &params = operation->params();
        const auto *local = dynamic_cast<const ILocalTPContext *>(params.tp_ctx);
        const auto native = params.device_id.is_cuda() ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL;
        if (!params.device_id.is_gpu() || reduce->device != params.device_id ||
            operation->device() != params.device_id || !local || local->degree() < 2 || local->backend() != native ||
            !params.tensor || !params.tensor_buffer_id || params.tensor->numel() == 0 ||
            params.stage_name != allreduce_node)
            throw std::invalid_argument("Allreduce overlap requires native arithmetic without sidebands and a bounded tensor");
        if constexpr (std::is_same_v<SumStage, TPAllreduceStage>)
            if (!params.sidebands.empty() || !params.sideband_workspace_bindings.empty() ||
                params.arithmetic_policy != TPAllreduceArithmeticPolicy::NativeCollective ||
                params.count > params.tensor->numel())
                throw std::invalid_argument("Allreduce overlap requires native bounded arithmetic without sidebands");
        if (std::none_of(local->devices().begin(), local->devices().end(), [&](const auto &endpoint) {
                return endpoint.toLocalDeviceId() == params.device_id;
            }))
            throw std::invalid_argument("Allreduce overlap endpoint is not a member of its native domain");
        const auto producer_name = reduce->dependencies.front();
        const auto *producer = graph.getNode(producer_name);
        if (!producer || !producer->stage || producer == compute ||
            producer->device != params.device_id || compute->device != params.device_id ||
            compute->stage->isCollectiveStage() ||
            !producer->stage->supportsLazyPrefillGraphCapturePreflight() ||
            !compute->stage->supportsLazyPrefillGraphCapturePreflight())
            throw std::invalid_argument("Allreduce overlap requires capturable same-device independent compute");
        for (const auto *node : {producer, compute, static_cast<const ComputeNode *>(reduce)})
            if (node->graph_capture_wave || node->heterogeneous_ticket_unit_contract ||
                requiresHeterogeneousTicketSegmentation(graph.nativeCaptureEnvelope()))
                throw std::invalid_argument("Allreduce overlap cannot cross native capture boundaries");
        const auto contract = compute->stage->bufferContract();
        const auto aliases = [&](const auto &bindings) {
            return std::any_of(bindings.begin(), bindings.end(), [&](const auto &binding) {
                if (binding.id == *params.tensor_buffer_id) return true;
                if constexpr (std::is_same_v<SumStage, TPColumnReduceScatterStage>)
                    return binding.id == *params.packing_buffer_id;
                return false;
            });
        };
        std::unordered_set<std::string> visited;
        if (aliases(contract.inputs) || aliases(contract.outputs) || aliases(contract.inouts) ||
            dependsOn(graph, producer_name, independent_compute, visited))
            throw std::invalid_argument("Allreduce overlap compute aliases or precedes its partial producer");
        visited.clear();
        if (dependsOn(graph, independent_compute, allreduce_node, visited))
            throw std::invalid_argument("Allreduce overlap compute already consumes collective completion");

        // Both edges keep the original tensor live. The join remains the only
        // publication visible to downstream consumers and the capture ledger.
        auto transaction = std::make_shared<ReduceTransaction>(params);
        const auto device = params.device_id;
        const auto submit_name = allreduce_node + "_submit";
        reduce->stage = std::make_unique<ReduceEdgeStage>(transaction, ReduceEdge::Join);
        graph.addNode(submit_name, std::make_unique<ReduceEdgeStage>(transaction, ReduceEdge::Submit), device);
        graph.addDependency(submit_name, producer_name);
        graph.addDependency(independent_compute, submit_name);
        graph.addDependency(allreduce_node, independent_compute);
    }

    void overlapTPLocalAllreduce(ComputeGraph &graph, const std::string &allreduce_node,
                                 const std::string &independent_compute)
    {
        overlapNativeSum<TPAllreduceStage>(graph, allreduce_node, independent_compute);
    }

    void overlapTPLocalColumnReduceScatter(ComputeGraph &graph, const std::string &scatter_node,
                                           const std::string &independent_compute)
    {
        overlapNativeSum<TPColumnReduceScatterStage>(graph, scatter_node, independent_compute);
    }
}
