/**
 * @file Test__NativeAllGatherStage.cpp
 * @brief CUDA/NCCL and ROCm/RCCL proof of lossless captured byte allgathers.
 *
 * One retained graph transports several exact, sometimes unaligned byte
 * prefixes. Twenty input resets check communicator order, all bits (including
 * floating-point special values), untouched capacity tails and tensor event
 * publication. Physical GPU order is reversed deliberately. Setup/readback
 * waits belong only to the fixture; the production stage enqueues no waits on
 * the host and performs no allocation, conversion or extra copy.
 */
#include <gtest/gtest.h>
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "collective/LocalTPContext.h"
#include "execution/compute_stages/stages/NativeAllGatherStage.h"
#include "execution/compute_stages/stages/TPLocalReduceOverlap.h"
#include "execution/compute_stages/stages/RMSNormStage.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "tensors/Tensors.h"
#include "transfer/TransferEngine.h"

#include <algorithm>
#include <array>
#include <barrier>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <string_view>
#include <optional>
#include <exception>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace llaminar2;

namespace
{
    /** @brief Raise a fixture failure while native resources still have owners. */
    void require(bool value, const char *message)
    { if (!value) throw std::runtime_error(message); }

    /** @brief Include NaN payloads and signed zeros without doing FP arithmetic. */
    std::uint8_t payloadByte(int participant, int request, std::size_t offset)
    {
        constexpr std::array<std::uint8_t, 16> special{
            0, 0, 0, 128, 1, 0, 192, 127, 0, 0, 128, 255, 255, 255, 255, 255};
        if (offset < special.size())
            return special[(offset + request + participant) % special.size()];
        return static_cast<std::uint8_t>(offset * 37 + participant * 101 + request * 17);
    }

    /** @brief Fixture-only peer failure points; never a transport/count policy. */
    enum class GatherPeerDrop { None, BeforeCapture, BeforeReplay };
    /** @brief Distinct intentional exception, so unrelated native failures cannot satisfy a control. */
    class InjectedGatherPeerDrop final:public std::runtime_error {
    public:
        /** @brief Identify the one deliberately failing fixture participant. */
        InjectedGatherPeerDrop():std::runtime_error("Intentional native gather peer drop"){}
    };
    /** @brief Passive owner-local observations, consumed only after thread join. */
    struct GatherObservation {std::size_t setupJoins=0,captureAttempts=0,instantiated=0,launchAttempts=0;};
    /** @brief Fence early gamma/setup uploads before the later graph-retirement guard exists. */
    class GatherSetupRetirement final {
    public:
        /** @brief Retain the exact setup stream while its tensor/workspace owners remain in scope. */
        GatherSetupRetirement(IWorkerGPUContext &worker,ILocalTPContext &tp,void *stream,GatherObservation &observation)
            :worker_(worker),tp_(tp),stream_(stream),observation_(observation),exceptions_(std::uncaught_exceptions()){}
        /** @brief Finish queued setup I/O before its host/native owner retirement, including failure. */
        ~GatherSetupRetirement()noexcept {
            if(std::uncaught_exceptions()>exceptions_)tp_.requestAbort();
            try{if(!worker_.synchronizeStreamChecked(stream_))std::terminate();++observation_.setupJoins;}
            catch(...){std::terminate();}
        }
    private:
        IWorkerGPUContext &worker_;ILocalTPContext &tp_;void *stream_;GatherObservation &observation_;int exceptions_;
    };

    /**
     * @brief Observe the existing native device receipt without implementing a transport.
     *
     * Only the optional raw native-send diagnostic pointer changes. The real LocalTPContext
     * owns membership, native communicator, row/count geometry, stream and
     * submissions. Counters belong to the fixture's admitted workspace and
     * remain stable through graph retirement. Native receipts ACCUMULATE actual
     * outgoing ring-send bytes (AllGather multiplicity P-1); they do not reset
     * themselves or report a per-invocation logical prefix. No host count is sampled here.
     */
    class ObservedNativeGatherContext final : public LocalTPContext
    {
    public:
        /** @brief Construct the same original ordered native communicator. */
        using LocalTPContext::LocalTPContext;
        /** @brief Bind one participant's four immutable receipt addresses before recording. */
        void bindReceipts(int participant, unsigned long long *base)
        {
            require(participant >= 0 && participant < 2 && base &&
                reinterpret_cast<std::uintptr_t>(base) % alignof(unsigned long long) == 0,
                "native receipt binding is invalid");
            require(!receipts_[participant], "native receipt owner cannot be replaced");
            receipts_[participant] = base;
        }
        /** @brief Forward the exact operation, native banks, count geometry and stream with its device receipt. */
        bool nativeRowsOnStream(NativeRowCollective operation,const void *send,void *receive,
            const NativeCollectiveRows &rows,CollectiveDataType dtype,CollectiveOp reduction,
            int participant,void *stream,const std::string &stage_name,
            unsigned long long *payload_bytes=nullptr) override
        {
            require(participant >= 0 && participant < 2 && receipts_[participant] &&
                !payload_bytes && !rows.payloadReceipt(), "unexpected native receipt source");
            constexpr std::array<std::string_view,4> names{"native_byte_gather_0","native_byte_gather_1",
                "native_byte_gather_2","native_byte_gather_3"};
            const auto found=std::find(names.begin(),names.end(),stage_name);
            require(found!=names.end(), "unexpected observed native operation");
            const auto observed=rows.withPayloadReceipt(receipts_[participant]+(found-names.begin()));
            return LocalTPContext::nativeRowsOnStream(operation,send,receive,observed,dtype,reduction,
                participant,stream,stage_name,payload_bytes);
        }
    private:
        std::array<unsigned long long *,2> receipts_{}; ///< Immutable borrowed workspace addresses, not observed byte values.
    };

    /** @brief Retire the real native executable before its overlap event/stage owners, including failure unwinding. */
    class OverlapCaptureRetirement final
    {
    public:
        /** @brief Borrow exact retained fixture owners; no allocation or submission occurs. */
        OverlapCaptureRetirement(IWorkerGPUContext &worker,ILocalTPContext &tp,
            std::unique_ptr<IGPUGraphCapture> &capture,void *stream)
            :worker_(worker),tp_(tp),capture_(capture),stream_(stream),exceptions_(std::uncaught_exceptions()) {}
        /** @brief Abort peers on failure, then fence only this terminal fixture boundary before retiring events. */
        ~OverlapCaptureRetirement() noexcept
        {
            if(std::uncaught_exceptions()>exceptions_)tp_.requestAbort();
            try
            {
                if(!worker_.synchronizeStreamChecked(stream_))std::terminate();
                if(capture_){capture_->reset();capture_.reset();}
            }
            catch(...){std::terminate();}
        }
    private:
        IWorkerGPUContext &worker_;ILocalTPContext &tp_;
        std::unique_ptr<IGPUGraphCapture> &capture_;void *stream_;int exceptions_;
    };

    /**
     * @brief Exercise ordinary or production-overlapped native byte allgather.
     * @param first Existing backend's first GPU; communicator order stays reversed.
     * @param arena_counted Whether one exact external arena scalar clips live rows.
     * @param row_capacity Original retained row capacity, including physical M1.
     * @param overlapped Use the production overlap builder around real RMSNorm work.
     * @param drop Explicit fixture peer failure; requires the actual overlap owner path.
     * Count values are fixture inputs; graph-produced zero-count publication is
     * separately required from the fixed-head serving pack stage.
     */
    void verifyNativeAllGather(DeviceId first, bool arena_counted = false, int row_capacity = 3, bool overlapped = false, GatherPeerDrop drop = GatherPeerDrop::None)
    {
        ASSERT_TRUE(drop==GatherPeerDrop::None || (overlapped && arena_counted));
        auto *backend = getBackendFor(first);
        ASSERT_NE(backend, nullptr);
        ASSERT_GE(backend->deviceCount(), 2) << "Native allgather proof requires two GPUs";
        const std::array<DeviceId, 2> devices{
            first.is_cuda() ? DeviceId::cuda(1) : DeviceId::rocm(1), first};
        const std::vector members{GlobalDeviceAddress::fromLocalDeviceId(devices[0]),
            GlobalDeviceAddress::fromLocalDeviceId(devices[1])};
        const auto native_backend=first.is_cuda()?CollectiveBackendType::NCCL:CollectiveBackendType::RCCL;
        std::unique_ptr<ILocalTPContext> tp;
        if(overlapped)
        {
            ASSERT_TRUE(arena_counted);
            tp=std::make_unique<ObservedNativeGatherContext>(members,std::vector<float>{},native_backend);
        }
        else tp=createLocalTPContext(members,{},native_backend);
        ASSERT_NE(tp, nullptr);
        WorkspaceRequirements receipt_requirements;
        std::shared_ptr<PhysicalMemoryAuthority> receipt_memory;
        if(overlapped)
        {
            // The only new diagnostic bank has an exact canonical PMA owner.
            receipt_requirements.buffers.push_back({"native_gather_payload_receipts",4*sizeof(unsigned long long),
                alignof(unsigned long long),true});
            const auto bytes=receipt_requirements.total_bytes_with_alignment();
            PhysicalMemoryPlanBuilder plan;
            for(const auto device:devices)
                plan.add({.world_rank=0,.device=device,.total_bytes=bytes,.admission_available_bytes=bytes},
                    PhysicalMemoryOwner::ExecutionWorkspace,bytes);
            receipt_memory=std::make_shared<PhysicalMemoryAuthority>(
                std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()),0);
        }
        std::array<std::size_t, 4> extents{1,37,4097,32768};
        if (arena_counted) for (auto &bytes : extents) bytes *= row_capacity;
        constexpr std::array<int, 10> live_rows{3,1,0,2,3,0,1,3,2,0};
        constexpr std::array input_ids{BufferId::HIDDEN_STATE, BufferId::MOE_GATE_SCRATCH,
            BufferId::MOE_UP_SCRATCH, BufferId::MOE_EXPERT_OUTPUT};
        constexpr std::array output_ids{BufferId::NORMALIZED, BufferId::MOE_COMBINED_OUTPUT,
            BufferId::MOE_SHARED_EXPERT_OUTPUT, BufferId::FFN_OUTPUT};
        std::barrier rendezvous(2);
        std::array<std::exception_ptr, 2> errors{};
        std::array<GatherObservation,2> observations{};
        std::array<std::array<std::vector<std::uint8_t>,4>,2> produced;
        std::array<std::array<unsigned long long,4>,2> receipts{};
        std::array<std::thread, 2> threads;
        for (int participant = 0; participant < 2; ++participant)
        {
            threads[participant] = std::thread([&, participant]
            {
                try
                {
                    auto &gpu = GPUDeviceContextPool::instance().getContext(devices[participant]);
                    gpu.submitAndWait([&]
                    {
                        const auto device = devices[participant];
                        const int ordinal = device.is_cuda() ? device.cuda_ordinal() : device.rocm_ordinal();
                        void *const stream = gpu.getOrCreateAuxiliaryStream("native_allgather_regression");
                        require(stream && reinterpret_cast<std::uintptr_t>(stream)>2,"Native gather requires its exact non-default stream");
                        auto context = IDeviceContext::create(device, 1);
                        std::array<std::unique_ptr<FP32Tensor>, 4> inputs, outputs;
                        std::array<std::unique_ptr<FP32Tensor>,4> packet_gamma, independent_inputs, independent_outputs;
                        std::unique_ptr<FP32Tensor> independent_gamma;
                        std::unique_ptr<DeviceWorkspaceManager> receipt_workspace;
                        unsigned long long *receipt_bank=nullptr;
                        std::optional<GatherSetupRetirement> setupRetirement;
                        if(overlapped)setupRetirement.emplace(gpu,*tp,stream,observations[participant]);
                        if(overlapped)
                        {
                            receipt_workspace=std::make_unique<DeviceWorkspaceManager>(device,
                                receipt_requirements.total_bytes_with_alignment(),receipt_memory);
                            require(receipt_workspace->allocate(receipt_requirements),"admit native payload receipt bank");
                            require(receipt_memory->claimedBytes(device,PhysicalMemoryOwner::ExecutionWorkspace,
                                PhysicalMemoryMaterializationKind::NewAllocation)==receipt_requirements.total_bytes_with_alignment(),
                                "native receipts must retain their exact actual PMA lease");
                            receipt_bank=static_cast<unsigned long long *>(receipt_workspace->getBuffer("native_gather_payload_receipts"));
                            auto *observed=dynamic_cast<ObservedNativeGatherContext *>(tp.get());
                            require(observed&&receipt_bank,"retain exact native receipt authority");
                            observed->bindReceipts(participant,receipt_bank);
                            independent_gamma=std::make_unique<FP32Tensor>(std::vector<std::size_t>{256},device);
                            std::fill_n(independent_gamma->mutable_data(),256,1.f);
                            TransferEngine::prepareDeviceInput(independent_gamma.get(),device,stream);
                        }
                        // The cold count starts with no GPU allocation. Its
                        // ordinary owner preparation precedes sealing; no stage
                        // constructor or execution path materializes storage.
                        std::unique_ptr<INT32Tensor> count;
                        if (arena_counted) count = std::make_unique<INT32Tensor>(std::vector<std::size_t>{2});
                        // Every asynchronous host frame outlives the inner
                        // retirement guard. On failure it fences pending DMA
                        // before these upload/download buffers can be destroyed.
                        std::array<std::vector<std::uint8_t>,4> sent,received;
                        std::int32_t live=0;
                        // Immutable setup/reset backing outlives all asynchronous
                        // uploads, including unwind through the retirement guard.
                        const std::array<unsigned long long,4> zero_receipts{};
                        std::uint64_t expected_window_rows=0;
                        std::array<std::array<float,256>,4> independent_actual{};
                        std::unique_ptr<IGPUGraphCapture> capture;
                        // Every asynchronous poison upload borrows this immutable
                        // worker-lifetime frame through the terminal stream wait.
                        const std::array<float,256> independent_poison=[] {
                            std::array<float,256> values{};values.fill(-999.f);return values;
                        }();
                        try
                        {
                            ComputeGraph graph;
                            std::optional<OverlapCaptureRetirement> retirement;
                            if(overlapped) {
                                retirement.emplace(gpu,*tp,capture,stream);
                                require(backend->hostToDeviceOnStream(receipt_bank,zero_receipts.data(),sizeof(zero_receipts),
                                    ordinal,stream),"initialize accumulating native ring-send receipt window");
                            }
                            std::vector<GraphCaptureDependencyLedger::StagePlan> plans;
                            for (std::size_t index = 0; index < extents.size(); ++index)
                            {
                                // Extra capacity is deliberate: the native count
                                // must be the message prefix, not size_bytes().
                                inputs[index] = std::make_unique<FP32Tensor>(
                                    overlapped ? std::vector<std::size_t>{1,(extents[index] + 3) / 4 + 16}
                                        : std::vector<std::size_t>{(extents[index] + 3) / 4 + 16}, device);
                                outputs[index] = std::make_unique<FP32Tensor>(
                                    std::vector<std::size_t>{(2 * extents[index] + 3) / 4 + 16}, device);
                                for (auto *tensor : {inputs[index].get(), outputs[index].get()})
                                {
                                    std::fill_n(tensor->mutable_data(), tensor->numel(), 0.0f);
                                    require(tensor->ensureOnDevice(device, stream), "fixture upload");
                                }
                                NativeAllGatherStage::Params params;
                                params.device_id = device;
                                params.tp_ctx = tp.get();
                                params.local_input = inputs[index].get();
                                params.rank_major_output = outputs[index].get();
                                params.bytes_per_participant = extents[index];
                                params.participant = participant;
                                params.stage_name = "native_byte_gather_" + std::to_string(index);
                                params.input_buffer_id = input_ids[index];
                                params.output_buffer_id = output_ids[index];
                                if (arena_counted)
                                    params.arena_rows = NativeAllGatherStage::ArenaRows{count.get(),
                                        BufferId::MTP_VERIFIER_REQUEST_LENGTHS, sizeof(std::int32_t), row_capacity,
                                        extents[index] / row_capacity};
                                auto stage = std::make_unique<NativeAllGatherStage>(params);
                                if (arena_counted && index == 0) {
                                    require(!count->gpu_data_ptr(), "count declaration must remain cold");
                                    require(!stage->isGraphCapturable(), "cold count must require ordinary preparation");
                                    count->mutable_typed_data()[0] = -777; // Offset must select the second scalar.
                                    count->mutable_typed_data()[1] = row_capacity;
                                    TransferEngine::prepareDeviceInput(count.get(), device, stream);
                                }
                                const auto declaration = stage->graphCaptureVariantSignature();
                                require(stage->prepareGraphLaunch(context.get(), stream), "stage stream binding");
                                require(stage->graphCaptureVariantSignature() == declaration, "storage sealing changed declaration identity");
                                require(stage->isGraphCapturable(), "native collective capture contract");
                                GraphCaptureDependencyLedger::StagePlan plan;
                                plan.stage_identity = stage.get();
                                plan.stage_name = params.stage_name;
                                plan.external_inputs = {inputs[index].get()};
                                if (count) plan.external_inputs.push_back(count.get());
                                plan.outputs = {outputs[index].get()};
                                plans.push_back(std::move(plan));
                                graph.addNode(params.stage_name, std::move(stage), device);
                                if(overlapped)
                                {
                                    const auto producer="packet_producer_"+std::to_string(index);
                                    const auto compute="independent_norm_"+std::to_string(index);
                                    packet_gamma[index]=std::make_unique<FP32Tensor>(std::vector<std::size_t>{inputs[index]->numel()},device);
                                    std::fill_n(packet_gamma[index]->mutable_data(),packet_gamma[index]->numel(),1.f);
                                    TransferEngine::prepareDeviceInput(packet_gamma[index].get(),device,stream);
                                    independent_inputs[index]=std::make_unique<FP32Tensor>(std::vector<std::size_t>{1,256},device);
                                    independent_outputs[index]=std::make_unique<FP32Tensor>(std::vector<std::size_t>{1,256},device);
                                    std::fill_n(independent_inputs[index]->mutable_data(),256,1.f);
                                    TransferEngine::prepareDeviceInput(independent_inputs[index].get(),device,stream);
                                    TransferEngine::prepareDeviceOutput(independent_outputs[index].get(),device,stream);
                                    RMSNormStage::Params production;
                                    production.device_id=device;production.input=inputs[index].get();production.output=inputs[index].get();
                                    production.gamma=packet_gamma[index].get();production.input_buffer_id=input_ids[index];
                                    production.output_buffer_id=input_ids[index];production.seq_len=1;
                                    graph.addNode(producer,std::make_unique<RMSNormStage>(production),device);
                                    RMSNormStage::Params independent;
                                    independent.device_id=device;independent.input=independent_inputs[index].get();
                                    independent.output=independent_outputs[index].get();independent.gamma=independent_gamma.get();
                                    independent.input_buffer_id=BufferId::GATE_PROJ;independent.output_buffer_id=BufferId::UP_PROJ;
                                    independent.seq_len=1;
                                    graph.addNode(compute,std::make_unique<RMSNormStage>(independent),device);
                                    graph.addDependency(params.stage_name,producer);
                                    if(index)graph.addDependency(producer,"native_byte_gather_"+std::to_string(index-1));
                                    overlapTPLocalAllGather(graph,params.stage_name,compute);
                                }
                                else if(index)
                                    graph.addDependency(params.stage_name, "native_byte_gather_" + std::to_string(index - 1));
                            }
                            if(overlapped)
                            {
                                plans.clear();
                                std::array<std::size_t,4> producer_indices{};
                                for(const auto &name:graph.getExecutionOrder())
                                {
                                    auto *operation=graph.getNode(name)->stage.get();
                                    require(operation->prepareGraphLaunch(context.get(),stream),"prepare real overlap edge");
                                    GraphCaptureDependencyLedger::StagePlan plan;
                                    plan.stage_identity=operation;plan.stage_name=name;
                                    bool matched=false;
                                    for(std::size_t index=0;index<extents.size();++index)
                                    {
                                        const auto suffix=std::to_string(index);
                                        if(name=="packet_producer_"+suffix)
                                        {
                                            plan.external_inputs={inputs[index].get(),packet_gamma[index].get()};
                                            plan.outputs={inputs[index].get()};producer_indices[index]=plans.size();
                                            require(operation->execute(context.get()),"prepare actual packet RMSNorm facade");matched=true;
                                        }
                                        else if(name=="independent_norm_"+suffix)
                                        {
                                            plan.external_inputs={independent_inputs[index].get(),independent_gamma.get()};
                                            plan.outputs={independent_outputs[index].get()};
                                            require(operation->execute(context.get()),"prepare actual independent RMSNorm facade");matched=true;
                                        }
                                        else if(name=="native_byte_gather_"+suffix || name=="native_byte_gather_"+suffix+"_submit")
                                        {
                                            require(producer_indices[index]<plans.size(),"overlap packet must have an earlier real producer");
                                            plan.internal_inputs.push_back({inputs[index].get(),producer_indices[index]});
                                            plan.external_inputs={count.get()};
                                            if(name=="native_byte_gather_"+suffix)plan.outputs={outputs[index].get()};
                                            matched=true;
                                        }
                                    }
                                    require(matched,"all native overlap nodes require exact producer/consumer ledger entries");
                                    plans.push_back(std::move(plan));
                                }
                            }
                            require(gpu.synchronizeStreamChecked(stream), "fixture upload completion");
                            for (const auto &input : inputs)
                                TransferEngine::requireDeviceInput(input.get(), device, stream);
                            if (count) TransferEngine::requireDeviceInput(count.get(), device, stream);
                            if(overlapped)
                            {
                                TransferEngine::requireDeviceInput(independent_gamma.get(),device,stream);
                                for(std::size_t index=0;index<extents.size();++index)
                                {
                                    TransferEngine::requireDeviceInput(packet_gamma[index].get(),device,stream);
                                    TransferEngine::requireDeviceInput(independent_inputs[index].get(),device,stream);
                                }
                            }
                            GraphCaptureDependencyLedger ledger(device, stream, std::move(plans), "native_allgather");
                            capture = gpu.createGraphCapture(stream);
                            if(drop==GatherPeerDrop::BeforeCapture) {
                                // Both real stage/identity preparations must finish
                                // before cancellation. Native capture entry has its
                                // own abort-checked rendezvous immediately below.
                                rendezvous.arrive_and_wait();
                                require(!tp->isAbortRequested(),"Peer failed before native gather capture");
                                if(participant==1)throw InjectedGatherPeerDrop{};
                            }
                            rendezvous.arrive_and_wait();
                            require(!tp->isAbortRequested(),"Peer failed before native gather capture");
                            ++observations[participant].captureAttempts;
                            // A removed guard must fail before entering an unmatched
                            // vendor capture, rather than waiting for its timeout.
                            require(drop!=GatherPeerDrop::BeforeCapture,"Dropped peer reached native gather capture");
                            ScopedBackendGraphCapture recording(gpu, *capture, "native_allgather", &ledger);
                            require(recording.begin(), "begin collective graph");
                            for (const auto &name : graph.getExecutionOrder())
                            {
                                auto *stage = graph.getNode(name)->stage.get();
                                ScopedGraphCaptureStage scope(stage);
                                require(stage->execute(context.get()), "record native allgather");
                                scope.complete();
                            }
                            recording.finish();
                            require(capture->instantiate(), "instantiate collective graph");
                            ++observations[participant].instantiated;

                            for (int request = 1; request <= 20; ++request)
                            {
                                live = arena_counted ? std::min(row_capacity,live_rows[(request - 1) % live_rows.size()]) : 1;
                                if(overlapped) {
                                    // All prior requests ended at the existing terminal
                                    // join. Reset only this diagnostic measurement
                                    // window once; request counts/graphs stay intact.
                                    if(request==11) {
                                        require(backend->hostToDeviceOnStream(receipt_bank,zero_receipts.data(),sizeof(zero_receipts),
                                            ordinal,stream),"reset retired native ring-send receipt window");
                                        expected_window_rows=0;
                                    }
                                    expected_window_rows+=static_cast<std::uint64_t>(live);
                                }
                                if (count)
                                    require(backend->hostToDeviceOnStream(static_cast<std::byte *>(count->gpu_data_ptr()) + sizeof(std::int32_t),
                                        &live, sizeof(live), ordinal, stream), "publish original live transport count");
                                for (std::size_t index = 0; index < extents.size(); ++index)
                                {
                                    sent[index].resize(inputs[index]->size_bytes());
                                    received[index].assign(outputs[index]->size_bytes(), 0xa5);
                                    if(overlapped)
                                    {
                                        std::vector<float> native(inputs[index]->numel());
                                        for(std::size_t element=0;element<native.size();++element)
                                            native[element]=1.f+float((element+participant+request)%7)*.125f;
                                        std::memcpy(sent[index].data(),native.data(),sent[index].size());
                                        require(backend->hostToDeviceOnStream(independent_outputs[index]->gpu_data_ptr(),independent_poison.data(),
                                            sizeof(independent_poison),ordinal,stream),"poison independent overlap output");
                                    }
                                    else for (std::size_t byte = 0; byte < sent[index].size(); ++byte)
                                        sent[index][byte] = payloadByte(participant, request, byte);
                                    require(backend->hostToDeviceOnStream(inputs[index]->gpu_data_ptr(), sent[index].data(),
                                        sent[index].size(), ordinal, stream), "reset input bytes");
                                    require(backend->hostToDeviceOnStream(outputs[index]->gpu_data_ptr(), received[index].data(),
                                        received[index].size(), ordinal, stream), "poison receive capacity");
                                }
                                if(drop==GatherPeerDrop::BeforeReplay && request==1) {
                                    // Both real executables must be instantiated
                                    // before a peer fails; cancellation must not
                                    // race a slower participant's native capture.
                                    rendezvous.arrive_and_wait();
                                    require(!tp->isAbortRequested(),"Peer failed before native gather replay");
                                    if(participant==1)throw InjectedGatherPeerDrop{};
                                }
                                rendezvous.arrive_and_wait();
                                require(!tp->isAbortRequested(),"Peer failed before native gather replay");
                                ++observations[participant].launchAttempts;
                                require(drop==GatherPeerDrop::None,"Dropped peer reached native gather submission");
                                require(capture->launch(), "replay unchanged collective graph");
                                for (std::size_t index = 0; index < extents.size(); ++index)
                                    require(backend->deviceToHostOnStream(received[index].data(), outputs[index]->gpu_data_ptr(),
                                        received[index].size(), ordinal, stream), "observe receive bytes");
                                if(overlapped)
                                {
                                    for(std::size_t index=0;index<extents.size();++index)
                                    {
                                        produced[participant][index].resize(inputs[index]->size_bytes());
                                        require(backend->deviceToHostOnStream(produced[participant][index].data(),inputs[index]->gpu_data_ptr(),
                                            produced[participant][index].size(),ordinal,stream),"observe real packet producer bits");
                                        require(backend->deviceToHostOnStream(independent_actual[index].data(),independent_outputs[index]->gpu_data_ptr(),
                                            sizeof(independent_actual[index]),ordinal,stream),"observe independent RMSNorm output");
                                    }
                                    require(backend->deviceToHostOnStream(receipts[participant].data(),receipt_bank,sizeof(receipts[participant]),
                                        ordinal,stream),"observe accumulated raw native ring-send receipts");
                                }
                                require(gpu.synchronizeStreamChecked(stream), "terminal observation");
                                if(overlapped){rendezvous.arrive_and_wait();
                                    require(!tp->isAbortRequested(),"Peer failed during native gather terminal observation");}
                                for (std::size_t index = 0; index < extents.size(); ++index)
                                {
                                    const auto bytes = extents[index];
                                    for (std::size_t byte = 0; byte < received[index].size(); ++byte)
                                    {
                                        const auto bank = byte / bytes;
                                        const auto offset = byte % bytes;
                                        const auto useful = arena_counted ? static_cast<std::size_t>(live) * (bytes / row_capacity) : bytes;
                                        const auto expected = bank < 2 && offset < useful
                                            ? (overlapped ? produced[bank][index][offset]
                                                : payloadByte(static_cast<int>(bank), request, offset))
                                            : std::uint8_t{0xa5};
                                        require(received[index][byte] == expected,
                                            "native allgather changed payload, participant order or unused capacity");
                                    }
                                    if(overlapped)
                                    {
                                        double squared=0;
                                        for(std::size_t element=0;element<inputs[index]->numel();++element)
                                        {
                                            float input;std::memcpy(&input,sent[index].data()+element*sizeof(float),sizeof(input));
                                            squared+=double(input)*input;
                                        }
                                        const auto inverse=1./std::sqrt(squared/inputs[index]->numel()+1e-6);
                                        for(std::size_t element=0;element<inputs[index]->numel();++element)
                                        {
                                            float input,actual;std::memcpy(&input,sent[index].data()+element*sizeof(float),sizeof(input));
                                            std::memcpy(&actual,produced[participant][index].data()+element*sizeof(float),sizeof(actual));
                                            require(std::isfinite(actual)&&std::abs(double(actual)-double(input)*inverse)<2e-5,
                                                "actual packet RMSNorm producer is invalid");
                                        }
                                        for(const auto value:independent_actual[index])
                                            require(std::isfinite(value)&&std::abs(double(value)-1./std::sqrt(1.+1e-6))<2e-5,
                                                "independent native overlap work was skipped or corrupted");
                                        // The source-authored native ring counter adds
                                        // actual clipped primitive sends, including P-1
                                        // outgoing links. Assert EACH owner: a sum could
                                        // hide a missing/duplicated participant receipt.
                                        const auto expected_sends=expected_window_rows*(extents[index]/row_capacity)*
                                            static_cast<std::uint64_t>(tp->degree()-1);
                                        for(const auto &owner_receipts:receipts)
                                            require(owner_receipts[index]==expected_sends,
                                                "overlapped native ring-send receipt lost accumulation or included inactive capacity");
                                    }
                                }
                                if(overlapped){rendezvous.arrive_and_wait();
                                    require(!tp->isAbortRequested(),"Peer failed after native gather validation");}
                            }
                            rendezvous.arrive_and_wait();
                            require(!tp->isAbortRequested(),"Peer failed before native gather count retirement");
                            if (count) {
                                require(count->releaseDeviceMemory(), "retire completed count allocation");
                                bool rejected = false;
                                try { (void)graph.getNode("native_byte_gather_0")->stage->graphCaptureVariantSignature(); }
                                catch (const std::logic_error &) { rejected = true; }
                                require(rejected, "changed count storage must invalidate prepared recording before enqueue");
                            }
                            capture->reset();
                        }
                        catch (...)
                        {
                            // Abort peers before retained graphs/tensors retire:
                            // their streams may still be inside this collective.
                            tp->requestAbort();
                            throw;
                        }
                    });
                }
                catch (...)
                {
                    errors[participant] = std::current_exception();
                    tp->requestAbort();
                    rendezvous.arrive_and_drop();
                }
            });
        }
        for (auto &thread : threads) thread.join();
        if(overlapped)
        {
            for(const auto device:devices)
            {
                EXPECT_EQ(receipt_memory->claimedBytes(device,PhysicalMemoryOwner::ExecutionWorkspace,
                    PhysicalMemoryMaterializationKind::NewAllocation),0u);
            }
        }
        if(drop!=GatherPeerDrop::None) {
            require(tp->isAbortRequested(),"Gather peer drop did not publish cancellation");
            std::size_t injected=0,guarded=0;
            for(int participant=0;participant<2;++participant) {
                const auto &observation=observations[participant];
                require(observation.setupJoins==1,"Dropped gather setup owners retired before their exact stream");
                require(observation.captureAttempts==(drop==GatherPeerDrop::BeforeCapture?0u:1u) &&
                    observation.instantiated==(drop==GatherPeerDrop::BeforeCapture?0u:1u) && !observation.launchAttempts,
                    "Surviving dropped peer reached a native capture/submission boundary");
                require(bool(errors[participant]),"Dropped native gather peer unexpectedly completed");
                try{std::rethrow_exception(errors[participant]);}
                catch(const InjectedGatherPeerDrop &){require(participant==1,"Wrong native gather peer failed");++injected;}
                catch(const std::runtime_error &failure){const std::string_view expected=drop==GatherPeerDrop::BeforeCapture?
                    "Peer failed before native gather capture":"Peer failed before native gather replay";
                    require(participant==0&&std::string_view(failure.what())==expected,"Unrelated failure satisfied dropped gather control");++guarded;}
            }
            require(injected==1&&guarded==1,"Native gather peer-drop control did not reject both owners");
            return;
        }
        for (const auto &error : errors)
            if (error)
            {
                try { std::rethrow_exception(error); }
                catch (const std::exception &failure) { ADD_FAILURE() << failure.what(); }
            }
    }
} // namespace

#ifdef HAVE_CUDA
TEST(Test__NativeAllGatherStage, CUDA) { verifyNativeAllGather(DeviceId::cuda(0)); }
TEST(Test__NativeAllGatherStage, CUDAArenaCount) { verifyNativeAllGather(DeviceId::cuda(0), true); }
TEST(Test__NativeAllGatherStage, CUDAArenaCountM1) { verifyNativeAllGather(DeviceId::cuda(0), true, 1); }
TEST(Test__NativeAllGatherStage, CUDAArenaCountOverlap) { verifyNativeAllGather(DeviceId::cuda(0), true, 3, true); }
TEST(Test__NativeAllGatherStage, CUDAArenaCountOverlapM1) {
    verifyNativeAllGather(DeviceId::cuda(0), true, 1, true);
    for(const auto drop:{GatherPeerDrop::BeforeCapture,GatherPeerDrop::BeforeReplay})
        verifyNativeAllGather(DeviceId::cuda(0),true,1,true,drop);
}
#endif
#ifdef HAVE_ROCM
TEST(Test__NativeAllGatherStage, ROCm) { verifyNativeAllGather(DeviceId::rocm(0)); }
TEST(Test__NativeAllGatherStage, ROCmArenaCount) { verifyNativeAllGather(DeviceId::rocm(0), true); }
TEST(Test__NativeAllGatherStage, ROCmArenaCountM1) { verifyNativeAllGather(DeviceId::rocm(0), true, 1); }
TEST(Test__NativeAllGatherStage, ROCmArenaCountOverlap) { verifyNativeAllGather(DeviceId::rocm(0), true, 3, true); }
TEST(Test__NativeAllGatherStage, ROCmArenaCountOverlapM1) {
    verifyNativeAllGather(DeviceId::rocm(0), true, 1, true);
    for(const auto drop:{GatherPeerDrop::BeforeCapture,GatherPeerDrop::BeforeReplay})
        verifyNativeAllGather(DeviceId::rocm(0),true,1,true,drop);
}
#endif
