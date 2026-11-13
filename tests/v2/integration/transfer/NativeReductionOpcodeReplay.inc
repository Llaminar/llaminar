/**
 * @file NativeReductionOpcodeReplay.inc
 * @brief Actual original-rank captured MIN/MAX semantics across reset and inactive participants.
 *
 * One observed NUMA-local source/PMA/context owns every scalar, native graph,
 * stream and terminal. Distinct signed inputs and opposite neutral sentinels
 * distinguish the native ABI operations; a one-GPU reduction cannot do so.
 * Twenty replays reuse one captured graph per device, with empty, skewed and
 * changing active membership. The rooted API publishes results only on the
 * declared root; every other receive slot and both guards remain untouched.
 * Scalar metadata is the entire live payload.
 */
#include <limits>
#include "collective/OriginalNativeGraphCaptureGroup.h"
namespace {
/** @brief Run two exact one-word reductions on each actual original native member. */
void originalNativeReductionReplay(DeviceType family, int degree)
{
    using Graph = llaminar2::test::RetainedParticipantGraphExecutableFixtureAccess;
    constexpr int reduction_root = 0;
    originalRankLocalReplay(family, degree, [&](const auto &devices) {
        auto original = llaminar2::test::PreparedNativeLookupFixture::prepare(devices,
            MPIContextFactory::global()->rank());
        std::vector<GlobalDeviceAddress> members;
        for (auto device : devices) members.push_back(GlobalDeviceAddress::fromLocalDeviceId(device));
        auto native = std::make_unique<LocalTPContext>(members, std::vector<float>{},
            family == DeviceType::CUDA ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL);
        const CPUTensorStorageDeclaration scalar(1, 6, BufferTensorType::INT32,
            PhysicalMemoryOwner::ExecutionWorkspace, DeviceId::cpu());
        PhysicalMemoryPlanBuilder plan;
        plan.add(original->resource(DeviceId::cpu()), PhysicalMemoryOwner::ExecutionWorkspace,
            devices.size() * scalar.physicalBytes());
        for (auto device : devices) {
            plan.add(original->resource(device), PhysicalMemoryOwner::ExecutionWorkspace, 6 * sizeof(int32_t));
            plan.add(original->resource(device), PhysicalMemoryOwner::LocalCollective,
                CollectiveMemoryEstimator::nativePipelineBoundaryBytes(
                    family == DeviceType::CUDA ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL));
            const auto graph_bom=RetainedParticipantGraphExecutable::generalBOM(
                original->resource(DeviceId::cpu()),original->resource(device),1);
            for(const auto& line:graph_bom.resources())plan.add(line);
        }
        original->admit(plan.build());
        auto memory = original->memory();
        require(native->reserveGraphCaptureBoundaryResources(memory), "Native reduction original cohort BOM failed");
        {
            std::vector<std::shared_ptr<IWorkerGPUContext>> workers;
            std::vector<void*> streams;
            std::vector<std::shared_ptr<TensorBase>> words;
            std::vector<std::shared_ptr<RetainedParticipantGraphExecutable>> graphs;
            std::vector<void*> addresses;
            for (auto device : devices) {
                workers.push_back(original->permit(device).worker()); streams.push_back(workers.back()->defaultStream());
                words.push_back(scalar.materialize(memory));
                const auto member=workers.size()-1;
                workers.back()->submitAndWait([&,member] {
                    auto* values=static_cast<int32_t*>(words[member]->raw_mutable_data());
                    std::fill_n(values,6,0);
                    TransferEngine::publishHostWrite(words[member].get());
                    TransferEngine::prepareDeviceInput(words[member].get(),devices[member],streams[member]);
                    TransferEngine::requireDeviceInput(words[member].get(),devices[member],streams[member]);
                });
                addresses.push_back(words.back()->gpu_data_ptr());
                graphs.push_back(Graph::original(original->permit(device),streams.back()));
            }
            const auto await = [&] {
                const auto deadline=std::chrono::steady_clock::now()+
                    std::chrono::milliseconds(collective_timeout_policy::kDefaultCollectiveTimeoutMs);
                for(std::size_t member=0;member<devices.size();++member) for(;;) {
                    GPUStreamExecutionState state=GPUStreamExecutionState::Pending;
                    workers[member]->submitAndWait([&]{state=workers[member]->queryStreamExecutionState(
                        streams[member],"original captured reduction terminal");});
                    if(state==GPUStreamExecutionState::Complete)break;
                    require(std::chrono::steady_clock::now()<deadline,"Native reduction original terminal timed out");
                    std::this_thread::yield();
                }
            };
            NativeReplayFailureDrain drain;
            drain.abort_and_join=[&] {
                require(native->signalNativeCompletionAbort(),"Native reduction original abort failed");
                await();
                for(const auto& graph:graphs)require(Graph::observeTerminal(graph),"Native reduction retirement was not terminal");
                graphs.clear();
            };
            std::vector<OriginalNativeGraphRecording> recordings;
            for(std::size_t member=0;member<devices.size();++member)
                recordings.push_back({workers[member],devices[member],&Graph::capture(graphs[member]),nullptr,[&,member] {
                    auto* values=static_cast<int32_t*>(words[member]->gpu_data_ptr());
                    require(native->reduceRawOnStream(values+1,values+3,1,CollectiveDataType::INT32,
                        CollectiveOp::ALLREDUCE_MIN,reduction_root,int(member),streams[member],"original_native_minimum"),
                        "Native captured MIN recording failed");
                    require(native->reduceRawOnStream(values+2,values+4,1,CollectiveDataType::INT32,
                        CollectiveOp::ALLREDUCE_MAX,reduction_root,int(member),streams[member],"original_native_maximum"),
                        "Native captured MAX recording failed");
                }});
            OriginalNativeGraphCaptureGroup::record(recordings,native->borrowOriginalNativeContext(),"original MIN/MAX ABI");
            for(std::size_t member=0;member<devices.size();++member)workers[member]->submitAndWait([&,member] {
                auto& graph=Graph::capture(graphs[member]);
                require(graph.prepareRuntimeContextStorage(memory)&&graph.instantiate(),"Native reduction original graph admission failed");
            });
            for(uint64_t replay=1;replay<=20;++replay) {
                int32_t minimum=std::numeric_limits<int32_t>::max(),maximum=std::numeric_limits<int32_t>::min();
                for(std::size_t member=0;member<devices.size();++member) {
                    // The last real member models a storage-only/inactive tail.
                    // Other rounds exercise all-empty, one-live and all-live input.
                    const bool active=replay%5!=0 && (replay%3==0 || member+1<devices.size()) &&
                        (replay%4!=0 || member==replay%devices.size());
                    const int32_t value=-int32_t(3*member+7*replay);
                    const int32_t min_input=active?value:std::numeric_limits<int32_t>::max();
                    const int32_t max_input=active?value:std::numeric_limits<int32_t>::min();
                    minimum=std::min(minimum,min_input); maximum=std::max(maximum,max_input);
                    workers[member]->submitAndWait([&,member,min_input,max_input] {
                        auto* values=static_cast<int32_t*>(words[member]->raw_mutable_data());
                        values[0]=0x1357;values[1]=min_input;values[2]=max_input;
                        values[3]=0x7788;values[4]=0x7788;values[5]=0x2468;
                        TransferEngine::publishHostWrite(words[member].get());
                        TransferEngine::prepareDeviceInput(words[member].get(),devices[member],streams[member]);
                        TransferEngine::requireDeviceInput(words[member].get(),devices[member],streams[member]);
                        require(words[member]->gpu_data_ptr()==addresses[member],"Native reduction input storage changed across reset");
                    });
                }
                std::vector<std::future<void>> pending;
                std::exception_ptr failure; std::mutex failure_mutex;
                const auto abort=[&](std::exception_ptr error) noexcept {
                    {std::lock_guard lock(failure_mutex);if(!failure)failure=std::move(error);}
                    try{if(!native->signalNativeCompletionAbort())std::terminate();}catch(...){std::terminate();}
                };
                try {
                    for(std::size_t member=0;member<devices.size();++member)
                        pending.push_back(workers[member]->submitAsync([&,member] {
                            try{require(Graph::submit(graphs[member],replay),"Native reduction original submit failed");}
                            catch(...){abort(std::current_exception());throw;}
                        }));
                }catch(...){abort(std::current_exception());}
                joinOriginalNativeWorkerSubmissions(pending,abort,
                    std::chrono::steady_clock::now()+std::chrono::milliseconds(collective_timeout_policy::kDefaultCollectiveTimeoutMs));
                if(failure)std::rethrow_exception(failure);
                await();
                for(const auto& graph:graphs)require(Graph::observeTerminal(graph),"Native reduction graph did not retire");
                for(std::size_t member=0;member<devices.size();++member) {
                    workers[member]->submitAndWait([&,member] {
                        // This component's whole-parent terminal is already acquired.
                        // Observe only its six original scalar words into the
                        // admitted CPU root; no runtime state feeds execution.
                        require(getBackendFor(devices[member])->deviceToHostOnStream(words[member]->raw_mutable_data(),
                            words[member]->gpu_data_ptr(),6*sizeof(int32_t),devices[member].ordinal,streams[member]),
                            "Native reduction original terminal observation failed");
                    });
                }
                await();
                for(std::size_t member=0;member<devices.size();++member) {
                    SCOPED_TRACE(::testing::Message() << "replay=" << replay << " member=" << member
                        << " root=" << reduction_root << " device=" << devices[member].toString());
                    const auto* values=static_cast<const int32_t*>(words[member]->raw_data());
                    EXPECT_EQ(values[0],0x1357);EXPECT_EQ(values[5],0x2468);
                    // reduceRawOnStream maps to native rooted Reduce, not AllReduce.
                    // Only the declared root owns a result; the other receive
                    // slots are ignored and must retain their original poison.
                    if (member == static_cast<std::size_t>(reduction_root)) {
                        EXPECT_EQ(values[3],minimum);EXPECT_EQ(values[4],maximum);
                    } else {
                        EXPECT_EQ(values[3],0x7788);EXPECT_EQ(values[4],0x7788);
                    }
                }
            }
            graphs.clear();drain.completed=true; words.clear(); workers.clear();
        }
        native.reset();original->requirePayloadRetired();original->retire();original->requireExactRetired();
    });
}
}
#ifdef HAVE_CUDA
TEST(NativeReductionOpcodes, CUDA2CapturedMinMaxReplay) { originalNativeReductionReplay(DeviceType::CUDA,2); }
#endif
#ifdef HAVE_ROCM
TEST(NativeReductionOpcodes, ROCm2CapturedMinMaxReplay) { originalNativeReductionReplay(DeviceType::ROCm,2); }
TEST(NativeReductionOpcodes, ROCm4CapturedMinMaxReplay) { originalNativeReductionReplay(DeviceType::ROCm,4); }
#endif
