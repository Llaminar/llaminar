/**
 * @file Test__LocalTPIndexedHostKV.cpp
 * @brief Actual host-owned indexed gather and phase-split KV row replay.
 *
 * Tiny CPU tensors, declared logical NUMA addresses and the existing workspace
 * manager exercise the production collective/stage without model weights or
 * native GPUs. Independent literal rows expose arrival-order aliases, stale
 * tails, wrong row strides, empty work and reuse of old collective generations.
 */
#include "collective/LocalTPContext.h"
#include "execution/compute_stages/stages/TPKVCacheStateAllGatherStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "backends/BackendManager.h"
#include "tensors/Tensors.h"
#include "utils/DebugEnv.h"
#include "utils/PerfStatsCollector.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <barrier>
#include <cstddef>
#include <cstdlib>
#include <map>
#include <future>
#include <chrono>
#include <optional>
#include <memory>
#include <numeric>
#include <span>
#include <thread>
#include <vector>

namespace llaminar2
{
/** @brief Synchronize the actual private wait boundary without model or payload access. */
struct LocalTPIndexedHostKVTestPeer
{
    /** @return An owning token for the same mutex guarding the production CV predicate. */
    static std::unique_lock<std::mutex> hold(LocalTPContext &context)
    { return std::unique_lock<std::mutex>(context.host_allgather_mutex_); }
    /** @return Whether the one canonical fatal-cancellation flag is published. */
    static bool aborted(const LocalTPContext &context)
    { return context.abort_requested_.load(std::memory_order_acquire); }
    /** @brief Release into the actual CV wait; abort must acquire this boundary before notification. */
    static bool waitForAbort(LocalTPContext &context,std::unique_lock<std::mutex> &lock)
    {
        return context.host_allgather_cv_.wait_for(lock,std::chrono::seconds(1),[&]{return aborted(context);});
    }
};
}
using namespace llaminar2;
namespace
{
/** @return Logical CPU endpoints only; no hardware detection or NUMA allocation is performed. */
std::vector<GlobalDeviceAddress> cpuMembers(int count, bool reverse = false)
{
    std::vector<GlobalDeviceAddress> addresses;
    for (int member = 0; member < count; ++member)
        addresses.push_back(GlobalDeviceAddress::cpu(member));
    if (reverse) std::reverse(addresses.begin(), addresses.end());
    return addresses;
}
/** @brief Launch participant workers in reverse order and join every real result. */
template <class Operation>
std::vector<int> invoke(int members, Operation operation)
{
    std::barrier<> start(members);
    std::vector<int> result(members, 0);
    std::vector<std::jthread> workers;
    workers.reserve(members);
    for (int member = members - 1; member >= 0; --member)
        workers.emplace_back([&, member] {
            start.arrive_and_wait();
            result[member] = operation(member) ? 1 : 0;
        });
    for (auto &worker : workers) worker.join();
    return result;
}
/** @brief Retain the process collector configuration while checking exact real communication. */
struct RestoreCollection
{
    std::map<std::string,std::optional<std::string>> prior;
    RestoreCollection()
    {
        for (const char *key : {"LLAMINAR_PERF_STATS_JSON","LLAMINAR_PERF_STATS_FILTER"})
        {
            const char *value=std::getenv(key);
            prior[key]=value ? std::optional<std::string>(value) : std::nullopt;
        }
        ::setenv("LLAMINAR_PERF_STATS_JSON","1",1);
        ::setenv("LLAMINAR_PERF_STATS_FILTER","tp_host_allgather_runtime",1);
        PerfStatsCollector::reset();
    }
    ~RestoreCollection()
    {
        for(const auto &[key,value]:prior)
            if(value) ::setenv(key.c_str(),value->c_str(),1); else ::unsetenv(key.c_str());
        PerfStatsCollector::reset();
    }
};
/** @brief Own the existing bounded-collective knob only during a timeout negative. */
struct RestoreTimeout
{
    int prior = debugEnv().tp_collect_timeout_ms;
    ~RestoreTimeout() { mutableDebugEnv().tp_collect_timeout_ms = prior; }
};
}
/** @test Communicator index, every output and exact live tails survive repeated 1/2/4/8-member gathers. */
TEST(LocalTPIndexedHostKV, DeclaredOrderEveryOutputAndLiveExtentsReplay)
{
    constexpr size_t capacity = 7;
    for (int members : {1, 2, 4, 8})
    for (bool reverse : {false, true})
    {
        SCOPED_TRACE(::testing::Message() << members << " reverse=" << reverse);
        auto context = createLocalTPContext(cpuMembers(members, reverse), {}, CollectiveBackendType::HOST);
        std::vector<std::array<int, capacity>> input(members);
        std::vector<std::vector<int>> output(members, std::vector<int>(capacity * members + 3));
        for (int round = 0; round < 20; ++round)
        {
            const size_t live = std::array<size_t, 4>{7, 1, 0, 3}[round % 4];
            for (int member = 0; member < members; ++member)
            {
                std::fill(output[member].begin(), output[member].end(), -991);
                for (size_t item = 0; item < capacity; ++item)
                    input[member][item] = round * 10000 + member * 100 + item;
            }
            const auto result = invoke(members, [&](int member) {
                return context->allgatherHostIndexed(
                    std::as_bytes(std::span(input[member].data(), live)),
                    std::as_writable_bytes(std::span(output[member].data(), live * members)), member, "literal_live_gather");
            });
            ASSERT_EQ(result, std::vector<int>(members, 1));
            for (const auto &destination : output)
            {
                for (int owner = 0; owner < members; ++owner)
                    for (size_t item = 0; item < live; ++item)
                        EXPECT_EQ(destination[owner * live + item], round * 10000 + owner * 100 + item);
                for (size_t tail = live * members; tail < destination.size(); ++tail)
                    EXPECT_EQ(destination[tail], -991);
            }
        }
    }
}
/** @test Duplicate identity, stage/count mismatches and aliases reject atomically before output writes. */
TEST(LocalTPIndexedHostKV, MalformedGenerationsAbortEveryParticipantWithoutPublication)
{
    for (int fault = 0; fault < 5; ++fault)
    {
        SCOPED_TRACE(fault);
        auto context = createLocalTPContext(cpuMembers(2), {}, CollectiveBackendType::HOST);
        std::array<int, 2> input{11, 22};
        std::array<std::array<int, 4>, 2> output;
        for (auto &row : output) row.fill(-881);
        const std::string common = "reject_round", foreign = "foreign_round";
        const auto result = invoke(2, [&](int member) {
            const int participant = fault == 0 ? 0 : member;
            const auto &name = fault == 1 && member == 1 ? foreign : common;
            const size_t live = fault == 2 && member == 1 ? 0 : 1;
            int *receive = fault == 3 ? output[0].data() : output[member].data();
            const int *send = fault == 4 ? output[member].data() : &input[member];
            return context->allgatherHostIndexed(std::as_bytes(std::span(send, live)),
                std::as_writable_bytes(std::span(receive, live * 2)), participant, name);
        });
        EXPECT_EQ(result, (std::vector<int>{0, 0}));
        for (const auto &row : output) for (int value : row) EXPECT_EQ(value, -881);
        EXPECT_FALSE(context->allgatherHostIndexed({}, {}, 0, common));
    }
    auto context = createLocalTPContext(cpuMembers(2), {}, CollectiveBackendType::HOST);
    std::array<std::byte, 3> input{}, output{};
    EXPECT_FALSE(context->allgatherHostIndexed(input, output, 0, "wrong_extent"));
    auto duplicate = createLocalTPContext({GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(0)}, {}, CollectiveBackendType::HOST);
    EXPECT_FALSE(duplicate->allgatherHostIndexed({}, {}, 0, "duplicate_address"));
}
/** @test Missing peers fail at the existing short configured timeout, with no partial output. */
TEST(LocalTPIndexedHostKV, MissingPeerUsesCanonicalBoundedFailureAndRetires)
{
    RestoreTimeout restore;
    mutableDebugEnv().tp_collect_timeout_ms = 2;
    auto context = createLocalTPContext(cpuMembers(2), {}, CollectiveBackendType::HOST);
    const std::array<int, 1> input{17};
    std::array<int, 2> output{-771, -771};
    EXPECT_FALSE(context->allgatherHostIndexed(std::as_bytes(std::span(input)),
        std::as_writable_bytes(std::span(output)), 0, "missing_peer"));
    EXPECT_EQ(output, (std::array<int, 2>{-771, -771}));
    EXPECT_FALSE(context->allgatherHostIndexed({}, {}, 1, "after_abort"));
}
/** @test The retired arrival-ordered CPU API cannot silently invent a participant. */
TEST(LocalTPIndexedHostKV, CoarseTensorIdentityCannotEnterLegacyGather)
{
    auto context = createLocalTPContext(cpuMembers(2), {}, CollectiveBackendType::HOST);
    FP32Tensor input({1, 3}), output({1, 6});
    EXPECT_THROW(context->allgather(&input, &output), std::invalid_argument);
}
/** @test Actual CPU stage gathers/deinterleaves only live rows through a retained admitted workspace. */
TEST(LocalTPIndexedHostKV, ProductionKVStageReplaysPartialEmptyAndDirtyFamilies)
{
    RestoreCollection collection;
    initCPUBackend(-1);
    constexpr int capacity = 8, width = 3, stride = 5;
    constexpr std::array<int, 4> counts{0, 1, 3, capacity};
    for (int members : {1, 2, 4, 8})
    for (bool reverse : {false, true})
    {
        SCOPED_TRACE(::testing::Message() << members << " reverse=" << reverse);
        auto collective = createLocalTPContext(cpuMembers(members, reverse), {}, CollectiveBackendType::HOST);
        struct Participant
        {
            FP32Tensor k{{capacity, stride}}, v{{capacity, stride}};
            std::unique_ptr<FP32Tensor> full_k, full_v;
            std::unique_ptr<DeviceWorkspaceManager> workspace;
            std::array<std::unique_ptr<TPKVCacheStateAllGatherStage>, 4> family;
        };
        std::vector<std::unique_ptr<Participant>> participants;
        for (int member = 0; member < members; ++member)
        {
            auto owner = std::make_unique<Participant>();
            owner->full_k = std::make_unique<FP32Tensor>(std::vector<size_t>{capacity, static_cast<size_t>(members * width)});
            owner->full_v = std::make_unique<FP32Tensor>(std::vector<size_t>{capacity, static_cast<size_t>(members * width)});
            for (size_t variant = 0; variant < counts.size(); ++variant)
                owner->family[variant] = std::make_unique<TPKVCacheStateAllGatherStage>(TPKVCacheStateAllGatherStage::Params{
                    .device_id=DeviceId::cpu(), .tp_ctx=collective.get(), .local_K=&owner->k, .local_V=&owner->v,
                    .full_K=owner->full_k.get(), .full_V=owner->full_v.get(), .layer_idx=0, .tp_device_idx=member,
                    .tokens=counts[variant], .local_kv_dim=width, .full_kv_dim=members*width,
                    .local_k_stride=stride, .local_v_stride=stride, .stage_name="actual_kv_rows"});
            auto requirements = owner->family.back()->getWorkspaceRequirements(0);
            ASSERT_EQ(requirements.buffers.size(), 4u);
            size_t literal_payload_bytes = 0;
            for (const auto &buffer : requirements.buffers)
            {
                const bool compact = buffer.name.find("compact") != std::string::npos;
                EXPECT_EQ(buffer.size_bytes, size_t(capacity * width * (compact ? 1 : members)) * sizeof(float));
                literal_payload_bytes += buffer.size_bytes;
            }
            EXPECT_EQ(literal_payload_bytes, size_t(2 * capacity * width * (members + 1)) * sizeof(float));
            EXPECT_EQ(owner->family.back()->estimatedMemoryBytes(),literal_payload_bytes);
            owner->workspace = std::make_unique<DeviceWorkspaceManager>(DeviceId::cpu(), 64 * 1024);
            ASSERT_TRUE(owner->workspace->allocate(requirements));
            EXPECT_EQ(owner->workspace->bufferCount(), 4u);
            for (auto &stage : owner->family) stage->bindWorkspace(owner->workspace.get());
            participants.push_back(std::move(owner));
        }
        for (int round = 0; round < 20; ++round)
        {
            const size_t variant = round % counts.size();
            const int live = counts[variant];
            for (int member = 0; member < members; ++member)
            {
                auto &owner = *participants[member];
                std::fill_n(owner.k.mutable_data(), owner.k.numel(), -661.0f);
                std::fill_n(owner.v.mutable_data(), owner.v.numel(), -662.0f);
                std::fill_n(owner.full_k->mutable_data(), owner.full_k->numel(), -663.0f);
                std::fill_n(owner.full_v->mutable_data(), owner.full_v->numel(), -664.0f);
                for (int row = 0; row < capacity; ++row)
                    for (int col = 0; col < width; ++col)
                    {
                        owner.k.mutable_data()[row * stride + col] = float(round * 10000 + member * 1000 + row * 10 + col);
                        owner.v.mutable_data()[row * stride + col] = -owner.k.data()[row * stride + col];
                    }
            }
            PerfStatsCollector::reset();
            const auto result = invoke(members, [&](int member) {
                CPUDeviceContext context(DeviceId::cpu(), 1);
                return participants[member]->family[variant]->execute(&context);
            });
            ASSERT_EQ(result, std::vector<int>(members, 1));
            std::map<std::string,double> observed;
            for(const auto &record:PerfStatsCollector::snapshot({"tp_host_allgather_runtime"}))
                observed[record.name]+=record.value;
            EXPECT_EQ(observed["calls"],2.0*members);
            EXPECT_EQ(observed["send_bytes"],double(2*members*live*width*sizeof(float)));
            EXPECT_EQ(observed["receive_bytes"],double(2*members*live*members*width*sizeof(float)));
            for (const auto &owner : participants)
                for (int row = 0; row < capacity; ++row)
                    for (int member = 0; member < members; ++member)
                        for (int col = 0; col < width; ++col)
                        {
                            const int index = row * members * width + member * width + col;
                            const float value = float(round * 10000 + member * 1000 + row * 10 + col);
                            EXPECT_EQ(owner->full_k->data()[index], row < live ? value : -663.0f);
                            EXPECT_EQ(owner->full_v->data()[index], row < live ? -value : -664.0f);
                        }
        }
    }
}
/** @test Actual stage fails before a collective if its retained workspace cannot cover live rows. */
TEST(LocalTPIndexedHostKV, ProductionKVStageRejectsInsufficientWorkspaceBeforePublication)
{
    initCPUBackend(-1);
    auto collective = createLocalTPContext(cpuMembers(1), {}, CollectiveBackendType::HOST);
    FP32Tensor k({8, 3}), v({8, 3}), full_k({8, 3}), full_v({8, 3});
    std::fill_n(full_k.mutable_data(), full_k.numel(), -551.0f);
    auto params = TPKVCacheStateAllGatherStage::Params{
        .device_id=DeviceId::cpu(), .tp_ctx=collective.get(), .local_K=&k, .local_V=&v,
        .full_K=&full_k, .full_V=&full_v, .layer_idx=0, .tp_device_idx=0,
        .tokens=1, .local_kv_dim=3, .full_kv_dim=3, .local_k_stride=3, .local_v_stride=3,
        .stage_name="admitted_one_row"};
    TPKVCacheStateAllGatherStage small(params);
    DeviceWorkspaceManager workspace(DeviceId::cpu(), 64 * 1024);
    ASSERT_TRUE(workspace.allocate(small.getWorkspaceRequirements(0)));
    params.tokens = 8;
    TPKVCacheStateAllGatherStage oversized(params);
    oversized.bindWorkspace(&workspace);
    CPUDeviceContext context(DeviceId::cpu(), 1);
    EXPECT_FALSE(oversized.execute(&context));
    for (size_t i = 0; i < full_k.numel(); ++i) EXPECT_EQ(full_k.data()[i], -551.0f);
}

/** @test Persistent workers reenter the same protocol with no inter-round barrier or recapture. */
TEST(LocalTPIndexedHostKV, ReusedWorkersRetireEveryGenerationBeforeBackToBackPublication)
{
    for(int members:{1,2,4,8})
    {
        auto context=createLocalTPContext(cpuMembers(members,true),{},CollectiveBackendType::HOST);
        std::vector<std::array<int,7>> inputs(members);
        std::vector<std::vector<int>> outputs(members,std::vector<int>(7*members,-441));
        const auto result=invoke(members,[&](int member){
            for(int round=0;round<40;++round)
            {
                const size_t live=std::array<size_t,4>{7,1,0,3}[round%4];
                for(size_t item=0;item<7;++item)inputs[member][item]=round*10000+member*100+item;
                std::fill(outputs[member].begin(),outputs[member].end(),-441);
                if(member==0 && round%2)std::this_thread::yield();
                if(!context->allgatherHostIndexed(std::as_bytes(std::span(inputs[member].data(),live)),
                    std::as_writable_bytes(std::span(outputs[member].data(),live*members)),member,"persistent_round"))
                    return false;
                for(int owner=0;owner<members;++owner)
                    for(size_t item=0;item<live;++item)
                        if(outputs[member][owner*live+item]!=round*10000+owner*100+int(item))return false;
                for(size_t item=live*members;item<outputs[member].size();++item)
                    if(outputs[member][item]!=-441)return false;
            }
            return true;
        });
        EXPECT_EQ(result,std::vector<int>(members,1));
    }
}
/** @test Packed full-row output width is a real layout contract, not tensor capacity. */
TEST(LocalTPIndexedHostKV, ProductionKVStageRejectsWrongFullRowLayoutWithoutWrites)
{
    initCPUBackend(-1);
    for(int wrong_role:{0,1})
    {
        auto collective=createLocalTPContext(cpuMembers(1),{},CollectiveBackendType::HOST);
        FP32Tensor k({3,3}),v({3,3}),full_k({3,size_t(wrong_role==0?4:3)}),
            full_v({3,size_t(wrong_role==1?4:3)});
        std::fill_n(full_k.mutable_data(),full_k.numel(),-331.0f);
        std::fill_n(full_v.mutable_data(),full_v.numel(),-332.0f);
        TPKVCacheStateAllGatherStage stage({.device_id=DeviceId::cpu(),.tp_ctx=collective.get(),
            .local_K=&k,.local_V=&v,.full_K=&full_k,.full_V=&full_v,.layer_idx=0,.tp_device_idx=0,
            .tokens=3,.local_kv_dim=3,.full_kv_dim=3,.local_k_stride=3,.local_v_stride=3,
            .stage_name="wrong_full_layout"});
        DeviceWorkspaceManager workspace(DeviceId::cpu(),64*1024);
        ASSERT_TRUE(workspace.allocate(stage.getWorkspaceRequirements(0)));
        stage.bindWorkspace(&workspace);CPUDeviceContext context(DeviceId::cpu(),1);
        EXPECT_FALSE(stage.execute(&context));
        for(size_t i=0;i<full_k.numel();++i)EXPECT_EQ(full_k.data()[i],-331.0f);
        for(size_t i=0;i<full_v.numel();++i)EXPECT_EQ(full_v.data()[i],-332.0f);
    }
}

/** @test CPU handoff cannot read a GPU owner or an invalid canonical host state. */
TEST(LocalTPIndexedHostKV, ProductionKVStageRejectsForeignOrInvalidStorageOwners)
{
    initCPUBackend(-1);
    enum class Fault { GPUAuthority, InvalidHost };
    /** @brief Device-free ownership negative; no accelerator allocation exists. */
    class InvalidOwner final : public FP32Tensor
    {
    public:
        /** @brief Keep real host storage while exposing the selected invalid authority. */
        explicit InvalidOwner(Fault fault) : FP32Tensor({1, 3}), fault_(fault) {}
        /** @return A foreign GPU authority for that negative control only. */
        std::optional<DeviceId> current_device() const override
        {
            return fault_ == Fault::GPUAuthority
                ? std::optional<DeviceId>{DeviceId::cuda(0)} : std::nullopt;
        }
        /** @return The canonical coherence violation independently of placement. */
        TensorCoherenceState coherenceState() const override
        {
            return fault_ == Fault::InvalidHost
                ? TensorCoherenceState::INVALID : TensorCoherenceState::HOST_ONLY;
        }
    private:
        Fault fault_;
    };
    for (const auto fault : {Fault::GPUAuthority, Fault::InvalidHost})
    for (int role = 0; role < 4; ++role)
    {
        auto collective = createLocalTPContext(cpuMembers(1), {}, CollectiveBackendType::HOST);
        FP32Tensor k({1, 3}), v({1, 3}), full_k({1, 3}), full_v({1, 3});
        InvalidOwner invalid(fault);
        std::fill_n(full_k.mutable_data(), full_k.numel(), -221.0f);
        std::fill_n(full_v.mutable_data(), full_v.numel(), -222.0f);
        auto params = TPKVCacheStateAllGatherStage::Params{
            .device_id = DeviceId::cpu(), .tp_ctx = collective.get(),
            .local_K = &k, .local_V = &v, .full_K = &full_k, .full_V = &full_v,
            .layer_idx = 0, .tp_device_idx = 0, .tokens = 1, .local_kv_dim = 3,
            .full_kv_dim = 3, .local_k_stride = 3, .local_v_stride = 3,
            .stage_name = "invalid_storage_owner"};
        switch (role)
        {
        case 0: params.local_K = &invalid; break;
        case 1: params.local_V = &invalid; break;
        case 2: params.full_K = &invalid; break;
        case 3: params.full_V = &invalid; break;
        }
        TPKVCacheStateAllGatherStage stage(params);
        DeviceWorkspaceManager workspace(DeviceId::cpu(), 64 * 1024);
        ASSERT_TRUE(workspace.allocate(stage.getWorkspaceRequirements(0)));
        stage.bindWorkspace(&workspace);
        CPUDeviceContext context(DeviceId::cpu(), 1);
        EXPECT_FALSE(stage.execute(&context));
        for (size_t i = 0; i < full_k.numel(); ++i) EXPECT_EQ(full_k.data()[i], -221.0f);
        for (size_t i = 0; i < full_v.numel(); ++i) EXPECT_EQ(full_v.data()[i], -222.0f);
    }
}

/** @test External cancellation owns the real predicate-to-wait boundary and retires every replay. */
TEST(LocalTPIndexedHostKV, ExternalAbortSerializesWithWaitBoundaryAndWakesWithoutTimeout)
{
    for(int replay=0;replay<20;++replay)
    {
        auto interface=createLocalTPContext(cpuMembers(2),{},CollectiveBackendType::HOST);
        auto &context=dynamic_cast<LocalTPContext &>(*interface);
        auto lock=LocalTPIndexedHostKVTestPeer::hold(context);
        std::promise<void> entered,finished;
        auto entered_future=entered.get_future(),finished_future=finished.get_future();
        std::jthread aborter([&]{entered.set_value();context.requestAbort();finished.set_value();});
        entered_future.wait();
        // This is a blocking-ownership control, not a throughput threshold:
        // publication is forbidden for the entire interval while we own the
        // mutex, regardless of how fast the cancellation thread is scheduled.
        EXPECT_EQ(finished_future.wait_for(std::chrono::milliseconds(5)),std::future_status::timeout);
        EXPECT_FALSE(LocalTPIndexedHostKVTestPeer::aborted(context));
        const bool woke=LocalTPIndexedHostKVTestPeer::waitForAbort(context,lock);
        lock.unlock();aborter.join();
        EXPECT_TRUE(woke);
        EXPECT_EQ(finished_future.wait_for(std::chrono::milliseconds(0)),std::future_status::ready);
        EXPECT_FALSE(interface->allgatherHostIndexed({}, {}, 0, "retired_abort"));
    }
}
