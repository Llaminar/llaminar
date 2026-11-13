/**
 * @file Test__ResidentImmutableRowTable.cpp
 * @brief Device-free proof of admitted native bytes, bounded upload and read lifetime.
 *
 * An injected GPU-shaped backend stores bytes in host test allocations. The real
 * canonical planner, PMA, ordinary loader finalization, TransferEngine and row
 * source materializer execute; no accelerator, engine DSO or model is required
 * by the test contract. Native runtime qualification is registered separately.
 */
#include "loaders/ResidentImmutableRowTable.h"
#include "models/qwen4exp/Qwen4ExpFixedHeadShards.h"
#include "loaders/ModelContext.h"
#include "loaders/ImmutableRowTableUploadGeometry.h"
#include "loaders/gpu_pipeline/LoadOrchestrator.h"
#include "planning/ImmutableRowTablePlacementPlanner.h"
#include "transfer/TransferEngine.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "mocks/MockBackend.h"
#include <gtest/gtest.h>
#include <array>
#include <algorithm>
#include <cstring>
#include <functional>
#include <optional>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <chrono>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
using namespace llaminar2;
namespace
{
/** @brief Deterministic original bytes, never converted into another table representation. */
class NativeFile final
{
public:
    /** @brief Create a deliberately unaligned original-file tensor extent. */
    explicit NativeFile(std::size_t payload) : bytes(37 + payload)
    {
        char name[] = "/tmp/llaminar-resident-native-XXXXXX";
        fd = ::mkstemp(name); if (fd < 0) throw std::runtime_error("mkstemp failed");
        path = name;
        for (std::size_t i = 0; i < bytes.size(); ++i) bytes[i] = std::byte((i * 29 + 7) % 251);
        if (::write(fd, bytes.data(), bytes.size()) != static_cast<ssize_t>(bytes.size()))
            throw std::runtime_error("write failed");
    }
    /** @brief Remove only this test's private source. */
    ~NativeFile() { ::close(fd); ::unlink(path.c_str()); }
    int fd = -1;
    std::string path;
    std::vector<std::byte> bytes;
};
/** @brief Track exact DMA extents and lifecycle with CPU-only native-shaped allocations. */
class NativeBackend final : public test::MockBackend
{
public:
    NativeBackend() : MockBackend(DeviceType::CUDA) {}
    /** @return Stable injectable runtime generation; no native startup. */
    std::uint64_t deviceRuntimeGeneration(int) const override { return runtime; }
    /** @brief Test pinned owner uses ordinary host backing with exact requested extent. */
    void *allocatePinned(std::size_t bytes, int) override
    { std::lock_guard lock(observations); pinned_sizes.push_back(bytes); return std::malloc(bytes); }
    /** @brief Release the exact mock pinned allocation. */
    void freePinned(void *pointer, int) override { std::free(pointer); }
    /** @brief Byte-preserving DMA with an exact non-null stream and optional failure injection. */
    bool hostToDeviceOnStream(void *dst, const void *src, size_t bytes, int device, void *stream) override
    {
        if (!stream) throw std::logic_error("default stream");
        std::size_t ordinal;
        { std::lock_guard lock(observations); copies.push_back(bytes); ordinal=copies.size(); }
        if (on_copy) on_copy(ordinal,device);
        if (ordinal == fail_copy) return false;
        std::memcpy(dst, src, bytes); return true;
    }
    /** @brief Record actual read-token event joins independently of copy submissions. */
    bool streamWaitEvent(void *stream, void *event, int) override
    { if (!stream || !event) return false; std::lock_guard lock(observations); ++joins; return true; }
    /** @brief Fail the setup drain without releasing any still-live allocation. */
    bool synchronize(int) override { return !fail_drain; }
    bool fail_drain = false;
    std::uint64_t runtime = 1;
    std::vector<size_t> copies, pinned_sizes;
    std::size_t fail_copy = 0, joins = 0;
    std::function<void(std::size_t,int)> on_copy;
    std::mutex observations;
};
NativeBackend *active = nullptr;
/** @brief Inject only the exact simulated CUDA resource. */
IBackend *resolve(DeviceId device)
{ return active && device.is_cuda() && device.ordinal < active->deviceCount() ? active : nullptr; }
/** @brief All resources fit; this does not bypass canonical PMA allocation claims. */
PhysicalMemoryResource resource(DeviceId device)
{ return {.world_rank=0, .device=device, .total_bytes=1u<<24, .admission_available_bytes=1u<<24}; }
/** @brief Real selected placement plus actual ordinary-loader released staging proof. */
class OwnerFixture final
{
public:
    /** @brief Admit source and ordinary loader obligations together, then actually drain staging. */
    OwnerFixture(GGUFTensorType format=GGUFTensorType::IQ4_NL, std::size_t rows=97,int devices=1,bool finalize=true)
        : source{.path="fixture", .tensor_name="per_layer_token_embd.weight", .file_offset=37,
                 .rows=rows, .row_elements=256, .format=format},
          file(source.tensorBytes()), transfers(resolve)
    {
        active = &backend; source.path = file.path; backend.setDeviceCount(devices);
        PhysicalMemoryPlanBuilder higher;
        std::vector<PhysicalMemoryResource> consumers;
        for(int device=0;device<devices;++device)
        {
            consumers.push_back(resource(DeviceId::cuda(device)));
            higher.add(consumers.back(), PhysicalMemoryOwner::PrimaryModelWeights, slot);
            higher.add(consumers.back(), PhysicalMemoryOwner::WeightLoadStaging, slot);
        }
        higher.add(resource(DeviceId::cpu()), PhysicalMemoryOwner::WeightLoadStaging, slot*devices);
        const auto input = ImmutableRowTablePlacementInput::forConsumers(
            ImmutableRowTableExtent::fromNative(source.format, source.rows, source.rowBytes()),
            resource(DeviceId::cpu()), consumers);
        const auto original_geometry=resolveGPUWeightLoadMemoryGeometry(slot,
            {.staging_stream_count=1,.staging_budget_bytes=slot});
        if (!residentImmutableRowUploadFits(source.rowBytes(),original_geometry))
            throw std::invalid_argument("fixture native row does not fit its original ordinary slot");
        const auto selected = ImmutableRowTablePlacementPlanner::select(
            {.storage=ImmutableRowStoragePreference::VRAM}, input, higher.build(),
            {.vram=residentImmutableRowUploadIncrementalBOM()});
        placement = selected.placement;
        memory = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(selected.physical_plan), 0);
        loader = std::make_unique<LoadOrchestrator>(&backend, memory, PhysicalMemoryOwner::PrimaryModelWeights);
        loader->addDevice(0); loader->planRawWeight(0, "ordinary.weight", 1, slot/sizeof(float), slot);
        loader->allocate(slot, 1);
        EXPECT_THROW((void)loader->releasedWeightLoadStaging(0), std::logic_error);
        if(finalize) { loader->finalize(); receipt.emplace(loader->releasedWeightLoadStaging(0)); }
        if(devices==2)
        {
            second_loader=std::make_unique<LoadOrchestrator>(&backend,memory,PhysicalMemoryOwner::PrimaryModelWeights);
            second_loader->addDevice(1);second_loader->planRawWeight(1,"ordinary.weight",1,slot/sizeof(float),slot);
            second_loader->allocate(slot,1);second_loader->finalize();
            second_receipt.emplace(second_loader->releasedWeightLoadStaging(1));
        }
    }
    /** @brief All local test borrowers are destroyed before the backend. */
    ~OwnerFixture() { second_loader.reset(); loader.reset(); active=nullptr; }
    /** @return Real loader/transfer preparation of original source-native rows. */
    std::shared_ptr<const ResidentImmutableRowTable> prepare(void *stream=reinterpret_cast<void *>(0x1234))
    { return ResidentImmutableRowTable::prepare(source, *placement, memory, DeviceId::cuda(0), *receipt, stream, transfers); }
    static constexpr std::size_t slot=4096;
    ImmutableRowTableSource source;
    NativeFile file;
    NativeBackend backend;
    TransferEngine transfers;
    std::shared_ptr<const ResolvedImmutableRowTablePlacement> placement;
    std::shared_ptr<PhysicalMemoryAuthority> memory;
    std::unique_ptr<LoadOrchestrator> loader;
    std::optional<ReleasedWeightLoadStaging> receipt;
    std::unique_ptr<LoadOrchestrator> second_loader;
    std::optional<ReleasedWeightLoadStaging> second_receipt;
};
/** @brief Directory-only metadata injection preserves the production source/materializer boundary. */
template<class Fixture>
std::shared_ptr<ModelContext> contextFor(Fixture &f)
{
    auto context=ModelContext::createForTesting(f.file.path,nullptr,1,false);
    auto &directory=const_cast<GGUFModel &>(context->model());
    directory.split_paths={f.file.path}; directory.data_offset=37;
    directory.tensors.push_back({.name="per_layer_token_embd.weight",.dimensions={f.source.rows,256},
        .type=f.source.format,.offset=0,.size_bytes=f.source.tensorBytes(),.split_idx=0});
    return context;
}
}

TEST(ResidentImmutableRowTable, OriginalNativeBytesUseOnlyDrainedBoundedSlot)
{
    for (const auto format : {GGUFTensorType::F32, GGUFTensorType::F16, GGUFTensorType::BF16,
        GGUFTensorType::Q4_0, GGUFTensorType::Q4_1, GGUFTensorType::Q5_0, GGUFTensorType::Q5_1,
        GGUFTensorType::Q8_0, GGUFTensorType::Q2_K, GGUFTensorType::Q3_K, GGUFTensorType::Q4_K,
        GGUFTensorType::Q5_K, GGUFTensorType::Q6_K, GGUFTensorType::Q8_K, GGUFTensorType::IQ1_S,
        GGUFTensorType::IQ1_M, GGUFTensorType::IQ2_XXS, GGUFTensorType::IQ2_XS, GGUFTensorType::IQ2_S,
        GGUFTensorType::IQ3_XXS, GGUFTensorType::IQ3_S, GGUFTensorType::IQ4_NL, GGUFTensorType::IQ4_XS})
    {
        SCOPED_TRACE(static_cast<int>(format)); OwnerFixture f(format);
        const auto table=f.prepare(); const auto view=table->view();
        EXPECT_EQ(table->nativeBytes(), f.source.tensorBytes());
        EXPECT_EQ(view.nativeBytes(), f.source.tensorBytes());
        EXPECT_EQ(view.format(), ggufToTensorType(format));
        EXPECT_EQ(std::memcmp(view.nativeData(), f.file.bytes.data()+37, f.source.tensorBytes()), 0);
        std::size_t sum=0;
        for (const auto bytes:f.backend.copies)
        { EXPECT_GT(bytes,0u); EXPECT_LE(bytes,f.slot); EXPECT_EQ(bytes%f.source.rowBytes(),0u); sum+=bytes; }
        EXPECT_EQ(sum,f.source.tensorBytes()); EXPECT_GT(f.backend.copies.size(),1u);
        for (const auto bytes:f.backend.pinned_sizes) EXPECT_EQ(bytes,f.slot);
        EXPECT_EQ(f.memory->claimedBytes(DeviceId::cuda(0),PhysicalMemoryOwner::ImmutableTableStorage,PhysicalMemoryMaterializationKind::NewAllocation), f.source.tensorBytes());
        EXPECT_EQ(f.memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::WeightLoadStaging,PhysicalMemoryMaterializationKind::NewAllocation),0u);
    }
}
TEST(ResidentImmutableRowTable, ExactReadTokenRetainsLastBorrowAndRejectsChangedIdentity)
{
    OwnerFixture f; auto table=f.prepare(); const auto bytes=table->nativeBytes();
    const auto copy_count=f.backend.copies.size();
    std::optional<ImmutableNativeDeviceRead> read(table->acquire(reinterpret_cast<void *>(0x2345),f.transfers));
    f.transfers.requireImmutableNativeDeviceInput(*read,DeviceId::cuda(0),reinterpret_cast<void *>(0x2345));
    EXPECT_EQ(f.backend.copies.size(),copy_count); EXPECT_EQ(f.backend.joins,1u);
    EXPECT_THROW(f.transfers.requireImmutableNativeDeviceInput(*read,DeviceId::cuda(0),reinterpret_cast<void *>(0x3456)),std::logic_error);
    EXPECT_THROW(f.transfers.requireImmutableNativeDeviceInput(*read,DeviceId::rocm(0),reinterpret_cast<void *>(0x2345)),std::logic_error);
    EXPECT_THROW(table->acquire(nullptr,f.transfers),std::invalid_argument);
    f.backend.runtime=2;
    EXPECT_THROW(f.transfers.requireImmutableNativeDeviceInput(*read,DeviceId::cuda(0),reinterpret_cast<void *>(0x2345)),std::logic_error);
    f.backend.runtime=1; table.reset();
    EXPECT_EQ(f.memory->claimedBytes(DeviceId::cuda(0),PhysicalMemoryOwner::ImmutableTableStorage,PhysicalMemoryMaterializationKind::NewAllocation),bytes);
    read.reset(); EXPECT_EQ(f.memory->claimedBytes(DeviceId::cuda(0),PhysicalMemoryOwner::ImmutableTableStorage,PhysicalMemoryMaterializationKind::NewAllocation),0u);
}
TEST(ResidentImmutableRowTable, MissingOrForeignAdmissionAndExpiredSlotCannotAllocate)
{
    OwnerFixture f;
    EXPECT_THROW(f.prepare(nullptr),std::invalid_argument);
    auto foreign=std::make_shared<PhysicalMemoryAuthority>(f.memory->admission(),0);
    EXPECT_THROW((void)ResidentImmutableRowTable::prepare(f.source,*f.placement,foreign,DeviceId::cuda(0),
        *f.receipt,reinterpret_cast<void *>(0x1234),f.transfers),std::logic_error);
    EXPECT_THROW((void)ResidentImmutableRowTable::prepare(f.source,*f.placement,f.memory,DeviceId::rocm(0),
        *f.receipt,reinterpret_cast<void *>(0x1234),f.transfers),std::invalid_argument);
    PhysicalMemoryPlanBuilder absent; absent.add(resource(DeviceId::cuda(0)),PhysicalMemoryOwner::PrimaryModelWeights,1);
    auto missing=std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(absent.build()),0);
    EXPECT_ANY_THROW((void)f.transfers.beginImmutableNativeDeviceUpload(missing,DeviceId::cuda(0),144,144,reinterpret_cast<void *>(0x1234)));
    EXPECT_TRUE(f.backend.copies.empty());
    f.loader.reset();
    EXPECT_THROW(f.receipt->requireFor(f.memory,DeviceId::cuda(0),f.source.rowBytes()),std::logic_error);
    EXPECT_THROW(f.prepare(),std::logic_error);
}
TEST(ResidentImmutableRowTable, PartialFailureRetiresBytesAndNeverPublishesSource)
{
    OwnerFixture f; f.backend.fail_copy=2;
    EXPECT_THROW(f.prepare(),std::runtime_error);
    EXPECT_EQ(f.memory->claimedBytes(DeviceId::cuda(0),PhysicalMemoryOwner::ImmutableTableStorage,PhysicalMemoryMaterializationKind::NewAllocation),0u);
    EXPECT_EQ(f.memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::WeightLoadStaging,PhysicalMemoryMaterializationKind::NewAllocation),0u);
    EXPECT_EQ(f.backend.copies.size(),2u);
}
TEST(ResidentImmutableRowTable, FinalSourceAuthenticationRejectsCompleteMixedFileUpload)
{
    OwnerFixture f;
    f.backend.on_copy=[&](std::size_t n,int) {
        if (n==1) ASSERT_EQ(::ftruncate(f.file.fd,static_cast<off_t>(f.file.bytes.size()+1)),0);
    };
    EXPECT_THROW(f.prepare(),std::runtime_error);
    std::size_t copied=0;for(auto bytes:f.backend.copies)copied+=bytes;
    EXPECT_EQ(copied,f.source.tensorBytes());
    EXPECT_EQ(f.memory->claimedBytes(DeviceId::cuda(0),PhysicalMemoryOwner::ImmutableTableStorage,PhysicalMemoryMaterializationKind::NewAllocation),0u);
}
TEST(ResidentImmutableRowTable, IncompleteAndMalformedNativeCoverageCannotSeal)
{
    OwnerFixture f; const auto stream=reinterpret_cast<void *>(0x1234);
    auto slot=f.transfers.prepareImmutableNativeUploadStaging(*f.receipt,f.memory,DeviceId::cuda(0),f.source.rowBytes());
    auto upload=f.transfers.beginImmutableNativeDeviceUpload(f.memory,DeviceId::cuda(0),288,144,stream);
    EXPECT_THROW((void)f.transfers.sealImmutableNativeDeviceUpload(std::move(upload)),std::logic_error);
    EXPECT_THROW(f.transfers.uploadImmutableNativeChunk(upload,*slot,143,stream),std::invalid_argument);
    EXPECT_THROW(f.transfers.uploadImmutableNativeChunk(upload,*slot,144,nullptr),std::invalid_argument);
    EXPECT_THROW(f.receipt->requireFor(f.memory,DeviceId::cuda(0),f.slot+1),std::logic_error);
    std::fill_n(slot->data(),288,std::byte{0x65});
    f.transfers.uploadImmutableNativeChunk(upload,*slot,144,stream);
    EXPECT_EQ(upload.copiedBytes(),144u);
    EXPECT_THROW((void)f.transfers.sealImmutableNativeDeviceUpload(std::move(upload)),std::logic_error);
    f.transfers.uploadImmutableNativeChunk(upload,*slot,144,stream);
    auto owner=f.transfers.sealImmutableNativeDeviceUpload(std::move(upload));
    EXPECT_EQ(owner->sizeBytes(),288u); EXPECT_EQ(upload.copiedBytes(),0u);
    EXPECT_THROW((void)f.transfers.sealImmutableNativeDeviceUpload(std::move(upload)),std::logic_error);
}

TEST(ResidentImmutableRowTable, ModelContextPublishesCompleteOwnerOnceAndAuthenticatesReuse)
{
    OwnerFixture f;
    auto context=ModelContext::createForTesting(f.file.path,nullptr,1,false);
    auto &directory=const_cast<GGUFModel &>(context->model());
    directory.split_paths={f.file.path}; directory.data_offset=37;
    directory.tensors.push_back({.name="per_layer_token_embd.weight",.dimensions={f.source.rows,256},
        .type=f.source.format,.offset=0,.size_bytes=f.source.tensorBytes(),.split_idx=0});
    const ImmutableRowTableDevicePreparation setup{DeviceId::cuda(0),reinterpret_cast<void *>(0x1234),*f.receipt};
    EXPECT_THROW(context->prepareImmutableRowTable(*f.placement,f.memory,{},f.transfers),std::invalid_argument);
    EXPECT_FALSE(context->residentImmutableRowTable(DeviceId::cuda(0)));
    EXPECT_EQ(f.memory->claimedBytes(DeviceId::cuda(0),PhysicalMemoryOwner::ImmutableTableStorage,
        PhysicalMemoryMaterializationKind::NewAllocation),0u);
    const std::array uploads{setup};
    context->prepareImmutableRowTable(*f.placement,f.memory,uploads,f.transfers);
    const auto first=context->residentImmutableRowTable(DeviceId::cuda(0)); ASSERT_TRUE(first);
    EXPECT_FALSE(context->hostImmutableRowProvider())<<"VRAM cannot retain a complete host table";
    EXPECT_EQ(*context->resolvedImmutableRowTablePlacement(),*f.placement);
    const auto copies=f.backend.copies.size();
    context->prepareImmutableRowTable(*f.placement,f.memory,{},f.transfers);
    EXPECT_EQ(context->residentImmutableRowTable(DeviceId::cuda(0)),first);
    EXPECT_EQ(context->residentImmutableRowTableBytes(DeviceId::cuda(0)),f.source.tensorBytes());
    EXPECT_EQ(context->residentImmutableRowTableBytes(DeviceId::cuda(1)),0u);
    EXPECT_EQ(f.backend.copies.size(),copies);
    auto foreign=std::make_shared<PhysicalMemoryAuthority>(f.memory->admission(),0);
    EXPECT_THROW(context->prepareImmutableRowTable(*f.placement,foreign,{},f.transfers),std::logic_error);
    ASSERT_EQ(::ftruncate(f.file.fd,static_cast<off_t>(f.file.bytes.size()+1)),0);
    EXPECT_THROW(context->prepareImmutableRowTable(*f.placement,f.memory,{},f.transfers),std::runtime_error);
    EXPECT_EQ(context->residentImmutableRowTable(DeviceId::cuda(0)),first);
    EXPECT_EQ(f.backend.copies.size(),copies);
}

TEST(ResidentImmutableRowTable, CaptureOnlyValidatesPreparedReadAndCannotPerformSetup)
{
    OwnerFixture f; const auto table=f.prepare(); const auto stream=reinterpret_cast<void *>(0x2345);
    const auto read=table->acquire(stream,f.transfers); const auto copies=f.backend.copies.size();
    const auto joins=f.backend.joins;
    {
        GraphCaptureGuard guard;
        EXPECT_NO_THROW(f.transfers.requireImmutableNativeDeviceInput(read,DeviceId::cuda(0),stream));
        EXPECT_THROW((void)table->acquire(stream,f.transfers),std::logic_error);
        EXPECT_THROW((void)f.transfers.beginImmutableNativeDeviceUpload(f.memory,DeviceId::cuda(0),
            288,144,stream),std::logic_error);
        EXPECT_THROW((void)f.transfers.prepareImmutableNativeUploadStaging(*f.receipt,f.memory,
            DeviceId::cuda(0),144),std::logic_error);
        EXPECT_THROW(table->authenticateSource(),std::logic_error);
    }
    EXPECT_EQ(f.backend.copies.size(),copies); EXPECT_EQ(f.backend.joins,joins);
}

TEST(ResidentImmutableRowTable, SameConsumerSetupJoinsOriginalWriterWithoutDuplicateUpload)
{
    OwnerFixture f;const auto context=contextFor(f);
    const ImmutableRowTableDevicePreparation setup{DeviceId::cuda(0),reinterpret_cast<void *>(0x1234),*f.receipt};
    std::mutex mutex;std::condition_variable ready;bool entered=false,release=false;
    f.backend.on_copy=[&](std::size_t ordinal,int) {
        if(ordinal!=1)return;
        // A getter can acquire the model mutex while this upload is paused.
        EXPECT_TRUE(context->resolvedImmutableRowTablePlacement());
        std::unique_lock lock(mutex);entered=true;ready.notify_all();
        ready.wait(lock,[&]{return release;});
    };
    std::exception_ptr first_error,second_error;
    std::thread first([&]{try{context->prepareImmutableRowTableForConsumer(*f.placement,f.memory,
        DeviceId::cuda(0),setup,f.transfers);}catch(...){first_error=std::current_exception();}});
    {std::unique_lock lock(mutex);EXPECT_TRUE(ready.wait_for(lock,std::chrono::seconds(1),[&]{return entered;}));}
    std::thread second([&]{try{context->prepareImmutableRowTableForConsumer(*f.placement,f.memory,
        DeviceId::cuda(0),std::nullopt,f.transfers);}catch(...){second_error=std::current_exception();}});
    {std::lock_guard lock(mutex);release=true;}ready.notify_all();first.join();second.join();
    EXPECT_FALSE(first_error);EXPECT_FALSE(second_error);
    EXPECT_NO_THROW(context->requireImmutableRowTablePlacementComplete(*f.placement,f.memory));
    std::size_t copied=0;for(auto bytes:f.backend.copies)copied+=bytes;
    EXPECT_EQ(copied,f.source.tensorBytes());
    const auto first_owner=context->residentImmutableRowTable(DeviceId::cuda(0));ASSERT_TRUE(first_owner);
    context->prepareImmutableRowTableForConsumer(*f.placement,f.memory,DeviceId::cuda(0),std::nullopt,f.transfers);
    EXPECT_EQ(context->residentImmutableRowTable(DeviceId::cuda(0)),first_owner);
}
TEST(ResidentImmutableRowTable, SiblingFailureWakesWaitersAndPreventsLateReadyPublication)
{
    OwnerFixture f(GGUFTensorType::IQ4_NL,97,2);const auto context=contextFor(f);
    const ImmutableRowTableDevicePreparation first_setup{DeviceId::cuda(0),reinterpret_cast<void *>(0x1234),*f.receipt};
    const ImmutableRowTableDevicePreparation second_setup{DeviceId::cuda(1),reinterpret_cast<void *>(0x5678),*f.second_receipt};
    std::mutex mutex;std::condition_variable ready;bool entered=false,release=false,peer_finished=false;
    f.backend.on_copy=[&](std::size_t,int device) {
        if(device==1)throw std::runtime_error("injected independent participant failure");
        std::unique_lock lock(mutex);
        if(entered)return;
        entered=true;ready.notify_all();ready.wait(lock,[&]{return release;});
    };
    std::exception_ptr first_error,second_error,waiter_error;
    std::thread first([&]{try{context->prepareImmutableRowTableForConsumer(*f.placement,f.memory,
        DeviceId::cuda(0),first_setup,f.transfers);}catch(...){first_error=std::current_exception();}});
    {std::unique_lock lock(mutex);EXPECT_TRUE(ready.wait_for(lock,std::chrono::seconds(1),[&]{return entered;}));}
    std::thread waiter([&]{try{context->prepareImmutableRowTableForConsumer(*f.placement,f.memory,
        DeviceId::cuda(0),std::nullopt,f.transfers);}catch(...){waiter_error=std::current_exception();}});
    std::thread second([&]{try{context->prepareImmutableRowTableForConsumer(*f.placement,f.memory,
        DeviceId::cuda(1),second_setup,f.transfers);}catch(...){second_error=std::current_exception();}
        {std::lock_guard lock(mutex);peer_finished=true;}ready.notify_all();});
    {std::unique_lock lock(mutex);EXPECT_TRUE(ready.wait_for(lock,std::chrono::seconds(1),[&]{return peer_finished;}));
        release=true;}ready.notify_all();
    first.join();second.join();waiter.join();
    EXPECT_TRUE(first_error);EXPECT_TRUE(second_error);EXPECT_TRUE(waiter_error);
    EXPECT_THROW(context->requireImmutableRowTablePlacementComplete(*f.placement,f.memory),std::runtime_error);
    EXPECT_THROW((void)context->residentImmutableRowTable(DeviceId::cuda(0)),std::runtime_error);
    for(int device:{0,1})
        EXPECT_EQ(f.memory->claimedBytes(DeviceId::cuda(device),PhysicalMemoryOwner::ImmutableTableStorage,
            PhysicalMemoryMaterializationKind::NewAllocation),0u);
    EXPECT_EQ(f.memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::WeightLoadStaging,
        PhysicalMemoryMaterializationKind::NewAllocation),0u);
}

TEST(ResidentImmutableRowTable, ParticipantCopiesMustAgreeOnOneSourceGenerationBeforeReady)
{
    OwnerFixture f(GGUFTensorType::IQ4_NL,97,2);auto context=contextFor(f);
    const ImmutableRowTableDevicePreparation first{DeviceId::cuda(0),reinterpret_cast<void *>(0x1234),*f.receipt};
    const ImmutableRowTableDevicePreparation second{DeviceId::cuda(1),reinterpret_cast<void *>(0x5678),*f.second_receipt};
    EXPECT_THROW(context->prepareImmutableRowTableForConsumer(*f.placement,f.memory,DeviceId::cuda(0),
        std::nullopt,f.transfers),std::invalid_argument);
    context->prepareImmutableRowTableForConsumer(*f.placement,f.memory,DeviceId::cuda(0),first,f.transfers);
    auto retained=context->residentImmutableRowTable(DeviceId::cuda(0));ASSERT_TRUE(retained);
    EXPECT_THROW(context->requireImmutableRowTablePlacementComplete(*f.placement,f.memory),std::logic_error);
    // The second copy can read every original byte successfully, but a newer
    // file generation cannot join an already-published model source generation.
    ASSERT_EQ(::ftruncate(f.file.fd,static_cast<off_t>(f.file.bytes.size()+1)),0);
    EXPECT_THROW(context->prepareImmutableRowTableForConsumer(*f.placement,f.memory,DeviceId::cuda(1),
        second,f.transfers),std::runtime_error);
    EXPECT_THROW(context->requireImmutableRowTablePlacementComplete(*f.placement,f.memory),std::runtime_error);
    EXPECT_EQ(f.memory->claimedBytes(DeviceId::cuda(1),PhysicalMemoryOwner::ImmutableTableStorage,
        PhysicalMemoryMaterializationKind::NewAllocation),0u);
    context.reset();
    EXPECT_EQ(f.memory->claimedBytes(DeviceId::cuda(0),PhysicalMemoryOwner::ImmutableTableStorage,
        PhysicalMemoryMaterializationKind::NewAllocation),f.source.tensorBytes());
    retained.reset();
    EXPECT_EQ(f.memory->claimedBytes(DeviceId::cuda(0),PhysicalMemoryOwner::ImmutableTableStorage,
        PhysicalMemoryMaterializationKind::NewAllocation),0u);
}

TEST(ResidentImmutableRowTable, FailedSetupDrainKeepsClaimsAndCannotPublishReleasedReceipt)
{
    OwnerFixture f(GGUFTensorType::IQ4_NL,97,1,false);
    const auto kind=PhysicalMemoryMaterializationKind::NewAllocation;
    ASSERT_EQ(f.memory->claimedBytes(DeviceId::cuda(0),PhysicalMemoryOwner::WeightLoadStaging,kind),f.slot);
    ASSERT_EQ(f.memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::WeightLoadStaging,kind),f.slot);
    f.backend.fail_drain=true;
    EXPECT_THROW(f.loader->finalize(),std::runtime_error);
    EXPECT_THROW((void)f.loader->releasedWeightLoadStaging(0),std::logic_error);
    EXPECT_EQ(f.memory->claimedBytes(DeviceId::cuda(0),PhysicalMemoryOwner::WeightLoadStaging,kind),f.slot);
    EXPECT_EQ(f.memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::WeightLoadStaging,kind),f.slot);
    // Restore the injected backend only for explicit test cleanup, not a production retry.
    f.backend.fail_drain=false;
}

namespace {
/** @brief Actual per-range PMA/upload fixture, retaining every original drained loader slot. */
class RangeOwnerFixture final {
public:
    /** @brief Admit unequal native shards plus real ordinary upload owners; no range-sized staging is invented. */
    explicit RangeOwnerFixture(std::size_t degree=2,GGUFTensorType format=GGUFTensorType::IQ4_NL)
        :source{.path="fixture",.tensor_name="per_layer_token_embd.weight",.file_offset=37,
            .rows=178,.row_elements=256,.format=format},file(source.tensorBytes()),transfers(resolve)
    {
        active=&backend;backend.setDeviceCount(static_cast<int>(degree));source.path=file.path;
        std::array<std::uint64_t,16> offsets{},sizes{};std::uint64_t row=0;
        for(std::size_t h=0;h<16;++h){if(h==8)row+=2;offsets[h]=row;sizes[h]=h+3;row+=sizes[h];}
        std::vector<PhysicalMemoryAllocatorIdentity> ids;std::vector<PhysicalMemoryResource> resources;
        PhysicalMemoryPlanBuilder plan;
        for(std::size_t n=0;n<degree;++n) {
            const auto d=DeviceId::cuda(static_cast<int>(n));ids.push_back({0,d});resources.push_back(resource(d));
            plan.add(resource(d),PhysicalMemoryOwner::PrimaryModelWeights,slot);
            plan.add(resource(d),PhysicalMemoryOwner::WeightLoadStaging,slot);
        }
        plan.add(resource(DeviceId::cpu()),PhysicalMemoryOwner::WeightLoadStaging,slot*degree);
        placement=Qwen4ExpFixedHeadShards::declare(
            ImmutableRowTableExtent::fromNative(source.format,source.rows,source.rowBytes()),offsets,sizes,ids);
        const auto native=placement->fixedHeadShardStorageBOM(resources);for(const auto &line:native.resources())plan.add(line);
        memory=std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()),0);
        for(std::size_t n=0;n<degree;++n) {
            auto loader=std::make_unique<LoadOrchestrator>(&backend,memory,PhysicalMemoryOwner::PrimaryModelWeights);
            loader->addDevice(static_cast<int>(n));loader->planRawWeight(static_cast<int>(n),"ordinary.weight",1,slot/sizeof(float),slot);
            loader->allocate(slot,1);loader->finalize();receipts.push_back(loader->releasedWeightLoadStaging(static_cast<int>(n)));
            loaders.push_back(std::move(loader));
        }
        backend.copies.clear();backend.pinned_sizes.clear();
    }
    /** @brief Callers retire range owners/read tokens before the fixture's actual backend and loader receipts. */
    ~RangeOwnerFixture(){loaders.clear();active=nullptr;}
    /** @return Real range loading through the canonical exact-stream TransferEngine path. */
    std::shared_ptr<const ResidentImmutableRowTable> prepare(int ordinal=0,void *stream=reinterpret_cast<void *>(0x1234))
    {return ResidentImmutableRowTable::prepareRange(source,*placement,memory,DeviceId::cuda(ordinal),receipts.at(ordinal),stream,transfers);}
    static constexpr std::size_t slot=4096;
    ImmutableRowTableSource source;NativeFile file;NativeBackend backend;TransferEngine transfers;
    std::shared_ptr<const ResolvedImmutableRowTablePlacement> placement;std::shared_ptr<PhysicalMemoryAuthority> memory;
    std::vector<std::unique_ptr<LoadOrchestrator>> loaders;std::vector<ReleasedWeightLoadStaging> receipts;
};
}
/** @test Complete fixed cohorts upload the original source once with exact unequal physical owner extents. */
TEST(ResidentImmutableRowTable, FixedRangesUploadOriginalBytesOnceAndCannotExposeWholeView)
{
    for(const std::size_t degree:{2u,4u,8u}) {
        RangeOwnerFixture f(degree);std::vector<std::shared_ptr<const ResidentImmutableRowTable>> tables;std::size_t total=0;
        for(std::size_t n=0;n<degree;++n) {
            auto table=f.prepare(static_cast<int>(n));const auto range=table->sourceRange();const auto view=table->rangeView();
            EXPECT_EQ(table->source(),f.source);EXPECT_EQ(range.sourceExtent().rows(),178u);
            EXPECT_EQ(view.sourceRowCount(),178u);EXPECT_EQ(view.globalRowBegin(),range.firstRow());EXPECT_EQ(view.rowCount(),range.rows());
            EXPECT_EQ(table->nativeBytes(),range.nativeBytes());EXPECT_EQ(view.nativeBytes(),range.nativeBytes());
            EXPECT_EQ(std::memcmp(view.nativeData(),f.file.bytes.data()+37+range.byteOffset(),range.nativeBytes()),0);
            EXPECT_THROW((void)table->view(),std::logic_error);
            EXPECT_EQ(f.memory->claimedBytes(table->device(),PhysicalMemoryOwner::ImmutableTableStorage,
                PhysicalMemoryMaterializationKind::NewAllocation),range.nativeBytes());
            if(!tables.empty()){EXPECT_EQ(table->identity(),tables.front()->identity());}
            total+=table->nativeBytes();tables.push_back(std::move(table));
        }
        EXPECT_EQ(total,f.source.tensorBytes());std::size_t copied=0;
        for(const auto count:f.backend.copies){EXPECT_GT(count,0u);EXPECT_LE(count,f.slot);EXPECT_EQ(count%f.source.rowBytes(),0u);copied+=count;}
        EXPECT_EQ(copied,f.source.tensorBytes());
        for(const auto pinned:f.backend.pinned_sizes)EXPECT_EQ(pinned,f.slot);
        EXPECT_EQ(f.memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::WeightLoadStaging,
            PhysicalMemoryMaterializationKind::NewAllocation),0u);
        tables.clear();
        for(std::size_t n=0;n<degree;++n)EXPECT_EQ(f.memory->claimedBytes(DeviceId::cuda(static_cast<int>(n)),
            PhysicalMemoryOwner::ImmutableTableStorage,PhysicalMemoryMaterializationKind::NewAllocation),0u);
    }
}
/** @test Every native codec stays byte-identical; range loading does not pad/requantize original rows. */
TEST(ResidentImmutableRowTable, FixedRangesPreserveEverySupportedOriginalNativeFormat)
{
    for(const auto format:{GGUFTensorType::F32,GGUFTensorType::F16,GGUFTensorType::BF16,
        GGUFTensorType::Q4_0,GGUFTensorType::Q4_1,GGUFTensorType::Q5_0,GGUFTensorType::Q5_1,
        GGUFTensorType::Q8_0,GGUFTensorType::Q2_K,GGUFTensorType::Q3_K,GGUFTensorType::Q4_K,
        GGUFTensorType::Q5_K,GGUFTensorType::Q6_K,GGUFTensorType::Q8_K,GGUFTensorType::IQ1_S,
        GGUFTensorType::IQ1_M,GGUFTensorType::IQ2_XXS,GGUFTensorType::IQ2_XS,GGUFTensorType::IQ2_S,
        GGUFTensorType::IQ3_XXS,GGUFTensorType::IQ3_S,GGUFTensorType::IQ4_NL,GGUFTensorType::IQ4_XS}) {
        SCOPED_TRACE(static_cast<int>(format));RangeOwnerFixture f(2,format);
        for(int n=0;n<2;++n) {
            const auto table=f.prepare(n);const auto r=table->sourceRange();const auto v=table->rangeView();
            EXPECT_EQ(v.format(),ggufToTensorType(format));EXPECT_EQ(v.rowBytes(),f.source.rowBytes());
            EXPECT_EQ(std::memcmp(v.nativeData(),f.file.bytes.data()+37+r.byteOffset(),r.nativeBytes()),0);
        }
    }
}
/** @test Exact-stream read ownership retains only the local range after facade retirement. */
TEST(ResidentImmutableRowTable, FixedRangeReadTokenRetainsExactLocalOwnerAndGeneration)
{
    RangeOwnerFixture f;auto table=f.prepare(1);const auto range=table->sourceRange();const auto view=table->rangeView();
    std::optional<ImmutableNativeDeviceRead> read(table->acquire(reinterpret_cast<void *>(0x2345),f.transfers));
    f.transfers.requireImmutableNativeDeviceInput(*read,DeviceId::cuda(1),reinterpret_cast<void *>(0x2345));
    EXPECT_EQ(read->owner()->sizeBytes(),range.nativeBytes());EXPECT_EQ(read->owner()->generation(),view.storageGeneration());
    EXPECT_THROW(f.transfers.requireImmutableNativeDeviceInput(*read,DeviceId::cuda(0),reinterpret_cast<void *>(0x2345)),std::logic_error);
    f.backend.runtime=2;EXPECT_THROW(f.transfers.requireImmutableNativeDeviceInput(*read,DeviceId::cuda(1),reinterpret_cast<void *>(0x2345)),std::logic_error);
    f.backend.runtime=1;table.reset();
    EXPECT_EQ(f.memory->claimedBytes(DeviceId::cuda(1),PhysicalMemoryOwner::ImmutableTableStorage,
        PhysicalMemoryMaterializationKind::NewAllocation),range.nativeBytes());
    read.reset();EXPECT_EQ(f.memory->claimedBytes(DeviceId::cuda(1),PhysicalMemoryOwner::ImmutableTableStorage,
        PhysicalMemoryMaterializationKind::NewAllocation),0u);
}
/** @test A shard never authorizes an old full upload, foreign source or truncated unowned tail. */
TEST(ResidentImmutableRowTable, FixedRangeRejectsWholeConsumerForeignSourceAndIncompleteFullFile)
{
    RangeOwnerFixture f;
    EXPECT_THROW(f.prepare(0,nullptr),std::invalid_argument);
    EXPECT_THROW((void)ResidentImmutableRowTable::prepare(f.source,*f.placement,f.memory,DeviceId::cuda(0),f.receipts[0],
        reinterpret_cast<void *>(0x1234),f.transfers),std::invalid_argument);
    auto changed=f.source;--changed.rows;
    EXPECT_THROW((void)ResidentImmutableRowTable::prepareRange(changed,*f.placement,f.memory,DeviceId::cuda(0),f.receipts[0],
        reinterpret_cast<void *>(0x1234),f.transfers),std::invalid_argument);
    auto foreign=std::make_shared<PhysicalMemoryAuthority>(f.memory->admission(),0);
    EXPECT_THROW((void)ResidentImmutableRowTable::prepareRange(f.source,*f.placement,foreign,DeviceId::cuda(0),f.receipts[0],
        reinterpret_cast<void *>(0x1234),f.transfers),std::logic_error);
    ASSERT_TRUE(f.backend.copies.empty());
    // Owner0's complete bytes still fit, but the ORIGINAL full source does not.
    ASSERT_EQ(::ftruncate(f.file.fd,static_cast<off_t>(f.file.bytes.size()-1)),0);
    EXPECT_THROW(f.prepare(0),std::invalid_argument);EXPECT_TRUE(f.backend.copies.empty());
    EXPECT_EQ(f.memory->claimedBytes(DeviceId::cuda(0),PhysicalMemoryOwner::ImmutableTableStorage,
        PhysicalMemoryMaterializationKind::NewAllocation),0u);
}
/** @test Partial copy or final full-source metadata drift never publishes a range and retires new claims. */
TEST(ResidentImmutableRowTable, FixedRangeCopyFailureAndFullSourceMutationNeverPublish)
{
    for(const bool mutation:{false,true}) {
        RangeOwnerFixture f;
        if(mutation) {
            f.backend.on_copy=[&](std::size_t copy,int) {
                if(copy==1){ASSERT_EQ(::ftruncate(f.file.fd,static_cast<off_t>(f.file.bytes.size()+1)),0);}
            };
        } else {f.backend.fail_copy=2;}
        EXPECT_THROW(f.prepare(),std::runtime_error);
        EXPECT_EQ(f.memory->claimedBytes(DeviceId::cuda(0),PhysicalMemoryOwner::ImmutableTableStorage,
            PhysicalMemoryMaterializationKind::NewAllocation),0u);
        EXPECT_EQ(f.memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::WeightLoadStaging,
            PhysicalMemoryMaterializationKind::NewAllocation),0u);
    }
}

/** @test ModelContext publishes each exact partial owner once and retains only its real PMA bytes. */
TEST(ResidentImmutableRowTable, FixedHeadContextMaterializesEveryRangeOnceAndRetainsExactOwner)
{
    for(const std::size_t degree:{2u,4u,8u})
    {
        RangeOwnerFixture f(degree);auto context=contextFor(f);
        EXPECT_THROW(context->requireImmutableRowTablePlacementComplete(*f.placement,f.memory),std::logic_error);
        std::vector<std::shared_ptr<const ResidentImmutableRowTable>> owners;
        for(std::size_t n=0;n<degree;++n)
        {
            const auto device=DeviceId::cuda(static_cast<int>(n));
            const ImmutableRowTableDevicePreparation upload{device,reinterpret_cast<void *>(0x1234),f.receipts[n]};
            context->prepareImmutableRowTableForConsumer(*f.placement,f.memory,device,upload,f.transfers);
            const auto table=context->residentImmutableRowTable(device);ASSERT_TRUE(table);
            const auto expected=f.placement->fixedHeadSourceRangeFor({0,device});
            EXPECT_EQ(table->source(),f.source);EXPECT_EQ(table->sourceRange(),expected);
            EXPECT_EQ(context->residentImmutableRowTableBytes(device),expected.nativeBytes());
            EXPECT_NE(context->residentImmutableRowTableBytes(device),f.source.tensorBytes());
            EXPECT_EQ(f.memory->claimedBytes(device,PhysicalMemoryOwner::ImmutableTableStorage,
                PhysicalMemoryMaterializationKind::NewAllocation),expected.nativeBytes());
            EXPECT_THROW((void)table->view(),std::logic_error);
            EXPECT_EQ(std::memcmp(table->rangeView().nativeData(),f.file.bytes.data()+37+expected.byteOffset(),expected.nativeBytes()),0);
            if(!owners.empty()){EXPECT_EQ(table->identity(),owners.front()->identity());}
            owners.push_back(table);
            if(n+1<degree)
            {
                EXPECT_THROW(context->requireImmutableRowTablePlacementComplete(*f.placement,f.memory),std::logic_error);
            }
        }
        EXPECT_NO_THROW(context->requireImmutableRowTablePlacementComplete(*f.placement,f.memory));
        EXPECT_FALSE(context->hostImmutableRowProvider());
        const auto copies=f.backend.copies.size();
        // Ready storage no longer needs the old loader or its released receipt;
        // only a fresh upload must prove that setup authority is still alive.
        f.loaders.clear();
        context->prepareImmutableRowTable(*f.placement,f.memory,{},f.transfers);
        EXPECT_EQ(f.backend.copies.size(),copies);
        for(std::size_t n=0;n<degree;++n)
            EXPECT_EQ(context->residentImmutableRowTable(DeviceId::cuda(static_cast<int>(n))),owners[n]);
        EXPECT_EQ(f.memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::WeightLoadStaging,
            PhysicalMemoryMaterializationKind::NewAllocation),0u);
        auto table=owners.back();
        std::optional<ImmutableNativeDeviceRead> read(table->acquire(reinterpret_cast<void *>(0x2345),f.transfers));
        const auto retained=table->sourceRange();const auto last=table->device();
        context.reset();owners.clear();table.reset();
        for(std::size_t n=0;n<degree-1;++n)
            EXPECT_EQ(f.memory->claimedBytes(DeviceId::cuda(static_cast<int>(n)),PhysicalMemoryOwner::ImmutableTableStorage,
                PhysicalMemoryMaterializationKind::NewAllocation),0u);
        EXPECT_EQ(f.memory->claimedBytes(last,PhysicalMemoryOwner::ImmutableTableStorage,
            PhysicalMemoryMaterializationKind::NewAllocation),retained.nativeBytes());
        read.reset();
        EXPECT_EQ(f.memory->claimedBytes(last,PhysicalMemoryOwner::ImmutableTableStorage,
            PhysicalMemoryMaterializationKind::NewAllocation),0u);
    }
}
/** @test A fresh fixed owner cannot borrow a peer's or expired upload receipt, change source/range, or replace PMA. */
TEST(ResidentImmutableRowTable, FixedHeadContextRejectsForeignMissingAndChangedOwnerAuthority)
{
    RangeOwnerFixture f;const auto context=contextFor(f);const auto device=DeviceId::cuda(0);
    EXPECT_THROW(context->prepareImmutableRowTableForConsumer(*f.placement,f.memory,device,std::nullopt,f.transfers),std::invalid_argument);
    const ImmutableRowTableDevicePreparation peer{device,reinterpret_cast<void *>(0x1234),f.receipts[1]};
    EXPECT_THROW(context->prepareImmutableRowTableForConsumer(*f.placement,f.memory,device,peer,f.transfers),std::logic_error);
    const ImmutableRowTableDevicePreparation setup{device,reinterpret_cast<void *>(0x1234),f.receipts[0]};
    EXPECT_THROW(context->prepareImmutableRowTableForConsumer(*f.placement,f.memory,DeviceId::cuda(7),setup,f.transfers),std::invalid_argument);
    auto foreign=std::make_shared<PhysicalMemoryAuthority>(f.memory->admission(),0);
    EXPECT_THROW(context->prepareImmutableRowTableForConsumer(*f.placement,foreign,device,setup,f.transfers),std::logic_error);
    EXPECT_FALSE(context->resolvedImmutableRowTablePlacement());EXPECT_TRUE(f.backend.copies.empty());
    context->prepareImmutableRowTableForConsumer(*f.placement,f.memory,device,setup,f.transfers);
    const auto first=context->residentImmutableRowTable(device);ASSERT_TRUE(first);const auto copies=f.backend.copies.size();
    std::array<std::uint64_t,16> offsets{},sizes{};std::uint64_t row=0;
    for(std::size_t h=0;h<16;++h){if(h==8)row+=2;offsets[h]=row;sizes[h]=h+3;row+=sizes[h];}
    // Reversing the original collective declaration swaps ranges, despite the
    // same full source and the same sorted physical owner inventory.
    const std::array reversed{PhysicalMemoryAllocatorIdentity{0,DeviceId::cuda(1)},PhysicalMemoryAllocatorIdentity{0,device}};
    const auto changed=Qwen4ExpFixedHeadShards::declare(f.placement->extent(),offsets,sizes,reversed);
    EXPECT_NE(changed->fixedHeadSourceRangeFor({0,device}),first->sourceRange());
    EXPECT_THROW(context->prepareImmutableRowTableForConsumer(*changed,f.memory,device,std::nullopt,f.transfers),std::logic_error);
    EXPECT_THROW(context->prepareImmutableRowTableForConsumer(*f.placement,foreign,device,std::nullopt,f.transfers),std::logic_error);
    auto &metadata=const_cast<GGUFModel &>(context->model()).tensors.back();
    --metadata.dimensions[0];metadata.size_bytes=(f.source.rows-1)*f.source.rowBytes();
    EXPECT_THROW(context->prepareImmutableRowTableForConsumer(*f.placement,f.memory,device,std::nullopt,f.transfers),std::invalid_argument);
    metadata.dimensions[0]=f.source.rows;metadata.size_bytes=f.source.tensorBytes();
    EXPECT_EQ(context->residentImmutableRowTable(device),first);EXPECT_EQ(f.backend.copies.size(),copies);
    // Revoking the actual ordinary loader's lifetime is different from merely
    // omitting its receipt for a Ready owner. A fresh owner must reject it.
    f.loaders[1].reset();
    const ImmutableRowTableDevicePreparation expired{DeviceId::cuda(1),reinterpret_cast<void *>(0x5678),f.receipts[1]};
    EXPECT_THROW(context->prepareImmutableRowTableForConsumer(*f.placement,f.memory,DeviceId::cuda(1),expired,f.transfers),std::logic_error);
    EXPECT_FALSE(context->residentImmutableRowTable(DeviceId::cuda(1)));
    EXPECT_EQ(f.memory->claimedBytes(DeviceId::cuda(1),PhysicalMemoryOwner::ImmutableTableStorage,
        PhysicalMemoryMaterializationKind::NewAllocation),0u);
}
/** @test A failed peer range or mismatched complete-file generation prevents any inference-ready publication. */
TEST(ResidentImmutableRowTable, FixedHeadContextSiblingFailureNeverPublishesCompletePlacement)
{
    for(const bool source_mutation:{false,true})
    {
        RangeOwnerFixture f;auto context=contextFor(f);
        const ImmutableRowTableDevicePreparation a{DeviceId::cuda(0),reinterpret_cast<void *>(0x1234),f.receipts[0]};
        const ImmutableRowTableDevicePreparation b{DeviceId::cuda(1),reinterpret_cast<void *>(0x5678),f.receipts[1]};
        context->prepareImmutableRowTableForConsumer(*f.placement,f.memory,DeviceId::cuda(0),a,f.transfers);
        auto retained=context->residentImmutableRowTable(DeviceId::cuda(0));ASSERT_TRUE(retained);
        const auto bytes=retained->nativeBytes();
        if(source_mutation){ASSERT_EQ(::ftruncate(f.file.fd,static_cast<off_t>(f.file.bytes.size()+1)),0);}
        else f.backend.on_copy=[](std::size_t,int device){if(device==1)throw std::runtime_error("injected fixed peer failure");};
        EXPECT_THROW(context->prepareImmutableRowTableForConsumer(*f.placement,f.memory,DeviceId::cuda(1),b,f.transfers),std::runtime_error);
        EXPECT_THROW(context->requireImmutableRowTablePlacementComplete(*f.placement,f.memory),std::runtime_error);
        EXPECT_THROW((void)context->residentImmutableRowTable(DeviceId::cuda(0)),std::runtime_error);
        EXPECT_EQ(f.memory->claimedBytes(DeviceId::cuda(1),PhysicalMemoryOwner::ImmutableTableStorage,
            PhysicalMemoryMaterializationKind::NewAllocation),0u);
        EXPECT_EQ(f.memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::WeightLoadStaging,
            PhysicalMemoryMaterializationKind::NewAllocation),0u);
        context.reset();
        EXPECT_EQ(f.memory->claimedBytes(DeviceId::cuda(0),PhysicalMemoryOwner::ImmutableTableStorage,
            PhysicalMemoryMaterializationKind::NewAllocation),bytes);
        retained.reset();
        EXPECT_EQ(f.memory->claimedBytes(DeviceId::cuda(0),PhysicalMemoryOwner::ImmutableTableStorage,
            PhysicalMemoryMaterializationKind::NewAllocation),0u);
    }
}
