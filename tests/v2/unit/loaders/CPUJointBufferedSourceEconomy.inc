/**
 * @file CPUJointBufferedSourceEconomy.inc
 * @brief Verify realistic strided source loading economy through kernel read-call observations.
 *
 * A sparse original E512 GGUF retains QFN H2560/F640 geometry. Only one native
 * down expert is populated/read. File holes are source fixtures, never model
 * numerical evidence. The actual public loader and PMA own the bounded window;
 * /proc/self/io observes kernel reads, not C++ row-loop calls or guessed timing.
 */
#include "../../utils/CPUJointSourceTestPlan.h"
#include "../../utils/NativeWeightSourceTestRead.h"
#include "execution/moe/MoEExpertOverlayPreparationPlan.h"
#include "planning/PhysicalMemoryAuthority.h"
#include <chrono>
#include <unistd.h>
namespace llaminar2::test {
namespace {
/** @brief Unique sparse native source, with no whole-parent allocation or copy. */
class BufferedEconomySource final {
public:
    /** @brief Author actual original directory and one deterministic native expert payload. */
    BufferedEconomySource() {
        char name[]="/tmp/llaminar-buffered-source-XXXXXX.gguf";
        const int fd=::mkstemps(name,5);
        if(fd<0)throw std::runtime_error("Cannot create buffered source fixture");
        ::close(fd);path_=name;
        try {
        GGUFTensorInfo gate{.type=GGUFTensorType::F32};
        GGUFTensorInfo down{.type=GGUFTensorType::Q5_1};
        source_row_bytes_=size_t(640)/down.getBlockSize()*down.getTypeSize();
        source_expert_bytes_=source_row_bytes_*2560;
        const uint64_t gate_bytes=uint64_t(2560)*640*512*gate.getTypeSize();
        const uint64_t down_bytes=uint64_t(source_expert_bytes_)*512;
        std::ofstream stream(path_,std::ios::binary|std::ios::trunc);
        stream.write("GGUF",4);writeScalar(stream,uint32_t{3});writeScalar(stream,uint64_t{3});writeScalar(stream,uint64_t{0});
        for(int role=0;role<3;++role) {
            writeString(stream,"blk.7.ffn_"+std::string(role==0?"gate":role==1?"up":"down")+"_exps.weight");
            writeScalar(stream,uint32_t{3});writeScalar(stream,uint64_t(role==2?640:2560));
            writeScalar(stream,uint64_t(role==2?2560:640));writeScalar(stream,uint64_t{512});
            writeScalar(stream,uint32_t(role==2?GGUFTensorType::Q5_1:GGUFTensorType::F32));
            writeScalar(stream,uint64_t(role)*gate_bytes);
        }
        const size_t end=size_t(stream.tellp());data_offset_=(end+31)/32*32;
        std::vector<char> padding(data_offset_-end,0);stream.write(padding.data(),padding.size());stream.close();
        std::filesystem::resize_file(path_,data_offset_+2*gate_bytes+down_bytes);
        original_.resize(source_expert_bytes_);
        for(size_t i=0;i<original_.size();++i)original_[i]=uint8_t((i*37u+i/71u)&255u);
        std::fstream payload(path_,std::ios::binary|std::ios::in|std::ios::out);
        payload.seekp(data_offset_+2*gate_bytes+source_expert_bytes_); // Original expert1, not ordinal0.
        payload.write(reinterpret_cast<const char*>(original_.data()),original_.size());
        if(!payload)throw std::runtime_error("Cannot populate original buffered source expert");
        } catch(...) {std::error_code error;std::filesystem::remove(path_,error);throw;}
    }
    BufferedEconomySource(const BufferedEconomySource&)=delete;
    BufferedEconomySource& operator=(const BufferedEconomySource&)=delete;
    /** @brief Retire only this unique fixture after all loader mappings are gone. */
    ~BufferedEconomySource(){std::error_code error;std::filesystem::remove(path_,error);}
    /** @return Original immutable file selected by the genuine ModelLoader. */
    const std::string& path()const noexcept{return path_;}
    /** @return Native down row extent, including both original F halves. */
    size_t sourceRowBytes()const noexcept{return source_row_bytes_;}
    /** @return The sole populated original native expert bytes. */
    const std::vector<uint8_t>& original()const noexcept{return original_;}
private:
    std::string path_;
    size_t data_offset_=0,source_row_bytes_=0,source_expert_bytes_=0;
    std::vector<uint8_t> original_;
};
/** @return Kernel read-system-call count for this process; no mmap residency inference. */
uint64_t bufferedSourceReadCalls() {
    std::ifstream stream("/proc/self/io");std::string key;uint64_t value;
    while(stream>>key>>value)if(key=="syscr:")return value;
    throw std::runtime_error("Cannot observe original buffered source read calls");
}
}
/** @test Realistic down-K loading reuses file buffering instead of seeking/reading each native row. */
TEST(CPUJointBufferedSource, RealisticStridedReadUsesBufferedSyscallsAndExactBytes) {
    BufferedEconomySource file;
    auto mpi=std::make_shared<MPIContext>(0,1,MPI_COMM_NULL);TensorFactory factory(*mpi);ModelLoader loader(&factory);
    loader.setPayloadAccessPattern(ModelPayloadAccessPattern::SparseSelection);
    // The fixture write supplies the same warm OS file cache to both reader
    // diagnostics; no source mmap is touched or pre-faulted for either one.
    loader.setSkipMmapCacheEviction(true);
    ASSERT_TRUE(loader.loadModel(file.path()));
    auto plan=localCPUJointSourceTestPlan();
    for(size_t expert=0;expert<512;++expert)plan.placements.front().routed_expert_tier[expert]=expert%8==0?0:1;
    const auto runtime=resolveMoEExpertOverlayRuntimePlan(std::make_shared<const MoERoutedExpertPlacementPlan>(plan),
        {.current_world_rank=0,.validate_mvp_root_reachability=false});
    const auto preparation=MoEExpertOverlayPreparationPlan::build(*runtime,loader);
    std::shared_ptr<const CPUExpertShardSourceSelection> selected;
    for(const auto& request:preparation.requests())if(request.cpu_joint_source &&
        request.cpu_joint_source->participant().world_rank==0) {selected=request.cpu_joint_source;break;}
    ASSERT_TRUE(selected);ASSERT_EQ(selected->expertIds().front(),1);
    const size_t row_bytes=file.sourceRowBytes()/2, payload_bytes=row_bytes*2560;
    const auto placement=CPUStoragePlacement::onNode(selected->participant().address.numa_node);
    const auto name=std::string("blk.7.ffn_down_exps.weight");
    const auto native=loader.nativeWeightSourceGeometry(name);
    const auto request=nativeJointRequirementForTest(selected,WeightRole::MoEExpertDown,name);
    const auto projections=projectNativeWeightSource(request,native);
    ASSERT_EQ(projections.size(),selected->expertIds().size());
    const auto range=projections.front().range;
    EXPECT_EQ(range.original_expert,1u);
    const auto read_request=nativeReadRequestForTest(native,range);
    const auto physical=resolveNativeWeightLoadStaging(read_request,placement,
        placement.physicalBytes<uint8_t>(payload_bytes)).physical_bytes;
    PhysicalMemoryPlanBuilder budget;budget.add({0,DeviceId::cpu(),physical,physical},PhysicalMemoryOwner::WeightLoadStaging,physical);
    const auto memory=std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(budget.build()),0);
    const auto before=bufferedSourceReadCalls();
    std::chrono::steady_clock::duration buffered_duration{};
    {
        const auto start=std::chrono::steady_clock::now();
        const auto visits=visitNativeSourceForTest(loader,name,range,read_request,placement,memory,
            selected->participant().device,physical,[&](const auto& window,auto bytes,auto) {
                for(size_t row=0;row<window.rows;++row) {
                    const auto source=(window.first_row+row)*file.sourceRowBytes()+
                        window.first_column/native.native_group_elements*native.native_group_bytes;
                    EXPECT_EQ(std::memcmp(bytes.data()+row*window.native_row_bytes,
                        file.original().data()+source,window.native_row_bytes),0);
                }
            });
        buffered_duration=std::chrono::steady_clock::now()-start;
        EXPECT_EQ(visits,1u);
    }
    const auto calls=bufferedSourceReadCalls()-before;
    EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::WeightLoadStaging,
        PhysicalMemoryMaterializationKind::NewAllocation),0u);
    // An unbuffered per-row implementation takes at least2560 reads (and seeks).
    // This actual process count includes the observation itself and setup reads.
    EXPECT_LT(calls,2560u/4u);
    const auto begin_mmap=std::chrono::steady_clock::now();
    auto slice=selected->sourceLoadSlice(WeightRole::MoEExpertDown);
    slice.expert_ids={selected->expertIds().front()};slice.expert_start=size_t(slice.expert_ids.front());slice.expert_count=1;
    auto mapped=loader.loadTensorExpertProjectionSlice(name,slice,
        DeviceId::cpu(),WeightPrecision::NATIVE);
    const auto end_mmap=std::chrono::steady_clock::now();
    ASSERT_TRUE(mapped);ASSERT_EQ(mapped->size_bytes(),payload_bytes);
    const auto *mapped_bytes=static_cast<const uint8_t*>(mapped->raw_data());
    for(size_t row=0;row<2560;++row)
        ASSERT_EQ(std::memcmp(mapped_bytes+row*row_bytes,file.original().data()+row*file.sourceRowBytes(),row_bytes),0);
    RecordProperty("source_rows",2560);RecordProperty("useful_native_bytes",std::to_string(payload_bytes));
    RecordProperty("stream_gap_bytes",std::to_string((2560u-1u)*row_bytes));
    RecordProperty("kernel_read_calls",std::to_string(calls));
    RecordProperty("buffered_diagnostic_us",std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(buffered_duration).count()));
    RecordProperty("mmap_diagnostic_us",std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(end_mmap-begin_mmap).count()));
}
}
