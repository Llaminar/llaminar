/**
 * @file GroupedHCUpShortKReplay.inc
 * @brief Counted production capture and independent exact M1 oracle for ROCm HC-up.
 */
#ifdef HAVE_ROCM
#include "kernels/common/VerifierDispatchTable.h"
namespace {
/** @brief Compare an actual production grouped capture with exact independent M1 diagnostic calls.
 * @param device Original physical ROCm test participant. @param m Retained physical row capacity.
 * @param source_file Optional original GGUF-native Q8_0 window for manual replay.
 */
void proveGroupedHCUpShortK(DeviceId device,int m,const std::filesystem::path& source_file) {
    auto* backend=getBackendFor(device);
    require(backend&&backend->deviceCount()>device.ordinal,"Exact rank-local ROCm participant required");
    const auto hardware=rocm::nativeQuantisedHardwareForDevice(device.ordinal);
    const auto table=native_quantised::selectVerifierDispatchTable(hardware);
    const bool measured=table==native_quantised::VerifierDispatchTable::MI50;
    constexpr int n=10240,k=320,blocks=10,guards=16;
    auto& worker=GPUDeviceContextPool::instance().getContext(device);
    worker.submitAndWait([&] {
        Storage storage{device,backend,ExplicitGPUStream(worker.getOrCreateAuxiliaryStream("grouped_hc_up")).get()};
        auto raw=nativeWeights(n,k);
        if(!source_file.empty()) {
            require(std::filesystem::file_size(source_file)==raw.size(),"Exact original HC-up source bytes");
            std::ifstream file(source_file,std::ios::binary);
            require(bool(file.read(reinterpret_cast<char*>(raw.data()),raw.size())),"Read original Q8_0 source");
        }
        Q8_0Tensor source({std::size_t(n),std::size_t(k)},raw);
        const auto weights_owner=prepare(source,raw,storage,n,k);
        DeviceNativeQuantisedMatrixDesc weights;
        require(weights_owner->exportNativeQuantisedMatrixDesc(weights)&&weights.source_identity_present&&weights.codebook_id==19,
                "Original HC-up physical layout");
        auto* live=storage.input(std::vector<int>{m});
        auto* input=storage.allocate<float>(m*k);
        const auto rows=DeviceRowRange::deviceCounted(m,live);
        std::array<GPUHCResidualDownPublication,2> arms{
          GPUHCResidualDownPublication{{weights,rows}},GPUHCResidualDownPublication{{weights,rows}}};
        for(auto& arm:arms) {
            auto& a=arm.projection;
            a.quantizer=Q8ResidualQuantizerPolicy::ROCmFiniteHalfNativeReciprocal;
            a.input=input;
            a.output=storage.allocate<float>(m*n+guards);
            a.primary=storage.allocate<signed char>(m*k+guards);
            a.residual=storage.allocate<signed char>(m*k+guards);
            a.primaryScales=storage.allocate<float>(m*blocks+guards);
            a.residualScales=storage.allocate<float>(m*blocks+guards);
            arm.activated=storage.allocate<float>(m*k+guards);
        }
        auto capture=[&] {
            auto graph=worker.createGraphCapture(storage.stream);
            ScopedBackendGraphCapture recording(worker,*graph,"Production grouped HC-up ordered short K");
            require(recording.begin(),"Capture exact grouped projection");
            rocmHCResidualDownQ8Projection(arms[0],storage.stream,{});
            recording.finish();
            require(graph->instantiate(),"Instantiate grouped HC-up");
            std::vector<GPUGraphKernelNodeInfo> nodes;
            std::string error;
            require(graph->inspectKernelNodes(nodes,&error)&&nodes.size()==2&&graph->nodeCount()==2,
                    "Exact native publisher and consumer without transfer nodes");
            bool selected=false;
            for(const auto& node:nodes) {
                require(node.name_resolved&&node.local_memory_bytes_per_thread==0,"Zero private bytes in both retained entries");
                if(node.name.find("publishHCResidualDown")!=std::string::npos)
                    continue;
                if(measured) {
                    const bool wide=m==4 && hardware.admits(640,20480);
                    require(node.name.find(wide?"projectHCGroupedShortKWide":"projectHCGroupedShortK")!=std::string::npos &&
                            node.block_x==(wide?640u:320u) && node.grid_x==(wide?160u:320u) &&
                            node.grid_y==static_cast<unsigned>((m+3)/4) &&
                            node.static_shared_memory_bytes==(wide?20480u:10240u),
                            "Observed retained HC-up schedule must match the measured table");
                } else {
                    require(table==native_quantised::VerifierDispatchTable::Portable &&
                            node.name.find("projectHCGroupedShortK")==std::string::npos &&
                            node.block_x==128u && node.grid_x==static_cast<unsigned>(80*((m+3)/4)) &&
                            node.grid_y==1 && node.static_shared_memory_bytes==0,
                            "Unmeasured ROCm participates through its original portable grouped projection");
                }
                selected=true;
            }
            require(selected,"Original production ingress did not select the table-owned HC-up projection");
            return graph;
        };
        const auto serial=[&](int count) {
            for(int i=0;i<count;++i) {
                auto row=arms[1];
                auto& a=row.projection;
                a.rows=DeviceRowRange::fullyActive(1);
                a.input+=i*k;
                a.output+=i*n;
                a.primary+=i*k;
                a.residual+=i*k;
                a.primaryScales+=i*blocks;
                a.residualScales+=i*blocks;
                row.activated+=i*k;
                rocmHCResidualDownQ8Projection(row,storage.stream,{});
            }
        };
        const auto poison=[&](auto& arm) {
            auto& a=arm.projection;
            storage.upload(a.output,std::vector<float>(m*n+guards,-990.125f));
            storage.upload(arm.activated,std::vector<float>(m*k+guards,-990.125f));
            storage.upload(a.primary,std::vector<signed char>(m*k+guards,113));
            storage.upload(a.residual,std::vector<signed char>(m*k+guards,113));
            storage.upload(a.primaryScales,std::vector<float>(m*blocks+guards,-990.125f));
            storage.upload(a.residualScales,std::vector<float>(m*blocks+guards,-990.125f));
        };
        const auto same=[&](auto* left,auto* right,int words) {
            const auto a=storage.download(left,words);
            const auto b=storage.download(right,words);
            require(std::memcmp(a.data(),b.data(),a.size()*sizeof(a[0]))==0,"Serial row byte equivalence and complete guards");
        };
        const auto input_for=[&](int history) {
            std::vector<float> in(m*k);
            for(int i=0;i<m*k;++i)
                in[i]=std::sin(float(i*7+history*31))*float(history%7+1)*0.125f;
            storage.upload(input,in);
        };
        std::unique_ptr<IGPUGraphCapture> graph;
        try {
            graph=capture();
            for(int history=0;history<2*(m+1)+8;++history) {
                input_for(history==2*(m+1)?0:history);
                const int count=history%(m+1);
                storage.upload(live,std::vector<int>{count});
                for(auto& arm:arms)poison(arm);
                for(int repeat=0;repeat<2;++repeat)
                    require(graph->launch(),"Retained grouped capture replay");
                serial(count);
                const auto& got=arms[0];
                const auto& expected=arms[1];
                same(got.projection.output,expected.projection.output,m*n+guards);
                same(got.activated,expected.activated,m*k+guards);
                same(got.projection.primary,expected.projection.primary,m*k+guards);
                same(got.projection.residual,expected.projection.residual,m*k+guards);
                same(got.projection.primaryScales,expected.projection.primaryScales,m*blocks+guards);
                same(got.projection.residualScales,expected.projection.residualScales,m*blocks+guards);
            }
            std::cout<<"GROUPED_HC_UP_ORACLE,M="<<m<<",counts=0.."<<m<<",production=1\n";
        } catch(...) {
            require(backend->synchronizeStream(storage.stream,device.ordinal),"Retire failed replay");
            throw;
        }
        require(backend->synchronizeStream(storage.stream,device.ordinal),"Retire all native launches");
    });
}
}
/** @test The explicit production dispatch retains every supported grouped HC-up row count. */
TEST(Q8ResidualCorrectionDevice, ROCmGroupedHCUpShortKProductionReplay)
{
    auto* backend=getBackendFor(DeviceId::rocm(0));
    require(backend&&backend->deviceCount()>0,"ROCm lane requires a visible physical device");
    const auto& roster=rocmHardwareMetadata();
    for(std::size_t index=0;index<roster.size();++index) {
        const auto& device=roster.at(static_cast<int>(index));
        if(device.compute_units!=60||device.native_wave_width!=64||
           (device.isa_target!="gfx906"&&!device.isa_target.starts_with("gfx906:")))continue;
        const auto owner=DeviceId::rocm(device.ordinal);
        if(const char* dir=std::getenv("LLAMINAR_TEST_GROUPED_HCU_SOURCE_DIR")) {
            for(int source=0;source<4;++source) {
                const auto file=std::filesystem::path(dir)/("role-"+std::to_string(source)+".bin");
                for(int m:{2,3,4,5,6,7,8})proveGroupedHCUpShortK(owner,m,file);
            }
        } else {
            for(int m:{2,3,4,5,6,7,8})proveGroupedHCUpShortK(owner,m,{});
        }
        return;
    }
    GTEST_SKIP()<<"Experimental grouped HC-up requires native gfx906 wave64 and 60 CUs";
}

/** @test Measured MI50 and portable ROCm schedules both keep original four-row bytes. */
TEST(Q8ResidualCorrectionDevice, ROCmGroupedHCUpMeasuredTableReplay)
{
    auto* backend=getBackendFor(DeviceId::rocm(0));
    require(backend&&backend->deviceCount()>0,"ROCm lane requires a visible physical device");
    const auto& roster=rocmHardwareMetadata();
    ASSERT_GT(roster.size(),0u);
    for(std::size_t index=0;index<roster.size();++index)
        proveGroupedHCUpShortK(DeviceId::rocm(roster.at(static_cast<int>(index)).ordinal),4,{});
}
#endif
