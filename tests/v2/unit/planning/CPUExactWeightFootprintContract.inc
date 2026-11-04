/**
 * @file CPUExactWeightFootprintContract.inc
 * @brief Canonical CPU admission preserves actual final packing, per-expert roots and resident layouts.
 *
 * These metadata contracts invoke the production estimator and allocator/packer
 * geometry without materializing sources or advertising native CPU readiness.
 * The QFN policy lifecycle independently checks actual live final claims.
 */
namespace {
/** @return One original matrix inventory with exact native block bytes. */
ModelMemoryProfile exactCPUProfile(std::string name,std::string type,size_t n,size_t k,size_t experts=1)
{
    ModelMemoryProfile profile;
    profile.architecture=experts==1?"qwen2":"qwen35moe";
    profile.n_layers=1;profile.d_model=int(k);profile.d_ff=int(n);
    profile.expert_count=experts==1?0:int(experts);profile.expert_feed_forward_length=int(n);
    const auto parsed=tensorTypeFromName(type);
    if(!parsed)throw std::invalid_argument("Missing original fixture source format");
    const bool floating=*parsed==TensorType::FP32 || *parsed==TensorType::FP16 || *parsed==TensorType::BF16;
    const size_t native=floating?n*k*(*parsed==TensorType::FP32?4u:2u):
        NativeEmbeddingLayout::forSource(*parsed,int(n),int(k)).byte_size;
    profile.tensors={{std::move(name),native*experts,std::move(type),n*k*experts,k,0}};
    return profile;
}
}
/** @test Every codebook retains its actual64-row padded root for small and partial matrices. */
TEST(CPUExactWeightFootprintContract, AllNativeFormatsRetainIndependentMatrixAndExpertPadding)
{
    for(const auto& format:native_vnni_formats::kAllSourceFormats)
      for(const size_t n:{3u,17u,65u})for(const size_t k:{256u,512u}) {
        SCOPED_TRACE(format.quant_type);
        const auto expected=CPUWeightPreparationMemory::sourceNative(format.quant_type,n,k).persistent_bytes;
        auto dense=exactCPUProfile("blk.0.ffn_up.weight",std::string(format.quant_type),n,k);
        EXPECT_EQ(WeightMemoryEstimator::estimate(dense,DeviceId::cpu()).device_bytes,expected);
        auto experts=exactCPUProfile("blk.0.ffn_up_exps.weight",std::string(format.quant_type),n,k,7);
        const auto selected=DeviceWeightResidency::selectedRoutedExpertsOnly(7,{3});
        const auto actual=WeightMemoryEstimator::estimate(experts,DeviceId::cpu(),0,1,0,0,selected);
        EXPECT_EQ(actual.device_bytes,3*expected);
        EXPECT_EQ(actual.native_bytes,experts.tensors[0].native_bytes/7*3);
      }
}
/** @test Typed TP axes remain separate from native final padding on every declared degree. */
TEST(CPUExactWeightFootprintContract, TensorShardsPreserveFinalNAndKThroughEight)
{
    for(const auto& format:native_vnni_formats::kAllSourceFormats)for(int degree:{1,2,4,8}) {
        SCOPED_TRACE(format.quant_type);
        auto gate=exactCPUProfile("blk.0.ffn_up.weight",std::string(format.quant_type),17u*degree,256);
        auto down=exactCPUProfile("blk.0.ffn_down.weight",std::string(format.quant_type),65,256u*degree);
        down.d_model=65;down.d_ff=256*degree;
        for(int shard=0;shard<degree;++shard) {
            EXPECT_EQ(WeightMemoryEstimator::estimate(gate,DeviceId::cpu(),shard,degree).device_bytes,
                CPUWeightPreparationMemory::sourceNative(format.quant_type,17,256).persistent_bytes);
            EXPECT_EQ(WeightMemoryEstimator::estimate(down,DeviceId::cpu(),shard,degree).device_bytes,
                CPUWeightPreparationMemory::sourceNative(format.quant_type,65,256).persistent_bytes);
        }
    }
}
/** @test Raw native lookup and scalar roles do not acquire GEMM padding; tied native output owns its distinct layout. */
TEST(CPUExactWeightFootprintContract, NativeResidentAndTiedLookupKeepActualFinalOwners)
{
    const auto placement=CPUStoragePlacement::local();
    for(const auto& format:native_vnni_formats::kAllSourceFormats) {
        auto router=exactCPUProfile("blk.0.ffn_gate_inp.weight",std::string(format.quant_type),17,256);
        EXPECT_EQ(WeightMemoryEstimator::estimate(router,DeviceId::cpu()).device_bytes,
            placement.physicalBytes<uint8_t>(router.tensors[0].native_bytes));
        auto tied=exactCPUProfile("token_embd.weight",std::string(format.quant_type),17,256);
        tied.vocab_size=17;tied.tensors[0].layer_index=-1;
        const auto actual=WeightMemoryEstimator::estimate(tied,DeviceId::cpu());
        const auto resident=placement.physicalBytes<uint8_t>(tied.tensors[0].native_bytes);
        EXPECT_EQ(actual.prepared_embedding_bytes,resident);
        EXPECT_EQ(actual.device_bytes,resident+CPUWeightPreparationMemory::sourceNative(format.quant_type,17,256).persistent_bytes);
    }
    for(const auto type:{std::string("F32"),std::string("F16"),std::string("BF16")}) {
        auto tied=exactCPUProfile("token_embd.weight",type,17,256);
        tied.vocab_size=17;tied.tensors[0].layer_index=-1;
        const auto elements=tied.tensors[0].elements;
        EXPECT_EQ(WeightMemoryEstimator::estimate(tied,DeviceId::cpu()).device_bytes,
            type=="F32"?placement.physicalBytes<float>(elements):placement.physicalBytes<uint16_t>(elements));
    }
    ModelMemoryProfile scalar;scalar.architecture="qwen2";scalar.n_layers=1;
    scalar.tensors={{"output_norm.weight",12,"F32",3,0,-1}};
    EXPECT_EQ(WeightMemoryEstimator::estimate(scalar,DeviceId::cpu()).device_bytes,placement.physicalBytes<float>(3));
}

/** @test Each final root rounds under the original typed placement, independently of allocator ordinal or expert population. */
TEST(CPUExactWeightFootprintContract, OriginalLocalAndNodePlacementRoundEveryActualRoot)
{
    const auto estimate=[](const auto& profile,CPUStoragePlacement placement,const DeviceWeightResidency& residency=DeviceWeightResidency{}) {
        return WeightMemoryEstimator::estimate(profile,DeviceId::cpu(),0,1,0,-1,residency,std::nullopt,
            WeightComponentScope::EmbeddingAndTerminal,placement);
    };
    for(const auto& format:native_vnni_formats::kAllSourceFormats) {
        const auto payload=CPUWeightPreparationMemory::sourceNative(format.quant_type,3,256).persistent_payload_bytes;
        auto experts=exactCPUProfile("blk.0.ffn_up_exps.weight",std::string(format.quant_type),3,256,7);
        const auto selected=DeviceWeightResidency::selectedRoutedExpertsOnly(7,{3});
        auto resident=exactCPUProfile("blk.0.ffn_gate_inp.weight",std::string(format.quant_type),3,256);
        for(const auto placement:{CPUStoragePlacement::local(),CPUStoragePlacement::onNode(7)}) {
            EXPECT_EQ(estimate(experts,placement,selected).device_bytes,3*placement.physicalBytes<uint8_t>(payload));
            EXPECT_EQ(estimate(resident,placement).device_bytes,
                placement.physicalBytes<uint8_t>(resident.tensors[0].native_bytes));
        }
    }
    auto floating=exactCPUProfile("blk.0.ffn_up_exps.weight","F32",3,64,7);
    const auto selected=DeviceWeightResidency::selectedRoutedExpertsOnly(7,{3});
    for(const auto placement:{CPUStoragePlacement::local(),CPUStoragePlacement::onNode(7)})
        EXPECT_EQ(estimate(floating,placement,selected).device_bytes,3*placement.physicalBytes<float>(3*64));
    EXPECT_LT(estimate(floating,CPUStoragePlacement::local(),selected).device_bytes,
        estimate(floating,CPUStoragePlacement::onNode(7),selected).device_bytes);
}
