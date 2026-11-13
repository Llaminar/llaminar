/** @file Test__PooledRemoteFamilyScope.cpp
 * @brief Versioned family eligibility preserves kernel-chain/ABI validity independently of observed size without per-capacity catalog rows.
 * These synthetic boundary inputs exercise the actual predicate; they are not native observations or installed limits.
 */
#include "backends/NativeAuxiliaryResourceCatalog.h"
#include <gtest/gtest.h>
namespace llaminar2::test {
TEST(PooledRemoteFamilyScope,SeveralMetadataSizesShareOneClosedEnvelope){
    const auto args=PooledRemoteGraphProgramClass::argumentLayout();
    const NativeAuxiliaryRemoteFamilyScope scope{1,{100,99,GPUFlatGraphLimits::bit(GPUFlatGraphNodeKind::Kernel)},args};
    ASSERT_TRUE(scope.valid());EXPECT_GT(args.count,0u);EXPECT_GT(args.bytes,0u);
    for(unsigned repeat=0;repeat<20;++repeat)for(const auto nodes:{1u,2u,4u,100u})
        EXPECT_TRUE(scope.accepts({nodes,nodes-1,GPUFlatGraphLimits::bit(GPUFlatGraphNodeKind::Kernel)},args));
}
TEST(PooledRemoteFamilyScope,VersionNativeKindGeometryAndArgumentsRemainValidWithoutSampleCeilings){
    const auto args=PooledRemoteGraphProgramClass::argumentLayout();
    const NativeAuxiliaryRemoteFamilyScope original{1,{100,99,GPUFlatGraphLimits::bit(GPUFlatGraphNodeKind::Kernel)},args};
    const GPUFlatGraphLimits nodes{93,92,GPUFlatGraphLimits::bit(GPUFlatGraphNodeKind::Kernel)};
    for(unsigned repeat=0;repeat<20;++repeat){
        auto scope=original;scope.version=2;EXPECT_FALSE(scope.accepts(nodes,args));
        scope=original;scope.version=0;EXPECT_FALSE(scope.valid());
        scope=original;scope.native.maximumNodes=92;EXPECT_TRUE(scope.accepts(nodes,args));
        scope=original;scope.native.maximumEdges=91;EXPECT_FALSE(scope.valid());
        auto larger=args;++larger.count;EXPECT_TRUE(original.accepts(nodes,larger));
        larger=args;++larger.bytes;EXPECT_TRUE(original.accepts(nodes,larger));
        larger=args;larger.alignment*=2;EXPECT_TRUE(original.accepts(nodes,larger));
        EXPECT_FALSE(original.accepts({93,91,nodes.allowedKinds},args));
        EXPECT_FALSE(original.accepts({93,92,nodes.allowedKinds|GPUFlatGraphLimits::bit(GPUFlatGraphNodeKind::EventRecord)},args));
    }
}
}
