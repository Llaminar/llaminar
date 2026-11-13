/**
 * @file Test__PooledNativeGraphInventory.cpp
 * @brief Replay the consumed cold SDK inspector against immutable and malformed native graph observations.
 *
 * Recorded adapters expose no GPU runtime. These tests exercise the exact
 * diagnostic query algorithm, including count drift and unsupported nesting.
 */
#include "integration/backends/PooledNativeGraphInventory.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <string>
#include <utility>

namespace llaminar2::test::pooled_program_observation {
namespace {
struct Observation {std::string event,error;std::size_t nodes,edges,index;const void* node;const void* other;int kind;};
std::vector<Observation> observations;
/** @brief Recorded original graph calls; changing query replies model invalid native evidence. */
struct RecordedAPI {
    using Graph=const void*;using Node=const void*;using Kind=int;
    static constexpr int success=0;
    static constexpr const char* backend="recorded",*operation="recorded_instantiate";
    static inline std::vector<Node> recordedNodes;
    static inline std::vector<std::pair<Node,Node>> recordedEdges;
    static inline bool nodeDrift=false,edgeDrift=false,queryFailure=false,throwHost=false;
    static inline int recordedKind=0;
    /** @brief Return the recorded graph membership, optionally changing its second observation. */
    static int nodes(Graph,Node* out,std::size_t* count){
        if(throwHost)throw std::bad_alloc();
        if(queryFailure)return 1;
        if(out){std::copy(recordedNodes.begin(),recordedNodes.end(),out);*count=recordedNodes.size()-(nodeDrift?1:0);}
        else *count=recordedNodes.size();
        return success;
    }
    /** @brief Return only the recorded dependency edges, including explicit malformed controls. */
    static int edges(Graph,Node* from,Node* to,std::size_t* count){
        if(from){for(std::size_t i=0;i<recordedEdges.size();++i){from[i]=recordedEdges[i].first;to[i]=recordedEdges[i].second;}
            *count=recordedEdges.size()-(edgeDrift?1:0);}
        else *count=recordedEdges.size();
        return success;
    }
    /** @brief Supply an explicit native kind instead of guessing from observed launches. */
    static int kind(Node,Kind* out){*out=recordedKind;return success;}
    /** @brief Mirror the closed supported diagnostic shape, not a production capability. */
    static bool flatLookupKind(Kind value){return value==0;}
};
/** @brief Restore one fixed two-node graph and clear every injected query failure. */
void reset(){
    observations.clear();RecordedAPI::recordedNodes={reinterpret_cast<void*>(0x10),reinterpret_cast<void*>(0x20)};
    RecordedAPI::recordedEdges={{RecordedAPI::recordedNodes[0],RecordedAPI::recordedNodes[1]}};
    RecordedAPI::nodeDrift=RecordedAPI::edgeDrift=RecordedAPI::queryFailure=RecordedAPI::throwHost=false;
    RecordedAPI::recordedKind=0;
}
/** @brief Consume the unchanged production diagnostic template with recorded SDK inputs. */
void inspect(){inspectFlatLookupGraph<RecordedAPI>(reinterpret_cast<void*>(0x30),1);}
/** @brief Require a failed receipt, never a truncated success. */
void requireFailed(const char* reason){
    inspect();ASSERT_EQ(observations.size(),1u);EXPECT_EQ(observations[0].event,"failed");EXPECT_EQ(observations[0].error,reason);
}
}
/** @brief Test sink for the actual inspector; production diagnostic emission remains separately native-tested. */
void inventory(const char*,const char*,std::uint64_t,const void*,const char* event,std::size_t nodes,std::size_t edges,
    std::size_t index,const void* node,const void* other,int kind,const char* error)noexcept{
    observations.push_back({event,error,nodes,edges,index,node,other,kind});
}
TEST(PooledNativeGraphInventory, CompleteResetAndRepeatedOriginalDag){
    for(unsigned replay=0;replay<20;++replay){reset();inspect();ASSERT_EQ(observations.size(),5u);
        EXPECT_EQ(observations.front().event,"begin");EXPECT_EQ(observations.back().event,"complete");
        EXPECT_EQ(observations[1].node,RecordedAPI::recordedNodes[0]);EXPECT_EQ(observations[2].node,RecordedAPI::recordedNodes[1]);
        EXPECT_EQ(observations[3].event,"edge");EXPECT_EQ(observations[3].other,RecordedAPI::recordedNodes[1]);
        for(const auto& row:observations){EXPECT_EQ(row.nodes,2u);EXPECT_EQ(row.edges,1u);}
    }
}
TEST(PooledNativeGraphInventory, ChangedCountsAndPartialQueriesCannotComplete){
    reset();RecordedAPI::nodeDrift=true;requireFailed("node_inventory_changed");
    reset();RecordedAPI::edgeDrift=true;requireFailed("edge_inventory_changed_or_nondefault");
    reset();RecordedAPI::queryFailure=true;requireFailed("node_count_query");
    reset();RecordedAPI::throwHost=true;requireFailed("host_inspection_failure");
}
TEST(PooledNativeGraphInventory, UnknownNestedOrInvalidMembershipCannotComplete){
    reset();RecordedAPI::recordedKind=4;requireFailed("node_kind_outside_flat_lookup");
    reset();RecordedAPI::recordedNodes[1]=RecordedAPI::recordedNodes[0];requireFailed("duplicate_or_null_node");
    reset();RecordedAPI::recordedEdges[0].second=reinterpret_cast<void*>(0x99);requireFailed("foreign_or_self_edge");
    reset();RecordedAPI::recordedEdges[0].second=RecordedAPI::recordedNodes[0];requireFailed("foreign_or_self_edge");
}
TEST(PooledNativeGraphInventory, SingletonWithoutEdgesHasACompleteInventory){
    reset();RecordedAPI::recordedNodes.resize(1);RecordedAPI::recordedEdges.clear();inspect();
    ASSERT_EQ(observations.size(),3u);EXPECT_EQ(observations.back().event,"complete");EXPECT_EQ(observations.back().edges,0u);
}
}
