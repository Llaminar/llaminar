/**
 * @file Test__GPUFlatGraphInspection.cpp
 * @brief Actual CUDA/HIP cold inspection bodies over recorded SDK metadata and bounded scratch.
 *
 * The build extracts the production methods unchanged. Only native query results
 * and owner activation are doubled. These tests neither load a GPU runtime nor
 * create a certificate; original-PMA graph ownership has a separate replay.
 */
#include "backends/detail/GPUFlatGraphInspection.h"
#include <gtest/gtest.h>
#include <array>
#include <algorithm>
namespace llaminar2::test::flat_graph_replay {
/** @brief Independent native query trace, not a graph builder or accounting authority. */
struct Graph {
    using Node=void*;
    std::array<Node,4> nodes{reinterpret_cast<Node>(0x10),reinterpret_cast<Node>(0x20),reinterpret_cast<Node>(0x30),reinterpret_cast<Node>(0x40)};
    std::array<Node,4> from{nodes[0],nodes[1],nullptr,nullptr},to{nodes[1],nodes[2],nullptr,nullptr};
    std::array<int,4> kinds{};
    std::size_t nodeCount=3,edgeCount=2;
    int nodesQueries=0,edgeQueries=0,kindQueries=0,activations=0;
    bool active=true,nodeFailure=false,edgeFailure=false,kindFailure=false,changedNodes=false,changedEdges=false,nondefaultEdges=false;
};
using cudaGraph_t=Graph*;using hipGraph_t=Graph*;using cudaGraphNode_t=void*;using hipGraphNode_t=void*;
using cudaGraphNodeType=int;using hipGraphNodeType=int;
constexpr int cudaSuccess=0,hipSuccess=0;
constexpr int cudaGraphNodeTypeKernel=0,cudaGraphNodeTypeMemcpy=1,cudaGraphNodeTypeMemset=2,cudaGraphNodeTypeEmpty=5,cudaGraphNodeTypeWaitEvent=6,cudaGraphNodeTypeEventRecord=7;
constexpr int hipGraphNodeTypeKernel=0,hipGraphNodeTypeMemcpy=1,hipGraphNodeTypeMemset=2,hipGraphNodeTypeEmpty=5,hipGraphNodeTypeWaitEvent=6,hipGraphNodeTypeEventRecord=7;
thread_local Graph* selected=nullptr;
/** @brief Recorded query preserves SDK capacity/output semantics and can expose an unstable definition. */
int nodes(Graph* graph,void** values,std::size_t* count){
    ++graph->nodesQueries;if(graph->nodeFailure)return 1;
    if(values)std::copy_n(graph->nodes.begin(),std::min(*count,graph->nodeCount),values);
    *count=graph->nodeCount+(values && graph->changedNodes);return 0;
}
/** @brief Record exact default-edge selection; no native edge or node is authored by the inspector. */
int edges(Graph* graph,void** from,void** to,std::size_t* count){
    ++graph->edgeQueries;if(graph->edgeFailure || graph->nondefaultEdges)return 1;
    if(from && to){const auto n=std::min(*count,graph->edgeCount);std::copy_n(graph->from.begin(),n,from);std::copy_n(graph->to.begin(),n,to);}
    *count=graph->edgeCount+(from && graph->changedEdges);return 0;
}
int kind(void* node,int* value){++selected->kindQueries;if(selected->kindFailure)return 1;const auto it=std::find(selected->nodes.begin(),selected->nodes.end(),node);if(it==selected->nodes.end())return 1;*value=selected->kinds[it-selected->nodes.begin()];return 0;}
int cudaGraphGetNodes(Graph* g,void** n,std::size_t* c){return nodes(g,n,c);}
int hipGraphGetNodes(Graph* g,void** n,std::size_t* c){return nodes(g,n,c);}
int cudaGraphGetEdges(Graph* g,void** f,void** t,void* data,std::size_t* c){if(data)return 1;return edges(g,f,t,c);}
int hipGraphGetEdges(Graph* g,void** f,void** t,std::size_t* c){return edges(g,f,t,c);}
int cudaGraphNodeGetType(void* n,int* k){return kind(n,k);}
int hipGraphNodeGetType(void* n,int* k){return kind(n,k);}
/** @brief SDK owner boundary only; the actual production method and shared algorithm follow below. */
class CUDAGraphCapture {
public:
    explicit CUDAGraphCapture(Graph* graph):graph_(graph){}
    bool inspectFlatGraphStructure(const GPUFlatGraphLimits&,std::span<std::byte>,GPUFlatGraphStructure&,std::string*)const;
private:
    bool activateOwner(const char*)const{if(!graph_)return false;++graph_->activations;selected=graph_;return graph_->active;}
    Graph* graph_;
};
class HIPGraphCapture {
public:
    explicit HIPGraphCapture(Graph* graph):graph_(graph){}
    bool inspectFlatGraphStructure(const GPUFlatGraphLimits&,std::span<std::byte>,GPUFlatGraphStructure&,std::string*)const;
private:
    bool activateOwner(const char*)const{if(!graph_)return false;++graph_->activations;selected=graph_;return graph_->active;}
    Graph* graph_;
};
#include "FlatGraphInspectionBodies.inc"
/** @brief Invoke both actual backend methods with the same independent trace. */
template<class Body>void each(Body body){body.template operator()<CUDAGraphCapture>();body.template operator()<HIPGraphCapture>();}
constexpr GPUFlatGraphLimits scope{3,2,GPUFlatGraphLimits::bit(GPUFlatGraphNodeKind::Kernel)};
struct alignas(std::uintptr_t) Scratch {std::array<std::byte,256> bytes;};
TEST(GPUFlatGraphInspection, ActualBackendBodiesInspectTheCompleteDefaultDAG){
    each([]<class Capture>{Graph graph;Capture capture(&graph);Scratch scratch;GPUFlatGraphStructure out;std::string error;
        ASSERT_TRUE(capture.inspectFlatGraphStructure(scope,scratch.bytes,out,&error));EXPECT_TRUE(error.empty());EXPECT_EQ(out.nodes,3u);EXPECT_EQ(out.edges,2u);EXPECT_EQ(out.kinds[0],3u);EXPECT_TRUE(out.linearChain);
        EXPECT_EQ(graph.nodesQueries,2);EXPECT_EQ(graph.edgeQueries,2);EXPECT_EQ(graph.kindQueries,3);});
}
TEST(GPUFlatGraphInspection, ResetEmptyAndLargeToSmallReuseTheSameBoundedScratch){
    each([]<class Capture>{Graph graph;Capture capture(&graph);Scratch scratch;GPUFlatGraphStructure out;std::string error;
        for(int repeat=0;repeat<20;++repeat){ASSERT_TRUE(capture.inspectFlatGraphStructure(scope,scratch.bytes,out,&error));EXPECT_EQ(out.nodes,3u);
            graph.nodeCount=1;graph.edgeCount=0;ASSERT_TRUE(capture.inspectFlatGraphStructure(scope,scratch.bytes,out,&error));EXPECT_EQ(out.nodes,1u);EXPECT_EQ(out.edges,0u);
            graph.nodeCount=0;EXPECT_FALSE(capture.inspectFlatGraphStructure(scope,scratch.bytes,out,&error));EXPECT_EQ(out,GPUFlatGraphStructure{});graph.nodeCount=3;graph.edgeCount=2;}});
}
TEST(GPUFlatGraphInspection, MissingMisalignedAndShortScratchRejectBeforeNativeQueries){
    each([]<class Capture>{for(unsigned fault=0;fault<3;++fault){Graph graph;Capture capture(&graph);Scratch scratch;GPUFlatGraphStructure out;std::string error;
        auto bytes=std::span<std::byte>(scratch.bytes);if(fault==0)bytes={};if(fault==1)bytes=bytes.subspan(1);if(fault==2)bytes=bytes.first(GPUFlatGraphScratchLayout::fromLimits(scope).bytes-1);
        EXPECT_FALSE(capture.inspectFlatGraphStructure(scope,bytes,out,&error));EXPECT_EQ(graph.nodesQueries,0);EXPECT_EQ(graph.edgeQueries,0);EXPECT_EQ(graph.kindQueries,0);}});
}
TEST(GPUFlatGraphInspection, InvalidOwnerAndScopeFailClosed){
    each([]<class Capture>{Graph graph;Capture capture(&graph);Scratch scratch;GPUFlatGraphStructure out;std::string error;
        graph.active=false;EXPECT_FALSE(capture.inspectFlatGraphStructure(scope,scratch.bytes,out,&error));EXPECT_EQ(graph.nodesQueries,0);
        graph.active=true;for(const auto bad:{GPUFlatGraphLimits{},GPUFlatGraphLimits{3,2,1u<<12}}){EXPECT_FALSE(capture.inspectFlatGraphStructure(bad,scratch.bytes,out,&error));EXPECT_EQ(graph.nodesQueries,0);}});
}
TEST(GPUFlatGraphInspection, NativeQueryFailureAndChangingInventoryReject){
    each([]<class Capture>{for(unsigned fault=0;fault<6;++fault){Graph graph;Capture capture(&graph);Scratch scratch;GPUFlatGraphStructure out;std::string error;
        if(fault==0)graph.nodeFailure=true;if(fault==1)graph.edgeFailure=true;if(fault==2)graph.kindFailure=true;if(fault==3)graph.changedNodes=true;if(fault==4)graph.changedEdges=true;if(fault==5)graph.nondefaultEdges=true;
        EXPECT_FALSE(capture.inspectFlatGraphStructure(scope,scratch.bytes,out,&error));EXPECT_FALSE(error.empty());EXPECT_EQ(out,GPUFlatGraphStructure{});}});
}
TEST(GPUFlatGraphInspection, NodeAndEdgeExtentsCannotGrowPastOriginalDeclaration){
    each([]<class Capture>{for(bool nodes:{false,true}){Graph graph;Capture capture(&graph);Scratch scratch;GPUFlatGraphStructure out;std::string error;auto bound=scope;
        if(nodes)bound.maximumNodes=2;else bound.maximumEdges=1;EXPECT_FALSE(capture.inspectFlatGraphStructure(bound,scratch.bytes,out,&error));EXPECT_EQ(graph.nodesQueries,nodes?1:2);EXPECT_EQ(out,GPUFlatGraphStructure{});}});
}
TEST(GPUFlatGraphInspection, DuplicateNullAndUnsupportedNodesReject){
    each([]<class Capture>{for(unsigned fault=0;fault<4;++fault){Graph graph;Capture capture(&graph);Scratch scratch;GPUFlatGraphStructure out;std::string error;
        if(fault==0)graph.nodes[1]=graph.nodes[0];if(fault==1)graph.nodes[1]=nullptr;if(fault==2)graph.kinds[1]=4;if(fault==3)graph.kinds[1]=1;
        EXPECT_FALSE(capture.inspectFlatGraphStructure(scope,scratch.bytes,out,&error));EXPECT_EQ(out,GPUFlatGraphStructure{});}});
}
TEST(GPUFlatGraphInspection, ForeignSelfDuplicateAndCyclicEdgesReject){
    each([]<class Capture>{for(unsigned fault=0;fault<4;++fault){Graph graph;Capture capture(&graph);Scratch scratch;GPUFlatGraphStructure out;std::string error;
        if(fault==0)graph.to[1]=reinterpret_cast<void*>(0xdead);if(fault==1)graph.to[1]=graph.from[1];if(fault==2){graph.from[1]=graph.from[0];graph.to[1]=graph.to[0];}if(fault==3)graph.to[1]=graph.nodes[0];
        EXPECT_FALSE(capture.inspectFlatGraphStructure(scope,scratch.bytes,out,&error));EXPECT_EQ(out,GPUFlatGraphStructure{});}});
}
TEST(GPUFlatGraphInspection, CanonicalKindsRequireTheExactDeclaredMask){
    each([]<class Capture>{for(const auto native:{0,1,2,5,6,7}){Graph graph;graph.nodeCount=1;graph.edgeCount=0;graph.kinds[0]=native;Capture capture(&graph);Scratch scratch;GPUFlatGraphStructure out;std::string error;
        auto bound=scope;bound.allowedKinds=(1u<<static_cast<unsigned>(GPUFlatGraphNodeKind::Count))-1;EXPECT_TRUE(capture.inspectFlatGraphStructure(bound,scratch.bytes,out,&error));EXPECT_EQ(out.nodes,1u);}});
}
TEST(GPUFlatGraphInspection, CompleteDAGDistinguishesSingleStreamChainFromFork){
    each([]<class Capture>{Graph graph;Capture capture(&graph);Scratch scratch;GPUFlatGraphStructure out;std::string error;
        graph.from[1]=graph.nodes[0];ASSERT_TRUE(capture.inspectFlatGraphStructure(scope,scratch.bytes,out,&error));EXPECT_FALSE(out.linearChain);
        graph.from[1]=graph.nodes[1];ASSERT_TRUE(capture.inspectFlatGraphStructure(scope,scratch.bytes,out,&error));EXPECT_TRUE(out.linearChain);});
}

}
