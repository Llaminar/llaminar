/**
 * @file QwenSparseIndexerPolicyPerfROCmTest.h
 * @brief Explicit Installed/bounded ROCm indexer-only repeated capture through 64K context.
 *
 * This private include follows QwenSparseDeviceTest.h in the ROCm QSA test TU.
 * Fixture storage owns setup, synthetic inputs, copies and terminal collection.
 * The timed graph contains only repeated complete production indexer calls:
 * Grouped1..16: fused preparation -> scorePools -> explicitly selected policy.
 * Prefill: separate pool/query preparation -> scorePools -> selected policy. Capture
 * validation authenticates every native function, launch shape and dependency.
 * Independent score sorting checks actual membership outside timing; it is not
 * a broad indexer arithmetic, stage, model, memory-admission or 64K-prefill proof.
 */
#pragma once

#include "QwenSparseSelectionCases.h"
#include "kernels/qwen4exp/QwenSparseBoundedSelection.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <cxxabi.h>
#include <fstream>
#include <filesystem>
#include <set>
#include <sstream>
#include <thread>
#include <sched.h>

namespace llaminar2::qwen4exp::ROCmQwenSparseAttention
{
namespace indexer_policy_perf108
{
/** @brief Reject a malformed explicit fixture request without another workload. */
inline void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

/**
 * @brief Parse one bounded decimal fixture selector before any GPU allocation.
 * @param name Environment variable naming the explicit selector.
 * @return Complete positive decimal value.
 * @throws std::invalid_argument For missing, signed, partial or oversized input.
 */
inline int positiveSelector(const char* name)
{
    const char* raw = std::getenv(name);
    if (!raw || !*raw) throw std::invalid_argument(std::string("Missing selector: ") + name);
    const std::string text(raw);
    if (text.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument(std::string("Malformed decimal selector: ") + name);
    std::size_t used = 0;
    const auto value = std::stoull(text, &used);
    if (used != text.size() || value == 0 || value > std::uint64_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument(std::string("Out-of-range selector: ") + name);
    return int(value);
}

/** @brief Closed explicit candidate tuple; it never chooses another policy on failure or cost. */
static_assert(sizeof(bounded_selection::Storage<1024,1024>)==42224);
struct SelectionPolicy final
{
    QwenSparsePoolSelection value;
    const char* name;
    int block, slack;
    /** @return Exact required environment policy, rejected before any native allocation. */
    static SelectionPolicy read()
    {
        const char* raw=std::getenv("QFN_INDEXER108_POLICY");
        require(raw && *raw,"Explicit indexer selector policy is required");
        const std::string name(raw);
        if(name=="Installed")return {QwenSparsePoolSelection::Installed,"Installed",0,0};
        if(name=="Bounded256_512")return {QwenSparsePoolSelection::Bounded256_512,"Bounded256_512",256,512};
        if(name=="Bounded512_512")return {QwenSparsePoolSelection::Bounded512_512,"Bounded512_512",512,512};
        if(name=="Bounded1024_1024")return {QwenSparsePoolSelection::Bounded1024_1024,"Bounded1024_1024",1024,1024};
        throw std::invalid_argument("Unknown explicit indexer selector policy");
    }
    /** @return Bounded lexical storage bytes only, not a reservation or physical traffic counter. */
    std::size_t sourceStorageBytes() const
    {
        switch(value){
        case QwenSparsePoolSelection::Installed:return 0;
        case QwenSparsePoolSelection::Bounded256_512:return sizeof(bounded_selection::Storage<256,512>);
        case QwenSparsePoolSelection::Bounded512_512:return sizeof(bounded_selection::Storage<512,512>);
        case QwenSparsePoolSelection::Bounded1024_1024:return sizeof(bounded_selection::Storage<1024,1024>);}
        throw std::invalid_argument("Unknown explicit indexer selector policy");
    }
};
/** @brief Preserve comma-delimited evidence while retaining the exact demangled template name. */
inline std::string markerFunction(const std::string& name)
{
    std::string encoded;
    for(char value:name){if(value=='%')encoded+="%25";else if(value==',')encoded+="%2C";else encoded+=value;}
    return encoded;
}

/** @return Explicit fixture demand; an omitted selector retains historical AllLive. */
inline QwenSparseQueryDemand readDemand()
{
    const char* value = std::getenv("QFN_INDEXER108_DEMAND");
    if (!value || std::string(value) == "AllLive") return QwenSparseQueryDemand::AllLive;
    if (std::string(value) == "RankingOnly") return QwenSparseQueryDemand::RankingOnly;
    throw std::invalid_argument("Unknown explicit indexer query demand");
}

/** @brief Immutable source-owned workload/capture bounds, not a physical memory ledger. */
struct Plan final
{
    static constexpr int MaxRepeats = 256;
    static constexpr std::uint64_t MaxRowVisits = 8192;
    static constexpr std::uint64_t MaxScoredPoolVisits = std::uint64_t{1} << 26;
    static constexpr std::uint64_t TargetScoredPoolVisits = std::uint64_t{1} << 25;
    static constexpr int Warmups = 5;
    static constexpr int Samples = 12;
    int rows, context, prefix, capacity, blocks, repeats;
    std::uint64_t causal_pools = 0, scored_pools = 0, selected_tokens = 0;
    SelectionPolicy policy;
    QwenSparseQueryDemand demand = readDemand();
    const char* demandName() const { return demand == QwenSparseQueryDemand::AllLive ? "AllLive" : "RankingOnly"; }
    bool wide;
    /** @brief Validate exact geometry and bound graph repeats by row and score work. */
    Plan(int m, int c,SelectionPolicy selected) : rows(m), context(c), prefix(0), capacity(c), blocks(0), repeats(1),
        policy(selected),wide(m<=attention::kMaxGroupedVerifierAttentionRows)
    {
        require((m>=1 && m<=attention::kMaxGroupedVerifierAttentionRows) || m==512 || m==1024 || m==2048,
            "Indexer timing admits grouped decode M1..16 and prefill M512/1024/2048");
        require(c>=m && c<=65536, "Indexer timing needs M <= context <= 65536");
        validateQwenSparsePoolSelection(policy.value,m);
        prefix=c-m; blocks=(c+3)/4;
        for (int row=0; row<rows; ++row)
        {
            const auto visible=std::uint64_t(prefix+row+1), complete=visible/4;
            causal_pools += complete;
            if (complete>512) scored_pools += complete;
            selected_tokens += std::min<std::uint64_t>(complete,512)*4+visible%4;
        }
        const auto byRows=MaxRowVisits/std::uint64_t(rows);
        const auto byScores=scored_pools ? MaxScoredPoolVisits/scored_pools : std::uint64_t(MaxRepeats);
        const auto maximum=std::min({std::uint64_t(MaxRepeats),byRows,byScores});
        require(maximum>=1, "Indexer tuple exceeds the source-owned capture work budget");
        const auto target=scored_pools ? std::max<std::uint64_t>(1,TargetScoredPoolVisits/scored_pools) : maximum;
        repeats=int(std::min(maximum,target));
        if (std::getenv("QFN_INDEXER108_REPEATS")) repeats=positiveSelector("QFN_INDEXER108_REPEATS");
        require(repeats>=1 && std::uint64_t(repeats)<=maximum,
            "Requested indexer repeats exceed the graph-node/row/score work budget");
    }
    /** @return Exact selector suffix, including immutable template arguments for candidates. */
    std::string selector() const
    {
        if(policy.value==QwenSparsePoolSelection::Installed)return wide?"selectPoolsWide":"selectPools";
        return "selectPoolsBounded<"+std::to_string(policy.block)+", "+std::to_string(policy.slack)+">";
    }
    /** @return Exact immutable preparation schedule and complete-role inventory. */
    bool fusedPreparation() const { return wide; }
    /** @return Exact total native kernels per captured indexer transaction. */
    int roleCount() const { return fusedPreparation() ? 3 : 4; }
    /** @return Role carrying the original replay-backed query row-work argument. */
    int queryRole() const { return fusedPreparation() ? 0 : 1; }
    /** @return Original score role, immediately before the selector. */
    int scoreRole() const { return roleCount()-2; }
    /** @return Stable marker spelling for the immutable geometry-selected program. */
    const char* preparationName() const { return fusedPreparation() ? "fused_grouped" : "separate_prefill"; }
    /** @return Exact native grid for one source-confirmed indexer role. */
    dim3 grid(int role) const
    {
        require(role>=0 && role<roleCount(),"Unknown geometry-selected indexer role");
        if (role < scoreRole())
        {
            if (fusedPreparation()) return dim3((rows+6)/4+rows*4,1,1);
            return dim3(role==0 ? (rows+6)/4 : rows*4,1,1);
        }
        if (role==scoreRole()) return dim3((blocks+7)/8,rows,1);
        return policy.value!=QwenSparsePoolSelection::Installed || wide ? dim3(rows,1,1) : dim3((rows+7)/8,1,1);
    }
    /** @return Exact original preparation/scoring thread geometry. */
    dim3 block(int role) const { return dim3(role<scoreRole() ? 128 : 256,1,1); }
    /** @return Exact demangled native name without its original parameter list. */
    std::string function(int role) const
    {
        require(role>=0 && role<roleCount(),"Unknown geometry-selected indexer role");
        const std::string ranked=demand==QwenSparseQueryDemand::AllLive ? "false" : "true";
        std::string name;
        if (role==roleCount()-1) name=selector();
        else if (role==scoreRole()) name="scorePools";
        else if (fusedPreparation()) name="preparePoolsAndQueries<"+ranked+">";
        else name=role==0 ? "preparePools" : "prepareQueries<"+ranked+">";
        const std::string returned=role==queryRole() || (role==roleCount()-1 && policy.block>0) ? "void " : "";
        return returned+"llaminar2::qwen4exp::rocm_sparse_detail::"+name;
    }

};

/** @brief One persistent complete-indexer graph with exact ordered native inventory. */
struct Capture final
{
    hipStream_t stream;
    const Plan& plan;
    const QwenSparseIndexer params;
    hipGraph_t graph=nullptr;
    hipGraphExec_t executable=nullptr;
    /** @brief Borrow exact fixture storage and freeze production indexer policy before capture. */
    Capture(Fixture& f, const Plan& p) : stream(f.stream), plan(p), params([&] {
        auto a=f.indexer;
        a.representation=QwenSparseScalar::FP32;
        a.rotary_source=QwenSparseRotarySource::AnalyticTheta;
        a.rope_theta=10000000.0f;
        a.inverse_frequencies=makeQwenSparsePowerFrequencies(a.rope_theta,64);
        a.cosine=nullptr; a.sine=nullptr;
        a.pool_selection=plan.policy.value;
        a.query_demand=plan.demand;
        a.selection_observations=nullptr;
        return a;
    }())
    {
        require(params.row_capacity==plan.rows && params.token_capacity==plan.capacity && params.block_capacity==plan.blocks,
            "Fixture and exact indexer capture plan differ");
        require(!params.transaction_prior && !params.work_counters.bound(),
            "Indexer timing excludes retained-acceptance and terminal-counter work");
    }
    Capture(const Capture&)=delete;
    /** @brief Join only at fixture retirement, then release retained graph objects. */
    ~Capture()
    {
        if (hipStreamSynchronize(stream)!=hipSuccess) std::terminate();
        if (executable) (void)hipGraphExecDestroy(executable);
        if (graph) (void)hipGraphDestroy(graph);
    }
    /** @brief Capture only the explicitly requested indexer calls and authenticate every native node. */
    void record()
    {
        require(!graph && !executable,"Indexer capture may only be recorded once");
        check(hipStreamBeginCapture(stream,hipStreamCaptureModeThreadLocal));
        try
        {
            for (int repeat=0; repeat<plan.repeats; ++repeat) rocmQwenSparseIndexer(params,stream);
        }
        catch (...)
        {
            hipGraph_t abandoned=nullptr;
            (void)hipStreamEndCapture(stream,&abandoned);
            if (abandoned) (void)hipGraphDestroy(abandoned);
            throw;
        }
        check(hipStreamEndCapture(stream,&graph));
        std::size_t count=0;
        check(hipGraphGetNodes(graph,nullptr,&count));
        require(count==std::size_t(plan.repeats)*plan.roleCount(),"Indexer graph must contain the exact geometry-selected kernels per repeat");
        std::vector<hipGraphNode_t> nodes(count);
        check(hipGraphGetNodes(graph,nodes.data(),&count));
        std::vector<int> roles(count,-1), next(count,-1);
        std::vector<void*> functions(std::size_t(plan.roleCount()));
        std::vector<std::string> actualNames(std::size_t(plan.roleCount()));
        std::vector<dim3> actualGrids(std::size_t(plan.roleCount())), actualBlocks(std::size_t(plan.roleCount()));
        std::vector<int> roleCounts(std::size_t(plan.roleCount()));
        std::size_t first=count;
        for (std::size_t index=0; index<count; ++index)
        {
            hipGraphNodeType type;
            check(hipGraphNodeGetType(nodes[index],&type));
            require(type==hipGraphNodeTypeKernel,"Indexer-only capture contains a copy/alloc/event/other node");
            hipKernelNodeParams native{};
            check(hipGraphKernelNodeGetParams(nodes[index],&native));
            const char* raw=hipKernelNameRefByPtr(native.func,stream);
            require(raw,"Captured indexer kernel has no authenticated name");
            int status=0;
            std::unique_ptr<char,decltype(&std::free)> demangled(abi::__cxa_demangle(raw,nullptr,nullptr,&status),&std::free);
            require(status==0 && demangled,"Cannot demangle the pinned HIP indexer kernel identity");
            const std::string actual(demangled.get());
            int role=-1;
            for (int candidate=0; candidate<plan.roleCount(); ++candidate)
                if (actual==plan.function(candidate)+(candidate==plan.queryRole() ?
                    "(llaminar2::qwen4exp::QwenSparseIndexer, llaminar2::ProjectionRowWork)" :
                    "(llaminar2::qwen4exp::QwenSparseIndexer)"))
                { require(role==-1,"Ambiguous native indexer role"); role=candidate; }
            require(role>=0,"Captured native function is outside the exact geometry-selected indexer inventory");
            if(role==plan.queryRole()) {
                require(native.kernelParams && native.kernelParams[0] && native.kernelParams[1],
                    "AllLive query capture lacks original native arguments");
                const auto &query=*static_cast<const QwenSparseIndexer *>(native.kernelParams[0]);
                const auto &work=*static_cast<const ProjectionRowWork *>(native.kernelParams[1]);
                require(params.query_demand==plan.demand &&
                    query.query_demand==plan.demand && query.replay==params.replay &&
                    query.row_capacity==params.row_capacity && query.token_capacity==params.token_capacity &&
                    query.query==params.query && query.prepared_query==params.prepared_query &&
                    work.captureIdentity()==qwenSparseIndexerQueryWork(params).repeatEachRow(4).captureIdentity(),
                    "AllLive query capture changed its original replay/prefix work identity");
            }
            const auto expectedGrid=plan.grid(role), expectedBlock=plan.block(role);
            require(native.gridDim.x==expectedGrid.x && native.gridDim.y==expectedGrid.y && native.gridDim.z==expectedGrid.z &&
                native.blockDim.x==expectedBlock.x && native.blockDim.y==expectedBlock.y && native.blockDim.z==expectedBlock.z &&
                native.sharedMemBytes==0,"Captured native indexer launch geometry differs from its source-owned plan");
            auto& function=functions[std::size_t(role)];
            if (!function)
            {
                function=native.func;
                actualNames[std::size_t(role)]=actual;
                actualGrids[std::size_t(role)]=native.gridDim;
                actualBlocks[std::size_t(role)]=native.blockDim;
            }
            require(function==native.func,"Native indexer function changes between captured repeats");
            roles[index]=role; ++roleCounts[std::size_t(role)];
            std::size_t dependencies=0;
            check(hipGraphNodeGetDependencies(nodes[index],nullptr,&dependencies));
            require(dependencies<=1,"Captured indexer chain has a multi-parent node");
            if (!dependencies) { require(first==count,"Captured indexer has multiple roots"); first=index; }
            else
            {
                hipGraphNode_t parent=nullptr;
                check(hipGraphNodeGetDependencies(nodes[index],&parent,&dependencies));
                const auto found=std::find(nodes.begin(),nodes.end(),parent);
                require(found!=nodes.end(),"Captured indexer names a foreign dependency");
                const auto previous=std::size_t(found-nodes.begin());
                require(next[previous]==-1,"Captured indexer dependency chain branches");
                next[previous]=int(index);
            }
        }
        for (int role=0; role<plan.roleCount(); ++role)
        {
            require(roleCounts[std::size_t(role)]==plan.repeats,"Captured indexer role count differs");
            for (int other=0; other<role; ++other)
                require(functions[std::size_t(role)]!=functions[std::size_t(other)],"Indexer roles alias one native function");
        }
        require(first<count,"Captured indexer has no unique root");
        std::vector<bool> visited(count,false);
        int cursor=int(first);
        for (std::size_t step=0; step<count; ++step)
        {
            require(cursor>=0 && !visited[std::size_t(cursor)],"Captured indexer chain is incomplete/cyclic");
            const auto index=std::size_t(cursor);
            visited[index]=true;
            require(roles[index]==int(step%std::size_t(plan.roleCount())),"Captured native indexer operations are out of order");
            cursor=next[index];
        }
        require(cursor==-1,"Captured indexer chain has excess operations");
        check(hipGraphInstantiate(&executable,graph,nullptr,nullptr,0));
        for (int role=0; role<plan.roleCount(); ++role)
        {
            const auto grid=actualGrids[std::size_t(role)], block=actualBlocks[std::size_t(role)];
            hipFuncAttributes attributes{};check(hipFuncGetAttributes(&attributes,functions[std::size_t(role)]));
            require(attributes.localSizeBytes==0,"Captured indexer kernel exposes forbidden private storage");
            if(role==plan.roleCount()-1)require(attributes.sharedSizeBytes>=plan.policy.sourceStorageBytes(),
                "Captured bounded selector lacks its actual lexical shared bank");
            std::cout<<"INDEXER108_CAPTURE,version=5,policy="<<plan.policy.name<<",policy_code="<<int(plan.policy.value)
                <<",bounded_block="<<plan.policy.block<<",bounded_slack="<<plan.policy.slack
                <<",selection_observations=off,preparation_policy="<<plan.preparationName()
                <<",role="<<role<<",function_encoding=percent-comma,function="<<markerFunction(actualNames[std::size_t(role)])
                <<",native_function="<<functions[std::size_t(role)]<<",grid="<<grid.x<<':'<<grid.y<<':'<<grid.z
                <<",block="<<block.x<<':'<<block.y<<':'<<block.z<<",occurrences="<<roleCounts[std::size_t(role)]
                <<",ordered_chain_nodes="<<count<<",native_attribute_num_regs="<<attributes.numRegs
                <<",native_attribute_shared_bytes="<<attributes.sharedSizeBytes<<",native_attribute_local_bytes="<<attributes.localSizeBytes
                <<",source_selector_storage_bytes="<<(role==plan.roleCount()-1?plan.policy.sourceStorageBytes():0)<<'\n';
        }
    }
    /** @brief Replay exactly the retained graph on its original non-default stream. */
    void launch() { require(executable,"Indexer graph was not recorded"); check(hipGraphLaunch(executable,stream)); }
};

/**
 * @brief Count effective physical host cores for outside-timing numeric-sort work.
 * @return Distinct package/core pairs in this Linux process's actual CPU affinity.
 * @throws std::runtime_error If explicit physical-core accounting is unavailable.
 */
inline unsigned physicalWorkers()
{
    cpu_set_t mask;
    CPU_ZERO(&mask);
    require(sched_getaffinity(0,sizeof(mask),&mask)==0,"Cannot observe indexer-oracle CPU affinity");
    std::set<std::pair<int,int>> cores;
    for (int cpu=0; cpu<CPU_SETSIZE; ++cpu) if (CPU_ISSET(cpu,&mask))
    {
        const auto topology="/sys/devices/system/cpu/cpu"+std::to_string(cpu)+"/topology/";
        int package=-1,core=-1;
        std::ifstream packageFile(topology+"physical_package_id"), coreFile(topology+"core_id");
        require(bool(packageFile>>package) && bool(coreFile>>core) && package>=0 && core>=0,
            "Cannot authenticate physical host cores for indexer oracle work");
        cores.emplace(package,core);
    }
    require(!cores.empty(),"Indexer oracle has no admitted physical host worker");
    return unsigned(cores.size());
}

/** @brief Membership checks projected from actual outputs, without a fabricated arithmetic reference. */
struct Membership final
{
    std::uint64_t selected_tokens=0;
    int largest_selected=-1;
    std::vector<std::int32_t> live_ids; ///< Only consumed selected IDs; no spare-capacity payload.
};

/**
 * @brief Validate every produced score-row selection with the independent numeric sort oracle.
 * @param f Fixture supplying actual terminal scores, selected IDs and native counts.
 * @param plan Exact immutable causal geometry of the measured transaction.
 * @return Actual verified membership metadata for the emitted evidence marker.
 * @details Downloads, finite checks and physical-core-parallel sorting happen
 * only after native completion, outside every timed interval and capture.
 */
inline Membership verifyMembership(Fixture& f, const Plan& plan)
{
    const auto counts=f.count.download(f.stream);
    std::vector<std::size_t> selectedOffsets(std::size_t(plan.rows)+1), scoreOffsets(std::size_t(plan.rows)+1);
    for (int row=0; row<plan.rows; ++row)
    {
        const int visible=plan.prefix+row+1, blocks=visible/4;
        require(counts[std::size_t(row)]==std::min(blocks,512)*4+visible%4,
            "Native indexer selected_count differs before terminal payload collection");
        selectedOffsets[std::size_t(row)+1]=selectedOffsets[std::size_t(row)]+std::size_t(counts[std::size_t(row)]);
        scoreOffsets[std::size_t(row)+1]=scoreOffsets[std::size_t(row)]+std::size_t(blocks>512 ? blocks : 0);
    }
    require(selectedOffsets.back()==plan.selected_tokens && scoreOffsets.back()==plan.scored_pools,
        "Terminal live indexer extents differ from exact causal geometry");
    std::vector<std::int32_t> selected(selectedOffsets.back());
    std::vector<float> scores(scoreOffsets.back());
    for (int row=0; row<plan.rows; ++row)
    {
        const auto selectedOffset=selectedOffsets[std::size_t(row)], scoreOffset=scoreOffsets[std::size_t(row)];
        const int blocks=(plan.prefix+row+1)/4;
        check(hipMemcpyAsync(selected.data()+selectedOffset, f.selected.data+std::size_t(row)*QwenSparseSelected,
            std::size_t(counts[std::size_t(row)])*sizeof(std::int32_t),hipMemcpyDeviceToHost,f.stream));
        if (blocks>512)
            check(hipMemcpyAsync(scores.data()+scoreOffset, f.scores.data+std::size_t(row)*plan.blocks,
                std::size_t(blocks)*sizeof(float),hipMemcpyDeviceToHost,f.stream));
    }
    check(hipStreamSynchronize(f.stream));
    const auto prepared=f.prepared.download(f.stream);
    for (float value:prepared) require(std::isfinite(value),"Indexer produced non-finite prepared Q");
    std::vector<std::string> errors(std::size_t(plan.rows));
    std::atomic<int> next{0};
    const unsigned workers=std::min<unsigned>(unsigned(plan.rows),physicalWorkers());
    std::vector<std::jthread> threads;
    threads.reserve(workers);
    for (unsigned worker=0; worker<workers; ++worker) threads.emplace_back([&] {
        for (int row=next.fetch_add(1); row<plan.rows; row=next.fetch_add(1))
        {
            const int visible=plan.prefix+row+1, blocks=visible/4;
            const auto* score=blocks>512 ? scores.data()+scoreOffsets[std::size_t(row)] : nullptr;
            const auto* actual=selected.data()+selectedOffsets[std::size_t(row)];
            const int expectedCount=std::min(blocks,512)*4+visible%4;
            if (counts[std::size_t(row)]!=expectedCount) { errors[std::size_t(row)]="selected_count"; continue; }
            bool finite=true;
            if (blocks>512) for (int block=0; block<blocks; ++block)
                finite &= std::isfinite(score[block]) && score[block]>=0.0f;
            if (!finite) { errors[std::size_t(row)]="non-finite/negative produced score"; continue; }
            const auto expected=selection_test::oracle(score,visible);
            if (expected.size()!=std::size_t(expectedCount) ||
                !std::equal(expected.begin(),expected.end(),actual))
            { errors[std::size_t(row)]="numeric-score-sort membership"; continue; }
            for (int index=0; index<expectedCount; ++index)
                if (actual[index]<0 || actual[index]>=visible || (index && actual[index]<=actual[index-1]))
                    errors[std::size_t(row)]="noncausal/duplicate/unordered membership";
        }
    });
    for (auto& thread:threads) thread.join();
    Membership metadata;
    for (int row=0; row<plan.rows; ++row)
    {
        if (!errors[std::size_t(row)].empty())
            throw std::runtime_error("Indexer row "+std::to_string(row)+" failed "+errors[std::size_t(row)]);
        metadata.selected_tokens+=std::uint64_t(counts[std::size_t(row)]);
        metadata.largest_selected=std::max(metadata.largest_selected,
            selected[selectedOffsets[std::size_t(row)+1]-1]);
    }
    require(metadata.selected_tokens==plan.selected_tokens,"Verified membership total differs from causal geometry");
    metadata.live_ids=std::move(selected);
    return metadata;
}

/** @brief Export bounded actual terminal banks only after final native join and checks.
 * @details Existing owners/staging survive each download join. No replay is submitted.
 * The directory must already exist; each file must be absent. This is component
 * evidence, not a public runtime dump interface or a model-accuracy receipt.
 */
inline void exportTerminal(Fixture& f, const Plan& plan)
{
    const char* raw = std::getenv("QFN_INDEXER108_EXPORT_DIR");
    if (!raw) return;
    const std::filesystem::path directory(raw);
    require(directory.is_absolute() && std::filesystem::is_directory(directory),
        "Indexer export requires an existing absolute directory");
    const std::uint64_t bytes = (f.pools.size + f.prepared.size + f.scores.size) * sizeof(float)
        + (f.selected.size + f.count.size) * sizeof(std::int32_t);
    require(bytes <= (std::uint64_t{64} << 20), "Indexer terminal export exceeds 64 MiB bound");
    const auto save = [&](const char* name, const auto& values) {
        const auto path = directory / name;
        require(!std::filesystem::exists(path), "Indexer export refuses to overwrite evidence");
        std::ofstream out(path,std::ios::binary | std::ios::out);
        out.write(reinterpret_cast<const char*>(values.data()),
            std::streamsize(values.size()*sizeof(values[0])));
        out.close();
        require(bool(out), "Indexer terminal export write failed");
    };
    save("pools.f32",f.pools.download(f.stream));
    const auto prepared = f.prepared.download(f.stream);
    // RankingOnly must retain the exact initial poison below its device threshold.
    for (int row=0; row<plan.rows; ++row) {
        if (plan.demand != QwenSparseQueryDemand::RankingOnly || plan.prefix+row+1 >= 2052) continue;
        for (int col=0; col<512; ++col)
            require(detail::floatBits(prepared[std::size_t(row)*512+col]) == 0xa5a5a5a5u,
                "RankingOnly skipped query changed its diagnostic poison");
    }
    save("prepared.f32",prepared);
    save("scores.f32",f.scores.download(f.stream));
    save("selected.i32",f.selected.download(f.stream));
    save("counts.i32",f.count.download(f.stream));
    const auto metadata = directory / "geometry.txt";
    require(!std::filesystem::exists(metadata), "Indexer export metadata already exists");
    std::ofstream out(metadata);
    out << "schema=1\nM=" << plan.rows << "\ncontext=" << plan.context
        << "\nprefix=" << plan.prefix << "\ncapacity=" << plan.capacity
        << "\nblocks=" << plan.blocks << "\nquery_demand=" << plan.demandName()
        << "\npolicy=" << plan.policy.name << "\nrepresentation=FP32\npoison_bits=a5a5a5a5\n";
    out.close();
    require(bool(out), "Indexer export metadata failed");
    std::cout << "INDEXER108_EXPORT,version=1,M=" << plan.rows << ",context=" << plan.context
        << ",query_demand=" << plan.demandName() << ",bytes=" << bytes << ",passed=1\n";
}

/** @brief Fixture-only native events around one graph, normalized once by complete transactions. */
struct Events final
{
    hipStream_t stream;
    hipEvent_t begin=nullptr,end=nullptr;
    /** @brief Create persistent event handles outside capture/timing. */
    explicit Events(hipStream_t s):stream(s)
    {
        check(hipEventCreate(&begin));
        try { check(hipEventCreate(&end)); }
        catch (...) { (void)hipEventDestroy(begin); begin=nullptr; throw; }
    }
    /** @brief Retire terminal event objects after all samples were collected. */
    ~Events() { if (end) (void)hipEventDestroy(end); if (begin) (void)hipEventDestroy(begin); }
    /** @brief Return total native graph microseconds; caller normalizes only once. */
    float sample(Capture& capture)
    {
        check(hipEventRecord(begin,stream)); capture.launch(); check(hipEventRecord(end,stream));
        check(hipEventSynchronize(end));
        float milliseconds=0;
        check(hipEventElapsedTime(&milliseconds,begin,end));
        return milliseconds*1000.0f;
    }
};
} // namespace indexer_policy_perf108

/**
 * @test Measure only the explicit FP32 analytic indexer policy through 64K.
 * @details This is an explicitly selected disabled Release component test. It
 * excludes Q/K projection GEMMs, attention, commit, model inference and indexer
 * CPU/reference work from capture/timing. C is the final horizon of one last
 * window [C-M,C), not a claim to have prefetched an entire 64K prompt.
 */
TEST(ROCmQwenSparseAttention,DISABLED_Indexer108PoliciesCapturedRepeatEvents)
{
    using namespace indexer_policy_perf108;
    const Plan plan(positiveSelector("QFN_INDEXER108_M"),positiveSelector("QFN_INDEXER108_CONTEXT"),SelectionPolicy::read());
    Fixture fixture(plan.rows,plan.prefix,plan.capacity,QwenSparseScalar::FP32);
    // Diagnostic initialization is outside capture and every event interval.
    check(hipMemsetAsync(fixture.prepared.data,0xa5,fixture.prepared.size*sizeof(float),fixture.stream));
    check(hipMemsetAsync(fixture.scores.data,0xa5,fixture.scores.size*sizeof(float),fixture.stream));
    check(hipMemsetAsync(fixture.selected.data,0xa5,fixture.selected.size*sizeof(std::int32_t),fixture.stream));
    check(hipMemsetAsync(fixture.count.data,0xa5,fixture.count.size*sizeof(std::int32_t),fixture.stream));
    Capture capture(fixture,plan);
    capture.record();
    capture.launch();
    const auto initial=verifyMembership(fixture,plan);
    const auto publishMembership=[&](const char* phase,const Membership& membership,const char* retained){
        std::cout<<"INDEXER108_MEMBERSHIP,version=2,phase="<<phase<<",query_demand="<<plan.demandName()<<",policy="<<plan.policy.name
            <<",M="<<plan.rows<<",prefix="<<plan.prefix<<",context="<<plan.context<<",rows="<<plan.rows
            <<",selected_tokens="<<membership.selected_tokens<<",largest_selected="<<membership.largest_selected
            <<",oracle=numeric-sort-produced-scores,tie_order=lower-original-pool-id,retained_equal="<<retained<<'\n';};
    publishMembership("initial",initial,"not-applicable");
    Events events(fixture.stream);
    for (int warmup=0; warmup<Plan::Warmups; ++warmup) (void)events.sample(capture);
    for (int sample=0; sample<Plan::Samples; ++sample)
    {
        const float graphUs=events.sample(capture);
        require(std::isfinite(graphUs) && graphUs>0.0f,"Native indexer event interval is invalid");
        std::cout<<"INDEXER108_EVENT,version=5,sample="<<sample
            <<",M="<<plan.rows<<",prefix="<<plan.prefix<<",context="<<plan.context<<",capacity="<<plan.capacity
            <<",blocks="<<plan.blocks<<",position_origin=17,repeats="<<plan.repeats<<",nodes="<<plan.repeats*plan.roleCount()
            <<",query_demand="<<plan.demandName()<<",policy="<<plan.policy.name<<",policy_code="<<int(plan.policy.value)
            <<",bounded_block="<<plan.policy.block<<",bounded_slack="<<plan.policy.slack
                <<",selection_observations=off,preparation_policy="<<plan.preparationName()
            <<",representation=FP32,rotary=AnalyticTheta,theta=10000000,selector="<<markerFunction(plan.selector())
            <<",source_causal_complete_pools="<<plan.causal_pools<<",source_scored_pools="<<plan.scored_pools
            <<",source_score_fmas="<<plan.scored_pools*512<<",selected_tokens="<<initial.selected_tokens
            <<",largest_selected="<<initial.largest_selected<<",oracle=numeric-sort-produced-scores,oracle_rows="<<plan.rows
            <<",graph_us="<<graphUs<<",transaction_us="<<graphUs/float(plan.repeats)
            <<",live_row_us="<<graphUs/float(plan.repeats)/float(plan.rows)<<'\n';
    }
    const auto final=verifyMembership(fixture,plan);
    EXPECT_EQ(final.selected_tokens,initial.selected_tokens);
    EXPECT_EQ(final.largest_selected,initial.largest_selected);
    EXPECT_EQ(final.live_ids,initial.live_ids);
    const bool retainedEqual=final.selected_tokens==initial.selected_tokens && final.largest_selected==initial.largest_selected &&
        final.live_ids==initial.live_ids;
    publishMembership("final",final,retainedEqual?"true":"false");
    require(retainedEqual,"Indexer terminal membership changed before export");
    exportTerminal(fixture,plan);
}
} // namespace llaminar2::qwen4exp::ROCmQwenSparseAttention
