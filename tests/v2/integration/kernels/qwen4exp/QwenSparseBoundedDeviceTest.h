/**
 * @file QwenSparseBoundedDeviceTest.h
 * @brief Real CUDA/HIP stored-score retained selector and paired timing harness.
 *
 * Each graph retains one or a bounded repeated chain of actual production
 * selection nodes. Each node owns its declared rows; event timing submits one
 * complete graph and normalizes by its authenticated repeat count. Fixture
 * upload/terminal reads and native inspection are outside replay timing. Retirement is declared
 * after all asynchronous host/device backing and joins the exact stream before
 * graph, buffer or host-span destruction, including assertions/exceptions.
 */
#pragma once
#include "QwenSparseSelectionCases.h"
#include "kernels/qwen4exp/QwenSparseBoundedSelection.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
namespace llaminar2::qwen4exp::QSB_SUITE
{
template<class Status> void check(Status status)
{
    if (status != QSB_RT(Success)) throw std::runtime_error(QSB_RT(GetErrorString)(status));
}
#if defined(QSB_CUDA_RUNTIME) && !defined(CUDART_VERSION)
#error "CUDA selector graph-edge helpers require the selected CUDA runtime version"
#endif
/**
 * @brief Query ordinary selector parent edges with the selected native SDK ABI.
 * @param node Actual native graph node from this retained selector graph.
 * @param parents Output parent array, or null for a count-only query.
 * @param count Input capacity/output parent count owned by the caller.
 * @return Exact CUDA/HIP status, including CUDA lossy-query failure for annotated edges.
 * @details CUDA 13 adds edgeData before count. Null deliberately requests ordinary
 * completion edges; the caller's check rejects an unsupported annotated edge.
 * Earlier CUDA and HIP retain the original ordinary-edge argument list.
 */
inline auto selectorDependencies(QSB_NODE node,QSB_NODE* parents,std::size_t* count)
{
#if defined(QSB_CUDA_RUNTIME) && CUDART_VERSION >= 13000
    return QSB_RT(GraphNodeGetDependencies)(node,parents,nullptr,count);
#else
    return QSB_RT(GraphNodeGetDependencies)(node,parents,count);
#endif
}
/**
 * @brief Remove exact ordinary selector edges for the focused malformed-chain control.
 * @param graph Native source graph; its executable is never launched while malformed.
 * @param from Exact original parent nodes.
 * @param to Exact original child nodes.
 * @param count Number of paired ordinary edges to remove.
 * @return Exact native status; failures remain visible through check.
 */
inline auto removeSelectorDependencies(QSB_GRAPH graph,const QSB_NODE* from,const QSB_NODE* to,std::size_t count)
{
#if defined(QSB_CUDA_RUNTIME) && CUDART_VERSION >= 13000
    return QSB_RT(GraphRemoveDependencies)(graph,from,to,nullptr,count);
#else
    return QSB_RT(GraphRemoveDependencies)(graph,from,to,count);
#endif
}
/**
 * @brief Restore the same ordinary selector edges after the malformed-chain control.
 * @param graph Native source graph whose complete original chain is being restored.
 * @param from Original parent nodes retained by the fixture.
 * @param to Original child nodes retained by the fixture.
 * @param count Number of paired ordinary edges to restore.
 * @return Exact native status; no retry or alternative graph is installed.
 */
inline auto addSelectorDependencies(QSB_GRAPH graph,const QSB_NODE* from,const QSB_NODE* to,std::size_t count)
{
#if defined(QSB_CUDA_RUNTIME) && CUDART_VERSION >= 13000
    return QSB_RT(GraphAddDependencies)(graph,from,to,nullptr,count);
#else
    return QSB_RT(GraphAddDependencies)(graph,from,to,count);
#endif
}
struct Stream final
{
    QSB_STREAM value = nullptr;
    Stream() { check(QSB_RT(StreamCreateWithFlags)(&value, QSB_RT(StreamNonBlocking))); }
    ~Stream() { if (value) (void)QSB_RT(StreamDestroy)(value); }
};
/** @brief Explicit diagnostic allocation only; production candidate uses no global scratch. */
template<class T> struct Buffer final
{
    T *value = nullptr; std::size_t size;
    explicit Buffer(std::size_t count) : size(count)
    { check(QSB_RT(Malloc)(reinterpret_cast<void **>(&value), count * sizeof(T))); }
    ~Buffer() { if (value) (void)QSB_RT(Free)(value); }
    Buffer(const Buffer &) = delete;
    void upload(const T *source, QSB_STREAM stream)
    { check(QSB_RT(MemcpyAsync)(value, source, size * sizeof(T), QSB_RT(MemcpyHostToDevice), stream)); }
    std::vector<T> read(QSB_STREAM stream)
    {
        std::vector<T> result(size);
        try
        {
            check(QSB_RT(MemcpyAsync)(result.data(), value, size * sizeof(T), QSB_RT(MemcpyDeviceToHost), stream));
            check(QSB_RT(StreamSynchronize)(stream));
        }
        catch (...) { if (QSB_RT(StreamSynchronize)(stream) != QSB_RT(Success)) std::terminate(); throw; }
        return result;
    }
};
/** @brief Mandatory last-declared owner: failures cannot unwind borrowed copy spans early. */
struct Retirement final
{
    QSB_STREAM stream;
    ~Retirement() { if (QSB_RT(StreamSynchronize)(stream) != QSB_RT(Success)) std::terminate(); }
};
struct Graph final
{
    QSB_GRAPH graph = nullptr; QSB_EXEC executable = nullptr;
    ~Graph()
    {
        if (executable) (void)QSB_RT(GraphExecDestroy)(executable);
        if (graph) (void)QSB_RT(GraphDestroy)(graph);
    }
};
struct NativeSelection final
{
    Stream stream;
    int rows, capacity, blocks;
    QwenSparseReplay replay;
    std::vector<float> scores;
    std::vector<std::int32_t> poison;
    Buffer<QwenSparseReplay> state;
    Buffer<float> input;
    Buffer<std::int32_t> selected, count;
    Buffer<QwenSparseSelectionObservation> observation;
    Graph graph;
    QwenSparseIndexer p;
    int capturedRepeats = 0; ///< Sealed node count, published only after full native-chain validation.
    Retirement retirement;
    NativeSelection(int m, int tokens, QwenSparsePoolSelection policy, bool observe = true)
        : rows(m), capacity(tokens), blocks((tokens + 3) / 4), replay{tokens - m, m, 0},
          scores(selection_test::scores(m, blocks, selection_test::Distribution::Hash)),
          poison(std::size_t(m) * QwenSparseSelected, -77), state(1), input(scores.size()),
          selected(poison.size()), count(m), observation(m), retirement{stream.value}
    {
        p.replay = state.value; p.scores = input.value; p.selected = selected.value;
        p.selected_count = count.value; p.row_capacity = m; p.token_capacity = tokens; p.block_capacity = blocks;
        p.pool_selection = policy;
        if (observe && policy != QwenSparsePoolSelection::Installed) p.selection_observations = observation.value;
        upload(); check(QSB_RT(StreamSynchronize)(stream.value));
    }
    void upload()
    {
        state.upload(&replay, stream.value); input.upload(scores.data(), stream.value);
        selected.upload(poison.data(), stream.value);
    }
    /**
     * @brief Admit immutable selector repeat geometry before beginning native capture.
     * @param repeats Actual selector nodes to retain, from 1 through 256.
     * @throws std::invalid_argument For an invalid repeat count or more than 8192 row-invocations.
     * @details No native call or execution-owned state read is needed for admission.
     */
    void requireRepeatShape(int repeats) const
    {
        if(repeats<1 || repeats>256 || rows<1 || rows>8192/repeats ||
           p.row_capacity!=rows || p.token_capacity!=capacity || p.block_capacity!=blocks)
            throw std::invalid_argument("Stored-score selector capture requires R 1..256 and M times R <= 8192 with exact bound shape");
    }
    /**
     * @brief Authenticate every repeated native selector and the complete linear chain.
     * @param repeats Expected exact node count from the source-admitted capture plan.
     * @return Actual native parameters shared by every authenticated selector node.
     * @throws std::logic_error For foreign nodes/functions, geometry/resources, or non-linear dependencies.
     * @details Graph enumeration order is not execution order. Parent links build
     * one successor map, then a unique-root traversal must cover every node.
     */
    QSB_PARAMS authenticateCapture(int repeats) const
    {
        requireRepeatShape(repeats);
        if(!graph.graph)throw std::logic_error("Stored-score selector graph is not captured");
        std::size_t size=0;check(QSB_RT(GraphGetNodes)(graph.graph,nullptr,&size));
        if(size!=std::size_t(repeats))
            throw std::logic_error("Stored-score selector graph has the wrong actual native node count");
        std::vector<QSB_NODE> nodes(size);check(QSB_RT(GraphGetNodes)(graph.graph,nodes.data(),&size));
        if(size!=nodes.size())throw std::logic_error("Stored-score selector native inventory changed during inspection");
        const auto expectedGrid=p.pool_selection==QwenSparsePoolSelection::Installed &&
            rows>attention::kMaxGroupedVerifierAttentionRows ? unsigned((rows+7)/8) : unsigned(rows);
        std::size_t expectedShared=0;
        switch(p.pool_selection) {
        case QwenSparsePoolSelection::Installed:
            expectedShared=rows<=attention::kMaxGroupedVerifierAttentionRows ? 3u*8u*sizeof(int) : 0u;break;
        case QwenSparsePoolSelection::Bounded256_512:expectedShared=sizeof(bounded_selection::Storage<256,512>);break;
        case QwenSparsePoolSelection::Bounded512_512:expectedShared=sizeof(bounded_selection::Storage<512,512>);break;
        case QwenSparsePoolSelection::Bounded1024_1024:expectedShared=sizeof(bounded_selection::Storage<1024,1024>);break;
        default:throw std::logic_error("Stored-score selector capture has an unknown policy");
        }
        QSB_PARAMS first{};QSB_ATTRIBUTES firstAttributes{};
        std::vector<int> successor(size,-1);int root=-1;
        for(std::size_t index=0;index<size;++index) {
            QSB_NODETYPE type;check(QSB_RT(GraphNodeGetType)(nodes[index],&type));
            if(type!=QSB_RT(GraphNodeTypeKernel))
                throw std::logic_error("Stored-score selector capture contains a non-kernel node");
            QSB_PARAMS params{};check(QSB_RT(GraphKernelNodeGetParams)(nodes[index],&params));
            if(!params.func)throw std::logic_error("Stored-score selector node has null native function identity");
            QSB_ATTRIBUTES attributes{};check(QSB_RT(FuncGetAttributes)(&attributes,params.func));
            if(params.gridDim.x!=expectedGrid || params.gridDim.y!=1 || params.gridDim.z!=1 ||
               params.blockDim.x!=256 || params.blockDim.y!=1 || params.blockDim.z!=1 || params.sharedMemBytes!=0 ||
               attributes.localSizeBytes!=0 || attributes.sharedSizeBytes!=expectedShared)
                throw std::logic_error("Stored-score selector node has incorrect native geometry/resources");
            if(index==0) {first=params;firstAttributes=attributes;}
            else if(params.func!=first.func || attributes.numRegs!=firstAttributes.numRegs ||
                    attributes.sharedSizeBytes!=firstAttributes.sharedSizeBytes || attributes.localSizeBytes!=firstAttributes.localSizeBytes)
                throw std::logic_error("Stored-score repeated selectors differ in actual native function/resources");
            std::size_t dependencies=0;check(selectorDependencies(nodes[index],nullptr,&dependencies));
            if(dependencies==0) {
                if(root!=-1)throw std::logic_error("Stored-score selector graph contains multiple roots");
                root=int(index);
            } else {
                if(dependencies!=1)throw std::logic_error("Stored-score selector graph is not a linear dependency chain");
                QSB_NODE parent=nullptr;check(selectorDependencies(nodes[index],&parent,&dependencies));
                const auto found=std::find(nodes.begin(),nodes.end(),parent);
                if(found==nodes.end())throw std::logic_error("Stored-score selector dependency is outside its graph inventory");
                const auto parentIndex=std::size_t(found-nodes.begin());
                if(parentIndex==index || successor[parentIndex]!=-1)
                    throw std::logic_error("Stored-score selector graph contains a self-edge or fork");
                successor[parentIndex]=int(index);
            }
        }
        std::vector<bool> visited(size,false);int cursor=root;std::size_t covered=0;
        while(cursor!=-1) {
            if(visited[std::size_t(cursor)])throw std::logic_error("Stored-score selector graph contains a cycle");
            visited[std::size_t(cursor)]=true;++covered;cursor=successor[std::size_t(cursor)];
        }
        if(root==-1 || covered!=size)
            throw std::logic_error("Stored-score selector chain does not cover every actual native node");
        return first;
    }
    /**
     * @brief Capture bounded actual selectors once; replay submits one complete native graph.
     * @param repeats Exact retained selector count; default 1 preserves single-node fixtures.
     * @throws std::invalid_argument For inadmissible repeat/row geometry before capture starts.
     * @throws std::logic_error For an already bound graph or incomplete native identity/chain.
     * @details Only the existing selector entrypoint executes inside capture; all
     * validation/metadata allocation and event setup stay outside timed replay.
     */
    void capture(int repeats=1)
    {
        requireRepeatShape(repeats);
        if(graph.graph || graph.executable || capturedRepeats)
            throw std::logic_error("Stored-score selector graph is already captured");
        validateQwenSparseSelection(p);
        check(QSB_RT(StreamBeginCapture)(stream.value,QSB_RT(StreamCaptureModeThreadLocal)));
        try { for(int repeat=0;repeat<repeats;++repeat)QSB_SELECT(p,stream.value); }
        catch (...)
        {
            QSB_GRAPH abandoned=nullptr;(void)QSB_RT(StreamEndCapture)(stream.value,&abandoned);
            if(abandoned)(void)QSB_RT(GraphDestroy)(abandoned);
            throw;
        }
        check(QSB_RT(StreamEndCapture)(stream.value,&graph.graph));
        const auto params=authenticateCapture(repeats);
        check(QSB_RT(GraphInstantiate)(&graph.executable,graph.graph,nullptr,nullptr,0));
        capturedRepeats=repeats;
        QSB_ATTRIBUTES attributes{};check(QSB_RT(FuncGetAttributes)(&attributes,params.func));
        std::cout<<"QSA_SELECTOR_RESOURCE,policy="<<int(p.pool_selection)<<",regs="<<attributes.numRegs
                 <<",shared="<<attributes.sharedSizeBytes<<",local="<<attributes.localSizeBytes
                 <<",block="<<params.blockDim.x<<",grid="<<params.gridDim.x
                 <<",captured_repeats="<<capturedRepeats<<",native_nodes="<<capturedRepeats
                 <<",dependency_chain=complete_linear"<<'\n';
    }
    void launch() { check(QSB_RT(GraphLaunch)(graph.executable, stream.value)); }
    /** @brief Terminal exact output/source/diagnostic checks; never a timed host probe. */
    void verify()
    {
        const auto result = selected.read(stream.value), counts = count.read(stream.value);
        const auto original = input.read(stream.value);
        ASSERT_EQ(std::memcmp(scores.data(), original.data(), scores.size() * sizeof(float)), 0);
        const auto observed = p.selection_observations ? observation.read(stream.value) :
            std::vector<QwenSparseSelectionObservation>{};
        for (int row = 0; row < rows; ++row)
        {
            const int visible = replay.prefix_tokens + row + 1;
            const auto expected = row < replay.live_rows ? selection_test::oracle(
                scores.data() + std::size_t(row) * blocks, visible) : std::vector<std::int32_t>{};
            ASSERT_EQ(counts[row], expected.size());
            for (std::size_t i = 0; i < expected.size(); ++i)
                ASSERT_EQ(result[std::size_t(row) * QwenSparseSelected + i], expected[i]);
            for (std::size_t i = expected.size(); i < QwenSparseSelected; ++i)
                ASSERT_EQ(result[std::size_t(row) * QwenSparseSelected + i], -77);
            if (p.selection_observations)
            {
                EXPECT_EQ(observed[row].score_elements,
                    row < replay.live_rows && visible / 4 > 512 ? std::uint64_t(visible / 4) : 0u);
                if (row < replay.live_rows && visible / 4 > 512) { EXPECT_GT(observed[row].compactions, 0u); }
                else { EXPECT_EQ(observed[row].compactions, 0u); }
                EXPECT_LE(observed[row].radix_passes,4u*observed[row].compactions);
                EXPECT_GE(observed[row].compacted_keys,512u*observed[row].compactions);
            }
        }
    }
};
/**
 * @test Every one of the 256 cutoff bins is owned by its actual native logical lane.
 * @details Row r targets bin base+r; 511 strictly higher pools leave a one-member
 * cutoff tie quota. All three bounded policies run the retained production graph.
 * The 255 boundary additionally descends from a higher score byte, while lower
 * bins exercise long zero gaps. Existing numeric-sort verification checks exact
 * selected IDs, causal tails, original score bytes and inactive output poison.
 */
TEST(QSB_SUITE, ParallelHistogramEveryBucketOwner)
{
    int devices = 0; check(QSB_RT(GetDeviceCount)(&devices));
    if (!devices) GTEST_SKIP() << "No native backend device";
    for (auto policy : selection_test::policies)
    {
        NativeSelection s(32, 641 * 4, policy);
        s.capture();
        for (int base = 0; base < 256; base += 32)
        {
            for (int row = 0; row < s.rows; ++row)
                for (int id = 0; id < s.blocks; ++id)
                {
                    const auto cutoff = std::uint32_t(base + row);
                    const std::uint32_t bits = 0x3f800000u + cutoff + (id < 511 ? 1u : 0u);
                    std::memcpy(&s.scores[std::size_t(row) * s.blocks + id], &bits, sizeof(bits));
                }
            s.replay = {s.capacity - s.rows, s.rows, 0};
            s.upload(); s.launch(); s.verify();
            const auto work = s.observation.read(s.stream.value);
            for (int row = 0; row < s.rows; ++row)
            {
                EXPECT_EQ(work[row].compactions, 1u);
                EXPECT_EQ(work[row].radix_passes, base + row == 255 ? 2u : 1u);
            }
        }
        for (int live : {17, 0, 32})
        {
            s.replay = {s.capacity - s.rows, live, 0};
            s.upload(); s.launch(); s.verify();
        }
    }
}

/**
 * @test Captured production selectors retain the four-digit frontier and exact carry tails.
 * @details One graph per policy covers all causal tails and aligned/unaligned
 * horizons through 64K. The independent numeric-sort oracle checks membership;
 * closed-form workload identities check actual optional diagnostic counters.
 */
TEST(QSB_SUITE, PersistentFourDigitFrontierWork)
{
    int devices=0;check(QSB_RT(GetDeviceCount)(&devices));
    if(!devices)GTEST_SKIP()<<"No native backend device";
    for(auto policy:selection_test::policies)
    {
        NativeSelection s(1,65536,policy);
        s.scores=selection_test::scores(s.rows,s.blocks,selection_test::Distribution::PersistentFourDigitFrontier);
        s.upload();s.capture();
        for(int pools:{767,768,769,1535,1536,1537,16383,16384})
            for(int tail=0;tail<4;++tail)
            {
                if(pools*4+tail>s.capacity)continue;
                SCOPED_TRACE(int(policy));
                SCOPED_TRACE(pools);
                SCOPED_TRACE(tail);
                s.replay={pools*4+tail-1,1,0};s.upload();s.launch();s.verify();
                const auto observed=s.observation.read(s.stream.value)[0];
                const auto expected=selection_test::persistentFrontierWork(pools,policy);
                EXPECT_EQ(observed.score_elements,expected.score_elements);
                EXPECT_EQ(observed.compactions,expected.compactions);
                EXPECT_EQ(observed.compacted_keys,expected.compacted_keys);
                EXPECT_EQ(observed.radix_passes,expected.radix_passes);
            }
        s.replay={0,0,0};s.upload();s.launch();s.verify();
    }
}

/**
 * @test Native one-warp selection preserves hot histogram frontier IDs and work.
 * @details One retained graph per policy covers aligned and unaligned horizons
 * with every available causal tail, then an empty replay. Actual optional work
 * proves N/C/W and aligned four digits; tail membership stays independent sort.
 */
TEST(QSB_SUITE, HotHistogramFrontierWork)
{
    int devices=0;check(QSB_RT(GetDeviceCount)(&devices));
    if(!devices)GTEST_SKIP()<<"No native backend device";
    for(auto policy:selection_test::policies)
    {
        NativeSelection s(1,65536,policy);
        s.scores=selection_test::scores(s.rows,s.blocks,selection_test::Distribution::HotHistogramFrontier);
        s.upload();s.capture();
        for(int pools:{767,768,769,1535,1536,1537,16383,16384})
            for(int tail=0;tail<4;++tail)
            {
                if(pools*4+tail>s.capacity)continue;
                SCOPED_TRACE(int(policy));
                SCOPED_TRACE(pools);
                SCOPED_TRACE(tail);
                s.replay={pools*4+tail-1,1,0};s.upload();s.launch();s.verify();
                const auto observed=s.observation.read(s.stream.value)[0];
                const auto expected=selection_test::persistentFrontierWork(pools,policy);
                EXPECT_EQ(observed.score_elements,expected.score_elements);
                EXPECT_EQ(observed.compactions,expected.compactions);
                EXPECT_EQ(observed.compacted_keys,expected.compacted_keys);
                EXPECT_LE(observed.radix_passes,4*observed.compactions);
                if(pools%256==0)EXPECT_EQ(observed.radix_passes,4*observed.compactions);
            }
        s.replay={0,0,0};s.upload();s.launch();s.verify();
    }
}

TEST(QSB_SUITE, ThresholdTiesAndEveryTail)
{
    int devices = 0; check(QSB_RT(GetDeviceCount)(&devices));
    if (!devices) GTEST_SKIP() << "No native backend device";
    for (auto policy : selection_test::policies)
    {
        NativeSelection s(1, 16391, policy); s.capture();
        for (auto distribution : selection_test::distributions)
            for (int pools : {511, 512, 513, 1025, 4097})
                for (int tail = 0; tail < 4; ++tail)
                {
                    SCOPED_TRACE(int(policy));
                    SCOPED_TRACE(int(distribution));
                    SCOPED_TRACE(pools);
                    SCOPED_TRACE(tail);
                    s.scores = selection_test::scores(s.rows, s.blocks, distribution);
                    s.replay = {pools * 4 + tail - 1, 1, 0};
                    s.upload(); s.launch(); s.verify();
                    ASSERT_FALSE(::testing::Test::HasFailure());
                }
    }
}
TEST(QSB_SUITE, RetainedPartialEmptyAndPositiveDispatch)
{
    int devices = 0; check(QSB_RT(GetDeviceCount)(&devices));
    if (!devices) GTEST_SKIP() << "No native backend device";
    for (int rows : {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,31,32,33,512,2048})
        for (auto policy : selection_test::policies)
        {
            NativeSelection s(rows, 4099 + rows, policy); s.capture();
            for (int repeat = 0; repeat < (rows == 33 ? 20 : 4); ++repeat)
            {
                s.replay.live_rows = repeat % 4 == 0 ? 0 : repeat % 4 == 1 ? 1 : repeat % 4 == 2 ? std::min(rows, 9) : rows;
                s.replay.prefix_tokens = repeat % 3 == 0 ? 2048 : 4099;
                s.upload(); s.launch(); s.verify();
                ASSERT_FALSE(::testing::Test::HasFailure());
            }
        }
}
/** @test Actual grouped native outputs equal one retained serial native row graph. */
TEST(QSB_SUITE, GroupedSerialRowBytes)
{
    int devices = 0; check(QSB_RT(GetDeviceCount)(&devices));
    if (!devices) GTEST_SKIP() << "No native backend device";
    for (int rows=1;rows<=16;++rows)
    for (auto policy : selection_test::policies)
    {
        NativeSelection grouped(rows,4096+rows,policy),serial(1,4096+rows,policy);
        grouped.capture(); serial.capture(); grouped.launch(); grouped.verify();
        const auto output = grouped.selected.read(grouped.stream.value);
        const auto counts = grouped.count.read(grouped.stream.value);
        for (int row = 0; row < rows; ++row)
        {
            std::copy_n(grouped.scores.data() + std::size_t(row) * grouped.blocks,
                grouped.blocks, serial.scores.data());
            serial.replay = {grouped.replay.prefix_tokens + row, 1, 0};
            serial.upload(); serial.launch(); serial.verify();
            const auto one = serial.selected.read(serial.stream.value);
            EXPECT_EQ(serial.count.read(serial.stream.value)[0], counts[row]);
            EXPECT_EQ(std::memcmp(one.data(), output.data() + std::size_t(row) * QwenSparseSelected,
                QwenSparseSelected * sizeof(std::int32_t)), 0);
            ASSERT_FALSE(::testing::Test::HasFailure());
        }
    }
}
/** @test Both signed zeros and one-ULP cutoff ties reach the real retained selector at64K. */
TEST(QSB_SUITE, AdversarialScoresAt64K)
{
    int devices=0;check(QSB_RT(GetDeviceCount)(&devices));if(!devices)GTEST_SKIP()<<"No native backend device";
    for(auto policy:selection_test::policies)for(int rows:{1,16}){
        NativeSelection s(rows,65536,policy);s.capture();
        for(auto distribution:{selection_test::Adversarial::SignedZero,selection_test::Adversarial::CutoffOneULP}){
            s.scores=selection_test::adversarialScores(rows,s.blocks,distribution);
            // Retain the same native graph through full/small/empty/full. Upload
            // poison before each replay proves inactive/suffix non-writes.
            for(int live:{rows,1,0,rows}){
                s.replay={65536-rows,live,0};s.upload();s.launch();s.verify();
                ASSERT_FALSE(::testing::Test::HasFailure());
            }
        }
    }
}
TEST(QSB_SUITE, BadBindingsAndStreamsBeforeEnqueue)
{
    int devices = 0; check(QSB_RT(GetDeviceCount)(&devices));
    if (!devices) GTEST_SKIP() << "No native backend device";
    NativeSelection s(1, 2052, QwenSparsePoolSelection::Bounded256_512);
    EXPECT_THROW(QSB_SELECT(s.p, nullptr), std::invalid_argument);
    EXPECT_THROW(QSB_SELECT(s.p, QSB_RT(StreamLegacy)), std::invalid_argument);
    EXPECT_THROW(QSB_SELECT(s.p, QSB_RT(StreamPerThread)), std::invalid_argument);
    auto p = s.p; p.pool_selection = static_cast<QwenSparsePoolSelection>(255);
    EXPECT_THROW(QSB_SELECT(p, s.stream.value), std::invalid_argument);
    p = s.p; p.selected_count = p.selected; EXPECT_THROW(QSB_SELECT(p, s.stream.value), std::invalid_argument);
    p = s.p; p.block_capacity = 1; EXPECT_THROW(QSB_SELECT(p, s.stream.value), std::invalid_argument);
}
/** @brief Persistent native events, outside correctness and metadata observation. */
struct Events final
{
    QSB_EVENT begin = nullptr, end = nullptr;
    QSB_STREAM stream;
    explicit Events(QSB_STREAM value) : stream(value)
    {
        check(QSB_RT(EventCreate)(&begin));
        try { check(QSB_RT(EventCreate)(&end)); }
        catch (...) { (void)QSB_RT(EventDestroy)(begin); throw; }
    }
    ~Events()
    {
        if (QSB_RT(StreamSynchronize)(stream) != QSB_RT(Success)) std::terminate();
        if (end) (void)QSB_RT(EventDestroy)(end);
        if (begin) (void)QSB_RT(EventDestroy)(begin);
    }
};
TEST(QSB_SUITE, PairedReleaseTiming)
{
    const char *rowsText = std::getenv("LLAMINAR_QSA_SELECT_M");
    const char *tokensText = std::getenv("LLAMINAR_QSA_SELECT_TOKENS");
    const char *repeatsText = std::getenv("LLAMINAR_QSA_SELECT_REPEATS");
    const char *distributionText = std::getenv("LLAMINAR_QSA_SELECT_DISTRIBUTION");
    if (!rowsText || !tokensText || !repeatsText || !distributionText)
        GTEST_SKIP() << "Explicit exact selector timing geometry/repeats/distribution required";
    const int rows = std::stoi(rowsText), tokens = std::stoi(tokensText), repeats = std::stoi(repeatsText);
    const int distributionId = std::stoi(distributionText);
    ASSERT_GT(rows, 0); ASSERT_LE(rows, 2048); ASSERT_GE(tokens, rows); ASSERT_LE(tokens, 262147);
    ASSERT_GT(repeats, 0); ASSERT_LE(repeats, 256);
    ASSERT_LE(rows,8192/repeats); // Source-admitted bound before any fixture/capture allocation.
    ASSERT_GE(distributionId, 0); ASSERT_LT(distributionId, int(selection_test::distributions.size()));
    if(distributionId==int(selection_test::Distribution::HotHistogramFrontier))
        ASSERT_LE(tokens,65536); // The authored contention identity is scoped through 16384 pools.
    for (auto policy : selection_test::policies)
    {
        NativeSelection a(rows, tokens, QwenSparsePoolSelection::Installed, false), b(rows, tokens, policy, false);
        a.scores = selection_test::scores(rows, a.blocks, selection_test::distributions[distributionId]); b.scores = a.scores;
        a.upload(); b.upload(); a.capture(repeats); b.capture(repeats);
        for (int warm = 0; warm < 3; ++warm) { a.launch(); b.launch(); }
        a.verify(); b.verify(); ASSERT_FALSE(::testing::Test::HasFailure());
        Events aEvents(a.stream.value), bEvents(b.stream.value);
        const auto time = [&](NativeSelection &s, Events &events)
        {
            check(QSB_RT(EventRecord)(events.begin, s.stream.value));
            s.launch(); // One host submission contains every authenticated selector repeat.
            check(QSB_RT(EventRecord)(events.end, s.stream.value));
            check(QSB_RT(EventSynchronize)(events.end)); // terminal fence OUTSIDE measured event interval
            float milliseconds = 0; check(QSB_RT(EventElapsedTime)(&milliseconds, events.begin, events.end));
            return milliseconds / s.capturedRepeats; // Normalize the complete graph interval exactly once.
        };
        for (int pair = 0; pair < 4; ++pair)
        {
            const bool ab = pair % 2 == 0;
            float first = ab ? time(a, aEvents) : time(b, bEvents);
            float second = ab ? time(b, bEvents) : time(a, aEvents);
            std::cout << "QSA_SELECT_PAIR,scope=stored_scores_captured_repeat,order=" << (ab ? "AB" : "BA")
                      << ",M=" << rows << ",tokens=" << tokens << ",distribution=" << distributionId
                      << ",captured_repeats=" << a.capturedRepeats << ",native_nodes=" << a.capturedRepeats
                      << ",graph_launches_per_event=1,normalization=graph_ms_over_captured_repeats"
                      << ",policy=" << int(policy) << ",installed_ms=" << (ab ? first : second)
                      << ",candidate_ms=" << (ab ? second : first) << '\n';
        }
        a.verify(); b.verify(); ASSERT_FALSE(::testing::Test::HasFailure());
        NativeSelection diagnostic(rows, tokens, policy, true);
        diagnostic.scores = a.scores; diagnostic.upload(); diagnostic.capture(); diagnostic.launch(); diagnostic.verify();
        const auto observations = diagnostic.observation.read(diagnostic.stream.value);
        QwenSparseSelectionObservation total{};
        for (const auto &row : observations)
        {
            total.score_elements += row.score_elements; total.compactions += row.compactions;
            total.compacted_keys += row.compacted_keys; total.radix_passes += row.radix_passes;
        }
        std::cout << "QSA_SELECT_WORK,policy=" << int(policy) << ",score_elements=" << total.score_elements
                  << ",compactions=" << total.compactions << ",compacted_keys=" << total.compacted_keys
                  << ",radix_passes=" << total.radix_passes << ",scope=diagnostic_not_timed_not_wire_bytes\n";
        ASSERT_FALSE(::testing::Test::HasFailure());
    }
}


/**
 * @test Repeated actual selectors retain one authenticated chain across full,
 * partial and empty device replay, with independent membership and dirty bytes.
 */
TEST(QSB_SUITE, CapturedRepeatsRetainFullPartialEmpty)
{
    int devices=0;check(QSB_RT(GetDeviceCount)(&devices));
    if(!devices)GTEST_SKIP()<<"No native backend device";
    constexpr std::array policies{QwenSparsePoolSelection::Installed,QwenSparsePoolSelection::Bounded256_512,
        QwenSparsePoolSelection::Bounded512_512,QwenSparsePoolSelection::Bounded1024_1024};
    for(const int rows:{1,16,33})for(const auto policy:policies) {
        SCOPED_TRACE(rows);
        SCOPED_TRACE(int(policy));
        NativeSelection repeated(rows,16384,policy,false),single(rows,16384,policy,false);
        repeated.capture(3);single.capture();
        EXPECT_EQ(repeated.capturedRepeats,3);EXPECT_EQ(single.capturedRepeats,1);
        const auto identity=repeated.authenticateCapture(3).func;
        EXPECT_EQ(identity,single.authenticateCapture(1).func);
        auto prior=repeated.poison;
        for(const auto replay:std::array<QwenSparseReplay,4>{{
            {12000,rows,0},{8194,std::min(rows,5),0},{0,0,0},{12000,rows,0}}}) {
            repeated.replay=replay;single.replay=replay;
            repeated.state.upload(&repeated.replay,repeated.stream.value);
            single.state.upload(&single.replay,single.stream.value);
            repeated.launch();single.launch();
            const auto output=repeated.selected.read(repeated.stream.value),counts=repeated.count.read(repeated.stream.value);
            EXPECT_EQ(output,single.selected.read(single.stream.value));
            EXPECT_EQ(counts,single.count.read(single.stream.value));
            for(int row=0;row<rows;++row) {
                const auto expected=row<replay.live_rows ? selection_test::oracle(
                    repeated.scores.data()+std::size_t(row)*repeated.blocks,replay.prefix_tokens+row+1) : std::vector<std::int32_t>{};
                ASSERT_EQ(counts[row],expected.size());
                for(int id=0;id<QwenSparseSelected;++id) {
                    const auto offset=std::size_t(row)*QwenSparseSelected+id;
                    EXPECT_EQ(output[offset],id<int(expected.size()) ? expected[id] : prior[offset]);
                }
            }
            const auto unchanged=repeated.input.read(repeated.stream.value);
            EXPECT_EQ(std::memcmp(unchanged.data(),repeated.scores.data(),unchanged.size()*sizeof(float)),0);
            EXPECT_EQ(repeated.authenticateCapture(3).func,identity);
            prior=output;
            ASSERT_FALSE(::testing::Test::HasFailure());
        }
        if(rows==1 && policy==QwenSparsePoolSelection::Installed) {
            // Mutation is diagnostic metadata only, after terminal joins. The
            // executable is not launched while the source graph is malformed.
            std::size_t size=3;std::array<QSB_NODE,3> nodes{};
            check(QSB_RT(GraphGetNodes)(repeated.graph.graph,nodes.data(),&size));
            QSB_NODE parent=nullptr,child=nullptr;
            for(const auto node:nodes) {
                std::size_t dependencies=0;check(selectorDependencies(node,nullptr,&dependencies));
                if(dependencies==1) {child=node;check(selectorDependencies(node,&parent,&dependencies));break;}
            }
            ASSERT_NE(child,nullptr);ASSERT_NE(parent,nullptr);
            check(removeSelectorDependencies(repeated.graph.graph,&parent,&child,1));
            EXPECT_THROW(repeated.authenticateCapture(3),std::logic_error);
            check(addSelectorDependencies(repeated.graph.graph,&parent,&child,1));
            EXPECT_EQ(repeated.authenticateCapture(3).func,identity);
        }
        EXPECT_THROW(repeated.capture(3),std::logic_error);
    }
    NativeSelection maximum(1,2052,QwenSparsePoolSelection::Installed,false);
    EXPECT_THROW(maximum.capture(0),std::invalid_argument);
    EXPECT_THROW(maximum.capture(257),std::invalid_argument);
    EXPECT_EQ(maximum.graph.graph,nullptr);EXPECT_EQ(maximum.capturedRepeats,0);
    maximum.capture(256);maximum.launch();maximum.verify();
    EXPECT_EQ(maximum.capturedRepeats,256);
    EXPECT_THROW(maximum.authenticateCapture(255),std::logic_error);
    NativeSelection rowBudget(33,16384,QwenSparsePoolSelection::Bounded512_512,false);
    EXPECT_THROW(rowBudget.capture(249),std::invalid_argument); // 33 times 249 = 8217.
    EXPECT_EQ(rowBudget.graph.graph,nullptr);EXPECT_EQ(rowBudget.capturedRepeats,0);
    rowBudget.capture();rowBudget.launch();rowBudget.verify(); // The rejected plan left the stream reusable.
}


/**
 * @test Score-only radix keeps canonical long ties and never visits ID digits.
 * @details The 640-pool cases make exactly one actual compaction per row,
 * allowing a per-call observed bound rather than only an aggregate average.
 */
TEST(QSB_SUITE, ScoreOnlyRadixCanonicalTiesAndFourDigitBound)
{
    int devices=0;check(QSB_RT(GetDeviceCount)(&devices));
    if(!devices)GTEST_SKIP()<<"No native backend device";
    for(const auto policy:selection_test::policies) {
        NativeSelection s(16,16384,policy,true);s.capture();
        for(const bool equal:{true,false}) {
            for(int row=0;row<s.rows;++row)for(int id=0;id<s.blocks;++id) {
                // Equal positive scores use no radix. Mixed scores place the
                // top tie at high original IDs; signed-zero coverage is also
                // retained in the existing adversarial native regression.
                const std::uint32_t bits=equal ? 0x3f800000u : id>=s.blocks-1024 ? 0x3f800001u : 0x3f800000u;
                std::memcpy(&s.scores[std::size_t(row)*s.blocks+id],&bits,sizeof(bits));
            }
            for(const int lastVisible:{640*4,16384}) {
                s.replay={lastVisible-s.rows,s.rows,0};s.upload();s.launch();s.verify();
                const auto observations=s.observation.read(s.stream.value);
                for(const auto& observed:observations) {
                    EXPECT_GT(observed.compactions,0u);
                    EXPECT_LE(observed.radix_passes,4u*observed.compactions);
                    EXPECT_GE(observed.compacted_keys,512u*observed.compactions);
                    if(equal || lastVisible==640*4)EXPECT_EQ(observed.radix_passes,0u);
                    if(lastVisible==640*4)EXPECT_EQ(observed.compactions,1u);
                }
                ASSERT_FALSE(::testing::Test::HasFailure());
            }
        }
        // Each distribution drives exactly one real compact call in the short
        // score frontier, including byte boundaries and finite extremes.
        for(auto distribution:selection_test::distributions) {
            s.scores=selection_test::scores(s.rows,s.blocks,distribution);
            s.replay={640*4-s.rows,s.rows,0};s.upload();s.launch();s.verify();
            for(const auto& observed:s.observation.read(s.stream.value)) {
                EXPECT_EQ(observed.compactions,1u);EXPECT_LE(observed.radix_passes,4u);
            }
            ASSERT_FALSE(::testing::Test::HasFailure());
        }
        // First case: 128 strict-high plus exactly 384 members of a score
        // prefix bucket fill rank 512 after one digit, despite a coarse pivot.
        // Second case: 255 strict-high plus a long exact tie selects 257 equals,
        // crossing both fixed 32-lane and 256-thread compaction boundaries.
        for(const bool coarse:{true,false}) {
            for(int row=0;row<s.rows;++row)for(int id=0;id<s.blocks;++id) {
                const std::uint32_t bits=coarse ?
                    (id<128 ? 0x40000000u : id<512 ? 0x3f001000u+std::uint32_t(id-128)*0x1000u : 0x3e000000u) :
                    (id<255 ? 0x40000000u : 0x3f800000u);
                std::memcpy(&s.scores[std::size_t(row)*s.blocks+id],&bits,sizeof(bits));
            }
            s.replay={640*4-s.rows,s.rows,0};s.upload();s.launch();s.verify();
            for(const auto& observed:s.observation.read(s.stream.value)) {
                EXPECT_EQ(observed.compactions,1u);EXPECT_EQ(observed.radix_passes,coarse ? 1u : 4u);
            }
            ASSERT_FALSE(::testing::Test::HasFailure());
        }
        // A 511-key strict upper bucket leaves rank 1 in a shared three-byte
        // score prefix; one final-byte winner forces all four score digits.
        for(int row=0;row<s.rows;++row)for(int id=0;id<s.blocks;++id) {
            const std::uint32_t bits=id<511 ? 0x7f7fffffu : id==511 ? 0x01010101u : 0x01010100u;
            std::memcpy(&s.scores[std::size_t(row)*s.blocks+id],&bits,sizeof(bits));
        }
        s.replay={640*4-s.rows,s.rows,0};s.upload();s.launch();s.verify();
        for(const auto& observed:s.observation.read(s.stream.value)) {
            EXPECT_EQ(observed.compactions,1u);EXPECT_EQ(observed.radix_passes,4u);
        }
        ASSERT_FALSE(::testing::Test::HasFailure());
    }
}

/** @test Both actual Installed GPU schedules preserve canonical numeric signed-zero ties. */
TEST(QSB_SUITE, InstalledSignedZeroCanonicalTies)
{
    int devices=0;check(QSB_RT(GetDeviceCount)(&devices));
    if(!devices)GTEST_SKIP()<<"No native backend device";
    for(const int rows:{1,16,17}) {
        SCOPED_TRACE(rows);
        NativeSelection s(rows,16384,QwenSparsePoolSelection::Installed,false);
        for(int row=0;row<rows;++row)for(int id=0;id<s.blocks;++id) {
            const std::uint32_t bits=id%2 ? 0x80000000u : 0u;
            std::memcpy(&s.scores[std::size_t(row)*s.blocks+id],&bits,sizeof(bits));
        }
        s.upload();s.capture();
        for(int lastVisible:{2052,8195,8196,16384}) {
            s.replay={lastVisible-rows,rows,0};s.upload();s.launch();s.verify();
            ASSERT_FALSE(::testing::Test::HasFailure());
        }
    }
}
} // namespace llaminar2::qwen4exp::QSB_SUITE
