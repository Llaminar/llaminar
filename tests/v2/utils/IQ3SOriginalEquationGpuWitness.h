/**
 * @file IQ3SOriginalEquationGpuWitness.h
 * @brief Actual captured IQ3_S dense projections versus independent source factors.
 *
 * Reuse the established prepared-GEMM owner and capture interfaces. Every retained
 * geometry observes exact consumed Q8 codes/scales; only terminal test reads join.
 * ROCm raw-pool construction must publish all512 IQ3 grid words before the first
 * multiply; the borrow-only inspection cannot initialize or repair those tables.
 * Every output word/code/scale is still checked on every intended replay. Constant
 * size summaries retain total mismatch counts and the first differing word at
 * each geometry/replay boundary, preventing a real failure from emitting millions
 * of assertion messages while preserving complete diagnostic population.
 */
#pragma once
#include "IQ3SOriginalEquationSource.h"
#include "GpuPreparedGemmHarness.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IGPUGraphCapture.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "kernels/common/NativeEmbeddingIQTables.h"
#include "transfer/TransferEngine.h"
#include <bit>
#include <iomanip>
#include <span>
#include <stdexcept>
#include <gtest/gtest.h>
#ifdef HAVE_CUDA
#include "kernels/cuda/gemm/CUDAQuantisedGemmKernel.h"
#endif
#ifdef HAVE_ROCM
/**
 * @brief Borrow existing native symbols without initializing their contents.
 * @param ordinal Exact owning device. @param tables Destination immutable view.
 * @return Whether all existing table addresses were resolved.
 */
extern "C" bool rocmNativeEmbeddingIQTableView(int ordinal, llaminar2::NativeEmbeddingIQTables *tables);
#endif
namespace llaminar2::test::iq3s_source_witness
{
/**
 * @brief Join a live terminal observation while all graph/staging owners survive.
 * @tparam T Observation element. @tparam Owners Retained backend/stream owner.
 * @param owners Exact device/stream authority. @param pointer Live device field.
 * @param count Exact live element count. @return Joined terminal bytes.
 */
template<class T,class Owners>
std::vector<T> terminal(Owners &owners,const void *pointer,size_t count)
{
    std::vector<T> values(count);
    struct Join
    {
        Owners &o; bool armed=true;
        ~Join(){if(armed)(void)o.backend->synchronizeStream(o.stream,o.device.ordinal);}
    } join{owners};
    if(!pointer||!owners.backend->deviceToHostOnStream(values.data(),pointer,
        count*sizeof(T),owners.device.ordinal,owners.stream)||
        !owners.backend->synchronizeStream(owners.stream,owners.device.ordinal))
        throw std::runtime_error("IQ3_S dense witness terminal publication failed");
    join.armed=false;
    return values;
}
/** @brief Join all dense replay/DMA users before member storage can retire. */
struct DenseOwners
{
    IBackend *backend;
    void *stream;
    DeviceId device;
    iq3s_original_equation::Source weights;
    GpuPreparedGemm prepared;
    std::unique_ptr<DeviceWorkspaceManager> workspace;
    std::unique_ptr<FP32Tensor> input,output;
    std::unique_ptr<IGPUGraphCapture> graph;
    std::vector<float> upload;
    /** @brief Join on failure as well as success before graph/host/device teardown. */
    ~DenseOwners()
    {
        if(stream&&backend)(void)backend->synchronizeStream(stream,device.ordinal);
        if(prepared.kernel)
        {
            if(auto *c=dynamic_cast<IWorkspaceConsumer*>(prepared.kernel))c->bindWorkspace(nullptr);
            prepared.kernel->clearGPUStreamBinding();
        }
    }
};
/**
 * @brief Constant-storage exact comparison census for one replay publication.
 *
 * Observe every element before emitting an expectation. Failure counts and the
 * first original word identify errors without retaining per-element messages or
 * aborting the remaining rows, replays or geometries. This helper owns no tensor,
 * stream, producer/consumer state or arithmetic policy.
 */
struct DenseMismatchSummary
{
    size_t checked=0; ///< Every observed element, including matches.
    size_t mismatches=0; ///< Total differing words/bytes, never clipped.
    size_t first_index=0; ///< Original flattened index of the first difference.
    uint32_t first_actual=0,first_expected=0; ///< Exact words (Q8 codes use raw bytes).

    /**
     * @brief Record one exact comparison without publishing an assertion yet.
     * @param actual Observed raw FP32 word or Q8 byte.
     * @param expected Original source-equation word or fixed Q8 byte.
     * @param index Original flattened publication index, in row order.
     */
    void observe(uint32_t actual,uint32_t expected,size_t index) noexcept
    {
        ++checked;
        if(actual!=expected)
        {
            if(mismatches==0)
            {
                first_index=index;first_actual=actual;first_expected=expected;
            }
            ++mismatches;
        }
    }

    /**
     * @brief Report complete counts and at most one first difference per field.
     * @param expected_count Exact required elements in this publication.
     * @param field Output/Q8-code/Q8-scale publication label.
     * @param rows Actual M for this retained graph.
     * @param replay Retained replay ordinal0..19.
     * @param active_block Original changed32-value activation block.
     * @param row_width Elements per row for lossless first-index coordinates.
     * @details Nonfatal expectations preserve all remaining replay/geometry work.
     */
    void report(size_t expected_count,const char *field,int rows,int replay,
        int active_block,size_t row_width) const
    {
        EXPECT_EQ(checked,expected_count)
            <<field<<" M="<<rows<<" replay="<<replay<<" block="<<active_block;
        EXPECT_EQ(mismatches,size_t{0})
            <<field<<" M="<<rows<<" replay="<<replay<<" block="<<active_block
            <<" checked="<<checked<<" mismatches="<<mismatches
            <<" first_index="<<first_index<<" first_row="<<first_index/row_width
            <<" first_column="<<first_index%row_width
            <<" actual_bits=0x"<<std::hex<<first_actual
            <<" expected_bits=0x"<<first_expected<<std::dec;
    }
};
/** @brief Explicit test-only scope; ordinary default continues production dispatch. */
enum class DenseExecutionScope { Ordinary, VerifierDecodeEquivalent };
/** @brief Existing ordinary dense population; retained backing stays 512 rows. */
inline constexpr std::array<int,4> defaultDenseRows{1,3,8,512};
/**
 * @brief Real captured dense source equations with exact observed Q8 publications.
 * @param device Explicit CUDA or ROCm endpoint; ordinary generated dispatch stays active.
 * @param row_populations Exact positive rows per retained geometry, at most512.
 * @param execution_scope Explicit verifier mode for separately named bypass diagnostics.
 * @details The IQ3_S all-code/domain sweep is representation-independent. Every graph
 * replays20changed blocks/signs, including32-value K tail and source-row view.
 */
inline void runDense(DeviceId device,
    std::span<const int> row_populations = defaultDenseRows,
    DenseExecutionScope execution_scope = DenseExecutionScope::Ordinary)
{
    ASSERT_FALSE(row_populations.empty());
    for (int rows : row_populations) ASSERT_TRUE(rows > 0 && rows <= 512);
    namespace eq=iq3s_original_equation;
    auto &context=GPUDeviceContextPool::instance().getContext(device);
    context.submitAndWait([&]{
        auto *backend=getBackendFor(device);ASSERT_NE(backend,nullptr);
        void *stream=context.getOrCreateAuxiliaryStream("iq3s.original_source.dense");
        ASSERT_NE(stream,nullptr);
        {
            DenseOwners o{backend,stream,device,eq::source()};
            o.prepared=makeGpuPreparedGemm(o.weights.view.get(),device,
                "iq3s.original_source.dense.",
                ModelContextId{uint64_t(5997000+(device.is_cuda()?0:1000))});
            ASSERT_NE(o.prepared.kernel,nullptr);o.prepared.kernel->setGPUStream(stream);
#ifdef HAVE_ROCM
            if (device.is_rocm())
            {
                // The focused CTest runs this witness alone in a fresh process.
                // Inspect before ANY multiply or embedding/table-preparation API:
                // a missing raw-constructor publication must fail, not self-warm.
                NativeEmbeddingIQTables tables{};
                ASSERT_TRUE(::rocmNativeEmbeddingIQTableView(device.ordinal, &tables));
                ASSERT_NE(tables.iq3s, nullptr);
                const auto published = terminal<uint32_t>(o, tables.iq3s, 512);
                DenseMismatchSummary grid_words;
                for (size_t index = 0; index < published.size(); ++index)
                    grid_words.observe(published[index], iq3s_grid[index], index);
                ASSERT_EQ(grid_words.checked, size_t{512});
                ASSERT_EQ(grid_words.mismatches, size_t{0})
                    << "cold_raw_pool_IQ3_grid_before_first_multiply"
                    << " checked=" << grid_words.checked
                    << " mismatches=" << grid_words.mismatches
                    << " first_index=" << grid_words.first_index
                    << " actual_bits=0x" << std::hex << grid_words.first_actual
                    << " expected_bits=0x" << grid_words.first_expected << std::dec;
            }
#endif
#ifdef HAVE_CUDA
            if (device.is_cuda() && execution_scope == DenseExecutionScope::VerifierDecodeEquivalent)
            {
                // Explicit bypass diagnostics own their immutable ROWPAR view
                // before capture; DIRECT may reuse it but never rewrites it.
                auto *cuda_kernel = dynamic_cast<cuda::CUDAQuantisedGemmKernel *>(o.prepared.kernel);
                ASSERT_NE(cuda_kernel, nullptr);
                cuda_kernel->prepareRowMajorWeights(stream);
            }
#endif

            auto *consumer=dynamic_cast<IWorkspaceConsumer*>(o.prepared.kernel);ASSERT_NE(consumer,nullptr);
            WorkspaceRequirements requirements;
            for(int m:row_populations)requirements.merge(consumer->getWorkspaceRequirements(m,o.weights.n,eq::k));
            o.workspace=std::make_unique<DeviceWorkspaceManager>(device,requirements.total_bytes_with_alignment());
            ASSERT_TRUE(o.workspace->allocate(requirements));consumer->bindWorkspace(o.workspace.get());
            o.input=std::make_unique<FP32Tensor>(std::vector<size_t>{512,eq::k});
            o.output=std::make_unique<FP32Tensor>(std::vector<size_t>{512,size_t(o.weights.n)});
            std::fill_n(o.input->mutable_data(),512*eq::k,0.0f);
            TransferEngine::prepareDeviceInput(o.input.get(),device,stream);
            ASSERT_TRUE(o.output->allocateOnDevice(device,stream));
            o.upload.resize(512*eq::k);
            DeviceNativeVNNIMatrixDesc desc{};
            ASSERT_TRUE(o.prepared.kernel->exportNativeVNNIMatrixDesc(desc));
            ASSERT_EQ(desc.n,o.weights.n);ASSERT_EQ(desc.k,eq::k);
            ASSERT_EQ(desc.blocks_per_row,eq::k/32);
            ASSERT_EQ(desc.source_identity_present,1);
            ASSERT_EQ(desc.codebook_id,11);
            ASSERT_EQ(desc.source_codebook_id,11);
            ASSERT_EQ(desc.source_is_superblock,1);
            ASSERT_EQ(eq::expected(o.weights,o.weights.n-1,0,1),384.375f);
            for(int m:row_populations)
            {
                // The scope is constructed on the actual context worker and
                // outlives setup, capture and every replay of this geometry.
                auto verifier = execution_scope == DenseExecutionScope::VerifierDecodeEquivalent
                    ? o.prepared.kernel->beginVerifierDecodeEquivalentScope() : nullptr;
                SCOPED_TRACE("IQ3_S M="+std::to_string(m));
                // Pay immutable pointer/launch setup before each retained geometry.
                ASSERT_TRUE(o.prepared.kernel->multiply_tensor(o.input.get(),o.output.get(),m,o.weights.n,eq::k));
                ASSERT_TRUE(backend->synchronizeStream(stream,device.ordinal));
                TransferEngine::requireDeviceInput(o.input.get(),device,stream);
                o.graph=context.createGraphCapture(stream);ASSERT_NE(o.graph,nullptr);
                {
                    ScopedBackendGraphCapture capture(context,*o.graph,"IQ3_S original source dense");
                    ASSERT_TRUE(capture.begin());
                    ASSERT_TRUE(o.prepared.kernel->multiply_tensor(o.input.get(),o.output.get(),m,o.weights.n,eq::k));
                    capture.finish();
                }
                ASSERT_TRUE(o.graph->instantiate());
                std::vector<GPUGraphKernelNodeInfo> nodes;std::string error;
                ASSERT_TRUE(o.graph->inspectKernelNodes(nodes,&error))<<error;ASSERT_FALSE(nodes.empty());
                for(int replay=0;replay<20;++replay)
                {
                    constexpr std::array<int,5> active{0,1,7,8,0};const int block=active[replay%active.size()];
                    std::fill(o.upload.begin(),o.upload.end(),0.0f);
                    for(int row=0;row<m;++row)
                    {
                        const int sign=(row+replay)&1?-1:1;
                        o.upload[row*eq::k+block*32]=float(sign);
                        o.upload[row*eq::k+block*32+31]=float(sign*127);
                    }
                    ASSERT_TRUE(backend->hostToDeviceOnStream(o.input->gpu_data_ptr(),o.upload.data(),m*eq::k*sizeof(float),device.ordinal,stream));
                    TransferEngine::publishCurrentDeviceWrite(o.input.get(),stream);
                    TransferEngine::requireDeviceInput(o.input.get(),device,stream);
                    ASSERT_TRUE(backend->memset(o.output->gpu_data_ptr(),0xff,m*o.weights.n*sizeof(float),device.ordinal,stream));
                    ASSERT_TRUE(o.graph->launch());
                    const auto actual=terminal<float>(o,o.output->gpu_data_ptr(),m*o.weights.n);
                    const auto codes=terminal<int8_t>(o,o.workspace->getBuffer(GemmWorkspaceBuffers::QUANT_A),m*eq::k);
                    const auto scales=terminal<float>(o,o.workspace->getBuffer(GemmWorkspaceBuffers::SCALES_A_BLOCKWISE),m*(eq::k/32));
                    DenseMismatchSummary output_words,q8_codes,q8_scales;
                    for(int row=0;row<m;++row)
                    {
                        const int sign=(row+replay)&1?-1:1;
                        for(int col=0;col<o.weights.n;++col)
                        {
                            const size_t index=size_t(row)*o.weights.n+col;
                            output_words.observe(std::bit_cast<uint32_t>(actual[index]),
                                std::bit_cast<uint32_t>(eq::expected(o.weights,col,block,sign)),index);
                        }
                        for(int j=0;j<eq::k;++j)
                        {
                            const size_t index=size_t(row)*eq::k+j;
                            q8_codes.observe(std::bit_cast<uint8_t>(codes[index]),
                                std::bit_cast<uint8_t>(int8_t(o.upload[index])),index);
                        }
                        for(int j=0;j<eq::k/32;++j)
                        {
                            const size_t index=size_t(row)*(eq::k/32)+j;
                            // Exactly1.0f has one FP32 representation, so raw word
                            // equality preserves the original exact scale test.
                            q8_scales.observe(std::bit_cast<uint32_t>(scales[index]),
                                std::bit_cast<uint32_t>(1.0f),index);
                        }
                    }
                    output_words.report(size_t(m)*o.weights.n,"output",m,replay,block,o.weights.n);
                    q8_codes.report(size_t(m)*eq::k,"Q8-codes",m,replay,block,eq::k);
                    q8_scales.report(size_t(m)*(eq::k/32),"Q8-scales",m,replay,block,eq::k/32);
                }
                o.graph.reset();
            }
            EXPECT_EQ(std::memcmp(o.weights.parent->raw_data(),o.weights.bytes.data(),o.weights.bytes.size()),0);
        }
    });
}
} // namespace llaminar2::test::iq3s_source_witness
