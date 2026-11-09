/**
 * @file Perf__CPUAttentionKeyQ8.cpp
 * @brief Isolated native AQ8 encode/decode/original-query dot timing and profiling.
 *
 * Each process measures one head width, ISA and operation. Allocation, fixture
 * construction, byte authentication and warmup precede the timed interval.
 * Output is compact CSV; perf/ISA evidence belongs to a separate invocation.
 * This harness measures kernel economics, not whole-model throughput or parity.
 */
#include "kernels/cpu/attention/CPUAttentionKeyQ8Dot.h"
#include "../../../../integration/kernels/AttentionKeyQ8DeviceTestCommon.h"
#include "../../../../utils/CPUAttentionFP32DotReference.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <omp.h>
#include <string_view>
#include <vector>

namespace
{
using namespace llaminar2;
using cpu::attention_key_q8::ISA;

/** @brief Exactly one production operation per timing/profiler launch. */
enum class Operation { Encode, Decode, Dot };

/** @brief Validate a complete positive integer argument without partial parsing.
 * @param text Original complete option token.
 * @return Positive representable integer.
 * @throws std::invalid_argument For malformed, overflowing or nonpositive input.
 */
int positive(std::string_view text)
{
    int value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || value <= 0)
        throw std::invalid_argument("expected a positive integer argument");
    return value;
}

/**
 * @brief Repeated production kernel work with a stable, named profiling symbol.
 * @tparam D Complete physical head width; @tparam Isa Authenticated native implementation.
 * @tparam Op Exactly one encode or decode program for this process.
 * @param source Immutable original keys, one contiguous head per block.
 * @param anchors Immutable bases with matching geometry.
 * @param blocks Persistent compressed outputs/inputs.
 * @param output Persistent reconstruction workspace.
 * @param iterations Number of complete passes over the selected heads.
 * @param workers Physical workers with disjoint contiguous head ranges.
 */
template <int D, ISA Isa, Operation Op>
__attribute__((noinline)) void kernelLoop(
    const std::vector<float> &source, const std::vector<float> &anchors,
    std::vector<AttentionKeyQ8Block<D>> &blocks, std::vector<float> &output, int iterations, int workers)
{
    // The harness creates one team, not one parallel region per head or pass.
    // Production callers own their own workshare; the primitive is deliberately
    // independent of OpenMP so it works inside an existing cache-append team.
#pragma omp parallel num_threads(workers) if(workers > 1)
    {
        const auto worker = static_cast<std::size_t>(omp_get_thread_num());
        const auto team = static_cast<std::size_t>(omp_get_num_threads());
        const auto begin = blocks.size() * worker / team;
        const auto end = blocks.size() * (worker + 1) / team;
        for (int iteration = 0; iteration < iterations; ++iteration)
        {
            for (std::size_t head = begin; head < end; ++head)
            {
                const auto anchor = std::span<const float, D>(anchors.data() + head * D, D);
                if constexpr (Op == Operation::Encode)
                    cpu::attention_key_q8::quantize<D>(
                        std::span<const float, D>(source.data() + head * D, D), anchor, blocks[head], Isa);
                else
                    cpu::attention_key_q8::dequantize<D>(
                        blocks[head], anchor, std::span<float, D>(output.data() + head * D, D), Isa);
            }
            // Keep repeated writes observable without adding timed checksum work.
            asm volatile("" ::: "memory");
        }
    }
}

/** @brief Authenticate exact bytes, warm the selected kernel, then time only it.
 * @tparam D Physical head width; @tparam Isa Explicit admitted native implementation.
 * @tparam Op One complete production program, unchanged across all samples.
 * @param heads Retained independent original heads.
 * @param iterations Complete passes per timed sample; @param workers Actual admitted team size.
 * @throws std::runtime_error If production output disagrees with the independent oracle.
 */
template <int D, ISA Isa, Operation Op>
void measure(int heads, int iterations, int workers)
{
    static_assert(Isa == ISA::NEON || Isa == ISA::AVX2 || Isa == ISA::AVX512);
    const auto source = test::makeAttentionKeyQ8DeviceInput<D>(heads);
    std::vector<float> anchors(source.size()), residual(source.size()), output(source.size());
    for (std::size_t index = 0; index < source.size(); ++index)
    {
        anchors[index] = source[(index % D)] * 0.875f;
        residual[index] = source[index] - anchors[index];
    }
    const auto expected = test::makeAttentionKeyQ8ReferenceBlocks<D>(residual);
    auto decoded = test::makeAttentionKeyQ8ReferenceDecoded<D>(expected);
    for (std::size_t index = 0; index < decoded.size(); ++index) decoded[index] += anchors[index];
    auto blocks = expected;
    kernelLoop<D, Isa, Op>(source, anchors, blocks, output, 1, workers);
    if constexpr (Op == Operation::Encode)
    {
        if (std::memcmp(blocks.data(), expected.data(), blocks.size() * sizeof(blocks[0])) != 0)
            throw std::runtime_error("AQ8 encode bytes disagree with scalar authority");
    }
    else if (std::memcmp(output.data(), decoded.data(), output.size() * sizeof(float)) != 0)
        throw std::runtime_error("AQ8 decode bytes disagree with scalar authority");
    kernelLoop<D, Isa, Op>(source, anchors, blocks, output, 256, workers);
    std::array<double, 7> elapsed{};
    for (double &sample : elapsed)
    {
        const auto start = std::chrono::steady_clock::now();
        kernelLoop<D, Isa, Op>(source, anchors, blocks, output, iterations, workers);
        sample = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }
    std::sort(elapsed.begin(), elapsed.end());
    const double ns_per_head = elapsed[elapsed.size() / 2] * 1e9 / iterations / heads;
    const double logical_gib_s = (2.0 * D * sizeof(float) + sizeof(blocks[0])) / ns_per_head * 1e9 / (1024.0 * 1024.0 * 1024.0);
    // Preserve the historical x86 CSV spellings while reporting the actual
    // native compiler envelope. NEON_DOTPROD compilation and this ASIMD-only
    // floating codec are distinct axes, not relabelled x86 observations.
    const auto compiled = compiledCPUISALevel();
    const char *codegen = compiled == ISALevel::AVX512 ? "AVX512" : compiled == ISALevel::AVX2 ? "AVX2" : isaLevelName(compiled);
    const char *runtime = Isa == ISA::NEON ? "NEON" : Isa == ISA::AVX512 ? "AVX512" : "AVX2";
    std::cout << "codegen_isa,runtime_isa,operation,head_dim,heads,iterations,samples,workers,ns_per_head,logical_gib_s,byte_exact\n"
              << codegen << ',' << runtime << ','
              << (Op == Operation::Encode ? "encode" : "decode") << ','
              << D << ',' << heads << ',' << iterations << ',' << elapsed.size() << ',' << workers << ','
              << ns_per_head << ',' << logical_gib_s << ",true\n";
}

/** @brief Execute one original compressed-key dot candidate with prebound query roots.
 * @tparam D Complete head width; @tparam Rows Shared original-query count1..4.
 * @param programs Cold-admitted immutable query/anchor/ISA programs.
 * @param blocks Original immutable compressed heads; @param output Persistent dot results.
 * @param iterations Complete passes; @param workers Explicit admitted physical-worker budget.
 */
template<int D,std::size_t Rows>
__attribute__((noinline)) void dotKernelLoop(
    const std::vector<cpu::attention_key_q8::QueryDotTile<D,Rows>>&programs,
    const std::vector<AttentionKeyQ8Block<D>>&blocks,std::vector<float>&output,int iterations,int workers)
{
#pragma omp parallel num_threads(workers) if(workers>1)
    {
        const auto worker=std::size_t(omp_get_thread_num()),team=std::size_t(omp_get_num_threads());
        const auto begin=blocks.size()*worker/team,end=blocks.size()*(worker+1)/team;
        for(int iteration=0;iteration<iterations;++iteration)
        {
            for(std::size_t head=begin;head<end;++head)
            {
                const auto result=programs[head].dot(blocks[head]);
                for(std::size_t row=0;row<Rows;++row)output[head*Rows+row]=result[row];
            }
            asm volatile("" ::: "memory");
        }
    }
}

/** @brief Authenticate original decode/dot words, then time one fixed query tile only.
 * @tparam D Complete physical head; @tparam Isa Explicit native program.
 * @tparam Rows Physical query width1..4; @param heads Original immutable key count.
 * @param iterations Passes per sample; @param workers Explicit admitted team.
 * @throws std::runtime_error For a wrong dot word before timing.
 * @details Whole decoded heads are diagnostic fixtures outside timing. No decode
 * mirror, bind, allocation or repacking runs inside the measured operation.
 */
template<int D,ISA Isa,std::size_t Rows>
void measureDot(int heads,int iterations,int workers)
{
    using Program=cpu::attention_key_q8::QueryDotTile<D,Rows>;
    const auto source=test::makeAttentionKeyQ8DeviceInput<D>(heads);
    std::vector<float> anchors(source.size()),residual(source.size());
    const std::size_t query_elements=std::size_t(heads)*Rows*D;
    auto query_storage=std::make_unique_for_overwrite<float[]>(query_elements);
    std::span<float> queries(query_storage.get(),query_elements);
    std::vector<float> output(std::size_t(heads)*Rows),expected(output.size());
    // Avoid serial value-initialization of the query bank: its first writes
    // belong to the consuming team, outside the one-candidate timing interval.
#pragma omp parallel for num_threads(workers) if(workers>1) schedule(static)
    for(std::ptrdiff_t i=0;i<std::ptrdiff_t(queries.size());++i)
        queries[std::size_t(i)]=float(int((std::size_t(i)*17+11)%113)-56)*0.003f;
    for(std::size_t i=0;i<source.size();++i)
    {
        anchors[i]=source[i%D]*0.875f;
        residual[i]=source[i]-anchors[i];
    }
    const auto blocks=test::makeAttentionKeyQ8ReferenceBlocks<D>(residual);
    auto decoded=test::makeAttentionKeyQ8ReferenceDecoded<D>(blocks);
    for(std::size_t i=0;i<decoded.size();++i)decoded[i]+=anchors[i];
    std::vector<Program> programs;programs.reserve(heads);
    for(int head=0;head<heads;++head)
    {
        std::array<std::span<const float>,Rows> roots;
        for(std::size_t row=0;row<Rows;++row)
        {
            roots[row]={queries.data()+(std::size_t(head)*Rows+row)*D,D};
            expected[std::size_t(head)*Rows+row]=test::cpuAttentionFP32DotReference<D>(
                roots[row],{decoded.data()+std::size_t(head)*D,D},Isa);
        }
        programs.emplace_back(Program::bind(roots,{anchors.data()+std::size_t(head)*D,D},Isa));
    }
    test::poisonAttentionDotOutput(output);
    dotKernelLoop<D,Rows>(programs,blocks,output,1,workers);
    if(!test::authenticateAttentionDotOutput(output,expected))
        throw std::runtime_error("AQ8 original-query dots disagree with declared decode/dot reference");
    dotKernelLoop<D,Rows>(programs,blocks,output,256,workers);
    std::array<double,7> elapsed{};
    for(auto &sample:elapsed)
    {
        const auto begin=std::chrono::steady_clock::now();
        dotKernelLoop<D,Rows>(programs,blocks,output,iterations,workers);
        sample=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    }
    const auto raw_elapsed=elapsed;
    std::sort(elapsed.begin(),elapsed.end());
    const double ns=elapsed[elapsed.size()/2]*1e9/iterations/heads;
    const double bytes=sizeof(AttentionKeyQ8Block<D>)+D*sizeof(float)*(1+Rows)+Rows*sizeof(float);
    const auto compiled=compiledCPUISALevel();
    const char *codegen=compiled==ISALevel::AVX512?"AVX512":compiled==ISALevel::AVX2?"AVX2":isaLevelName(compiled);
    const char *runtime=Isa==ISA::NEON?"NEON":Isa==ISA::AVX512?"AVX512":"AVX2";
    // One head here means one compressed key shared by exactly Rows queries;
    // divide latency by Rows only when comparing per-query dot economics.
    std::cout<<"codegen_isa,runtime_isa,operation,head_dim,heads,iterations,samples,workers,ns_per_head,logical_gib_s,byte_exact,"
                "sample0_ns,sample1_ns,sample2_ns,sample3_ns,sample4_ns,sample5_ns,sample6_ns\n"
             <<codegen<<','<<runtime<<",dot_rows"<<Rows<<','<<D<<','<<heads<<','<<iterations<<','<<elapsed.size()<<','<<workers<<','
             <<ns<<','<<bytes/ns*1e9/(1024.0*1024.0*1024.0)<<",true";
    for(const double sample:raw_elapsed)std::cout<<','<<sample*1e9/iterations/heads;
    std::cout<<'\n';
}

/** @brief Select one complete operation/query width outside timing.
 * @tparam D Head width; @tparam Isa Authenticated native program.
 * @param operation Encode/decode/dot; @param heads Retained original heads.
 * @param iterations Passes per sample; @param workers Actual admitted workers.
 * @param rows Exact query tile width, meaningful only for dot.
 * @throws std::invalid_argument For an unimplemented query geometry.
 */
template<int D,ISA Isa>
void dispatchOperation(Operation operation,int heads,int iterations,int workers,int rows)
{
    if(operation==Operation::Encode)measure<D,Isa,Operation::Encode>(heads,iterations,workers);
    else if(operation==Operation::Decode)measure<D,Isa,Operation::Decode>(heads,iterations,workers);
    else switch(rows)
    {
    case 1:measureDot<D,Isa,1>(heads,iterations,workers);break;
    case 2:measureDot<D,Isa,2>(heads,iterations,workers);break;
    case 3:measureDot<D,Isa,3>(heads,iterations,workers);break;
    case 4:measureDot<D,Isa,4>(heads,iterations,workers);break;
    default:throw std::invalid_argument("AQ8 dot query rows must be1..4");
    }
}

/** @brief Dispatch once outside timing, preserving a single-operation process.
 * @tparam D Complete supported physical head width.
 * @param isa Authenticated explicit implementation; @param operation Sole program to time.
 * @param heads Number of original heads; @param iterations Complete passes per sample.
 * @param workers Admitted team size, not an inferred physical-core affinity.
 * @param rows Explicit query width for dot; exactly1 for encode/decode.
 * @throws std::invalid_argument If the selected native implementation was not compiled.
 */
template <int D>
void dispatch(ISA isa, Operation operation, int heads, int iterations, int workers, int rows)
{
#if LLAMINAR_COMPILED_WITH_NEON
    if (isa == ISA::NEON)
    {
        dispatchOperation<D,ISA::NEON>(operation,heads,iterations,workers,rows);
        return;
    }
#endif
#if defined(__AVX512F__)
    if (isa == ISA::AVX512)
    {
        dispatchOperation<D,ISA::AVX512>(operation,heads,iterations,workers,rows);
        return;
    }
#endif
#if defined(__AVX2__)
    if (isa == ISA::AVX2)
    {
        dispatchOperation<D,ISA::AVX2>(operation,heads,iterations,workers,rows);
        return;
    }
#endif
    throw std::invalid_argument("AQ8 benchmark ISA has no compiled production implementation");
}
} // namespace

/** @brief Fail closed on incomplete or unsupported benchmark configuration.
 * @param argc Complete argument count; @param argv Paired option/value tokens.
 * @return Zero after authenticated timing, or one with a precise diagnostic.
 */
int main(int argc, char **argv)
{
    try
    {
        ISA isa = ISA::Automatic;
        Operation operation = Operation::Encode;
        int dimensions = 64, heads = 128, iterations = 1000, workers = 1, query_rows = 1;
        for (int index = 1; index < argc; ++index)
        {
            const std::string_view option(argv[index]);
            if (option == "--help")
            {
                std::cout << "CPU AQ8: --operation encode|decode|dot --query-rows 1..4 (dot only) "
                    "--head-dim 64|128|256 --heads N --iterations N --workers N --isa auto|neon|avx2|avx512\n";
                return 0;
            }
            if (++index >= argc) throw std::invalid_argument("missing option value");
            const std::string_view value(argv[index]);
            if (option == "--head-dim") dimensions = positive(value);
            else if (option == "--heads") heads = positive(value);
            else if (option == "--iterations") iterations = positive(value);
            else if (option == "--workers") workers = positive(value);
            else if (option == "--query-rows") query_rows = positive(value);
            else if (option == "--isa")
            {
                if (value == "avx2") isa = ISA::AVX2;
                else if (value == "avx512") isa = ISA::AVX512;
                else if (value == "neon") isa = ISA::NEON;
                else if (value != "auto") throw std::invalid_argument("unsupported ISA");
            }
            else if (option == "--operation")
            {
                if (value == "encode") operation = Operation::Encode;
                else if (value == "decode") operation = Operation::Decode;
                else if (value == "dot") operation = Operation::Dot;
                else throw std::invalid_argument("unsupported operation");
            }
            else throw std::invalid_argument("unknown benchmark option");
        }
        if (query_rows>4||(operation!=Operation::Dot&&query_rows!=1))
            throw std::invalid_argument("AQ8 query rows1..4 are admitted only for the dot operation");
        // A reduced OpenMP team would make the reported worker count false.
        // Authenticate it before fixture construction and before any timing.
        omp_set_dynamic(0);
        int admitted_workers = 0;
#pragma omp parallel num_threads(workers) if(workers > 1)
        {
#pragma omp single
            admitted_workers = omp_get_num_threads();
        }
        if (admitted_workers != workers)
            throw std::runtime_error("OpenMP did not admit the requested benchmark worker count");
        isa = cpu::attention_key_q8::resolveISA(isa);
        switch (dimensions)
        {
        case 64: dispatch<64>(isa, operation, heads, iterations, workers, query_rows); break;
        case 128: dispatch<128>(isa, operation, heads, iterations, workers, query_rows); break;
        case 256: dispatch<256>(isa, operation, heads, iterations, workers, query_rows); break;
        default: throw std::invalid_argument("AQ8 head dimension must be 64, 128 or 256");
        }
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "CPU AQ8 benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
