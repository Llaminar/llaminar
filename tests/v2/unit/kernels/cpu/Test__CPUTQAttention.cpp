/**
 * @file Test__CPUTQAttention.cpp
 * @brief Exact native centroid programs, checked cache borrows and original-head work ownership.
 *
 * Small original-byte fixtures are authored before invocation. Independent
 * codebook/FMA oracles certify native arithmetic; real public serial calls
 * certify same-backend worker grouping. No Tensor/factory/cache/request or
 * full-attention performance certificate is implied by these leaf proofs.
 * Supplied-codebook tests preserve original table words (including nonfinite
 * payloads) as decode-only data, distinct from the finite numerical domain.
 */
#include "kernels/cpu/attention/TQFusedAttentionPrimitives.h"
#include "utils/CPUExplicitRoundingOracle.h"
#include <gtest/gtest.h>
#include <omp.h>
#include <array>
#include <bit>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace llaminar2::test
{
namespace
{
using cpu::tq_attention::Format;
using cpu::tq_attention::NormRole;
static_assert(!std::is_default_constructible_v<cpu::tq_attention::DotOperand<Format::TQ4>>);
static_assert(!std::is_default_constructible_v<cpu::tq_attention::AccumOperand<Format::TQ8>>);
static_assert(!std::is_constructible_v<cpu::tq_attention::HeadView<Format::TQ4, NormRole::Source>, const std::uint8_t *, int, float>);

/** @brief Retain original complete physical bytes and independently known source indices. */
struct Fixture
{
    int dim;
    Format format;
    std::vector<std::uint8_t> bytes, indices;
    /** @brief Author bounded diagnostic physical storage, not a production preparer.
     * @param d Complete ABI width; @param f Existing encoding; @param seed Original fixture identity.
     */
    Fixture(int d, Format f, int seed) : dim(d), format(f), bytes(8 + d / (f == Format::TQ4 ? 2 : 1) + 16, 0x65), indices(d)
    {
        norms((seed + 3) * 0.03125f, (seed + 7) * 0.01713f);
        const int codes = f == Format::TQ4 ? 16 : 256;
        for (int i = 0; i < d; ++i) indices[i] = static_cast<std::uint8_t>((seed * 17 + i * 73) % codes);
        if (f == Format::TQ8) std::memcpy(bytes.data() + 8, indices.data(), d);
        else
        {
            for (int group = 0; group < d / 8; ++group)
            {
                std::uint8_t low[8], high = 0;
                for (int i = 0; i < 8; ++i)
                { low[i] = indices[group * 8 + i] & 7; high |= (indices[group * 8 + i] >> 3) << i; }
                tq3_pack_8(low, bytes.data() + 8 + group * 3);
                bytes[8 + d * 3 / 8 + group] = high;
            }
        }
    }
    /** @brief Author distinct original source and fitted value radii.
     * @param source Original source L2 norm; @param reconstruction Fitted value radius.
     */
    void norms(float source, float reconstruction)
    { std::memcpy(bytes.data(), &source, sizeof(source)); std::memcpy(bytes.data() + 4, &reconstruction, sizeof(reconstruction)); }
    /** @return Original selected physical norm through an independent byte read. */
    float norm(NormRole role) const
    { float x; std::memcpy(&x, bytes.data() + (role == NormRole::Source ? 0 : 4), sizeof(x)); return x; }
    /** @return Independently known original centroid for one coordinate. */
    float centroid(int coordinate) const
    { return format == Format::TQ4 ? TQ4_CENTROIDS[indices[coordinate]] : TQ8_CENTROIDS[indices[coordinate]]; }
};

/** @brief Invoke the sole production public numerator entrypoint.
 * @param f Original complete physical head; @param out Original writable numerator; @param weight Original softmax weight.
 */
void accumulate(const Fixture &f, float *out, float weight)
{
    if (f.format == Format::TQ4) tq4_accum_weighted(out, f.bytes.data(), weight, f.dim);
    else tq8_accum_weighted(out, f.bytes.data(), weight, f.dim);
}
/** @brief Invoke the sole production public score entrypoint.
 * @param f Original complete physical head; @param query Original rotated query.
 * @return Actual same-backend ordered score.
 */
float score(const Fixture &f, const float *query)
{ return f.format == Format::TQ4 ? tq4_dot_rotated_q(query, f.bytes.data(), f.dim) : tq8_dot_rotated_q(query, f.bytes.data(), f.dim); }

#if LLAMINAR_COMPILED_WITH_NEON
/** @brief Independently execute the native fixed two-chain score program with memory-rounded folds.
 * @param f Original diagnostic head/indices; @param query Original FP32 query.
 * @return Exact native reference score, not an x86 cross-ISA reduction claim.
 */
float referenceNativeScore(const Fixture &f, const float *query)
{
    const float norm = f.norm(NormRole::Source);
    if (norm < 1e-30f) return 0;
    float lo[4]{}, hi[4]{};
    for (int i = 0; i < f.dim; i += 8)
        for (int lane = 0; lane < 4; ++lane)
        {
            volatile float a = std::fma(f.centroid(i + lane), query[i + lane], lo[lane]);
            volatile float b = std::fma(f.centroid(i + 4 + lane), query[i + 4 + lane], hi[lane]);
            lo[lane] = a; hi[lane] = b;
        }
    float sum[4]; for (int i = 0; i < 4; ++i) sum[i] = memoryRoundedAdd(lo[i], hi[i]);
    return memoryRoundedMultiply(memoryRoundedAdd(memoryRoundedAdd(sum[0], sum[1]), memoryRoundedAdd(sum[2], sum[3])), norm);
}
#endif
/** @brief Independently update diagnostic original numerator coordinates.
 * @param f Original fixture; @param values Complete original output/canary bank.
 * @param weight Original finite softmax weight.
 */
void referenceAccumulation(const Fixture &f, std::vector<float> &values, float weight)
{
    const float norm = f.norm(NormRole::Reconstruction);
    if (norm < 1e-30f || std::abs(weight) < 1e-30f) return;
    const float combined = memoryRoundedMultiply(weight, norm);
    for (int i = 0; i < f.dim; ++i)
    { volatile float value = std::fma(f.centroid(i), combined, values[i]); values[i] = value; }
}

/** @test Actual native numerator FMA uses fitted radii, threshold edges and exact original codebook words. */
TEST(CPUTQAttention, OriginalMetadataRolesThresholdsAndNumeratorBytes)
{
    for (auto format : {Format::TQ4, Format::TQ8})
        for (int dim : {64, 128, 256})
            for (int seed = 0; seed < 32; ++seed)
                for (float weight : {0.0f, -0.0f, 1e-31f, -1e-31f, std::nextafter(1e-30f, 0.0f), 1e-30f, -1e-30f, 0.3333333f, -0.67231f, 1.0f, 2.0f})
                {
                    Fixture fixture(dim, format, seed);
                    if (seed % 7 == 0) fixture.norms(0.0f, 0.217f);
                    if (seed % 7 == 1) fixture.norms(0.991f, 0.0f);
                    if (seed % 7 == 2) fixture.norms(0.121f, std::nextafter(1e-30f, 0.0f));
                    if (seed % 7 == 3) fixture.norms(0.625f, 1e-30f);
                    const auto original = fixture.bytes;
                    std::vector<float> actual(dim + 4, 919.25f);
                    for (int i = 0; i < dim; ++i) actual[i] = float((i * 13 + seed * 11) % 251 - 125) * 0.003127f;
                    auto expected = actual;
                    referenceAccumulation(fixture, expected, weight);
                    accumulate(fixture, actual.data(), weight);
                    EXPECT_EQ(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)), 0);
                    EXPECT_EQ(fixture.bytes, original);
                }
}

#if LLAMINAR_COMPILED_WITH_NEON
/** @test Native original-order score bytes follow an independent memory-rounded FMA/fold oracle. */
TEST(CPUTQAttention, NativeScoreMatchesIndependentOrderedFMABytes)
{
    for (auto format : {Format::TQ4, Format::TQ8})
        for (int dim : {64, 128, 256})
            for (int seed = 0; seed < 64; ++seed)
            {
                Fixture f(dim, format, seed);
                if (seed % 5 == 0) f.norms(0, 0.7f);
                if (seed % 5 == 1) f.norms(1e-30f, 0.7f);
                std::vector<float> query(dim);
                for (int i = 0; i < dim; ++i) query[i] = float((i * 131 + seed * 17) % 257 - 128) * 0.017321f;
                EXPECT_EQ(std::bit_cast<std::uint32_t>(score(f, query.data())), std::bit_cast<std::uint32_t>(referenceNativeScore(f, query.data())));
            }
}

/** @test Every original TQ4 code/bit position and complete-byte domain matches independent canonical unpacking. */
TEST(CPUTQAttention, NativeCentroidDecodeByteDomains)
{
    const cpu::neon_tq_attention::TQ4Lookup lookup;
    for (int byte = 0; byte < 4; ++byte)
        for (int value = 0; value < 256; ++value)
            for (int pattern : {0, 0x55, 0xaa, 0xff})
            {
                std::uint8_t low[3] = {std::uint8_t(pattern), std::uint8_t(pattern), std::uint8_t(pattern)};
                std::uint8_t high = static_cast<std::uint8_t>(pattern);
                if (byte < 3) low[byte] = static_cast<std::uint8_t>(value); else high = static_cast<std::uint8_t>(value);
                std::uint8_t indices[8]; tq3_unpack_8(low, indices);
                const auto got = lookup.decode(low, high);
                float actual[8]; vst1q_f32(actual, got.lo); vst1q_f32(actual + 4, got.hi);
                for (int lane = 0; lane < 8; ++lane)
                {
                    const auto code = indices[lane] | (((high >> lane) & 1) << 3);
                    EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[lane]), std::bit_cast<std::uint32_t>(TQ4_CENTROIDS[code]));
                }
            }
    for (int lane = 0; lane < 4; ++lane)
        for (int code = 0; code < 256; ++code)
        {
            std::uint8_t indices[4] = {0, 85, 170, 255}; indices[lane] = static_cast<std::uint8_t>(code);
            float values[4]; vst1q_f32(values, cpu::neon_tq_attention::lookupTQ8(indices));
            for (int i = 0; i < 4; ++i) EXPECT_EQ(std::bit_cast<std::uint32_t>(values[i]), std::bit_cast<std::uint32_t>(TQ8_CENTROIDS[indices[i]]));
        }
}

/** @test Supplied original tables retain every FP32 word, not a replacement global codebook. */
TEST(CPUTQAttention, NativeOriginalCodebookWordAuthorities)
{
    const auto isa=activeISALevel();
    if((isa!=ISALevel::NEON&&isa!=ISALevel::NEONDotProd)||
        !isNativeCPUExecutionISAPair(compiledCPUISALevel(),isa))
        throw std::runtime_error("Original TQ codebook conformance requires an admitted native ARM program");
    constexpr std::array<std::uint32_t,16>words={0,0x80000000u,1,0x007fffffu,0x00800000u,0x3f800001u,
        0xbf800001u,0x7fc12345u,0xffc23456u,0x7f812345u,0xff812345u,0x7f800000u,0xff800000u,
        0x3eaaaaabu,0x4b800001u,0xcb800001u};
    std::array<float,16>table4;std::array<float,256>table8;
    constexpr std::size_t prefix=7,suffix=9;constexpr std::uint32_t guard=0x45b639deu;
    for(unsigned seed=0;seed<256;++seed)
    {
        for(unsigned i=0;i<table4.size();++i)table4[i]=std::bit_cast<float>(words[(i+seed)%words.size()]);
        for(unsigned i=0;i<table8.size();++i)table8[i]=std::bit_cast<float>(words[(i+seed)%words.size()]^((i>>4)&0x7fu));
        const auto original4=table4;const auto original8=table8;
        const cpu::neon_tq_attention::TQ4Lookup lookup{std::span<const float,16>(table4)};
        std::uint8_t low[8],indices4[8],indices8[8],packed[3],high=0;
        for(unsigned lane=0;lane<8;++lane)
        {
            indices4[lane]=std::uint8_t((seed*73+lane*11)&15);low[lane]=indices4[lane]&7;
            high|=std::uint8_t((indices4[lane]>>3)<<lane);indices8[lane]=std::uint8_t(seed+lane*73);
        }
        tq3_pack_8(low,packed);
        std::array<float,prefix+8+suffix>out4,out8;out4.fill(std::bit_cast<float>(guard));out8=out4;
        for(unsigned lane=0;lane<8;++lane)
        {
            out4[prefix+lane]=std::bit_cast<float>(~std::bit_cast<std::uint32_t>(table4[indices4[lane]]));
            out8[prefix+lane]=std::bit_cast<float>(~std::bit_cast<std::uint32_t>(table8[indices8[lane]]));
        }
        const auto values=lookup.decode(packed,high);
        vst1q_f32(out4.data()+prefix,values.lo);vst1q_f32(out4.data()+prefix+4,values.hi);
        vst1q_f32(out8.data()+prefix,cpu::neon_tq_attention::lookupTQ8(std::span<const float,256>(table8),indices8));
        vst1q_f32(out8.data()+prefix+4,cpu::neon_tq_attention::lookupTQ8(std::span<const float,256>(table8),indices8+4));
        for(unsigned lane=0;lane<8;++lane)
        {
            EXPECT_EQ(std::bit_cast<std::uint32_t>(out4[prefix+lane]),std::bit_cast<std::uint32_t>(table4[indices4[lane]]));
            EXPECT_EQ(std::bit_cast<std::uint32_t>(out8[prefix+lane]),std::bit_cast<std::uint32_t>(table8[indices8[lane]]));
        }
        for(unsigned i=0;i<out4.size();++i)if(i<prefix||i>=prefix+8)
        {
            EXPECT_EQ(std::bit_cast<std::uint32_t>(out4[i]),guard);EXPECT_EQ(std::bit_cast<std::uint32_t>(out8[i]),guard);
        }
        EXPECT_EQ(std::memcmp(table4.data(),original4.data(),sizeof(table4)),0);
        EXPECT_EQ(std::memcmp(table8.data(),original8.data(),sizeof(table8)),0);
    }
}
#endif

/** @brief Restore original OpenMP policy even when a worker assertion exits early. */
class WorkerBudget final
{
public:
    /** @brief Retain the actual caller budget while admitting exact requested teams. */
    WorkerBudget() : threads_(omp_get_max_threads()), dynamic_(omp_get_dynamic()) { omp_set_dynamic(0); }
    /** @brief Restore unchanged caller policy at every exit. */
    ~WorkerBudget() { omp_set_num_threads(threads_); omp_set_dynamic(dynamic_); }
    /** @return Actual original positive worker capacity, not a fabricated topology width. */
    int threads() const noexcept { return threads_; }
private:
    int threads_, dynamic_;
};

/** @test Original physical heads remain byte-equivalent when an existing team changes head ownership. */
TEST(CPUTQAttention, AllPositiveWorkersRetainOriginalSerialHeadBytes)
{
    const WorkerBudget budget;
    for (auto format : {Format::TQ4, Format::TQ8})
        for (int dim : {64, 128, 256})
            for (int rows : {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,31,33})
            {
                const int heads = rows * 3;
                Fixture f(dim, format, rows);
                std::vector<float> query(std::size_t(heads) * dim), initial(query.size() + 4, 919.25f), expected_scores(heads);
                for (std::size_t i = 0; i < query.size(); ++i) { query[i] = float(int(i % 251) - 125) * 0.03125f; initial[i] = query[i] * 0.125f; }
                auto expected = initial;
                for (int head = 0; head < heads; ++head)
                { expected_scores[head] = score(f, query.data() + std::size_t(head) * dim); accumulate(f, expected.data() + std::size_t(head) * dim, 0.3333333f); }
                for (int workers = 1; workers <= budget.threads(); ++workers)
                {
                    auto actual = initial;
                    std::vector<float> actual_scores(heads, 919.25f);
                    int observed = 0;
#pragma omp parallel num_threads(workers)
                    {
#pragma omp single
                        observed = omp_get_num_threads();
#pragma omp for schedule(static)
                        for (int head = 0; head < heads; ++head)
                        { actual_scores[head] = score(f, query.data() + std::size_t(head) * dim); accumulate(f, actual.data() + std::size_t(head) * dim, 0.3333333f); }
                    }
                    ASSERT_EQ(observed, workers);
                    EXPECT_EQ(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)), 0);
                    EXPECT_EQ(std::memcmp(actual_scores.data(), expected_scores.data(), actual_scores.size() * sizeof(float)), 0);
                }
            }
}

/** @test Original storage authorities and typed head geometry reject before numerator mutation. */
TEST(CPUTQAttention, StorageAndGeometryAdmissionPrecedeWrites)
{
    for (auto format : {Format::TQ4, Format::TQ8})
    {
        Fixture f(64, format, 1); const auto original = f.bytes;
        std::vector<float> output(68, 919.25f); const auto before = output;
        auto *aliased = reinterpret_cast<float *>(f.bytes.data());
        EXPECT_THROW(accumulate(f, aliased, 0.5f), std::invalid_argument);
        EXPECT_THROW(score(f, aliased), std::invalid_argument);
        EXPECT_EQ(f.bytes, original);
        EXPECT_THROW(accumulate(f, nullptr, 0.5f), std::invalid_argument);
        EXPECT_THROW(score(f, nullptr), std::invalid_argument);
        const auto call = [&](int dim) {
            if (format == Format::TQ4) tq4_accum_weighted(output.data(), reinterpret_cast<const std::uint8_t *>(1), 0.5f, dim);
            else tq8_accum_weighted(output.data(), reinterpret_cast<const std::uint8_t *>(1), 0.5f, dim);
        };
        for (int bad : {-1,1,8,31,32,63,65,127,129,255,257,std::numeric_limits<int>::max()}) EXPECT_THROW(call(bad), std::invalid_argument);
        float *book = const_cast<float *>(format == Format::TQ4 ? TQ4_CENTROIDS.data() : TQ8_CENTROIDS.data());
        EXPECT_THROW(accumulate(f, book, 0.5f), std::invalid_argument);
        EXPECT_EQ(output, before);
    }
    EXPECT_EQ(tq4_dot_rotated_q(nullptr, nullptr, 0), 0);
    EXPECT_EQ(tq8_dot_rotated_q(nullptr, nullptr, 0), 0);
    EXPECT_NO_THROW(tq4_accum_weighted(nullptr, nullptr, 1.0f, 0));
    EXPECT_NO_THROW(tq8_accum_weighted(nullptr, nullptr, 1.0f, 0));
}

/** @test Scalar metadata failures cannot partially update an original numerator head. */
TEST(CPUTQAttention, InvalidSelectedMetadataAndWeightRejectBeforeWrites)
{
    for (auto format : {Format::TQ4, Format::TQ8})
        for (int dim : {64,128,256})
        {
            Fixture f(dim, format, 1);
            std::vector<float> out(dim + 4, 919.25f), query(dim, 0.125f); const auto original = out;
            for (float bad : {-1.0f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()})
            {
                f.norms(bad, 0.5f); EXPECT_THROW(score(f, query.data()), std::domain_error);
                f.norms(0.5f, bad); EXPECT_THROW(accumulate(f, out.data(), 0.5f), std::domain_error);
            }
            f.norms(0.5f, 0.75f);
            for (float bad : {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()})
                EXPECT_THROW(accumulate(f, out.data(), bad), std::domain_error);
            f.norms(0.5f, std::numeric_limits<float>::max());
            EXPECT_THROW(accumulate(f, out.data(), 2.0f), std::domain_error);
            EXPECT_EQ(out, original);
        }
}
} // namespace
} // namespace llaminar2::test
