/**
 * @file Test__CPUTQRotation.cpp
 * @brief Original-matrix rotation bounds, native arithmetic and worker-ownership regressions.
 *
 * Diagnostic finite matrices expose every row/K fringe without model loading or
 * expensive repeated QR generation. Native forward references model independent
 * memory-rounded two-chain dots; transpose references retain every increasing-row
 * FMA. Public same-backend serial calls certify head ownership, not full FA2/KV
 * state or cross-standard-library random-matrix identity.
 */
#include "kernels/cpu/turboquant/TurboQuantRotation.h"
#include "utils/CPUExplicitRoundingOracle.h"
#include <gtest/gtest.h>
#include <omp.h>
#include <array>
#include <bit>
#include <cstring>
#include <limits>
#include <type_traits>

namespace llaminar2::test
{
namespace
{
static_assert(!std::is_default_constructible_v<rotation_detail::Invocation>);
/** @brief Author finite original coefficient storage before invocation.
 * @param dim Complete physical square dimension; @param seed Diagnostic coefficient identity.
 * @return One original row-major owner with no alternate transpose/head representation.
 */
TurboQuantRotation matrix(int dim, int seed)
{
    TurboQuantRotation result{.dim=dim, .matrix=std::vector<float>(std::size_t(dim) * dim), .seed=std::uint64_t(seed)};
    for (std::size_t i = 0; i < result.matrix.size(); ++i)
        result.matrix[i] = float(int((i * 131 + seed * 17) % 257) - 128) * 0.001731f;
    return result;
}
/** @brief Check complete output and retained trailing canaries.
 * @param actual Original authored bank; @param expected Independently authored reference bank.
 */
void bytes(const std::vector<float> &actual, const std::vector<float> &expected)
{
    ASSERT_EQ(actual.size(), expected.size());
    EXPECT_EQ(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)), 0);
}
#if LLAMINAR_COMPILED_WITH_NEON
/** @brief Independently evaluate one original matrix row in the native vector primitive's exact tree.
 * @param rot Original matrix; @param input Original immutable query; @param row Original output row.
 * @return Same-backend exact reference word, including bounded K tails.
 */
float referenceForward(const TurboQuantRotation &rot, const float *input, int row)
{
    float lo[4]{}, hi[4]{};
    int k = 0;
    for (; k + 8 <= rot.dim; k += 8)
        for (int lane = 0; lane < 4; ++lane)
        {
            volatile float a = std::fma(input[k + lane], rot.at(row, k + lane), lo[lane]);
            volatile float b = std::fma(input[k + 4 + lane], rot.at(row, k + 4 + lane), hi[lane]);
            lo[lane] = a; hi[lane] = b;
        }
    float sum[4]; for (int lane = 0; lane < 4; ++lane) sum[lane] = memoryRoundedAdd(lo[lane], hi[lane]);
    float result = memoryRoundedAdd(memoryRoundedAdd(sum[0], sum[1]), memoryRoundedAdd(sum[2], sum[3]));
    for (; k < rot.dim; ++k) { volatile float value = std::fma(input[k], rot.at(row, k), result); result = value; }
    return result;
}
/** @brief Independently retain increasing-row FMA order for one transposed output coordinate.
 * @param rot Original matrix; @param input Original input; @param column Original output column.
 * @return Ordered transpose reference word.
 */
float referenceTranspose(const TurboQuantRotation &rot, const float *input, int column)
{
    float result = 0;
    for (int row = 0; row < rot.dim; ++row)
    { volatile float value = std::fma(input[row], rot.at(row, column), result); result = value; }
    return result;
}
#endif

/** @test Public forward/transpose kernels never cross original row/K/output fringes. */
TEST(CPUTQRotation, OriginalGeometryFringesRetainArithmeticAndCanaries)
{
    std::vector<int> dimensions;
    for (int d = 0; d <= 33; ++d) dimensions.push_back(d);
    for (int d : {63,64,65,127,128,129,255,256,257}) dimensions.push_back(d);
    for (int dim : dimensions)
        for (int seed = 0; seed < 8; ++seed)
        {
            auto rot = matrix(dim, seed); const auto original = rot.matrix;
            std::vector<float> input(dim + 4, 919.25f), forward(dim + 4, 919.25f), transposed = forward;
            for (int i = 0; i < dim; ++i) input[i] = float((i * 73 + seed * 13) % 251 - 125) * 0.017123f;
            const auto original_input = input;
            apply_rotation(rot, input.data(), forward.data());
            apply_rotation_transpose(rot, input.data(), transposed.data());
#if LLAMINAR_COMPILED_WITH_NEON
            auto expected_forward = std::vector<float>(dim + 4, 919.25f), expected_transpose = expected_forward;
            for (int i = 0; i < dim; ++i)
            { expected_forward[i] = referenceForward(rot, input.data(), i); expected_transpose[i] = referenceTranspose(rot, input.data(), i); }
            bytes(forward, expected_forward); bytes(transposed, expected_transpose);
#else
            for (int i = 0; i < dim; ++i)
            {
                double f = 0, t = 0;
                for (int j = 0; j < dim; ++j) { f += double(rot.at(i,j)) * input[j]; t += double(rot.at(j,i)) * input[j]; }
                EXPECT_NEAR(forward[i], f, 1e-5 * (1 + std::abs(f)));
                EXPECT_NEAR(transposed[i], t, 1e-5 * (1 + std::abs(t)));
            }
            for (int i = dim; i < dim + 4; ++i) { EXPECT_EQ(forward[i], 919.25f); EXPECT_EQ(transposed[i], 919.25f); }
#endif
            EXPECT_EQ(rot.matrix, original); bytes(input, original_input);
        }
}

/** @brief Restore original OpenMP policy on every test exit. */
class TeamBudget final
{
public:
    /** @brief Retain the actual caller budget and admit exact requested teams. */
    TeamBudget() : threads_(omp_get_max_threads()), dynamic_(omp_get_dynamic()) { omp_set_dynamic(0); }
    /** @brief Restore unchanged caller policy, including on assertions. */
    ~TeamBudget() { omp_set_num_threads(threads_); omp_set_dynamic(dynamic_); }
    /** @return Original actual positive worker capacity. */
    int threads() const noexcept { return threads_; }
private:
    int threads_, dynamic_;
};

/** @test Grouping original head invocations never changes matrix/input/output ownership or native bytes. */
TEST(CPUTQRotation, AllPositiveWorkersRetainOriginalSerialHeadBytes)
{
    const TeamBudget budget;
    for (int dim : {64,128,256})
        for (int rows : {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,31,33})
        {
            const int heads = rows * 3;
            const std::array<TurboQuantRotation,3> matrices = {matrix(dim,1),matrix(dim,7),matrix(dim,31)};
            std::vector<float> input(std::size_t(heads) * dim + 4, 919.25f), expected_f(input.size(), 919.25f), expected_t = expected_f;
            for (std::size_t i = 0; i < input.size() - 4; ++i) input[i] = float(int(i % 257) - 128) * 0.003173f;
            for (int head = 0; head < heads; ++head)
            {
                const auto offset = std::size_t(head) * dim;
                apply_rotation(matrices[head % 3], input.data() + offset, expected_f.data() + offset);
                apply_rotation_transpose(matrices[head % 3], expected_f.data() + offset, expected_t.data() + offset);
            }
            for (int workers = 1; workers <= budget.threads(); ++workers)
            {
                std::vector<float> actual_f(input.size(), 919.25f), actual_t = actual_f;
                int observed = 0;
#pragma omp parallel num_threads(workers)
                {
#pragma omp single
                    observed = omp_get_num_threads();
#pragma omp for schedule(static)
                    for (int head = 0; head < heads; ++head)
                    {
                        const auto offset = std::size_t(head) * dim;
                        apply_rotation(matrices[head % 3], input.data() + offset, actual_f.data() + offset);
                        apply_rotation_transpose(matrices[head % 3], actual_f.data() + offset, actual_t.data() + offset);
                    }
                }
                ASSERT_EQ(observed, workers); bytes(actual_f, expected_f); bytes(actual_t, expected_t);
            }
        }
}

/** @test Malformed or overlapping original operands fail before authoring any result. */
TEST(CPUTQRotation, OriginalStorageAndGeometryRejectBeforeMutation)
{
    auto rot = matrix(64, 1);
    rot.matrix.reserve(rot.matrix.size() + 64);
    const auto original = rot.matrix;
    std::vector<float> input(68, 0.125f), output(68, 919.25f);
    const auto original_input = input, original_output = output;
    for (bool transpose : {false,true})
    {
        const auto invoke = [&](const TurboQuantRotation &r, const float *x, float *y) {
            if (transpose) apply_rotation_transpose(r,x,y); else apply_rotation(r,x,y);
        };
        EXPECT_THROW(invoke(rot,input.data(),input.data()), std::invalid_argument);
        EXPECT_THROW(invoke(rot,input.data(),input.data()+1), std::invalid_argument);
        EXPECT_THROW(invoke(rot,input.data(),rot.matrix.data()+1), std::invalid_argument);
        EXPECT_THROW(invoke(rot,input.data(),rot.matrix.data()+rot.matrix.size()), std::invalid_argument);
        EXPECT_THROW(invoke(rot,input.data(),reinterpret_cast<float *>(&rot)), std::invalid_argument);
        EXPECT_THROW(invoke(rot,nullptr,output.data()), std::invalid_argument);
        EXPECT_THROW(invoke(rot,input.data(),nullptr), std::invalid_argument);
        TurboQuantRotation malformed{.dim=-1,.matrix={},.seed=1};
        EXPECT_THROW(invoke(malformed,nullptr,nullptr), std::invalid_argument);
        EXPECT_THROW(verify_orthogonality(malformed), std::invalid_argument);
        malformed.dim=64; malformed.matrix.resize(4095);
        EXPECT_THROW(invoke(malformed,input.data(),output.data()), std::invalid_argument);
        malformed.matrix.resize(4097);
        EXPECT_THROW(invoke(malformed,input.data(),output.data()), std::invalid_argument);
        malformed.dim=std::numeric_limits<int>::max(); malformed.matrix.clear();
        EXPECT_THROW(invoke(malformed,input.data(),output.data()), std::invalid_argument);
        const TurboQuantRotation empty{.dim=0,.matrix={},.seed=1};
        EXPECT_NO_THROW(invoke(empty,nullptr,nullptr));
    }
    EXPECT_EQ(rot.matrix,original); bytes(input,original_input); bytes(output,original_output);
    EXPECT_THROW(generate_rotation_matrix(-1), std::invalid_argument);
    EXPECT_THROW(generate_rotation_matrix(0), std::invalid_argument);
}

/** @test Cold matrix verification cannot certify NaN, infinity or finite-overflow observations as zero error. */
TEST(CPUTQRotation, NonfiniteVerifierObservationsFailClosed)
{
    for (int dim : {1,2,3,64,128,256})
        for (int row : {0,dim/2,dim-1})
            for (float invalid : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
                                  -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::max()})
            {
                TurboQuantRotation rot{.dim=dim,.matrix=std::vector<float>(std::size_t(dim)*dim,0.0f),.seed=1};
                for (int i=0;i<dim;++i) rot.matrix[std::size_t(i)*dim+i]=1.0f;
                rot.matrix[std::size_t(row)*dim+row]=invalid;
                const auto original = rot.matrix;
                EXPECT_THROW(verify_orthogonality(rot),std::domain_error);
                EXPECT_EQ(std::memcmp(rot.matrix.data(),original.data(),original.size()*sizeof(float)),0);
            }
}

/** @test The real cold generator still supplies the original deterministic same-library matrix program. */
TEST(CPUTQRotation, ExistingGeneratorRetainsSameSeedAndOrthogonality)
{
    const auto a = generate_rotation_matrix(64,31), b = generate_rotation_matrix(64,31);
    EXPECT_EQ(a.matrix,b.matrix); EXPECT_EQ(a.seed,b.seed);
    EXPECT_LT(verify_orthogonality(a),1e-4f);
    std::vector<float> input(64), rotated(64), recovered(64);
    for (int i=0;i<64;++i) input[i] = float(i-32)*0.0173f;
    apply_rotation(a,input.data(),rotated.data());
    apply_rotation_transpose(a,rotated.data(),recovered.data());
    for(int i=0;i<64;++i) EXPECT_NEAR(recovered[i],input[i],2e-5f);
}
} // namespace
} // namespace llaminar2::test
