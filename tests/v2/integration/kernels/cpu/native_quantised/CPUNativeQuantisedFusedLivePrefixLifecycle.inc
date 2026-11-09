/**
 * @file CPUNativeQuantisedFusedLivePrefixLifecycle.inc
 * @brief Actual batched projections retain only the current live descriptor prefix.
 *
 * Authored native lifecycle inputs reuse prepared matrices and output capacity
 * across large, small and reset invocations. Invalid inactive descriptors and
 * poisoned inactive output rows may never enter planning, LUT access or work.
 * The public serial kernel remains the independent byte oracle; this fixture
 * implements no alternative dot arithmetic and makes no timing assertion.
 */
#include <utility>

namespace llaminar2::test
{
namespace
{
/** @brief Check all-format native prefix reuse in standalone and existing teams. */
void proveLiveFusedPrefixLifecycle()
{
    using namespace cpu::native_quantised;
    using Kernel = CPUNativeQuantisedGemmKernel;
    using Desc = Kernel::BatchedPrequantizedProjectionDesc;
    constexpr int projections = 37, max_rows = 7, n = 80, k = 1024, stride = n + 7;
    constexpr int blocks = k / Q8_1Block::BLOCK_SIZE;
    constexpr float guard = -12345.0f;
    constexpr auto policy = CPUProjectionNumericalPolicy::GPUAlignedExpert;
    FusedCompletionScope restore(/*workers=*/1, std::nullopt);
    const auto &formats = quantizedVerifierFormats();
    std::vector<float> source(static_cast<size_t>(projections) * max_rows * k);
    for (size_t index = 0; index < source.size(); ++index)
        source[index] = float(int((index * 37 + 13) % 257) - 128) * 0.003125f;
    std::vector<Q8_1Block> q8(static_cast<size_t>(projections) * max_rows * blocks);
    quantize_activations_to_q8_1(source.data(), q8.data(), projections * max_rows, k, blocks, policy);

    for (size_t format = 0; format < formats.size(); ++format)
    {
        SCOPED_TRACE(formats[format].label);
        omp_set_num_threads(1);
        auto weights0 = formats[format].create({size_t(n), size_t(k)}, 9109 + format);
        auto weights1 = formats[(format + 1) % formats.size()].create({size_t(n), size_t(k)}, 9127 + format);
        Kernel kernel0(weights0.get(), 0, -1, policy), kernel1(weights1.get(), 0, -1, policy);
        ASSERT_TRUE(kernel0.isValid() && kernel1.isValid());
        std::array<Kernel *, 2> kernels{&kernel0, &kernel1};
        std::array<std::vector<float>, projections> outputs, serial;
        for (int projection = 0; projection < projections; ++projection)
        {
            outputs[projection].assign(size_t(max_rows) * stride + 17u, guard);
            serial[projection] = outputs[projection];
            auto &packed = kernels[projection % 2]->packedWeights();
            NativeQuantisedTestPartialStorage partial(packed, 1);
            for (int row = 0; row < max_rows; ++row)
                gemv_native_quantised_preq(packed,
                    q8.data() + size_t(projection * max_rows + row) * blocks,
                    serial[projection].data() + size_t(row) * stride, partial.span());
        }
        for (int workers : {1, 3, 7})
        {
            omp_set_num_threads(workers);
            // Inactive capacity is deliberately malformed, rather than a
            // second valid vector whose accidental execution could go unseen.
            std::array<Desc, projections + 4> descriptors;
            for (const auto [live, rows] : std::array<std::pair<int, int>, 4>{{
                    {projections, max_rows}, {1, 1}, {projections, 3}, {3, 2}}})
            {
                SCOPED_TRACE("workers=" + std::to_string(workers) + " live=" +
                    std::to_string(live) + " rows=" + std::to_string(rows));
                for (auto &output : outputs)
                    std::fill(output.begin(), output.end(), guard);
                for (size_t projection = 0; projection < descriptors.size(); ++projection)
                {
                    descriptors[projection] = {.kernel = nullptr, .rows = -11, .n = -13};
                    if (projection < static_cast<size_t>(live))
                        descriptors[projection] = {
                            .kernel = kernels[projection % 2],
                            .input_q8 = q8.data() + projection * max_rows * blocks,
                            .output = outputs[projection].data(),
                            .rows = rows, .n = n, .ldc = stride,
                            .purpose = ProjectionRowsPurpose::Prefill};
                }
                NativeQuantisedTestPartialStorage partial(
                    NativeQuantisedTestPartialStorage::bundleFloats(descriptors.data(), live, 1));
                const auto complete = [&]()
                {
                    for (int projection = 0; projection < projections; ++projection)
                        for (size_t index = 0; index < outputs[projection].size(); ++index)
                        {
                            const bool active = projection < live && index / stride < static_cast<size_t>(rows)
                                && index % stride < static_cast<size_t>(n);
                            const float expected = active ? serial[projection][index] : guard;
                            if (std::memcmp(&outputs[projection][index], &expected, sizeof(float)) != 0)
                                return false;
                        }
                    return true;
                };
                const auto invoke = [&]()
                {
                    return Kernel::multiply_batched_preq_decode_equivalent(
                        descriptors.data(), live, k, partial.span());
                };
                ASSERT_TRUE(invoke());
                ASSERT_TRUE(complete());
                for (auto &output : outputs)
                    std::fill(output.begin(), output.end(), guard);
                std::array<int, 7> success;
                success.fill(1);
#pragma omp parallel
                {
                    const int worker = omp_get_thread_num();
                    if (!invoke() || !complete()) success[worker] = 0;
                }
                EXPECT_TRUE(std::all_of(success.begin(), success.end(), [](int value) { return value == 1; }));
            }
        }
    }
}
} // namespace

/** @test Live-only metadata/LUT storage ignores invalid inactive capacity after reset. */
TEST(CPUNativeQuantisedFusedCompletion, LivePrefixStorageLargeSmallReset)
{
    proveLiveFusedPrefixLifecycle();
}
} // namespace llaminar2::test
