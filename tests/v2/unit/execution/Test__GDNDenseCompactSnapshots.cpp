/**
 * @file Test__GDNDenseCompactSnapshots.cpp
 * @brief Device-free exact-correction retention, selection and storage contracts.
 *
 * Authored normalized K/delta/decay records advance an independent full Dense
 * snapshot sequence. A separate bounded decoder reconstructs every prefix from
 * retained checkpoints. This proves the host storage/selection contract only:
 * it neither computes GDN gates/dots nor certifies CUDA FTZ, NaN or codegen.
 * No device, model, allocation authority or production publisher is created.
 */
#include "kernels/gdn/GDNDenseCompactSnapshots.h"
#include "kernels/gdn/GDNCUDADenseCapturePlan.h"
#include "kernels/gdn/GDNPreparedRetainedInvocation.h"
#include <functional>
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{
using namespace llaminar2;
using namespace llaminar2::gdn;
constexpr auto arithmetic = GDNDenseCompactArithmetic::CUDAEightPartFusedUpdateV1;
constexpr auto key_major = GDNRecurrenceStateLayout::KeyMajorValueContiguous;
constexpr auto value_major = GDNRecurrenceStateLayout::ValueMajorKeyContiguous;
constexpr std::uint64_t generation = 0x12345678abcdef09ULL;
constexpr float poison = -12345.25f;

/** @brief Test-only aligned backing with owned guards; never a PMA substitute. */
class GuardedCompact final
{
  public:
    /** @param count Live test view elements, excluding both physical guards. */
    explicit GuardedCompact(std::size_t count) : allocation_(count + 32, poison), count_(count)
    {
        const auto start = reinterpret_cast<std::uintptr_t>(allocation_.data() + 1);
        data_ = reinterpret_cast<float *>((start + 63) & ~std::uintptr_t(63));
    }
    /** @return Aligned view whose allocation outlives every borrowed descriptor. */
    float *data() { return data_; }
    /** @return Read-only aligned view for independent decoding. */
    const float *data() const { return data_; }
    /** @return Whether both adjacent guards retain their original bits. */
    bool guards() const { return data_[-1] == poison && data_[count_] == poison; }
  private:
    std::vector<float> allocation_;
    std::size_t count_;
    float *data_;
};

/** @brief Access one protocol word without pointer-object aliasing.
 * @param head Compact head base.
 * @param word Word coordinate.
 * @return Exact raw32-bit protocol value. */
std::uint32_t wordAt(const float *head, unsigned word)
{
    std::uint32_t value;
    std::memcpy(&value, head + word, sizeof(value));
    return value;
}
/** @brief Independently decode the authored seven-word protocol.
 * @param head Completed compact head.
 * @return Raw full-width words, before enum narrowing. */
GDNDenseCompactPublication publicationAt(const float *head)
{
    return {wordAt(head, 0), wordAt(head, 1), wordAt(head, 2),
        std::uint64_t(wordAt(head, 3)) | (std::uint64_t(wordAt(head, 4)) << 32),
        wordAt(head, 5), wordAt(head, 6)};
}
/** @brief Write one raw metadata witness without treating integers as floats.
 * @param head Compact head base.
 * @param word Protocol coordinate.
 * @param value Exact bits to publish. */
void writeWord(float *head, unsigned word, std::uint32_t value)
{
    std::memcpy(head + word, &value, sizeof(value));
}
/** @brief Independent Dense physical coordinate, unrelated to the production update helper.
 * @param layout Explicit matrix axes.
 * @param dk Key width.
 * @param dv Value width.
 * @param k Key coordinate.
 * @param v Value coordinate.
 * @return One matrix element offset. */
std::size_t matrixIndex(GDNRecurrenceStateLayout layout, int dk, int dv, int k, int v)
{
    return layout == key_major ? std::size_t(k) * dv + v : std::size_t(v) * dk + k;
}

/** @brief Author known corrections and full sequential snapshots without GDN forward math. */
struct AuthoredCapture final
{
    GDNDenseCompactGeometry geometry;
    GDNRecurrenceStateLayout layout;
    std::vector<float> initial;
    GuardedCompact compact;
    std::vector<std::vector<float>> snapshots;
    std::vector<int> live;

    /** @brief Retain exact finite update operands and independently materialized snapshots.
     * @param shape Checked immutable storage geometry.
     * @param axes Explicit Dense axes.
     * @param lengths Per-request completed prefixes.
     * @details Every capacity-only/padding word remains poisoned. cp0 is authored
     * even for live0; no original-bank read is needed by a later decoder. */
    AuthoredCapture(GDNDenseCompactGeometry shape, GDNRecurrenceStateLayout axes, std::vector<int> lengths)
        : geometry(shape), layout(axes), initial(shape.denseBankFloats()), compact(shape.bankFloats()),
          snapshots(std::size_t(shape.rowsPerRequest()) + 1, initial), live(std::move(lengths))
    {
        const int dk = shape.keyWidth(), dv = shape.valueWidth();
        for (std::size_t i = 0; i < initial.size(); ++i)
            initial[i] = float(int((i * 37) % 97) - 48) / 128.0f;
        snapshots[0] = initial;
        for (int row = 0; row < shape.rowsPerRequest(); ++row)
        {
            snapshots[row + 1] = snapshots[row];
            for (int request = 0; request < shape.requests(); ++request)
                for (int head = 0; head < shape.heads(); ++head)
                {
                    if (row >= live[request]) continue;
                    const auto index = std::size_t(request) * shape.heads() + head;
                    float *record = compact.data() + index * shape.headStrideFloats() +
                        shape.recordsOffsetFloats() + std::size_t(row) * shape.recordStrideFloats();
                    for (int k = 0; k < dk; ++k)
                        record[k] = float(((k * 11 + row * 7 + head * 3 + request) % 61) - 30) / 128.0f;
                    for (int v = 0; v < dv; ++v)
                        record[dk + v] = float(((v * 13 + row * 19 + request * 5 + head) % 79) - 39) / 64.0f;
                    const float decay = 0.8125f + float((row + head + request) % 5) / 32.0f;
                    record[dk + dv] = decay;
                    // Full sequential snapshots execute each cell once per row.
                    // Volatile spells the independently rounded FP32 multiply.
                    for (int k = 0; k < dk; ++k)
                        for (int v = 0; v < dv; ++v)
                        {
                            const auto cell = index * shape.matrixFloats() + matrixIndex(axes, dk, dv, k, v);
                            volatile float decayed = snapshots[row][cell] * decay;
                            snapshots[row + 1][cell] = std::fma(record[k], record[dk + v], float(decayed));
                        }
                }
        }
        for (int request = 0; request < shape.requests(); ++request)
            for (int head = 0; head < shape.heads(); ++head)
            {
                const auto index = std::size_t(request) * shape.heads() + head;
                float *base = compact.data() + index * shape.headStrideFloats();
                const std::uint32_t words[]{0x44464331u, unsigned(live[request]), unsigned(live[request]),
                    std::uint32_t(generation), std::uint32_t(generation >> 32), std::uint32_t(axes), 1};
                std::memcpy(base, words, sizeof(words));
                for (int cp = 0; cp <= live[request] / 8; ++cp)
                    std::copy_n(snapshots[cp * 8].data() + index * shape.matrixFloats(), shape.matrixFloats(),
                        base + shape.checkpointOffsetFloats() + std::size_t(cp) * shape.checkpointStrideFloats());
            }
    }
    /** @return Immutable transaction descriptor; the host owns these test buffers. */
    GDNDenseCompactTransaction transaction()
    {
        return {geometry, initial.data(), compact.data(), geometry.bankFloats(), generation, layout, arithmetic};
    }
};

/** @brief Test decoder: validate ALL headers/counts before mutating any Dense destination.
 * @param capture Authored immutable transaction.
 * @param accepted One accepted count per independent request.
 * @param destination Complete Dense bank.
 * @param expected_generation Explicit expected immutable generation.
 * @return False without writes for any malformed protocol/count; no fallback.
 * @details This host oracle tests selection/retention only. It decodes at most
 * seven logged cell updates and never reads original Dense, raw V, beta or Q. */
bool decode(const AuthoredCapture &capture, const std::vector<int> &accepted,
    float *destination, std::uint64_t expected_generation = generation)
{
    const auto &g = capture.geometry;
    if (accepted.size() != std::size_t(g.requests())) return false;
    std::vector<GDNDenseCompactSelection> selected(std::size_t(g.requests()) * g.heads());
    for (int r = 0; r < g.requests(); ++r)
    {
        std::uint32_t live = 0;
        for (int h = 0; h < g.heads(); ++h)
        {
            const auto index = std::size_t(r) * g.heads() + h;
            const auto publication = publicationAt(capture.compact.data() + index * g.headStrideFloats());
            if ((h && publication.published_rows != live) ||
                !g.select(publication, expected_generation, capture.layout, arithmetic, accepted[r], selected[index]))
                return false;
            live = publication.published_rows;
        }
    }
    for (int r = 0; r < g.requests(); ++r)
        for (int h = 0; h < g.heads(); ++h)
        {
            const auto index = std::size_t(r) * g.heads() + h;
            const auto &selection = selected[index];
            const float *base = capture.compact.data() + index * g.headStrideFloats();
            const float *checkpoint = base + g.checkpointOffsetFloats() + selection.checkpoint * g.checkpointStrideFloats();
            // Different cell traversal from the full sequential author. Each
            // cell still sees precisely the original chronological update order.
            for (int v = 0; v < g.valueWidth(); ++v)
                for (int k = 0; k < g.keyWidth(); ++k)
                {
                    const auto cell = matrixIndex(capture.layout, g.keyWidth(), g.valueWidth(), k, v);
                    float state = checkpoint[cell];
                    for (unsigned step = 0; step < selection.update_count; ++step)
                    {
                        const float *record = base + g.recordsOffsetFloats() +
                            (selection.first_record + step) * g.recordStrideFloats();
                        volatile float decayed = state * record[g.recordDecayOffsetFloats()];
                        state = std::fma(record[k], record[g.recordCorrectionOffsetFloats() + v], float(decayed));
                    }
                    destination[index * g.matrixFloats() + cell] = state;
                }
        }
    return true;
}
}

TEST(GDNDenseCompactStorage, LiteralGeometryAndLiveSpansExcludePaddingAndUnusedCapacity)
{
    const GDNDenseCompactGeometry g(2, 64, 17, 3, 31);
    EXPECT_EQ(g.matrixFloats(), 1088u);
    EXPECT_EQ(g.checkpointStrideFloats(), 1088u);
    EXPECT_EQ(g.checkpointCount(), 4u);
    EXPECT_EQ(g.recordFloats(), 82u);
    EXPECT_EQ(g.recordStrideFloats(), 96u);
    EXPECT_EQ(g.recordsOffsetFloats(), 4368u);
    EXPECT_EQ(g.headStrideFloats(), 7344u);
    EXPECT_EQ(g.bankFloats(), 44064u);
    EXPECT_EQ(g.bankBytes(), 176256u);
    EXPECT_EQ(g.denseBankFloats(), 6528u);
    EXPECT_EQ(g.denseBankBytes(), 26112u);
    EXPECT_EQ(g.recordCorrectionOffsetFloats(), 64u);
    EXPECT_EQ(g.recordDecayOffsetFloats(), 81u);
    for (const unsigned live : {0u, 1u, 7u, 8u, 9u, 15u, 16u, 17u, 31u})
    {
        AuthoredCapture capture(g, value_major, {int(live), 0, int(live / 2)});
        std::vector<bool> covered(g.headStrideFloats(), false);
        const auto *base = capture.compact.data();
        std::size_t total = 0;
        for (std::size_t index = 0;; ++index)
        {
            GDNDenseCompactByteSpan span;
            if (!g.publishedSpanFor(live, index, span)) break;
            ASSERT_EQ(span.offset_bytes % 4, 0u);
            ASSERT_EQ(span.live_bytes % 4, 0u);
            ASSERT_LE(span.offset_bytes + span.live_bytes, g.headStrideFloats() * 4);
            total += span.live_bytes;
            for (std::size_t word = span.offset_bytes / 4; word < (span.offset_bytes + span.live_bytes) / 4; ++word)
            {
                ASSERT_FALSE(covered[word]);
                covered[word] = true;
            }
        }
        EXPECT_EQ(total, 28u + (1u + live / 8) * 1088u * 4u + live * 82u * 4u);
        for (std::size_t word = 0; word < covered.size(); ++word)
            if (!covered[word]) EXPECT_EQ(base[word], poison) << word;
        EXPECT_TRUE(capture.compact.guards());
    }
    GDNDenseCompactByteSpan span{99, 99};
    EXPECT_FALSE(g.publishedSpanFor(32, 0, span));
    EXPECT_EQ(span.offset_bytes, 0u);
    EXPECT_EQ(span.live_bytes, 0u);
    EXPECT_FALSE(g.publishedSpanFor(17, std::numeric_limits<std::size_t>::max(), span));
    EXPECT_EQ(span.live_bytes, 0u);
}

TEST(GDNDenseCompactStorage, AuthoredCorrectionsMatchEverySequentialPrefixAndRevisions)
{
    for (const auto axes : {key_major, value_major})
        for (const int dk : {64, 128})
            for (const int live : {0, 1, 7, 8, 9, 15, 16, 17, 31})
            {
                const GDNDenseCompactGeometry g(2, dk, dk == 64 ? 3 : 17, 3, 31);
                AuthoredCapture capture(g, axes, {live, 0, live / 2});
                const std::vector<float> retained(capture.compact.data(), capture.compact.data() + g.bankFloats());
                // Destroy original data before ANY decode, including live0.
                std::fill(capture.initial.begin(), capture.initial.end(), 9876.5f);
                std::vector<float> destination(g.denseBankFloats() + 2, poison);
                std::vector<int> revisions;
                for (int a = 0; a <= live; ++a) revisions.push_back(a);
                revisions.insert(revisions.end(), {live, live / 2, 0, live});
                for (const int a : revisions)
                {
                    SCOPED_TRACE(::testing::Message() << "DK=" << dk << " live=" << live << " A=" << a);
                    std::fill(destination.begin(), destination.end(), poison);
                    const std::vector<int> accepted{a, 0, std::min(a, live / 2)};
                    ASSERT_TRUE(decode(capture, accepted, destination.data() + 1));
                    for (int r = 0; r < g.requests(); ++r)
                    {
                        const auto offset = std::size_t(r) * g.heads() * g.matrixFloats();
                        ASSERT_EQ(0, std::memcmp(destination.data() + 1 + offset,
                            capture.snapshots[accepted[r]].data() + offset,
                            std::size_t(g.heads()) * g.matrixFloats() * sizeof(float)));
                    }
                    EXPECT_EQ(destination.front(), poison);
                    EXPECT_EQ(destination.back(), poison);
                    EXPECT_EQ(0, std::memcmp(retained.data(), capture.compact.data(), g.bankBytes()));
                }
                // Exact destination==original remains legal, without reading its poisoned old bytes.
                ASSERT_TRUE(decode(capture, {live, 0, live / 2}, capture.initial.data()));
                EXPECT_EQ(0, std::memcmp(capture.initial.data(), capture.snapshots[live].data(), g.denseBankBytes()));
                EXPECT_TRUE(capture.compact.guards());
            }
}

TEST(GDNDenseCompactStorage, MalformedLateProtocolOrCountRejectsBeforeAnyDestinationWrite)
{
    const GDNDenseCompactGeometry g(2, 64, 3, 3, 17);
    AuthoredCapture capture(g, key_major, {17, 0, 9});
    const std::vector<float> original(capture.compact.data(), capture.compact.data() + g.bankFloats());
    std::vector<float> destination(g.denseBankFloats(), poison);
    const auto unchanged = destination;
    float *late = capture.compact.data() + 5 * g.headStrideFloats();
    for (const auto [word, bad] : std::array<std::array<std::uint32_t, 2>, 10>{{
        {0, 0}, {0, 0x45504f43u}, {1, 18}, {2, 8}, {3, 0}, {4, 0}, {5, 1}, {5, 0x100}, {6, 0x101}, {6, 0}}})
    {
        SCOPED_TRACE(word);
        std::copy(original.begin(), original.end(), capture.compact.data());
        writeWord(late, word, bad);
        EXPECT_FALSE(decode(capture, {17, 0, 9}, destination.data()));
        EXPECT_EQ(destination, unchanged);
    }
    std::copy(original.begin(), original.end(), capture.compact.data());
    const std::vector<std::vector<int>> invalid_accepts{{-1, 0, 9}, {18, 0, 9}, {17, 1, 9}, {17, 0, 10}, {17, 0}};
    for (const auto &accepted : invalid_accepts)
    {
        EXPECT_FALSE(decode(capture, accepted, destination.data()));
        EXPECT_EQ(destination, unchanged);
    }
    EXPECT_FALSE(decode(capture, {17, 0, 9}, destination.data(), 0));
    EXPECT_FALSE(decode(capture, {17, 0, 9}, destination.data(), generation + 1));
    EXPECT_EQ(destination, unchanged);
    // A valid, internally consistent but different head count is still foreign
    // to the single request-owned length; no early head may be written.
    writeWord(late, 1, 8); writeWord(late, 2, 8);
    EXPECT_FALSE(decode(capture, {17, 0, 8}, destination.data()));
    EXPECT_EQ(destination, unchanged);
}

TEST(GDNDenseCompactStorage, ColdStorageAdmissionRejectsAliasesAndIncompleteSourceContracts)
{
    const GDNDenseCompactGeometry g(2, 64, 17, 3, 9);
    AuthoredCapture capture(g, value_major, {9, 0, 4});
    auto transaction = capture.transaction();
    std::array<std::int32_t, 3> counts{9, 0, 4}, accepted{9, 0, 4};
    std::vector<float> merged(3 * 9 * (2 * 64 + 2 * 17)), alpha(g.gateFloats()), beta(g.gateFloats());
    std::vector<float> A(2), dt(2), output(g.outputFloats() + 1), terminal(g.denseBankFloats() + 1);
    MergedRecurrenceInput sources{.rows = DeviceRequestRowRanges::deviceCounted(3, 9, counts.data()),
        .key_heads = 1, .value_heads = 2, .key_width = 64, .value_width = 17, .global_value_offset = -1,
        .use_qk_l2norm = true, .merged = merged.data(), .alpha = alpha.data(), .beta_raw = beta.data(),
        .A_log = A.data(), .dt_bias = dt.data(), .output = output.data(), .initial_state = nullptr,
        .updated_state = nullptr, .snapshots = nullptr, .snapshot_stride_floats = 0, .max_snapshot_rows = 0,
        .recurrence_layout = value_major};
    const GDNDenseCompactCaptureInput valid{sources, transaction,
        GDNDenseCompactTerminalOutput::denseDestination(terminal.data())};
    ASSERT_TRUE(validGDNDenseCompactTransactionStorage(transaction));
    ASSERT_TRUE(validCUDADenseCompactCaptureStorage(valid));
    static_assert(!std::is_default_constructible_v<GDNDenseCompactTerminalOutput>);
    static_assert(!std::is_constructible_v<GDNDenseCompactTerminalOutput, float *, GDNDenseCompactTerminalKind>);
    EXPECT_EQ(valid.terminal.kind(), GDNDenseCompactTerminalKind::DenseDestination);
    auto retained_only = valid;
    retained_only.terminal = GDNDenseCompactTerminalOutput::retainedPrefixesOnly();
    EXPECT_EQ(retained_only.terminal.kind(), GDNDenseCompactTerminalKind::RetainedPrefixesOnly);
    EXPECT_EQ(retained_only.terminal.destination(), nullptr);
    EXPECT_TRUE(validCUDADenseCompactCaptureStorage(retained_only));
    EXPECT_FALSE(validROCmDenseCompactCaptureStorage(retained_only));
    auto retained_rocm = retained_only;
    retained_rocm.transaction.arithmetic = GDNDenseCompactArithmetic::ROCmFourPartFusedUpdateV1;
    EXPECT_TRUE(validROCmDenseCompactCaptureStorage(retained_rocm));
    EXPECT_FALSE(validCUDADenseCompactCaptureStorage(retained_rocm));
    // No terminal span does not excuse a partial/foreign retained transaction
    // or any overlap of the remaining actual writes with their read authority.
    for (int defect = 0; defect < 4; ++defect)
    {
        auto bad = retained_only;
        if (defect == 0) --bad.transaction.storage_floats;
        if (defect == 1) bad.transaction.generation = 0;
        if (defect == 2) bad.sources.output = capture.initial.data();
        if (defect == 3) bad.sources.snapshots = terminal.data();
        EXPECT_FALSE(validCUDADenseCompactCaptureStorage(bad));
    }
    // One format can own two explicit backend arithmetic identities, but each
    // native admission must reject the other program instead of converting it.
    constexpr auto rocm = GDNDenseCompactArithmetic::ROCmFourPartFusedUpdateV1;
    auto rocm_input = valid;
    rocm_input.transaction.arithmetic = rocm;
    EXPECT_TRUE(validGDNDenseCompactTransactionStorage(rocm_input.transaction));
    EXPECT_TRUE(validROCmDenseCompactCaptureStorage(rocm_input));
    EXPECT_FALSE(validCUDADenseCompactCaptureStorage(rocm_input));
    EXPECT_FALSE(validROCmDenseCompactCaptureStorage(valid));
    GDNDenseCompactPublication rocm_publication{0x44464331u, 9, 9, generation,
        std::uint32_t(value_major), std::uint32_t(rocm)};
    EXPECT_TRUE(g.validPublication(rocm_publication, generation, value_major, rocm));
    EXPECT_FALSE(g.validPublication(rocm_publication, generation, value_major, arithmetic));
    for (int accepted_count = 0; accepted_count <= 9; ++accepted_count)
    {
        GDNDenseCompactSelection selection{};
        EXPECT_TRUE(g.select(rocm_publication, generation, value_major, rocm, accepted_count, selection));
        EXPECT_EQ(selection.checkpoint, std::size_t(accepted_count / 8));
        EXPECT_EQ(selection.first_record, std::size_t((accepted_count / 8) * 8));
        EXPECT_EQ(selection.update_count, unsigned(accepted_count % 8));
        EXPECT_FALSE(g.select(rocm_publication, generation, value_major, arithmetic, accepted_count, selection));
        EXPECT_EQ(selection.update_count, 0u);
    }
    const GDNDenseCompactAcceptedPrefixInput rocm_publish{
        rocm_input.transaction, GDNDenseAcceptedSelection::acceptedCounts(g, accepted.data(), accepted.size()), terminal.data()};
    EXPECT_TRUE(validROCmDenseCompactAcceptedPrefixStorage(rocm_publish));
    EXPECT_FALSE(validCUDADenseCompactAcceptedPrefixStorage(rocm_publish));
    EXPECT_FALSE(validROCmDenseCompactAcceptedPrefixStorage(
        {transaction, GDNDenseAcceptedSelection::acceptedCounts(g, accepted.data(), accepted.size()), terminal.data()}));
    auto input = valid;
    input.terminal = GDNDenseCompactTerminalOutput::denseDestination(capture.initial.data());
    EXPECT_TRUE(validCUDADenseCompactCaptureStorage(input));
    for (int field = 0; field < 5; ++field)
    {
        input = valid;
        if (field == 0) input.sources.initial_state = capture.initial.data();
        if (field == 1) input.sources.updated_state = terminal.data();
        if (field == 2) input.sources.snapshots = terminal.data();
        if (field == 3) input.sources.snapshot_stride_floats = 1;
        if (field == 4) input.sources.max_snapshot_rows = 1;
        EXPECT_FALSE(validCUDADenseCompactCaptureStorage(input));
    }
    input = valid; input.sources.key_width = 128;
    EXPECT_FALSE(validCUDADenseCompactCaptureStorage(input));
    input = valid; input.sources.recurrence_layout = key_major;
    EXPECT_FALSE(validCUDADenseCompactCaptureStorage(input));
    input = valid; input.sources.output = merged.data();
    EXPECT_FALSE(validCUDADenseCompactCaptureStorage(input));
    for (float *destination : {capture.initial.data() + 1, capture.compact.data(), alpha.data(), output.data()})
    {
        input = valid; input.terminal = GDNDenseCompactTerminalOutput::denseDestination(destination);
        EXPECT_FALSE(validCUDADenseCompactCaptureStorage(input));
    }
    EXPECT_THROW((void)GDNDenseCompactTerminalOutput::denseDestination(nullptr), std::invalid_argument);
    const GDNDenseCompactAcceptedPrefixInput publish{transaction, GDNDenseAcceptedSelection::acceptedCounts(g, accepted.data(), accepted.size()), terminal.data()};
    ASSERT_TRUE(validCUDADenseCompactAcceptedPrefixStorage(publish));
    auto publication = publish;
    EXPECT_THROW((void)GDNDenseAcceptedSelection::acceptedCounts(g, nullptr, accepted.size()), std::invalid_argument);
    publication = publish;
    publication.destination_state = capture.initial.data();
    EXPECT_TRUE(validCUDADenseCompactAcceptedPrefixStorage(publication));
    for (const auto *count_owner : {reinterpret_cast<const std::int32_t *>(capture.initial.data()),
        reinterpret_cast<const std::int32_t *>(capture.compact.data()),
        reinterpret_cast<const std::int32_t *>(terminal.data())})
    {
        publication = publish; publication.selection = GDNDenseAcceptedSelection::acceptedCounts(g, count_owner, accepted.size());
        EXPECT_FALSE(validCUDADenseCompactAcceptedPrefixStorage(publication));
    }
    for (float *destination : {capture.initial.data() + 1, capture.compact.data(),
        reinterpret_cast<float *>(accepted.data())})
    {
        publication = publish; publication.destination_state = destination;
        EXPECT_FALSE(validCUDADenseCompactAcceptedPrefixStorage(publication));
    }
    for (int failure = 0; failure < 7; ++failure)
    {
        auto bad = transaction;
        if (failure == 0) bad.generation = 0;
        if (failure == 1) bad.encoding = static_cast<GDNDenseCompactEncoding>(0);
        if (failure == 2) bad.arithmetic = static_cast<GDNDenseCompactArithmetic>(255);
        if (failure == 3) bad.layout = static_cast<GDNRecurrenceStateLayout>(2);
        if (failure == 4) --bad.storage_floats;
        if (failure == 5) ++bad.storage;
        if (failure == 6) bad.storage_floats = std::numeric_limits<std::size_t>::max();
        EXPECT_FALSE(validGDNDenseCompactTransactionStorage(bad));
    }
    auto overlapping = transaction;
    overlapping.initial_state = overlapping.storage;
    EXPECT_FALSE(validGDNDenseCompactTransactionStorage(overlapping));
    auto wrapping = transaction;
    wrapping.initial_state = reinterpret_cast<const float *>(std::numeric_limits<std::uintptr_t>::max() - 3);
    EXPECT_FALSE(validGDNDenseCompactTransactionStorage(wrapping));
    EXPECT_TRUE(capture.compact.guards());
}

TEST(GDNDenseCompactStorage, CheckedGeometryRejectsInvalidAndOverflowingNativeCoordinates)
{
    for (const auto dims : std::array<std::array<int, 5>, 7>{{
        {0, 64, 17, 1, 1}, {1, 0, 17, 1, 1}, {1, 513, 17, 1, 1},
        {1, 64, 0, 1, 1}, {1, 64, 513, 1, 1}, {1, 64, 17, 0, 1}, {1, 64, 17, 1, 0}}})
        EXPECT_THROW((void)GDNDenseCompactGeometry(dims[0], dims[1], dims[2], dims[3], dims[4]), std::invalid_argument);
    constexpr int maximum = std::numeric_limits<int>::max();
    EXPECT_THROW((void)GDNDenseCompactGeometry(2, 64, 17, maximum, 1), std::overflow_error);
    EXPECT_THROW((void)GDNDenseCompactGeometry(1, 64, 17, maximum, 2), std::overflow_error);
    EXPECT_THROW((void)GDNDenseCompactGeometry(maximum, 512, 512, 1, maximum), std::overflow_error);
    // General checked size_t geometry can be representable while the inherited
    // CUDA signed per-head matrix coordinate is not. No huge allocation occurs.
    const GDNDenseCompactGeometry too_wide(maximum / (64 * 17) + 1, 64, 17, 1, 1);
    const GDNDenseCompactTransaction unsupported{too_wide, reinterpret_cast<const float *>(0x10000),
        reinterpret_cast<float *>(0x10000000000ULL), too_wide.bankFloats(), generation, key_major, arithmetic};
    EXPECT_FALSE(validGDNDenseCompactTransactionStorage(unsupported));
}

/** @test Pure publication tiles partition every matrix exactly once, including partial tails. */
TEST(GDNDenseCompactStorage, PublicationTilesHaveExclusiveCompleteMatrixOwnership)
{
    for (int dk : {64, 128})
        for (int dv : {1, 17, 65, 128, 129, 512})
        {
            const GDNDenseCompactGeometry geometry(3, dk, dv, 3, 16);
            for (std::size_t width : {std::size_t{31}, std::size_t{4096},
                std::numeric_limits<std::size_t>::max()})
            {
                const auto count = geometry.matrixTileCount(width);
                ASSERT_GT(count, 0u);
                std::size_t covered = 0;
                for (std::size_t tile = 0; tile < count; ++tile)
                {
                    GDNDenseCompactMatrixTileSpan span{};
                    ASSERT_TRUE(geometry.matrixTileFor(tile, width, span));
                    EXPECT_EQ(span.first_element, covered);
                    ASSERT_GT(span.live_elements, 0u);
                    EXPECT_LE(span.live_elements, width);
                    EXPECT_LE(span.live_elements, geometry.matrixFloats() - covered);
                    covered += span.live_elements;
                }
                EXPECT_EQ(covered, geometry.matrixFloats());
                GDNDenseCompactMatrixTileSpan foreign{17, 19};
                EXPECT_FALSE(geometry.matrixTileFor(count, width, foreign));
                EXPECT_EQ(foreign.first_element, 0u);
                EXPECT_EQ(foreign.live_elements, 0u);
                foreign = {17, 19};
                EXPECT_FALSE(geometry.matrixTileFor(std::numeric_limits<std::size_t>::max(), width, foreign));
                EXPECT_EQ(foreign.first_element, 0u);
                EXPECT_EQ(foreign.live_elements, 0u);
            }
        }
    const GDNDenseCompactGeometry full(12, 128, 128, 1, 16);
    EXPECT_EQ(full.matrixTileCount(4096), 4u);
    const GDNDenseCompactGeometry tail(3, 128, 129, 3, 16);
    EXPECT_EQ(tail.matrixTileCount(4096), 5u);
    GDNDenseCompactMatrixTileSpan last{};
    ASSERT_TRUE(tail.matrixTileFor(4, 4096, last));
    EXPECT_EQ(last.first_element, 16384u);
    EXPECT_EQ(last.live_elements, 128u);
}

/** @test Invalid widths/owners cannot produce a tile; oversized widths remain overflow-safe. */
TEST(GDNDenseCompactStorage, PublicationTileRejectsZeroWidthAndResetsInvalidSpan)
{
    const GDNDenseCompactGeometry geometry(1, 64, 1, 1, 1);
    EXPECT_EQ(geometry.matrixTileCount(0), 0u);
    GDNDenseCompactMatrixTileSpan span{17, 19};
    EXPECT_FALSE(geometry.matrixTileFor(0, 0, span));
    EXPECT_EQ(span.first_element, 0u);
    EXPECT_EQ(span.live_elements, 0u);
    EXPECT_EQ(geometry.matrixTileCount(1), 64u);
    ASSERT_TRUE(geometry.matrixTileFor(63, 1, span));
    EXPECT_EQ(span.first_element, 63u);
    EXPECT_EQ(span.live_elements, 1u);
    const auto maximum = std::numeric_limits<std::size_t>::max();
    EXPECT_EQ(geometry.matrixTileCount(maximum), 1u);
    ASSERT_TRUE(geometry.matrixTileFor(0, maximum, span));
    EXPECT_EQ(span.first_element, 0u);
    EXPECT_EQ(span.live_elements, 64u);
}

/** @test Every request-local live count selects one immutable representation and exact spans. */
TEST(GDNDenseCompactStorage, SplitRetentionModesSelectEveryLivePrefixAndExcludeInactiveCapacity)
{
    for (const auto program : {GDNDenseRetentionProgram{},
        GDNDenseRetentionProgram{GDNDenseRetentionMode::FullSnapshots, 0},
        GDNDenseRetentionProgram{GDNDenseRetentionMode::LiveRowThreshold, 1},
        GDNDenseRetentionProgram{GDNDenseRetentionMode::LiveRowThreshold, 8},
        GDNDenseRetentionProgram{GDNDenseRetentionMode::LiveRowThreshold, 31}})
        for (const auto axes : {key_major, value_major})
        {
            const GDNDenseCompactGeometry g(2, 64, 17, 3, 31, program);
            const GDNDenseCompactGeometry corrections(2, 64, 17, 3, 31);
            if (program.mode == GDNDenseRetentionMode::LiveRowThreshold)
                EXPECT_EQ(g.headStrideFloats(), std::max(corrections.headStrideFloats(),
                    g.checkpointOffsetFloats() + std::size_t(program.compact_min_live_rows) *
                        g.checkpointStrideFloats()));
            for (int live = 0; live <= 31; ++live)
            {
                const bool dense = program.mode == GDNDenseRetentionMode::FullSnapshots ||
                    (program.mode == GDNDenseRetentionMode::LiveRowThreshold && unsigned(live) < program.compact_min_live_rows);
                const auto expected = dense ? GDNDenseRetentionMode::FullSnapshots :
                    GDNDenseRetentionMode::ExactCorrections;
                GDNDenseRetentionMode actual{};
                ASSERT_TRUE(g.retentionForLiveRows(live, actual));
                EXPECT_EQ(actual, expected);
                GDNDenseCompactPublication publication{std::uint32_t(g.encoding()), unsigned(live),
                    unsigned(live), generation, std::uint32_t(axes), std::uint32_t(arithmetic),
                    std::uint32_t(expected), std::uint32_t(program.mode), program.compact_min_live_rows};
                ASSERT_TRUE(g.validPublication(publication, generation, axes, arithmetic));
                for (int accepted = 0; accepted <= live; ++accepted)
                {
                    GDNDenseCompactSelection selected{};
                    ASSERT_TRUE(g.select(publication, generation, axes, arithmetic, accepted, selected));
                    GDNDenseCompactSelection coordinates{};
                    ASSERT_TRUE(g.selectForLiveRows(live, accepted, coordinates));
                    EXPECT_EQ(coordinates.retention_mode, selected.retention_mode);
                    EXPECT_EQ(coordinates.checkpoint, selected.checkpoint);
                    EXPECT_EQ(coordinates.first_record, selected.first_record);
                    EXPECT_EQ(coordinates.update_count, selected.update_count);
                    EXPECT_EQ(selected.retention_mode, expected);
                    EXPECT_EQ(selected.checkpoint, std::size_t(dense ? accepted : accepted / 8));
                    EXPECT_EQ(selected.first_record, std::size_t(dense ? 0 : accepted / 8 * 8));
                    EXPECT_EQ(selected.update_count, unsigned(dense ? 0 : accepted % 8));
                }
                for (int row = -1; row <= 31; ++row)
                {
                    EXPECT_EQ(g.retainRecordForRow(live, row), !dense && row >= 0 && row < live);
                    std::size_t slot = 999;
                    const bool retained = row > 0 && row <= live && (dense || row % 8 == 0);
                    EXPECT_EQ(g.checkpointForCompletedRow(live, row, slot), retained);
                    EXPECT_EQ(slot, retained ? std::size_t(dense ? row : row / 8) : 0u);
                }
                std::vector<bool> covered(g.headStrideFloats(), false);
                std::size_t bytes = 0;
                for (std::size_t i = 0;; ++i)
                {
                    GDNDenseCompactByteSpan span{};
                    if (!g.publishedSpanFor(unsigned(live), i, span)) break;
                    ASSERT_LE(span.offset_bytes + span.live_bytes, g.headStrideFloats() * sizeof(float));
                    for (std::size_t word = span.offset_bytes / 4;
                        word < (span.offset_bytes + span.live_bytes) / 4; ++word)
                    {
                        ASSERT_FALSE(covered[word]);
                        covered[word] = true;
                    }
                    bytes += span.live_bytes;
                }
                const auto header = program.mode == GDNDenseRetentionMode::ExactCorrections ? 28u : 40u;
                EXPECT_EQ(bytes, header + (1u + (dense ? unsigned(live) : unsigned(live) / 8)) *
                    g.matrixFloats() * 4u + (dense ? 0u : unsigned(live) * g.recordFloats() * 4u));
                // cp0 remains live even for empty requests and every repeated A0 revision.
                EXPECT_TRUE(covered[g.checkpointOffsetFloats()]);
            }
            GDNDenseRetentionMode invalid = GDNDenseRetentionMode::FullSnapshots;
            EXPECT_FALSE(g.retentionForLiveRows(-1, invalid));
            EXPECT_EQ(invalid, GDNDenseRetentionMode::ExactCorrections);
            EXPECT_FALSE(g.retentionForLiveRows(32, invalid));
            GDNDenseCompactSelection rejected{99, 99, 99};
            EXPECT_FALSE(g.selectForLiveRows(-1, 0, rejected));
            EXPECT_EQ(rejected.checkpoint, 0u);
            EXPECT_FALSE(g.selectForLiveRows(32, 0, rejected));
            EXPECT_FALSE(g.selectForLiveRows(0, 1, rejected));
        }
}

/** @test Split headers authenticate the actual mode, captured policy and complete live identity. */
TEST(GDNDenseCompactStorage, SplitRetentionRejectsForeignIdentityCountsAndThresholds)
{
    const GDNDenseCompactGeometry g(2, 128, 129, 3, 16,
        {GDNDenseRetentionMode::LiveRowThreshold, 8});
    for (unsigned live : {0u, 7u, 8u, 16u})
    {
        GDNDenseRetentionMode mode{};
        ASSERT_TRUE(g.retentionForLiveRows(int(live), mode));
        const GDNDenseCompactPublication good{std::uint32_t(g.encoding()), live, live, generation,
            std::uint32_t(value_major), std::uint32_t(arithmetic), std::uint32_t(mode),
            std::uint32_t(GDNDenseRetentionMode::LiveRowThreshold), 8};
        ASSERT_TRUE(g.validPublication(good, generation, value_major, arithmetic));
        for (int field = 0; field < 10; ++field)
        {
            auto bad = good;
            if (field == 0) bad.format = std::uint32_t(GDNDenseCompactEncoding::ExactCorrectionsV1);
            if (field == 1) bad.published_rows = 17;
            if (field == 2) ++bad.planned_rows;
            if (field == 3) ++bad.generation;
            if (field == 4) bad.layout = 0x100;
            if (field == 5) bad.arithmetic = 0x100;
            if (field == 6) bad.retention_mode = std::uint32_t(mode) ^ 1u;
            if (field == 7) bad.retention_program = 0;
            if (field == 8) bad.compact_min_live_rows = 7;
            if (field == 9) bad.retention_mode = 0x100;
            GDNDenseCompactSelection selected{99, 99, 99};
            EXPECT_FALSE(g.select(bad, generation, value_major, arithmetic, 0, selected));
            EXPECT_EQ(selected.checkpoint, 0u);
            EXPECT_EQ(selected.update_count, 0u);
        }
        GDNDenseCompactSelection selected{};
        EXPECT_FALSE(g.select(good, generation, value_major, arithmetic, -1, selected));
        EXPECT_FALSE(g.select(good, generation, value_major, arithmetic, int(live) + 1, selected));
    }
    for (auto program : {GDNDenseRetentionProgram{GDNDenseRetentionMode::FullSnapshots, 1},
        GDNDenseRetentionProgram{GDNDenseRetentionMode::ExactCorrections, 1},
        GDNDenseRetentionProgram{GDNDenseRetentionMode::LiveRowThreshold, 0},
        GDNDenseRetentionProgram{GDNDenseRetentionMode::LiveRowThreshold, 17},
        GDNDenseRetentionProgram{static_cast<GDNDenseRetentionMode>(255), 0}})
        EXPECT_THROW((void)GDNDenseCompactGeometry(1, 64, 17, 1, 16, program), std::invalid_argument);
    constexpr int maximum = std::numeric_limits<int>::max();
    // Exact-only geometry must not evaluate unreachable full-image capacity.
    EXPECT_NO_THROW((void)GDNDenseCompactGeometry(10000, 512, 512, 1, maximum));
    EXPECT_THROW((void)GDNDenseCompactGeometry(10000, 512, 512, 1, maximum,
        {GDNDenseRetentionMode::FullSnapshots, 0}), std::overflow_error);
    EXPECT_THROW((void)GDNDenseCompactGeometry(maximum, 512, 512, 1, maximum,
        {GDNDenseRetentionMode::FullSnapshots, 0}), std::overflow_error);
    EXPECT_THROW((void)GDNDenseCompactGeometry(maximum, 512, 512, 1, maximum,
        {GDNDenseRetentionMode::LiveRowThreshold, 8}), std::overflow_error);
}

/** @test Full retention copies immutable cp0 and every accepted row after repeated target mutation. */
TEST(GDNDenseCompactStorage, FullRetentionRevisionsNeverReadMutatedInitialOrTerminalBanks)
{
    const GDNDenseCompactGeometry g(1, 64, 3, 1, 17,
        {GDNDenseRetentionMode::FullSnapshots, 0});
    GuardedCompact retained(g.bankFloats());
    for (int row = 0; row <= 17; ++row)
        for (std::size_t cell = 0; cell < g.matrixFloats(); ++cell)
            retained.data()[g.checkpointOffsetFloats() + std::size_t(row) *
                g.checkpointStrideFloats() + cell] = float(row * 1000 + cell);
    const std::vector<float> immutable(retained.data(), retained.data() + g.bankFloats());
    const GDNDenseCompactPublication publication{std::uint32_t(g.encoding()), 17, 17, generation,
        std::uint32_t(key_major), std::uint32_t(arithmetic),
        std::uint32_t(GDNDenseRetentionMode::FullSnapshots),
        std::uint32_t(GDNDenseRetentionMode::FullSnapshots), 0};
    std::vector<float> destination(g.denseBankFloats(), poison);
    for (int accepted : {17, 0, 7, 1, 17, 0})
    {
        GDNDenseCompactSelection selected{};
        ASSERT_TRUE(g.select(publication, generation, key_major, arithmetic, accepted, selected));
        ASSERT_EQ(selected.update_count, 0u);
        const auto *source = retained.data() + g.checkpointOffsetFloats() +
            selected.checkpoint * g.checkpointStrideFloats();
        std::copy_n(source, g.matrixFloats(), destination.data());
        for (std::size_t cell = 0; cell < g.matrixFloats(); ++cell)
            EXPECT_EQ(destination[cell], float(accepted * 1000 + cell));
        EXPECT_EQ(0, std::memcmp(immutable.data(), retained.data(), g.bankBytes()));
    }
    EXPECT_TRUE(retained.guards());
    EXPECT_THROW((void)GDNDenseCompactTerminalOutput::denseDestination(nullptr), std::invalid_argument);
}

/** @test Split storage requires its explicit encoding and keeps persistent Dense/terminal ownership. */
TEST(GDNDenseCompactStorage, SplitRetentionStorageCannotAliasLegacyFormatOrDropRequiredTerminal)
{
    const GDNDenseCompactGeometry geometry(2, 64, 17, 3, 16,
        {GDNDenseRetentionMode::LiveRowThreshold, 4});
    GuardedCompact retained(geometry.bankFloats());
    std::vector<float> initial(geometry.denseBankFloats(), poison);
    GDNDenseCompactTransaction transaction{geometry, initial.data(), retained.data(),
        geometry.bankFloats(), generation, key_major, arithmetic};
    EXPECT_FALSE(validGDNDenseCompactTransactionStorage(transaction));
    transaction.encoding = geometry.encoding();
    EXPECT_TRUE(validGDNDenseCompactTransactionStorage(transaction));
    EXPECT_EQ(geometry.denseBankFloats(), std::size_t{3} * 2 * 64 * 17);
    auto short_storage = transaction;
    --short_storage.storage_floats;
    EXPECT_FALSE(validGDNDenseCompactTransactionStorage(short_storage));
    // Request-local live choices may differ without changing geometry or captured policy.
    for (int live : {0, 3, 4, 16, 0})
    {
        GDNDenseRetentionMode mode{};
        ASSERT_TRUE(geometry.retentionForLiveRows(live, mode));
        EXPECT_EQ(mode, live < 4 ? GDNDenseRetentionMode::FullSnapshots :
            GDNDenseRetentionMode::ExactCorrections);
    }
    EXPECT_THROW((void)GDNDenseCompactTerminalOutput::denseDestination(nullptr), std::invalid_argument);
    EXPECT_TRUE(retained.guards());
}


/** @test Existing metadata meanings remain distinct, including explicit cp0 versus skip. */
TEST(GDNDenseCompactStorage, DeviceSelectionKindsPreserveNoOpVersusExplicitPrefixZero)
{
    const GDNDenseCompactGeometry geometry(2, 64, 17, 3, 16);
    std::array<std::int32_t, 9> metadata{};
    const auto counts = GDNDenseAcceptedSelection::acceptedCounts(geometry, metadata.data(), metadata.size(), 3);
    const auto rows = GDNDenseAcceptedSelection::ownRequestFlattenedRows(geometry, metadata.data(), metadata.size(), 3);
    const auto lengths = GDNDenseAcceptedSelection::requestTerminalLengths(geometry, metadata.data(), metadata.size(), 3);
    for (int request = 0; request < 3; ++request)
        for (int live : {0, 1, 3, 7, 8, 15, 16})
        {
            GDNDenseAcceptedSelectionResult selected;
            ASSERT_TRUE(counts.resolveObserved(request, live, 0, selected));
            EXPECT_EQ(selected.action(), GDNDenseAcceptedSelectionAction::Restore);
            EXPECT_EQ(selected.acceptedCount(), 0u);
            ASSERT_TRUE(lengths.resolveObserved(request, live, 0, selected));
            EXPECT_EQ(selected.action(), GDNDenseAcceptedSelectionAction::NoOp);
            ASSERT_TRUE(rows.resolveObserved(request, live, -1, selected));
            EXPECT_EQ(selected.action(), GDNDenseAcceptedSelectionAction::NoOp);
            ASSERT_TRUE(rows.resolveObserved(request, live, std::numeric_limits<std::int32_t>::min(), selected));
            EXPECT_EQ(selected.action(), GDNDenseAcceptedSelectionAction::NoOp);
            for (int accepted = 1; accepted <= live; ++accepted)
            {
                for (const auto &view : {counts, lengths, rows})
                {
                    const int value = view.kind() == GDNDenseAcceptedSelectionKind::OwnRequestFlattenedRows ?
                        request * 16 + accepted - 1 : accepted;
                    ASSERT_TRUE(view.resolveObserved(request, live, value, selected));
                    ASSERT_EQ(selected.action(), GDNDenseAcceptedSelectionAction::Restore);
                    EXPECT_EQ(selected.acceptedCount(), unsigned(accepted));
                    GDNDenseCompactSelection coordinates{};
                    ASSERT_TRUE(geometry.selectForLiveRows(live, int(selected.acceptedCount()), coordinates));
                    EXPECT_EQ(coordinates.checkpoint, std::size_t(accepted / 8));
                    EXPECT_EQ(coordinates.update_count, unsigned(accepted % 8));
                }
            }
        }
    // Factories and resolution never read the bound metadata payload.
    metadata.fill(std::numeric_limits<std::int32_t>::min());
    GDNDenseAcceptedSelectionResult selected;
    ASSERT_TRUE(counts.resolveObserved(2, 3, 2, selected));
    EXPECT_EQ(selected.acceptedCount(), 2u);
}

/** @test Positive flattened rows must belong to this request AND its published live prefix. */
TEST(GDNDenseCompactStorage, DeviceSelectionRejectsForeignWindowsMalformedCountsAndStaleResults)
{
    const GDNDenseCompactGeometry geometry(2, 128, 129, 3, 16);
    std::array<std::int32_t, 3> metadata{};
    const auto rows = GDNDenseAcceptedSelection::ownRequestFlattenedRows(geometry, metadata.data(), metadata.size());
    const auto counts = GDNDenseAcceptedSelection::acceptedCounts(geometry, metadata.data(), metadata.size());
    const auto lengths = GDNDenseAcceptedSelection::requestTerminalLengths(geometry, metadata.data(), metadata.size());
    for (const auto &view : {rows, counts, lengths})
    {
        GDNDenseAcceptedSelectionResult result;
        ASSERT_TRUE(counts.resolveObserved(1, 3, 2, result));
        for (const auto invalid : std::array<std::array<int, 3>, 4>{{{-1, 3, 0}, {3, 3, 0}, {0, -1, 0}, {0, 17, 0}}})
        {
            EXPECT_FALSE(view.resolveObserved(invalid[0], invalid[1], invalid[2], result));
            EXPECT_EQ(result.action(), GDNDenseAcceptedSelectionAction::NoOp);
            EXPECT_EQ(result.acceptedCount(), 0u);
        }
    }
    for (int request = 0; request < 3; ++request)
        for (int live : {0, 1, 4, 16})
        {
            GDNDenseAcceptedSelectionResult result;
            const int first = request * 16;
            for (int foreign : {first + live, first + 16, 48, std::numeric_limits<int>::max()})
            {
                EXPECT_FALSE(rows.resolveObserved(request, live, foreign, result));
                EXPECT_EQ(result.action(), GDNDenseAcceptedSelectionAction::NoOp);
            }
            if (request > 0)
                for (int foreign : {0, first - 1})
                    EXPECT_FALSE(rows.resolveObserved(request, live, foreign, result));
            EXPECT_FALSE(counts.resolveObserved(request, live, -1, result));
            EXPECT_FALSE(lengths.resolveObserved(request, live, -1, result));
            EXPECT_FALSE(counts.resolveObserved(request, live, live + 1, result));
            EXPECT_FALSE(lengths.resolveObserved(request, live, live + 1, result));
        }
}

/** @test View construction authenticates capacity, stride and nonwrapping borrowed addresses. */
TEST(GDNDenseCompactStorage, DeviceSelectionColdBindingChecksCapacityOverflowAndExactIdentity)
{
    const GDNDenseCompactGeometry geometry(2, 64, 17, 3, 16);
    std::array<std::int32_t, 12> values{}, other{};
    const auto a = GDNDenseAcceptedSelection::ownRequestFlattenedRows(geometry, values.data(), values.size(), 3);
    EXPECT_EQ(a.addressedBytes(), 7u * sizeof(std::int32_t));
    EXPECT_EQ(a.capacityElements(), values.size());
    EXPECT_EQ(a.stride(), 3);
    EXPECT_EQ(a.data(), values.data());
    EXPECT_TRUE(a.matchesGeometry(geometry));
    EXPECT_FALSE(a.matchesGeometry(GDNDenseCompactGeometry(2, 64, 17, 2, 16)));
    EXPECT_FALSE(a.matchesGeometry(GDNDenseCompactGeometry(2, 64, 17, 3, 15)));
    EXPECT_TRUE(a.sameBinding(GDNDenseAcceptedSelection::ownRequestFlattenedRows(geometry, values.data(), values.size(), 3)));
    EXPECT_FALSE(a.sameBinding(GDNDenseAcceptedSelection::acceptedCounts(geometry, values.data(), values.size(), 3)));
    EXPECT_FALSE(a.sameBinding(GDNDenseAcceptedSelection::ownRequestFlattenedRows(geometry, other.data(), other.size(), 3)));
    EXPECT_FALSE(a.sameBinding(GDNDenseAcceptedSelection::ownRequestFlattenedRows(geometry, values.data(), values.size(), 2)));
    EXPECT_FALSE(a.sameBinding(GDNDenseAcceptedSelection::ownRequestFlattenedRows(geometry, values.data(), values.size() - 1, 3)));
    for (int r = 0; r < 3; ++r)
    {
        std::size_t offset = 99;
        ASSERT_TRUE(a.valueOffset(r, offset));
        EXPECT_EQ(offset, std::size_t(r * 3));
    }
    std::size_t offset = 99;
    EXPECT_FALSE(a.valueOffset(-1, offset)); EXPECT_EQ(offset, 0u);
    EXPECT_FALSE(a.valueOffset(3, offset)); EXPECT_EQ(offset, 0u);
    EXPECT_THROW((void)GDNDenseAcceptedSelection::acceptedCounts(geometry, nullptr, 3), std::invalid_argument);
    EXPECT_THROW((void)GDNDenseAcceptedSelection::acceptedCounts(geometry,
        reinterpret_cast<const std::int32_t *>(std::uintptr_t(0x1001)), 3), std::invalid_argument);
    EXPECT_THROW((void)GDNDenseAcceptedSelection::acceptedCounts(geometry, values.data(), 3, 0), std::invalid_argument);
    EXPECT_THROW((void)GDNDenseAcceptedSelection::acceptedCounts(geometry, values.data(), 3, -1), std::invalid_argument);
    EXPECT_THROW((void)GDNDenseAcceptedSelection::acceptedCounts(geometry, values.data(), 6, 3), std::invalid_argument);
    EXPECT_THROW((void)GDNDenseAcceptedSelection::acceptedCounts(geometry, values.data(),
        std::numeric_limits<std::size_t>::max()), std::overflow_error);
    EXPECT_THROW((void)GDNDenseAcceptedSelection::acceptedCounts(geometry,
        reinterpret_cast<const std::int32_t *>(std::numeric_limits<std::uintptr_t>::max() - 3), 3), std::overflow_error);
    // Pure coordinates may be large; neither int multiplication nor a payload read is permitted.
    const GDNDenseCompactGeometry large(1, 1, 1, std::numeric_limits<int>::max(), 1);
    const auto capacity = std::size_t(std::numeric_limits<int>::max() - 1) *
        std::size_t(std::numeric_limits<int>::max()) + 1;
    const auto wide = GDNDenseAcceptedSelection::acceptedCounts(large,
        reinterpret_cast<const std::int32_t *>(std::uintptr_t(0x1000)), capacity,
        std::numeric_limits<int>::max());
    ASSERT_TRUE(wide.valueOffset(std::numeric_limits<int>::max() - 1, offset));
    EXPECT_EQ(offset, capacity - 1);
}

/** @test Accepted-prefix admission binds typed request geometry and exact addressed stride span. */
TEST(GDNDenseCompactStorage, TypedPublisherAdmissionUsesSelectionGeometryAndAddressedSpan)
{
    const GDNDenseCompactGeometry geometry(2, 64, 17, 3, 9);
    AuthoredCapture capture(geometry, value_major, {9, 0, 4});
    std::array<std::int32_t, 9> metadata{};
    std::vector<float> destination(geometry.denseBankFloats());
    const auto selection = GDNDenseAcceptedSelection::ownRequestFlattenedRows(
        geometry, metadata.data(), metadata.size(), 3);
    GDNDenseCompactAcceptedPrefixInput input{capture.transaction(), selection, destination.data()};
    EXPECT_EQ(input.selection.addressedBytes(), 7u * sizeof(std::int32_t));
    EXPECT_TRUE(validCUDADenseCompactAcceptedPrefixStorage(input));
    const GDNDenseCompactGeometry foreign_requests(2, 64, 17, 2, 9);
    input.selection = GDNDenseAcceptedSelection::acceptedCounts(foreign_requests, metadata.data(), metadata.size());
    EXPECT_FALSE(validCUDADenseCompactAcceptedPrefixStorage(input));
    const GDNDenseCompactGeometry foreign_width(2, 64, 17, 3, 8);
    input.selection = GDNDenseAcceptedSelection::requestTerminalLengths(foreign_width, metadata.data(), metadata.size());
    EXPECT_FALSE(validCUDADenseCompactAcceptedPrefixStorage(input));
    for (const auto *owner : {reinterpret_cast<const std::int32_t *>(capture.initial.data()),
        reinterpret_cast<const std::int32_t *>(capture.compact.data()),
        reinterpret_cast<const std::int32_t *>(destination.data())})
    {
        input.selection = GDNDenseAcceptedSelection::acceptedCounts(geometry, owner, metadata.size(), 3);
        EXPECT_FALSE(validCUDADenseCompactAcceptedPrefixStorage(input));
    }
}

/** @test Physical grid/SM selection is geometry-derived, including equality and alternate factorizations. */
TEST(GDNDenseCompactStorage, CUDACaptureScheduleUsesPhysicalGridRatherThanHeadcountOrLiveRows)
{
    using Plan = CUDADenseRetentionCapturePlan;
    using Schedule = CUDADenseRetentionSchedule;
    for (int sm : {1, 7, 14, 48, 82, 108, 132, 256})
    {
        const GDNDenseCompactGeometry one(1, 128, 1, 1, 16);
        EXPECT_EQ(Plan::captureCTAs(one), 1u);
        EXPECT_EQ(Plan::chooseSchedule(one, key_major, sm), Schedule::PartitionOneCTA);
        const GDNDenseCompactGeometry equal(sm, 128, 1, 1, 3);
        const GDNDenseCompactGeometry above(sm + 1, 128, 1, 1, 3);
        EXPECT_EQ(Plan::chooseSchedule(equal, key_major, sm), Schedule::PartitionOneCTA);
        EXPECT_EQ(Plan::chooseSchedule(above, key_major, sm), Schedule::CheckpointWordTwoCTA);
        EXPECT_EQ(Plan::chooseSchedule(above, value_major, sm), Schedule::PartitionOneCTA);
        EXPECT_EQ(Plan::chooseSchedule(GDNDenseCompactGeometry(sm + 1, 64, 1, 1, 3), key_major, sm),
            Schedule::PartitionOneCTA);
    }
    const GDNDenseCompactGeometry a(24, 128, 128, 1, 16);
    const GDNDenseCompactGeometry b(8, 128, 128, 3, 3);
    const GDNDenseCompactGeometry c(12, 128, 256, 1, 31);
    EXPECT_EQ(Plan::captureCTAs(a), 96u);
    EXPECT_EQ(Plan::captureCTAs(b), 96u);
    EXPECT_EQ(Plan::captureCTAs(c), 96u);
    for (int sm : {48, 96, 132})
    {
        EXPECT_EQ(Plan::chooseSchedule(a, key_major, sm), Plan::chooseSchedule(b, key_major, sm));
        EXPECT_EQ(Plan::chooseSchedule(a, key_major, sm), Plan::chooseSchedule(c, key_major, sm));
    }
    EXPECT_THROW((void)Plan::chooseSchedule(a, key_major, 0), std::invalid_argument);
    EXPECT_THROW((void)Plan::chooseSchedule(a, key_major, -1), std::invalid_argument);
    EXPECT_THROW((void)Plan::chooseSchedule(a, static_cast<GDNRecurrenceStateLayout>(255), 82), std::invalid_argument);
    EXPECT_THROW((void)Plan::chooseSchedule(GDNDenseCompactGeometry(1, 32, 128, 1, 16), key_major, 82), std::invalid_argument);
    // Checked grid multiplication, independent of any allocation or accelerator initialization.
    const GDNDenseCompactGeometry large(1, 64, 512, std::numeric_limits<int>::max(), 1);
    if (std::numeric_limits<std::size_t>::max() / 16 >= std::size_t(std::numeric_limits<int>::max()))
        EXPECT_EQ(Plan::captureCTAs(large), std::size_t(std::numeric_limits<int>::max()) * 16);
    else
        EXPECT_THROW((void)Plan::captureCTAs(large), std::overflow_error);
}

namespace
{
/** @brief Host-only pointer/extent witnesses; no device or prepared backend is constructed. */
struct PreparedHelperCase final
{
    const GDNDenseCompactGeometry geometry{2, 64, 17, 3, 9};
    GuardedCompact retained{geometry.bankFloats()};
    std::array<std::int32_t, 3> counts{9, 0, 4};
    std::vector<float> initial = std::vector<float>(geometry.denseBankFloats());
    std::vector<float> merged = std::vector<float>(3 * 9 * (2 * 64 + 2 * 17));
    std::vector<float> alpha = std::vector<float>(geometry.gateFloats());
    std::vector<float> beta = std::vector<float>(geometry.gateFloats());
    std::array<float, 2> A{}, dt{};
    std::vector<float> output = std::vector<float>(geometry.outputFloats());
    std::vector<float> terminal = std::vector<float>(geometry.denseBankFloats());
    int stream_tag = 0; ///< Host identity tag for pure validation only; never a GPU stream claim.

    /** @return Complete source-only tuple with independent owned host witness spans. */
    MergedRecurrenceInput sources()
    {
        return {.rows = DeviceRequestRowRanges::deviceCounted(3, 9, counts.data()),
                .key_heads = 1,
                .value_heads = 2,
                .key_width = 64,
                .value_width = 17,
                .global_value_offset = -1,
                .use_qk_l2norm = true,
                .merged = merged.data(),
                .alpha = alpha.data(),
                .beta_raw = beta.data(),
                .A_log = A.data(),
                .dt_bias = dt.data(),
                .output = output.data(),
                .initial_state = nullptr,
                .updated_state = nullptr,
                .snapshots = nullptr,
                .snapshot_stride_floats = 0,
                .max_snapshot_rows = 0,
                .recurrence_layout = value_major};
    }
    /** @return Pure immutable binding metadata; no backend cold preparation is invoked. */
    GDNRetainedInvocationBinding binding()
    {
        return {geometry,
                GDNDenseCompactTerminalKind::DenseDestination,
                {retained.data(), geometry.bankFloats()},
                terminal,
                generation,
                &stream_tag};
    }
    /** @return Validated native metadata tuple without executing or reading any payload. */
    GDNDenseCompactCaptureInput input()
    {
        return makeGDNRetainedCaptureInput(sources(), binding(), initial.data(), value_major,
                                           arithmetic);
    }
};
} // namespace

/** @test Prepared helper admission rejects missing, short, aliased and contradictory cold views. */
TEST(GDNDenseCompactStorage, PreparedHelperAdmissionRejectsIncompleteViews)
{
    PreparedHelperCase c;
    const auto binding = c.binding();
    const auto sources = c.sources();
    const auto make = [&](const GDNRetainedInvocationBinding &b)
    { return makeGDNRetainedCaptureInput(sources, b, c.initial.data(), value_major, arithmetic); };
    EXPECT_NO_THROW((void)make(binding));
    auto bad = binding;
    bad.workspace_owner_identity = 0;
    EXPECT_THROW((void)make(bad), std::invalid_argument);
    for (std::uintptr_t stream : {0u, 1u, 2u})
    {
        bad = binding;
        bad.capture_stream = reinterpret_cast<void *>(stream);
        EXPECT_THROW((void)make(bad), std::invalid_argument);
    }
    bad = binding;
    bad.storage = {binding.storage.data(), binding.geometry.bankFloats() - 1};
    EXPECT_THROW((void)make(bad), std::invalid_argument);
    bad = binding;
    bad.terminal = {binding.terminal.data(), binding.geometry.denseBankFloats() - 1};
    EXPECT_THROW((void)make(bad), std::invalid_argument);
    bad = binding;
    bad.storage = {binding.storage.data() + 1, binding.storage.size() - 1};
    EXPECT_THROW((void)make(bad), std::invalid_argument);
    bad = binding;
    bad.terminal = {binding.storage.data(), binding.geometry.denseBankFloats()};
    EXPECT_THROW((void)make(bad), std::invalid_argument);
    bad = binding;
    bad.terminal_kind = static_cast<GDNDenseCompactTerminalKind>(255);
    EXPECT_THROW((void)make(bad), std::invalid_argument);
    bad = binding;
    bad.terminal_kind = GDNDenseCompactTerminalKind::RetainedPrefixesOnly;
    EXPECT_THROW((void)make(bad), std::invalid_argument);
    bad.terminal = {};
    EXPECT_NO_THROW((void)make(bad));
    bad.terminal = {c.terminal.data(), 0};
    EXPECT_THROW((void)make(bad), std::invalid_argument);
    bad = binding;
    bad.terminal = {};
    EXPECT_THROW((void)make(bad), std::invalid_argument);
    EXPECT_TRUE(c.retained.guards());
}

/** @test Every embedded native input field participates in prepared capture identity. */
TEST(GDNDenseCompactStorage, PreparedHelperIdentityCoversEveryCaptureField)
{
    PreparedHelperCase c;
    const auto original = c.input();
    using Input = GDNDenseCompactCaptureInput;
    const std::vector<std::function<void(Input &)>> changes{
        [](Input &x) { x.transaction.geometry = GDNDenseCompactGeometry(3, 64, 17, 3, 9); },
        [](Input &x) { x.transaction.geometry = GDNDenseCompactGeometry(2, 128, 17, 3, 9); },
        [](Input &x) { x.transaction.geometry = GDNDenseCompactGeometry(2, 64, 18, 3, 9); },
        [](Input &x) { x.transaction.geometry = GDNDenseCompactGeometry(2, 64, 17, 2, 9); },
        [](Input &x) { x.transaction.geometry = GDNDenseCompactGeometry(2, 64, 17, 3, 8); },
        [](Input &x)
        {
            x.transaction.geometry =
                GDNDenseCompactGeometry(2, 64, 17, 3, 9, {GDNDenseRetentionMode::FullSnapshots, 0});
        },
        [](Input &x)
        {
            x.transaction.geometry = GDNDenseCompactGeometry(
                2, 64, 17, 3, 9, {GDNDenseRetentionMode::LiveRowThreshold, 4});
        },
        [](Input &x) { ++x.transaction.initial_state; },
        [](Input &x) { ++x.transaction.storage; },
        [](Input &x) { ++x.transaction.storage_floats; },
        [](Input &x) { ++x.transaction.generation; },
        [](Input &x) { x.transaction.layout = key_major; },
        [](Input &x)
        { x.transaction.arithmetic = GDNDenseCompactArithmetic::ROCmFourPartFusedUpdateV1; },
        [](Input &x) { x.transaction.encoding = static_cast<GDNDenseCompactEncoding>(0); },
        [](Input &x) { x.terminal = GDNDenseCompactTerminalOutput::retainedPrefixesOnly(); },
        [](Input &x)
        {
            x.terminal =
                GDNDenseCompactTerminalOutput::denseDestination(x.terminal.destination() + 1);
        },
        [](Input &x) { x.sources.rows = DeviceRequestRowRanges::fullyActiveRequests(2, 9); },
        [](Input &x) { x.sources.rows = DeviceRequestRowRanges::fullyActiveRequests(3, 8); },
        [](Input &x) { x.sources.rows = DeviceRequestRowRanges::fullyActiveRequests(3, 9); },
        [](Input &x) { ++x.sources.key_heads; },
        [](Input &x) { ++x.sources.value_heads; },
        [](Input &x) { ++x.sources.key_width; },
        [](Input &x) { ++x.sources.value_width; },
        [](Input &x) { ++x.sources.global_value_offset; },
        [](Input &x) { x.sources.use_qk_l2norm = !x.sources.use_qk_l2norm; },
        [](Input &x) { ++x.sources.merged; },
        [](Input &x) { ++x.sources.alpha; },
        [](Input &x) { ++x.sources.beta_raw; },
        [](Input &x) { ++x.sources.A_log; },
        [](Input &x) { ++x.sources.dt_bias; },
        [](Input &x) { ++x.sources.output; },
        [](Input &x) { x.sources.initial_state = x.transaction.initial_state; },
        [](Input &x) { x.sources.updated_state = x.terminal.destination(); },
        [](Input &x) { x.sources.snapshots = x.transaction.storage; },
        [](Input &x) { ++x.sources.snapshot_stride_floats; },
        [](Input &x) { ++x.sources.max_snapshot_rows; },
        [](Input &x) { x.sources.recurrence_layout = key_major; }};
    EXPECT_TRUE(sameGDNRetainedCaptureInput(original, original));
    for (std::size_t i = 0; i < changes.size(); ++i)
    {
        SCOPED_TRACE(i);
        auto changed = original;
        changes[i](changed);
        EXPECT_FALSE(sameGDNRetainedCaptureInput(original, changed));
        EXPECT_FALSE(sameGDNRetainedCaptureInput(changed, original));
    }
    auto threshold4 = original;
    threshold4.transaction.geometry =
        GDNDenseCompactGeometry(2, 64, 17, 3, 9, {GDNDenseRetentionMode::LiveRowThreshold, 4});
    auto threshold8 = threshold4;
    threshold8.transaction.geometry =
        GDNDenseCompactGeometry(2, 64, 17, 3, 9, {GDNDenseRetentionMode::LiveRowThreshold, 8});
    EXPECT_FALSE(sameGDNRetainedCaptureInput(threshold4, threshold8));
    const auto binding = c.binding();
    const std::vector<std::function<void(GDNRetainedInvocationBinding &)>> view_changes{
        [](auto &x) { x.storage = {x.storage.data() + 1, x.storage.size() - 1}; },
        [](auto &x) { x.storage = {x.storage.data(), x.storage.size() - 1}; },
        [](auto &x) { x.terminal = {x.terminal.data() + 1, x.terminal.size() - 1}; },
        [](auto &x) { x.terminal = {x.terminal.data(), x.terminal.size() - 1}; },
        [](auto &x) { ++x.workspace_owner_identity; },
        [](auto &x) { x.capture_stream = nullptr; },
        [](auto &x) { x.terminal_kind = GDNDenseCompactTerminalKind::RetainedPrefixesOnly; }};
    EXPECT_TRUE(sameGDNRetainedBindingViews(binding, binding));
    for (const auto &change : view_changes)
    {
        auto changed = binding;
        change(changed);
        EXPECT_FALSE(sameGDNRetainedBindingViews(binding, changed));
    }
}

/** @test Pure protection covers every speculative write and immutable selection source. */
TEST(GDNDenseCompactStorage, PreparedHelpersProtectCanonicalStateOwners)
{
    PreparedHelperCase c;
    const auto original = c.input();
    const std::span<const float> state(c.initial);
    EXPECT_TRUE(retainedCaptureWritesDisjointFromState(original, state));
    auto bad = original;
    bad.sources.output = c.initial.data();
    EXPECT_FALSE(retainedCaptureWritesDisjointFromState(bad, state));
    bad = original;
    bad.transaction.storage = c.initial.data();
    EXPECT_FALSE(retainedCaptureWritesDisjointFromState(bad, state));
    bad = original;
    bad.terminal = GDNDenseCompactTerminalOutput::denseDestination(c.initial.data());
    EXPECT_FALSE(retainedCaptureWritesDisjointFromState(bad, state));
    EXPECT_TRUE(retainedCaptureWritesDisjointFromState(original, {}));
    EXPECT_FALSE(retainedCaptureWritesDisjointFromState(original, {c.initial.data(), 0}));
    const auto selection =
        GDNDenseAcceptedSelection::acceptedCounts(c.geometry, c.counts.data(), c.counts.size());
    EXPECT_TRUE(retainedSelectionDisjointFromState(selection, state));
    const auto alias = GDNDenseAcceptedSelection::acceptedCounts(
        c.geometry, reinterpret_cast<const std::int32_t *>(c.initial.data()), c.initial.size());
    EXPECT_FALSE(retainedSelectionDisjointFromState(alias, state));
    EXPECT_TRUE(retainedSelectionDisjointFromState(selection, {}));
    EXPECT_FALSE(retainedSelectionDisjointFromState(selection, {c.initial.data(), 0}));
    EXPECT_TRUE(c.retained.guards());
}
