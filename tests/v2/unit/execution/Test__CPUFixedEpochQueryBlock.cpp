/**
 * @file Test__CPUFixedEpochQueryBlock.cpp
 * @brief Byte oracle for independent-query SIMD over every supported key tail.
 *
 * The scalar interpreter here spells out backward record traversal and forward
 * key FMAs independently of both production contraction implementations. Every
 * admissible phase/block size is covered, and untouched output padding remains
 * poisoned so SIMD lanes cannot accidentally publish inactive history or rows.
 * A separate prepared-row interpreter authenticates direct retained checkpoints,
 * every materialized accepted prefix, and ordinary continuation from that prefix.
 * Isolated flush checks also cover exact-alias/disjoint predecessor storage,
 * unaligned tails, signed-zero/subnormal gates and immutable input guards.
 */
#include "kernels/cpu/gdn/CPUFixedEpochQueryBlock.h"
#include "kernels/cpu/gdn/CPUFixedEpochGDN.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <limits>
#include <cmath>
#include <cstring>
#include <vector>
#include <type_traits>

namespace
{
    /**
     * @brief Compare one fixed-size physical block with an independent interpreter.
     * @tparam Rows Live row count.
     * @param width Live key width.
     * @param phase History length before this block.
     */
    template<int Rows>
    void checkBlock(int width, int phase)
    {
        const llaminar2::CPUGDNStateLayout layout(1, width, 3);
        std::vector<float> state(layout.headFloats(), 0.0f), q(Rows * (width + 3));
        for (int p = 0; p < phase + Rows; ++p)
        {
            float *raw = layout.record(state.data(), p);
            for (int j = 0; j < width; ++j)
                raw[j] = static_cast<float>((j * 19 + p * 31) % 101 - 50) / 1024.0f;
            raw[width + 3] = 0.73f + p * 0.017f;
            raw[width + 4] = 0.19f + p * 0.031f;
        }
        for (int r = 0; r < Rows; ++r)
            for (int j = 0; j < width; ++j)
                q[r * (width + 3) + j] = static_cast<float>((j * 37 + r * 17) % 127 - 63) / 64.0f;
        std::array<float, Rows * 512> actual, expected;
        std::array<float, Rows * 8> ca, ce;
        actual.fill(-1234.0f); expected = actual;
        ca.fill(-1234.0f); ce = ca;
        for (int r = 0; r < Rows; ++r)
        {
            float *query = expected.data() + r * 512;
            std::memcpy(query, q.data() + r * (width + 3), width * sizeof(float));
            for (int p = phase + r; p >= 0; --p)
            {
                const float *raw = layout.record(state.data(), p);
                float dot = 0.0f;
                for (int j = 0; j < width; ++j)
                    dot = std::fma(raw[j], query[j], dot);
                const float eta = raw[width + 4] * dot;
                ce[r * 8 + p] = eta;
                for (int j = 0; j < width; ++j)
                    query[j] = raw[width + 3] * std::fma(-eta, raw[j], query[j]);
            }
        }
        llaminar2::cpu_gdn_epoch::contractQueries<Rows>(layout, layout.record(state.data(), 0), q.data(),
            width + 3, phase, actual.data(), ca.data());
        ASSERT_EQ(0, std::memcmp(actual.data(), expected.data(), sizeof(actual)));
        ASSERT_EQ(0, std::memcmp(ca.data(), ce.data(), sizeof(ca)));
    }
}

TEST(CPUFixedEpochQueryBlock, EveryKeyTailAndAdmissibleHistoryMatchesIndependentProgram)
{
    for (int width = 1; width <= 512; ++width)
        for (int phase = 0; phase < 8; ++phase)
        {
            SCOPED_TRACE(::testing::Message() << "width=" << width << " phase=" << phase);
            checkBlock<1>(width, phase);
            if (phase <= 6) checkBlock<2>(width, phase);
            if (phase <= 5) checkBlock<3>(width, phase);
            if (phase <= 4) checkBlock<4>(width, phase);
        }
}

namespace
{
    /**
     * @brief Interpret one prepared update without production numeric helpers.
     * @param layout Canonical encoded head geometry.
     * @param state In/out checkpoint and pending raw history.
     * @param q Prepared normalized/scaled query.
     * @param k Prepared normalized key.
     * @param v Raw value vector.
     * @param decay Rounded decay gate.
     * @param beta Rounded write strength.
     * @return Independent canonical readout for this row.
     */
    std::vector<float> interpretPrepared(const llaminar2::CPUGDNStateLayout &layout,
        std::vector<float> &state, const float *q, const float *k, const float *v, float decay, float beta)
    {
        const int dk = layout.keyWidth(), dv = layout.valueWidth();
        const int count = static_cast<int>(state[0]) + 1;
        float *raw = layout.record(state.data(), count - 1);
        std::copy_n(k, dk, raw); std::copy_n(v, dv, raw + dk);
        raw[dk + dv] = decay; raw[dk + dv + 1] = beta;
        std::vector<float> x(q, q + dk), c(count), output(dv);
        for (int p = count - 1; p >= 0; --p)
        {
            const float *u = layout.record(state.data(), p);
            float dot = 0;
            for (int j = 0; j < dk; ++j) dot = std::fma(u[j], x[j], dot);
            c[p] = u[dk + dv + 1] * dot;
            for (int j = 0; j < dk; ++j) x[j] = u[dk + dv] * std::fma(-c[p], u[j], x[j]);
        }
        for (int vi = 0; vi < dv; ++vi)
        {
            float sum = 0;
            for (int j = 0; j < dk; ++j) sum = std::fma(state[1 + j * dv + vi], x[j], sum);
            for (int p = 0; p < count; ++p) sum = std::fma(c[p], layout.record(state.data(), p)[dk + vi], sum);
            output[vi] = sum;
        }
        state[0] = static_cast<float>(count);
        if (count == 8)
        {
            for (int p = 0; p < 8; ++p)
            {
                const float *u = layout.record(state.data(), p);
                for (int vi = 0; vi < dv; ++vi)
                {
                    float sum = 0;
                    for (int j = 0; j < dk; ++j)
                    {
                        float &s = state[1 + j * dv + vi]; s *= u[dk + dv];
                        sum = std::fma(s, u[j], sum);
                    }
                    const float delta = (u[dk + vi] - sum) * u[dk + dv + 1];
                    for (int j = 0; j < dk; ++j)
                    {
                        float &s = state[1 + j * dv + vi]; s = std::fma(u[j], delta, s);
                    }
                }
            }
            state[0] = 0;
            std::fill(layout.record(state.data(), 0), state.data() + state.size(), 0.0f);
        }
        return output;
    }
    /**
     * @brief Compare a complete eight-record checkpoint and its next continuation.
     * @param dk Positive key width, including SIMD/tail neighbors.
     * @param dv Positive value width, including unaligned scalar endpoints.
     * @param edge_gates Include signed zero and gradual-underflow decay factors.
     * @note The independent prepared-row interpreter owns expected state bytes.
     * Source history/predecessor words and destination guards remain immutable.
     */
    void checkFlushPipeline(int dk, int dv, bool edge_gates)
    {
        using namespace llaminar2;
        const CPUGDNStateLayout layout(1, dk, dv);
        std::vector<float> q(dk), k(8 * dk), v(8 * dv), decay(8), beta(8),
            initial(layout.headFloats(), 0.0f);
        for (int j = 0; j < dk; ++j)
            q[j] = static_cast<float>(j % 19 - 9) / 64.0f;
        for (int j = 0; j < layout.matrixFloats(); ++j)
            initial[1 + j] = static_cast<float>((j * 17) % 47 - 23) / 256.0f;
        for (int p = 0; p < 8; ++p)
        {
            for (int j = 0; j < dk; ++j)
                k[p * dk + j] = static_cast<float>((j * 13 + p * 7) % 37 - 18) / 1024.0f;
            for (int j = 0; j < dv; ++j)
                v[p * dv + j] = static_cast<float>((j * 11 + p * 23) % 43 - 21) / 32.0f;
            decay[p] = edge_gates ? std::array<float, 8>{1.0f, -0.0f,
                std::numeric_limits<float>::min(), std::numeric_limits<float>::denorm_min(),
                0.0f, 0.75f, 1.0f, 0.875f}[p] : 0.69f + p * 0.031f;
            beta[p] = 0.17f + p * 0.023f;
        }
        auto expected = initial;
        for (int p = 0; p < 8; ++p)
            interpretPrepared(layout, expected, q.data(), k.data() + p * dk,
                v.data() + p * dv, decay[p], beta[p]);
        std::vector<float> history(8 * layout.recordFloats(), 0.0f);
        for (int p = 0; p < 8; ++p)
        {
            float *raw = history.data() + p * layout.recordFloats();
            std::copy_n(k.data() + p * dk, dk, raw);
            std::copy_n(v.data() + p * dv, dv, raw + dk);
            raw[dk + dv] = decay[p]; raw[dk + dv + 1] = beta[p];
        }
        const auto original_history = history, original_initial = initial;
        for (const bool inplace : {false, true})
        {
            SCOPED_TRACE(::testing::Message() << "dk=" << dk << " dv=" << dv
                << " edge_gates=" << edge_gates << " inplace=" << inplace);
            // Deliberate one-float offset exercises valid unaligned endpoints.
            std::vector<float> destination(layout.matrixFloats() + 2, -1234.0f);
            float *matrix = destination.data() + 1;
            if (inplace) std::copy_n(initial.data() + 1, layout.matrixFloats(), matrix);
            cpu_gdn_epoch::flushCheckpoint(layout, matrix, history.data(),
                inplace ? matrix : initial.data() + 1);
            ASSERT_EQ(0, std::memcmp(matrix, expected.data() + 1, layout.matrixFloats() * sizeof(float)));
            ASSERT_EQ(destination.front(), -1234.0f); ASSERT_EQ(destination.back(), -1234.0f);
            ASSERT_EQ(0, std::memcmp(initial.data(), original_initial.data(), initial.size() * sizeof(float)));
            ASSERT_EQ(0, std::memcmp(history.data(), original_history.data(), history.size() * sizeof(float)));
            auto continued = expected;
            std::copy_n(matrix, layout.matrixFloats(), continued.data() + 1);
            auto continued_oracle = expected;
            const auto output_oracle = interpretPrepared(layout, continued_oracle,
                q.data(), k.data(), v.data(), decay[0], beta[0]);
            std::vector<float> output(dv);
            cpu_gdn_epoch::run(layout, continued.data(), continued.data(), q.data(),
                k.data(), v.data(), decay.data(), beta.data(), 1, dk, dv, 1, output.data(), dv);
            ASSERT_EQ(0, std::memcmp(output.data(), output_oracle.data(), dv * sizeof(float)));
            ASSERT_EQ(0, std::memcmp(continued.data(), continued_oracle.data(), continued.size() * sizeof(float)));
        }
    }

    /**
     * @brief Compare ordinary whole/chunked prefill against the independent interpreter.
     * @param dk Positive key width with dense cancellation inputs.
     * @param dv Positive value width, including full wide tiles and their tails.
     * @param rows Live input rows.
     * @param phase Initial canonical pending-history length.
     * @param split Positive first-call row count, strictly below rows.
     *
     * Production state stays in place; output row padding and both end guards
     * remain poisoned. The interpreter spells every FP32 operation independently.
     * Chunking must not add a checkpoint boundary or alter the final raw history.
     */
    void checkOrdinaryPartitions(int dk, int dv, int rows, int phase, int split)
    {
        using namespace llaminar2;
        SCOPED_TRACE(::testing::Message() << "dk=" << dk << " dv=" << dv
            << " rows=" << rows << " phase=" << phase << " split=" << split);
        const CPUGDNStateLayout layout(1, dk, dv);
        const int qstride = dk + 3, vstride = dv + 3;
        std::vector<float> q(rows * qstride, -1234.0f), k(q.size(), -1234.0f),
            v(rows * vstride, -1234.0f), decay(rows), beta(rows), initial(layout.headFloats(), 0.0f);
        for (int j = 0; j < layout.matrixFloats(); ++j)
            initial[1 + j] = static_cast<float>((j * 17) % 47 - 23) / 256.0f;
        for (int row = 0; row < rows; ++row)
        {
            for (int j = 0; j < dk; ++j)
            {
                q[row * qstride + j] = static_cast<float>((row * 7 + j * 13) % 31 - 15) / 64.0f;
                k[row * qstride + j] = static_cast<float>((row * 11 + j * 17) % 29 - 14) / 1024.0f;
            }
            for (int vi = 0; vi < dv; ++vi)
                v[row * vstride + vi] = static_cast<float>((row * 19 + vi * 7) % 23 - 11) / 32.0f;
            decay[row] = 0.71f + static_cast<float>(row % 7) * 0.021f;
            beta[row] = 0.17f + static_cast<float>(row % 5) * 0.037f;
        }
        for (int p = 0; p < phase; ++p)
            interpretPrepared(layout, initial, q.data(), k.data(), v.data(), decay[0], beta[0]);
        auto oracle_state = initial, whole_state = initial, split_state = initial;
        std::vector<float> expected(rows * vstride + 2, -1234.0f), whole = expected, chunked = expected;
        for (int row = 0; row < rows; ++row)
        {
            const auto output = interpretPrepared(layout, oracle_state, q.data() + row * qstride,
                k.data() + row * qstride, v.data() + row * vstride, decay[row], beta[row]);
            std::copy(output.begin(), output.end(), expected.begin() + 1 + row * vstride);
        }
        cpu_gdn_epoch::run(layout, whole_state.data(), whole_state.data(), q.data(), k.data(),
            v.data(), decay.data(), beta.data(), rows, qstride, vstride, 1, whole.data() + 1, vstride);
        cpu_gdn_epoch::run(layout, split_state.data(), split_state.data(), q.data(), k.data(),
            v.data(), decay.data(), beta.data(), split, qstride, vstride, 1, chunked.data() + 1, vstride);
        cpu_gdn_epoch::run(layout, split_state.data(), split_state.data(), q.data() + split * qstride,
            k.data() + split * qstride, v.data() + split * vstride, decay.data() + split, beta.data() + split,
            rows - split, qstride, vstride, 1, chunked.data() + 1 + split * vstride, vstride);
        ASSERT_EQ(0, std::memcmp(whole.data(), expected.data(), expected.size() * sizeof(float)));
        ASSERT_EQ(0, std::memcmp(chunked.data(), expected.data(), expected.size() * sizeof(float)));
        ASSERT_EQ(0, std::memcmp(whole_state.data(), oracle_state.data(), oracle_state.size() * sizeof(float)));
        ASSERT_EQ(0, std::memcmp(split_state.data(), oracle_state.data(), oracle_state.size() * sizeof(float)));
    }
}

/** @brief Direct retained transactions preserve every accepted state and continuation. */
TEST(CPUFixedEpochQueryBlock, DirectRetainedTransactionsMatchIndependentProgram)
{
    // Real CPU benchmark extent is one595-row call; 512+83 is its explicit
    // continuation control, not a claim that CPU capture used two buckets.
    for (const int phase : {0, 7})
        checkOrdinaryPartitions(128, 128, 595, phase, 512);
    // Wide complete tiles plus16-lane and scalar tails; a non-epoch split also
    // proves that changing physical call boundaries cannot change arithmetic.
    for (const int dv : {128, 129, 255, 256, 257})
        {
            checkFlushPipeline(7, dv, false);
            for (int phase = 0; phase < 8; ++phase)
            {
                SCOPED_TRACE(::testing::Message() << "blocking dv=" << dv << " phase=" << phase);
                checkOrdinaryPartitions(7, dv, 17, phase, 5);
            }
        }
    for (const int dk : {1, 7, 64, 128, 257})
        for (const int dv : {1, 15, 16, 17, 65, 128, 129})
            checkFlushPipeline(dk, dv, false);
    for (const int dk : {1, 64, 128})
        for (const int dv : {1, 17, 128})
            checkFlushPipeline(dk, dv, true);
    for (const int dk : {1, 7, 8, 15, 17, 65, 128, 257, 512})
        for (const int dv : {1, 5, 17, 65, 129})
            for (int phase = 0; phase < 8; ++phase)
                for (const int rows : {1, 2, 3, 4, 8, 17})
                {
                    SCOPED_TRACE(::testing::Message() << "dk=" << dk << " dv=" << dv
                        << " phase=" << phase << " rows=" << rows);
                    const llaminar2::CPUGDNStateLayout layout(1, dk, dv);
                    std::vector<float> q(rows * dk), k(q.size()), v(rows * dv),
                        decay(rows), beta(rows), initial(layout.headFloats(), 0.0f);
                    for (int i = 0; i < layout.matrixFloats(); ++i) initial[1 + i] = static_cast<float>(i % 17 - 8) / 256.0f;
                    for (int row = 0; row < rows; ++row)
                    {
                        for (int j = 0; j < dk; ++j)
                        {
                            q[row * dk + j] = static_cast<float>((row * 7 + j * 13) % 31 - 15) / 64.0f;
                            k[row * dk + j] = static_cast<float>((row * 11 + j * 17) % 29 - 14) / 1024.0f;
                        }
                        for (int vi = 0; vi < dv; ++vi) v[row * dv + vi] = static_cast<float>((row * 19 + vi * 7) % 23 - 11) / 32.0f;
                        decay[row] = 0.71f + static_cast<float>(row % 7) * 0.021f;
                        beta[row] = 0.17f + static_cast<float>(row % 5) * 0.037f;
                    }
                    for (int p = 0; p < phase; ++p)
                        interpretPrepared(layout, initial, q.data(), k.data(), v.data(), decay[0], beta[0]);
                    const auto untouched = initial;
                    std::vector<float> slots(llaminar2::CPUSharedGDNSnapshots::headFloats(layout, rows), -1234.0f),
                        output(rows * dv), oracle = initial, accepted(layout.headFloats());
                    llaminar2::CPUSharedGDNSnapshots transaction(layout, slots.data(), rows);
                    llaminar2::cpu_gdn_epoch::runShared(layout, initial.data(), q.data(), k.data(), v.data(),
                        decay.data(), beta.data(), rows, dk, dv, 1, output.data(), dv, transaction);
                    ASSERT_EQ(0, std::memcmp(initial.data(), untouched.data(), initial.size() * sizeof(float)));
                    // Every retained prefix must survive unrelated mutation of
                    // the former canonical source after synchronous capture.
                    std::fill(initial.begin(),initial.end(),-9876.0f);
                    for (int row = 0; row < rows; ++row)
                    {
                        const auto expected = interpretPrepared(layout, oracle, q.data() + row * dk,
                            k.data() + row * dk, v.data() + row * dv, decay[row], beta[row]);
                        ASSERT_EQ(0, std::memcmp(output.data() + row * dv, expected.data(), dv * sizeof(float)));
                        ASSERT_TRUE(transaction.validRow(row));
                        transaction.materialize(row, accepted.data());
                        ASSERT_EQ(0, std::memcmp(accepted.data(), oracle.data(), oracle.size() * sizeof(float)));
                        // Every accepted prefix must continue with identical bytes
                        // through the ordinary encoded-state numeric helpers.
                        auto continued_oracle = oracle;
                        const auto expected_next = interpretPrepared(layout, continued_oracle,
                            q.data(), k.data(), v.data(), decay[0], beta[0]);
                        std::vector<float> continued_output(dv);
                        llaminar2::cpu_gdn_epoch::run(layout, accepted.data(), accepted.data(), q.data(), k.data(),
                            v.data(), decay.data(), beta.data(), 1, dk, dv, 1, continued_output.data(), dv);
                        ASSERT_EQ(0, std::memcmp(continued_output.data(), expected_next.data(), dv * sizeof(float)));
                        ASSERT_EQ(0, std::memcmp(accepted.data(), continued_oracle.data(), accepted.size() * sizeof(float)));
                    }
                }
}

/** @brief Synchronous retention cannot publish until its complete readout returns. */
TEST(CPUFixedEpochQueryBlock, InitialReadoutLifecycleRejectsEarlyPublicationAndReaderWrites)
{
    using namespace llaminar2;
    // Local build permission must have exactly one owner. Fresh read-only
    // views still resolve persisted publication metadata through construction.
    static_assert(!std::is_copy_constructible_v<CPUSharedGDNSnapshots>);
    static_assert(!std::is_move_constructible_v<CPUSharedGDNSnapshots>);
    static_assert(!std::is_copy_assignable_v<CPUSharedGDNSnapshots>);
    static_assert(!std::is_move_assignable_v<CPUSharedGDNSnapshots>);
    const CPUGDNStateLayout layout(1,7,17);
    std::vector<float> initial(layout.headFloats(),0.0f);
    for(int j=0;j<layout.matrixFloats();++j) initial[1+j]=(j+1)*0.125f;
    std::vector<float> slots(CPUSharedGDNSnapshots::headFloats(layout,1),-777.0f);
    std::vector<float> raw(layout.recordFloats(),0.25f),accepted(layout.headFloats());
    CPUSharedGDNSnapshots transaction(layout,slots.data(),1);
    EXPECT_THROW(transaction.complete(),std::logic_error);
    bool called=false;
    transaction.beginWithInitialReadout(initial.data(),1,[&](const float *source,float *retained) {
        called=true;
        EXPECT_EQ(source,initial.data()+1);
        EXPECT_EQ(retained,transaction.checkpoint(0));
        EXPECT_FALSE(transaction.validRow(0));
        EXPECT_THROW(transaction.complete(),std::logic_error);
        EXPECT_THROW(transaction.begin(initial.data(),1),std::logic_error);
        EXPECT_THROW(transaction.retire(),std::logic_error);
        transaction.append(0,raw.data());
        std::copy_n(source,layout.matrixFloats(),retained);
    });
    ASSERT_TRUE(called);
    EXPECT_FALSE(transaction.validRow(0));
    transaction.complete();
    ASSERT_TRUE(transaction.validRow(0));
    std::fill(initial.begin(),initial.end(),-999.0f);
    CPUSharedGDNSnapshots reader(layout,slots.data(),1);
    ASSERT_TRUE(reader.validRow(0));
    const auto before=slots;
    EXPECT_THROW(reader.complete(),std::logic_error);
    EXPECT_THROW(reader.append(0,raw.data()),std::logic_error);
    EXPECT_EQ(slots,before);
    reader.materialize(0,accepted.data());
    EXPECT_EQ(accepted[0],1.0f);
    for(int j=0;j<layout.matrixFloats();++j) EXPECT_EQ(accepted[1+j],(j+1)*0.125f);
    EXPECT_EQ(std::memcmp(layout.record(accepted.data(),0),raw.data(),raw.size()*sizeof(float)),0);
}

/** @brief Failed first readouts keep both old and new candidates unpublished. */
TEST(CPUFixedEpochQueryBlock, InitialReadoutExceptionRetiresPublicationAndCanRestart)
{
    using namespace llaminar2;
    const CPUGDNStateLayout layout(1,7,17);
    std::vector<float> initial(layout.headFloats(),0.0f),raw(layout.recordFloats(),0.25f);
    std::vector<float> slots(CPUSharedGDNSnapshots::headFloats(layout,1),-777.0f);
    CPUSharedGDNSnapshots transaction(layout,slots.data(),1);
    transaction.begin(initial.data(),1); transaction.append(0,raw.data()); transaction.complete();
    ASSERT_TRUE(transaction.validRow(0));
    EXPECT_THROW(transaction.beginWithInitialReadout(initial.data(),1,[&](const float *,float *retained) {
        EXPECT_FALSE(transaction.validRow(0));
        retained[0]=123.0f;
        throw std::runtime_error("forced incomplete checkpoint retention");
    }),std::runtime_error);
    EXPECT_FALSE(transaction.validRow(0));
    EXPECT_FALSE(CPUSharedGDNSnapshots(layout,slots.data(),1).validRow(0));
    EXPECT_THROW(transaction.complete(),std::logic_error);
    transaction.beginWithInitialReadout(initial.data(),1,[&](const float *source,float *retained) {
        std::copy_n(source,layout.matrixFloats(),retained);
        transaction.append(0,raw.data());
    });
    transaction.complete();
    ASSERT_TRUE(transaction.validRow(0));
}
