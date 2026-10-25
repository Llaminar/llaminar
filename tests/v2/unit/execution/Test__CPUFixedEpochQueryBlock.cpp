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
 */
#include "kernels/cpu/gdn/CPUFixedEpochQueryBlock.h"
#include "kernels/cpu/gdn/CPUFixedEpochGDN.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
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
}

/** @brief Direct retained transactions preserve every accepted state and continuation. */
TEST(CPUFixedEpochQueryBlock, DirectRetainedTransactionsMatchIndependentProgram)
{
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
