/**
 * @file Test__CPUFixedEpochGDN.cpp
 * @brief Independent scalar interpreter of the fixed-epoch CPU GDN contract.
 *
 * Authenticate production M=1 and grouped outputs plus complete serialized
 * state at every checkpoint phase, including SIMD tails and acceptance. The
 * interpreter does not call production contraction or checkpoint helpers.
 * A premature-flush negative control proves history is part of arithmetic.
 */
#include "kernels/cpu/gdn/CPUGatedDeltaNet.h"
#include "kernels/CPUSharedGDNSnapshots.h"
#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace
{
    /**
     * @brief Interpret one new readout and ordinary rounded state transition.
     * @param state In/out key-major parent.
     * @param query Raw query before head scaling.
     * @param key Supplied normalized key.
     * @param value Value vector.
     * @param alpha Raw decay input.
     * @param beta_raw Raw sigmoid write input.
     * @param a Negative exponential A parameter.
     * @param bias Decay time bias.
     * @return Canonical readout with no implementation helper calls.
     */
    std::vector<float> reference(std::vector<float> &state,
        const std::vector<float> &query, const std::vector<float> &key,
        const std::vector<float> &value, float alpha, float beta_raw, float a, float bias)
    {
        constexpr int period = 8;
        const int dk = query.size(), dv = value.size(), record = dk + dv + 2;
        const int count = static_cast<int>(state[0]) + 1;
        const auto raw = [&](int i) { return state.data() + 1 + dk*dv + i*record; };
        const float x = alpha + bias;
        const float decay = std::exp(a * (x > 20.0f ? x : std::log1p(std::exp(x))));
        const float beta = 1.0f / (1.0f + std::exp(-beta_raw));
        const float scale = 1.0f / std::sqrt(static_cast<float>(dk));
        std::copy(key.begin(), key.end(), raw(count-1));
        std::copy(value.begin(), value.end(), raw(count-1)+dk);
        raw(count-1)[dk+dv] = decay; raw(count-1)[dk+dv+1] = beta;
        std::vector<float> adjusted(dk), coefficients(count), output(dv);
        for (int j = 0; j < dk; ++j) adjusted[j] = query[j] * scale;
        for (int p = count-1; p >= 0; --p)
        {
            float dot = 0;
            for (int j = 0; j < dk; ++j) dot = std::fma(raw(p)[j], adjusted[j], dot);
            coefficients[p] = raw(p)[dk+dv+1] * dot;
            for (int j = 0; j < dk; ++j)
                adjusted[j] = raw(p)[dk+dv] * std::fma(-coefficients[p], raw(p)[j], adjusted[j]);
        }
        for (int vi = 0; vi < dv; ++vi)
        {
            float readout = 0;
            for (int j = 0; j < dk; ++j) readout = std::fma(state[1+j*dv+vi], adjusted[j], readout);
            for (int p = 0; p < count; ++p) readout = std::fma(coefficients[p], raw(p)[dk+vi], readout);
            output[vi] = readout;
        }
        state[0] = static_cast<float>(count);
        if (count == period)
        {
            for (int p = 0; p < period; ++p)
                for (int vi = 0; vi < dv; ++vi)
                {
                    float projection = 0;
                    for (int j = 0; j < dk; ++j)
                    {
                        float &s = state[1+j*dv+vi]; s *= raw(p)[dk+dv];
                        projection = std::fma(s, raw(p)[j], projection);
                    }
                    const float delta = (raw(p)[dk+vi]-projection)*raw(p)[dk+dv+1];
                    for (int j = 0; j < dk; ++j)
                    {
                        float &s = state[1+j*dv+vi]; s = std::fma(raw(p)[j], delta, s);
                    }
                }
            std::fill(raw(0), state.data()+state.size(), 0.0f);
            state[0] = 0;
        }
        return output;
    }
}

TEST(CPUFixedEpochGDN, IndependentProgramTailsAndContinuation)
{
    for (const int dk : {1, 7, 64, 128, 512})
        for (const int dv : {1, 5, 8, 16, 17, 32, 64, 65, 128, 257, 512})
        {
            SCOPED_TRACE(::testing::Message() << "dk=" << dk << " dv=" << dv);
            llaminar2::CPUGatedDeltaNet kernel;
            std::vector<float> state(1+dk*dv+8*(dk+dv+2), 0.0f), q(dk), k(dk), v(dv), output(dv);
            for (int j = 0; j < dk*dv; ++j) state[1+j] = ((j*13)%31 - 15) / 128.0f;
            auto expected_state = state;
            for (int row = 0; row < 17; ++row)
            {
                for (int j = 0; j < dk; ++j)
                {
                    q[j] = ((j*7 + row)%17 - 8) / 16.0f;
                    k[j] = ((j*11 + row)%19 - 9) / (16.0f*std::sqrt(static_cast<float>(dk)));
                }
                for (int vi = 0; vi < dv; ++vi) v[vi] = ((vi*3 + row)%13 - 6) / 32.0f;
                const float alpha = row / 4.0f, beta = row - 2.0f, a = -1.0f, bias = 0;
                const auto expected = reference(expected_state, q, k, v, alpha, beta, a, bias);
                ASSERT_TRUE(kernel.recurrent_step(q.data(), k.data(), v.data(),
                    &alpha, &beta, &a, &bias, output.data(), state.data(), 1, dk, dv, false));
                EXPECT_EQ(std::memcmp(output.data(), expected.data(), dv*sizeof(float)), 0);
                EXPECT_EQ(std::memcmp(state.data(), expected_state.data(), state.size()*sizeof(float)), 0);
            }
        }
}

TEST(CPUFixedEpochGDN, CancellationWitnessRetainsUnflushedHistory)
{
    llaminar2::CPUGatedDeltaNet kernel;
    const float q = 1, k = 1, alpha = 0, beta = 100, a = 0, bias = 0;
    std::vector<float> value(16, std::ldexp(1.0f, -25)), state(llaminar2::CPUGDNStateLayout(1,1,16).floats(), 0.0f), output(16);
    std::fill_n(state.data()+1,16,1.0f);
    ASSERT_TRUE(kernel.recurrent_step(&q, &k, value.data(), &alpha, &beta, &a, &bias,
        output.data(), state.data(), 1, 1, 16, false));
    for (int i = 0; i < 16; ++i)
    {
        EXPECT_EQ(output[i], value[i]);
        EXPECT_EQ(state[1+i], 1.0f);
    }
    const float zero_key = 0;
    const auto saved = state;
    ASSERT_TRUE(kernel.recurrent_step(&q, &zero_key, value.data(), &alpha, &beta, &a, &bias,
        output.data(), state.data(), 1, 1, 16, false));
    for (float element : output) EXPECT_EQ(element, std::ldexp(1.0f, -25));
    // A dense-only snapshot prematurely rounds the erased checkpoint to zero.
    // Repeating this continuation must expose that invalid publication contract.
    std::vector<float> prematurely_flushed(saved.size(), 0.0f), wrong(16);
    ASSERT_TRUE(kernel.recurrent_step(&q, &zero_key, value.data(), &alpha, &beta, &a, &bias,
        wrong.data(), prematurely_flushed.data(), 1, 1, 16, false));
    for (float element : wrong) EXPECT_EQ(element, 0.0f);
    EXPECT_NE(std::memcmp(output.data(), wrong.data(), 16*sizeof(float)), 0);
}

TEST(CPUFixedEpochGDN, RejectsMalformedPhaseBeforeMutatingOutputs)
{
    const llaminar2::CPUGDNStateLayout layout(1,1,1);
    llaminar2::CPUGatedDeltaNet kernel;
    const float value=1,alpha=0,beta=0,a=-1,bias=0;
    for (float phase : {-1.0f,0.5f,8.0f,std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity()})
    {
        std::vector<float> state(layout.floats(),0.0f);state[0]=phase;
        const auto initial=state;float output=123;
        EXPECT_FALSE(kernel.recurrent_step(&value,&value,&value,&alpha,&beta,&a,&bias,
            &output,state.data(),1,1,1,false));
        EXPECT_EQ(output,123);
        EXPECT_EQ(std::memcmp(initial.data(),state.data(),state.size()*sizeof(float)),0);
    }
    EXPECT_THROW((llaminar2::CPUGDNStateLayout(0,1,1)), std::invalid_argument);
    EXPECT_THROW((llaminar2::CPUGDNStateLayout(std::numeric_limits<int>::max(),512,512)), std::overflow_error);
}

TEST(CPUFixedEpochGDN, GroupedEveryPhaseSnapshotsAndRejectedContinuation)
{
    constexpr int dk=7,dv=17,heads=1;
    const llaminar2::CPUGDNStateLayout layout(heads,dk,dv);
    for (int phase=0;phase<8;++phase)
        for (int rows : {1,2,3,4,7,8,9,15,16,31,63})
        {
            SCOPED_TRACE(::testing::Message()<<"phase="<<phase<<" rows="<<rows);
            std::vector<float> initial(layout.floats(),0.0f),q(rows*dk),k(rows*dk),v(rows*dv),alpha(rows),beta(rows);
            const float a=-0.5f,bias=0.1f;
            for (int j=0;j<dk*dv;++j) initial[1+j]=((j*13)%31-15)/128.0f;
            for (int t=0;t<rows;++t)
            {
                alpha[t]=t/9.0f; beta[t]=t/11.0f;
                for (int j=0;j<dk;++j) { q[t*dk+j]=((t+j*3)%17-8)/32.0f; k[t*dk+j]=((t+j*7)%19-9)/64.0f; }
                for (int j=0;j<dv;++j) v[t*dv+j]=((t+j*11)%23-11)/32.0f;
            }
            std::vector<float> first_q(q.begin(),q.begin()+dk),first_k(k.begin(),k.begin()+dk),first_v(v.begin(),v.begin()+dv);
            for (int t=0;t<phase;++t) reference(initial,first_q,first_k,first_v,alpha[0],beta[0],a,bias);
            auto expected_state=initial;
            std::vector<float> expected(rows*dv),states(rows*layout.floats());
            for (int t=0;t<rows;++t)
            {
                const auto out=reference(expected_state,{q.begin()+t*dk,q.begin()+(t+1)*dk},
                    {k.begin()+t*dk,k.begin()+(t+1)*dk},{v.begin()+t*dv,v.begin()+(t+1)*dv},alpha[t],beta[t],a,bias);
                std::copy(out.begin(),out.end(),expected.begin()+t*dv);
                std::copy(expected_state.begin(),expected_state.end(),states.begin()+t*layout.floats());
            }
            llaminar2::CPUGatedDeltaNet grouped;
            auto live=initial;
            std::vector<float> output(rows*dv),snapshots(rows*layout.floats());
            ASSERT_TRUE(grouped.chunkForwardWithStateSnapshots(q.data(),k.data(),v.data(),alpha.data(),beta.data(),&a,&bias,
                output.data(),live.data(),rows,heads,dk,dv,64,false,snapshots.data(),layout.floats(),rows));
            live=initial; // The explicit serialization boundary publishes its terminal value.
            grouped.bindVerifierStateCaptureWorkspace(snapshots.data(),rows,layout.floats());
            EXPECT_EQ(std::memcmp(output.data(),expected.data(),output.size()*sizeof(float)),0);
            EXPECT_EQ(std::memcmp(snapshots.data(),states.data(),states.size()*sizeof(float)),0);
            for (int accepted=-1;accepted<rows;++accepted)
            {
                auto committed=initial;
                if (accepted>=0) ASSERT_TRUE(grouped.restoreVerifierStateCaptureRow(committed.data(),accepted,nullptr));
                const auto oracle=accepted<0 ? initial : std::vector<float>(states.begin()+accepted*layout.floats(),states.begin()+(accepted+1)*layout.floats());
                EXPECT_EQ(committed,oracle);
                auto next=oracle; const auto expected_next=reference(next,first_q,first_k,first_v,alpha[0],beta[0],a,bias);
                llaminar2::CPUGatedDeltaNet serial; std::vector<float> actual(dv);
                ASSERT_TRUE(serial.recurrent_step(q.data(),k.data(),v.data(),alpha.data(),beta.data(),&a,&bias,
                    actual.data(),committed.data(),heads,dk,dv,false));
                EXPECT_EQ(std::memcmp(actual.data(),expected_next.data(),dv*sizeof(float)),0);
                EXPECT_EQ(committed,next);
            }
        }
}

TEST(CPUFixedEpochGDN, RejectsAliasedOrUndersizedCaptureBeforeStateMutation)
{
    const llaminar2::CPUGDNStateLayout layout(1, 1, 1);
    const float value = 1, alpha = 0, beta = 0, a = -1, bias = 0;
    std::vector<float> state(layout.floats(), 0.0f), slots(layout.floats(), 0.0f);
    const auto initial = state;
    float output = 123;
    llaminar2::CPUGatedDeltaNet kernel;
    kernel.bindVerifierStateCaptureWorkspace(state.data(), 1, layout.floats());
    EXPECT_THROW(kernel.recurrent_step(&value, &value, &value, &alpha, &beta, &a, &bias,
        &output, state.data(), 1, 1, 1, false),std::invalid_argument);
    EXPECT_EQ(output, 123);
    EXPECT_EQ(state, initial);
    kernel.bindVerifierStateCaptureWorkspace(slots.data(), 1, layout.floats() - 1);
    EXPECT_THROW(kernel.chunk_forward(&value, &value, &value, &alpha, &beta, &a, &bias,
        &output, state.data(), 1, 1, 1, 1, 1, false),std::invalid_argument);
    EXPECT_EQ(output, 123);
    EXPECT_EQ(state, initial);
    EXPECT_EQ(slots, initial);
}

TEST(CPUFixedEpochGDN, SharedSnapshotsEveryPhaseAndAcceptedContinuation)
{
    constexpr int dk=7,dv=17,heads=1;
    const llaminar2::CPUGDNStateLayout layout(heads,dk,dv);
    for (int phase=0;phase<8;++phase)
        for (int rows : {1,2,3,4,7,8,9,15,16,31,63})
        {
            SCOPED_TRACE(::testing::Message()<<"phase="<<phase<<" rows="<<rows);
            std::vector<float> initial(layout.floats(),0.0f),q(rows*dk),k(rows*dk),v(rows*dv),alpha(rows),beta(rows);
            const float a=-0.5f,bias=0.1f;
            for (int j=0;j<dk*dv;++j) initial[1+j]=((j*13)%31-15)/128.0f;
            for (int t=0;t<rows;++t)
            {
                alpha[t]=t/9.0f; beta[t]=t/11.0f;
                for (int j=0;j<dk;++j) { q[t*dk+j]=((t+j*3)%17-8)/32.0f; k[t*dk+j]=((t+j*7)%19-9)/64.0f; }
                for (int j=0;j<dv;++j) v[t*dv+j]=((t+j*11)%23-11)/32.0f;
            }
            std::vector<float> first_q(q.begin(),q.begin()+dk),first_k(k.begin(),k.begin()+dk),first_v(v.begin(),v.begin()+dv);
            for (int t=0;t<phase;++t) reference(initial,first_q,first_k,first_v,alpha[0],beta[0],a,bias);
            auto expected_state=initial;
            std::vector<float> expected(rows*dv),states(rows*layout.floats());
            for (int t=0;t<rows;++t)
            {
                const auto out=reference(expected_state,{q.begin()+t*dk,q.begin()+(t+1)*dk},
                    {k.begin()+t*dk,k.begin()+(t+1)*dk},{v.begin()+t*dv,v.begin()+(t+1)*dv},alpha[t],beta[t],a,bias);
                std::copy(out.begin(),out.end(),expected.begin()+t*dv);
                std::copy(expected_state.begin(),expected_state.end(),states.begin()+t*layout.floats());
            }
            llaminar2::CPUGatedDeltaNet grouped;
            auto live=initial;
            std::vector<float> output(rows*dv),snapshots(rows*layout.floats());
            std::vector<float> slots(llaminar2::CPUSharedGDNSnapshots::headFloats(layout,rows));
            grouped.bindVerifierStateCaptureWorkspace(slots.data(),rows,layout.floats());
            grouped.bindCPUSharedStateCapture(slots.data(),slots.size(),rows);
            ASSERT_TRUE(grouped.chunk_forward(q.data(),k.data(),v.data(),alpha.data(),beta.data(),&a,&bias,
                output.data(),live.data(),rows,heads,dk,dv,64,false));
            EXPECT_EQ(live,initial);
            EXPECT_EQ(std::memcmp(output.data(),expected.data(),output.size()*sizeof(float)),0);
            for (int row=0;row<rows;++row)
                ASSERT_TRUE(grouped.restoreVerifierStateCaptureRow(snapshots.data()+row*layout.floats(),row,nullptr));
            EXPECT_EQ(std::memcmp(snapshots.data(),states.data(),states.size()*sizeof(float)),0);
            for (int accepted=-1;accepted<rows;++accepted)
            {
                auto committed=initial;
                if (accepted>=0) ASSERT_TRUE(grouped.restoreVerifierStateCaptureRow(committed.data(),accepted,nullptr));
                const auto oracle=accepted<0 ? initial : std::vector<float>(states.begin()+accepted*layout.floats(),states.begin()+(accepted+1)*layout.floats());
                EXPECT_EQ(committed,oracle);
                auto next=oracle; const auto expected_next=reference(next,first_q,first_k,first_v,alpha[0],beta[0],a,bias);
                llaminar2::CPUGatedDeltaNet serial; std::vector<float> actual(dv);
                ASSERT_TRUE(serial.recurrent_step(q.data(),k.data(),v.data(),alpha.data(),beta.data(),&a,&bias,
                    actual.data(),committed.data(),heads,dk,dv,false));
                EXPECT_EQ(std::memcmp(actual.data(),expected_next.data(),dv*sizeof(float)),0);
                EXPECT_EQ(committed,next);
            }
        }
}

/** @brief Compact capture rejects invalid admission, stale lifetimes and undersized arenas. */
TEST(CPUFixedEpochGDN, SharedSnapshotsRejectIncompleteBindingAndRetiredTransactions)
{
    using namespace llaminar2;
    const CPUGDNStateLayout layout(1,7,17);
    constexpr int rows=9;
    std::vector<float> slots(CPUSharedGDNSnapshots::headFloats(layout,rows));
    std::vector<float> state(layout.floats()),output(rows*17,123.0f);
    std::vector<float> q(rows*7,0.1f),k(rows*7,0.2f),v(rows*17,0.3f),gate(rows,0.1f);
    const float a=-0.5f,bias=0.0f;
    CPUGatedDeltaNet kernel;
    EXPECT_THROW(kernel.bindCPUSharedStateCapture(slots.data(),slots.size(),0),std::invalid_argument);
    kernel.bindVerifierStateCaptureWorkspace(slots.data(),rows,layout.floats());
    kernel.bindCPUSharedStateCapture(slots.data(),slots.size()-1,rows);
    EXPECT_FALSE(kernel.chunk_forward(q.data(),k.data(),v.data(),gate.data(),gate.data(),&a,&bias,
        output.data(),state.data(),rows,1,7,17,64,false));
    EXPECT_EQ(output,std::vector<float>(rows*17,123.0f));
    EXPECT_FALSE(kernel.restoreVerifierStateCaptureRow(state.data(),0,nullptr));
    kernel.bindCPUSharedStateCapture(slots.data(),slots.size(),rows);
    ASSERT_TRUE(kernel.chunk_forward(q.data(),k.data(),v.data(),gate.data(),gate.data(),&a,&bias,
        output.data(),state.data(),rows,1,7,17,64,false));
    EXPECT_FALSE(kernel.restoreVerifierStateCaptureRow(state.data(),rows,nullptr));
    EXPECT_FALSE(kernel.restoreVerifierStateCaptureRow(slots.data(),0,nullptr));
    for (const uint32_t invalid : {0x7fc00000u,0xffffffffu,0x3f000000u,8u}) {
        const auto before=state;
        const uint32_t saved=std::bit_cast<uint32_t>(slots[0]);
        std::memcpy(slots.data(),&invalid,sizeof(invalid));
        EXPECT_FALSE(kernel.restoreVerifierStateCaptureRow(state.data(),0,nullptr));
        EXPECT_EQ(state,before);
        std::memcpy(slots.data(),&saved,sizeof(saved));
    }
    ASSERT_TRUE(kernel.resetGPUState(nullptr));
    EXPECT_FALSE(kernel.restoreVerifierStateCaptureRow(state.data(),0,nullptr));
}

/** @brief Production geometry retains checkpoints per epoch rather than per row. */
TEST(CPUFixedEpochGDN, SharedSnapshotBOMCountsEpochsAndRawUpdates)
{
    using namespace llaminar2;
    const CPUGDNStateLayout layout(64,128,128);
    const size_t per_head=(16+2*128*128+15*(128+128+2)+15)&~size_t{15};
    EXPECT_EQ(CPUSharedGDNSnapshots::headFloats(layout,8),per_head);
    EXPECT_LT(per_head*64,static_cast<size_t>(8)*layout.floats());
}

/** @brief Admitted checkpoint/log strides survive large-to-small-to-large replay. */
TEST(CPUFixedEpochGDN, SharedSnapshotCapacityReplayAndExplicitOwnerGeneration)
{
    using namespace llaminar2;
    constexpr int dk=7,dv=17,capacity=9;
    const CPUGDNStateLayout layout(1,dk,dv);
    std::vector<float> slots(CPUSharedGDNSnapshots::headFloats(layout,capacity));
    std::vector<float> initial(layout.floats()),q(capacity*dk,0.1f),k(capacity*dk,0.2f);
    std::vector<float> v(capacity*dv,0.3f),gate(capacity,0.1f),output(capacity*dv);
    const float a=-0.5f,bias=0.0f;
    CPUGatedDeltaNet kernel;
    kernel.bindVerifierStateCaptureWorkspace(slots.data(),capacity,layout.floats());
    kernel.bindCPUSharedStateCapture(slots.data(),slots.size(),capacity);
    uint64_t owner_generation=7;
    for (const int rows : {capacity,1,capacity}) {
        auto live=initial,oracle=initial;
        ASSERT_TRUE(kernel.chunk_forward(q.data(),k.data(),v.data(),gate.data(),gate.data(),&a,&bias,
            output.data(),live.data(),rows,1,dk,dv,64,false));
        EXPECT_EQ(live,initial);
        for(int row=0;row<rows;++row) {
            const auto expected=reference(oracle,{q.begin()+row*dk,q.begin()+(row+1)*dk},
                {k.begin()+row*dk,k.begin()+(row+1)*dk},{v.begin()+row*dv,v.begin()+(row+1)*dv},gate[row],gate[row],a,bias);
            EXPECT_EQ(std::memcmp(output.data()+row*dv,expected.data(),dv*sizeof(float)),0);
            auto accepted=initial;
            ASSERT_TRUE(kernel.restoreVerifierStateCaptureRow(accepted.data(),row,nullptr));
            EXPECT_EQ(accepted,oracle);
        }
        auto invalid=initial;
        EXPECT_FALSE(kernel.restoreVerifierStateCaptureRow(invalid.data(),rows,nullptr));
        const CPUSharedGDNTransaction transaction{.slots=slots.data(),.slot_floats=slots.size(),
            .width=capacity,.requests=1,.live_width=rows,.heads=1,.dk=dk,.dv=dv,
            .generation_authority=&owner_generation,.generation=owner_generation};
        const int selected=rows-1;
        kernel.bindVerifierStateCaptureWorkspace(nullptr,0,layout.floats());
        EXPECT_FALSE(kernel.restoreVerifierStateCaptureRow(invalid.data(),selected,nullptr));
        // Stage-owned immutable transactions survive another graph clearing the
        // shared kernel binding; only their own lifecycle authority retires them.
        ASSERT_TRUE(kernel.restoreCPUSharedStateRows(transaction,invalid.data(),&selected));
        EXPECT_EQ(invalid,oracle);
        ++owner_generation;
        const auto unchanged=invalid;
        EXPECT_FALSE(kernel.restoreCPUSharedStateRows(transaction,invalid.data(),&selected));
        EXPECT_EQ(invalid,unchanged);
        kernel.bindVerifierStateCaptureWorkspace(slots.data(),capacity,layout.floats());
    }
}

/** @brief Cache-line padding never changes explicitly admitted logical row ownership. */
TEST(CPUFixedEpochGDN, SharedSnapshotAlignedPayloadUsesExplicitLogicalCapacity)
{
    using namespace llaminar2;
    EXPECT_THROW(CPUSharedGDNLogicalCapacity(0,1),std::invalid_argument);
    EXPECT_THROW(CPUSharedGDNLogicalCapacity(5,2),std::invalid_argument);
    EXPECT_THROW(CPUSharedGDNLogicalCapacity(8,0),std::invalid_argument);
    for(int dk=1;dk<=7;++dk)
        for(int dv=1;dv<=7;++dv)
            for(int width=1;width<=17;++width)
            {
                const CPUGDNStateLayout layout(2,dk,dv);
                const CPUSharedGDNLogicalCapacity capacity(3*width,3);
                const size_t span=CPUSharedGDNSnapshots::headFloats(layout,width);
                EXPECT_EQ(span%16,0u);
                EXPECT_EQ(capacity.flatRows(),3*width);
                EXPECT_EQ(capacity.rowsPerRequest(),width);
                EXPECT_EQ(capacity.slotFloats(layout),3u*2u*span);
                // An independently constructed reader retains the same physical
                // head/log offsets; no row capacity is inferred from padding.
                std::vector<float> storage(capacity.slotFloats(layout)+16);
                const auto address=(reinterpret_cast<uintptr_t>(storage.data())+63u)&~uintptr_t{63u};
                auto *base=reinterpret_cast<float *>(address);
                for(int head=0;head<6;++head)
                {
                    CPUSharedGDNSnapshots view(layout,base+head*span,width);
                    EXPECT_EQ(reinterpret_cast<uintptr_t>(view.checkpoint(0))%64u,0u);
                }
            }
    const CPUGDNStateLayout smallest(1,1,1),boundary(1,2,2);
    EXPECT_EQ(CPUSharedGDNSnapshots::headFloats(smallest,1),CPUSharedGDNSnapshots::headFloats(smallest,2));
    EXPECT_EQ(CPUSharedGDNSnapshots::headFloats(boundary,8),CPUSharedGDNSnapshots::headFloats(boundary,9));
    EXPECT_NE(CPUSharedGDNSnapshots::checkpoints(8),CPUSharedGDNSnapshots::checkpoints(9));
}
