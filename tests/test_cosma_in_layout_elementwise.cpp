#include "cosma_prefill_manager.h"
#include <gtest/gtest.h>
#include <random>
#include <cmath>

using namespace llaminar;

static void reference_rmsnorm(const std::vector<float>& in, std::vector<float>& out, const std::vector<float>& w, int seq, int hid, float eps) {
    for (int r=0;r<seq;++r) {
        double sum=0.0; const float* row=&in[r*hid];
        for (int c=0;c<hid;++c) sum += double(row[c])*row[c];
        double inv = 1.0/std::sqrt(sum/hid + eps);
        for (int c=0;c<hid;++c) out[r*hid+c] = float(row[c]*inv*w[c]);
    }
}

static inline float silu(float x){return x/(1.0f+std::exp(-x));}
static void reference_swiglu(const std::vector<float>& gate, const std::vector<float>& up, std::vector<float>& out, int seq, int hid) {
    for (int r=0;r<seq;++r) for(int c=0;c<hid;++c) out[r*hid+c]=silu(up[r*hid+c])*gate[r*hid+c];
}

TEST(CosmaInLayoutElementwiseTest, RMSNormAndSwiGLUConsistencySingleRank)
{
    const int seq=8; const int hid=16; float eps=1e-5f;
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.f,1.f);
    std::vector<float> input(seq*hid), gamma(hid), gate(seq*hid), up(seq*hid);
    for (auto &v: input) v=dist(rng);
    for (auto &v: gamma) v=0.5f+dist(rng);
    for (auto &v: gate) v=dist(rng);
    for (auto &v: up) v=dist(rng);

    auto &mgr = CosmaPrefillManager::instance();
    // Force disable COSMA gating so we can create views with world_size=1 path.
    // Create activation view via conversion (single-rank path leaves original pointer).
    auto act = mgr.convert_activation_in(input.data(), seq, hid);
    std::vector<float> rms_out(seq*hid, 0.f);
    auto act_out = mgr.convert_activation_in(rms_out.data(), seq, hid); // separate output buffer

    ASSERT_TRUE(mgr.rmsnorm_in_layout(act, act_out, gamma.data(), seq, hid, eps));

    std::vector<float> ref_rms(seq*hid);
    reference_rmsnorm(input, ref_rms, gamma, seq, hid, eps);

    // Reconstruct row-major from act_out (single rank: direct pointer).
    std::vector<float> got_rms(seq*hid);
    mgr.to_row_major(act_out, got_rms.data());

    double num=0.0, den=0.0; for(int i=0;i<seq*hid;++i){ double d=double(got_rms[i])-ref_rms[i]; num+=d*d; den+=double(ref_rms[i])*ref_rms[i]; }
    double rel_l2 = std::sqrt(num/(den+1e-30));
    EXPECT_LT(rel_l2, 5e-4) << "RMSNorm rel_l2 too large";

    // SwiGLU
    auto gate_v = mgr.convert_activation_in(gate.data(), seq, hid);
    auto up_v = mgr.convert_activation_in(up.data(), seq, hid);
    std::vector<float> sw_out(seq*hid, 0.f);
    auto out_v = mgr.convert_activation_in(sw_out.data(), seq, hid);
    ASSERT_TRUE(mgr.swiglu_in_layout(gate_v, up_v, out_v, seq, hid));

    std::vector<float> ref_sw(seq*hid); reference_swiglu(gate, up, ref_sw, seq, hid);
    std::vector<float> got_sw(seq*hid); mgr.to_row_major(out_v, got_sw.data());
    num=0.0; den=0.0; for(int i=0;i<seq*hid;++i){ double d=double(got_sw[i])-ref_sw[i]; num+=d*d; den+=double(ref_sw[i])*ref_sw[i]; }
    rel_l2 = std::sqrt(num/(den+1e-30));
    EXPECT_LT(rel_l2, 5e-4) << "SwiGLU rel_l2 too large";
}
