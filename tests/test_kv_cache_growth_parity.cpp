// Parity test: prefill + incremental decodes (with KV cache growth) vs replaying full sequence.
#include "distributed_transformer_pipeline.h" // DistributedTransformerPipeline
#include "gtest/gtest.h"
#include <cstdlib>
#include <cmath>

using namespace llaminar;

namespace
{
    struct TestWeightsBuilder
    {
        static DistributedTransformerPipeline::ModelWeights build(std::unique_ptr<DistributedTransformerPipeline> &pipeline,
                                                                  const DistributedTransformerPipeline::LayerConfig &cfg)
        {
            DistributedTransformerPipeline::ModelWeights w;
            auto make_matrix = [&](int rows, int cols)
            {
                auto t = pipeline->allocateTestLocalTensor({rows, cols});
                for (int r = 0; r < rows; ++r)
                {
                    for (int c = 0; c < cols; ++c)
                    {
                        t->data()[r * cols + c] = 0.001f * (r + 1) + 0.0001f * (c + 1);
                    }
                }
                return t;
            };
            auto make_vector = [&](int dim)
            { auto t = pipeline->allocateTestLocalTensor({dim}); for(int i=0;i<dim;++i) t->data()[i] = 1.0f; return t; };
            w.token_embedding = make_matrix(cfg.vocab_size, cfg.d_model);
            for (int l = 0; l < cfg.n_layers; ++l)
            {
                w.attn_norm_weight.push_back(make_vector(cfg.d_model));
                w.wq.push_back(make_matrix(cfg.d_model, cfg.n_head * cfg.head_dim));
                w.wk.push_back(make_matrix(cfg.d_model, cfg.n_head_kv * cfg.head_dim));
                w.wv.push_back(make_matrix(cfg.d_model, cfg.n_head_kv * cfg.head_dim));
                w.wo.push_back(make_matrix(cfg.n_head * cfg.head_dim, cfg.d_model));
                w.ffn_norm_weight.push_back(make_vector(cfg.d_model));
                w.w_gate.push_back(make_matrix(cfg.d_model, cfg.d_ff));
                w.w_up.push_back(make_matrix(cfg.d_model, cfg.d_ff));
                w.w_down.push_back(make_matrix(cfg.d_ff, cfg.d_model));
            }
            w.output_norm_weight = make_vector(cfg.d_model);
            w.lm_head = make_matrix(cfg.d_model, cfg.vocab_size);
            return w;
        }
    };
}

class KVCacheGrowthParityTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Ensure global capture store starts empty for this test (avoid contamination from prior tests)
        DistributedTransformerPipeline::resetLayerTokenRows();
        setenv("LLAMINAR_KV_DYNAMIC_INIT", "1", 1);
        setenv("LLAMINAR_KV_GROWTH_FACTOR", "2", 1);
        setenv("LLAMINAR_PIPELINE_LAYER_TOKEN_DIFF", "1", 1); // enable per-layer last-token capture
        // Also capture pre-LM head hidden state (post final RMSNorm) to isolate whether drift appears before or after LM head.
        setenv("LLAMINAR_PIPELINE_CAPTURE_PRE_LM", "1", 1);
        // Ensure noisy per-call LM head diff instrumentation is OFF unless explicitly enabled elsewhere.
        unsetenv("LLAMINAR_PIPELINE_PRE_LM_ROW_DIFF");
        debugEnvRefresh();
        cfg_.n_layers = 2;
        cfg_.n_head = 2;
        cfg_.n_head_kv = 2;
        cfg_.head_dim = 4;
        cfg_.d_model = 8;
        cfg_.d_ff = 16;
        cfg_.vocab_size = 32;
        cfg_.max_seq_len = 64;
        cfg_.eps = 1e-5f;
        pipeline_dynamic_ = createDistributedTransformerPipeline(cfg_);
        weights_ = TestWeightsBuilder::build(pipeline_dynamic_, cfg_);
        // Replay pipeline (allocate full cache up front by disabling dynamic init)
        unsetenv("LLAMINAR_KV_DYNAMIC_INIT");
        debugEnvRefresh();
        pipeline_replay_ = createDistributedTransformerPipeline(cfg_);
        weights_replay_ = TestWeightsBuilder::build(pipeline_replay_, cfg_);
        // Re-enable dynamic init for remainder (for subsequent tests if any)
        setenv("LLAMINAR_KV_DYNAMIC_INIT", "1", 1);
        debugEnvRefresh();
    }
    DistributedTransformerPipeline::LayerConfig cfg_;
    std::unique_ptr<DistributedTransformerPipeline> pipeline_dynamic_;
    std::unique_ptr<DistributedTransformerPipeline> pipeline_replay_;
    DistributedTransformerPipeline::ModelWeights weights_;
    DistributedTransformerPipeline::ModelWeights weights_replay_;
};

TEST_F(KVCacheGrowthParityTest, LogitsParityAcrossGrowth)
{
    ASSERT_NE(pipeline_dynamic_, nullptr);
    ASSERT_NE(pipeline_replay_, nullptr);
    // Initial prefill of 4 tokens
    std::vector<int> prefix = {1, 2, 3, 4};
    std::shared_ptr<TensorBase> prefill_logits_dyn;
    std::shared_ptr<TensorBase> prefill_logits_rep;
    ASSERT_TRUE(pipeline_dynamic_->execute(prefix, weights_, prefill_logits_dyn));
    ASSERT_TRUE(pipeline_replay_->execute(prefix, weights_replay_, prefill_logits_rep));
    // Generate 6 more tokens (will trigger at least one growth: capacity starts 4 -> 8 -> maybe 16)
    std::vector<int> generated;
    std::shared_ptr<TensorBase> last_step_logits; // capture logits of final incremental token
    int vocab = cfg_.vocab_size;
    for (int i = 0; i < 6; ++i)
    {
        int tok = (i + 5) % vocab;
        std::shared_ptr<TensorBase> step_logits;
        ASSERT_TRUE(pipeline_dynamic_->incrementalDecodeToken(tok, weights_, step_logits));
        generated.push_back(tok);
        // Per-step replay up to this point (prefix + generated so far)
        std::vector<int> partial = prefix;
        partial.insert(partial.end(), generated.begin(), generated.end());
        std::shared_ptr<TensorBase> partial_replay_logits;
        ASSERT_TRUE(pipeline_replay_->execute(partial, weights_replay_, partial_replay_logits));
        // Extract last row replay vs incremental current step
        // NOTE: incremental decode returns a logits tensor shaped [current_seq_len, vocab].
        // The test previously compared the FIRST row (token 0) of incremental logits against the
        // LAST row (new token) of the replay path – producing an artificial systematic drift.
        // Fix: point to the final row (current newly generated token) for the incremental path.
        int inc_seq_len = step_logits->shape()[0];
        const float *replay_step = partial_replay_logits->data() + (partial.size() - 1) * vocab;
        const float *inc_step = step_logits->data() + (inc_seq_len - 1) * vocab;
        double sum_sq = 0, ref_sq = 0, max_abs = 0;
        for (int j = 0; j < vocab; ++j)
        {
            double diff = (double)inc_step[j] - replay_step[j];
            sum_sq += diff * diff;
            ref_sq += (double)replay_step[j] * replay_step[j];
            double ad = fabs(diff);
            if (ad > max_abs)
                max_abs = ad;
        }
        double rel_l2 = ref_sq > 0 ? std::sqrt(sum_sq) / std::sqrt(ref_sq) : 0.0;
        if (::testing::Test::HasFailure())
        { /* leave early if assertion above failed */
        }
        // Log diagnostic without failing early to observe progression
        if (getenv("GTEST_PARITY_VERBOSE"))
        {
            std::ostringstream oss;
            oss << "[KVParityStep] step=" << i << " rel_l2=" << rel_l2 << " max_abs=" << max_abs;
            LOG_INFO(oss.str());
        }
        last_step_logits = step_logits; // retain last logits for final comparison
    }
    // Replay full sequence (prefix + generated)
    std::vector<int> full_seq = prefix;
    full_seq.insert(full_seq.end(), generated.begin(), generated.end());

    // Capture incremental pre-LM hidden (static buffer) BEFORE running the full replay execute which will overwrite it.
    std::vector<float> inc_pre_lm = DistributedTransformerPipeline::getLastPreLMHidden();
    if (getenv("GTEST_PARITY_VERBOSE") && pipeline_dynamic_->getRank() == 0)
    {
        LOG_INFO(std::string("[PreLMIncCapture] size=") + std::to_string(inc_pre_lm.size()));
    }

    // New diagnostic: compute manual LM head logits reference for incremental path BEFORE replay
    if (pipeline_dynamic_->getRank() == 0 && !inc_pre_lm.empty() && weights_.lm_head && last_step_logits)
    {
        int d_model = cfg_.d_model;
        int seq_len_full = (int)full_seq.size();
        int vocab = cfg_.vocab_size;
        size_t expect_seq = (size_t)seq_len_full * d_model;
        const float *hidden_final = nullptr;
        if (inc_pre_lm.size() == (size_t)d_model)
            hidden_final = inc_pre_lm.data();
        else if (inc_pre_lm.size() == expect_seq)
            hidden_final = inc_pre_lm.data() + (seq_len_full - 1) * d_model;
        if (hidden_final)
        {
            const float *W = weights_.lm_head->data();
            std::vector<float> inc_manual(vocab, 0.f);
            for (int j = 0; j < vocab; ++j)
            {
                double acc = 0.0;
                for (int i = 0; i < d_model; ++i)
                {
                    acc += (double)hidden_final[i] * (double)W[i * vocab + j];
                }
                inc_manual[j] = (float)acc;
            }
            int inc_seq_len = last_step_logits->shape()[0];
            const float *inc_logits_last = last_step_logits->data() + (size_t)(inc_seq_len - 1) * vocab;
            double sum_sq = 0, ref_sq = 0, max_abs = 0;
            for (int j = 0; j < vocab; ++j)
            {
                double diff = (double)inc_logits_last[j] - (double)inc_manual[j];
                sum_sq += diff * diff;
                ref_sq += (double)inc_manual[j] * (double)inc_manual[j];
                double a = fabs(diff);
                if (a > max_abs)
                    max_abs = a;
            }
            double rel_l2 = ref_sq > 0 ? std::sqrt(sum_sq) / std::sqrt(ref_sq) : 0.0;
            LOG_INFO("[IncManualLMHeadDiff] rel_l2=" << rel_l2 << " max_abs=" << max_abs
                                                     << " inc_seq_len=" << inc_seq_len
                                                     << " vocab=" << vocab
                                                     << " inc_logits_ptr=" << (const void *)inc_logits_last
                                                     << " manual_rows=" << 1);
            if (getenv("GTEST_PARITY_VERBOSE"))
            {
                for (int j = 0; j < std::min(vocab, 8); ++j)
                {
                    LOG_INFO("[IncManualLMHeadPreview] j=" << j
                                                           << " manual=" << inc_manual[j]
                                                           << " inc=" << inc_logits_last[j]
                                                           << " diff=" << (inc_logits_last[j] - inc_manual[j]));
                }
            }
        }
        else
        {
            LOG_WARN("[IncManualLMHeadDiff] Hidden final unavailable for manual incremental reference hidden_size=" << inc_pre_lm.size());
        }
    }

    std::shared_ptr<TensorBase> full_logits_replay;
    ASSERT_TRUE(pipeline_replay_->execute(full_seq, weights_replay_, full_logits_replay));

    // Capture replay pre-LM hidden after full execute
    std::vector<float> rep_pre_lm = DistributedTransformerPipeline::getLastPreLMHidden();
    if (getenv("GTEST_PARITY_VERBOSE") && pipeline_dynamic_->getRank() == 0)
    {
        LOG_INFO(std::string("[PreLMRepCapture] size=") + std::to_string(rep_pre_lm.size()));
    }

    // Compute diff of pre-LM hidden last token row (dimension d_model) if sizes match expected (seq_len * d_model for each path)
    if (pipeline_dynamic_->getRank() == 0)
    {
        int d_model = cfg_.d_model;
        int seq_len_full = (int)full_seq.size();
        size_t expect_seq = (size_t)seq_len_full * d_model;
        if (!inc_pre_lm.empty() && !rep_pre_lm.empty())
        {
            const float *inc_vec = nullptr;
            const float *rep_vec = nullptr;
            bool shape_ok = false;
            // Case A: incremental stored only last row, replay stored full sequence
            if (inc_pre_lm.size() == (size_t)d_model && rep_pre_lm.size() == expect_seq)
            {
                inc_vec = inc_pre_lm.data();
                rep_vec = rep_pre_lm.data() + (seq_len_full - 1) * d_model;
                shape_ok = true;
            }
            // Case B: both stored full sequence (e.g., implementation changed to keep rolling buffer for incremental path)
            else if (inc_pre_lm.size() == expect_seq && rep_pre_lm.size() == expect_seq)
            {
                inc_vec = inc_pre_lm.data() + (seq_len_full - 1) * d_model;
                rep_vec = rep_pre_lm.data() + (seq_len_full - 1) * d_model;
                shape_ok = true;
            }
            // Case C: both only last row (defensive)
            else if (inc_pre_lm.size() == (size_t)d_model && rep_pre_lm.size() == (size_t)d_model)
            {
                inc_vec = inc_pre_lm.data();
                rep_vec = rep_pre_lm.data();
                shape_ok = true;
            }
            if (shape_ok)
            {
                double sum_sq = 0, ref_sq = 0, max_abs = 0;
                for (int i = 0; i < d_model; ++i)
                {
                    double diff = (double)inc_vec[i] - rep_vec[i];
                    sum_sq += diff * diff;
                    ref_sq += (double)rep_vec[i] * rep_vec[i];
                    double a = fabs(diff);
                    if (a > max_abs)
                        max_abs = a;
                }
                double rel_l2 = ref_sq > 0 ? std::sqrt(sum_sq) / std::sqrt(ref_sq) : 0.0;
                LOG_INFO("[PreLMDiff] rel_l2=" << rel_l2 << " max_abs=" << max_abs << " d_model=" << d_model
                                               << " inc_size=" << inc_pre_lm.size() << " rep_size=" << rep_pre_lm.size() << " seq_len=" << seq_len_full);
            }
            else
            {
                LOG_WARN("[PreLMDiff] Unexpected capture sizes inc=" << inc_pre_lm.size() << " rep=" << rep_pre_lm.size()
                                                                     << " expected one of {inc=" << d_model << ", rep=" << expect_seq << "} OR {inc=" << expect_seq << ", rep=" << expect_seq
                                                                     << "} OR {inc=" << d_model << ", rep=" << d_model << "}");
            }
        }
        else
        {
            LOG_WARN("[PreLMDiff] Missing pre-LM captures inc_empty=" << (inc_pre_lm.empty() ? 1 : 0) << " rep_empty=" << (rep_pre_lm.empty() ? 1 : 0));
        }
    }

    // LM head reference verification: compute manual logits from captured final token hidden state and weight matrix.
    if (pipeline_dynamic_->getRank() == 0)
    {
        int d_model = cfg_.d_model;
        int seq_len_full = (int)full_seq.size();
        size_t expect_seq = (size_t)seq_len_full * d_model;
        const float *hidden_final = nullptr;
        if (!inc_pre_lm.empty())
        {
            if (inc_pre_lm.size() == (size_t)d_model)
                hidden_final = inc_pre_lm.data();
            else if (inc_pre_lm.size() == expect_seq)
                hidden_final = inc_pre_lm.data() + (seq_len_full - 1) * d_model;
        }
        if (hidden_final && weights_.lm_head)
        {
            int vocab = cfg_.vocab_size;
            const float *W = weights_.lm_head->data(); // layout [d_model, vocab]
            std::vector<float> ref_logits(vocab, 0.f);
            for (int j = 0; j < vocab; ++j)
            {
                double acc = 0.0;
                for (int i = 0; i < d_model; ++i)
                {
                    acc += (double)hidden_final[i] * (double)W[i * vocab + j];
                }
                ref_logits[j] = (float)acc;
            }
            // Extract incremental last logits (already computed in loop)
            // Extract replay last logits from full logits replay tensor captured later
            // We haven't yet computed full replay logits at this point? (Yes, full_logits_replay computed above.)
            if (full_logits_replay && last_step_logits)
            {
                const float *replay_last = full_logits_replay->data() + (size_t)(full_seq.size() - 1) * vocab;
                // incremental logits tensor is shaped [seq_len, vocab]; select its last row
                int inc_seq_len = last_step_logits->shape()[0];
                const float *inc_last = last_step_logits->data() + (size_t)(inc_seq_len - 1) * vocab;
                auto rel_l2 = [&](const float *a)
                { double sum_sq=0, ref_sq=0; for(int j=0;j<vocab;++j){ double diff=(double)a[j] - (double)ref_logits[j]; sum_sq+=diff*diff; ref_sq += (double)ref_logits[j]*(double)ref_logits[j]; } return ref_sq>0? std::sqrt(sum_sq)/std::sqrt(ref_sq):0.0; };
                auto max_abs = [&](const float *a)
                { double m=0; for(int j=0;j<vocab;++j){ double d=fabs((double)a[j]-(double)ref_logits[j]); if(d>m) m=d; } return m; };
                double rel_inc = rel_l2(inc_last);
                double rel_rep = rel_l2(replay_last);
                double max_inc = max_abs(inc_last);
                double max_rep = max_abs(replay_last);
                // Also direct inc vs rep rel_l2 (already computed later but include here for consolidated view)
                double sum_sq = 0, ref_sq = 0;
                for (int j = 0; j < vocab; ++j)
                {
                    double diff = (double)inc_last[j] - (double)replay_last[j];
                    sum_sq += diff * diff;
                    ref_sq += (double)replay_last[j] * (double)replay_last[j];
                }
                double rel_inc_rep = ref_sq > 0 ? std::sqrt(sum_sq) / std::sqrt(ref_sq) : 0.0;
                LOG_INFO("[LMHeadRefDiff] inc_rel_l2=" << rel_inc
                                                       << " rep_rel_l2=" << rel_rep
                                                       << " inc_max_abs=" << max_inc
                                                       << " rep_max_abs=" << max_rep
                                                       << " inc_vs_rep_rel_l2=" << rel_inc_rep
                                                       << " hidden_src_size=" << inc_pre_lm.size()
                                                       << " lm_head_rows=" << weights_.lm_head->shape()[0]
                                                       << " lm_head_cols=" << weights_.lm_head->shape()[1]);
                if (getenv("GTEST_PARITY_VERBOSE"))
                {
                    LOG_INFO(std::string("[LMHeadRefPreview] j ref inc rep d_inc d_rep"));
                    for (int j = 0; j < std::min(vocab, 8); ++j)
                    {
                        double refv = ref_logits[j];
                        double incv = inc_last[j];
                        double repv = replay_last[j];
                        LOG_INFO("[LMHeadRefPreview] j=" << j << " ref=" << refv << " inc=" << incv << " rep=" << repv
                                                         << " d_inc=" << (incv - refv) << " d_rep=" << (repv - refv));
                    }
                }
            }
            else
            {
                LOG_WARN("[LMHeadRefDiff] Missing logits tensors for reference comparison full_logits_replay=" << (full_logits_replay ? 1 : 0)
                                                                                                               << " last_step_logits=" << (last_step_logits ? 1 : 0));
            }
        }
        else
        {
            LOG_WARN("[LMHeadRefDiff] Missing hidden_final or lm_head weight for reference computation hidden_final=" << (hidden_final ? 1 : 0)
                                                                                                                      << " lm_head_present=" << (weights_.lm_head ? 1 : 0)
                                                                                                                      << " inc_pre_lm_size=" << inc_pre_lm.size());
        }
    }
    if (getenv("GTEST_PARITY_VERBOSE") && pipeline_dynamic_->getRank() == 0)
    {
        const char *lt = std::getenv("LLAMINAR_PIPELINE_LAYER_TOKEN_DIFF");
        LOG_INFO(std::string("[KVParityDebug] env.LLAMINAR_PIPELINE_LAYER_TOKEN_DIFF=") + (lt ? lt : "<unset>"));
    }
    // Per-layer last-token diff diagnostics (always on rank 0 for this test to aid debugging)
    if (pipeline_dynamic_->getRank() == 0)
    {
        const auto &rows_all = DistributedTransformerPipeline::getLastLayerTokenRows();
        if (getenv("GTEST_PARITY_VERBOSE"))
        {
            LOG_INFO(std::string("[LayerTokenDiff] captured_rows=") + std::to_string(rows_all.size()));
        }
        const void *dyn_ptr = pipeline_dynamic_.get();
        const void *rep_ptr = pipeline_replay_.get();
        // For incremental path, seq_len == 1 always (single token). We want the LAST captured incremental row for each layer.
        std::vector<const DistributedTransformerPipeline::LayerTokenDiffRow *> dyn_last_inc(cfg_.n_layers, nullptr);
        std::vector<const DistributedTransformerPipeline::LayerTokenDiffRow *> rep_final(cfg_.n_layers, nullptr);
        int final_seq = (int)full_seq.size();
        for (const auto &r : rows_all)
        {
            if (r.pipeline == dyn_ptr && r.incremental)
            {
                if (r.layer >= 0 && r.layer < cfg_.n_layers)
                    dyn_last_inc[r.layer] = &r; // overwrite -> last occurrence
            }
            else if (r.pipeline == rep_ptr && !r.incremental && r.seq_len == final_seq)
            {
                if (r.layer >= 0 && r.layer < cfg_.n_layers)
                    rep_final[r.layer] = &r;
            }
        }
        if (getenv("GTEST_PARITY_VERBOSE"))
        {
            for (int l = 0; l < cfg_.n_layers; ++l)
            {
                LOG_INFO(std::string("[LayerTokenDiffDiag] layer=") + std::to_string(l) +
                         " dyn_inc_present=" + (dyn_last_inc[l] ? "1" : "0") +
                         " rep_final_present=" + (rep_final[l] ? "1" : "0"));
            }
        }
        bool have_all = true;
        for (int l = 0; l < cfg_.n_layers; ++l)
        {
            if (!dyn_last_inc[l] || !rep_final[l])
            {
                have_all = false;
                break;
            }
        }
        LOG_INFO(std::string("[LayerTokenDiffDiag] have_all=") + (have_all ? "1" : "0"));
        // Always attempt per-layer diff for any layer where both rows exist, even if some other layer missing.
        for (int l = 0; l < cfg_.n_layers; ++l)
        {
            if (dyn_last_inc[l] && rep_final[l])
            {
                const auto &inc_row = dyn_last_inc[l]->values;
                const auto &rep_row = rep_final[l]->values;
                size_t d = std::min(inc_row.size(), rep_row.size());
                double sum_sq = 0, ref_sq = 0, max_abs = 0;
                for (size_t k = 0; k < d; ++k)
                {
                    double diff = (double)inc_row[k] - rep_row[k];
                    sum_sq += diff * diff;
                    ref_sq += (double)rep_row[k] * rep_row[k];
                    double a = fabs(diff);
                    if (a > max_abs)
                        max_abs = a;
                }
                double rel_l2 = ref_sq > 0 ? std::sqrt(sum_sq) / std::sqrt(ref_sq) : 0.0;
                LOG_INFO("[LayerTokenDiff] layer=" << l
                                                   << " inc_seq_len=" << dyn_last_inc[l]->seq_len
                                                   << " rep_seq_len=" << rep_final[l]->seq_len
                                                   << " inc_dim=" << inc_row.size()
                                                   << " rep_dim=" << rep_row.size()
                                                   << " used_dim=" << d
                                                   << " rel_l2=" << rel_l2
                                                   << " max_abs=" << max_abs);
            }
            else
            {
                LOG_WARN("[LayerTokenDiff] layer=" << l << " missing="
                                                   << (dyn_last_inc[l] ? "" : "inc")
                                                   << (dyn_last_inc[l] && !rep_final[l] ? " rep" : "")
                                                   << (!dyn_last_inc[l] && rep_final[l] ? " inc" : "")
                                                   << (!dyn_last_inc[l] && !rep_final[l] ? " inc+rep" : ""));
            }
        }
    }
    // Decode final token via incremental path has logits in last step_logits. For reproducibility reuse final step decode.
    // For simplicity compare only last token logits parity (shape [1, vocab]) between incremental and replay execution.
    // Extract last row from replay logits.
    ASSERT_TRUE(full_logits_replay != nullptr);
    int seq_len_full = (int)full_seq.size();
    ASSERT_EQ(full_logits_replay->shape()[0], seq_len_full);
    int vocab_size = full_logits_replay->shape()[1];
    // Get incremental final logits from the last incremental decode (already captured)
    ASSERT_TRUE(last_step_logits != nullptr);
    const float *replay_last = full_logits_replay->data() + (size_t)(seq_len_full - 1) * vocab_size;
    const float *inc_last = last_step_logits->data();
    double sum_sq = 0, ref_sq = 0, max_abs = 0;
    int n = vocab_size;
    for (int i = 0; i < n; ++i)
    {
        double diff = (double)inc_last[i] - replay_last[i];
        sum_sq += diff * diff;
        ref_sq += (double)replay_last[i] * replay_last[i];
        double ad = fabs(diff);
        if (ad > max_abs)
            max_abs = ad;
    }
    double rel_l2 = ref_sq > 0 ? sqrt(sum_sq) / sqrt(ref_sq) : 0.0;
    if (pipeline_dynamic_->getRank() == 0)
    {
        // Redundant final reference check to ensure visibility even if earlier block skipped
        int d_model = cfg_.d_model;
        const float *hidden_final2 = nullptr;
        size_t expect_seq2 = (size_t)seq_len_full * d_model;
        if (!inc_pre_lm.empty())
        {
            if (inc_pre_lm.size() == (size_t)d_model)
                hidden_final2 = inc_pre_lm.data();
            else if (inc_pre_lm.size() == expect_seq2)
                hidden_final2 = inc_pre_lm.data() + (seq_len_full - 1) * d_model;
        }
        if (hidden_final2 && weights_.lm_head)
        {
            int vocab = cfg_.vocab_size;
            const float *W = weights_.lm_head->data();
            std::vector<float> ref_logits(vocab, 0.f);
            for (int j = 0; j < vocab; ++j)
            {
                double acc = 0.0;
                for (int i = 0; i < d_model; ++i)
                {
                    acc += (double)hidden_final2[i] * (double)W[i * vocab + j];
                }
                ref_logits[j] = (float)acc;
            }
            auto rel_l2_ref = [&](const float *a)
            { double s=0,r=0; for(int j=0;j<vocab;++j){ double d=(double)a[j]-ref_logits[j]; s+=d*d; r += (double)ref_logits[j]*ref_logits[j]; } return r>0? std::sqrt(s)/std::sqrt(r):0.0; };
            double rel_inc_ref = rel_l2_ref(inc_last);
            double rel_rep_ref = rel_l2_ref(replay_last);
            LOG_INFO("[LMHeadRefDiff(Final)] rel_inc_ref=" << rel_inc_ref << " rel_rep_ref=" << rel_rep_ref);
        }
        else
        {
            LOG_WARN("[LMHeadRefDiff(Final)] Skipped hidden_final2=" << (hidden_final2 ? 1 : 0) << " lm_head=" << (weights_.lm_head ? 1 : 0));
        }
    }
    // Allow relaxed tolerance configurable via environment. Default tighter than
    // abstract parity because deterministic weights reduce accumulation error, but
    // still permit small FP32 ordering differences as cache grows.
    double rel_tol = 0.0;
    if (const char *env_tol = std::getenv("LLAMINAR_PARITY_TOL"))
    {
        rel_tol = std::strtod(env_tol, nullptr);
    }
    else
    {
        rel_tol = 1e-2; // align with abstract parity default; revisit after numerical audit
    }
    if (rel_l2 > rel_tol)
    {
        EXPECT_LT(rel_l2, rel_tol) << "Relative L2 drift too large (rel_l2=" << rel_l2 << ", tol=" << rel_tol << ")";
    }
    else if (rel_l2 > 5e-3 && std::getenv("LLAMINAR_PARITY_WARN"))
    {
        LOG_WARN("[KVGrowthParityWarn] rel_l2=" << rel_l2 << " tol=" << rel_tol);
    }
    EXPECT_LT(max_abs, 1e-2) << "Max abs drift too large (max_abs=" << max_abs << ")";
}
