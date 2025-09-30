#include <gtest/gtest.h>
#include <filesystem>
#include <vector>
#include <string>
#include <cstdlib>
#include <limits>
#include <mpi.h>

#include "logger.h"
#include "model_loader.h"
#include "mpi_transformer_pipeline.h"

// Forward declaration of helper (inside llaminar namespace)
namespace llaminar
{
    MPITransformerPipeline::ModelWeights loadModelWeights(const std::string &model_path, const MPITransformerPipeline::LayerConfig &config);
}

namespace
{
    std::string pick_small_model()
    {
        const char *env = std::getenv("LLAMINAR_ROLE_TAG_MODEL");
        if (env)
            return env;
        std::filesystem::path dir{"models"};
        if (!std::filesystem::exists(dir))
            return {};
        uintmax_t best = (std::numeric_limits<uintmax_t>::max)();
        std::string choice;
        for (auto &p : std::filesystem::directory_iterator(dir))
        {
            if (!p.is_regular_file())
                continue;
            if (p.path().extension() == ".gguf")
            {
                auto sz = std::filesystem::file_size(p.path());
                if (sz < best)
                {
                    best = sz;
                    choice = p.path().string();
                }
            }
        }
        return choice;
    }
}

// Helpers for extended validation
static bool log_contains(const std::vector<std::string> &lines, const std::string &needle)
{
    for (const auto &l : lines)
        if (l.find(needle) != std::string::npos)
            return true;
    return false;
}

static void assert_embedding_stats_reasonable(const std::vector<std::string> &lines)
{
    for (const auto &l : lines)
    {
        if (l.find("First 10 token embedding values:") != std::string::npos)
        {
            // We only want to flag actual NaN/Inf tokens in the embedding value dump, not the
            // log level prefix (e.g. "INFO" contains substring "inf"). Perform a simple
            // word-boundary style scan so that sequences like "info" do not trip the check.
            auto has_token = [](const std::string &s, const std::string &tok)
            {
                const std::string lower_tok = tok; // already lowercase constant
                for (size_t i = 0; i < s.size(); ++i)
                {
                    if (std::tolower((unsigned char)s[i]) != lower_tok[0])
                        continue;
                    bool match = true;
                    size_t j = 0;
                    for (; j < lower_tok.size() && i + j < s.size(); ++j)
                    {
                        if (std::tolower((unsigned char)s[i + j]) != lower_tok[j])
                        {
                            match = false;
                            break;
                        }
                    }
                    if (!match)
                        continue;
                    // Check "word" boundary: preceding and following chars not alphabetic
                    bool left_ok = (i == 0) || !std::isalpha((unsigned char)s[i - 1]);
                    bool right_ok = (i + j >= s.size()) || !std::isalpha((unsigned char)s[i + j]);
                    if (left_ok && right_ok)
                        return true;
                }
                return false;
            };
            ASSERT_FALSE(has_token(l, "nan")) << "Embedding dump contains NaN";
            ASSERT_FALSE(has_token(l, "inf")) << "Embedding dump contains Inf";
        }
    }
}

static void assert_topology_logged(const std::vector<std::string> &lines)
{
    ASSERT_TRUE(log_contains(lines, "-- Sharded Tensor Registry (post model load) --"))
        << "Missing sharded tensor registry header";
    ASSERT_TRUE(log_contains(lines, "Aggregate shard footprint:"))
        << "Missing aggregate shard footprint line";
    ASSERT_TRUE(log_contains(lines, "Estimated global (logical) elements"))
        << "Missing estimated global elements line";
}

// ---- Refactored: group models by token embedding quant type and test each group separately ----

namespace
{
    using GroupMap = std::unordered_map<GGUFTensorType, std::vector<std::string>>;

    static const GGUFTensorInfo *find_token_embd(const ModelLoader &L)
    {
        return L.getModel().findTensor("token_embd.weight");
    }

    static GroupMap &model_groups()
    {
        static GroupMap groups;
        static std::once_flag once;
        std::call_once(once, []()
                       {
        namespace fs = std::filesystem;
        if(const char* only = std::getenv("LLAMINAR_ROLE_TAG_MODEL")) {
            ModelLoader loader; if(loader.loadModel(only)) {
                if(auto *info = find_token_embd(loader)) groups[info->type].push_back(only);
            }
            return;
        }
        fs::path dir{"models"};
        if(!fs::exists(dir)) return;
        for(auto &p : fs::directory_iterator(dir)) {
            if(!p.is_regular_file() || p.path().extension() != ".gguf") continue;
            auto path = p.path().string();
            ModelLoader loader; if(!loader.loadModel(path)) continue;
            if(auto *info = find_token_embd(loader)) {
                groups[info->type].push_back(path);
            }
        } });
        return groups;
    }

    static void run_role_tag_assertions_for_group(GGUFTensorType type)
    {
        auto &groups = model_groups();
        auto it = groups.find(type);
        if (it == groups.end() || it->second.empty())
        {
            GTEST_SKIP() << "No models found for quant type enum=" << static_cast<int>(type);
        }
        int provided = 0, init_flag = 0;
        MPI_Initialized(&init_flag);
        if (!init_flag)
            MPI_Init_thread(nullptr, nullptr, MPI_THREAD_SINGLE, &provided);
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        Logger::getInstance().setLogLevel(LogLevel::TRACE);
        setenv("LLAMINAR_SHARD_LOAD_DIAG", "1", 1);
        for (const auto &model_path : it->second)
        {
            if (rank == 0)
                fprintf(stderr, "[ROLE-TAG-GROUP enum=%d] %s\n", (int)type, model_path.c_str());
            ModelLoader loader;
            ASSERT_TRUE(loader.loadModel(model_path)) << "Failed to load model: " << model_path;
            if (const char *req = std::getenv("REQUIRE_KNOWN_TYPES"))
            {
                bool saw_unsupported = false;
                for (auto &line : Logger::getInstance().recent_lines())
                    if (line.find("Unsupported non-quantized type") != std::string::npos)
                    {
                        saw_unsupported = true;
                        break;
                    }
                EXPECT_FALSE(saw_unsupported) << "Unsupported tensor encountered under REQUIRE_KNOWN_TYPES in model " << model_path;
            }
            auto config = loader.createLayerConfig();
            auto pipeline = std::make_unique<llaminar::MPITransformerPipeline>(config);
            auto weights = llaminar::loadModelWeights(model_path, config);
            (void)weights;
            (void)pipeline;
            auto lines = Logger::getInstance().recent_lines();
            if (rank == 0)
            {
                EXPECT_TRUE(log_contains(lines, "token_embd.weight -> Embedding")) << "Missing embedding role tag log (" << model_path << ")";
                EXPECT_TRUE(log_contains(lines, "ffn_gate -> W1")) << "Missing FFN gate W1 role tag (" << model_path << ")";
                EXPECT_TRUE(log_contains(lines, "ffn_up   -> W3")) << "Missing FFN up W3 role tag (" << model_path << ")";
                EXPECT_TRUE(log_contains(lines, "ffn_down -> W2")) << "Missing FFN down W2 role tag (" << model_path << ")";
                EXPECT_TRUE(log_contains(lines, "W_Q") || log_contains(lines, "role=W_Q")) << "Missing W_Q shard role (" << model_path << ")";
                EXPECT_TRUE(log_contains(lines, "W_K") || log_contains(lines, "role=W_K")) << "Missing W_K shard role (" << model_path << ")";
                EXPECT_TRUE(log_contains(lines, "W_V") || log_contains(lines, "role=W_V")) << "Missing W_V shard role (" << model_path << ")";
                EXPECT_TRUE(log_contains(lines, "W_O") || log_contains(lines, "role=W_O")) << "Missing W_O shard role (" << model_path << ")";
                assert_topology_logged(lines);
                assert_embedding_stats_reasonable(lines);
            }
        }
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (!finalized)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            if (!init_flag)
                MPI_Finalize();
        }
    }
} // namespace

// Individual group tests (skip automatically if no models of that type)
TEST(ModelLoadRoleTags, Q4_0_Group) { run_role_tag_assertions_for_group(GGUFTensorType::Q4_0); }
TEST(ModelLoadRoleTags, Q5_0_Group) { run_role_tag_assertions_for_group(GGUFTensorType::Q5_0); }
TEST(ModelLoadRoleTags, Q8_0_Group) { run_role_tag_assertions_for_group(GGUFTensorType::Q8_0); }
TEST(ModelLoadRoleTags, Q2_K_Group) { run_role_tag_assertions_for_group(GGUFTensorType::Q2_K); }
TEST(ModelLoadRoleTags, Q3_K_Group) { run_role_tag_assertions_for_group(GGUFTensorType::Q3_K); }
TEST(ModelLoadRoleTags, Q4_K_Group) { run_role_tag_assertions_for_group(GGUFTensorType::Q4_K); }
TEST(ModelLoadRoleTags, Q4_K_M_Group) { run_role_tag_assertions_for_group(GGUFTensorType::Q4_K_M); }
TEST(ModelLoadRoleTags, Q5_K_Group) { run_role_tag_assertions_for_group(GGUFTensorType::Q5_K); }
TEST(ModelLoadRoleTags, Q6_K_Group) { run_role_tag_assertions_for_group(GGUFTensorType::Q6_K); }
