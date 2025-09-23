#include "../src/cosma_prefill_manager.h"
#include "../src/logger.h"
#include <gtest/gtest.h>
#include <mpi.h>
#include <unordered_set>
#include <cstdlib>
using namespace llaminar;

TEST(CosmaPrefillManagerEnvAuditTest, RecognizedVariablesDocumented)
{
    int init=0; MPI_Initialized(&init); if(!init){ int argc=0; char** argv=nullptr; MPI_Init(&argc,&argv);}    
    auto &mgr = CosmaPrefillManager::instance();
    const auto &vars = CosmaPrefillManager::recognized_env_vars();
    // Simple assertion: list not empty and contains key gating variable names
    ASSERT_FALSE(vars.empty());
    std::unordered_set<std::string> set(vars.begin(), vars.end());
    EXPECT_TRUE(set.count("LLAMINAR_COSMA_PREFILL_THRESHOLD"));
    EXPECT_TRUE(set.count("ADAPTIVE_DISABLE_COSMA"));
    EXPECT_TRUE(set.count("LLAMINAR_COSMA_MAX_RESIDENT_MB"));
}
