#include "logger.h"
#include "utils/debug_env.h"
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <cassert>

// This test simulates role tag logging by emitting representative log lines
// In full integration it would load a minimal model; here we directly log to validate ring buffer capture.

int main()
{
    // Force log level high enough and enable shard_load_diag semantics indirectly by simulating condition
    Logger::getInstance().setLogLevel(LogLevel::TRACE);

    // Emit synthetic role tag lines (mirroring real loader output)
    LOG_INFO("[RoleTag] token_embd.weight -> Embedding (vocab_size x d_model)");
    LOG_INFO("[RoleTag] layer=0 ffn_gate -> W1  (d_model x d_ff)");
    LOG_INFO("[RoleTag] layer=0 ffn_up   -> W3  (d_model x d_ff)");
    LOG_INFO("[RoleTag] layer=0 ffn_down -> W2  (d_ff x d_model)");

    auto lines = Logger::getInstance().recent_lines();
    auto has = [&](const std::string &needle)
    { return std::any_of(lines.begin(), lines.end(), [&](const std::string &l)
                         { return l.find(needle) != std::string::npos; }); };

    bool ok = true;
    ok &= has("token_embd.weight -> Embedding");
    ok &= has("ffn_gate -> W1");
    ok &= has("ffn_up   -> W3");
    ok &= has("ffn_down -> W2");

    if (!ok)
    {
        std::cerr << "Role tag log assertions failed. Captured lines:\n";
        for (const auto &l : lines)
            std::cerr << l << '\n';
        return 1;
    }
    std::cout << "Role tagging log test passed (synthetic)." << std::endl;
    return 0;
}
