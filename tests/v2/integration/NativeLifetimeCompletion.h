/**
 * @file NativeLifetimeCompletion.h
 * @brief Source-gate terminal evidence after the original lexical native GPU lifetime.
 *
 * Runner/model receipts precede this boundary. This opt-in test protocol emits
 * no native operation and performs no cleanup: it observes that the existing
 * lifetime destructor returned. Failed assertions may still retire cleanly;
 * an exception or fatal teardown cannot reach publication.
 */
#pragma once
#include <charconv>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unistd.h>
namespace llaminar2::test {
/** @brief Sequence the actual lifetime, preserving an ordinary failed test result. */
template<class NativeLifetime, class Run, class Publish>
int runWithNativeLifetimeCompletion(Run&& run, Publish&& publish)
{
    int result;
    { NativeLifetime original_lifetime; result = run(); }
    publish(result);
    return result;
}
/** @brief Immutable opt-in publication target, distinct from resource ownership. */
class NativeLifetimeCompletion final {
public:
    /** @return Canonical source invocation target, or no publication for ordinary tests.
     * @throws std::invalid_argument For incomplete opt-in metadata or stale output.
     */
    static std::optional<NativeLifetimeCompletion> fromEnvironment()
    {
        const char* path = std::getenv("LLAMINAR_SOURCE_ADMISSION_PROCESS_RECEIPT_ROOT");
        const char* token = std::getenv("LLAMINAR_SOURCE_ADMISSION_PROCESS_INVOCATION");
        if (!path && !token) return {};
        if (!path || !token || !validToken(token))
            throw std::invalid_argument("Source native lifetime publication lacks its original invocation");
        // Canonical source cells are launched by the existing OpenMPI owner.
        // The daemon-authored world rank names this process, never another rank.
        const char* rank_text = std::getenv("OMPI_COMM_WORLD_RANK");
        int rank = -1;
        if (!rank_text) throw std::invalid_argument("Source native lifetime publication lacks original MPI rank");
        const std::string_view text(rank_text);
        const auto parsed = std::from_chars(text.data(), text.data()+text.size(), rank);
        if (parsed.ec != std::errc{} || parsed.ptr != text.data()+text.size() || rank < 0)
            throw std::invalid_argument("Source native lifetime publication has invalid original MPI rank");
        const std::filesystem::path root(path);
        const auto output = root / ("rank-"+std::to_string(rank)+".csv");
        if (!root.is_absolute() || !std::filesystem::is_directory(root) || std::filesystem::exists(output))
            throw std::invalid_argument("Source native lifetime publication requires a fresh original directory");
        return NativeLifetimeCompletion(output, token, rank);
    }
    /** @brief Publish only after the original lifetime returned; no reset or native query occurs here. */
    void publish(int result) const
    {
        if (result != 0 && result != 1)
            throw std::invalid_argument("Source native lifetime publication has a non-GTest result");
        std::ofstream output(output_);
        output << "schema,invocation,rank,pid,test_result,native_lifetime_retired\n"
               << "1," << token_ << ',' << rank_ << ',' << ::getpid() << ',' << result << ",1\n";
        output.flush();
        if (!output.good()) throw std::runtime_error("Source native lifetime terminal publication failed");
    }
private:
    /** @brief Bind the already validated original invocation and rank. */
    NativeLifetimeCompletion(std::filesystem::path output, std::string token, int rank)
        : output_(std::move(output)), token_(std::move(token)), rank_(rank) {}
    /** @return A complete fresh invocation nonce, not a resource identity or policy hash. */
    static bool validToken(std::string_view token)
    {
        return token.size()==32 && token.find_first_not_of("0123456789abcdef")==token.npos;
    }
    std::filesystem::path output_;
    std::string token_;
    int rank_;
};
}
