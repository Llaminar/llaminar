/**
 * @file Test__PhysicalPagePoolReplay.cpp
 * @brief Replay recorded unpublished page-authority inputs and ownership results.
 *
 * Traces explicitly start with an empty, zero-generation CPU pool. Expected
 * decisions/frame generations are captured from real authority executions, not
 * regenerated during ordinary replay. Pool incarnation is rebound once to the
 * new process instance; physical frame and request transitions are unchanged.
 * This does not replay native KV kernels, published views or GPU accesses.
 */
#include "planning/PhysicalPagePool.h"
#include "PhysicalPagePoolTraceData.h"
#include <gtest/gtest.h>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace llaminar2;
namespace
{
/** @brief One replay instance, using actual PMA materialization and private grant operations. */
class PrivatePageReplay
{
public:
    /** @brief Bind the trace's exact empty initial pool through the production factory. */
    explicit PrivatePageReplay(PhysicalPagePoolGeometry geometry)
    {
        const auto device = DeviceId::cpu();
        const auto bytes = PhysicalPagePool::physicalBytes(geometry, device);
        PhysicalMemoryPlanBuilder plan;
        plan.add({.world_rank=0, .device=device, .total_bytes=bytes, .admission_available_bytes=bytes},
            PhysicalMemoryOwner::KVCache, bytes);
        authority_ = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
        pool_ = authority_->createCPUPagePool(device, geometry, CPUStoragePlacement::local());
    }
    /** @brief Execute a recorded command, returning only actual observed results. */
    std::string execute(const std::string &command)
    {
        std::istringstream input(command);
        std::string kind, name;
        input >> kind;
        std::string outcome;
        if (kind == "admit")
        {
            size_t pages;
            if (!(input >> name >> pages) || requests_.contains(name)) fail();
            auto grant = pool_->tryAdmit(pages);
            outcome = grant ? "accepted" : "refused";
            if (grant) requests_.emplace(name, std::move(*grant));
        }
        else if (kind == "assign")
        {
            if (!(input >> name) || !requests_.contains(name)) fail();
            const auto frame = requests_.at(name).assignFrame();
            outcome = std::to_string(frame.index) + ":" + std::to_string(frame.generation);
        }
        else if (kind == "pin")
        {
            std::string request;
            size_t ordinal;
            if (!(input >> name >> request >> ordinal) || pins_.contains(name) || !requests_.contains(request)) fail();
            auto &grant = requests_.at(request);
            const auto frames = grant.assignedFrames();
            if (ordinal >= frames.size()) fail();
            pins_.emplace(name, grant.pinFrame(frames[ordinal]));
            outcome = "pinned";
        }
        else if (kind == "drop-request")
        {
            if (!(input >> name) || requests_.erase(name) != 1) fail();
            outcome = "retired";
        }
        else if (kind == "drop-pin")
        {
            if (!(input >> name) || pins_.erase(name) != 1) fail();
            outcome = "released";
        }
        else if (kind == "plan")
        {
            std::string request;
            size_t ordinal, bytes;
            if (!(input >> name >> request >> ordinal >> bytes) || plans_.contains(name) || !requests_.contains(request)) fail();
            auto &grant = requests_.at(request);
            if (ordinal >= grant.assignedFrames().size()) fail();
            auto plan = grant.prepareAppend(grant.assignedFrames()[ordinal], bytes);
            outcome = plan ? "ready" : "busy";
            if (plan) plans_.emplace(name, std::move(*plan));
        }
        else if (kind == "copy")
        {
            std::string hex;
            if (!(input >> name >> hex) || !plans_.contains(name) || hex.size() % 2) fail();
            std::vector<std::byte> bytes;
            for (size_t i = 0; i < hex.size(); i += 2)
            {
                const auto digit = [](char c) -> int {
                    if (c >= '0' && c <= '9') return c - '0';
                    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
                    throw std::invalid_argument("Noncanonical recorded page byte");
                };
                bytes.push_back(static_cast<std::byte>(16 * digit(hex[i]) + digit(hex[i + 1])));
            }
            plans_.at(name).copyFrom(bytes);
            outcome = "copied";
        }
        else if (kind == "commit")
        {
            std::string page;
            if (!(input >> name >> page) || !plans_.contains(name) || pages_.contains(page)) fail();
            pages_.emplace(page, plans_.at(name).commit());
            plans_.erase(name);
            outcome = "published";
        }
        else if (kind == "read")
        {
            if (!(input >> name) || !pages_.contains(name)) fail();
            const auto &page = pages_.at(name);
            std::ostringstream result;
            result << page.version() << ':';
            for (const auto byte : page.read()) result << std::hex << std::setw(2) << std::setfill('0') << std::to_integer<unsigned>(byte);
            outcome = result.str();
        }
        else if (kind == "register")
        {
            std::string page;
            if (!(input >> name >> page) || reads_.contains(name) || !pages_.contains(page)) fail();
            reads_.emplace(name, pages_.at(page).registerRead());
            outcome = "registered";
        }
        else if (kind == "close")
        {
            size_t ordinal;
            if (!(input >> name >> ordinal) || !requests_.contains(name)) fail();
            auto &grant = requests_.at(name);
            if (ordinal >= grant.assignedFrames().size()) fail();
            grant.closeReaders(grant.assignedFrames()[ordinal]);
            outcome = "closed";
        }
        else if (kind == "consume")
        {
            if (!(input >> name) || !reads_.contains(name)) fail();
            auto &access = reads_.at(name);
            std::ostringstream result;
            const auto bytes = access.beginCPU();
            try
            {
                for (const auto byte : bytes)
                    result << std::hex << std::setw(2) << std::setfill('0') << std::to_integer<unsigned>(byte);
                outcome = result.str();
            }
            catch (...)
            {
                access.completeCPU();
                reads_.erase(name);
                throw;
            }
            access.completeCPU();
            reads_.erase(name);
        }
        else if (kind == "drop-read")
        {
            if (!(input >> name) || reads_.erase(name) != 1) fail();
            outcome = "aborted";
        }
        else if (kind == "drop-plan")
        {
            if (!(input >> name) || plans_.erase(name) != 1) fail();
            outcome = "aborted";
        }
        else if (kind == "drop-page")
        {
            if (!(input >> name) || pages_.erase(name) != 1) fail();
            outcome = "released";
        }
        else if (kind == "end")
        {
            if (!requests_.empty() || !pins_.empty() || !plans_.empty() || !pages_.empty() || !reads_.empty()) fail();
            outcome = "idle";
        }
        else fail();
        std::string extra;
        if (input >> extra) fail();
        return outcome;
    }
private:
    /** @brief Reject incompatible or incomplete command grammar rather than invent state. */
    [[noreturn]] static void fail() { throw std::invalid_argument("Invalid private-page replay command/state"); }
    std::shared_ptr<PhysicalMemoryAuthority> authority_;
    std::shared_ptr<PhysicalPagePool> pool_;
    std::map<std::string, PhysicalPagePool::Grant> requests_;
    std::map<std::string, PhysicalPagePool::FramePin> pins_;
    std::map<std::string, PhysicalPagePool::AppendPlan> plans_;
    std::map<std::string, PhysicalPagePool::CommittedPage> pages_;
    std::map<std::string, PhysicalPagePool::ReadAccess> reads_;
};

/** @brief Parse an explicitly versioned complete trace and compare the production authority's observations. */
void replay(std::istream &input)
{
    std::string header;
    if (!std::getline(input, header)) throw std::invalid_argument("Missing private-page trace header");
    std::istringstream geometry_line(header);
    std::string version, initial, extra;
    PhysicalPagePoolGeometry geometry;
    if (!(geometry_line >> version >> geometry.frames >> geometry.frame_bytes >> geometry.request_records >>
        geometry.max_pages_per_request >> initial) || version != "pma-private-pages-v1" ||
        initial != "empty-zero-generations" || (geometry_line >> extra))
        throw std::invalid_argument("Incompatible private-page trace initial state");
    PrivatePageReplay runner(geometry);
    std::string line;
    bool ended = false;
    while (std::getline(input, line))
    {
        if (ended || line.empty()) throw std::invalid_argument("Trailing/empty private-page trace row");
        const auto separator = line.find(" | ");
        if (separator == std::string::npos) throw std::invalid_argument("Missing private-page trace observation");
        const auto command = line.substr(0, separator);
        const auto expected = line.substr(separator + 3);
        const auto observed = runner.execute(command);
        if (observed != expected)
            throw std::runtime_error("Private-page replay mismatch: " + command + ": expected " + expected + ", got " + observed);
        ended = command == "end";
    }
    if (!ended) throw std::invalid_argument("Incomplete private-page trace");
}
}

/** @test Recorded private growth/refusal/pin/retirement/frame-reuse outcomes replay through PMA. */
TEST(PhysicalMemoryAuthority, PagePoolRecordedUnpublishedOwnershipReplay)
{
    // Explicit diagnostic capture only. Ordinary replay never rewrites expected results.
    if (const char *record = std::getenv("LLAMINAR_RECORD_PRIVATE_PAGE_TRACE"))
    {
        std::ofstream output(record);
        ASSERT_TRUE(output.good());
        output << "pma-private-pages-v1 3 128 4 3 empty-zero-generations\n";
        PrivatePageReplay runner({3, 128, 4, 3});
        for (const std::string command : {"admit A 2", "assign A", "pin P A 0", "admit B 1", "assign B",
            "admit C 1", "drop-request A", "admit C 1", "assign C", "admit D 1", "drop-pin P",
            "admit D 1", "assign D", "drop-request B", "drop-request C", "drop-request D",
            "admit E 1", "assign E", "plan W E 0 4", "copy W 01020304", "commit W V", "read V",
            "plan busy E 0 1", "drop-page V", "plan abort E 0 4", "copy abort ffffffff", "drop-plan abort",
            "plan correct E 0 2", "copy correct 0506", "commit correct V2", "read V2", "drop-request E",
            "read V2", "drop-page V2", "admit F 1", "assign F", "plan W2 F 0 4", "copy W2 0708090a",
            "commit W2 V3", "register R V3", "close F 0", "drop-page V3", "drop-request F", "admit G 3",
            "consume R", "admit G 3", "assign G", "drop-request G", "end"})
            output << command << " | " << runner.execute(command) << '\n';
        output.close();
        ASSERT_TRUE(output.good());
        std::ifstream captured(record);
        EXPECT_NO_THROW(replay(captured));
        return;
    }
    std::istringstream input(llaminar2::test::privatePageOwnershipTrace);
    EXPECT_NO_THROW(replay(input));
    std::istringstream appended(llaminar2::test::privatePageAppendTrace);
    EXPECT_NO_THROW(replay(appended));
    std::istringstream readers(llaminar2::test::privatePageReaderTrace);
    EXPECT_NO_THROW(replay(readers));
}

/** @test Incomplete and incompatible recordings are rejected before inventing observations. */
TEST(PhysicalMemoryAuthority, PagePoolReplayRejectsIncompleteInitialState)
{
    for (const std::string trace : {"", "pma-private-pages-v2 3 128 4 3 empty-zero-generations\nend | idle\n",
        "pma-private-pages-v1 3 128 4 3 unspecified\nend | idle\n",
        "pma-private-pages-v1 3 128 4 3 empty-zero-generations\nadmit A 1 | accepted\n",
        "pma-private-pages-v1 3 128 4 3 empty-zero-generations\nend | idle\nextra\n"})
    {
        std::istringstream input(trace);
        EXPECT_THROW(replay(input), std::exception);
    }
}
