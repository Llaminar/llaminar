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
#include "planning/PageTableAdmission.h"
#include "kernels/cpu/attention/CPUFP32PagedKVTable.h"
#include "PhysicalPagePoolTraceData.h"
#include <gtest/gtest.h>
#include <cstdlib>
#include <bit>
#include <limits>
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
    explicit PrivatePageReplay(PhysicalPagePoolGeometry geometry,
        std::optional<CPUFP32PagedKVTableGeometry> table_geometry = std::nullopt)
    {
        const auto device = DeviceId::cpu();
        const auto page_bytes = PhysicalPagePool::physicalBytes(geometry, device);
        const auto table_bytes = table_geometry ? CPUFP32PagedKVTable::physicalBytes(*table_geometry, device) : 0u;
        if (table_bytes > std::numeric_limits<size_t>::max() - page_bytes)
            throw std::overflow_error("Recorded page/table BOM overflows");
        const auto bytes = page_bytes + table_bytes;
        PhysicalMemoryPlanBuilder plan;
        plan.add({.world_rank=0, .device=device, .total_bytes=bytes, .admission_available_bytes=bytes},
            PhysicalMemoryOwner::KVCache, bytes);
        authority_ = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
        pool_ = authority_->createCPUPagePool(device, geometry, CPUStoragePlacement::local());
        if (table_geometry) table_ = CPUFP32PagedKVTable::create(authority_, device, *table_geometry);
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
        else if (kind == "discard")
        {
            size_t ordinal;
            if (!(input >> name >> ordinal) || !requests_.contains(name)) fail();
            auto &grant = requests_.at(name);
            if (ordinal >= grant.assignedFrames().size()) fail();
            grant.discardFrame(grant.assignedFrames()[ordinal]);
            outcome = std::to_string(grant.remainingCommitment());
        }
        else if (kind == "remaining")
        {
            if (!(input >> name) || !requests_.contains(name)) fail();
            outcome = std::to_string(requests_.at(name).remainingCommitment());
        }
        else if (kind == "table-begin")
        {
            if (!(input >> name) || !table_ || updates_.contains(name)) fail();
            auto update = table_->tryBeginUpdate();
            outcome = update ? "ready" : "busy";
            if (update) updates_.emplace(name, std::move(*update));
        }
        else if (kind == "table-append")
        {
            std::string page; int heads, dimension;
            if (!(input >> name >> page >> heads >> dimension) || !updates_.contains(name) || !pages_.contains(page)) fail();
            updates_.at(name).append(CPUFP32KVPage(std::move(pages_.at(page)), heads, dimension));
            pages_.erase(page);
            outcome = "attached";
        }
        else if (kind == "table-commit")
        {
            std::string snapshot;
            if (!(input >> name >> snapshot) || !updates_.contains(name) || snapshots_.contains(snapshot)) fail();
            snapshots_.emplace(snapshot, updates_.at(name).commit());
            updates_.erase(name);
            outcome = std::to_string(snapshots_.at(snapshot).version());
        }
        else if (kind == "table-borrow")
        {
            std::string source;
            if (!(input >> name >> source) || snapshots_.contains(name) || !snapshots_.contains(source)) fail();
            auto borrow = snapshots_.at(source).tryBorrow();
            outcome = borrow ? "accepted" : "refused";
            if (borrow) snapshots_.emplace(name, std::move(*borrow));
        }
        else if (kind == "table-close")
        {
            if (!(input >> name) || !table_ || !snapshots_.contains(name)) fail();
            table_->closeBorrowers(snapshots_.at(name));
            outcome = "closed";
        }
        else if (kind == "table-read")
        {
            if (!(input >> name) || !snapshots_.contains(name)) fail();
            const auto &snapshot = snapshots_.at(name);
            const auto view = snapshot.view();
            outcome = std::to_string(snapshot.version()) + ":" + std::to_string(view.rows()) + ":" +
                std::to_string(view.key(0, 0)[0]) + ":" + std::to_string(view.value(0, 0)[0]);
        }
        else if (kind == "table-drop")
        {
            if (!(input >> name) || snapshots_.erase(name) != 1) fail();
            outcome = "retired";
        }
        else if (kind == "table-abort")
        {
            if (!(input >> name) || updates_.erase(name) != 1) fail();
            outcome = "aborted";
        }
        else if (kind == "share")
        {
            std::string page;
            if (!(input >> name >> page) || shareds_.contains(name) || !pages_.contains(page)) fail();
            // Sharing converts the private publication into one immutable shared
            // reference; the original publication remains a pin until dropped.
            shareds_.emplace(name, pages_.at(page).share());
            outcome = std::to_string(shareds_.at(name).version()) + ":" + std::to_string(shareds_.at(name).bytes());
        }
        else if (kind == "drop-shared")
        {
            if (!(input >> name) || shareds_.erase(name) != 1) fail();
            outcome = "released";
        }
        else if (kind == "adopt")
        {
            std::string source, request;
            if (!(input >> name >> request >> source) || adopted_.contains(name) || !shareds_.contains(source) || !requests_.contains(request)) fail();
            auto reference = requests_.at(request).adoptShared(shareds_.at(source));
            outcome = reference ? "accepted" : "refused";
            if (reference) adopted_.emplace(name, std::move(*reference));
        }
        else if (kind == "drop-adopted")
        {
            if (!(input >> name) || adopted_.erase(name) != 1) fail();
            outcome = "released";
        }
        else if (kind == "cow")
        {
            std::string request, source;
            size_t fresh;
            if (!(input >> name >> request >> source >> fresh) || cowplans_.contains(name) || !shareds_.contains(source) || !requests_.contains(request)) fail();
            auto plan = requests_.at(request).prepareCOW(shareds_.at(source), fresh);
            outcome = plan ? "ready" : "busy";
            if (plan) cowplans_.emplace(name, std::move(*plan));
        }
        else if (kind == "cow-source")
        {
            if (!(input >> name) || !cowplans_.contains(name)) fail();
            cowplans_.at(name).copySource();
            outcome = "copied-source";
        }
        else if (kind == "cow-next")
        {
            std::string hex;
            if (!(input >> name >> hex) || !cowplans_.contains(name) || hex.size() % 2) fail();
            std::vector<std::byte> bytes;
            for (size_t i = 0; i < hex.size(); i += 2)
            {
                const auto digit = [](char c) -> int {
                    if (c >= '0' && c <= '9') return c - '0';
                    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
                    throw std::invalid_argument("Noncanonical recorded cow byte");
                };
                bytes.push_back(static_cast<std::byte>(16 * digit(hex[i]) + digit(hex[i + 1])));
            }
            cowplans_.at(name).copyNext(bytes);
            outcome = "copied-fresh";
        }
        else if (kind == "cow-commit")
        {
            std::string page;
            if (!(input >> name >> page) || !cowplans_.contains(name) || pages_.contains(page)) fail();
            pages_.emplace(page, cowplans_.at(name).commit());
            cowplans_.erase(name);
            outcome = "published";
        }
        else if (kind == "drop-cow")
        {
            if (!(input >> name) || cowplans_.erase(name) != 1) fail();
            outcome = "aborted";
        }
        else if (kind == "combined")
        {
            size_t pages;
            if (!(input >> name >> pages) || combinations_.contains(name)) fail();
            auto admitted = admitPoolAndTable(*pool_, *table_, pages);
            outcome = admitted ? "accepted" : "refused";
            if (admitted)
                combinations_.emplace(name, PageTableAdmission{
                    std::move(admitted->grant), std::move(admitted->table_update)});
        }
        else if (kind == "combined-abort")
        {
            if (!(input >> name) || combinations_.erase(name) != 1) fail();
            outcome = "aborted";
        }
        else if (kind == "admit-witness")
        {
            std::string request;
            size_t peak, grant_pages;
            if (!(input >> name >> request >> peak >> grant_pages) || requests_.contains(name)) fail();
            auto granted = pool_->tryAdmitWithWitness(peak, grant_pages);
            outcome = granted ? "accepted" : "refused";
            if (granted) requests_.emplace(name, std::move(*granted));
        }
        else if (kind == "extend-witness")
        {
            std::string request;
            size_t pages;
            if (!(input >> name >> request >> pages) || !requests_.contains(request)) fail();
            auto &grant = requests_.at(request);
            outcome = pool_->extendWithWitness(grant, pages) ? "accepted" : "refused";
        }
        else if (kind == "end")
        {
            if (!requests_.empty() || !pins_.empty() || !plans_.empty() || !pages_.empty() || !reads_.empty() ||
                !updates_.empty() || !snapshots_.empty() || !shareds_.empty() || !adopted_.empty() || !cowplans_.empty() ||
                !combinations_.empty()) fail();
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
    std::shared_ptr<CPUFP32PagedKVTable> table_;
    std::map<std::string, PhysicalPagePool::Grant> requests_;
    std::map<std::string, PhysicalPagePool::FramePin> pins_;
    std::map<std::string, PhysicalPagePool::AppendPlan> plans_;
    std::map<std::string, PhysicalPagePool::CommittedPage> pages_;
    std::map<std::string, PhysicalPagePool::ReadAccess> reads_;
    std::map<std::string, CPUFP32PagedKVTable::Update> updates_;
    std::map<std::string, CPUFP32PagedKVTable::Snapshot> snapshots_;
    std::map<std::string, PhysicalPagePool::SharedPage> shareds_;
    std::map<std::string, PhysicalPagePool::SharedPage> adopted_;
    std::map<std::string, PhysicalPagePool::COWPlan> cowplans_;
    std::map<std::string, PageTableAdmission> combinations_;
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
        geometry.max_pages_per_request >> initial) ||
        (version != "pma-private-pages-v1" && version != "pma-private-pages-v2") || initial != "empty-zero-generations")
        throw std::invalid_argument("Incompatible private-page trace initial state");
    std::optional<CPUFP32PagedKVTableGeometry> table;
    if (version == "pma-private-pages-v2")
    {
        CPUFP32PagedKVTableGeometry parsed;
        std::string tag, state, codec;
        if (!(geometry_line >> tag >> parsed.page_capacity >> parsed.bank_count >> parsed.readers_per_bank >>
            parsed.page_tokens >> state >> codec) || tag != "table" || state != "empty-zero-versions" ||
            codec != "fp32-le" || std::endian::native != std::endian::little)
            throw std::invalid_argument("Incompatible native table initial state or byte order");
        parsed.validate();
        table = parsed;
    }
    if (geometry_line >> extra) throw std::invalid_argument("Trailing page/table geometry");
    PrivatePageReplay runner(geometry, table);
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

/** @test Recorded sharing, adoption and partial-tail COW transitions replay through real owners. */
TEST(PhysicalMemoryAuthority, PagePoolRecordedSharingAndCOWReplayExactly)
{
    if (const char *record = std::getenv("LLAMINAR_RECORD_PRIVATE_SHARE_TRACE"))
    {
        std::ofstream output(record);
        ASSERT_TRUE(output.good());
        output << "pma-private-pages-v1 3 128 4 3 empty-zero-generations\n";
        PrivatePageReplay runner({3, 128, 4, 3});
        for (const std::string command : {"admit Exporter 1", "assign Exporter",
            "plan W Exporter 0 9", "copy W 010203040506070809", "commit W P0",
            "share SH0 P0", "read P0", "adopt P0A Exporter SH0", "drop-adopted P0A",
            "admit Writer1 1", "cow C1 Writer1 SH0 7", "cow-source C1", "cow-next C1 0a0b0c0d0e0f10",
            "cow-commit C1 PW1", "read PW1", "read P0",
            "admit Writer2 1", "cow C2 Writer2 SH0 7", "cow-source C2", "cow-next C2 11121314151617",
            "cow-commit C2 PW2", "read PW2", "read P0",
            "adopt P0B Exporter SH0", "drop-adopted P0B", "drop-shared SH0", "drop-page P0",
            "drop-page PW1", "drop-page PW2", "drop-request Exporter", "drop-request Writer1",
            "drop-request Writer2", "end"})
            output << command << " | " << runner.execute(command) << '\n';
        output.close();
        ASSERT_TRUE(output.good());
        std::ifstream captured(record);
        EXPECT_NO_THROW(replay(captured));
        return;
    }
    std::istringstream input(llaminar2::test::privatePageShareTrace);
    EXPECT_NO_THROW(replay(input));
}

/** @test Shared/COW replay rejects invented observations and missing terminal state. */
TEST(PhysicalMemoryAuthority, PagePoolSharingReplayRejectsAlteredObservations)
{
    for (const auto &[original, changed] : std::vector<std::pair<std::string, std::string>>{
        {"cow C2 Writer2 SH0 7 | ready", "cow C2 Writer2 SH0 7 | busy"},
        {"share SH0 P0 | 1:9", "share SH0 P0 | 2:9"},
        {"adopt P0A Exporter SH0 | accepted", "adopt P0A Exporter SH0 | refused"},
        {"end | idle\n", ""}})
    {
        std::string corrupt(llaminar2::test::privatePageShareTrace);
        const auto position = corrupt.find(original);
        ASSERT_NE(position, std::string::npos);
        corrupt.replace(position, original.size(), changed);
        std::istringstream input(corrupt);
        EXPECT_THROW(replay(input), std::exception);
    }
}

/** @test Recorded finite-grant witness admission and extension replay through real owners. */
TEST(PhysicalMemoryAuthority, PagePoolRecordedWitnessAdmissionReplayExactly)
{
    if (const char *record = std::getenv("LLAMINAR_RECORD_PRIVATE_WITNESS_TRACE"))
    {
        std::ofstream output(record);
        ASSERT_TRUE(output.good());
        output << "pma-private-pages-v1 32 64 4 32 empty-zero-generations\n";
        PrivatePageReplay runner({32, 64, 4, 32});
        const std::string assign_a = [&]() {
            std::string result;
            for (int i = 0; i < 31; ++i) result += "assign A\n";
            return result;
        }();
        std::ostringstream commands;
        // A holds 31 of 32 frames (peak 32); B cannot enter with any finite
        // grant because A would need one more frame before it can complete.
        commands << "admit-witness A A 32 31\n" << assign_a
                 << "admit-witness B B 32 1\n"
                 << "drop-request A\n" << "drop-request B\n" << "end\n";
        std::istringstream sequence(commands.str());
        bool admitted_b_ = false;
        std::string line;
        while (std::getline(sequence, line))
        {
            if (line.empty()) continue;
            std::istringstream row(line);
            std::string kind, name;
            row >> kind >> name;
            if (kind == "admit-witness")
            {
                size_t peak, grant_pages;
                row >> name >> peak >> grant_pages;
                const auto observed = runner.execute("admit-witness " + name + " " + name + " " +
                    std::to_string(peak) + " " + std::to_string(grant_pages));
                output << "admit-witness " << name << " " << name << " " << peak << " " << grant_pages
                       << " | " << observed << '\n';
                admitted_b_ = admitted_b_ || (name == "B" && observed == "accepted");
            }
            else if (kind == "drop-request" && name == "B" && !admitted_b_)
                continue; // Refused requests own no record; nothing to retire.
            else
                output << line << " | " << runner.execute(line) << '\n';
        }
        output.close();
        ASSERT_TRUE(output.good());
        std::ifstream captured(record);
        EXPECT_NO_THROW(replay(captured));
        return;
    }
    std::istringstream input(llaminar2::test::privatePageWitnessTrace);
    EXPECT_NO_THROW(replay(input));
}

/** @test Witness replay rejects invented acceptance/refusal observations. */
TEST(PhysicalMemoryAuthority, PagePoolWitnessReplayRejectsAlteredObservations)
{
    for (const auto &[original, changed] : std::vector<std::pair<std::string, std::string>>{
        {"admit-witness B B 32 1 | refused", "admit-witness B B 32 1 | accepted"},
        {"drop-request A | retired", "drop-request A | refused"},
        {"end | idle\n", ""}})
    {
        std::string corrupt(llaminar2::test::privatePageWitnessTrace);
        const auto position = corrupt.find(original);
        ASSERT_NE(position, std::string::npos);
        corrupt.replace(position, original.size(), changed);
        std::istringstream input(corrupt);
        EXPECT_THROW(replay(input), std::exception);
    }
}

/** @test Recorded all-class pool+table admission replay through real owners. */
TEST(PhysicalMemoryAuthority, PagePoolRecordedCombinedAdmissionReplayExactly)
{
    if (const char *record = std::getenv("LLAMINAR_RECORD_PRIVATE_COMBINED_TRACE"))
    {
        std::ofstream output(record);
        ASSERT_TRUE(output.good());
        output << "pma-private-pages-v2 2 64 3 2 empty-zero-generations table 1 2 2 1 empty-zero-versions fp32-le\n";
        PrivatePageReplay runner({2, 64, 3, 2}, CPUFP32PagedKVTableGeometry{1, 2, 2, 1});
        for (const std::string command : {"combined one 1", "combined two 1", "combined three 1",
            "combined-abort one", "combined-abort two", "combined three 1",
            "combined-abort three", "end"})
            output << command << " | " << runner.execute(command) << '\n';
        output.close();
        ASSERT_TRUE(output.good());
        std::ifstream captured(record);
        EXPECT_NO_THROW(replay(captured));
        return;
    }
    std::istringstream input(llaminar2::test::privatePageCombinedTrace);
    EXPECT_NO_THROW(replay(input));
}

/** @test Combined-admission replay rejects invented acceptance/refusal observations. */
TEST(PhysicalMemoryAuthority, PagePoolCombinedReplayRejectsAlteredObservations)
{
    for (const auto &[original, changed] : std::vector<std::pair<std::string, std::string>>{
        {"combined three 1 | refused", "combined three 1 | accepted"},
        {"combined-abort one | aborted", "combined-abort one | retired"},
        {"end | idle\n", ""}})
    {
        std::string corrupt(llaminar2::test::privatePageCombinedTrace);
        const auto position = corrupt.find(original);
        ASSERT_NE(position, std::string::npos);
        corrupt.replace(position, original.size(), changed);
        std::istringstream input(corrupt);
        EXPECT_THROW(replay(input), std::exception);
    }
}

/** @test Recorded immutable table banks, late borrowers and private-frame return replay through real owners. */
TEST(PhysicalMemoryAuthority, PagePoolRecordedTableBanksReplayExactly)
{
    if (const char *record = std::getenv("LLAMINAR_RECORD_PRIVATE_TABLE_TRACE"))
    {
        std::ofstream output(record);
        ASSERT_TRUE(output.good());
        output << "pma-private-pages-v2 3 128 4 3 empty-zero-generations table 1 2 2 2 empty-zero-versions fp32-le\n";
        PrivatePageReplay runner({3, 128, 4, 3}, CPUFP32PagedKVTableGeometry{1, 2, 2, 2});
        for (const std::string command : {"admit A 2", "assign A", "plan W A 0 32",
            "copy W 0000803f00000040000030410000404100004040000080400000504100006041", "commit W P",
            "table-begin U", "table-append U P 1 2", "table-commit U S", "table-borrow B S",
            "table-begin U2", "assign A", "plan W2 A 1 32",
            "copy W2 0000a0400000c04000007041000080410000e040000000410000884100009041", "commit W2 P2",
            "table-append U2 P2 1 2", "table-commit U2 T", "table-begin Busy", "table-close S", "table-drop S",
            "table-borrow C B", "table-read B", "table-drop B", "table-begin Abort", "table-abort Abort",
            "table-begin U3", "discard A 0", "remaining A", "assign A", "plan W3 A 1 32",
            "copy W3 0000803f00000040000030410000404100004040000080400000504100006041", "commit W3 P3",
            "table-append U3 P3 1 2", "table-commit U3 New", "table-read New", "table-read T",
            "table-drop T", "table-drop New", "drop-request A", "admit Hold 1", "assign Hold",
            "pin H Hold 0", "discard Hold 0", "remaining Hold", "drop-request Hold", "admit Replace 1",
            "remaining Replace", "drop-pin H", "remaining Replace", "assign Replace", "drop-request Replace", "end"})
            output << command << " | " << runner.execute(command) << '\n';
        output.close(); ASSERT_TRUE(output.good());
        std::ifstream captured(record);
        EXPECT_NO_THROW(replay(captured));
        return;
    }
    std::istringstream input(llaminar2::test::privatePageTableTrace);
    EXPECT_NO_THROW(replay(input));
}

/** @test Table replay rejects invented version/credit observations and absent initial/terminal state. */
TEST(PhysicalMemoryAuthority, PagePoolTableReplayRejectsAlteredStateAndObservations)
{
    for (const auto &[original, changed] : std::vector<std::pair<std::string, std::string>>{
        {"table-commit U3 New | 4", "table-commit U3 New | 3"},
        {"table-borrow C B | refused", "table-borrow C B | accepted"},
        {"remaining Replace | 1", "remaining Replace | 2"},
        {"empty-zero-versions", "missing-table-state"},
        {"end | idle\n", ""}})
    {
        std::string corrupt(llaminar2::test::privatePageTableTrace);
        const auto position = corrupt.find(original);
        ASSERT_NE(position, std::string::npos);
        corrupt.replace(position, original.size(), changed);
        std::istringstream input(corrupt);
        EXPECT_THROW(replay(input), std::exception);
    }
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
