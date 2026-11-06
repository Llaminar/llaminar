/**
 * @file Test__Qwen4ExpLocalHotTerminalEvidence.cpp
 * @brief Device-free malformed-window regressions for actual local cache reports.
 *
 * Fixtures describe immutable completed counters only. They never claim native
 * execution or replace the real model's original storage/transport witnesses.
 */
#include "integration/parity/qwen4exp/Qwen4ExpLocalHotTerminalEvidence.h"
#include <gtest/gtest.h>

namespace llaminar2::test::parity::qwen4exp {
namespace {
namespace h=::llaminar2::qwen4exp::local_hot;
namespace t=::llaminar2::qwen4exp::local_hot_transport;
/** @return Distinct original coordinator/reader fixture identities, independent of native inventory. */
std::array<Qwen4ExpLocalHotTerminalOwner,2> owners()
{
    return {{{DeviceId::cuda(0),101,8,90,h::TerminalReceiveRole::Coordinator},
             {DeviceId::rocm(3),202,8,90,h::TerminalReceiveRole::Reader}}};
}
/** @return A complete producer-schema fixture with genuinely different local miss sets. */
std::vector<PerfStatRecord> evidence(bool zero=false)
{
    std::vector<PerfStatRecord> result;
    for(const auto& owner:owners()) {
        h::TerminalObservation observation;
        observation.values[0]=64;observation.values[1]=zero?0:16;
        observation.values[3]=64;observation.values[8]=zero?0:2;
        if(owner.receive_role==h::TerminalReceiveRole::Coordinator) {
            observation.values[10]=12*sizeof(std::uint64_t);
            observation.values[12]=2*sizeof(t::RequestHeader);
            observation.values[13]=2*sizeof(t::ResultHeader);
        } else {
            observation.values[11]=12*owner.native_row_bytes;
            observation.values[12]=2*sizeof(t::ResultHeader);
            observation.values[14]=sizeof(t::SourceAdmissionHeader);
        }
        for(std::size_t i=0;i<15;++i) {
            PerfStatRecord record;
            record.domain="ngram_lookup";record.name=h::TerminalMetricNames[i];
            record.device=owner.device.toString();record.phase="generation_terminal";
            record.tags={{"authority","completed_local_hot_terminal"},
                {"storage_identity",std::to_string(owner.storage_identity)},
                {"unit",i<4?"original_l1_ids":i<10?"native_rows":"bytes"},
                {"wire_attribution","receive_completed_only"},
                {"window","since_original_storage_initialization"}};
            record.count=1;record.value=observation.values[i];record.sequence_word_count=1;
            record.sequence_minimum_words={observation.values[i]};
            record.sequence_maximum_words={observation.values[i]};
            result.push_back(std::move(record));
        }
    }
    return result;
}
/** @brief Alter the value with its native integer observation, isolating semantic rejection. */
void setValue(PerfStatRecord& record,std::uint64_t value)
{record.value=value;record.sequence_minimum_words={value};record.sequence_maximum_words={value};}
}

TEST(Qwen4ExpLocalHotTerminalEvidence, CompleteDistinctConsumersAndCompletedWindowsPass)
{
    auto records=evidence();const auto original=owners();
    EXPECT_NO_THROW(requireQwen4ExpLocalHotTerminalEvidence(records,original));
    auto later=records;
    for(auto& record:later)record.tags["window"]="between_completed_terminals";
    records.insert(records.end(),later.begin(),later.end());
    EXPECT_NO_THROW(requireQwen4ExpLocalHotTerminalEvidence(records,original));
    for(auto& record:records) {
        if(record.tags.at("window")=="between_completed_terminals") {
            record.value*=2;record.count=2;record.sequence_word_count=2;
        }
    }
    EXPECT_NO_THROW(requireQwen4ExpLocalHotTerminalEvidence(records,original));
}
TEST(Qwen4ExpLocalHotTerminalEvidence, EveryConsumerAndEveryFieldIsMandatory)
{
    const auto original=owners();
    for(std::size_t missing=0;missing<30;++missing) {
        auto records=evidence();records.erase(records.begin()+missing);
        EXPECT_THROW(requireQwen4ExpLocalHotTerminalEvidence(records,original),std::invalid_argument)<<missing;
    }
    auto records=evidence();records.resize(15);
    EXPECT_THROW(requireQwen4ExpLocalHotTerminalEvidence(records,original),std::invalid_argument);
    records=evidence();records.push_back(records.front());
    EXPECT_THROW(requireQwen4ExpLocalHotTerminalEvidence(records,original),std::invalid_argument);
    auto duplicate=original;duplicate[1]=duplicate[0];
    EXPECT_THROW(requireQwen4ExpLocalHotTerminalEvidence(evidence(),duplicate),std::invalid_argument);
}
TEST(Qwen4ExpLocalHotTerminalEvidence, ForeignIdentityUnitsWindowAndIncompleteCountsReject)
{
    const auto original=owners();
    for(const auto& [key,value]:std::vector<std::pair<std::string,std::string>>{
        {"authority","controller_alias"},{"storage_identity","999"},{"unit","bytes"},
        {"wire_attribution","allocated_capacity"},{"window","per_token"}}) {
        auto records=evidence();records.front().tags[key]=value;
        EXPECT_THROW(requireQwen4ExpLocalHotTerminalEvidence(records,original),std::invalid_argument)<<key;
    }
    auto records=evidence();records.front().device="ROCm:2";
    EXPECT_THROW(requireQwen4ExpLocalHotTerminalEvidence(records,original),std::invalid_argument);
    records=evidence();records.front().name="local_invented_metric";
    EXPECT_THROW(requireQwen4ExpLocalHotTerminalEvidence(records,original),std::invalid_argument);
    records=evidence();records.front().count=2;records.front().sequence_word_count=2;
    records.front().value*=2;
    EXPECT_THROW(requireQwen4ExpLocalHotTerminalEvidence(records,original),std::invalid_argument);
    records=evidence();records.front().sequence_word_count=0;
    EXPECT_THROW(requireQwen4ExpLocalHotTerminalEvidence(records,original),std::invalid_argument);
    records=evidence();
    for(auto& record:records) {
        record.count=2;record.sequence_word_count=2;record.value*=2;
    }
    EXPECT_THROW(requireQwen4ExpLocalHotTerminalEvidence(records,original),std::invalid_argument);
}
TEST(Qwen4ExpLocalHotTerminalEvidence, ZeroCapacityNativeExtentAndActualMissBoundsAreEnforced)
{
    auto original=owners();for(auto& owner:original)owner.slots=0;
    EXPECT_NO_THROW(requireQwen4ExpLocalHotTerminalEvidence(evidence(true),original));
    EXPECT_THROW(requireQwen4ExpLocalHotTerminalEvidence(evidence(),original),std::invalid_argument);
    original=owners();
    for(const auto value:{std::uint64_t{1},std::uint64_t{640}*64,std::uint64_t{65}*90}) {
        auto records=evidence();setValue(records[15+11],value);
        EXPECT_THROW(requireQwen4ExpLocalHotTerminalEvidence(records,original),std::invalid_argument)<<value;
    }
    auto records=evidence();setValue(records[11],90);
    EXPECT_THROW(requireQwen4ExpLocalHotTerminalEvidence(records,original),std::invalid_argument);
    records=evidence();setValue(records[1],65);
    EXPECT_THROW(requireQwen4ExpLocalHotTerminalEvidence(records,original),std::invalid_argument);
    records=evidence();setValue(records[2],17);
    EXPECT_THROW(requireQwen4ExpLocalHotTerminalEvidence(records,original),std::invalid_argument);
}
} // namespace llaminar2::test::parity::qwen4exp
