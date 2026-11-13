#!/usr/bin/env python3
"""Replay the actual pipeline retirement body without a GPU or model.

Generation runs at build time; the resulting binary needs no compiler or SDK.
Only native completion and stage endpoints are doubled. The complete production
retirement function, terminal authority transition, and post-retirement guard
are compiled unchanged, including the original failing fixed-head branch.
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / 'src/v2/execution/local_execution/orchestrators/PipelineDeviceGeneration.cpp'


def body(text, signature):
    """Extract one complete original definition, retaining every nested branch."""
    start = text.index(signature)
    opening = text.index('{', start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]


def generate(output):
    original = SOURCE.read_text()
    retirement = body(original, 'void PipelineDeviceGeneration::retireForwardOnlyBeforeStageRelease() noexcept')
    guard = body(original, 'void PipelineDeviceGeneration::requireLiveParticipantBorrow() const')
    # Doubles retain the actual production field/API spellings. A second
    # stage access after revocation is an explicit failure, not freed memory.
    harness = r'''
#include <array>
#include <cassert>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#undef assert
#define assert(expression) do { if (!(expression)) throw std::runtime_error("Replay invariant failed: " #expression); } while(false)
#define LOG_ERROR(message) do {} while(false)
namespace collective_timeout_policy { int effectiveCollectTimeoutMs(int) { return 30; } }
enum class MoEOverlayInferenceCompletionFenceProgress { Ready, Failed };
struct Completion { bool recorded=false; auto poll(std::string*) { return MoEOverlayInferenceCompletionFenceProgress::Ready; } };
struct Admission {};
struct Device { int ordinal=0; };
int native_waits=0;
struct Backend { bool waitForEvent(void*,int) { ++native_waits; return true; } };
Backend* getBackendFor(Device) { static Backend backend; return &backend; }
struct Stage {
struct { Device device_id; } state_;
struct { std::shared_ptr<void> event; } device_generation_state_ready_;
std::shared_ptr<Admission> pipeline_maintenance_admission_;
int pipeline_maintenance_participant_=1;
void* pipeline_forward_edges_=reinterpret_cast<void*>(64);
bool alive=true; int retired=0;
void retireExecutionTopologyBeforeSharedStorageDetach() { assert(alive); ++retired; }
};
struct Family { struct Participant { bool submitted=false; }; std::vector<Participant> participants; };
struct PipelineDeviceGeneration {
enum class ParticipantBorrowLifecycle { Live, Retired };
ParticipantBorrowLifecycle participant_borrow_lifecycle_=ParticipantBorrowLifecycle::Live;
std::vector<std::unique_ptr<Completion>> rank_completion_slots_;
std::vector<int> rank_source_completions_;
std::unique_ptr<int> workers_=std::make_unique<int>(1);
std::array<std::unique_ptr<Family>,2> forward_only_families_;
std::vector<Stage*> stages_;
std::vector<Stage*> domains_;
std::shared_ptr<Admission> maintenance_admission_;
bool fixed_head_pipeline_=true;
std::vector<int> forward_edges_;
void retireForwardOnlyBeforeStageRelease() noexcept;
void requireLiveParticipantBorrow() const;
~PipelineDeviceGeneration() { retireForwardOnlyBeforeStageRelease(); }
};
'''
    main = r'''
int main() {
// Actual flattened populations: PP(1+1), PP(1+2), and PP(EP4+EP4).
for (int population : {2,3,8})
for (bool fixed : {false,true}) for (bool maintenance : {false,true})
for (int submitted_pattern : {0,1,2,3}) {
    std::vector<Stage> participants(population);
    auto pipeline=std::make_unique<PipelineDeviceGeneration>();
    pipeline->fixed_head_pipeline_=fixed;
    for (int index=0; index<population; ++index) {
        participants[index].state_.device_id.ordinal=index;
        pipeline->stages_.push_back(&participants[index]);
        pipeline->domains_.push_back(&participants[index]);
        pipeline->forward_edges_.push_back(index);
    }
    if (maintenance) {
        pipeline->maintenance_admission_=std::make_shared<Admission>();
        for (auto &stage : participants)
            stage.pipeline_maintenance_admission_=pipeline->maintenance_admission_;
    }
    int expected_waits=0;
    if (submitted_pattern) {
        pipeline->forward_only_families_[0]=std::make_unique<Family>();
        for (int index=0; index<population; ++index) {
            // First-only, alternating, and last-only submissions prove unequal
            // sets and inactive peers retain no invented terminal event.
            const bool submitted=submitted_pattern==1 ? index==0 :
                submitted_pattern==2 ? index%2==0 : index+1==population;
            pipeline->forward_only_families_[0]->participants.push_back({submitted});
            if (submitted) {
                participants[index].device_generation_state_ready_.event=std::make_shared<int>(1);
                ++expected_waits;
            }
        }
    }
    const int prior_waits=native_waits;
    // Cold failure has no submitted family. Partial submission retains only
    // actual producer events, independent of the participant population.
    pipeline->retireForwardOnlyBeforeStageRelease();
    assert(native_waits-prior_waits==expected_waits);
    assert(pipeline->participant_borrow_lifecycle_==PipelineDeviceGeneration::ParticipantBorrowLifecycle::Retired);
    assert(pipeline->stages_.empty() && pipeline->domains_.empty());
    assert(pipeline->fixed_head_pipeline_==fixed);
    for (const auto &stage : participants)
        assert(stage.retired==int(fixed||maintenance));
    bool rejected=false;
    try { pipeline->requireLiveParticipantBorrow(); } catch(const std::logic_error&) { rejected=true; }
    assert(rejected);
    for (auto &stage : participants) stage.alive=false; // Rank released stage owners.
    pipeline->retireForwardOnlyBeforeStageRelease();
    pipeline.reset(); // The real member-order second destructor boundary.
    assert(native_waits-prior_waits==expected_waits);
}
}

'''
    output.write_text(harness + retirement + "\n" + guard + main)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    generate(parser.parse_args().output)
