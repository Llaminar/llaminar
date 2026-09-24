/**
 * @file Ornith15AccuracyWorkload.h
 * @brief Frozen natural-language workload reproducing Ornith Q8 token drift.
 *
 * These are the exact 424 raw-completion prompt tokens used by the CPU and
 * four-ROCm comparison, without chat templating or synthetic repeated padding.
 * The HF reference owns the decode continuation; no generated answer is
 * embedded here or silently rebaselined. Eighty-nine incremental steps cover
 * both observed disagreement frontiers (38 and 88) and their next state.
 */
#pragma once

#include <string_view>
#include <vector>

namespace llaminar2::test::parity::qwen36
{
    /** Exact UTF-8 prompt authenticated by the reference metadata. */
    inline constexpr std::string_view kOrnith15AccuracyPrompt = R"ORNITH(You are reviewing a production large-language-model inference engine that uses speculative multi-token prediction on two GPUs. Analyze the design under four constraints: generated tokens must match serial stochastic decode for the same random draws, all recurrent state remains device-resident, collectives are captured inside one reusable GPU graph, and the optimization must improve end-to-end throughput rather than merely move latency between stages. Identify the most likely performance bottlenecks, explain how you would measure each one, and propose an ordered tuning plan.

The model is a mixture-of-experts transformer. Every device holds complete decode-time replicas of the routed experts and the multi-token prediction head, while ordinary prefill apportions expert rows across devices. A device-side least-loaded assignment planner may update row ownership between decode windows. The hot path must not allocate memory, synchronize a whole device or stream, transfer intermediate values through the host, or launch an eager fallback. Producer and consumer streams communicate through explicit events. A homogeneous pair of GPUs must execute one monolithic captured graph, including NCCL collectives, and segmented execution is considered a correctness failure.

Treat stochastic verification as the production case. Candidate rows are evaluated together, but each row must consume the same random draw and produce the same token that serial decode would have produced at that position. Acceptance, recurrent-state advancement, key-value cache advancement, rollback after a rejected suffix, and terminal hidden-state publication must remain ordered and byte-exact. The controller may select depths from one through fifteen, although this benchmark fixes the depth so graph and communication costs can be measured independently from controller hysteresis.

For each suspected bottleneck, name the evidence needed from end-to-end PerfStats, Nsight Systems, and Nsight Compute. Distinguish launch latency, collective latency, memory bandwidth, tensor-core utilization, occupancy limits, register spilling, redundant graph work, and acceptance-rate effects. Explain which metric would falsify your hypothesis. Then describe the smallest correctness regression and the smallest representative performance benchmark that should accompany each change. Use concrete engineering language and finish with a short list of numerical acceptance criteria.)ORNITH";

    /** Decode horizon includes both observed mismatches and their successor. */
    inline constexpr int kOrnith15AccuracyDecodeSteps = 89;

    /** @return Model-tokenizer identity, checked against the HF reference pack. */
    inline std::vector<int> ornith15AccuracyTokenIds()
    {
        return {
            2523, 513, 32767, 264, 5492, 3349, 42117, 27327, 42903, 4560, 421, 5533,
            63520, 7072, 33686, 19039, 383, 1330, 68022, 13, 36189, 2891, 279, 2790,
            1172, 2943, 16484, 25, 7658, 10885, 1902, 2353, 5953, 91954, 16401, 364,
            279, 1788, 4055, 25791, 11, 660, 61911, 1528, 8198, 3545, 11255, 1653,
            11, 6327, 1821, 513, 16508, 4613, 799, 59493, 21966, 4618, 11, 321,
            279, 24460, 1902, 7042, 809, 4534, 12692, 61610, 4598, 1056, 15756, 3166,
            37972, 1881, 17102, 13, 62366, 279, 1379, 4222, 4906, 10617, 33059, 14134,
            11, 10033, 1204, 488, 1000, 6420, 1754, 799, 11, 321, 28647, 449,
            11128, 39972, 3019, 13, 271, 760, 1558, 369, 264, 20340, 8404, 17830,
            15089, 41163, 13, 6983, 3545, 9687, 4434, 16401, 7019, 78014, 314, 279,
            69764, 11312, 321, 279, 7072, 33686, 19039, 1901, 11, 1345, 18541, 829,
            7320, 182874, 880, 6009, 6761, 3808, 7370, 13, 357, 3545, 23314, 3140,
            75395, 15840, 48038, 1189, 2560, 2713, 14834, 1881, 16401, 10708, 13, 561,
            3882, 1752, 1902, 524, 21405, 4779, 11, 61646, 264, 4220, 3545, 466,
            4129, 11, 8059, 27510, 2663, 1472, 279, 3357, 11, 466, 6830, 449,
            22806, 31686, 13, 41926, 321, 11171, 22327, 18459, 1472, 11135, 4216, 13,
            357, 83208, 6505, 314, 68022, 1902, 8754, 799, 1559, 63164, 16508, 4618,
            11, 2583, 19448, 3040, 6327, 1821, 11, 321, 82015, 10993, 369, 6306,
            264, 55404, 7652, 13, 271, 51, 1180, 91954, 22188, 430, 279, 5492,
            1105, 13, 47914, 6761, 513, 24276, 3658, 11, 694, 1754, 2713, 1902,
            23304, 279, 1788, 4055, 3902, 321, 7936, 279, 1788, 3817, 421, 5953,
            16401, 1000, 599, 8677, 506, 421, 2234, 13, 20195, 663, 11, 61911,
            20105, 48150, 11, 1328, 18509, 6297, 48150, 11, 58377, 1238, 264, 17030,
            19900, 11, 321, 14586, 7920, 20105, 16111, 1902, 6922, 11128, 321, 4763,
            9884, 519, 13, 561, 6259, 1189, 3186, 41439, 494, 799, 1472, 35442,
            11, 7643, 411, 27502, 25689, 279, 7739, 748, 4618, 321, 10222, 6829,
            628, 381, 16384, 27222, 494, 6259, 304, 582, 12547, 284, 13, 271,
            2381, 1754, 23059, 84810, 11, 803, 279, 5721, 4221, 494, 809, 4534,
            12692, 69024, 16145, 11, 443, 82, 481, 14486, 11, 321, 443, 82,
            481, 21902, 13, 414, 85596, 6830, 37972, 11, 21079, 37972, 11, 4779,
            32288, 11, 15167, 22728, 47785, 11, 63113, 13001, 11, 4025, 947, 9121,
            11, 46142, 4618, 944, 11, 321, 24693, 41606, 6043, 13, 79091, 864,
            17723, 1000, 30882, 1386, 678, 29094, 13, 4844, 7276, 279, 23856, 55404,
            29551, 321, 279, 23856, 17697, 4906, 27502, 421, 1220, 18646, 1754, 2222,
            13, 5272, 13769, 14246, 3992, 321, 6052, 440, 264, 2716, 1103, 314,
            33625, 24693, 12521, 13,
        };
    }
} // namespace llaminar2::test::parity::qwen36
