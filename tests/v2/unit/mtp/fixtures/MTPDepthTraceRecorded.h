/**
 * @file MTPDepthTraceRecorded.h
 * @brief Genuine model-recorded repetition inputs with immutable producer identity.
 *
 * Both original TRAIN captures executed Qwen3.8-Flash-Next UD-Q4_K_XL (qwen4exp
 * MoE, hidden2560/main48/expert512/top10) with its shared Q8_0 MTP predictor,
 * ROCm gfx906/MI50 TP4, PP1, ExactLogicalRows retained-capacity15, context4096,
 * prompt512/output256, activationFP32/KVFP16, zero warmups/one measured request.
 * Greedy uses T0/k1/p1/seed42; stochastic uses T0.7/k40/p0.9/seed4242 (the
 * native config retains its actual round-trip float values). Both contain43
 * physical transactions, two evaluated windows, no utility probes/keeps/returns,
 * and an actual budget-clipped last transaction. These unchanged depth5 paths
 * complement authored arithmetic and future nontrivial utility regressions;
 * their diagnostic timings are never performance evidence.
 *
 * Original passed10-capture/40-participant baseline manifest:
 * /tmp/qfn-mtp-train-controller-trace-v2-utility-baseline-replays-r1/manifest.json
 * Greedy capture receipt:20f859e39a5925000f282f4b049fa9c82ac10cea2a767d7786f15633bd327133
 * Stochastic receipt:65b4d085b1e659ed4740fbab01aad2e3b8c5894ad58601f1640ac578027f364b
 * Original native captures are mtp-train-controller-trace-v2-utility-greedy-r1
 * and mtp-train-controller-trace-v2-utility-stochastic-r2, train-repetition cell,
 * ROCm:0. All four participants agree except device/output custody; extraction
 * verified every saved request/replay file stat against the original manifest.
 *
 * Producer signatures, capture fields, all108 initial/final words and every
 * ordered transaction are retained exactly. Only intent changes baseline to
 * candidate and output changes to an unused test path. No compiledIdentity()
 * substitution, current source hash, model payload or runtime payload is added.
 * Candidate replay must obey its original ABI and stop unsupported frontiers.
 */
#pragma once
#include <string_view>

namespace llaminar2::test::mtp_trace_replay::recorded
{
/** @brief Exact historical ROCm:0 greedy request; only intent/output are test-local. */
inline constexpr std::string_view kGreedyRepetition = R"MTP_RECORD(
{
  "schema": "llaminar.mtp.depth_trace_replay_request.v2",
  "intent": "candidate",
  "authority": "native_resident",
  "compiled_identity": {
    "schema": "llaminar.mtp.depth_replay_source_identity.v1",
    "sources": {
      "control_layout": "1331c08379adae4b1ffe204dd2d0a4d9fb0a4105e5e08b273ac87a89a8143ccc",
      "depth_defaults": "14b57d8bbfe813b1e0a721f676c2f1fbd42ba7842cdc5fbb9074f84ac195e1e6",
      "device_policy_projection": "9b9b7367a551015af30288fe6be91160cdaf8ad11b7568afc09072a0d0ac5b5d",
      "host_controller": "d2aef299023b9214150c7f3d3bf7107865c4923fb1793af1caa153a2f89d5b8d",
      "host_controller_header": "17399218dc4a0c60ccd9599a9b7b3074545dedf1aa4cd5a8c09be0c08ce5261a",
      "initialization": "4d7da4f4f9b1d144f824fa985ffc493957eda9ffdf24d1cda10e9cf249cd5e37",
      "learned_interpreter": "5ca62068b6652cc5d70119b07620608c1bc37f1a3ca22c0b8d588db5933aa0c0",
      "policy_scope": "eaed3544fd4bcb45509f5042dd8a7bb25bae9b50d3b90368083ab7106061b343",
      "policy_table": "087e62a0b2a392e245cccefe00e5b32feb7ee2aae590bc3d065af9fca65068eb",
      "policy_types": "ed17c71245961b263541323da3466cb9907e540e086bad33c70b0499fa173b9e",
      "resident_controller": "455191bd05da96d9664fadaa754fb32e63911a1ac42b0f5724df7a4c7e7a040f",
      "reversible_proposal_gate": "bcc7b39ac42f9dcbcbb12dd8e656cd03a07e4732fa333b344f3dfdc0e04673c0",
      "runtime_policy": "66447b7a921af2799e648e2aeea3878e1afb0fc8f8ea3d762836f2c089af51c6",
      "transaction_cost_profile": "c21f3ac989ac004b41e410be9647badde6c305aa60ce4a05b2219be4b031ecf0",
      "transaction_trace": "fdcea3e4656fc34e095a9fd4d73f0a93cbc66e7b57bd7adbf53c01ae63859916",
      "verifier_width_policy": "825dc58e4d75ed175ba90e938e2bdc2c6889e147557b3c0097883f89e6a69175"
    }
  },
  "capture": {
    "observation_identity": "7:1:0",
    "device": "ROCm:0",
    "receipt_sha256": "20f859e39a5925000f282f4b049fa9c82ac10cea2a767d7786f15633bd327133",
    "configuration_sha256": "47134785dab7f81d5e2016eefad98dbb2f78fbee749c628a722f34ca86e7907f"
  },
  "initial": {
    "phase": "validated_pre_first_response_append",
    "control": [
      1, 0, 256, 0, 0, 0, 0, 6, 0, 0, 0, 0,
      0, 5, 6, 1, 15, 2, 16, 4, 8, 3, 1000000, 450000,
      550000, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 1, 3,
      2, 0, 0, 5, 192, 0, 0, 0, 36864, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 2, 4, 1, 15, 2560, 48,
      512, 10, 90, 6, 0, 0, 0, 0, 0, 0, 0, 0
    ]
  },
  "transactions": [
    {"ordinal": 1, "packed": 167773525, "token_witness": 6, "evaluated_windows": 0},
    {"ordinal": 2, "packed": 167773525, "token_witness": 12, "evaluated_windows": 0},
    {"ordinal": 3, "packed": 167773525, "token_witness": 18, "evaluated_windows": 0},
    {"ordinal": 4, "packed": 167773525, "token_witness": 24, "evaluated_windows": 0},
    {"ordinal": 5, "packed": 167773525, "token_witness": 30, "evaluated_windows": 0},
    {"ordinal": 6, "packed": 167773525, "token_witness": 36, "evaluated_windows": 0},
    {"ordinal": 7, "packed": 167773525, "token_witness": 42, "evaluated_windows": 0},
    {"ordinal": 8, "packed": 167773525, "token_witness": 48, "evaluated_windows": 0},
    {"ordinal": 9, "packed": 167773525, "token_witness": 54, "evaluated_windows": 0},
    {"ordinal": 10, "packed": 167773525, "token_witness": 60, "evaluated_windows": 0},
    {"ordinal": 11, "packed": 167773525, "token_witness": 66, "evaluated_windows": 0},
    {"ordinal": 12, "packed": 167773525, "token_witness": 72, "evaluated_windows": 0},
    {"ordinal": 13, "packed": 167773525, "token_witness": 78, "evaluated_windows": 0},
    {"ordinal": 14, "packed": 167773525, "token_witness": 84, "evaluated_windows": 0},
    {"ordinal": 15, "packed": 167773525, "token_witness": 90, "evaluated_windows": 0},
    {"ordinal": 16, "packed": 167773525, "token_witness": 96, "evaluated_windows": 1},
    {"ordinal": 17, "packed": 167773525, "token_witness": 102, "evaluated_windows": 1},
    {"ordinal": 18, "packed": 167773525, "token_witness": 108, "evaluated_windows": 1},
    {"ordinal": 19, "packed": 167773525, "token_witness": 114, "evaluated_windows": 1},
    {"ordinal": 20, "packed": 167773525, "token_witness": 120, "evaluated_windows": 1},
    {"ordinal": 21, "packed": 167773525, "token_witness": 126, "evaluated_windows": 1},
    {"ordinal": 22, "packed": 167773525, "token_witness": 132, "evaluated_windows": 1},
    {"ordinal": 23, "packed": 167773525, "token_witness": 138, "evaluated_windows": 1},
    {"ordinal": 24, "packed": 167773525, "token_witness": 144, "evaluated_windows": 1},
    {"ordinal": 25, "packed": 167773525, "token_witness": 150, "evaluated_windows": 1},
    {"ordinal": 26, "packed": 167773525, "token_witness": 156, "evaluated_windows": 1},
    {"ordinal": 27, "packed": 167773525, "token_witness": 162, "evaluated_windows": 1},
    {"ordinal": 28, "packed": 167773525, "token_witness": 168, "evaluated_windows": 1},
    {"ordinal": 29, "packed": 167777605, "token_witness": 174, "evaluated_windows": 1},
    {"ordinal": 30, "packed": 167773525, "token_witness": 179, "evaluated_windows": 1},
    {"ordinal": 31, "packed": 167773525, "token_witness": 185, "evaluated_windows": 1},
    {"ordinal": 32, "packed": 167773525, "token_witness": 191, "evaluated_windows": 2},
    {"ordinal": 33, "packed": 167773525, "token_witness": 197, "evaluated_windows": 2},
    {"ordinal": 34, "packed": 167773525, "token_witness": 203, "evaluated_windows": 2},
    {"ordinal": 35, "packed": 167773525, "token_witness": 209, "evaluated_windows": 2},
    {"ordinal": 36, "packed": 167773525, "token_witness": 215, "evaluated_windows": 2},
    {"ordinal": 37, "packed": 167773525, "token_witness": 221, "evaluated_windows": 2},
    {"ordinal": 38, "packed": 167773525, "token_witness": 227, "evaluated_windows": 2},
    {"ordinal": 39, "packed": 167773525, "token_witness": 233, "evaluated_windows": 2},
    {"ordinal": 40, "packed": 167773525, "token_witness": 239, "evaluated_windows": 2},
    {"ordinal": 41, "packed": 167773525, "token_witness": 245, "evaluated_windows": 2},
    {"ordinal": 42, "packed": 167773525, "token_witness": 251, "evaluated_windows": 2},
    {"ordinal": 43, "packed": 167781701, "token_witness": 256, "evaluated_windows": 2}
  ],
  "terminal": {
    "transactions": 43,
    "control": [
      1, 256, 0, 0, 1, 43, 0, 0, 213, 1, 214, 0,
      256, 5, 6, 1, 15, 2, 16, 4, 8, 3, 1000000, 450000,
      550000, 50, 0, 10, 50, 50, 0, 0, 10, 0, 50, 2,
      0, 0, 0, 5, 215, 258, 5, 5, 0, 0, 1, 3,
      2, 0, 2, 5, 8256, 0, 344064, 0, 1585152, 0, 344064, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 2, 4, 1, 15, 2560, 48,
      512, 10, 90, 6, 0, 0, 0, 0, 60, 0, 0, 0
    ]
  },
  "output": "/tmp/llaminar-recorded-train-repetition-greedy-unused.json"
}
)MTP_RECORD";

/** @brief Exact historical ROCm:0 stochastic request; only intent/output are test-local. */
inline constexpr std::string_view kStochasticRepetition = R"MTP_RECORD(
{
  "schema": "llaminar.mtp.depth_trace_replay_request.v2",
  "intent": "candidate",
  "authority": "native_resident",
  "compiled_identity": {
    "schema": "llaminar.mtp.depth_replay_source_identity.v1",
    "sources": {
      "control_layout": "1331c08379adae4b1ffe204dd2d0a4d9fb0a4105e5e08b273ac87a89a8143ccc",
      "depth_defaults": "14b57d8bbfe813b1e0a721f676c2f1fbd42ba7842cdc5fbb9074f84ac195e1e6",
      "device_policy_projection": "9b9b7367a551015af30288fe6be91160cdaf8ad11b7568afc09072a0d0ac5b5d",
      "host_controller": "d2aef299023b9214150c7f3d3bf7107865c4923fb1793af1caa153a2f89d5b8d",
      "host_controller_header": "17399218dc4a0c60ccd9599a9b7b3074545dedf1aa4cd5a8c09be0c08ce5261a",
      "initialization": "4d7da4f4f9b1d144f824fa985ffc493957eda9ffdf24d1cda10e9cf249cd5e37",
      "learned_interpreter": "5ca62068b6652cc5d70119b07620608c1bc37f1a3ca22c0b8d588db5933aa0c0",
      "policy_scope": "eaed3544fd4bcb45509f5042dd8a7bb25bae9b50d3b90368083ab7106061b343",
      "policy_table": "087e62a0b2a392e245cccefe00e5b32feb7ee2aae590bc3d065af9fca65068eb",
      "policy_types": "ed17c71245961b263541323da3466cb9907e540e086bad33c70b0499fa173b9e",
      "resident_controller": "455191bd05da96d9664fadaa754fb32e63911a1ac42b0f5724df7a4c7e7a040f",
      "reversible_proposal_gate": "bcc7b39ac42f9dcbcbb12dd8e656cd03a07e4732fa333b344f3dfdc0e04673c0",
      "runtime_policy": "66447b7a921af2799e648e2aeea3878e1afb0fc8f8ea3d762836f2c089af51c6",
      "transaction_cost_profile": "c21f3ac989ac004b41e410be9647badde6c305aa60ce4a05b2219be4b031ecf0",
      "transaction_trace": "fdcea3e4656fc34e095a9fd4d73f0a93cbc66e7b57bd7adbf53c01ae63859916",
      "verifier_width_policy": "825dc58e4d75ed175ba90e938e2bdc2c6889e147557b3c0097883f89e6a69175"
    }
  },
  "capture": {
    "observation_identity": "7:1:0",
    "device": "ROCm:0",
    "receipt_sha256": "65b4d085b1e659ed4740fbab01aad2e3b8c5894ad58601f1640ac578027f364b",
    "configuration_sha256": "4f7a5a36d24166d8b86fb15748731bc0acb47b9ab435cfe3d42fe7497eb448bb"
  },
  "initial": {
    "phase": "validated_pre_first_response_append",
    "control": [
      1, 0, 256, 0, 0, 0, 0, 6, 0, 0, 0, 0,
      0, 5, 6, 1, 15, 2, 16, 4, 8, 3, 1000000, 450000,
      550000, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 1, 3,
      2, 1, 0, 5, 192, 0, 0, 0, 36864, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 2, 4, 1, 15, 2560, 48,
      512, 10, 90, 6, 0, 0, 0, 0, 0, 0, 0, 0
    ]
  },
  "transactions": [
    {"ordinal": 1, "packed": 167773525, "token_witness": 6, "evaluated_windows": 0},
    {"ordinal": 2, "packed": 167773525, "token_witness": 12, "evaluated_windows": 0},
    {"ordinal": 3, "packed": 167773525, "token_witness": 18, "evaluated_windows": 0},
    {"ordinal": 4, "packed": 167773525, "token_witness": 24, "evaluated_windows": 0},
    {"ordinal": 5, "packed": 167773525, "token_witness": 30, "evaluated_windows": 0},
    {"ordinal": 6, "packed": 167773525, "token_witness": 36, "evaluated_windows": 0},
    {"ordinal": 7, "packed": 167773525, "token_witness": 42, "evaluated_windows": 0},
    {"ordinal": 8, "packed": 167773525, "token_witness": 48, "evaluated_windows": 0},
    {"ordinal": 9, "packed": 167773525, "token_witness": 54, "evaluated_windows": 0},
    {"ordinal": 10, "packed": 167773525, "token_witness": 60, "evaluated_windows": 0},
    {"ordinal": 11, "packed": 167773525, "token_witness": 66, "evaluated_windows": 0},
    {"ordinal": 12, "packed": 167773525, "token_witness": 72, "evaluated_windows": 0},
    {"ordinal": 13, "packed": 167773525, "token_witness": 78, "evaluated_windows": 0},
    {"ordinal": 14, "packed": 167773525, "token_witness": 84, "evaluated_windows": 0},
    {"ordinal": 15, "packed": 167773525, "token_witness": 90, "evaluated_windows": 0},
    {"ordinal": 16, "packed": 167773525, "token_witness": 96, "evaluated_windows": 1},
    {"ordinal": 17, "packed": 167773525, "token_witness": 102, "evaluated_windows": 1},
    {"ordinal": 18, "packed": 167773525, "token_witness": 108, "evaluated_windows": 1},
    {"ordinal": 19, "packed": 167773525, "token_witness": 114, "evaluated_windows": 1},
    {"ordinal": 20, "packed": 167773525, "token_witness": 120, "evaluated_windows": 1},
    {"ordinal": 21, "packed": 167773525, "token_witness": 126, "evaluated_windows": 1},
    {"ordinal": 22, "packed": 167773525, "token_witness": 132, "evaluated_windows": 1},
    {"ordinal": 23, "packed": 167773525, "token_witness": 138, "evaluated_windows": 1},
    {"ordinal": 24, "packed": 167773525, "token_witness": 144, "evaluated_windows": 1},
    {"ordinal": 25, "packed": 167773525, "token_witness": 150, "evaluated_windows": 1},
    {"ordinal": 26, "packed": 167773525, "token_witness": 156, "evaluated_windows": 1},
    {"ordinal": 27, "packed": 167773525, "token_witness": 162, "evaluated_windows": 1},
    {"ordinal": 28, "packed": 167773525, "token_witness": 168, "evaluated_windows": 1},
    {"ordinal": 29, "packed": 167777605, "token_witness": 174, "evaluated_windows": 1},
    {"ordinal": 30, "packed": 167773525, "token_witness": 179, "evaluated_windows": 1},
    {"ordinal": 31, "packed": 167773525, "token_witness": 185, "evaluated_windows": 1},
    {"ordinal": 32, "packed": 167773525, "token_witness": 191, "evaluated_windows": 2},
    {"ordinal": 33, "packed": 167773525, "token_witness": 197, "evaluated_windows": 2},
    {"ordinal": 34, "packed": 167773525, "token_witness": 203, "evaluated_windows": 2},
    {"ordinal": 35, "packed": 167773525, "token_witness": 209, "evaluated_windows": 2},
    {"ordinal": 36, "packed": 167773525, "token_witness": 215, "evaluated_windows": 2},
    {"ordinal": 37, "packed": 167773525, "token_witness": 221, "evaluated_windows": 2},
    {"ordinal": 38, "packed": 167773525, "token_witness": 227, "evaluated_windows": 2},
    {"ordinal": 39, "packed": 167773525, "token_witness": 233, "evaluated_windows": 2},
    {"ordinal": 40, "packed": 167773525, "token_witness": 239, "evaluated_windows": 2},
    {"ordinal": 41, "packed": 167773525, "token_witness": 245, "evaluated_windows": 2},
    {"ordinal": 42, "packed": 167773525, "token_witness": 251, "evaluated_windows": 2},
    {"ordinal": 43, "packed": 167781701, "token_witness": 256, "evaluated_windows": 2}
  ],
  "terminal": {
    "transactions": 43,
    "control": [
      1, 256, 0, 0, 1, 43, 0, 0, 213, 1, 214, 0,
      256, 5, 6, 1, 15, 2, 16, 4, 8, 3, 1000000, 450000,
      550000, 50, 0, 10, 50, 50, 0, 0, 10, 0, 50, 2,
      0, 0, 0, 5, 215, 258, 5, 5, 0, 0, 1, 3,
      2, 1, 2, 5, 8256, 0, 344064, 0, 1585152, 0, 344064, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 2, 4, 1, 15, 2560, 48,
      512, 10, 90, 6, 0, 0, 0, 0, 60, 0, 0, 0
    ]
  },
  "output": "/tmp/llaminar-recorded-train-repetition-stochastic-unused.json"
}
)MTP_RECORD";

}
