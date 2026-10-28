/**
 * @file MTPDepthTraceDirectionalRecorded.h
 * @brief Genuine stochastic long-Python inputs at a protected downward boundary.
 *
 * Captured Qwen3.8-Flash-Next UD-Q4_K_XL, shared Q8_0 MTP predictor, qwen4exp
 * MoE (hidden2560/main48/expert512/top10), ROCm gfx906 TP4/PP1, ExactLogicalRows
 * retained-capacity15, context4096/prompt2306/output256, activationFP32/KVFP16,
 * T0.7/k40/p0.9/seed4242, zero warmups and one measured diagnostic request.
 * This is the complete original73-transaction record,108-word initial/final
 * state, and producer/capture identity. It is not a new model run or timing claim.
 *
 * Original source manifest:
 * /tmp/qfn-mtp-train-controller-trace-v2-utility-baseline-replays-r1/manifest.json
 * Native capture: mtp-train-controller-trace-v2-utility-stochastic-r2,
 * train-long-python--controllerTrace, ROCm:0, observation11:1:0.
 * Native receipt SHA256:
 * 65b4d085b1e659ed4740fbab01aad2e3b8c5894ad58601f1640ac578027f364b
 * Initial/final policy and trace signatures below are the ORIGINAL producer's.
 * All four saved participants agree except device/output custody, and extraction
 * verified their original manifest-issued request/replay stat identities.
 *
 * Historical T4 selects5-to4 with probe1 after actual emittedY14; T8 returns5
 * with return1 after four target transactions emittedY6. The saved whole trace
 * finishes at depth7/E14 with probe5/keep2/return3, but a changed candidate is
 * supported ONLY while it follows observed depths. Tests must stop at the first
 * unsupported selector rather than consuming invented future observations.
 *
 * Only intent changes baseline-to-candidate and output becomes an unused test
 * path. Every other JSON value is retained exactly. No compiledIdentity()
 * substitution or current source stamp is permitted; replay under new code is
 * explicit candidate intent, never historical-baseline certification.
 */
#pragma once
#include <string_view>

namespace llaminar2::test::mtp_trace_replay::recorded
{
/** @brief Exact historical ROCm:0 stochastic request; only intent/output are test-local. */
inline constexpr std::string_view kStochasticLongPythonDirectional = R"MTP_RECORD(
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
    "observation_identity": "11:1:0",
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
      2, 1, 0, 5, 192, 0, 12288, 0, 36864, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 2, 4, 1, 15, 2560, 48,
      512, 10, 90, 6, 0, 0, 0, 0, 0, 0, 0, 0
    ]
  },
  "transactions": [
    {"ordinal": 1, "packed": 167777605, "token_witness": 6, "evaluated_windows": 0},
    {"ordinal": 2, "packed": 167777605, "token_witness": 11, "evaluated_windows": 0},
    {"ordinal": 3, "packed": 167777557, "token_witness": 13, "evaluated_windows": 0},
    {"ordinal": 4, "packed": 134222853, "token_witness": 14, "evaluated_windows": 1},
    {"ordinal": 5, "packed": 134222884, "token_witness": 17, "evaluated_windows": 1},
    {"ordinal": 6, "packed": 134222852, "token_witness": 18, "evaluated_windows": 1},
    {"ordinal": 7, "packed": 134222852, "token_witness": 19, "evaluated_windows": 1},
    {"ordinal": 8, "packed": 167777540, "token_witness": 20, "evaluated_windows": 2},
    {"ordinal": 9, "packed": 167777557, "token_witness": 22, "evaluated_windows": 2},
    {"ordinal": 10, "packed": 167777541, "token_witness": 23, "evaluated_windows": 2},
    {"ordinal": 11, "packed": 167777557, "token_witness": 25, "evaluated_windows": 2},
    {"ordinal": 12, "packed": 167777589, "token_witness": 29, "evaluated_windows": 2},
    {"ordinal": 13, "packed": 167773525, "token_witness": 34, "evaluated_windows": 2},
    {"ordinal": 14, "packed": 167777573, "token_witness": 38, "evaluated_windows": 2},
    {"ordinal": 15, "packed": 167777573, "token_witness": 41, "evaluated_windows": 2},
    {"ordinal": 16, "packed": 167773525, "token_witness": 46, "evaluated_windows": 3},
    {"ordinal": 17, "packed": 167777541, "token_witness": 48, "evaluated_windows": 3},
    {"ordinal": 18, "packed": 167777573, "token_witness": 51, "evaluated_windows": 3},
    {"ordinal": 19, "packed": 167777605, "token_witness": 56, "evaluated_windows": 3},
    {"ordinal": 20, "packed": 167777589, "token_witness": 60, "evaluated_windows": 4},
    {"ordinal": 21, "packed": 167777573, "token_witness": 63, "evaluated_windows": 4},
    {"ordinal": 22, "packed": 167773525, "token_witness": 68, "evaluated_windows": 4},
    {"ordinal": 23, "packed": 167777573, "token_witness": 72, "evaluated_windows": 4},
    {"ordinal": 24, "packed": 167777605, "token_witness": 77, "evaluated_windows": 4},
    {"ordinal": 25, "packed": 167777557, "token_witness": 79, "evaluated_windows": 5},
    {"ordinal": 26, "packed": 167777557, "token_witness": 81, "evaluated_windows": 5},
    {"ordinal": 27, "packed": 167777541, "token_witness": 82, "evaluated_windows": 5},
    {"ordinal": 28, "packed": 167777557, "token_witness": 84, "evaluated_windows": 5},
    {"ordinal": 29, "packed": 134222853, "token_witness": 85, "evaluated_windows": 6},
    {"ordinal": 30, "packed": 134218820, "token_witness": 89, "evaluated_windows": 6},
    {"ordinal": 31, "packed": 134222884, "token_witness": 93, "evaluated_windows": 6},
    {"ordinal": 32, "packed": 134222852, "token_witness": 94, "evaluated_windows": 6},
    {"ordinal": 33, "packed": 134222884, "token_witness": 97, "evaluated_windows": 7},
    {"ordinal": 34, "packed": 134222868, "token_witness": 99, "evaluated_windows": 7},
    {"ordinal": 35, "packed": 134218820, "token_witness": 103, "evaluated_windows": 7},
    {"ordinal": 36, "packed": 134222900, "token_witness": 108, "evaluated_windows": 7},
    {"ordinal": 37, "packed": 134218820, "token_witness": 112, "evaluated_windows": 7},
    {"ordinal": 38, "packed": 134218820, "token_witness": 117, "evaluated_windows": 7},
    {"ordinal": 39, "packed": 234882884, "token_witness": 122, "evaluated_windows": 8},
    {"ordinal": 40, "packed": 234882935, "token_witness": 130, "evaluated_windows": 8},
    {"ordinal": 41, "packed": 234882935, "token_witness": 138, "evaluated_windows": 8},
    {"ordinal": 42, "packed": 234882935, "token_witness": 146, "evaluated_windows": 8},
    {"ordinal": 43, "packed": 234882935, "token_witness": 154, "evaluated_windows": 9},
    {"ordinal": 44, "packed": 234882935, "token_witness": 162, "evaluated_windows": 9},
    {"ordinal": 45, "packed": 234886999, "token_witness": 169, "evaluated_windows": 9},
    {"ordinal": 46, "packed": 234886919, "token_witness": 170, "evaluated_windows": 9},
    {"ordinal": 47, "packed": 134222871, "token_witness": 172, "evaluated_windows": 10},
    {"ordinal": 48, "packed": 134222852, "token_witness": 173, "evaluated_windows": 10},
    {"ordinal": 49, "packed": 134218820, "token_witness": 177, "evaluated_windows": 10},
    {"ordinal": 50, "packed": 134222884, "token_witness": 181, "evaluated_windows": 10},
    {"ordinal": 51, "packed": 234886932, "token_witness": 183, "evaluated_windows": 11},
    {"ordinal": 52, "packed": 234886999, "token_witness": 189, "evaluated_windows": 11},
    {"ordinal": 53, "packed": 234886919, "token_witness": 190, "evaluated_windows": 11},
    {"ordinal": 54, "packed": 234886983, "token_witness": 195, "evaluated_windows": 11},
    {"ordinal": 55, "packed": 234886919, "token_witness": 196, "evaluated_windows": 11},
    {"ordinal": 56, "packed": 234886967, "token_witness": 200, "evaluated_windows": 11},
    {"ordinal": 57, "packed": 234886967, "token_witness": 204, "evaluated_windows": 11},
    {"ordinal": 58, "packed": 234886983, "token_witness": 209, "evaluated_windows": 11},
    {"ordinal": 59, "packed": 234886999, "token_witness": 215, "evaluated_windows": 12},
    {"ordinal": 60, "packed": 234882935, "token_witness": 222, "evaluated_windows": 12},
    {"ordinal": 61, "packed": 234882935, "token_witness": 230, "evaluated_windows": 12},
    {"ordinal": 62, "packed": 234886951, "token_witness": 234, "evaluated_windows": 12},
    {"ordinal": 63, "packed": 167777543, "token_witness": 235, "evaluated_windows": 13},
    {"ordinal": 64, "packed": 167777541, "token_witness": 236, "evaluated_windows": 13},
    {"ordinal": 65, "packed": 167777589, "token_witness": 240, "evaluated_windows": 13},
    {"ordinal": 66, "packed": 167777541, "token_witness": 241, "evaluated_windows": 13},
    {"ordinal": 67, "packed": 234886917, "token_witness": 242, "evaluated_windows": 14},
    {"ordinal": 68, "packed": 234886935, "token_witness": 244, "evaluated_windows": 14},
    {"ordinal": 69, "packed": 234886919, "token_witness": 245, "evaluated_windows": 14},
    {"ordinal": 70, "packed": 234886935, "token_witness": 247, "evaluated_windows": 14},
    {"ordinal": 71, "packed": 234886967, "token_witness": 251, "evaluated_windows": 14},
    {"ordinal": 72, "packed": 234895143, "token_witness": 254, "evaluated_windows": 14},
    {"ordinal": 73, "packed": 234895127, "token_witness": 256, "evaluated_windows": 14}
  ],
  "terminal": {
    "transactions": 73,
    "control": [
      1, 256, 0, 0, 1, 73, 1, 0, 182, 57, 239, 0,
      255, 7, 8, 1, 15, 2, 16, 4, 8, 3, 1000000, 450000,
      550000, 4, 0, 4, 28, 5, 23, 4, 0, 1, 5, 14,
      8, 4, 4, 7, 399, 472, 7, 2, 0, 0, 1, 3,
      2, 1, 9, 5, 14016, 0, 614400, 0, 2899968, 0, 608256, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 2, 4, 1, 15, 2560, 48,
      512, 10, 90, 6, 2, 5, 4, 7, 9, 5, 2, 3
    ]
  },
  "output": "/tmp/llaminar-recorded-stochastic-long-python-directional-unused.json"
}
)MTP_RECORD";

}
