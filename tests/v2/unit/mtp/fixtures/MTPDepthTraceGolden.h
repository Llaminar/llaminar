/**
 * @file MTPDepthTraceGolden.h
 * @brief Independently authored v3 publication chronology, not controller-generated expectations.
 *
 * Initial incoming debt0. O/L/P per transaction are authored source counts;
 * Y=O-L totals34 and P totals33 with final raw outgoing debt1. The clipped
 * second transaction contributes no eligible sample. Known windows hold
 * (R4/A8/F1/Z0), promote (R4/A12/F4/Z0), demote (R4/A0/F0/Z4).
 * Expected counters E3/U2/P1/D1 remain independent of the implementation.
 * CPU rollback is false as at the actual host caller; native rejection flags
 * include partial outcomes. The fixture is NEW authored v3 data, never an
 * upgraded or restamped genuine v2 capture.
 */
#pragma once
#include "kernels/common/MTPDepthTransactionTrace.h"
#include <array>
namespace llaminar2::test::mtp_trace_replay::golden
{
using Record = sampling_math::MTPDepthTransactionTraceRecord;
inline constexpr std::array<Record,13> kResident{{
    {1174410019u, 4, 0}, // tx1: d3, A2, O4/L0/P3, Y4, post3
    {1711289091u, 5, 0}, // tx2: d3, A0, O2/L1/P1, Y1, post3, clipped
    {637535027u, 8, 0}, // tx3: d3, A3, O4/L1/P4, Y3, post3
    {1174410003u, 11, 0}, // tx4: d3, A1, O3/L0/P2, Y3, post3
    {1711280931u, 14, 1}, // tx5: d3, A2, O4/L1/P3, Y3, post3
    {637535027u, 17, 1}, // tx6: d3, A3, O4/L1/P4, Y3, post3
    {100664115u, 21, 1}, // tx7: d3, A3, O4/L0/P4, Y4, post3
    {100664115u, 25, 1}, // tx8: d3, A3, O4/L0/P4, Y4, post3
    {134218803u, 29, 2}, // tx9: d3, A3, O4/L0/P4, Y4, post4
    {1207964676u, 31, 2}, // tx10: d4, A0, O2/L0/P1, Y2, post4
    {1744835588u, 32, 2}, // tx11: d4, A0, O2/L1/P1, Y1, post4
    {1744835588u, 33, 2}, // tx12: d4, A0, O2/L1/P1, Y1, post4
    {1711280900u, 34, 3}, // tx13: d4, A0, O2/L1/P1, Y1, post3
}};
inline constexpr std::array<Record,13> kHost{{
    {1192805155u, 4, 0}, // tx1: d3, A2, O4/L0/P3, Y4, post3
    {1728635651u, 1, 0}, // tx2: d3, A0, O2/L1/P1, Y1, post3, clipped
    {655934259u, 3, 0}, // tx3: d3, A3, O4/L1/P4, Y3, post3
    {1192805139u, 3, 0}, // tx4: d3, A1, O3/L0/P2, Y3, post3
    {1729676067u, 3, 1}, // tx5: d3, A2, O4/L1/P3, Y3, post3
    {655934259u, 3, 1}, // tx6: d3, A3, O4/L1/P4, Y3, post3
    {119063347u, 4, 1}, // tx7: d3, A3, O4/L0/P4, Y4, post3
    {119063347u, 4, 1}, // tx8: d3, A3, O4/L0/P4, Y4, post3
    {152618035u, 4, 2}, // tx9: d3, A3, O4/L0/P4, Y4, post4
    {1226900484u, 2, 2}, // tx10: d4, A0, O2/L0/P1, Y2, post4
    {1763771396u, 1, 2}, // tx11: d4, A0, O2/L1/P1, Y1, post4
    {1763771396u, 1, 2}, // tx12: d4, A0, O2/L1/P1, Y1, post4
    {1730216708u, 1, 3}, // tx13: d4, A0, O2/L1/P1, Y1, post3
}};
inline constexpr std::array<Record,2> kStoppedResident{{
    {1174410003u, 3, 0}, // actual rejection O3/L0/P2/Y3; one eligible partial window row
    {3858760451u, 4, 0}, // stopped correction O2/L1/P1/Y1, raw D1, rollbackfalse
}};
inline constexpr std::array<Record,2> kStoppedHost{{
    {1192805139u, 3, 0}, // actual rejection O3/L0/P2/Y3; one eligible partial window row
    {3877159683u, 1, 0}, // stopped correction O2/L1/P1/Y1, raw D1, rollbackfalse
}};
}
