/**
 * @file rccl_selector_probe.cpp
 * @brief Execute original native selector parser and enum taxonomy without GPU initialization.
 *
 * The builder compiles the real dependency translation units independently.
 * Expected matrices are keyed by named native enum constants, not string-table
 * indices, so the historical misordered array fails this functional oracle.
 */
#include "core.h"
#include "device.h"
#include "collectives.h"
#include <array>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <iterator>
#include <utility>
#include <cstdarg>
#include "debug.h"
extern ncclResult_t parseList(const char*, const char*[], int, const char*[], int, int*);
namespace {
unsigned probeWarningCalls = 0;
unsigned probeOtherLogCalls = 0;
}
/**
 * @brief Probe-local host logging boundary: retain actual parser diagnostics without HIP queries.
 * @param level Original native logger level.
 * @param flags Original logger subsystem mask (not interpreted by this CPU-only probe).
 * @param filefunc Original source/function diagnostic label.
 * @param line Original source line.
 * @param fmt Original native printf format and subsequent variadic arguments.
 * @note Returns no value; diagnostics are emitted only to the probe stderr.
 *
 * This exact strong native signature is localized in the probe executable by
 * its existing version script, so native objects resolve their calls here while
 * the selected DSO keeps its own production logger unchanged. No debug.cc is
 * compiled into this probe. The old parser must enter this seam exactly four
 * times; zero warnings is not accepted as evidence of a disconnected logger.
 */
void ncclDebugLog(ncclDebugLogLevel level, unsigned long flags,
                  const char *filefunc, int line, const char *fmt, ...)
{
    (void)flags;
    if (level == NCCL_LOG_WARN) ++probeWarningCalls;
    else ++probeOtherLogCalls;
    char message[512];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(message, sizeof(message), fmt, args);
    va_end(args);
    std::fprintf(stderr, "selector-probe %s:%d: %s\n", filefunc, line, message);
}
namespace {
/** @brief Independent public collective names paired with actual native enum authority. */
constexpr std::array<std::pair<ncclFunc_t, const char *>, 8> functions{{
    {ncclFuncBroadcast, "Broadcast"},
    {ncclFuncReduce, "Reduce"},
    {ncclFuncAllGather, "AllGather"},
    {ncclFuncReduceScatter, "ReduceScatter"},
    {ncclFuncAllReduce, "AllReduce"},
    {ncclFuncSendRecv, "SendRecv"},
    {ncclFuncSend, "Send"},
    {ncclFuncRecv, "Recv"}
}};

/** @brief Accumulate complete coverage even when historical code selects the wrong row. */
struct Oracle {
    int checks = 0, failures = 0;

    /**
     * @brief Count one independently expected mapping or parser matrix cell.
     * @param ok Whether the actual native result satisfies the independent oracle.
     * @return No value; increments the retained assertion and failure counters.
     */
    void check(bool ok)
    {
        ++checks;
        failures += !ok;
    }

    /**
     * @brief Check every native row and protocol/algorithm cell after actual parsing.
     * @param got Complete flattened matrix returned by the original native parser.
     * @param expected Independent enum-keyed expected matrix with the same dimensions.
     * @return No value; records every size and element assertion.
     */
    void matrix(const std::vector<int> &got, const std::vector<int> &expected)
    {
        check(got.size() == expected.size());
        for (size_t i = 0; i < got.size(); ++i)
        {
            check(got[i] == expected[i]);
        }
    }
};

/**
 * @brief Exercise all tunable prefixes, compound expressions, exclusions and case folding.
 * @param o Collector retaining every result even when the original table misroutes it.
 * @param choices Original native protocol or algorithm name array.
 * @param count Number of choices in one native matrix row.
 * @param selected Two independent native enum indices used by compound expressions.
 * @return No value; appends the complete parser-matrix assertions to the oracle.
 */
void selectors(Oracle &o, const char **choices, int count, const std::vector<int> &selected)
{
    for (int target = 0; target < NCCL_NUM_FUNCTIONS; ++target)
    {
        const char *name = nullptr;
        for (auto [id, text] : functions)
        {
            if (int(id) == target)
            {
                name = text;
            }
        }
        std::vector<int> got(NCCL_NUM_FUNCTIONS * count, 7), expected = got;
        std::string text = std::string(name) + ":" + choices[selected[0]];
        for (int c = 0; c < count; ++c)
        {
            expected[target * count + c] = (c == selected[0]);
        }
        o.check(parseList(text.c_str(), std::data(ncclFuncStr), NCCL_NUM_FUNCTIONS,
                          choices, count, got.data()) == ncclSuccess);
        o.matrix(got, expected);
    }

    // The global baseline and each named override affect independent enum rows.
    // Untouched functions must retain their original global settings.
    std::string expr = std::string(choices[selected[0]]) + ";aLlGaThEr:" + choices[selected[1]] +
        ";ReduceScatter:^" + choices[selected[0]] + ";AllReduce:" + choices[selected[0]] +
        "," + choices[selected[1]];
    std::vector<int> got(NCCL_NUM_FUNCTIONS * count, 7), expected(got.size());
    for (int f = 0; f < NCCL_NUM_FUNCTIONS; ++f)
    {
        for (int c = 0; c < count; ++c)
        {
            int v = c == selected[0];
            if (f == ncclFuncAllGather)
            {
                v = c == selected[1];
            }
            if (f == ncclFuncReduceScatter)
            {
                v = c != selected[0];
            }
            if (f == ncclFuncAllReduce)
            {
                v = c == selected[0] || c == selected[1];
            }
            expected[f * count + c] = v;
        }
    }
    o.check(parseList(expr.c_str(), std::data(ncclFuncStr), NCCL_NUM_FUNCTIONS,
                      choices, count, got.data()) == ncclSuccess);
    o.matrix(got, expected);
}
} // namespace

/**
 * @brief Execute actual native parser/table/string authority without communicator or device APIs.
 * @return Zero for complete native success, one when the original table fails the oracle.
 */
int main()
{
    Oracle o;
    for (auto [id, name] : functions)
    {
        o.check(std::strcmp(ncclFuncToString(id), name) == 0);
    }
    for (size_t index = 0; index < std::size(ncclFuncStr); ++index)
    {
        o.check(std::strcmp(ncclFuncStr[index],
                            ncclFuncToString(static_cast<ncclFunc_t>(index))) == 0);
    }
    o.check(std::strcmp(ncclFuncToString(static_cast<ncclFunc_t>(-1)), "Invalid") == 0);
    o.check(std::strcmp(ncclFuncToString(static_cast<ncclFunc_t>(ncclNumFuncs)), "Invalid") == 0);
    selectors(o, ncclProtoStr, NCCL_NUM_PROTOCOLS, {NCCL_PROTO_LL, NCCL_PROTO_SIMPLE});
    selectors(o, ncclAlgoStr, NCCL_NUM_ALGORITHMS, {NCCL_ALGO_RING, NCCL_ALGO_TREE});
    std::printf("{\"checks\":%d,\"failures\":%d,\"logging_warning_calls\":%u,\"logging_other_calls\":%u}\n",
                o.checks, o.failures, probeWarningCalls, probeOtherLogCalls);
    return o.failures ? 1 : 0;
}
