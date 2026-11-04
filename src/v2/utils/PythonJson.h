/**
 * @file PythonJson.h
 * @brief Byte-exact serialization of JSON values as Python's json.dumps writes them.
 *
 * Model prompt formats defined by Python reference encoders (for example the
 * DeepSeek-V4 encoder used by vLLM) embed tool schemas and tool-call arguments
 * through @c json.dumps(value, ensure_ascii=False). Its spelling differs from
 * nlohmann's compact dump in three ways that change prompt tokens:
 *  - separators are ", " and ": ";
 *  - floats use Python's shortest round-trip repr (always a ".0" or exponent,
 *    scientific notation outside 1e-4 <= |x| < 1e16, two-digit exponents);
 *  - only '"', '\\' and control characters are escaped (lower-case \\u00xx),
 *    every other code point is written as raw UTF-8.
 * Object member order is the caller's: pass nlohmann::ordered_json parsed from
 * the client's bytes so insertion order survives, as it does in Python dicts.
 */
#pragma once

#include <nlohmann/json.hpp>
#include <string>

namespace llaminar2
{
    /**
     * @brief Serialize @p value exactly as Python's json.dumps(value, ensure_ascii=False).
     * @param value Order-preserving JSON value.
     * @return UTF-8 text identical to the Python reference serialization.
     * @throws std::invalid_argument For non-finite floats, which Python would spell
     *         as the non-JSON tokens NaN/Infinity and no client request can carry.
     */
    std::string pythonJsonDumps(const nlohmann::ordered_json &value);

    /**
     * @brief Python repr() of a finite double (shortest round-trip spelling).
     * @param value Finite double.
     * @return For example "0.5", "100.0", "1e-07", "1.5e+16".
     */
    std::string pythonFloatRepr(double value);
} // namespace llaminar2
