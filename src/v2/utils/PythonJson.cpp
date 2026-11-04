/**
 * @file PythonJson.cpp
 * @brief Python json.dumps(ensure_ascii=False) spelling for order-preserving JSON.
 *
 * Float spelling follows CPython's float_repr_style "short": the shortest
 * decimal digits that round-trip (obtained here from std::to_chars), written in
 * fixed notation when the decimal exponent e of d.ddd x 10^e satisfies
 * -4 <= e < 16 and in scientific notation otherwise. Fixed notation always has
 * a fractional part ("100.0"); scientific notation omits a lone ".0"
 * ("1e-07") and uses a signed exponent of at least two digits.
 */
#include "PythonJson.h"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace llaminar2
{
    namespace
    {
        void appendString(std::string &out, const std::string &text)
        {
            out.push_back('"');
            for (const unsigned char c : text)
            {
                switch (c)
                {
                case '"': out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\n': out += "\\n"; break;
                case '\r': out += "\\r"; break;
                case '\t': out += "\\t"; break;
                case '\b': out += "\\b"; break;
                case '\f': out += "\\f"; break;
                default:
                    if (c < 0x20)
                    {
                        char escaped[8];
                        std::snprintf(escaped, sizeof(escaped), "\\u%04x", c);
                        out += escaped;
                    }
                    else
                    {
                        // ensure_ascii=False: multi-byte UTF-8 passes through verbatim.
                        out.push_back(static_cast<char>(c));
                    }
                }
            }
            out.push_back('"');
        }

        void append(std::string &out, const nlohmann::ordered_json &value)
        {
            switch (value.type())
            {
            case nlohmann::ordered_json::value_t::null: out += "null"; break;
            case nlohmann::ordered_json::value_t::boolean: out += value.get<bool>() ? "true" : "false"; break;
            case nlohmann::ordered_json::value_t::number_integer: out += std::to_string(value.get<std::int64_t>()); break;
            case nlohmann::ordered_json::value_t::number_unsigned: out += std::to_string(value.get<std::uint64_t>()); break;
            case nlohmann::ordered_json::value_t::number_float: out += pythonFloatRepr(value.get<double>()); break;
            case nlohmann::ordered_json::value_t::string: appendString(out, value.get_ref<const std::string &>()); break;
            case nlohmann::ordered_json::value_t::array:
            {
                out.push_back('[');
                bool first = true;
                for (const auto &item : value)
                {
                    if (!first) out += ", ";
                    first = false;
                    append(out, item);
                }
                out.push_back(']');
                break;
            }
            case nlohmann::ordered_json::value_t::object:
            {
                out.push_back('{');
                bool first = true;
                for (const auto &[key, item] : value.items())
                {
                    if (!first) out += ", ";
                    first = false;
                    appendString(out, key);
                    out += ": ";
                    append(out, item);
                }
                out.push_back('}');
                break;
            }
            case nlohmann::ordered_json::value_t::binary:
            case nlohmann::ordered_json::value_t::discarded:
                throw std::invalid_argument("pythonJsonDumps: binary/discarded values have no JSON spelling");
            }
        }
    } // namespace

    std::string pythonFloatRepr(double value)
    {
        if (!std::isfinite(value))
            throw std::invalid_argument("pythonFloatRepr: non-finite floats have no JSON spelling");
        if (value == 0.0) return std::signbit(value) ? "-0.0" : "0.0";

        // Shortest round-trip digits in scientific form: [-]d[.ddd]e(+|-)xx.
        char buffer[64];
        const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value, std::chars_format::scientific);
        if (result.ec != std::errc())
            throw std::logic_error("pythonFloatRepr: to_chars failed");
        const std::string scientific(buffer, result.ptr);
        const auto e_pos = scientific.find('e');
        const bool negative = scientific[0] == '-';
        std::string digits;
        for (std::size_t i = negative ? 1 : 0; i < e_pos; ++i)
            if (scientific[i] != '.') digits.push_back(scientific[i]);
        const int exponent = std::stoi(scientific.substr(e_pos + 1));

        std::string out = negative ? "-" : "";
        if (exponent < -4 || exponent >= 16)
        {
            out += digits[0];
            if (digits.size() > 1)
            {
                out.push_back('.');
                out.append(digits, 1, std::string::npos);
            }
            char exp_text[8];
            std::snprintf(exp_text, sizeof(exp_text), "e%c%02d", exponent < 0 ? '-' : '+', std::abs(exponent));
            out += exp_text;
            return out;
        }
        if (exponent < 0)
        {
            out += "0.";
            out.append(static_cast<std::size_t>(-exponent - 1), '0');
            out += digits;
            return out;
        }
        const auto integer_digits = static_cast<std::size_t>(exponent) + 1;
        if (digits.size() <= integer_digits)
        {
            out += digits;
            out.append(integer_digits - digits.size(), '0');
            out += ".0";
        }
        else
        {
            out.append(digits, 0, integer_digits);
            out.push_back('.');
            out.append(digits, integer_digits, std::string::npos);
        }
        return out;
    }

    std::string pythonJsonDumps(const nlohmann::ordered_json &value)
    {
        std::string out;
        append(out, value);
        return out;
    }
} // namespace llaminar2
