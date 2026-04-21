/**
 * @file CliSpec.h
 * @brief Structured, declarative command-line flag specification.
 *
 * This header introduces a tiny, testable CLI framework used by the
 * OrchestrationConfigParser.  Instead of a 700-line sea of if/else string
 * matching, each flag is described by a `CliOption` value that carries its
 * names, category, human-readable description, optional value label, optional
 * list of accepted values, and a setter callback that knows how to apply the
 * parsed value to an OrchestrationConfig.
 *
 * The `CliSpec` collects these options, provides generic argument parsing
 * (handling `--flag value`, `--flag=value`, short aliases, and enum value
 * validation in one place), and auto-generates the `--help` text so it can
 * never drift from the actual wiring.
 *
 * Design notes:
 *   - Setters receive the raw string for value-carrying flags and an empty
 *     string for bare flags; they throw `std::invalid_argument` on bad input.
 *   - "Valid values" whitelisting happens before the setter runs, producing a
 *     single, uniform error message.
 *   - Options can be marked `not_yet_implemented`; they are still accepted so
 *     we don't break existing scripts, but they are rendered in a dedicated
 *     "Not yet implemented" section of the help.
 *   - The CliSpec itself is decoupled from OrchestrationConfig via a template
 *     so it can be unit-tested with a throwaway config struct.
 *
 * @author David Sanftenberg
 * @date April 2026
 */

#pragma once

#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace llaminar2
{

    /**
     * @brief Describes a single command-line option.
     *
     * The template parameter is the configuration struct the setter mutates;
     * for production use this is `OrchestrationConfig`, but tests can
     * instantiate with their own struct.
     */
    template <typename Config>
    struct CliOption
    {
        using Setter = std::function<void(Config &, const std::string &)>;

        /// Short form, e.g. "-m". Empty if the option has no short form.
        std::string short_name;
        /// Primary long form, e.g. "--model". Required.
        std::string long_name;
        /// Additional long-form aliases, e.g. {"--act-prec"}.
        std::vector<std::string> aliases;

        /// Section the flag appears under in --help output.
        std::string category = "Options";

        /// Placeholder for the value in --help, e.g. "<path>". Empty for bare flags.
        std::string value_label;

        /// One-line description for --help.
        std::string description;

        /// If non-empty, the parsed value must be a member; enforced before
        /// the setter is called. Produces a unified error message.
        std::vector<std::string> valid_values;

        /// Marks the option as accepted-but-inert; rendered in a "Not yet
        /// implemented" section of the help.
        bool not_yet_implemented = false;

        /// Callback that applies the parsed value to the config. For bare
        /// flags (empty value_label) this is invoked with an empty string.
        Setter setter;

        /// True for bare flags (no value expected).
        bool isFlag() const { return value_label.empty(); }

        /// Returns every accepted name (short + long + aliases), skipping empties.
        std::vector<std::string> allNames() const
        {
            std::vector<std::string> out;
            if (!short_name.empty())
                out.push_back(short_name);
            if (!long_name.empty())
                out.push_back(long_name);
            for (const auto &a : aliases)
                if (!a.empty())
                    out.push_back(a);
            return out;
        }

        /**
         * @brief Does the given raw argument refer to this option?
         *
         * Matches exact names, and for long-form flags also accepts the
         * `--flag=value` form.
         */
        bool matches(const std::string &arg) const
        {
            for (const auto &name : allNames())
            {
                if (arg == name)
                    return true;
                // --flag=value form only makes sense for value-carrying options
                // and only for long forms.
                if (!isFlag() && name.size() >= 2 && name[0] == '-' && name[1] == '-' &&
                    arg.size() > name.size() && arg.compare(0, name.size(), name) == 0 &&
                    arg[name.size()] == '=')
                {
                    return true;
                }
            }
            return false;
        }
    };

    /**
     * @brief Collection of CliOption entries plus generic parse/help logic.
     *
     * Typical usage:
     *   CliSpec<OrchestrationConfig> spec;
     *   spec.addCategory("Model Configuration");
     *   spec.add({...});
     *   spec.parse(args, config);
     *   std::cout << spec.getHelpText("Llaminar V2");
     */
    template <typename Config>
    class CliSpec
    {
    public:
        using Option = CliOption<Config>;

        /// Register a category in the order it should appear in --help.
        /// Options with a category that was never registered are appended
        /// after all registered categories, in first-seen order.
        CliSpec &addCategory(std::string name)
        {
            categories_.push_back(std::move(name));
            return *this;
        }

        /// Register an option. Duplicate names are flagged at parse time via
        /// the first match wins.
        CliSpec &add(Option opt)
        {
            options_.push_back(std::move(opt));
            return *this;
        }

        /**
         * @brief Walk `args` and mutate `config` accordingly.
         *
         * `args` should NOT include argv[0]. Unknown arguments throw
         * std::invalid_argument.
         *
         * @param allow_unknown  If true, unrecognised arguments are silently
         *                       skipped instead of throwing. Used during the
         *                       first-pass `--config` discovery.
         */
        void parse(const std::vector<std::string> &args,
                   Config &config,
                   bool allow_unknown = false) const;

        /**
         * @brief Auto-generate the --help text.
         *
         * @param header   Optional free-form text rendered before the option
         *                 list (e.g. usage line, examples).
         * @param footer   Optional free-form text rendered after the option
         *                 list (e.g. examples block).
         */
        std::string getHelpText(const std::string &header = "",
                                const std::string &footer = "") const;

        /// Accessor for inspection (used by tests).
        const std::vector<Option> &options() const { return options_; }

    private:
        std::vector<Option> options_;
        std::vector<std::string> categories_;

        /// Locate an option matching `arg`. Returns nullptr if none match.
        const Option *find(const std::string &arg) const;

        /// For a matched option, extract the value and advance `idx`.
        /// Returns empty for bare flags. Throws if a value is required but
        /// missing.
        std::string extractValue(const Option &opt,
                                 const std::vector<std::string> &args,
                                 size_t &idx) const;

        /// Validate an extracted value against opt.valid_values.
        void validateValue(const Option &opt, const std::string &value) const;

        /// Compute the rendered flag column ("-m, --model <path>") width.
        static std::string formatFlagColumn(const Option &opt);
    };

    // =========================================================================
    // Inline/template implementation
    // =========================================================================

    template <typename Config>
    const CliOption<Config> *CliSpec<Config>::find(const std::string &arg) const
    {
        for (const auto &opt : options_)
        {
            if (opt.matches(arg))
                return &opt;
        }
        return nullptr;
    }

    template <typename Config>
    std::string CliSpec<Config>::extractValue(const Option &opt,
                                              const std::vector<std::string> &args,
                                              size_t &idx) const
    {
        if (opt.isFlag())
            return "";

        const std::string &arg = args[idx];

        // --flag=value form
        auto eq = arg.find('=');
        if (eq != std::string::npos)
        {
            std::string value = arg.substr(eq + 1);
            if (value.empty())
                throw std::invalid_argument(opt.long_name + " requires a value");
            return value;
        }

        // --flag value form
        if (idx + 1 < args.size())
        {
            const std::string &next = args[idx + 1];
            // Accept arbitrary next token as the value, including values that
            // start with '-' (e.g. negative numbers for --seed -1). The
            // previous parser only accepted non-dash tokens; we preserve that
            // behaviour for bare flags but relax it here because numeric
            // options legitimately want negatives.
            ++idx;
            return next;
        }

        throw std::invalid_argument(opt.long_name + " requires a value");
    }

    template <typename Config>
    void CliSpec<Config>::validateValue(const Option &opt, const std::string &value) const
    {
        if (opt.valid_values.empty())
            return;
        for (const auto &v : opt.valid_values)
        {
            if (value == v)
                return;
        }
        std::string allowed;
        for (size_t i = 0; i < opt.valid_values.size(); ++i)
        {
            if (i)
                allowed += ", ";
            allowed += opt.valid_values[i];
        }
        throw std::invalid_argument("Invalid value for " + opt.long_name + ": '" +
                                    value + "' (valid: " + allowed + ")");
    }

    template <typename Config>
    void CliSpec<Config>::parse(const std::vector<std::string> &args,
                                Config &config,
                                bool allow_unknown) const
    {
        for (size_t i = 0; i < args.size(); ++i)
        {
            const std::string &arg = args[i];
            const Option *opt = find(arg);
            if (!opt)
            {
                if (allow_unknown)
                    continue;
                throw std::invalid_argument(
                    "Unknown argument: '" + arg + "'. Use --help to see available options.");
            }

            std::string value = extractValue(*opt, args, i);
            validateValue(*opt, value);
            if (opt->setter)
                opt->setter(config, value);
        }
    }

    template <typename Config>
    std::string CliSpec<Config>::formatFlagColumn(const Option &opt)
    {
        std::string s;
        if (!opt.short_name.empty())
        {
            s = opt.short_name;
            if (!opt.long_name.empty())
                s += ", ";
        }
        if (!opt.long_name.empty())
            s += opt.long_name;
        if (!opt.value_label.empty())
        {
            s += ' ';
            s += opt.value_label;
        }
        return s;
    }

    template <typename Config>
    std::string CliSpec<Config>::getHelpText(const std::string &header,
                                             const std::string &footer) const
    {
        std::string out;
        if (!header.empty())
        {
            out += header;
            if (header.back() != '\n')
                out += '\n';
        }

        // Determine the column width for the flag column. Cap it only when a
        // single option name would push the description absurdly far right.
        size_t flag_col_width = 0;
        for (const auto &opt : options_)
        {
            size_t w = formatFlagColumn(opt).size();
            if (w > flag_col_width)
                flag_col_width = w;
        }
        if (flag_col_width > 38)
            flag_col_width = 38;

        auto renderOption = [&](const Option &opt) {
            std::string flag_col = formatFlagColumn(opt);
            out += "  ";
            out += flag_col;
            if (flag_col.size() < flag_col_width)
                out.append(flag_col_width - flag_col.size(), ' ');
            else
                out += '\n' + std::string(2 + flag_col_width, ' ');
            out += "  ";
            out += opt.description;
            out += '\n';
        };

        // Build the category ordering: explicitly declared categories first,
        // then any additional categories seen in declaration order.
        std::vector<std::string> ordered = categories_;
        for (const auto &opt : options_)
        {
            bool seen = false;
            for (const auto &c : ordered)
                if (c == opt.category)
                {
                    seen = true;
                    break;
                }
            if (!seen)
                ordered.push_back(opt.category);
        }

        // Emit each category's active options.
        for (const auto &cat : ordered)
        {
            bool any = false;
            for (const auto &opt : options_)
            {
                if (opt.category != cat || opt.not_yet_implemented)
                    continue;
                if (!any)
                {
                    out += '\n';
                    out += cat;
                    out += ":\n";
                    any = true;
                }
                renderOption(opt);
            }
        }

        // NYI section.
        bool any_nyi = false;
        for (const auto &opt : options_)
        {
            if (!opt.not_yet_implemented)
                continue;
            if (!any_nyi)
            {
                out += "\nNot yet implemented (accepted but currently inert):\n";
                any_nyi = true;
            }
            renderOption(opt);
        }

        if (!footer.empty())
        {
            out += '\n';
            out += footer;
            if (footer.back() != '\n')
                out += '\n';
        }
        return out;
    }

    // =========================================================================
    // Setter helpers
    // =========================================================================
    //
    // Free functions that build common setter callbacks so option definitions
    // stay readable (`.setter = setters::assignBoolTrue(&Config::chat_mode)`).
    // These are header-inline so tests and the production parser share them.

    namespace setters
    {

        template <typename Config>
        inline typename CliOption<Config>::Setter assignBoolTrue(bool Config::*member)
        {
            return [member](Config &c, const std::string &) { c.*member = true; };
        }

        template <typename Config>
        inline typename CliOption<Config>::Setter assignBoolFalse(bool Config::*member)
        {
            return [member](Config &c, const std::string &) { c.*member = false; };
        }

        template <typename Config>
        inline typename CliOption<Config>::Setter incrementInt(int Config::*member,
                                                               int max_value = 3)
        {
            return [member, max_value](Config &c, const std::string &) {
                if (c.*member < max_value)
                    ++(c.*member);
            };
        }

        template <typename Config>
        inline typename CliOption<Config>::Setter assignIntLiteral(int Config::*member, int value)
        {
            return [member, value](Config &c, const std::string &) { c.*member = value; };
        }

        template <typename Config>
        inline typename CliOption<Config>::Setter parseInt(int Config::*member,
                                                           std::string option_name)
        {
            return [member, option_name](Config &c, const std::string &v) {
                try
                {
                    c.*member = std::stoi(v);
                }
                catch (const std::exception &)
                {
                    throw std::invalid_argument("Invalid value for " + option_name + ": '" + v + "'");
                }
            };
        }

        template <typename Config>
        inline typename CliOption<Config>::Setter parseFloat(float Config::*member,
                                                             std::string option_name)
        {
            return [member, option_name](Config &c, const std::string &v) {
                try
                {
                    c.*member = std::stof(v);
                }
                catch (const std::exception &)
                {
                    throw std::invalid_argument("Invalid value for " + option_name + ": '" + v + "'");
                }
            };
        }

        template <typename Config>
        inline typename CliOption<Config>::Setter assignString(std::string Config::*member)
        {
            return [member](Config &c, const std::string &v) { c.*member = v; };
        }

        /// Wraps a user-supplied lambda for cases that touch multiple fields
        /// or call into external parsers. Just sugar for readability.
        template <typename Config, typename Fn>
        inline typename CliOption<Config>::Setter custom(Fn fn)
        {
            return typename CliOption<Config>::Setter(std::move(fn));
        }

    } // namespace setters

} // namespace llaminar2
