/**
 * @file ARMCPUFeatures.h
 * @brief Immutable native AArch64 ISA capabilities, independent of kernel dispatch.
 *
 * Darwin sysctl and Linux auxiliary-vector observations authenticate optional
 * instructions. Compiler macros describe code generation, not the executing
 * CPU, and cannot substitute for these observations. The mandatory ASIMD ABI
 * is checked separately from DotProd, full FP16 arithmetic, BF16 and I8MM.
 * This device-free CPU authority owns no accelerator or mutable model state.
 */
#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <stdexcept>
#include <system_error>

#if !defined(__aarch64__)
#error "ARMCPUFeatures requires a native AArch64 compilation target"
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#else
#error "Native AArch64 capability observation is implemented for Darwin and Linux"
#endif

namespace llaminar2::cpu
{
    /** @brief Optional instruction families, not interchangeable kernel implementations. */
    enum class ARMCPUFeature : std::uint32_t
    {
        ASIMD = 1u << 0,
        DotProd = 1u << 1,
        FP16Arithmetic = 1u << 2,
        BF16Arithmetic = 1u << 3,
        IntegerMatMul = 1u << 4,
        FP16FML = 1u << 5,
    };

    /** @brief Process-wide immutable native ISA observation with checked feature queries. */
    class ARMCPUCapabilities final
    {
    public:
        /**
         * @brief Observe native CPU features once, without GPU/runtime initialization.
         * @return Immutable process-lifetime capabilities from the native OS ABI.
         * @throws std::system_error If a present OS observation cannot be read.
         * @throws std::runtime_error If the mandatory AArch64 ASIMD ABI is absent.
         */
        [[nodiscard]] static const ARMCPUCapabilities &native()
        {
            static const ARMCPUCapabilities capabilities = observe();
            return capabilities;
        }

        /**
         * @brief Test one explicitly named native instruction family.
         * @param feature Known ARM feature identifier.
         * @return Whether the executing CPU and OS advertise the feature.
         * @throws std::invalid_argument For an unknown feature enum value.
         */
        [[nodiscard]] bool supports(ARMCPUFeature feature) const
        {
            switch (feature)
            {
            case ARMCPUFeature::ASIMD:
            case ARMCPUFeature::DotProd:
            case ARMCPUFeature::FP16Arithmetic:
            case ARMCPUFeature::BF16Arithmetic:
            case ARMCPUFeature::IntegerMatMul:
            case ARMCPUFeature::FP16FML:
                return (features_ & static_cast<std::uint32_t>(feature)) != 0;
            }
            throw std::invalid_argument("Unknown native ARM CPU instruction feature");
        }

        /**
         * @brief Reject an instruction envelope absent from the actual machine.
         * @param feature Required instruction family.
         * @throws std::runtime_error If the native machine does not advertise it.
         * @throws std::invalid_argument For an unknown feature identifier.
         */
        void require(ARMCPUFeature feature) const
        {
            if (!supports(feature))
                throw std::runtime_error(std::string("Required native ARM CPU feature is unavailable: ") + name(feature));
        }

        /**
         * @brief Optional features the including translation unit was compiled to use.
         * @return ASIMD plus DotProd and I8MM when the compiler envelope enables them.
         * @details Compiler macros never establish availability; they name what
         * the machine must provide before code from this envelope may run.
         */
        [[nodiscard]] static constexpr std::uint32_t compiledEnvelopeFeatures() noexcept
        {
            std::uint32_t features = static_cast<std::uint32_t>(ARMCPUFeature::ASIMD);
#if defined(__ARM_FEATURE_DOTPROD)
            features |= static_cast<std::uint32_t>(ARMCPUFeature::DotProd);
#endif
#if defined(__ARM_FEATURE_FP16FML)
            features |= static_cast<std::uint32_t>(ARMCPUFeature::FP16FML);
#endif
#if defined(__ARM_FEATURE_MATMUL_INT8)
            features |= static_cast<std::uint32_t>(ARMCPUFeature::IntegerMatMul);
#endif
            return features;
        }

        /**
         * @brief Admit one complete compiler envelope against observed features.
         * @param observed Native feature bits.
         * @param envelope Features the executing code was compiled to use.
         * @throws std::runtime_error Naming the first envelope feature the machine lacks.
         * @details The single admission rule shared by every native leaf entry,
         * device-free testable with any observed feature set.
         */
        static void requireEnvelope(std::uint32_t observed, std::uint32_t envelope)
        {
            for (const auto feature : {ARMCPUFeature::ASIMD, ARMCPUFeature::DotProd, ARMCPUFeature::FP16Arithmetic,
                                       ARMCPUFeature::BF16Arithmetic, ARMCPUFeature::IntegerMatMul,
                                       ARMCPUFeature::FP16FML})
            {
                const auto bit = static_cast<std::uint32_t>(feature);
                if ((envelope & bit) != 0 && (observed & bit) == 0)
                    throw std::runtime_error(std::string("Required native ARM CPU feature is unavailable: ") + name(feature));
            }
        }

        /**
         * @brief Admit the including translation unit's complete envelope on this machine.
         * @throws std::runtime_error If any compiled optional feature is not advertised.
         * @details Auto-generated code may use any instruction of the envelope,
         * so admission covers all of it, not only the explicitly selected leaf.
         */
        void requireCompiledEnvelope() const { requireEnvelope(features_, compiledEnvelopeFeatures()); }

        /**
         * @brief Return one stable diagnostic spelling without implying availability.
         * @param feature Known ARM instruction feature.
         * @return Static feature name.
         * @throws std::invalid_argument For an unknown feature identifier.
         */
        [[nodiscard]] static constexpr const char *name(ARMCPUFeature feature)
        {
            switch (feature)
            {
            case ARMCPUFeature::ASIMD: return "ASIMD";
            case ARMCPUFeature::DotProd: return "DotProd";
            case ARMCPUFeature::FP16Arithmetic: return "FP16 arithmetic";
            case ARMCPUFeature::BF16Arithmetic: return "BF16 arithmetic";
            case ARMCPUFeature::IntegerMatMul: return "I8MM";
            case ARMCPUFeature::FP16FML: return "FP16 FML";
            }
            throw std::invalid_argument("Unknown native ARM CPU instruction feature");
        }

    private:
        /** @brief Seal the complete feature word; only the native observer constructs it. */
        explicit constexpr ARMCPUCapabilities(std::uint32_t features) : features_(features) {}

#if defined(__APPLE__)
        /**
         * @brief Read one Darwin optional-feature word without guessing from the compiler.
         * @param name Canonical hw.optional.arm feature key.
         * @return True only when the native ABI publishes one as the feature value.
         * @throws std::system_error For unreadable observations other than an absent optional key.
         * @throws std::runtime_error For an unexpected native ABI word length/value.
         */
        [[nodiscard]] static bool optionalFeature(const char *name)
        {
            int value = 0;
            std::size_t size = sizeof(value);
            if (sysctlbyname(name, &value, &size, nullptr, 0) != 0)
            {
                const int error = errno;
                // An older OS does not advertise an unknown optional feature.
                // It cannot authorize that instruction, but ASIMD remains an
                // independent mandatory native implementation.
                if (error == ENOENT)
                    return false;
                throw std::system_error(error, std::generic_category(), name);
            }
            if (size != sizeof(value) || (value != 0 && value != 1))
                throw std::runtime_error("Unexpected Darwin ARM feature observation ABI");
            return value == 1;
        }
#endif

        /**
         * @brief Construct capabilities solely from the executing OS instruction ABI.
         * @return Checked native feature set.
         * @throws std::system_error For failed OS feature observations.
         * @throws std::runtime_error If mandatory ASIMD is not available.
         */
        [[nodiscard]] static ARMCPUCapabilities observe()
        {
            std::uint32_t features = 0;
            const auto add = [&](ARMCPUFeature feature, bool present)
            {
                if (present)
                    features |= static_cast<std::uint32_t>(feature);
            };
#if defined(__APPLE__)
            // arm64 Darwin's userspace ABI guarantees FP/ASIMD state support;
            // optional arithmetic extensions still require their own keys.
            add(ARMCPUFeature::ASIMD, true);
            add(ARMCPUFeature::DotProd, optionalFeature("hw.optional.arm.FEAT_DotProd"));
            add(ARMCPUFeature::FP16Arithmetic, optionalFeature("hw.optional.arm.FEAT_FP16"));
            add(ARMCPUFeature::BF16Arithmetic, optionalFeature("hw.optional.arm.FEAT_BF16"));
            add(ARMCPUFeature::IntegerMatMul, optionalFeature("hw.optional.arm.FEAT_I8MM"));
            add(ARMCPUFeature::FP16FML, optionalFeature("hw.optional.arm.FEAT_FHM"));
#elif defined(__linux__)
            errno = 0;
            const unsigned long hwcap = getauxval(AT_HWCAP);
            if (errno != 0)
                throw std::system_error(errno, std::generic_category(), "AT_HWCAP");
            add(ARMCPUFeature::ASIMD, (hwcap & HWCAP_ASIMD) != 0);
#ifdef HWCAP_ASIMDDP
            add(ARMCPUFeature::DotProd, (hwcap & HWCAP_ASIMDDP) != 0);
#endif
#if defined(HWCAP_FPHP) && defined(HWCAP_ASIMDHP)
            add(ARMCPUFeature::FP16Arithmetic,
                (hwcap & (HWCAP_FPHP | HWCAP_ASIMDHP)) == (HWCAP_FPHP | HWCAP_ASIMDHP));
#endif
#ifdef HWCAP_ASIMDFHM
            add(ARMCPUFeature::FP16FML, (hwcap & HWCAP_ASIMDFHM) != 0);
#endif
#if defined(HWCAP2_BF16) || defined(HWCAP2_I8MM)
            errno = 0;
            const unsigned long hwcap2 = getauxval(AT_HWCAP2);
            if (errno != 0)
                throw std::system_error(errno, std::generic_category(), "AT_HWCAP2");
#ifdef HWCAP2_BF16
            add(ARMCPUFeature::BF16Arithmetic, (hwcap2 & HWCAP2_BF16) != 0);
#endif
#ifdef HWCAP2_I8MM
            add(ARMCPUFeature::IntegerMatMul, (hwcap2 & HWCAP2_I8MM) != 0);
#endif
#endif
#endif
            ARMCPUCapabilities result(features);
            result.require(ARMCPUFeature::ASIMD);
            return result;
        }

        std::uint32_t features_ = 0; ///< Native observation bits, immutable after construction.
    };
} // namespace llaminar2::cpu
