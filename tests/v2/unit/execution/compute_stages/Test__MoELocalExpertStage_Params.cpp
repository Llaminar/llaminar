#include "execution/compute_stages/stages/MoELocalExpertStage.h"

#include <gtest/gtest.h>

#include <type_traits>
#include <utility>

namespace llaminar2::test
{
    namespace
    {
        template <typename, typename = void>
        struct has_runtime : std::false_type
        {
        };
        template <typename T>
        struct has_runtime<T, std::void_t<decltype(std::declval<T &>().runtime)>> : std::true_type
        {
        };

        template <typename, typename = void>
        struct has_domain_runtime : std::false_type
        {
        };
        template <typename T>
        struct has_domain_runtime<T, std::void_t<decltype(std::declval<T &>().domain_runtime)>> : std::true_type
        {
        };

        template <typename, typename = void>
        struct has_runtime_service : std::false_type
        {
        };
        template <typename T>
        struct has_runtime_service<T, std::void_t<decltype(std::declval<T &>().runtime_service)>> : std::true_type
        {
        };

        template <typename, typename = void>
        struct has_runner : std::false_type
        {
        };
        template <typename T>
        struct has_runner<T, std::void_t<decltype(std::declval<T &>().runner)>> : std::true_type
        {
        };

        template <typename, typename = void>
        struct has_role_runner : std::false_type
        {
        };
        template <typename T>
        struct has_role_runner<T, std::void_t<decltype(std::declval<T &>().role_runner)>> : std::true_type
        {
        };

        template <typename, typename = void>
        struct has_participants : std::false_type
        {
        };
        template <typename T>
        struct has_participants<T, std::void_t<decltype(std::declval<T &>().participants)>> : std::true_type
        {
        };

        template <typename, typename = void>
        struct has_prepared_participants : std::false_type
        {
        };
        template <typename T>
        struct has_prepared_participants<T, std::void_t<decltype(std::declval<T &>().prepared_participants)>> : std::true_type
        {
        };

        template <typename, typename = void>
        struct has_peer_participants : std::false_type
        {
        };
        template <typename T>
        struct has_peer_participants<T, std::void_t<decltype(std::declval<T &>().peer_participants)>> : std::true_type
        {
        };

    } // namespace

    TEST(Test__MoELocalExpertStage_Params, ConstructibleAndNoRuntimeOrPeerParticipantFields)
    {
        using Params = MoELocalExpertStage::Params;

        static_assert(StageParamsRequired<Params>);
        EXPECT_TRUE(std::is_default_constructible_v<Params>);
        EXPECT_TRUE((std::is_constructible_v<MoELocalExpertStage, Params>));

        EXPECT_FALSE(has_runtime<Params>::value);
        EXPECT_FALSE(has_domain_runtime<Params>::value);
        EXPECT_FALSE(has_runtime_service<Params>::value);
        EXPECT_FALSE(has_runner<Params>::value);
        EXPECT_FALSE(has_role_runner<Params>::value);
        EXPECT_FALSE(has_participants<Params>::value);
        EXPECT_FALSE(has_prepared_participants<Params>::value);
        EXPECT_FALSE(has_peer_participants<Params>::value);
    }

} // namespace llaminar2::test