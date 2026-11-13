/**
 * @file NodeLocalRouteExchangeReplayAccess.h
 * @brief Test-only raw storage boundary for isolated single-sided route kernel controls.
 *
 * These controls intentionally do not represent an admitted model or a distributed
 * source. Production cannot construct the raw owner: its only public constructor
 * requires the immutable storage declaration, original PMA and original inventory.
 */
#pragma once
#include "execution/moe/MoEOverlayNodeLocalRouteExchange.h"
namespace llaminar2::test {
/** @brief Friend access kept exclusively under the test include root. */
class NodeLocalRouteExchangeReplayAccess final {
public:
    /** @return Raw isolated component storage; never valid as a production source owner. */
    static std::unique_ptr<MoEOverlayNodeLocalRouteExchange> createUnique(MoEOverlayNodeLocalRouteExchange::Config config) {
        return std::unique_ptr<MoEOverlayNodeLocalRouteExchange>(new MoEOverlayNodeLocalRouteExchange(std::move(config)));
    }
    /** @return Shared raw isolated component storage for declarative kernel controls. */
    static std::shared_ptr<MoEOverlayNodeLocalRouteExchange> create(MoEOverlayNodeLocalRouteExchange::Config config) {
        return std::shared_ptr<MoEOverlayNodeLocalRouteExchange>(new MoEOverlayNodeLocalRouteExchange(std::move(config)));
    }
};
}
