/**
 * @file ModelKLDQFNOwner.h
 * @brief Keep the QFN observer's sole exported original model alive until final retirement.
 *
 * Export is a lifecycle operation, not a passive metadata read. The same contract
 * is borrowed throughout observation; runner and final model are retired before
 * the surrounding application's MPI session can unwind. No alternate reset or
 * physical-memory authority is introduced by this diagnostic frontend.
 */
#pragma once
#include "execution/runner/ModelContextRetirement.h"
#include "execution/config/RuntimeConfig.h"
#include "loaders/SharedPredictorGGUFSource.h"
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
namespace llaminar2::accuracy {
/** @brief Resolve the actual paired predictor's source policy from retained graph capacity, including disabled execution. */
inline SharedPredictorMaterialization qfnPredictorMaterialization(const MTPRuntimeConfig& config) {
    return retainsMTPGraphCapacity(config) ? SharedPredictorMaterialization::Active : SharedPredictorMaterialization::Inactive;
}
/** @brief Describe an original failure without replacing its exception when cleanup succeeds. */
inline std::string qfnObserverFailureText(const std::exception_ptr& failure) {
    try {std::rethrow_exception(failure);} catch(const std::exception& error) {return error.what();}
    catch(...) {return "non-standard QFN observer failure";}
}
/**
 * @brief Retire an observer's existing original runner and sole exported model in order.
 * @param runner Actual runner; shutdown and destruction precede final model retirement.
 * @param owner Same original export borrowed by observations; consumed after physical retirement.
 * @throws std::exception When either original retirement boundary fails.
 */
inline void retireQFNFinalOwner(std::unique_ptr<IOrchestrationRunner>& runner,
                               std::optional<ModelContextReuseContract>& owner) {
    if(runner){
        runner->shutdown();
        if(runner->isInitialized())throw std::runtime_error("QFN observer runner remains initialized after shutdown");
        runner.reset();
    }
    if(owner && owner->context)(void)retireExclusiveModelContextReuseContract(*owner);
    owner.reset();
}
/**
 * @brief Observe under one exported owner, then retire runner and original final model on every exit.
 * @param runner Actual initialized application runner; consumed before this call returns.
 * @param observe Callback borrowing only the same exported contract, with no escaping model/workspace aliases.
 * @throws Original observer failure after successful cleanup; combined primary/cleanup diagnostic if cleanup also fails.
 */
template<class Observe>
void observeQFNWithFinalOwner(std::unique_ptr<IOrchestrationRunner>& runner,Observe&& observe) {
    std::optional<ModelContextReuseContract> owner;
    std::exception_ptr primary;
    try {
        if(!runner)throw std::invalid_argument("QFN observer lost its original runner");
        owner=runner->modelContextReuseContract();
        if(!owner || !owner->context || !owner->reuse_authority || !owner->physical_memory_authority)
            throw std::invalid_argument("QFN observer needs the actual admitted model/prepared/PMA owner");
        std::forward<Observe>(observe)(std::as_const(*owner));
    }catch(...){
        primary=std::current_exception();
        // Native retirement can fail fatally after a partially submitted
        // request. Preserve the original observation error before cleanup.
        std::cerr << "QFN observation failed before final retirement: "
                  << qfnObserverFailureText(primary) << '\n';
    }
    try {
        retireQFNFinalOwner(runner,owner);
    }catch(...) {
        if(primary)throw std::runtime_error(qfnObserverFailureText(primary)+"; QFN final retirement failed: "+qfnObserverFailureText(std::current_exception()));
        throw;
    }
    if(primary)std::rethrow_exception(primary);
}
}
