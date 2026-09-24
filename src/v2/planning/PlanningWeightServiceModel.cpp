/**
 * @file PlanningWeightServiceModel.cpp
 * @brief Source-family sampling and compute/streaming composition without model warmups.
 *
 * Selection is bounded by executed format families, not layer count, expert count
 * or candidate count. Timed cache-resident work never supplies DRAM bandwidth.
 * Predictions retain the observed operation family and extrapolate logical
 * shape explicitly; no vendor peak table, inferred topology, synthetic speed
 * bonus, live placement state or parallel physical-memory ledger is introduced.
 */
#include "PlanningWeightServiceModel.h"
#include "AutomaticPlanningStartup.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
#include <type_traits>

namespace llaminar2
{
    namespace
    {
        /** @return Strictly finite positive prediction; overflow cannot silently win/lose a search. */
        double positive(double seconds)
        {
            if (!std::isfinite(seconds) || seconds <= 0)
                throw std::overflow_error("Weight service prediction is not finite and positive");
            return seconds;
        }

        /** @brief Retain exact shape frequency and stable source name, without holding payloads. */
        struct Representative
        {
            size_t count = 0;
            const TensorSizeInfo *source = nullptr;
        };
    }

    std::vector<PlanningWeightServiceSample> PlanningWeightServiceModel::samples(const PlanningModelMetadata &model)
    {
        const auto &profile = model.memoryProfile();
        using Shape = std::pair<size_t, size_t>;
        std::map<std::string, std::map<Shape, Representative>> ordinary;
        std::map<int, std::array<const TensorSizeInfo *, 3>> layers;
        for (const auto &tensor : profile.tensors)
        {
            if (tensor.layer_index >= model.mainLayerCount()) continue;
            const auto role = inferWeightRole(tensor.name);
            const auto kind = planningOrdinaryWeightKind(role);
            if (isRoutedExpertRole(role))
            {
                auto &triplet = layers[tensor.layer_index];
                const size_t slot = role == WeightRole::MoEExpertGate ? 0 : role == WeightRole::MoEExpertUp ? 1 : 2;
                if (triplet[slot]) throw std::invalid_argument("Duplicate expert source in planning samples");
                triplet[slot] = &tensor;
            }
            // An embedding is also the source for a tied head. Including it
            // guarantees that the head's format is observed, without charging
            // an embedding lookup as a vocabulary-sized matrix multiply.
            else if (kind == PlanningOrdinaryWeightKind::Projection || kind == PlanningOrdinaryWeightKind::Router ||
                kind == PlanningOrdinaryWeightKind::Embedding)
            {
                if (!tensor.K || !tensor.elements || tensor.elements % tensor.K)
                    throw std::invalid_argument("Planning projection source has no matrix geometry");
                const auto execution_format = PreparedWeightRepresentationContract::resolve(role, tensor.quant_type) ==
                    ModelPreparedWeightRepresentation::FP32 ? std::string("F32") : tensor.quant_type;
                auto &entry = ordinary[execution_format][{tensor.elements / tensor.K, tensor.K}];
                ++entry.count;
                if (!entry.source || tensor.name < entry.source->name) entry.source = &tensor;
            }
        }
        std::vector<PlanningWeightServiceSample> result;
        for (const auto &[format, shapes] : ordinary)
        {
            // Weight frequency by arithmetic volume. Repeated tiny K/V or
            // router projections must not displace the substantial repeated
            // FFN work just because their source directory has more entries.
            const auto best = std::max_element(shapes.begin(), shapes.end(), [](const auto &a, const auto &b) {
                const long double left = static_cast<long double>(a.second.count) * a.first.first * a.first.second;
                const long double right = static_cast<long double>(b.second.count) * b.first.first * b.first.second;
                if (left != right) return left < right;
                return a.first > b.first;
            });
            const auto &tensor = *best->second.source;
            const size_t n = tensor.elements / tensor.K;
            constexpr size_t kMaximumSourceRows = 1024;
            if (n <= kMaximumSourceRows) result.emplace_back(PlanningModelSampleRequest{tensor.name, PlanningWholeMatrix{}});
            else result.emplace_back(PlanningModelSampleRequest{tensor.name, PlanningMatrixRows{0, kMaximumSourceRows}});
        }

        using Formats = std::array<std::string, 3>;
        std::map<Formats, std::map<Shape, std::vector<int>>> experts;
        for (const auto &[layer, triplet] : layers)
        {
            if (layer < 0 || !triplet[0] || !triplet[1] || !triplet[2] || profile.expert_count <= 0)
                throw std::invalid_argument("Planning samples require a complete main-layer expert triplet");
            const auto *gate = triplet[0];
            if (!gate->K || !gate->elements || gate->elements % gate->K ||
                (gate->elements / gate->K) % profile.expert_count)
                throw std::invalid_argument("Planning expert sample has invalid source geometry");
            experts[{triplet[0]->quant_type, triplet[1]->quant_type, triplet[2]->quant_type}]
                [{gate->elements / gate->K / profile.expert_count, gate->K}].push_back(layer);
        }
        for (const auto &[formats, shapes] : experts)
        {
            const auto best = std::max_element(shapes.begin(), shapes.end(), [](const auto &a, const auto &b) {
                const long double left = static_cast<long double>(a.second.size()) * a.first.first * a.first.second;
                const long double right = static_cast<long double>(b.second.size()) * b.first.first * b.first.second;
                if (left != right) return left < right;
                return a.first > b.first;
            });
            const auto &triplet = layers.at(best->second.front());
            result.emplace_back(PlanningExpertSampleRequest{
                {triplet[0]->name, PlanningExpertMatrix{0}}, {triplet[1]->name, PlanningExpertMatrix{0}},
                {triplet[2]->name, PlanningExpertMatrix{0}}});
        }
        if (result.empty()) throw std::invalid_argument("Model has no main-forward source service samples");
        return result;
    }

    std::optional<PlanningWeightServiceModel> PlanningWeightServiceModel::collect(const AutomaticPlanningPreparation &context)
    {
        std::vector<PlanningWeightServiceSample> selected;
        std::optional<AutomaticOrchestrationRequest> request;
        acceptPlanningCostPreparation(context.mpi(), [&] {
            selected = samples(context.metadata());
            request = std::get<AutomaticOrchestrationRequest>(resolveOrchestrationIntent(context.request()));
        });
        auto memory = PlanningKernelServiceCatalog::collect(context.mpi(), context.inventory(), *request,
            PlanningStreamingServicePlan{});
        std::vector<PlanningKernelServiceCatalog> kernels;
        for (const auto &sample : selected)
        {
            auto catalog = std::visit([&](const auto &value) {
                using Sample = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<Sample, PlanningModelSampleRequest>)
                {
                    auto matrix = PlanningMatrixSamplePublication::describe(context.mpi(), [&] {
                        return PlanningMatrixSamplePlan::resolve(context.rootSource(), value);
                    });
                    // M is bounded setup work, not a model prompt or a new CLI
                    // horizon. Price other M explicitly using the family model.
                    const PlanningProjectionServicePlan plan(std::move(matrix), 64);
                    return PlanningKernelServiceCatalog::collect(context.mpi(), context.inventory(), *request, plan,
                        [&](const auto &authority) -> PlanningLoadedKernelSample {
                            return plan.matrix().load(context.rootSource(), authority, DeviceId::cpu());
                        });
                }
                else
                {
                    const auto plan = PlanningExpertSamplePublication::describe(context.mpi(), [&] {
                        return PlanningExpertSamplePlan::resolve(context.rootSource(), value);
                    });
                    return PlanningKernelServiceCatalog::collect(context.mpi(), context.inventory(), *request, plan,
                        [&](const auto &authority) -> PlanningLoadedKernelSample {
                            return plan.load(context.rootSource(), authority, DeviceId::cpu());
                        });
                }
            }, sample);
            // Even root's vector growth is fallible. Finish that local edge in
            // consensus before anyone starts the next source publication.
            acceptPlanningCostPreparation(context.mpi(), [&] {
                if (context.isRoot()) kernels.push_back(std::move(*catalog));
            });
        }
        if (!context.isRoot()) return std::nullopt;
        return PlanningWeightServiceModel(std::move(*memory), std::move(kernels));
    }

    PlanningWeightServiceModel::PlanningWeightServiceModel(PlanningKernelServiceCatalog memory,
        std::vector<PlanningKernelServiceCatalog> kernels) : memory_(std::move(memory)), kernels_(std::move(kernels))
    {
        if (!std::holds_alternative<PlanningStreamingServicePlan>(memory_.source()) || kernels_.empty())
            throw std::invalid_argument("Weight service needs independent streaming and kernel catalogs");
        std::set<std::string> ordinary;
        std::set<std::array<std::string, 3>> experts;
        for (const auto &kernel : kernels_)
        {
            if (const auto *plan = std::get_if<PlanningProjectionServicePlan>(&kernel.source()))
            {
                if (plan->prefillRows() <= 1 || !ordinary.insert(plan->matrix().executionFormat()).second)
                    throw std::invalid_argument("Missing grouped rows or duplicate projection format evidence");
            }
            else if (const auto *plan = std::get_if<PlanningExpertSamplePlan>(&kernel.source()))
            {
                if (!experts.insert(plan->description().formats).second)
                    throw std::invalid_argument("Duplicate expert family evidence");
            }
            else throw std::invalid_argument("Streaming bandwidth is not arithmetic evidence");
            if (kernel.records().size() != memory_.records().size())
                throw std::invalid_argument("Kernel and memory evidence cover different endpoint sets");
            for (const auto &record : memory_.records())
            {
                const auto &observer = record.observer;
                const auto &measured = kernel.serviceFor(observer.discoveryRank(), observer.device());
                if (measured.observer != observer)
                    throw std::invalid_argument("Kernel and memory evidence disagree on physical observer identity");
                if (observer.device().is_cpu())
                {
                    // Identity is not workshare geometry. Two separately
                    // authenticated batches can observe the same rank/NUMA
                    // endpoint with different thread budgets; never combine
                    // their rates as if they measured one CPU configuration.
                    const auto &stream = std::get<PlanningMemoryBandwidthObservation>(record.observation).request;
                    if (const auto *expert = std::get_if<PlanningCPUExpertObservations>(&measured.observation))
                    {
                        if (expert->worker_threads != stream.workers())
                            throw std::invalid_argument("Expert and streaming service use different CPU workshares");
                    }
                    else
                    {
                        const auto &cpu = std::get<PlanningProjectionObservations>(measured.observation).cpu;
                        if (!cpu || cpu->workers != stream.workers() || cpu->execution != stream.cpuGeometry())
                            throw std::invalid_argument("Projection and streaming service use different CPU execution geometry");
                    }
                }
            }
        }
    }

    const PlanningKernelServiceCatalog &PlanningWeightServiceModel::projection(std::string_view format) const
    {
        for (const auto &kernel : kernels_)
            if (const auto *plan = std::get_if<PlanningProjectionServicePlan>(&kernel.source());
                plan && plan->matrix().executionFormat() == format) return kernel;
        throw std::invalid_argument("No observed ordinary projection service for execution format " + std::string(format));
    }

    const PlanningKernelServiceCatalog &PlanningWeightServiceModel::expert(const std::array<std::string, 3> &formats) const
    {
        for (const auto &kernel : kernels_)
            if (const auto *plan = std::get_if<PlanningExpertSamplePlan>(&kernel.source());
                plan && plan->description().formats == formats) return kernel;
        throw std::invalid_argument("No observed complete expert service for execution format triplet");
    }

    double PlanningWeightServiceModel::memorySeconds(int rank, DeviceId device, double bytes) const
    {
        return std::get<PlanningMemoryBandwidthObservation>(memory_.serviceFor(rank, device).observation)
            .service.secondsFor(PlanningWorkUnit::Bytes, bytes);
    }

    double PlanningWeightServiceModel::projectionSeconds(int rank, DeviceId device,
        const PlanningWeightOperand &weight, size_t rows) const
    {
        const auto &shape = weight.geometry.matrix();
        if (!rows || !shape || shape->instances != 1 || !shape->rows || !shape->columns)
            throw std::invalid_argument("Projection service requires a positive single-matrix invocation");
        const auto &catalog = projection(weight.executionFormat());
        const auto &plan = std::get<PlanningProjectionServicePlan>(catalog.source());
        const auto &observed = std::get<PlanningProjectionObservations>(catalog.serviceFor(rank, device).observation);
        const auto &decode = observed.phases.at(0);
        const auto &prefill = observed.phases.at(1);
        const double elements = static_cast<double>(shape->rows) * shape->columns;
        const double ratio = elements / (static_cast<double>(plan.matrix().geometry().n) * plan.matrix().geometry().k);
        const double arithmetic = planningArithmeticSeconds(decode.service, prefill.rows, prefill.service,
            rows, 2.0 * rows * elements);
        // Prepared byte extent is an explicit traffic proxy, never an allocation
        // admission. Borrowing floating kernels report zero owned preparation;
        // their already-admitted runtime source still has to be read, including
        // the larger FP32 representation when the model contract promotes it.
        const double weight_bytes = (observed.prepared_bytes ? observed.prepared_bytes : plan.matrix().executionPayloadBytes()) * ratio;
        const double activation_bytes = sizeof(float) * static_cast<double>(rows) * (shape->rows + shape->columns);
        return positive(std::max(arithmetic, memorySeconds(rank, device, weight_bytes + activation_bytes)));
    }

    double PlanningWeightServiceModel::expertSeconds(int rank, DeviceId device,
        const PlanningRoutedExpertWeightWork &work, size_t token_rows) const
    {
        const auto expectation = work.uniformExpectation(token_rows);
        if (expectation.routed_rows == 0) return 0;
        std::array<std::string, 3> formats;
        double elements = 0;
        for (size_t i = 0; i < formats.size(); ++i)
        {
            formats[i] = work.gate_up_down[i].executionFormat();
            const auto &shape = work.gate_up_down[i].geometry.matrix();
            if (!shape || shape->instances != 1 || !shape->rows || !shape->columns)
                throw std::invalid_argument("Expert service requires complete single-expert operands");
            elements += static_cast<double>(shape->rows) * shape->columns;
        }
        const auto &catalog = expert(formats);
        const auto &plan = std::get<PlanningExpertSamplePlan>(catalog.source());
        const auto &observation = catalog.serviceFor(rank, device).observation;
        return std::visit([&](const auto &observed) -> double {
            using Observation = std::decay_t<decltype(observed)>;
            if constexpr (std::is_same_v<Observation, PlanningCPUExpertObservations> ||
                std::is_same_v<Observation, PlanningGPUExpertObservations>)
            {
                const auto phase = [&](ExpertHistogramSource name) -> const auto & {
                    const auto found = std::find_if(observed.phases.begin(), observed.phases.end(),
                        [&](const auto &entry) { return entry.phase == name; });
                    if (found == observed.phases.end()) throw std::invalid_argument("Expert service lacks required phase");
                    return *found;
                };
                const auto &decode = phase(ExpertHistogramSource::DecodeToken);
                const auto &prefill = phase(ExpertHistogramSource::PrefillChunk);
                // A nonempty group necessarily contains at least one row.
                // log1p/expm1 hit probabilities can round a few ULPs above the
                // one-token expectation; preserve that mathematical boundary.
                const double rows_per_group = std::max(1.0, expectation.routed_rows / expectation.nonempty_experts);
                const double arithmetic = planningArithmeticSeconds(decode.service, prefill.rows, prefill.service,
                    rows_per_group, 2 * expectation.routed_rows * elements);
                const double source_elements = 3.0 * observed.input_width * observed.intermediate_width;
                const double bytes = observed.prepared_bytes ? observed.prepared_bytes : plan.description().source_bytes;
                const double traffic = bytes * (elements / source_elements) * expectation.nonempty_experts +
                    2.0 * sizeof(float) * work.gate_up_down[0].geometry.matrix()->columns * expectation.routed_rows;
                return positive(std::max(arithmetic, memorySeconds(rank, device, traffic)));
            }
            else throw std::logic_error("Expert catalog contains a different operation family");
        }, observation);
    }
}
