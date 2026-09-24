/**
 * @file PlanningExpertSample.cpp
 * @brief Immutable native expert sample plans and local source materialization.
 *
 * The wire record carries exact source selections, not pointers or device
 * ordinals. Decoding validates the loader's canonical GGUF block geometry
 * before any receive allocation. Only the original file owner performs I/O.
 */
#include "PlanningExpertSample.h"
#include "PlanningSampleWire.h"
#include "loaders/ModelLoader.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace llaminar2
{
    using planning_sample_wire::unsignedField;

    PlanningExpertSamplePlan PlanningExpertSamplePlan::resolve(const PlanningModelSource &source,
        const PlanningExpertSampleRequest &request)
    {
        PlanningExpertSamplePlan plan;
        plan.model_path_ = source.path();
        plan.architecture_ = source.metadata().memoryProfile().architecture;
        plan.request_ = request;
        plan.description_ = PlanningExpertSampleDescription::resolve(source, request);
        const auto projections = request.projections();
        for (size_t i = 0; i < projections.size(); ++i)
            plan.gguf_types_[i] = static_cast<uint32_t>(source.loader().getModel().findTensor(projections[i]->tensor_name)->type);
        // The local path and the wire path use precisely the same validator.
        return deserialize(plan.serialize());
    }

    TensorType PlanningExpertSamplePlan::tensorType(size_t projection) const
    {
        return ggufToTensorType(static_cast<GGUFTensorType>(gguf_types_.at(projection)));
    }

    std::vector<uint8_t> PlanningExpertSamplePlan::serialize() const
    {
        nlohmann::json matrices = nlohmann::json::array();
        const auto inputs = request_.projections();
        for (size_t i = 0; i < inputs.size(); ++i)
            matrices.push_back(planning_sample_wire::encode(
                {inputs[i]->tensor_name, gguf_types_[i], description_.matrices[i], description_.formats[i]}));
        const auto text = nlohmann::json{{"schema", "llaminar.planning-expert-sample.v1"},
            {"model", model_path_}, {"architecture", architecture_}, {"layer", description_.layer},
            {"expert", std::get<PlanningExpertMatrix>(request_.gate.selection).index},
            {"matrices", matrices}}.dump();
        return {text.begin(), text.end()};
    }

    PlanningExpertSamplePlan PlanningExpertSamplePlan::deserialize(std::span<const uint8_t> bytes)
    {
        const auto wire = nlohmann::json::parse(bytes.begin(), bytes.end());
        if (!wire.is_object() || wire.size() != 6 ||
            wire.at("schema") != "llaminar.planning-expert-sample.v1")
            throw std::invalid_argument("Invalid planning expert sample envelope");
        PlanningExpertSamplePlan plan;
        plan.model_path_ = wire.at("model").get<std::string>();
        plan.architecture_ = wire.at("architecture").get<std::string>();
        if (plan.model_path_.empty() || plan.architecture_.empty())
            throw std::invalid_argument("Planning expert sample omits source identity");
        plan.description_.layer = static_cast<int>(unsignedField(wire.at("layer"), std::numeric_limits<int>::max()));
        const size_t expert = unsignedField(wire.at("expert"), std::numeric_limits<size_t>::max());
        const auto &matrices = wire.at("matrices");
        if (!matrices.is_array() || matrices.size() != 3)
            throw std::invalid_argument("Planning expert sample requires exactly three projections");
        const std::array inputs{&plan.request_.gate, &plan.request_.up, &plan.request_.down};
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            const auto matrix = planning_sample_wire::decode(matrices.at(i));
            *inputs[i] = {matrix.name, PlanningExpertMatrix{expert}};
            plan.gguf_types_[i] = matrix.gguf_type;
            plan.description_.formats[i] = matrix.format;
            plan.description_.matrices[i] = matrix.geometry;
            if (matrix.geometry.source_bytes > std::numeric_limits<size_t>::max() - plan.description_.source_bytes)
                throw std::invalid_argument("Planning expert sample source extent overflow");
            plan.description_.source_bytes += matrix.geometry.source_bytes;
            plan.description_.largest_source_bytes = std::max(plan.description_.largest_source_bytes, matrix.geometry.source_bytes);
        }
        const auto &[gate, up, down] = plan.description_.matrices;
        if (gate.n != up.n || gate.k != up.k || down.n != gate.k || down.k != gate.n ||
            inputs[0]->tensor_name == inputs[1]->tensor_name || inputs[0]->tensor_name == inputs[2]->tensor_name ||
            inputs[1]->tensor_name == inputs[2]->tensor_name)
            throw std::invalid_argument("Planning expert sample has incompatible FFN projections");
        return plan;
    }

    PlanningLoadedExpertSample PlanningExpertSamplePlan::load(const PlanningModelSource &source,
        const std::shared_ptr<PhysicalMemoryAuthority> &memory, DeviceId host_device) const
    {
        if (resolve(source, request_).serialize() != serialize())
            throw std::invalid_argument("Planning source no longer matches the admitted sample plan");
        return PlanningLoadedExpertSample(*this, {source.loadSample(request_.gate, memory, host_device),
            source.loadSample(request_.up, memory, host_device), source.loadSample(request_.down, memory, host_device)});
    }
}
