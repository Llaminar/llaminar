/**
 * @file PlanningMatrixSample.cpp
 * @brief Strict matrix selection publication using canonical GGUF geometry and loading.
 *
 * Matrix slices retain their original source coordinates; equivalent output
 * extents cannot disguise a different shard. The shared sample codec validates
 * native bytes for every format, and load revalidates the sealed plan against
 * the same retained directory before allocating or reading anything.
 */
#include "PlanningMatrixSample.h"
#include "PlanningSampleWire.h"
#include "loaders/PreparedWeightRepresentationContract.h"
#include "tensors/NativeTensorExtent.h"
#include <array>
#include <limits>
#include <type_traits>

namespace llaminar2
{
    PlanningMatrixSamplePlan PlanningMatrixSamplePlan::resolve(const PlanningModelSource &source,
        const PlanningModelSampleRequest &request)
    {
        PlanningMatrixSamplePlan plan;
        plan.model_path_ = source.path();
        plan.architecture_ = source.metadata().memoryProfile().architecture;
        plan.request_ = request;
        plan.geometry_ = source.sampleGeometry(request);
        const auto *tensor = source.loader().getModel().findTensor(request.tensor_name);
        plan.gguf_type_ = static_cast<uint32_t>(tensor->type);
        plan.format_ = planning_sample_wire::sourceFormat(plan.tensorType());
        return deserialize(plan.serialize());
    }

    TensorType PlanningMatrixSamplePlan::tensorType() const
    {
        return ggufToTensorType(static_cast<GGUFTensorType>(gguf_type_));
    }

    TensorType PlanningMatrixSamplePlan::executionType() const
    {
        return PreparedWeightRepresentationContract::resolve(inferWeightRole(request_.tensor_name), tensorType()) ==
            ModelPreparedWeightRepresentation::FP32 ? TensorType::FP32 : tensorType();
    }

    std::string PlanningMatrixSamplePlan::executionFormat() const
    {
        return planning_sample_wire::sourceFormat(executionType());
    }

    size_t PlanningMatrixSamplePlan::executionPayloadBytes() const
    {
        if (executionType() == tensorType()) return geometry_.source_bytes;
        return nativeTensorExtent(std::array{geometry_.n, geometry_.k}, 1, sizeof(float));
    }

    std::vector<uint8_t> PlanningMatrixSamplePlan::serialize() const
    {
        using Json = nlohmann::json;
        const auto selection = std::visit([](const auto &value) -> Json {
            using Selection = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<Selection, PlanningWholeMatrix>) return {{"kind", "whole"}};
            else if constexpr (std::is_same_v<Selection, PlanningExpertMatrix>) return {{"kind", "expert"}, {"index", value.index}};
            else return {{"kind", std::is_same_v<Selection, PlanningMatrixRows> ? "rows" : "columns"},
                {"first", value.first}, {"last", value.last}};
        }, request_.selection);
        const auto text = Json{{"schema", "llaminar.planning-matrix-sample.v1"}, {"model", model_path_},
            {"architecture", architecture_}, {"selection", selection},
            {"matrix", planning_sample_wire::encode({request_.tensor_name, gguf_type_, geometry_, format_})}}.dump();
        return {text.begin(), text.end()};
    }

    PlanningMatrixSamplePlan PlanningMatrixSamplePlan::deserialize(std::span<const uint8_t> bytes)
    {
        using planning_sample_wire::unsignedField;
        const auto wire = nlohmann::json::parse(bytes.begin(), bytes.end());
        if (!wire.is_object() || wire.size() != 5 || wire.at("schema") != "llaminar.planning-matrix-sample.v1")
            throw std::invalid_argument("Invalid planning matrix sample envelope");
        PlanningMatrixSamplePlan plan;
        plan.model_path_ = wire.at("model").get<std::string>();
        plan.architecture_ = wire.at("architecture").get<std::string>();
        if (plan.model_path_.empty() || plan.architecture_.empty())
            throw std::invalid_argument("Planning matrix sample omits source identity");
        const auto matrix = planning_sample_wire::decode(wire.at("matrix"));
        plan.request_.tensor_name = matrix.name;
        plan.gguf_type_ = matrix.gguf_type;
        plan.geometry_ = matrix.geometry;
        plan.format_ = matrix.format;
        const auto &selection = wire.at("selection");
        const auto kind = selection.at("kind").get<std::string>();
        if (kind == "whole" && selection.size() == 1) plan.request_.selection = PlanningWholeMatrix{};
        else if (kind == "expert" && selection.size() == 2)
            plan.request_.selection = PlanningExpertMatrix{unsignedField(selection.at("index"), std::numeric_limits<size_t>::max())};
        else if ((kind == "rows" || kind == "columns") && selection.size() == 3)
        {
            const size_t first = unsignedField(selection.at("first"), std::numeric_limits<int>::max());
            const size_t last = unsignedField(selection.at("last"), std::numeric_limits<int>::max());
            const size_t extent = kind == "rows" ? plan.geometry_.n : plan.geometry_.k;
            if (first >= last || last - first != extent)
                throw std::invalid_argument("Planning matrix source interval disagrees with sample extent");
            if (kind == "rows") plan.request_.selection = PlanningMatrixRows{first, last};
            else
            {
                GGUFTensorInfo info{};
                info.type = static_cast<GGUFTensorType>(plan.gguf_type_);
                const size_t block = std::max(size_t{1}, info.getBlockSize());
                if (first % block || last % block)
                    throw std::invalid_argument("Planning matrix source columns cross native blocks");
                plan.request_.selection = PlanningMatrixColumns{first, last};
            }
        }
        else throw std::invalid_argument("Unknown or malformed planning matrix source selection");
        return plan;
    }

    PlanningLoadedMatrixSample PlanningMatrixSamplePlan::load(const PlanningModelSource &source,
        const std::shared_ptr<PhysicalMemoryAuthority> &memory, DeviceId host_device) const
    {
        if (resolve(source, request_).serialize() != serialize())
            throw std::invalid_argument("Planning matrix source no longer matches its admitted selection");
        return PlanningLoadedMatrixSample(*this, source.loadSample(request_, memory, host_device));
    }
}
