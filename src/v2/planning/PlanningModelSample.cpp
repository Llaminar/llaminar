/**
 * @file PlanningModelSample.cpp
 * @brief Backend-independent expert identity validation for bounded source samples.
 *
 * CPU and GPU preparation consume the same immutable source geometry. This
 * boundary performs no payload I/O and never guesses topology from source paths
 * or timing. Backend-specific packing and execution contribute separate BOMs.
 */
#include "PlanningModelMetadata.h"
#include <algorithm>
#include <limits>
#include <optional>
#include <stdexcept>

namespace llaminar2
{
    PlanningExpertSampleDescription PlanningExpertSampleDescription::resolve(
        const PlanningModelSource &source, const PlanningExpertSampleRequest &request)
    {
        PlanningExpertSampleDescription result{};
        const auto inputs = request.projections();
        const auto &directory = source.metadata().memoryProfile().tensors;
        std::optional<size_t> expert;
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            const auto *selection = std::get_if<PlanningExpertMatrix>(&inputs[i]->selection);
            if (!selection || (expert && *expert != selection->index))
                throw std::invalid_argument("Expert measurement requires one complete, common source expert");
            expert = selection->index;
            result.matrices[i] = source.sampleGeometry(*inputs[i]);
            const auto entry = std::find_if(directory.begin(), directory.end(), [&](const auto &tensor) {
                return tensor.name == inputs[i]->tensor_name;
            });
            if (entry == directory.end() || entry->layer_index < 0 ||
                (i && entry->layer_index != result.layer))
                throw std::invalid_argument("Expert sample projections must belong to the same model layer");
            result.layer = entry->layer_index;
            result.formats[i] = entry->quant_type;
            const size_t bytes = result.matrices[i].source_bytes;
            if (bytes > std::numeric_limits<size_t>::max() - result.source_bytes)
                throw std::overflow_error("Expert source payload overflow");
            result.source_bytes += bytes;
            result.largest_source_bytes = std::max(result.largest_source_bytes, bytes);
        }
        const auto &[gate, up, down] = result.matrices;
        if (gate.n != up.n || gate.k != up.k || down.n != gate.k || down.k != gate.n ||
            request.gate.tensor_name == request.up.tensor_name ||
            request.gate.tensor_name == request.down.tensor_name || request.up.tensor_name == request.down.tensor_name)
            throw std::invalid_argument("Expert sample requires three distinct, compatible FFN projections");
        return result;
    }
}
