/**
 * @file PlanningModelMetadata.cpp
 * @brief Root-only GGUF discovery and authenticated planning-metadata publication.
 *
 * The descriptor wraps the existing ModelMemoryProfile wire format rather than
 * maintaining a second tensor directory. Read, receive allocation and decode
 * each reach the common planning-publication consensus before any rank advances.
 * Selected apply configurations use that same transport and failure lifecycle.
 * These are startup collectives, not host decisions in a captured request loop.
 */
#include "planning/PlanningModelMetadata.h"
#include "planning/PlanningPublication.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "execution/mpi_orchestration/IExecutionPlanBuilder.h"
#include "execution/mtp/MTPWeightManifest.h"
#include "loaders/ModelLoader.h"
#include "tensors/TensorFactory.h"
#include "utils/MPIContext.h"
#include <array>
#include <limits>
#include <stdexcept>

namespace llaminar2
{
    namespace
    {
        constexpr std::array<uint8_t, 4> kDescriptorMagic{'L', 'P', 'M', 1};
        constexpr size_t kDescriptorHeaderBytes = 8;

    }

    PlanningModelMetadata::PlanningModelMetadata(ModelMemoryProfile profile, int main_layer_count)
        : profile_(std::move(profile)), main_layer_count_(main_layer_count)
    {
        if (main_layer_count <= 0 || main_layer_count > profile_.n_layers)
            throw std::invalid_argument("Planning metadata has an invalid main-layer interval");
        const auto errors = executionModelConfig().validate();
        if (!errors.empty())
            throw std::invalid_argument("Invalid planning model geometry: " + errors.front());
    }

    ModelConfig PlanningModelMetadata::executionModelConfig() const
    {
        return {
            .name = profile_.architecture,
            .n_layers = main_layer_count_,
            .n_heads = profile_.n_heads,
            .n_kv_heads = profile_.n_kv_heads,
            .hidden_size = profile_.d_model,
            .intermediate_size = profile_.d_ff,
            .vocab_size = profile_.vocab_size,
            .head_dim = profile_.head_dim,
            .estimated_weight_bytes = profile_.total_native_bytes,
        };
    }

    PlanningModelMetadata PlanningModelMetadata::fromLoader(const ModelLoader &loader)
    {
        const int layers = mainLayerCountExcludingMTP(
            loader, loader.architecture(), static_cast<int>(loader.blockCount()));
        return PlanningModelMetadata(ModelMemoryProfile::fromGGUF(loader.getModel()), layers);
    }

    std::vector<uint8_t> PlanningModelMetadata::serialize() const
    {
        auto payload = profile_.serialize();
        std::vector<uint8_t> bytes(kDescriptorHeaderBytes);
        std::copy(kDescriptorMagic.begin(), kDescriptorMagic.end(), bytes.begin());
        const auto layers = static_cast<uint32_t>(main_layer_count_);
        // Explicit byte order keeps descriptor identity independent of the
        // initiating host's integer layout. The profile owns its own version.
        for (unsigned byte = 0; byte < 4; ++byte)
            bytes[4 + byte] = static_cast<uint8_t>(layers >> (8 * byte));
        bytes.insert(bytes.end(), payload.begin(), payload.end());
        return bytes;
    }

    PlanningModelMetadata PlanningModelMetadata::deserialize(std::span<const uint8_t> bytes)
    {
        if (bytes.size() <= kDescriptorHeaderBytes ||
            !std::equal(kDescriptorMagic.begin(), kDescriptorMagic.end(), bytes.begin()))
            throw std::runtime_error("Invalid or stale planning model descriptor header");
        uint32_t layers = 0;
        for (unsigned byte = 0; byte < 4; ++byte)
            layers |= static_cast<uint32_t>(bytes[4 + byte]) << (8 * byte);
        if (layers > static_cast<uint32_t>(std::numeric_limits<int>::max()))
            throw std::runtime_error("Planning main-layer count exceeds supported geometry");
        return PlanningModelMetadata(
            ModelMemoryProfile::deserialize(bytes.data() + kDescriptorHeaderBytes,
                                            bytes.size() - kDescriptorHeaderBytes),
            static_cast<int>(layers));
    }

    /** @brief One destruction-ordered metadata transaction with stable borrowed addresses. */
    struct PlanningModelSource::Storage
    {
        std::string path;
        std::shared_ptr<IMPIContext> local_context;
        TensorFactory factory;
        ModelLoader loader;
        PlanningModelMetadata metadata;

        /** @brief Bind process-local metadata infrastructure before reading once. */
        explicit Storage(const std::string &model_path)
            : path(model_path), local_context(MPIContextFactory::self()), factory(*local_context),
              loader(&factory), metadata(read(loader, model_path)) {}

        /** @return Metadata from this exact loader, without touching weight payloads. */
        static PlanningModelMetadata read(ModelLoader &loader, const std::string &path)
        {
            loader.setUseMmap(false);
            if (!loader.loadModel(path))
                throw std::runtime_error("Cannot read planning metadata from " + path);
            return PlanningModelMetadata::fromLoader(loader);
        }
    };

    PlanningModelSource::PlanningModelSource(const std::string &model_path)
    {
        if (model_path.empty()) throw std::invalid_argument("Planning requires a GGUF model path");
        storage_ = std::make_unique<Storage>(model_path);
    }
    PlanningModelSource::~PlanningModelSource() = default;
    PlanningModelSource::PlanningModelSource(PlanningModelSource &&) noexcept = default;
    PlanningModelSource &PlanningModelSource::operator=(PlanningModelSource &&) noexcept = default;

    const PlanningModelMetadata &PlanningModelSource::metadata() const
    {
        if (!storage_) throw std::logic_error("Planning model source was moved from");
        return storage_->metadata;
    }
    const ModelLoader &PlanningModelSource::loader() const
    {
        if (!storage_) throw std::logic_error("Planning model source was moved from");
        return storage_->loader;
    }

    const std::string &PlanningModelSource::path() const
    {
        if (!storage_) throw std::logic_error("Planning model source was moved from");
        return storage_->path;
    }

    PlanningModelSampleGeometry PlanningModelSource::sampleGeometry(const PlanningModelSampleRequest &request) const
    {
        const auto *info = loader().getModel().findTensor(request.tensor_name);
        if (!info) throw std::invalid_argument("Planning sample tensor is absent: " + request.tensor_name);
        const bool expert = std::holds_alternative<PlanningExpertMatrix>(request.selection);
        if (info->dimensions.size() != (expert ? 3u : 2u))
            throw std::invalid_argument("Planning sample selection does not match the source tensor rank");
        // ModelLoader normalizes ordinary 2-D GGUF weights to [N,K], but keeps
        // routed parent directories in [K,N,experts] order. Do not flatten all
        // experts into N or use the expert count as the inner GEMM dimension.
        size_t n = info->dimensions[expert ? 1 : 0];
        size_t k = info->dimensions[expert ? 0 : 1];
        const size_t block = std::max(size_t{1}, info->getBlockSize());
        const size_t block_bytes = info->getTypeSize();
        if (n == 0 || k == 0 || block_bytes == 0 || k % block != 0)
            throw std::invalid_argument("Planning sample has unsupported native format or unaligned K");
        const auto multiply = [](size_t a, size_t b) {
            if (b && a > std::numeric_limits<size_t>::max() / b)
                throw std::overflow_error("Planning sample source extent overflow");
            return a * b;
        };
        const size_t matrix_bytes = multiply(n, multiply(k / block, block_bytes));
        const size_t directory_bytes = expert ? multiply(matrix_bytes, info->dimensions[2]) : matrix_bytes;
        if (directory_bytes != info->size_bytes)
            throw std::invalid_argument("Planning sample directory extent disagrees with its native geometry");
        const auto requireRange = [](size_t first, size_t last, size_t extent) {
            if (first >= last || last > extent)
                throw std::invalid_argument("Planning sample interval is empty or outside its source");
        };
        std::visit([&](const auto &selection) {
            using Selection = std::decay_t<decltype(selection)>;
            if constexpr (std::is_same_v<Selection, PlanningMatrixRows>)
            {
                requireRange(selection.first, selection.last, n);
                n = selection.last - selection.first;
            }
            else if constexpr (std::is_same_v<Selection, PlanningMatrixColumns>)
            {
                requireRange(selection.first, selection.last, k);
                if (selection.first % block || selection.last % block)
                    throw std::invalid_argument("Planning sample column interval splits a source codebook block");
                k = selection.last - selection.first;
            }
            else if constexpr (std::is_same_v<Selection, PlanningExpertMatrix>)
            {
                if (selection.index >= info->dimensions[2])
                    throw std::invalid_argument("Planning sample expert is outside its source parent");
            }
        }, request.selection);
        if (n > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            k > static_cast<size_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument("Planning sample exceeds supported GEMM geometry");
        return {n, k, multiply(n, multiply(k / block, block_bytes))};
    }

    /** @brief Keep the matrix borrow, parent allocation and claim in retirement order. */
    struct PlanningLoadedModelSample::Storage final
    {
        PhysicalMemoryAllocationLease claim;
        std::shared_ptr<TensorBase> parent;
        std::shared_ptr<TensorBase> matrix;
    };

    PlanningLoadedModelSample::PlanningLoadedModelSample(std::shared_ptr<Storage> storage)
        : storage_(std::move(storage)) {}

    const TensorBase &PlanningLoadedModelSample::tensor() const
    {
        if (!storage_) throw std::logic_error("Loaded planning sample was moved from");
        return *storage_->matrix;
    }

    PlanningLoadedModelSample PlanningLoadedModelSample::allocateReceive(TensorType type,
        PlanningModelSampleGeometry geometry, const std::shared_ptr<PhysicalMemoryAuthority> &memory,
        DeviceId host_device)
    {
        if (!memory || !host_device.is_cpu())
            throw std::invalid_argument("Planning sample reception requires an admitted CPU allocator");
        auto payload = std::make_shared<Storage>();
        payload->claim = memory->claimNewAllocation(host_device, PhysicalMemoryOwner::ModelSourcePayload,
            geometry.source_bytes);
        const auto context = MPIContextFactory::self();
        TensorFactory factory(*context);
        const std::vector<size_t> shape{geometry.n, geometry.k};
        // Allocation/first touch runs on the rank's affined setup thread. The
        // initialized tensor is the final receive address, not a staging copy.
        if (type == TensorType::FP32) payload->matrix = factory.createFP32(shape, host_device);
        else if (type == TensorType::FP16) payload->matrix = factory.createFP16(shape);
        else if (type == TensorType::BF16) payload->matrix = factory.createBF16(shape);
        else payload->matrix = factory.createQuantizedOwned(type, shape, AlignedVector<uint8_t>(geometry.source_bytes));
        if (!payload->matrix || payload->matrix->size_bytes() != geometry.source_bytes)
            throw std::runtime_error("Planning sample receive allocation disagrees with native extent");
        return PlanningLoadedModelSample(std::move(payload));
    }

    void *PlanningLoadedModelSample::receiveData()
    {
        if (!storage_) throw std::logic_error("Planning receive sample was moved from");
        return storage_->matrix->raw_mutable_data();
    }

    PlanningLoadedModelSample PlanningModelSource::loadSample(const PlanningModelSampleRequest &request,
        const std::shared_ptr<PhysicalMemoryAuthority> &memory, DeviceId host_device) const
    {
        const auto geometry = sampleGeometry(request);
        if (!memory || !host_device.is_cpu())
            throw std::invalid_argument("Planning source sampling requires an admitted CPU allocator");
        auto payload = std::make_shared<PlanningLoadedModelSample::Storage>();
        payload->claim = memory->claimNewAllocation(host_device, PhysicalMemoryOwner::ModelSourcePayload,
            geometry.source_bytes);
        const auto reader_claim = memory->claimNewAllocation(host_device, PhysicalMemoryOwner::WeightLoadStaging,
            geometry.source_bytes);
        payload->parent = std::visit([&](const auto &selection) -> std::shared_ptr<TensorBase> {
            using Selection = std::decay_t<decltype(selection)>;
            if constexpr (std::is_same_v<Selection, PlanningMatrixColumns>)
                return storage_->loader.loadTensorColumnSlice(request.tensor_name, selection.first, selection.last, host_device);
            else if constexpr (std::is_same_v<Selection, PlanningExpertMatrix>)
                return storage_->loader.loadTensorExpertSlice(request.tensor_name, selection.index, selection.index + 1, host_device);
            else
            {
                const size_t first = [&] {
                    if constexpr (std::is_same_v<Selection, PlanningMatrixRows>) return selection.first;
                    else return size_t{0};
                }();
                return storage_->loader.loadTensorRowSlice(request.tensor_name, first, first + geometry.n, host_device);
            }
        }, request.selection);
        if (!payload->parent || !payload->parent->raw_data() || payload->parent->size_bytes() != geometry.source_bytes)
            throw std::runtime_error("Planning source sample did not materialize its exact native extent");
        payload->matrix = std::holds_alternative<PlanningExpertMatrix>(request.selection)
            ? payload->parent->create_view({geometry.n, geometry.k}) : payload->parent;
        if (!payload->matrix || payload->matrix->shape() != std::vector<size_t>{geometry.n, geometry.k})
            throw std::runtime_error("Planning source sample did not publish its exact matrix geometry");
        return PlanningLoadedModelSample(std::move(payload));
    }

    size_t maximumGGUFTensorPayloadBytes(const GGUFModel &model)
    {
        uint64_t maximum = 0;
        for (const auto &tensor : model.tensors)
            maximum = std::max(maximum, tensor.size_bytes);
        if (maximum == 0)
            throw std::invalid_argument("GPU upload admission requires a nonempty GGUF tensor manifest");
        if (maximum > std::numeric_limits<size_t>::max())
            throw std::overflow_error("Maximum GGUF tensor payload exceeds size_t");
        return static_cast<size_t>(maximum);
    }

    PlanningModelMetadata readPlanningModelMetadata(const std::string &model_path)
    {
        return PlanningModelSource(model_path).metadata();
    }

    PlanningModelMetadata exchangePlanningModelMetadata(
        const std::shared_ptr<IMPIContext> &mpi,
        const std::function<PlanningModelMetadata()> &root_reader)
    {
        std::optional<PlanningModelMetadata> descriptor;
        exchangePlanningArtifact(mpi, PlanningArtifact::ModelMetadata,
            [&] { return root_reader().serialize(); },
            [&](std::span<const uint8_t> payload) {
                descriptor.emplace(PlanningModelMetadata::deserialize(payload));
            });
        return std::move(*descriptor);
    }
}
