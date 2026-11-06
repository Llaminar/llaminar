/**
 * @file PlanningModelMetadata.cpp
 * @brief Root-only GGUF discovery and authenticated planning-metadata publication.
 *
 * The descriptor wraps the existing ModelMemoryProfile wire format rather than
 * maintaining a second tensor directory. Read, receive allocation and decode
 * each reach the common planning-publication consensus before any rank advances.
 * Selected apply configurations use that same transport and failure lifecycle.
 * These are startup collectives, not host decisions in a captured request loop.
 * Native calibration samples use the same original-coordinate window reader as
 * model preparation. Reads populate the existing final sample owner directly;
 * there is no second matrix-sized reader allocation or retained source parent.
 * Unpublished receivers become tensors only after complete I/O and native
 * admission. The original loader owns whole-source online-A4 policy admission,
 * borrowing this final allocation before returning selected bytes. Planning
 * neither narrows that policy nor duplicates its scan or validation authority.
 */
#include "planning/PlanningModelMetadata.h"
#include "planning/PlanningPublication.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "execution/mpi_orchestration/IExecutionPlanBuilder.h"
#include "execution/mtp/MTPWeightManifest.h"
#include "loaders/ModelLoader.h"
#include "loaders/WeightIdentity.h"
#include "loaders/NativeWeightLoadStaging.h"
#include "tensors/NativeWeightSourceRange.h"
#include "tensors/TensorFactory.h"
#include "tensors/NativeTensorExtent.h"
#include "tensors/W4A4Tensor.h"
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
        NativeVnniSourceIdentity terminal_head_source;
        // Tensor-directory provenance bounds measured defaults without loading
        // weights or inferring a quantizer from a model filename.
        for (const auto &tensor : profile_.tensors)
            if (tensor.name == "output.weight")
                if (const auto *format = native_vnni_formats::forQuantType(tensor.quant_type))
                    terminal_head_source = {.codebook_id = format->codebook_id,
                        .is_superblock = format->is_superblock, .present = true};
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
            .terminal_head_source = terminal_head_source,
        };
    }

    PlanningModelMetadata PlanningModelMetadata::fromLoader(const ModelLoader &loader)
    {
        const int layers = mainLayerCountExcludingMTP(
            loader, loader.architecture(), static_cast<int>(loader.blockCount()));
        return PlanningModelMetadata(ModelMemoryProfile::fromLoader(loader), layers);
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
        std::optional<SharedPredictorGGUFSource> predictor;
        std::shared_ptr<IMPIContext> local_context;
        TensorFactory factory;
        ModelLoader loader;
        PlanningModelMetadata metadata;

        /** @brief Bind process-local metadata infrastructure before reading once. */
        Storage(const std::string &model_path, const std::optional<SharedPredictorGGUFSource> &predictor_source)
            : path(model_path), predictor(predictor_source), local_context(MPIContextFactory::self()), factory(*local_context),
              loader(&factory), metadata(read(loader, model_path, predictor)) {}

        /** @return Metadata from this exact loader, without touching weight payloads. */
        static PlanningModelMetadata read(ModelLoader &loader, const std::string &path,
            const std::optional<SharedPredictorGGUFSource> &predictor)
        {
            loader.setUseMmap(false);
            if (!loader.loadModel(path))
                throw std::runtime_error("Cannot read planning metadata from " + path);
            if (predictor) loader.bindSharedPredictor(*predictor, SharedPredictorMaterialization::Active);
            loader.sealTensorSources();
            return PlanningModelMetadata::fromLoader(loader);
        }
    };

    PlanningModelSource::PlanningModelSource(const std::string &model_path,
        const std::optional<SharedPredictorGGUFSource> &predictor)
    {
        if (model_path.empty()) throw std::invalid_argument("Planning requires a GGUF model path");
        storage_ = std::make_unique<Storage>(model_path, predictor);
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

    const std::optional<SharedPredictorGGUFSource> &PlanningModelSource::predictorSource() const
    {
        if (!storage_) throw std::logic_error("Planning model source was moved from");
        return storage_->predictor;
    }

    PlanningModelSampleGeometry PlanningModelSource::sampleGeometry(const PlanningModelSampleRequest &request) const
    {
        const auto *info = loader().tensorInfo(request.tensor_name);
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
                if (isW4A4Format(ggufToTensorType(info->type)) &&
                    (selection.first != 0 || selection.last != k))
                    throw std::invalid_argument("W4A4/H planning samples require complete original K");
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

    /** @brief Retain the final native sample whose storage carries its own admission lease. */
    struct PlanningLoadedModelSample::Storage final
    {
        std::shared_ptr<TensorBase> matrix;
    };

    PlanningLoadedModelSample::PlanningLoadedModelSample(std::shared_ptr<Storage> storage)
        : storage_(std::move(storage)) {}

    const TensorBase &PlanningLoadedModelSample::tensor() const
    {
        if (!storage_) throw std::logic_error("Loaded planning sample was moved from");
        return *storage_->matrix;
    }

    /** @brief Exclusive final roots retain their scalar ABI and original allocator through sealing. */
    struct PlanningModelSampleReceiver::Storage final
    {
        using Bytes = std::variant<AlignedVector<float>, AlignedVector<uint16_t>, AlignedVector<uint8_t>>;
        TensorType type; ///< Source format, not a requested conversion.
        PlanningModelSampleGeometry geometry; ///< Already validated source selection.
        DeviceId host_device; ///< Original admitted CPU allocator.
        Bytes bytes; ///< One final payload; no tensor can borrow it before finish.
    };

    PlanningModelSampleReceiver::PlanningModelSampleReceiver(std::unique_ptr<Storage> storage)
        : storage_(std::move(storage)) {}
    PlanningModelSampleReceiver::~PlanningModelSampleReceiver() = default;
    PlanningModelSampleReceiver::PlanningModelSampleReceiver(PlanningModelSampleReceiver &&) noexcept = default;
    PlanningModelSampleReceiver &PlanningModelSampleReceiver::operator=(PlanningModelSampleReceiver &&) noexcept = default;

    PlanningModelSampleReceiver PlanningModelSampleReceiver::allocate(TensorType type,
        PlanningModelSampleGeometry geometry, const std::shared_ptr<PhysicalMemoryAuthority> &memory,
        DeviceId host_device)
    {
        if (!memory || !host_device.is_cpu())
            throw std::invalid_argument("Planning sample reception requires an admitted CPU allocator");
        if (!geometry.n || !geometry.k || !geometry.source_bytes ||
            geometry.n > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            geometry.k > static_cast<size_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument("Planning sample reception requires complete matrix geometry");
        const auto allocate = [&]<typename T>(size_t count) {
            return CPUStorageRoot<T>(count, PhysicalMemoryOwner::ModelSourcePayload, host_device,
                CPUStoragePlacement::local(), CPUStorageInitialization::Uninitialized).allocate(memory);
        };
        const std::array shape{geometry.n, geometry.k};
        Storage::Bytes bytes;
        if (type == TensorType::FP32 || type == TensorType::FP16 || type == TensorType::BF16)
        {
            const size_t scalar_bytes = type == TensorType::FP32 ? sizeof(float) : sizeof(uint16_t);
            if (geometry.source_bytes != nativeTensorExtent(shape, 1, scalar_bytes))
                throw std::invalid_argument("Planning floating receive extent disagrees with geometry");
            if (type == TensorType::FP32)
                bytes = allocate.template operator()<float>(geometry.source_bytes / sizeof(float));
            else bytes = allocate.template operator()<uint16_t>(geometry.source_bytes / sizeof(uint16_t));
        }
        else
        {
            if (isW4A4Format(type))
            {
                const auto extent = w4a4::MatrixContract::sourceExtent(geometry.n, geometry.k);
                if (!extent || *extent != geometry.source_bytes)
                    throw std::invalid_argument("Planning online-A4 receive extent disagrees with geometry");
            }
            bytes = allocate.template operator()<uint8_t>(geometry.source_bytes);
        }
        return PlanningModelSampleReceiver(std::make_unique<Storage>(
            Storage{type, geometry, host_device, std::move(bytes)}));
    }

    std::span<std::byte> PlanningModelSampleReceiver::receiveBytes()
    {
        if (!storage_) throw std::logic_error("Planning native receiver has retired");
        return std::visit([](auto &bytes) { return std::as_writable_bytes(std::span(bytes.data(), bytes.size())); },
            storage_->bytes);
    }

    PlanningLoadedModelSample PlanningModelSampleReceiver::finish()
    {
        if (!storage_) throw std::logic_error("Planning native receiver has retired");
        auto source = std::move(storage_); // Failure consumes the writer and retires its original lease.
        auto payload = std::make_shared<PlanningLoadedModelSample::Storage>();
        const std::vector<size_t> shape{source->geometry.n, source->geometry.k};
        if (source->type == TensorType::FP32) payload->matrix = std::make_shared<FP32Tensor>(
            shape, std::move(std::get<AlignedVector<float>>(source->bytes)), source->host_device);
        else if (source->type == TensorType::FP16) payload->matrix = std::make_shared<FP16Tensor>(
            shape, std::move(std::get<AlignedVector<uint16_t>>(source->bytes)), source->host_device);
        else if (source->type == TensorType::BF16) payload->matrix = std::make_shared<BF16Tensor>(
            shape, std::move(std::get<AlignedVector<uint16_t>>(source->bytes)), source->host_device);
        else if (isW4A4Format(source->type)) payload->matrix = std::make_shared<W4A4Tensor>(w4a4::WeightStorage::adopt(
            source->type == TensorType::W4A4 ? w4a4::Transform::Identity : w4a4::Transform::PaleySylvester,
            source->geometry.n, source->geometry.k, std::move(std::get<AlignedVector<uint8_t>>(source->bytes)),
            PhysicalMemoryOwner::ModelSourcePayload));
        else payload->matrix = TensorFactory::adoptQuantizedFinalStorage(source->type, shape,
            std::move(std::get<AlignedVector<uint8_t>>(source->bytes)), source->host_device);
        if (!payload->matrix || payload->matrix->size_bytes() != source->geometry.source_bytes)
            throw std::runtime_error("Planning sample receive allocation disagrees with native extent");
        return PlanningLoadedModelSample(std::move(payload));
    }

    PlanningLoadedModelSample PlanningModelSource::loadSample(const PlanningModelSampleRequest &request,
        const std::shared_ptr<PhysicalMemoryAuthority> &memory, DeviceId host_device) const
    {
        const auto geometry = sampleGeometry(request);
        if (!memory || !host_device.is_cpu())
            throw std::invalid_argument("Planning source sampling requires an admitted CPU allocator");
        // Share the original authority's source transaction with every model
        // manager. The final sample survives this scope, but no temporary input
        // allocation escapes it (the reader writes directly to final storage).
        const auto transaction = memory->acquireCPUWeightSourceTransaction(host_device);
        const auto original = storage_->loader.nativeWeightSourceGeometry(request.tensor_name);
        auto receiver = PlanningModelSampleReceiver::allocate(
            original.original_type, geometry, memory, host_device);
        NativeWeightSourceWindow range{
            .rows = geometry.n,
            .columns = geometry.k,
            .native_row_bytes = geometry.source_bytes / geometry.n};
        std::visit([&](const auto &selection) {
            using Selection = std::decay_t<decltype(selection)>;
            if constexpr (std::is_same_v<Selection, PlanningMatrixColumns>)
                range.first_column = selection.first;
            else if constexpr (std::is_same_v<Selection, PlanningExpertMatrix>)
                range.original_expert = selection.index;
            else if constexpr (std::is_same_v<Selection, PlanningMatrixRows>)
                range.first_row = selection.first;
        }, request.selection);
        const auto budget = configuredNativeWeightLoadStagingBudgetBytes();
        const size_t capacity = budget ? std::min(*budget, geometry.source_bytes) : geometry.source_bytes;
        auto destination = receiver.receiveBytes();
        const auto read_range = [&](const NativeWeightSourceWindow &selected) {
            forEachNativeWeightSourceWindow(original, selected, capacity, [&](const NativeWeightSourceWindow &window) {
                // Transport may split K, but admission below sees complete rows.
                const size_t offset = (window.first_row - selected.first_row) * selected.native_row_bytes +
                    ((window.first_column - selected.first_column) / original.native_group_elements) * original.native_group_bytes;
                if (offset > destination.size() || window.nativeBytes() > destination.size() - offset)
                    throw std::logic_error("Planning source window exceeds its final native sample");
                storage_->loader.readNativeWeightSourceWindow(request.tensor_name, window,
                    destination.subspan(offset, window.nativeBytes()));
            });
        };
        // The same source reader admits every original header before its first
        // selected read, using this unpublished destination. Receiver sealing
        // then validates the actual received matrix, including MPI corruption.
        read_range(range);
        return receiver.finish();
    }

    size_t maximumGGUFTensorPayloadBytes(const GGUFModel &model)
    {
        uint64_t maximum = 0;
        for (const auto &tensor : model.tensors)
        {
            if (inferWeightRole(tensor.name) == WeightRole::ImmutableRowTable) continue;
            maximum = std::max(maximum, tensor.size_bytes);
        }
        if (maximum == 0)
            throw std::invalid_argument("GPU upload admission requires a nonempty GGUF tensor manifest");
        if (maximum > std::numeric_limits<size_t>::max())
            throw std::overflow_error("Maximum GGUF tensor payload exceeds size_t");
        return static_cast<size_t>(maximum);
    }

    PlanningModelMetadata readPlanningModelMetadata(const std::string &model_path,
        const std::optional<SharedPredictorGGUFSource> &predictor)
    {
        return PlanningModelSource(model_path, predictor).metadata();
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
