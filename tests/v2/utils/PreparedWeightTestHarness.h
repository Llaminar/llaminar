#pragma once

#include "loaders/PreparedWeightStore.h"

#include <atomic>
#include <memory>
#include <string>

namespace llaminar2::test
{
    inline PreparedWeightKind preparedKindForDevice(DeviceId device)
    {
        if (device.is_cuda())
            return PreparedWeightKind::CudaInt8PackedGemm;
        if (device.is_rocm())
            return PreparedWeightKind::RocmInt8PackedGemm;
        return PreparedWeightKind::CpuPackedGemm;
    }

    inline uint64_t nextPreparedWeightTestBindingId()
    {
        static std::atomic<uint64_t> next_id{100000};
        return next_id.fetch_add(1, std::memory_order_relaxed);
    }

    inline WeightBinding makePreparedWeightTestBinding(
        TensorBase *tensor,
        DeviceId device,
        std::string canonical_name,
        ModelContextId model_id = ModelContextId{9900})
    {
        WeightBinding binding;
        binding.binding_id = nextPreparedWeightTestBindingId();
        binding.identity = makeSourceWeightIdentity(canonical_name, model_id, binding.binding_id);
        binding.residency.home_device = device;
        binding.residency.resident_device = device;
        binding.tensor = tensor;
        binding.immutable = true;
        return binding;
    }

    struct PreparedGemmFixture
    {
        ModelContextId model_id{9900};
        WeightBinding binding;
        std::unique_ptr<PreparedWeightStore> store;
        PreparedWeightRef ref;
    };

    inline PreparedGemmFixture makePreparedGemmFixture(
        TensorBase *tensor,
        DeviceId device,
        const std::string &canonical_name,
        ModelContextId model_id = ModelContextId{9900})
    {
        PreparedGemmFixture fixture;
        fixture.model_id = model_id;
        fixture.binding = makePreparedWeightTestBinding(tensor, device, canonical_name, model_id);
        fixture.store = std::make_unique<PreparedWeightStore>(model_id);
        fixture.ref = fixture.store->prepareGemm(fixture.binding);
        return fixture;
    }

    inline PreparedGemmFixture makeRegisteredPreparedGemmFixture(
        TensorBase *tensor,
        DeviceId device,
        const std::string &canonical_name,
        ModelContextId model_id = ModelContextId{9900})
    {
        PreparedGemmFixture fixture;
        fixture.model_id = model_id;
        fixture.binding = makePreparedWeightTestBinding(tensor, device, canonical_name, model_id);
        fixture.store = std::make_unique<PreparedWeightStore>(model_id);
        fixture.ref = fixture.store->registerPreparedForTest(
            fixture.binding,
            preparedKindForDevice(device),
            device);
        return fixture;
    }

    struct PreparedGateUpFixture
    {
        ModelContextId model_id{9900};
        WeightBinding gate_binding;
        WeightBinding up_binding;
        std::unique_ptr<PreparedWeightStore> store;
        PreparedWeightRef gate_ref;
        PreparedWeightRef up_ref;
    };

    inline PreparedGateUpFixture makePreparedGateUpFixture(
        TensorBase *gate,
        TensorBase *up,
        DeviceId device,
        int layer,
        ModelContextId model_id = ModelContextId{9900})
    {
        PreparedGateUpFixture fixture;
        fixture.model_id = model_id;
        fixture.gate_binding = makePreparedWeightTestBinding(
            gate,
            device,
            "blk." + std::to_string(layer) + ".ffn_gate.weight",
            model_id);
        fixture.up_binding = makePreparedWeightTestBinding(
            up,
            device,
            "blk." + std::to_string(layer) + ".ffn_up.weight",
            model_id);
        fixture.store = std::make_unique<PreparedWeightStore>(model_id);
        fixture.gate_ref = fixture.store->prepareGemm(fixture.gate_binding);
        fixture.up_ref = fixture.store->prepareGemm(fixture.up_binding);
        return fixture;
    }

    struct PreparedFFNFixture
    {
        ModelContextId model_id{9900};
        WeightBinding gate_binding;
        WeightBinding up_binding;
        WeightBinding down_binding;
        std::unique_ptr<PreparedWeightStore> store;
        PreparedWeightRef gate_ref;
        PreparedWeightRef up_ref;
        PreparedWeightRef down_ref;
    };

    inline PreparedFFNFixture makePreparedFFNFixture(
        TensorBase *gate,
        TensorBase *up,
        TensorBase *down,
        DeviceId device,
        int layer,
        const std::string &name_prefix = "ffn",
        ModelContextId model_id = ModelContextId{9900})
    {
        PreparedFFNFixture fixture;
        fixture.model_id = model_id;
        fixture.gate_binding = makePreparedWeightTestBinding(
            gate,
            device,
            "blk." + std::to_string(layer) + "." + name_prefix + "_gate.weight",
            model_id);
        fixture.up_binding = makePreparedWeightTestBinding(
            up,
            device,
            "blk." + std::to_string(layer) + "." + name_prefix + "_up.weight",
            model_id);
        fixture.down_binding = makePreparedWeightTestBinding(
            down,
            device,
            "blk." + std::to_string(layer) + "." + name_prefix + "_down.weight",
            model_id);
        fixture.store = std::make_unique<PreparedWeightStore>(model_id);
        fixture.gate_ref = fixture.store->prepareGemm(fixture.gate_binding);
        fixture.up_ref = fixture.store->prepareGemm(fixture.up_binding);
        fixture.down_ref = fixture.store->prepareGemm(fixture.down_binding);
        return fixture;
    }

    struct PreparedQKVFixture
    {
        ModelContextId model_id{9900};
        WeightBinding q_binding;
        WeightBinding k_binding;
        WeightBinding v_binding;
        std::unique_ptr<PreparedWeightStore> store;
        PreparedWeightRef q_ref;
        PreparedWeightRef k_ref;
        PreparedWeightRef v_ref;
    };

    inline PreparedQKVFixture makePreparedQKVFixture(
        TensorBase *q,
        TensorBase *k,
        TensorBase *v,
        DeviceId device,
        int layer,
        ModelContextId model_id = ModelContextId{9900})
    {
        PreparedQKVFixture fixture;
        fixture.model_id = model_id;
        fixture.q_binding = makePreparedWeightTestBinding(
            q,
            device,
            "blk." + std::to_string(layer) + ".attn_q.weight",
            model_id);
        fixture.k_binding = makePreparedWeightTestBinding(
            k,
            device,
            "blk." + std::to_string(layer) + ".attn_k.weight",
            model_id);
        fixture.v_binding = makePreparedWeightTestBinding(
            v,
            device,
            "blk." + std::to_string(layer) + ".attn_v.weight",
            model_id);
        fixture.store = std::make_unique<PreparedWeightStore>(model_id);
        fixture.q_ref = fixture.store->prepareGemm(fixture.q_binding);
        fixture.k_ref = fixture.store->prepareGemm(fixture.k_binding);
        fixture.v_ref = fixture.store->prepareGemm(fixture.v_binding);
        return fixture;
    }
}
