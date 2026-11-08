/**
 * @file W4A4CPUReplayDriver.cpp
 * @brief Bounded device-free transport into the actual native CPU W4A4 implementation.
 *
 * The Python replay gate authenticates immutable recorded inputs and expected
 * outputs. This driver executes production preparation/projection and returns
 * actual intermediate/output bytes. It has no expected-value generator or
 * alternate arithmetic. The canonical test session owns this CPU-only child.
 */
#include "W4A4ReplayTransport.h"
#include "kernels/cpu/gemm/CPUW4A4Arithmetic.h"
#include <iostream>
#include <omp.h>

/**
 * @brief Execute one record through ordinary or existing-team CPU arithmetic.
 * @param argc Must name the input and output files only.
 * @param argv Owned temporary transport paths, never the immutable trace destination.
 * @return Zero only after complete native execution and result publication.
 */
int main(int argc, char **argv)
{
    try
    {
        using namespace llaminar2::w4a4;
        using namespace test_transport;
        if (argc != 3) throw std::runtime_error("Expected replay input and output paths");
        const auto record = read(argv[1]);
        if (record.entry() != Entry::Arithmetic && record.entry() != Entry::ExistingTeam)
            throw std::runtime_error("Arithmetic replay cannot certify a tensor-engine entry");
        const auto source = SourceView::admit(record.transform(), record.n(), record.k(), record.weights);
        if (!source) throw std::runtime_error(source.error().message());
        const auto counts = cpuWorkspaceCounts(source->contract(), record.m());
        auto packed = storage<ActivationBlock>(counts.blocks);
        auto ranges = storage<ActivationRange>(counts.ranges);
        auto row_ranges = storage<float>(counts.row_ranges);
        auto butterfly = storage<float>(counts.butterfly_floats);
        auto rotated = storage<float>(counts.transformed_floats);
        if (static_cast<std::size_t>(record.m()) > kMaximumBytes / sizeof(float) / record.n())
            throw std::runtime_error("Replay output geometry exceeds the fixture bound");
        auto output = storage<float>(static_cast<std::size_t>(record.m()) * record.n());
        CPUActivationBuffers buffers{packed, ranges, row_ranges, butterfly, rotated};
        omp_set_dynamic(0);
        const auto execute = [&] {
            const auto prepared = prepareCPUActivations(source->contract(), record.input, record.m(), record.k(), buffers);
            projectCPU(*source, prepared, output, record.n());
        };
        if (record.entry() == Entry::ExistingTeam)
        {
#pragma omp parallel
            { execute(); }
        }
        else execute();
        std::ofstream stream(argv[2], std::ios::binary | std::ios::trunc);
        writeExact(stream, record.header.data(), sizeof(record.header));
        writeExact(stream, record.transform() == Transform::PaleySylvester ? rotated.data() : record.input.data(),
                   record.input.size() * sizeof(float));
        writeExact(stream, packed.data(), packed.size() * sizeof(ActivationBlock));
        writeExact(stream, output.data(), output.size() * sizeof(float));
        stream.flush();
        if (!stream) throw std::runtime_error("Replay output flush failed");
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "W4A4 CPU replay rejected: " << error.what() << '\n';
        return 1;
    }
}
