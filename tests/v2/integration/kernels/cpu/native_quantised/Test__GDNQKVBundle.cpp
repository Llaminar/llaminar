/**
 * @file Test__GDNQKVBundle.cpp
 * @brief Full-QFN-shape synthetic four-original-source CPU component regression.
 *
 * The real prepared store, native Q8 and FP16 control engines, GDN stage and
 * exact PMA workspace are used. No model file, GPU initialization, serving
 * topology, graph admission or end-to-end numerical certificate is claimed.
 */
#include "../../../../utils/GDNQKVBundleTestHarness.h"
#include "utils/OpenMPUtils.h"

using namespace llaminar2;

/** @brief Exercise compact serial/grouped CPU rows and explicitly host-counted QKV only. */
TEST(GDNQKVBundle, CPUOriginalSourcesCompactBundle)
{
    CPUDeviceContext context(DeviceId::cpu(), omp_get_max_threads());
    test::GDNQKVBundleFixture fixture(DeviceId::cpu(), context, nullptr, nullptr,
        2560, {10240, 6144, 48, 48});
    fixture.prove();
}
