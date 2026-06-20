#include <gtest/gtest.h>

#include <cstdlib>
#include <optional>
#include <string>
#include <utility>

#include "execution/compute_stages/IComputeStage.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "tensors/Tensors.h"
#include "utils/DebugEnv.h"

namespace llaminar2
{
    namespace
    {
        class ScopedEnvVar
        {
        public:
            ScopedEnvVar(const char *name, const char *value)
                : name_(name)
            {
                if (const char *existing = std::getenv(name))
                    previous_ = std::string(existing);
                ::setenv(name, value, 1);
                mutableDebugEnv().reload();
            }

            ~ScopedEnvVar()
            {
                if (previous_)
                    ::setenv(name_.c_str(), previous_->c_str(), 1);
                else
                    ::unsetenv(name_.c_str());
                mutableDebugEnv().reload();
            }

            ScopedEnvVar(const ScopedEnvVar &) = delete;
            ScopedEnvVar &operator=(const ScopedEnvVar &) = delete;

        private:
            std::string name_;
            std::optional<std::string> previous_;
        };

        class CountingFP32Tensor final : public FP32Tensor
        {
        public:
            explicit CountingFP32Tensor(const std::vector<size_t> &shape)
                : FP32Tensor(shape)
            {
            }

            const float *fp32_data() const override
            {
                ++fp32_data_calls;
                return data();
            }

            mutable int fp32_data_calls = 0;
        };

        class TraceProbeStage final : public IComputeStage
        {
        public:
            TraceProbeStage() : IComputeStage(DeviceId::cpu()) {}

            bool execute(IDeviceContext * /*ctx*/) override { return true; }
            ComputeStageType type() const override { return ComputeStageType::COPY; }
            bool supportsBackend(ComputeBackendType /*backend*/) const override { return true; }

            void traceInputForTest(const std::string &name, const ITensor *tensor) const
            {
                traceInput(name, tensor);
            }

            void traceOutputForTest(const std::string &name, const ITensor *tensor) const
            {
                traceOutput(name, tensor);
            }

            void traceIntermediateForTest(const std::string &name, const float *data, size_t count) const
            {
                traceIntermediate(name, data, count);
            }

        protected:
            StageDumpInfo buildDumpInfoImpl() const override { return StageDumpInfo{}; }
        };

        void fillTensor(CountingFP32Tensor &tensor)
        {
            float *data = tensor.mutable_data();
            for (size_t i = 0; i < tensor.numel(); ++i)
                data[i] = static_cast<float>(i + 1);
        }
    }

    TEST(Test__ComputeStageTraceGraphCapture, TracePayloadUsesFP32DataOutsideGraphCapture)
    {
        ScopedEnvVar trace("LLAMINAR_TRACE_STAGES", "1");
        ScopedEnvVar shapes("LLAMINAR_TRACE_SHAPES", "1");
        ScopedEnvVar samples("LLAMINAR_TRACE_SAMPLE_COUNT", "4");

        CountingFP32Tensor tensor({1, 4});
        fillTensor(tensor);

        TraceProbeStage stage;
        stage.traceInputForTest("x", &tensor);
        stage.traceOutputForTest("y", &tensor);

        EXPECT_EQ(tensor.fp32_data_calls, 2);
    }

    TEST(Test__ComputeStageTraceGraphCapture, TracePayloadSkipsFP32DataDuringGraphCapture)
    {
        ScopedEnvVar trace("LLAMINAR_TRACE_STAGES", "1");
        ScopedEnvVar shapes("LLAMINAR_TRACE_SHAPES", "1");
        ScopedEnvVar samples("LLAMINAR_TRACE_SAMPLE_COUNT", "4");

        CountingFP32Tensor tensor({1, 4});
        fillTensor(tensor);
        float intermediate[2] = {1.0f, 2.0f};

        TraceProbeStage stage;
        {
            GraphCaptureGuard guard;
            stage.traceInputForTest("x", &tensor);
            stage.traceOutputForTest("y", &tensor);
            stage.traceIntermediateForTest("scratch", intermediate, 2);
        }

        EXPECT_EQ(tensor.fp32_data_calls, 0);
    }
} // namespace llaminar2
