/**
 * @file CPUGDNStateTestUtils.h
 * @brief Explicit CPU epoch-state fixtures and a diagnostic dense interpreter.
 *
 * Mathematical/GPU comparisons observe a materialized copy while live CPU state
 * retains its pending history. Grouped tests compare the entire encoded state.
 * These helpers never enter production and never flush a live kernel bank.
 */
#pragma once
#include "kernels/CPUGDNStateLayout.h"
#include "kernels/CPUSharedGDNSnapshots.h"
#include <algorithm>
#include <cmath>
#include <span>
#include <vector>

namespace llaminar2
{
    /**
     * @brief Own compact production capture storage for a device-free CPU fixture.
     *
     * Dense observation vectors remain separate: tests materialize each candidate
     * through the production restore API and compare the complete encoded bytes.
     */
    struct CPUGDNCompactTestCapture
    {
        std::vector<float> slots; ///< Immutable checkpoint/log transaction bytes.

        /**
         * @brief Admit and bind one exact request/head geometry before execution.
         * @tparam Kernel CPU production recurrence implementation.
         * @param kernel Kernel receiving both capture and compact arena bindings.
         * @param heads Value-head count per request.
         * @param dk Live key width.
         * @param dv Live value width.
         * @param width Candidate capacity per request.
         * @param requests Number of independently owned request banks.
         */
        template<class Kernel>
        CPUGDNCompactTestCapture(Kernel &kernel, int heads, int dk, int dv,
            int width, int requests = 1)
        {
            const CPUGDNStateLayout layout(heads, dk, dv);
            slots.resize(static_cast<size_t>(requests) * heads *
                CPUSharedGDNSnapshots::headFloats(layout, width), -777.0f);
            kernel.bindVerifierStateCaptureWorkspace(slots.data(), requests * width, layout.floats());
            kernel.bindCPUSharedStateCapture(slots.data(),slots.size(),requests*width);
        }
    };

    /**
     * @brief Seed a compact mock transaction with distinct canonical head bytes.
     * @param slots Admitted compact per-head arena for one request.
     * @param heads Value-head count.
     * @param dk Key width.
     * @param dv Value width.
     * @param rows Live row count within the initial epoch, from one through seven.
     * @param pattern Distinguishes independently owned stage transactions.
     * @return Terminal candidate materialized independently for byte comparisons.
     */
    inline std::vector<float> seedCPUGDNCompactTestTransaction(float *slots,
        int heads, int dk, int dv, int rows, float pattern)
    {
        if (!slots || rows <= 0 || rows >= CPUGDNStateLayout::kPeriod)
            throw std::invalid_argument("CPU compact mock transaction extent");
        const CPUGDNStateLayout layout(heads, dk, dv);
        const size_t span = CPUSharedGDNSnapshots::headFloats(layout, rows);
        std::vector<float> expected(layout.floats(), 0.0f), initial(layout.headFloats(), 0.0f), raw(layout.recordFloats());
        for (int h = 0; h < heads; ++h)
        {
            for (int i = 0; i < layout.matrixFloats(); ++i) initial[1 + i] = pattern + h * 100 + i;
            CPUSharedGDNSnapshots transaction(layout, slots + static_cast<size_t>(h) * span, rows);
            transaction.begin(initial.data(), rows);
            for (int row = 0; row < rows; ++row)
            {
                for (int i = 0; i < layout.recordFloats(); ++i) raw[i] = pattern + h * 100 + row * 20 + i;
                transaction.append(row, raw.data());
            }
            transaction.complete();
            transaction.materialize(rows - 1, expected.data() + static_cast<size_t>(h) * layout.headFloats());
        }
        return expected;
    }

    /**
     * @brief Retire fixture history after filling checkpoint bytes with a pattern.
     * @param state Complete encoded bank with the supplied geometry.
     * @param heads Value-head count.
     * @param dk Key width.
     * @param dv Value width.
     */
    inline void clearCPUGDNTestHistory(std::span<float> state, int heads, int dk, int dv)
    {
        const CPUGDNStateLayout layout(heads,dk,dv);
        if (state.size()!=static_cast<size_t>(layout.floats())) throw std::invalid_argument("CPU GDN fixture extent");
        for (int h=0;h<heads;++h)
        {
            float *head=state.data()+static_cast<size_t>(h)*layout.headFloats();
            head[0]=0; std::fill(layout.record(head,0),head+layout.headFloats(),0.0f);
        }
    }

    /**
     * @brief Encode an independently owned dense initial checkpoint at phase zero.
     * @param dense Initial key-major matrices, exactly heads*dk*dv elements.
     * @param heads Value-head count.
     * @param dk Key width.
     * @param dv Value width.
     * @return Complete state with zero inactive history and phase.
     */
    inline std::vector<float> encodeCPUGDNTestState(const std::vector<float> &dense, int heads, int dk, int dv)
    {
        if (dense.size()!=static_cast<size_t>(heads)*dk*dv) throw std::invalid_argument("CPU GDN dense fixture extent");
        const CPUGDNStateLayout layout(heads,dk,dv);
        std::vector<float> result(layout.floats(),0.0f);
        for (int h=0;h<heads;++h)
            std::copy_n(dense.data()+static_cast<size_t>(h)*dk*dv,dk*dv,result.data()+static_cast<size_t>(h)*layout.headFloats()+1);
        return result;
    }

    /**
     * @brief Observe logical dense state through an independent scalar interpreter.
     * @param state Complete immutable encoded state.
     * @param heads Value-head count.
     * @param dk Key width.
     * @param dv Value width.
     * @return Materialized copy; the original history and phase remain unchanged.
     */
    inline std::vector<float> materializeCPUGDNTestState(const std::vector<float> &state, int heads, int dk, int dv)
    {
        const CPUGDNStateLayout layout(heads,dk,dv);
        if (state.size()!=static_cast<size_t>(layout.floats()) || !layout.valid(state.data())) throw std::invalid_argument("CPU GDN observed fixture extent/phase");
        std::vector<float> result(static_cast<size_t>(heads)*dk*dv);
        for (int h=0;h<heads;++h)
        {
            const float *head=state.data()+static_cast<size_t>(h)*layout.headFloats();
            float *dense=result.data()+static_cast<size_t>(h)*dk*dv;
            std::copy_n(head+1,dk*dv,dense);
            for (int p=0;p<static_cast<int>(head[0]);++p)
            {
                const float *raw=layout.record(head,p);
                for (int vi=0;vi<dv;++vi)
                {
                    float projection=0;
                    for (int j=0;j<dk;++j)
                    {
                        float &s=dense[j*dv+vi];s*=raw[dk+dv];projection=std::fma(s,raw[j],projection);
                    }
                    const float delta=(raw[dk+vi]-projection)*raw[dk+dv+1];
                    for (int j=0;j<dk;++j) dense[j*dv+vi]=std::fma(raw[j],delta,dense[j*dv+vi]);
                }
            }
        }
        return result;
    }
}
