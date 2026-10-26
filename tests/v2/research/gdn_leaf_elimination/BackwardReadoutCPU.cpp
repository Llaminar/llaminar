/**
 * @file BackwardReadoutCPU.cpp
 * @brief Research-only comparison of leaves, backward queries and bounded suffixes.
 *
 * An immutable topology plan groups up to four queries sharing a checkpoint.
 * Leaf mode retains every internal rounded state; backward mode retains none;
 * combined mode retains a top prefix and contracts each remaining suffix.
 * Only accepted paths resolve missing states, in original FP32 update order.
 * Every allocation and topology decision precedes the timed interval. This is
 * not a production execution path: multi-update contraction changes rounding.
 */
#include "kernels/cpu/gdn/CPUAdjustedGDNReadout.h"
#include <immintrin.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

extern "C" int gdn_shared_group(int, int, int, int,
    const float *, const float *, const float *, const float *, const float *, const float *,
    float *, float *, float *, float *);
extern "C" int gdn_shared_resolve(int, int, const float *, const float *, const float *, float, float, float *);

namespace
{
    /** @brief Explicit experiments, never recovery choices. */
    enum class Mode { Leaves, Backward, Combined };
    /** @brief Literal y accumulation or a single-hop-compatible final FMA chain. */
    enum class Sum { LiteralY, NestedFMA };
    /** @brief At most four live outputs with one immutable checkpoint. */
    struct Group { int anchor; std::vector<int> nodes; };
    /** @brief All topology and ownership decisions, prepared once before timing. */
    struct Plan
    {
        std::vector<int> parent, depth, anchor, offset;
        std::vector<bool> retained;
        std::vector<std::vector<int>> suffix;
        std::vector<Group> prefixes, readouts;
        int terms = 0;

        /** @brief Validate topological ancestry and partition exact prefixes/suffixes. */
        Plan(Mode mode, int count, int cutoff, const int *parents)
            : parent(parents, parents+count), depth(count), anchor(count), offset(count),
              retained(count, false), suffix(count)
        {
            std::vector<bool> internal(count, false);
            for (int i=0; i<count; ++i)
            {
                if (parent[i] < -1 || parent[i] >= i) throw std::invalid_argument("ancestry");
                depth[i] = parent[i] < 0 ? 1 : depth[parent[i]]+1;
                if (parent[i] >= 0) internal[parent[i]] = true;
            }
            for (int i=0; i<count; ++i)
                retained[i] = internal[i] && (mode == Mode::Leaves ||
                    (mode == Mode::Combined && depth[i] <= cutoff));
            // Prefix groups are topological and physically contiguous, allowing
            // the existing fused internal kernel to publish directly in place.
            for (int p=-1; p<count; ++p)
            {
                for (int i=0; i<count;)
                {
                    if (!retained[i] || parent[i] != p) { ++i; continue; }
                    Group group{p,{i++}};
                    while (i<count && group.nodes.size()<4 && retained[i] && parent[i]==p)
                        group.nodes.push_back(i++);
                    prefixes.push_back(std::move(group));
                }
            }
            for (int i=0; i<count; ++i)
            {
                offset[i]=terms;
                if (retained[i]) continue;
                int current=i;
                while (current>=0 && !retained[current])
                {
                    suffix[i].push_back(current);
                    current=parent[current];
                }
                anchor[i]=current;
                terms += static_cast<int>(suffix[i].size());
            }
            for (int p=-1; p<count; ++p)
            {
                Group group{p,{}};
                for (int i=0; i<count; ++i)
                    if (!retained[i] && anchor[i]==p)
                    {
                        group.nodes.push_back(i);
                        if (group.nodes.size()==4)
                        {
                            readouts.push_back(std::move(group));
                            group={p,{}};
                        }
                    }
                if (!group.nodes.empty()) readouts.push_back(std::move(group));
            }
        }
    };

    /**
     * @brief Reuse each checkpoint load across four fixed-order query reductions.
     * @tparam Rows Physical live query count, with no padded consumers.
     * @param plan Prepared immutable suffix ownership.
     * @param group Query indices sharing this state.
     * @param state Checkpoint [dk,dv], stored in key-major order.
     * @param x Contracted queries [N,dk].
     * @param coefficients Packed path coefficients, newest to oldest.
     * @param y Literal backward value sums [N,dv], used only by LiteralY.
     * @param values Original candidate values [N,dv].
     * @param output Readout destination [N,dv].
     * @param dk Key width; each lane reduces increasing keys.
     * @param dv Value width, a positive multiple of eight.
     * @param sum Chosen explicit FP32 addition program.
     */
    template<int Rows>
    __attribute__((noinline))
    void action(const Plan &plan, const Group &group, const float *state,
        const float *x, const float *coefficients, const float *y, const float *values,
        float *output, int dk, int dv, Sum sum)
    {
        for (int vi=0; vi<dv; vi+=8)
        {
            __m256 result[Rows];
#pragma GCC unroll 4
            for (int r=0; r<Rows; ++r) result[r]=_mm256_setzero_ps();
            for (int j=0; j<dk; ++j)
            {
                const __m256 s=_mm256_loadu_ps(state+j*dv+vi);
#pragma GCC unroll 4
                for (int r=0; r<Rows; ++r)
                    result[r]=_mm256_fmadd_ps(s,_mm256_set1_ps(x[group.nodes[r]*dk+j]),result[r]);
            }
#pragma GCC unroll 4
            for (int r=0; r<Rows; ++r)
            {
                const int i=group.nodes[r];
                if (sum==Sum::LiteralY)
                    result[r]=_mm256_add_ps(result[r],_mm256_loadu_ps(y+i*dv+vi));
                else
                    for (int t=static_cast<int>(plan.suffix[i].size())-1; t>=0; --t)
                        result[r]=_mm256_fmadd_ps(_mm256_set1_ps(coefficients[plan.offset[i]+t]),
                            _mm256_loadu_ps(values+plan.suffix[i][t]*dv+vi),result[r]);
                _mm256_storeu_ps(output+i*dv+vi,result[r]);
            }
        }
    }

    /** @brief Verify all candidates with the explicit prefix/suffix arithmetic. */
    bool verify(const Plan &plan, Sum sum, int dk, int dv, const float *root,
        const float *q, const float *k, const float *v, const float *decay, const float *beta,
        float *output, float *states, float *x, float *coefficients, float *y)
    {
        const int matrix=dk*dv;
        for (const Group &group: plan.prefixes)
        {
            const int i=group.nodes.front(), rows=static_cast<int>(group.nodes.size());
            std::array<float,4> eta{};
            if (gdn_shared_group(1,rows,dk,dv,group.anchor<0 ? root : states+group.anchor*matrix,
                q+i*dk,k+i*dk,v+i*dv,decay+i,beta+i,x+i*dk,eta.data(),output+i*dv,states+i*matrix)!=1)
                return false;
        }
        for (int i=0; i<static_cast<int>(plan.parent.size()); ++i)
        {
            if (plan.retained[i]) continue;
            std::memcpy(x+i*dk,q+i*dk,dk*sizeof(float));
            if (sum==Sum::LiteralY) std::fill(y+i*dv,y+(i+1)*dv,0.0f);
            for (int t=0; t<static_cast<int>(plan.suffix[i].size()); ++t)
            {
                const int j=plan.suffix[i][t];
                const float c=llaminar2::prepareAdjustedGDNReadout(x+i*dk,k+j*dk,decay[j],beta[j],dk,x+i*dk);
                coefficients[plan.offset[i]+t]=c;
                if (sum==Sum::LiteralY)
                    for (int vi=0; vi<dv; vi+=8)
                        _mm256_storeu_ps(y+i*dv+vi,_mm256_fmadd_ps(_mm256_set1_ps(c),
                            _mm256_loadu_ps(v+j*dv+vi),_mm256_loadu_ps(y+i*dv+vi)));
            }
        }
        for (const Group &group: plan.readouts)
        {
            const float *state=group.anchor<0 ? root : states+group.anchor*matrix;
            switch (group.nodes.size())
            {
                case 1: action<1>(plan,group,state,x,coefficients,y,v,output,dk,dv,sum); break;
                case 2: action<2>(plan,group,state,x,coefficients,y,v,output,dk,dv,sum); break;
                case 3: action<3>(plan,group,state,x,coefficients,y,v,output,dk,dv,sum); break;
                case 4: action<4>(plan,group,state,x,coefficients,y,v,output,dk,dv,sum); break;
                default: return false;
            }
        }
        return true;
    }

    /** @brief Resolve only missing accepted updates, retaining rounded dense state. */
    bool commit(const Plan &plan, int endpoint, int dk, int dv, const float *root,
        const float *k, const float *v, const float *decay, const float *beta,
        const float *states, float *destination, float *scratch)
    {
        const int matrix=dk*dv;
        if (endpoint<0 || plan.retained[endpoint])
        {
            std::memcpy(destination,endpoint<0 ? root : states+endpoint*matrix,matrix*sizeof(float));
            return true;
        }
        const auto &path=plan.suffix[endpoint];
        const float *parent=plan.anchor[endpoint]<0 ? root : states+plan.anchor[endpoint]*matrix;
        for (int t=static_cast<int>(path.size())-1; t>=0; --t)
        {
            const int i=path[t];
            float *next=t==0 ? destination : scratch+(t%2)*matrix;
            if (gdn_shared_resolve(dk,dv,parent,k+i*dk,v+i*dv,decay[i],beta[i],next)!=1) return false;
            parent=next;
        }
        return true;
    }
}

/**
 * @brief Time two verify/accept rounds and terminal publication on fixed operands.
 * @param mode Leaves=0, Backward=1, Combined=2.
 * @param accumulation LiteralY=0, NestedFMA=1. NestedFMA matches a one-hop leaf.
 * @param iterations Positive transaction repetitions; preparation is untimed.
 * @param rows Live nodes in [1,256].
 * @param dk Key width in [1,512].
 * @param dv Value width in [8,512], multiple of eight.
 * @param cutoff Retained prefix depth for Combined; ignored by other modes.
 * @param first First-round accepted endpoint, -1 means reject all.
 * @param last Second-round accepted endpoint, -1 means reject all.
 * @param parents Topological parent indices [rows]; -1 denotes checkpoint.
 * @param root Initial immutable dense checkpoint [dk,dv].
 * @param q Prepared queries [2,rows,dk].
 * @param k Prepared keys [2,rows,dk].
 * @param v Raw values [2,rows,dv].
 * @param decay Rounded decay gates [2,rows].
 * @param beta Rounded write gates [2,rows].
 * @param output Both rounds' output [2,rows,dv].
 * @param states Reusable prefix-state bank [rows,dk,dv]. Unretained slots untouched.
 * @param resolved First accepted dense state [dk,dv].
 * @param terminal Second accepted dense state [dk,dv].
 * @param x Query workspace [rows,dk].
 * @param y Literal accumulation workspace [rows,dv].
 * @param nanoseconds Mean duration of the complete two-round transaction.
 * @return 1 on success, -1 for any invalid contract or failed native operation.
 */
extern "C" int gdn_backward_lifecycle(int mode, int accumulation, int iterations,
    int rows, int dk, int dv, int cutoff, int first, int last, const int *parents,
    const float *root, const float *q, const float *k, const float *v,
    const float *decay, const float *beta, float *output, float *states,
    float *resolved, float *terminal, float *x, float *y, double *nanoseconds)
{
    if (mode<0 || mode>2 || accumulation<0 || accumulation>1 || iterations<1 ||
        rows<1 || rows>256 || dk<1 || dk>512 || dv<8 || dv>512 || dv%8 || cutoff<0 ||
        first< -1 || first>=rows || last< -1 || last>=rows || !parents || !root || !q ||
        !k || !v || !decay || !beta || !output || !states || !resolved || !terminal || !x || !y || !nanoseconds)
        return -1;
    try
    {
        const Plan plan(static_cast<Mode>(mode),rows,cutoff,parents);
        std::vector<float> coefficients(plan.terms), scratch(2*dk*dv);
        const Sum sum=static_cast<Sum>(accumulation);
        const auto start=std::chrono::steady_clock::now();
        for (int repeat=0; repeat<iterations; ++repeat)
        {
            if (!verify(plan,sum,dk,dv,root,q,k,v,decay,beta,output,states,x,coefficients.data(),y) ||
                !commit(plan,first,dk,dv,root,k,v,decay,beta,states,resolved,scratch.data()) ||
                !verify(plan,sum,dk,dv,resolved,q+rows*dk,k+rows*dk,v+rows*dv,decay+rows,beta+rows,
                    output+rows*dv,states,x,coefficients.data(),y) ||
                !commit(plan,last,dk,dv,resolved,k+rows*dk,v+rows*dv,decay+rows,beta+rows,states,terminal,scratch.data()))
                return -1;
        }
        *nanoseconds=std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-start).count()/iterations;
        return 1;
    }
    catch (...) { return -1; }
}
