/**
 * @file EpochReadoutCPU.cpp
 * @brief Canonical fixed-epoch backward arithmetic shared by serial and trees.
 *
 * State is a checkpoint, a bounded raw-update log, and its live count. Only
 * reaching the fixed period materializes a checkpoint. Acceptance and cache
 * serialization preserve the log and never introduce a rounding boundary.
 * A scalar serial interpreter and two grouped algorithms share this contract.
 * The grouped hybrid caches rounded prefix states only to construct required
 * checkpoints; those auxiliary states never replace a readout's canonical base.
 * This research implementation does not change production state ownership.
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

namespace
{
    /** @brief One versioned research-state layout; inactive slots are zero. */
    struct Layout
    {
        int period, dk, dv, matrix, record, size;
        /** @brief Reject invalid geometry before deriving any physical extent. */
        Layout(int p,int k,int v):period(p),dk(k),dv(v),matrix(0),record(0),size(0)
        {
            if (p<1 || p>32 || k<1 || k>512 || v<8 || v>512 || v%8)
                throw std::invalid_argument("epoch geometry");
            matrix=k*v; record=k+v+2; size=1+matrix+p*record;
        }
        /** @brief Validate the immutable state's live raw-history count. */
        int count(const float *state) const
        {
            if (!state || !std::isfinite(state[0]) || state[0]<0 || state[0]>=period ||
                state[0]!=std::floor(state[0])) throw std::invalid_argument("epoch state count");
            return static_cast<int>(state[0]);
        }
        /** @brief Locate one admitted raw-update record in serialized storage. */
        float *raw(float *state,int i) const { return state+1+matrix+i*record; }
        /** @brief Locate one immutable raw-update record. */
        const float *raw(const float *state,int i) const { return state+1+matrix+i*record; }
    };
    /** @brief Borrowed live operands; no second state authority is created. */
    struct Update { const float *key; const float *value; float decay; float beta; };
    /** @brief Raw candidate arrays plus the authoritative initial history. */
    struct Inputs
    {
        const Layout &layout;
        int nodes;
        const float *root,*q,*k,*v,*decay,*beta;
        /** @brief Resolve a candidate or initial-history index into its owner. */
        Update at(int id) const
        {
            if (id<nodes) return {k+id*layout.dk,v+id*layout.dv,decay[id],beta[id]};
            const float *record=layout.raw(root,id-nodes);
            return {record,record+layout.dk,record[layout.dk+layout.dv],record[layout.dk+layout.dv+1]};
        }
    };
    /**
     * @brief Preserve original rounded update order, allowing exact in-place state.
     * Each eight-value tile finishes its entire key projection before changing
     * any element in that tile. Tiles own disjoint columns, so alias is safe.
     */
    void advance(const Layout &l,const float *source,float *destination,Update u)
    {
        for (int vi=0;vi<l.dv;vi+=8)
        {
            const __m256 decay=_mm256_set1_ps(u.decay);
            __m256 projection=_mm256_setzero_ps();
            for (int j=0;j<l.dk;++j)
                projection=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_loadu_ps(source+j*l.dv+vi),decay),
                    _mm256_set1_ps(u.key[j]),projection);
            const __m256 delta=_mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(u.value+vi),projection),_mm256_set1_ps(u.beta));
            for (int j=0;j<l.dk;++j)
                _mm256_storeu_ps(destination+j*l.dv+vi,_mm256_fmadd_ps(_mm256_set1_ps(u.key[j]),delta,
                    _mm256_mul_ps(_mm256_loadu_ps(source+j*l.dv+vi),decay)));
        }
    }
    /** @brief Copy one raw update into the sole serialized history authority. */
    void storeUpdate(const Layout &l,float *target,Update u)
    {
        std::memcpy(target,u.key,l.dk*sizeof(float));
        std::memcpy(target+l.dk,u.value,l.dv*sizeof(float));
        target[l.dk+l.dv]=u.decay; target[l.dk+l.dv+1]=u.beta;
    }
    /**
     * @brief Independent scalar serial oracle for the new fixed-epoch contract.
     * Workspace is preallocated. Serial history construction never calls the
     * grouped topology, contraction helper or SIMD projection implementation.
     */
    void serial(const Layout &l,const float *query,Update update,float *state,float *output,
        float *x,float *coefficients)
    {
        const int count=l.count(state)+1;
        storeUpdate(l,l.raw(state,count-1),update);
        std::memcpy(x,query,l.dk*sizeof(float));
        for (int p=count-1;p>=0;--p)
        {
            const float *record=l.raw(state,p);
            float dot=0;
            for (int j=0;j<l.dk;++j) dot=std::fma(record[j],x[j],dot);
            const float c=record[l.dk+l.dv+1]*dot;
            coefficients[p]=c;
            for (int j=0;j<l.dk;++j) x[j]=record[l.dk+l.dv]*std::fma(-c,record[j],x[j]);
        }
        for (int vi=0;vi<l.dv;++vi)
        {
            float value=0;
            for (int j=0;j<l.dk;++j) value=std::fma(state[1+j*l.dv+vi],x[j],value);
            for (int p=0;p<count;++p) value=std::fma(coefficients[p],l.raw(state,p)[l.dk+vi],value);
            output[vi]=value;
        }
        if (count==l.period)
        {
            // Independent scalar checkpoint interpreter, including every
            // rounding boundary. It does not reuse the candidate advance body.
            for (int p=0;p<count;++p)
            {
                const float *record=l.raw(state,p);
                for (int vi=0;vi<l.dv;++vi)
                {
                    float projection=0;
                    for (int j=0;j<l.dk;++j)
                    {
                        float &s=state[1+j*l.dv+vi];
                        s*=record[l.dk+l.dv];
                        projection=std::fma(s,record[j],projection);
                    }
                    const float delta=(record[l.dk+vi]-projection)*record[l.dk+l.dv+1];
                    for (int j=0;j<l.dk;++j)
                    {
                        float &s=state[1+j*l.dv+vi];
                        s=std::fma(record[j],delta,s);
                    }
                }
            }
            std::fill(l.raw(state,0),state+l.size,0.0f);
            state[0]=0;
        }
        else state[0]=static_cast<float>(count);
    }
    /** @brief All query groups and checkpoint dependencies for a fixed root phase. */
    struct Plan
    {
        std::vector<int> parent,depth,anchor;
        std::vector<bool> boundary,internal,auxiliary;
        std::vector<std::vector<int>> path,groups;
        /** @brief Derive boundaries exclusively from accepted position and period. */
        Plan(const Layout &l,int n,int phase,const int *parents)
            :parent(parents,parents+n),depth(n),anchor(n),boundary(n),internal(n,false),auxiliary(n,false),path(n),groups(n+1)
        {
            for (int i=0;i<n;++i)
            {
                const int p=parent[i];
                if (p< -1 || p>=i) throw std::invalid_argument("epoch ancestry");
                depth[i]=p<0 ? 1 : depth[p]+1;
                boundary[i]=(phase+depth[i])%l.period==0;
                anchor[i]=p<0 ? -1 : boundary[p] ? p : anchor[p];
                if (p>=0) internal[p]=true;
                path[i].push_back(i);
                if (p>=0 && !boundary[p]) path[i].insert(path[i].end(),path[p].begin(),path[p].end());
                else if (p<0) for (int h=phase-1;h>=0;--h) path[i].push_back(n+h);
                if (path[i].size()>static_cast<size_t>(l.period)) throw std::logic_error("epoch boundary missing");
                groups[anchor[i]+1].push_back(i);
            }
            // Auxiliary rounded prefixes may accelerate only canonical flushes.
            // They never become a readout anchor and never change its arithmetic.
            for (int i=0;i<n;++i)
                if (boundary[i] && internal[i])
                    for (int node:path[i]) if (node<n) auxiliary[node]=true;
        }
    };
    /** @brief Query contraction workspace and checkpoint banks admitted before timing. */
    struct Workspace
    {
        std::vector<float> checkpoints,x,c,history_state;
        /** @brief Reserve every live candidate surface once. */
        Workspace(const Layout &l,int n):checkpoints(n*l.matrix),x(n*l.dk),c(n*l.period),history_state(l.matrix) {}
    };
    /** @brief One fixed SIMD action with explicit independent row reductions. */
    template<int Rows>
    __attribute__((noinline))
    void project(const Layout &l,const Plan &plan,const Inputs &in,Workspace &work,
        const int *ids,const float *base,float *output)
    {
        for (int vi=0;vi<l.dv;vi+=8)
        {
            __m256 sums[Rows];
#pragma GCC unroll 4
            for (int r=0;r<Rows;++r) sums[r]=_mm256_setzero_ps();
            for (int j=0;j<l.dk;++j)
            {
                const __m256 s=_mm256_loadu_ps(base+j*l.dv+vi);
#pragma GCC unroll 4
                for (int r=0;r<Rows;++r)
                    sums[r]=_mm256_fmadd_ps(s,_mm256_set1_ps(work.x[ids[r]*l.dk+j]),sums[r]);
            }
#pragma GCC unroll 4
            for (int r=0;r<Rows;++r)
            {
                const int i=ids[r];
                for (int t=static_cast<int>(plan.path[i].size())-1;t>=0;--t)
                    sums[r]=_mm256_fmadd_ps(_mm256_set1_ps(work.c[i*l.period+t]),
                        _mm256_loadu_ps(in.at(plan.path[i][t]).value+vi),sums[r]);
                _mm256_storeu_ps(output+i*l.dv+vi,sums[r]);
            }
        }
    }
    /** @brief Materialize a required canonical boundary with fixed update ordering. */
    void flushPath(const Layout &l,const Plan &plan,const Inputs &in,int i,
        const float *base,float *destination)
    {
        const float *source=base;
        for (auto it=plan.path[i].rbegin();it!=plan.path[i].rend();++it)
        {
            advance(l,source,destination,in.at(*it));
            source=destination;
        }
    }
    /** @brief Execute all outputs and only the checkpoint state each descendant needs. */
    void verify(const Layout &l,const Plan &plan,const Inputs &in,Workspace &work,float *output,bool hybrid)
    {
        const int n=in.nodes,phase=l.count(in.root);
        bool history_ready=false;
        for (int a=-1;a<n;++a)
        {
            const auto &ids=plan.groups[a+1];
            if (ids.empty()) continue;
            const float *base=a<0 ? in.root+1 : work.checkpoints.data()+a*l.matrix;
            for (int i:ids)
            {
                float *x=work.x.data()+i*l.dk;
                std::memcpy(x,in.q+i*l.dk,l.dk*sizeof(float));
                for (int t=0;t<static_cast<int>(plan.path[i].size());++t)
                {
                    const Update u=in.at(plan.path[i][t]);
                    work.c[i*l.period+t]=llaminar2::prepareAdjustedGDNReadout(x,u.key,u.decay,u.beta,l.dk,x);
                }
            }
            for (size_t begin=0;begin<ids.size();begin+=4)
                switch (std::min<size_t>(4,ids.size()-begin))
                {
                    case 1: project<1>(l,plan,in,work,ids.data()+begin,base,output); break;
                    case 2: project<2>(l,plan,in,work,ids.data()+begin,base,output); break;
                    case 3: project<3>(l,plan,in,work,ids.data()+begin,base,output); break;
                    case 4: project<4>(l,plan,in,work,ids.data()+begin,base,output); break;
                }
            if (!hybrid)
            {
                for (int i:ids) if (plan.boundary[i] && plan.internal[i])
                    flushPath(l,plan,in,i,base,work.checkpoints.data()+i*l.matrix);
            }
            else for (int i:ids) if (plan.auxiliary[i])
            {
                const int p=plan.parent[i];
                const float *source=p<0 ? in.root+1 : work.checkpoints.data()+p*l.matrix;
                if (p<0 && phase>0)
                {
                    if (!history_ready)
                    {
                        const float *s=in.root+1;
                        for (int h=0;h<phase;++h)
                        {
                            advance(l,s,work.history_state.data(),in.at(n+h));
                            s=work.history_state.data();
                        }
                        history_ready=true;
                    }
                    source=work.history_state.data();
                }
                advance(l,source,work.checkpoints.data()+i*l.matrix,in.at(i));
            }
        }
    }
    /** @brief Serialize an accepted canonical state without inventing a flush boundary. */
    void commit(const Layout &l,const Plan &plan,const Inputs &in,const Workspace &work,
        int endpoint,float *destination,bool hybrid)
    {
        if (endpoint<0) { std::memcpy(destination,in.root,l.size*sizeof(float)); return; }
        const int a=plan.anchor[endpoint];
        const float *base=a<0 ? in.root+1 : work.checkpoints.data()+a*l.matrix;
        destination[0]=0;
        std::fill(l.raw(destination,0),destination+l.size,0.0f);
        if (plan.boundary[endpoint])
        {
            if (plan.internal[endpoint])
                std::memcpy(destination+1,work.checkpoints.data()+endpoint*l.matrix,l.matrix*sizeof(float));
            else if (hybrid && plan.parent[endpoint]>=0 && plan.auxiliary[plan.parent[endpoint]])
                advance(l,work.checkpoints.data()+plan.parent[endpoint]*l.matrix,destination+1,in.at(endpoint));
            else flushPath(l,plan,in,endpoint,base,destination+1);
        }
        else
        {
            std::memcpy(destination+1,base,l.matrix*sizeof(float));
            int h=0;
            for (auto it=plan.path[endpoint].rbegin();it!=plan.path[endpoint].rend();++it)
                storeUpdate(l,l.raw(destination,h++),in.at(*it));
            destination[0]=static_cast<float>(h);
        }
    }
}

/** @brief Return exact serialized state extent, or -1 for unsupported geometry. */
extern "C" int gdn_epoch_state_size(int period,int dk,int dv)
{
    try { return Layout(period,dk,dv).size; } catch (...) { return -1; }
}

/**
 * @brief Execute one independent canonical serial step with prepared operands.
 * @param period Fixed checkpoint interval, 1..32; part of arithmetic identity.
 * @param dk Key width, 1..512.
 * @param dv Value width, 8..512 and divisible by eight.
 * @param q Prepared query [dk].
 * @param k Prepared key [dk].
 * @param v Raw value [dv].
 * @param decay Rounded state decay.
 * @param beta Rounded write strength.
 * @param state In/out complete serialized checkpoint/history state.
 * @param output Canonical output [dv].
 * @return 1 on success, -1 for a rejected contract.
 */
extern "C" int gdn_epoch_serial(int period,int dk,int dv,const float *q,const float *k,
    const float *v,float decay,float beta,float *state,float *output)
{
    if (!q || !k || !v || !state || !output) return -1;
    try
    {
        const Layout l(period,dk,dv);
        std::vector<float> x(dk),c(period);
        serial(l,q,{k,v,decay,beta},state,output,x.data(),c.data());
        return 1;
    }
    catch (...) { return -1; }
}

/**
 * @brief Time two complete canonical tree transactions and state serialization.
 * @param mode 0=serial snapshot oracle; 1=backward; 2=backward plus shared flush prefixes.
 * @param period Fixed arithmetic checkpoint interval, independent of tree shape.
 * @param iterations Positive repeated two-round lifecycles.
 * @param nodes Number of live candidate nodes, 1..256.
 * @param dk Key width.
 * @param dv Value width.
 * @param first First-round accepted node, -1 rejects all candidates.
 * @param last Second-round accepted node, -1 rejects all candidates.
 * @param parents Topological ancestry [nodes].
 * @param root Immutable complete initial canonical state.
 * @param q Prepared queries [2,nodes,dk].
 * @param k Prepared keys [2,nodes,dk].
 * @param v Raw values [2,nodes,dv].
 * @param decay Rounded decay gates [2,nodes].
 * @param beta Rounded write gates [2,nodes].
 * @param output Both rounds' canonical output [2,nodes,dv].
 * @param resolved Complete first accepted state.
 * @param terminal Complete second accepted state, including unflushed history.
 * @param nanoseconds Mean native transaction latency, excluding admission/allocation.
 * @return 1 on success, -1 for invalid state, geometry or ancestry.
 */
extern "C" int gdn_epoch_lifecycle(int mode,int period,int iterations,int nodes,int dk,int dv,
    int first,int last,const int *parents,const float *root,const float *q,const float *k,
    const float *v,const float *decay,const float *beta,float *output,float *resolved,float *terminal,double *nanoseconds)
{
    if (mode<0 || mode>2 || iterations<1 || nodes<1 || nodes>256 || first< -1 || first>=nodes ||
        last< -1 || last>=nodes || !parents || !root || !q || !k || !v || !decay || !beta ||
        !output || !resolved || !terminal || !nanoseconds) return -1;
    try
    {
        const Layout l(period,dk,dv);
        const int phase=l.count(root);
        const Plan first_plan(l,nodes,phase,parents);
        const int next_phase=first<0 ? phase : (phase+first_plan.depth[first])%period;
        const Plan second_plan(l,nodes,next_phase,parents);
        Workspace workspace(l,nodes);
        std::vector<float> serial_states(mode==0 ? nodes*l.size : 0),serial_x(dk),serial_c(period);
        const auto run=[&](const Plan &plan,const Inputs &in,int selected,float *out,float *state)
        {
            if (mode==0)
            {
                for (int i=0;i<nodes;++i)
                {
                    float *s=serial_states.data()+i*l.size;
                    const float *p=plan.parent[i]<0 ? in.root : serial_states.data()+plan.parent[i]*l.size;
                    std::memcpy(s,p,l.size*sizeof(float));
                    serial(l,in.q+i*dk,in.at(i),s,out+i*dv,serial_x.data(),serial_c.data());
                }
                std::memcpy(state,selected<0 ? in.root : serial_states.data()+selected*l.size,l.size*sizeof(float));
            }
            else
            {
                verify(l,plan,in,workspace,out,mode==2);
                commit(l,plan,in,workspace,selected,state,mode==2);
            }
        };
        const auto start=std::chrono::steady_clock::now();
        for (int repeat=0;repeat<iterations;++repeat)
        {
            run(first_plan,{l,nodes,root,q,k,v,decay,beta},first,output,resolved);
            run(second_plan,{l,nodes,resolved,q+nodes*dk,k+nodes*dk,v+nodes*dv,decay+nodes,beta+nodes},
                last,output+nodes*dv,terminal);
        }
        *nanoseconds=std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-start).count()/iterations;
        return 1;
    }
    catch (...) { return -1; }
}
