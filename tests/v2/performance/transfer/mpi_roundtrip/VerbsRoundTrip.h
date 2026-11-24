/**
 * @file VerbsRoundTrip.h
 * @brief Explicit regular-MR and CUDA DMA-BUF-MR data planes for the isolated GPU experiment.
 *
 * MPI exchanges cold endpoint metadata only. RC WRITE_WITH_IMM moves exactly the
 * live payload and produces receiver completion before GPU publication. Both
 * registration modes use identical QPs, striping, GPU work and completion logic.
 * DMA-BUF never substitutes ibv_reg_mr after an export/import error. Allocations
 * outlive this borrowed transport; QPs retire before MRs, FDs and original pages.
 */
#pragma once
#include <cuda.h>
#include <infiniband/verbs.h>
#include <mpi.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <dlfcn.h>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <arpa/inet.h>
#include <unistd.h>
#include "RoundTripContract.h"

namespace llaminar2::mpi_roundtrip {
/** @brief Borrowed original allocation and live payload views; no second byte authority. */
struct VerbsBufferView {
    void* hostBase;
    void* gpuBase;
    float* payload;
    std::size_t allocationBytes;
};
/** @brief Required native-call success or immediate failure of the selected experiment. */
inline void verbsRequire(bool good, std::string_view what) {
    if (!good) checkNativeStatus(errno ? errno : -1, what);
}
/** @brief Exact scope owner for one native object with a fallible destruction API. */
template<class T, auto Destroy> struct VerbsDelete {
    /** @brief Destruction failure is fatal before dependent backing can retire. */
    void operator()(T* object) const noexcept { if (object && Destroy(object)) std::abort(); }
};
/** @brief Two retained CUDA exports shared across the selected NIC protection domains. */
class DmaBufExports {
    void* driver_ = nullptr;
    std::array<int, 2> fds_{-1, -1};
public:
    /** @brief Export complete page-aligned original allocations with the proven default mapping flag. */
    DmaBufExports(VerbsBufferView tx, VerbsBufferView rx) {
        driver_ = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!driver_) throw std::runtime_error(dlerror());
        try {
            const auto api = reinterpret_cast<decltype(&cuMemGetHandleForAddressRange)>(
                dlsym(driver_, "cuMemGetHandleForAddressRange"));
            if (!api) throw std::runtime_error("driver DMA-BUF export API missing");
            std::array views{tx, rx};
            const auto page = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
            for (std::size_t i = 0; i < views.size(); ++i) {
                if (!page || reinterpret_cast<std::uintptr_t>(views[i].gpuBase) % page || views[i].allocationBytes % page)
                    throw std::invalid_argument("DMA-BUF range must be native-page aligned");
                int exported = -1;
                const auto status = api(&exported, reinterpret_cast<CUdeviceptr>(views[i].gpuBase),
                    views[i].allocationBytes, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
                if (status != CUDA_SUCCESS || exported < 0)
                    throw std::runtime_error("selected CUDA DMA-BUF export failed: " + std::to_string(status));
                fds_[i] = exported;
            }
        } catch (...) { retire(); throw; }
    }
    DmaBufExports(const DmaBufExports&) = delete;
    /** @brief FDs outlive every imported MR and are retired before original allocation release. */
    ~DmaBufExports() { retire(); }
    /** @return Original transmit FD, not a separately registered host pointer. */
    int tx() const { return fds_[0]; }
    /** @return Original receive FD. */
    int rx() const { return fds_[1]; }
private:
    /** @brief Release a partially or completely constructed export closure exactly once. */
    void retire() noexcept {
        for (int& fd : fds_) if (fd >= 0) { if (close(fd)) std::abort(); fd = -1; }
        if (driver_) { dlclose(driver_); driver_ = nullptr; }
    }
};
/** @brief One RC rail, original NIC registration, and persistent completion state. */
class VerbsRail {
    std::unique_ptr<ibv_context, VerbsDelete<ibv_context, ibv_close_device>> context_;
    std::unique_ptr<ibv_pd, VerbsDelete<ibv_pd, ibv_dealloc_pd>> pd_;
    std::unique_ptr<ibv_cq, VerbsDelete<ibv_cq, ibv_destroy_cq>> cq_;
    std::unique_ptr<ibv_mr, VerbsDelete<ibv_mr, ibv_dereg_mr>> txMr_, rxMr_;
    std::unique_ptr<ibv_qp, VerbsDelete<ibv_qp, ibv_destroy_qp>> qp_;
    std::uintptr_t txAddress_ = 0, remoteAddress_ = 0;
    std::uint32_t remoteKey_ = 0, bytes_ = 0;
    unsigned receivePosted_ = 0, epoch_ = 0;
    TransportReceiptFrontier received_;
    bool sent_ = false;
    std::array<ibv_wc, 2> completions_{};
public:
    /** @brief Admit one live slice and connect its actual RC endpoint through cold MPI metadata. */
    VerbsRail(const std::string& spec, VerbsRegistration registration, const DmaBufExports* exports,
              VerbsBufferView tx, VerbsBufferView rx, std::size_t offsetWords, std::size_t words,
              std::size_t totalWords, int peer) {
        const auto colon = spec.find(':'), last = spec.rfind(':');
        if (colon == std::string::npos || colon == last) throw std::invalid_argument("rail must be HCA:PORT:GID_INDEX");
        const auto name = spec.substr(0, colon);
        const auto portNumber = positive(spec.substr(colon + 1, last - colon - 1));
        const auto gidText = spec.substr(last + 1);
        // GID zero is a valid native index, unlike the positive port/count grammar.
        const auto gidIndex = gidText == "0" ? 0 : positive(gidText);
        if (portNumber > UINT8_MAX || gidIndex > UINT8_MAX) throw std::invalid_argument("native port/GID field overflow");
        std::ifstream type("/sys/class/infiniband/" + name + "/ports/" + std::to_string(portNumber) +
                           "/gid_attrs/types/" + std::to_string(gidIndex));
        std::string gidType; std::getline(type, gidType);
        if (gidType != "RoCE v2") throw std::invalid_argument("selected rail must observe a RoCE v2 GID");
        int deviceCount = 0; ibv_device** devices = ibv_get_device_list(&deviceCount);
        verbsRequire(devices != nullptr, "enumerate RDMA devices");
        ibv_device* selected = nullptr;
        for (int i = 0; i < deviceCount; ++i) if (name == ibv_get_device_name(devices[i])) selected = devices[i];
        if (selected) context_.reset(ibv_open_device(selected));
        ibv_free_device_list(devices);
        verbsRequire(context_ != nullptr, "open exact HCA " + name);
        ibv_port_attr port{};
        checkNativeStatus(ibv_query_port(context_.get(), static_cast<std::uint8_t>(portNumber), &port), "query actual port");
        if (port.state != IBV_PORT_ACTIVE || !words || words > UINT32_MAX / sizeof(float) || words * sizeof(float) > port.max_msg_sz)
            throw std::invalid_argument("inactive port or live slice outside native message limits");
        bytes_ = static_cast<std::uint32_t>(words * sizeof(float));
        ibv_gid gid{};
        checkNativeStatus(ibv_query_gid(context_.get(), static_cast<std::uint8_t>(portNumber), static_cast<int>(gidIndex), &gid), "query original GID");
        pd_.reset(ibv_alloc_pd(context_.get())); verbsRequire(pd_ != nullptr, "allocate PD");
        cq_.reset(ibv_create_cq(context_.get(), 4, nullptr, nullptr, 0)); verbsRequire(cq_ != nullptr, "allocate bounded CQ");
        const int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
        const auto registerView = [&](VerbsBufferView view, int fd) {
            decltype(txMr_) mr(registration == VerbsRegistration::DmaBuf
                ? ibv_reg_dmabuf_mr(pd_.get(), 0, view.allocationBytes, reinterpret_cast<std::uintptr_t>(view.hostBase), fd, access)
                : ibv_reg_mr(pd_.get(), view.hostBase, view.allocationBytes, access));
            verbsRequire(mr != nullptr, "selected native MR registration");
            validateNativeMr(registration, reinterpret_cast<std::uintptr_t>(mr->addr), mr->length,
                             reinterpret_cast<std::uintptr_t>(view.hostBase), view.allocationBytes);
            return mr;
        };
        if ((registration == VerbsRegistration::DmaBuf) != (exports != nullptr)) throw std::logic_error("incomplete explicit registration contract");
        txMr_ = registerView(tx, exports ? exports->tx() : -1);
        rxMr_ = registerView(rx, exports ? exports->rx() : -1);
        ibv_qp_init_attr init{}; init.send_cq = cq_.get(); init.recv_cq = cq_.get(); init.qp_type = IBV_QPT_RC;
        init.cap.max_send_wr = 1; init.cap.max_recv_wr = 2; init.cap.max_send_sge = 1; init.cap.max_recv_sge = 1;
        qp_.reset(ibv_create_qp(pd_.get(), &init)); verbsRequire(qp_ != nullptr, "create bounded RC QP");
        ibv_qp_attr state{}; state.qp_state = IBV_QPS_INIT; state.pkey_index = 0;
        state.port_num = static_cast<std::uint8_t>(portNumber); state.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
        checkNativeStatus(ibv_modify_qp(qp_.get(), &state, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS), "QP INIT");
        std::array<std::uint64_t, 10> local{}, remote{};
        for (unsigned i = 0; i < sizeof(gid.raw); ++i) local[i / 8] |= std::uint64_t(gid.raw[i]) << ((i % 8) * 8);
        local[2] = qp_->qp_num; local[3] = rxMr_->rkey;
        local[4] = reinterpret_cast<std::uintptr_t>(rx.payload + offsetWords);
        local[5] = words; local[6] = offsetWords; local[7] = totalWords; local[8] = port.active_mtu; local[9] = port.lid;
        if (MPI_Sendrecv(local.data(), local.size(), MPI_UINT64_T, peer, 71, remote.data(), remote.size(), MPI_UINT64_T,
                         peer, 71, MPI_COMM_WORLD, MPI_STATUS_IGNORE) != MPI_SUCCESS)
            throw std::runtime_error("original peer endpoint exchange failed");
        if (remote[5] != words || remote[6] != offsetWords || remote[7] != totalWords)
            throw std::invalid_argument("asymmetric live slice geometry");
        state = {}; state.qp_state = IBV_QPS_RTR;
        state.path_mtu = static_cast<ibv_mtu>(std::min<std::uint64_t>(port.active_mtu, remote[8]));
        state.dest_qp_num = static_cast<std::uint32_t>(remote[2]); state.rq_psn = 0;
        state.max_dest_rd_atomic = 1; state.min_rnr_timer = 1;
        state.ah_attr.is_global = 1; state.ah_attr.dlid = static_cast<std::uint16_t>(remote[9]);
        state.ah_attr.port_num = static_cast<std::uint8_t>(portNumber);
        for (unsigned i = 0; i < sizeof(gid.raw); ++i) state.ah_attr.grh.dgid.raw[i] = static_cast<std::uint8_t>(remote[i / 8] >> ((i % 8) * 8));
        state.ah_attr.grh.sgid_index = static_cast<std::uint8_t>(gidIndex); state.ah_attr.grh.hop_limit = 1;
        checkNativeStatus(ibv_modify_qp(qp_.get(), &state, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
            IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER), "QP RTR");
        state = {}; state.qp_state = IBV_QPS_RTS; state.timeout = 14; state.retry_cnt = 3; state.rnr_retry = 3;
        state.sq_psn = 0; state.max_rd_atomic = 1;
        checkNativeStatus(ibv_modify_qp(qp_.get(), &state, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
            IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC), "QP RTS");
        txAddress_ = reinterpret_cast<std::uintptr_t>(tx.payload + offsetWords);
        remoteAddress_ = remote[4]; remoteKey_ = static_cast<std::uint32_t>(remote[3]);
        std::cerr << "verbs rail=" << spec << " registration=" << (exports ? "dmabuf" : "ordinary-mr")
                  << " offset_bytes=" << offsetWords * sizeof(float) << " live_bytes=" << bytes_
                  << " tx_allocation=" << tx.allocationBytes << " rx_allocation=" << rx.allocationBytes << '\n';
    }
    /** @brief Prepost only the two metadata credits needed to avoid a completion/repost race. */
    void prepare(unsigned iterations) { for (unsigned i = 0; i < std::min(2U, iterations); ++i) receive(); }
    /** @brief Start the next fixed live transaction after both previous CQ receipts were consumed. */
    void beginEpoch(unsigned epoch) { sent_ = false; epoch_ = epoch; }
    /** @brief Submit exactly this rail's live bytes; readiness was acquired from the original GPU. */
    void send(unsigned epoch) {
        ibv_sge sge{}; sge.addr = txAddress_; sge.length = bytes_; sge.lkey = txMr_->lkey;
        ibv_send_wr wr{}, *bad = nullptr; wr.wr_id = std::uint64_t(epoch) * 2 + 1; wr.sg_list = &sge; wr.num_sge = 1;
        wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM; wr.send_flags = IBV_SEND_SIGNALED;
        wr.imm_data = htonl(epoch); wr.wr.rdma.remote_addr = remoteAddress_; wr.wr.rdma.rkey = remoteKey_;
        checkNativeStatus(ibv_post_send(qp_.get(), &wr, &bad), "submit live RDMA write");
    }
    /** @brief Consume exact native send/receive completions and replenish bounded metadata credit. */
    void poll(unsigned epoch, unsigned iterations) {
        const int count = ibv_poll_cq(cq_.get(), completions_.size(), completions_.data());
        checkNativeStatus(count >= 0 ? 0 : count, "poll original CQ");
        for (int i = 0; i < count; ++i) {
            const auto& completion = completions_[i];
            if (completion.status != IBV_WC_SUCCESS)
                throw std::runtime_error(std::string("RDMA completion failure: ") + ibv_wc_status_str(completion.status));
            if (completion.wr_id == std::uint64_t(epoch) * 2 + 1 && completion.opcode == IBV_WC_RDMA_WRITE && !sent_) sent_ = true;
            else if (completion.wr_id == 2 && completion.opcode == IBV_WC_RECV_RDMA_WITH_IMM &&
                     (completion.wc_flags & IBV_WC_WITH_IMM)) {
                const unsigned observed = ntohl(completion.imm_data);
                if (observed > iterations || observed > epoch + 1) throw std::runtime_error("remote write exceeds retained receive credit");
                // The peer may deliver its next payload before our previous send
                // CQE arrives. Keep that completion rather than dropping/resetting it.
                received_.observe(observed);
                if (receivePosted_ < iterations) receive();
            } else throw std::runtime_error("duplicate, stale or foreign RDMA completion");
        }
    }
    /** @return Whether the NIC has retired its original source-buffer read. */
    bool sent() const { return sent_; }
    /** @return Whether this destination slice has its original remote-write completion. */
    bool received() const { return received_.ready(epoch_); }
private:
    /** @brief Post an immediate-only receive: no payload bounce buffer or capacity-sized exchange. */
    void receive() {
        ibv_recv_wr wr{}, *bad = nullptr; wr.wr_id = 2;
        checkNativeStatus(ibv_post_recv(qp_.get(), &wr, &bad), "post metadata receive credit"); ++receivePosted_;
    }
};
/** @brief Persistent direct-verbs follower; only device readiness and native completion cross the boundary. */
class VerbsRoundTrip {
    std::unique_ptr<DmaBufExports> exports_;
    std::vector<std::unique_ptr<VerbsRail>> rails_;
public:
    /** @brief Partition live FP32 words across selected rails before capture; each used rail owns work. */
    VerbsRoundTrip(VerbsRegistration registration, const std::vector<std::string>& specs,
                  VerbsBufferView tx, VerbsBufferView rx, Geometry geometry, int peer) {
        if (specs.empty()) throw std::invalid_argument("explicit verbs rails required");
        if (registration == VerbsRegistration::DmaBuf) exports_ = std::make_unique<DmaBufExports>(tx, rx);
        const LiveRailPlan plan(geometry, specs.size());
        rails_.reserve(plan.activeRails());
        for (std::size_t i = 0; i < specs.size(); ++i)
            if (std::find(specs.begin(), specs.begin() + i, specs[i]) != specs.begin() + i)
                throw std::invalid_argument("duplicate explicit rail");
        for (std::size_t i = 0; i < plan.activeRails(); ++i) {
            const auto slice = plan.slice(i);
            rails_.push_back(std::make_unique<VerbsRail>(specs[i], registration, exports_.get(), tx, rx,
                slice.offsetWords, slice.words, geometry.words, peer));
        }
    }
    /** @brief Failed distributed progress cannot release NIC/GPU backing through ordinary unwinding. */
    ~VerbsRoundTrip() {
        if (std::uncaught_exceptions()) { MPI_Abort(MPI_COMM_WORLD, 1); std::_Exit(1); }
    }
    /** @brief Prepost initial bounded receives before either GPU can publish an epoch. */
    void prepare(unsigned iterations) { for (auto& rail : rails_) rail->prepare(iterations); }
    /** @brief Execute fixed admitted transactions without hot allocations or CUDA API calls. */
    template<class Control> void progress(Control& control, unsigned iterations) {
        for (unsigned epoch = 1; epoch <= iterations; ++epoch) {
            for (auto& rail : rails_) rail->beginEpoch(epoch);
            bool posted = false, txPublished = false, rxPublished = false;
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
            while (!txPublished || !rxPublished) {
                if (!posted && std::atomic_ref<unsigned long long>(control.txReady).load(std::memory_order_acquire) >= epoch) {
                    for (auto& rail : rails_) rail->send(epoch);
                    posted = true;
                }
                bool sent = true, received = true;
                for (auto& rail : rails_) { rail->poll(epoch, iterations); sent &= rail->sent(); received &= rail->received(); }
                if (posted && sent && !txPublished) {
                    std::atomic_ref<unsigned long long>(control.txRetired).store(epoch, std::memory_order_release); txPublished = true;
                }
                if (received && !rxPublished) {
                    std::atomic_ref<unsigned long long>(control.rxReady).store(epoch, std::memory_order_release); rxPublished = true;
                }
                if (std::chrono::steady_clock::now() > deadline) {
                    std::atomic_ref<unsigned long long>(control.abort).store(1, std::memory_order_release);
                    throw std::runtime_error("30-second native RDMA progress deadline");
                }
            }
        }
    }
};
} // namespace llaminar2::mpi_roundtrip
