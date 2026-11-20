/**
 * @file SelectedDescriptorObserver.inc
 * @brief Original graph reader declaring its exact selected descriptor dependency.
 */
extern "C" bool launchSelectedDescriptorObserver(const void*,MoEDecodeDescriptorPublication,int,void*,void*);

constexpr std::array kSelectedRoleNames{MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_GATE_DESC_TABLE,
    MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_UP_DESC_TABLE,MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_DOWN_DESC_TABLE};
constexpr std::size_t kSelectedRoleBytes=3*sizeof(DeviceNativeQuantisedMatrixDesc);

/** @brief Consume only local selected descriptors on the graph's ordered stream. */
class SelectedDescriptorObserver final : public IComputeStage,public IWorkspaceConsumer,
    public IMoESelectedDescriptorConsumer {
public:
    /** @brief Borrow original table owner and a fixture-owned observable output. */
    SelectedDescriptorObserver(DeviceId device,DeviceMoERuntimeTable& runtime,int experts,int top_k,INT32Tensor& output)
        :IComputeStage(device),read_{"shared",&runtime,0},experts_(experts),top_k_(top_k),output_(output),runtime_layer_(runtime.deviceLayerState(0)){}
    /** @return The exact selected descriptor read wired into the fixture graph. */
    std::span<const MoESelectedDescriptorRead> selectedDescriptorReads() const noexcept override {
        return {&read_,1};
    }
    /** @brief Borrow the original selected role storage prepared by the producer. */
    void bindWorkspace(DeviceWorkspaceManager* workspace) override {workspace_=workspace;}
    /** @return Existing role storage is owned by the producer. */
    WorkspaceRequirements getWorkspaceRequirements(int,int,int) const override {return {};}
    /** @return Whether the borrowed original workspace is bound. */
    bool hasWorkspace() const override {return workspace_!=nullptr;}
    /** @return The workspace that contains the original role tables. */
    DeviceWorkspaceManager* getWorkspace() const override {return workspace_;}
    /** @return ROCm is the implemented replay backend. */
    bool supportsBackend(ComputeBackendType backend) const override {return backend==ComputeBackendType::GPU_ROCM;}
    /** @return The role-table reader is a participant-local projection operation. */
    ComputeStageType type() const override {return ComputeStageType::MOE_PROJECTION_PHASE;}
    /** @brief Record one reader on the same exact device stream after the producer. */
    bool execute(IDeviceContext* context) override {
        require(context&&context->deviceId()==device()&&workspace_,"exact selected descriptor reader bindings");
        const auto table=[&](const auto name) {
            return static_cast<DeviceNativeQuantisedMatrixDesc*>(workspace_->getBuffer(name));
        };
        const MoEDecodeDescriptorPublication publication{table(kSelectedRoleNames[0]),table(kSelectedRoleNames[1]),
            table(kSelectedRoleNames[2]),experts_};
        return launchSelectedDescriptorObserver(runtime_layer_,publication,
            top_k_,output_.gpu_data_ptr(),requireGPUStream());
    }
protected:
    /** @return The fixture verifies its private result after capture completion. */
    StageDumpInfo buildDumpInfoImpl() const override {return {};}
private:
    const MoESelectedDescriptorRead read_;
    const int experts_,top_k_;
    INT32Tensor& output_;
    const DeviceMoELayerRuntime* runtime_layer_;
    DeviceWorkspaceManager* workspace_=nullptr;
};
