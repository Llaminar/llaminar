/**
 * @file RealRetainedTPSourceWindow.inc
 * @brief Original TP source windows for one local retained-versus-staged graph.
 *
 * This local benchmark retains one authentic source shard and uses the original
 * WeightManager preparation. It makes no claim about the distributed graph or
 * the fork's separately partitioned shared expert.
 */
/** @brief Select only source roles materialized by the destination authority. */
enum class RealRetainedWindowRoles { IncludingRouter, NativeProjections };
enum class RealRetainedWindowExtent { OriginalShard, WholeLayer };

class RealRetainedTPSourceWindow final {
public:
    /** @brief Load original metadata and resolve participant cuts with the planner policy. */
    RealRetainedTPSourceWindow(const char* path, int layer, int participant, const std::vector<DeviceId>& members)
        : layer_(layer), participant_(participant)
    {
        const int width=static_cast<int>(members.size());
        require((width==1||width==2||width==4||width==8)&&participant>=0&&participant<width,
            "real TP continuation coordinate");
        loader_.setUseMmap(true);
        require(loader_.loadModel(path),"load original split GGUF for TP source slices");
        const auto gate=geometryOf(kRoutedNames[0]),up=geometryOf(kRoutedNames[1]);
        const auto down=geometryOf(kRoutedNames[2]);
        gate_=gate.original_type;
        down_=down.original_type;
        require(gate_==up.original_type&&
            ((gate_==TensorType::Q4_K&&(down_==TensorType::Q5_1||down_==TensorType::Q8_0))||
             (gate_==TensorType::Q5_K&&down_==TensorType::Q8_0)),
            "original supported QFN routed source tuple");
        const auto& original=loader_.getModel();
        const auto profile=ModelMemoryProfile::fromGGUF(original);
        require(layer>=0&&layer<profile.mainLayerCount(),"original main layer coordinate");
        const auto declaration=resolveQfnTPDeclaration(original,TensorParallelMainPlacement::FullMainRankLocal,
            MTPRuntimeConfig{},DenseParallelPolicy::TensorParallel,members);
        const auto assignment=declaration.partition(members,{},profile.d_ff,profile.vocab_size);
        const auto shared=assignment.forRank(participant);
        shared_first_=shared.d_ff_start;
        shared_width_=shared.d_ff_count;
        const auto manifest=buildMoEOverlayLayerWeightManifestFromLoader(loader_,1,profile.expert_count,layer);
        routed_=MoEExpertShardCohort::gpuIntervals(manifest.front(),width,layer).at(participant);
        source_routed_width_=manifest.front().projectionForRole(ExpertTierWeightProjection::Gate).N;
        source_shared_width_=profile.d_ff;
        require(shared_width_>0&&shared_width_%32==0&&shared_first_%32==0,
            "original shared projection cut must preserve native Q8 blocks");
    }

    /** @return Geometry of the original source interval, realized by one test participant. */
    OriginalFFNFixtureGeometry geometry() const
    {
        return {.hidden=2560,.routed=routed_.columns,.shared=shared_width_,.experts=512,.top_k=10,
            .router_source=OriginalFFNRouterSource::Dense};
    }

    /** @return Original source format identities used by the ordinary preparation owner. */
    int originalLayer() const noexcept {return layer_;}

    /** @return The original gate, down and Q8 shared tuple for this model layer. */
    SourceFormat format() const { return sourceFormat(gate_, down_); }

    /** @brief Bind both real GGUF encodings to their canonical native test source factories.
     * @param gate_type Original routed gate/up dtype. @param down_type Original down dtype.
     * @return A source fixture with the same independent gate/up, down and shared encodings. */
    static SourceFormat sourceFormat(TensorType gate_type, TensorType down_type)
    {
        const auto& entries=quantizedVerifierFormats();
        const auto pick=[&](TensorType type) {
            const auto it=std::find_if(entries.begin(),entries.end(),[&](const auto& entry){return entry.tensor_type==type;});
            require(it!=entries.end(),"real TP4 source format exists in production native fixture inventory");
            return &*it;
        };
        const auto* gate=pick(gate_type);
        return {.name="QFN_TP_source_window",.type=gate_type,
            .identity=ExpertWeightFormat::nativeQuantised({gate->source_codebook_id,gate->source_is_superblock,true}),
            .create=gate->create,.shared_quantized=pick(TensorType::Q8_0),.down_quantized=pick(down_type)};
    }

    /** @brief Copy unchanged declared natives to the test component's one-layer source. */
    void copyTo(PlanningGGUFFixture& destination, OriginalFFNFixtureGeometry shape, const SourceFormat& selected,
        RealRetainedWindowRoles roles=RealRetainedWindowRoles::IncludingRouter,
        RealRetainedWindowExtent scope=RealRetainedWindowExtent::OriginalShard)
    {
        const bool full=scope==RealRetainedWindowExtent::WholeLayer;
        require(shape.hidden==2560&&shape.routed==(full?source_routed_width_:routed_.columns)&&
            shape.shared==(full?source_shared_width_:shared_width_)&&
            selected.type==gate_&&selected.down_quantized&&selected.down_quantized->tensor_type==down_,
            "exact original model source coordinates reach the execution owner");
        const auto copy=[&](const char* target,TensorType type,std::vector<std::size_t> original_shape,
            std::size_t experts,std::size_t first_row,std::size_t rows,std::size_t first_column,std::size_t columns) {
            const auto original=originalName(target);
            const auto source=loader_.nativeWeightSourceGeometry(original);
            const bool original_column_vector=original_shape==std::vector<std::size_t>{2560}&&
                source.original_shape==std::vector<std::size_t>{2560,1};
            require(source.original_type==type&&(source.original_shape==original_shape||original_column_vector)&&
                source.native_group_elements&&first_column%source.native_group_elements==0&&
                columns%source.native_group_elements==0,"native source window format, geometry and exact groups");
            const auto row_bytes=columns/source.native_group_elements*source.native_group_bytes;
            std::vector<std::uint8_t> bytes(rows*row_bytes);
            for(std::size_t expert=0;expert<experts;++expert) {
                const NativeWeightSourceWindow window{.original_expert=expert,.selected_expert_ordinal=expert,
                    .first_row=first_row,.rows=rows,.first_column=first_column,
                    .columns=columns,.native_row_bytes=row_bytes};
                loader_.readNativeWeightSourceWindow(original,window,
                    {reinterpret_cast<std::byte*>(bytes.data()),bytes.size()});
                destination.writePayload(target,expert*bytes.size(),{bytes.data(),bytes.size()});
            }
        };
        const auto start=full?0u:std::size_t(routed_.first_column),width=std::size_t(shape.routed);
        const auto shared=full?0u:std::size_t(shared_first_),shared_width=std::size_t(shape.shared);
        copy(kRoutedNames[0],gate_,{2560,640,512},512,start,width,0,2560);
        copy(kRoutedNames[1],gate_,{2560,640,512},512,start,width,0,2560);
        copy(kRoutedNames[2],down_,{640,2560,512},512,0,2560,start,width);
        copy(kSharedNames[0],TensorType::Q8_0,{2560,640},1,shared,shared_width,0,2560);
        copy(kSharedNames[1],TensorType::Q8_0,{2560,640},1,shared,shared_width,0,2560);
        copy(kSharedNames[2],TensorType::Q8_0,{640,2560},1,0,2560,shared,shared_width);
        if(roles==RealRetainedWindowRoles::IncludingRouter) {
            copy(kSharedNames[3],TensorType::FP32,{2560},1,0,1,0,2560);
            copy(kSharedNames[4],TensorType::FP32,{2560,512},1,0,512,0,2560);
        }
        std::osyncstream(std::cout)<<"QFN_RETAINED_REAL_TP_WINDOW,layer="<<layer_<<",participant="<<participant_
            <<",gate="<<tensorTypeName(gate_)<<",down="<<tensorTypeName(down_)
            <<",routed_first="<<start<<",F="<<width<<",shared_first="<<shared<<",S="<<shared_width
            <<",source_extent="<<(full?"whole_layer":"original_shard")<<'\n';
    }

    /** @return Original native HC operand from the same actual model layer. */
    std::shared_ptr<TensorBase> loadHC(const char* suffix,TensorType type,std::vector<std::size_t> shape)
    {
        const auto name="blk."+std::to_string(layer_)+"."+suffix;
        const auto geometry=loader_.nativeWeightSourceGeometry(name);
        require(geometry.original_type==type&&geometry.original_shape==shape,"exact original HC source geometry");
        auto source=loader_.loadTensor(name,DeviceId::cpu(),WeightPrecision::NATIVE);
        require(source&&source->native_type()==type,"original HC source without conversion");
        return source;
    }

private:
    /** @return Original model tensor name for a canonical one-layer fixture role. */
    std::string originalName(std::string_view fixture_name) const
    {
        require(fixture_name.starts_with("blk.0."),"canonical test layer role");
        return "blk."+std::to_string(layer_)+"."+std::string(fixture_name.substr(6));
    }

    /** @return Native group and source tensor identity from the unchanged GGUF header. */
    NativeWeightSourceGeometry geometryOf(const char* fixture_name)
    {
        return loader_.nativeWeightSourceGeometry(originalName(fixture_name));
    }

    ModelLoader loader_; ///< Original on-disk model authority.
    int layer_,participant_;
    TensorType gate_{},down_{};
    MoEExpertProjectionOwnership::IntermediateInterval routed_{};
    int shared_width_{}; ///< Original shared output-channel assignment width.
    int shared_first_=0; ///< Original shared output-channel assignment offset.
    int source_routed_width_{}; ///< Uncut original routed intermediate extent.
    int source_shared_width_{}; ///< Uncut original shared intermediate extent.
};
