/**
 * @file GroupedFFNM4SourceMatrix.inc
 * @brief Compare the original counted M4 FFN capture against its native serial oracle.
 *
 * The production source catalog owns the independent sweep. The existing
 * family recipes provide a separate mixed-decoder sweep. Only the explicit
 * original window copies bytes from an actual QFN GGUF weight shard.
 */
namespace llaminar2::test {
namespace {

enum class M4SourceRole { RoutedGateUp, RoutedDown, SharedGateUp, SharedDown, Family, RuntimeOnly, FixedQ51Down };

struct M4SourceCase {
    std::string name; ///< Filter and receipt identity.
    M4SourceRole role; ///< One varied codebook or an original decoder-family recipe.
    std::string_view source; ///< Canonical source name when varying a single role.
    std::optional<FullNativeHCFFNFamilyCase> family; ///< Source-authenticated mixed recipe.
    int fixed_down_pitch=0; ///< Independently expected original Q5_1 Down specialization.
};

/** @return Immutable catalog entries and the independent decoder-family recipes. */
std::vector<M4SourceCase> groupedM4Cases()
{
    const std::array roles{
        std::pair{M4SourceRole::RoutedGateUp,"routed_gate_up"},
        std::pair{M4SourceRole::RoutedDown,"routed_down"},
        std::pair{M4SourceRole::SharedGateUp,"shared_gate_up"},
        std::pair{M4SourceRole::SharedDown,"shared_down"}};
    std::vector<M4SourceCase> cases;
    for (const auto& entry:native_quantised_formats::kAllSourceFormats) {
        if (entry.quant_type=="Q8_1") {
            cases.push_back({"Q8_1_source_boundary",M4SourceRole::RuntimeOnly,entry.quant_type,{}});
            continue;
        }
        for (const auto& [role,name]:roles)
            cases.push_back({std::string(entry.quant_type)+"_"+name,role,entry.quant_type,{}});
    }
    cases.push_back({"Q8_1_shared_down_prepared",M4SourceRole::SharedDown,"Q8_1",{}});
    cases.push_back({"Q4_K_repeat_routes",M4SourceRole::RoutedGateUp,"Q4_K",{}});
    cases.push_back({"Q4_K_sparse_routes",M4SourceRole::RoutedGateUp,"Q4_K",{}});
    cases.push_back({"Q4_K_deep_grid",M4SourceRole::RoutedGateUp,"Q4_K",{}});
    cases.push_back({"Q4_K_Q5_1_Down_F128",M4SourceRole::FixedQ51Down,"Q4_K",{},128});
    cases.push_back({"Q4_K_Q5_1_Down_F256",M4SourceRole::FixedQ51Down,"Q4_K",{},256});
    cases.push_back({"Q5_K_sparse_routes",M4SourceRole::RoutedGateUp,"Q5_K",{}});
    cases.push_back({"Q2_K_sparse_routes",M4SourceRole::RoutedDown,"Q2_K",{}});
    cases.push_back({"IQ1_M_sparse_routes",M4SourceRole::RoutedDown,"IQ1_M",{}});
    cases.push_back({"IQ4_NL_up_IQ4_XS",M4SourceRole::RoutedGateUp,"IQ4_NL",{}});
    cases.push_back({"IQ4_XS_up_IQ4_NL",M4SourceRole::RoutedGateUp,"IQ4_XS",{}});
    cases.push_back({"IQ4_NL_shared_up_IQ4_XS",M4SourceRole::SharedGateUp,"IQ4_NL",{}});
    cases.push_back({"IQ4_XS_shared_up_IQ4_NL",M4SourceRole::SharedGateUp,"IQ4_XS",{}});
    for (const auto& family:fullNativeHCFFNFamilyCases())
        cases.push_back({family.name,M4SourceRole::Family,{},family});
    return cases;
}

/**
 * @brief Read the independent expected producer and graph program for this fixture.
 * @param shape Mutable test geometry before production graph construction.
 */
void expectGroupedM4Program(OriginalFFNFixtureGeometry& shape)
{
    const char* requested=std::getenv("LLAMINAR_TEST_GROUPED_M4_PROGRAM");
    if (requested) {
        const std::string_view program=requested;
        require(program=="fixed" || program=="all_formats","Use a known grouped M4 fixture program");
        shape.grouped_matrix_program=program=="fixed"?GroupedM4FixtureProgram::Fixed:
            GroupedM4FixtureProgram::AllFormats;
    }
    const char* symbol=std::getenv("LLAMINAR_TEST_GROUPED_M4_REQUIRED_SYMBOL");
    if (symbol)
        shape.grouped_matrix_symbol=symbol;
    else if (requested)
        shape.grouped_matrix_symbol=shape.grouped_matrix_program==GroupedM4FixtureProgram::Fixed?
            "groupedFFNPairedGatePublish":"groupedFFNFormatProgramGate";
    const char* present=std::getenv("LLAMINAR_TEST_GROUPED_M4_SYMBOL_PRESENT");
    require(!present || std::string_view(present)=="0" || std::string_view(present)=="1",
        "Native expected marker must be present (1) or absent (0)");
    shape.grouped_matrix_symbol_present=!present || std::string_view(present)=="1";
}

/** @brief Use the original source preparer and run the counted M4 production graph. */
void runGroupedM4Source(const M4SourceCase& item,bool deferred,bool timing)
{
    if (item.role==M4SourceRole::RuntimeOnly) {
        require(!timing,"Q8_1 has no GGUF source-backed timing arm");
        const auto& format=quantizedVerifierFormat(item.source);
        EXPECT_THROW((void)PlanningGGUFFixture::sourceType(format.tensor_type),std::invalid_argument);
        return;
    }
    const auto& q8=quantizedVerifierFormat("Q8_0");
    const auto& selected=item.family?q8:quantizedVerifierFormat(item.source);
    const bool fixedQ51Down=item.role==M4SourceRole::FixedQ51Down;
    const auto& gate=(item.role==M4SourceRole::RoutedGateUp || fixedQ51Down)?selected:q8;
    const auto& down=fixedQ51Down?quantizedVerifierFormat("Q5_1"):
        item.role==M4SourceRole::RoutedDown?selected:q8;
    const auto& shared=item.role==M4SourceRole::SharedGateUp?selected:q8;
    const bool prepared_q8=item.name=="Q8_1_shared_down_prepared";
    const auto& shared_down=item.role==M4SourceRole::SharedDown&&!prepared_q8?selected:q8;
    const auto variants=sourceFormats(OriginalSourceQualification::ResidentFFN);
    const auto original=std::find_if(variants.begin(),variants.end(),[&](const auto& format){
        return format.type==gate.tensor_type;
    });
    require(original!=variants.end(),"Routed gate/up has its original source preparer");
    auto shape=fullNativeHCFFNFamilyGeometry();
    if (!item.family) {
        shape.routed=fixedQ51Down?item.fixed_down_pitch:down.source_is_superblock?256:160;
        shape.shared=shared_down.source_is_superblock?256:160;
    }
    if(item.name=="Q4_K_deep_grid") {
        // The original unequal TP2 cut gives this participant 384 routed and
        // 320 shared columns, enough tiles to exercise the long-grid program.
        shape.routed=384;
        shape.shared=320;
    }
    auto format=item.family?fullNativeHCFFNFamilySource(*item.family,shape):*original;
    format.name=item.name;
    if (!item.family) {
        requireQuantizedVerifierSourceShape(gate,shape.routed,shape.hidden);
        requireQuantizedVerifierSourceShape(down,shape.hidden,shape.routed);
        requireQuantizedVerifierSourceShape(shared,shape.shared,shape.hidden);
        requireQuantizedVerifierSourceShape(shared_down,shape.hidden,shape.shared);
        format.down_quantized=&down;
        format.shared_quantized=&shared;
        format.shared_down_quantized=&shared_down;
    }
    if (item.name.ends_with("_up_IQ4_XS") || item.name.ends_with("_up_IQ4_NL")) {
        const auto& up=quantizedVerifierFormat(item.name.ends_with("_IQ4_XS")?"IQ4_XS":"IQ4_NL");
        if (item.role==M4SourceRole::RoutedGateUp)
            format.up_quantized=&up;
        else
            format.shared_up_quantized=&up;
    }
    if (item.name=="Q4_K_repeat_routes")
        shape.router_source=OriginalFFNRouterSource::TieControl;
    if (item.name.ends_with("_sparse_routes"))
        shape.router_source=OriginalFFNRouterSource::SparseSharedRoutes;
    const char* pci=std::getenv("LLAMINAR_TEST_RETAINED_HC_FFN_PCI");
    const std::array<std::string_view,1> requested{pci?pci:""};
    shape.grouped_matrix_case=item.name;
    expectGroupedM4Program(shape);
    if(fixedQ51Down) {
        require(item.fixed_down_pitch==128 || item.fixed_down_pitch==256,"independent supported Q5_1 Down fixture pitch");
        shape.expected_q51_down_pitch=item.fixed_down_pitch;
        shape.grouped_matrix_symbol="groupedFFNFormatProgramGate";
    }
    if(item.name=="Q4_K_deep_grid")
        shape.grouped_matrix_symbol="groupedFFNFormatProgramGateILj1ELi5ELi0ELi2EE";
    if(item.name=="Q5_K_sparse_routes" && shape.grouped_matrix_program==GroupedM4FixtureProgram::AllFormats)
        shape.grouped_matrix_symbol="groupedFFNFormatProgramGateILj1ELin1ELi160ELi1EE";
    shape.grouped_matrix_timing=timing;
    shape.deferred_frontier_history=deferred;
    std::osyncstream(std::cout)<<"M4_SOURCE_INPUT,case="<<item.name
        <<",origin="<<(prepared_q8?"prepared_tensor_plus_generated_gguf":"generated_gguf")
        <<",H="<<shape.hidden<<",F="<<shape.routed<<",S="<<shape.shared
        <<",gate="<<(item.family?item.family->gate:gate.label)
        <<",up="<<(format.up_quantized?format.up_quantized->label:(item.family?item.family->gate:gate.label))
        <<",down="<<format.down_quantized->label<<",shared_gate="<<format.shared_quantized->label
        <<",shared_up="<<(format.shared_up_quantized?format.shared_up_quantized->label:format.shared_quantized->label)
        <<",shared_down="<<(prepared_q8?"Q8_1":(format.shared_down_quantized?format.shared_down_quantized->label:format.shared_quantized->label))<<'\n';
    originalRankLocalReplay(DeviceType::ROCm,1,[&](const auto& devices) {
        proveOriginalFinalization(devices,false,OriginalSourceQualification::GroupedFFNSourceMatrix,
            &format,LocalFFNProgram::Original,shape);
    },"gfx906",requested.front().empty()?std::span<const std::string_view>{}:std::span<const std::string_view>{requested});
}

/** @test Independent quantized verifier registry cannot silently lose a compiled source/role case. */
TEST(GroupedM4SourceInventory, CoversOriginalCatalog)
{
    const auto cases=groupedM4Cases();
    const auto& known=quantizedVerifierFormats();
    ASSERT_GE(known.size(),21u);
    constexpr std::array roles{"routed_gate_up","routed_down","shared_gate_up","shared_down"};
    for (const auto& source:known) {
        if (source.tensor_type==TensorType::Q8_1) {
            EXPECT_EQ(std::count_if(cases.begin(),cases.end(),[](const auto& item) {
                return item.role==M4SourceRole::RuntimeOnly && item.source=="Q8_1";
            }),1);
            continue;
        }
        for (const auto* role:roles) {
            const auto name=std::string(source.label)+"_"+role;
            EXPECT_EQ(std::count_if(cases.begin(),cases.end(),[&](const auto& item) {
                return item.name==name;
            }),1)<<name;
        }
    }
}

/** @brief Independent process-selectable source/demand cohorts. */
class GroupedM4Source : public ::testing::TestWithParam<M4SourceCase> {};

/** @test Static device count, empty and partial work, dirty reset and bank epoch. */
TEST_P(GroupedM4Source, Static) { runGroupedM4Source(GetParam(),false,false); }
/** @test Native acceptance commits exactly the live prefix after movement. */
TEST_P(GroupedM4Source, Deferred) { runGroupedM4Source(GetParam(),true,false); }
/** @test A separate unprofiled event window with the same exact parent and byte oracle. */
TEST_P(GroupedM4Source, DISABLED_Events) { runGroupedM4Source(GetParam(),false,true); }

INSTANTIATE_TEST_SUITE_P(OriginalSources,GroupedM4Source,::testing::ValuesIn(groupedM4Cases()),
    [](const ::testing::TestParamInfo<M4SourceCase>& item){return item.param.name;});

} // namespace
} // namespace llaminar2::test

namespace llaminar2::test {

/** @test The real shard's original TP coordinates supply all FFN source bytes. */
TEST(GroupedM4SourceWindow, DISABLED_OriginalQFN)
{
    const char* pci=std::getenv("LLAMINAR_TEST_RETAINED_HC_FFN_PCI");
    require(pci&&*pci,"Real M4 source window requires an explicit PCI");
    auto window=realProductionTPInputFromEnv();
    auto format=window.format();
    auto shape=window.geometry();
    const char* demand=std::getenv("LLAMINAR_TEST_GROUPED_M4_DEMAND");
    const char* events=std::getenv("LLAMINAR_TEST_QFN_WINDOW_EVENTS");
    require(demand && (std::string_view(demand)=="static" || std::string_view(demand)=="deferred"),
        "Real M4 source window requires static or deferred demand");
    shape.grouped_matrix_case="OriginalQFN";
    expectGroupedM4Program(shape);
    shape.deferred_frontier_history=std::string_view(demand)=="deferred";
    shape.grouped_matrix_timing=events&&std::string_view(events)=="1";
    require(!shape.grouped_matrix_timing || !shape.deferred_frontier_history,"Event brackets use static demand");
    const std::array<std::string_view,1> address{pci};
    std::osyncstream(std::cout)<<"M4_SOURCE_INPUT,case=OriginalQFN,origin=original_qfn_gguf"
        <<",H="<<shape.hidden<<",F="<<shape.routed<<",S="<<shape.shared
        <<",format="<<format.name<<'\n';
    originalRankLocalReplay(DeviceType::ROCm,1,[&](const auto& devices) {
        proveOriginalFinalization(devices,false,OriginalSourceQualification::GroupedFFNSourceMatrix,
            &format,LocalFFNProgram::Original,shape,CastagnaPhaseObservation::Disabled,
            FullProductionHCProbe::Disabled,FullProductionHCStorageReuse::Distinct,false,
            FullProductionHCObservation::ObservedAll,&window);
    },"gfx906",address);
}

} // namespace llaminar2::test
