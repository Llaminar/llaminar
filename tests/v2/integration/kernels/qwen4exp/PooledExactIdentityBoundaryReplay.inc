/**
 * @file PooledExactIdentityBoundaryReplay.inc
 * @brief Actual captured owner Pin and retained-pin/allocation identity thresholds.
 *
 * ProviderReplay records the real complete local lookup graph and original
 * provider/source publication. Only input IDs are uploaded; final owner logs,
 * source counts and native decoded bytes are observed after retirement. This
 * integer-policy boundary sweep complements the existing all-format native
 * gates; it is neither a custom identity kernel nor a helper-only substitute.
 */
enum class ExactIdentityBoundaryTrace {Pin,RetainedPinsAndAllocation};
/**
 * @brief Execute one partition of the production identity boundary on one exact backend.
 * @param device Actual native execution endpoint; absence is a failing unqualified cell.
 * @param live Actual identity count at original small-set, CTA and tiled-sort boundaries.
 * @param trace Either a pure warm Pin batch or32 retained pins plus fresh accepted grants.
 * @details Two admitted generations reuse the retained graph. Every generation
 * repeats the threshold with shrinking and empty replays between large work.
 * Allocation work uses admitted free slots for all six disjoint cohorts, so
 * exact identity thresholds do not depend on the separate TinyLFU victim
 * policy. Native victim/rejection coverage remains in the provider/all-format
 * and OwnerPlan replays; no host changes owner counts or physical metadata.
 */
void exactIdentityNativeBoundary(DeviceId device,std::uint32_t live,ExactIdentityBoundaryTrace trace){
    requireProvider((live>=63 && live<=65) || live==127 || live==128 || live==129 || live==255 || live==256 || live==257 || live==513,"Exact identity replay requires its actual threshold neighbor");
    if(!hasCPUBackend())initCPUBackend(-1);
    const auto maximum=std::max(65u,live);
    std::vector<std::uint64_t> authored;
    for(std::uint64_t row=100;row<100+maximum;++row)authored.push_back(row);
    for(std::uint64_t row=1000;row<1000+6*(maximum-32);++row)authored.push_back(row);
    const auto slots=trace==ExactIdentityBoundaryTrace::Pin?maximum:32u+6u*(live-32u);
    ProviderReplayPrograms programs({{std::make_shared<ProviderFile>(4096,authored),maximum,slots,2,n::OriginalIdWidth::UInt64,false,OwnerAdmission::Original,ACKInterleaving::None,ActivationPreparation::BeforeGraph,maximum,true}});
    programs.prepare(device);programs.run(0,[&](ProviderReplay& replay){auto& file=replay.file;
        const auto evidence=[&](std::uint32_t expectedPins,std::uint32_t expectedGrants){
            const auto roots=replay.owner->roots();const auto& plan=replay.owner->plan().parameters();
            const auto frame=replay.read<p::OwnerFrame>(static_cast<const std::byte*>(roots.frame->gpu_data_ptr())+plan.frame.control,1).front();
            const auto progress=replay.read<n::OwnerProgress>(static_cast<const std::byte*>(roots.native->gpu_data_ptr())+plan.native.progress,1).front();
            EXPECT_EQ(frame.phase,p::OwnerPhase::Idle);EXPECT_EQ(frame.status,q::adaptive_lookup::Status::Success);EXPECT_EQ(frame.initial_pin_count,expectedPins);EXPECT_EQ(frame.held_pins,0u);EXPECT_EQ(frame.reservations,0u);
            EXPECT_EQ(progress.released,expectedPins+expectedGrants);EXPECT_EQ(progress.release_bytes,std::uint64_t{expectedPins+expectedGrants}*sizeof(p::ReleaseReceipt));
            const auto pins=replay.read<p::Grant>(static_cast<const std::byte*>(roots.frame->gpu_data_ptr())+plan.frame.initial_pins,expectedPins);
            const auto grants=replay.read<p::Grant>(static_cast<const std::byte*>(roots.frame->gpu_data_ptr())+plan.frame.replies,expectedGrants);
            for(const auto& pin:pins){EXPECT_EQ(pin.kind,p::GrantKind::Read);EXPECT_EQ(pin.owner,0u);EXPECT_EQ(pin.destination_token,0u);}
            for(const auto& grant:grants){EXPECT_EQ(grant.kind,p::GrantKind::Reservation);EXPECT_EQ(grant.owner,0u);EXPECT_EQ(grant.destination_token,std::uint64_t{grant.unique}+1);}
            // Release receipts count the actual combined retained prefix even
            // though the allocation control is reset to Idle by retirement.
            const auto receipts=replay.read<p::ReleaseReceipt>(static_cast<const std::byte*>(roots.frame->gpu_data_ptr())+plan.frame.release_receipts,expectedPins+expectedGrants);
            for(std::size_t i=0;i<receipts.size();++i){EXPECT_NE(receipts[i].disposition,p::ReleaseDisposition::Failed);const auto& expected=i<pins.size()?pins[i]:grants[i-pins.size()];EXPECT_EQ(receipts[i].original.row,expected.row);EXPECT_EQ(receipts[i].original.unique,expected.unique);EXPECT_EQ(receipts[i].original.owner,expected.owner);EXPECT_EQ(receipts[i].original.slot,expected.slot);EXPECT_EQ(receipts[i].original.generation,expected.generation);EXPECT_EQ(receipts[i].original.kind,expected.kind);EXPECT_EQ(receipts[i].original.destination_token,expected.destination_token);}
            EXPECT_EQ(replay.lastObservation.gpu_hits,expectedPins);EXPECT_EQ(replay.lastObservation.source_rows,expectedGrants);EXPECT_EQ(replay.lastObservation.source_native_bytes,std::uint64_t{expectedGrants}*file.row_bytes);
            std::cout<<"POOLED_EXACT_IDENTITY_BOUNDARY device="<<device.toString()<<" expected_combined="<<live<<" expected_pins="<<expectedPins<<" expected_grants="<<expectedGrants<<" actual_initial_pins="<<frame.initial_pin_count<<" actual_released="<<progress.released<<" release_bytes="<<progress.release_bytes<<" captured=1\n";
        };
        for(std::uint64_t generation=1;generation<=2;++generation){replay.admit(generation);
            if(trace==ExactIdentityBoundaryTrace::Pin){std::vector<std::uint64_t> ids(live);for(std::size_t i=0;i<ids.size();++i)ids[i]=100+i;replay.replay(ids);
                for(unsigned repeat=0;repeat<3;++repeat){replay.replay(ids);evidence(live,0);replay.replay(std::array<std::uint64_t,1>{100});evidence(1,0);replay.replay({});evidence(0,0);}
            }else{
                std::vector<std::uint64_t> warm(32);for(std::size_t i=0;i<warm.size();++i)warm[i]=100+i;replay.replay(warm);
                // Six disjoint cohorts fit in the declared physical capacity.
                // This isolates combined identity cardinality from TinyLFU's
                // lawful victim rejection: every fresh grant uses a free slot.
                // Only the32 requested warm rows are pinned, regardless of the
                // unrelated retained rows occupying other physical slots.
                for(unsigned repeat=0;repeat<3;++repeat){std::vector<std::uint64_t> mixed=warm;
                    for(std::uint32_t i=0;i<live-32;++i)mixed.push_back(1000+(generation-1)*3*(live-32)+repeat*(live-32)+i);
                    replay.replay(mixed);evidence(32,live-32);
                    replay.replay(std::array<std::uint64_t,1>{100});evidence(1,0);replay.replay({});evidence(0,0);}
            }
            replay.close();
        }
    });programs.retire();
}
/**
 * @brief Capture the real selected native phase with rejected original inputs and a scalar phase oracle.
 * @param device Original selected CUDA/ROCm participant.
 * @param live Full hot identity prefix, crossing64 or256.
 * @param action Actual production Pin, AcceptFree or AcceptVictim operation.
 * @param row64 True for original UInt64 physical source metadata, false for UInt32.
 * @note Recorded GPU state restoration is kernel-replay input, not production
 * recovery or proof of failed-request retirement. Original cancel/release drains
 * retained rights where supported; ordinary partial/empty and request close use
 * the unchanged complete provider graph and its existing separate lifecycle proof.
 */
void exactIdentityNativeRejection(DeviceId device,std::uint32_t live,n::OwnerAction action,bool row64){
    requireProvider(live==65 || live==257,"Captured identity rejection requires original64/256 neighbor");
    if(!hasCPUBackend())initCPUBackend(-1);
    std::vector<identity_fixture::IdentityFault> faults;
    if(action==n::OwnerAction::Pin)faults={identity_fixture::IdentityFault::DuplicateRow,identity_fixture::IdentityFault::DuplicateUnique,identity_fixture::IdentityFault::DuplicateSlot};
    else faults={identity_fixture::IdentityFault::DuplicateRow,identity_fixture::IdentityFault::DuplicateUnique,identity_fixture::IdentityFault::ForeignOwner,identity_fixture::IdentityFault::ForeignInterval};
    const auto base=row64?std::uint64_t{0x100000000ULL}:0u;
    std::vector<std::uint64_t> rows(live);for(std::uint32_t i=0;i<live;++i)rows[i]=base+100+i;
    auto file=std::make_shared<ProviderFile>(base+4096,rows);
    std::vector<ProviderReplaySpec> specs;
    for(const auto fault:faults){
        ProviderReplaySpec spec{file,live+17,live+17,2,n::OriginalIdWidth::UInt64,false,OwnerAdmission::Original,ACKInterleaving::None,ActivationPreparation::BeforeGraph,live+17,true};
        spec.identityRejection=IdentityRejectionCase{action,fault,live-1};spec.requireComplete();specs.push_back(std::move(spec));
    }
    ProviderReplayPrograms programs(specs);programs.prepare(device);
    for(std::size_t index=0;index<specs.size();++index)programs.run(index,[&](ProviderReplay& replay){
        EXPECT_EQ(replay.owner->plan().parameters().owner.rowWidth(),row64?q::adaptive_lookup::RowKeyWidth::UInt64:q::adaptive_lookup::RowKeyWidth::UInt32);
        for(std::uint64_t generation=1;generation<=2;++generation){
            replay.admit(generation);replay.replay(rows);
            for(unsigned repeat=0;repeat<20;++repeat){
                replay.replayIdentityRejection(rows);
                // These are real successful production transactions, not a
                // truncated-prefix restore standing in for normal reset/close.
                replay.replay(std::span<const std::uint64_t>(rows.data(),1));
                replay.replay({});
            }
            replay.close();
        }
    });
    programs.retire();
}
