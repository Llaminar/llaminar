/**
 * @file MoEOverlayMigrationMeasurementLedger.cpp
 * @brief Bounded sample retention and robust medians for migration economy.
 */

#include "MoEOverlayMigrationMeasurementLedger.h"
#include "MoEOverlayMigrationEvidenceContext.h"

#include "../../utils/FNV1a.h"

#include <algorithm>
#include <array>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace llaminar2
{
    namespace
    {
        /** Sentinel returned when a required coordinate is absent. */
        constexpr std::size_t kMissingCoordinate =
            std::numeric_limits<std::size_t>::max();

        /** @brief Store one failure diagnostic only when requested. */
        void assignLedgerError(
            std::string *error,
            const char *message) noexcept
        {
            if (error)
                *error = message;
        }

        /** @brief Compute the deterministic lower median of a copied sample set. */
        template <typename Value>
        Value median(std::vector<Value> values)
        {
            std::sort(values.begin(), values.end());
            return values[(values.size() - 1u) / 2u];
        }

        /** @brief Add a scalar to the platform-independent corpus digest. */
        void hashUnsigned(std::uint64_t &hash, std::uint64_t value) noexcept
        {
            std::array<unsigned char, sizeof(value)> bytes{};
            for (std::size_t index = 0; index < bytes.size(); ++index)
            {
                bytes[index] = static_cast<unsigned char>(value & 0xffu);
                value >>= 8u;
            }
            hash = fnv1a64(bytes.data(), bytes.size(), hash);
        }

        /** @brief Add a length-delimited string to the corpus digest. */
        void hashString(std::uint64_t &hash, const std::string &value) noexcept
        {
            hashUnsigned(hash, static_cast<std::uint64_t>(value.size()));
            hash = fnv1a64(value.data(), value.size(), hash);
        }
    } // namespace

    bool MoEOverlayMigrationMeasurementCoordinate::valid() const noexcept
    {
        return source_participant >= 0 && destination_participant >= 0 &&
               source_participant != destination_participant && layer >= 0;
    }

    bool MoEOverlaySealedMigrationMeasurements::valid() const noexcept
    {
        return !identity.empty() && !rows.empty() && std::all_of(rows.begin(),rows.end(),[](const auto &row){return row.valid();});
    }

    std::vector<MoEOverlayParticipantLayerMigrationMeasurement> MoEOverlaySealedMigrationMeasurements::wholeRows() const
    {
        if(!valid())throw std::invalid_argument("Whole-only consumer requires complete sealed movement evidence");
        std::vector<MoEOverlayParticipantLayerMigrationMeasurement> result;result.reserve(rows.size());
        for(const auto &row:rows)result.push_back(row.whole());
        return result;
    }

    MoEOverlayMigrationMeasurementLedger::MoEOverlayMigrationMeasurementLedger(Config config):config_(std::move(config))
    {
        const bool typed=!config_.required_target_measurements.empty();
        if((typed && !config_.required_coordinates.empty()) || (!typed && config_.required_coordinates.empty()) ||
            config_.measurement_identity.empty() || config_.measured_samples_per_coordinate<MoEOverlayEconomyProfileComposer::kMinimumMigrationSamples ||
            !validExpertHistogramProductionSourceMask(config_.required_sources) ||
            config_.warmup_samples_per_coordinate>std::numeric_limits<std::uint64_t>::max()-config_.measured_samples_per_coordinate)
            throw std::invalid_argument("Migration ledger requires one original coordinate declaration set and bounded robust sample policy");
        if(!typed) {
            for(const auto &coordinate:config_.required_coordinates) {
                if(!coordinate.valid())throw std::invalid_argument("Invalid legacy whole coordinate");
                config_.required_target_measurements.push_back({{MoEOverlayRouteTarget::whole(coordinate.source_participant),
                    MoEOverlayRouteTarget::whole(coordinate.destination_participant),coordinate.layer},{},{}});
            }
        }
        auto &declared=config_.required_target_measurements;
        std::sort(declared.begin(),declared.end(),[](const auto &a,const auto &b){return a.coordinate<b.coordinate;});
        coordinate_state_=std::make_unique<CoordinateState[]>(declared.size());coordinates_.reserve(declared.size());
        std::size_t piece_count=0;
        for(std::size_t i=0;i<declared.size();++i) {
            const auto &declaration=declared[i];
            if(!declaration.coordinate.valid() || (i && declared[i-1].coordinate==declaration.coordinate) ||
                (typed && (!declaration.geometry || !MoEOverlayTargetLayerMigrationMeasurement::geometryMatches(declaration.coordinate,*declaration.geometry))))
                throw std::invalid_argument("Ledger target declarations require unique coordinates and matching original geometry");
            auto &state=coordinate_state_[i];state.geometry=declaration.geometry;
            state.piece_count=typed?declaration.geometry->pieces().size():3;state.first_piece=piece_count;
            if(!state.piece_count || piece_count>std::numeric_limits<std::size_t>::max()-state.piece_count)
                throw std::overflow_error("Ledger original piece prefix overflows host size");
            if(!declaration.quoted_payload_bytes.empty() &&
                (declaration.quoted_payload_bytes.size()!=state.piece_count || std::any_of(declaration.quoted_payload_bytes.begin(),declaration.quoted_payload_bytes.end(),[](auto bytes){return !bytes;})))
                throw std::invalid_argument("Ledger setup quotes must cover every original piece exactly");
            state.setup_quotes_bound=!declaration.quoted_payload_bytes.empty();
            piece_count+=state.piece_count;coordinates_.push_back(declaration.coordinate);
        }
        const auto samples=static_cast<std::size_t>(config_.measured_samples_per_coordinate);
        if(samples!=config_.measured_samples_per_coordinate || coordinates_.size()>std::numeric_limits<std::size_t>::max()/samples ||
            piece_count>std::numeric_limits<std::size_t>::max()/samples ||
            coordinates_.size()*samples>std::numeric_limits<std::size_t>::max()/kExpertHistogramProductionSourceCount)
            throw std::overflow_error("Ledger original sample geometry overflows host size");
        wave_samples_.resize(coordinates_.size()*samples);piece_samples_.resize(piece_count*samples);piece_bytes_.resize(piece_count);
        interference_samples_.resize(coordinates_.size()*samples*kExpertHistogramProductionSourceCount);
        for(std::size_t i=0;i<declared.size();++i)if(coordinate_state_[i].setup_quotes_bound)
            std::copy(declared[i].quoted_payload_bytes.begin(),declared[i].quoted_payload_bytes.end(),piece_bytes_.begin()+coordinate_state_[i].first_piece);
        // The declaration inputs are adapters, not retained competing coordinate/corpus tables.
        std::vector<MoEOverlayRequiredTargetMigrationMeasurement>().swap(config_.required_target_measurements);
        std::vector<MoEOverlayMigrationMeasurementCoordinate>().swap(config_.required_coordinates);
    }

    std::vector<MoEOverlayMigrationMeasurementCoordinate> MoEOverlayMigrationMeasurementLedger::requiredCoordinates() const
    {
        std::vector<MoEOverlayMigrationMeasurementCoordinate> result;result.reserve(coordinates_.size());
        for(const auto &coordinate:coordinates_) {
            if(!coordinate.source_target.isWhole() || !coordinate.destination_target.isWhole())
                throw std::invalid_argument("Cohort declarations have no whole-participant coordinate projection");
            result.push_back({coordinate.source_target.word(),coordinate.destination_target.word(),coordinate.layer});
        }
        return result;
    }

    std::size_t MoEOverlayMigrationMeasurementLedger::coordinateIndex(const MoEOverlayMigrationTargetCoordinate &coordinate) const noexcept
    {
        const auto found=std::lower_bound(coordinates_.begin(),coordinates_.end(),coordinate);
        return found==coordinates_.end() || *found!=coordinate?kMissingCoordinate:static_cast<std::size_t>(found-coordinates_.begin());
    }

    std::size_t MoEOverlayMigrationMeasurementLedger::pieceSampleOffset(std::size_t coordinate,std::size_t piece,std::uint64_t measured) const noexcept
    {
        return (coordinate_state_[coordinate].first_piece+piece)*static_cast<std::size_t>(config_.measured_samples_per_coordinate)+static_cast<std::size_t>(measured);
    }

    std::size_t MoEOverlayMigrationMeasurementLedger::sampleOffset(
        std::size_t coordinate_index,
        std::uint64_t measured_sample_index) const noexcept
    {
        return coordinate_index *
                   static_cast<std::size_t>(
                       config_.measured_samples_per_coordinate) +
               static_cast<std::size_t>(measured_sample_index);
    }

    std::size_t
    MoEOverlayMigrationMeasurementLedger::interferenceSampleOffset(
        std::size_t coordinate_index,
        std::size_t phase_index,
        std::uint64_t measured_sample_index) const noexcept
    {
        return (coordinate_index * kExpertHistogramProductionSourceCount +
                phase_index) *
                   static_cast<std::size_t>(
                       config_.measured_samples_per_coordinate) +
               static_cast<std::size_t>(measured_sample_index);
    }

    bool MoEOverlayMigrationMeasurementLedger::recordCompletedWave(std::vector<MoEOverlayCompletedMigrationMeasurement> &&rows,std::string *error) noexcept
    {
        return recordWave(rows,{},error);
    }
    bool MoEOverlayMigrationMeasurementLedger::recordCompletedWave(const std::vector<MoEOverlayCompletedMigrationMeasurement> &rows,std::string *error) noexcept
    {
        return recordWave(rows,{},error);
    }
    bool MoEOverlayMigrationMeasurementLedger::recordCompletedTargetWave(std::span<const MoEOverlayBorrowedTargetMigrationObservation> rows,std::string *error) noexcept
    {
        return recordWave({},rows,error);
    }

    bool MoEOverlayMigrationMeasurementLedger::recordWave(std::span<const MoEOverlayCompletedMigrationMeasurement> whole,
        std::span<const MoEOverlayBorrowedTargetMigrationObservation> targets,std::string *error) noexcept
    {
        if(error)error->clear();
        const auto reject=[&](const char *message){rejected_observations_.fetch_add(1,std::memory_order_relaxed);assignLedgerError(error,message);return false;};
        const auto count=targets.empty()?whole.size():targets.size();
        if(sealed_.load(std::memory_order_acquire))
            return reject("ExpertOverlay migration ledger is sealed");
        if(!count || (!whole.empty() && !targets.empty()))
            return reject("Migration ledger received an invalid empty input arm");
        const auto rowAt=[&](std::size_t i)->const MoEOverlayCompletedMigrationMeasurement&{return targets.empty()?whole[i]:targets[i].measurement;};
        const auto keyAt=[&](std::size_t i)->MoEOverlayMigrationTargetCoordinate {
            if(!targets.empty())return targets[i].coordinate;
            const auto &row=whole[i];
            if(row.source_participant<0 || row.destination_participant<0)return {};
            return {MoEOverlayRouteTarget::fromWord(row.source_participant),MoEOverlayRouteTarget::fromWord(row.destination_participant),row.layer};
        };
        const auto sampleAt=[](const auto &row,std::size_t piece)->std::optional<ExpertTierProjectionTransferMeasurement>{
            return row.geometry?row.piece_observations[piece].observation:row.projections[piece];};
        const auto quoteAt=[](const auto &row,std::size_t piece)->std::uint64_t {
            return row.geometry?row.piece_observations[piece].quoted_payload_bytes.value_or(0):row.projections[piece]->bytes;};
        // Validate the entire wave before any accepted counter, invariant quote or sample mutation.
        for(std::size_t i=0;i<count;++i) {
            const auto key=keyAt(i);const auto slot=coordinateIndex(key);const auto &row=rowAt(i);
            if(!key.valid() || slot==kMissingCoordinate || !row.valid() || row.layer!=key.layer)
                return reject("Migration wave has an invalid or undeclared original target coordinate");
            const auto &state=coordinate_state_[slot];
            if(state.geometry) {
                if(!row.geometry || !state.geometry->sameIdentity(*row.geometry) ||
                    !MoEOverlayTargetLayerMigrationMeasurement::geometryMatches(key,*row.geometry))
                    return reject("Migration observation differs from setup original members, cuts or source formats");
            } else if(!key.source_target.isWhole() || !key.destination_target.isWhole() ||
                row.source_participant!=key.source_target.word() || row.destination_participant!=key.destination_target.word() ||
                (row.geometry && (row.geometry->targets().first.members.size()!=1 || row.geometry->targets().second.members.size()!=1)))
                return reject("Whole-only ledger adapter cannot certify a cohort target");
            if(row.geometry && row.piece_observations.size()!=state.piece_count)
                return reject("Migration observation does not cover the original canonical pieces");
            for(std::size_t earlier=0;earlier<i;++earlier)if(keyAt(earlier)==key)
                return reject("Migration wave repeats a directed target coordinate");
            for(std::size_t piece=0;piece<state.piece_count;++piece) {
                const auto sample=sampleAt(row,piece);
                if(!sample || !sample->valid() || (!sample->device_nanoseconds && !sample->host_nanoseconds && !sample->transport_nanoseconds))
                    return reject("Migration ledger received a partial observation without complete real piece timing evidence");
                const auto quote=quoteAt(row,piece),expected=piece_bytes_[state.first_piece+piece];
                if(!quote || quote!=sample->bytes)
                    return reject("Migration piece differs from independent original quoted bytes");
                if(expected && expected!=quote)
                    return reject("Migration ledger observed changing projection bytes for an original piece");
            }
        }
        for(std::size_t i=0;i<count;++i) {
            const auto slot=coordinateIndex(keyAt(i));auto &state=coordinate_state_[slot];const auto &row=rowAt(i);
            const auto observed=state.migration_observations.load(std::memory_order_relaxed);
            migration_observations_seen_.fetch_add(1,std::memory_order_relaxed);
            if(observed<config_.warmup_samples_per_coordinate)warmup_migration_observations_.fetch_add(1,std::memory_order_relaxed);
            else {
                const auto measured=observed-config_.warmup_samples_per_coordinate;
                if(measured<config_.measured_samples_per_coordinate) {
                    wave_samples_[sampleOffset(slot,measured)]=row.wave_wall_nanoseconds;
                    for(std::size_t piece=0;piece<state.piece_count;++piece) {
                        const auto sample=*sampleAt(row,piece);piece_samples_[pieceSampleOffset(slot,piece,measured)]=sample;
                        if(!piece_bytes_[state.first_piece+piece])piece_bytes_[state.first_piece+piece]=quoteAt(row,piece);
                    }
                    stored_migration_observations_.fetch_add(1,std::memory_order_relaxed);
                }else excess_migration_observations_.fetch_add(1,std::memory_order_relaxed);
            }
            state.migration_observations.store(observed+1,std::memory_order_release);
        }
        completed_waves_observed_.fetch_add(1,std::memory_order_relaxed);return true;
    }

    bool MoEOverlayMigrationMeasurementLedger::recordInterferenceSample(
        MoEOverlayMigrationMeasurementCoordinate coordinate,
        ExpertHistogramSource source,
        std::uint64_t inference_only_nanoseconds,
        std::uint64_t concurrent_nanoseconds,
        std::string *error) noexcept
    {
        if (error)
            error->clear();
        const std::size_t coordinate_index = coordinate.valid()?coordinateIndex({MoEOverlayRouteTarget::fromWord(coordinate.source_participant),
            MoEOverlayRouteTarget::fromWord(coordinate.destination_participant),coordinate.layer}):kMissingCoordinate;
        const auto phase_index = static_cast<std::size_t>(source);
        if (sealed_.load(std::memory_order_acquire) ||
            coordinate_index == kMissingCoordinate ||
            source == ExpertHistogramSource::SyntheticTest ||
            phase_index >= kExpertHistogramProductionSourceCount ||
            !config_.required_sources[phase_index] ||
            inference_only_nanoseconds == 0 || concurrent_nanoseconds == 0)
        {
            rejected_observations_.fetch_add(1, std::memory_order_relaxed);
            assignLedgerError(
                error,
                "ExpertOverlay migration ledger received an invalid interference pair or is sealed");
            return false;
        }

        auto &state = coordinate_state_[coordinate_index];
        const std::uint64_t observed =
            state.interference_observations[phase_index].load(
                std::memory_order_relaxed);
        if (observed < config_.warmup_samples_per_coordinate)
        {
            warmup_interference_observations_.fetch_add(
                1, std::memory_order_relaxed);
        }
        else
        {
            const std::uint64_t measured_index =
                observed - config_.warmup_samples_per_coordinate;
            if (measured_index < config_.measured_samples_per_coordinate)
            {
                interference_samples_[interferenceSampleOffset(
                    coordinate_index, phase_index, measured_index)] =
                    concurrent_nanoseconds > inference_only_nanoseconds
                        ? concurrent_nanoseconds - inference_only_nanoseconds
                        : 0u;
                stored_interference_observations_.fetch_add(
                    1, std::memory_order_relaxed);
            }
            else
            {
                excess_interference_observations_.fetch_add(
                    1, std::memory_order_relaxed);
            }
        }
        state.interference_observations[phase_index].store(
            observed + 1u, std::memory_order_release);
        return true;
    }

    bool MoEOverlayMigrationMeasurementLedger::ready() const noexcept
    {
        if (sealed_.load(std::memory_order_acquire))
            return false;
        const std::uint64_t required =
            config_.warmup_samples_per_coordinate +
            config_.measured_samples_per_coordinate;
        for (std::size_t index = 0; index < coordinates_.size(); ++index)
        {
            if (coordinate_state_[index].migration_observations.load(
                    std::memory_order_acquire) < required)
            {
                return false;
            }
        }
        return true;
    }

    bool MoEOverlayMigrationMeasurementLedger::matchesDeclarations(const MoEOverlaySealedMigrationMeasurements &measurements,
        std::string *error) const noexcept
    {
        if(error)error->clear();
        const auto reject=[&](const char *message){assignLedgerError(error,message);return false;};
        if(!measurements.valid() || measurements.rows.size()!=coordinates_.size())
            return reject("Presealed movement evidence does not cover the original declared target set");
        for(std::size_t i=0;i<coordinates_.size();++i) {
            const MoEOverlayTargetLayerMigrationMeasurement *row=nullptr;
            for(const auto &candidate:measurements.rows)if(candidate.coordinate==coordinates_[i]) {
                if(row)return reject("Presealed movement evidence repeats an original target coordinate");
                row=&candidate;
            }
            if(!row)return reject("Presealed movement evidence has a foreign or missing original target coordinate");
            if (row->measured_coordinate)
                return reject("Equivalent-layer projection is not a directly measured calibration corpus");
            const auto &state=coordinate_state_[i];
            if(state.geometry) {
                if(!row->geometry || !state.geometry->sameIdentity(*row->geometry))
                    return reject("Presealed movement evidence differs from original members, cuts or formats");
            } else if(!row->coordinate.source_target.isWhole() || !row->coordinate.destination_target.isWhole())
                return reject("Legacy whole declarations cannot authenticate cohort reuse");
            if((row->coordinate.source_target.isCPUCohort() || row->coordinate.destination_target.isCPUCohort()) && !state.setup_quotes_bound)
                return reject("Cohort presealed reuse requires original setup-bound piece quotes");
            if(row->pieces.size()!=state.piece_count)return reject("Presealed movement evidence has incomplete original piece coverage");
            for(std::size_t piece=0;piece<state.piece_count;++piece) {
                const auto expected=piece_bytes_[state.first_piece+piece];
                if(!row->pieces[piece].valid() || (expected && expected!=row->pieces[piece].quoted_payload_bytes))
                    return reject("Presealed movement evidence differs from original bound piece quotes");
            }
        }
        return true;
    }

    MoEOverlaySealedMigrationMeasurements MoEOverlayMigrationMeasurementLedger::seal()
    {
        if(sealed_.load(std::memory_order_acquire))throw std::logic_error("Migration ledger was already sealed");
        if(!ready())throw std::logic_error("Migration ledger cannot seal incomplete target evidence");
        MoEOverlaySealedMigrationMeasurements result;result.rows.reserve(coordinates_.size());
        const auto sample_count=static_cast<std::size_t>(config_.measured_samples_per_coordinate);
        for(std::size_t coordinate=0;coordinate<coordinates_.size();++coordinate) {
            const auto &state=coordinate_state_[coordinate];
            const auto begin=sampleOffset(coordinate,0),end=begin+sample_count;
            MoEOverlayTargetLayerMigrationMeasurement row{.coordinate=coordinates_[coordinate],.geometry=state.geometry,
                .wave_wall_nanoseconds=median(std::vector<std::uint64_t>(wave_samples_.begin()+begin,wave_samples_.begin()+end)),
                .wave_sample_count=config_.measured_samples_per_coordinate};
            row.pieces.reserve(state.piece_count);
            for(std::size_t piece=0;piece<state.piece_count;++piece) {
                std::array<std::vector<std::uint64_t>,4> components;
                for(auto &component:components)component.reserve(sample_count);
                for(std::size_t sample=0;sample<sample_count;++sample) {
                    const auto &value=piece_samples_[pieceSampleOffset(coordinate,piece,sample)];
                    components[0].push_back(value.wall_nanoseconds);components[1].push_back(value.device_nanoseconds);
                    components[2].push_back(value.host_nanoseconds);components[3].push_back(value.transport_nanoseconds);
                }
                const auto bytes=piece_bytes_[state.first_piece+piece];
                row.pieces.push_back({bytes,{.sequence=1,.bytes=bytes,.wall_nanoseconds=median(std::move(components[0])),
                    .device_nanoseconds=median(std::move(components[1])),.host_nanoseconds=median(std::move(components[2])),
                    .transport_nanoseconds=median(std::move(components[3]))}});
            }
            result.rows.push_back(std::move(row));
        }
        // Evidence identity covers metadata and medians, never model/source payload bytes.
        std::uint64_t hash=kFNV1a64OffsetBasis;hashString(hash,"MoEOverlayMigrationMeasurementLedger/v4");
        hashString(hash,config_.measurement_identity);hashUnsigned(hash,config_.warmup_samples_per_coordinate);
        hashUnsigned(hash,config_.measured_samples_per_coordinate);
        for(const auto &row:result.rows) {
            hashUnsigned(hash,static_cast<std::uint64_t>(static_cast<std::int64_t>(row.coordinate.source_target.word())));
            hashUnsigned(hash,static_cast<std::uint64_t>(static_cast<std::int64_t>(row.coordinate.destination_target.word())));
            hashUnsigned(hash,row.coordinate.layer);hashUnsigned(hash,bool(row.geometry));
            if(row.geometry) {
                for(const auto *target:{&row.geometry->targets().first,&row.geometry->targets().second}) {
                    hashUnsigned(hash,target->tier_idx);hashUnsigned(hash,target->members.size());
                    for(const auto &member:target->members) {
                        hashUnsigned(hash,member.participant_id);hashUnsigned(hash,member.world_rank);
                        hashString(hash,member.address.toString());
                        hashUnsigned(hash,static_cast<std::uint64_t>(member.device.type));hashUnsigned(hash,member.device.ordinal);
                        for(const auto &projection:member.projections) {
                            hashUnsigned(hash,static_cast<std::uint64_t>(projection.residence));
                            for(int value:{projection.source_rows,projection.source_columns,projection.first_row,projection.rows,projection.first_column,projection.columns})hashUnsigned(hash,value);
                        }
                    }
                }
                for(const auto &format:row.geometry->formats()) {
                    hashUnsigned(hash,static_cast<std::uint64_t>(format.kind));hashUnsigned(hash,format.native_vnni.codebook_id);
                    hashUnsigned(hash,format.native_vnni.is_superblock);hashUnsigned(hash,format.native_vnni.present);
                }
            }
            hashUnsigned(hash,row.wave_wall_nanoseconds);hashUnsigned(hash,row.wave_sample_count);
            for(const auto &piece:row.pieces) {
                hashUnsigned(hash,piece.quoted_payload_bytes);
                const auto &sample=piece.measurement;
                for(auto value:{sample.bytes,sample.wall_nanoseconds,sample.device_nanoseconds,sample.host_nanoseconds,sample.transport_nanoseconds})hashUnsigned(hash,value);
            }
        }
        std::ostringstream identity;identity<<"expert-overlay-migration-measurements-v4/"<<std::hex<<std::setfill('0')<<std::setw(16)<<hash;
        result.identity=identity.str();
        if(!result.valid())throw std::logic_error("Migration ledger produced incomplete sealed original-target evidence");
        bool expected=false;
        if(!sealed_.compare_exchange_strong(expected,true,std::memory_order_acq_rel))throw std::logic_error("Migration ledger raced with another seal");
        return result;
    }

    MoEOverlayMigrationMeasurementLedgerStats
    MoEOverlayMigrationMeasurementLedger::stats() const noexcept
    {
        return {
            .completed_waves_observed = completed_waves_observed_.load(
                std::memory_order_relaxed),
            .migration_observations_seen = migration_observations_seen_.load(
                std::memory_order_relaxed),
            .warmup_migration_observations =
                warmup_migration_observations_.load(
                    std::memory_order_relaxed),
            .stored_migration_observations =
                stored_migration_observations_.load(
                    std::memory_order_relaxed),
            .excess_migration_observations =
                excess_migration_observations_.load(
                    std::memory_order_relaxed),
            .warmup_interference_observations =
                warmup_interference_observations_.load(
                    std::memory_order_relaxed),
            .stored_interference_observations =
                stored_interference_observations_.load(
                    std::memory_order_relaxed),
            .excess_interference_observations =
                excess_interference_observations_.load(
                    std::memory_order_relaxed),
            .rejected_observations = rejected_observations_.load(
                std::memory_order_relaxed),
        };
    }
} // namespace llaminar2
