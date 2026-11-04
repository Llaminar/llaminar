/**
 * @file MoEOverlayEconomyEvidenceCodec.h
 * @brief Exact-live observation codec bound to original local wave metadata.
 *
 * The packet names canonical piece ordinals and redundant original coordinates;
 * none of those fields can construct or replace receiver geometry. Empty
 * observers encode no payload. Buffers are maintenance-owned, never model data.
 */
#pragma once
#include "MoEOverlayMigrationEvidenceContext.h"
#include "MoEOverlayPreparedTargetService.h"
#include <algorithm>
#include <map>
#include <cstring>
#include <limits>
#include <span>
#include <type_traits>
namespace llaminar2 {
/** @brief Pure codec shared by MPI production and device-free replay tests. */
class MoEOverlayEconomyEvidenceCodec final {
    /** @brief Canonical identity fingerprint fields, not sender-authored layouts. */
    struct Coordinate {
        std::uint64_t migration=0,piece=0;
        std::uint64_t role=0,source=0,destination=0;
        std::uint64_t format=0,codebook=0,superblock=0;
        std::array<std::uint64_t,14> rectangle{};
        bool operator==(const Coordinate &) const = default;
    };
    /** @brief One live independently quoted completed endpoint observation. */
    struct Record {Coordinate coordinate;std::uint64_t quote=0;};
    /** @brief Compare exact original overlap and both local strides/origins.
     * @param migration Transaction ordinal. @param piece Canonical piece ordinal.
     * @param identity Locally retained original piece. @return Redundant wire identity. */
    static Coordinate coordinate(std::size_t migration,std::size_t piece,const MoEOverlayMigrationPieceMeasurement &identity) {
        const auto &x=*identity.intersection;
        return {migration,piece,static_cast<std::uint64_t>(identity.projection),static_cast<std::uint64_t>(identity.source_participant),
            static_cast<std::uint64_t>(identity.destination_participant),static_cast<std::uint64_t>(identity.format.kind),
            identity.format.native_vnni.codebook_id,identity.format.native_vnni.is_superblock?1u:0u,{static_cast<std::uint64_t>(x.source().source_rows),static_cast<std::uint64_t>(x.source().source_columns),
            static_cast<std::uint64_t>(x.firstRow()),static_cast<std::uint64_t>(x.firstColumn()),static_cast<std::uint64_t>(x.rows()),static_cast<std::uint64_t>(x.columns()),static_cast<std::uint64_t>(x.source().rows),static_cast<std::uint64_t>(x.source().columns),
            static_cast<std::uint64_t>(x.sourceRowOrigin()),static_cast<std::uint64_t>(x.sourceColumnOrigin()),static_cast<std::uint64_t>(x.destinationRowOrigin()),static_cast<std::uint64_t>(x.destinationColumnOrigin()),
            static_cast<std::uint64_t>(x.destination().rows),static_cast<std::uint64_t>(x.destination().columns)}};
    }
    /** @brief Append only one initialized protocol value. @param out Live packet.
     * @param value Exact protocol value. */
    template<class T> static void put(std::vector<std::byte> &out,const T &value) {
        static_assert(std::is_trivially_copyable_v<T>);const auto n=out.size();
        if(n>std::numeric_limits<std::size_t>::max()-sizeof(T))throw std::overflow_error("Economy evidence packet overflow");
        out.resize(n+sizeof(T));std::memcpy(out.data()+n,&value,sizeof(T));
    }
    /** @brief Consume one complete protocol value. @param in Remaining live bytes.
     * @return Decoded value. @throws std::invalid_argument For truncation. */
    template<class T> static T take(std::span<const std::byte> &in) {
        static_assert(std::is_trivially_copyable_v<T>);
        if(in.size()<sizeof(T))throw std::invalid_argument("Truncated economy observation packet");
        T value{};std::memcpy(&value,in.data(),sizeof(T));in=in.subspan(sizeof(T));return value;
    }
    /** @brief Authenticate a physical observer against original member ranks.
     * @param g Original local geometry. @param p Canonical piece. @param rank Actual MPI rank.
     * @return Whether this rank is an endpoint of this piece. */
    static bool owns(const MoEOverlayMigrationMeasurementGeometry &g,const MoEOverlayMigrationPieceMeasurement &p,int rank) {
        for(const auto *target:{&g.targets().first,&g.targets().second})for(const auto &m:target->members)
            if((m.participant_id==p.source_participant || m.participant_id==p.destination_participant) && m.world_rank==rank)return true;
        return false;
    }
public:
    /** @brief Quote the maximum protocol bytes for this rank's actual original incidence.
     * @param context Original local wave metadata. @param rank Actual observer rank.
     * @return Exact all-present upper bound; empty ranks require zero payload bytes. */
    static std::size_t maximumEncodedBytes(const MoEOverlayMigrationEvidenceContext &context,int rank) {
        std::size_t total=0;
        for(const auto &row:context.identities())for(const auto &piece:row.geometry->pieces())if(owns(*row.geometry,piece,rank)) {
            constexpr auto unit=sizeof(Record)+sizeof(ExpertTierProjectionTransferMeasurement);
            if(total>std::numeric_limits<std::size_t>::max()-unit)throw std::overflow_error("Economy evidence incidence overflows host size");
            total+=unit;
        }
        return total;
    }
    /**
     * @brief Encode present observations only after checking the local original wave.
     * @param context Original prepared-wave metadata. @param rows Actual local observations.
     * @param rank Actual observing rank. @return Exact live quote/sample bytes, empty for an uninvolved rank.
     * @throws std::invalid_argument For mismatched geometry or foreign observers.
     */
    static std::vector<std::byte> encode(const MoEOverlayMigrationEvidenceContext &context,
        const std::vector<MoEOverlayCompletedMigrationMeasurement> &rows,int rank) {
        if(rank<0 || !context.matches(rows))throw std::invalid_argument("Economy evidence differs from original local context");
        std::vector<std::byte> out;
        for(std::size_t m=0;m<rows.size();++m)for(std::size_t p=0;p<rows[m].piece_observations.size();++p){
            const auto &slot=rows[m].piece_observations[p];
            const auto &g=*context.identities()[m].geometry;const auto &id=g.pieces()[p];
            const bool required=owns(g,id,rank);
            if(!required && !slot.quoted_payload_bytes && !slot.observation)continue;
            if(!required || !slot.quoted_payload_bytes || !slot.observation)
                throw std::invalid_argument("Economy endpoint observer omitted a required local quote/sample");
            if(!slot.valid() || !owns(g,id,rank))throw std::invalid_argument("Economy evidence supplied by a nonendpoint observer");
            put(out,Record{coordinate(m,p,id),*slot.quoted_payload_bytes});
            put(out,*slot.observation);
        }
        return out;
    }
    /**
     * @brief Decode exact live bytes against local geometry; never accept sender target definitions.
     * @param context Original local prepared metadata. @param bytes Exact sender payload.
     * @param rank Actual collective sender rank. @param wave_wall_nanoseconds Sender's measured whole-wave duration.
     * @return Complete context-shaped partial rows, including empty observers.
     * @throws std::invalid_argument For duplicates, foreign coordinates, invalid samples or truncation.
     */
    static std::vector<MoEOverlayCompletedMigrationMeasurement> decode(const MoEOverlayMigrationEvidenceContext &context,
        std::span<const std::byte> bytes,int rank,std::uint64_t wave_wall_nanoseconds) {
        if(rank<0 || !wave_wall_nanoseconds)throw std::invalid_argument("Economy evidence missing observer or wall duration");
        std::vector<MoEOverlayCompletedMigrationMeasurement> rows;
        rows.reserve(context.identities().size());
        for(const auto &id:context.identities()) {
            rows.push_back({.expected_epoch=id.expected_epoch,.candidate_epoch=id.candidate_epoch,
                .source_participant=id.source_participant,.destination_participant=id.destination_participant,
                .layer=id.layer,.expert=id.expert,.wave_wall_nanoseconds=wave_wall_nanoseconds,.geometry=id.geometry,
                .piece_observations=std::vector<MoEOverlayMigrationPieceObservation>(id.geometry->pieces().size())});
        }
        while(!bytes.empty()) {
            const auto r=take<Record>(bytes);
            if(r.coordinate.migration>=rows.size())throw std::invalid_argument("Economy evidence migration ordinal outside original wave");
            auto &row=rows[r.coordinate.migration];const auto &g=*row.geometry;
            if(r.coordinate.piece>=g.pieces().size())throw std::invalid_argument("Economy evidence piece ordinal outside original geometry");
            const auto &id=g.pieces()[r.coordinate.piece];auto &slot=row.piece_observations[r.coordinate.piece];
            if(r.coordinate!=coordinate(r.coordinate.migration,r.coordinate.piece,id) || !owns(g,id,rank) ||
               slot.quoted_payload_bytes || !r.quote)
                throw std::invalid_argument("Economy evidence repeated or foreign original piece coordinate");
            slot.quoted_payload_bytes=r.quote;
            slot.observation=take<ExpertTierProjectionTransferMeasurement>(bytes);
            if(!slot.valid())throw std::invalid_argument("Economy observation differs from independent bound quote");
        }
        for(const auto &row:rows)for(std::size_t p=0;p<row.piece_observations.size();++p)
            if(owns(*row.geometry,row.geometry->pieces()[p],rank) &&
               (!row.piece_observations[p].quoted_payload_bytes || !row.piece_observations[p].observation))
                throw std::invalid_argument("Economy endpoint payload omitted required original piece evidence");
        return rows;
    }
    /** @brief Exact metadata bytes owned by one original root; inactive phases never travel.
     * @param plan Original receiver geometry. @param rank Actual observer.
     * @return One initialized 40-byte record per actual sampled phase. */
    static size_t preparedServiceBytes(const MoEOverlayPreparedTargetServicePlan& plan,int rank) {
        constexpr size_t sample_bytes=3*sizeof(int32_t)+sizeof(uint32_t)+3*sizeof(uint64_t);
        size_t count=0;
        for(const auto& entry:plan.entries())if(entry.root_world_rank==rank)
            for(const int layer:entry.sample_layers)if(layer==entry.coordinate.layer)++count;
        if(count>std::numeric_limits<size_t>::max()/sample_bytes)throw std::overflow_error("Prepared service wire extent overflows");
        return count*sample_bytes;
    }
    /** @brief Encode actual prepared phases against original geometry, not sender layouts.
     * @param plan Original checked metadata. @param rows Exact local root samples. @param rank Actual observer.
     * @return Live-only initialized values; unrelated/contributor observers are empty. */
    static std::vector<std::byte> encodePreparedService(const MoEOverlayPreparedTargetServicePlan& plan,
        std::span<const MoEOverlayPreparedTargetServiceTotals> rows,int rank) {
        plan.requireLocalSamples(rows,rank);std::vector<std::byte> result;result.reserve(preparedServiceBytes(plan,rank));
        for(const auto& row:rows)for(size_t phase=0;phase<kExpertHistogramProductionSourceCount;++phase)
            if(row.sample_count[phase]) {
                put(result,int32_t(row.coordinate.target.word()));put(result,int32_t(row.coordinate.layer));put(result,uint32_t(phase));
                put(result,row.total_nanoseconds[phase]);put(result,row.activation_count[phase]);
                put(result,row.sample_count[phase]);put(result,int32_t(row.expert_ids[phase]));
            }
        if(result.size()!=preparedServiceBytes(plan,rank))throw std::logic_error("Prepared service did not encode its exact live extent");
        return result;
    }
    /** @brief Decode coordinates only against original local declarations.
     * @param plan Original checked metadata. @param bytes Actual observer payload. @param rank Actual MPI rank.
     * @return Complete local root samples. @throws std::invalid_argument For missing/foreign/trailing evidence. */
    static std::vector<MoEOverlayPreparedTargetServiceTotals> decodePreparedService(
        const MoEOverlayPreparedTargetServicePlan& plan,std::span<const std::byte> bytes,int rank) {
        if(bytes.size()!=preparedServiceBytes(plan,rank))throw std::invalid_argument("Prepared service payload differs from original exact live phase extent");
        std::map<MoEOverlayTargetServiceCoordinate,MoEOverlayPreparedTargetServiceTotals> decoded;
        while(!bytes.empty()) {
            MoEOverlayTargetServiceCoordinate coordinate{MoEOverlayRouteTarget::fromWord(take<int32_t>(bytes)),take<int32_t>(bytes)};
            const auto phase=take<uint32_t>(bytes);
            const auto& declaration=plan.entry(coordinate);
            if(phase>=kExpertHistogramProductionSourceCount || declaration.root_world_rank!=rank ||
                declaration.sample_layers[phase]!=coordinate.layer)
                throw std::invalid_argument("Prepared service phase is outside original root declarations");
            auto& row=decoded[coordinate];row.coordinate=coordinate;
            if(row.sample_count[phase])throw std::invalid_argument("Prepared service repeats an original root phase");
            row.total_nanoseconds[phase]=take<uint64_t>(bytes);row.activation_count[phase]=take<uint64_t>(bytes);
            row.sample_count[phase]=take<uint64_t>(bytes);row.expert_ids[phase]=take<int32_t>(bytes);
        }
        std::vector<MoEOverlayPreparedTargetServiceTotals> rows;rows.reserve(decoded.size());
        for(auto& [coordinate,row]:decoded)rows.push_back(std::move(row));
        plan.requireLocalSamples(rows,rank);return rows;
    }

};
}
