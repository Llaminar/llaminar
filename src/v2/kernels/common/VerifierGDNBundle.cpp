/**
 * @file VerifierGDNBundle.cpp
 * @brief Validate four-source GDN ownership and exact copied subgroup views.
 *
 * Original and subgroup scopes exist only on the synchronous recording worker.
 * Empty or stale subgroups cannot inherit another operation's measured key.
 */
#include "VerifierGDNBundle.h"
#include <algorithm>
#include <exception>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
VerifierGDNBundleBinding::VerifierGDNBundleBinding(DeviceId device,const TensorBase *input,
    DeviceRowRange rows,int reduction,Projections projections,QuantizedActivationOperandPrecision precision,
    std::optional<DeviceRowRange> correction_rows) noexcept
    :device_(device),input_(input),rows_(rows),reduction_(reduction),projections_(projections),
     precision_(precision),correction_rows_(correction_rows) {}

VerifierGDNBundleBinding VerifierGDNBundleBinding::bind(DeviceId device,const TensorBase *input,
    DeviceRowRange rows,int reduction,Projections projections,QuantizedActivationOperandPrecision precision,
    std::optional<DeviceRowRange> correction_rows)
{
    if((!device.is_cpu() && !device.is_gpu()) || !input || rows.physicalRows()<1 || reduction<1 || projections.size()!=4)
        throw std::invalid_argument("GDN verifier requires its complete original four-role operation");
    if(precision==QuantizedActivationOperandPrecision::TwoTermQ8) {
        if(!correction_rows || correction_rows->physicalRows()!=rows.physicalRows())
            throw std::invalid_argument("GDN verifier correction must retain the original physical width");
    } else if(precision!=QuantizedActivationOperandPrecision::SingleQ8 || correction_rows)
        throw std::invalid_argument("GDN verifier has an inconsistent QKV precision/row policy");
    for(size_t role=0;role<4;++role) {
        const auto &p=projections[role];
        if(!p.kernel || !p.output || p.n<1)
            throw std::invalid_argument("GDN verifier has an incomplete original role");
        for(size_t prior=0;prior<role;++prior)if(p.output==projections[prior].output)
            throw std::invalid_argument("GDN verifier aliases semantic outputs");
    }
    return {device,input,rows,reduction,projections,precision,correction_rows};
}

VerifierGDNSubgroup::VerifierGDNSubgroup(const VerifierGDNBundleBinding &original,
    std::span<const VerifierGDNRole> roles,VerifierGDNBundleBinding::Projections copied) noexcept
    :original_(&original),copied_(copied)
{std::copy(roles.begin(),roles.end(),roles_.begin());}

bool VerifierGDNSubgroup::membersMatch() const noexcept
{
    int previous=-1;
    for(size_t index=0;index<copied_.size();++index) {
        const int role=static_cast<int>(roles_[index]);
        if(role<=previous || role>=4 || (role==0 && original_->precision()==QuantizedActivationOperandPrecision::TwoTermQ8))return false;
        const auto &a=original_->projections()[role],&b=copied_[index];
        if(a.kernel!=b.kernel || a.output!=b.output || a.n!=b.n || a.bias!=b.bias || a.original_output_slice!=b.original_output_slice)return false;
        previous=role;
    }
    return true;
}

VerifierGDNSubgroup VerifierGDNBundleBinding::subgroup(std::span<const VerifierGDNRole> roles,Projections copied) const
{
    if(roles.empty() || roles.size()>4 || roles.size()!=copied.size())
        throw std::invalid_argument("GDN verifier subgroup needs a bounded original role for every copied member");
    VerifierGDNSubgroup view(*this,roles,copied);
    if(!view.membersMatch())throw std::invalid_argument("GDN verifier subgroup changed or reordered its original roles");
    return view;
}

bool VerifierGDNSubgroup::matches(DeviceId device,const TensorBase *input,const DeviceRowRange &rows,
    int reduction,VerifierGDNBundleBinding::Projections copied) const noexcept
{
    return ScopedVerifierGDNBundle::current()==original_ && original_->device()==device &&
        original_->input()==input && original_->rows().captureIdentity()==rows.captureIdentity() &&
        original_->reduction()==reduction && copied_.data()==copied.data() && copied_.size()==copied.size() && membersMatch();
}

thread_local const ScopedVerifierGDNBundle *ScopedVerifierGDNBundle::active_=nullptr;
ScopedVerifierGDNBundle::ScopedVerifierGDNBundle(std::optional<VerifierGDNBundleBinding> binding) noexcept
    :binding_(std::move(binding)),previous_(active_) {active_=this;}
ScopedVerifierGDNBundle::~ScopedVerifierGDNBundle()
{
    const auto *subgroup=ScopedVerifierGDNSubgroup::current();
    if(active_!=this || (binding_ && subgroup && &subgroup->original()==&*binding_))std::terminate();
    active_=previous_;
}
const VerifierGDNBundleBinding *ScopedVerifierGDNBundle::current() noexcept
{return active_ && active_->binding_?&*active_->binding_:nullptr;}

thread_local const ScopedVerifierGDNSubgroup *ScopedVerifierGDNSubgroup::active_=nullptr;
ScopedVerifierGDNSubgroup::ScopedVerifierGDNSubgroup(std::optional<VerifierGDNSubgroup> subgroup)
    :subgroup_(std::move(subgroup)),previous_(active_)
{
    if(subgroup_ && ScopedVerifierGDNBundle::current()!=subgroup_->original_)
        throw std::invalid_argument("GDN subgroup's original scope is no longer current");
    active_=this;
}
ScopedVerifierGDNSubgroup::~ScopedVerifierGDNSubgroup()
{if(active_!=this)std::terminate();active_=previous_;}
const VerifierGDNSubgroup *ScopedVerifierGDNSubgroup::current() noexcept
{return active_ && active_->subgroup_?&*active_->subgroup_:nullptr;}
}
