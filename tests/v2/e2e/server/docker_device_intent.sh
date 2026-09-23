#!/usr/bin/env bash
# =============================================================================
# Docker accelerator visibility for the production HTTP E2E harness.
#
# The model runner receives a CLI request, not a preselected device list. In
# automatic mode, a CUDA candidate can exist without any literal `cuda:N`
# address in argv. This parser therefore honors the backend constraint first,
# then explicit placement, and otherwise exposes CUDA so discovery can make
# the same decision that an ordinary `llaminar serve` invocation would make.
# =============================================================================

docker_args_need_cuda() {
    local -a args=("$@")
    local only_backends=""
    local has_only_backends=0
    local explicit_placement=0
    local has_cuda_address=0
    local index arg

    for ((index = 0; index < ${#args[@]}; ++index)); do
        arg="${args[index]}"
        case "$arg" in
            --only-backends)
                has_only_backends=1
                only_backends="${args[index+1]-}"
                ((++index))
                ;;
            --only-backends=*)
                has_only_backends=1
                only_backends="${arg#*=}"
                ;;
            --device|--tp-devices|--define-domain|--pp-stage|--expert-tier|\
            --moe-routed-expert-domain|--moe-routed-expert-tier)
                explicit_placement=1
                ;;
            --device=*|--tp-devices=*|--define-domain=*|--pp-stage=*|--expert-tier=*|\
            --moe-routed-expert-domain=*|--moe-routed-expert-tier=*)
                explicit_placement=1
                ;;
        esac
        case "$arg" in
            cuda:*|*cuda:*) has_cuda_address=1 ;;
        esac
    done

    # An explicit backend filter is the public authority for auto discovery.
    # Bracket commas to avoid matching a backend name merely containing cuda.
    if ((has_only_backends)); then
        case ",${only_backends}," in
            *,cuda,*) return 0 ;;
            *) return 1 ;;
        esac
    fi

    # Authored topology names its endpoints. Without authored placement, serve
    # defaults to auto, so CUDA must be visible for inventory discovery even
    # when no candidate has been selected yet. An applied config document is
    # intentionally treated as unknown here and gets device visibility too.
    if ((has_cuda_address)); then return 0; fi
    if ((explicit_placement)); then return 1; fi
    return 0
}
