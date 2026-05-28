#!/usr/bin/env bash
# ===========================================================================
# Hygiene check: no usage of the default (nullptr/0) CUDA/HIP stream in
# production GPU kernels and stages.
#
# The default stream serializes all GPU work across the device. Every kernel
# and async memcpy MUST use an explicit stream for correctness and overlap.
#
# Scanned directories:
#   src/v2/kernels/cuda/    src/v2/kernels/rocm/
#   src/v2/execution/       src/v2/backends/rocm/
#   src/v2/backends/cuda/
#
# Detected patterns:
#   1. CUDA kernel launch with 3 args (missing stream): <<<grid, block, smem>>>
#   2. hipLaunchKernelGGL with literal 0/nullptr as stream (5th positional arg)
#   3. hipMemcpyAsync / cudaMemcpyAsync / hipMemsetAsync / cudaMemsetAsync
#      with nullptr or literal 0 as stream argument
#   4. Default parameter "hipStream_t ... = 0" or "cudaStream_t ... = nullptr"
#      in function/method signatures (allows callers to accidentally omit stream)
#
# Files with known violations are tracked in the allowlist below. The goal is
# to shrink this list over time — never add to it without a strong reason.
# ===========================================================================
set -euo pipefail

if [[ $# -gt 0 ]]; then
    repo_root="$1"
else
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    repo_root="$(cd "${script_dir}/.." && pwd)"
fi

# Directories to scan (relative to repo_root)
scan_dirs=(
    "src/v2/kernels/cuda"
    "src/v2/kernels/rocm"
    "src/v2/execution"
    "src/v2/backends/cuda"
    "src/v2/backends/rocm"
)

# --- Known violations (legacy code to be fixed) ---
# Each entry is a relative path from repo_root.
# These files are SKIPPED during scanning. Remove once fixed.
allowed_exception_files=(
    # Tensor validation utilities (debug-only) use default stream.
    "src/v2/backends/rocm/ROCmTensorValidation.cpp"
    "src/v2/backends/cuda/CUDATensorValidation.cu"
)

is_allowed_file()
{
    local rel_path="$1"
    for allowed in "${allowed_exception_files[@]}"; do
        if [[ "${rel_path}" == "${allowed}" ]]; then
            return 0
        fi
    done
    return 1
}

# ==========================================================================
# Pattern 1: CUDA kernel launch with only 3 args (no stream)
#   Matches: <<<expr, expr, expr>>>  (exactly 2 commas inside <<<>>>)
#   Good:    <<<expr, expr, expr, stream>>>  (3 commas = has stream)
# ==========================================================================
check_cuda_launch_no_stream()
{
    local file="$1"
    # Find all <<<...>>> launches and filter to those with exactly 2 commas (no stream)
    grep -n '<<<' "$file" 2>/dev/null | while IFS= read -r line; do
        # Extract the content between <<< and >>>
        launch_args=$(echo "$line" | sed -n 's/.*<<<\([^>]*\)>>>.*/\1/p')
        if [[ -z "$launch_args" ]]; then
            continue
        fi
        # Count commas — 2 commas = 3 args (grid, block, smem) = missing stream
        comma_count=$(echo "$launch_args" | tr -cd ',' | wc -c)
        if [[ "$comma_count" -lt 3 ]]; then
            echo "$line"
        fi
    done
}

# ==========================================================================
# Pattern 2: hipLaunchKernelGGL with 0/nullptr as stream
#   Format: hipLaunchKernelGGL(kernel, grid, block, smem, stream, ...)
#   Bad:    hipLaunchKernelGGL(kernel, dim3(N), dim3(N), 0, 0, ...)
#                                                           ^ stream = 0
#
#   Note: kernel arg may contain template commas like (kernel<64, TILE>).
#   We handle this by matching the ", 0, 0," pattern (smem=0, stream=0)
#   or by using perl to count balanced parens.
# ==========================================================================
check_hip_launch_default_stream()
{
    local file="$1"
    grep -n 'hipLaunchKernelGGL' "$file" 2>/dev/null | while IFS= read -r line; do
        # Use perl to properly parse the 5th argument accounting for balanced parens
        # hipLaunchKernelGGL(kernel, grid, block, smem, STREAM, ...)
        stream_arg=$(echo "$line" | LC_ALL=C perl -ne '
            if (/hipLaunchKernelGGL\s*\(/) {
                my $rest = $'\'';  # after the match
                $rest = substr($_, pos($_));
                # Remove "hipLaunchKernelGGL(" prefix
                $rest =~ s/^.*?hipLaunchKernelGGL\s*\(//;
                # Parse args respecting balanced parens and angle brackets
                my @args;
                my $depth_p = 0; my $depth_a = 0;
                my $current = "";
                for my $ch (split //, $rest) {
                    if ($ch eq "(" ) { $depth_p++; $current .= $ch; }
                    elsif ($ch eq ")") {
                        if ($depth_p == 0) { push @args, $current; last; }
                        $depth_p--; $current .= $ch;
                    }
                    elsif ($ch eq "<") { $depth_a++; $current .= $ch; }
                    elsif ($ch eq ">") { $depth_a--; $current .= $ch; }
                    elsif ($ch eq "," && $depth_p == 0 && $depth_a == 0) {
                        push @args, $current; $current = "";
                    }
                    else { $current .= $ch; }
                }
                # 5th arg (index 4) is the stream
                if (scalar(@args) >= 5) {
                    my $stream = $args[4];
                    $stream =~ s/^\s+|\s+$//g;
                    if ($stream eq "0" || $stream eq "nullptr" || $stream eq "NULL") {
                        print "VIOLATION";
                    }
                }
            }
        ')
        if [[ "$stream_arg" == "VIOLATION" ]]; then
            echo "$line"
        fi
    done
}

# ==========================================================================
# Pattern 3: Async API with nullptr/0 as stream
#   hipMemcpyAsync(..., nullptr)  hipMemcpyAsync(..., 0)
#   cudaMemcpyAsync(..., nullptr) cudaMemsetAsync(..., nullptr)
# ==========================================================================
pattern_async_nullptr='(hip|cuda)(Memcpy|Memset|Memcpy2D)Async\s*\([^;]*,\s*(nullptr|0)\s*\)'

# ==========================================================================
# Pattern 4: Default stream parameter in function signatures
#   hipStream_t stream = 0    hipStream_t stream = nullptr
#   cudaStream_t stream = 0   cudaStream_t stream = nullptr
#
#   Excludes:
#   - Member variable initializers (name ends with _)
#   - Local variable declarations (line ends with ;)
#   - Comments
# ==========================================================================
pattern_default_param='(hip|cuda)Stream_t\s+\w+\s*=\s*(0|nullptr)'

is_default_param_violation()
{
    local line="$1"
    # Skip member variable initializers (variable name ending with _)
    if echo "$line" | grep -qE '(hip|cuda)Stream_t\s+\w+_\s*='; then
        return 1
    fi
    # Skip comments
    if echo "$line" | grep -qE '^\s*(//|/\*|\*)'; then
        return 1
    fi
    # Skip local variable declarations: line ends with "= 0;" or "= nullptr;"
    # (these are assigned before use, not default parameters)
    if echo "$line" | grep -qE '(hip|cuda)Stream_t\s+\w+\s*=\s*(0|nullptr)\s*;'; then
        return 1
    fi
    return 0
}

# ==========================================================================
# Scan
# ==========================================================================
violations=()
declare -A seen_exception_files=()

for dir in "${scan_dirs[@]}"; do
    abs_dir="${repo_root}/${dir}"
    if [[ ! -d "$abs_dir" ]]; then
        continue
    fi

    # Find GPU source files (.cu, .hip, .cpp, .h in GPU dirs)
    while IFS= read -r -d '' file; do
        rel_path="${file#"${repo_root}/"}"

        if is_allowed_file "${rel_path}"; then
            seen_exception_files["${rel_path}"]=1
            continue
        fi

        # Skip test files that end up in these dirs
        if [[ "$rel_path" == *"/test/"* || "$rel_path" == *"/tests/"* || "$rel_path" == *"Test__"* ]]; then
            continue
        fi

        file_violations=""

        # Check Pattern 1: CUDA 3-arg launch (only for .cu files)
        if [[ "$file" == *.cu ]]; then
            p1_hits=$(check_cuda_launch_no_stream "$file" || true)
            if [[ -n "$p1_hits" ]]; then
                file_violations+="$p1_hits"$'\n'
            fi
        fi

        # Check Pattern 2: hipLaunchKernelGGL with 0/nullptr stream
        if [[ "$file" == *.hip || "$file" == *.cpp ]]; then
            p2_hits=$(check_hip_launch_default_stream "$file" || true)
            if [[ -n "$p2_hits" ]]; then
                file_violations+="$p2_hits"$'\n'
            fi
        fi

        # Check Pattern 3: Async API with nullptr/0
        p3_hits=$(grep -n -E "$pattern_async_nullptr" "$file" 2>/dev/null || true)
        if [[ -n "$p3_hits" ]]; then
            file_violations+="$p3_hits"$'\n'
        fi

        # Check Pattern 4: Default stream parameter in function signatures
        p4_hits=$(grep -n -E "$pattern_default_param" "$file" 2>/dev/null || true)
        if [[ -n "$p4_hits" ]]; then
            while IFS= read -r hit; do
                if is_default_param_violation "$hit"; then
                    file_violations+="$hit"$'\n'
                fi
            done <<< "$p4_hits"
        fi

        if [[ -n "$file_violations" ]]; then
            while IFS= read -r v; do
                [[ -z "$v" ]] && continue
                violations+=("${rel_path}:${v}")
            done <<< "$file_violations"
        fi
    done < <(find "$abs_dir" -type f \( -name '*.cu' -o -name '*.hip' -o -name '*.cpp' -o -name '*.h' \) -print0)
done

# Check for stale exception entries
stale_exceptions=()
for allowed in "${allowed_exception_files[@]}"; do
    if [[ -z "${seen_exception_files[${allowed}]:-}" ]]; then
        # Only flag as stale if the file still exists (could have been deleted)
        if [[ -f "${repo_root}/${allowed}" ]]; then
            stale_exceptions+=("${allowed}")
        fi
    fi
done

# ==========================================================================
# Report
# ==========================================================================
if (( ${#violations[@]} > 0 )); then
    echo "╔══════════════════════════════════════════════════════════════════╗" >&2
    echo "║  DEFAULT STREAM USAGE DETECTED IN GPU CODE                     ║" >&2
    echo "╚══════════════════════════════════════════════════════════════════╝" >&2
    echo "" >&2
    echo "All CUDA/HIP kernel launches and async API calls MUST use an" >&2
    echo "explicit stream. The default stream (nullptr/0) serializes all" >&2
    echo "GPU work and prevents overlap with compute." >&2
    echo "" >&2
    echo "Violations:" >&2
    printf '  %s\n' "${violations[@]}" >&2
    echo "" >&2
    echo "Fix: Pass the stage/kernel stream explicitly to all GPU calls." >&2
    echo "If this is intentional legacy code, add to the allowlist in:" >&2
    echo "  scripts/check_no_default_stream_in_gpu_code.sh" >&2
    exit 1
fi

if (( ${#stale_exceptions[@]} > 0 )); then
    echo "Default-stream allowlist contains files that no longer have violations." >&2
    echo "Remove these fixed files from scripts/check_no_default_stream_in_gpu_code.sh:" >&2
    echo "" >&2
    printf '  %s\n' "${stale_exceptions[@]}" >&2
    exit 1
fi

echo "✓ No default-stream usage in GPU code (checked ${#scan_dirs[@]} directories)"
exit 0
