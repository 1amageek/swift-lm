#!/bin/bash
set -euo pipefail

timeout_seconds=120
artifacts_root="${PWD}/.test-artifacts/lfm25-a1b-readiness"
custom_filters=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --timeout)
      timeout_seconds="$2"
      shift 2
      ;;
    --artifacts-root)
      artifacts_root="$2"
      shift 2
      ;;
    --filter)
      custom_filters+=("$2")
      shift 2
      ;;
    --help|-h)
      cat <<'EOF'
usage: scripts/benchmarks/run-lfm25-a1b-readiness.sh [options]

options:
  --timeout <seconds>        Per-gate timeout. Defaults to 120.
  --artifacts-root <path>    Output directory. Defaults to .test-artifacts/lfm25-a1b-readiness.
  --filter <Swift test filter>
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 64
      ;;
  esac
done

mkdir -p "$artifacts_root"
export SWIFTPM_MODULECACHE_OVERRIDE="${SWIFTPM_MODULECACHE_OVERRIDE:-${PWD}/.cache/clang}"
mkdir -p "$SWIFTPM_MODULECACHE_OVERRIDE"
timestamp="$(date +%Y%m%d-%H%M%S)"
run_dir="${artifacts_root}/${timestamp}"
mkdir -p "$run_dir"
summary="${run_dir}/summary.csv"
metrics_log="${run_dir}/metrics.log"
printf "filter,status,elapsed_seconds,log\n" > "$summary"
printf "" > "$metrics_log"

now_seconds() {
  perl -MTime::HiRes=time -e 'printf "%.3f", time'
}

elapsed_since() {
  perl -e 'printf "%.3f", $ARGV[1] - $ARGV[0]' "$1" "$2"
}

if [ "${#custom_filters[@]}" -gt 0 ]; then
  filters=("${custom_filters[@]}")
else
  filters=(
    "LFM25A1BRealBundleTests/localLFM25A1BLoadsAndPreparesText"
    "LFM25A1BRealBundleTests/localLFM25A1BPreparesPromptVariantsAndRejectsImages"
    "LFM25A1BRealBundleTests/localLFM25A1BEmitsOneGreedyToken"
    "LFM25A1BRealBundleTests/localLFM25A1BMatchesHFShortTraceForStrictCapitalChat"
    "LFM25A1BRealBundleTests/localLFM25A1BPromptStateRestorePreservesVisibleOutput"
    "LFM25A1BRealBundleTests/defaultSparseMoERouteStaysBoundedAcrossPromptLengths"
    "LFM25A1BRealBundleTests/defaultSparseMoERouteStaysBoundedAcrossDecodeLengths"
    "LFM25A1BRealBundleTests/defaultSparseMoERouteMatchesHFTracesAcrossMultiplePrompts"
    "LFM25A1BRealBundleTests/defaultSparseMoERouteReportsDecodeTimingBreakdown"
    "LFM25A1BRealBundleTests/realPackedSparseMoEKernelMatchesCPUReference"
    "LFM25A1BRealBundleTests/splitSparseMoERouteMatchesHFFirstTokenAndClearsLegacySpeedGate"
    "MetalSourceGeneratorTests/sparseMoECompiles"
    "MetalSourceGeneratorTests/sparseMoEMonolithicRouteIsDiagnosticOnly"
    "MetalSourceGeneratorTests/sparseMoEPrefillMatchesCPUReference"
    "MetalSourceGeneratorTests/sparseMoESharedActivationTailRowsMatchCPUReference"
  )
fi

for filter in "${filters[@]}"; do
  slug="${filter//\//-}"
  log_path="${run_dir}/${slug}.log"
  echo "[lfm25-a1b-readiness] ${filter}"
  start_seconds="$(now_seconds)"
  if scripts/xcodebuild/test-timeout.sh "$timeout_seconds" -- swift test --filter "$filter" 2>&1 | tee "$log_path"; then
    end_seconds="$(now_seconds)"
    elapsed_seconds="$(elapsed_since "$start_seconds" "$end_seconds")"
    if grep -q "\[Skip\] LFM2.5-8B-A1B not cached" "$log_path"; then
      printf '"%s",skip,%s,"%s"\n' "$filter" "$elapsed_seconds" "$log_path" >> "$summary"
      echo "[lfm25-a1b-readiness] missing required local LFM2.5-8B-A1B bundle: ${filter}" >&2
      echo "[lfm25-a1b-readiness] logs: ${run_dir}" >&2
      exit 66
    fi
    printf '"%s",pass,%s,"%s"\n' "$filter" "$elapsed_seconds" "$log_path" >> "$summary"
    awk -v filter="$filter" '/^\[LFM2\.5/ { print "[" filter "] " $0 }' "$log_path" >> "$metrics_log"
  else
    status=$?
    end_seconds="$(now_seconds)"
    elapsed_seconds="$(elapsed_since "$start_seconds" "$end_seconds")"
    printf '"%s",fail,%s,"%s"\n' "$filter" "$elapsed_seconds" "$log_path" >> "$summary"
    awk -v filter="$filter" '/^\[LFM2\.5/ { print "[" filter "] " $0 }' "$log_path" >> "$metrics_log"
    echo "[lfm25-a1b-readiness] failed: ${filter} (status ${status})" >&2
    echo "[lfm25-a1b-readiness] logs: ${run_dir}" >&2
    exit "$status"
  fi
done

echo "[lfm25-a1b-readiness] summary: ${summary}"
