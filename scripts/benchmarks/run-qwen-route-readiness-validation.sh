#!/bin/bash
set -euo pipefail

timeout_seconds=120
artifacts_root="${PWD}/.test-artifacts/qwen-route-readiness-validation"
baseline_dir=""
experimental_dir=""
speed_gate_output="${PWD}/.test-artifacts/prefill-profile/qwen35-prefill-full-profile-speed-gate.csv"

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
    --baseline-dir)
      baseline_dir="$2"
      shift 2
      ;;
    --experimental-dir)
      experimental_dir="$2"
      shift 2
      ;;
    --speed-gate-output)
      speed_gate_output="$2"
      shift 2
      ;;
    --help|-h)
      cat <<'EOF'
usage: scripts/benchmarks/run-qwen-route-readiness-validation.sh [options]

Validates Qwen3.5 prefill route-readiness artifacts and records replayable logs.

options:
  --timeout <seconds>      Per-test timeout. Default: 120.
  --artifacts-root <dir>   Directory for validation summary and logs.
                           Default: .test-artifacts/qwen-route-readiness-validation.
  --baseline-dir <dir>     Optional baseline full-profile artifact directory.
  --experimental-dir <dir> Optional experimental full-profile artifact directory.
  --speed-gate-output <file>
                           Output path for generated full-profile speed gate.
                           Default: .test-artifacts/prefill-profile/qwen35-prefill-full-profile-speed-gate.csv.

When both --baseline-dir and --experimental-dir are supplied, this script first
generates qwen35-prefill-full-profile-speed-gate.csv with the default 10%
minimum full-profile speedup gate before validating current route-readiness
reconstruction.
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 64
      ;;
  esac
done

if { [ -n "$baseline_dir" ] && [ -z "$experimental_dir" ]; } || { [ -z "$baseline_dir" ] && [ -n "$experimental_dir" ]; }; then
  echo "--baseline-dir and --experimental-dir must be supplied together." >&2
  exit 64
fi

run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="${artifacts_root}/${run_stamp}"
summary_csv="${run_dir}/summary.csv"

mkdir -p "$run_dir"
printf "phase,command,environment,result,log\n" > "$summary_csv"

append_result() {
  local phase="$1"
  local command_name="$2"
  local environment="$3"
  local result="$4"
  local log_file="$5"

  printf "%s,%s,%s,%s,%s\n" \
    "$phase" \
    "$command_name" \
    "$environment" \
    "$result" \
    "$log_file" >> "$summary_csv"
}

run_command() {
  local phase="$1"
  local command_name="$2"
  local environment="$3"
  shift 3

  local log_name="${phase}_${command_name}"
  log_name="${log_name//\//_}"
  log_name="${log_name//:/_}"
  log_name="${log_name// /_}"
  local log_file="${run_dir}/${log_name}.log"

  echo "[qwen-route-readiness] ${command_name}"
  if "$@" > "$log_file" 2>&1; then
    cat "$log_file"
    append_result "$phase" "$command_name" "$environment" "pass" "$log_file"
  else
    cat "$log_file"
    append_result "$phase" "$command_name" "$environment" "fail" "$log_file"
    echo "[qwen-route-readiness] FAILED: ${command_name}" >&2
    echo "[qwen-route-readiness] summary: ${summary_csv}" >&2
    exit 1
  fi
}

run_swift_test() {
  local phase="$1"
  local filter="$2"
  shift 2

  local environment="$*"
  if [ -z "$environment" ]; then
    environment="-"
  fi

  run_command "$phase" "$filter" "$environment" \
    env "$@" perl -e 'alarm shift; exec @ARGV' "$timeout_seconds" \
    swift test -Xswiftc -DENABLE_METAL_PROBES --filter "$filter"
}

if [ -n "$baseline_dir" ]; then
  run_command "speed_gate" "compare-qwen35-prefill-speed-gate.py" "-" \
    perl -e 'alarm shift; exec @ARGV' "$timeout_seconds" \
    scripts/benchmarks/compare-qwen35-prefill-speed-gate.py \
      --baseline-dir "$baseline_dir" \
      --experimental-dir "$experimental_dir" \
      --output "$speed_gate_output"
fi

run_swift_test "contract" "Qwen35PrefillProfileTests/routeReadinessCanBeReconstructedFromArtifactCSVs" \
  ENABLE_METAL_PROBES=1
run_swift_test "current_artifacts" "Qwen35PrefillProfileTests/currentRouteReadinessArtifactsCanBeReconstructedWhenRequested" \
  ENABLE_METAL_PROBES=1 \
  SWIFTLM_VALIDATE_QWEN_ROUTE_READINESS_ARTIFACTS=1

echo "[qwen-route-readiness] summary: ${summary_csv}"
echo "[qwen-route-readiness] OK"
