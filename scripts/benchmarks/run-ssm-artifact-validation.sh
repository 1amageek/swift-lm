#!/bin/bash
set -euo pipefail

timeout_seconds=120
generate=0
artifacts_root="${PWD}/.test-artifacts/ssm-artifact-validation"

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
    --generate)
      generate=1
      shift
      ;;
    --help|-h)
      cat <<'EOF'
usage: scripts/benchmarks/run-ssm-artifact-validation.sh [options]

Validates SSM recurrence microbenchmark artifacts and their reconstruction gates.

options:
  --timeout <seconds>   Per-test timeout. Default: 120.
  --artifacts-root <dir> Directory for validation summary and logs.
                        Default: .test-artifacts/ssm-artifact-validation.
  --generate            Regenerate SSM microbenchmark artifacts before validation.
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 64
      ;;
  esac
done

run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="${artifacts_root}/${run_stamp}"
summary_csv="${run_dir}/summary.csv"

mkdir -p "$run_dir"
printf "phase,filter,environment,result,log\n" > "$summary_csv"

append_result() {
  local phase="$1"
  local filter="$2"
  local environment="$3"
  local result="$4"
  local log_file="$5"

  printf "%s,%s,%s,%s,%s\n" \
    "$phase" \
    "$filter" \
    "$environment" \
    "$result" \
    "$log_file" >> "$summary_csv"
}

run_swift_test() {
  local phase="$1"
  local filter="$2"
  shift 2

  local environment="$*"
  if [ -z "$environment" ]; then
    environment="-"
  fi

  local log_name="${phase}_${filter}"
  log_name="${log_name//\//_}"
  log_name="${log_name//:/_}"
  log_name="${log_name// /_}"
  local log_file="${run_dir}/${log_name}.log"

  echo "[ssm-artifacts] swift test --filter ${filter}"
  if env "$@" perl -e 'alarm shift; exec @ARGV' "$timeout_seconds" swift test --filter "$filter" > "$log_file" 2>&1; then
    cat "$log_file"
    append_result "$phase" "$filter" "$environment" "pass" "$log_file"
  else
    cat "$log_file"
    append_result "$phase" "$filter" "$environment" "fail" "$log_file"
    echo "[ssm-artifacts] FAILED: ${filter}" >&2
    echo "[ssm-artifacts] summary: ${summary_csv}" >&2
    exit 1
  fi
}

if [ "$generate" -eq 1 ]; then
  run_swift_test "generate" "SSMRecurrenceMicrobenchmarkTests/bf16SSMRecurrenceRealShapeMicrobench"
  run_swift_test "generate" "SSMRecurrenceMicrobenchmarkTests/bf16SSMRecurrencePhaseIsolationMicrobench"
  run_swift_test "generate" "SSMRecurrenceMicrobenchmarkTests/bf16SSMStateRecurrenceCandidateStabilityMicrobench"
fi

run_swift_test "manifest" "SSMRecurrenceMicrobenchmarkTests/ssmArtifactManifestCoversHarnessOutputs"
run_swift_test "reconstruct" "SSMRecurrenceMicrobenchmarkTests/ssmRouteArtifactsCanBeReconstructedWhenRequested" \
  SWIFTLM_VALIDATE_SSM_ROUTE_ARTIFACTS=1
run_swift_test "reconstruct" "SSMRecurrenceMicrobenchmarkTests/ssmThreadgroupPolicyArtifactCanBeReconstructedWhenRequested" \
  SWIFTLM_VALIDATE_SSM_THREADGROUP_POLICY_ARTIFACTS=1
run_swift_test "reconstruct" "SSMRecurrenceMicrobenchmarkTests/ssmStateCandidateFeasibilityArtifactCanBeReconstructedWhenRequested" \
  SWIFTLM_VALIDATE_SSM_STATE_FEASIBILITY_ARTIFACTS=1
run_swift_test "reconstruct" "SSMRecurrenceMicrobenchmarkTests/ssmStateCandidateBridgeArtifactCanBeReconstructedWhenRequested" \
  SWIFTLM_VALIDATE_SSM_STATE_BRIDGE_ARTIFACTS=1
run_swift_test "reconstruct" "SSMRecurrenceMicrobenchmarkTests/ssmPhaseFullBridgeArtifactsCanBeReconstructedWhenRequested" \
  SWIFTLM_VALIDATE_SSM_RECURRENCE_BRIDGE_ARTIFACTS=1
run_swift_test "manifest" "SSMRecurrenceMicrobenchmarkTests/ssmArtifactManifestFilesCanBeParsedWhenRequested" \
  SWIFTLM_VALIDATE_SSM_ARTIFACT_MANIFEST=1

echo "[ssm-artifacts] summary: ${summary_csv}"
echo "[ssm-artifacts] OK"
