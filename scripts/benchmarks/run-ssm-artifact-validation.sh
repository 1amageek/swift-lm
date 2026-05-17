#!/bin/bash
set -euo pipefail

timeout_seconds=120
generate=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --timeout)
      timeout_seconds="$2"
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

run_swift_test() {
  local filter="$1"
  shift
  echo "[ssm-artifacts] swift test --filter ${filter}"
  env "$@" perl -e 'alarm shift; exec @ARGV' "$timeout_seconds" swift test --filter "$filter"
}

if [ "$generate" -eq 1 ]; then
  run_swift_test "SSMRecurrenceMicrobenchmarkTests/bf16SSMRecurrenceRealShapeMicrobench"
  run_swift_test "SSMRecurrenceMicrobenchmarkTests/bf16SSMRecurrencePhaseIsolationMicrobench"
  run_swift_test "SSMRecurrenceMicrobenchmarkTests/bf16SSMStateRecurrenceCandidateStabilityMicrobench"
fi

run_swift_test "SSMRecurrenceMicrobenchmarkTests/ssmArtifactManifestCoversHarnessOutputs"
run_swift_test "SSMRecurrenceMicrobenchmarkTests/ssmRouteArtifactsCanBeReconstructedWhenRequested" \
  SWIFTLM_VALIDATE_SSM_ROUTE_ARTIFACTS=1
run_swift_test "SSMRecurrenceMicrobenchmarkTests/ssmThreadgroupPolicyArtifactCanBeReconstructedWhenRequested" \
  SWIFTLM_VALIDATE_SSM_THREADGROUP_POLICY_ARTIFACTS=1
run_swift_test "SSMRecurrenceMicrobenchmarkTests/ssmStateCandidateFeasibilityArtifactCanBeReconstructedWhenRequested" \
  SWIFTLM_VALIDATE_SSM_STATE_FEASIBILITY_ARTIFACTS=1
run_swift_test "SSMRecurrenceMicrobenchmarkTests/ssmStateCandidateBridgeArtifactCanBeReconstructedWhenRequested" \
  SWIFTLM_VALIDATE_SSM_STATE_BRIDGE_ARTIFACTS=1
run_swift_test "SSMRecurrenceMicrobenchmarkTests/ssmPhaseFullBridgeArtifactsCanBeReconstructedWhenRequested" \
  SWIFTLM_VALIDATE_SSM_RECURRENCE_BRIDGE_ARTIFACTS=1
run_swift_test "SSMRecurrenceMicrobenchmarkTests/ssmArtifactManifestFilesCanBeParsedWhenRequested" \
  SWIFTLM_VALIDATE_SSM_ARTIFACT_MANIFEST=1

echo "[ssm-artifacts] OK"
