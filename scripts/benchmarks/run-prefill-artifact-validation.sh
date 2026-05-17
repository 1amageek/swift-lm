#!/bin/bash
set -euo pipefail

timeout_seconds=120
artifacts_root="${PWD}/.test-artifacts/prefill-artifact-validation"
qwen_baseline_dir=""
qwen_experimental_dir=""

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
    --qwen-baseline-dir)
      qwen_baseline_dir="$2"
      shift 2
      ;;
    --qwen-experimental-dir)
      qwen_experimental_dir="$2"
      shift 2
      ;;
    --help|-h)
      cat <<'EOF'
usage: scripts/benchmarks/run-prefill-artifact-validation.sh [options]

Runs the lightweight prefill artifact gates used before promoting prefill routes.

options:
  --timeout <seconds>        Per-child timeout. Default: 120.
  --artifacts-root <dir>     Directory for validation summary and logs.
                             Default: .test-artifacts/prefill-artifact-validation.
  --qwen-baseline-dir <dir>  Optional Qwen baseline full-profile artifact directory.
  --qwen-experimental-dir <dir>
                             Optional Qwen experimental full-profile artifact directory.

When both Qwen profile directories are supplied, the Qwen route-readiness gate
also regenerates the full-profile speed gate with the default 10% minimum
full-profile speedup requirement before validating route readiness.
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 64
      ;;
  esac
done

if { [ -n "$qwen_baseline_dir" ] && [ -z "$qwen_experimental_dir" ]; } || { [ -z "$qwen_baseline_dir" ] && [ -n "$qwen_experimental_dir" ]; }; then
  echo "--qwen-baseline-dir and --qwen-experimental-dir must be supplied together." >&2
  exit 64
fi

run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="${artifacts_root}/${run_stamp}"
summary_csv="${run_dir}/summary.csv"

mkdir -p "$run_dir"
printf "gate,command,result,log\n" > "$summary_csv"

append_result() {
  local gate="$1"
  local command_name="$2"
  local result="$3"
  local log_file="$4"

  printf "%s,%s,%s,%s\n" "$gate" "$command_name" "$result" "$log_file" >> "$summary_csv"
}

run_gate() {
  local gate="$1"
  local command_name="$2"
  shift 2

  local log_name="${gate}_${command_name}"
  log_name="${log_name//\//_}"
  log_name="${log_name//:/_}"
  log_name="${log_name// /_}"
  local log_file="${run_dir}/${log_name}.log"

  echo "[prefill-artifacts] ${command_name}"
  if "$@" > "$log_file" 2>&1; then
    cat "$log_file"
    append_result "$gate" "$command_name" "pass" "$log_file"
  else
    cat "$log_file"
    append_result "$gate" "$command_name" "fail" "$log_file"
    echo "[prefill-artifacts] FAILED: ${command_name}" >&2
    echo "[prefill-artifacts] summary: ${summary_csv}" >&2
    exit 1
  fi
}

run_gate "ssm" "run-ssm-artifact-validation.sh" \
  scripts/benchmarks/run-ssm-artifact-validation.sh \
    --timeout "$timeout_seconds" \
    --artifacts-root "${run_dir}/ssm"

qwen_args=(
  scripts/benchmarks/run-qwen-route-readiness-validation.sh
  --timeout "$timeout_seconds"
  --artifacts-root "${run_dir}/qwen-route-readiness"
)
if [ -n "$qwen_baseline_dir" ]; then
  qwen_args+=(--baseline-dir "$qwen_baseline_dir")
  qwen_args+=(--experimental-dir "$qwen_experimental_dir")
fi

run_gate "qwen_route_readiness" "run-qwen-route-readiness-validation.sh" "${qwen_args[@]}"

echo "[prefill-artifacts] summary: ${summary_csv}"
echo "[prefill-artifacts] OK"
