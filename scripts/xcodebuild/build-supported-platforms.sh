#!/bin/bash
set -euo pipefail

timeout_seconds=120

while [ "$#" -gt 0 ]; do
  case "$1" in
    --timeout)
      timeout_seconds="$2"
      shift 2
      ;;
    *)
      echo "usage: $0 [--timeout SECONDS]" >&2
      exit 64
      ;;
  esac
done

if ! [[ "$timeout_seconds" =~ ^[1-9][0-9]*$ ]] || [ "$timeout_seconds" -gt 120 ]; then
  echo "--timeout must be an integer between 1 and 120" >&2
  exit 64
fi

repository_root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$repository_root"

build_destination() {
  local label="$1"
  local destination="$2"

  echo "[platform-build] ${label}: ${destination}"
  scripts/xcodebuild/test-timeout.sh "$timeout_seconds" -- \
    xcodebuild build \
      -quiet \
      -scheme swift-lm-Package \
      -destination "$destination"
}

build_destination "iOS" "generic/platform=iOS"
build_destination "Mac Catalyst" "generic/platform=macOS,variant=Mac Catalyst"

echo "[platform-build] supported platform builds passed"
