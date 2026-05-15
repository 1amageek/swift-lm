#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def parse_sequence_lengths(value: str) -> list[int]:
    lengths: list[int] = []
    for part in value.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        lengths.append(int(stripped))
    if not lengths:
        raise ValueError("at least one sequence length is required")
    return lengths


def profile_path(directory: Path, pattern: str, sequence_length: int) -> Path:
    return directory / pattern.format(seq=sequence_length)


def profile_total_gpu_microseconds(path: Path) -> float:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is empty")
        if "averageGpuMicroseconds" not in reader.fieldnames:
            raise ValueError(f"{path} is missing averageGpuMicroseconds")
        total = 0.0
        for row in reader:
            value = row["averageGpuMicroseconds"]
            if value is None or value == "":
                raise ValueError(f"{path} contains an empty averageGpuMicroseconds value")
            total += float(value)
        return total


def collect_totals(directory: Path, pattern: str, sequence_lengths: list[int]) -> dict[int, float]:
    totals: dict[int, float] = {}
    for sequence_length in sequence_lengths:
        path = profile_path(directory, pattern, sequence_length)
        if path.exists():
            totals[sequence_length] = profile_total_gpu_microseconds(path)
    return totals


def format_float_list(values: list[float]) -> str:
    return "|".join(f"{value:.3f}" for value in values)


def write_speed_gate(
    output: Path,
    baseline_totals: dict[int, float],
    experimental_totals: dict[int, float],
    sequence_lengths: list[int],
    route_family: str,
    role: str,
    variant: str,
    minimum_speedup_percent: float,
) -> None:
    observed_sequence_lengths: list[int] = []
    baseline_values: list[float] = []
    experimental_values: list[float] = []
    speedup_percents: list[float] = []
    failing_sequence_lengths: list[int] = []
    missing_sequence_lengths: list[int] = []
    threshold_shortfall_percent = 0.0
    passing_sequence_count = 0

    for sequence_length in sequence_lengths:
        baseline = baseline_totals.get(sequence_length)
        experimental = experimental_totals.get(sequence_length)
        if baseline is None or experimental is None:
            missing_sequence_lengths.append(sequence_length)
            continue

        speedup_percent = ((baseline - experimental) / baseline * 100.0) if baseline > 0 else 0.0
        observed_sequence_lengths.append(sequence_length)
        baseline_values.append(baseline)
        experimental_values.append(experimental)
        speedup_percents.append(speedup_percent)
        if speedup_percent >= minimum_speedup_percent:
            passing_sequence_count += 1
        else:
            failing_sequence_lengths.append(sequence_length)
            threshold_shortfall_percent = max(
                threshold_shortfall_percent,
                minimum_speedup_percent - speedup_percent,
            )

    if missing_sequence_lengths:
        profile_speed_gate = "missing-production-sequence"
    elif passing_sequence_count == len(sequence_lengths):
        profile_speed_gate = "full-profile-speedup-observed"
    else:
        profile_speed_gate = "full-profile-regression-observed"

    all_failing_sequence_lengths = failing_sequence_lengths + missing_sequence_lengths
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "routeFamily",
                "role",
                "variant",
                "productionSequenceLengths",
                "baselineTotalGpuMicroseconds",
                "experimentalTotalGpuMicroseconds",
                "speedupPercents",
                "passingSequenceCount",
                "requiredSequenceCount",
                "minimumSpeedupPercent",
                "thresholdShortfallPercent",
                "failingSequenceLengths",
                "profileSpeedGate",
            ]
        )
        writer.writerow(
            [
                route_family,
                role,
                variant,
                "|".join(str(value) for value in observed_sequence_lengths),
                format_float_list(baseline_values),
                format_float_list(experimental_values),
                format_float_list(speedup_percents),
                str(passing_sequence_count),
                str(len(sequence_lengths)),
                f"{minimum_speedup_percent:.3f}",
                f"{max(0.0, threshold_shortfall_percent):.3f}",
                "|".join(str(value) for value in all_failing_sequence_lengths),
                profile_speed_gate,
            ]
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a Qwen3.5 prefill full-profile speed gate from persisted profile CSVs."
    )
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--experimental-dir", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".test-artifacts/prefill-profile/qwen35-prefill-full-profile-speed-gate.csv"),
    )
    parser.add_argument("--sequence-lengths", default="64,128")
    parser.add_argument("--profile-pattern", default="qwen35-prefill-steps-seq{seq}.csv")
    parser.add_argument("--route-family", default="recurrent_block_row_grid_fan_in")
    parser.add_argument("--role", default="linear_attn.out_proj")
    parser.add_argument("--variant", default="row_grid_fan_in")
    parser.add_argument("--minimum-speedup-percent", type=float, default=5.0)
    args = parser.parse_args()

    sequence_lengths = parse_sequence_lengths(args.sequence_lengths)
    baseline_totals = collect_totals(args.baseline_dir, args.profile_pattern, sequence_lengths)
    experimental_totals = collect_totals(args.experimental_dir, args.profile_pattern, sequence_lengths)
    write_speed_gate(
        output=args.output,
        baseline_totals=baseline_totals,
        experimental_totals=experimental_totals,
        sequence_lengths=sequence_lengths,
        route_family=args.route_family,
        role=args.role,
        variant=args.variant,
        minimum_speedup_percent=args.minimum_speedup_percent,
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
