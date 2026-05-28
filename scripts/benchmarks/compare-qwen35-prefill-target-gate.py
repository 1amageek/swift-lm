#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def parse_sequence_lengths(value: str) -> list[int]:
    lengths: list[int] = []
    for part in value.split(","):
        stripped = part.strip()
        if stripped:
            lengths.append(int(stripped))
    if not lengths:
        raise ValueError("at least one sequence length is required")
    return lengths


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
        path = directory / pattern.format(seq=sequence_length)
        if path.exists():
            totals[sequence_length] = profile_total_gpu_microseconds(path)
    return totals


def format_float_list(values: list[float]) -> str:
    return "|".join(f"{value:.3f}" for value in values)


def write_target_gate(
    output: Path,
    baseline_totals: dict[int, float],
    observed_totals: dict[int, float],
    sequence_lengths: list[int],
    target_name: str,
    target_reduction_percent: float,
) -> None:
    observed_sequence_lengths: list[int] = []
    baseline_values: list[float] = []
    observed_values: list[float] = []
    target_values: list[float] = []
    reduction_percents: list[float] = []
    failing_sequence_lengths: list[int] = []
    missing_sequence_lengths: list[int] = []
    threshold_shortfall_percent = 0.0
    passing_sequence_count = 0

    for sequence_length in sequence_lengths:
        baseline = baseline_totals.get(sequence_length)
        observed = observed_totals.get(sequence_length)
        if baseline is None or observed is None:
            missing_sequence_lengths.append(sequence_length)
            continue

        target = baseline * (1.0 - target_reduction_percent / 100.0)
        reduction_percent = ((baseline - observed) / baseline * 100.0) if baseline > 0 else 0.0
        observed_sequence_lengths.append(sequence_length)
        baseline_values.append(baseline)
        observed_values.append(observed)
        target_values.append(target)
        reduction_percents.append(reduction_percent)
        if observed <= target:
            passing_sequence_count += 1
        else:
            failing_sequence_lengths.append(sequence_length)
            threshold_shortfall_percent = max(
                threshold_shortfall_percent,
                target_reduction_percent - reduction_percent,
            )

    if missing_sequence_lengths:
        profile_target_gate = "missing-production-sequence"
    elif passing_sequence_count == len(sequence_lengths):
        profile_target_gate = "full-profile-target-observed"
    else:
        profile_target_gate = "full-profile-target-missed"

    all_failing_sequence_lengths = failing_sequence_lengths + missing_sequence_lengths
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "targetName",
                "productionSequenceLengths",
                "baselineTotalGpuMicroseconds",
                "observedTotalGpuMicroseconds",
                "targetTotalGpuMicroseconds",
                "reductionPercents",
                "passingSequenceCount",
                "requiredSequenceCount",
                "targetReductionPercent",
                "thresholdShortfallPercent",
                "failingSequenceLengths",
                "profileTargetGate",
            ]
        )
        writer.writerow(
            [
                target_name,
                "|".join(str(value) for value in observed_sequence_lengths),
                format_float_list(baseline_values),
                format_float_list(observed_values),
                format_float_list(target_values),
                format_float_list(reduction_percents),
                str(passing_sequence_count),
                str(len(sequence_lengths)),
                f"{target_reduction_percent:.3f}",
                f"{max(0.0, threshold_shortfall_percent):.3f}",
                "|".join(str(value) for value in all_failing_sequence_lengths),
                profile_target_gate,
            ]
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a Qwen3.5 prefill full-profile absolute target gate from persisted profile CSVs."
    )
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--observed-dir", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".test-artifacts/prefill-profile/qwen35-prefill-50pct-target-gate.csv"),
    )
    parser.add_argument("--sequence-lengths", default="64,128")
    parser.add_argument("--profile-pattern", default="qwen35-prefill-steps-seq{seq}.csv")
    parser.add_argument("--target-name", default="qwen35-prefill-total-50pct")
    parser.add_argument("--target-reduction-percent", type=float, default=50.0)
    args = parser.parse_args()

    sequence_lengths = parse_sequence_lengths(args.sequence_lengths)
    baseline_totals = collect_totals(args.baseline_dir, args.profile_pattern, sequence_lengths)
    observed_totals = collect_totals(args.observed_dir, args.profile_pattern, sequence_lengths)
    write_target_gate(
        output=args.output,
        baseline_totals=baseline_totals,
        observed_totals=observed_totals,
        sequence_lengths=sequence_lengths,
        target_name=args.target_name,
        target_reduction_percent=args.target_reduction_percent,
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
