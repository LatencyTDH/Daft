from __future__ import annotations

import argparse
import shutil
import statistics
import tempfile
import time
from pathlib import Path

import deltalake
import pyarrow as pa

import daft
from daft import col

DEFAULT_ROWS = 5_000_000
DEFAULT_PARTITIONS = 2048
DEFAULT_WRITE_PARTITIONS = 64
DEFAULT_DELTA_PARTITIONS = 512
DEFAULT_REPEATS = 3


def build_input_df(rows: int, partitions: int, write_partitions: int, delta_partitions: int) -> daft.DataFrame:
    return (
        daft.range(rows, partitions=partitions)
        .with_columns(
            {
                "part": col("id") % delta_partitions,
                "grp": col("id") % 97,
                "value2": (col("id") * 3) % 100_000,
                "new_value": (col("id") * 7) % 1_000_000,
                "new_value2": (col("id") * 11) % 1_000_000,
            }
        )
        .select("part", "id", "grp", "value2", "new_value", "new_value2")
        .repartition(write_partitions, "part")
    )


def write_base_table(path: str) -> None:
    deltalake.write_deltalake(
        path,
        pa.table(
            {
                "part": pa.array([0, 1, 2], type=pa.int64()),
                "id": pa.array([1, 2, 3], type=pa.int64()),
                "grp": pa.array([1, 2, 3], type=pa.int64()),
                "value2": pa.array([10, 20, 30], type=pa.int64()),
            }
        ),
        partition_by=["part"],
    )


# Exact baseline shape from commit 4d10f3966 (the first implementation):
# 1) build a RecordBatchReader directly from self.to_arrow_iter(results_buffer_size=None)
# 2) hand that reader to deltalake.writer.write_deltalake(..., schema_mode="merge")
def write_schema_merge_baseline(path: str, df: daft.DataFrame) -> None:
    schema = pa.schema((field.name, field.dtype.to_arrow_dtype()) for field in df.schema())
    reader = pa.RecordBatchReader.from_batches(schema, df.to_arrow_iter(results_buffer_size=None))
    deltalake.writer.write_deltalake(
        path,
        reader,
        mode="append",
        schema_mode="merge",
        partition_by=["part"],
    )


# Optimized path under test: the current Daft implementation.
def write_schema_merge_optimized(path: str, df: daft.DataFrame) -> None:
    df.write_deltalake(path, schema_mode="merge")


def verify_table(path: str, expected_rows: int) -> tuple[int, str, int]:
    table = deltalake.DeltaTable(path)
    schema_repr = str(table.schema().to_arrow())
    expected_fields = {
        "part": pa.int64(),
        "id": pa.int64(),
        "grp": pa.int64(),
        "value2": pa.int64(),
        "new_value": pa.int64(),
        "new_value2": pa.int64(),
    }
    actual_schema = pa.schema(table.schema().to_arrow())
    actual_fields = {field.name: field.type for field in actual_schema}
    if actual_fields != expected_fields:
        raise AssertionError(f"unexpected schema: {actual_schema}")

    row_count = table.to_pyarrow_table(columns=["id"]).num_rows
    if row_count != expected_rows:
        raise AssertionError(f"unexpected row count: {row_count} != {expected_rows}")

    version = table.version()
    if version != 1:
        raise AssertionError(f"schema merge should advance the table by one version, got {version}")

    active_files = len(table.file_uris())
    return version, schema_repr, active_files


def run_once(
    kind: str,
    rows: int,
    partitions: int,
    write_partitions: int,
    delta_partitions: int,
    keep_temp: bool,
) -> tuple[float, int, str, int, str]:
    temp_dir = Path(tempfile.mkdtemp(prefix=f"daft-delta-merge-{kind}-"))
    path = str(temp_dir / "tbl")
    try:
        write_base_table(path)
        df = build_input_df(rows, partitions, write_partitions, delta_partitions)
        writer = {
            "baseline": write_schema_merge_baseline,
            "optimized": write_schema_merge_optimized,
        }[kind]
        start = time.perf_counter()
        writer(path, df)
        elapsed = time.perf_counter() - start
        version, schema_repr, active_files = verify_table(path, expected_rows=rows + 3)
        return elapsed, version, schema_repr, active_files, path
    finally:
        if not keep_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)


def format_seconds(values: list[float]) -> str:
    return ", ".join(f"{value:.3f}s" for value in values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Delta Lake schema_mode='merge' write paths.")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--partitions", type=int, default=DEFAULT_PARTITIONS)
    parser.add_argument("--write-partitions", type=int, default=DEFAULT_WRITE_PARTITIONS)
    parser.add_argument("--delta-partitions", type=int, default=DEFAULT_DELTA_PARTITIONS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()

    results: dict[str, list[float]] = {"baseline": [], "optimized": []}
    versions: dict[str, list[int]] = {"baseline": [], "optimized": []}
    active_files: dict[str, list[int]] = {"baseline": [], "optimized": []}
    schema_repr: str | None = None
    temp_paths: dict[str, list[str]] = {"baseline": [], "optimized": []}

    print("Benchmark configuration")
    print(f"  rows={args.rows}")
    print(f"  input_partitions={args.partitions}")
    print(f"  write_partitions={args.write_partitions}")
    print(f"  delta_partitions={args.delta_partitions}")
    print(f"  repeats={args.repeats}")
    print("  merge_case=append to a partitioned Delta table with additive schema evolution")
    print(
        "  execution_shape=high-partition upstream input, then hash repartition on the Delta partition key "
        "before the sink; optimized path stays on Daft's native/distributed writer"
    )
    print()

    for kind in ("baseline", "optimized"):
        for run_idx in range(args.repeats):
            elapsed, version, current_schema_repr, current_active_files, path = run_once(
                kind,
                rows=args.rows,
                partitions=args.partitions,
                write_partitions=args.write_partitions,
                delta_partitions=args.delta_partitions,
                keep_temp=args.keep_temp,
            )
            results[kind].append(elapsed)
            versions[kind].append(version)
            active_files[kind].append(current_active_files)
            schema_repr = current_schema_repr
            temp_paths[kind].append(path)
            print(
                f"{kind:>9} run {run_idx + 1}: {elapsed:.3f}s "
                f"(table_version={version}, active_files={current_active_files})"
            )
        print()

    baseline_mean = statistics.mean(results["baseline"])
    optimized_mean = statistics.mean(results["optimized"])
    speedup = baseline_mean / optimized_mean
    improvement_pct = (baseline_mean - optimized_mean) / baseline_mean * 100

    print("Summary")
    print(f"  baseline_times = [{format_seconds(results['baseline'])}]")
    print(f"  optimized_times = [{format_seconds(results['optimized'])}]")
    print(f"  baseline_mean = {baseline_mean:.3f}s")
    print(f"  optimized_mean = {optimized_mean:.3f}s")
    print(f"  speedup = {speedup:.3f}x")
    print(f"  improvement = {improvement_pct:.1f}%")
    print(f"  baseline_versions = {versions['baseline']}")
    print(f"  optimized_versions = {versions['optimized']}")
    print(f"  baseline_active_files = {active_files['baseline']}")
    print(f"  optimized_active_files = {active_files['optimized']}")
    print("  final_schema =")
    for line in (schema_repr or "").splitlines():
        print(f"    {line}")

    if args.keep_temp:
        print("  temp_paths =")
        for kind in ("baseline", "optimized"):
            for path in temp_paths[kind]:
                print(f"    {kind}: {path}")


if __name__ == "__main__":
    main()
