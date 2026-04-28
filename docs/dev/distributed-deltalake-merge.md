# Distributed Delta Lake Merge in Daft

## Overview

Delta Lake MERGE allows users to update, delete, and insert data into an existing Delta table based on a source DataFrame. This document outlines the architecture for supporting truly distributed Delta MERGE operations in Daft, ensuring that source rows are processed in a distributed manner without being collected on the driver.

## Goals
- Support `DataFrame.write_deltalake(..., mode="merge", merge_condition=..., when_matched_update=..., ...)` with standard Delta MERGE semantics.
- Process source rows in a distributed fashion (e.g., via Ray workers or native distributed runners).
- Prevent Out-Of-Memory (OOM) issues on the driver by only collecting compact Delta transaction metadata (`AddAction` and `RemoveAction`).
- Preserve backwards compatibility for `append`, `overwrite`, `error`, and `ignore` write modes.

## Non-Goals
- Full schema evolution during MERGE (to be handled in a separate design).
- Handling extremely large transaction metadata payloads (Delta transaction metadata must fit in driver memory, which is standard).

## Architecture: Distributed MERGE Execution

Since `deltalake` (delta-rs) 1.5 exposes `DeltaTable.merge(source, predicate, streamed_exec=True)`, passing a local ArrowStreamExportable works well for single-node workloads. However, for a distributed execution, we cannot collect the distributed Daft DataFrame into a single driver-side Arrow stream.

### Proposed Strategy: The Anti-Join and Distributed Rewrite Method

A true distributed MERGE can be modeled as a distributed join and rewrite operation:

1. **File Skipping & Target Scanning:** Evaluate the `merge_condition` using statistics from the target Delta table's transaction log to prune target Parquet files that definitely do not overlap with the source DataFrame.
2. **Target File Read:** Read the remaining target Parquet files as a distributed Daft DataFrame.
3. **Distributed Join & Evaluate:** Perform a distributed full outer join (or similar, depending on MERGE clauses) between the target DataFrame and the source DataFrame on the `merge_condition`.
4. **Apply Actions:** For each joined row, evaluate the `when_matched_update`, `when_matched_delete`, and `when_not_matched_insert` conditions and expressions to compute the new row values (or filter out deleted rows).
5. **Distributed Write:** Write the resulting rows to new Parquet files in a distributed manner, generating `AddAction` objects.
6. **Transaction Commit:** Send the `AddAction` objects (for new files) and `RemoveAction` objects (for the old target files that were rewritten) to the driver. The driver then atomically commits these actions to the Delta log using `deltalake`'s low-level commit APIs.

### Alternative Strategy: Staging Table (Evaluated)

An alternative is to write the source DataFrame to a temporary staging Delta table using the existing distributed write path, and then invoke a target-side merge reading from the staging table. However, delta-rs typically still needs to read the source table into memory or a stream to perform the join. Unless delta-rs pushes the join into a distributed query engine (which it does not by default), this still bottlenecks on the machine executing the merge. Therefore, the Distributed Join & Rewrite method is superior.

## API Changes

`DataFrame.write_deltalake` will be extended:
```python
def write_deltalake(
    self,
    target: str | Path | deltalake.DeltaTable,
    mode: Literal["append", "overwrite", "error", "ignore", "merge"] = "append",
    *,
    merge_condition: Optional[Expression] = None,
    when_matched_update: Optional[Dict[str, Expression]] = None,
    when_matched_delete: Optional[Expression] = None,
    when_not_matched_insert: Optional[Dict[str, Expression]] = None,
    # ... existing parameters ...
): ...
```

## Validation and Concurrency
- **Concurrency:** Delta Lake's optimistic concurrency control applies. If concurrent transactions modify the same target files, the transaction must be retried or failed.
- **Distributed Guarantee:** We must include assertions in our test suite (e.g., using Ray worker mocking or analyzing memory/network usage) to ensure the driver node never loads the full source DataFrame.

## Limitations and Future Work
- Row-level tracking: Naively rewriting entire target files may cause write amplification. We will explore advanced pruning or Deletion Vectors in the future.
- Complex MERGE predicates might not map cleanly to equi-joins, requiring broadcast nested loop joins which are expensive.
