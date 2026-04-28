# Distributed Delta Lake Merge in Daft

## Summary

Daft now supports a first distributed Delta Lake MERGE path through
`DataFrame.write_deltalake(..., mode="merge", ...)`. The implementation keeps
Daft source rows in the normal write pipeline, stages them as a temporary Delta
table, and then gives delta-rs a streaming Arrow reader over that staged table.
The driver coordinates the target transaction, but it does not materialize the
source DataFrame with `to_arrow()` or `to_pydict()` before merge execution.

This is intentionally a staged-source implementation rather than a fake
`schema_mode="merge"` relaxation. `schema_mode` remains schema evolution
terminology; `mode="merge"` is the Delta MERGE/upsert operation.

## Public API

`DataFrame.write_deltalake` accepts `mode="merge"` plus explicit merge clauses:

```python
df.write_deltalake(
    target_path,
    mode="merge",
    merge_predicate="target.id = source.id",
    merge_when_matched_update_all=True,
    merge_when_not_matched_insert_all=True,
)
```

Supported clauses in this phase:

- `merge_when_matched_update_all=True`
- `merge_when_matched_updates={"col": "source.expr"}`
- `merge_when_matched_delete=<optional predicate>`
- `merge_when_not_matched_insert_all=True`
- `merge_when_not_matched_insert={"col": "source.expr"}`
- `merge_source_alias` and `merge_target_alias` for SQL predicate/expression aliases

The API rejects ambiguous combinations such as update-all plus explicit update
assignments. It also rejects `schema_mode="merge"` with a message pointing users
to `mode="merge"`, because schema merging and Delta MERGE are different
operations.

## Execution Architecture

### 1. Validate and bind the target table

The target table must already exist. Daft validates that the source schema
matches the target schema unless the existing overwrite schema rule applies.
Merge does not create tables.

### 2. Distributed source staging

The source DataFrame is written to a temporary sibling Delta table via Daft's
existing `write_deltalake` execution path:

```text
source Daft plan -> distributed sink tasks -> staged Delta table AddActions ->
staged Delta commit
```

This reuses the production writer machinery that already writes source
micropartitions on workers and returns only compact Delta add-action metadata to
the coordinator. Source rows are not collected on the Python driver.

### 3. Streaming merge input

After staging commits, Daft opens the staged Delta table and builds a PyArrow
`RecordBatchReader` from `DeltaTable.to_pyarrow_dataset().scanner().to_reader()`.
That reader exposes Arrow's C stream interface, which delta-rs 1.5 accepts in
`DeltaTable.merge(..., streamed_exec=True)`. This avoids constructing a single
driver-side Arrow table for the merge source.

### 4. Target transaction

Delta-rs performs the MERGE against the target table, including file scanning,
rewrites, optimistic concurrency control, and Delta log commit. Daft passes
custom commit metadata through the delta-rs `CommitProperties` path.

### 5. Cleanup

For local filesystems, Daft removes the temporary staging table after the target
merge succeeds or fails. Remote-object-store cleanup is intentionally left for a
follow-up because safe recursive deletion needs object-store-specific handling
and should not be implemented with broad best-effort filesystem assumptions.

## Distributed Guarantees

- Source execution and staging are distributed using Daft's existing sink
  pipeline.
- The driver collects staged add-action metadata and merge metrics only.
- The merge source is provided to delta-rs as a `RecordBatchReader` stream, not
  as a driver-collected `pyarrow.Table`.
- Target-side merge execution is currently performed by delta-rs in the process
  coordinating the transaction. That is acceptable for this phase because the
  source is streamed and spill-aware (`streamed_exec=True`), but it is not the
  final lowest-write-amplification architecture.

## Limitations and Follow-ups

This phase deliberately chooses a safe staged-source merge over directly
rewriting target files in Daft. Remaining work for a fully native Daft merge
engine:

1. Represent merge clauses in the logical/physical plan instead of using a
   blocking Python wrapper.
2. Use Delta log statistics plus source-domain statistics to prune target files
   before scan.
3. Execute target/source joins in Daft workers and generate `RemoveAction` plus
   `AddAction` records directly.
4. Add remote staging cleanup through Daft object-store abstractions.
5. Support schema evolution during merge.

## Concurrency Semantics

The target commit is delegated to delta-rs, so Delta Lake optimistic concurrency
control applies. If a concurrent writer modifies files that the merge read or
rewrote, delta-rs raises the appropriate transaction conflict rather than Daft
silently corrupting the table.

## Test Plan

Coverage should include:

- update+insert correctness for `update_all` and `insert_all`;
- explicit update/insert assignment clauses;
- argument validation for missing predicates and ambiguous clauses;
- local staging cleanup;
- distributed-runner coverage showing repartitioned sources flow through the
  staging write path before delta-rs merge execution.
