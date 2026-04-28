# Distributed Delta Lake Merge

## Problem
Currently, `write_deltalake` does not support distributed merge semantics. The previous approach bypassed schema mode checks and was removed.

## Goals
- Add distributed data merge support for Delta Lake tables.
- Keep execution distributed without driver bottlenecks for large data.

## Non-goals
- Schema evolution during merge.

## Design
We will introduce a `merge` write mode that executes distributed joins and upserts.
Transactions will be coordinated on the driver.

