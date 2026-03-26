from __future__ import annotations

import contextlib
import datetime
import decimal
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pytest

import daft
from daft.io.object_store_options import io_config_to_storage_options
from daft.logical.schema import Schema
from tests.conftest import get_tests_daft_runner_name


class _FakeCommitProperties:
    def __init__(self, custom_metadata):
        self.custom_metadata = custom_metadata


@pytest.fixture
def custom_metadata():
    return {"CUSTOM_METADATA": "1"}


@pytest.fixture()
def commit_properties():
    @contextlib.contextmanager
    def _(deltalake):
        setattr(deltalake, "CommitProperties", _FakeCommitProperties)
        try:
            yield
        finally:
            delattr(deltalake, "CommitProperties")

    return _


def test_deltalake_write_basic(tmp_path, base_table):
    deltalake = pytest.importorskip("deltalake")
    path = tmp_path / "some_table"
    df = daft.from_arrow(base_table)
    result = df.write_deltalake(str(path))
    result = result.to_pydict()
    assert result["operation"] == ["ADD"]
    assert result["rows"] == [base_table.num_rows]

    read_delta = deltalake.DeltaTable(str(path))
    expected_schema = Schema.from_pyarrow_schema(pa.schema(read_delta.schema().to_arrow()))
    assert df.schema() == expected_schema
    assert read_delta.to_pyarrow_table() == base_table


def test_deltalake_multi_write_basic(tmp_path, base_table):
    deltalake = pytest.importorskip("deltalake")
    path = tmp_path / "some_table"
    df = daft.from_arrow(base_table)
    df.write_deltalake(str(path))

    result = df.write_deltalake(str(path))
    result = result.to_pydict()
    assert result["operation"] == ["ADD"]
    assert result["rows"] == [base_table.num_rows]

    read_delta = deltalake.DeltaTable(str(path))
    expected_schema = Schema.from_pyarrow_schema(pa.schema(read_delta.schema().to_arrow()))
    assert df.schema() == expected_schema
    assert read_delta.version() == 1
    assert read_delta.to_pyarrow_table() == pa.concat_tables([base_table, base_table])


def test_deltalake_write_cloud(base_table, cloud_paths):
    deltalake = pytest.importorskip("deltalake")
    path, io_config = cloud_paths
    df = daft.from_arrow(base_table)
    result = df.write_deltalake(str(path), io_config=io_config)
    result = result.to_pydict()
    assert result["operation"] == ["ADD"]
    assert result["rows"] == [base_table.num_rows]
    storage_options = io_config_to_storage_options(io_config, path) if io_config is not None else None
    read_delta = deltalake.DeltaTable(str(path), storage_options=storage_options)
    expected_schema = Schema.from_pyarrow_schema(pa.schema(read_delta.schema().to_arrow()))
    assert df.schema() == expected_schema
    assert read_delta.to_pyarrow_table() == base_table


def test_deltalake_write_overwrite_basic(tmp_path):
    deltalake = pytest.importorskip("deltalake")
    path = tmp_path / "some_table"
    df1 = daft.from_pydict({"a": [1, 2]})
    df1.write_deltalake(str(path))

    df2 = daft.from_pydict({"a": [3, 4]})
    result = df2.write_deltalake(str(path), mode="overwrite")
    result = result.to_pydict()
    assert result["operation"] == ["ADD", "DELETE"]
    assert result["rows"] == [2, 2]

    read_delta = deltalake.DeltaTable(str(path))
    expected_schema = Schema.from_pyarrow_schema(pa.schema(read_delta.schema().to_arrow()))
    assert df2.schema() == expected_schema
    assert read_delta.to_pyarrow_table() == df2.to_arrow()


def test_deltalake_write_overwrite_cloud(cloud_paths):
    deltalake = pytest.importorskip("deltalake")
    path, io_config = cloud_paths
    df1 = daft.from_pydict({"a": [1, 2]})
    df1.write_deltalake(str(path), io_config=io_config)

    df2 = daft.from_pydict({"a": [3, 4]})
    result = df2.write_deltalake(str(path), mode="overwrite", io_config=io_config)
    result = result.to_pydict()
    assert result["operation"] == ["ADD", "DELETE"]
    assert result["rows"] == [2, 2]

    storage_options = io_config_to_storage_options(io_config, path) if io_config is not None else None
    read_delta = deltalake.DeltaTable(str(path), storage_options=storage_options)
    expected_schema = Schema.from_pyarrow_schema(pa.schema(read_delta.schema().to_arrow()))
    assert df2.schema() == expected_schema
    assert read_delta.to_pyarrow_table() == df2.to_arrow()


@pytest.mark.skipif(
    get_tests_daft_runner_name() == "native",
    reason="Native executor does not support repartitioning",
)
def test_deltalake_write_overwrite_multi_partition(tmp_path):
    deltalake = pytest.importorskip("deltalake")
    path = tmp_path / "some_table"
    df1 = daft.from_pydict({"a": [1, 2, 3, 4]})
    df1 = df1.repartition(2)
    df1.write_deltalake(str(path))

    df2 = daft.from_pydict({"a": [5, 6, 7, 8]})
    df2 = df2.repartition(2)
    result = df2.write_deltalake(str(path), mode="overwrite")
    result = result.to_pydict()
    assert result["operation"] == ["ADD", "ADD", "DELETE", "DELETE"]

    read_delta = deltalake.DeltaTable(str(path))
    expected_schema = Schema.from_pyarrow_schema(pa.schema(read_delta.schema().to_arrow()))
    assert df2.schema() == expected_schema
    assert read_delta.to_pyarrow_table() == df2.to_arrow()


def test_deltalake_write_overwrite_schema(tmp_path):
    deltalake = pytest.importorskip("deltalake")
    path = tmp_path / "some_table"
    df1 = daft.from_pydict({"a": [1, 2]})
    df1.write_deltalake(str(path))

    df2 = daft.from_pydict({"b": [3, 4]})
    result = df2.write_deltalake(str(path), mode="overwrite", schema_mode="overwrite")
    result = result.to_pydict()
    assert result["operation"] == ["ADD", "DELETE"]

    read_delta = deltalake.DeltaTable(str(path))
    expected_schema = Schema.from_pyarrow_schema(pa.schema(read_delta.schema().to_arrow()))
    assert df2.schema() == expected_schema
    assert read_delta.to_pyarrow_table() == df2.to_arrow()


def test_deltalake_write_overwrite_error_schema(tmp_path):
    path = tmp_path / "some_table"
    df1 = daft.from_pydict({"a": [1, 2]})
    df1.write_deltalake(str(path), mode="overwrite")
    df2 = daft.from_pydict({"b": [3, 4]})
    with pytest.raises(ValueError):
        df2.write_deltalake(str(path), mode="overwrite")


def test_deltalake_write_append_schema_merge_add_column(tmp_path):
    deltalake = pytest.importorskip("deltalake")
    path = tmp_path / "some_table"
    deltalake.write_deltalake(str(path), pa.table({"a": [1, 2]}))

    df2 = daft.from_pydict({"a": [3, 4], "b": ["x", "y"]})
    result = df2.write_deltalake(str(path), schema_mode="merge")
    result = result.to_pydict()
    assert result["operation"] == ["ADD"]
    assert result["rows"] == [2]

    read_delta = deltalake.DeltaTable(str(path))
    assert read_delta.version() == 1
    assert Schema.from_pyarrow_schema(pa.schema(read_delta.schema().to_arrow())) == Schema.from_pyarrow_schema(
        pa.schema([("a", pa.int64()), ("b", pa.string())])
    )
    assert sorted(read_delta.to_pyarrow_table().to_pylist(), key=lambda row: row["a"]) == [
        {"a": 1, "b": None},
        {"a": 2, "b": None},
        {"a": 3, "b": "x"},
        {"a": 4, "b": "y"},
    ]


def test_deltalake_write_append_schema_merge_missing_existing_column(tmp_path):
    deltalake = pytest.importorskip("deltalake")
    path = tmp_path / "some_table"
    deltalake.write_deltalake(str(path), pa.table({"a": [1, 2]}))

    df2 = daft.from_pydict({"b": ["x", "y"]})
    result = df2.write_deltalake(str(path), schema_mode="merge")
    result = result.to_pydict()
    assert result["operation"] == ["ADD"]
    assert result["rows"] == [2]

    read_delta = deltalake.DeltaTable(str(path))
    assert Schema.from_pyarrow_schema(pa.schema(read_delta.schema().to_arrow())) == Schema.from_pyarrow_schema(
        pa.schema([("a", pa.int64()), ("b", pa.string())])
    )
    assert sorted(
        read_delta.to_pyarrow_table().to_pylist(),
        key=lambda row: (row["a"] is None, row["a"] if row["a"] is not None else row["b"]),
    ) == [
        {"a": 1, "b": None},
        {"a": 2, "b": None},
        {"a": None, "b": "x"},
        {"a": None, "b": "y"},
    ]


def test_deltalake_write_append_schema_merge_cast_existing_column_type(tmp_path):
    deltalake = pytest.importorskip("deltalake")
    path = tmp_path / "some_table"
    deltalake.write_deltalake(str(path), pa.table({"a": [1, 2]}))

    df2 = daft.from_pydict({"a": ["3", "4"]})
    result = df2.write_deltalake(str(path), schema_mode="merge")
    result = result.to_pydict()
    assert result["operation"] == ["ADD"]
    assert result["rows"] == [2]

    read_delta = deltalake.DeltaTable(str(path))
    expected_schema = Schema.from_pyarrow_schema(pa.schema(read_delta.schema().to_arrow()))
    assert expected_schema == Schema.from_pyarrow_schema(pa.schema([("a", pa.int64())]))
    assert sorted(read_delta.to_pyarrow_table().to_pylist(), key=lambda row: row["a"]) == [
        {"a": 1},
        {"a": 2},
        {"a": 3},
        {"a": 4},
    ]


def test_deltalake_write_overwrite_schema_merge_preserves_existing_columns(tmp_path):
    deltalake = pytest.importorskip("deltalake")
    path = tmp_path / "some_table"
    deltalake.write_deltalake(str(path), pa.table({"a": [1, 2]}))

    df2 = daft.from_pydict({"b": ["x", "y"]})
    result = df2.write_deltalake(str(path), mode="overwrite", schema_mode="merge")
    result = result.to_pydict()
    assert result["operation"] == ["ADD", "DELETE"]
    assert result["rows"] == [2, 2]

    read_delta = deltalake.DeltaTable(str(path))
    assert read_delta.version() == 1
    assert Schema.from_pyarrow_schema(pa.schema(read_delta.schema().to_arrow())) == Schema.from_pyarrow_schema(
        pa.schema([("a", pa.int64()), ("b", pa.string())])
    )
    assert sorted(read_delta.to_pyarrow_table().to_pylist(), key=lambda row: row["b"]) == [
        {"a": None, "b": "x"},
        {"a": None, "b": "y"},
    ]


def test_deltalake_write_append_schema_merge_add_column_partitioned(tmp_path):
    deltalake = pytest.importorskip("deltalake")
    path = tmp_path / "some_table"

    df1 = daft.from_pydict({"part": [1, 2], "a": [1, 2]})
    deltalake.write_deltalake(str(path), df1.to_arrow(), partition_by=["part"])

    df2 = daft.from_pydict({"part": [1, 2], "a": [3, 4], "b": ["x", "y"]})
    result = df2.write_deltalake(str(path), schema_mode="merge")
    result = result.to_pydict()
    assert result["operation"] == ["ADD", "ADD"]
    assert sorted(result["rows"]) == [1, 1]

    expected = daft.from_pydict(
        {
            "part": [1, 2, 1, 2],
            "a": [1, 2, 3, 4],
            "b": [None, None, "x", "y"],
        }
    )
    read_delta = deltalake.DeltaTable(str(path))
    assert read_delta.version() == 1
    assert Schema.from_pyarrow_schema(pa.schema(read_delta.schema().to_arrow())) == expected.schema()
    check_equal_both_daft_and_delta_rs(expected, path, [("part", "ascending"), ("a", "ascending")])


@pytest.mark.skipif(
    get_tests_daft_runner_name() == "native",
    reason="Native executor does not support repartitioning",
)
def test_deltalake_write_append_schema_merge_distributed_partitioned_cast_missing_and_additive(tmp_path):
    deltalake = pytest.importorskip("deltalake")
    path = tmp_path / "some_table"

    base = pa.table(
        {
            "part": pa.array([0, 1], type=pa.int64()),
            "a": pa.array([1, 2], type=pa.int64()),
            "existing_only": pa.array(["left", "right"], type=pa.string()),
        }
    )
    deltalake.write_deltalake(str(path), base, partition_by=["part"])

    df = (
        daft.range(64, partitions=16)
        .with_columns(
            {
                "part": daft.col("id") % 8,
                "a": (daft.col("id") + 10).cast(daft.DataType.string()),
                "new_value": daft.col("id") * 2,
            }
        )
        .select("part", "a", "new_value")
        .into_partitions(32)
    )

    result = df.write_deltalake(str(path), schema_mode="merge").to_pydict()
    assert set(result["operation"]) == {"ADD"}
    assert len(result["operation"]) > 1
    assert sum(result["rows"]) == 64

    expected_base = (
        daft.from_arrow(base)
        .with_columns({"new_value": daft.lit(None).cast(daft.DataType.int64())})
        .select("part", "a", "existing_only", "new_value")
    )
    expected_append = df.with_columns(
        {
            "a": daft.col("a").cast(daft.DataType.int64()),
            "existing_only": daft.lit(None).cast(daft.DataType.string()),
        }
    ).select("part", "a", "existing_only", "new_value")
    expected = expected_base.concat(expected_append)

    read_delta = deltalake.DeltaTable(str(path))
    assert read_delta.version() == 1
    assert Schema.from_pyarrow_schema(pa.schema(read_delta.schema().to_arrow())) == expected.schema()
    check_equal_both_daft_and_delta_rs(expected, path, [("part", "ascending"), ("a", "ascending")])


def test_deltalake_write_append_schema_merge_falls_back_when_active_actions_are_unsupported(tmp_path, monkeypatch):
    deltalake = pytest.importorskip("deltalake")
    path = tmp_path / "some_table"
    deltalake.write_deltalake(str(path), pa.table({"a": [1, 2]}))

    original_get_add_actions = deltalake.DeltaTable.get_add_actions

    def patched_get_add_actions(self, flatten: bool = False):
        record_batch = pa.record_batch(original_get_add_actions(self, flatten=flatten))
        extra = pa.array(["present"] * record_batch.num_rows, type=pa.string())
        return pa.RecordBatch.from_arrays(
            [*record_batch.columns, extra],
            names=[*record_batch.schema.names, "tags"],
        )

    monkeypatch.setattr(deltalake.DeltaTable, "get_add_actions", patched_get_add_actions)

    df2 = daft.from_pydict({"a": [3, 4], "b": ["x", "y"]})
    result = df2.write_deltalake(str(path), schema_mode="merge").to_pydict()
    assert result["operation"] == ["ADD"]
    assert result["rows"] == [2]

    read_delta = deltalake.DeltaTable(str(path))
    assert read_delta.version() == 1
    assert Schema.from_pyarrow_schema(pa.schema(read_delta.schema().to_arrow())) == Schema.from_pyarrow_schema(
        pa.schema([("a", pa.int64()), ("b", pa.string())])
    )
    assert sorted(read_delta.to_pyarrow_table().to_pylist(), key=lambda row: row["a"]) == [
        {"a": 1, "b": None},
        {"a": 2, "b": None},
        {"a": 3, "b": "x"},
        {"a": 4, "b": "y"},
    ]
    assert read_delta.history(1)[0]["operationParameters"]["mode"] == "Append"


def test_deltalake_write_schema_merge_raises_on_invalid_cast(tmp_path):
    deltalake = pytest.importorskip("deltalake")
    path = tmp_path / "some_table"
    deltalake.write_deltalake(str(path), pa.table({"a": [1, 2]}))

    df = daft.from_pydict({"a": ["x", "y"]})
    with pytest.raises(Exception, match="Cannot cast string"):
        df.write_deltalake(str(path), schema_mode="merge")


def test_deltalake_write_error(tmp_path, base_table):
    path = tmp_path / "some_table"
    df = daft.from_arrow(base_table)
    df.write_deltalake(str(path), mode="error")
    with pytest.raises(AssertionError):
        df.write_deltalake(str(path), mode="error")


def test_deltalake_write_ignore(tmp_path):
    deltalake = pytest.importorskip("deltalake")
    path = tmp_path / "some_table"
    df1 = daft.from_pydict({"a": [1, 2]})
    df1.write_deltalake(str(path), mode="ignore")
    df2 = daft.from_pydict({"a": [3, 4]})
    result = df2.write_deltalake(str(path), mode="ignore")
    result = result.to_arrow()
    assert result.num_rows == 0

    read_delta = deltalake.DeltaTable(str(path))
    expected_schema = Schema.from_pyarrow_schema(pa.schema(read_delta.schema().to_arrow()))
    assert df1.schema() == expected_schema
    assert read_delta.to_pyarrow_table() == df1.to_arrow()


@pytest.mark.skipif(
    get_tests_daft_runner_name() == "native",
    reason="Native executor does not support repartitioning",
)
def test_deltalake_write_with_empty_partition(tmp_path, base_table):
    deltalake = pytest.importorskip("deltalake")
    path = tmp_path / "some_table"
    df = daft.from_arrow(base_table).into_partitions(4)
    result = df.write_deltalake(str(path))
    result = result.to_pydict()
    assert result["operation"] == ["ADD", "ADD", "ADD"]
    assert result["rows"] == [1, 1, 1]

    read_delta = deltalake.DeltaTable(str(path))
    expected_schema = Schema.from_pyarrow_schema(pa.schema(read_delta.schema().to_arrow()))
    assert df.schema() == expected_schema
    assert read_delta.to_pyarrow_table() == base_table


def check_equal_both_daft_and_delta_rs(df: daft.DataFrame, path: Path, sort_order: list[tuple[str, str]]):
    deltalake = pytest.importorskip("deltalake")

    arrow_df = df.to_arrow().sort_by(sort_order)

    read_daft = daft.read_deltalake(str(path))
    assert read_daft.schema() == df.schema()
    assert read_daft.to_arrow().sort_by(sort_order) == arrow_df

    read_delta = deltalake.DeltaTable(str(path))
    expected_schema = Schema.from_pyarrow_schema(pa.schema(read_delta.schema().to_arrow()))
    assert df.schema() == expected_schema
    assert read_delta.to_pyarrow_table().cast(expected_schema.to_pyarrow_schema()).sort_by(sort_order) == arrow_df


@pytest.mark.parametrize(
    "partition_cols,num_partitions",
    [
        (["int"], 3),
        (["float"], 3),
        (["str"], 3),
        pytest.param(["bin"], 3, marks=pytest.mark.xfail(reason="Binary partitioning is not yet supported")),
        (["bool"], 3),
        (["datetime"], 3),
        (["date"], 3),
        (["decimal"], 3),
        (["int", "float"], 4),
    ],
)
def test_deltalake_write_partitioned(tmp_path, partition_cols, num_partitions):
    path = tmp_path / "some_table"
    df = daft.from_pydict(
        {
            "int": [1, 1, 2, None],
            "float": [1.1, 2.2, 2.2, None],
            "str": ["foo", "foo", "bar", None],
            "bin": [b"foo", b"foo", b"bar", None],
            "bool": [True, True, False, None],
            "datetime": [
                datetime.datetime(2024, 2, 10),
                datetime.datetime(2024, 2, 10),
                datetime.datetime(2024, 2, 11),
                None,
            ],
            "date": [datetime.date(2024, 2, 10), datetime.date(2024, 2, 10), datetime.date(2024, 2, 11), None],
            "decimal": pa.array(
                [decimal.Decimal("1111.111"), decimal.Decimal("1111.111"), decimal.Decimal("2222.222"), None],
                type=pa.decimal128(7, 3),
            ),
        }
    )
    result = df.write_deltalake(str(path), partition_cols=partition_cols)
    result = result.to_pydict()
    assert len(result["operation"]) == num_partitions
    assert all(op == "ADD" for op in result["operation"])
    assert sum(result["rows"]) == len(df)

    sort_order = [("int", "ascending"), ("float", "ascending")]
    check_equal_both_daft_and_delta_rs(df, path, sort_order)


def test_deltalake_write_partitioned_empty(tmp_path):
    path = tmp_path / "some_table"

    df = daft.from_arrow(pa.schema([("int", pa.int64()), ("string", pa.string())]).empty_table())

    df.write_deltalake(str(path), partition_cols=["int"])

    check_equal_both_daft_and_delta_rs(df, path, [("int", "ascending")])


def test_deltalake_write_partitioned_some_empty(tmp_path):
    path = tmp_path / "some_table"

    df = daft.from_pydict({"int": [1, 2, 3, None], "string": ["foo", "foo", "bar", None]}).into_partitions(5)

    df.write_deltalake(str(path), partition_cols=["int"])

    check_equal_both_daft_and_delta_rs(df, path, [("int", "ascending")])


def test_deltalake_write_partitioned_existing_table(tmp_path):
    path = tmp_path / "some_table"

    df1 = daft.from_pydict({"int": [1], "string": ["foo"]})
    result = df1.write_deltalake(str(path), partition_cols=["int"])
    result = result.to_pydict()
    assert result["operation"] == ["ADD"]
    assert result["rows"] == [1]

    df2 = daft.from_pydict({"int": [1, 2], "string": ["bar", "bar"]})
    with pytest.raises(ValueError):
        df2.write_deltalake(str(path), partition_cols=["string"])

    result = df2.write_deltalake(str(path))
    result = result.to_pydict()
    assert result["operation"] == ["ADD", "ADD"]
    assert result["rows"] == [1, 1]

    check_equal_both_daft_and_delta_rs(df1.concat(df2), path, [("int", "ascending"), ("string", "ascending")])


def test_deltalake_write_roundtrip(tmp_path):
    path = tmp_path / "some_table"
    df = daft.from_pydict({"a": [1, 2, 3, 4]})
    df.write_deltalake(str(path))

    read_df = daft.read_deltalake(str(path))
    assert df.schema() == read_df.schema()
    assert df.to_arrow() == read_df.to_arrow()


def test_custom_metadata_added_for_new_table(tmp_path, custom_metadata):
    # import deltalake
    deltalake = pytest.importorskip("deltalake")

    path = tmp_path / "some_table"
    df = daft.from_pydict({"a": [1, 2, 3, 4]})
    df.write_deltalake(str(path), custom_metadata=custom_metadata)

    table = deltalake.DeltaTable(path)
    history = table.history(1)

    assert custom_metadata.items() <= history[0].items()


def test_custom_metadata_updated_for_existing_table(tmp_path, custom_metadata):
    """Tests for deltalake version installed in the current environment (currently 0.19.2)."""
    # import deltalake
    deltalake = pytest.importorskip("deltalake")

    path = tmp_path / "some_table"
    df = daft.from_pydict({"a": [1, 2, 3, 4]})
    df.write_deltalake(str(path))

    df = daft.from_pydict({"a": [5, 6]})
    df.write_deltalake(str(path), custom_metadata=custom_metadata, mode="append")

    table = deltalake.DeltaTable(path)
    history = table.history(1)

    assert custom_metadata.items() <= history[0].items()


def test_custom_metadata_updated_for_existing_table_with_commit_properties(
    tmp_path, custom_metadata, commit_properties
):
    deltalake = pytest.importorskip("deltalake")
    from deltalake._internal import RawDeltaTable

    # write once to get into the table is not None path
    path = tmp_path / "some_table"
    df = daft.from_pydict({"a": [1, 2, 3, 4]})
    df.write_deltalake(str(path))

    # Add mocked CommitProperties class introduced in 0.20.0
    with commit_properties(deltalake), patch.object(RawDeltaTable, "create_write_transaction") as mock_method:
        df = daft.from_pydict({"a": [5, 6]})
        df.write_deltalake(str(path), custom_metadata=custom_metadata, mode="append")

        mock_method.assert_called_once()
        (_, _, _, _, _, custom_metadata_arg) = mock_method.call_args[0]

        assert isinstance(custom_metadata_arg, _FakeCommitProperties)
        assert custom_metadata_arg.custom_metadata == custom_metadata
