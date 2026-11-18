import os
from datetime import datetime

import pyarrow as pa
import pytest

from triad.utils.arrow_dataset import (
    ArrowIPCReader,
    LocalArrowIPCDataset,
    _decode_scalar,
    _encode_scalar,
)
from triad.utils.batch_reslicers import ArrowTableBatchReslicer


def _schema():
    return pa.schema(
        [
            ("p", pa.int64()),
            ("s", pa.string()),
            ("d", pa.timestamp("us")),
            ("b", pa.binary()),
            ("lst", pa.list_(pa.int64())),
        ]
    )


def _table(p_values, schema=None, seed=0):
    schema = schema or _schema()
    n = len(p_values)
    return pa.table(
        {
            "p": pa.array(p_values, schema.field("p").type),
            "s": [f"s{seed + i}" for i in range(n)],
            "d": pa.array(
                [None if j % 2 else datetime(2020, 1, 1 + j) for j in range(n)],
                pa.timestamp("us"),
            ),
            "b": [bytes([j % 256]) for j in range(n)],
            "lst": [[j, j + 1] for j in range(n)],
        },
        schema=schema,
    )


def _batches(p_values, schema=None, seed=0):
    return _table(p_values, schema, seed).to_batches()


def test_scalar_key_codec_roundtrip():
    for value in [
        None,
        0,
        -123,
        "plain",
        "a/b/c",
        "中国",
        b"\x00\x01\xff",
        True,
        3.25,
    ]:
        encoded = _encode_scalar(value)
        assert "/" not in encoded and os.sep not in encoded
        assert _decode_scalar(encoded) == value


def test_publish_and_read_basic(tmp_path):
    ds = LocalArrowIPCDataset(str(tmp_path / "ds"), partition_key="p")
    assert not ds.exists
    assert ds.partition_key == "p"

    version = ds.publish(_batches([1, 1, 2]), partitions=[1, 2], schema=_schema())
    assert isinstance(version, str) and len(version) == 32
    assert ds.exists

    reader = ds.open_reader()
    assert isinstance(reader, ArrowIPCReader)
    assert reader.version == version
    assert reader.partition_key == "p"
    assert reader.partition_values == [1, 2]
    assert reader.schema.equals(_schema())

    table = reader.read_table()
    expected = _table([1, 1, 2]).to_pydict()
    assert table.to_pydict() == expected
    # native record batches, consumable directly
    batches = list(reader.read_batches())
    assert all(isinstance(b, pa.RecordBatch) for b in batches)
    assert pa.Table.from_batches(batches).to_pydict() == expected
    reader.close()
    assert reader.closed


def test_rerun_replaces_only_declared_partitions(tmp_path):
    ds = LocalArrowIPCDataset(str(tmp_path / "ds"), partition_key="p")
    ds.publish(_batches([1, 2, 3], seed=0), partitions=[1, 2, 3], schema=_schema())

    # rerun only partition 2; 1 and 3 stay byte-identical
    ds.publish(_batches([2, 2], seed=100), partitions=[2], schema=_schema())
    reader = ds.open_reader()
    assert sorted(reader.partition_values) == [1, 2, 3]
    table = reader.read_table()
    by_part = {
        v: table.filter(pa.compute.equal(table.column("p"), v)).to_pydict()
        for v in [1, 2, 3]
    }
    assert by_part[1]["s"] == ["s0"]
    assert by_part[2]["s"] == ["s100", "s101"]
    assert by_part[3]["s"] == ["s2"]


def test_zero_row_rerun_deletes_partition(tmp_path):
    ds = LocalArrowIPCDataset(str(tmp_path / "ds"), partition_key="p")
    ds.publish(_batches([1, 2]), partitions=[1, 2], schema=_schema())

    ds.publish([], partitions=[1], schema=_schema())
    reader = ds.open_reader()
    assert reader.partition_values == [2]
    assert reader.read_table().column("p").to_pylist() == [2]

    # deleting every partition leaves an empty but schema-bearing dataset
    ds.publish([], partitions=[2], schema=_schema())
    reader = ds.open_reader()
    assert reader.partition_values == []
    table = reader.read_table()
    assert table.num_rows == 0
    assert table.schema.equals(_schema())


def test_null_partition_value(tmp_path):
    schema = pa.schema([("p", pa.string()), ("v", pa.int64())])
    ds = LocalArrowIPCDataset(str(tmp_path / "ds"), partition_key="p")
    table = pa.table({"p": pa.array([None, None, "x"]), "v": [1, 1, 2]}, schema=schema)
    ds.publish(table.to_batches(), partitions=[None, "x"], schema=schema)

    reader = ds.open_reader()
    assert None in reader.partition_values and "x" in reader.partition_values
    # rows are grouped by partition; within-partition order and content survive
    assert reader.read_table().sort_by([("v", "ascending")]).to_pydict() == (
        table.sort_by([("v", "ascending")]).to_pydict()
    )
    null_only = reader.read_table(partitions=[None])
    assert null_only.num_rows == 2
    assert null_only.to_pydict()["p"] == [None, None]
    reader.close()

    # zero-row rerun of the NULL partition deletes it
    ds.publish([], partitions=[None], schema=schema)
    reader = ds.open_reader()
    assert reader.partition_values == ["x"]
    reader.close()


def test_duplicate_rows_and_field_types_preserved(tmp_path):
    schema = _schema()
    ds = LocalArrowIPCDataset(str(tmp_path / "ds"), partition_key="p")
    table = _table([1, 1, 1])
    ds.publish(table.to_batches(), partitions=[1], schema=schema)

    reader = ds.open_reader()
    back = reader.read_table()
    assert back.schema.equals(schema)
    assert back.to_pydict() == table.to_pydict()
    assert back.num_rows == 3
    reader.close()


def test_empty_first_publication_has_native_schema(tmp_path):
    ds = LocalArrowIPCDataset(str(tmp_path / "ds"), partition_key="p")
    ds.publish([], partitions=["ghost"], schema=_schema())

    reader = ds.open_reader()
    assert reader.schema.equals(_schema())
    table = reader.read_table()
    assert table.num_rows == 0 and table.schema.equals(_schema())
    assert list(reader.read_batches()) == []
    reader.close()


def test_reader_pinned_across_replacements_and_deletes(tmp_path):
    ds = LocalArrowIPCDataset(str(tmp_path / "ds"), partition_key="p")
    ds.publish(_batches([1, 2], seed=0), partitions=[1, 2], schema=_schema())
    pinned = ds.open_reader()
    pinned_version = pinned.version
    expected = pinned.read_table().to_pydict()

    ds.publish(_batches([1], seed=50), partitions=[1], schema=_schema())
    ds.publish([], partitions=[2], schema=_schema())
    ds.publish(_batches([3], seed=70), partitions=[3], schema=_schema())

    # the old reader keeps reading the version it opened
    assert pinned.version == pinned_version
    assert pinned.read_table().to_pydict() == expected
    assert pinned.read_table(partitions=[2]).num_rows == 1

    # a new reader sees the new current version
    fresh = ds.open_reader()
    assert fresh.version != pinned_version
    assert sorted(fresh.partition_values) == [1, 3]
    pinned.close()
    fresh.close()


def test_pinned_versions_protected_from_gc(tmp_path):
    ds = LocalArrowIPCDataset(str(tmp_path / "ds"), partition_key="p")
    ds.publish(_batches([1]), partitions=[1], schema=_schema())
    pinned = ds.open_reader()
    pinned_id = pinned.version

    for i in range(3):
        ds.publish(_batches([1], seed=i + 10), partitions=[1], schema=_schema())
    versions_dir = os.path.join(ds.path, "versions")
    assert pinned_id in os.listdir(versions_dir)
    # pinned reader data still readable
    assert pinned.read_table().num_rows == 1

    removed = ds.gc()
    assert pinned_id not in removed
    assert pinned_id in os.listdir(versions_dir)
    pinned.close()

    removed = ds.gc()
    assert pinned_id in removed
    assert pinned_id not in os.listdir(versions_dir)


def test_selective_partition_read(tmp_path):
    ds = LocalArrowIPCDataset(str(tmp_path / "ds"), partition_key="p")
    ds.publish(_batches([1, 2, 3]), partitions=[1, 2, 3], schema=_schema())
    reader = ds.open_reader()
    table = reader.read_table(partitions=[3, 1])
    assert sorted(table.column("p").to_pylist()) == [1, 3]
    with pytest.raises(KeyError):
        reader.read_table(partitions=[99])
    reader.close()


def test_undeclared_partition_in_batch_rejected(tmp_path):
    ds = LocalArrowIPCDataset(str(tmp_path / "ds"), partition_key="p")
    ds.publish(_batches([1, 2]), partitions=[1, 2], schema=_schema())
    with pytest.raises(ValueError, match="not declared"):
        ds.publish(_batches([9]), partitions=[1], schema=_schema())
    # old version still visible and intact
    reader = ds.open_reader()
    assert sorted(reader.partition_values) == [1, 2]
    reader.close()
    # no staging leftovers
    staging = os.path.join(ds.path, "staging")
    assert not os.path.exists(staging) or os.listdir(staging) == []


def test_schema_mismatch_keeps_old_version(tmp_path):
    ds = LocalArrowIPCDataset(str(tmp_path / "ds"), partition_key="p")
    ds.publish(_batches([1]), partitions=[1], schema=_schema())
    other = pa.schema([("p", pa.int64()), ("v", pa.int64())])
    with pytest.raises(ValueError, match="schema mismatch"):
        ds.publish(
            pa.table({"p": [1], "v": [1]}, schema=other).to_batches(),
            partitions=[1],
            schema=other,
        )
    assert ds.open_reader().read_table().schema.equals(_schema())


def test_failed_generator_aborts_publication(tmp_path):
    ds = LocalArrowIPCDataset(str(tmp_path / "ds"), partition_key="p")
    ds.publish(_batches([1], seed=0), partitions=[1], schema=_schema())

    def failing_batches():
        yield _batches([1], seed=99)[0]
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        ds.publish(failing_batches(), partitions=[1], schema=_schema())
    reader = ds.open_reader()
    assert reader.read_table().to_pydict() == _table([1], seed=0).to_pydict()
    reader.close()
    assert len(os.listdir(os.path.join(ds.path, "versions"))) == 1


def test_input_validation_errors(tmp_path):
    ds = LocalArrowIPCDataset(str(tmp_path / "ds"), partition_key="p")
    # first publication requires schema
    with pytest.raises(ValueError, match="schema must be provided"):
        ds.publish(_batches([1]), partitions=[1])
    # partition key not in schema
    with pytest.raises(ValueError, match="is not in schema"):
        ds.publish(
            [],
            partitions=[1],
            schema=pa.schema([("x", pa.int64())]),
        )
    # partitioned dataset requires declared partitions
    ds.publish(_batches([1]), partitions=[1], schema=_schema())
    with pytest.raises(ValueError, match="partitions must be provided"):
        ds.publish(_batches([1]), schema=_schema())
    with pytest.raises(ValueError, match="can't be empty"):
        ds.publish([], partitions=[], schema=_schema())
    # duplicate declarations
    with pytest.raises(ValueError, match="duplicate partition value"):
        ds.publish([], partitions=[1, 1], schema=_schema())
    # incompatible partition value type
    with pytest.raises(TypeError, match="not compatible"):
        ds.publish([], partitions=["not-an-int"], schema=_schema())
    # silent truncation and overflow are also rejected
    with pytest.raises(TypeError, match="not compatible"):
        ds.publish([], partitions=[1.9], schema=_schema())
    with pytest.raises(TypeError, match="not compatible"):
        ds.publish([], partitions=[10**20], schema=_schema())
    # non RecordBatch items
    with pytest.raises(TypeError, match="expected pyarrow.RecordBatch"):
        ds.publish([_table([1])], partitions=[1], schema=_schema())


def test_partition_key_mismatch_on_reopen(tmp_path):
    path = str(tmp_path / "ds")
    ds = LocalArrowIPCDataset(path, partition_key="p")
    ds.publish(_batches([1]), partitions=[1], schema=_schema())

    wrong = LocalArrowIPCDataset(path, partition_key="other")
    with pytest.raises(ValueError, match="partition_key"):
        wrong.open_reader()
    with pytest.raises(ValueError, match="partition_key"):
        wrong.publish(_batches([1]), partitions=[1], schema=_schema())


def test_unpartitioned_dataset(tmp_path):
    schema = pa.schema([("a", pa.int64()), ("b", pa.string())])
    ds = LocalArrowIPCDataset(str(tmp_path / "ds"))
    assert ds.partition_key is None
    table = pa.table({"a": [1, 1, 2], "b": ["x", "x", "y"]}, schema=schema)
    ds.publish(table.to_batches(), schema=schema)

    reader = ds.open_reader()
    assert reader.partition_key is None
    assert reader.partition_values == []
    assert reader.read_table().to_pydict() == table.to_pydict()
    reader.close()

    # whole dataset replacement
    ds.publish(
        pa.table({"a": [3], "b": ["z"]}, schema=schema).to_batches(),
        schema=schema,
    )
    assert ds.open_reader().read_table().to_pydict() == {
        "a": [3],
        "b": ["z"],
    }

    # zero rows still carries the schema
    ds.publish([], schema=schema)
    empty = ds.open_reader().read_table()
    assert empty.num_rows == 0 and empty.schema.equals(schema)

    with pytest.raises(ValueError, match="must be None"):
        ds.publish([], partitions=[1], schema=schema)


def test_reopen_dataset_from_disk(tmp_path):
    path = str(tmp_path / "ds")
    ds = LocalArrowIPCDataset(path, partition_key="p")
    ds.publish(_batches([1, 2]), partitions=[1, 2], schema=_schema())

    reopened = LocalArrowIPCDataset(path, partition_key="p")
    assert reopened.exists
    reader = reopened.open_reader()
    assert reader.schema.equals(_schema())
    assert sorted(reader.partition_values) == [1, 2]
    reader.close()

    with pytest.raises(FileNotFoundError):
        LocalArrowIPCDataset(str(tmp_path / "never"), partition_key="p").open_reader()


def test_zero_row_batches_are_ignored(tmp_path):
    ds = LocalArrowIPCDataset(str(tmp_path / "ds"), partition_key="p")
    empty_batch = pa.RecordBatch.from_arrays(
        [pa.array([], pa.int64()), pa.array([], pa.string())],
        names=["p", "s"],
    )
    schema = pa.schema([("p", pa.int64()), ("s", pa.string())])
    real = pa.table({"p": [1, 1], "s": ["a", "a"]}, schema=schema).to_batches()
    ds.publish([empty_batch, *real, empty_batch], partitions=[1], schema=schema)
    reader = ds.open_reader()
    assert reader.read_table().to_pydict() == {"p": [1, 1], "s": ["a", "a"]}
    data_files = os.listdir(os.path.join(ds.path, "versions", reader.version, "data"))
    assert len(data_files) == 1
    reader.close()


def test_batch_stream_works_with_reslicer(tmp_path):
    schema = pa.schema([("p", pa.int64()), ("v", pa.int64())])
    ds = LocalArrowIPCDataset(str(tmp_path / "ds"), partition_key="p")
    big = pa.table({"p": [1] * 10, "v": list(range(10))}, schema=schema)
    ds.publish(big.to_batches(), partitions=[1], schema=schema)

    reader = ds.open_reader()
    tables = [pa.Table.from_batches([b]) for b in reader.read_batches()]
    resliced = list(ArrowTableBatchReslicer(row_limit=3).reslice(tables))
    assert [t.num_rows for t in resliced] == [3, 3, 3, 1]
    assert pa.concat_tables(resliced).to_pydict() == big.to_pydict()
    reader.close()


def test_reader_context_manager_and_close_guard(tmp_path):
    ds = LocalArrowIPCDataset(str(tmp_path / "ds"), partition_key="p")
    ds.publish(_batches([1]), partitions=[1], schema=_schema())
    with ds.open_reader() as reader:
        assert reader.read_table().num_rows == 1
    assert reader.closed
    with pytest.raises(ValueError, match="reader is closed"):
        list(reader.read_batches())


def test_partition_with_int32_column_accepts_python_int(tmp_path):
    schema = pa.schema([("p", pa.int32()), ("v", pa.string())])
    ds = LocalArrowIPCDataset(str(tmp_path / "ds"), partition_key="p")
    ds.publish(
        pa.table({"p": [1, 2], "v": ["a", "b"]}, schema=schema).to_batches(),
        partitions=[1, 2],
        schema=schema,
    )
    # rerun partition 1 declared with a plain python int
    ds.publish(
        pa.table({"p": [1], "v": ["c"]}, schema=schema).to_batches(),
        partitions=[1],
        schema=schema,
    )
    reader = ds.open_reader()
    assert sorted(reader.partition_values) == [1, 2]
    assert reader.read_table(partitions=[1]).to_pydict() == {
        "p": [1],
        "v": ["c"],
    }
    reader.close()


def test_metadata_is_preserved(tmp_path):
    schema = pa.schema(
        [("p", pa.int64()), ("v", pa.string())],
        metadata={b"owner": b"team-a", b"note": b"keep"},
    )
    ds = LocalArrowIPCDataset(str(tmp_path / "ds"), partition_key="p")
    ds.publish(
        pa.table({"p": [1], "v": ["x"]}, schema=schema).to_batches(),
        partitions=[1],
        schema=schema,
    )
    reader = ds.open_reader()
    assert reader.schema.metadata == {b"owner": b"team-a", b"note": b"keep"}
    assert reader.read_table().schema.metadata == schema.metadata
    reader.close()


def test_gc_force_removes_pinned_but_keeps_current(tmp_path):
    ds = LocalArrowIPCDataset(str(tmp_path / "ds"), partition_key="p")
    ds.publish(_batches([1]), partitions=[1], schema=_schema())
    pinned = ds.open_reader()
    pinned_id = pinned.version
    ds.publish(_batches([1], seed=2), partitions=[1], schema=_schema())
    current_id = ds.open_reader().version

    removed = ds.gc(force=True)
    assert pinned_id in removed
    # the current version is always retained
    assert current_id in os.listdir(os.path.join(ds.path, "versions"))
    pinned.close()


def test_untouched_partition_files_carry_to_new_version(tmp_path, monkeypatch):
    ds = LocalArrowIPCDataset(str(tmp_path / "ds"), partition_key="p")
    ds.publish(_batches([1, 2]), partitions=[1, 2], schema=_schema())
    first_reader = ds.open_reader()
    first_version = first_reader.version
    original = sorted(
        os.listdir(os.path.join(ds.path, "versions", first_version, "data"))
    )
    first_reader.close()

    # force the copy fallback instead of hard links
    def raise_link(source, target):
        raise OSError("hard links unsupported")

    monkeypatch.setattr(os, "link", raise_link)
    ds.publish(_batches([1], seed=9), partitions=[1], schema=_schema())
    reader = ds.open_reader()
    data_dir = os.path.join(ds.path, "versions", reader.version, "data")
    carried = sorted(os.listdir(data_dir))
    # partition 2's file keeps its name and its content in the new version dir
    assert set(original).intersection(carried)  # the carried p2 file
    assert reader.read_table(partitions=[2]).num_rows == 1
    reader.close()
