"""Local Arrow IPC partitioned dataset with atomic, versioned publication.

The module provides a local file based dataset that stores each partition in
Arrow IPC (streaming) files. Publications are atomic: all input batches are
validated and fully written before the visible version switches, so readers
never observe a half-written partition rerun.

A publication explicitly declares the partitions it replaces. Input batches
may only contain rows of those partitions, untouched partitions keep their
old data, and a declared partition that receives zero rows is deleted.

Readers pin the version they open: later publications, partition replacements
or deletions never change the data a reader has not consumed yet, while newly
opened readers always see the latest version.
"""

import base64
import json
import os
import shutil
import threading
import uuid
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pyarrow as pa
import pyarrow.compute as pc

from triad.utils.assertion import assert_arg_not_none, assert_or_throw
from triad.utils.pyarrow import schemas_equal

_MANIFEST_NAME = "manifest.json"
_CURRENT_NAME = "current"
_CURRENT_DIR = "meta"
_VERSIONS_DIR = "versions"
_STAGING_DIR = "staging"
_DATA_DIR = "data"
_SCHEMA_KEY = "schema"
_PARTITION_KEY = "partition_key"
_PARTITIONS_KEY = "partitions"
_VERSION_KEY = "version"


def _encode_scalar(value: Any) -> str:
    """Encode a python value (including None) into a filesystem safe string."""
    scalar = pa.scalar(value)
    schema = pa.schema([pa.field("v", scalar.type)])
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, schema) as writer:
        writer.write_batch(
            pa.record_batch([pa.array([value], type=scalar.type)], schema=schema)
        )
    return base64.urlsafe_b64encode(sink.getvalue()).decode("ascii")


def _decode_scalar(encoded: str) -> Any:
    """Reverse of :func:`_encode_scalar`, returns the python native value."""
    reader = pa.ipc.open_stream(pa.BufferReader(base64.urlsafe_b64decode(encoded)))
    return reader.read_next_batch().column(0)[0].as_py()


def _schema_to_json(schema: pa.Schema) -> str:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, schema):
        pass
    return base64.urlsafe_b64encode(sink.getvalue()).decode("ascii")


def _schema_from_json(encoded: str) -> pa.Schema:
    reader = pa.ipc.open_stream(pa.BufferReader(base64.urlsafe_b64decode(encoded)))
    return reader.schema


# manifest key of the single anonymous group of an unpartitioned dataset
_ANONYMOUS_KEY = _encode_scalar(None)


class ArrowIPCReader:
    """A version pinned reader of a local Arrow IPC partitioned dataset.

    The reader is bound to the version that was current when it was opened.
    Concurrent or later publications do not affect its data.

    :param version_path: the absolute path of the pinned version directory
    :param manifest: the parsed manifest of the pinned version
    """

    def __init__(self, version_path: str, manifest: Dict[str, Any]):
        self._version_path = version_path
        self._manifest = manifest
        self._schema = _schema_from_json(manifest[_SCHEMA_KEY])
        self._closed = False

    @property
    def schema(self) -> pa.Schema:
        """The native Arrow schema of the dataset, even when it has zero rows."""
        return self._schema

    @property
    def version(self) -> str:
        """The pinned version id."""
        return self._manifest[_VERSION_KEY]

    @property
    def partition_key(self) -> Optional[str]:
        """The partition column name, or None when the dataset is unpartitioned."""
        return self._manifest[_PARTITION_KEY]

    @property
    def partition_values(self) -> List[Any]:
        """The partition values existing in this version, in manifest order."""
        if self._manifest[_PARTITION_KEY] is None:
            return []
        return [_decode_scalar(item["key"]) for item in self._manifest[_PARTITIONS_KEY]]

    @property
    def closed(self) -> bool:
        """Whether the reader has been closed."""
        return self._closed

    def read_batches(
        self, partitions: Optional[Iterable[Any]] = None
    ) -> Iterable[pa.RecordBatch]:
        """Read the pinned version as a stream of Arrow record batches.

        The yielded objects are native :class:`pyarrow.RecordBatch` instances,
        so the stream can be directly consumed by the existing slicing and
        conversion utilities (for example
        :class:`~triad.utils.batch_reslicers.ArrowTableBatchReslicer` after
        ``pa.Table.from_batches``).

        :param partitions: optional partition values to restrict the read to.
            Each value must exist in this version. When None, all partitions
            are read in manifest order.
        :raises KeyError: if a requested partition does not exist
        :yield: Arrow record batches
        """
        assert_or_throw(not self._closed, lambda: ValueError("reader is closed"))
        selected = self._resolve_partition_files(partitions)
        for file_name in selected:
            with pa.memory_map(
                os.path.join(self._version_path, _DATA_DIR, file_name), "r"
            ) as source:
                with pa.ipc.open_stream(source) as reader:
                    for batch in reader:
                        if batch.num_rows > 0:
                            yield batch

    def read_table(self, partitions: Optional[Iterable[Any]] = None) -> pa.Table:
        """Read the pinned version as a single Arrow table.

        An empty dataset still returns a table carrying the native schema.

        :param partitions: optional partition values to restrict the read to
        :return: the dataset (or selected partitions) as an Arrow table
        """
        batches = list(self.read_batches(partitions))
        if len(batches) == 0:
            return pa.Table.from_batches([], schema=self._schema)
        return pa.Table.from_batches(batches, schema=self._schema)

    def close(self) -> None:
        """Release the reader. The pinned version stays on disk until gc."""
        self._closed = True

    def __enter__(self) -> "ArrowIPCReader":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _resolve_partition_files(
        self, partitions: Optional[Iterable[Any]]
    ) -> List[str]:
        entries = self._manifest[_PARTITIONS_KEY]
        if partitions is None:
            return [item["file"] for item in entries]
        assert_or_throw(
            self._manifest[_PARTITION_KEY] is not None,
            lambda: ValueError(
                "partitions filter is not supported on an unpartitioned dataset"
            ),
        )
        wanted = {_encode_scalar(v) for v in partitions}
        by_key = {item["key"]: item["file"] for item in entries}
        missing = [k for k in wanted if k not in by_key]
        assert_or_throw(
            len(missing) == 0,
            lambda: KeyError(
                "partitions not found in this version: "
                + str([_decode_scalar(k) for k in missing])
            ),
        )
        return [by_key[k] for k in by_key if k in wanted]


class LocalArrowIPCDataset:
    """A local, versioned, partitioned Arrow IPC dataset.

    Data is stored under ``path`` with one immutable directory per version.
    A ``current`` pointer file atomically switches the visible version, so a
    publication is all-or-nothing: schema validation and every file write must
    succeed before the pointer moves, and a failed publication leaves the
    previously visible version untouched.

    :param path: the dataset directory, created on the first publication
    :param partition_key: the column used as partition key. When None, the
        dataset holds a single unpartitioned dataset and each publication
        replaces the whole dataset.
    """

    def __init__(self, path: str, partition_key: Optional[str] = None):
        assert_arg_not_none(path, "path")
        self._path = os.path.abspath(path)
        self._partition_key = partition_key
        self._lock = threading.RLock()
        self._pins: Dict[str, int] = {}

    @property
    def path(self) -> str:
        """The absolute dataset directory."""
        return self._path

    @property
    def partition_key(self) -> Optional[str]:
        """The partition column name, None for unpartitioned datasets."""
        return self._partition_key

    @property
    def exists(self) -> bool:
        """Whether at least one version has been published."""
        return os.path.isfile(self._current_file())

    def publish(
        self,
        batches: Iterable[pa.RecordBatch],
        partitions: Optional[Iterable[Any]] = None,
        schema: Optional[pa.Schema] = None,
    ) -> str:
        """Atomically publish a new version replacing the given partitions.

        Every input batch must have the same schema as the dataset and may
        only contain rows whose partition value is in ``partitions``. Rows of
        other partitions are rejected, so a rerun can never silently overwrite
        data it did not declare. A declared partition that receives zero rows
        is deleted in the new version; all other partitions keep their old
        data. On any failure the previous version remains visible and the
        staging data is removed.

        :param batches: iterable of Arrow record batches. It is fully consumed
            before the version switch, so generator failures abort the
            publication.
        :param partitions: the partition values this publication replaces. For
            a partitioned dataset it must be a non-empty iterable. For an
            unpartitioned dataset it must be None.
        :param schema: required when publishing the first version or when
            ``batches`` may be empty; otherwise taken from the first batch and
            checked against the existing dataset schema.
        :raises ValueError: for invalid arguments or schema mismatches
        :return: the new version id
        """
        with self._lock:
            old_manifest = self._read_current_manifest()
            self._assert_partition_key_consistent(old_manifest)
            effective_schema = self._resolve_publish_schema(schema, old_manifest)
            declared = self._validate_declared_partitions(partitions, old_manifest)
            version_id = uuid.uuid4().hex
            staging_path = os.path.join(self._path, _STAGING_DIR, version_id)
            final_path = os.path.join(self._path, _VERSIONS_DIR, version_id)
            try:
                os.makedirs(os.path.join(staging_path, _DATA_DIR))
                new_entries = self._write_staging(
                    staging_path, batches, declared, effective_schema
                )
                entries = self._merge_manifest_entries(
                    old_manifest, declared, new_entries, staging_path
                )
                manifest = {
                    _VERSION_KEY: version_id,
                    _SCHEMA_KEY: _schema_to_json(effective_schema),
                    _PARTITION_KEY: self._partition_key,
                    _PARTITIONS_KEY: entries,
                }
                self._write_manifest(staging_path, manifest)
                os.makedirs(os.path.dirname(final_path), exist_ok=True)
                os.replace(staging_path, final_path)
                self._switch_current(version_id)
            except BaseException:
                shutil.rmtree(staging_path, ignore_errors=True)
                if "final_path" in locals() and os.path.isdir(final_path):
                    shutil.rmtree(final_path, ignore_errors=True)
                raise
            self._cleanup_staging_dir()
            self._gc_versions()
            return version_id

    def open_reader(self) -> ArrowIPCReader:
        """Open a reader pinned to the currently visible version.

        Publications happening afterwards never change the reader's data; a
        reader opened later sees the new version.

        :raises FileNotFoundError: if no version has been published
        :return: a version pinned :class:`ArrowIPCReader`
        """
        with self._lock:
            version_id, manifest = self._read_current_version()
            self._assert_partition_key_consistent(manifest)
            version_path = os.path.join(self._path, _VERSIONS_DIR, version_id)
            assert_or_throw(
                os.path.isdir(version_path),
                lambda: FileNotFoundError(f"version {version_id} is missing on disk"),
            )
            self._pins[version_id] = self._pins.get(version_id, 0) + 1
            return _PinnedReader(self, version_id, version_path, manifest)

    def gc(self, force: bool = False) -> List[str]:
        """Remove version directories that are no longer reachable.

        The current version and versions pinned by open in-process readers are
        always kept.

        :param force: if True, also remove versions pinned by in-process
            readers (the current version is still kept)
        :return: the removed version ids
        """
        with self._lock:
            return self._gc_versions(force=force)

    def _write_staging(
        self,
        staging_path: str,
        batches: Iterable[pa.RecordBatch],
        declared: Set[Any],
        schema: pa.Schema,
    ) -> List[Dict[str, str]]:
        partition_index = (
            schema.get_field_index(self._partition_key)
            if self._partition_key is not None
            else -1
        )
        writers: Dict[Any, pa.ipc.RecordBatchStreamWriter] = {}
        files: Dict[Any, str] = {}
        counts: Dict[Any, int] = {}
        try:
            for batch in batches:
                assert_or_throw(
                    isinstance(batch, pa.RecordBatch),
                    lambda: TypeError(
                        f"expected pyarrow.RecordBatch, got {type(batch)}"
                    ),
                )
                assert_or_throw(
                    schemas_equal(batch.schema, schema),
                    lambda: ValueError(
                        "batch schema mismatch.\n"
                        f"expected: {schema}\n"
                        f"got: {batch.schema}"
                    ),
                )
                if self._partition_key is None:
                    self._append_unpartitioned(
                        staging_path, batch, writers, files, counts, schema
                    )
                else:
                    self._append_partitioned(
                        staging_path,
                        batch,
                        partition_index,
                        declared,
                        writers,
                        files,
                        counts,
                        schema,
                    )
        finally:
            for writer in writers.values():
                writer.close()
        if self._partition_key is None:
            # unpartitioned datasets keep a single anonymous data group
            values: Iterable[Any] = [None]
        else:
            values = declared
        entries: List[Dict[str, str]] = []
        for value in values:
            if counts.get(value, 0) <= 0:
                continue
            if self._partition_key is None:
                key = _ANONYMOUS_KEY
            else:
                key = _encode_scalar(value)
            entries.append({"key": key, "file": files[value]})
        return entries

    def _append_unpartitioned(
        self,
        staging_path: str,
        batch: pa.RecordBatch,
        writers: Dict[Any, pa.ipc.RecordBatchStreamWriter],
        files: Dict[Any, str],
        counts: Dict[Any, int],
        schema: pa.Schema,
    ) -> None:
        if batch.num_rows == 0:
            return
        if None not in writers:
            file_name = self._new_file_name()
            files[None] = file_name
            writers[None] = pa.ipc.new_stream(
                os.path.join(staging_path, _DATA_DIR, file_name), schema
            )
            counts[None] = 0
        writers[None].write_batch(batch)
        counts[None] += batch.num_rows

    def _append_partitioned(
        self,
        staging_path: str,
        batch: pa.RecordBatch,
        partition_index: int,
        declared: Set[Any],
        writers: Dict[Any, pa.ipc.RecordBatchStreamWriter],
        files: Dict[Any, str],
        counts: Dict[Any, int],
        schema: pa.Schema,
    ) -> None:
        if batch.num_rows == 0:
            return
        keys = batch.column(partition_index)
        unique = pc.unique(keys).to_pylist()
        undeclared = [v for v in unique if v not in declared]
        assert_or_throw(
            len(undeclared) == 0,
            lambda: ValueError(
                "batch contains partitions not declared for replacement: "
                + str(undeclared)
            ),
        )
        for value in unique:
            # pc.equal never matches NULLs, so compare nullness explicitly
            if value is None:
                mask = pc.is_null(keys)
            else:
                mask = pc.equal(keys, pa.scalar(value, type=keys.type))
            sub = batch.filter(mask)
            if sub.num_rows == 0:
                continue
            if value not in writers:
                file_name = self._new_file_name()
                files[value] = file_name
                writers[value] = pa.ipc.new_stream(
                    os.path.join(staging_path, _DATA_DIR, file_name), schema
                )
                counts[value] = 0
            writers[value].write_batch(sub)
            counts[value] += sub.num_rows

    def _merge_manifest_entries(
        self,
        old_manifest: Optional[Dict[str, Any]],
        declared: Set[Any],
        new_entries: List[Dict[str, str]],
        staging_path: str,
    ) -> List[Dict[str, str]]:
        if self._partition_key is None:
            # unpartitioned datasets keep a single anonymous data group, and
            # every publication replaces that group wholesale
            return new_entries
        if old_manifest is None:
            return new_entries
        declared_encoded = {_encode_scalar(v) for v in declared}
        old_version_path = os.path.join(
            self._path, _VERSIONS_DIR, old_manifest[_VERSION_KEY]
        )
        kept: List[Dict[str, str]] = []
        for item in old_manifest[_PARTITIONS_KEY]:
            if item["key"] in declared_encoded:
                continue
            # every version directory is self contained: carry the untouched
            # partition file over by hard link (zero copy) or plain copy
            source = os.path.join(old_version_path, _DATA_DIR, item["file"])
            target = os.path.join(staging_path, _DATA_DIR, item["file"])
            self._link_or_copy(source, target)
            kept.append(dict(item))
        return kept + new_entries

    @staticmethod
    def _link_or_copy(source: str, target: str) -> None:
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)

    def _resolve_publish_schema(
        self,
        schema: Optional[pa.Schema],
        old_manifest: Optional[Dict[str, Any]],
    ) -> pa.Schema:
        if old_manifest is not None:
            existing = _schema_from_json(old_manifest[_SCHEMA_KEY])
            if schema is None:
                return existing
            assert_or_throw(
                schemas_equal(schema, existing),
                lambda: ValueError(
                    "schema mismatch with existing dataset.\n"
                    f"existing: {existing}\n"
                    f"given: {schema}"
                ),
            )
            return existing
        assert_or_throw(
            schema is not None,
            lambda: ValueError(
                "schema must be provided when publishing the first version"
            ),
        )
        if self._partition_key is not None:
            assert_or_throw(
                schema.get_field_index(self._partition_key) >= 0,
                lambda: ValueError(
                    f"partition_key '{self._partition_key}' is not in schema {schema}"
                ),
            )
        return schema

    def _validate_declared_partitions(
        self,
        partitions: Optional[Iterable[Any]],
        old_manifest: Optional[Dict[str, Any]],
    ) -> Set[Any]:
        if self._partition_key is None:
            assert_or_throw(
                partitions is None,
                lambda: ValueError(
                    "partitions must be None for an unpartitioned dataset"
                ),
            )
            return set()
        assert_or_throw(
            partitions is not None,
            lambda: ValueError("partitions must be provided for a partitioned dataset"),
        )
        declared = list(partitions)
        assert_or_throw(
            len(declared) > 0,
            lambda: ValueError("partitions to replace can't be empty"),
        )
        seen: Set[str] = set()
        unique: Set[Any] = set()
        partition_type: Optional[pa.DataType] = None
        if old_manifest is not None:
            partition_type = (
                _schema_from_json(old_manifest[_SCHEMA_KEY])
                .field(
                    self._partition_key  # type: ignore
                )
                .type
            )
        for value in declared:
            if partition_type is not None and value is not None:
                self._assert_partition_value_compatible(value, partition_type)
            encoded = _encode_scalar(value)
            assert_or_throw(
                encoded not in seen,
                lambda: ValueError(f"duplicate partition value: {value}"),
            )
            seen.add(encoded)
            unique.add(value)
        return unique

    @staticmethod
    def _assert_partition_value_compatible(
        value: Any, partition_type: pa.DataType
    ) -> None:
        if isinstance(value, bool) and not pa.types.is_boolean(partition_type):
            raise TypeError(
                f"partition value {value!r} is not compatible with "
                f"partition column type {partition_type}"
            )
        if isinstance(value, float) and pa.types.is_integer(partition_type):
            # pa.scalar would otherwise silently truncate, e.g. 1.9 -> 1
            if not float(value).is_integer():
                raise TypeError(
                    f"partition value {value!r} is not compatible with "
                    f"partition column type {partition_type}"
                )
        try:
            pa.scalar(value, partition_type)
        except (
            pa.ArrowInvalid,
            pa.ArrowTypeError,
            TypeError,
            OverflowError,
        ) as e:
            raise TypeError(
                f"partition value {value!r} is not compatible with "
                f"partition column type {partition_type}: {e}"
            ) from e

    def _read_current_manifest(self) -> Optional[Dict[str, Any]]:
        if not os.path.isfile(self._current_file()):
            return None
        with open(self._current_file(), "r", encoding="utf-8") as f:
            version_id = f.read().strip()
        return self._read_manifest(version_id)

    def _read_current_version(self) -> Tuple[str, Dict[str, Any]]:
        manifest = self._read_current_manifest()
        assert_or_throw(
            manifest is not None,
            lambda: FileNotFoundError(f"no published version exists at {self._path}"),
        )
        return manifest[_VERSION_KEY], manifest  # type: ignore

    def _read_manifest(self, version_id: str) -> Dict[str, Any]:
        manifest_path = os.path.join(
            self._path, _VERSIONS_DIR, version_id, _MANIFEST_NAME
        )
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_manifest(self, version_path: str, manifest: Dict[str, Any]) -> None:
        manifest_path = os.path.join(version_path, _MANIFEST_NAME)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())

    def _switch_current(self, version_id: str) -> None:
        os.makedirs(os.path.dirname(self._current_file()), exist_ok=True)
        tmp_file = os.path.join(self._path, _CURRENT_DIR, f".{uuid.uuid4().hex}.tmp")
        with open(tmp_file, "w", encoding="utf-8") as f:
            f.write(version_id)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_file, self._current_file())

    def _gc_versions(self, force: bool = False) -> List[str]:
        versions_dir = os.path.join(self._path, _VERSIONS_DIR)
        if not os.path.isdir(versions_dir):
            return []
        current_id = None
        if os.path.isfile(self._current_file()):
            with open(self._current_file(), "r", encoding="utf-8") as f:
                current_id = f.read().strip()
        protected = {current_id}
        if not force:
            protected.update(
                version_id for version_id, count in self._pins.items() if count > 0
            )
        removed: List[str] = []
        for entry in os.listdir(versions_dir):
            if entry in protected:
                continue
            shutil.rmtree(os.path.join(versions_dir, entry), ignore_errors=True)
            removed.append(entry)
            self._pins.pop(entry, None)
        return removed

    def _cleanup_staging_dir(self) -> None:
        staging_dir = os.path.join(self._path, _STAGING_DIR)
        if os.path.isdir(staging_dir) and len(os.listdir(staging_dir)) == 0:
            shutil.rmtree(staging_dir, ignore_errors=True)

    def _release_pin(self, version_id: str) -> None:
        with self._lock:
            count = self._pins.get(version_id, 0)
            if count <= 1:
                self._pins.pop(version_id, None)
            else:
                self._pins[version_id] = count - 1

    def _current_file(self) -> str:
        return os.path.join(self._path, _CURRENT_DIR, _CURRENT_NAME)

    def _assert_partition_key_consistent(
        self, manifest: Optional[Dict[str, Any]]
    ) -> None:
        if manifest is None:
            return
        stored = manifest[_PARTITION_KEY]
        assert_or_throw(
            stored == self._partition_key,
            lambda: ValueError(
                f"dataset at {self._path} was created with partition_key "
                f"{stored!r}, but {self._partition_key!r} was given"
            ),
        )

    @staticmethod
    def _new_file_name() -> str:
        return uuid.uuid4().hex + ".arrows"


class _PinnedReader(ArrowIPCReader):
    """Reader that unregisters its in-process pin on close."""

    def __init__(
        self,
        dataset: LocalArrowIPCDataset,
        version_id: str,
        version_path: str,
        manifest: Dict[str, Any],
    ):
        super().__init__(version_path, manifest)
        self._dataset = dataset
        self._version_id = version_id

    def close(self) -> None:
        if not self._closed:
            super().close()
            self._dataset._release_pin(self._version_id)
