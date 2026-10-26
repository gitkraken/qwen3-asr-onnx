#!/usr/bin/env python3
"""Share decoder data without invalidating the source models during publication.

Data and both graphs are validated in staging. The canonical data file is
published before either graph, and source data is retained until both graphs
have been replaced. Individual replacements are atomic; the group is not.
An interrupted run can retain an unreferenced custom data file. Conventional
per-decoder data files are cleaned on retry, without deleting live references.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import onnx
from google.protobuf.descriptor import FieldDescriptor

SHARED_DATA_NAME = "decoder_weights.data"
_CHUNK_BYTES = 8 * 1024 * 1024


def _shared_data_name(suffix=None):
    return f"decoder_weights.{suffix}.data" if suffix else SHARED_DATA_NAME


def _decoder_filenames(suffix=None):
    tag = f".{suffix}" if suffix else ""
    return f"decoder_init{tag}.onnx", f"decoder_step{tag}.onnx"


def _storage_tensors(message, initializer=False):
    if isinstance(message, onnx.TensorProto):
        yield message, initializer
        return
    for field, value in message.ListFields():
        if field.type != FieldDescriptor.TYPE_MESSAGE:
            continue
        if field.is_repeated:
            for item in value:
                yield from _storage_tensors(item, field.name == "initializer")
        else:
            yield from _storage_tensors(value, field.name == "initializer")


def model_tensors(message):
    for tensor, _ in _storage_tensors(message):
        yield tensor


def _can_share(tensor, root, initializer):
    # ORT materializes attributes and shape/control inputs before loading external weights.
    if not initializer:
        return False
    if tensor.data_type in (
        onnx.TensorProto.INT16,
        onnx.TensorProto.INT32,
        onnx.TensorProto.INT64,
        onnx.TensorProto.UINT16,
        onnx.TensorProto.UINT32,
        onnx.TensorProto.UINT64,
        onnx.TensorProto.BOOL,
        onnx.TensorProto.STRING,
    ):
        return False
    length = (
        _external_range(root, tensor)[2] if tensor.data_location == onnx.TensorProto.EXTERNAL else len(tensor.raw_data)
    )
    return length >= 1024


def get_external_info(tensor):
    return {entry.key: entry.value for entry in tensor.external_data}


def _external_range(root, tensor):
    info = get_external_info(tensor)
    location = info.get("location", "")
    if not location or "\\" in location or Path(location).is_absolute():
        raise ValueError(f"Invalid external data location: {location!r}")
    root = Path(root).resolve()
    if ".." in Path(location).parts:
        raise ValueError(f"Invalid external data location: {location!r}")
    path = root / location
    if path.resolve() != path or not path.is_file():
        raise ValueError(f"External data missing or outside model directory: {location}")
    size = path.stat().st_size
    offset = int(info.get("offset", "0"))
    length = int(info.get("length", str(size - offset)))
    if offset < 0 or length < 0 or offset + length > size:
        raise ValueError(f"Invalid external data range for {tensor.name}")
    dtype_name = onnx.TensorProto.DataType.Name(tensor.data_type)
    if dtype_name == "STRING":
        raise ValueError("External string tensors are unsupported")
    if dtype_name in ("INT4", "UINT4", "FLOAT4E2M1"):
        bits = 4
    elif dtype_name in ("INT2", "UINT2"):
        bits = 2
    else:
        bits = onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type).itemsize * 8
    expected = (math.prod(tensor.dims) * bits + 7) // 8
    if length != expected:
        raise ValueError(f"External tensor {tensor.name} has {length} bytes; expected {expected}")
    return path, offset, length


def external_data_files(model_path):
    path = Path(model_path).resolve()
    model = onnx.load(path, load_external_data=False)
    return {
        _external_range(path.parent, tensor)[0]
        for tensor in model_tensors(model)
        if tensor.data_location == onnx.TensorProto.EXTERNAL
    }


def graph_file_closure(root, names):
    root = Path(root).resolve()
    graphs = set(names)
    files = set(graphs)
    for name in graphs:
        for path in external_data_files(root / name):
            files.add(path.relative_to(root).as_posix())
    return files


def transport_graph_files(root, graphs, generated=(), rewritten=()):
    graphs = set(graphs)
    files = graph_file_closure(root, graphs)
    conflicts = (files & set(generated)) | ((files - graphs) & set(rewritten))
    if conflicts:
        raise ValueError(f"ONNX transport destination conflict: {', '.join(sorted(conflicts))}")
    return files


def validate_generated_destinations(output, names):
    root = Path(output).absolute()
    if root.is_symlink():
        raise ValueError(f"ONNX transport destination conflict: linked output directory {root}")
    resolved_root = root.resolve()
    for name in names:
        destination = root / name
        if not destination.parent.resolve().is_relative_to(resolved_root):
            raise ValueError(f"ONNX transport destination conflict: {name}")
        if destination.is_symlink() or (destination.exists() and destination.stat().st_nlink > 1):
            raise ValueError(f"ONNX transport destination conflict: linked {name}")


def copy_artifact_entry(source, output, name):
    root = Path(output).resolve()
    destination = Path(output) / name
    if not destination.parent.resolve().is_relative_to(root):
        raise ValueError(f"Artifact destination escapes output directory: {name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.parent.resolve().is_relative_to(root):
        raise ValueError(f"Artifact destination escapes output directory: {name}")
    with tempfile.NamedTemporaryFile(prefix=".artifact-", dir=destination.parent, delete=False) as temporary:
        staged = Path(temporary.name)
    try:
        shutil.copy2(Path(source) / name, staged)
        os.replace(staged, destination)
    finally:
        staged.unlink(missing_ok=True)


def write_artifact_json(output, name, value):
    destination = Path(output) / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".artifact-", dir=destination.parent) as temp:
        staged = Path(temp) / "value.json"
        with staged.open("w") as stream:
            json.dump(value, stream, indent=2)
        staged.chmod(0o644)
        os.replace(staged, destination)


def _tensor_chunks(tensor, root):
    if tensor.data_location != onnx.TensorProto.EXTERNAL:
        yield tensor.raw_data
        return
    path, offset, remaining = _external_range(root, tensor)
    with path.open("rb") as stream:
        stream.seek(offset)
        while remaining:
            chunk = stream.read(min(remaining, _CHUNK_BYTES))
            if not chunk:
                raise ValueError(f"Unexpected end of data for {tensor.name}")
            remaining -= len(chunk)
            yield chunk


def set_tensor_external(tensor, location, offset, length):
    tensor.ClearField("raw_data")
    tensor.ClearField("external_data")
    tensor.data_location = onnx.TensorProto.EXTERNAL
    for key, value in (("location", location), ("offset", str(offset)), ("length", str(length))):
        entry = tensor.external_data.add()
        entry.key = key
        entry.value = value


def _write_shared_data(storage, root, path):
    index = {}
    with path.open("wb") as stream:
        for tensor, shareable in storage:
            if tensor.data_location != onnx.TensorProto.EXTERNAL and not tensor.raw_data:
                continue
            if not shareable:
                if tensor.data_location == onnx.TensorProto.EXTERNAL:
                    data = b"".join(_tensor_chunks(tensor, root))
                    tensor.ClearField("external_data")
                    tensor.data_location = onnx.TensorProto.DEFAULT
                    tensor.raw_data = data
                continue
            digest = hashlib.sha256()
            length = 0
            for chunk in _tensor_chunks(tensor, root):
                digest.update(chunk)
                length += len(chunk)
            key = (tensor.data_type, tuple(tensor.dims), length, digest.digest())
            if key not in index:
                index[key] = stream.tell()
                for chunk in _tensor_chunks(tensor, root):
                    stream.write(chunk)
            set_tensor_external(tensor, path.name, index[key], length)
        stream.flush()
        os.fsync(stream.fileno())


def _file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").digest()


def _live_data_files(root, excluded):
    live = set()
    for path in root.rglob("*.onnx"):
        if path not in excluded:
            try:
                live.update(external_data_files(path))
            except Exception as error:
                raise ValueError(
                    f"Cannot inventory independent graph {path}; preserve it and repair its references "
                    "or move abandoned staging outside the model directory before retrying."
                ) from error
    return live


def _publish_data(staged, target):
    if target.is_symlink():
        raise ValueError("Canonical shared data cannot be a symlink; rebuild in a clean output directory")
    if target.exists():
        identical = (
            target.is_file()
            and staged.stat().st_size == target.stat().st_size
            and _file_digest(staged) == _file_digest(target)
        )
        if identical:
            return
        raise ValueError("Canonical shared data has incompatible bytes; rebuild in a clean output directory")
    os.replace(staged, target)


def _cleanup_data(root, names, live):
    candidates = {root / f"{name}.data" for name in names}
    for path in candidates - live:
        if path.exists():
            path.unlink()


def _validate_models(root, names):
    for name in names:
        external_data_files(root / name)
        onnx.checker.check_model(str(root / name))


def _validate_runtime_models(root, names):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    for name in names:
        ort.InferenceSession(str(root / name), options, providers=["CPUExecutionProvider"])


def share_external_models(model_dir, suffix=None):
    root = Path(model_dir).resolve()
    names = _decoder_filenames(suffix)
    if not all((root / name).is_file() for name in names):
        raise FileNotFoundError(f"Both split decoder models are required in {root}")
    models = [onnx.load(root / name, load_external_data=False) for name in names]
    old_files = set().union(*(external_data_files(root / name) for name in names))
    shared_name = _shared_data_name(suffix)
    _validate_models(root, names)
    live = _live_data_files(root, {root / name for name in names})
    storage = [
        (tensor, _can_share(tensor, root, initializer))
        for model in models
        for tensor, initializer in _storage_tensors(model)
    ]
    already_shared = (
        (root / shared_name).is_file()
        and bool(old_files)
        and old_files <= {root / shared_name}
        and all((tensor.data_location == onnx.TensorProto.EXTERNAL) == shareable for tensor, shareable in storage)
    )
    if already_shared:
        _validate_runtime_models(root, names)
        _cleanup_data(root, names, live)
        return True
    # A process exit must not leave transient graphs inside the recursive consumer inventory.
    with tempfile.TemporaryDirectory(prefix=".share-", dir=root.parent) as tmp:
        stage = Path(tmp)
        _write_shared_data(storage, root, stage / shared_name)
        for model, name in zip(models, names):
            with open(stage / name, "wb") as stream:
                stream.write(model.SerializeToString())
                stream.flush()
                os.fsync(stream.fileno())
        _validate_models(stage, names)
        _validate_runtime_models(stage, names)
        _publish_data(stage / shared_name, root / shared_name)
        for name in names:
            os.replace(stage / name, root / name)
        _validate_models(root, names)
    _cleanup_data(root, names, live)
    return True


def share_inline_models(model_dir, suffix=None):
    return share_external_models(model_dir, suffix=suffix)


def verify_model(model_dir, filename):
    try:
        _validate_models(Path(model_dir), [filename])
        return True
    except Exception as error:
        print(f"{filename}: FAILED: {error}")
        return False


def verify_inference(model_dir, suffix=None):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    try:
        for name in _decoder_filenames(suffix):
            ort.InferenceSession(str(Path(model_dir) / name), opts, providers=["CPUExecutionProvider"])
        return True
    except Exception as error:
        print(f"ORT load FAILED: {error}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Share weights between split decoder models")
    parser.add_argument("model_dir")
    parser.add_argument("--suffix", default=None)
    parser.add_argument("--also-int8", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    dirs = [args.model_dir]
    if args.also_int8:
        dirs.append(args.model_dir.rstrip("/") + "-int8")
    try:
        for model_dir in dirs:
            share_external_models(model_dir, suffix=args.suffix)
            if args.verify and not verify_inference(model_dir, suffix=args.suffix):
                return 1
    except Exception as error:
        print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
