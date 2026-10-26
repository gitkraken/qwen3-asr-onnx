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
import math
import os
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


def model_tensors(message):
    if isinstance(message, onnx.TensorProto):
        yield message
        return
    for field, value in message.ListFields():
        if field.type != FieldDescriptor.TYPE_MESSAGE:
            continue
        if field.is_repeated:
            for item in value:
                yield from model_tensors(item)
        else:
            yield from model_tensors(value)


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


def _write_shared_data(models, root, path):
    index = {}
    with path.open("wb") as stream:
        for model in models:
            for tensor in model_tensors(model):
                if tensor.data_location != onnx.TensorProto.EXTERNAL and not tensor.raw_data:
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


def _live_data_files(root):
    live = set()
    for path in root.glob("*.onnx"):
        live.update(external_data_files(path))
    return live


def _publish_data(staged, target, root):
    if target.exists():
        identical = staged.stat().st_size == target.stat().st_size and _file_digest(staged) == _file_digest(target)
        if identical:
            return
        if target in _live_data_files(root):
            raise ValueError(
                "Canonical shared data is in use with incompatible bytes; rebuild in a clean output directory"
            )
    os.replace(staged, target)


def _cleanup_data(root, names, old_files, shared_name):
    candidates = old_files | {root / f"{name}.data" for name in names}
    live = _live_data_files(root)
    for path in candidates - live - {root / shared_name}:
        if path.exists():
            path.unlink()


def _validate_models(root, names):
    for name in names:
        external_data_files(root / name)
        onnx.checker.check_model(str(root / name))


def share_external_models(model_dir, suffix=None):
    root = Path(model_dir).resolve()
    names = _decoder_filenames(suffix)
    if not all((root / name).is_file() for name in names):
        raise FileNotFoundError(f"Both split decoder models are required in {root}")
    models = [onnx.load(root / name, load_external_data=False) for name in names]
    old_files = set().union(*(external_data_files(root / name) for name in names))
    shared_name = _shared_data_name(suffix)
    _validate_models(root, names)
    already_shared = old_files == {root / shared_name} and all(
        not tensor.raw_data for model in models for tensor in model_tensors(model)
    )
    if already_shared:
        _cleanup_data(root, names, old_files, shared_name)
        return True
    with tempfile.TemporaryDirectory(prefix=".share-", dir=root) as tmp:
        stage = Path(tmp)
        _write_shared_data(models, root, stage / shared_name)
        for model, name in zip(models, names):
            with open(stage / name, "wb") as stream:
                stream.write(model.SerializeToString())
                stream.flush()
                os.fsync(stream.fileno())
        _validate_models(stage, names)
        _publish_data(stage / shared_name, root / shared_name, root)
        for name in names:
            os.replace(stage / name, root / name)
        _validate_models(root, names)
        _cleanup_data(root, names, old_files, shared_name)
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
