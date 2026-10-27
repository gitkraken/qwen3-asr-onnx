import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import onnx
from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.message import DecodeError


def storage_tensors(message, initializer=False):
    if isinstance(message, onnx.TensorProto):
        yield message, initializer
        return
    for field, value in message.ListFields():
        if field.type != FieldDescriptor.TYPE_MESSAGE:
            continue
        if field.is_repeated:
            for item in value:
                yield from storage_tensors(item, field.name == "initializer")
        else:
            yield from storage_tensors(value, field.name == "initializer")


def model_tensors(message):
    for tensor, _ in storage_tensors(message):
        yield tensor


def get_external_info(tensor):
    return {entry.key: entry.value for entry in tensor.external_data}


def external_tensor_range(root, tensor):
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
    if not model.HasField("graph") or not model.ir_version:
        raise ValueError(f"Not an ONNX graph: {path}")
    return {
        external_tensor_range(path.parent, tensor)[0]
        for tensor in model_tensors(model)
        if tensor.data_location == onnx.TensorProto.EXTERNAL
    }


def inventory_graphs(root, exclude=()):
    root = Path(root).resolve()
    graphs = {}
    exclude = set(exclude)
    for path in sorted(root.rglob("*.onnx")):
        name = path.relative_to(root).as_posix()
        if name in exclude:
            continue
        if not path.is_file():
            continue
        if path.stat().st_size >= onnx.checker.MAXIMUM_PROTOBUF:
            continue
        # Tensor storage can use this suffix; only decoded models establish graph roles.
        try:
            model = onnx.load(path, load_external_data=False)
        except DecodeError:
            continue
        if not model.HasField("graph") or not model.ir_version:
            continue
        try:
            graphs[name] = {
                external_tensor_range(path.parent, tensor)[0].relative_to(root).as_posix()
                for tensor in model_tensors(model)
                if tensor.data_location == onnx.TensorProto.EXTERNAL
            }
        except Exception as error:
            raise ValueError(
                f"Cannot inventory independent graph {path}; preserve it and repair its references "
                "or move abandoned staging outside the model directory before retrying."
            ) from error
    return graphs


def preflight_artifact_mutations(
    source, output, copies, *, generated=(), rewritten=(), removed=(), reads=(), existing=None, copied_assets=()
):
    destination_root = Path(output).absolute()
    source, output = Path(source).resolve(), destination_root.resolve()
    generated, rewritten, removed = set(generated), set(rewritten), set(removed)
    validate_generated_destinations(destination_root, generated)
    assets = set(copied_assets)
    files = {name for name in set(copies) | assets if (source / name).is_file()} if source != output else set()
    incoming = inventory_graphs(source, set(generated) - set(reads) if source == output else ())
    copied_graphs = files & incoming.keys()
    dependencies = set().union(*(incoming[name] for name in copied_graphs))
    files.update(dependencies)
    conflicts = (files & generated) | (dependencies & (rewritten | removed))
    if source == output:
        read_dependencies = set().union(*(incoming[name] for name in reads if name in incoming))
        conflicts.update(read_dependencies & (generated | rewritten | removed))
    if conflicts:
        raise ValueError(f"ONNX transport destination conflict: {', '.join(sorted(conflicts))}")
    if existing is None:
        existing = (
            incoming
            if source == output
            else (inventory_graphs(output, generated | copied_graphs) if output.is_dir() else {})
        )
    surviving = existing.keys() - copied_graphs - generated
    reserved = set(surviving) | set().union(*(existing[name] for name in surviving))
    conflicts = reserved & (generated | rewritten | removed)
    for name in files & reserved:
        original, retained = source / name, output / name
        if original.stat().st_size == retained.stat().st_size and file_digest(original) == file_digest(retained):
            files.remove(name)
        else:
            conflicts.add(name)
    if conflicts:
        raise ValueError(f"ONNX retained graph destination conflict: {', '.join(sorted(conflicts))}")
    validate_generated_destinations(destination_root, generated)
    for name in rewritten | removed | assets:
        if (output / name).is_dir():
            raise ValueError(f"Artifact destination is a directory: {output / name}")
    return files - assets


def graph_file_closure(root, names):
    root = Path(root).resolve()
    graphs = set(names)
    return graphs | _graph_dependencies(root, graphs)


def _graph_dependencies(root, names):
    return {path.relative_to(root).as_posix() for name in names for path in external_data_files(root / name)}


def transport_graph_files(root, graphs, generated=(), rewritten=()):
    root = Path(root).resolve()
    graphs = set(graphs)
    dependencies = _graph_dependencies(root, graphs)
    files = graphs | dependencies
    conflicts = (files & set(generated)) | (dependencies & set(rewritten))
    if conflicts:
        raise ValueError(f"ONNX transport destination conflict: {', '.join(sorted(conflicts))}")
    return files


def protect_retained_graphs(source, output, graphs, copies, *, generated=(), rewritten=(), removed=()):
    source, output = Path(source).resolve(), Path(output).resolve()
    reserved = graph_file_closure(output, graphs)
    copies = set(copies)
    conflicts = reserved & (set(generated) | set(rewritten) | set(removed))
    for name in copies & reserved:
        original, retained = source / name, output / name
        if original.stat().st_size == retained.stat().st_size and file_digest(original) == file_digest(retained):
            copies.remove(name)
        else:
            conflicts.add(name)
    if conflicts:
        raise ValueError(f"ONNX retained graph destination conflict: {', '.join(sorted(conflicts))}")
    return copies


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


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").digest()
