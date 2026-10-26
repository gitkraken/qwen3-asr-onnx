#!/usr/bin/env python3
"""Validate complete FP32/int4 releases before replacing published artifacts."""

import argparse
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

import onnx
import onnxruntime as ort

from share_weights import external_data_files, share_external_models
from src.embeddings import load_embed_tokens

REQUIRED_METADATA = ["config.json", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json"]
OPTIONAL_METADATA = ["added_tokens.json", "vocab.json", "merges.txt", "special_tokens_map.json"]
SHARED_FILES = REQUIRED_METADATA + OPTIONAL_METADATA


def format_size(size_bytes):
    if size_bytes >= 1e9:
        return f"{size_bytes / 1e9:.2f} GB"
    if size_bytes >= 1e6:
        return f"{size_bytes / 1e6:.1f} MB"
    return f"{size_bytes / 1e3:.1f} KB"


def _graph_names(suffix=""):
    tag = f".{suffix}" if suffix else ""
    return [f"encoder{tag}.onnx", f"decoder_init{tag}.onnx", f"decoder_step{tag}.onnx"]


def _dependencies(root, names):
    files = set(names)
    for name in names:
        for path in external_data_files(root / name):
            files.add(path.relative_to(root).as_posix())
    return files


def copy_file(src, dst, hardlink=False):
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if hardlink:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)
    return dst.stat().st_size


def verify_ort_load(output_dir):
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    try:
        for path in sorted(Path(output_dir).glob("*.onnx")):
            onnx.checker.check_model(str(path))
            ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        return True
    except Exception as error:
        print(f"ORT verification failed: {error}")
        return False


def _create_archive(root, target, names, archive_root):
    with tarfile.open(target, "w:gz", compresslevel=1) as archive:
        for name in sorted(names):
            archive.add(root / name, arcname=f"{archive_root}/{name}", recursive=False)


def _publish_artifacts(replacements):
    backup = Path(tempfile.mkdtemp(prefix=".release-backup-", dir=replacements[0][1].parent))
    saved = []
    installed = []
    clean = False
    try:
        for index, (source, target) in enumerate(replacements):
            if target.exists():
                old = backup / str(index)
                os.replace(target, old)
                saved.append((old, target))
            if source is not None:
                os.replace(source, target)
                installed.append(target)
        clean = True
    except BaseException:
        try:
            for target in reversed(installed):
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            for old, target in reversed(saved):
                os.replace(old, target)
            clean = True
        except Exception as recovery_error:
            raise RuntimeError(
                f"Publication rollback failed; previous artifacts remain in {backup}"
            ) from recovery_error
        raise
    finally:
        if clean:
            shutil.rmtree(backup)


def _package(input_dir, output_dir, test_wavs_src=None, hardlink=False):
    source = Path(input_dir).resolve()
    output = Path(output_dir).absolute()
    resolved_output = output.resolve()
    if source.is_relative_to(resolved_output) or resolved_output.is_relative_to(source):
        raise ValueError("Input and output directories must be separate")
    if output.is_symlink():
        raise ValueError("Output cannot be a symlink")
    for name in REQUIRED_METADATA + ["embed_tokens.bin"]:
        if not (source / name).is_file():
            raise ValueError(f"Required artifact missing: {name}")
    variants = [variant for variant in ("", "int4") if any((source / name).exists() for name in _graph_names(variant))]
    if not variants:
        raise ValueError("No model variants found")
    for variant in variants:
        for name in _graph_names(variant):
            if not (source / name).is_file():
                raise ValueError(f"Incomplete {variant or 'FP32'} variant: {name}")
    embed = load_embed_tokens(source, output_dtype="float16")
    sidecar = source / "embed_tokens.fp16.bin"
    if sidecar.exists():
        candidate = load_embed_tokens(source, filename=sidecar.name, dtype="float16", output_dtype="float16")
        if not (candidate == embed).all():
            raise ValueError("FP16 embedding sidecar disagrees with canonical data")
        embed = candidate
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".package-", dir=output.parent) as tmp:
        stage = Path(tmp)
        release = stage / output.name
        release.mkdir()
        metadata = set(REQUIRED_METADATA + [name for name in OPTIONAL_METADATA if (source / name).is_file()])
        names = set(metadata)
        for variant in variants:
            names.update(_dependencies(source, _graph_names(variant)))
        for name in sorted(names):
            copy_file(source / name, release / name, hardlink=hardlink and name in metadata)
        embed.tofile(release / "embed_tokens.bin")
        config = json.loads((release / "config.json").read_text())
        config["embed_tokens_dtype"] = "float16"
        config["embed_tokens_shape"] = list(embed.shape)
        config_path = release / "config.json"
        config_path.unlink()
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        for variant in variants:
            share_external_models(release, suffix=variant or None)
        shared = metadata | {"embed_tokens.bin"}
        if test_wavs_src is not None:
            copy_file(test_wavs_src, release / "test_wavs" / "0.wav")
            shared.add("test_wavs/0.wav")
        complete = set(shared)
        for variant in variants:
            complete.update(_dependencies(release, _graph_names(variant)))
        for path in release.rglob("*"):
            if path.is_file() and path.relative_to(release).as_posix() not in complete:
                path.unlink()
        if not verify_ort_load(release):
            raise ValueError("Release failed ONNX/ORT verification")
        load_embed_tokens(release)
        replacements = [(release, output)]
        for variant in variants:
            base = output.name + ("-int4" if variant else "")
            archive_path = stage / f"{base}.tar.gz"
            members = shared | _dependencies(release, _graph_names(variant))
            _create_archive(release, archive_path, members, base)
            replacements.append((archive_path, output.parent / archive_path.name))
        for variant in ("", "int4"):
            if variant not in variants:
                base = output.name + ("-int4" if variant else "")
                replacements.append((None, output.parent / f"{base}.tar.gz"))
        _publish_artifacts(replacements)
    print(f"Release directory: {output}")
    return True


def package(input_dir, output_dir, test_wavs_src=None, hardlink=False):
    try:
        return _package(input_dir, output_dir, test_wavs_src=test_wavs_src, hardlink=hardlink)
    except Exception as error:
        print(f"ERROR: {error}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Package complete FP32 + int4 model releases")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", choices=["0.6b", "1.7b"])
    group.add_argument("--input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--test-wavs", default=None)
    parser.add_argument("--hardlink", action="store_true")
    args = parser.parse_args()
    input_dir = args.input or f"output/qwen3-asr-{args.model}"
    return 0 if package(input_dir, args.output, test_wavs_src=args.test_wavs, hardlink=args.hardlink) else 1


if __name__ == "__main__":
    raise SystemExit(main())
