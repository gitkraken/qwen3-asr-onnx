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

from share_weights import graph_file_closure, share_external_models, transport_graph_files
from src.embeddings import load_embed_tokens
from src.tokenizer_assets import tokenizer_asset_names, validate_tokenizer_assets

REQUIRED_METADATA = ["config.json", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json"]


def _graph_names(suffix=""):
    tag = f".{suffix}" if suffix else ""
    return [f"encoder{tag}.onnx", f"decoder_init{tag}.onnx", f"decoder_step{tag}.onnx"]


def copy_file(src, dst, hardlink=False):
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if hardlink and not src.is_symlink():
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)


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
    with tarfile.open(target, "w:gz", compresslevel=1, dereference=True) as archive:
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
    archives = {
        variant: output.parent / f"{output.name}{'-int4' if variant else ''}.tar.gz" for variant in ("", "int4")
    }
    for target in [output, *archives.values()]:
        resolved_target = target.resolve()
        if source.is_relative_to(resolved_target) or resolved_target.is_relative_to(source):
            raise ValueError(f"Input directory overlaps publication destination: {target}")
    if output.is_symlink():
        raise ValueError("Output cannot be a symlink")
    for name in REQUIRED_METADATA + ["embed_tokens.bin"]:
        if not (source / name).is_file():
            raise ValueError(f"Required artifact missing: {name}")
    for name in REQUIRED_METADATA:
        with (source / name).open() as stream:
            metadata = json.load(stream)
        if not isinstance(metadata, dict):
            raise ValueError(f"Required metadata must be a JSON object: {name}")
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
    graph_files = transport_graph_files(
        source,
        (name for variant in variants for name in _graph_names(variant)),
        rewritten={"config.json", "embed_tokens.bin"},
    )
    for variant in variants:
        canonical = f"decoder_weights.{variant}.data" if variant else "decoder_weights.data"
        unaffected_graphs = ["encoder" + (f".{variant}" if variant else "") + ".onnx"]
        unaffected_graphs.extend(name for other in variants if other != variant for name in _graph_names(other))
        if canonical in graph_file_closure(source, unaffected_graphs):
            raise ValueError(f"ONNX transport destination conflict: {canonical}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".package-", dir=output.parent) as tmp:
        stage = Path(tmp)
        release = stage / output.name
        release.mkdir()
        metadata = {"config.json", "preprocessor_config.json"} | tokenizer_asset_names(source)
        names = set(metadata)
        names.update(graph_files)
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
        variant_files = {variant: graph_file_closure(release, _graph_names(variant)) for variant in variants}
        complete = shared | set().union(*variant_files.values())
        for path in release.rglob("*"):
            if path.is_file() and path.relative_to(release).as_posix() not in complete:
                path.unlink()
        if not verify_ort_load(release):
            raise ValueError("Release failed ONNX/ORT verification")
        load_embed_tokens(release)
        validate_tokenizer_assets(release)
        replacements = [(release, output)]
        for variant in variants:
            base = output.name + ("-int4" if variant else "")
            archive_path = stage / f"{base}.tar.gz"
            members = shared | variant_files[variant]
            _create_archive(release, archive_path, members, base)
            replacements.append((archive_path, archives[variant]))
        for variant in ("", "int4"):
            if variant not in variants:
                replacements.append((None, archives[variant]))
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
