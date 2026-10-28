#!/usr/bin/env python3
"""
Convert FP32 ONNX decoder models to FP16 for reduced memory bandwidth.

Uses onnxconverter_common.float16_converter for proper graph-level conversion
(updates both weights AND all intermediate tensor types to float16).

ORT CPU EP will internally cast to FP32 for compute but benefits from reduced
memory bandwidth on weight loads.

Usage:
    uv run python convert_fp16.py \
        --input output/qwen3-asr-0.6b \
        --output output/trial-fp16-dec
"""

import argparse
import json
import os

import onnx
from onnxconverter_common import convert_float_to_float16

from share_weights import share_external_models
from src.artifact_metadata import MODEL_METADATA_FILES
from src.onnx_artifacts import (
    copy_artifact_entry,
    preflight_artifact_mutations,
    write_artifact_json,
)
from src.tokenizer_assets import (
    TOKENIZER_FILES,
    copy_tokenizer_assets,
    tokenizer_asset_names,
    validate_tokenizer_assets,
    validate_tokenizer_destination,
)


def convert_model_fp16(input_path: str, output_path: str) -> None:
    """Convert a single ONNX model to FP16 using graph-level conversion."""
    print(f"  Loading {os.path.basename(input_path)}...")
    model = onnx.load(input_path, load_external_data=True)

    print("    Converting graph to FP16...")
    model_fp16 = convert_float_to_float16(
        model,
        keep_io_types=True,  # Keep inputs/outputs as float32 for compatibility
    )

    # Save with external data
    onnx.save(
        model_fp16,
        output_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=os.path.basename(output_path) + ".data",
        size_threshold=1024,
    )

    in_size = os.path.getsize(input_path)
    data_in = input_path + ".data"
    if os.path.exists(data_in):
        in_size += os.path.getsize(data_in)

    out_size = os.path.getsize(output_path)
    data_out = output_path + ".data"
    if os.path.exists(data_out):
        out_size += os.path.getsize(data_out)

    print(f"    {in_size / 1e6:.1f} MB -> {out_size / 1e6:.1f} MB ({out_size / in_size:.1%})")


def main():
    parser = argparse.ArgumentParser(description="Convert ONNX decoder to FP16")
    parser.add_argument("--input", required=True, help="Input directory with FP32 ONNX files")
    parser.add_argument("--output", required=True, help="Output directory for FP16 files")
    parser.add_argument(
        "--files",
        nargs="+",
        default=["decoder_init.onnx", "decoder_step.onnx"],
        help="ONNX files to convert (default: decoder_init.onnx decoder_step.onnx)",
    )
    parser.add_argument(
        "--no-share-weights",
        action="store_true",
        help="Skip weight sharing for split decoder after conversion",
    )
    args = parser.parse_args()
    if tokenizer_asset_names(args.input):
        validate_tokenizer_assets(args.input)
    validate_tokenizer_destination(args.output)

    converted = {name for name in args.files if os.path.isfile(os.path.join(args.input, name))}
    generated = converted | {f"{name}.data" for name in converted}
    share_pair = {"decoder_init.onnx", "decoder_step.onnx"} <= converted and not args.no_share_weights
    if share_pair:
        generated.add("decoder_weights.data")
    files = {"embed_tokens.bin", *MODEL_METADATA_FILES}
    if os.path.isfile(os.path.join(args.input, "encoder.onnx")) and "encoder.onnx" not in converted:
        files.add("encoder.onnx")
    tokenizer_files = tokenizer_asset_names(args.input)
    files.difference_update(tokenizer_files)
    files = preflight_artifact_mutations(
        args.input,
        args.output,
        files,
        generated=generated,
        rewritten={"config.json"},
        copied_assets=tokenizer_files,
        removed=TOKENIZER_FILES - tokenizer_files if tokenizer_files else (),
        reads=converted,
    )
    os.makedirs(args.output, exist_ok=True)

    print("Converting ONNX files to FP16...")
    for filename in args.files:
        input_path = os.path.join(args.input, filename)
        if not os.path.exists(input_path):
            print(f"  Skipping {filename} (not found)")
            continue
        output_path = os.path.join(args.output, filename)
        convert_model_fp16(input_path, output_path)

    if share_pair:
        print("\nSharing FP16 decoder weights...")
        share_external_models(args.output)

    # Copy non-converted files
    print("\nCopying supporting files...")
    for filename in sorted(files):
        src = os.path.join(args.input, filename)
        if os.path.exists(src):
            copy_artifact_entry(args.input, args.output, filename)
            print(f"  Copied {filename}")

    if tokenizer_asset_names(args.input):
        copy_tokenizer_assets(args.input, args.output)

    # Update config
    config_path = os.path.join(args.output, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        config["decoder_dtype"] = "float16"
        write_artifact_json(args.output, "config.json", config)
        print("  Updated config.json with decoder_dtype=float16")

    print(f"\nConversion complete. Output: {args.output}")


if __name__ == "__main__":
    main()
