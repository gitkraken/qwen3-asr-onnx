"""
quantize_nbits.py — Apply MatMulNBits (int4 or int8) block quantization to Qwen3-ASR decoder.

Uses ORT's MatMulNBitsQuantizer with RTN (default) or GPTQ algorithm.
Produces a MatMulNBits ONNX graph — weights are stored as packed uint8 with per-group scales.

Usage (RTN, in-place — adds int4 files alongside FP32):
    uv run python quantize_nbits.py \
        --input output/qwen3-asr-1.7b \
        --output output/qwen3-asr-1.7b \
        --bits 4 --block-size 64 --accuracy-level 4

Usage (GPTQ decoder_init + RTN decoder_step, split runs):
    uv run python quantize_nbits.py \
        --input output/qwen3-asr-1.7b \
        --output output/qwen3-asr-1.7b \
        --bits 4 --block-size 64 --accuracy-level 4 \
        --algo gptq --calib-data calibration_cache/1.7b_gptq_init.npz \
        --decoders decoder_init

    uv run python quantize_nbits.py \
        --input output/qwen3-asr-1.7b \
        --output output/qwen3-asr-1.7b \
        --bits 4 --block-size 64 --accuracy-level 4 \
        --algo rtn --decoders decoder_step

Output files use suffixed names (e.g. decoder_init.int4.onnx) so multiple
quantization variants can coexist in one directory alongside the FP32 source.
When --output differs from --input, all non-quantized files (encoder, tokenizer,
config, FP32 decoders) are copied to the output directory.
"""

import argparse
import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization.calibrate import CalibrationDataReader
from onnxruntime.quantization.matmul_nbits_quantizer import (
    GPTQWeightOnlyQuantConfig,
    MatMulNBitsQuantizer,
    RTNWeightOnlyQuantConfig,
)
from onnxruntime.quantization.quant_utils import QuantFormat

from share_weights import (
    copy_artifact_entry,
    external_data_files,
    get_external_info,
    model_tensors,
    protect_retained_graphs,
    transport_graph_files,
    validate_generated_destinations,
    write_artifact_json,
)
from src.tokenizer_assets import (
    TOKENIZER_FILES,
    copy_tokenizer_assets,
    tokenizer_asset_names,
    validate_tokenizer_assets,
)


class NpzCalibrationReader(CalibrationDataReader):
    """Feeds pre-collected input dicts from a numpy .npz file to GPTQ."""

    def __init__(self, path: str):
        data = np.load(path)
        n = int(data["_n_samples"])
        self._data: list[dict] = []
        for i in range(n):
            prefix = f"{i}_"
            sample_keys = [k[len(prefix) :] for k in data.files if k.startswith(prefix)]
            self._data.append({k: data[f"{prefix}{k}"] for k in sample_keys})
        self._idx = 0

    def get_next(self) -> dict | None:
        if self._idx >= len(self._data):
            return None
        item = self._data[self._idx]
        self._idx += 1
        return item

    def __len__(self):
        return len(self._data)


def validate_quantization_request(bits, block_size, algo, calib_data_path, nodes_to_exclude=None):
    if algo not in {"rtn", "gptq"}:
        raise ValueError(f"Unsupported quantization algorithm: {algo}")
    if bits != 4:
        raise ValueError(f"{algo.upper()} supports only 4-bit MatMulNBits in the supported ORT environment; got {bits}")
    if block_size < 16 or block_size & (block_size - 1):
        raise ValueError("MatMulNBits block size must be a power of two and at least 16")
    if algo == "gptq":
        if calib_data_path is None:
            raise ValueError("--calib-data required for --algo gptq")
        if nodes_to_exclude:
            raise ValueError("GPTQ node exclusions are unsupported by the supported ORT algorithm; use RTN")


def _quantized_nodes(graph):
    for node in graph.node:
        if node.domain == "com.microsoft" and node.op_type == "MatMulNBits":
            yield node
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                yield from _quantized_nodes(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for subgraph in attribute.graphs:
                    yield from _quantized_nodes(subgraph)


def verify_quantized_precision(model, bits, block_size):
    nodes = list(_quantized_nodes(model.graph))
    if not nodes:
        raise ValueError("No MatMulNBits operators were produced; refusing to label an unquantized model")
    for node in nodes:
        attributes = {attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in node.attribute}
        actual_bits = attributes.get("bits")
        actual_block_size = attributes.get("block_size")
        if actual_bits != bits or actual_block_size != block_size:
            raise ValueError(
                f"Quantized operator {node.name!r} has bits={actual_bits}, block_size={actual_block_size}; "
                f"requested bits={bits}, block_size={block_size}. No output was written."
            )


@contextmanager
def _quantization_input(input_path: str, algo: str) -> Iterator[tuple[onnx.ModelProto | str, str | None]]:
    if algo != "gptq":
        yield onnx.load(input_path), None
        return
    source = Path(input_path).resolve()
    dependencies = sorted(external_data_files(source))
    with tempfile.TemporaryDirectory(prefix="qwen-gptq-") as temporary:
        stage = Path(temporary)
        locations = {path: f"weights/{index}.data" for index, path in enumerate(dependencies)}
        for path, location in locations.items():
            target = stage / location
            target.parent.mkdir(exist_ok=True)
            # ONNX rejects external tensor files with multiple hard links.
            shutil.copy2(path, target)
        model = onnx.load(source, load_external_data=False)
        external_bytes = 0
        for tensor in model_tensors(model):
            if tensor.data_location != onnx.TensorProto.EXTERNAL:
                continue
            info = get_external_info(tensor)
            dependency = (source.parent / info["location"]).resolve()
            external_bytes += int(info.get("length", str(dependency.stat().st_size - int(info.get("offset", "0")))))
            for entry in tensor.external_data:
                if entry.key == "location":
                    entry.value = locations[dependency]
        if model.ByteSize() + external_bytes < onnx.checker.MAXIMUM_PROTOBUF:
            # GPTQ runs smaller graphs from bytes, which cannot resolve adjacent data files.
            onnx.external_data_helper.load_external_data_for_model(model, str(stage))
        staged_model = stage / "model.onnx"
        staged_model.write_bytes(model.SerializeToString())
        yield str(staged_model), str(stage)


def quantize_decoder(
    input_path: str,
    output_path: str,
    bits: int,
    block_size: int,
    accuracy_level: int | None,
    algo: str,
    calib_data_path: str | None,
    nodes_to_exclude: list[str] | None = None,
) -> None:
    validate_quantization_request(bits, block_size, algo, calib_data_path, nodes_to_exclude)
    print(f"  Loading {os.path.basename(input_path)} ...")
    if algo == "gptq":
        assert calib_data_path is not None
        algo_config = GPTQWeightOnlyQuantConfig(
            calibration_data_reader=NpzCalibrationReader(calib_data_path),
            block_size=block_size,
            quant_format=QuantFormat.QOperator,
        )
    else:
        algo_config = RTNWeightOnlyQuantConfig(quant_format=QuantFormat.QOperator)

    with _quantization_input(input_path, algo) as (model_arg, external_root):
        quantizer = MatMulNBitsQuantizer(
            model=model_arg,
            bits=bits,
            block_size=block_size,
            is_symmetric=False,
            accuracy_level=accuracy_level,
            algo_config=algo_config,
            nodes_to_exclude=nodes_to_exclude or None,
        )
        quantizer.process()
        model = quantizer.model.model
        verify_quantized_precision(model, bits, block_size)

        if not any(opset.domain == "com.microsoft" for opset in model.opset_import):
            opset = model.opset_import.add()
            opset.domain = "com.microsoft"
            opset.version = 1

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        if external_root is not None:
            onnx.external_data_helper.load_external_data_for_model(model, external_root)
        quantizer.model.save_model_to_file(output_path, use_external_data_format=True)
        print(f"  Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input model directory (FP32 ONNX)")
    parser.add_argument("--output", required=True, help="Output model directory")
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        choices=[4, 8],
        help="Requested bits (RTN/GPTQ support 4; unsupported requests fail before output)",
    )
    parser.add_argument("--block-size", type=int, default=64, help="Block size (group size)")
    parser.add_argument(
        "--accuracy-level",
        type=int,
        default=None,
        choices=[0, 1, 2, 3, 4],
        help="MatMulNBits accuracy level (0=default fp32 accum; 1-4 may use VNNI/AVX-512 paths)",
    )
    parser.add_argument(
        "--algo",
        default="rtn",
        choices=["rtn", "gptq"],
        help="Quantization algorithm: rtn (no calibration) or gptq (requires --calib-data)",
    )
    parser.add_argument(
        "--calib-data",
        default=None,
        help="Path to .npz file with calibration inputs (required for --algo gptq)",
    )
    parser.add_argument(
        "--decoders",
        nargs="+",
        default=["decoder_init", "decoder_step"],
        help="Model file stems to quantize (default: decoder_init decoder_step)",
    )
    parser.add_argument(
        "--nodes-to-exclude",
        nargs="+",
        default=None,
        help="MatMul node names to exclude from quantization (keep at FP32)",
    )
    args = parser.parse_args()

    validate_quantization_request(args.bits, args.block_size, args.algo, args.calib_data, args.nodes_to_exclude)
    for name in args.decoders:
        if not name or name in {".", ".."} or any(character in name for character in "/\\:"):
            raise ValueError(f"Unsafe model name: {name}")
        path = os.path.join(args.input, f"{name}.onnx")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required decoder not found: {path}")

    tokenizer_files = tokenizer_asset_names(args.input)
    if tokenizer_files:
        validate_tokenizer_assets(args.input)
    if tokenizer_asset_names(args.output):
        validate_tokenizer_assets(args.output)

    bits_suffix = f".int{args.bits}"
    generated_graphs = {f"{name}{bits_suffix}.onnx" for name in args.decoders}
    retained_graphs = {
        path.name for path in Path(args.output).glob(f"*{bits_suffix}.onnx") if path.is_file()
    } - generated_graphs
    excluded_graphs = (
        generated_graphs | retained_graphs | {f"{name}{bits_suffix}.onnx" for name in ("decoder_init", "decoder_step")}
    )
    excluded_files = excluded_graphs | {f"{name}.data" for name in excluded_graphs}
    previous_quantization = {}
    prior_config = Path(args.output) / "config.json"
    if prior_config.is_file():
        prior = json.loads(prior_config.read_text())
        prior_quantization = prior.get("quantization", {})
        if isinstance(prior_quantization, dict):
            previous_quantization = {
                name: tag for name, tag in prior_quantization.items() if f"{name}{bits_suffix}.onnx" in retained_graphs
            }
    files = {
        fname
        for fname in os.listdir(args.input)
        if fname not in TOKENIZER_FILES
        and fname not in excluded_files
        and os.path.isfile(os.path.join(args.input, fname))
    }
    generated = generated_graphs | {f"{name}.data" for name in generated_graphs}
    files.update(
        transport_graph_files(
            args.input, (name for name in files if name.endswith(".onnx")), generated, {"config.json"}
        )
    )
    files.difference_update(TOKENIZER_FILES)
    files = protect_retained_graphs(
        args.input,
        args.output,
        retained_graphs,
        files,
        generated=generated,
        rewritten={"config.json", *tokenizer_files},
        removed=TOKENIZER_FILES - tokenizer_files if tokenizer_files else (),
    )
    validate_generated_destinations(args.output, generated)
    os.makedirs(args.output, exist_ok=True)

    # Quantize each decoder file
    quantized_names = set()
    for name in args.decoders:
        src = os.path.join(args.input, f"{name}.onnx")
        dst = os.path.join(args.output, f"{name}{bits_suffix}.onnx")
        quantize_decoder(
            src, dst, args.bits, args.block_size, args.accuracy_level, args.algo, args.calib_data, args.nodes_to_exclude
        )
        quantized_names.add(name)

    # Copy non-quantized files from input to output (encoder, tokenizer, config, etc.)
    # Skip if input == output (in-place quantization).
    if os.path.realpath(args.input) != os.path.realpath(args.output):
        for fname in sorted(files):
            copy_artifact_entry(args.input, args.output, fname)
            print(f"  Copied {fname}")

    if tokenizer_files:
        copy_tokenizer_assets(args.input, args.output)

    if tokenizer_asset_names(args.output):
        validate_tokenizer_assets(args.output)

    # Update config.json quantization field (merge with existing for split-decoder runs)
    config_path = os.path.join(args.output, "config.json")
    if os.path.exists(config_path) and quantized_names:
        with open(config_path) as f:
            cfg = json.load(f)
        al_tag = f"_al{args.accuracy_level}" if args.accuracy_level is not None else ""
        tag = f"int{args.bits}_{args.algo}_block{args.block_size}{al_tag}"

        quant_info = cfg.get("quantization", {})
        if isinstance(quant_info, str):
            quant_info = {}  # migrate from old scalar format
        for graph in retained_graphs:
            quant_info.pop(graph.removesuffix(f"{bits_suffix}.onnx"), None)
        quant_info.update(previous_quantization)
        for name in quantized_names:
            quant_info[name] = tag
        cfg["quantization"] = quant_info

        write_artifact_json(args.output, "config.json", cfg)
        print(f"  Updated config.json: quantization={json.dumps(quant_info)}")

    # Report sizes
    in_mb = (
        sum(
            os.path.getsize(os.path.join(args.input, f))
            for f in os.listdir(args.input)
            if os.path.isfile(os.path.join(args.input, f))
        )
        / 1024**2
    )
    out_mb = (
        sum(
            os.path.getsize(os.path.join(args.output, f))
            for f in os.listdir(args.output)
            if os.path.isfile(os.path.join(args.output, f))
        )
        / 1024**2
    )
    print("\nDone.")
    print(f"  Input:  {in_mb:.1f} MB")
    print(f"  Output: {out_mb:.1f} MB")
    print(f"  Ratio:  {out_mb / in_mb * 100:.1f}%")


if __name__ == "__main__":
    main()
