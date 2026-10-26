import sys
from types import SimpleNamespace

import numpy as np
import onnx
import onnxruntime as ort
import torch
from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig
from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRAudioEncoder

import export
import export_encoder_native_fp16 as native_fp16
from tests.encoder_helpers import FRAME_LENGTHS


def test_native_fp16_main_exports_dynamic_fp32_io(tmp_path, monkeypatch, capsys):
    with torch.random.fork_rng():
        torch.manual_seed(4342)
        config = Qwen3ASRAudioEncoderConfig(
            d_model=32,
            encoder_attention_heads=4,
            encoder_layers=2,
            encoder_ffn_dim=64,
            downsample_hidden_size=4,
            output_dim=16,
            n_window=50,
            n_window_infer=800,
            initializer_range=0.2,
        )
        config._attn_implementation = "eager"
        model = SimpleNamespace(thinker=SimpleNamespace(audio_tower=Qwen3ASRAudioEncoder(config).eval()))

        def load_tiny_model(model_name, dtype):
            assert model_name == "fixture/tiny"
            assert dtype == torch.float32
            return model

        path = tmp_path / "encoder.fp16.onnx"
        monkeypatch.setattr(export, "load_model", load_tiny_model)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "export_encoder_native_fp16.py",
                "--model",
                "fixture/tiny",
                "--output",
                str(path),
                "--opset",
                "17",
                "--verify",
            ],
        )
        native_fp16.main()

    assert "ORT OK:" in capsys.readouterr().out
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    assert [(item.domain, item.version) for item in graph.opset_import] == [("", 17)]
    assert [(item.name, item.type.tensor_type.elem_type) for item in graph.graph.input] == [
        ("mel", onnx.TensorProto.FLOAT)
    ]
    assert [(item.name, item.type.tensor_type.elem_type) for item in graph.graph.output] == [
        ("audio_features", onnx.TensorProto.FLOAT)
    ]
    assert any(item.data_type == onnx.TensorProto.FLOAT16 for item in graph.graph.initializer)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    for frames in FRAME_LENGTHS:
        mel = np.random.default_rng(frames).standard_normal((1, 128, frames), dtype=np.float32)
        features = session.run(["audio_features"], {"mel": mel})[0]
        expected_tokens = (frames // 100) * 13 + ((frames % 100) + 7) // 8
        assert features.shape == (1, expected_tokens, 16)
        assert features.dtype == np.float32
        assert np.isfinite(features).all()
