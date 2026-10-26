# Maintaining the GitKraken fork

## Ownership and provenance

The fork retains the history, Apache-2.0 license and attribution of
[andrewleech/qwen3-asr-onnx](https://github.com/andrewleech/qwen3-asr-onnx).
The audited starting commit is `4aee2ae0c355c7bc33bd5c09befec9e9eabd4b50`.
The consolidated defects and original reproduction scripts are in
[upstream issue 3](https://github.com/andrewleech/qwen3-asr-onnx/issues/3);
embedding shape compatibility also addresses
[upstream issue 1](https://github.com/andrewleech/qwen3-asr-onnx/issues/1).

This repository owns exportable wrappers and the standalone conversion,
quantization, validation and packaging tools. Kepler owns its orchestration,
layout-v2 prefill contract, current q4/q4f16 recipe, independent corpus/gates,
manifest, licensing objects and release uploader. Keep those protections when
updating the fork pin. Do not rewrite provenance of older exports to name this
fork. No improvement to transcription accuracy is implied by a feature parity fix.
The separate int4 sensitivity report in upstream issue 2 remains unclaimed.

## Supported environment and checks

Use Python 3.12.13 and uv 0.11.20. `pyproject.toml` declares the runtime and test
dependencies; `uv.lock` fixes their complete resolution. Numerical dependencies
initially match Kepler's reviewed export environment: torch 2.14.1,
transformers 4.57.6, qwen-asr 0.0.6, ONNX 1.23.2 and ORT 1.31.0.
The standalone FP16 converter uses pinned onnxconverter-common 1.16.0.
Dataset evaluation additionally needs the platform's FFmpeg libraries for
`datasets[audio]`/TorchCodec. Do not infer compatibility with another ORT or
checkpoint version from constructor signatures.

```sh
uv sync --locked --python 3.12.13
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy .
uv run --locked pytest -q --strict-markers
```

The default suite uses small official Qwen architectures with seeded weights,
small ONNX graphs, synthetic embeddings and an attributed chat-template fixture.
It runs without checkpoint downloads. CI runs the same suite with Hugging Face
offline mode enabled on the standard `macos-15` ARM64 CPU runner, whose architecture
is listed in [GitHub's runner documentation](https://docs.github.com/en/actions/how-tos/write-workflows/choose-where-workflows-run/choose-the-runner-for-a-job).
The job records and asserts its architecture and CPU execution. Unexpected Python
warnings fail the suite. The two pinned TorchScript exporter deprecations are
filtered because the legacy exporter preserves the requested opset and dynamic
contracts; the new default fails those tested cases. The pinned qwen-asr dependency
nagisa also emits invalid-escape SyntaxWarnings on first Python 3.12 compilation;
that specific dependency warning is filtered. This choice avoids
Linux's default CUDA wheel dependency stack and matches the measured Apple Silicon
environment. The universal lock contains Linux-specific CUDA dependencies; it does
not establish numerical validation or a lightweight CPU installation on Linux. Legacy full-export tests and the explicit checkpoint suite
carry the `integration` marker and are excluded by default. Missing full-model
fixtures or skipped integration cases do not establish successful validation.

Update dependencies intentionally, regenerate the lock with `uv lock`, rerun the
numerical and artifact checks, then regenerate the pip runtime lock:

```sh
uv export --locked --no-dev --format requirements-txt --output-file requirements.txt
```

## Precision and artifacts

| Algorithm | Bits | Exclusions | Supported behavior in ORT 1.31.0 |
| --- | --- | --- | --- |
| RTN | 4 | yes | Quantize, check every MatMulNBits operator, then label output |
| GPTQ | 4 | no | Quantize with calibration, check every operator, then label output |
| RTN or GPTQ | 8 | any | Reject before output creation |
| GPTQ | 4 | requested | Reject before output creation |

An all-excluded/no-quantized-node model is rejected instead of receiving an int4
label. GPTQ receives an isolated model and its external-data closure, never the
exporter's adjacent configuration. External data is copied, because ONNX rejects
hardlinks. Smaller models are materialized inline for ORT's byte-based calibration
sessions; larger models keep the external-data path. Staging can temporarily
require the source weights' size in additional disk space. Input configuration
and weights stay untouched on success or failure. This matrix does not change Kepler's own explicit 4-bit recipe.

Embedding metadata declares `embed_tokens_shape` and `embed_tokens_dtype`.
Readers validate dimensions, dtype, byte count and finite values. Legacy exports
may derive shape from `decoder.vocab_size` and `decoder.hidden_size`; absent dtype
means the historical FP32 format. An explicit unknown dtype or conflicting shape
is an error. FP16 input is preserved when repackaged; FP32 conversion must remain
finite. The raw binary is separate from ONNX session creation and needs its own
validation.

Converters preserve available configuration and preprocessing metadata from the
shared model-metadata inventory; graph-only conversions remain supported.
Converters, quantizers and packages transport each ONNX graph with its external-data
closure. They reject dependencies that would be replaced by a generated graph or
rewritten metadata before writing output. Decoder canonical data may be reused by
its owning decoders; an encoder or another variant cannot claim that destination.
Copied assets and rewritten configuration replace directory entries so an existing
link cannot change retained bytes.

Packages require complete graph sets plus their actual external-data references,
configuration, tokenizer and embeddings. Inline, per-decoder and shared weights
are alternatives. The release and archives are staged and validated before
replacing a prior usable release. Every publication destination must be separate from the source. Existing emitted
archives must be regular, recognizable model archives with the expected root and
required artifacts; inspection never extracts members or follows archive links.
An archive for an inactive variant is preserved and causes a preflight error, so
a successful release cannot leave an ambiguous stale variant beside it. Choose a
clean output name or move that file aside explicitly before retrying. The shared tokenizer-asset contract
preserves embedded/JSON/Jinja templates across conversion and archiving, removes
stale destination template assets when the source has tokenizer assets, and requires a locally renderable ASR prompt
before a package can be published. An archive or validation failure is a failed
command. Standalone archives include the ONNX external-data closure and replace a
prior tar only after a complete staged write. Tokenizer validation uses built-in
classes; local asset metadata cannot authorize Python modules. Kepler continues
using its own manifest-driven publisher.

## Shared-weight recovery

Sharing publishes the canonical `decoder_weights[.<suffix>].data` only after
preparing and validating its data and graph references. Original data remains
available while each graph is replaced. Required shape/control tensors and
operator attributes remain inline, and staged CPU ORT sessions must load before
publication. The supported q4 and q4f16 CPU paths are covered by real quantized
fixtures; missing kernels on another target need an explicit reviewed exception. Those replacements are individually
atomic; the filesystem does not provide a multi-file transaction. A retry must
validate the real graphs and bytes before declaring an already-shared result.
Any existing canonical file with incompatible bytes requires an explicit rebuild
error rather than being overwritten, even when no live graph refers to it.
Canonical symlink entries are rejected before graph publication.

Successful conventional decoder exports retain the two graph files and canonical
shared data file. Unreferenced conventional per-decoder data is cleaned on retry after checking
nested graph references. Sharing stages in a sibling directory so abrupt process
exit cannot contaminate the model inventory or leave directories in the canonical
triplet handed to Kepler. The model directory must have a writable parent on the
same filesystem for staging and atomic replacement. A terminated process can leave that sibling behind;
only remove it after establishing ownership. Historical unmarked staging inside
a model directory is preserved and inventoried like other foreign content;
invalid references produce an actionable error before publication. Arbitrary data
dependencies are retained because a reference does not establish exclusive
ownership; unreferenced data with custom original names may remain in the working directory;
keep it until its ownership is known. Packaging includes only the verified file
closure. Do not remove another graph's data merely because the current pair no
longer references it. Use Kepler's output lock, stage fingerprints and invalidation
when sharing is called by its pipeline; standalone callers must serialize writes
to a model directory.

## Full checkpoints and release evidence

Use these original, immutable checkpoints:

| Size | Repository | Revision |
| --- | --- | --- |
| 0.6B | Qwen/Qwen3-ASR-0.6B | 5eb144179a02acc5e5ba31e748d22b0cf3e303b0 |
| 1.7B | Qwen/Qwen3-ASR-1.7B | 7278e1e70fe206f11671096ffdd38061171dd6e5 |

Run the explicit checkpoint encoder suite separately from default CI:

```sh
uv run --locked pytest -m integration tests/full/test_checkpoint_encoder.py -s
```

Run one model at a time when another export is using memory. The suite must
compare the official native encoder, wrapper and dynamic ONNX at very short
inputs, 49/50/51, 74/75/76, 99/100/101, 799/800/801 and longer controls. The same
ONNX graph must execute lengths other than its trace example. Long-audio native
attention requires the official window mask; eager attention alone ignores the
FlashAttention chunk boundaries. Record finite values, shape/token count,
absolute and relative errors, runtime versions and checkpoint identities.

The validator uses the official native encoder independently of the exported
wrapper, checks shapes and finite values, enforces the existing absolute 1e-4
encoder threshold directly between native and ONNX as well as both intermediate comparisons, and fails on unacceptable transcript divergence or missing
sessions. Wrapper-to-ONNX parity remains a separate check. Prompts use the official
checkpoint chat template and tokenizer for roles, context and forced language.
Check both original checkpoint tokenizers, including system=8948/user=872 and the
former IDs' ` Current`/` time` decoding, rather than maintaining role IDs in code.

After the branch is committed and reviewed, run the real CLI/fault regressions
and both checkpoint checks again. Recheck the changed tree if live evidence leads
to another fix. Kepler's release process then rebuilds both sizes and evaluates
its expanded corpus under all four prompt forms, retains the hard fidelity/WER
gates, and tests the measured candidate bytes in an isolated application before
publishing. Keep previous content-addressed revisions for rollback. A CPU
checkpoint check is not evidence for WebGPU behavior.

## Focused upstream proposals

Prepare small branches from the audited upstream base, grouping tests with the
behavior they prove:

1. `fix/short-encoder-native-padding`: encoder masks, independent native reference
   and eager/dynamic-ONNX regression tests (issue 3 section 1).
2. `fix/nbits-effective-precision`: supported request matrix and real CLI precision
   checks (section 2).
3. `fix/embedding-and-package-contract`: dtype/shape readers and writer, complete
   staged packaging and archive failure checks (sections 3/5 and issue 1).
4. `fix/independent-validation`: native reference, finite/shape/tolerance verdicts
   and nonzero CLI exits (section 4).
5. `fix/template-derived-prompts`: official template construction and all
   calibration/evaluation callers (section 6).
6. `fix/recoverable-shared-weights`: staged canonical data, graph replacement,
   conservative cleanup and failure/retry tests (section 7).

Resolve shared helper dependencies explicitly when preparing each proposal.
Each proposed branch needs its own committed-tree review, relevant numerical
and artifact checks, and an accurate report of tested environments. Keep
GitKraken maintenance/publishing policy separate from reusable upstream fixes.
Do not merge upstream or the fork's main branch as part of preparing a proposal.
