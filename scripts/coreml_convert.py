"""Convert the OpenReshot reconstruction network to a Core ML fp16 .mlpackage.

The default path uses torch.export + iOS18 ML Program ops so Core ML can keep
ViT attention as scaled_dot_product_attention instead of materializing the
large matmul/softmax attention matrices produced by legacy TorchScript tracing.

Only the neural forward goes into Core ML. The NDC->metric unprojection,
covariance decomposition, and 3DGS rendering stay outside the model.
"""
from __future__ import annotations

import argparse
import shutil
import time
from collections import Counter
from pathlib import Path

import coremltools as ct
import coremltools.optimize as cto
import numpy as np
import torch

from sharp.models import PredictorParams, create_predictor

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WIDTH = 1536
DEFAULT_CHECKPOINT = ROOT / "model" / "sharp_2572gikvuh.pt"
DEFAULT_OUTPUT = ROOT / "out" / "SHARP.mlpackage"
DEFAULT_TRACED_OUTPUT = ROOT / "out" / "sharp_traced.pt"
OUTPUT_NAMES = ["mean", "scale", "quat", "color", "opacity"]


def log(message: str) -> None:
    print(message, flush=True)


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument(
        "--mode",
        choices=("sdpa", "trace"),
        default="sdpa",
        help="sdpa uses torch.export+iOS18 fused attention; trace matches the old iOS17 TorchScript path.",
    )
    parser.add_argument(
        "--skip-forward-sanity",
        action="store_true",
        help="Skip the pre-conversion PyTorch forward pass.",
    )
    parser.add_argument(
        "--save-traced",
        type=Path,
        default=DEFAULT_TRACED_OUTPUT,
        help="TorchScript path to save when --mode trace is used. Pass an empty string to skip.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove an existing output package before saving.",
    )
    parser.add_argument(
        "--weight-compression",
        choices=("none", "int4", "uint4", "palettize4", "palettize4-uniform"),
        default="none",
        help=(
            "Optional Core ML weight-only 4-bit compression. This reduces model storage/mmap size, "
            "but does not quantize activations or intermediate tensors."
        ),
    )
    parser.add_argument(
        "--compression-threshold",
        type=int,
        default=2048,
        help="Only compress weight tensors with more than this many elements.",
    )
    parser.add_argument(
        "--compression-granularity",
        choices=("per_channel", "per_block"),
        default="per_channel",
        help="Granularity for int4/uint4 linear weight quantization.",
    )
    parser.add_argument(
        "--compression-block-size",
        type=int,
        default=32,
        help="Block size for int4/uint4 per-block linear weight quantization.",
    )
    parser.add_argument(
        "--palettize-workers",
        type=int,
        default=1,
        help="Worker count for 4-bit k-means palettization.",
    )
    return parser.parse_args()


def load_model(checkpoint: Path) -> torch.nn.Module:
    log("[1/5] loading PyTorch model on CPU ...")
    t0 = time.time()
    model = create_predictor(PredictorParams())
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    log(f"  loaded in {time.time() - t0:.1f}s")
    return model


def make_example_inputs(width: int, device: torch.device = torch.device("cpu")):
    image = torch.rand(1, 3, width, width, device=device)
    disp = torch.tensor([1246.6742 / width], device=device)
    return image, disp


def run_forward_sanity(model: torch.nn.Module, width: int) -> None:
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    log(f"[2/5] forward sanity on {device} ...")
    t0 = time.time()
    model = model.to(device)
    image, disp = make_example_inputs(width, device=device)
    with torch.inference_mode():
        ref = model(image, disp)
    log(
        "  ok in %.1fs | outputs: %s"
        % (
            time.time() - t0,
            {
                name: tuple(getattr(ref, attr).shape)
                for name, attr in [
                    ("mean", "mean_vectors"),
                    ("scale", "singular_values"),
                    ("quat", "quaternions"),
                    ("color", "colors"),
                    ("opacity", "opacities"),
                ]
            },
        )
    )
    model.to("cpu")
    if device.type == "mps":
        torch.mps.empty_cache()


def export_sdpa_program(model: torch.nn.Module, width: int):
    log("[3/5] exporting with torch.export (keeps scaled_dot_product_attention) ...")
    t0 = time.time()
    image, disp = make_example_inputs(width)
    with torch.no_grad():
        exported = torch.export.export(model, (image, disp)).run_decompositions({})
    exported = strip_aten_aliases(exported)
    graph_text = str(exported.graph_module.graph)
    sdpa_count = graph_text.count("scaled_dot_product_attention")
    matmul_count = graph_text.count("matmul")
    softmax_count = graph_text.count("softmax")
    log(
        "  exported in %.1fs | dialect=%s | sdpa refs=%d, matmul refs=%d, softmax refs=%d"
        % (time.time() - t0, getattr(exported, "dialect", "unknown"), sdpa_count, matmul_count, softmax_count)
    )
    if sdpa_count == 0:
        raise RuntimeError("torch.export graph did not preserve scaled_dot_product_attention")
    return exported


def strip_aten_aliases(exported):
    """Remove identity alias nodes that coremltools cannot ingest."""
    graph = exported.graph_module.graph
    removed = 0
    for node in list(graph.nodes):
        if node.op == "call_function" and node.target == torch.ops.aten.alias.default:
            node.replace_all_uses_with(node.args[0])
            graph.erase_node(node)
            removed += 1
    if removed:
        graph.lint()
        exported.graph_module.recompile()
        log(f"  stripped aten.alias identity nodes: {removed}")
    return exported


def trace_legacy_program(model: torch.nn.Module, width: int, save_path: Path | None):
    log("[3/5] tracing with torch.jit.trace (legacy explicit attention) ...")
    t0 = time.time()
    image, disp = make_example_inputs(width)
    with torch.no_grad():
        traced = torch.jit.trace(model, (image, disp), check_trace=False)
    log(f"  traced in {time.time() - t0:.1f}s")
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        traced.save(str(save_path))
        log(f"  saved {display_path(save_path)}")
    return traced


def convert_to_coreml(program, mode: str, width: int):
    target = ct.target.iOS18 if mode == "sdpa" else ct.target.iOS17
    log(f"[4/5] converting to Core ML fp16 ({mode}, minimum_deployment_target={target}) ...")
    t0 = time.time()
    kwargs = dict(
        outputs=[ct.TensorType(name=name) for name in OUTPUT_NAMES],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=target,
        skip_model_load=True,
    )
    if mode == "trace":
        kwargs["inputs"] = [
            ct.TensorType(name="image", shape=(1, 3, width, width), dtype=np.float32),
            ct.TensorType(name="disparity_factor", shape=(1,), dtype=np.float32),
        ]
    mlmodel = ct.convert(program, **kwargs)
    mlmodel.short_description = f"OpenReshot SHARP Core ML reconstruction model ({mode})"
    mlmodel.user_defined_metadata.update(
        {
            "OpenReshotCoreMLMode": mode,
            "OpenReshotMinimumDeploymentTarget": "iOS18" if mode == "sdpa" else "iOS17",
            "OpenReshotAttention": "scaled_dot_product_attention" if mode == "sdpa" else "explicit_matmul_softmax",
            "OpenReshotInternalResolution": str(width),
        }
    )
    log(f"  converted in {time.time() - t0:.1f}s")
    return mlmodel


def mlprogram_op_counts(mlmodel) -> Counter[str]:
    spec = mlmodel.get_spec()
    if spec.WhichOneof("Type") != "mlProgram":
        return Counter()
    counts: Counter[str] = Counter()
    for function in spec.mlProgram.functions.values():
        for block in function.block_specializations.values():
            counts.update(op.type for op in block.operations)
    return counts


def validate_converted_spec(mlmodel, mode: str) -> None:
    spec = mlmodel.get_spec()
    counts = mlprogram_op_counts(mlmodel)
    log(f"  spec type: {spec.WhichOneof('Type')}")
    log(f"  output names: {[output.name for output in spec.description.output]}")
    if counts:
        log(f"  top ops: {counts.most_common(8)}")
    if mode == "sdpa" and counts.get("scaled_dot_product_attention", 0) == 0:
        raise RuntimeError("Core ML spec does not contain scaled_dot_product_attention")
    if mode == "sdpa":
        log(f"  scaled_dot_product_attention ops: {counts['scaled_dot_product_attention']}")


def compress_coreml_weights(mlmodel, args: argparse.Namespace):
    compression = args.weight_compression
    if compression == "none":
        mlmodel.user_defined_metadata["OpenReshotWeightCompression"] = "none"
        return mlmodel

    log(f"  applying Core ML weight compression: {compression} ...")
    t0 = time.time()

    if compression in {"int4", "uint4"}:
        config = cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpLinearQuantizerConfig(
                mode="linear_symmetric",
                dtype=compression,
                granularity=args.compression_granularity,
                block_size=args.compression_block_size,
                weight_threshold=args.compression_threshold,
            )
        )
        compressed = cto.coreml.linear_quantize_weights(mlmodel, config)
        if args.compression_granularity == "per_block":
            compression_label = f"{compression}_linear_symmetric_per_block{args.compression_block_size}"
        else:
            compression_label = f"{compression}_linear_symmetric_per_channel"
    else:
        mode = "uniform" if compression == "palettize4-uniform" else "kmeans"
        config = cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpPalettizerConfig(
                mode=mode,
                nbits=4,
                num_kmeans_workers=args.palettize_workers,
                weight_threshold=args.compression_threshold,
            )
        )
        compressed = cto.coreml.palettize_weights(mlmodel, config)
        compression_label = f"palettize4_{mode}"

    compressed.user_defined_metadata.update(mlmodel.user_defined_metadata)
    compressed.user_defined_metadata.update(
        {
            "OpenReshotWeightCompression": compression_label,
            "OpenReshotWeightCompressionTool": f"coremltools {ct.__version__}",
        }
    )
    counts = mlprogram_op_counts(compressed)
    compressed_ops = {
        name: counts.get(name, 0)
        for name in [
            "constexpr_blockwise_shift_scale",
            "constexpr_affine_dequantize",
            "constexpr_lut_to_dense",
        ]
    }
    log(f"  compressed in {time.time() - t0:.1f}s | compression ops: {compressed_ops}")
    return compressed


def save_model(mlmodel, output: Path, force: bool) -> None:
    log(f"[5/5] saving {display_path(output)} ...")
    if output.exists():
        if not force:
            raise FileExistsError(f"{output} already exists; pass --force to replace it")
        if output.is_dir():
            shutil.rmtree(output)
        else:
            output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output))
    log(f"DONE -> {display_path(output)}")


def main() -> None:
    args = parse_args()
    model = load_model(args.checkpoint)
    if args.skip_forward_sanity:
        log("[2/5] forward sanity skipped")
    else:
        run_forward_sanity(model, args.width)

    if args.mode == "sdpa":
        program = export_sdpa_program(model, args.width)
    else:
        save_traced = args.save_traced if str(args.save_traced) else None
        program = trace_legacy_program(model, args.width, save_traced)

    mlmodel = convert_to_coreml(program, args.mode, args.width)
    mlmodel = compress_coreml_weights(mlmodel, args)
    validate_converted_spec(mlmodel, args.mode)
    save_model(mlmodel, args.output, args.force)


if __name__ == "__main__":
    main()
