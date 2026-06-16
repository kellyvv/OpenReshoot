"""Apply optional 4-bit Core ML weight compression to an existing mlpackage.

This is intended for A/B testing storage and runtime behavior without rerunning
the full PyTorch -> Core ML conversion.
"""
from __future__ import annotations

import argparse
import shutil
import time
from collections import Counter
from pathlib import Path

import coremltools as ct
import coremltools.optimize as cto

ROOT = Path(__file__).resolve().parents[1]


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
    parser.add_argument("--input", type=Path, default=ROOT / "out" / "SHARP.mlpackage")
    parser.add_argument("--output", type=Path, default=ROOT / "out" / "SHARP-int4.mlpackage")
    parser.add_argument(
        "--weight-compression",
        choices=("int4", "uint4", "palettize4", "palettize4-uniform"),
        default="int4",
    )
    parser.add_argument("--compression-threshold", type=int, default=2048)
    parser.add_argument(
        "--compression-granularity",
        choices=("per_channel", "per_block"),
        default="per_channel",
    )
    parser.add_argument("--compression-block-size", type=int, default=32)
    parser.add_argument("--palettize-workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def mlprogram_op_counts(mlmodel) -> Counter[str]:
    spec = mlmodel.get_spec()
    if spec.WhichOneof("Type") != "mlProgram":
        return Counter()
    counts: Counter[str] = Counter()
    for function in spec.mlProgram.functions.values():
        for block in function.block_specializations.values():
            counts.update(op.type for op in block.operations)
    return counts


def package_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for file in path.rglob("*"):
        if file.is_file():
            total += file.stat().st_size
    return total


def format_bytes(byte_count: int) -> str:
    return f"{byte_count / 1024 / 1024:.1f}MB"


def compress_weights(mlmodel, args: argparse.Namespace):
    compression = args.weight_compression
    log(f"applying Core ML weight compression: {compression}")
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
    log(f"compressed in {time.time() - t0:.1f}s | compression ops: {compressed_ops}")
    return compressed


def save_model(mlmodel, output: Path, force: bool) -> None:
    if output.exists():
        if not force:
            raise FileExistsError(f"{output} already exists; pass --force to replace it")
        if output.is_dir():
            shutil.rmtree(output)
        else:
            output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)
    log(f"saving {display_path(output)}")
    mlmodel.save(str(output))


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    log(f"loading {display_path(args.input)} ({format_bytes(package_size(args.input))})")
    mlmodel = ct.models.MLModel(str(args.input), skip_model_load=True)
    counts = mlprogram_op_counts(mlmodel)
    log(f"source spec={mlmodel.get_spec().WhichOneof('Type')} | top ops={counts.most_common(8)}")

    compressed = compress_weights(mlmodel, args)
    save_model(compressed, args.output, args.force)
    log(f"DONE -> {display_path(args.output)} ({format_bytes(package_size(args.output))})")


if __name__ == "__main__":
    main()
