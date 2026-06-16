#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
CHECKPOINT="$ROOT/model/sharp_2572gikvuh.pt"
CONVERT_MODE="${OPENRESHOT_COREML_MODE:-sdpa}"
COREML_WIDTH="${OPENRESHOT_COREML_WIDTH:-1536}"
WEIGHT_COMPRESSION="${OPENRESHOT_WEIGHT_COMPRESSION:-none}"
COMPRESSION_GRANULARITY="${OPENRESHOT_COMPRESSION_GRANULARITY:-per_channel}"
COMPRESSION_BLOCK_SIZE="${OPENRESHOT_COMPRESSION_BLOCK_SIZE:-32}"
FORCE_REBUILD="${FORCE:-0}"

if ! [[ "$COREML_WIDTH" =~ ^[0-9]+$ ]] || [ "$COREML_WIDTH" -le 0 ]; then
  echo "OPENRESHOT_COREML_WIDTH must be a positive integer, got: $COREML_WIDTH"
  exit 1
fi

if [ -n "${OPENRESHOT_BASE_MLPACKAGE_OUT:-}" ]; then
  BASE_MLPACKAGE_OUT="$OPENRESHOT_BASE_MLPACKAGE_OUT"
elif [ "$COREML_WIDTH" = "1536" ]; then
  BASE_MLPACKAGE_OUT="$ROOT/out/SHARP-$CONVERT_MODE.mlpackage"
else
  BASE_MLPACKAGE_OUT="$ROOT/out/SHARP-$CONVERT_MODE-${COREML_WIDTH}.mlpackage"
fi

if [ "$WEIGHT_COMPRESSION" = "none" ]; then
  MLPACKAGE_OUT="${OPENRESHOT_MLPACKAGE_OUT:-$BASE_MLPACKAGE_OUT}"
elif [ -n "${OPENRESHOT_MLPACKAGE_OUT:-}" ]; then
  MLPACKAGE_OUT="$OPENRESHOT_MLPACKAGE_OUT"
elif [ "$COREML_WIDTH" = "1536" ]; then
  MLPACKAGE_OUT="$ROOT/out/SHARP-$CONVERT_MODE-$WEIGHT_COMPRESSION.mlpackage"
else
  MLPACKAGE_OUT="$ROOT/out/SHARP-$CONVERT_MODE-${COREML_WIDTH}-$WEIGHT_COMPRESSION.mlpackage"
fi

mkdir -p "$ROOT/model" "$ROOT/out"

if ! "$PYTHON_BIN" -c "import coremltools" >/dev/null 2>&1; then
  echo "coremltools is missing."
  echo "Install iOS conversion dependencies first:"
  echo "  $PYTHON_BIN -m pip install -r \"$ROOT/requirements/requirements-ios.txt\""
  exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
  echo "[1/3] downloading reconstruction checkpoint..."
  curl -L "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt" -o "$CHECKPOINT"
else
  echo "[1/3] checkpoint already exists."
fi

if [ "$FORCE_REBUILD" = "1" ] || [ ! -d "$BASE_MLPACKAGE_OUT" ]; then
  echo "[2/3] converting reconstruction model to Core ML ($CONVERT_MODE, ${COREML_WIDTH}x${COREML_WIDTH}). This can take a while..."
  CONVERT_ARGS=(--mode "$CONVERT_MODE" --width "$COREML_WIDTH" --output "$BASE_MLPACKAGE_OUT")
  if [ "$FORCE_REBUILD" = "1" ]; then
    CONVERT_ARGS+=(--force)
  fi
  "$PYTHON_BIN" "$ROOT/scripts/coreml_convert.py" "${CONVERT_ARGS[@]}"
else
  echo "[2/3] Core ML base package already exists."
fi

if [ "$WEIGHT_COMPRESSION" = "none" ]; then
  echo "[3/3] weight compression disabled."
else
  if [ "$FORCE_REBUILD" = "1" ] || [ ! -d "$MLPACKAGE_OUT" ]; then
    echo "[3/3] applying Core ML weight compression ($WEIGHT_COMPRESSION). This can take a while..."
    COMPRESS_ARGS=(
      --input "$BASE_MLPACKAGE_OUT"
      --output "$MLPACKAGE_OUT"
      --weight-compression "$WEIGHT_COMPRESSION"
      --compression-granularity "$COMPRESSION_GRANULARITY"
      --compression-block-size "$COMPRESSION_BLOCK_SIZE"
    )
    if [ "$FORCE_REBUILD" = "1" ]; then
      COMPRESS_ARGS+=(--force)
    fi
    "$PYTHON_BIN" "$ROOT/scripts/coreml_compress.py" "${COMPRESS_ARGS[@]}"
  else
    echo "[3/3] compressed Core ML package already exists."
  fi
fi

echo
echo "Ready. Core ML package:"
echo "  $MLPACKAGE_OUT"
echo
echo "App builds stay model-empty; users download the model inside the app."
echo "Next:"
echo "  cd ios"
echo "  xcodegen generate"
echo "  open OpenReshot.xcodeproj"
