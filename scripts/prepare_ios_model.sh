#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
CHECKPOINT="$ROOT/model/sharp_2572gikvuh.pt"
MLPACKAGE_OUT="$ROOT/out/SHARP.mlpackage"
MLPACKAGE_IOS="$ROOT/ios/OpenReshot/SHARP.mlpackage"

mkdir -p "$ROOT/model" "$ROOT/out" "$ROOT/ios/OpenReshot"

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

if [ ! -d "$MLPACKAGE_OUT" ]; then
  echo "[2/3] converting reconstruction model to Core ML. This can take a while..."
  "$PYTHON_BIN" "$ROOT/scripts/coreml_convert.py"
else
  echo "[2/3] Core ML package already exists."
fi

if [ ! -d "$MLPACKAGE_IOS" ]; then
  echo "[3/3] copying Core ML package into the iOS target..."
  cp -R "$MLPACKAGE_OUT" "$MLPACKAGE_IOS"
else
  echo "[3/3] iOS Core ML package already exists."
fi

echo
echo "Ready. Next:"
echo "  cd ios"
echo "  xcodegen generate"
echo "  open OpenReshot.xcodeproj"
