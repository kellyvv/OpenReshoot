#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
FAILURES=0
WARNINGS=0

pass() {
  printf "[OK] %s\n" "$1"
}

warn() {
  WARNINGS=$((WARNINGS + 1))
  printf "[WARN] %s\n" "$1"
}

fail() {
  FAILURES=$((FAILURES + 1))
  printf "[FAIL] %s\n" "$1"
}

check_command() {
  if command -v "$1" >/dev/null 2>&1; then
    pass "$2"
  else
    fail "$3"
  fi
}

check_python_import() {
  if "$PYTHON_BIN" -c "import $1" >/dev/null 2>&1; then
    pass "$2"
  else
    fail "$3"
  fi
}

check_python_import_optional() {
  if "$PYTHON_BIN" -c "import $1" >/dev/null 2>&1; then
    pass "$2"
  else
    warn "$3"
  fi
}

echo "OpenReshot setup check"
echo "Repo: $ROOT"
echo

if [ -f "$ROOT/README.md" ] && [ -d "$ROOT/ios/OpenReshot" ]; then
  pass "repository layout looks correct"
else
  fail "run this script from a complete OpenReshot checkout"
fi

if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  pass "python available: $($PYTHON_BIN --version 2>&1)"
else
  fail "python not found; set PYTHON=/path/to/python or install Python"
fi

check_command "curl" "curl available" "curl is required to download the model checkpoint"
check_command "xcodegen" "XcodeGen available" "install XcodeGen: brew install xcodegen"
check_command "xcodebuild" "xcodebuild available" "install Xcode from the App Store or Apple Developer"

if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  check_python_import "torch" "torch import works" "install base dependencies: pip install -r requirements/requirements.txt"
  check_python_import "coremltools" "coremltools import works" "install iOS conversion dependencies: pip install -r requirements/requirements-ios.txt"
  check_python_import_optional "google.genai" "google-genai import works" "optional web prototype dependency missing: pip install -r requirements/requirements-web.txt"
fi

if [ -f "$ROOT/model/sharp_2572gikvuh.pt" ]; then
  pass "checkpoint exists: model/sharp_2572gikvuh.pt"
else
  warn "checkpoint missing; run scripts/prepare_ios_model.sh"
fi

if [ -d "$ROOT/out/SHARP.mlpackage" ]; then
  pass "converted Core ML model exists: out/SHARP.mlpackage"
else
  warn "converted Core ML model missing; run scripts/prepare_ios_model.sh"
fi

if [ -d "$ROOT/ios/OpenReshot/SHARP.mlpackage" ]; then
  warn "local iOS model package exists but is not included by project.yml: ios/OpenReshot/SHARP.mlpackage"
else
  pass "iOS app target is model-empty; users download the model in app"
fi

if [ -d "$ROOT/ios/OpenReshot.xcodeproj" ]; then
  pass "Xcode project exists: ios/OpenReshot.xcodeproj"
else
  warn "Xcode project missing; run: cd ios && xcodegen generate"
fi

echo
if [ "$FAILURES" -eq 0 ]; then
  if [ "$WARNINGS" -eq 0 ]; then
    echo "Ready."
  else
    echo "Ready with $WARNINGS warning(s)."
  fi
  exit 0
else
  echo "Blocked by $FAILURES failure(s), with $WARNINGS warning(s)."
  exit 1
fi
