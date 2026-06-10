#!/bin/bash
# Launch the OpenReshoot local comparison viewer.
#
# 1) Install web prototype dependencies:
#       pip install -r requirements-web.txt
# 2) Set your Gemini key first (only needed for the "重构" / AI step):
#       export GEMINI_API_KEY="你的key"
# 3) Then run this script:
#       bash run.sh
# 4) Open http://127.0.0.1:8765/splat/ in your browser.
#
# Keep this terminal open — the server runs here (so it stays alive and can see your key).

set -e
cd "$(dirname "$0")"

# free the port if an old server is still running
pkill -f "serve_viewer.py" 2>/dev/null || true
sleep 0.5

if [ -z "$GEMINI_API_KEY" ]; then
  echo "⚠️  GEMINI_API_KEY 未设置 —— 3D 查看与上传可用，但点「重构」会报错。"
  echo "    先执行: export GEMINI_API_KEY=\"你的key\"  再重跑本脚本即可启用 AI 重构。"
fi

export PYTORCH_ENABLE_MPS_FALLBACK=1
exec ./.venv/bin/python ./scripts/serve_viewer.py
