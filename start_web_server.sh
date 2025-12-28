#!/bin/bash
# TechFlow 语音客服 Web 服务器启动脚本

echo "======================================"
echo "TechFlow 语音客服 Web 服务器"
echo "======================================"
echo ""

# 检查依赖
if ! command -v uv &> /dev/null; then
    echo "❌ 错误: 未安装 uv"
    echo "请运行: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# 设置环境变量（如需要）
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 启动服务器
echo "🚀 启动 FastAPI 服务器..."
echo "📍 Web 界面: http://localhost:8000"
echo "📍 API 文档: http://localhost:8000/docs"
echo "📍 WebSocket: ws://localhost:8000/ws"
echo ""
echo "按 Ctrl+C 停止服务器"
echo ""

uv run uvicorn src.server.api:app --host 0.0.0.0 --port 8000 --reload
