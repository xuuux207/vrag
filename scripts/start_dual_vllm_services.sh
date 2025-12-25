#!/bin/bash
# 启动双vLLM服务：8B用于总结，14B用于回复

set -e

echo "=========================================="
echo "Starting Dual vLLM Services"
echo "=========================================="
echo ""

# 停止现有服务
echo "🛑 停止现有vLLM服务..."
pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 3

# 激活conda环境
echo "🔧 激活conda环境..."
source ~/miniconda3/bin/activate
export PATH=~/miniconda3/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com

# 启动 Qwen3-8B (Port 8000, GPU 0) - 用于总结
echo ""
echo "🚀 启动 Qwen3-8B @ localhost:8000 (总结模型, GPU 0)..."
CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B \
    --served-model-name Qwen/Qwen3-8B \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 16384 \
    > ~/tts/logs/vllm_8b.log 2>&1 &

echo "  PID: $!"

# 等待8B服务启动
echo "  等待8B服务启动..."
for i in {1..30}; do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "  ✓ 8B服务启动成功"
        break
    fi
    sleep 2
done

# 启动 Qwen3-14B (Port 8001, GPU 1) - 用于回复
echo ""
echo "🚀 启动 Qwen3-14B @ localhost:8001 (回复模型, GPU 1)..."
CUDA_VISIBLE_DEVICES=1 nohup python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-14B \
    --served-model-name Qwen/Qwen3-14B \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8001 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 16384 \
    > ~/tts/logs/vllm_14b.log 2>&1 &

echo "  PID: $!"

# 等待14B服务启动
echo "  等待14B服务启动..."
for i in {1..30}; do
    if curl -s http://localhost:8001/v1/models > /dev/null 2>&1; then
        echo "  ✓ 14B服务启动成功"
        break
    fi
    sleep 2
done

echo ""
echo "=========================================="
echo "✅ 双vLLM服务启动完成"
echo "=========================================="
echo ""
echo "📋 服务列表:"
echo "  - Qwen3-8B:  http://localhost:8000 (总结)"
echo "  - Qwen3-14B: http://localhost:8001 (回复)"
echo ""
echo "📊 检查服务状态:"
curl -s http://localhost:8000/v1/models | python3 -c "import sys, json; data=json.load(sys.stdin); print(f\"  ✓ Port 8000: {data['data'][0]['id']}\")" 2>/dev/null || echo "  ✗ Port 8000 未响应"
curl -s http://localhost:8001/v1/models | python3 -c "import sys, json; data=json.load(sys.stdin); print(f\"  ✓ Port 8001: {data['data'][0]['id']}\")" 2>/dev/null || echo "  ✗ Port 8001 未响应"
echo ""
echo "📝 日志位置:"
echo "  - 8B日志:  ~/tts/logs/vllm_8b.log"
echo "  - 14B日志: ~/tts/logs/vllm_14b.log"
echo ""
