#!/bin/bash
# 运行实验3 双模型版本 - 服务器版本
# 8B用于总结，14B用于回复

set -e

echo "=========================================="
echo "Experiment 3 Dual Model Version"
echo "8B for Summary, 14B for Response"
echo "=========================================="
echo ""

# 设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 激活conda环境
source ~/miniconda3/bin/activate
export PATH=~/miniconda3/bin:$PATH

# 进入项目目录
cd ~/tts

# 检查vLLM服务状态
echo "🔍 检查vLLM服务..."
echo ""

if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    model_8b=$(curl -s http://localhost:8000/v1/models | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['data'][0]['id'])")
    echo "✓ Port 8000: $model_8b"
else
    echo "❌ Port 8000 服务未运行！"
    echo "请先运行: ./scripts/start_dual_vllm_services.sh"
    exit 1
fi

if curl -s http://localhost:8001/v1/models > /dev/null 2>&1; then
    model_14b=$(curl -s http://localhost:8001/v1/models | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['data'][0]['id'])")
    echo "✓ Port 8001: $model_14b"
else
    echo "❌ Port 8001 服务未运行！"
    echo "请先运行: ./scripts/start_dual_vllm_services.sh"
    exit 1
fi

# 检查.env配置
echo ""
echo "🔍 检查配置文件..."
if [ ! -f .env ]; then
    echo "❌ .env文件不存在！"
    echo "请创建.env文件并配置Embedding/Reranking API"
    exit 1
fi

# 显示配置
echo ""
echo "📋 实验配置:"
echo "  Summary LLM: $model_8b @ localhost:8000"
echo "  Response LLM: $model_14b @ localhost:8001"
echo "  Embedding: $(grep EMBEDDING_MODEL .env | cut -d'=' -f2)"
echo "  测试用例: 5个长文本场景"
echo "  方法数量: 4个"
echo ""

# 创建输出目录
mkdir -p outputs
mkdir -p logs

# 运行实验
echo "=========================================="
echo "🚀 开始运行实验..."
echo "=========================================="
echo ""

python experiments/test_03_v3_dual_model.py 2>&1 | tee logs/experiment3_dual_model_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "=========================================="
echo "✅ 实验完成！"
echo "=========================================="
echo ""
echo "结果文件: outputs/experiment3_dual_model_results_*.json"
echo "日志文件: logs/experiment3_dual_model_*.log"
echo ""
echo "分析结果:"
echo "  python experiments/analyze_exp3_v3_results.py"
echo ""
