#!/bin/bash
# 实验3快速运行脚本

set -e

echo "========================================"
echo "实验3：长时间语音输入处理"
echo "========================================"
echo ""

# 检查vLLM服务
echo "1. 检查vLLM服务状态..."
if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "✓ vLLM服务运行中"
else
    echo "✗ vLLM服务未运行"
    echo ""
    echo "请先启动vLLM服务："
    echo "  bash scripts/start_local_services.sh"
    exit 1
fi

# 检查环境变量
echo ""
echo "2. 检查环境配置..."
if [ -f .env ]; then
    echo "✓ .env 文件存在"

    if grep -q "EMBEDDING_TOKEN" .env && grep -q "RERANK_TOKEN" .env; then
        echo "✓ API tokens 已配置"
    else
        echo "⚠ 警告: .env 中缺少 EMBEDDING_TOKEN 或 RERANK_TOKEN"
    fi
else
    echo "✗ .env 文件不存在"
    echo "请复制 .env.example 并配置token"
    exit 1
fi

# 创建输出目录
echo ""
echo "3. 准备输出目录..."
mkdir -p outputs logs
echo "✓ 输出目录已创建"

# 运行测试
echo ""
echo "4. 运行实验3测试..."
echo "========================================"
echo ""

uv run python experiments/test_03_long_audio_rag.py

echo ""
echo "========================================"
echo "实验3完成！"
echo ""
echo "查看结果："
echo "  - JSON: outputs/experiment3_long_audio_results_*.json"
echo "  - 报告: outputs/experiment3_long_audio_report_*.md"
echo ""
echo "查看文档："
echo "  - 设计: docs/5. 实验3-长语音理解实验设计.md"
echo "  - 指南: docs/6. 实验3-使用指南.md"
echo "  - 总结: docs/7. 实验3-准备总结.md"
echo "========================================"
