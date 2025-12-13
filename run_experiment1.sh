#!/bin/bash

# 实验1运行脚本
# 使用流式输出测试模型对比

# 默认并发数
WORKERS=${1:-8}

echo "=========================================="
echo "实验1：模型对比实验"
echo "=========================================="
echo ""
echo "测试配置："
echo "- 模型: qwen3-8b, qwen3-14b, qwen3-32b"
echo "- 场景: 6 个测试场景"
echo "- 模式: 有/无 RAG"
echo "- 总测试数: 36"
echo "- 并发数: $WORKERS"
echo ""
echo "输出指标："
echo "- 首 token 延迟"
echo "- Token 生成速度"
echo "- 总延迟"
echo "- 回答质量评分"
echo ""
echo "日志文件: outputs/experiment1_run.log"
echo "结果文件: outputs/experiment1_results_*.json"
echo "报告文件: outputs/experiment1_report_*.md"
echo ""
echo "=========================================="
echo ""

# 清空旧日志
rm -f outputs/experiment1_run.log

# 运行实验（无缓冲输出，指定并发数）
PYTHONUNBUFFERED=1 uv run python experiments/test_01_model_comparison.py --workers $WORKERS 2>&1 | tee outputs/experiment1_run.log

echo ""
echo "=========================================="
echo "实验完成！"
echo "=========================================="
echo ""
echo "查看结果："
echo "- 日志: cat outputs/experiment1_run.log"
echo "- 结果: ls -lt outputs/experiment1_results_*.json | head -1"
echo "- 报告: ls -lt outputs/experiment1_report_*.md | head -1"

