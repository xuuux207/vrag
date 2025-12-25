#!/bin/bash
# 监控双模型实验运行状态

echo "=========================================="
echo "Dual Model Experiment - Status Monitor"
echo "=========================================="
echo ""

cd ~/tts

# 检查进程
echo "📊 进程状态:"
if ps aux | grep -v grep | grep "test_03_v3_dual_model.py" > /dev/null; then
    echo "  ✓ 实验进程运行中"
    ps aux | grep -v grep | grep "test_03_v3_dual_model.py" | awk '{print "    PID:", $2, "  CPU:", $3"%", "  MEM:", $4"%"}'
else
    echo "  ✗ 实验进程未运行"
fi

echo ""
echo "🎯 GPU使用情况:"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader | head -2

echo ""
echo "📝 最新日志 (最后15行):"
echo "----------------------------------------"
tail -15 logs/exp3_dual_run_*.log 2>/dev/null | tail -15 || echo "  (暂无日志)"
echo "----------------------------------------"

echo ""
echo "📂 输出文件:"
ls -lht outputs/experiment3_dual_model_results_*.json 2>/dev/null | head -3 || echo "  (暂无输出文件)"

echo ""
echo "💡 提示:"
echo "  查看完整日志: tail -f logs/exp3_dual_run_*.log"
echo "  查看实验日志: tail -f logs/experiment3_dual_model_*.log"
echo "  停止实验: pkill -f test_03_v3_dual_model.py"
echo ""
