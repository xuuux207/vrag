#!/bin/bash
# 监控实验3 v3运行状态

echo "=========================================="
echo "Experiment 3 v3 - Status Monitor"
echo "=========================================="
echo ""

cd ~/tts

# 检查进程
echo "📊 进程状态:"
if ps aux | grep -v grep | grep "test_03_v3_server.py" > /dev/null; then
    echo "  ✓ 实验进程运行中"
    ps aux | grep -v grep | grep "test_03_v3_server.py" | awk '{print "    PID:", $2, "  CPU:", $3"%", "  MEM:", $4"%"}'
else
    echo "  ✗ 实验进程未运行"
fi

echo ""
echo "📝 最新日志 (最后20行):"
echo "----------------------------------------"
tail -20 logs/exp3_run_*.log 2>/dev/null | tail -20 || echo "  (暂无日志)"
echo "----------------------------------------"

echo ""
echo "📂 输出文件:"
ls -lht outputs/experiment3_v3_server_results_*.json 2>/dev/null | head -5 || echo "  (暂无输出文件)"

echo ""
echo "💡 提示:"
echo "  查看完整日志: tail -f logs/exp3_run_*.log"
echo "  查看实验日志: tail -f logs/experiment3_v3_*.log"
echo "  停止实验: pkill -f test_03_v3_server.py"
echo ""
