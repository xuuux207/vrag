#!/usr/bin/env python3
"""
Web服务器启动脚本
使用 uvicorn 启动 FastAPI 应用
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    import uvicorn

    print("=" * 70)
    print("启动 TechFlow AI 客服 Web 服务...")
    print("=" * 70)
    print("\n访问地址:")
    print("  - Web 界面: http://localhost:8000")
    print("  - API 文档: http://localhost:8000/docs")
    print("  - 健康检查: http://localhost:8000/health")
    print("\n按 Ctrl+C 停止服务\n")
    print("=" * 70)

    uvicorn.run(
        "src.server.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # 生产环境关闭热重载
        log_level="info",
    )
