"""pytest 配置文件"""
import os
from dotenv import load_dotenv

# 在所有测试前加载 .env
load_dotenv()

# 检查必要的环境变量
def pytest_configure(config):
    """pytest 启动时检查"""
    qwen_token = os.getenv("QWEN_TOKEN")
    if not qwen_token:
        raise ValueError("❌ 缺少 QWEN_TOKEN 配置，请在 .env 中设置")
    print("✓ 环境配置正确")
