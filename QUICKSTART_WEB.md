# 快速开始 - Web API 服务器

## 启动服务

```bash
# 启动服务器
uv run python server.py
```

服务启动后访问：
- **文本版界面**: http://localhost:8000
- **语音版界面**: http://localhost:8000/voice.html
- **API文档**: http://localhost:8000/docs

## 两种交互模式

### 1. 文本版（/index.html）
- 传统聊天界面
- 文字输入输出
- 适合快速测试

### 2. 语音版（/voice.html）✨
- 实时语音交互界面
- WebSocket 双向通信
- 麦克风输入（可按 T 键测试文本）
- 语音输出（待集成 TTS）

## 快速测试

### 文本版测试
1. 打开 http://localhost:8000
2. 输入框输入"你好"
3. 点击发送，查看回复

### 语音版测试（文本模式）
1. 打开 http://localhost:8000/voice.html
2. 点击"开始对话"
3. 允许麦克风权限
4. **按键盘 T 键**
5. 输入测试消息："介绍一下FlowMind"
6. 查看 AI 回复

> 💡 当前语音版支持文本测试（按T键），完整语音功能需要集成 STT/TTS 服务。

## 测试接口

### 1. 健康检查
```bash
curl http://localhost:8000/health
```

### 2. 发送消息（新会话）
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "你好"}'
```

### 3. 发送消息（指定会话）
```bash
# 使用上一步返回的 session_id
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "your-session-id", "message": "介绍一下FlowMind"}'
```

### 4. 查看会话历史
```bash
curl http://localhost:8000/api/session/your-session-id/history
```

### 5. 删除会话
```bash
curl -X DELETE http://localhost:8000/api/session/your-session-id
```

## 前端使用

1. 浏览器打开 http://localhost:8000
2. 直接在输入框输入问题
3. 点击"发送"或按回车发送
4. 查看AI回复（带RAG检索标识）

## 开发模式

启用热重载（代码修改自动重启）：
```bash
uv run uvicorn src.server.api:app --reload --host 0.0.0.0 --port 8000
```

## 配置

确保 `.env` 文件包含以下配置：
- `AZURE_SEARCH_*` - RAG检索
- `QWEN_*` - LLM服务
- `EMBEDDING_*` - 向量化
- `RERANKING_*` - 重排序

详细部署文档见 [WEB_DEPLOYMENT.md](WEB_DEPLOYMENT.md)
