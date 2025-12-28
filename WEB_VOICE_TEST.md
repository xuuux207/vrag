# TechFlow 语音客服 Web 版测试指南

## 系统架构

```
浏览器端 (Azure Speech SDK)
    ↓ STT 语音识别
    ↓ WebSocket 发送文本
后端 (FastAPI + VoiceAssistant)
    ↓ RAG 决策 Agent
    ↓ 输入完整性 Agent
    ↓ LLM 流式生成
    ↓ WebSocket 返回文本
浏览器端 (Azure Speech SDK)
    ↓ TTS 语音合成播放
```

## 快速启动

### 1. 启动服务器

```bash
./start_web_server.sh
```

或手动启动：
```bash
uv run uvicorn src.server.api:app --host 0.0.0.0 --port 8000 --reload
```

### 2. 打开浏览器

访问：http://localhost:8000

### 3. 测试语音对话

1. 点击 **"开始通话"** 按钮
2. 授权麦克风访问
3. 直接说话（例如："你好"）
4. 等待 AI 回复并语音播放

## 测试场景

### 场景 1：简单问候
- **说**: "你好"
- **预期**: AI 回复 "你好呀，有什么可以帮你的吗？" 并播放语音
- **检查**:
  - 前端显示用户消息和 AI 回复
  - 听到语音播放
  - 控制台无错误

### 场景 2：知识库查询（触发 RAG）
- **说**: "FlowMind 是什么？"
- **预期**: AI 从知识库检索相关文档并回答
- **检查**:
  - 服务器日志显示 "RAG 判断: 需要检索"
  - 回复包含 FlowMind 的详细信息
  - 响应时间 2-3 秒内

### 场景 3：闲聊（不触发 RAG）
- **说**: "今天天气不错"
- **预期**: AI 简短回复，不检索知识库
- **检查**:
  - 服务器日志显示 "RAG 判断: 无需检索"
  - 快速响应（1-2 秒）

### 场景 4：连续对话
1. **说**: "FlowMind 是什么？"
2. **说**: "它有什么功能？"（测试上下文理解）
3. **预期**: AI 能理解 "它" 指的是 FlowMind

## 日志检查

### 正常日志流程

启动时：
```
2025-12-28 22:00:00 [INFO] src.server.api: 初始化 TechFlow AI 客服 API...
2025-12-28 22:00:01 [INFO] src.server.api: ✓ TTS 服务已初始化
2025-12-28 22:00:02 [INFO] src.server.api: ✓ RAG检索服务已初始化
2025-12-28 22:00:03 [INFO] src.server.api: ✓ API服务初始化完成！
```

WebSocket 连接：
```
2025-12-28 22:01:00 [INFO] src.server.websocket: WebSocket连接已建立
2025-12-28 22:01:00 [INFO] src.server.session_manager: 创建会话: xxx, 当前会话数: 1
```

处理消息：
```
2025-12-28 22:01:05 [INFO] src.server.websocket: 收到文本: 你好
2025-12-28 22:01:05 [INFO] src.pipeline.agents.input_completion_agent: 用户输入完整
2025-12-28 22:01:05 [INFO] src.pipeline.agents.rag_decision_agent: RAG 判断: 无需检索
2025-12-28 22:01:06 [INFO] src.pipeline.voice_assistant: 助手: 你好呀，有什么可以帮你的吗？
2025-12-28 22:01:06 [INFO] src.server.websocket: 文本处理完成
```

## 常见问题排查

### 问题 1: 麦克风无法访问
**症状**: 点击"开始通话"后无响应
**原因**: 浏览器未授权麦克风
**解决**:
- 检查浏览器地址栏是否有麦克风图标
- 允许麦克风访问
- 使用 Chrome/Edge 浏览器（Safari 可能有兼容问题）

### 问题 2: WebSocket 连接失败
**症状**: 控制台显示 "WebSocket 错误"
**原因**: 服务器未启动或端口被占用
**解决**:
```bash
# 检查服务器是否运行
curl http://localhost:8000/health

# 检查端口占用
lsof -i :8000
```

### 问题 3: Azure Speech SDK 错误
**症状**: 控制台显示 "无法启动" 或 "识别取消"
**原因**: Azure Speech Key/Region 配置错误
**解决**:
- 检查 `web/index.html` 第 109-110 行
- 确认 SPEECH_KEY 和 SPEECH_REGION 正确
- 中国区域应使用 `chinanorth3` 或类似

### 问题 4: AI 不回复
**症状**: 说话后无响应
**原因**: 后端处理错误
**解决**:
1. 检查服务器日志是否有错误
2. 确认 RAG/LLM 服务正常初始化
3. 查看是否有 "处理失败" 错误

### 问题 5: 事件循环错误（已修复）
**症状**: `RuntimeError: no running event loop`
**原因**: VoiceAssistant 回调从线程调用异步函数
**解决**: 已使用 `asyncio.run_coroutine_threadsafe` 修复

## 性能指标

- **首次响应时间**: 2-3 秒（STT + RAG + LLM）
- **后续响应**: 1-2 秒（无需 RAG）
- **语音播放**: 实时流式（TTS 生成后立即播放）
- **并发会话**: 支持最多 100 个同时在线

## 核心功能验证清单

- [ ] WebSocket 连接成功
- [ ] 麦克风识别工作正常
- [ ] RAG 决策 Agent 正确判断（有/无检索）
- [ ] 输入完整性 Agent 工作（可选测试）
- [ ] LLM 生成合理回复
- [ ] TTS 语音播放流畅
- [ ] 多轮对话上下文保持
- [ ] 错误处理和重连机制

## 对比本地版本

| 功能 | 本地版 (main.py) | Web 版 |
|------|-----------------|--------|
| STT | Azure SDK (Python) | Azure SDK (Browser) |
| TTS | Azure SDK (Python) | Azure SDK (Browser) |
| RAG 决策 | ✅ 完整 Agent | ✅ 完整 Agent |
| 输入完整性 | ✅ 完整 Agent | ✅ 完整 Agent |
| LLM 流式 | ✅ | ✅ |
| 上下文压缩 | ✅ | ✅ |
| 并发支持 | ❌ | ✅ 多会话 |
| 部署方式 | 本地运行 | Web 服务器 |

## 下一步优化（可选）

1. **流式响应**: LLM 生成时实时发送文本块
2. **分句 TTS**: 边生成边播放，降低延迟
3. **VAD 优化**: 实时打断和连续对话
4. **密钥安全**: 后端生成临时 token，避免暴露密钥
5. **会话持久化**: 保存对话历史到数据库
6. **负载均衡**: 多实例部署支持高并发

## 技术支持

如遇问题，请提供：
1. 浏览器控制台日志（F12 → Console）
2. 服务器日志（终端输出）
3. 具体操作步骤和错误信息
