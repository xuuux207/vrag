"""
语音助手主流程编排
整合 STT + RAG + LLM + TTS + 上下文管理
"""

import logging
import re
import time
import uuid
from typing import List, Dict, Optional, Callable
from threading import Event, Thread, Lock, Condition
from queue import Queue, Empty

logger = logging.getLogger(__name__)


class VoiceAssistant:
    """语音助手主控制器"""

    def __init__(
        self,
        stt_service,
        tts_service,
        rag_searcher,
        llm_service,
        context_manager,
        system_prompt: Optional[str] = None,
    ):
        """
        初始化语音助手

        Args:
            stt_service: STT服务
            tts_service: TTS服务
            rag_searcher: RAG检索器
            llm_service: LLM服务
            context_manager: 上下文管理器
            system_prompt: 系统提示词
        """
        self.stt = stt_service
        self.tts = tts_service
        self.rag = rag_searcher
        self.llm = llm_service
        self.context_mgr = context_manager

        self.system_prompt = system_prompt or self._default_system_prompt()

        # 对话状态
        self.messages: List[Dict[str, str]] = []
        self.is_running = False
        self.is_processing = False

        # 打断控制
        self.interrupt_requested = False  # 打断请求标志
        self.interrupted_response = None  # 被打断的不完整回复
        self.abort_event = Event()  # 中断信号：用于立即停止LLM生成和TTS合成

        # TTS任务队列管理（使用session ID避免频繁启停线程）
        self.tts_queue = Queue()
        self.tts_worker_thread: Optional[Thread] = None
        self.current_session_id = str(uuid.uuid4())  # 当前会话ID
        self.session_lock = Lock()  # 保护session_id
        self.is_tts_playing = False  # 当前是否正在播放

        # 临时变量（用于处理单轮对话）
        self._current_user_text = None
        self._current_assistant_text = ""
        self._recognition_complete_event = Event()

        # 回调函数
        self.on_user_speech: Optional[Callable[[str], None]] = None
        self.on_assistant_response: Optional[Callable[[str], None]] = None
        self.on_rag_retrieved: Optional[Callable[[Dict], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None

    def _default_system_prompt(self) -> str:
        """默认系统提示词"""
        return """
你是一个专业的智能客服助手，具备以下特点：
1. 友好、耐心、专业
2. 基于提供的知识库内容准确回答问题
3. 如果知识库中没有相关信息，诚实告知用户
4. 回答简洁明了，避免冗长
5. 对于直接问答，直接给出答案
""".strip()

    def start(self):
        """启动语音助手"""
        if self.is_running:
            logger.warning("语音助手已在运行")
            return

        logger.info("启动语音助手...")
        self.is_running = True

        # 启动持久TTS worker线程
        self._start_tts_worker()

        # 启动STT连续识别
        try:
            self.stt.start_continuous_recognition(
                on_recognizing=self._on_stt_recognizing,
                on_recognized=self._on_stt_recognized,
                on_session_started=self._on_stt_session_started,
                on_session_stopped=self._on_stt_session_stopped,
                on_canceled=self._on_stt_canceled,
                on_speech_started=self._on_speech_started,  # VAD检测到语音开始
            )
        except Exception as e:
            logger.error(f"启动STT失败: {str(e)}")
            self.is_running = False
            if self.on_error:
                self.on_error(f"启动失败: {str(e)}")
            raise

    def stop(self):
        """停止语音助手"""
        if not self.is_running:
            return

        logger.info("停止语音助手...")
        self.is_running = False

        try:
            self.stt.stop_continuous_recognition()
        except Exception as e:
            logger.error(f"停止STT失败: {str(e)}")

        # 停止TTS worker
        self._stop_tts_worker()

    def _start_tts_worker(self):
        """启动持久TTS工作线程"""
        if self.tts_worker_thread and self.tts_worker_thread.is_alive():
            logger.debug("TTS worker已在运行")
            return

        self.tts_worker_thread = Thread(target=self._tts_worker_loop, daemon=True)
        self.tts_worker_thread.start()
        logger.info("TTS worker线程已启动")

    def _stop_tts_worker(self):
        """停止TTS工作线程"""
        if not self.tts_worker_thread:
            return

        # 发送停止信号
        self.tts_queue.put(None)
        logger.info("TTS worker线程停止信号已发送")

    def _tts_worker_loop(self):
        """
        TTS工作线程主循环（持久运行）

        使用session ID管理任务：
        - 每个任务带有session_id
        - 播放前检查session_id是否匹配当前session
        - 打断时只需要更新session_id，无需重启线程
        - 检查abort_event，打断时立即停止合成
        """
        logger.info("TTS worker循环开始")

        while self.is_running:
            try:
                # 从队列获取 (session_id, index, sentence)
                item = self.tts_queue.get(timeout=0.5)

                # None表示结束信号
                if item is None:
                    logger.debug("TTS worker收到停止信号")
                    break

                session_id, index, sentence = item

                # 检查session是否匹配（打断后session_id会变化）
                with self.session_lock:
                    if session_id != self.current_session_id:
                        logger.debug(f"跳过旧session的任务 #{index} (session: {session_id[:8]})")
                        continue

                # 检查中断信号
                if self.abort_event.is_set():
                    logger.debug(f"检测到中断信号，跳过TTS任务 #{index}")
                    continue

                # 合成并播放
                try:
                    logger.debug(f"TTS合成 #{index}: {sentence[:30]}...")
                    audio_data = self.tts.synthesize_to_bytes(sentence)
                    logger.debug(f"TTS合成完成 #{index}, 大小: {len(audio_data)} bytes")

                    # 再次检查session（合成可能耗时较长）
                    with self.session_lock:
                        if session_id != self.current_session_id:
                            logger.debug(f"合成完成后session已变化，跳过播放 #{index}")
                            continue

                    # 再次检查中断信号
                    if self.abort_event.is_set():
                        logger.debug(f"合成完成后检测到中断信号，跳过播放 #{index}")
                        continue

                    # 播放
                    self.is_tts_playing = True
                    logger.debug(f"TTS播放 #{index}: {sentence[:30]}...")
                    self.tts.reset_playback_flag()
                    self.tts.play_audio_bytes(audio_data)
                    logger.debug(f"TTS播放完成 #{index}")

                except Exception as e:
                    logger.error(f"TTS #{index} 失败: {str(e)}")

            except Empty:
                # 队列空，继续等待
                self.is_tts_playing = False
                continue
            except Exception as e:
                logger.error(f"TTS worker错误: {str(e)}")

        self.is_tts_playing = False
        logger.info("TTS worker循环结束")

    def _on_stt_recognizing(self, text: str):
        """STT部分识别结果"""
        logger.debug(f"识别中: {text}")

    def _on_speech_started(self):
        """VAD检测到语音开始（用于快速打断）"""
        # 如果正在播放TTS，立即打断（不等待STT识别完成）
        if self.is_tts_playing:
            logger.info(f"⚠️ VAD检测到语音开始，立即打断！")
            self._handle_interruption()

    def _on_stt_recognized(self, text: str):
        """STT最终识别结果"""
        if not self.is_running:
            return

        logger.info(f"用户: {text}")

        # 检查退出命令
        exit_keywords = ["退出", "再见", "结束对话", "拜拜"]
        if any(keyword in text for keyword in exit_keywords):
            logger.info("用户请求退出")
            self.stop()
            return

        # 触发用户语音回调
        if self.on_user_speech:
            self.on_user_speech(text)

        # 处理用户输入（异步）
        self._current_user_text = text
        self._recognition_complete_event.set()

        # 启动处理线程
        Thread(target=self._process_user_input, args=(text,), daemon=True).start()

    def _on_stt_session_started(self):
        """STT会话启动"""
        logger.info("语音识别会话已启动")

    def _on_stt_session_stopped(self):
        """STT会话停止"""
        logger.info("语音识别会话已停止")

    def _on_stt_canceled(self, reason: str):
        """STT取消/错误"""
        logger.error(f"STT错误: {reason}")
        if self.on_error:
            self.on_error(f"语音识别错误: {reason}")

    def _process_user_input(self, user_text: str):
        """
        处理用户输入（核心流程）

        流程：
        1. RAG检索
        2. 构建上下文
        3. LLM生成（流式）
        4. TTS播放
        5. 更新历史
        6. 压缩上下文
        """
        if self.is_processing and not self.interrupt_requested:
            logger.warning("正在处理中，请稍候...")
            return

        # 检查是否是打断后的新输入
        was_interrupted = self.interrupt_requested
        if was_interrupted:
            logger.info(f"处理打断后的新输入: {user_text}")
            self.interrupt_requested = False  # 重置打断标志
            self._current_user_text = user_text  # 更新当前用户文本

        # 清除中断信号（开始处理新任务）
        self.abort_event.clear()
        logger.debug("已清除中断信号，开始处理新任务")

        self.is_processing = True
        self._current_assistant_text = ""

        try:
            # ===== 步骤1: RAG检索 =====
            logger.info("[1/5] RAG检索中...")
            rag_start_time = time.time()
            rag_result = self.rag.search(user_text)
            rag_elapsed = time.time() - rag_start_time
            logger.info(f"[1/5] RAG检索完成，耗时: {rag_elapsed:.3f}秒")

            if self.on_rag_retrieved:
                self.on_rag_retrieved(rag_result)

            # ===== 步骤2: 判断直接回答还是RAG生成 =====
            if rag_result["type"] == "direct_answer":
                # 高置信度QA，直接返回答案
                logger.info(f"[2/5] 直接回答（置信度: {rag_result['confidence']:.2f}）")
                assistant_text = rag_result["answer"]
                self._current_assistant_text = assistant_text

                # ===== 步骤3: TTS播放（仅直接回答需要）=====
                logger.info("[3/5] 播放回答...")
                self._play_response(assistant_text)

            else:
                # 需要LLM生成
                logger.info("[2/5] LLM生成回答中...")

                # 构建RAG上下文
                rag_context = self._build_rag_context(rag_result)

                # 管理上下文（检查是否需要压缩）
                managed_messages, summary = self.context_mgr.manage_context(
                    self.messages
                )

                # 构建完整上下文
                if summary:
                    managed_messages = self.context_mgr.build_context_with_summary(
                        managed_messages, summary
                    )

                # LLM流式生成（内部已实时TTS播放）
                logger.info("[3/5] 流式生成并播放...")
                assistant_text = self._generate_with_llm(
                    user_text, rag_context, managed_messages
                )
                self._current_assistant_text = assistant_text

            # ===== 步骤4: 更新对话历史 =====
            logger.info("[4/5] 更新对话历史...")
            if not was_interrupted:
                # 正常情况：添加用户输入和assistant回复
                self.messages.append({"role": "user", "content": user_text})
                self.messages.append({"role": "assistant", "content": assistant_text})
            else:
                # 打断后：只添加新的用户输入和assistant回复（被打断的回复已在_handle_interruption中保存）
                self.messages.append({"role": "user", "content": user_text})
                self.messages.append({"role": "assistant", "content": assistant_text})
                logger.info(f"已添加打断后的新对话到历史")

            # ===== 步骤5: 异步压缩上下文（如果需要）=====
            if self.context_mgr.should_compress(self.messages):
                logger.info("[5/5] 触发异步上下文压缩...")
                Thread(target=self._compress_context_async, daemon=True).start()

            # 等待TTS播放完成（防止回音）
            logger.info("等待TTS播放完成...")
            while self.is_tts_playing:
                time.sleep(0.1)

            logger.info("处理完成")

        except Exception as e:
            logger.error(f"处理失败: {str(e)}")
            if self.on_error:
                self.on_error(f"处理失败: {str(e)}")

        finally:
            self.is_processing = False

    def _build_rag_context(self, rag_result: Dict) -> str:
        """构建RAG上下文"""
        if rag_result["type"] != "rag_context" or not rag_result.get("docs"):
            return ""

        context_parts = ["以下是相关的知识库内容：\n"]

        for i, doc in enumerate(rag_result["docs"], 1):
            context_parts.append(f"【文档 {i}】{doc['title']}")
            context_parts.append(doc["content"])
            context_parts.append("")

        context_parts.append("请基于以上内容回答用户问题。如果内容中没有相关信息，请如实告知。")

        return "\n".join(context_parts)

    def _generate_with_llm(
        self, user_text: str, rag_context: str, messages: List[Dict[str, str]]
    ) -> str:
        """使用LLM生成回答（流式 + 实时TTS）"""
        # 构建增强的用户消息
        if rag_context:
            enhanced_query = f"{rag_context}\n\n用户问题：{user_text}"
        else:
            enhanced_query = user_text

        # 添加到消息列表
        current_messages = messages + [{"role": "user", "content": enhanced_query}]

        # 获取当前session ID（用于任务标识）
        with self.session_lock:
            session_id = self.current_session_id

        sentence_index = 0

        # 流式生成 + 分句处理
        sentence_buffer = ""
        full_response = ""

        for chunk in self.llm.chat_stream(
            current_messages, system_prompt=self.system_prompt
        ):
            # 检查中断信号
            if self.abort_event.is_set():
                logger.info("检测到中断信号，停止LLM生成")
                break

            logger.info(f"[LLM] chunk: {repr(chunk)}")
            full_response += chunk
            sentence_buffer += chunk

            # 优先检测强句子边界（。！？\n）
            strong_parts = re.split(r'([。！？\n])', sentence_buffer)

            if len(strong_parts) > 1:
                # 有强分隔符，立即分句
                for i in range(0, len(strong_parts) - 1, 2):
                    if i + 1 < len(strong_parts):
                        sentence = strong_parts[i] + strong_parts[i+1]
                        if sentence.strip():
                            self.tts_queue.put((session_id, sentence_index, sentence.strip()))
                            logger.info(f"[TTS入队-强分 #{sentence_index}] {sentence.strip()[:30]}...")
                            sentence_index += 1

                # 保留未完成的部分
                sentence_buffer = strong_parts[-1] if len(strong_parts) % 2 == 1 else ""

            else:
                # 没有强分隔符，检测弱分隔符（，、；：）
                weak_parts = re.split(r'([，、；：])', sentence_buffer)
                if len(weak_parts) > 1:
                    # 有弱分隔符，立即分句
                    for i in range(0, len(weak_parts) - 1, 2):
                        if i + 1 < len(weak_parts):
                            sentence = weak_parts[i] + weak_parts[i+1]
                            if sentence.strip():
                                self.tts_queue.put((session_id, sentence_index, sentence.strip()))
                                logger.info(f"[TTS入队-弱分 #{sentence_index}] {sentence.strip()[:30]}...")
                                sentence_index += 1

                    # 保留未完成的部分
                    sentence_buffer = weak_parts[-1] if len(weak_parts) % 2 == 1 else ""
                elif len(sentence_buffer) > 60:
                    # 既无分隔符又buffer过长，强制分句
                    self.tts_queue.put((session_id, sentence_index, sentence_buffer.strip()))
                    logger.info(f"[TTS入队-强制 #{sentence_index}] {sentence_buffer.strip()[:30]}...")
                    sentence_index += 1
                    sentence_buffer = ""

        # 处理剩余文本（如果未被中断）
        if not self.abort_event.is_set() and sentence_buffer.strip():
            self.tts_queue.put((session_id, sentence_index, sentence_buffer.strip()))
            logger.debug(f"TTS队列(剩余 #{sentence_index}): {sentence_buffer.strip()[:30]}...")

        logger.info(f"助手: {full_response}")

        # 触发回调
        if self.on_assistant_response:
            self.on_assistant_response(full_response)

        return full_response

    def _play_response(self, text: str):
        """播放回答"""
        try:
            self.tts.synthesize_and_play(
                text,
                on_synthesis_started=lambda: logger.debug("TTS开始"),
                on_synthesis_completed=lambda: logger.debug("TTS完成"),
                on_canceled=lambda reason: logger.error(f"TTS失败: {reason}"),
            )
        except Exception as e:
            logger.error(f"TTS播放失败: {str(e)}")

    def _compress_context_async(self):
        """异步压缩上下文"""
        try:
            logger.info("开始异步压缩上下文...")
            managed_messages, summary = self.context_mgr.manage_context(
                self.messages, force_compress=True
            )

            if summary:
                # 更新消息列表
                self.messages = self.context_mgr.build_context_with_summary(
                    managed_messages, summary
                )
                logger.info(f"压缩完成，保留 {len(self.messages)} 条消息")

        except Exception as e:
            logger.error(f"异步压缩失败: {str(e)}")

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """获取对话历史"""
        return self.messages.copy()

    def _handle_interruption(self):
        """
        处理打断：更新session ID，停止当前播放

        使用session ID + abort Event机制：
        - 设置abort_event，立即中断LLM生成和TTS合成
        - 更新session_id，后续任务会自动跳过旧session的内容
        - 停止当前播放
        - 清空队列中的旧任务（可选，因为worker会自动跳过）
        """
        logger.info("⚠️ 处理打断...")

        # 1. 设置中断信号（立即停止LLM生成和TTS合成）
        self.abort_event.set()
        logger.debug("已设置中断信号")

        # 2. 停止当前TTS播放
        self.tts.stop_playback()

        # 3. 更新session ID（这是关键：后续任务会跳过旧session）
        with self.session_lock:
            old_session_id = self.current_session_id
            self.current_session_id = str(uuid.uuid4())
            logger.debug(f"Session ID已更新: {old_session_id[:8]} -> {self.current_session_id[:8]}")

        # 4. 清空队列中的旧任务（可选优化，避免worker处理无用任务）
        cleared_count = 0
        while not self.tts_queue.empty():
            try:
                self.tts_queue.get_nowait()
                cleared_count += 1
            except:
                break
        if cleared_count > 0:
            logger.debug(f"已清空队列中的 {cleared_count} 个旧任务")

        # 5. 保存被打断的不完整回复
        if self._current_assistant_text:
            interrupted_text = self._current_assistant_text + " [被打断]"
            self.interrupted_response = interrupted_text

            # 添加到对话历史
            self.messages.append({"role": "user", "content": self._current_user_text})
            self.messages.append({"role": "assistant", "content": interrupted_text})
            logger.info(f"已保存被打断的回复到上下文: {interrupted_text[:50]}...")

        # 6. 设置打断标志
        self.interrupt_requested = True

        # 7. 重置处理状态
        self.is_processing = False
        self.is_tts_playing = False

        logger.info("✓ 打断处理完成")

    def clear_history(self):
        """清空对话历史"""
        self.messages.clear()
        logger.info("对话历史已清空")
