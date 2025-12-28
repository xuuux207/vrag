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
from concurrent.futures import ThreadPoolExecutor

from src.agents.rag_decision_agent import RAGDecisionAgent
from src.agents.input_completion_agent import InputCompletionAgent

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
        enable_tts_playback: bool = True,
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
            enable_tts_playback: 是否启用TTS播放（Web模式设为False）
        """
        self.stt = stt_service
        self.tts = tts_service
        self.rag = rag_searcher
        self.llm = llm_service
        self.context_mgr = context_manager

        # Web模式配置
        self.enable_tts_playback = enable_tts_playback

        # 初始化RAG判断Agent
        self.rag_decision_agent = RAGDecisionAgent()
        # 初始化输入完整性判断Agent
        self.input_completion_agent = InputCompletionAgent()

        self.system_prompt = system_prompt or self._default_system_prompt()

        # 对话状态
        self.messages: List[Dict[str, str]] = []
        self.is_running = False
        self.is_processing = False

        # 输入缓冲（用于处理输入不完整的情况 - 防抖机制）
        self.input_buffer = ""  # 缓冲的用户输入
        self.input_buffer_lock = Lock()  # 保护input_buffer
        self.pending_input_event = Event()  # 等待新输入的事件
        self.input_wait_timeout = 4.0  # 最多等待4秒

        # 打断控制
        self.interrupt_requested = False  # 打断请求标志
        self.interrupted_response = None  # 被打断的不完整回复
        self.abort_event = Event()  # 中断信号：用于立即停止LLM生成和TTS合成

        # TTS任务队列管理（使用session ID避免频繁启停线程）
        self.tts_queue = Queue()
        self.tts_synthesis_thread: Optional[Thread] = None  # 合成调度线程
        self.tts_playback_thread: Optional[Thread] = None  # 播放线程
        self.current_session_id = str(uuid.uuid4())  # 当前会话ID
        self.session_lock = Lock()  # 保护session_id
        self.is_tts_playing = False  # 当前是否正在播放
        self.expected_sentences: Dict[str, int] = {}  # 记录每个session预期的句子总数 {session_id: count}

        # TTS并行合成（加速语音生成）
        self.tts_synthesis_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="TTS-Synthesis")
        self.audio_buffer: Dict[tuple, bytes] = {}  # {(session_id, index): audio_data}
        self.audio_buffer_lock = Lock()  # 保护audio_buffer
        self.audio_ready_event = Condition()  # 通知播放线程有新音频ready

        # 临时变量（用于处理单轮对话）
        self._current_user_text = None
        self._current_assistant_text = ""
        self._recognition_complete_event = Event()

        # 回调函数
        self.on_user_speech: Optional[Callable[[str], None]] = None
        self.on_assistant_response: Optional[Callable[[str], None]] = None
        self.on_rag_retrieved: Optional[Callable[[Dict], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        self.on_audio_ready: Optional[Callable[[str, int, bytes], None]] = None  # Web模式音频回调
        self.on_generation_complete: Optional[Callable[[str], None]] = None  # LLM生成完成回调（通知前端所有音频已入队）

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

    def process_text_input(self, text: str):
        """
        处理文本输入（供Web模式调用）

        Args:
            text: 用户输入文本（已通过STT识别）
        """
        if not text or not text.strip():
            logger.warning("收到空文本输入")
            return

        logger.info(f"处理文本输入: {text[:50]}...")

        # 确保TTS worker已启动
        if not self.is_running:
            self.is_running = True
            self._start_tts_worker()

        # Web模式：STT已返回完整句子，直接处理（跳过完整性判断，避免4秒等待）
        self._start_processing(text)

    def _start_tts_worker(self):
        """启动持久TTS工作线程（合成+播放）"""
        if self.tts_synthesis_thread and self.tts_synthesis_thread.is_alive():
            logger.debug("TTS synthesis thread已在运行")
            return
        if self.tts_playback_thread and self.tts_playback_thread.is_alive():
            logger.debug("TTS playback thread已在运行")
            return

        self.tts_synthesis_thread = Thread(target=self._tts_synthesis_loop, daemon=True)
        self.tts_playback_thread = Thread(target=self._tts_playback_loop, daemon=True)
        self.tts_synthesis_thread.start()
        self.tts_playback_thread.start()
        logger.info("TTS worker线程已启动（合成+播放）")

    def _stop_tts_worker(self):
        """停止TTS工作线程"""
        # 发送停止信号
        self.tts_queue.put(None)
        logger.info("TTS worker线程停止信号已发送")

    def _tts_synthesis_loop(self):
        """
        TTS合成调度循环（持久运行）

        职责：
        - 从队列取任务
        - 立即提交到线程池合成（不等待）
        - 多个句子并行合成
        """
        logger.info("TTS合成调度循环开始")

        while self.is_running:
            try:
                # 从队列获取 (session_id, index, sentence)
                item = self.tts_queue.get(timeout=0.5)

                # None表示结束信号
                if item is None:
                    logger.debug("TTS合成线程收到停止信号")
                    break

                session_id, index, sentence = item

                # 检查session是否匹配
                with self.session_lock:
                    if session_id != self.current_session_id:
                        logger.debug(f"跳过旧session的合成任务 #{index}")
                        continue

                # 检查中断信号
                if self.abort_event.is_set():
                    logger.debug(f"检测到中断信号，跳过合成 #{index}")
                    continue

                # 提交到线程池异步合成
                buffer_key = (session_id, index)
                with self.audio_buffer_lock:
                    if buffer_key not in self.audio_buffer:
                        logger.debug(f"提交合成任务 #{index}: {sentence[:30]}...")
                        self.tts_synthesis_executor.submit(
                            self._synthesize_sentence, session_id, index, sentence
                        )

            except Empty:
                continue
            except Exception as e:
                logger.error(f"TTS合成调度错误: {str(e)}")

        logger.info("TTS合成调度循环结束")

    def _tts_playback_loop(self):
        """
        TTS播放循环（持久运行）

        职责：
        - 按序号等待buffer中的音频
        - 顺序播放
        """
        logger.info("TTS播放循环开始")

        current_session = None
        next_play_index = 0

        while self.is_running:
            try:
                # 获取当前session
                with self.session_lock:
                    session_id = self.current_session_id

                # 检测到新session，重置
                if current_session != session_id:
                    current_session = session_id
                    next_play_index = 0
                    logger.debug(f"切换到新session: {session_id[:8]}")

                # 等待下一个要播放的音频
                buffer_key = (session_id, next_play_index)
                audio_data = None

                # 检查是否知道预期的句子总数
                with self.session_lock:
                    expected_count = self.expected_sentences.get(session_id, None)

                # 根据是否知道预期数量，决定等待策略
                if expected_count is not None and next_play_index < expected_count:
                    # 还有句子未播放，持续等待（最多30秒，因为Azure TTS可能很慢）
                    max_wait_rounds = 600  # 600 * 0.05 = 30秒
                    wait_reason = f"等待 #{next_play_index}/{expected_count}"
                else:
                    # 不知道预期数量，或已播放完所有句子，等待5秒
                    max_wait_rounds = 100  # 100 * 0.05 = 5秒
                    wait_reason = f"探测 #{next_play_index}"

                for round_num in range(max_wait_rounds):
                    with self.audio_buffer_lock:
                        if buffer_key in self.audio_buffer:
                            audio_data = self.audio_buffer.pop(buffer_key)
                            break
                    time.sleep(0.05)

                    # 每5秒打印一次等待日志（避免刷屏）
                    if round_num > 0 and round_num % 100 == 0:
                        elapsed = round_num * 0.05
                        logger.info(f"[TTS播放] {wait_reason}，已等待{elapsed:.1f}秒...")

                if audio_data is None:
                    # 等待超时，检查是否已播放完所有预期句子
                    if expected_count is not None and next_play_index >= expected_count:
                        logger.info(f"[TTS播放] 已播放完所有 {expected_count} 个句子")
                        self.is_tts_playing = False
                        time.sleep(0.1)
                        continue
                    else:
                        # 不知道预期数量，或超时，认为播放完毕
                        logger.debug(f"[TTS播放] 等待超时，跳过 #{next_play_index}")
                        self.is_tts_playing = False
                        time.sleep(0.1)
                        continue

                # 检查session和中断信号
                with self.session_lock:
                    if session_id != self.current_session_id:
                        logger.debug(f"session已变化，跳过播放 #{next_play_index}")
                        continue

                if self.abort_event.is_set():
                    logger.debug(f"检测到中断信号，跳过播放 #{next_play_index}")
                    continue

                # 播放或回调
                try:
                    self.is_tts_playing = True
                    logger.debug(f"TTS播放 #{next_play_index}")

                    if self.enable_tts_playback:
                        # 本地模式：播放音频
                        self.tts.reset_playback_flag()
                        self.tts.play_audio_bytes(audio_data)
                        logger.debug(f"TTS播放完成 #{next_play_index}")
                    else:
                        # Web模式：触发回调
                        if self.on_audio_ready:
                            self.on_audio_ready(session_id, next_play_index, audio_data)
                            logger.debug(f"TTS回调完成 #{next_play_index}")

                    next_play_index += 1
                except Exception as e:
                    logger.error(f"TTS播放 #{next_play_index} 失败: {str(e)}")
                    next_play_index += 1

            except Exception as e:
                logger.error(f"TTS播放循环错误: {str(e)}")

        self.is_tts_playing = False
        logger.info("TTS播放循环结束")

    def _synthesize_sentence(self, session_id: str, index: int, sentence: str) -> None:
        """
        合成单个句子（在线程池中运行）

        Args:
            session_id: 会话ID
            index: 句子序号
            sentence: 句子文本
        """
        try:
            logger.debug(f"合成中 #{index}: {sentence[:30]}...")
            audio_data = self.tts.synthesize_to_bytes(sentence)
            logger.debug(f"合成完成 #{index}, 大小: {len(audio_data)} bytes")

            # 存入buffer
            with self.audio_buffer_lock:
                self.audio_buffer[(session_id, index)] = audio_data

        except Exception as e:
            logger.error(f"TTS合成 #{index} 失败: {str(e)}")

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

        # 处理用户输入（带完整性判断和等待）
        self._handle_user_input_with_completion_check(text)

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

    def _handle_user_input_with_completion_check(self, text: str):
        """
        处理用户输入（带完整性判断和等待 - 防抖机制）

        逻辑：
        1. 判断输入是否完整
        2. 如果不完整，等待新输入（最多4秒）
        3. 有新输入立即合并并重新判断
        4. 4秒后无新输入则直接处理
        """
        # 如果正在处理中，说明是打断，不需要完整性判断
        if self.is_processing:
            logger.info("检测到打断，直接处理新输入")
            self._start_processing(text)
            return

        with self.input_buffer_lock:
            # 如果缓冲区已有内容，合并
            if self.input_buffer:
                self.input_buffer += " " + text  # 用空格分隔
                logger.info(f"合并输入: {self.input_buffer}")
                # 触发等待事件（有新输入了）
                self.pending_input_event.set()
            else:
                # 第一次输入
                self.input_buffer = text
                # 启动完整性判断流程（异步）
                Thread(
                    target=self._check_input_completion_and_process,
                    daemon=True
                ).start()

    def _check_input_completion_and_process(self):
        """
        检查输入完整性并处理（防抖机制）

        在独立线程中运行，负责判断输入完整性，并等待可能的补充输入
        """
        with self.input_buffer_lock:
            current_input = self.input_buffer
            logger.info(f"[完整性判断] 当前输入: {current_input[:50]}")

        # 判断是否完整
        is_complete = self.input_completion_agent.is_input_complete(
            current_input,
            self.messages[-4:] if self.messages else []
        )

        if is_complete:
            # 输入完整，启动处理
            logger.info(f"✓ 输入完整，启动处理")
            self._start_processing(current_input)
            return

        # 输入不完整，等待新输入
        logger.info(f"✗ 输入不完整，等待{self.input_wait_timeout}秒（或新输入）...")
        self.pending_input_event.clear()

        # 等待新输入或超时
        has_new_input = self.pending_input_event.wait(timeout=self.input_wait_timeout)

        if has_new_input:
            logger.info("收到新输入，重新判断")
            # 重新进入判断流程
            self._check_input_completion_and_process()
        else:
            # 超时，直接处理当前输入
            with self.input_buffer_lock:
                final_input = self.input_buffer
            logger.info(f"等待超时，直接处理: {final_input[:50]}")
            self._start_processing(final_input)

    def _start_processing(self, user_text: str):
        """启动用户输入处理流程"""
        # 清空缓冲区
        with self.input_buffer_lock:
            self.input_buffer = ""

        # 更新当前用户文本
        self._current_user_text = user_text
        self._recognition_complete_event.set()

        # 启动处理线程
        Thread(target=self._process_user_input, args=(user_text,), daemon=True).start()

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

        # ===== 创建新的session ID（每个请求独立session，避免expected_sentences被覆盖）=====
        with self.session_lock:
            old_session_id = self.current_session_id
            self.current_session_id = str(uuid.uuid4())
            logger.info(f"创建新session: {old_session_id[:8]} -> {self.current_session_id[:8]}")

        self.is_processing = True
        self._current_assistant_text = ""

        try:
            # ===== 步骤1: 判断是否需要RAG =====
            should_rag = self.rag_decision_agent.should_use_rag(
                user_text,
                self.messages[-6:] if self.messages else []
            )
            logger.info(f"[1/6] RAG需求判断: {'需要' if should_rag else '不需要'}")

            # ===== 步骤2: RAG检索（如果需要）=====
            if should_rag:
                logger.info("[2/6] RAG检索中...")
                rag_start_time = time.time()
                rag_result = self.rag.search(user_text)
                rag_elapsed = time.time() - rag_start_time
                logger.info(f"[2/6] RAG检索完成，耗时: {rag_elapsed:.3f}秒")
            else:
                # 跳过RAG，构造空结果
                logger.info("[2/6] 跳过RAG检索（闲聊模式）")
                rag_result = {"type": "no_rag", "docs": []}

            if self.on_rag_retrieved:
                self.on_rag_retrieved(rag_result)

            # ===== 步骤3: 判断直接回答还是RAG生成 =====
            if rag_result["type"] == "direct_answer":
                # 高置信度QA，直接返回答案
                logger.info(f"[3/6] 直接回答（置信度: {rag_result['confidence']:.2f}）")
                assistant_text = rag_result["answer"]
                self._current_assistant_text = assistant_text

                # ===== 步骤4: TTS播放（仅直接回答需要）=====
                logger.info("[4/6] 播放回答...")
                self._play_response(assistant_text)

            else:
                # 需要LLM生成
                logger.info("[3/6] LLM生成回答中...")

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
                logger.info("[4/6] 流式生成并播放...")
                assistant_text = self._generate_with_llm(
                    user_text, rag_context, managed_messages
                )
                self._current_assistant_text = assistant_text

            # ===== 步骤5: 更新对话历史 =====
            logger.info("[5/6] 更新对话历史...")
            if not was_interrupted:
                # 正常情况：添加用户输入和assistant回复
                self.messages.append({"role": "user", "content": user_text})
                self.messages.append({"role": "assistant", "content": assistant_text})
            else:
                # 打断后：只添加新的用户输入和assistant回复（被打断的回复已在_handle_interruption中保存）
                self.messages.append({"role": "user", "content": user_text})
                self.messages.append({"role": "assistant", "content": assistant_text})
                logger.info(f"已添加打断后的新对话到历史")

            # ===== 步骤6: 异步压缩上下文（如果需要）=====
            if self.context_mgr.should_compress(self.messages):
                logger.info("[6/6] 触发异步上下文压缩...")
                Thread(target=self._compress_context_async, daemon=True).start()

            # 等待TTS播放完成（防止回音）
            logger.info("等待TTS播放完成...")

            # 获取当前session ID
            with self.session_lock:
                current_session = self.current_session_id

            # 等待条件：播放完成 + 队列为空 + buffer中无当前session的音频
            while True:
                # 检查播放状态
                if not self.is_tts_playing:
                    # 检查队列是否为空
                    if self.tts_queue.empty():
                        # 检查buffer中是否还有当前session的音频
                        with self.audio_buffer_lock:
                            has_pending = any(k[0] == current_session for k in self.audio_buffer.keys())

                        if not has_pending:
                            break  # 所有条件满足，退出等待

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

            # logger.info(f"[LLM] chunk: {repr(chunk)}")
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
            sentence_index += 1

        # 记录这个session预期的句子总数（播放线程需要知道何时停止等待）
        if not self.abort_event.is_set() and sentence_index > 0:
            with self.session_lock:
                self.expected_sentences[session_id] = sentence_index
            logger.info(f"[TTS] 预期播放 {sentence_index} 个句子")

        logger.info(f"助手: {full_response}")

        # 触发回调
        if self.on_assistant_response:
            self.on_assistant_response(full_response)

        # 触发生成完成回调（通知前端所有音频已入队，可以等待播放完成）
        if self.on_generation_complete:
            self.on_generation_complete(session_id)

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
        - 清空队列、buffer和expected_sentences中的旧任务
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

        # 5. 清空音频缓冲区中的旧音频
        with self.audio_buffer_lock:
            keys_to_remove = [k for k in self.audio_buffer.keys() if k[0] == old_session_id]
            for key in keys_to_remove:
                self.audio_buffer.pop(key, None)
            if len(keys_to_remove) > 0:
                logger.debug(f"已清空buffer中的 {len(keys_to_remove)} 个旧音频")

        # 6. 清空旧session的预期句子数记录
        with self.session_lock:
            if old_session_id in self.expected_sentences:
                self.expected_sentences.pop(old_session_id)
                logger.debug(f"已清空session {old_session_id[:8]} 的预期句子数")

        # 7. 保存被打断的不完整回复
        if self._current_assistant_text:
            interrupted_text = self._current_assistant_text + " [被打断]"
            self.interrupted_response = interrupted_text

            # 添加到对话历史
            self.messages.append({"role": "user", "content": self._current_user_text})
            self.messages.append({"role": "assistant", "content": interrupted_text})
            logger.info(f"已保存被打断的回复到上下文: {interrupted_text[:50]}...")

        # 8. 设置打断标志
        self.interrupt_requested = True

        # 9. 重置处理状态
        self.is_processing = False
        self.is_tts_playing = False

        logger.info("✓ 打断处理完成")

    def clear_history(self):
        """清空对话历史"""
        self.messages.clear()
        logger.info("对话历史已清空")
