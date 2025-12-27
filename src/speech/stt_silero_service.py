"""
Speech-to-Text 服务（Silero VAD + Azure STT）
使用官方silero-vad库进行语音活动检测，Azure Speech SDK进行识别
"""

import logging
import threading
from typing import Callable, Optional
import numpy as np
import torch
import pyaudio
import azure.cognitiveservices.speech as speechsdk
from silero_vad import load_silero_vad

logger = logging.getLogger(__name__)


class STTSileroService:
    """语音识别服务（Silero VAD + Azure STT）"""

    def __init__(
        self,
        key: str,
        region: str,
        language: str = "zh-CN",
        sample_rate: int = 16000,
        vad_threshold: float = 0.5,
        model_path: str = None,
        min_speech_duration: float = 0.3,
        min_silence_duration: float = 1.0,
    ):
        """
        初始化STT服务

        Args:
            key: Azure Speech API密钥
            region: Azure区域
            language: 识别语言（默认zh-CN）
            sample_rate: 采样率（默认16000Hz）
            vad_threshold: VAD阈值（0-1，默认0.5）
            model_path: 保留参数，兼容性用（官方库自动下载模型）
            min_speech_duration: 最小语音时长（秒，默认0.3）
            min_silence_duration: 最小静音时长（秒，默认1.0，停顿多久算结束）
        """
        self.key = key
        self.region = region
        self.language = language
        self.sample_rate = sample_rate
        self.vad_threshold = vad_threshold
        self.vad_sample_rate = 16000  # Silero VAD固定16kHz
        self.chunk_size = 512  # 每次处理的帧数
        self.audio_gain = 50.0  # 音频增益倍数（提高麦克风灵敏度）

        # 初始化Silero VAD（官方库）
        self._init_silero_vad()

        # VAD状态
        self.min_speech_frames = int(min_speech_duration * sample_rate / self.chunk_size)
        self.min_silence_frames = int(min_silence_duration * sample_rate / self.chunk_size)
        self.is_speech = False
        self.speech_counter = 0
        self.silence_counter = 0

        # AGC（自动增益控制）参数
        self.agc_enabled = True  # 启用AGC
        self.agc_target_rms = 0.15  # 目标RMS（15%电平）
        self.agc_history_size = 100  # 历史窗口大小（帧数）
        self.agc_rms_history = []  # RMS历史记录
        self.agc_current_gain = 1.0  # 当前增益系数
        self.agc_max_gain = 50.0  # 最大增益
        self.agc_min_gain = 1.0  # 最小增益
        self.agc_adaptation_rate = 0.05  # 增益调整速率

        # PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream = None

        # 识别器
        self.recognizer = None
        self._is_recognizing = False
        self._stop_event = threading.Event()

        # 回调
        self.on_recognizing = None
        self.on_recognized = None
        self.on_session_started = None
        self.on_session_stopped = None
        self.on_canceled = None
        self.on_speech_started = None  # VAD检测到语音开始的回调（用于打断检测）

    def _init_silero_vad(self):
        """初始化Silero VAD（官方库）"""
        try:
            logger.info("加载Silero VAD模型（官方库）...")
            self.vad_model = load_silero_vad()
            logger.info("✓ Silero VAD模型加载完成")
        except Exception as e:
            logger.error(f"加载Silero VAD模型失败: {str(e)}")
            raise

    def start_continuous_recognition(
        self,
        on_recognizing: Callable[[str], None],
        on_recognized: Callable[[str], None],
        on_session_started: Optional[Callable[[], None]] = None,
        on_session_stopped: Optional[Callable[[], None]] = None,
        on_canceled: Optional[Callable[[str], None]] = None,
        on_speech_started: Optional[Callable[[], None]] = None,
    ):
        """
        启动连续识别（Silero VAD + Azure STT）

        Args:
            on_recognizing: 部分识别结果回调
            on_recognized: 最终识别结果回调
            on_session_started: 会话开始回调
            on_session_stopped: 会话停止回调
            on_canceled: 取消/错误回调
            on_speech_started: VAD检测到语音开始回调（用于打断检测）
        """
        if self._is_recognizing:
            logger.warning("识别已在运行中")
            return

        self.on_recognizing = on_recognizing
        self.on_recognized = on_recognized
        self.on_session_started = on_session_started
        self.on_session_stopped = on_session_stopped
        self.on_canceled = on_canceled
        self.on_speech_started = on_speech_started

        try:
            # 配置Azure Speech
            speech_config = speechsdk.SpeechConfig(
                subscription=self.key, region=self.region
            )
            speech_config.speech_recognition_language = self.language

            # 使用PushAudioInputStream（从Silero VAD输出）
            self.push_stream = speechsdk.audio.PushAudioInputStream()
            audio_config = speechsdk.audio.AudioConfig(stream=self.push_stream)

            self.recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config, audio_config=audio_config
            )

            # 绑定Azure STT事件
            self.recognizer.recognizing.connect(self._on_recognizing_event)
            self.recognizer.recognized.connect(self._on_recognized_event)
            self.recognizer.session_started.connect(self._on_session_started_event)
            self.recognizer.session_stopped.connect(self._on_session_stopped_event)
            self.recognizer.canceled.connect(self._on_canceled_event)

            # 启动Azure识别器
            self.recognizer.start_continuous_recognition()
            self._is_recognizing = True
            logger.info("Silero VAD + Azure STT 已启动")

            # 启动Silero VAD音频采集线程
            self._stop_event.clear()
            self._vad_thread = threading.Thread(target=self._vad_loop, daemon=True)
            self._vad_thread.start()

            if self.on_session_started:
                self.on_session_started()

        except Exception as e:
            logger.error(f"启动失败: {str(e)}")
            self._is_recognizing = False
            if self.on_canceled:
                self.on_canceled(f"启动失败: {str(e)}")
            raise

    def _vad_loop(self):
        """Silero VAD音频采集主循环"""
        try:
            # 打开麦克风流
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
            )

            logger.info("Silero VAD 音频采集已启动")

            speech_buffer = []

            frame_count = 0
            while not self._stop_event.is_set():
                # 读取音频帧
                audio_chunk = self.stream.read(self.chunk_size, exception_on_overflow=False)
                audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)

                # 计算音频电平（用于调试）
                amplitude = np.abs(audio_int16).mean()
                amplitude_pct = (amplitude / 32768.0) * 100

                # Silero VAD检测
                speech_prob = self._process_vad_chunk(audio_int16)
                is_speech_frame = speech_prob > self.vad_threshold

                # 实时显示（每10帧，DEBUG级别）
                frame_count += 1
                if frame_count % 10 == 0:
                    status = "🗣️ 语音" if is_speech_frame else "  静音"
                    logger.debug(f"[实时] 电平:{amplitude_pct:4.1f}% VAD:{speech_prob:.4f} {status}")

                if is_speech_frame:
                    # 检测到语音
                    speech_buffer.append(audio_chunk)
                    self.speech_counter += 1
                    self.silence_counter = 0

                    if not self.is_speech and self.speech_counter >= self.min_speech_frames:
                        self.is_speech = True
                        logger.info(f"[VAD] 语音开始 (prob={speech_prob:.4f}, counter={self.speech_counter})")
                        # 触发语音开始回调（用于打断检测）
                        if self.on_speech_started:
                            self.on_speech_started()
                else:
                    # 静音
                    self.silence_counter += 1
                    self.speech_counter = 0

                    if self.is_speech:
                        speech_buffer.append(audio_chunk)

                        if self.silence_counter >= self.min_silence_frames:
                            # 语音结束，推送到Azure STT
                            if speech_buffer:
                                full_audio = b"".join(speech_buffer)
                                self.push_stream.write(full_audio)
                                logger.info(f"[VAD] 语音结束，推送 {len(speech_buffer)} 帧 ({len(full_audio)} bytes) 到Azure STT")

                            speech_buffer = []
                            self.is_speech = False
                            logger.info("[VAD] 等待下一段语音...")

            # 关闭流
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                logger.info("Silero VAD 音频采集已停止")

        except Exception as e:
            logger.error(f"VAD循环错误: {str(e)}")
            if self.on_canceled:
                self.on_canceled(f"VAD错误: {str(e)}")

    def _process_vad_chunk(self, audio_data: np.ndarray) -> float:
        """
        处理音频块，返回语音概率（带AGC自动增益控制）

        Args:
            audio_data: int16音频数据

        Returns:
            语音概率（0-1）
        """
        try:
            # 转换为float32并归一化
            audio_float32 = audio_data.astype(np.float32) / 32768.0

            # 计算原始RMS
            original_rms = np.sqrt(np.mean(audio_float32 ** 2))

            # AGC自动增益控制
            if self.agc_enabled and original_rms > 0.0001:
                # 更新RMS历史
                self.agc_rms_history.append(original_rms)
                if len(self.agc_rms_history) > self.agc_history_size:
                    self.agc_rms_history.pop(0)

                # 计算历史平均RMS（用于稳定增益调整）
                if len(self.agc_rms_history) >= 10:  # 至少10帧后才开始调整
                    avg_rms = np.mean(self.agc_rms_history[-50:])  # 使用最近50帧

                    # 计算理想增益
                    ideal_gain = self.agc_target_rms / avg_rms if avg_rms > 0.0001 else self.agc_min_gain
                    ideal_gain = np.clip(ideal_gain, self.agc_min_gain, self.agc_max_gain)

                    # 平滑调整当前增益（避免突变）
                    self.agc_current_gain += (ideal_gain - self.agc_current_gain) * self.agc_adaptation_rate
                    self.agc_current_gain = np.clip(self.agc_current_gain, self.agc_min_gain, self.agc_max_gain)

                # 应用增益
                audio_float32 = audio_float32 * self.agc_current_gain

                # 软限幅（防止削波）
                audio_float32 = np.tanh(audio_float32)

                # 记录增益后的RMS
                gained_rms = np.sqrt(np.mean(audio_float32 ** 2))

                # 每100帧打印一次AGC状态（DEBUG级别）
                if not hasattr(self, '_vad_frame_count'):
                    self._vad_frame_count = 0
                self._vad_frame_count += 1

                if self._vad_frame_count % 100 == 0:
                    logger.debug(f"[AGC] 原始RMS:{original_rms:.4f} 增益:{self.agc_current_gain:.1f}x 输出RMS:{gained_rms:.4f}")
            else:
                # AGC未启用或信号太弱，使用固定增益
                audio_float32 = audio_float32 * self.audio_gain
                audio_float32 = np.tanh(audio_float32)

            # 转换为torch tensor并推理（官方库）
            audio_tensor = torch.from_numpy(audio_float32)

            with torch.no_grad():
                speech_prob = self.vad_model(audio_tensor, self.vad_sample_rate).item()

            return speech_prob

        except Exception as e:
            logger.error(f"VAD处理错误: {str(e)}")
            return 0.0

    def stop_continuous_recognition(self):
        """停止连续识别"""
        if not self._is_recognizing:
            return

        try:
            # 停止VAD线程
            self._stop_event.set()
            if self._vad_thread:
                self._vad_thread.join(timeout=2.0)

            # 停止Azure识别器
            if self.recognizer:
                self.recognizer.stop_continuous_recognition()

            self._is_recognizing = False
            logger.info("已停止连续语音识别")

        except Exception as e:
            logger.error(f"停止识别失败: {str(e)}")

    def _on_recognizing_event(self, evt):
        """Azure STT部分识别事件"""
        if evt.result.reason == speechsdk.ResultReason.RecognizingSpeech:
            text = evt.result.text
            if text.strip() and self.on_recognizing:
                self.on_recognizing(text)

    def _on_recognized_event(self, evt):
        """Azure STT最终识别事件"""
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            text = evt.result.text
            if text.strip() and self.on_recognized:
                self.on_recognized(text)
        elif evt.result.reason == speechsdk.ResultReason.NoMatch:
            logger.debug("未识别到语音")

    def _on_session_started_event(self, evt):
        """会话启动事件"""
        logger.info("Azure STT 会话已启动")

    def _on_session_stopped_event(self, evt):
        """会话停止事件"""
        self._is_recognizing = False
        logger.info("Azure STT 会话已停止")
        if self.on_session_stopped:
            self.on_session_stopped()

    def _on_canceled_event(self, evt):
        """取消事件"""
        self._is_recognizing = False
        reason = f"识别取消: {evt.result.cancellation_details.reason}"
        if evt.result.cancellation_details.reason == speechsdk.CancellationReason.Error:
            error_details = evt.result.cancellation_details.error_details
            reason = f"识别错误: {error_details}"
            logger.error(reason)
        if self.on_canceled:
            self.on_canceled(reason)

    def __del__(self):
        """清理资源"""
        if self.audio:
            self.audio.terminate()
