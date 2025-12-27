"""
Azure Text-to-Speech 服务
支持流式合成 + 语速调节
"""

import logging
from typing import Callable, Optional, Iterator
import azure.cognitiveservices.speech as speechsdk

logger = logging.getLogger(__name__)


class TTSService:
    """语音合成服务（Azure TTS）"""

    def __init__(
        self,
        key: str,
        region: str,
        voice_name: str = "zh-CN-XiaoxiaoNeural",
        rate: float = 1.0,
    ):
        """
        初始化TTS服务

        Args:
            key: Azure Speech API密钥
            region: Azure区域
            voice_name: 语音名称（默认晓晓）
            rate: 语速（0.5-2.0，默认1.0）
        """
        self.key = key
        self.region = region
        self.voice_name = voice_name
        self.rate = max(0.5, min(2.0, rate))  # 限制范围
        self.synthesizer = None
        self._is_synthesizing = False

    def _build_ssml(self, text: str) -> str:
        """
        构建SSML（用于语速控制）

        Args:
            text: 要合成的文本

        Returns:
            SSML字符串
        """
        # 计算prosody rate（相对百分比）
        rate_percent = f"{int((self.rate - 1.0) * 100):+d}%"

        ssml = f"""
<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="zh-CN">
    <voice name="{self.voice_name}">
        <prosody rate="{rate_percent}">
            {text}
        </prosody>
    </voice>
</speak>
""".strip()
        return ssml

    def synthesize_and_play(
        self,
        text: str,
        on_synthesis_started: Optional[Callable[[], None]] = None,
        on_synthesis_completed: Optional[Callable[[], None]] = None,
        on_canceled: Optional[Callable[[str], None]] = None,
    ):
        """
        合成语音并直接播放到扬声器

        Args:
            text: 要合成的文本
            on_synthesis_started: 合成开始回调
            on_synthesis_completed: 合成完成回调
            on_canceled: 取消/错误回调
        """
        if self._is_synthesizing:
            logger.warning("合成正在进行中")
            return

        try:
            # 配置语音合成
            speech_config = speechsdk.SpeechConfig(
                subscription=self.key, region=self.region
            )

            # 使用默认扬声器
            audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

            # 创建合成器
            self.synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=speech_config, audio_config=audio_config
            )

            # 绑定事件回调
            def _on_started(evt):
                self._is_synthesizing = True
                logger.info("语音合成已启动")
                if on_synthesis_started:
                    on_synthesis_started()

            def _on_completed(evt):
                self._is_synthesizing = False
                logger.info("语音合成已完成")
                if on_synthesis_completed:
                    on_synthesis_completed()

            def _on_canceled(evt):
                self._is_synthesizing = False
                reason = f"合成取消: {evt.cancellation_details.reason}"
                if evt.cancellation_details.reason == speechsdk.CancellationReason.Error:
                    error_details = evt.cancellation_details.error_details
                    reason = f"合成错误: {error_details}"
                    logger.error(reason)
                if on_canceled:
                    on_canceled(reason)

            # 连接回调
            self.synthesizer.synthesis_started.connect(_on_started)
            self.synthesizer.synthesis_completed.connect(_on_completed)
            self.synthesizer.synthesis_canceled.connect(_on_canceled)

            # 构建SSML并合成
            ssml = self._build_ssml(text)
            result = self.synthesizer.speak_ssml_async(ssml).get()

            # 检查结果
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                logger.info(f"合成成功: {len(result.audio_data)} 字节")
            elif result.reason == speechsdk.ResultReason.Canceled:
                error_msg = f"合成失败: {result.cancellation_details.reason}"
                logger.error(error_msg)
                if on_canceled:
                    on_canceled(error_msg)

        except Exception as e:
            self._is_synthesizing = False
            error_msg = f"合成失败: {str(e)}"
            logger.error(error_msg)
            if on_canceled:
                on_canceled(error_msg)
            raise

    def synthesize_to_bytes(self, text: str) -> bytes:
        """
        合成语音并返回完整音频数据（不播放）

        Args:
            text: 要合成的文本

        Returns:
            音频数据（WAV格式）
        """
        try:
            # 配置语音合成（无音频输出）
            speech_config = speechsdk.SpeechConfig(
                subscription=self.key, region=self.region
            )

            # 使用pull stream
            pull_stream = speechsdk.audio.PullAudioOutputStream()
            audio_config = speechsdk.audio.AudioOutputConfig(stream=pull_stream)

            # 创建合成器
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=speech_config, audio_config=audio_config
            )

            # 构建SSML并合成
            ssml = self._build_ssml(text)
            result = synthesizer.speak_ssml_async(ssml).get()

            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                # 直接返回音频数据
                return result.audio_data
            else:
                error_msg = f"合成失败: {result.cancellation_details.reason}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

        except Exception as e:
            logger.error(f"合成失败: {str(e)}")
            raise

    def play_audio_bytes(self, audio_data: bytes):
        """
        播放音频数据到默认扬声器

        Args:
            audio_data: 音频字节数据（WAV格式）
        """
        try:
            # 配置语音合成（使用默认扬声器）
            speech_config = speechsdk.SpeechConfig(
                subscription=self.key, region=self.region
            )
            audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

            # 创建合成器
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=speech_config, audio_config=audio_config
            )

            # 从音频数据创建音频流
            audio_stream = speechsdk.audio.PushAudioInputStream()
            audio_stream.write(audio_data)
            audio_stream.close()

            # 播放
            audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
            # 注意：Azure SDK没有直接播放bytes的API，需要用其他方式
            # 这里用PyAudio作为替代
            import wave
            import io
            import pyaudio

            # 解析WAV数据
            with io.BytesIO(audio_data) as wav_io:
                with wave.open(wav_io, 'rb') as wf:
                    p = pyaudio.PyAudio()
                    stream = p.open(
                        format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True
                    )

                    # 播放
                    chunk_size = 1024
                    data = wf.readframes(chunk_size)
                    while data:
                        stream.write(data)
                        data = wf.readframes(chunk_size)

                    stream.stop_stream()
                    stream.close()
                    p.terminate()

        except Exception as e:
            logger.error(f"播放失败: {str(e)}")
            raise

    def synthesize_to_stream(self, text: str) -> Iterator[bytes]:
        """
        合成语音并返回音频流（用于自定义处理）

        Args:
            text: 要合成的文本

        Yields:
            音频块（bytes）
        """
        try:
            # 配置语音合成（无音频输出）
            speech_config = speechsdk.SpeechConfig(
                subscription=self.key, region=self.region
            )

            # 使用pull stream
            pull_stream = speechsdk.audio.PullAudioOutputStream()
            audio_config = speechsdk.audio.AudioOutputConfig(stream=pull_stream)

            # 创建合成器
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=speech_config, audio_config=audio_config
            )

            # 构建SSML并合成
            ssml = self._build_ssml(text)
            result = synthesizer.speak_ssml_async(ssml).get()

            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                # 读取音频流
                audio_buffer = bytes(32000)  # 32KB buffer
                while True:
                    filled_size = pull_stream.read(audio_buffer)
                    if filled_size == 0:
                        break
                    yield audio_buffer[:filled_size]
            else:
                error_msg = f"合成失败: {result.cancellation_details.reason}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

        except Exception as e:
            logger.error(f"流式合成失败: {str(e)}")
            raise

    def stop(self):
        """停止合成"""
        if self.synthesizer and self._is_synthesizing:
            try:
                # Azure SDK没有直接的stop方法，通过清空synthesizer来中断
                self.synthesizer = None
                self._is_synthesizing = False
                logger.info("已停止语音合成")
            except Exception as e:
                logger.error(f"停止合成失败: {str(e)}")
