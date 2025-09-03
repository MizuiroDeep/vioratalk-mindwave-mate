"""
Audio Capture Module

音声入力デバイスの管理と録音機能を提供。
インフラストラクチャ層として音声ハードウェアを抽象化。

開発規約書 v1.12準拠
エラーハンドリング指針 v1.20準拠
"""

import asyncio
import logging
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np

# サードパーティライブラリのインポート
try:
    import sounddevice as sd

    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    warnings.warn("sounddevice not installed. Audio capture will use fallback methods.")

try:
    import pyaudio

    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    warnings.warn("pyaudio not installed. Some features may be limited.")

# プロジェクト内インポート
from vioratalk.core.base import ComponentState, VioraTalkComponent
from vioratalk.core.exceptions import AudioError
from vioratalk.core.i18n_manager import I18nManager
from vioratalk.core.stt.base import AudioData, AudioMetadata

# ============================================================================
# 定数定義
# ============================================================================

# デフォルト設定
DEFAULT_SAMPLE_RATE = 16000  # Hz (Whisper推奨)
DEFAULT_CHANNELS = 1  # モノラル
DEFAULT_DTYPE = np.float32  # 音声データ型
DEFAULT_CHUNK_SIZE = 1024  # チャンクサイズ
DEFAULT_RECORD_SECONDS = 5  # デフォルト録音時間

# 音声パラメータ
MIN_AUDIO_LEVEL = -60  # dB (最小音量)
MAX_AUDIO_LEVEL = 0  # dB (最大音量)
SILENCE_THRESHOLD = -40  # dB (無音判定閾値)
AGC_TARGET_LEVEL = -20  # dB (自動ゲイン調整目標)

# リトライ設定
MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY = 0.5  # 秒


# ============================================================================
# データクラス定義
# ============================================================================


@dataclass
class AudioDevice:
    """音声デバイス情報

    Attributes:
        id: デバイスID
        name: デバイス名
        channels: チャンネル数
        sample_rate: サンプリングレート
        is_default: デフォルトデバイスかどうか
        host_api: ホストAPI名
        latency: レイテンシ（秒）
    """

    id: int
    name: str
    channels: int
    sample_rate: int
    is_default: bool = False
    host_api: str = ""
    latency: float = 0.0

    def __str__(self) -> str:
        """文字列表現"""
        default_mark = " [DEFAULT]" if self.is_default else ""
        return f"{self.id}: {self.name} ({self.channels}ch, {self.sample_rate}Hz){default_mark}"


@dataclass
class RecordingConfig:
    """録音設定

    Attributes:
        device_id: 使用するデバイスID（Noneでデフォルト）
        sample_rate: サンプリングレート
        channels: チャンネル数
        duration: 録音時間（秒）
        chunk_size: バッファサイズ
        enable_agc: 自動ゲイン調整の有効化
        enable_noise_reduction: ノイズリダクションの有効化
        silence_threshold: 無音判定閾値（dB）
    """

    device_id: Optional[int] = None
    sample_rate: int = DEFAULT_SAMPLE_RATE
    channels: int = DEFAULT_CHANNELS
    duration: float = DEFAULT_RECORD_SECONDS
    chunk_size: int = DEFAULT_CHUNK_SIZE
    enable_agc: bool = False
    enable_noise_reduction: bool = False
    silence_threshold: float = SILENCE_THRESHOLD


# ============================================================================
# AudioCaptureクラス
# ============================================================================


class AudioCapture(VioraTalkComponent):
    """音声入力デバイス管理クラス

    マイクデバイスの管理と録音機能を提供。
    Phase 4実装として基本機能のみ実装、高度な機能は将来拡張。

    Attributes:
        config: 録音設定
        devices: 利用可能なデバイスリスト
        current_device: 現在選択されているデバイス
        i18n: 国際化マネージャー
    """

    def __init__(self, config: Optional[RecordingConfig] = None):
        """初期化

        Args:
            config: 録音設定（Noneの場合はデフォルト使用）

        Raises:
            InitializationError: 初期化失敗（E0100）
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.config = config or RecordingConfig()
        self.devices: List[AudioDevice] = []
        self.current_device: Optional[AudioDevice] = None
        self.i18n = I18nManager()
        self._audio_interface = None
        self._last_used: Optional[datetime] = None  # 最後に使用した時刻

        # 初期化時にデバイスをスキャン
        try:
            self._scan_devices()
        except Exception as e:
            self.logger.error(
                "Failed to scan audio devices",
                extra={"error_type": type(e).__name__},
                exc_info=True,
            )
            # 初期化エラーだが、後でデバイス選択可能にするため続行

    async def initialize(self) -> None:
        """非同期初期化

        VioraTalkComponentの抽象メソッド実装。
        状態遷移はsafe_initialize()が管理するため、ここでは実処理のみ。

        Raises:
            AudioError: 音声デバイス初期化失敗（E1001, E1002）
        """
        try:
            # デバイスリストを更新
            self._scan_devices()

            # デフォルトデバイスを選択
            if not self.current_device:
                self._select_default_device()

            if not self.current_device:
                raise AudioError(self.i18n.get_error_message("E1002"), error_code="E1002")

            self.logger.info(
                "AudioCapture initialized",
                extra={"device": self.current_device.name, "sample_rate": self.config.sample_rate},
            )

        except AudioError:
            raise
        except Exception as e:
            self.logger.error(
                "AudioCapture initialization failed",
                extra={"error_type": type(e).__name__},
                exc_info=True,
            )
            raise AudioError(self.i18n.get_error_message("E1001"), error_code="E1001") from e

    async def cleanup(self) -> None:
        """リソースのクリーンアップ

        VioraTalkComponentの抽象メソッド実装。
        状態遷移はsafe_cleanup()が管理するため、ここでは実処理のみ。
        """
        if self._audio_interface:
            try:
                if PYAUDIO_AVAILABLE and hasattr(self._audio_interface, "terminate"):
                    self._audio_interface.terminate()
            except Exception as e:
                self.logger.warning(f"Error during audio interface cleanup: {e}")
            finally:
                self._audio_interface = None

    def is_available(self) -> bool:
        """利用可能状態の確認

        VioraTalkComponentの抽象メソッド実装。

        Returns:
            bool: READYまたはRUNNING状態の場合True
        """
        return ComponentState.is_operational(self._state)

    def get_status(self) -> Dict[str, Any]:
        """コンポーネントの詳細ステータス取得

        VioraTalkComponentの抽象メソッド実装。
        基底クラスの実装を拡張してAudioCapture固有の情報を追加。

        Returns:
            Dict[str, Any]: 状態情報を含む辞書
        """
        # 基底クラスのステータスを取得
        status = super().get_status()

        # AudioCapture固有の情報を追加
        status.update(
            {
                "audio_capture": {
                    "device_count": len(self.devices),
                    "current_device": self.current_device.name if self.current_device else None,
                    "sample_rate": self.config.sample_rate,
                    "channels": self.config.channels,
                    "agc_enabled": self.config.enable_agc,
                    "noise_reduction_enabled": self.config.enable_noise_reduction,
                    "sounddevice_available": SOUNDDEVICE_AVAILABLE,
                    "pyaudio_available": PYAUDIO_AVAILABLE,
                },
                "last_used": self._last_used.isoformat() if self._last_used else None,
            }
        )

        return status

    # ========================================================================
    # デバイス管理
    # ========================================================================

    def list_devices(self) -> List[AudioDevice]:
        """利用可能な音声入力デバイスを取得

        Returns:
            List[AudioDevice]: デバイスリスト
        """
        self._scan_devices()
        return self.devices.copy()

    def select_device(self, device_id: int) -> None:
        """デバイスを選択

        Args:
            device_id: デバイスID

        Raises:
            AudioError: デバイスが見つからない（E1002）
        """
        for device in self.devices:
            if device.id == device_id:
                self.current_device = device
                self.config.device_id = device_id
                self.logger.info(f"Selected audio device: {device.name}")
                return

        raise AudioError(f"Device with ID {device_id} not found", error_code="E1002")

    def get_current_device(self) -> Optional[AudioDevice]:
        """現在のデバイスを取得

        Returns:
            Optional[AudioDevice]: 現在のデバイス（未選択の場合None）
        """
        return self.current_device

    # ========================================================================
    # 録音機能
    # ========================================================================

    async def record_from_microphone(
        self,
        duration: Optional[float] = None,
        callback: Optional[Callable[[np.ndarray], None]] = None,
    ) -> AudioData:
        """マイクから音声を録音

        Args:
            duration: 録音時間（秒）。Noneの場合は設定値を使用
            callback: チャンクごとのコールバック関数

        Returns:
            AudioData: 録音された音声データ

        Raises:
            AudioError: 録音失敗（E1001, E1002, E1003）
        """
        if self._state != ComponentState.READY:
            await self.safe_initialize()

        # 最後に使用した時刻を更新
        self._last_used = datetime.now()

        duration = duration or self.config.duration

        try:
            # sounddeviceを優先使用
            if SOUNDDEVICE_AVAILABLE:
                audio_array = await self._record_with_sounddevice(duration, callback)
            elif PYAUDIO_AVAILABLE:
                audio_array = await self._record_with_pyaudio(duration, callback)
            else:
                raise AudioError(
                    "No audio library available (sounddevice or pyaudio required)",
                    error_code="E1002",
                )

            # 音声レベルチェック
            if self._is_silent(audio_array):
                self.logger.warning(
                    "Recorded audio is too quiet", extra={"level": self._calculate_db(audio_array)}
                )
                # E1003: 音声が小さい（警告のみ、エラーにはしない）

            # AGC適用（有効な場合）
            if self.config.enable_agc:
                audio_array = self._apply_agc(audio_array)

            # ノイズリダクション適用（有効な場合）
            if self.config.enable_noise_reduction:
                audio_array = self._apply_noise_reduction(audio_array)

            # AudioDataオブジェクト作成
            metadata = AudioMetadata(
                sample_rate=self.config.sample_rate,
                channels=self.config.channels,
                bit_depth=16,
                duration=duration,
                format="pcm_float32",
                filename=None,
            )

            return AudioData(raw_data=audio_array, encoded_data=None, metadata=metadata)

        except AudioError:
            raise
        except Exception as e:
            self.logger.error(
                "Recording failed", extra={"error_type": type(e).__name__}, exc_info=True
            )
            raise AudioError(self.i18n.get_error_message("E1001"), error_code="E1001") from e

    # ========================================================================
    # プライベートメソッド
    # ========================================================================

    def _scan_devices(self) -> None:
        """デバイスをスキャン"""
        self.devices.clear()

        if SOUNDDEVICE_AVAILABLE:
            self._scan_sounddevice()
        elif PYAUDIO_AVAILABLE:
            self._scan_pyaudio()

    def _scan_sounddevice(self) -> None:
        """sounddeviceでデバイスをスキャン"""
        try:
            devices = sd.query_devices()
            default_input = sd.default.device[0]

            for idx, device in enumerate(devices):
                if device["max_input_channels"] > 0:  # 入力デバイスのみ
                    audio_device = AudioDevice(
                        id=idx,
                        name=device["name"],
                        channels=device["max_input_channels"],
                        sample_rate=int(device["default_samplerate"]),
                        is_default=(idx == default_input),
                        host_api=sd.query_hostapis(device["hostapi"])["name"],
                        latency=device["default_low_input_latency"],
                    )
                    self.devices.append(audio_device)

        except Exception as e:
            self.logger.error(f"Error scanning sounddevice: {e}")

    def _scan_pyaudio(self) -> None:
        """PyAudioでデバイスをスキャン"""
        try:
            # ローカルインポート
            import pyaudio

            if not self._audio_interface:
                self._audio_interface = pyaudio.PyAudio()

            p = self._audio_interface
            default_input = p.get_default_input_device_info()["index"]

            for i in range(p.get_device_count()):
                try:
                    info = p.get_device_info_by_index(i)
                    if info["maxInputChannels"] > 0:
                        audio_device = AudioDevice(
                            id=i,
                            name=info["name"],
                            channels=info["maxInputChannels"],
                            sample_rate=int(info["defaultSampleRate"]),
                            is_default=(i == default_input),
                            host_api=p.get_host_api_info_by_index(info["hostApi"])["name"],
                        )
                        self.devices.append(audio_device)
                except Exception:
                    continue

        except Exception as e:
            self.logger.error(f"Error scanning pyaudio: {e}")

    def _select_default_device(self) -> None:
        """デフォルトデバイスを選択"""
        for device in self.devices:
            if device.is_default:
                self.current_device = device
                self.config.device_id = device.id
                return

        # デフォルトが見つからない場合は最初のデバイス
        if self.devices:
            self.current_device = self.devices[0]
            self.config.device_id = self.devices[0].id

    async def _record_with_sounddevice(
        self, duration: float, callback: Optional[Callable] = None
    ) -> np.ndarray:
        """sounddeviceで録音"""
        loop = asyncio.get_event_loop()

        def record_sync():
            recording = sd.rec(
                int(duration * self.config.sample_rate),
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype=DEFAULT_DTYPE,
                device=self.config.device_id,
            )
            sd.wait()
            return recording.flatten() if self.config.channels == 1 else recording

        # ブロッキング処理を非同期実行
        audio_data = await loop.run_in_executor(None, record_sync)

        if callback:
            callback(audio_data)

        return audio_data

    async def _record_with_pyaudio(
        self, duration: float, callback: Optional[Callable] = None
    ) -> np.ndarray:
        """PyAudioで録音"""
        # ローカルインポート（メソッドの先頭で実行）
        import pyaudio

        if not self._audio_interface:
            self._audio_interface = pyaudio.PyAudio()

        p = self._audio_interface
        frames = []

        stream = p.open(
            format=pyaudio.paFloat32,
            channels=self.config.channels,
            rate=self.config.sample_rate,
            input=True,
            input_device_index=self.config.device_id,
            frames_per_buffer=self.config.chunk_size,
        )

        try:
            num_chunks = int(self.config.sample_rate / self.config.chunk_size * duration)

            for _ in range(num_chunks):
                data = stream.read(self.config.chunk_size)
                chunk = np.frombuffer(data, dtype=np.float32)
                frames.append(chunk)

                if callback:
                    callback(chunk)

            return np.concatenate(frames)

        finally:
            stream.stop_stream()
            stream.close()

    def _is_silent(self, audio_data: np.ndarray) -> bool:
        """無音判定

        Args:
            audio_data: 音声データ

        Returns:
            bool: 無音の場合True
        """
        db_level = self._calculate_db(audio_data)
        # bool型にキャストして返す（numpy.bool_ではなく）
        return bool(db_level < self.config.silence_threshold)

    def _calculate_db(self, audio_data: np.ndarray) -> float:
        """音声レベルをdBで計算

        Args:
            audio_data: 音声データ

        Returns:
            float: dB値
        """
        rms = np.sqrt(np.mean(audio_data**2))
        if rms > 0:
            return 20 * np.log10(rms)
        return MIN_AUDIO_LEVEL

    def _apply_agc(self, audio_data: np.ndarray) -> np.ndarray:
        """自動ゲイン調整（基本実装）

        Args:
            audio_data: 入力音声データ

        Returns:
            np.ndarray: 調整後の音声データ
        """
        current_db = self._calculate_db(audio_data)

        if current_db < MIN_AUDIO_LEVEL:
            return audio_data

        # ゲイン計算（シンプルな実装）
        gain_db = AGC_TARGET_LEVEL - current_db
        gain_linear = 10 ** (gain_db / 20)

        # クリッピング防止
        adjusted = audio_data * gain_linear
        adjusted = np.clip(adjusted, -1.0, 1.0)

        return adjusted

    def _apply_noise_reduction(self, audio_data: np.ndarray) -> np.ndarray:
        """ノイズリダクション（基本実装）

        Args:
            audio_data: 入力音声データ（1次元または2次元配列）

        Returns:
            np.ndarray: ノイズ除去後の音声データ
        """
        # 簡単なハイパスフィルタ（将来的にはnoisereduceライブラリ使用）
        # 現在は基本的な実装のみ

        # 多チャンネル対応: 2次元配列の場合は各チャンネルを個別に処理
        if audio_data.ndim == 2:
            # 各チャンネルを個別に処理
            processed_channels = []
            for channel in range(audio_data.shape[1]):
                channel_data = audio_data[:, channel]
                processed = self._apply_noise_reduction_single_channel(channel_data)
                processed_channels.append(processed)
            return np.column_stack(processed_channels)
        else:
            # 1次元配列の場合はそのまま処理
            return self._apply_noise_reduction_single_channel(audio_data)

    def _apply_noise_reduction_single_channel(self, audio_data: np.ndarray) -> np.ndarray:
        """単一チャンネルのノイズリダクション

        Args:
            audio_data: 1次元の音声データ

        Returns:
            np.ndarray: ノイズ除去後の音声データ
        """
        # DC成分除去
        audio_data = audio_data - np.mean(audio_data)

        # 簡単なスムージング（移動平均）
        window_size = 3
        if len(audio_data) > window_size:
            smoothed = np.convolve(audio_data, np.ones(window_size) / window_size, mode="same")
            return smoothed

        return audio_data


# ============================================================================
# ユーティリティ関数
# ============================================================================


def get_audio_capture(config: Optional[RecordingConfig] = None) -> AudioCapture:
    """AudioCaptureインスタンスを取得

    ファクトリ関数として、設定に基づいて適切なインスタンスを作成。

    Args:
        config: 録音設定

    Returns:
        AudioCapture: 音声キャプチャインスタンス
    """
    return AudioCapture(config=config)


async def quick_record(duration: float = 3.0) -> AudioData:
    """簡単な録音関数

    デフォルト設定で素早く録音を行うユーティリティ関数。

    Args:
        duration: 録音時間（秒）

    Returns:
        AudioData: 録音された音声データ

    Raises:
        AudioError: 録音失敗
    """
    capture = AudioCapture()
    await capture.safe_initialize()  # safe_initialize()を使用

    try:
        return await capture.record_from_microphone(duration=duration)
    finally:
        await capture.safe_cleanup()  # safe_cleanup()を使用


# ============================================================================
# エクスポート定義
# ============================================================================

__all__ = [
    "AudioCapture",
    "AudioDevice",
    "RecordingConfig",
    "get_audio_capture",
    "quick_record",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_CHANNELS",
]
