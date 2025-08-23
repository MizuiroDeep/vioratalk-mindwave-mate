"""MockTTSEngine実装

音声合成エンジンのモック実装。
テスト用にダミー音声データを生成する。

インターフェース定義書 v1.34準拠
テストデータ・モック完全仕様書 v1.1準拠
エラーハンドリング指針 v1.20準拠
開発規約書 v1.12準拠
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

# プロジェクト内インポート（絶対インポート）
from vioratalk.core.base import ComponentState, VioraTalkComponent
from vioratalk.core.exceptions import AudioError, InvalidVoiceError, TTSError
from vioratalk.utils.logger_manager import LoggerManager


@dataclass
class SynthesisResult:
    """音声合成結果

    データフォーマット仕様書 v1.5準拠
    """

    audio_data: bytes
    sample_rate: int
    duration: float
    format: str  # "wav", "mp3", etc.
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class VoiceInfo:
    """音声情報

    インターフェース定義書 v1.34準拠
    """

    id: str
    name: str
    language: str
    gender: str
    sample_url: Optional[str] = None


@dataclass
class StyleInfo:
    """音声スタイル情報（Phase 7-8準備）

    インターフェース定義書 v1.34準拠
    """

    id: str
    name: str
    description: str
    voice_id: str


@dataclass
class VoiceParameters:
    """音声パラメータ（Phase 7-8準備）

    テストデータ・モック完全仕様書 v1.1準拠
    """

    voice_id: str
    speed: float = 1.0
    pitch: float = 1.0
    volume: float = 0.9
    style_id: Optional[str] = None
    intonation_scale: Optional[float] = None
    pre_phoneme_length: Optional[float] = None
    post_phoneme_length: Optional[float] = None


class MockTTSEngine(VioraTalkComponent):
    """TTSエンジンのモック実装

    Phase 3のテスト用にダミー音声データを生成するTTSエンジン。
    Phase 7-8に向けたVoiceParameters対応の準備も含む。

    Attributes:
        synthesis_delay: 音声合成の遅延シミュレーション（秒）
        error_mode: エラーモードの有効/無効
        current_voice_id: 現在使用中の音声ID
        available_voices: 利用可能な音声のリスト
        available_styles: 利用可能なスタイルのマッピング
        sample_rate: サンプリングレート
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初期化

        Args:
            config: 設定辞書（オプション）
        """
        # VioraTalkComponentの初期化（引数なし）
        super().__init__()

        # configを保存
        self.config = config or {}

        # ロガーの初期化
        self.logger = LoggerManager().get_logger(self.__class__.__name__)

        # 設定の初期化
        self.synthesis_delay = self.config.get("delay", 0.1)
        self.error_mode = False
        self.current_voice_id = self.config.get("voice_id", "ja-JP-Female-1")
        self.sample_rate = self.config.get("sample_rate", 22050)

        # VoiceParametersの保存用
        self.last_parameters: Optional[VoiceParameters] = None

        # 利用可能な音声
        self.available_voices = [
            VoiceInfo(id="ja-JP-Female-1", name="日本語女性1", language="ja", gender="female"),
            VoiceInfo(id="ja-JP-Female-2", name="日本語女性2", language="ja", gender="female"),
            VoiceInfo(id="ja-JP-Male-1", name="日本語男性1", language="ja", gender="male"),
            VoiceInfo(id="en-US-Female-1", name="英語女性1", language="en", gender="female"),
            VoiceInfo(id="en-US-Male-1", name="英語男性1", language="en", gender="male"),
        ]

        # 利用可能なスタイル（Phase 7-8準備）
        self.available_styles = {
            "ja-JP-Female-1": [
                StyleInfo(
                    id="happy", name="楽しい", description="明るく楽しげな声", voice_id="ja-JP-Female-1"
                ),
                StyleInfo(id="sad", name="悲しい", description="控えめで悲しげな声", voice_id="ja-JP-Female-1"),
            ]
        }

        self.logger.info(
            "MockTTSEngine initialized",
            extra={"voice_id": self.current_voice_id, "sample_rate": self.sample_rate},
        )

    # 状態プロパティ（VioraTalkComponent準拠）
    @property
    def state(self) -> ComponentState:
        """現在の状態を取得"""
        return self._state

    @state.setter
    def state(self, value: ComponentState) -> None:
        """状態を設定"""
        self._state = value

    # 抽象メソッドの実装

    async def initialize(self) -> None:
        """エンジンの初期化

        Raises:
            TTSError: 初期化エラー
        """
        self._state = ComponentState.INITIALIZING
        self.logger.info("Initializing MockTTSEngine")

        # 初期化処理のシミュレーション
        await asyncio.sleep(0.01)

        self._state = ComponentState.READY
        self.logger.info("MockTTSEngine initialization completed")

    async def cleanup(self) -> None:
        """リソースのクリーンアップ"""
        self._state = ComponentState.TERMINATING
        self.logger.info("Cleaning up MockTTSEngine")

        # クリーンアップ処理のシミュレーション
        await asyncio.sleep(0.01)

        self._state = ComponentState.TERMINATED
        self.logger.info("MockTTSEngine cleanup completed")

    def is_available(self) -> bool:
        """利用可能状態の確認

        Returns:
            bool: 利用可能な場合True
        """
        return self._state in [ComponentState.READY, ComponentState.RUNNING]

    def get_status(self) -> Dict[str, Any]:
        """ステータス情報の取得

        Returns:
            Dict[str, Any]: ステータス情報
        """
        return {
            "state": self._state.value,
            "is_available": self.is_available(),
            "error": None,
            "current_voice": self.current_voice_id,
            "sample_rate": self.sample_rate,
            "error_mode": self.error_mode,
        }

    # TTSエンジン固有のメソッド

    async def synthesize(
        self, text: str, voice_id: Optional[str] = None, style: Optional[str] = None, **kwargs
    ) -> SynthesisResult:
        """テキストを音声に変換

        Args:
            text: 変換するテキスト
            voice_id: 音声ID（オプション）
            style: スタイル（オプション）
            **kwargs: 追加パラメータ

        Returns:
            SynthesisResult: 音声合成結果

        Raises:
            AudioError: エラーモード時またはオーディオエラー（E3001）
            TTSError: 音声合成エラー（E3000）
        """
        # 状態チェック
        if self._state != ComponentState.READY:
            raise TTSError(
                "TTS engine is not ready", error_code="E3000", details={"state": self._state.value}
            )

        # エラーモードチェック
        if self.error_mode:
            self.logger.error("Mock TTS error mode is enabled")
            raise AudioError(
                "Mock audio synthesis error",
                error_code="E3001",
                details={"mode": "error_simulation"},
            )

        # 音声IDの決定
        if voice_id is None:
            voice_id = self.current_voice_id

        # 音声IDの検証
        if not self._is_valid_voice_id(voice_id):
            self.logger.warning(f"Invalid voice_id: {voice_id}, using default")
            voice_id = self.current_voice_id

        # 遅延シミュレーション
        await asyncio.sleep(self.synthesis_delay)

        # ダミー音声データの生成
        audio_data = self._generate_dummy_audio(text, voice_id, style)

        # 音声の長さを計算（文字数ベースの簡易計算）
        duration = len(text) * 0.1  # 1文字あたり0.1秒と仮定

        # 結果の作成
        result = SynthesisResult(
            audio_data=audio_data,
            sample_rate=self.sample_rate,
            duration=duration,
            format="wav",
            metadata={
                "voice_id": voice_id,
                "style": style,
                "text_length": len(text),
                "engine": "mock",
            },
        )

        self.logger.info(
            "Synthesis completed",
            extra={
                "text_length": len(text),
                "voice_id": voice_id,
                "style": style,
                "duration": duration,
            },
        )

        return result

    async def synthesize_with_parameters(
        self, text: str, voice_params: VoiceParameters
    ) -> SynthesisResult:
        """VoiceParametersを使用した音声合成（Phase 7-8準備）

        Args:
            text: 変換するテキスト
            voice_params: 音声パラメータ

        Returns:
            SynthesisResult: 音声合成結果
        """
        # パラメータをキャッシュ
        self.last_parameters = voice_params

        # 通常のsynthesizeに委譲
        return await self.synthesize(
            text, voice_id=voice_params.voice_id, style=voice_params.style_id
        )

    def get_available_voices(self) -> List[VoiceInfo]:
        """利用可能な音声のリスト

        Returns:
            List[VoiceInfo]: 音声情報のリスト
        """
        return self.available_voices.copy()

    def set_voice(self, voice_id: str) -> None:
        """使用する音声を設定

        Args:
            voice_id: 音声ID

        Raises:
            InvalidVoiceError: 無効な音声ID（E3001）
        """
        if not self._is_valid_voice_id(voice_id):
            raise InvalidVoiceError(
                f"Voice ID not found: {voice_id}", error_code="E3001", voice_id=voice_id
            )

        self.current_voice_id = voice_id
        self.logger.info(f"Voice changed to: {voice_id}")

    async def test_availability(self) -> bool:
        """エンジンが利用可能かテスト

        Returns:
            bool: 利用可能な場合True
        """
        try:
            # 簡単なテスト合成
            result = await self.synthesize("test")
            return result is not None
        except Exception as e:
            self.logger.error(f"Availability test failed: {e}")
            return False

    def get_available_styles(self, voice_id: str) -> List[StyleInfo]:
        """利用可能なスタイル一覧を取得（Phase 7-8準備）

        Args:
            voice_id: 音声ID

        Returns:
            List[StyleInfo]: スタイル情報のリスト
        """
        return self.available_styles.get(voice_id, []).copy()

    # プライベートメソッド

    def _is_valid_voice_id(self, voice_id: str) -> bool:
        """音声IDの有効性をチェック

        Args:
            voice_id: 音声ID

        Returns:
            bool: 有効な場合True
        """
        return any(v.id == voice_id for v in self.available_voices)

    def _generate_dummy_audio(self, text: str, voice_id: str, style: Optional[str]) -> bytes:
        """ダミー音声データを生成

        Args:
            text: テキスト
            voice_id: 音声ID
            style: スタイル

        Returns:
            bytes: ダミー音声データ
        """
        # テキストの長さに基づいてサンプル数を決定
        num_samples = len(text) * self.sample_rate // 10  # 簡易計算

        # サイン波を生成（周波数は音声IDによって変更）
        frequency = 440.0  # A4音
        if "Female" in voice_id:
            frequency = 520.0  # 女性声は高め
        elif "Male" in voice_id:
            frequency = 220.0  # 男性声は低め

        # スタイルによって周波数を調整
        if style == "happy":
            frequency *= 1.1
        elif style == "sad":
            frequency *= 0.9

        # サイン波生成
        t = np.linspace(0, num_samples / self.sample_rate, num_samples)
        wave = np.sin(2 * np.pi * frequency * t)

        # エンベロープ適用（フェードイン・フェードアウト）
        envelope = np.ones_like(wave)
        fade_samples = min(1000, num_samples // 10)
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
        wave = wave * envelope * 0.5  # 音量調整

        # 16ビット整数に変換
        wave_int16 = (wave * 32767).astype(np.int16)

        # WAVヘッダーを追加（簡易版）
        wav_header = self._create_wav_header(len(wave_int16) * 2, self.sample_rate)

        return wav_header + wave_int16.tobytes()

    def _create_wav_header(self, data_size: int, sample_rate: int) -> bytes:
        """WAVヘッダーを作成（簡易版）

        Args:
            data_size: データサイズ（バイト）
            sample_rate: サンプリングレート

        Returns:
            bytes: WAVヘッダー
        """
        # 簡易的なWAVヘッダー（44バイト）
        header = b"RIFF"
        header += (data_size + 36).to_bytes(4, "little")
        header += b"WAVE"
        header += b"fmt "
        header += (16).to_bytes(4, "little")  # fmt chunk size
        header += (1).to_bytes(2, "little")  # PCM
        header += (1).to_bytes(2, "little")  # channels
        header += sample_rate.to_bytes(4, "little")
        header += (sample_rate * 2).to_bytes(4, "little")  # byte rate
        header += (2).to_bytes(2, "little")  # block align
        header += (16).to_bytes(2, "little")  # bits per sample
        header += b"data"
        header += data_size.to_bytes(4, "little")

        return header

    # テスト用メソッド

    def set_error_mode(self, enabled: bool) -> None:
        """エラーモードの設定（テスト用）

        Args:
            enabled: エラーモードの有効/無効
        """
        self.error_mode = enabled
        self.logger.debug(f"Error mode set to: {enabled}")

    def set_synthesis_delay(self, delay: float) -> None:
        """音声合成遅延の設定（テスト用）

        Args:
            delay: 遅延時間（秒）
        """
        self.synthesis_delay = max(0, delay)
        self.logger.debug(f"Synthesis delay set to: {delay}s")

    def add_custom_voice(self, voice_info: VoiceInfo) -> None:
        """カスタム音声の追加（テスト用）

        Args:
            voice_info: 音声情報
        """
        self.available_voices.append(voice_info)
        self.logger.debug(f"Custom voice added: {voice_info.id}")

    def get_statistics(self) -> Dict[str, Any]:
        """統計情報の取得（テスト用）

        Returns:
            Dict[str, Any]: 統計情報
        """
        return {
            "current_voice": self.current_voice_id,
            "available_voices_count": len(self.available_voices),
            "sample_rate": self.sample_rate,
            "error_mode": self.error_mode,
            "synthesis_delay": self.synthesis_delay,
            "last_parameters": self.last_parameters.__dict__ if self.last_parameters else None,
        }
