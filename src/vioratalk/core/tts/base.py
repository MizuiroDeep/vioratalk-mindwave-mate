"""
音声合成エンジンの基底クラスと関連データ構造

このモジュールは、すべてのTTSエンジンが継承すべき基底クラスと、
音声合成に関連するデータ構造を定義します。

インターフェース定義書 v1.34準拠
データフォーマット仕様書 v1.5準拠
開発規約書 v1.12準拠
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from vioratalk.core.base import VioraTalkComponent

# ============================================================================
# データ構造定義（データフォーマット仕様書 v1.5準拠）
# ============================================================================


@dataclass
class SynthesisResult:
    """音声合成結果を表すデータクラス

    データフォーマット仕様書 v1.5準拠
    TTSエンジンが返す音声合成結果を統一的に扱うための構造。

    Attributes:
        audio_data: 音声データ（バイナリ形式、WAVやPCMなど）
        sample_rate: サンプリングレート（Hz）
        duration: 音声の長さ（秒）
        format: 音声フォーマット（"wav", "mp3", "pcm", "direct_output"など）
        metadata: 追加メタデータ（エンジン固有情報、処理時間など）
    """

    audio_data: bytes
    sample_rate: int
    duration: float
    format: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """初期化後の検証処理"""
        if self.sample_rate <= 0:
            raise ValueError(f"Invalid sample_rate: {self.sample_rate}")
        if self.duration < 0:
            raise ValueError(f"Invalid duration: {self.duration}")

        # formatが"direct_output"の場合、audio_dataは空でもOK
        if self.format != "direct_output" and not self.audio_data:
            raise ValueError("audio_data is required for non-direct output")


@dataclass
class VoiceInfo:
    """利用可能な音声の情報

    データフォーマット仕様書 v1.5準拠
    各TTSエンジンで利用可能な音声の詳細情報。

    Attributes:
        id: 音声を識別する一意のID
        name: 音声の表示名
        language: 対応言語（ISO 639-1形式、例: "ja", "en"）
        gender: 性別（"male", "female", "neutral"）
        sample_url: サンプル音声のURL（オプション）
        metadata: エンジン固有の追加情報
    """

    id: str
    name: str
    language: str
    gender: str
    sample_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """文字列表現"""
        return f"{self.name} ({self.language}, {self.gender})"


@dataclass
class TTSConfig:
    """TTSエンジンの設定

    設定ファイル完全仕様書 v1.2準拠
    TTSエンジンの動作を制御する設定項目。

    Attributes:
        engine: エンジン名（"pyttsx3", "windows_sapi", "edge_tts"など）
        voice_id: デフォルト音声ID
        language: デフォルト言語（ISO 639-1形式）
        speed: 話速（1.0が標準）
        pitch: ピッチ（1.0が標準）
        volume: 音量（0.0-1.0）
        save_audio_data: 音声データを保存するか（True: WAV取得、False: 直接出力）
        output_format: 出力フォーマット（"wav", "mp3"など）
        cache_enabled: キャッシュの有効/無効
        engine_specific: エンジン固有の設定
    """

    engine: str = "pyttsx3"
    voice_id: Optional[str] = None
    language: str = "ja"
    speed: float = 1.0
    pitch: float = 1.0
    volume: float = 0.9
    save_audio_data: bool = False  # Phase 4 Part 39追加：デフォルトは直接出力
    output_format: str = "wav"
    cache_enabled: bool = False
    engine_specific: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """初期化後の検証処理"""
        # 値の範囲チェック
        if not 0.1 <= self.speed <= 3.0:
            raise ValueError(f"Invalid speed: {self.speed} (must be 0.1-3.0)")
        if not 0.5 <= self.pitch <= 2.0:
            raise ValueError(f"Invalid pitch: {self.pitch} (must be 0.5-2.0)")
        if not 0.0 <= self.volume <= 1.0:
            raise ValueError(f"Invalid volume: {self.volume} (must be 0.0-1.0)")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TTSConfig":
        """辞書から設定を作成

        Args:
            config_dict: 設定の辞書

        Returns:
            TTSConfig: 設定インスタンス
        """
        # 既知のフィールドのみを抽出
        known_fields = {
            "engine",
            "voice_id",
            "language",
            "speed",
            "pitch",
            "volume",
            "save_audio_data",
            "output_format",
            "cache_enabled",
            "engine_specific",
        }
        filtered_dict = {k: v for k, v in config_dict.items() if k in known_fields}
        return cls(**filtered_dict)


# ============================================================================
# 基底クラス定義
# ============================================================================


class BaseTTSEngine(VioraTalkComponent):
    """音声合成エンジンの基底インターフェース

    インターフェース定義書 v1.34準拠
    すべてのTTSエンジンはこのクラスを継承して実装する。

    実装時期:
    - Phase 3: MockTTSEngine（テスト用）
    - Phase 4: Pyttsx3Engine（フォールバック用）
    - Phase 6: WindowsSAPIEngine, EdgeTTSEngine, MultiTTSEngine
    - Phase 7-8: AivisSpeechEngine（高品質音声）

    Attributes:
        config: TTSエンジン設定
        available_voices: 利用可能な音声リスト
        current_voice_id: 現在選択されている音声ID
    """

    def __init__(self, config: Optional[TTSConfig] = None):
        """初期化

        Args:
            config: TTSエンジン設定（Noneの場合はデフォルト使用）
        """
        super().__init__()
        self.config = config or TTSConfig()
        self.available_voices: List[VoiceInfo] = []
        self.current_voice_id: Optional[str] = config.voice_id if config else None

    @abstractmethod
    async def synthesize(
        self, text: str, voice_id: Optional[str] = None, style: Optional[str] = None, **kwargs
    ) -> SynthesisResult:
        """テキストを音声に変換

        Args:
            text: 変換するテキスト
            voice_id: 音声ID（Noneの場合は現在の設定を使用）
            style: スタイル（Phase 7-8で実装）
            **kwargs: エンジン固有のパラメータ

        Returns:
            SynthesisResult: 音声合成結果

        Raises:
            TTSError: 音声合成エラー（E3000番台）
            InvalidVoiceError: 無効な音声ID（E3001）
        """
        pass

    @abstractmethod
    def get_available_voices(self) -> List[VoiceInfo]:
        """利用可能な音声のリストを取得

        Returns:
            List[VoiceInfo]: 音声情報のリスト
        """
        pass

    @abstractmethod
    def set_voice(self, voice_id: str) -> None:
        """使用する音声を設定

        Args:
            voice_id: 音声ID

        Raises:
            InvalidVoiceError: 無効な音声ID（E3001）
        """
        pass

    @abstractmethod
    async def test_availability(self) -> bool:
        """エンジンが利用可能かテスト

        Returns:
            bool: 利用可能な場合True

        Note:
            このメソッドは起動時の自動選択で使用される。
            実際の音声合成を試みて、エンジンの可用性を確認する。
        """
        pass

    def get_available_styles(self, voice_id: str) -> List[str]:
        """利用可能なスタイル一覧を取得（Phase 7-8で実装）

        Args:
            voice_id: 音声ID

        Returns:
            List[str]: スタイルIDのリスト

        Note:
            Phase 7-8のAivisSpeechEngineでオーバーライド。
            それまでは空リストを返す。
        """
        return []

    def get_config(self) -> TTSConfig:
        """現在の設定を取得

        Returns:
            TTSConfig: 現在の設定
        """
        return self.config

    def update_config(self, **kwargs) -> None:
        """設定を部分的に更新

        Args:
            **kwargs: 更新する設定項目
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
