"""STTエンジン基底クラスとデータ構造定義

音声認識エンジンの抽象基底クラスと関連データ構造を定義。
すべてのSTTエンジン実装はこのインターフェースに準拠する。

インターフェース定義書 v1.34準拠
データフォーマット仕様書 v1.5準拠
開発規約書 v1.12準拠
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

# プロジェクト内インポート（絶対インポート）
from vioratalk.core.base import VioraTalkComponent

# ============================================================================
# データ構造定義
# ============================================================================


@dataclass
class AudioMetadata:
    """音声メタデータ

    データフォーマット仕様書 v1.5準拠
    音声データの属性情報を保持
    """

    sample_rate: int = 16000  # サンプリングレート（Hz）
    channels: int = 1  # チャンネル数（1: モノラル, 2: ステレオ）
    bit_depth: int = 16  # ビット深度
    duration: float = 0.0  # 音声の長さ（秒）
    format: str = "pcm"  # 音声フォーマット: "pcm", "wav", "mp3"
    filename: Optional[str] = None  # 元のファイル名（あれば）


@dataclass
class AudioData:
    """音声データコンテナ

    Phase間での音声データ受け渡しに使用。
    生データ（numpy配列）またはエンコード済みバイト列を保持。

    Attributes:
        raw_data: PCMデータ（float32, -1.0 to 1.0）
        encoded_data: エンコード済みデータ
        metadata: 音声メタデータ
        timestamp: タイムスタンプ
        source: データソース（"microphone", "file", "synthesis"）
    """

    # 音声データ本体（いずれか一つ）
    raw_data: Optional[np.ndarray] = None  # PCMデータ（float32, -1.0 to 1.0）
    encoded_data: Optional[bytes] = None  # エンコード済みデータ

    # メタデータ
    metadata: AudioMetadata = field(default_factory=AudioMetadata)

    # 処理情報
    timestamp: Optional[datetime] = None
    source: str = ""  # "microphone", "file", "synthesis"

    def __post_init__(self):
        """初期化後の処理"""
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

        # データ検証
        if self.raw_data is None and self.encoded_data is None:
            raise ValueError("Either raw_data or encoded_data must be provided")

        # durationの自動計算
        if self.raw_data is not None and self.metadata.duration == 0.0:
            self.metadata.duration = len(self.raw_data) / self.metadata.sample_rate

    def to_bytes(self) -> bytes:
        """バイト列に変換

        Returns:
            bytes: 音声データのバイト列表現

        Raises:
            ValueError: 音声データが存在しない場合
        """
        if self.encoded_data is not None:
            return self.encoded_data
        elif self.raw_data is not None:
            # PCMデータをバイト列に変換
            return (self.raw_data * 32767).astype(np.int16).tobytes()
        else:
            raise ValueError("No audio data available")


@dataclass
class TranscriptionResult:
    """音声認識結果

    STTエンジンの出力形式。
    認識テキストと付加情報を含む。

    Attributes:
        text: 認識されたテキスト
        confidence: 認識信頼度（0.0-1.0）
        language: 認識言語コード（ISO 639-1）
        duration: 処理時間（秒）
        alternatives: 代替候補リスト
        timestamp: 処理タイムスタンプ
    """

    text: str
    confidence: float
    language: str
    duration: float
    alternatives: List[Dict[str, float]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """初期化後の検証"""
        # 信頼度の範囲チェック
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

        # 言語コードの基本チェック（2文字または5文字）
        if len(self.language) not in [2, 5]:
            raise ValueError(f"Invalid language code: {self.language}")


@dataclass
class STTConfig:
    """STTエンジン設定

    データフォーマット仕様書 v1.5準拠
    STTエンジンの動作パラメータを定義

    Attributes:
        engine: エンジン名
        model: モデル名/サイズ
        language: デフォルト言語
        device: デバイス（"cpu", "cuda", "auto"）
        vad_threshold: 音声検出閾値
        max_recording_duration: 最大録音時間（秒）
        model_path: モデルファイルパス（オプション）
        compute_type: 計算精度（"int8", "float16", "float32"）
    """

    engine: str = "faster-whisper"
    model: str = "base"
    language: str = "ja"
    device: str = "auto"
    vad_threshold: float = 0.5
    max_recording_duration: int = 30
    model_path: Optional[str] = None
    compute_type: str = "int8"  # faster-whisper用

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "STTConfig":
        """辞書から生成

        Args:
            config_dict: 設定辞書

        Returns:
            STTConfig: 設定インスタンス
        """
        # 既知のフィールドのみを抽出
        known_fields = {
            "engine",
            "model",
            "language",
            "device",
            "vad_threshold",
            "max_recording_duration",
            "model_path",
            "compute_type",
        }
        filtered_dict = {k: v for k, v in config_dict.items() if k in known_fields}
        return cls(**filtered_dict)


# ============================================================================
# 基底クラス定義
# ============================================================================


class BaseSTTEngine(VioraTalkComponent):
    """音声認識エンジンの基底インターフェース

    インターフェース定義書 v1.34準拠
    すべてのSTTエンジンはこのクラスを継承して実装する。

    実装時期:
    - Phase 3: MockSTTEngine（テスト用）
    - Phase 4: FasterWhisperEngine（本実装）
    - Phase 6: MultiSTTEngine（複数エンジン管理）

    Attributes:
        config: STTエンジン設定
        supported_languages: サポート言語リスト
        current_model: 現在のモデル名
    """

    def __init__(self, config: Optional[STTConfig] = None):
        """初期化

        Args:
            config: STTエンジン設定（Noneの場合はデフォルト使用）
        """
        super().__init__()
        self.config = config or STTConfig()
        self.supported_languages: List[str] = []
        self.current_model: Optional[str] = None

    @abstractmethod
    async def transcribe(
        self, audio_data: AudioData, language: Optional[str] = None
    ) -> TranscriptionResult:
        """音声をテキストに変換

        Args:
            audio_data: 音声データ
            language: 認識言語（Noneの場合は自動検出またはデフォルト使用）

        Returns:
            TranscriptionResult: 認識結果

        Raises:
            STTError: 音声認識エラー（E1000番台）
            AudioError: 音声データエラー（E1001）
            TimeoutError: タイムアウト（E1004）
        """
        pass

    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """サポートする言語のリスト取得

        Returns:
            List[str]: 言語コードのリスト（ISO 639-1形式）

        Example:
            ["ja", "en", "zh", "ko"]
        """
        pass

    @abstractmethod
    def set_model(self, model_name: str) -> None:
        """使用するモデルを設定

        Args:
            model_name: モデル名（"tiny", "base", "small", "medium", "large"）

        Raises:
            ModelNotFoundError: モデルが見つからない（E2100）
            ValueError: 無効なモデル名
        """
        pass

    async def is_available(self) -> bool:
        """エンジンが利用可能かチェック

        Returns:
            bool: 利用可能な場合True

        Note:
            デフォルト実装では常にTrueを返す。
            必要に応じてオーバーライド。
        """
        return self._state.is_operational()

    def get_model_info(self) -> Dict[str, Any]:
        """現在のモデル情報を取得

        Returns:
            Dict[str, Any]: モデル情報

        Example:
            {
                "name": "base",
                "size_mb": 142,
                "languages": ["ja", "en"],
                "parameters": "74M"
            }
        """
        return {
            "name": self.current_model,
            "engine": self.config.engine,
            "device": self.config.device,
            "languages": self.supported_languages,
        }

    async def validate_audio(self, audio_data: AudioData) -> bool:
        """音声データの妥当性チェック

        Args:
            audio_data: チェック対象の音声データ

        Returns:
            bool: 妥当な場合True

        Note:
            最小限のチェックを実装。
            必要に応じてオーバーライド。
        """
        # 基本的な検証
        if audio_data is None:
            return False

        # データの存在確認
        has_data = audio_data.raw_data is not None or audio_data.encoded_data is not None

        # メタデータの妥当性
        valid_metadata = (
            audio_data.metadata.sample_rate > 0
            and audio_data.metadata.channels in [1, 2]
            and audio_data.metadata.bit_depth in [8, 16, 24, 32]
        )

        return has_data and valid_metadata


# ============================================================================
# エクスポート定義
# ============================================================================

__all__ = [
    "BaseSTTEngine",
    "AudioData",
    "AudioMetadata",
    "TranscriptionResult",
    "STTConfig",
]
