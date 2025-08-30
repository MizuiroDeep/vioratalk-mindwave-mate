"""VioraTalk STT (Speech-to-Text) パッケージ

音声認識エンジンの実装を提供。
Phase 4でFasterWhisperEngineを実装。

インポート規約 v1.1準拠
開発規約書 v1.12準拠
"""

# 基底クラスとデータ構造のインポート
from vioratalk.core.stt.base import (
    AudioData,
    AudioMetadata,
    BaseSTTEngine,
    STTConfig,
    TranscriptionResult,
)

# FasterWhisperEngineのインポート（利用可能な場合のみ）
try:
    from vioratalk.core.stt.faster_whisper_engine import (
        FASTER_WHISPER_AVAILABLE,
        SUPPORTED_MODELS,
        FasterWhisperEngine,
        create_faster_whisper_engine,
    )

    _has_faster_whisper = True
except ImportError:
    # faster-whisperがインストールされていない場合
    _has_faster_whisper = False
    FASTER_WHISPER_AVAILABLE = False
    SUPPORTED_MODELS = {}

    # ダミークラスを定義（エラーメッセージ改善のため）
    class FasterWhisperEngine:
        """FasterWhisperEngine placeholder when faster-whisper is not installed"""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "faster-whisper is not installed. "
                "Please install it with: pip install faster-whisper"
            )

    def create_faster_whisper_engine(*args, **kwargs):
        """Factory function placeholder"""
        raise ImportError(
            "faster-whisper is not installed. " "Please install it with: pip install faster-whisper"
        )


# ============================================================================
# デフォルトエンジンの設定
# ============================================================================


def get_default_stt_engine(config: dict = None) -> BaseSTTEngine:
    """デフォルトのSTTエンジンを取得

    利用可能なエンジンから自動的に選択。
    Phase 4ではFasterWhisperEngineのみ。

    Args:
        config: エンジン設定（オプション）

    Returns:
        BaseSTTEngine: STTエンジンインスタンス

    Raises:
        ImportError: 利用可能なエンジンがない場合
    """
    # FasterWhisperが利用可能な場合
    if _has_faster_whisper:
        return create_faster_whisper_engine(config)

    # 利用可能なエンジンがない
    raise ImportError(
        "No STT engine available. Please install faster-whisper: " "pip install faster-whisper"
    )


# ============================================================================
# エンジン登録（将来の拡張用）
# ============================================================================

# 利用可能なエンジンのレジストリ
_AVAILABLE_ENGINES = {}

if _has_faster_whisper:
    _AVAILABLE_ENGINES["faster-whisper"] = FasterWhisperEngine


def get_available_engines() -> dict:
    """利用可能なSTTエンジンの一覧を取得

    Returns:
        dict: エンジン名とクラスのマッピング
    """
    return _AVAILABLE_ENGINES.copy()


def register_engine(name: str, engine_class: type) -> None:
    """カスタムSTTエンジンを登録

    Args:
        name: エンジン名
        engine_class: エンジンクラス（BaseSTTEngineを継承）

    Raises:
        TypeError: BaseSTTEngineを継承していない場合
        ValueError: 既に登録済みの名前の場合
    """
    if not issubclass(engine_class, BaseSTTEngine):
        raise TypeError(f"{engine_class.__name__} must inherit from BaseSTTEngine")

    if name in _AVAILABLE_ENGINES:
        raise ValueError(f"Engine '{name}' is already registered")

    _AVAILABLE_ENGINES[name] = engine_class


def create_engine(engine_name: str, config: dict = None) -> BaseSTTEngine:
    """指定された名前のエンジンを作成

    Args:
        engine_name: エンジン名
        config: エンジン設定

    Returns:
        BaseSTTEngine: エンジンインスタンス

    Raises:
        ValueError: 指定されたエンジンが見つからない場合
    """
    if engine_name not in _AVAILABLE_ENGINES:
        available = list(_AVAILABLE_ENGINES.keys())
        raise ValueError(f"Unknown engine '{engine_name}'. " f"Available engines: {available}")

    engine_class = _AVAILABLE_ENGINES[engine_name]

    # STTConfigの作成
    from vioratalk.core.stt.base import STTConfig

    stt_config = STTConfig.from_dict(config) if config else STTConfig()

    return engine_class(stt_config)


# ============================================================================
# 公開API定義
# ============================================================================

__all__ = [
    # 基底クラスとデータ構造
    "BaseSTTEngine",
    "AudioData",
    "AudioMetadata",
    "TranscriptionResult",
    "STTConfig",
    # ヘルパー関数
    "get_default_stt_engine",
    "get_available_engines",
    "register_engine",
    "create_engine",
    # 定数
    "FASTER_WHISPER_AVAILABLE",
    "SUPPORTED_MODELS",
]

# FasterWhisperEngineが利用可能な場合は追加
if _has_faster_whisper:
    __all__.extend(
        [
            "FasterWhisperEngine",
            "create_faster_whisper_engine",
        ]
    )


# ============================================================================
# パッケージ情報
# ============================================================================

__version__ = "0.3.0"  # Phase 4
__author__ = "VioraTalk Development Team"
__license__ = "MIT"
