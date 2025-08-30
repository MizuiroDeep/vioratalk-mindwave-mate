"""
tts パッケージ - 音声合成エンジン

VioraTalkのTTS（Text-to-Speech）エンジン実装。
テキストから音声への変換機能を提供。

インポート規約 v1.1準拠
インターフェース定義書 v1.34準拠
開発規約書 v1.12準拠

Phase 4実装範囲：
- BaseTTSEngine: 抽象基底クラス
- Pyttsx3Engine: pyttsx3実装（フォールバック用）
- データクラス: SynthesisResult, VoiceInfo, TTSConfig

Phase 6実装予定：
- WindowsSAPIEngine: Windows SAPI実装
- EdgeTTSEngine: Microsoft Edge TTS実装
- MultiTTSEngine: 複数エンジン管理

Phase 7-8実装予定：
- AivisSpeechEngine: 高品質音声合成

使用例:
    >>> from vioratalk.core.tts import Pyttsx3Engine, TTSConfig
    >>> 
    >>> # 設定作成
    >>> config = TTSConfig(
    ...     engine="pyttsx3",
    ...     language="ja",
    ...     save_audio_data=False  # 直接出力モード
    ... )
    >>> 
    >>> # エンジン初期化
    >>> engine = Pyttsx3Engine(config=config)
    >>> await engine.initialize()
    >>> 
    >>> # 音声合成（スピーカー出力）
    >>> result = await engine.synthesize("こんにちは、今日は良い天気ですね。")
    >>> print(f"Duration: {result.duration}s")
    >>> 
    >>> # 音声データ取得モード
    >>> result = await engine.synthesize(
    ...     "テスト音声です",
    ...     save_audio=True
    ... )
    >>> print(f"Audio data size: {len(result.audio_data)} bytes")

Package Contents:
    BaseTTSEngine: TTSエンジンの抽象基底クラス
    SynthesisResult: 音声合成結果データクラス
    VoiceInfo: 音声情報データクラス
    TTSConfig: TTS設定データクラス
    Pyttsx3Engine: pyttsx3実装
    SUPPORTED_LANGUAGES: サポート言語リスト
    DEFAULT_VOICE: デフォルト音声設定
"""

# インポート規約 v1.1準拠：絶対インポート、パッケージレベル

from typing import Dict, List, Optional, Type

# 基底クラスとデータクラス
from vioratalk.core.tts.base import BaseTTSEngine, SynthesisResult, TTSConfig, VoiceInfo

# Pyttsx3Engine実装（Phase 4）
from vioratalk.core.tts.pyttsx3_engine import Pyttsx3Engine

# Phase 6で追加予定
# from vioratalk.core.tts.windows_sapi_engine import WindowsSAPIEngine
# from vioratalk.core.tts.edge_tts_engine import EdgeTTSEngine
# from vioratalk.core.tts.multi_tts_engine import MultiTTSEngine

# Phase 7-8で追加予定
# from vioratalk.core.tts.aivisspeech_engine import AivisSpeechEngine


# ============================================================================
# 定数定義
# ============================================================================

# サポートする言語（Phase 4時点）
SUPPORTED_LANGUAGES: List[str] = ["ja", "en"]

# デフォルト音声設定
DEFAULT_VOICE: Dict[str, str] = {"ja": "Japanese", "en": "English"}

# 利用可能なエンジン（Phase 4時点）
AVAILABLE_ENGINES: List[str] = ["pyttsx3"]


# ============================================================================
# エンジン管理ユーティリティ（Phase 4簡易版）
# ============================================================================


def get_tts_engine(
    engine_name: str = "pyttsx3", config: Optional[TTSConfig] = None, **kwargs
) -> BaseTTSEngine:
    """TTSエンジンのファクトリ関数

    指定された名前のTTSエンジンインスタンスを作成。

    Args:
        engine_name: エンジン名（'pyttsx3', 'sapi', 'edge', 'aivis'）
        config: TTS設定（省略時はデフォルト設定）
        **kwargs: エンジン固有の追加設定

    Returns:
        BaseTTSEngine: TTSエンジンインスタンス

    Raises:
        ValueError: サポートされていないエンジン名
        TTSError: エンジン初期化失敗

    Example:
        >>> # デフォルト（pyttsx3）
        >>> engine = get_tts_engine()
        >>>
        >>> # 設定指定
        >>> config = TTSConfig(save_audio_data=True)
        >>> engine = get_tts_engine("pyttsx3", config=config)
    """
    engine_name = engine_name.lower()

    # 設定がない場合はデフォルト作成
    if config is None:
        config = TTSConfig(engine=engine_name, **kwargs)

    if engine_name == "pyttsx3":
        return Pyttsx3Engine(config=config)
    # Phase 6で追加
    # elif engine_name in ["sapi", "windows_sapi"]:
    #     return WindowsSAPIEngine(config=config)
    # elif engine_name in ["edge", "edge_tts"]:
    #     return EdgeTTSEngine(config=config)
    # Phase 7-8で追加
    # elif engine_name in ["aivis", "aivisspeech"]:
    #     return AivisSpeechEngine(config=config)
    else:
        raise ValueError(
            f"Unsupported TTS engine: {engine_name}. "
            f"Available engines: {', '.join(AVAILABLE_ENGINES)}"
        )


async def test_tts_engines() -> Dict[str, bool]:
    """利用可能なTTSエンジンをテスト

    各TTSエンジンの可用性をテストし、結果を返す。
    アプリケーション起動時の自動選択で使用。

    Returns:
        Dict[str, bool]: エンジン名と可用性のマップ

    Example:
        >>> results = await test_tts_engines()
        >>> print(results)
        {'pyttsx3': True, 'sapi': False, 'edge': False}
    """
    results = {}

    # Pyttsx3のテスト
    try:
        engine = Pyttsx3Engine()
        results["pyttsx3"] = await engine.test_availability()
    except Exception:
        results["pyttsx3"] = False

    # Phase 6で追加
    # try:
    #     engine = WindowsSAPIEngine()
    #     results["sapi"] = await engine.test_availability()
    # except Exception:
    #     results["sapi"] = False

    return results


def get_engine_for_platform() -> str:
    """プラットフォームに最適なエンジンを選択

    OSとインストール状況に基づいて最適なTTSエンジンを選択。

    Returns:
        str: 推奨エンジン名

    Example:
        >>> engine_name = get_engine_for_platform()
        >>> engine = get_tts_engine(engine_name)
    """
    import sys

    # Windows環境
    if sys.platform == "win32":
        # Phase 6以降はSAPI → Edge → Pyttsx3の順で試す
        # 現在はPyttsx3のみ
        return "pyttsx3"

    # macOS/Linux環境
    else:
        # Phase 6以降はEdge → Pyttsx3の順で試す
        # 現在はPyttsx3のみ
        return "pyttsx3"


# ============================================================================
# 公開API定義（インポート規約 v1.1準拠）
# ============================================================================

__all__ = [
    # 基底クラス
    "BaseTTSEngine",
    # データクラス
    "SynthesisResult",
    "VoiceInfo",
    "TTSConfig",
    # エンジン実装
    "Pyttsx3Engine",
    # ユーティリティ関数
    "get_tts_engine",
    "test_tts_engines",
    "get_engine_for_platform",
    # 定数
    "SUPPORTED_LANGUAGES",
    "DEFAULT_VOICE",
    "AVAILABLE_ENGINES",
]
