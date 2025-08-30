"""llm パッケージ - 大規模言語モデルエンジン

VioraTalkのLLM（Large Language Model）エンジン実装。
テキスト生成、対話処理、プロンプト処理などの機能を提供。

インポート規約 v1.1準拠
インターフェース定義書 v1.34準拠
開発規約書 v1.12準拠

Phase 4実装範囲：
- BaseLLMEngine: 抽象基底クラス
- GeminiEngine: Google Gemini API実装
- データクラス: LLMRequest, LLMResponse, LLMConfig

Phase 6実装予定：
- ClaudeEngine: Anthropic Claude API
- ChatGPTEngine: OpenAI ChatGPT API
- OllamaEngine: ローカルLLM

使用例:
    >>> from vioratalk.core.llm import GeminiEngine, LLMConfig
    >>> 
    >>> # 設定作成
    >>> config = LLMConfig(
    ...     engine="gemini",
    ...     model="gemini-pro",
    ...     temperature=0.7
    ... )
    >>> 
    >>> # エンジン初期化
    >>> engine = GeminiEngine(config=config)
    >>> await engine.initialize()
    >>> 
    >>> # テキスト生成
    >>> response = await engine.generate(
    ...     prompt="こんにちは、今日の天気はどうですか？",
    ...     system_prompt="あなたは親切なアシスタントです。"
    ... )
    >>> print(response.content)
    >>> 
    >>> # ストリーミング生成
    >>> async for chunk in engine.stream_generate(prompt="長い話をして"):
    ...     print(chunk, end="", flush=True)

Package Contents:
    BaseLLMEngine: LLMエンジンの抽象基底クラス
    LLMRequest: リクエストデータクラス
    LLMResponse: レスポンスデータクラス
    LLMConfig: 設定データクラス
    GeminiEngine: Gemini API実装
    AVAILABLE_MODELS: 利用可能なGeminiモデルリスト
    DEFAULT_MODEL: デフォルトモデル名
"""

# インポート規約 v1.1準拠：絶対インポート、パッケージレベル

# 基底クラスとデータクラス
from vioratalk.core.llm.base import BaseLLMEngine, LLMConfig, LLMRequest, LLMResponse

# GeminiEngine実装（Phase 4）
from vioratalk.core.llm.gemini_engine import AVAILABLE_MODELS, DEFAULT_MODEL, GeminiEngine

# Phase 6で追加予定
# from vioratalk.core.llm.claude_engine import ClaudeEngine
# from vioratalk.core.llm.chatgpt_engine import ChatGPTEngine
# from vioratalk.core.llm.ollama_engine import OllamaEngine


# ============================================================================
# エンジン管理ユーティリティ（Phase 4簡易版）
# ============================================================================


def get_llm_engine(engine_name: str = "gemini", **kwargs) -> BaseLLMEngine:
    """LLMエンジンのファクトリ関数

    指定された名前のLLMエンジンインスタンスを作成。

    Args:
        engine_name: エンジン名（'gemini', 'claude', 'chatgpt', 'ollama'）
        **kwargs: エンジン固有の設定

    Returns:
        BaseLLMEngine: LLMエンジンインスタンス

    Raises:
        ValueError: サポートされていないエンジン名

    Example:
        >>> engine = get_llm_engine("gemini", model="gemini-pro")
        >>> await engine.initialize()
    """
    engine_name = engine_name.lower()

    if engine_name == "gemini":
        return GeminiEngine(**kwargs)
    # Phase 6で追加
    # elif engine_name == "claude":
    #     return ClaudeEngine(**kwargs)
    # elif engine_name in ["chatgpt", "openai"]:
    #     return ChatGPTEngine(**kwargs)
    # elif engine_name == "ollama":
    #     return OllamaEngine(**kwargs)
    else:
        raise ValueError(
            f"Unsupported LLM engine: {engine_name}. " f"Currently supported: ['gemini']"
        )


def get_supported_engines() -> list[str]:
    """サポートされているエンジンのリスト取得

    Returns:
        list[str]: エンジン名のリスト

    Example:
        >>> engines = get_supported_engines()
        >>> print(engines)
        ['gemini']
    """
    # Phase 4では Gemini のみ
    return ["gemini"]
    # Phase 6で拡張
    # return ["gemini", "claude", "chatgpt", "openai", "ollama"]


def get_engine_info(engine_name: str) -> dict:
    """エンジン情報の取得

    Args:
        engine_name: エンジン名

    Returns:
        dict: エンジン情報（モデル、制限、特徴など）

    Example:
        >>> info = get_engine_info("gemini")
        >>> print(info["models"])
        ['gemini-pro', 'gemini-pro-vision', ...]
    """
    engine_name = engine_name.lower()

    if engine_name == "gemini":
        return {
            "name": "Google Gemini",
            "models": AVAILABLE_MODELS,
            "default_model": DEFAULT_MODEL,
            "max_tokens": {
                "gemini-pro": 30720,
                "gemini-1.5-pro": 1048576,
            },
            "features": [
                "text_generation",
                "streaming",
                "vision_support",  # gemini-pro-vision
                "long_context",  # gemini-1.5-pro
            ],
            "requires_api_key": True,
            "byok": True,
        }
    # Phase 6で追加
    else:
        raise ValueError(f"Unknown engine: {engine_name}")


# ============================================================================
# パッケージエクスポート定義（インポート規約 v1.1準拠）
# ============================================================================

__all__ = [
    # 基底クラスとデータ構造
    "BaseLLMEngine",
    "LLMRequest",
    "LLMResponse",
    "LLMConfig",
    # GeminiEngine実装
    "GeminiEngine",
    "AVAILABLE_MODELS",
    "DEFAULT_MODEL",
    # ユーティリティ関数
    "get_llm_engine",
    "get_supported_engines",
    "get_engine_info",
    # Phase 6で追加予定
    # 'ClaudeEngine',
    # 'ChatGPTEngine',
    # 'OllamaEngine',
]
