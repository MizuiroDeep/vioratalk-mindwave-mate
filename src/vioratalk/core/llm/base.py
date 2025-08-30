"""base.py - LLMエンジンの基底クラスとデータ構造

大規模言語モデル（LLM）エンジンの抽象基底クラスと
関連データクラスの定義。

インターフェース定義書 v1.34準拠
データフォーマット仕様書 v1.5準拠
開発規約書 v1.12準拠
エラーハンドリング指針 v1.20準拠

Phase 4実装範囲：
- BaseLLMEngine抽象クラス
- LLMRequest/LLMResponseデータクラス
- LLMConfig設定クラス
"""

# 標準ライブラリ
import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

# プロジェクト内インポート
from vioratalk.core.base import VioraTalkComponent

# ============================================================================
# ロガー設定
# ============================================================================

logger = logging.getLogger(__name__)


# ============================================================================
# データクラス定義
# ============================================================================


@dataclass
class LLMRequest:
    """LLMリクエストデータ

    データフォーマット仕様書 v1.5準拠

    Attributes:
        prompt: ユーザープロンプト
        system_prompt: システムプロンプト（オプション）
        temperature: 生成の温度パラメータ（0.0-2.0）
        max_tokens: 最大生成トークン数
        stream: ストリーミング生成の有効/無効
        metadata: 追加メタデータ
        timestamp: リクエストタイムスタンプ
    """

    prompt: str
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換

        Returns:
            辞書形式のリクエストデータ
        """
        return {
            "prompt": self.prompt,
            "system_prompt": self.system_prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": self.stream,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }

    def validate(self) -> None:
        """リクエストデータの検証

        Raises:
            ValueError: 無効なパラメータの場合
        """
        if not self.prompt:
            raise ValueError("Prompt is required")

        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"Temperature must be between 0.0 and 2.0, got {self.temperature}")

        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")


@dataclass
class LLMResponse:
    """LLM応答データ

    データフォーマット仕様書 v1.5準拠
    インターフェース定義書 v1.34準拠

    Attributes:
        content: 生成されたテキスト
        usage: トークン使用量
        model: 使用されたモデル名
        finish_reason: 生成終了理由
        metadata: 追加メタデータ
        timestamp: 応答タイムスタンプ
    """

    content: str
    usage: Dict[str, int]  # tokens_used, prompt_tokens, completion_tokens等
    model: str
    finish_reason: str  # "stop", "length", "error"等
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換

        Returns:
            辞書形式の応答データ
        """
        return {
            "content": self.content,
            "usage": self.usage,
            "model": self.model,
            "finish_reason": self.finish_reason,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_api_response(cls, response: Dict[str, Any], model: str) -> "LLMResponse":
        """API応答から変換

        Args:
            response: API応答データ
            model: 使用モデル名

        Returns:
            LLMResponseインスタンス
        """
        # 各APIの形式に応じて変換（実装時に詳細化）
        return cls(
            content=response.get("content", ""),
            usage=response.get("usage", {}),
            model=model,
            finish_reason=response.get("finish_reason", "stop"),
            metadata=response.get("metadata", {}),
        )


@dataclass
class LLMConfig:
    """LLMエンジン設定

    データフォーマット仕様書 v1.5準拠

    Attributes:
        engine: エンジン名（'gemini', 'claude', 'chatgpt', 'ollama'）
        model: モデル名
        api_key: APIキー（BYOK対応）
        temperature: デフォルト温度
        max_tokens: デフォルト最大トークン数
        timeout: タイムアウト秒数
        retry_count: リトライ回数
        streaming: ストリーミングのデフォルト設定
    """

    engine: str = "gemini"
    model: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: float = 60.0
    retry_count: int = 3
    streaming: bool = False

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LLMConfig":
        """辞書から生成（設定ファイル読み込み用）

        Args:
            config_dict: 設定辞書

        Returns:
            LLMConfigインスタンス
        """
        # 既知のフィールドのみを抽出
        known_fields = {
            "engine",
            "model",
            "api_key",
            "temperature",
            "max_tokens",
            "timeout",
            "retry_count",
            "streaming",
        }
        filtered_dict = {k: v for k, v in config_dict.items() if k in known_fields}
        return cls(**filtered_dict)


# ============================================================================
# 基底クラス定義
# ============================================================================


class BaseLLMEngine(VioraTalkComponent):
    """LLMエンジンの基底インターフェース

    インターフェース定義書 v1.34準拠
    すべてのLLMエンジンはこのクラスを継承して実装する。

    実装クラス名:
    - GeminiEngine    # Phase 4の主要エンジン
    - ClaudeEngine    # Phase 6
    - ChatGPTEngine   # Phase 6
    - OllamaEngine    # Phase 6

    Attributes:
        config: LLMエンジン設定
        available_models: 利用可能なモデルリスト
        current_model: 現在のモデル名
        max_tokens: 最大トークン数
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        """初期化

        Args:
            config: LLMエンジン設定（Noneの場合はデフォルト使用）
        """
        super().__init__()
        self.config = config or LLMConfig()
        self.available_models: List[str] = []
        self.current_model: Optional[str] = None
        self.max_tokens: int = self.config.max_tokens

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """テキスト生成

        Args:
            prompt: ユーザープロンプト
            system_prompt: システムプロンプト（オプション）
            temperature: 生成の温度パラメータ
            max_tokens: 最大生成トークン数

        Returns:
            LLMResponse: 生成結果

        Raises:
            LLMError: LLM処理エラー（E2000番台）
            APIError: API通信エラー（E2001）
            RateLimitError: レート制限（E2002）
            AuthenticationError: 認証エラー（E2003）
        """
        pass

    @abstractmethod
    async def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """ストリーミング生成

        Args:
            prompt: ユーザープロンプト
            system_prompt: システムプロンプト（オプション）
            temperature: 生成の温度パラメータ
            max_tokens: 最大生成トークン数

        Yields:
            str: 生成されたテキストの断片

        Raises:
            LLMError: LLM処理エラー
            APIError: API通信エラー
        """
        pass

    @abstractmethod
    def get_available_models(self) -> List[str]:
        """利用可能なモデルのリスト取得

        Returns:
            List[str]: モデル名のリスト

        Example:
            ["gemini-pro", "gemini-pro-vision", "gemini-1.5-pro"]
        """
        pass

    @abstractmethod
    def set_model(self, model_name: str) -> None:
        """使用するモデルを設定

        Args:
            model_name: モデル名

        Raises:
            ModelNotFoundError: 指定されたモデルが存在しない（E2004）
        """
        pass

    @abstractmethod
    def get_max_tokens(self) -> int:
        """最大トークン数を取得

        Returns:
            int: 現在のモデルの最大トークン数
        """
        pass

    # ========================================================================
    # 共通ヘルパーメソッド（Phase 4実装）
    # ========================================================================

    def validate_request(self, request: LLMRequest) -> None:
        """リクエストの検証

        Args:
            request: LLMリクエスト

        Raises:
            ValueError: 無効なリクエスト
        """
        request.validate()

        # モデル固有の検証
        if request.max_tokens and request.max_tokens > self.get_max_tokens():
            logger.warning(
                f"Requested tokens ({request.max_tokens}) exceeds model limit "
                f"({self.get_max_tokens()}). Using model limit."
            )

    async def prepare_request(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMRequest:
        """リクエストの準備

        Args:
            prompt: ユーザープロンプト
            system_prompt: システムプロンプト
            temperature: 温度パラメータ
            max_tokens: 最大トークン数

        Returns:
            LLMRequest: 準備されたリクエスト
        """
        request = LLMRequest(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens or self.config.max_tokens,
        )

        self.validate_request(request)
        return request

    def format_messages(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """メッセージ形式に変換（ChatML形式）

        Args:
            prompt: ユーザープロンプト
            system_prompt: システムプロンプト

        Returns:
            List[Dict[str, str]]: メッセージリスト
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        return messages


# ============================================================================
# エクスポート
# ============================================================================

__all__ = [
    "BaseLLMEngine",
    "LLMRequest",
    "LLMResponse",
    "LLMConfig",
]
