"""gemini_engine.py - Gemini APIを使用したLLMエンジン（新SDK対応）

Google Gemini APIを使用した大規模言語モデルエンジンの実装。
BYOK（Bring Your Own Key）対応。
google-genai SDK使用（2025年8月版）。

インターフェース定義書 v1.34準拠
API通信実装ガイド v1.4準拠
エラーハンドリング指針 v1.20準拠
開発規約書 v1.12準拠

Phase 4実装範囲：
- GeminiEngine本体
- generate/stream_generate実装
- APIキー管理（credential_manager連携）
- エラーハンドリング
"""

# 標準ライブラリ
import asyncio
import logging
from datetime import datetime
from typing import Any, AsyncGenerator, List, Optional

# サードパーティライブラリ - 新SDK
try:
    from google import genai
    from google.genai import types

    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None
    types = None

# プロジェクト内インポート
from vioratalk.core.base import ComponentState
from vioratalk.core.error_handler import get_default_error_handler
from vioratalk.core.exceptions import (
    APIError,
    AuthenticationError,
    ConfigurationError,
    LLMError,
    ModelNotFoundError,
    RateLimitError,
)
from vioratalk.core.llm.base import BaseLLMEngine, LLMConfig, LLMRequest, LLMResponse
from vioratalk.infrastructure.credential_manager import get_api_key_manager
from vioratalk.utils.progress import ProgressBar

# ============================================================================
# ロガー設定
# ============================================================================

logger = logging.getLogger(__name__)


# ============================================================================
# 定数定義
# ============================================================================

# Geminiモデル定義（2025年8月現在）
AVAILABLE_MODELS = [
    "gemini-2.0-flash",  # 無料枠最大（1日100万トークン）
    "gemini-2.5-flash",  # 高速・バランス型（1日25万トークン）
    "gemini-2.5-flash-lite",  # 最軽量・最安価
    "gemini-2.5-pro",  # 最高性能
    # レガシーモデル（2025年9月廃止予定）
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

# モデル別最大トークン数
MODEL_MAX_TOKENS = {
    "gemini-2.0-flash": 1048576,  # 1M
    "gemini-2.5-flash": 1048576,  # 1M
    "gemini-2.5-flash-lite": 1048576,  # 1M
    "gemini-2.5-pro": 1048576,  # 1M
    # レガシー
    "gemini-1.5-flash": 1048576,
    "gemini-1.5-pro": 1048576,
}

# デフォルトモデル（無料枠優先）
DEFAULT_MODEL = "gemini-2.0-flash"

# API設定
GEMINI_API_TIMEOUT = 60.0  # 秒
GEMINI_STREAMING_TIMEOUT = 300.0  # ストリーミング用


# ============================================================================
# GeminiEngineクラス
# ============================================================================


class GeminiEngine(BaseLLMEngine):
    """Gemini APIを使用したLLMエンジン（新SDK対応）

    Google Gemini APIとの通信を管理し、テキスト生成機能を提供。
    BYOK（Bring Your Own Key）対応で、APIキーは外部から設定可能。

    Attributes:
        config: LLMエンジン設定
        api_key: Gemini APIキー
        client: Gemini APIクライアント
        error_handler: エラーハンドラー
    """

    def __init__(self, config: Optional[LLMConfig] = None, api_key: Optional[str] = None):
        """初期化

        Args:
            config: LLMエンジン設定
            api_key: Gemini APIキー（Noneの場合はcredential_managerから取得）

        Raises:
            ConfigurationError: google-genaiが未インストール（E2010）
        """
        logger.debug("GeminiEngine インスタンス作成")

        # google-genaiライブラリの確認
        if not GENAI_AVAILABLE:
            error_msg = (
                "google-genai is not installed. " "Please install it with: pip install google-genai"
            )
            logger.error(f"[E2010] {error_msg}")
            raise ConfigurationError(error_msg, error_code="E2010")

        # 設定の初期化
        config = config or LLMConfig(engine="gemini", model=DEFAULT_MODEL)

        # 基底クラスの初期化
        super().__init__(config)

        # APIキーの取得（BYOK対応）
        if api_key is None:
            api_key_manager = get_api_key_manager()
            api_key = api_key_manager.get_api_key("gemini")

        self.api_key = api_key

        # Gemini API設定
        self.client = None
        self.error_handler = get_default_error_handler()

        # モデル設定
        self.available_models = AVAILABLE_MODELS.copy()
        self.current_model = config.model or DEFAULT_MODEL
        self.max_tokens = MODEL_MAX_TOKENS.get(self.current_model, 1048576)

        # APIキーが設定されている場合は初期化
        if self.api_key:
            self._initialize_client()

        logger.info(f"GeminiEngine initialized with model: {self.current_model}")

    def _initialize_client(self) -> None:
        """Geminiクライアントの初期化（新SDK）"""
        if not self.api_key:
            return

        try:
            # 新SDK: genai.Client使用
            self.client = genai.Client(api_key=self.api_key)
            logger.debug("Gemini client initialized successfully")
        except Exception as e:
            error_msg = f"Failed to initialize Gemini client: {e}"
            logger.error(f"[E2011] {error_msg}")
            self.error_handler.handle_error(
                APIError(error_msg, error_code="E2011"),
                context={"component": "GeminiEngine", "operation": "initialize_client"},
            )

    async def initialize(self) -> None:
        """非同期初期化

        Raises:
            AuthenticationError: APIキー未設定または無効（E2003）
        """
        # 状態を初期化中に設定
        self._state = ComponentState.INITIALIZING

        await super().initialize()

        # APIキーの確認
        if not self.api_key:
            api_key_manager = get_api_key_manager()
            self.api_key = api_key_manager.require_api_key("gemini")

        # クライアントの初期化
        if not self.client:
            self._initialize_client()

        # APIキーの検証（テスト生成）
        try:
            await self._test_connection()
            # 成功時は状態をREADYに設定
            self._state = ComponentState.READY
            self._initialized_at = datetime.now()
            logger.info("GeminiEngine initialization completed")
        except Exception as e:
            error_msg = f"API key validation failed: {e}"
            logger.error(f"[E2003] {error_msg}")
            self._state = ComponentState.ERROR
            raise AuthenticationError(error_msg)

    async def _test_connection(self) -> None:
        """接続テスト（新SDK対応）

        非同期処理実装ガイド v1.1準拠：
        CPUバウンドタスクを別スレッドで実行するパターン
        """
        try:
            # 新SDK: 設定オブジェクトを使用
            config = types.GenerateContentConfig(
                system_instruction="You are a helpful assistant.",
                max_output_tokens=10,
                temperature=0.0,
            )

            # 同期処理用の内部関数を定義
            def test_sync():
                """同期的な接続テスト"""
                response = self.client.models.generate_content(
                    model=self.current_model, contents="Hello, this is a test", config=config
                )
                return response

            # 別スレッドで実行
            response = await asyncio.get_event_loop().run_in_executor(None, test_sync)

            # レスポンス確認
            if response and hasattr(response, "text"):
                logger.debug("Connection test successful")
            else:
                logger.warning("Connection test returned empty response")

        except Exception as e:
            logger.warning(f"Connection test failed: {e}")
            raise

    # ========================================================================
    # BaseLLMEngineインターフェース実装
    # ========================================================================

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """テキスト生成（新SDK対応）

        Args:
            prompt: ユーザープロンプト
            system_prompt: システムプロンプト（オプション）
            temperature: 生成の温度パラメータ（0.0-2.0）
            max_tokens: 最大生成トークン数

        Returns:
            LLMResponse: 生成結果

        Raises:
            LLMError: 生成処理エラー（E2000）
            APIError: API通信エラー（E2001）
            RateLimitError: レート制限（E2002）
            AuthenticationError: 認証エラー（E2003）
        """
        # リクエスト準備
        request = await self.prepare_request(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # クライアント確認
        if not self.client:
            raise LLMError(
                "Gemini client not initialized. Call initialize() first.", error_code="E2000"
            )

        try:
            # プロンプトの結合
            full_prompt = self._combine_prompts(prompt, system_prompt)

            # 新SDK: GenerateContentConfig使用
            generation_config = types.GenerateContentConfig(
                system_instruction=system_prompt if system_prompt else None,
                temperature=temperature,
                max_output_tokens=max_tokens or self.config.max_tokens,
                candidate_count=1,
            )

            # 同期処理用の内部関数
            def generate_sync():
                """同期的な生成処理"""
                return self.client.models.generate_content(
                    model=self.current_model, contents=full_prompt, config=generation_config
                )

            # API呼び出し（同期処理を非同期で実行）
            logger.debug(f"Generating with model {self.current_model}")
            response = await asyncio.get_event_loop().run_in_executor(None, generate_sync)

            # レスポンスの処理
            return self._process_response(response, request)

        except Exception as e:
            # エラーハンドリング
            self._handle_generation_error(e, request)

    async def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """ストリーミング生成（新SDK対応）

        Args:
            prompt: ユーザープロンプト
            system_prompt: システムプロンプト（オプション）
            temperature: 生成の温度パラメータ
            max_tokens: 最大生成トークン数

        Yields:
            str: 生成されたテキストの断片

        Raises:
            LLMError: 生成処理エラー
            APIError: API通信エラー
        """
        # クライアント確認
        if not self.client:
            raise LLMError(
                "Gemini client not initialized. Call initialize() first.", error_code="E2000"
            )

        try:
            # プロンプトの結合
            full_prompt = self._combine_prompts(prompt, system_prompt)

            # 新SDK: GenerateContentConfig使用
            generation_config = types.GenerateContentConfig(
                system_instruction=system_prompt if system_prompt else None,
                temperature=temperature,
                max_output_tokens=max_tokens or self.config.max_tokens,
                candidate_count=1,
            )

            # ストリーミング生成
            logger.debug(f"Starting streaming generation with model {self.current_model}")

            # プログレスバー（オプション）
            progress = None
            if logger.isEnabledFor(logging.INFO):
                progress = ProgressBar(prefix="Generating", show_time=True)

            # 新SDK: generate_content_streamを使用
            def stream_sync():
                """同期的なストリーミング生成"""
                return self.client.models.generate_content_stream(
                    model=self.current_model, contents=full_prompt, config=generation_config
                )

            # ストリーミングレスポンスの取得
            response_stream = await asyncio.get_event_loop().run_in_executor(None, stream_sync)

            # チャンクごとに処理
            total_tokens = 0
            for chunk in response_stream:
                if hasattr(chunk, "text") and chunk.text:
                    text_chunk = chunk.text
                    total_tokens += len(text_chunk.split())

                    # プログレスバー更新（概算）
                    if progress and max_tokens:
                        progress.update(min(total_tokens / max_tokens * 100, 100))

                    yield text_chunk

            # プログレスバー完了
            if progress:
                progress.update(100)
                progress.close()

            logger.debug(f"Streaming generation completed: {total_tokens} tokens")

        except Exception as e:
            error_msg = f"Streaming generation failed: {e}"
            logger.error(f"[E2000] {error_msg}")
            raise LLMError(error_msg, error_code="E2000")

    def get_available_models(self) -> List[str]:
        """利用可能なモデルのリスト取得

        Returns:
            List[str]: モデル名のリスト
        """
        return self.available_models.copy()

    def set_model(self, model_name: str) -> None:
        """使用するモデルを設定

        Args:
            model_name: モデル名

        Raises:
            ModelNotFoundError: 指定されたモデルが存在しない（E2004）
        """
        if model_name not in self.available_models:
            raise ModelNotFoundError(
                f"Model '{model_name}' not found. Available models: {self.available_models}",
                error_code="E2004",
            )

        self.current_model = model_name
        self.max_tokens = MODEL_MAX_TOKENS.get(model_name, 1048576)

        logger.info(f"Model changed to: {model_name}")

    def get_max_tokens(self) -> int:
        """最大トークン数を取得

        Returns:
            int: 現在のモデルの最大トークン数
        """
        return self.max_tokens

    # ========================================================================
    # ヘルパーメソッド
    # ========================================================================

    def _combine_prompts(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """プロンプトの結合

        Args:
            prompt: ユーザープロンプト
            system_prompt: システムプロンプト

        Returns:
            str: 結合されたプロンプト
        """
        if system_prompt:
            return f"{system_prompt}\n\n{prompt}"
        return prompt

    def _process_response(self, response: Any, request: LLMRequest) -> LLMResponse:
        """レスポンスの処理（新SDK対応）

        Args:
            response: Gemini APIレスポンス
            request: 元のリクエスト

        Returns:
            LLMResponse: 処理済みレスポンス
        """
        # responseがNoneの場合の特別処理
        if response is None:
            return LLMResponse(
                content="",
                usage={"total_tokens": 0},
                model=self.current_model,
                finish_reason="error",
                metadata={"error": "No response received"},
            )

        try:
            # テキスト取得（新SDKではresponse.textで直接アクセス）
            text = ""
            if hasattr(response, "text"):
                text = response.text if response.text is not None else ""
            elif hasattr(response, "candidates") and response.candidates:
                # 候補がある場合は最初の候補を使用
                candidate = response.candidates[0]
                if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                    text = "".join(
                        part.text for part in candidate.content.parts if hasattr(part, "text")
                    )

            # 使用量の推定（新SDKでは詳細な使用量が取得可能な場合がある）
            usage = {}
            if hasattr(response, "usage_metadata"):
                usage = {
                    "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
                    "completion_tokens": getattr(
                        response.usage_metadata, "candidates_token_count", 0
                    ),
                    "total_tokens": getattr(response.usage_metadata, "total_token_count", 0),
                }
            else:
                # 使用量が取得できない場合は推定
                prompt_tokens = len(request.prompt.split())
                completion_tokens = len(text.split()) if text else 0
                usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }

            # LLMResponse作成
            return LLMResponse(
                content=text,
                usage=usage,
                model=self.current_model,
                finish_reason="stop",
                metadata={"temperature": request.temperature, "max_tokens": request.max_tokens},
            )

        except Exception as e:
            logger.error(f"Failed to process response: {e}")
            # エラー時のレスポンス
            return LLMResponse(
                content="",
                usage={"total_tokens": 0},
                model=self.current_model,
                finish_reason="error",
                metadata={"error": str(e)},
            )

    def _handle_generation_error(self, error: Exception, request: LLMRequest) -> None:
        """エラーハンドリング（新SDK対応）

        Args:
            error: エラー
            request: 元のリクエスト

        Raises:
            RateLimitError: レート制限
            AuthenticationError: 認証エラー
            APIError: その他のAPIエラー
            LLMError: その他のLLMエラー
        """
        error_msg = str(error)

        # エラーメッセージの解析
        if "quota" in error_msg.lower() or "rate" in error_msg.lower() or "429" in error_msg:
            logger.warning(f"[E2002] Rate limit exceeded: {error_msg}")
            raise RateLimitError(
                "API rate limit exceeded. Please try again later.", retry_after=60  # デフォルト60秒
            )
        elif "api_key" in error_msg.lower() or "auth" in error_msg.lower() or "401" in error_msg:
            logger.error(f"[E2003] Authentication failed: {error_msg}")
            raise AuthenticationError("Invalid or missing API key")
        elif "404" in error_msg and "model" in error_msg.lower():
            logger.error(f"[E2004] Model not found: {error_msg}")
            raise ModelNotFoundError(
                f"Model '{self.current_model}' not found or no longer available", error_code="E2004"
            )
        # Google特有のエラー判定を改善（開発規約書v1.12準拠）
        elif hasattr(error, "__class__") and (
            "google" in str(error.__class__).lower()
            or "GoogleGenerativeAIError" in error.__class__.__name__
        ):
            # Google API固有のエラー
            logger.error(f"[E2001] API error: {error_msg}")
            api_error = APIError(
                f"Gemini API error: {error_msg}", status_code=500, error_code="E2001"
            )
            api_error.details["model"] = self.current_model
            api_error.details["prompt_length"] = len(request.prompt)
            raise api_error
        else:
            # その他の一般的なエラー
            logger.error(f"[E2000] Generation failed: {error_msg}")
            raise LLMError(
                f"Generation failed: {error_msg}",
                error_code="E2000",
                details={"prompt": request.prompt[:100], "model": self.current_model},
            )

    async def cleanup(self) -> None:
        """リソースのクリーンアップ"""
        await super().cleanup()
        self.client = None
        # 状態をTERMINATEDに設定
        self._state = ComponentState.TERMINATED
        logger.debug("GeminiEngine cleaned up")


# ============================================================================
# エクスポート
# ============================================================================

__all__ = [
    "GeminiEngine",
    "AVAILABLE_MODELS",
    "DEFAULT_MODEL",
]
