"""MockLLMEngine実装

大規模言語モデルエンジンのモック実装。
テスト用にキャラクター別の固定応答を返す。

インターフェース定義書 v1.34準拠
テストデータ・モック完全仕様書 v1.1準拠
エラーハンドリング指針 v1.20準拠
開発規約書 v1.12準拠
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

# プロジェクト内インポート（絶対インポート）
from vioratalk.core.base import ComponentState, VioraTalkComponent
from vioratalk.core.exceptions import APIError, LLMError, ModelNotFoundError


@dataclass
class LLMResponse:
    """LLM応答

    データフォーマット仕様書 v1.5準拠
    """

    content: str
    usage: Dict[str, int]  # tokens used, etc.
    model: str
    finish_reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Message:
    """会話メッセージ（簡易版）

    実際のMessageクラスがPhase 2で実装されるまでの代替
    """

    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


class MockLLMEngine(VioraTalkComponent):
    """LLMエンジンのモック実装

    Phase 3のテスト用にキャラクター別の固定応答を返すLLMエンジン。
    ストリーミング対応、エラーモード、カスタムレスポンス設定機能を提供。

    Attributes:
        config: 設定辞書
        response_delay: 応答生成の遅延シミュレーション（秒）
        streaming_enabled: ストリーミングモードの有効/無効
        error_mode: エラーモードの有効/無効
        character_responses: キャラクター別の応答パターン
        current_model: 現在使用中のモデル名
        max_tokens: 最大トークン数
        logger: ロガー
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初期化

        Args:
            config: 設定辞書（オプション）
        """
        # 基底クラスの初期化（引数なし）
        super().__init__()

        # 状態の初期化
        self._state = ComponentState.NOT_INITIALIZED
        self._initialized_at = None
        self._error = None

        # 設定を保存
        self.config = config or {}

        # ロガー設定
        self.logger = logging.getLogger(__name__)

        # 設定の初期化
        self.response_delay = self.config.get("delay", 0.1)
        self.streaming_enabled = self.config.get("streaming", False)
        self.error_mode = False
        self.current_model = self.config.get("model", "mock-gpt-3.5")
        self.max_tokens = self.config.get("max_tokens", 2048)

        # カスタムレスポンス設定（プロンプトごとのカスタムレスポンス）
        self.custom_responses: Dict[str, str] = {}

        # 統計情報
        self.total_requests = 0
        self.total_tokens = 0

        # 利用可能なモデル
        self.available_models = ["mock-gpt-3.5", "mock-gpt-4", "mock-claude-3", "mock-gemini-pro"]

        # キャラクター別の応答パターン（テストデータ・モック完全仕様書 v1.1準拠）
        self.character_responses = {
            "001_aoi": {
                "greeting": "こんにちは！碧衣です。今日はどんなお話をしましょうか？",
                "question": "それは興味深い質問ですね。私なりの考えをお話しします。",
                "command": "はい、承知いたしました。すぐに対応いたしますね。",
                "default": "なるほど、そういうことですね。もう少し詳しく教えていただけますか？",
            },
            "002_haru": {
                "greeting": "やあ！春人だよ！今日も元気いっぱい話そう！",
                "question": "えーっと、それはね...うん、きっとこうだと思う！",
                "command": "りょうかい！すぐにやるね！",
                "default": "へぇ〜、そうなんだ！おもしろいね！",
            },
            "003_yui": {
                "greeting": "こんにちは...結衣です。よろしくお願いします...",
                "question": "えっと...その質問は...たぶん、こうかもしれません...",
                "command": "はい...やってみますね...",
                "default": "そ、そうですか...なるほど...",
            },
            "default": {
                "greeting": "こんにちは。何かお手伝いできることはありますか？",
                "question": "その質問について考えてみました。",
                "command": "承知しました。実行します。",
                "default": "了解しました。続けてください。",
            },
        }

        # 会話履歴（簡易実装）
        self.conversation_history: List[Message] = []

    async def initialize(self) -> None:
        """初期化処理

        エンジン初期化仕様書 v1.4準拠
        """
        self.logger.info("MockLLMEngine initialization started")

        # 初期化シミュレーション
        await asyncio.sleep(0.05)

        self._state = ComponentState.READY
        self._initialized_at = datetime.utcnow()
        self.logger.info(
            "MockLLMEngine initialized",
            extra={"model": self.current_model, "max_tokens": self.max_tokens},
        )

    async def cleanup(self) -> None:
        """クリーンアップ処理"""
        self.logger.info("MockLLMEngine cleanup started")

        # 会話履歴のクリア
        self.conversation_history.clear()

        self._state = ComponentState.TERMINATED
        self.logger.info("MockLLMEngine cleaned up")

    @property
    def state(self) -> ComponentState:
        """現在の状態（プロパティ）

        Returns:
            ComponentState: 現在の状態
        """
        return self._state

    def is_available(self) -> bool:
        """利用可能状態の確認

        Returns:
            bool: READYまたはRUNNING状態の場合True
        """
        return self._state in [ComponentState.READY, ComponentState.RUNNING]

    def get_status(self) -> Dict[str, Any]:
        """ステータス情報取得

        Returns:
            Dict[str, Any]: ステータス情報
        """
        return {
            "state": self._state.value,
            "is_available": self.is_available(),
            "error": str(self._error) if self._error else None,
            "last_used": self._initialized_at.isoformat() if self._initialized_at else None,
            "current_model": self.current_model,
            "streaming_enabled": self.streaming_enabled,
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
        }

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """テキスト生成（モック実装）

        Args:
            prompt: 入力プロンプト
            system_prompt: システムプロンプト（オプション）
            temperature: 生成温度
            max_tokens: 最大トークン数

        Returns:
            LLMResponse: 生成結果

        Raises:
            APIError: エラーモード時またはAPI呼び出しエラー（E2001）
            LLMError: LLM処理エラー（E2000）
        """
        # 状態チェック
        if self._state != ComponentState.READY:
            raise LLMError(
                "LLM engine is not ready", error_code="E2000", details={"state": self._state.value}
            )

        # エラーモードチェック
        if self.error_mode:
            self.logger.error("Mock LLM error mode is enabled")
            raise APIError(
                "Mock API error", error_code="E2001", details={"mode": "error_simulation"}
            )

        # 統計更新
        self.total_requests += 1

        # 最大トークン数の決定
        if max_tokens is None:
            max_tokens = self.max_tokens

        # 遅延シミュレーション
        await asyncio.sleep(self.response_delay)

        # カスタムレスポンスチェック
        if prompt in self.custom_responses:
            response_text = self.custom_responses[prompt]
            response_type = "custom"
            character_id = "custom"
        else:
            # キャラクターIDの抽出（system_promptから推定）
            character_id = self._extract_character_id(system_prompt)

            # 応答タイプの判定
            response_type = self._determine_response_type(prompt)

            # 応答テキストの取得
            response_text = self._get_response_text(character_id, response_type)

        # 会話履歴に追加
        self.conversation_history.append(Message(role="user", content=prompt))
        self.conversation_history.append(Message(role="assistant", content=response_text))

        # トークン使用量のモック計算（より現実的な計算）
        # 日本語は1文字≒2トークン、英語は4文字≒1トークンとして計算
        if any(ord(c) > 0x3000 for c in prompt):  # 日本語チェック
            prompt_tokens = len(prompt) * 2
            completion_tokens = min(len(response_text) * 2, max_tokens)
        else:
            prompt_tokens = max(1, len(prompt) // 4)
            completion_tokens = min(max(1, len(response_text) // 4), max_tokens)

        total_tokens = prompt_tokens + completion_tokens

        self.total_tokens += total_tokens

        # 結果の作成
        result = LLMResponse(
            content=response_text,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            model=self.current_model,
            finish_reason="stop" if completion_tokens < max_tokens else "length",
            metadata={
                "character_id": character_id,
                "response_type": response_type,
                "temperature": temperature,
            },
        )

        self.logger.debug(f"Generated response: {len(response_text)} chars, {total_tokens} tokens")

        return result

    async def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """ストリーミング生成（モック実装）

        Args:
            prompt: 入力プロンプト
            system_prompt: システムプロンプト
            temperature: 生成温度
            max_tokens: 最大トークン数

        Yields:
            str: 生成されたテキストの断片

        Raises:
            APIError: エラーモード時（E2001）
            LLMError: 準備未完了時（E2000）
        """
        # 通常の生成を実行
        response = await self.generate(prompt, system_prompt, temperature, max_tokens)

        # 文字ごとにストリーミング
        for char in response.content:
            await asyncio.sleep(0.01)  # ストリーミング遅延
            yield char

    def _extract_character_id(self, system_prompt: Optional[str]) -> str:
        """キャラクターIDの抽出

        Args:
            system_prompt: システムプロンプト

        Returns:
            str: キャラクターID
        """
        if not system_prompt:
            return "default"

        # "character_id:xxx"形式から抽出
        if "character_id:" in system_prompt:
            parts = system_prompt.split("character_id:")
            if len(parts) > 1:
                char_id = parts[1].split()[0].strip()
                if char_id in self.character_responses:
                    return char_id

        return "default"

    def _determine_response_type(self, prompt: str) -> str:
        """応答タイプの判定

        Args:
            prompt: 入力プロンプト

        Returns:
            str: 応答タイプ（greeting/question/command/default）
        """
        prompt_lower = prompt.lower()

        # 挨拶判定
        greetings = ["こんにちは", "おはよう", "こんばんは", "やあ", "hello", "hi"]
        if any(g in prompt_lower for g in greetings):
            return "greeting"

        # 質問判定（"？"があれば必ず質問として扱う）
        questions = ["？", "?", "ですか", "ますか", "何", "どう", "なぜ", "いつ"]
        if any(q in prompt for q in questions):  # prompt_lowerではなくpromptで"？"をチェック
            return "question"

        # コマンド判定
        commands = ["して", "ください", "お願い", "実行", "再生", "停止"]
        if any(c in prompt_lower for c in commands):
            return "command"

        return "default"

    def _get_response_text(self, character_id: str, response_type: str) -> str:
        """応答テキストの取得

        Args:
            character_id: キャラクターID
            response_type: 応答タイプ

        Returns:
            str: 応答テキスト
        """
        character = self.character_responses.get(character_id, self.character_responses["default"])
        return character.get(response_type, character["default"])

    # ========================================================================
    # モデル管理メソッド
    # ========================================================================

    def get_available_models(self) -> List[str]:
        """利用可能なモデルのリスト取得

        Returns:
            List[str]: モデル名のリスト
        """
        return self.available_models.copy()

    def set_model(self, model_name: str) -> None:
        """モデルの切り替え

        Args:
            model_name: モデル名

        Raises:
            ModelNotFoundError: 指定されたモデルが存在しない場合（E2404）
        """
        if model_name not in self.available_models:
            raise ModelNotFoundError(
                f"Model '{model_name}' not found",
                error_code="E2404",
                details={"available_models": self.available_models},
            )

        self.current_model = model_name
        self.logger.info(f"Model changed to {model_name}")

    # ========================================================================
    # 会話履歴管理
    # ========================================================================

    def add_message(self, role: str, content: str) -> None:
        """メッセージを会話履歴に追加

        Args:
            role: ロール（user/assistant/system）
            content: メッセージ内容
        """
        self.conversation_history.append(Message(role=role, content=content))

    def clear_history(self) -> None:
        """会話履歴のクリア"""
        self.conversation_history.clear()
        self.logger.debug("Conversation history cleared")

    def get_history(self) -> List[Message]:
        """会話履歴の取得

        Returns:
            List[Message]: 会話履歴
        """
        return self.conversation_history.copy()

    # ========================================================================
    # テスト用ユーティリティメソッド
    # ========================================================================

    def set_error_mode(self, enabled: bool) -> None:
        """エラーモードの設定（テスト用）

        Args:
            enabled: エラーモードの有効/無効
        """
        self.error_mode = enabled
        self.logger.warning(f"Error mode {'enabled' if enabled else 'disabled'}")

    def set_custom_response(self, prompt: str, response: str) -> None:
        """カスタムレスポンスの設定（テスト用）

        プロンプトに対するカスタムレスポンスを設定する。
        MockSTTEngineと同様のインターフェースを提供。

        Args:
            prompt: 対象プロンプト
            response: カスタムレスポンステキスト
        """
        self.custom_responses[prompt] = response
        self.logger.debug(f"Custom response set for prompt: {prompt[:50]}...")

    def set_response_delay(self, delay: float) -> None:
        """応答遅延の設定（テスト用）

        Args:
            delay: 遅延時間（秒）、負の値は0にクリップ
        """
        self.response_delay = max(0.0, delay)
        self.logger.debug(f"Response delay set to {self.response_delay}s")

    def get_statistics(self) -> Dict[str, Any]:
        """統計情報の取得（テスト用）

        Returns:
            Dict[str, Any]: 統計情報
        """
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "average_tokens_per_request": (
                self.total_tokens / self.total_requests if self.total_requests > 0 else 0
            ),
            "conversation_length": len(self.conversation_history),
        }
