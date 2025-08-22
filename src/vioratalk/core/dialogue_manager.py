"""対話システム管理モジュール

DialogueManagerクラスを定義し、対話フローを制御する。
Phase 2の中核コンポーネントとして、会話の管理と処理を担当。

インターフェース定義書 v1.34準拠
DialogueManager統合ガイド v1.2準拠
データフォーマット仕様書 v1.5準拠
開発規約書 v1.12準拠
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from vioratalk.core.base import ComponentState, VioraTalkComponent
from vioratalk.core.dialogue_config import DialogueConfig
from vioratalk.core.dialogue_state import DialogueTurn


class ConversationState(Enum):
    """会話の状態を表すEnum

    Phase 2の最小実装として基本的な状態のみ定義。
    """

    IDLE = "idle"  # アイドル状態
    PROCESSING = "processing"  # 処理中
    WAITING = "waiting"  # ユーザー入力待ち
    ERROR = "error"  # エラー状態


@dataclass
class ConversationContext:
    """会話のコンテキストを管理する内部クラス

    DialogueManagerで使用される会話状態を保持。
    Phase 2では最小限の実装。

    Attributes:
        conversation_id: 会話セッションID
        character_id: 使用中のキャラクターID
        turns: 対話ターンのリスト
        current_turn_number: 現在のターン番号
        state: 会話の状態
        started_at: 会話開始時刻
        last_activity: 最後の活動時刻
        metadata: 追加メタデータ
    """

    conversation_id: str
    character_id: str
    turns: List[DialogueTurn] = field(default_factory=list)
    current_turn_number: int = 0
    state: ConversationState = ConversationState.IDLE
    started_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_turn(self, turn: DialogueTurn) -> None:
        """対話ターンを追加

        Args:
            turn: 追加する対話ターン
        """
        self.turns.append(turn)
        self.current_turn_number += 1
        self.last_activity = datetime.now()

    def get_recent_turns(self, limit: int = 10) -> List[DialogueTurn]:
        """最近の対話ターンを取得

        Args:
            limit: 取得する最大ターン数

        Returns:
            List[DialogueTurn]: 最近の対話ターンのリスト
        """
        if limit <= 0:
            return []
        return self.turns[-limit:] if self.turns else []

    def clear(self) -> None:
        """会話履歴をクリア"""
        self.turns.clear()
        self.current_turn_number = 0
        self.state = ConversationState.IDLE
        self.last_activity = datetime.now()

    def is_active(self) -> bool:
        """会話がアクティブかチェック

        Returns:
            bool: アクティブな場合True
        """
        return self.state in [ConversationState.PROCESSING, ConversationState.WAITING]


class DialogueManager(VioraTalkComponent):
    """対話フロー管理クラス

    Phase 2の中核コンポーネント。
    ユーザー入力を処理し、応答を生成する。

    インターフェース定義書 v1.34 セクション3.9準拠
    DialogueManager統合ガイド v1.2準拠

    Attributes:
        config: 対話設定
        character_manager: キャラクター管理（Phase 2ではMock）
        _context: 現在の会話コンテキスト
        _initialized: 初期化済みフラグ
        _conversation_count: 生成した会話ID用カウンタ

    Example:
        >>> config = DialogueConfig()
        >>> char_manager = MockCharacterManager()
        >>> manager = DialogueManager(config, char_manager)
        >>> await manager.initialize()
        >>> turn = await manager.process_text_input("こんにちは")
        >>> print(turn.assistant_response)
        こんにちは！今日はどんなお話をしましょうか？
    """

    def __init__(self, config: DialogueConfig, character_manager: Optional[Any] = None) -> None:
        """コンストラクタ

        Args:
            config: 対話設定
            character_manager: キャラクター管理（Phase 2ではオプション）
        """
        super().__init__()
        self.config = config
        self.character_manager = character_manager
        self._context: Optional[ConversationContext] = None
        self._initialized: bool = False
        self._conversation_count: int = 0

    async def initialize(self) -> None:
        """非同期初期化

        コンポーネントを初期化し、使用可能な状態にする。

        Raises:
            RuntimeError: 既に初期化済みの場合
        """
        if self._state == ComponentState.READY:
            raise RuntimeError("DialogueManager is already initialized")

        self._state = ComponentState.INITIALIZING

        # キャラクターマネージャーの初期化（存在する場合）
        if self.character_manager and hasattr(self.character_manager, "initialize"):
            if self.character_manager._state != ComponentState.READY:
                await self.character_manager.initialize()

        # 新しい会話コンテキストを作成
        self._create_new_context()

        self._initialized = True
        self._state = ComponentState.READY

    async def cleanup(self) -> None:
        """リソースのクリーンアップ

        使用したリソースを解放する。
        """
        if self._state == ComponentState.TERMINATED:
            return

        self._state = ComponentState.TERMINATING

        # キャラクターマネージャーのクリーンアップ
        if self.character_manager and hasattr(self.character_manager, "cleanup"):
            await self.character_manager.cleanup()

        # コンテキストのクリア
        if self._context:
            self._context.clear()
            self._context = None

        self._initialized = False
        self._state = ComponentState.TERMINATED

    def _create_new_context(self) -> None:
        """新しい会話コンテキストを作成"""
        self._conversation_count += 1
        conversation_id = (
            f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._conversation_count:04d}"
        )

        # 現在のキャラクターIDを取得
        character_id = self.config.default_character_id
        if self.character_manager and hasattr(self.character_manager, "get_current_character"):
            current_char = self.character_manager.get_current_character()
            if current_char:
                character_id = current_char.character_id

        self._context = ConversationContext(
            conversation_id=conversation_id, character_id=character_id
        )

    async def process_text_input(self, text: str) -> DialogueTurn:
        """テキスト入力を処理して応答を生成

        Phase 2のメイン機能。テキスト入力を受け取り、
        キャラクターに応じた応答を生成する。

        Args:
            text: ユーザーからの入力テキスト

        Returns:
            DialogueTurn: ユーザー入力とアシスタント応答のペア

        Raises:
            RuntimeError: 未初期化の場合
            ValueError: 入力テキストが空の場合

        Example:
            >>> turn = await manager.process_text_input("お元気ですか？")
            >>> print(turn.assistant_response)
            はい、元気です！あなたはいかがですか？
        """
        if not self._initialized:
            raise RuntimeError("DialogueManager is not initialized")

        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")

        # 最大ターン数チェック
        if self._context and len(self._context.turns) >= self.config.max_turns:
            # 最古のターンを削除
            self._context.turns.pop(0)

        # 会話状態を処理中に変更
        if self._context:
            self._context.state = ConversationState.PROCESSING

        try:
            # Phase 2では簡単な応答生成
            # 実際のLLM統合はPhase 3-4で実装
            assistant_response = await self._generate_mock_response(text)

            # DialogueTurnを作成
            turn = DialogueTurn(
                user_input=text,
                assistant_response=assistant_response,
                timestamp=datetime.now(),
                turn_number=self._context.current_turn_number + 1 if self._context else 1,
                character_id=self._context.character_id
                if self._context
                else self.config.default_character_id,
                emotion="neutral",  # Phase 7で感情分析実装
                confidence=0.95,  # Phase 2では固定値
                processing_time=0.1,  # Phase 2では固定値
            )

            # コンテキストに追加
            if self._context:
                self._context.add_turn(turn)
                self._context.state = ConversationState.WAITING

            return turn

        except Exception as e:
            # エラー時の処理
            if self._context:
                self._context.state = ConversationState.ERROR
            raise RuntimeError(f"Failed to process text input: {str(e)}")

    async def _generate_mock_response(self, user_input: str) -> str:
        """モック応答を生成（Phase 2用）

        Args:
            user_input: ユーザー入力

        Returns:
            str: 生成された応答
        """
        # キャラクターに応じた応答パターン
        responses = {
            "こんにちは": "こんにちは！今日はどんなお話をしましょうか？",
            "お元気ですか": "はい、元気です！あなたはいかがですか？",
            "ありがとう": "どういたしまして！お役に立てて嬉しいです。",
            "さようなら": "さようなら！またお話ししましょうね。",
            "おはよう": "おはようございます！今日も良い一日になりますように。",
            "おやすみ": "おやすみなさい。良い夢を見てくださいね。",
        }

        # 簡単なパターンマッチング
        for pattern, response in responses.items():
            if pattern in user_input:
                return response

        # デフォルト応答
        if self.character_manager and hasattr(self.character_manager, "get_current_character"):
            character = self.character_manager.get_current_character()
            if character and character.name == "あおい":
                return "はい、承知いたしました。他に何かお手伝いできることはありますか？"

        return "なるほど、そうですね。もう少し詳しく教えていただけますか？"

    async def process_audio_input(self, audio_data: bytes) -> DialogueTurn:
        """音声入力を処理して応答を生成（Phase 3以降で実装）

        Phase 2ではスタブ実装。

        Args:
            audio_data: 音声データ（バイト列）

        Returns:
            DialogueTurn: ユーザー入力とアシスタント応答のペア

        Raises:
            NotImplementedError: Phase 2では未実装
        """
        raise NotImplementedError("Audio input processing will be implemented in Phase 3")

    def get_conversation_history(self, limit: int = 10) -> List[DialogueTurn]:
        """会話履歴を取得

        Args:
            limit: 取得する最大ターン数（デフォルト: 10）

        Returns:
            List[DialogueTurn]: 会話履歴のリスト

        Raises:
            RuntimeError: 未初期化の場合

        Example:
            >>> history = manager.get_conversation_history(5)
            >>> for turn in history:
            ...     print(f"User: {turn.user_input}")
            ...     print(f"Assistant: {turn.assistant_response}")
        """
        if not self._initialized:
            raise RuntimeError("DialogueManager is not initialized")

        if not self._context:
            return []

        return self._context.get_recent_turns(limit)

    async def reset_conversation(self) -> None:
        """会話をリセット

        現在の会話をクリアし、新しい会話を開始する。

        Raises:
            RuntimeError: 未初期化の場合
        """
        if not self._initialized:
            raise RuntimeError("DialogueManager is not initialized")

        # 現在のコンテキストをクリア
        if self._context:
            self._context.clear()

        # 新しいコンテキストを作成
        self._create_new_context()

    def clear_conversation(self) -> None:
        """会話履歴をクリア

        現在の会話履歴を削除する。
        reset_conversationとは異なり、同じセッション内で履歴のみクリア。

        Raises:
            RuntimeError: 未初期化の場合
        """
        if not self._initialized:
            raise RuntimeError("DialogueManager is not initialized")

        if self._context:
            self._context.clear()

    def can_switch_character(self) -> bool:
        """キャラクター切り替え可能かチェック

        Returns:
            bool: 切り替え可能な場合True

        Note:
            インターフェース定義書 v1.34準拠
            会話がアクティブでない場合のみ切り替え可能
        """
        if not self._initialized:
            return False

        if not self._context:
            return True

        # 会話がアクティブでない場合のみ切り替え可能
        return not self._context.is_active()

    def is_conversation_active(self) -> bool:
        """会話がアクティブか確認

        Returns:
            bool: アクティブな場合True
        """
        if not self._initialized:
            return False

        if not self._context:
            return False

        return self._context.is_active()

    def get_current_context(self) -> Optional[ConversationContext]:
        """現在の会話コンテキストを取得（デバッグ用）

        Returns:
            Optional[ConversationContext]: 現在のコンテキスト

        Note:
            主にテストやデバッグで使用
        """
        return self._context

    def get_conversation_stats(self) -> Dict[str, Any]:
        """会話統計を取得

        Returns:
            Dict[str, Any]: 統計情報の辞書

        Example:
            >>> stats = manager.get_conversation_stats()
            >>> print(f"Total turns: {stats['total_turns']}")
        """
        if not self._context:
            return {
                "conversation_id": None,
                "total_turns": 0,
                "character_id": None,
                "state": "no_context",
                "duration_seconds": 0,
            }

        duration = (datetime.now() - self._context.started_at).total_seconds()

        return {
            "conversation_id": self._context.conversation_id,
            "total_turns": len(self._context.turns),
            "current_turn_number": self._context.current_turn_number,
            "character_id": self._context.character_id,
            "state": self._context.state.value,
            "duration_seconds": duration,
            "started_at": self._context.started_at.isoformat(),
            "last_activity": self._context.last_activity.isoformat(),
        }

    def __repr__(self) -> str:
        """開発用の詳細文字列表現

        Returns:
            str: オブジェクトの詳細情報
        """
        context_info = "no_context"
        if self._context:
            context_info = f"id={self._context.conversation_id}, turns={len(self._context.turns)}"

        return (
            f"DialogueManager("
            f"state={self._state.value}, "
            f"initialized={self._initialized}, "
            f"context=[{context_info}], "
            f"config=use_mock={self.config.use_mock_engines})"
        )
