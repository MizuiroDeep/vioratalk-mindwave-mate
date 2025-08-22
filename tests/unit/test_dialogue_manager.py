"""DialogueManagerクラスの単体テスト

DialogueManagerのすべての機能を網羅的にテストします。
Phase 2の中核コンポーネントの品質を保証します。

テスト実装ガイド v1.3準拠
テスト戦略ガイドライン v1.7準拠
開発規約書 v1.12準拠
"""

from datetime import datetime

import pytest

from tests.mocks.mock_character_manager import MockCharacterManager
from vioratalk.core.base import ComponentState
from vioratalk.core.dialogue_config import DialogueConfig
from vioratalk.core.dialogue_manager import ConversationContext, ConversationState, DialogueManager
from vioratalk.core.dialogue_state import DialogueTurn


@pytest.mark.unit
@pytest.mark.phase(2)
class TestDialogueManager:
    """DialogueManagerクラスのテストスイート

    インターフェース定義書 v1.34準拠
    DialogueManager統合ガイド v1.2準拠
    """

    # ------------------------------------------------------------------------
    # フィクスチャ
    # ------------------------------------------------------------------------

    @pytest.fixture
    def config(self) -> DialogueConfig:
        """テスト用設定のフィクスチャ

        Returns:
            DialogueConfig: テスト用設定
        """
        return DialogueConfig(max_turns=5, debug_mode=True, use_mock_engines=True)  # テスト用に小さく設定

    @pytest.fixture
    async def character_manager(self) -> MockCharacterManager:
        """初期化済みMockCharacterManagerのフィクスチャ

        Returns:
            MockCharacterManager: 初期化済みマネージャー
        """
        manager = MockCharacterManager()
        await manager.initialize()
        return manager

    @pytest.fixture
    def manager(self, config, character_manager) -> DialogueManager:
        """DialogueManagerのフィクスチャ（未初期化）

        Returns:
            DialogueManager: 未初期化のマネージャー
        """
        return DialogueManager(config, character_manager)

    @pytest.fixture
    async def initialized_manager(self, config, character_manager) -> DialogueManager:
        """初期化済みDialogueManagerのフィクスチャ

        Returns:
            DialogueManager: 初期化済みマネージャー
        """
        manager = DialogueManager(config, character_manager)
        await manager.initialize()
        return manager

    # ------------------------------------------------------------------------
    # 初期化のテスト
    # ------------------------------------------------------------------------

    def test_constructor(self, config, character_manager):
        """コンストラクタのテスト"""
        manager = DialogueManager(config, character_manager)

        assert manager.config == config
        assert manager.character_manager == character_manager
        assert manager._state == ComponentState.NOT_INITIALIZED
        assert manager._context is None
        assert not manager._initialized
        assert manager._conversation_count == 0

    def test_constructor_without_character_manager(self, config):
        """キャラクターマネージャーなしでのコンストラクタ"""
        manager = DialogueManager(config)

        assert manager.config == config
        assert manager.character_manager is None
        assert manager._state == ComponentState.NOT_INITIALIZED

    @pytest.mark.asyncio
    async def test_initialize(self, manager):
        """初期化メソッドのテスト"""
        assert manager._state == ComponentState.NOT_INITIALIZED
        assert manager._context is None

        await manager.initialize()

        assert manager._state == ComponentState.READY
        assert manager._initialized
        assert manager._context is not None
        assert manager._context.conversation_id.startswith("conv_")
        assert manager._context.character_id == "001_aoi"
        assert len(manager._context.turns) == 0

    @pytest.mark.asyncio
    async def test_initialize_twice(self, initialized_manager):
        """二重初期化のエラーテスト"""
        with pytest.raises(RuntimeError, match="already initialized"):
            await initialized_manager.initialize()

    @pytest.mark.asyncio
    async def test_cleanup(self, initialized_manager):
        """クリーンアップメソッドのテスト"""
        assert initialized_manager._state == ComponentState.READY
        assert initialized_manager._context is not None

        await initialized_manager.cleanup()

        assert initialized_manager._state == ComponentState.TERMINATED
        assert not initialized_manager._initialized
        assert initialized_manager._context is None

    @pytest.mark.asyncio
    async def test_cleanup_twice(self, initialized_manager):
        """二重クリーンアップのテスト"""
        await initialized_manager.cleanup()
        assert initialized_manager._state == ComponentState.TERMINATED

        # 二回目は安全に何もしない
        await initialized_manager.cleanup()
        assert initialized_manager._state == ComponentState.TERMINATED

    # ------------------------------------------------------------------------
    # process_text_input()のテスト
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_process_text_input_basic(self, initialized_manager):
        """基本的なテキスト入力処理のテスト"""
        user_input = "こんにちは"
        turn = await initialized_manager.process_text_input(user_input)

        assert isinstance(turn, DialogueTurn)
        assert turn.user_input == user_input
        assert turn.assistant_response == "こんにちは！今日はどんなお話をしましょうか？"
        assert turn.turn_number == 1
        assert turn.character_id == "001_aoi"
        assert turn.emotion == "neutral"
        assert turn.confidence == 0.95
        assert turn.processing_time == 0.1

    @pytest.mark.asyncio
    async def test_process_text_input_multiple_turns(self, initialized_manager):
        """複数ターンの処理テスト"""
        inputs = ["おはよう", "お元気ですか", "ありがとう"]
        expected_responses = [
            "おはようございます！今日も良い一日になりますように。",
            "はい、元気です！あなたはいかがですか？",
            "どういたしまして！お役に立てて嬉しいです。",
        ]

        for i, (user_input, expected) in enumerate(zip(inputs, expected_responses), 1):
            turn = await initialized_manager.process_text_input(user_input)
            assert turn.user_input == user_input
            assert turn.assistant_response == expected
            assert turn.turn_number == i

        # 履歴確認
        history = initialized_manager.get_conversation_history()
        assert len(history) == 3

    @pytest.mark.asyncio
    async def test_process_text_input_default_response(self, initialized_manager):
        """デフォルト応答のテスト"""
        turn = await initialized_manager.process_text_input("知らない入力")

        # あおいキャラクターのデフォルト応答
        assert turn.assistant_response == "はい、承知いたしました。他に何かお手伝いできることはありますか？"

    @pytest.mark.asyncio
    async def test_process_text_input_empty(self, initialized_manager):
        """空文字列入力のエラーテスト"""
        with pytest.raises(ValueError, match="Input text cannot be empty"):
            await initialized_manager.process_text_input("")

        with pytest.raises(ValueError, match="Input text cannot be empty"):
            await initialized_manager.process_text_input("   ")

    @pytest.mark.asyncio
    async def test_process_text_input_not_initialized(self, manager):
        """未初期化状態での処理エラーテスト"""
        with pytest.raises(RuntimeError, match="not initialized"):
            await manager.process_text_input("テスト")

    @pytest.mark.asyncio
    async def test_process_text_input_max_turns(self, initialized_manager):
        """最大ターン数制限のテスト"""
        # max_turns=5の設定で6ターン入力
        for i in range(6):
            turn = await initialized_manager.process_text_input(f"入力{i+1}")
            assert turn.turn_number == i + 1

        # 履歴は最大5ターンまで
        history = initialized_manager.get_conversation_history()
        assert len(history) == 5
        # 最古のターンが削除されている
        assert history[0].user_input == "入力2"

    # ------------------------------------------------------------------------
    # process_audio_input()のテスト（Phase 2ではスタブ）
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_process_audio_input_not_implemented(self, initialized_manager):
        """音声入力処理の未実装テスト"""
        with pytest.raises(NotImplementedError, match="Phase 3"):
            await initialized_manager.process_audio_input(b"audio_data")

    # ------------------------------------------------------------------------
    # 会話履歴管理のテスト
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_conversation_history_empty(self, initialized_manager):
        """空の履歴取得テスト"""
        history = initialized_manager.get_conversation_history()
        assert isinstance(history, list)
        assert len(history) == 0

    @pytest.mark.asyncio
    async def test_get_conversation_history_with_turns(self, initialized_manager):
        """ターンありの履歴取得テスト"""
        # 3ターン追加
        await initialized_manager.process_text_input("こんにちは")
        await initialized_manager.process_text_input("お元気ですか")
        await initialized_manager.process_text_input("ありがとう")

        # 全履歴取得
        history = initialized_manager.get_conversation_history()
        assert len(history) == 3

        # 制限付き取得
        history_limited = initialized_manager.get_conversation_history(limit=2)
        assert len(history_limited) == 2
        # 最新の2つが取得される
        assert history_limited[0].user_input == "お元気ですか"
        assert history_limited[1].user_input == "ありがとう"

    def test_get_conversation_history_not_initialized(self, manager):
        """未初期化状態での履歴取得エラーテスト"""
        with pytest.raises(RuntimeError, match="not initialized"):
            manager.get_conversation_history()

    @pytest.mark.asyncio
    async def test_clear_conversation(self, initialized_manager):
        """会話クリアのテスト"""
        # ターン追加
        await initialized_manager.process_text_input("こんにちは")
        await initialized_manager.process_text_input("お元気ですか")

        history = initialized_manager.get_conversation_history()
        assert len(history) == 2

        # クリア
        initialized_manager.clear_conversation()

        history = initialized_manager.get_conversation_history()
        assert len(history) == 0

        # コンテキストは維持される
        assert initialized_manager._context is not None
        assert initialized_manager._context.current_turn_number == 0

    def test_clear_conversation_not_initialized(self, manager):
        """未初期化状態でのクリアエラーテスト"""
        with pytest.raises(RuntimeError, match="not initialized"):
            manager.clear_conversation()

    @pytest.mark.asyncio
    async def test_reset_conversation(self, initialized_manager):
        """会話リセットのテスト"""
        # 初期のconversation_id記録
        initial_id = initialized_manager._context.conversation_id

        # ターン追加
        await initialized_manager.process_text_input("こんにちは")
        assert len(initialized_manager.get_conversation_history()) == 1

        # リセット
        await initialized_manager.reset_conversation()

        # 新しいコンテキストが作成される
        assert initialized_manager._context.conversation_id != initial_id
        assert len(initialized_manager.get_conversation_history()) == 0
        assert initialized_manager._context.current_turn_number == 0

    @pytest.mark.asyncio
    async def test_reset_conversation_not_initialized(self, manager):
        """未初期化状態でのリセットエラーテスト"""
        with pytest.raises(RuntimeError, match="not initialized"):
            await manager.reset_conversation()

    # ------------------------------------------------------------------------
    # キャラクター切り替え関連のテスト
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_can_switch_character_idle(self, initialized_manager):
        """アイドル状態でのキャラクター切り替え可能性テスト"""
        assert initialized_manager.can_switch_character() is True

    @pytest.mark.asyncio
    async def test_can_switch_character_processing(self, initialized_manager):
        """処理中状態でのキャラクター切り替え不可テスト"""
        # 処理状態をシミュレート
        initialized_manager._context.state = ConversationState.PROCESSING
        assert initialized_manager.can_switch_character() is False

    def test_can_switch_character_not_initialized(self, manager):
        """未初期化状態でのキャラクター切り替え不可テスト"""
        assert manager.can_switch_character() is False

    # ------------------------------------------------------------------------
    # 会話状態確認のテスト
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_is_conversation_active(self, initialized_manager):
        """会話アクティブ状態のテスト"""
        # 初期状態（IDLE）
        assert initialized_manager.is_conversation_active() is False

        # 処理中状態
        initialized_manager._context.state = ConversationState.PROCESSING
        assert initialized_manager.is_conversation_active() is True

        # 待機状態
        initialized_manager._context.state = ConversationState.WAITING
        assert initialized_manager.is_conversation_active() is True

        # エラー状態
        initialized_manager._context.state = ConversationState.ERROR
        assert initialized_manager.is_conversation_active() is False

    def test_is_conversation_active_not_initialized(self, manager):
        """未初期化状態での会話アクティブチェック"""
        assert manager.is_conversation_active() is False

    # ------------------------------------------------------------------------
    # 統計情報のテスト
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_conversation_stats(self, initialized_manager):
        """会話統計取得のテスト"""
        # 初期状態
        stats = initialized_manager.get_conversation_stats()
        assert stats["total_turns"] == 0
        assert stats["character_id"] == "001_aoi"
        assert stats["state"] == "idle"

        # ターン追加後
        await initialized_manager.process_text_input("こんにちは")
        await initialized_manager.process_text_input("お元気ですか")

        stats = initialized_manager.get_conversation_stats()
        assert stats["total_turns"] == 2
        assert stats["current_turn_number"] == 2
        assert stats["duration_seconds"] >= 0
        assert "started_at" in stats
        assert "last_activity" in stats

    def test_get_conversation_stats_no_context(self, manager):
        """コンテキストなしでの統計取得テスト"""
        stats = manager.get_conversation_stats()
        assert stats["conversation_id"] is None
        assert stats["total_turns"] == 0
        assert stats["state"] == "no_context"

    # ------------------------------------------------------------------------
    # ConversationContextのテスト
    # ------------------------------------------------------------------------

    def test_conversation_context_initialization(self):
        """ConversationContext初期化のテスト"""
        context = ConversationContext(conversation_id="test_001", character_id="001_aoi")

        assert context.conversation_id == "test_001"
        assert context.character_id == "001_aoi"
        assert len(context.turns) == 0
        assert context.current_turn_number == 0
        assert context.state == ConversationState.IDLE

    def test_conversation_context_add_turn(self):
        """ConversationContextへのターン追加テスト"""
        context = ConversationContext(conversation_id="test_001", character_id="001_aoi")

        turn = DialogueTurn(
            user_input="テスト",
            assistant_response="応答",
            timestamp=datetime.now(),
            turn_number=1,
            character_id="001_aoi",
        )

        context.add_turn(turn)

        assert len(context.turns) == 1
        assert context.current_turn_number == 1
        assert context.turns[0] == turn

    def test_conversation_context_get_recent_turns(self):
        """最近のターン取得テスト"""
        context = ConversationContext(conversation_id="test_001", character_id="001_aoi")

        # 5ターン追加
        for i in range(5):
            turn = DialogueTurn(
                user_input=f"入力{i+1}",
                assistant_response=f"応答{i+1}",
                timestamp=datetime.now(),
                turn_number=i + 1,
                character_id="001_aoi",
            )
            context.add_turn(turn)

        # 最新3ターン取得
        recent = context.get_recent_turns(3)
        assert len(recent) == 3
        assert recent[0].user_input == "入力3"
        assert recent[2].user_input == "入力5"

        # 制限が0の場合
        recent = context.get_recent_turns(0)
        assert len(recent) == 0

    def test_conversation_context_clear(self):
        """ConversationContextクリアのテスト"""
        context = ConversationContext(conversation_id="test_001", character_id="001_aoi")

        # ターン追加
        turn = DialogueTurn(
            user_input="テスト",
            assistant_response="応答",
            timestamp=datetime.now(),
            turn_number=1,
            character_id="001_aoi",
        )
        context.add_turn(turn)
        context.state = ConversationState.PROCESSING

        # クリア
        context.clear()

        assert len(context.turns) == 0
        assert context.current_turn_number == 0
        assert context.state == ConversationState.IDLE

    def test_conversation_context_is_active(self):
        """ConversationContextのアクティブ状態テスト"""
        context = ConversationContext(conversation_id="test_001", character_id="001_aoi")

        # IDLE状態
        assert context.is_active() is False

        # PROCESSING状態
        context.state = ConversationState.PROCESSING
        assert context.is_active() is True

        # WAITING状態
        context.state = ConversationState.WAITING
        assert context.is_active() is True

        # ERROR状態
        context.state = ConversationState.ERROR
        assert context.is_active() is False

    # ------------------------------------------------------------------------
    # デバッグ用メソッドのテスト
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_current_context(self, initialized_manager):
        """現在のコンテキスト取得テスト"""
        context = initialized_manager.get_current_context()

        assert isinstance(context, ConversationContext)
        assert context.conversation_id.startswith("conv_")
        assert context.character_id == "001_aoi"

    @pytest.mark.asyncio
    async def test_repr(self, initialized_manager):
        """__repr__メソッドのテスト"""
        repr_str = repr(initialized_manager)

        assert "DialogueManager" in repr_str
        assert "state=ready" in repr_str
        assert "initialized=True" in repr_str
        assert "conv_" in repr_str
        assert "use_mock=True" in repr_str

    # ------------------------------------------------------------------------
    # エッジケースのテスト
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_state_transitions(self, manager):
        """状態遷移の正常性テスト"""
        # 初期状態
        assert manager._state == ComponentState.NOT_INITIALIZED

        # 初期化
        await manager.initialize()
        assert manager._state == ComponentState.READY

        # クリーンアップ
        await manager.cleanup()
        assert manager._state == ComponentState.TERMINATED

    @pytest.mark.asyncio
    async def test_operations_after_cleanup(self):
        """クリーンアップ後の操作テスト"""
        config = DialogueConfig()
        manager = DialogueManager(config)
        await manager.initialize()
        await manager.cleanup()

        # クリーンアップ後は操作不可
        with pytest.raises(RuntimeError, match="not initialized"):
            await manager.process_text_input("テスト")

        with pytest.raises(RuntimeError, match="not initialized"):
            manager.get_conversation_history()

        with pytest.raises(RuntimeError, match="not initialized"):
            manager.clear_conversation()

    @pytest.mark.asyncio
    async def test_without_character_manager(self):
        """キャラクターマネージャーなしでの動作テスト"""
        config = DialogueConfig()
        manager = DialogueManager(config)
        await manager.initialize()

        # キャラクターマネージャーなしでも動作
        turn = await manager.process_text_input("こんにちは")
        assert turn.assistant_response == "こんにちは！今日はどんなお話をしましょうか？"
        assert turn.character_id == "001_aoi"  # デフォルトID使用

        await manager.cleanup()
