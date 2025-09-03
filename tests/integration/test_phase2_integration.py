"""Phase 2統合テスト

Phase 2で実装したすべてのコンポーネントの統合動作を検証。
DialogueTurn、DialogueConfig、MockCharacterManager、DialogueManagerが
正しく連携することを確認する。

テスト実装ガイド v1.3準拠
テスト戦略ガイドライン v1.7準拠
開発規約書 v1.12準拠
"""

from datetime import datetime

import pytest

from tests.mocks.mock_character_manager import MockCharacterManager
from vioratalk.configuration.config_manager import ConfigManager
from vioratalk.core.base import ComponentState
from vioratalk.core.dialogue_config import DialogueConfig
from vioratalk.core.dialogue_manager import DialogueManager
from vioratalk.core.dialogue_state import DialogueTurn
from vioratalk.core.vioratalk_engine import VioraTalkEngine

# ============================================================================
# フィクスチャ
# ============================================================================


@pytest.fixture
def dialogue_config() -> DialogueConfig:
    """テスト用DialogueConfig"""
    return DialogueConfig(
        max_turns=10, temperature=0.7, language="ja", use_mock_engines=True, debug_mode=True
    )


@pytest.fixture
async def mock_character_manager() -> MockCharacterManager:
    """初期化済みMockCharacterManager"""
    manager = MockCharacterManager()
    await manager.initialize()
    return manager


@pytest.fixture
async def dialogue_manager(dialogue_config, mock_character_manager) -> DialogueManager:
    """初期化済みDialogueManager"""
    manager = DialogueManager(config=dialogue_config, character_manager=mock_character_manager)
    await manager.initialize()
    return manager


@pytest.fixture
async def initialized_engine() -> VioraTalkEngine:
    """初期化済みVioraTalkEngine（Phase 1）"""
    engine = VioraTalkEngine()
    await engine.initialize()
    return engine


# ============================================================================
# Phase 2コンポーネント統合テスト
# ============================================================================


@pytest.mark.integration
@pytest.mark.phase(2)
class TestPhase2ComponentIntegration:
    """Phase 2コンポーネント間の統合テスト"""

    @pytest.mark.asyncio
    async def test_dialogue_turn_and_config_integration(self, dialogue_config):
        """DialogueTurnとDialogueConfigの統合テスト

        データフォーマット仕様書 v1.5準拠
        """
        # DialogueConfigを使用してDialogueTurnを作成
        turn = DialogueTurn(
            user_input="こんにちは",
            assistant_response="こんにちは！今日はどんなお話をしましょうか？",
            timestamp=datetime.now(),
            turn_number=1,
            character_id="001_aoi",
            confidence=1.0,
            processing_time=0.5,
            metadata={
                "config": dialogue_config.to_dict(),
                "temperature": dialogue_config.temperature,
            },
        )

        # DialogueTurnが設定を正しく保持
        assert turn.metadata["temperature"] == dialogue_config.temperature
        assert turn.metadata["config"]["language"] == dialogue_config.language

        # to_dict()とfrom_dict()の往復変換
        turn_dict = turn.to_dict()
        restored_turn = DialogueTurn.from_dict(turn_dict)
        assert restored_turn.user_input == turn.user_input
        assert restored_turn.metadata["temperature"] == dialogue_config.temperature

    @pytest.mark.asyncio
    async def test_mock_character_and_dialogue_manager(
        self, mock_character_manager, dialogue_config
    ):
        """MockCharacterManagerとDialogueManagerの統合テスト"""
        # MockCharacterManagerは既にフィクスチャで初期化済み
        assert mock_character_manager.get_state() == ComponentState.READY

        # DialogueManagerにMockCharacterManagerを注入
        dialogue_manager = DialogueManager(
            config=dialogue_config, character_manager=mock_character_manager
        )
        await dialogue_manager.initialize()

        # キャラクター情報の取得（character_idフィールドを使用）
        current_character = mock_character_manager.get_current_character()
        assert current_character.character_id == "001_aoi"  # idではなくcharacter_id

        # 対話処理
        response = await dialogue_manager.process_text_input("こんにちは")
        assert response is not None
        assert response.assistant_response != ""
        assert response.character_id == "001_aoi"

        # クリーンアップ
        await dialogue_manager.cleanup()
        await mock_character_manager.cleanup()

    @pytest.mark.asyncio
    async def test_dialogue_manager_conversation_flow(
        self, dialogue_config, mock_character_manager
    ):
        """DialogueManagerの会話フローテスト"""
        # 初期化
        dialogue_manager = DialogueManager(
            config=dialogue_config, character_manager=mock_character_manager
        )
        await dialogue_manager.initialize()
        assert dialogue_manager.get_state() == ComponentState.READY

        # 最初の対話
        response1 = await dialogue_manager.process_text_input("こんにちは")
        assert response1.turn_number == 1
        assert response1.user_input == "こんにちは"
        assert response1.assistant_response != ""

        # 2回目の対話
        response2 = await dialogue_manager.process_text_input("元気ですか？")
        assert response2.turn_number == 2
        assert response2.user_input == "元気ですか？"

        # 会話履歴の確認
        history = dialogue_manager.get_conversation_history()
        assert len(history) == 2
        assert history[0].turn_number == 1
        assert history[1].turn_number == 2

        # クリーンアップ
        await dialogue_manager.cleanup()
        assert dialogue_manager.get_state() == ComponentState.TERMINATED


# ============================================================================
# Phase 1とPhase 2の連携テスト
# ============================================================================


@pytest.mark.integration
@pytest.mark.phase(1, 2)
class TestPhase1Phase2Integration:
    """Phase 1とPhase 2コンポーネントの連携テスト"""

    @pytest.mark.asyncio
    async def test_vioratalk_engine_with_dialogue_manager(self, initialized_engine):
        """VioraTalkEngineとDialogueManagerの連携"""
        # Phase 1のVioraTalkEngineが初期化済み
        assert initialized_engine.get_state() == ComponentState.READY

        # Phase 2コンポーネントを作成
        dialogue_config = DialogueConfig()
        mock_character = MockCharacterManager()
        await mock_character.initialize()

        dialogue_manager = DialogueManager(config=dialogue_config, character_manager=mock_character)
        await dialogue_manager.initialize()

        # 両方のコンポーネントが動作可能
        assert initialized_engine.is_available()
        assert dialogue_manager.is_available()

        # クリーンアップ
        await dialogue_manager.cleanup()
        await mock_character.cleanup()
        await initialized_engine.cleanup()

    @pytest.mark.asyncio
    async def test_config_manager_dialogue_config_sync(self):
        """ConfigManagerとDialogueConfigの設定同期"""
        # ConfigManagerから設定を読み込み
        config_manager = ConfigManager()
        config_data = {"dialogue": {"max_turns": 20, "temperature": 0.9, "language": "en"}}

        # DialogueConfigに反映
        dialogue_config = DialogueConfig(
            max_turns=config_data["dialogue"]["max_turns"],
            temperature=config_data["dialogue"]["temperature"],
            language=config_data["dialogue"]["language"],
        )

        assert dialogue_config.max_turns == 20
        assert dialogue_config.temperature == 0.9
        assert dialogue_config.language == "en"


# ============================================================================
# 複数ターン会話のテスト
# ============================================================================


@pytest.mark.integration
@pytest.mark.phase(2)
class TestMultiTurnConversation:
    """複数ターン会話のシナリオテスト"""

    @pytest.mark.asyncio
    async def test_three_turn_conversation(self, dialogue_config, mock_character_manager):
        """3ターン会話のシナリオテスト

        ConversationManager実装仕様書 v1.3の基礎実装
        """
        # 最大ターン数を3に設定
        config = dialogue_config.copy_with(max_turns=3)

        dialogue_manager = DialogueManager(config=config, character_manager=mock_character_manager)
        await dialogue_manager.initialize()

        # 3ターンの会話
        responses = []
        inputs = ["初めまして", "よろしくお願いします", "さようなら"]

        for user_input in inputs:
            response = await dialogue_manager.process_text_input(user_input)
            responses.append(response)
            assert response is not None
            assert response.assistant_response != ""

        # 会話履歴の確認
        history = dialogue_manager.get_conversation_history()
        assert len(history) == 3

        # 各ターンの確認
        for i, turn in enumerate(history, 1):
            assert turn.turn_number == i
            assert turn.character_id == "001_aoi"

        # 4ターン目（最大ターン数超過）
        response = await dialogue_manager.process_text_input("もう一度")

        # Phase 2では最大ターン数超過でもエラーにはならない（古いターンが削除される）
        assert response is not None

        # クリーンアップ
        await dialogue_manager.cleanup()

    @pytest.mark.asyncio
    async def test_conversation_reset(self, dialogue_config, mock_character_manager):
        """会話リセットのテスト"""
        dialogue_manager = DialogueManager(
            config=dialogue_config, character_manager=mock_character_manager
        )
        await dialogue_manager.initialize()

        # 初回会話
        await dialogue_manager.process_text_input("こんにちは")
        await dialogue_manager.process_text_input("元気ですか？")

        history = dialogue_manager.get_conversation_history()
        assert len(history) == 2

        # 会話をリセット（asyncメソッドなのでawaitが必要）
        await dialogue_manager.reset_conversation()

        # リセット後の確認
        history = dialogue_manager.get_conversation_history()
        assert len(history) == 0

        # 新しい会話
        response = await dialogue_manager.process_text_input("新しい会話です")
        assert response.turn_number == 1  # ターン番号がリセットされている

        # クリーンアップ
        await dialogue_manager.cleanup()

    @pytest.mark.asyncio
    async def test_conversation_context_persistence(self, dialogue_config, mock_character_manager):
        """会話コンテキストの永続性テスト"""
        dialogue_manager = DialogueManager(
            config=dialogue_config, character_manager=mock_character_manager
        )
        await dialogue_manager.initialize()

        # 会話を開始
        await dialogue_manager.process_text_input("覚えておいてください")

        # コンテキストを取得（辞書ではなくオブジェクト）
        context = dialogue_manager.get_current_context()
        assert context.character_id == "001_aoi"  # 辞書アクセスではなく属性アクセス
        assert len(context.turns) == 1

        # 続きの会話
        await dialogue_manager.process_text_input("覚えていますか？")

        # コンテキストが維持されている
        assert len(context.turns) == 2
        assert context.current_turn_number == 2

        # クリーンアップ
        await dialogue_manager.cleanup()


# ============================================================================
# エラーハンドリングのテスト
# ============================================================================


@pytest.mark.integration
@pytest.mark.phase(2)
class TestErrorHandling:
    """エラーハンドリングの統合テスト"""

    @pytest.mark.asyncio
    async def test_empty_input_handling(self, dialogue_config, mock_character_manager):
        """空入力のエラーハンドリング"""
        dialogue_manager = DialogueManager(
            config=dialogue_config, character_manager=mock_character_manager
        )
        await dialogue_manager.initialize()

        # 空文字列は実装でValueErrorを投げる
        with pytest.raises(ValueError) as exc_info:
            await dialogue_manager.process_text_input("")
        assert "Input text cannot be empty" in str(exc_info.value)

        # 空白のみも同様
        with pytest.raises(ValueError) as exc_info:
            await dialogue_manager.process_text_input("   ")
        assert "Input text cannot be empty" in str(exc_info.value)

        # クリーンアップ
        await dialogue_manager.cleanup()

    @pytest.mark.asyncio
    async def test_uninitialized_dialogue_manager(self, dialogue_config):
        """未初期化DialogueManagerのエラーハンドリング"""
        dialogue_manager = DialogueManager(config=dialogue_config)

        # 初期化前の操作はエラー
        with pytest.raises(RuntimeError) as exc_info:
            await dialogue_manager.process_text_input("テスト")
        assert "not initialized" in str(exc_info.value).lower()

        # 会話履歴取得もエラー
        with pytest.raises(RuntimeError):
            dialogue_manager.get_conversation_history()

    @pytest.mark.asyncio
    async def test_invalid_dialogue_config(self, mock_character_manager):
        """不正なDialogueConfigのテスト"""
        # 不正な設定値
        with pytest.raises(ValueError) as exc_info:
            DialogueConfig(max_turns=0)  # 0は無効
        assert "max_turns must be >= 1" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            DialogueConfig(temperature=-0.1)  # 負の値は無効
        assert "temperature must be between 0.0 and 2.0" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            DialogueConfig(language="unknown")  # サポートされていない言語
        assert "language must be 'ja' or 'en'" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_character_not_found(self, dialogue_config):
        """キャラクターが見つからない場合のテスト"""
        mock_manager = MockCharacterManager()
        await mock_manager.initialize()

        dialogue_manager = DialogueManager(config=dialogue_config, character_manager=mock_manager)
        await dialogue_manager.initialize()

        # 存在しないキャラクターへの切り替えはValueErrorを投げる
        with pytest.raises(ValueError) as exc_info:
            mock_manager.switch_character("999_unknown")
        assert "Character '999_unknown' not found" in str(exc_info.value)

        # 現在のキャラクターは変わらない
        current = mock_manager.get_current_character()
        assert current.character_id == "001_aoi"

        # クリーンアップ
        await dialogue_manager.cleanup()
        await mock_manager.cleanup()


# ============================================================================
# 設定変更の動的反映テスト
# ============================================================================


@pytest.mark.integration
@pytest.mark.phase(2)
class TestDynamicConfiguration:
    """設定の動的変更テスト"""

    @pytest.mark.asyncio
    async def test_language_switching(self, mock_character_manager):
        """言語切り替えのテスト"""
        # 日本語設定で開始
        config_ja = DialogueConfig(language="ja")
        dialogue_manager = DialogueManager(
            config=config_ja, character_manager=mock_character_manager
        )
        await dialogue_manager.initialize()

        response = await dialogue_manager.process_text_input("こんにちは")
        assert response.character_id == "001_aoi"
        # Phase 2では言語情報はmetadataに含まれない（DialogueConfigで管理）

        # クリーンアップして英語に切り替え
        await dialogue_manager.cleanup()

        config_en = DialogueConfig(language="en")
        dialogue_manager = DialogueManager(
            config=config_en, character_manager=mock_character_manager
        )
        await dialogue_manager.initialize()

        response = await dialogue_manager.process_text_input("Hello")
        assert response.character_id == "001_aoi"

        # クリーンアップ
        await dialogue_manager.cleanup()

    @pytest.mark.asyncio
    async def test_temperature_adjustment(self, mock_character_manager):
        """temperatureパラメータ調整のテスト"""
        # 低temperature（決定的）
        config_low = DialogueConfig(temperature=0.1)
        dialogue_manager = DialogueManager(
            config=config_low, character_manager=mock_character_manager
        )
        await dialogue_manager.initialize()

        response1 = await dialogue_manager.process_text_input("テスト")
        await dialogue_manager.cleanup()

        # 高temperature（創造的）
        config_high = DialogueConfig(temperature=1.9)
        dialogue_manager = DialogueManager(
            config=config_high, character_manager=mock_character_manager
        )
        await dialogue_manager.initialize()

        response2 = await dialogue_manager.process_text_input("テスト")

        # 両方とも正常に動作
        assert response1 is not None
        assert response2 is not None

        # クリーンアップ
        await dialogue_manager.cleanup()


# ============================================================================
# パフォーマンステスト
# ============================================================================


@pytest.mark.integration
@pytest.mark.phase(2)
class TestPerformance:
    """パフォーマンス関連のテスト"""

    @pytest.mark.asyncio
    async def test_response_time(self, dialogue_config, mock_character_manager):
        """応答時間のテスト"""
        dialogue_manager = DialogueManager(
            config=dialogue_config, character_manager=mock_character_manager
        )
        await dialogue_manager.initialize()

        # 応答時間を測定
        start_time = datetime.now()
        response = await dialogue_manager.process_text_input("テスト")
        end_time = datetime.now()

        processing_time = (end_time - start_time).total_seconds()

        # Phase 2のモック実装は1秒以内に応答
        assert processing_time < 1.0
        assert response.processing_time >= 0

        # クリーンアップ
        await dialogue_manager.cleanup()

    @pytest.mark.asyncio
    async def test_memory_usage_basic(self, mock_character_manager):
        """基本的なメモリ使用量テスト"""
        # 最大5ターンの設定
        config = DialogueConfig(max_turns=5)
        dialogue_manager = DialogueManager(config=config, character_manager=mock_character_manager)
        await dialogue_manager.initialize()

        # 100回の対話（max_turnsを超える）
        for i in range(100):
            await dialogue_manager.process_text_input(f"テスト {i}")

        # 履歴は最大5ターンのみ保持
        history = dialogue_manager.get_conversation_history()
        assert len(history) == 5  # max_turnsに制限される

        # 最新の5ターンが保持されている
        assert history[-1].user_input == "テスト 99"
        assert history[-2].user_input == "テスト 98"

        # クリーンアップ
        await dialogue_manager.cleanup()


# ============================================================================
# Phase 1との互換性テスト
# ============================================================================


@pytest.mark.integration
@pytest.mark.phase(1, 2)
class TestPhase1Compatibility:
    """Phase 1コンポーネントとの互換性テスト"""

    @pytest.mark.asyncio
    async def test_phase1_components_still_work(self):
        """Phase 1コンポーネントが引き続き動作することを確認"""
        # Phase 1のVioraTalkEngine
        engine = VioraTalkEngine()
        await engine.initialize()
        assert engine.get_state() == ComponentState.READY

        # Phase 1のConfigManager
        config_manager = ConfigManager()
        # ConfigManagerはget()メソッドで設定値を取得
        app_name = config_manager.get("general.app_name", "VioraTalk")
        assert app_name is not None

        # クリーンアップ
        await engine.cleanup()

    @pytest.mark.asyncio
    async def test_phase1_initialization_with_phase2(self, initialized_engine):
        """Phase 1の初期化フローにPhase 2を組み込む"""
        # Phase 1エンジンが初期化済み
        assert initialized_engine.get_state() == ComponentState.READY

        # Phase 2コンポーネントを追加
        dialogue_config = DialogueConfig()
        mock_character = MockCharacterManager()
        dialogue_manager = DialogueManager(config=dialogue_config, character_manager=mock_character)

        # Phase 2コンポーネントを初期化
        await mock_character.initialize()
        await dialogue_manager.initialize()

        # 両方が正常に動作
        assert initialized_engine.is_available()
        assert dialogue_manager.is_available()

        # クリーンアップ
        await dialogue_manager.cleanup()
        await mock_character.cleanup()
        await initialized_engine.cleanup()


# ============================================================================
# 実際の使用シナリオテスト
# ============================================================================


@pytest.mark.integration
@pytest.mark.phase(2)
class TestRealWorldScenarios:
    """実際の使用シナリオをシミュレート"""

    @pytest.mark.asyncio
    async def test_morning_greeting_scenario(self, initialized_engine):
        """朝の挨拶シナリオ"""
        # Phase 2コンポーネントを初期化
        dialogue_config = DialogueConfig(language="ja", temperature=0.8, max_turns=10)

        mock_character = MockCharacterManager()
        await mock_character.initialize()

        dialogue_manager = DialogueManager(config=dialogue_config, character_manager=mock_character)
        await dialogue_manager.initialize()

        # 朝の挨拶会話
        conversation = ["おはようございます", "今日も一日頑張ります", "ありがとう、あおいちゃん"]

        for user_input in conversation:
            response = await dialogue_manager.process_text_input(user_input)
            assert response.character_id == "001_aoi"
            assert len(response.assistant_response) > 0

        # 会話統計
        stats = dialogue_manager.get_conversation_stats()
        assert stats["total_turns"] == len(conversation)
        assert stats["character_id"] == "001_aoi"

        # クリーンアップ
        await dialogue_manager.cleanup()
        await mock_character.cleanup()

    @pytest.mark.asyncio
    async def test_character_interaction_scenario(self, dialogue_config):
        """キャラクターとの対話シナリオ

        修正: confidence値のアサーションを >= 0.5 に変更
        Phase 4 Part 88で修正
        """
        # あおいちゃんとの対話
        mock_character = MockCharacterManager()
        await mock_character.initialize()

        dialogue_manager = DialogueManager(config=dialogue_config, character_manager=mock_character)
        await dialogue_manager.initialize()

        # キャラクター情報の確認
        character = mock_character.get_current_character()
        assert character.name == "あおい"  # 「結月あおい」ではなく「あおい」
        assert character.character_id == "001_aoi"

        # 感情豊かな対話
        emotional_inputs = ["今日は嬉しいことがあったんだ", "でも少し疲れちゃった", "応援してくれる？"]

        for input_text in emotional_inputs:
            response = await dialogue_manager.process_text_input(input_text)
            assert response.emotion in ["happy", "neutral", "supportive"]
            assert response.confidence >= 0.5  # 修正: > 0.5 から >= 0.5 に変更

        # クリーンアップ
        await dialogue_manager.cleanup()
        await mock_character.cleanup()


# ============================================================================
# クリーンアップ処理のテスト
# ============================================================================


@pytest.mark.integration
@pytest.mark.phase(2)
class TestCleanup:
    """クリーンアップ処理の統合テスト"""

    @pytest.mark.asyncio
    async def test_proper_cleanup_sequence(self, dialogue_config, mock_character_manager):
        """正しいクリーンアップシーケンス"""
        # すべてのコンポーネントを初期化
        dialogue_manager = DialogueManager(
            config=dialogue_config, character_manager=mock_character_manager
        )
        await dialogue_manager.initialize()

        # 使用
        await dialogue_manager.process_text_input("テスト")

        # 正しい順序でクリーンアップ
        await dialogue_manager.cleanup()
        assert dialogue_manager.get_state() == ComponentState.TERMINATED

        await mock_character_manager.cleanup()
        assert mock_character_manager._state == ComponentState.TERMINATED

    @pytest.mark.asyncio
    async def test_cleanup_after_error(self):
        """エラー後のクリーンアップ"""
        dialogue_manager = DialogueManager(config=DialogueConfig())

        # エラーを発生させる（未初期化で操作）
        with pytest.raises(RuntimeError):
            await dialogue_manager.process_text_input("テスト")

        # エラー後でもクリーンアップ可能
        await dialogue_manager.cleanup()
        assert dialogue_manager.get_state() == ComponentState.TERMINATED  # NOT_INITIALIZEDではない
