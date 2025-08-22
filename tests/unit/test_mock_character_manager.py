"""MockCharacterManagerクラスの単体テスト

MockCharacterManagerのすべての機能を網羅的にテストします。
Phase 2のモック実装の品質を保証します。

テスト実装ガイド v1.3準拠
テスト戦略ガイドライン v1.7準拠
開発規約書 v1.12準拠
"""

from datetime import datetime

import pytest

from tests.mocks.mock_character_manager import Character, MockCharacterManager
from vioratalk.core.base import ComponentState


@pytest.mark.unit
@pytest.mark.phase(2)
class TestMockCharacterManager:
    """MockCharacterManagerクラスのテストスイート

    インターフェース定義書 v1.34準拠
    テスト戦略ガイドライン v1.7準拠
    """

    # ------------------------------------------------------------------------
    # フィクスチャ
    # ------------------------------------------------------------------------

    @pytest.fixture
    def manager(self) -> MockCharacterManager:
        """MockCharacterManagerのフィクスチャ

        Returns:
            MockCharacterManager: 未初期化のマネージャーインスタンス
        """
        return MockCharacterManager()

    @pytest.fixture
    async def initialized_manager(self) -> MockCharacterManager:
        """初期化済みMockCharacterManagerのフィクスチャ

        Returns:
            MockCharacterManager: 初期化済みのマネージャーインスタンス
        """
        manager = MockCharacterManager()
        await manager.initialize()
        return manager

    # ------------------------------------------------------------------------
    # 初期化のテスト
    # ------------------------------------------------------------------------

    def test_constructor(self, manager):
        """コンストラクタのテスト"""
        assert manager._state == ComponentState.NOT_INITIALIZED
        assert manager._current_character_id == "001_aoi"
        assert "001_aoi" in manager._characters
        assert not manager._initialized

    def test_default_character_creation(self, manager):
        """デフォルトキャラクター001_aoiの作成確認"""
        aoi = manager._characters["001_aoi"]

        assert aoi.character_id == "001_aoi"
        assert aoi.name == "あおい"
        assert aoi.description == "フレンドリーで親しみやすいアシスタント"
        assert aoi.voice_model_id == "ja-JP-NanamiNeural"
        assert aoi.language == "ja"
        assert aoi.is_default is True

        # 性格パラメータの確認
        assert aoi.personality_traits["formality"] == 60
        assert aoi.personality_traits["friendliness"] == 80
        assert aoi.personality_traits["humor"] == 50
        assert aoi.personality_traits["empathy"] == 85
        assert aoi.personality_traits["curiosity"] == 70
        assert aoi.personality_traits["assertiveness"] == 40
        assert aoi.personality_traits["creativity"] == 60
        assert aoi.personality_traits["analytical"] == 45

        # メタデータの確認
        assert aoi.metadata["version"] == "1.0"
        assert aoi.metadata["phase"] == 2
        assert aoi.metadata["mock"] is True

    @pytest.mark.asyncio
    async def test_initialize(self, manager):
        """初期化メソッドのテスト"""
        assert manager._state == ComponentState.NOT_INITIALIZED
        assert not manager._initialized

        await manager.initialize()

        assert manager._state == ComponentState.READY
        assert manager._initialized

    @pytest.mark.asyncio
    async def test_initialize_twice(self, initialized_manager):
        """二重初期化のエラーテスト"""
        with pytest.raises(RuntimeError, match="already initialized"):
            await initialized_manager.initialize()

    @pytest.mark.asyncio
    async def test_cleanup(self, initialized_manager):
        """クリーンアップメソッドのテスト"""
        assert initialized_manager._state == ComponentState.READY
        assert initialized_manager._initialized

        await initialized_manager.cleanup()

        assert initialized_manager._state == ComponentState.TERMINATED
        assert not initialized_manager._initialized
        assert initialized_manager._current_character_id == "001_aoi"

    @pytest.mark.asyncio
    async def test_cleanup_twice(self, initialized_manager):
        """二重クリーンアップのテスト"""
        await initialized_manager.cleanup()
        assert initialized_manager._state == ComponentState.TERMINATED

        # 二回目のクリーンアップは安全に何もしない
        await initialized_manager.cleanup()
        assert initialized_manager._state == ComponentState.TERMINATED

    # ------------------------------------------------------------------------
    # キャラクター取得のテスト
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_character(self, initialized_manager):
        """get_character()メソッドのテスト"""
        character = initialized_manager.get_character("001_aoi")

        assert character is not None
        assert character.character_id == "001_aoi"
        assert character.name == "あおい"

    @pytest.mark.asyncio
    async def test_get_character_not_found(self, initialized_manager):
        """存在しないキャラクターの取得テスト"""
        character = initialized_manager.get_character("999_unknown")
        assert character is None

    def test_get_character_not_initialized(self, manager):
        """未初期化状態でのget_character()エラーテスト"""
        with pytest.raises(RuntimeError, match="not initialized"):
            manager.get_character("001_aoi")

    @pytest.mark.asyncio
    async def test_get_current_character(self, initialized_manager):
        """get_current_character()メソッドのテスト"""
        character = initialized_manager.get_current_character()

        assert character is not None
        assert character.character_id == "001_aoi"
        assert character.name == "あおい"

    def test_get_current_character_not_initialized(self, manager):
        """未初期化状態でのget_current_character()エラーテスト"""
        with pytest.raises(RuntimeError, match="not initialized"):
            manager.get_current_character()

    # ------------------------------------------------------------------------
    # リスト取得のテスト
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_list_characters(self, initialized_manager):
        """list_characters()メソッドのテスト"""
        characters = initialized_manager.list_characters()

        assert isinstance(characters, list)
        assert len(characters) == 1
        assert "001_aoi" in characters

    def test_list_characters_not_initialized(self, manager):
        """未初期化状態でのlist_characters()エラーテスト"""
        with pytest.raises(RuntimeError, match="not initialized"):
            manager.list_characters()

    # ------------------------------------------------------------------------
    # キャラクター切り替えのテスト
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_switch_character(self, initialized_manager):
        """switch_character()メソッドのテスト"""
        # 001_aoiへの切り替え（既に選択されているが成功するはず）
        result = initialized_manager.switch_character("001_aoi")

        assert result is True
        assert initialized_manager._current_character_id == "001_aoi"

    @pytest.mark.asyncio
    async def test_switch_character_not_found(self, initialized_manager):
        """存在しないキャラクターへの切り替えテスト"""
        with pytest.raises(ValueError, match="Character '999_unknown' not found"):
            initialized_manager.switch_character("999_unknown")

    def test_switch_character_not_initialized(self, manager):
        """未初期化状態でのswitch_character()エラーテスト"""
        with pytest.raises(RuntimeError, match="not initialized"):
            manager.switch_character("001_aoi")

    # ------------------------------------------------------------------------
    # 利用可能性チェックのテスト
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_is_character_available(self, initialized_manager):
        """is_character_available()メソッドのテスト"""
        # 存在するキャラクター
        assert initialized_manager.is_character_available("001_aoi") is True

        # 存在しないキャラクター
        assert initialized_manager.is_character_available("999_unknown") is False

    def test_is_character_available_not_initialized(self, manager):
        """未初期化状態でのis_character_available()エラーテスト"""
        with pytest.raises(RuntimeError, match="not initialized"):
            manager.is_character_available("001_aoi")

    # ------------------------------------------------------------------------
    # デフォルトキャラクターのテスト
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_default_character_id(self, initialized_manager):
        """get_default_character_id()メソッドのテスト"""
        default_id = initialized_manager.get_default_character_id()
        assert default_id == "001_aoi"

    def test_get_default_character_id_not_initialized(self, manager):
        """未初期化状態でのget_default_character_id()エラーテスト"""
        with pytest.raises(RuntimeError, match="not initialized"):
            manager.get_default_character_id()

    # ------------------------------------------------------------------------
    # Phase 7用メソッドのテスト（Phase 2では未実装）
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_add_character_not_implemented(self, initialized_manager):
        """add_character()メソッドのテスト（Phase 2では未実装）"""
        new_character = Character(character_id="002_yuki", name="ゆき", description="テスト用キャラクター")

        result = initialized_manager.add_character(new_character)
        assert result is False  # Phase 2では常にFalse

    @pytest.mark.asyncio
    async def test_remove_character_not_implemented(self, initialized_manager):
        """remove_character()メソッドのテスト（Phase 2では未実装）"""
        result = initialized_manager.remove_character("001_aoi")
        assert result is False  # Phase 2では常にFalse

    @pytest.mark.asyncio
    async def test_update_character_not_implemented(self, initialized_manager):
        """update_character()メソッドのテスト（Phase 2では未実装）"""
        updates = {"name": "新しい名前"}
        result = initialized_manager.update_character("001_aoi", updates)
        assert result is False  # Phase 2では常にFalse

    # ------------------------------------------------------------------------
    # 文字列表現のテスト
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_repr(self, initialized_manager):
        """__repr__()メソッドのテスト"""
        repr_str = repr(initialized_manager)

        assert "MockCharacterManager" in repr_str
        assert "state=ready" in repr_str
        assert "initialized=True" in repr_str
        assert "current_character=001_aoi" in repr_str
        assert "['001_aoi']" in repr_str

    # ------------------------------------------------------------------------
    # Characterデータクラスのテスト
    # ------------------------------------------------------------------------

    def test_character_dataclass_default_traits(self):
        """Characterデータクラスのデフォルト性格パラメータテスト"""
        character = Character(character_id="test_001", name="テスト")

        # デフォルトの性格パラメータが設定される
        assert character.personality_traits["formality"] == 50
        assert character.personality_traits["friendliness"] == 70
        assert character.personality_traits["humor"] == 40
        assert character.personality_traits["empathy"] == 80
        assert character.personality_traits["curiosity"] == 60
        assert character.personality_traits["assertiveness"] == 30
        assert character.personality_traits["creativity"] == 50
        assert character.personality_traits["analytical"] == 40

    def test_character_dataclass_custom_traits(self):
        """Characterデータクラスのカスタム性格パラメータテスト"""
        custom_traits = {
            "formality": 100,
            "friendliness": 0,
            "humor": 50,
            "empathy": 50,
            "curiosity": 50,
            "assertiveness": 50,
            "creativity": 50,
            "analytical": 50,
        }

        character = Character(
            character_id="test_002", name="カスタム", personality_traits=custom_traits
        )

        assert character.personality_traits == custom_traits

    def test_character_dataclass_defaults(self):
        """Characterデータクラスのデフォルト値テスト"""
        character = Character(character_id="test_003", name="デフォルト")

        assert character.description == ""
        assert character.voice_model_id == "ja-JP-NanamiNeural"
        assert character.language == "ja"
        assert character.is_default is False
        assert isinstance(character.created_at, datetime)
        assert character.metadata == {}

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
        manager = MockCharacterManager()
        await manager.initialize()
        await manager.cleanup()

        # クリーンアップ後は操作不可
        with pytest.raises(RuntimeError, match="not initialized"):
            manager.get_character("001_aoi")

        with pytest.raises(RuntimeError, match="not initialized"):
            manager.list_characters()
