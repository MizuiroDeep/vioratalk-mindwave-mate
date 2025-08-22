"""モックキャラクター管理モジュール

Phase 2のテスト用にCharacterManagerのモック実装を提供する。
最小実装として「001_aoi」キャラクターのみをサポート。

インターフェース定義書 v1.34準拠
開発規約書 v1.12準拠
テストデータ・モック完全仕様書 v1.1準拠
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from vioratalk.core.base import ComponentState, VioraTalkComponent


@dataclass
class Character:
    """キャラクター情報を保持するデータクラス（Phase 2最小実装）

    Phase 2用の最小実装。Phase 7で完全実装予定。

    Attributes:
        character_id: キャラクター識別子（形式: 番号_名前）
        name: キャラクター表示名
        description: キャラクターの説明
        personality_traits: 性格パラメータ（0-100）
        voice_model_id: 音声モデルID
        language: 対応言語
        is_default: デフォルトキャラクターフラグ
        created_at: 作成日時
        metadata: 追加メタデータ
    """

    character_id: str
    name: str
    description: str = ""
    personality_traits: Dict[str, int] = field(default_factory=dict)
    voice_model_id: str = "ja-JP-NanamiNeural"
    language: str = "ja"
    is_default: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """初期化後の処理"""
        # デフォルトの性格パラメータを設定
        if not self.personality_traits:
            self.personality_traits = {
                "formality": 50,  # 丁寧さ
                "friendliness": 70,  # 親しみやすさ
                "humor": 40,  # ユーモア
                "empathy": 80,  # 共感性
                "curiosity": 60,  # 好奇心
                "assertiveness": 30,  # 積極性
                "creativity": 50,  # 創造性
                "analytical": 40,  # 分析的思考
            }


class MockCharacterManager(VioraTalkComponent):
    """キャラクター管理のモック実装

    Phase 2テスト用の最小実装。
    「001_aoi」キャラクターのみをサポート。

    インターフェース定義書 v1.34 セクション3.1準拠

    Attributes:
        _current_character_id: 現在のキャラクターID
        _characters: 利用可能なキャラクターの辞書
        _initialized: 初期化済みフラグ

    Example:
        >>> manager = MockCharacterManager()
        >>> await manager.initialize()
        >>> character = manager.get_character("001_aoi")
        >>> print(character.name)
        あおい
    """

    def __init__(self) -> None:
        """コンストラクタ

        MockCharacterManagerを初期化する。
        Phase 2では「001_aoi」のみを登録。
        """
        super().__init__()
        self._current_character_id: str = "001_aoi"
        self._characters: Dict[str, Character] = {}
        self._initialized: bool = False

        # デフォルトキャラクター「001_aoi」を作成
        self._create_default_character()

    def _create_default_character(self) -> None:
        """デフォルトキャラクター「001_aoi」を作成

        Phase 2用の最小実装キャラクターを作成する。
        """
        aoi = Character(
            character_id="001_aoi",
            name="あおい",
            description="フレンドリーで親しみやすいアシスタント",
            personality_traits={
                "formality": 60,  # やや丁寧
                "friendliness": 80,  # とてもフレンドリー
                "humor": 50,  # 適度なユーモア
                "empathy": 85,  # 高い共感性
                "curiosity": 70,  # 好奇心旺盛
                "assertiveness": 40,  # 控えめ
                "creativity": 60,  # 創造的
                "analytical": 45,  # バランス型
            },
            voice_model_id="ja-JP-NanamiNeural",
            language="ja",
            is_default=True,
            metadata={"version": "1.0", "phase": 2, "mock": True},
        )
        self._characters["001_aoi"] = aoi

    async def initialize(self) -> None:
        """非同期初期化

        コンポーネントを初期化し、使用可能な状態にする。
        Phase 2では即座に成功する。

        Raises:
            RuntimeError: 既に初期化済みの場合
        """
        if self._state == ComponentState.READY:
            raise RuntimeError("MockCharacterManager is already initialized")

        self._state = ComponentState.INITIALIZING

        # Phase 2では即座に初期化完了
        self._initialized = True
        self._state = ComponentState.READY

    async def cleanup(self) -> None:
        """リソースのクリーンアップ

        使用したリソースを解放する。
        Phase 2では特にクリーンアップ処理なし。
        """
        if self._state == ComponentState.TERMINATED:
            return

        self._state = ComponentState.TERMINATING

        # Phase 2では特別なクリーンアップ不要
        self._initialized = False
        self._current_character_id = "001_aoi"

        self._state = ComponentState.TERMINATED

    def get_character(self, character_id: str) -> Optional[Character]:
        """指定IDのキャラクターを取得

        Args:
            character_id: キャラクターID（形式: 番号_名前）

        Returns:
            Character: キャラクター情報。存在しない場合はNone

        Raises:
            RuntimeError: 未初期化の場合

        Example:
            >>> character = manager.get_character("001_aoi")
            >>> print(character.name)
            あおい
        """
        if not self._initialized:
            raise RuntimeError("MockCharacterManager is not initialized")

        return self._characters.get(character_id)

    def get_current_character(self) -> Optional[Character]:
        """現在のキャラクターを取得

        Returns:
            Character: 現在のキャラクター情報

        Raises:
            RuntimeError: 未初期化の場合
        """
        if not self._initialized:
            raise RuntimeError("MockCharacterManager is not initialized")

        return self._characters.get(self._current_character_id)

    def list_characters(self) -> List[str]:
        """利用可能なキャラクターIDのリストを取得

        Returns:
            List[str]: キャラクターIDのリスト

        Raises:
            RuntimeError: 未初期化の場合

        Example:
            >>> ids = manager.list_characters()
            >>> print(ids)
            ['001_aoi']
        """
        if not self._initialized:
            raise RuntimeError("MockCharacterManager is not initialized")

        return list(self._characters.keys())

    def switch_character(self, character_id: str) -> bool:
        """キャラクターを切り替え

        Args:
            character_id: 切り替え先のキャラクターID

        Returns:
            bool: 切り替えに成功した場合True

        Raises:
            RuntimeError: 未初期化の場合
            ValueError: 存在しないキャラクターIDの場合

        Example:
            >>> success = manager.switch_character("001_aoi")
            >>> print(success)
            True
        """
        if not self._initialized:
            raise RuntimeError("MockCharacterManager is not initialized")

        if character_id not in self._characters:
            raise ValueError(f"Character '{character_id}' not found")

        self._current_character_id = character_id
        return True

    def is_character_available(self, character_id: str) -> bool:
        """キャラクターが利用可能か確認

        Args:
            character_id: 確認するキャラクターID

        Returns:
            bool: 利用可能な場合True

        Raises:
            RuntimeError: 未初期化の場合
        """
        if not self._initialized:
            raise RuntimeError("MockCharacterManager is not initialized")

        return character_id in self._characters

    def get_default_character_id(self) -> str:
        """デフォルトキャラクターIDを取得

        Returns:
            str: デフォルトキャラクターID

        Raises:
            RuntimeError: 未初期化の場合
        """
        if not self._initialized:
            raise RuntimeError("MockCharacterManager is not initialized")

        # Phase 2では常に001_aoiを返す
        return "001_aoi"

    def add_character(self, character: Character) -> bool:
        """新しいキャラクターを追加（Phase 2では未実装）

        Args:
            character: 追加するキャラクター

        Returns:
            bool: 常にFalse（Phase 2では追加不可）

        Note:
            Phase 7で実装予定。Phase 2では001_aoiのみサポート。
        """
        # Phase 2では新規追加は不可
        return False

    def remove_character(self, character_id: str) -> bool:
        """キャラクターを削除（Phase 2では未実装）

        Args:
            character_id: 削除するキャラクターID

        Returns:
            bool: 常にFalse（Phase 2では削除不可）

        Note:
            Phase 7で実装予定。Phase 2では001_aoiのみサポート。
        """
        # Phase 2では削除は不可
        return False

    def update_character(self, character_id: str, updates: Dict[str, Any]) -> bool:
        """キャラクター情報を更新（Phase 2では未実装）

        Args:
            character_id: 更新するキャラクターID
            updates: 更新内容の辞書

        Returns:
            bool: 常にFalse（Phase 2では更新不可）

        Note:
            Phase 7で実装予定。Phase 2では静的な001_aoiのみサポート。
        """
        # Phase 2では更新は不可
        return False

    def __repr__(self) -> str:
        """開発用の詳細文字列表現

        Returns:
            str: オブジェクトの詳細情報
        """
        return (
            f"MockCharacterManager("
            f"state={self._state.value}, "
            f"initialized={self._initialized}, "
            f"current_character={self._current_character_id}, "
            f"available_characters={list(self._characters.keys())})"
        )
