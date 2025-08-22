"""DialogueConfigクラスの単体テスト

DialogueConfig設定クラスのすべての機能を網羅的にテストします。
Phase 2の対話システム設定の品質を保証します。

テスト実装ガイド v1.3準拠
テスト戦略ガイドライン v1.7準拠
開発規約書 v1.12準拠
"""

import json
from typing import Any, Dict

import pytest

from vioratalk.core.dialogue_config import DialogueConfig, ResponseMode


@pytest.mark.unit
@pytest.mark.phase(2)
class TestDialogueConfig:
    """DialogueConfigクラスのテストスイート

    設定ファイル完全仕様書 v1.2準拠
    DialogueManager統合ガイド v1.2準拠
    """

    # ------------------------------------------------------------------------
    # フィクスチャ
    # ------------------------------------------------------------------------

    @pytest.fixture
    def default_config(self) -> DialogueConfig:
        """デフォルト設定のフィクスチャ

        Returns:
            DialogueConfig: デフォルト値で初期化されたインスタンス
        """
        return DialogueConfig()

    @pytest.fixture
    def custom_config_data(self) -> Dict[str, Any]:
        """カスタム設定データのフィクスチャ

        Returns:
            Dict[str, Any]: カスタム設定値の辞書
        """
        return {
            "max_turns": 50,
            "turn_timeout": 60.0,
            "response_mode": "simple",
            "temperature": 0.8,
            "max_response_length": 1000,
            "enable_memory": True,
            "enable_emotion": True,
            "default_character_id": "002_yuki",
            "language": "en",
            "debug_mode": True,
            "log_conversation": False,
            "use_mock_engines": False,
            "auto_save_interval": 600,
            "metadata": {"session_id": "test_001"},
        }

    # ------------------------------------------------------------------------
    # デフォルト値のテスト
    # ------------------------------------------------------------------------

    def test_default_values(self, default_config):
        """デフォルト値が正しく設定されているか"""
        assert default_config.max_turns == 100
        assert default_config.turn_timeout == 30.0
        assert default_config.response_mode == ResponseMode.SIMPLE
        assert default_config.temperature == 0.7
        assert default_config.max_response_length == 500

        assert default_config.enable_memory is False
        assert default_config.enable_emotion is False
        assert default_config.default_character_id == "001_aoi"

        assert default_config.language == "ja"
        assert default_config.debug_mode is False
        assert default_config.log_conversation is True
        assert default_config.use_mock_engines is True
        assert default_config.auto_save_interval == 300
        assert default_config.metadata == {}

    # ------------------------------------------------------------------------
    # カスタム値での初期化テスト
    # ------------------------------------------------------------------------

    def test_custom_initialization(self, custom_config_data):
        """カスタム値での初期化"""
        # ResponseModeはEnumなので変換
        custom_config_data["response_mode"] = ResponseMode.SIMPLE
        config = DialogueConfig(**custom_config_data)

        assert config.max_turns == 50
        assert config.turn_timeout == 60.0
        assert config.response_mode == ResponseMode.SIMPLE
        assert config.temperature == 0.8
        assert config.max_response_length == 1000
        assert config.enable_memory is True
        assert config.enable_emotion is True
        assert config.default_character_id == "002_yuki"
        assert config.language == "en"
        assert config.debug_mode is True
        assert config.log_conversation is False
        assert config.use_mock_engines is False
        assert config.auto_save_interval == 600
        assert config.metadata == {"session_id": "test_001"}

    def test_partial_initialization(self):
        """一部の値のみ指定した初期化"""
        config = DialogueConfig(max_turns=200, debug_mode=True, temperature=1.2)

        # 指定した値
        assert config.max_turns == 200
        assert config.debug_mode is True
        assert config.temperature == 1.2

        # デフォルト値
        assert config.turn_timeout == 30.0
        assert config.response_mode == ResponseMode.SIMPLE
        assert config.language == "ja"

    # ------------------------------------------------------------------------
    # 入力値検証のテスト（__post_init__）
    # ------------------------------------------------------------------------

    def test_invalid_max_turns_zero(self):
        """max_turnsが0の場合のエラー"""
        with pytest.raises(ValueError, match="max_turns must be >= 1"):
            DialogueConfig(max_turns=0)

    def test_invalid_max_turns_negative(self):
        """max_turnsが負の場合のエラー"""
        with pytest.raises(ValueError, match="max_turns must be >= 1"):
            DialogueConfig(max_turns=-1)

    def test_invalid_turn_timeout_zero(self):
        """turn_timeoutが0の場合のエラー"""
        with pytest.raises(ValueError, match="turn_timeout must be > 0"):
            DialogueConfig(turn_timeout=0.0)

    def test_invalid_turn_timeout_negative(self):
        """turn_timeoutが負の場合のエラー"""
        with pytest.raises(ValueError, match="turn_timeout must be > 0"):
            DialogueConfig(turn_timeout=-1.0)

    def test_invalid_temperature_below_zero(self):
        """temperatureが0未満の場合のエラー"""
        with pytest.raises(ValueError, match="temperature must be between 0.0 and 2.0"):
            DialogueConfig(temperature=-0.1)

    def test_invalid_temperature_above_two(self):
        """temperatureが2を超える場合のエラー"""
        with pytest.raises(ValueError, match="temperature must be between 0.0 and 2.0"):
            DialogueConfig(temperature=2.1)

    def test_valid_temperature_boundaries(self):
        """temperatureの境界値テスト（0.0と2.0は有効）"""
        # 0.0は有効
        config = DialogueConfig(temperature=0.0)
        assert config.temperature == 0.0

        # 2.0も有効
        config = DialogueConfig(temperature=2.0)
        assert config.temperature == 2.0

    def test_invalid_max_response_length_zero(self):
        """max_response_lengthが0の場合のエラー"""
        with pytest.raises(ValueError, match="max_response_length must be >= 1"):
            DialogueConfig(max_response_length=0)

    def test_invalid_max_response_length_negative(self):
        """max_response_lengthが負の場合のエラー"""
        with pytest.raises(ValueError, match="max_response_length must be >= 1"):
            DialogueConfig(max_response_length=-1)

    def test_invalid_language(self):
        """サポートされていない言語の場合のエラー"""
        with pytest.raises(ValueError, match="language must be 'ja' or 'en'"):
            DialogueConfig(language="fr")

    def test_valid_languages(self):
        """サポートされている言語の確認"""
        # 日本語
        config = DialogueConfig(language="ja")
        assert config.language == "ja"

        # 英語
        config = DialogueConfig(language="en")
        assert config.language == "en"

    def test_invalid_auto_save_interval_negative(self):
        """auto_save_intervalが負の場合のエラー"""
        with pytest.raises(ValueError, match="auto_save_interval must be >= 0"):
            DialogueConfig(auto_save_interval=-1)

    def test_valid_auto_save_interval_zero(self):
        """auto_save_intervalが0は有効（自動保存無効）"""
        config = DialogueConfig(auto_save_interval=0)
        assert config.auto_save_interval == 0

    # ------------------------------------------------------------------------
    # to_dict()メソッドのテスト
    # ------------------------------------------------------------------------

    def test_to_dict_default(self, default_config):
        """デフォルト設定の辞書変換"""
        result = default_config.to_dict()

        assert isinstance(result, dict)
        assert result["max_turns"] == 100
        assert result["turn_timeout"] == 30.0
        assert result["response_mode"] == "simple"  # Enumの値
        assert result["temperature"] == 0.7
        assert result["max_response_length"] == 500
        assert result["enable_memory"] is False
        assert result["enable_emotion"] is False
        assert result["default_character_id"] == "001_aoi"
        assert result["language"] == "ja"
        assert result["debug_mode"] is False
        assert result["log_conversation"] is True
        assert result["use_mock_engines"] is True
        assert result["auto_save_interval"] == 300
        assert result["metadata"] == {}

    def test_to_dict_custom(self, custom_config_data):
        """カスタム設定の辞書変換"""
        custom_config_data["response_mode"] = ResponseMode.SIMPLE
        config = DialogueConfig(**custom_config_data)
        result = config.to_dict()

        assert result["max_turns"] == 50
        assert result["turn_timeout"] == 60.0
        assert result["response_mode"] == "simple"
        assert result["temperature"] == 0.8
        assert result["metadata"] == {"session_id": "test_001"}

    def test_to_dict_json_serializable(self, default_config):
        """to_dict()の結果がJSON変換可能であること"""
        result = default_config.to_dict()

        # JSON変換が例外を発生させないことを確認
        json_str = json.dumps(result)
        assert isinstance(json_str, str)

        # 逆変換も可能
        restored = json.loads(json_str)
        assert restored["max_turns"] == default_config.max_turns

    # ------------------------------------------------------------------------
    # from_dict()メソッドのテスト
    # ------------------------------------------------------------------------

    def test_from_dict_full(self, custom_config_data):
        """完全な辞書からの生成"""
        config = DialogueConfig.from_dict(custom_config_data)

        assert config.max_turns == 50
        assert config.turn_timeout == 60.0
        assert config.response_mode == ResponseMode.SIMPLE
        assert config.temperature == 0.8
        assert config.max_response_length == 1000
        assert config.metadata == {"session_id": "test_001"}

    def test_from_dict_partial(self):
        """部分的な辞書からの生成（デフォルト値使用）"""
        data = {"max_turns": 75, "debug_mode": True}
        config = DialogueConfig.from_dict(data)

        # 指定された値
        assert config.max_turns == 75
        assert config.debug_mode is True

        # デフォルト値
        assert config.turn_timeout == 30.0
        assert config.temperature == 0.7
        assert config.language == "ja"

    def test_from_dict_empty(self):
        """空の辞書からの生成（すべてデフォルト値）"""
        config = DialogueConfig.from_dict({})

        assert config.max_turns == 100
        assert config.turn_timeout == 30.0
        assert config.response_mode == ResponseMode.SIMPLE
        assert config.temperature == 0.7

    def test_from_dict_unknown_fields(self):
        """未知のフィールドを含む辞書からの生成（無視される）"""
        data = {"max_turns": 50, "unknown_field": "value", "another_unknown": 123}
        config = DialogueConfig.from_dict(data)

        assert config.max_turns == 50
        # 未知のフィールドは無視される
        assert not hasattr(config, "unknown_field")
        assert not hasattr(config, "another_unknown")

    def test_from_dict_invalid_response_mode(self):
        """無効なresponse_modeのフォールバック"""
        data = {"response_mode": "invalid_mode"}
        config = DialogueConfig.from_dict(data)

        # 無効なモードはSIMPLEにフォールバック
        assert config.response_mode == ResponseMode.SIMPLE

    def test_from_dict_with_invalid_values(self):
        """不正な値を含む辞書からの生成時のエラー"""
        data = {"max_turns": 0}  # 不正な値

        with pytest.raises(ValueError, match="max_turns must be >= 1"):
            DialogueConfig.from_dict(data)

    # ------------------------------------------------------------------------
    # validate()メソッドのテスト
    # ------------------------------------------------------------------------

    def test_validate_no_warnings(self, default_config):
        """警告なしの場合"""
        warnings = default_config.validate()
        assert warnings == []

    def test_validate_high_max_turns(self):
        """max_turnsが多すぎる場合の警告"""
        config = DialogueConfig(max_turns=1500)
        warnings = config.validate()

        assert len(warnings) > 0
        assert any("max_turns" in w and "1500" in w for w in warnings)

    def test_validate_long_timeout(self):
        """turn_timeoutが長すぎる場合の警告"""
        config = DialogueConfig(turn_timeout=400.0)
        warnings = config.validate()

        assert len(warnings) > 0
        assert any("turn_timeout" in w and "400" in w for w in warnings)

    def test_validate_low_temperature(self):
        """temperatureが低すぎる場合の警告"""
        config = DialogueConfig(temperature=0.05)
        warnings = config.validate()

        assert len(warnings) > 0
        assert any("temperature" in w and "deterministic" in w for w in warnings)

    def test_validate_high_temperature(self):
        """temperatureが高すぎる場合の警告"""
        config = DialogueConfig(temperature=1.8)
        warnings = config.validate()

        assert len(warnings) > 0
        assert any("temperature" in w and "random" in w for w in warnings)

    def test_validate_phase2_limitations(self):
        """Phase 2の機能制限に関する警告"""
        config = DialogueConfig(enable_memory=True, enable_emotion=True, use_mock_engines=False)
        warnings = config.validate()

        # 3つの警告が出るはず
        assert len(warnings) >= 3
        assert any("memory" in w and "not implemented" in w for w in warnings)
        assert any("emotion" in w and "not implemented" in w for w in warnings)
        assert any("mock_engines" in w and "not implemented" in w for w in warnings)

    # ------------------------------------------------------------------------
    # get_summary()メソッドのテスト
    # ------------------------------------------------------------------------

    def test_get_summary_default(self, default_config):
        """デフォルト設定の概要"""
        summary = default_config.get_summary()

        assert "DialogueConfig:" in summary
        assert "max_turns=100" in summary
        assert "timeout=30.0s" in summary
        assert "mode=simple" in summary
        assert "temp=0.7" in summary
        assert "lang=ja" in summary
        assert "debug=False" in summary
        assert "mock=True" in summary

    def test_get_summary_custom(self):
        """カスタム設定の概要"""
        config = DialogueConfig(max_turns=50, debug_mode=True, language="en")
        summary = config.get_summary()

        assert "max_turns=50" in summary
        assert "debug=True" in summary
        assert "lang=en" in summary

    # ------------------------------------------------------------------------
    # copy_with()メソッドのテスト
    # ------------------------------------------------------------------------

    def test_copy_with_single_change(self, default_config):
        """単一の設定変更"""
        new_config = default_config.copy_with(debug_mode=True)

        # 変更された値
        assert new_config.debug_mode is True

        # 変更されていない値
        assert new_config.max_turns == default_config.max_turns
        assert new_config.temperature == default_config.temperature

        # 元のインスタンスは変更されない
        assert default_config.debug_mode is False

    def test_copy_with_multiple_changes(self, default_config):
        """複数の設定変更"""
        new_config = default_config.copy_with(
            max_turns=200, temperature=1.0, language="en", debug_mode=True
        )

        assert new_config.max_turns == 200
        assert new_config.temperature == 1.0
        assert new_config.language == "en"
        assert new_config.debug_mode is True

        # 元のインスタンスは変更されない
        assert default_config.max_turns == 100
        assert default_config.temperature == 0.7

    def test_copy_with_response_mode(self, default_config):
        """ResponseModeの変更"""
        new_config = default_config.copy_with(response_mode=ResponseMode.SIMPLE)

        assert new_config.response_mode == ResponseMode.SIMPLE

    def test_copy_with_metadata(self, default_config):
        """メタデータの変更"""
        new_metadata = {"session": "test", "version": 2}
        new_config = default_config.copy_with(metadata=new_metadata)

        assert new_config.metadata == new_metadata
        assert default_config.metadata == {}

    def test_copy_with_invalid_values(self, default_config):
        """不正な値での変更時のエラー"""
        with pytest.raises(ValueError, match="max_turns must be >= 1"):
            default_config.copy_with(max_turns=0)

    # ------------------------------------------------------------------------
    # 文字列表現のテスト
    # ------------------------------------------------------------------------

    def test_str_representation(self, default_config):
        """__str__メソッドのテスト"""
        str_repr = str(default_config)

        # get_summary()と同じ結果
        assert str_repr == default_config.get_summary()
        assert "DialogueConfig:" in str_repr

    def test_repr_representation(self, default_config):
        """__repr__メソッドのテスト"""
        repr_str = repr(default_config)

        assert "DialogueConfig(" in repr_str
        assert "max_turns=100" in repr_str
        assert "turn_timeout=30.0" in repr_str
        assert "response_mode=ResponseMode.SIMPLE" in repr_str
        assert "temperature=0.7" in repr_str
        assert "language='ja'" in repr_str
        assert "debug_mode=False" in repr_str

    # ------------------------------------------------------------------------
    # ResponseMode Enumのテスト
    # ------------------------------------------------------------------------

    def test_response_mode_enum_values(self):
        """ResponseMode Enumの値確認"""
        assert ResponseMode.SIMPLE.value == "simple"
        assert ResponseMode.CREATIVE.value == "creative"
        assert ResponseMode.BALANCED.value == "balanced"

    def test_response_mode_enum_membership(self):
        """ResponseMode Enumのメンバーシップ"""
        assert ResponseMode.SIMPLE in ResponseMode
        assert ResponseMode.CREATIVE in ResponseMode
        assert ResponseMode.BALANCED in ResponseMode

    # ------------------------------------------------------------------------
    # 相互運用性のテスト
    # ------------------------------------------------------------------------

    def test_round_trip_conversion(self, custom_config_data):
        """to_dict → from_dict の往復変換"""
        original = DialogueConfig.from_dict(custom_config_data)

        # 辞書に変換
        data = original.to_dict()

        # 辞書から復元
        restored = DialogueConfig.from_dict(data)

        # すべてのフィールドが一致
        assert restored.max_turns == original.max_turns
        assert restored.turn_timeout == original.turn_timeout
        assert restored.response_mode == original.response_mode
        assert restored.temperature == original.temperature
        assert restored.max_response_length == original.max_response_length
        assert restored.enable_memory == original.enable_memory
        assert restored.enable_emotion == original.enable_emotion
        assert restored.default_character_id == original.default_character_id
        assert restored.language == original.language
        assert restored.debug_mode == original.debug_mode
        assert restored.log_conversation == original.log_conversation
        assert restored.use_mock_engines == original.use_mock_engines
        assert restored.auto_save_interval == original.auto_save_interval
        assert restored.metadata == original.metadata

    def test_json_round_trip(self, default_config):
        """JSON経由の往復変換"""
        # オブジェクト → 辞書 → JSON文字列
        data = default_config.to_dict()
        json_str = json.dumps(data)

        # JSON文字列 → 辞書 → オブジェクト
        loaded_data = json.loads(json_str)
        restored = DialogueConfig.from_dict(loaded_data)

        assert restored.max_turns == default_config.max_turns
        assert restored.turn_timeout == default_config.turn_timeout
        assert restored.response_mode == default_config.response_mode
        assert restored.temperature == default_config.temperature
