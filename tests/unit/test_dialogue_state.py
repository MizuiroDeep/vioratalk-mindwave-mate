"""DialogueTurnクラスの単体テスト

DialogueTurnデータクラスのすべての機能を網羅的にテストします。
Phase 2の対話システム構築における中核データ構造の品質を保証します。

テスト実装ガイド v1.3準拠
テスト戦略ガイドライン v1.7準拠
開発規約書 v1.12準拠
"""

import json
from datetime import datetime, timedelta
from typing import Any, Dict

import pytest

from vioratalk.core.dialogue_state import DialogueTurn


@pytest.mark.unit
@pytest.mark.phase(2)
class TestDialogueTurn:
    """DialogueTurnクラスのテストスイート

    インターフェース定義書 v1.34 セクション2.4準拠
    データフォーマット仕様書 v1.5準拠
    """

    # ------------------------------------------------------------------------
    # フィクスチャ
    # ------------------------------------------------------------------------

    @pytest.fixture
    def valid_turn_data(self) -> Dict[str, Any]:
        """有効なDialogueTurnデータのフィクスチャ

        Returns:
            Dict[str, Any]: テスト用の有効なデータ
        """
        return {
            "user_input": "こんにちは、今日の天気はどうですか？",
            "assistant_response": "こんにちは！今日は晴れて気持ちの良い天気ですね。",
            "timestamp": datetime.now(),
            "turn_number": 1,
            "character_id": "001_aoi",
        }

    @pytest.fixture
    def complete_turn_data(self, valid_turn_data) -> Dict[str, Any]:
        """すべてのフィールドを含むDialogueTurnデータ

        Args:
            valid_turn_data: 有効なデータの基本セット

        Returns:
            Dict[str, Any]: オプションフィールドも含む完全なデータ
        """
        data = valid_turn_data.copy()
        data.update(
            {
                "emotion": "happy",
                "confidence": 0.95,
                "processing_time": 1.234,
                "metadata": {"session_id": "test_session_001", "context": "weather"},
            }
        )
        return data

    @pytest.fixture
    def dialogue_turn(self, valid_turn_data) -> DialogueTurn:
        """基本的なDialogueTurnインスタンス

        Args:
            valid_turn_data: 有効なデータ

        Returns:
            DialogueTurn: テスト用インスタンス
        """
        return DialogueTurn(**valid_turn_data)

    # ------------------------------------------------------------------------
    # 初期化とフィールドのテスト
    # ------------------------------------------------------------------------

    def test_initialization_with_required_fields(self, valid_turn_data):
        """必須フィールドのみでの初期化"""
        turn = DialogueTurn(**valid_turn_data)

        assert turn.user_input == valid_turn_data["user_input"]
        assert turn.assistant_response == valid_turn_data["assistant_response"]
        assert turn.timestamp == valid_turn_data["timestamp"]
        assert turn.turn_number == valid_turn_data["turn_number"]
        assert turn.character_id == valid_turn_data["character_id"]

        # オプションフィールドのデフォルト値確認
        assert turn.emotion is None
        assert turn.confidence is None
        assert turn.processing_time is None
        assert turn.metadata == {}

    def test_initialization_with_all_fields(self, complete_turn_data):
        """すべてのフィールドを含む初期化"""
        turn = DialogueTurn(**complete_turn_data)

        # 必須フィールド
        assert turn.user_input == complete_turn_data["user_input"]
        assert turn.assistant_response == complete_turn_data["assistant_response"]
        assert turn.timestamp == complete_turn_data["timestamp"]
        assert turn.turn_number == complete_turn_data["turn_number"]
        assert turn.character_id == complete_turn_data["character_id"]

        # オプションフィールド
        assert turn.emotion == "happy"
        assert turn.confidence == 0.95
        assert turn.processing_time == 1.234
        assert turn.metadata == {"session_id": "test_session_001", "context": "weather"}

    # ------------------------------------------------------------------------
    # 入力値検証のテスト（__post_init__）
    # ------------------------------------------------------------------------

    def test_invalid_turn_number_zero(self, valid_turn_data):
        """ターン番号が0の場合のエラー"""
        valid_turn_data["turn_number"] = 0
        with pytest.raises(ValueError, match="turn_number must be >= 1"):
            DialogueTurn(**valid_turn_data)

    def test_invalid_turn_number_negative(self, valid_turn_data):
        """ターン番号が負の場合のエラー"""
        valid_turn_data["turn_number"] = -1
        with pytest.raises(ValueError, match="turn_number must be >= 1"):
            DialogueTurn(**valid_turn_data)

    def test_invalid_confidence_below_zero(self, complete_turn_data):
        """確信度が0未満の場合のエラー"""
        complete_turn_data["confidence"] = -0.1
        with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
            DialogueTurn(**complete_turn_data)

    def test_invalid_confidence_above_one(self, complete_turn_data):
        """確信度が1を超える場合のエラー"""
        complete_turn_data["confidence"] = 1.1
        with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
            DialogueTurn(**complete_turn_data)

    def test_valid_confidence_boundaries(self, complete_turn_data):
        """確信度の境界値テスト（0.0と1.0は有効）"""
        # 0.0は有効
        complete_turn_data["confidence"] = 0.0
        turn = DialogueTurn(**complete_turn_data)
        assert turn.confidence == 0.0

        # 1.0も有効
        complete_turn_data["confidence"] = 1.0
        turn = DialogueTurn(**complete_turn_data)
        assert turn.confidence == 1.0

    def test_invalid_processing_time_negative(self, complete_turn_data):
        """処理時間が負の場合のエラー"""
        complete_turn_data["processing_time"] = -0.001
        with pytest.raises(ValueError, match="processing_time must be >= 0"):
            DialogueTurn(**complete_turn_data)

    def test_valid_processing_time_zero(self, complete_turn_data):
        """処理時間0は有効"""
        complete_turn_data["processing_time"] = 0.0
        turn = DialogueTurn(**complete_turn_data)
        assert turn.processing_time == 0.0

    def test_invalid_character_id_empty(self, valid_turn_data):
        """キャラクターIDが空文字列の場合のエラー"""
        valid_turn_data["character_id"] = ""
        with pytest.raises(ValueError, match="character_id cannot be empty"):
            DialogueTurn(**valid_turn_data)

    def test_invalid_character_id_whitespace(self, valid_turn_data):
        """キャラクターIDが空白のみの場合のエラー"""
        valid_turn_data["character_id"] = "   "
        with pytest.raises(ValueError, match="character_id cannot be empty"):
            DialogueTurn(**valid_turn_data)

    # ------------------------------------------------------------------------
    # to_dict()メソッドのテスト
    # ------------------------------------------------------------------------

    def test_to_dict_with_required_fields(self, dialogue_turn):
        """必須フィールドのみの辞書変換"""
        result = dialogue_turn.to_dict()

        assert isinstance(result, dict)
        assert result["user_input"] == dialogue_turn.user_input
        assert result["assistant_response"] == dialogue_turn.assistant_response
        assert result["timestamp"] == dialogue_turn.timestamp.isoformat()
        assert result["turn_number"] == dialogue_turn.turn_number
        assert result["character_id"] == dialogue_turn.character_id

        # オプションフィールドもキーとして存在（値はNone）
        assert "emotion" in result
        assert result["emotion"] is None
        assert "confidence" in result
        assert result["confidence"] is None
        assert "processing_time" in result
        assert result["processing_time"] is None
        assert "metadata" in result
        assert result["metadata"] == {}

    def test_to_dict_with_all_fields(self, complete_turn_data):
        """すべてのフィールドを含む辞書変換"""
        turn = DialogueTurn(**complete_turn_data)
        result = turn.to_dict()

        assert result["user_input"] == complete_turn_data["user_input"]
        assert result["assistant_response"] == complete_turn_data["assistant_response"]
        assert result["timestamp"] == complete_turn_data["timestamp"].isoformat()
        assert result["turn_number"] == complete_turn_data["turn_number"]
        assert result["character_id"] == complete_turn_data["character_id"]
        assert result["emotion"] == "happy"
        assert result["confidence"] == 0.95
        assert result["processing_time"] == 1.234
        assert result["metadata"] == {"session_id": "test_session_001", "context": "weather"}

    def test_to_dict_json_serializable(self, dialogue_turn):
        """to_dict()の結果がJSON変換可能であること"""
        result = dialogue_turn.to_dict()

        # JSON変換が例外を発生させないことを確認
        json_str = json.dumps(result)
        assert isinstance(json_str, str)

        # 逆変換も可能
        restored = json.loads(json_str)
        assert restored["user_input"] == dialogue_turn.user_input

    # ------------------------------------------------------------------------
    # from_dict()メソッドのテスト
    # ------------------------------------------------------------------------

    def test_from_dict_with_required_fields(self, dialogue_turn):
        """必須フィールドのみからの復元"""
        data = dialogue_turn.to_dict()
        restored = DialogueTurn.from_dict(data)

        assert restored.user_input == dialogue_turn.user_input
        assert restored.assistant_response == dialogue_turn.assistant_response
        assert restored.timestamp == dialogue_turn.timestamp
        assert restored.turn_number == dialogue_turn.turn_number
        assert restored.character_id == dialogue_turn.character_id
        assert restored.emotion is None
        assert restored.confidence is None
        assert restored.processing_time is None
        assert restored.metadata == {}

    def test_from_dict_with_all_fields(self, complete_turn_data):
        """すべてのフィールドからの復元"""
        turn = DialogueTurn(**complete_turn_data)
        data = turn.to_dict()
        restored = DialogueTurn.from_dict(data)

        assert restored.user_input == turn.user_input
        assert restored.assistant_response == turn.assistant_response
        assert restored.timestamp == turn.timestamp
        assert restored.turn_number == turn.turn_number
        assert restored.character_id == turn.character_id
        assert restored.emotion == turn.emotion
        assert restored.confidence == turn.confidence
        assert restored.processing_time == turn.processing_time
        assert restored.metadata == turn.metadata

    def test_from_dict_with_string_timestamp(self, dialogue_turn):
        """文字列形式のタイムスタンプからの復元"""
        data = dialogue_turn.to_dict()
        # 文字列形式のタイムスタンプ
        assert isinstance(data["timestamp"], str)

        restored = DialogueTurn.from_dict(data)
        assert isinstance(restored.timestamp, datetime)
        assert restored.timestamp == dialogue_turn.timestamp

    def test_from_dict_with_datetime_timestamp(self, valid_turn_data):
        """datetime形式のタイムスタンプからの復元"""
        # datetimeオブジェクトをそのまま含むデータ
        data = {
            "user_input": valid_turn_data["user_input"],
            "assistant_response": valid_turn_data["assistant_response"],
            "timestamp": valid_turn_data["timestamp"],  # datetimeオブジェクト
            "turn_number": valid_turn_data["turn_number"],
            "character_id": valid_turn_data["character_id"],
        }

        restored = DialogueTurn.from_dict(data)
        assert isinstance(restored.timestamp, datetime)
        assert restored.timestamp == valid_turn_data["timestamp"]

    def test_from_dict_missing_required_field(self):
        """必須フィールドが不足している場合のエラー"""
        incomplete_data = {
            "user_input": "test",
            "assistant_response": "response",
            # timestampが不足
            "turn_number": 1,
            "character_id": "001_aoi",
        }

        with pytest.raises(KeyError):
            DialogueTurn.from_dict(incomplete_data)

    def test_from_dict_with_invalid_values(self, valid_turn_data):
        """不正な値を含むデータからの復元時のエラー"""
        data = {
            "user_input": valid_turn_data["user_input"],
            "assistant_response": valid_turn_data["assistant_response"],
            "timestamp": datetime.now().isoformat(),
            "turn_number": 0,  # 不正な値
            "character_id": valid_turn_data["character_id"],
        }

        with pytest.raises(ValueError, match="turn_number must be >= 1"):
            DialogueTurn.from_dict(data)

    # ------------------------------------------------------------------------
    # get_summary()メソッドのテスト
    # ------------------------------------------------------------------------

    def test_get_summary_short_text(self, dialogue_turn):
        """短いテキストの概要"""
        summary = dialogue_turn.get_summary()

        assert f"Turn #{dialogue_turn.turn_number}" in summary
        assert f"[{dialogue_turn.character_id}]" in summary
        assert dialogue_turn.user_input in summary
        # 短いテキストなので省略されない
        assert "..." not in summary

    def test_get_summary_long_text(self, valid_turn_data):
        """長いテキストの概要（省略あり）"""
        valid_turn_data["user_input"] = "あ" * 50  # 50文字
        valid_turn_data["assistant_response"] = "い" * 50  # 50文字
        turn = DialogueTurn(**valid_turn_data)

        summary = turn.get_summary()

        # 30文字で切られて...が付く
        assert ("あ" * 30 + "...") in summary
        assert ("い" * 30 + "...") in summary

    def test_get_summary_exact_30_chars(self, valid_turn_data):
        """ちょうど30文字のテキストの概要"""
        valid_turn_data["user_input"] = "x" * 30  # ちょうど30文字
        valid_turn_data["assistant_response"] = "y" * 30
        turn = DialogueTurn(**valid_turn_data)

        summary = turn.get_summary()

        # ちょうど30文字なので省略されない
        assert ("x" * 30) in summary
        assert ("y" * 30) in summary
        assert "..." not in summary

    def test_get_summary_format(self, dialogue_turn):
        """概要のフォーマット確認"""
        summary = dialogue_turn.get_summary()

        # 期待されるフォーマット: Turn #N [character_id]: "input" -> "response"
        expected_pattern = (
            f"Turn #{dialogue_turn.turn_number} "
            f"[{dialogue_turn.character_id}]: "
            f'"{dialogue_turn.user_input}" -> '
        )
        assert expected_pattern in summary

    # ------------------------------------------------------------------------
    # 文字列表現のテスト
    # ------------------------------------------------------------------------

    def test_str_representation(self, dialogue_turn):
        """__str__メソッドのテスト"""
        str_repr = str(dialogue_turn)

        # get_summary()と同じ結果
        assert str_repr == dialogue_turn.get_summary()

    def test_repr_representation(self, dialogue_turn):
        """__repr__メソッドのテスト"""
        repr_str = repr(dialogue_turn)

        # デバッグ用の詳細情報を含む
        assert "DialogueTurn(" in repr_str
        assert f"turn_number={dialogue_turn.turn_number}" in repr_str
        assert f"character_id='{dialogue_turn.character_id}'" in repr_str
        assert "timestamp=" in repr_str
        assert dialogue_turn.timestamp.isoformat() in repr_str
        assert "emotion=None" in repr_str
        assert "confidence=None" in repr_str

    def test_repr_with_optional_fields(self, complete_turn_data):
        """オプションフィールドを含む__repr__"""
        turn = DialogueTurn(**complete_turn_data)
        repr_str = repr(turn)

        assert "emotion=happy" in repr_str
        assert "confidence=0.95" in repr_str

    # ------------------------------------------------------------------------
    # エッジケースと境界値のテスト
    # ------------------------------------------------------------------------

    def test_empty_strings_in_text_fields(self, valid_turn_data):
        """テキストフィールドが空文字列の場合（有効）"""
        valid_turn_data["user_input"] = ""
        valid_turn_data["assistant_response"] = ""

        # 空文字列は有効（character_id以外）
        turn = DialogueTurn(**valid_turn_data)
        assert turn.user_input == ""
        assert turn.assistant_response == ""

    def test_very_long_text(self, valid_turn_data):
        """非常に長いテキストの処理"""
        long_text = "テスト" * 1000  # 2000文字
        valid_turn_data["user_input"] = long_text
        valid_turn_data["assistant_response"] = long_text

        turn = DialogueTurn(**valid_turn_data)
        assert turn.user_input == long_text
        assert turn.assistant_response == long_text

        # 概要では省略される
        summary = turn.get_summary()
        assert len(summary) < len(long_text) * 2

    def test_special_characters_in_text(self, valid_turn_data):
        """特殊文字を含むテキスト"""
        special_text = '改行\nタブ\t引用符"エスケープ\\'
        valid_turn_data["user_input"] = special_text
        valid_turn_data["assistant_response"] = special_text

        turn = DialogueTurn(**valid_turn_data)
        assert turn.user_input == special_text
        assert turn.assistant_response == special_text

    def test_unicode_characters(self, valid_turn_data):
        """Unicode文字（絵文字など）を含むテキスト"""
        unicode_text = "こんにちは😊 🎉✨"
        valid_turn_data["user_input"] = unicode_text
        valid_turn_data["assistant_response"] = unicode_text

        turn = DialogueTurn(**valid_turn_data)
        assert turn.user_input == unicode_text
        assert turn.assistant_response == unicode_text

    def test_large_turn_number(self, valid_turn_data):
        """大きなターン番号"""
        valid_turn_data["turn_number"] = 999999
        turn = DialogueTurn(**valid_turn_data)
        assert turn.turn_number == 999999

    def test_future_timestamp(self, valid_turn_data):
        """未来のタイムスタンプ（有効）"""
        future_time = datetime.now() + timedelta(days=365)
        valid_turn_data["timestamp"] = future_time

        turn = DialogueTurn(**valid_turn_data)
        assert turn.timestamp == future_time

    def test_past_timestamp(self, valid_turn_data):
        """過去のタイムスタンプ（有効）"""
        past_time = datetime(2000, 1, 1, 0, 0, 0)
        valid_turn_data["timestamp"] = past_time

        turn = DialogueTurn(**valid_turn_data)
        assert turn.timestamp == past_time

    def test_metadata_with_nested_structure(self, complete_turn_data):
        """ネストした構造を持つメタデータ"""
        complete_turn_data["metadata"] = {
            "session": {"id": "test", "user": {"name": "TestUser", "preferences": ["ja", "en"]}},
            "flags": [True, False, None],
        }

        turn = DialogueTurn(**complete_turn_data)
        assert turn.metadata["session"]["id"] == "test"
        assert turn.metadata["session"]["user"]["name"] == "TestUser"
        assert turn.metadata["flags"] == [True, False, None]

    # ------------------------------------------------------------------------
    # 相互運用性のテスト
    # ------------------------------------------------------------------------

    def test_round_trip_conversion(self, complete_turn_data):
        """to_dict → from_dict の往復変換"""
        original = DialogueTurn(**complete_turn_data)

        # 辞書に変換
        data = original.to_dict()

        # 辞書から復元
        restored = DialogueTurn.from_dict(data)

        # すべてのフィールドが一致
        assert restored.user_input == original.user_input
        assert restored.assistant_response == original.assistant_response
        assert restored.timestamp == original.timestamp
        assert restored.turn_number == original.turn_number
        assert restored.character_id == original.character_id
        assert restored.emotion == original.emotion
        assert restored.confidence == original.confidence
        assert restored.processing_time == original.processing_time
        assert restored.metadata == original.metadata

    def test_json_round_trip(self, dialogue_turn):
        """JSON経由の往復変換"""
        # オブジェクト → 辞書 → JSON文字列
        data = dialogue_turn.to_dict()
        json_str = json.dumps(data)

        # JSON文字列 → 辞書 → オブジェクト
        loaded_data = json.loads(json_str)
        restored = DialogueTurn.from_dict(loaded_data)

        assert restored.user_input == dialogue_turn.user_input
        assert restored.assistant_response == dialogue_turn.assistant_response
        assert restored.timestamp == dialogue_turn.timestamp
        assert restored.turn_number == dialogue_turn.turn_number
        assert restored.character_id == dialogue_turn.character_id
