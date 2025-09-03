"""DialogueConfigのテスト

DialogueConfigクラスの動作を検証する。
Phase 2の基本機能とPhase 4で追加された音声パラメータの両方をテスト。

テスト戦略ガイドライン v1.7準拠
テスト実装ガイド v1.3準拠
開発規約書 v1.12準拠
"""


import pytest

from vioratalk.core.dialogue_config import DialogueConfig, ResponseMode

# ============================================================================
# DialogueConfigの基本テスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(2)
class TestDialogueConfigBasics:
    """DialogueConfigの基本機能テスト"""

    def test_default_initialization(self):
        """デフォルト値での初期化"""
        config = DialogueConfig()

        # Phase 2基本設定
        assert config.max_turns == 100
        assert config.turn_timeout == 30.0
        assert config.response_mode == ResponseMode.SIMPLE
        assert config.temperature == 0.7
        assert config.max_response_length == 500
        assert config.enable_memory is False
        assert config.enable_emotion is False
        assert config.default_character_id == "001_aoi"
        assert config.language == "ja"
        assert config.debug_mode is False
        assert config.log_conversation is True
        assert config.use_mock_engines is True
        assert config.auto_save_interval == 300
        assert config.metadata == {}

        # Phase 4音声設定（暫定追加）
        assert config.tts_enabled is True
        assert config.tts_enabled_for_text is True
        assert config.voice_id is None
        assert config.voice_style is None
        assert config.recording_duration == 5.0
        assert config.sample_rate == 16000

    def test_custom_initialization(self):
        """カスタム値での初期化"""
        config = DialogueConfig(
            max_turns=50,
            temperature=0.9,
            debug_mode=True,
            language="en",
            # Phase 4音声設定
            tts_enabled=False,
            recording_duration=10.0,
            sample_rate=44100,
            voice_id="voice_001",
            voice_style="cheerful",
        )

        # Phase 2設定
        assert config.max_turns == 50
        assert config.temperature == 0.9
        assert config.debug_mode is True
        assert config.language == "en"

        # Phase 4音声設定
        assert config.tts_enabled is False
        assert config.recording_duration == 10.0
        assert config.sample_rate == 44100
        assert config.voice_id == "voice_001"
        assert config.voice_style == "cheerful"

    def test_metadata_initialization(self):
        """メタデータの初期化"""
        metadata = {"version": "1.0", "custom_key": "custom_value"}
        config = DialogueConfig(metadata=metadata)

        assert config.metadata == metadata
        # dataclassは引数をそのまま使用する（標準的な動作）
        assert config.metadata is metadata

        # default_factoryの動作確認（デフォルト値の場合）
        config2 = DialogueConfig()  # metadataを指定しない
        config3 = DialogueConfig()  # metadataを指定しない
        # それぞれ独立した辞書が作成される
        assert config2.metadata is not config3.metadata
        assert config2.metadata == {}
        assert config3.metadata == {}


# ============================================================================
# Phase 4音声パラメータのテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestDialogueConfigPhase4Audio:
    """Phase 4で追加された音声パラメータのテスト"""

    def test_tts_configuration(self):
        """TTS設定のテスト"""
        config = DialogueConfig(
            tts_enabled=False,
            tts_enabled_for_text=False,
            voice_id="japanese_female_001",
            voice_style="calm",
        )

        assert config.tts_enabled is False
        assert config.tts_enabled_for_text is False
        assert config.voice_id == "japanese_female_001"
        assert config.voice_style == "calm"

    def test_audio_input_configuration(self):
        """音声入力設定のテスト"""
        config = DialogueConfig(recording_duration=15.0, sample_rate=48000)

        assert config.recording_duration == 15.0
        assert config.sample_rate == 48000

    def test_valid_sample_rates(self):
        """有効なサンプリングレートのテスト"""
        valid_rates = [8000, 16000, 22050, 44100, 48000]

        for rate in valid_rates:
            config = DialogueConfig(sample_rate=rate)
            assert config.sample_rate == rate

    def test_invalid_sample_rate(self):
        """無効なサンプリングレートのテスト"""
        with pytest.raises(ValueError) as exc_info:
            DialogueConfig(sample_rate=12000)  # 無効な値

        assert "sample_rate must be one of" in str(exc_info.value)

    def test_invalid_recording_duration(self):
        """無効な録音時間のテスト"""
        with pytest.raises(ValueError) as exc_info:
            DialogueConfig(recording_duration=0)

        assert "recording_duration must be > 0" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            DialogueConfig(recording_duration=-1.0)

        assert "recording_duration must be > 0" in str(exc_info.value)


# ============================================================================
# バリデーションのテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(2, 4)
class TestDialogueConfigValidation:
    """設定値検証のテスト"""

    def test_post_init_validation(self):
        """__post_init__での検証"""
        # max_turnsの検証
        with pytest.raises(ValueError) as exc_info:
            DialogueConfig(max_turns=0)
        assert "max_turns must be >= 1" in str(exc_info.value)

        # turn_timeoutの検証
        with pytest.raises(ValueError) as exc_info:
            DialogueConfig(turn_timeout=0)
        assert "turn_timeout must be > 0" in str(exc_info.value)

        # temperatureの検証
        with pytest.raises(ValueError) as exc_info:
            DialogueConfig(temperature=-0.1)
        assert "temperature must be between 0.0 and 2.0" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            DialogueConfig(temperature=2.1)
        assert "temperature must be between 0.0 and 2.0" in str(exc_info.value)

        # max_response_lengthの検証
        with pytest.raises(ValueError) as exc_info:
            DialogueConfig(max_response_length=0)
        assert "max_response_length must be >= 1" in str(exc_info.value)

        # languageの検証
        with pytest.raises(ValueError) as exc_info:
            DialogueConfig(language="fr")  # フランス語は未サポート
        assert "language must be 'ja' or 'en'" in str(exc_info.value)

        # auto_save_intervalの検証
        with pytest.raises(ValueError) as exc_info:
            DialogueConfig(auto_save_interval=-1)
        assert "auto_save_interval must be >= 0" in str(exc_info.value)

    def test_validate_method_warnings(self):
        """validate()メソッドの警告テスト"""
        # Phase 2警告
        config = DialogueConfig(
            max_turns=1500,  # 多すぎる
            turn_timeout=400,  # 長すぎる
            temperature=0.05,  # 低すぎる
            enable_memory=True,  # Phase 2では未実装
            enable_emotion=True,  # Phase 2では未実装
            use_mock_engines=False,  # Phase 2ではモックのみ
        )

        warnings = config.validate()
        assert len(warnings) >= 6

        assert any("max_turns" in w and "1500" in w for w in warnings)
        assert any("turn_timeout" in w and "400" in w for w in warnings)
        assert any("temperature" in w and "0.05" in w for w in warnings)
        assert any("enable_memory" in w for w in warnings)
        assert any("enable_emotion" in w for w in warnings)
        assert any("use_mock_engines" in w for w in warnings)

    def test_validate_phase4_warnings(self):
        """Phase 4音声設定の警告テスト"""
        # 録音時間が長すぎる
        config = DialogueConfig(recording_duration=60.0)
        warnings = config.validate()
        assert any("recording_duration" in w and "60.0" in w for w in warnings)

        # サンプリングレートが高すぎる
        config = DialogueConfig(sample_rate=48000)
        warnings = config.validate()
        assert any("sample_rate" in w and "48000" in w for w in warnings)

        # TTSが無効なのにvoice設定がある
        config = DialogueConfig(tts_enabled=False, voice_id="test_voice", voice_style="happy")
        warnings = config.validate()
        assert any("TTS is disabled" in w for w in warnings)

    def test_validate_no_warnings(self):
        """警告なしの場合"""
        config = DialogueConfig()  # デフォルト値は妥当
        warnings = config.validate()
        assert warnings == []


# ============================================================================
# 辞書変換のテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(2, 4)
class TestDialogueConfigSerialization:
    """シリアライゼーションのテスト"""

    def test_to_dict(self):
        """to_dict()メソッドのテスト"""
        config = DialogueConfig(
            max_turns=50,
            debug_mode=True,
            metadata={"key": "value"},
            # Phase 4音声設定
            tts_enabled=False,
            voice_id="test_voice",
            recording_duration=10.0,
        )

        data = config.to_dict()

        # Phase 2設定
        assert data["max_turns"] == 50
        assert data["debug_mode"] is True
        assert data["response_mode"] == "simple"
        assert data["metadata"] == {"key": "value"}

        # Phase 4音声設定
        assert data["tts_enabled"] is False
        assert data["voice_id"] == "test_voice"
        assert data["recording_duration"] == 10.0
        assert data["sample_rate"] == 16000  # デフォルト値

    def test_from_dict(self):
        """from_dict()メソッドのテスト"""
        data = {
            "max_turns": 75,
            "temperature": 0.8,
            "response_mode": "simple",
            "language": "en",
            "metadata": {"version": "2.0"},
            # Phase 4音声設定
            "tts_enabled_for_text": False,
            "voice_style": "energetic",
            "sample_rate": 44100,
        }

        config = DialogueConfig.from_dict(data)

        # Phase 2設定
        assert config.max_turns == 75
        assert config.temperature == 0.8
        assert config.response_mode == ResponseMode.SIMPLE
        assert config.language == "en"
        assert config.metadata == {"version": "2.0"}

        # Phase 4音声設定
        assert config.tts_enabled_for_text is False
        assert config.voice_style == "energetic"
        assert config.sample_rate == 44100

        # 未指定のフィールドはデフォルト値
        assert config.debug_mode is False
        assert config.tts_enabled is True
        assert config.recording_duration == 5.0

    def test_from_dict_with_unknown_fields(self):
        """未知のフィールドを含む辞書からの生成"""
        data = {
            "max_turns": 100,
            "unknown_field": "ignored",  # 未知のフィールド
            "future_feature": True,  # 将来の機能
            # Phase 4音声設定
            "sample_rate": 22050,
        }

        config = DialogueConfig.from_dict(data)

        assert config.max_turns == 100
        assert config.sample_rate == 22050
        # 未知のフィールドは無視される
        assert not hasattr(config, "unknown_field")
        assert not hasattr(config, "future_feature")

    def test_from_dict_response_mode_conversion(self):
        """ResponseModeの変換テスト"""
        # 文字列からEnum変換
        data = {"response_mode": "simple"}
        config = DialogueConfig.from_dict(data)
        assert config.response_mode == ResponseMode.SIMPLE

        # 無効なモードはSIMPLEにフォールバック
        data = {"response_mode": "invalid_mode"}
        config = DialogueConfig.from_dict(data)
        assert config.response_mode == ResponseMode.SIMPLE

    def test_round_trip_conversion(self):
        """to_dict → from_dict の往復変換"""
        original = DialogueConfig(
            max_turns=200,
            temperature=1.2,
            debug_mode=True,
            language="en",
            metadata={"test": "data"},
            # Phase 4音声設定
            tts_enabled=False,
            voice_id="custom_voice",
            voice_style="neutral",
            recording_duration=8.0,
            sample_rate=22050,
        )

        # 辞書に変換して再構築
        data = original.to_dict()
        restored = DialogueConfig.from_dict(data)

        # Phase 2設定の確認
        assert restored.max_turns == original.max_turns
        assert restored.temperature == original.temperature
        assert restored.debug_mode == original.debug_mode
        assert restored.language == original.language
        assert restored.metadata == original.metadata

        # Phase 4音声設定の確認
        assert restored.tts_enabled == original.tts_enabled
        assert restored.voice_id == original.voice_id
        assert restored.voice_style == original.voice_style
        assert restored.recording_duration == original.recording_duration
        assert restored.sample_rate == original.sample_rate


# ============================================================================
# ユーティリティメソッドのテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(2, 4)
class TestDialogueConfigUtilities:
    """ユーティリティメソッドのテスト"""

    def test_get_summary(self):
        """get_summary()メソッドのテスト"""
        config = DialogueConfig(
            max_turns=50,
            debug_mode=True,
            # Phase 4音声設定
            tts_enabled=False,
            recording_duration=10.0,
            sample_rate=44100,
        )

        summary = config.get_summary()

        # 主要な情報が含まれていることを確認
        assert "max_turns=50" in summary
        assert "debug=True" in summary
        assert "mode=simple" in summary
        # Phase 4追加
        assert "tts=False" in summary
        assert "rec_dur=10.0s" in summary
        assert "sr=44100Hz" in summary

    def test_copy_with(self):
        """copy_with()メソッドのテスト"""
        original = DialogueConfig(max_turns=100, debug_mode=False)

        # Phase 2設定の変更
        modified = original.copy_with(max_turns=50, debug_mode=True, temperature=0.9)

        assert modified.max_turns == 50
        assert modified.debug_mode is True
        assert modified.temperature == 0.9
        # 変更されていない項目は元の値を保持
        assert modified.language == original.language
        assert modified.use_mock_engines == original.use_mock_engines

        # Phase 4音声設定の変更
        audio_modified = original.copy_with(
            recording_duration=15.0, sample_rate=48000, voice_id="new_voice"
        )

        assert audio_modified.recording_duration == 15.0
        assert audio_modified.sample_rate == 48000
        assert audio_modified.voice_id == "new_voice"
        # 元のインスタンスは変更されない
        assert original.recording_duration == 5.0
        assert original.sample_rate == 16000
        assert original.voice_id is None

    def test_str_representation(self):
        """文字列表現のテスト"""
        config = DialogueConfig()

        str_repr = str(config)
        assert "DialogueConfig:" in str_repr
        assert "max_turns=" in str_repr
        # Phase 4追加
        assert "tts=" in str_repr
        assert "sr=" in str_repr

    def test_repr_representation(self):
        """開発用文字列表現のテスト"""
        config = DialogueConfig(max_turns=50, debug_mode=True, tts_enabled=False, sample_rate=44100)

        repr_str = repr(config)
        assert "DialogueConfig(" in repr_str
        assert "max_turns=50" in repr_str
        assert "debug_mode=True" in repr_str
        # Phase 4追加
        assert "tts_enabled=False" in repr_str
        assert "sample_rate=44100" in repr_str


# ============================================================================
# エッジケースのテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(2, 4)
class TestDialogueConfigEdgeCases:
    """エッジケースのテスト"""

    def test_extreme_values(self):
        """極端な値のテスト"""
        # 最小値
        config = DialogueConfig(
            max_turns=1,
            turn_timeout=0.1,
            temperature=0.0,
            max_response_length=1,
            auto_save_interval=0,
            # Phase 4
            recording_duration=0.1,
            sample_rate=8000,  # 最小の有効値
        )

        assert config.max_turns == 1
        assert config.turn_timeout == 0.1
        assert config.temperature == 0.0
        assert config.recording_duration == 0.1
        assert config.sample_rate == 8000

        # 最大値
        config = DialogueConfig(
            max_turns=10000,
            turn_timeout=3600.0,
            temperature=2.0,
            max_response_length=100000,
            # Phase 4
            recording_duration=3600.0,
            sample_rate=48000,  # 最大の有効値
        )

        assert config.max_turns == 10000
        assert config.turn_timeout == 3600.0
        assert config.temperature == 2.0
        assert config.recording_duration == 3600.0
        assert config.sample_rate == 48000

        # 警告は出るが、エラーにはならない
        warnings = config.validate()
        assert len(warnings) > 0  # 警告あり

    def test_immutability_protection(self):
        """データクラスの不変性は保証されない（フィールドは変更可能）"""
        config = DialogueConfig()

        # フィールドは変更可能
        config.max_turns = 200
        assert config.max_turns == 200

        # Phase 4フィールドも変更可能
        config.recording_duration = 20.0
        assert config.recording_duration == 20.0

        # メタデータも変更可能
        config.metadata["new_key"] = "new_value"
        assert config.metadata["new_key"] == "new_value"

    def test_response_mode_enum_values(self):
        """ResponseModeのEnum値テスト"""
        assert ResponseMode.SIMPLE.value == "simple"
        assert ResponseMode.CREATIVE.value == "creative"
        assert ResponseMode.BALANCED.value == "balanced"

        # 全てのモードを列挙
        all_modes = list(ResponseMode)
        assert len(all_modes) == 3
        assert ResponseMode.SIMPLE in all_modes
        assert ResponseMode.CREATIVE in all_modes
        assert ResponseMode.BALANCED in all_modes
