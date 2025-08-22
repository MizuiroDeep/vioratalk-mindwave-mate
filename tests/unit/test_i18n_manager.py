"""I18nManagerの単体テスト

国際化管理機能の動作を検証。
Phase 2実装として、基本的な機能のテストに焦点を当てる。

テスト戦略ガイドライン v1.7準拠
テスト実装ガイド v1.3準拠
開発規約書 v1.12準拠
"""

from unittest.mock import patch

import pytest

from vioratalk.core.i18n_manager import I18nManager

# ============================================================================
# I18nManagerのテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(2)
class TestI18nManager:
    """I18nManagerクラスのテスト"""

    @pytest.fixture
    def i18n_ja(self):
        """日本語設定のI18nManagerフィクスチャ"""
        return I18nManager("ja")

    @pytest.fixture
    def i18n_en(self):
        """英語設定のI18nManagerフィクスチャ"""
        return I18nManager("en")

    # ------------------------------------------------------------------------
    # 初期化のテスト
    # ------------------------------------------------------------------------

    def test_initialization_with_japanese(self):
        """日本語での初期化"""
        i18n = I18nManager("ja")
        assert i18n.current_language == "ja"

    def test_initialization_with_english(self):
        """英語での初期化"""
        i18n = I18nManager("en")
        assert i18n.current_language == "en"

    def test_initialization_with_default(self):
        """デフォルト言語での初期化"""
        i18n = I18nManager()
        assert i18n.current_language == "ja"  # デフォルトは日本語

    def test_initialization_with_unsupported_language(self):
        """未サポート言語での初期化（フォールバック）"""
        i18n = I18nManager("fr")  # フランス語は未サポート
        assert i18n.current_language == "ja"  # 日本語にフォールバック

    def test_supported_languages_property(self):
        """サポート言語リストの取得"""
        i18n = I18nManager()
        languages = i18n.supported_languages

        assert isinstance(languages, list)
        assert "ja" in languages
        assert "en" in languages
        assert len(languages) == 2

        # リストのコピーが返されることを確認
        languages.append("fr")
        assert len(i18n.supported_languages) == 2

    # ------------------------------------------------------------------------
    # get_error_message()のテスト
    # ------------------------------------------------------------------------

    def test_get_error_message_japanese(self, i18n_ja):
        """日本語エラーメッセージの取得"""
        # E1001: マイクアクセスエラー（仕様書準拠）
        msg = i18n_ja.get_error_message("E1001")
        assert msg == "マイクにアクセスできません。"

        # E1002: 音声入力デバイスエラー（仕様書準拠）
        msg = i18n_ja.get_error_message("E1002")
        assert msg == "音声入力デバイスが見つかりません。"

        # E0001: 設定ファイルエラー
        msg = i18n_ja.get_error_message("E0001")
        assert msg == "設定ファイルが見つかりません。"

    def test_get_error_message_english(self, i18n_en):
        """英語エラーメッセージの取得"""
        # E1001: マイクアクセスエラー（仕様書準拠）
        msg = i18n_en.get_error_message("E1001")
        assert msg == "Cannot access microphone."

        # E1002: 音声入力デバイスエラー（仕様書準拠）
        msg = i18n_en.get_error_message("E1002")
        assert msg == "Audio input device not found."

        # E0001: 設定ファイルエラー
        msg = i18n_en.get_error_message("E0001")
        assert msg == "Configuration file not found."

    def test_get_error_message_with_variables(self, i18n_ja):
        """変数埋め込みのエラーメッセージ"""
        # 変数を使用する場合（Phase 2では基本的な変数展開のみ）
        # E1001は変数を持たないので、character_idは無視される
        msg = i18n_ja.get_error_message("E1001", character_id="001_aoi")
        assert msg == "マイクにアクセスできません。"

    def test_get_error_message_unknown_code(self, i18n_ja):
        """未定義のエラーコード"""
        msg = i18n_ja.get_error_message("E9999")
        assert "E9999" in msg  # エラーコードが含まれる
        assert "不明なエラー" in msg

    def test_get_error_message_with_character_id(self, i18n_ja):
        """character_idパラメータ（Phase 2では無視される）"""
        msg1 = i18n_ja.get_error_message("E1001")
        msg2 = i18n_ja.get_error_message("E1001", character_id="001_aoi")
        assert msg1 == msg2  # Phase 2では同じメッセージ

    def test_get_error_message_fallback_to_english(self):
        """英語へのフォールバック"""
        i18n = I18nManager("unknown")  # 未知の言語
        # 日本語にフォールバック（デフォルト）
        msg = i18n.get_error_message("E1001")
        assert msg == "マイクにアクセスできません。"

    def test_get_error_message_format_error_handling(self, i18n_ja):
        """フォーマットエラーの処理"""
        # 変数が不正でもエラーにならない
        msg = i18n_ja.get_error_message("E1001", variables={"invalid": "value"})
        assert msg == "マイクにアクセスできません。"

    # ------------------------------------------------------------------------
    # get_ui_message()のテスト
    # ------------------------------------------------------------------------

    def test_get_ui_message_japanese(self, i18n_ja):
        """日本語UIメッセージの取得"""
        assert i18n_ja.get_ui_message("ui.menu.settings") == "設定"
        assert i18n_ja.get_ui_message("ui.buttons.start") == "開始"
        assert i18n_ja.get_ui_message("ui.status.ready") == "準備完了"

    def test_get_ui_message_english(self, i18n_en):
        """英語UIメッセージの取得"""
        assert i18n_en.get_ui_message("ui.menu.settings") == "Settings"
        assert i18n_en.get_ui_message("ui.buttons.start") == "Start"
        assert i18n_en.get_ui_message("ui.status.ready") == "Ready"

    def test_get_ui_message_nested_keys(self, i18n_ja):
        """ネストされたキーの処理"""
        assert i18n_ja.get_ui_message("ui.menu.help") == "ヘルプ"
        assert i18n_ja.get_ui_message("ui.buttons.stop") == "停止"

    def test_get_ui_message_unknown_key(self, i18n_ja):
        """未定義のUIキー"""
        msg = i18n_ja.get_ui_message("ui.nonexistent.key")
        assert msg == "ui.nonexistent.key"  # キーがそのまま返される

    def test_get_ui_message_with_variables(self, i18n_ja):
        """変数埋め込みのUIメッセージ"""
        # Phase 2では変数を持つUIメッセージはないが、機能はテスト
        msg = i18n_ja.get_ui_message("ui.status.ready", count=5)
        assert msg == "準備完了"  # 変数は無視される

    def test_get_ui_message_fallback(self):
        """UIメッセージのフォールバック"""
        i18n = I18nManager("unknown")
        msg = i18n.get_ui_message("ui.menu.settings")
        assert msg == "設定"  # 日本語にフォールバック

    # ------------------------------------------------------------------------
    # get_log_message()のテスト
    # ------------------------------------------------------------------------

    def test_get_log_message_always_english(self, i18n_ja):
        """ログメッセージは常に英語"""
        # 日本語設定でも英語のログメッセージ
        msg = i18n_ja.get_log_message("L0001")
        assert msg == "Component initialized successfully."

        msg = i18n_ja.get_log_message("L0002")
        assert msg == "Microphone access denied."  # 仕様書準拠

    def test_get_log_message_with_variables(self, i18n_en):
        """変数埋め込みのログメッセージ"""
        # Phase 2では変数を持つログメッセージはないが、機能はテスト
        msg = i18n_en.get_log_message("L0002", device_name="test_device")
        assert msg == "Microphone access denied."  # Phase 2では変数は無視

    def test_get_log_message_unknown_code(self, i18n_ja):
        """未定義のログコード"""
        msg = i18n_ja.get_log_message("L9999")
        assert "L9999" in msg

    # ------------------------------------------------------------------------
    # set_language()のテスト
    # ------------------------------------------------------------------------

    def test_set_language_to_english(self, i18n_ja):
        """日本語から英語への切り替え"""
        assert i18n_ja.set_language("en") is True
        assert i18n_ja.current_language == "en"

        # 切り替え後のメッセージ確認
        msg = i18n_ja.get_error_message("E1001")
        assert msg == "Cannot access microphone."

    def test_set_language_to_japanese(self, i18n_en):
        """英語から日本語への切り替え"""
        assert i18n_en.set_language("ja") is True
        assert i18n_en.current_language == "ja"

        # 切り替え後のメッセージ確認
        msg = i18n_en.get_error_message("E1001")
        assert msg == "マイクにアクセスできません。"

    def test_set_language_unsupported(self, i18n_ja):
        """未サポート言語への切り替え試行"""
        assert i18n_ja.set_language("fr") is False
        assert i18n_ja.current_language == "ja"  # 変更されない

    def test_set_language_same_language(self, i18n_ja):
        """同じ言語への切り替え"""
        assert i18n_ja.set_language("ja") is True
        assert i18n_ja.current_language == "ja"

    # ------------------------------------------------------------------------
    # その他のメソッドのテスト
    # ------------------------------------------------------------------------

    def test_load_character_messages_stub(self, i18n_ja):
        """キャラクターメッセージ読み込み（Phase 2ではスタブ）"""
        # Phase 2では何もしない
        i18n_ja.load_character_messages("001_aoi")
        # エラーが発生しないことを確認
        assert True

    def test_language_switching_workflow(self):
        """言語切り替えワークフロー"""
        i18n = I18nManager("ja")

        # 日本語メッセージ
        assert i18n.get_error_message("E1001") == "マイクにアクセスできません。"

        # 英語に切り替え
        i18n.set_language("en")
        assert i18n.get_error_message("E1001") == "Cannot access microphone."

        # 日本語に戻す
        i18n.set_language("ja")
        assert i18n.get_error_message("E1001") == "マイクにアクセスできません。"

    def test_error_recovery(self):
        """エラー処理の回復性"""
        i18n = I18nManager("ja")

        # 不正な値を渡してもクラッシュしない
        i18n.set_language(None)  # Noneを渡す
        assert i18n.current_language == "ja"  # 変更されない

        # 不正なエラーコード
        msg = i18n.get_error_message(None)
        assert isinstance(msg, str)

        # 不正なUIキー
        msg = i18n.get_ui_message(None)
        assert isinstance(msg, str)

    @patch("vioratalk.core.i18n_manager.logger")
    def test_logging_on_errors(self, mock_logger):
        """エラー時のログ出力"""
        i18n = I18nManager("ja")

        # 未定義のエラーコード
        i18n.get_error_message("E9999")
        # 警告ログが出力される
        assert mock_logger.debug.called

        # 未サポート言語
        i18n.set_language("fr")
        # 警告ログが出力される
        mock_logger.warning.assert_called()


# Phase 1最小実装の確認テストは削除（現在Phase 2のため不要）
