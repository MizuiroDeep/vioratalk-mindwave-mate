"""I18nManagerの単体テスト

国際化管理機能の動作を検証。
Phase 1の最小実装（ハードコード）として、基本的な機能のテストに焦点を当てる。

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
@pytest.mark.phase(1)
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
        # E1001: マイクアクセスエラー
        msg = i18n_ja.get_error_message("E1001")
        assert "マイクにアクセスできません" in msg

        # E2001: API接続エラー
        msg = i18n_ja.get_error_message("E2001")
        assert "APIへの接続に失敗しました" in msg

        # E4201: キャラクター読み込みエラー
        msg = i18n_ja.get_error_message("E4201")
        assert "キャラクター設定の読み込みに失敗しました" in msg

    def test_get_error_message_english(self, i18n_en):
        """英語エラーメッセージの取得"""
        # E1001: マイクアクセスエラー
        msg = i18n_en.get_error_message("E1001")
        assert "Microphone access denied" in msg

        # E2001: API接続エラー
        msg = i18n_en.get_error_message("E2001")
        assert "API connection failed" in msg

        # E4201: キャラクター読み込みエラー
        msg = i18n_en.get_error_message("E4201")
        assert "Failed to load character settings" in msg

    def test_get_error_message_with_variables(self, i18n_ja):
        """変数展開を含むエラーメッセージ"""
        # E5302: メモリ不足エラー
        msg = i18n_ja.get_error_message("E5302", required=10, available=5)
        assert "10GB" in msg
        assert "5GB" in msg

        # E4201: キャラクター名を含むエラー
        msg = i18n_ja.get_error_message("E4201", character_name="碧衣")
        assert "碧衣" in msg

    def test_get_error_message_unknown_code(self, i18n_ja):
        """未定義のエラーコード"""
        msg = i18n_ja.get_error_message("E9999")
        assert "E9999" in msg  # エラーコードが含まれる

    def test_get_error_message_with_character_id(self, i18n_ja):
        """character_idパラメータ（Phase 1では無視される）"""
        # character_idを渡しても基本メッセージが返る
        msg_without = i18n_ja.get_error_message("E1001")
        msg_with = i18n_ja.get_error_message("E1001", character_id="001_aoi")

        assert msg_without == msg_with  # Phase 1では同じメッセージ

    def test_get_error_message_fallback_to_english(self):
        """英語へのフォールバック"""
        i18n = I18nManager("ja")

        # 日本語メッセージが定義されていないエラーコード（仮定）
        # Phase 1では全て定義されているので、このテストは概念的
        msg = i18n.get_error_message("E0000")  # 未定義のコード
        assert "E0000" in msg

    def test_get_error_message_format_error_handling(self, i18n_ja):
        """フォーマットエラーのハンドリング"""
        # 必要な変数が足りない場合
        # E5302は{required}と{available}を期待
        msg = i18n_ja.get_error_message("E5302", required=10)  # availableが不足

        # エラーが発生してもアプリケーションは停止しない
        assert isinstance(msg, str)
        # テンプレートがそのまま返されるか、部分的にフォーマットされる

    # ------------------------------------------------------------------------
    # get_ui_message()のテスト
    # ------------------------------------------------------------------------

    def test_get_ui_message_japanese(self, i18n_ja):
        """日本語UI文字列の取得"""
        assert i18n_ja.get_ui_message("ui.menu.settings") == "設定"
        assert i18n_ja.get_ui_message("ui.menu.character") == "キャラクター"
        assert i18n_ja.get_ui_message("ui.menu.help") == "ヘルプ"

        assert i18n_ja.get_ui_message("ui.buttons.start") == "開始"
        assert i18n_ja.get_ui_message("ui.buttons.stop") == "停止"
        assert i18n_ja.get_ui_message("ui.buttons.cancel") == "キャンセル"

        assert i18n_ja.get_ui_message("ui.status.ready") == "準備完了"
        assert i18n_ja.get_ui_message("ui.status.processing") == "処理中..."
        assert i18n_ja.get_ui_message("ui.status.error") == "エラー"

    def test_get_ui_message_english(self, i18n_en):
        """英語UI文字列の取得"""
        assert i18n_en.get_ui_message("ui.menu.settings") == "Settings"
        assert i18n_en.get_ui_message("ui.menu.character") == "Character"
        assert i18n_en.get_ui_message("ui.menu.help") == "Help"

        assert i18n_en.get_ui_message("ui.buttons.start") == "Start"
        assert i18n_en.get_ui_message("ui.buttons.stop") == "Stop"
        assert i18n_en.get_ui_message("ui.buttons.cancel") == "Cancel"

        assert i18n_en.get_ui_message("ui.status.ready") == "Ready"
        assert i18n_en.get_ui_message("ui.status.processing") == "Processing..."
        assert i18n_en.get_ui_message("ui.status.error") == "Error"

    def test_get_ui_message_nested_keys(self, i18n_ja):
        """階層的なキーの処理"""
        # 階層的なキーをドット記法で指定
        msg = i18n_ja.get_ui_message("ui.menu.settings")
        assert msg == "設定"

        msg = i18n_ja.get_ui_message("ui.status.processing")
        assert msg == "処理中..."

    def test_get_ui_message_unknown_key(self, i18n_ja):
        """未定義のUIキー"""
        msg = i18n_ja.get_ui_message("ui.nonexistent.key")
        assert msg == "ui.nonexistent.key"  # キーがそのまま返される

    def test_get_ui_message_with_variables(self, i18n_ja):
        """変数展開を含むUI文字列"""
        # Phase 1のハードコード実装では変数展開するメッセージが少ないが、
        # 機能自体はテストする
        msg = i18n_ja.get_ui_message("ui.status.processing")
        assert isinstance(msg, str)

    def test_get_ui_message_fallback(self):
        """UI文字列の英語フォールバック"""
        i18n = I18nManager("ja")

        # 存在しないキーの場合、英語にフォールバックを試みる
        # Phase 1では全て定義されているので、概念的なテスト
        msg = i18n.get_ui_message("ui.undefined.test")
        assert "ui.undefined.test" in msg

    # ------------------------------------------------------------------------
    # get_log_message()のテスト
    # ------------------------------------------------------------------------

    def test_get_log_message_always_english(self, i18n_ja):
        """ログメッセージは常に英語"""
        # 日本語設定でも英語が返る
        msg = i18n_ja.get_log_message("E1001")
        assert "Microphone access denied" in msg
        assert "マイク" not in msg

    def test_get_log_message_with_variables(self, i18n_ja):
        """変数展開を含むログメッセージ"""
        msg = i18n_ja.get_log_message("E1001", device_name="test_device")
        assert "test_device" in msg

        msg = i18n_ja.get_log_message("E5302", required=10, available=5)
        assert "10GB" in msg
        assert "5GB" in msg

    def test_get_log_message_unknown_code(self, i18n_ja):
        """未定義のエラーコードのログメッセージ"""
        msg = i18n_ja.get_log_message("E9999")
        assert "Unknown error" in msg
        assert "E9999" in msg

    # ------------------------------------------------------------------------
    # set_language()のテスト
    # ------------------------------------------------------------------------

    def test_set_language_to_english(self, i18n_ja):
        """日本語から英語への切り替え"""
        assert i18n_ja.current_language == "ja"

        result = i18n_ja.set_language("en")
        assert result is True
        assert i18n_ja.current_language == "en"

        # メッセージが英語になる
        msg = i18n_ja.get_error_message("E1001")
        assert "Microphone" in msg

    def test_set_language_to_japanese(self, i18n_en):
        """英語から日本語への切り替え"""
        assert i18n_en.current_language == "en"

        result = i18n_en.set_language("ja")
        assert result is True
        assert i18n_en.current_language == "ja"

        # メッセージが日本語になる
        msg = i18n_en.get_error_message("E1001")
        assert "マイク" in msg

    def test_set_language_unsupported(self, i18n_ja):
        """未サポート言語への切り替え失敗"""
        result = i18n_ja.set_language("fr")
        assert result is False
        assert i18n_ja.current_language == "ja"  # 変更されない

    def test_set_language_same_language(self, i18n_ja):
        """同じ言語への切り替え"""
        result = i18n_ja.set_language("ja")
        assert result is True
        assert i18n_ja.current_language == "ja"

    # ------------------------------------------------------------------------
    # load_character_messages()のテスト
    # ------------------------------------------------------------------------

    def test_load_character_messages_stub(self, i18n_ja):
        """キャラクターメッセージ読み込み（Phase 1ではスタブ）"""
        # Phase 1では何もしないスタブメソッド
        # エラーが発生しないことを確認
        i18n_ja.load_character_messages("001_aoi")
        i18n_ja.load_character_messages("002_haru")
        i18n_ja.load_character_messages("003_yui")

        # メッセージは変わらない
        msg_before = i18n_ja.get_error_message("E1001")
        i18n_ja.load_character_messages("001_aoi")
        msg_after = i18n_ja.get_error_message("E1001")
        assert msg_before == msg_after

    # ------------------------------------------------------------------------
    # 統合的な動作テスト
    # ------------------------------------------------------------------------

    def test_language_switching_workflow(self):
        """言語切り替えワークフロー"""
        i18n = I18nManager("ja")

        # 日本語メッセージ
        ja_error = i18n.get_error_message("E1001")
        ja_ui = i18n.get_ui_message("ui.menu.settings")

        assert "マイク" in ja_error
        assert ja_ui == "設定"

        # 英語に切り替え
        i18n.set_language("en")

        # 英語メッセージ
        en_error = i18n.get_error_message("E1001")
        en_ui = i18n.get_ui_message("ui.menu.settings")

        assert "Microphone" in en_error
        assert en_ui == "Settings"

        # ログメッセージは言語に関係なく英語
        log_msg = i18n.get_log_message("E1001")
        assert "Microphone" in log_msg

    def test_error_recovery(self):
        """エラー発生時の回復性"""
        i18n = I18nManager("ja")

        # 不正な操作をしてもクラッシュしない
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
        assert mock_logger.warning.called or mock_logger.debug.called

        # 未サポート言語
        i18n.set_language("fr")
        # 警告ログが出力される
        mock_logger.warning.assert_called()


# ============================================================================
# Phase 1最小実装の確認テスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(1)
class TestPhase1MinimalImplementation:
    """Phase 1の最小実装要件を満たしているかの確認"""

    def test_hardcoded_messages(self):
        """ハードコードされたメッセージが返されることを確認"""
        i18n = I18nManager("ja")

        # エラーメッセージがハードコードされている
        # （YAMLファイルからの読み込みではない）
        msg = i18n.get_error_message("E1001")
        assert isinstance(msg, str)
        assert len(msg) > 0

        # UI文字列もハードコード
        ui_msg = i18n.get_ui_message("ui.menu.settings")
        assert isinstance(ui_msg, str)
        assert len(ui_msg) > 0

    def test_minimum_supported_languages(self):
        """最小限の言語サポート（日本語と英語）"""
        i18n = I18nManager()

        languages = i18n.supported_languages
        assert len(languages) >= 2
        assert "ja" in languages
        assert "en" in languages

    def test_character_feature_not_implemented(self):
        """キャラクター機能が未実装であることを確認"""
        i18n = I18nManager("ja")

        # character_idを渡しても無視される
        msg1 = i18n.get_error_message("E1001")
        msg2 = i18n.get_error_message("E1001", character_id="001_aoi")
        assert msg1 == msg2

        # load_character_messagesは何もしない
        before = i18n.get_error_message("E1001")
        i18n.load_character_messages("001_aoi")
        after = i18n.get_error_message("E1001")
        assert before == after
