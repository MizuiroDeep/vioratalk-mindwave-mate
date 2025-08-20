"""
国際化管理マネージャー

多言語対応のメッセージ管理を行う。
Phase 1では最小実装として、ハードコードされたメッセージを返す。

関連ドキュメント:
    - I18nManager実装ガイド v1.0
    - エラーハンドリング指針 v1.20
    - インターフェース定義書 v1.33
    - 開発規約書 v1.12

Author: MizuiroDeep
Copyright (c) 2025 MizuiroDeep
"""

import logging
from typing import Any, Dict, List, Optional

from vioratalk.core.base import ComponentState, VioraTalkComponent

# ロガー取得（Phase 1では標準のloggingモジュールを使用）
logger = logging.getLogger(__name__)


class I18nManager(VioraTalkComponent):
    """
    国際化管理マネージャー

    Phase 1では最小実装として、ハードコードされたメッセージを提供。
    将来的にはJSONファイルからの読み込みに対応予定。

    Attributes:
        current_language: 現在の言語設定（ja/en）
        _messages: ハードコードされたメッセージ辞書
    """

    def __init__(self, language: str = "ja"):
        """
        I18nManagerの初期化

        Args:
            language: 初期言語設定（ja/en）、デフォルトは日本語
        """
        super().__init__()

        # サポート言語の定義（Phase 1では日本語と英語のみ）
        self._supported_languages = ["ja", "en"]

        # 言語設定（サポートされていない場合は日本語にフォールバック）
        if language in self._supported_languages:
            self.current_language = language
        else:
            self.current_language = "ja"
            if language:  # languageが指定されていて、かつサポート外の場合のみ警告
                logger.warning(f"Unsupported language: {language}, falling back to Japanese")

        # Phase 1: ハードコードされたメッセージ
        self._messages = self._create_hardcoded_messages()

        self._state = ComponentState.READY
        logger.info(f"I18nManager initialized with language: {self.current_language}")

    def _create_hardcoded_messages(self) -> Dict[str, Dict[str, Any]]:
        """
        Phase 1用のハードコードされたメッセージを作成

        Returns:
            言語別のメッセージ辞書
        """
        return {
            "ja": {
                "errors": {
                    "E0001": "設定ファイルが見つかりません。",
                    "E0002": "設定ファイルの形式が無効です。",
                    "E0100": "コンポーネントの初期化に失敗しました。",
                    "E0101": "依存コンポーネントが見つかりません。",
                    "E1001": "マイクにアクセスできません。",
                    "E1002": "音声入力デバイスが見つかりません。",
                    "E1101": "ホットワードが設定されていません。",
                    "E2001": "APIへの接続に失敗しました。",
                    "E3001": "TTSエンジンの初期化に失敗しました。",
                    "E3301": "録音に失敗しました。",
                    "E4001": "対話マネージャーの初期化に失敗しました。",
                    "E4201": "キャラクター設定の読み込みに失敗しました。キャラクター名: {character_name}",
                    "E5001": "ストレージアクセスエラー。",
                    "E5302": "ディスク容量が不足しています。必要: {required}GB、利用可能: {available}GB",
                    "unknown": "不明なエラーが発生しました。",
                },
                "ui": {
                    "menu": {
                        "settings": "設定",
                        "file": "ファイル",
                        "help": "ヘルプ",
                        "character": "キャラクター",
                    },
                    "button": {"ok": "OK", "cancel": "キャンセル", "apply": "適用"},
                    "buttons": {"start": "開始", "stop": "停止", "cancel": "キャンセル"},
                    "status": {
                        "ready": "準備完了",
                        "listening": "聞き取り中...",
                        "processing": "処理中...",
                        "error": "エラー",
                    },
                },
                "log": {},  # ログは常に英語
            },
            "en": {
                "errors": {
                    "E0001": "Configuration file not found.",
                    "E0002": "Invalid configuration file format.",
                    "E0100": "Component initialization failed.",
                    "E0101": "Dependency component not found.",
                    "E1001": "Microphone access denied. Device: {device_name}",
                    "E1002": "Audio input device not found.",
                    "E1101": "Hotword not configured.",
                    "E2001": "API connection failed.",
                    "E3001": "TTS engine initialization failed.",
                    "E3301": "Recording failed.",
                    "E4001": "Dialogue manager initialization failed.",
                    "E4201": "Failed to load character settings.",
                    "E5001": "Storage access error.",
                    "E5302": "Insufficient disk space. Required: {required}GB, Available: {available}GB",
                    "unknown": "An unknown error occurred.",
                },
                "ui": {
                    "menu": {
                        "settings": "Settings",
                        "file": "File",
                        "help": "Help",
                        "character": "Character",
                    },
                    "button": {"ok": "OK", "cancel": "Cancel", "apply": "Apply"},
                    "buttons": {"start": "Start", "stop": "Stop", "cancel": "Cancel"},
                    "status": {
                        "ready": "Ready",
                        "listening": "Listening...",
                        "processing": "Processing...",
                        "error": "Error",
                    },
                },
                "log": {},  # ログは常に英語
            },
        }

    async def initialize(self) -> None:
        """初期化処理（Phase 1では特に何もしない）"""
        self._state = ComponentState.INITIALIZING
        # Phase 1では設定ファイルの読み込みはスキップ
        self._state = ComponentState.READY
        logger.info("I18nManager initialization complete")

    @property
    def supported_languages(self) -> List[str]:
        """サポートされている言語のリストを返す"""
        return self._supported_languages.copy()

    def set_language(self, language: str) -> bool:
        """
        言語を変更

        Args:
            language: 変更先の言語コード

        Returns:
            変更に成功した場合True
        """
        if language not in self._supported_languages:
            logger.warning(f"Unsupported language: {language}")
            return False

        if language == self.current_language:
            logger.debug(f"Language already set to: {language}")
            return True

        self.current_language = language
        logger.info("Language changed")
        return True

    def get_error_message(
        self,
        error_code: str,
        variables: Optional[Dict[str, Any]] = None,
        character_id: Optional[str] = None,
        **kwargs,  # 追加のキーワード引数をサポート
    ) -> str:
        """
        エラーメッセージを取得

        Args:
            error_code: エラーコード（例: "E0001"）
            variables: メッセージに埋め込む変数（辞書形式）
            character_id: キャラクターID（Phase 1では未使用）
            **kwargs: 追加の変数（variables引数の代替）

        Returns:
            フォーマット済みのエラーメッセージ
        """
        # Noneチェック
        if error_code is None:
            return "Error: None"

        # Phase 1: ハードコードされたメッセージを返す
        messages = self._messages.get(self.current_language, self._messages["ja"])
        error_messages = messages.get("errors", {})

        # エラーコードからメッセージを取得
        if error_code in error_messages:
            message = error_messages[error_code]
        else:
            # 不明なエラーコードの場合
            logger.debug(f"Unknown error code: {error_code}")
            message = error_messages.get("unknown", f"Error: {error_code}")
            # エラーコードを末尾に追加
            message = f"{message}({error_code})"

        # 変数の埋め込み（Phase 1では簡易実装）
        # variablesとkwargsの両方をサポート
        format_vars = {}
        if variables:
            format_vars.update(variables)
        if kwargs:
            format_vars.update(kwargs)

        if format_vars:
            try:
                message = message.format(**format_vars)
            except Exception as e:
                logger.debug(f"Error formatting message: {e}")

        return message

    def get_ui_message(self, key: str, variables: Optional[Dict[str, Any]] = None, **kwargs) -> str:
        """
        UIメッセージを取得

        Args:
            key: メッセージキー（例: "ui.menu.settings"）
            variables: メッセージに埋め込む変数
            **kwargs: 追加の変数

        Returns:
            フォーマット済みのUIメッセージ
        """
        # Noneチェック
        if key is None:
            return "None"

        # Phase 1: ハードコードされたメッセージを返す
        messages = self._messages.get(self.current_language, self._messages["ja"])

        # ドット記法でネストされたキーをたどる
        keys = key.split(".")
        result = messages

        try:
            for k in keys:
                if isinstance(result, dict) and k in result:
                    result = result[k]
                else:
                    # キーが見つからない場合はキーをそのまま返す（[]なし）
                    return key

            # resultが文字列でない場合
            if not isinstance(result, str):
                return key

            # 変数の埋め込み
            format_vars = {}
            if variables:
                format_vars.update(variables)
            if kwargs:
                format_vars.update(kwargs)

            if format_vars:
                result = result.format(**format_vars)

            return result

        except Exception as e:
            logger.debug(f"Error getting UI message for key '{key}': {e}")
            return key

    def get_log_message(
        self, code: str, variables: Optional[Dict[str, Any]] = None, **kwargs
    ) -> str:
        """
        ログメッセージを取得（常に英語）

        Args:
            code: ログメッセージコード
            variables: メッセージに埋め込む変数
            **kwargs: 追加の変数

        Returns:
            フォーマット済みのログメッセージ（英語）
        """
        # Phase 1: ログメッセージは英語のエラーメッセージを流用
        en_messages = self._messages.get("en", {})

        # まずログ専用メッセージを探す（Phase 1では空）
        log_messages = en_messages.get("log", {})
        if code in log_messages:
            message = log_messages[code]
        # エラーメッセージから探す
        elif code in en_messages.get("errors", {}):
            message = en_messages["errors"][code]
        else:
            # デフォルトメッセージ（Unknown errorを含める）
            message = f"Unknown error: {code}"

        # 変数の埋め込み
        format_vars = {}
        if variables:
            format_vars.update(variables)
        if kwargs:
            format_vars.update(kwargs)

        if format_vars:
            try:
                message = message.format(**format_vars)
            except Exception:
                pass

        return message

    def load_character_messages(self, character_id: str) -> None:
        """
        キャラクター固有のメッセージを読み込む

        Phase 1ではスタブ実装。Phase 8で本実装予定。

        Args:
            character_id: キャラクターID
        """
        # Phase 1: スタブ実装
        logger.debug(f"Character messages loading skipped for: {character_id} (Phase 1)")
        pass

    async def cleanup(self) -> None:
        """クリーンアップ処理"""
        self._state = ComponentState.SHUTDOWN
        logger.info("I18nManager cleanup complete")


# Phase 1での利用例
if __name__ == "__main__":
    import asyncio

    async def test_i18n():
        """I18nManagerのテスト"""
        # 日本語で初期化
        i18n = I18nManager("ja")
        await i18n.initialize()

        # エラーメッセージ取得
        print("=== Japanese Messages ===")
        print(f"E0001: {i18n.get_error_message('E0001')}")
        print(f"E1002: {i18n.get_error_message('E1002')}")

        # UIメッセージ取得
        print(f"Settings: {i18n.get_ui_message('ui.menu.settings')}")
        print(f"Ready: {i18n.get_ui_message('ui.status.ready')}")

        # 英語に切り替え
        i18n.set_language("en")
        print("\n=== English Messages ===")
        print(f"E0001: {i18n.get_error_message('E0001')}")
        print(f"E1002: {i18n.get_error_message('E1002')}")
        print(f"Settings: {i18n.get_ui_message('ui.menu.settings')}")

        # ログメッセージ（常に英語）
        print(f"\nLog E1002: {i18n.get_log_message('E1002')}")

        await i18n.cleanup()

    asyncio.run(test_i18n())
