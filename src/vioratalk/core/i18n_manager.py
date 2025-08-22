"""
国際化管理マネージャー

多言語対応のメッセージ管理を行う。
Phase 2実装として、ハードコードされたメッセージを返す。

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

# ロガー取得（Phase 2では標準のloggingモジュールを使用）
logger = logging.getLogger(__name__)


class I18nManager(VioraTalkComponent):
    """
    国際化管理マネージャー

    Phase 2実装として、ハードコードされたメッセージを提供。
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

        # サポート言語の定義（Phase 2では日本語と英語のみ）
        self._supported_languages = ["ja", "en"]

        # 言語設定（サポートされていない場合は日本語にフォールバック）
        if language in self._supported_languages:
            self.current_language = language
        else:
            self.current_language = "ja"
            if language:  # languageが指定されていて、かつサポート外の場合のみ警告
                logger.warning(f"Unsupported language: {language}, falling back to Japanese")

        # Phase 2: ハードコードされたメッセージ
        self._messages = self._create_hardcoded_messages()

        self._state = ComponentState.READY
        logger.info(f"I18nManager initialized with language: {self.current_language}")

    @property
    def supported_languages(self) -> List[str]:
        """
        サポートされている言語のリストを取得

        Returns:
            サポート言語のリスト（コピー）
        """
        return self._supported_languages.copy()

    def _create_hardcoded_messages(self) -> Dict[str, Dict[str, Any]]:
        """
        Phase 2用のハードコードされたメッセージを作成

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
                    "E1001": "マイクにアクセスできません。",  # 仕様書準拠
                    "E1002": "音声入力デバイスが見つかりません。",  # 仕様書準拠
                    "E1101": "ホットワードが設定されていません。",
                    "E2001": "APIへの接続に失敗しました。",
                    "E3001": "TTSエンジンの初期化に失敗しました。",
                    "E4100": "キャラクター設定の読み込みに失敗しました。",
                    "E4101": "このキャラクターはPro版限定です。",
                    "E5001": "メモリが不足しています。",
                    "E7001": "サービスの起動に失敗しました。",
                    "E8001": "セットアップディレクトリの作成に失敗しました。",
                    "E9400": "プロンプト生成に失敗しました。",
                    "unknown": "不明なエラー",
                },
                "logs": {
                    "L0001": "Component initialized successfully.",
                    "L0002": "Microphone access denied.",  # 仕様書準拠（英語ログ）
                    "L0003": "API connection established.",
                    "L0004": "Service started.",
                    "L0005": "Prompt generated.",
                },
                "ui": {
                    "menu": {
                        "settings": "設定",
                        "help": "ヘルプ",
                        "about": "VioraTalkについて",
                        "exit": "終了",
                    },
                    "buttons": {"start": "開始", "stop": "停止", "record": "録音", "play": "再生"},
                    "status": {
                        "ready": "準備完了",
                        "recording": "録音中",
                        "processing": "処理中",
                        "error": "エラー",
                    },
                },
            },
            "en": {
                "errors": {
                    "E0001": "Configuration file not found.",
                    "E0002": "Invalid configuration file format.",
                    "E0100": "Component initialization failed.",
                    "E0101": "Dependency component not found.",
                    "E1001": "Cannot access microphone.",  # 仕様書準拠
                    "E1002": "Audio input device not found.",  # 仕様書準拠
                    "E1101": "Hotword not configured.",
                    "E2001": "Failed to connect to API.",
                    "E3001": "TTS engine initialization failed.",
                    "E4100": "Failed to load character settings.",
                    "E4101": "This character is Pro edition exclusive.",
                    "E5001": "Out of memory.",
                    "E7001": "Failed to start service.",
                    "E8001": "Failed to create setup directory.",
                    "E9400": "Failed to generate prompt.",
                    "unknown": "Unknown error",
                },
                "logs": {
                    "L0001": "Component initialized successfully.",
                    "L0002": "Microphone access denied.",  # 仕様書準拠
                    "L0003": "API connection established.",
                    "L0004": "Service started.",
                    "L0005": "Prompt generated.",
                },
                "ui": {
                    "menu": {
                        "settings": "Settings",
                        "help": "Help",
                        "about": "About VioraTalk",
                        "exit": "Exit",
                    },
                    "buttons": {
                        "start": "Start",
                        "stop": "Stop",
                        "record": "Record",
                        "play": "Play",
                    },
                    "status": {
                        "ready": "Ready",
                        "recording": "Recording",
                        "processing": "Processing",
                        "error": "Error",
                    },
                },
            },
        }

    async def initialize(self) -> None:
        """
        非同期初期化処理

        Phase 2では特に何もしない（将来的に設定ファイル読み込みなど）
        """
        logger.debug("I18nManager async initialization complete")

    async def cleanup(self) -> None:
        """
        非同期クリーンアップ処理

        Phase 2では特に何もしない
        """
        self._state = ComponentState.TERMINATED
        logger.info("I18nManager cleanup complete")

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
            character_id: キャラクターID（Phase 2では未使用）
            **kwargs: 追加の変数（variables引数の代替）

        Returns:
            フォーマット済みのエラーメッセージ
        """
        # Noneチェック
        if error_code is None:
            return "Error: None"

        # Phase 2: ハードコードされたメッセージを返す
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

        # 変数の埋め込み（Phase 2では簡易実装）
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

        # Phase 2: ハードコードされたメッセージを返す
        messages = self._messages.get(self.current_language, self._messages["ja"])

        # ドット記法でネストされたキーをたどる
        keys = key.split(".")
        result = messages

        for k in keys:
            if isinstance(result, dict) and k in result:
                result = result[k]
            else:
                # キーが見つからない場合
                logger.debug(f"UI message key not found: {key}")
                return key

        # 文字列でない場合（辞書など）はキーを返す
        if not isinstance(result, str):
            return key

        # 変数の埋め込み
        format_vars = {}
        if variables:
            format_vars.update(variables)
        if kwargs:
            format_vars.update(kwargs)

        if format_vars:
            try:
                result = result.format(**format_vars)
            except Exception as e:
                logger.debug(f"Error formatting UI message: {e}")

        return result

    def get_log_message(
        self, log_code: str, variables: Optional[Dict[str, Any]] = None, **kwargs
    ) -> str:
        """
        ログメッセージを取得（常に英語）

        Args:
            log_code: ログコード（例: "L0001"）
            variables: メッセージに埋め込む変数
            **kwargs: 追加の変数

        Returns:
            フォーマット済みのログメッセージ（英語）
        """
        # Noneチェック
        if log_code is None:
            return "Log: None"

        # ログメッセージは常に英語
        messages = self._messages.get("en", self._messages["ja"])
        log_messages = messages.get("logs", {})

        if log_code in log_messages:
            message = log_messages[log_code]
        else:
            logger.debug(f"Unknown log code: {log_code}")
            message = f"Log: {log_code}"

        # 変数の埋め込み
        format_vars = {}
        if variables:
            format_vars.update(variables)
        if kwargs:
            format_vars.update(kwargs)

        if format_vars:
            try:
                message = message.format(**format_vars)
            except Exception as e:
                logger.debug(f"Error formatting log message: {e}")

        return message

    def get_message(
        self,
        key: str,
        message_type: str = "ui",
        variables: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """
        汎用メッセージ取得メソッド

        Args:
            key: メッセージキー
            message_type: メッセージタイプ（ui/error/log）
            variables: メッセージに埋め込む変数
            **kwargs: 追加の変数

        Returns:
            フォーマット済みのメッセージ
        """
        if message_type == "error":
            return self.get_error_message(key, variables, **kwargs)
        elif message_type == "log":
            return self.get_log_message(key, variables, **kwargs)
        else:
            return self.get_ui_message(key, variables, **kwargs)

    def load_character_messages(self, character_id: str) -> None:
        """
        キャラクター固有のメッセージを読み込む

        Phase 2では何もしない（Phase 7-9で実装予定）

        Args:
            character_id: キャラクターID
        """
        logger.debug(f"load_character_messages called with {character_id} (stub)")
        # Phase 7-9で実装予定
