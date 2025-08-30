"""
国際化管理マネージャー

多言語対応のメッセージ管理を行う。
Phase 3-4実装として、YAMLファイルからメッセージを読み込み。
YAMLが読めない場合はハードコードにフォールバック。

関連ドキュメント:
    - I18nManager実装ガイド v1.0
    - エラーハンドリング指針 v1.20
    - インターフェース定義書 v1.33
    - 開発規約書 v1.12

Author: MizuiroDeep
Copyright (c) 2025 MizuiroDeep
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from vioratalk.core.base import ComponentState, VioraTalkComponent

# ロガー取得
logger = logging.getLogger(__name__)


class I18nManager(VioraTalkComponent):
    """
    国際化管理マネージャー

    Phase 3-4実装として、YAMLファイルからメッセージを読み込み。
    YAMLが利用できない場合はハードコードされたメッセージにフォールバック。

    Attributes:
        current_language: 現在の言語設定（ja/en）
        messages_dir: メッセージファイルのディレクトリ
        _messages: メッセージ辞書（YAML優先、ハードコードはフォールバック）
        _character_messages: キャラクター別メッセージ（Phase 7-9用の枠組み）
    """

    # サポート言語（ISO 639-1準拠）
    SUPPORTED_LANGUAGES = ["ja", "en"]

    def __init__(self, language: str = "ja", messages_dir: Optional[Path] = None):
        """
        I18nManagerの初期化

        Args:
            language: 初期言語設定（ja/en）、デフォルトは日本語
            messages_dir: メッセージファイルディレクトリ（省略時はmessages/）
        """
        super().__init__()

        # 言語設定（サポートされていない場合は日本語にフォールバック）
        if language in self.SUPPORTED_LANGUAGES:
            self.current_language = language
        else:
            self.current_language = "ja"
            if language:
                logger.warning(f"Unsupported language: {language}, falling back to Japanese")

        # メッセージディレクトリの設定
        if messages_dir is None:
            # プロジェクトルートからの相対パスを想定
            self.messages_dir = Path("messages")
        else:
            self.messages_dir = Path(messages_dir)

        # メッセージキャッシュの初期化
        self._messages: Dict[str, Dict[str, Any]] = {}
        self._character_messages: Dict[str, Dict[str, Any]] = {}  # Phase 7-9用の枠組み

        # Phase 2: ハードコードされたメッセージ（フォールバック用）
        self._messages = self._create_hardcoded_messages()

        # Phase 3-4: YAMLファイルから読み込み（上書き）
        self._load_messages_from_yaml()

        self._state = ComponentState.READY
        logger.info(f"I18nManager initialized with language: {self.current_language}")

    @property
    def supported_languages(self) -> List[str]:
        """
        サポートされている言語のリストを取得

        Returns:
            サポート言語のリスト（コピー）
        """
        return self.SUPPORTED_LANGUAGES.copy()

    def _create_hardcoded_messages(self) -> Dict[str, Dict[str, Any]]:
        """
        Phase 2用のハードコードされたメッセージを作成
        YAMLが読めない場合のフォールバックとして使用

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
                    "L0002": "Microphone access denied.",
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
                    "E1001": "Cannot access microphone.",
                    "E1002": "Audio input device not found.",
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
                    "L0002": "Microphone access denied.",
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

    def _load_messages_from_yaml(self) -> None:
        """
        YAMLファイルからメッセージを読み込み（Phase 3-4実装）

        エラーが発生してもアプリケーションの起動は継続する。
        読み込みに失敗した場合はハードコードされたメッセージを使用。
        """
        for lang in self.SUPPORTED_LANGUAGES:
            # エラーメッセージの読み込み
            self._load_yaml_file(lang, "errors.yaml", "errors")

            # UI文字列の読み込み
            self._load_yaml_file(lang, "ui.yaml", "ui")

            # キャラクターメッセージの枠組み準備（Phase 7-9用）
            self._prepare_character_messages_framework(lang)

    def _load_yaml_file(self, language: str, filename: str, message_type: str) -> None:
        """
        個別のYAMLファイルを読み込み

        Args:
            language: 言語コード（ja/en）
            filename: ファイル名（errors.yaml、ui.yaml等）
            message_type: メッセージタイプ（errors、ui等）
        """
        file_path = self.messages_dir / language / filename

        try:
            if not file_path.exists():
                logger.debug(f"Message file not found (using hardcoded): {file_path}")
                return

            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

                if data is None:
                    logger.warning(f"Empty YAML file: {file_path}")
                    return

                # メッセージをマージ（YAMLが優先）
                if language not in self._messages:
                    self._messages[language] = {}

                if message_type == "ui":
                    # ui.yamlの内容を直接マージ
                    self._merge_ui_messages(self._messages[language], data)
                else:
                    # errors.yamlなどはそのままマージ
                    if message_type not in self._messages[language]:
                        self._messages[language][message_type] = {}
                    self._messages[language][message_type].update(data)

                logger.info(f"Loaded {message_type} messages for {language} from {filename}")

        except yaml.YAMLError as e:
            logger.warning(
                f"Failed to parse YAML file {file_path}: {e}. " "Using hardcoded messages."
            )
        except UnicodeDecodeError as e:
            logger.warning(f"Encoding error in {file_path}: {e}. " "Using hardcoded messages.")
        except Exception as e:
            logger.warning(
                f"Unexpected error loading {file_path}: {e}. " "Using hardcoded messages."
            )

    def _merge_ui_messages(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        UI文字列を階層的にマージ

        ui.yamlは階層構造を持つため、特別な処理が必要。

        Args:
            target: マージ先の辞書
            source: マージ元の辞書（YAMLから読み込んだデータ）
        """
        if "ui" not in target:
            target["ui"] = {}

        # 再帰的にマージ
        self._deep_merge(target["ui"], source)

    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        辞書を再帰的にマージ

        Args:
            target: マージ先の辞書
            source: マージ元の辞書
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                # 両方が辞書の場合は再帰的にマージ
                self._deep_merge(target[key], value)
            else:
                # それ以外は上書き
                target[key] = value

    def _prepare_character_messages_framework(self, language: str) -> None:
        """
        キャラクターメッセージの枠組みを準備（Phase 3-4: 空実装）

        Phase 7-9で実際の実装を行う。

        Args:
            language: 言語コード
        """
        if language not in self._character_messages:
            self._character_messages[language] = {}
        logger.debug(f"Prepared character message framework for {language}")

    async def initialize(self) -> None:
        """
        非同期初期化処理

        Phase 3-4では特に追加処理なし（将来の拡張用）
        """
        logger.debug("I18nManager async initialization complete")

    async def cleanup(self) -> None:
        """
        非同期クリーンアップ処理
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
        if language not in self.SUPPORTED_LANGUAGES:
            logger.warning(f"Unsupported language: {language}")
            return False

        if language == self.current_language:
            logger.debug(f"Language already set to: {language}")
            return True

        old_language = self.current_language
        self.current_language = language

        # YAMLファイルを再読み込み（新しい言語用）
        if language not in self._messages or not self._messages[language]:
            # 新しい言語のメッセージがキャッシュにない場合
            self._load_yaml_file(language, "errors.yaml", "errors")
            self._load_yaml_file(language, "ui.yaml", "ui")

        logger.info(f"Language changed from {old_language} to {language}")
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
            character_id: キャラクターID（Phase 3-4では未使用、Phase 7-9で実装）
            **kwargs: 追加の変数（variables引数の代替）

        Returns:
            フォーマット済みのエラーメッセージ（ユーザー向け）
        """
        # Noneチェック
        if error_code is None:
            return "Error: None"

        # 現在の言語のメッセージを取得（フォールバック付き）
        messages = self._messages.get(self.current_language, self._messages.get("ja", {}))
        error_messages = messages.get("errors", {})

        # エラーコードからメッセージを取得
        message_data = error_messages.get(error_code)

        if message_data is None:
            # 不明なエラーコードの場合
            logger.debug(f"Unknown error code: {error_code}")
            message = error_messages.get("unknown", f"Error: {error_code}")
            # エラーコードを末尾に追加
            message = f"{message}({error_code})"
        elif isinstance(message_data, dict):
            # 辞書形式（{user: '...', log: '...'}）の場合はuserフィールドを使用
            message = message_data.get("user", str(message_data))
        else:
            # 文字列形式（ハードコードまたは旧形式）の場合はそのまま使用
            message = message_data

        # 変数の埋め込み
        format_vars = {}
        if variables:
            format_vars.update(variables)
        if kwargs:
            format_vars.update(kwargs)

        if format_vars:
            try:
                message = message.format(**format_vars)
            except KeyError as e:
                logger.debug(f"Missing format variable: {e}")
            except Exception as e:
                logger.debug(f"Error formatting message: {e}")

        # キャラクター対応の枠組み（Phase 7-9で実装）
        if character_id:
            # TODO(Phase 7-9): キャラクター別メッセージの選択
            # 現時点では基本メッセージを返す
            pass

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

        # 現在の言語のメッセージを取得
        messages = self._messages.get(self.current_language, self._messages.get("ja", {}))

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
            except KeyError as e:
                logger.debug(f"Missing format variable: {e}")
            except Exception as e:
                logger.debug(f"Error formatting UI message: {e}")

        return result

    def get_log_message(
        self, log_code: str, variables: Optional[Dict[str, Any]] = None, **kwargs
    ) -> str:
        """
        ログメッセージを取得（常に英語）

        Args:
            log_code: ログコード（例: "L0001"または"E1001"）
            variables: メッセージに埋め込む変数
            **kwargs: 追加の変数

        Returns:
            フォーマット済みのログメッセージ（英語）
        """
        # Noneチェック
        if log_code is None:
            return "Log: None"

        # ログメッセージは常に英語
        messages = self._messages.get("en", self._messages.get("ja", {}))

        # まずログメッセージから探す
        log_messages = messages.get("logs", {})
        if log_code in log_messages:
            message = log_messages[log_code]
        else:
            # Eで始まるエラーコードの場合、エラーメッセージのlogフィールドを使用
            if log_code.startswith("E"):
                error_messages = messages.get("errors", {})
                error_data = error_messages.get(log_code)

                if error_data and isinstance(error_data, dict):
                    # 辞書形式の場合はlogフィールドを使用
                    message = error_data.get("log", f"Error occurred: {log_code}")
                elif error_data:
                    # 文字列形式の場合（ハードコード）はそのまま使用
                    message = error_data
                else:
                    logger.debug(f"Unknown log code: {log_code}")
                    message = f"Log: {log_code}"
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
            except KeyError as e:
                logger.debug(f"Missing format variable: {e}")
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

        Phase 3-4では枠組みのみ（Phase 7-9で実装予定）

        Args:
            character_id: キャラクターID
        """
        logger.debug(f"load_character_messages called with {character_id} (stub)")
        # TODO(Phase 7-9): キャラクター別メッセージの読み込み実装
        # - messages/{language}/characters.yamlを読み込み
        # - character_idに対応するメッセージを抽出
        # - self._character_messagesに格納

    def get_available_languages(self) -> List[str]:
        """
        利用可能な言語のリストを取得

        Returns:
            利用可能な言語コードのリスト
        """
        available = []
        for lang in self.SUPPORTED_LANGUAGES:
            lang_dir = self.messages_dir / lang
            if lang_dir.exists() and lang_dir.is_dir():
                # ディレクトリが存在すれば利用可能とみなす
                available.append(lang)
            elif lang in self._messages and self._messages[lang]:
                # またはハードコードメッセージがあれば利用可能
                available.append(lang)

        return available if available else ["ja"]  # 最低限日本語は返す
