"""設定管理マネージャー

YAMLベースの設定ファイル管理とバリデーション機能を提供。
環境変数や設定ファイルの優先順位を管理し、動的な設定変更をサポート。

設定管理仕様書 v1.3準拠
インターフェース定義書 v1.34準拠
開発規約書 v1.12準拠
"""

import copy
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from vioratalk.core.base import ComponentState, VioraTalkComponent
from vioratalk.core.exceptions import ConfigurationError

# ロガー設定
logger = logging.getLogger(__name__)


class ConfigManager(VioraTalkComponent):
    """設定管理マネージャー

    YAMLベースの設定ファイル管理とバリデーション機能を提供する。
    設定の階層的な管理と環境変数によるオーバーライドをサポート。

    設定管理仕様書 v1.3準拠：
    - 設定ファイルの読み込みと保存
    - 環境変数によるオーバーライド
    - 設定値のバリデーション
    - デフォルト値の提供

    Attributes:
        config_path: 設定ファイルのパス
        _config: 現在の設定値
        _defaults: デフォルト設定値
    """

    def __init__(self, config_path: Optional[Path] = None):
        """初期化

        Args:
            config_path: 設定ファイルのパス（Noneの場合はデフォルトパス使用）
        """
        super().__init__()

        # 設定ファイルパスの決定
        if config_path is None:
            # デフォルトパス: settings.pyのDEFAULT_CONFIG_PATH
            from vioratalk.configuration.settings import DEFAULT_CONFIG_PATH

            config_path = DEFAULT_CONFIG_PATH

        self.config_path = config_path
        self._config: Dict[str, Any] = {}
        self._defaults = self._get_default_config()

        logger.info(f"ConfigManager initialized with path: {self.config_path}")

    async def initialize(self) -> None:
        """非同期初期化処理

        設定ファイルを読み込み、環境変数でオーバーライドする。

        Raises:
            ConfigurationError: 設定ファイルの読み込みやバリデーションに失敗
        """
        self._state = ComponentState.INITIALIZING
        logger.info("ConfigManager initialization started")

        try:
            # 設定ファイルの読み込み（存在しない場合はデフォルト使用）
            self._config = self.load_config(self.config_path)

            # 環境変数でオーバーライド
            self._apply_env_overrides()

            # 設定値のバリデーション
            errors = self.validate_config(self._config)
            if errors:
                logger.warning(f"Configuration validation warnings: {errors}")

            self._state = ComponentState.READY
            logger.info("ConfigManager initialization completed")

        except yaml.YAMLError as e:
            self._state = ComponentState.ERROR
            raise ConfigurationError(
                f"Failed to parse YAML config: {e}",
                config_file=str(self.config_path),
                error_code="E0001",
            )
        except Exception as e:
            self._state = ComponentState.ERROR
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(
                f"Configuration initialization failed: {e}", error_code="E0001"
            )

    def get(self, key: str, default: Any = None) -> Any:
        """設定値を取得

        ドット記法で階層的なキーを指定可能。
        例: "llm.engine" → config["llm"]["engine"]

        Args:
            key: 設定キー（ドット記法対応）
            default: デフォルト値

        Returns:
            Any: 設定値またはデフォルト値
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """設定値を設定

        ドット記法で階層的なキーを指定可能。
        存在しない階層は自動的に作成される。

        Args:
            key: 設定キー（ドット記法対応）
            value: 設定する値
        """
        keys = key.split(".")
        config = self._config

        # 最後のキー以外を処理
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            elif not isinstance(config[k], dict):
                # 既存の値が辞書でない場合は辞書に置き換え
                config[k] = {}
            config = config[k]

        # 最後のキーに値を設定
        config[keys[-1]] = value

        logger.debug(f"Configuration updated: {key} = {value}")

    def get_all(self) -> Dict[str, Any]:
        """全設定を取得

        Returns:
            Dict[str, Any]: 全設定の深いコピー
        """
        return copy.deepcopy(self._config)

    def get_config(self) -> Dict[str, Any]:
        """全設定を取得（互換性のための別名）

        Returns:
            Dict[str, Any]: 全設定の深いコピー
        """
        return self.get_all()

    async def cleanup(self) -> None:
        """クリーンアップ処理

        設定の保存など、終了時の処理を行う。
        """
        self._state = ComponentState.TERMINATING
        logger.info("ConfigManager cleanup started")

        # Phase 1では特に処理なし
        # Phase 2以降: 変更された設定の自動保存など

        self._state = ComponentState.TERMINATED
        logger.info("ConfigManager cleanup completed")

    def is_available(self) -> bool:
        """利用可能状態の確認

        Returns:
            bool: 利用可能な場合True
        """
        return self._state in [ComponentState.READY, ComponentState.RUNNING]

    def get_status(self) -> Dict[str, Any]:
        """コンポーネントの状態を取得

        Returns:
            Dict[str, Any]: 状態情報を含む辞書
        """

        return {
            "state": self._state,
            "is_available": self.is_available(),
            "error": None,
            "last_used": datetime.now(),
            "config_path": str(self.config_path),
            "config_loaded": bool(self._config),
        }

    def load_config(self, config_path: Path) -> Dict[str, Any]:
        """設定ファイルを読み込み

        Args:
            config_path: 読み込む設定ファイルのパス

        Returns:
            Dict[str, Any]: 読み込んだ設定（デフォルト値とマージ済み）

        Raises:
            ConfigurationError: ファイル読み込みエラー（E0001）
        """
        logger.debug(f"Loading config from: {config_path}")

        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return copy.deepcopy(self._defaults)

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

            # デフォルト値とマージ
            merged_config = self._deep_merge(copy.deepcopy(self._defaults), config)

            logger.info(f"Configuration loaded successfully from {config_path}")
            return merged_config

        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Failed to parse YAML config file: {e}",
                config_file=str(config_path),
                error_code="E0001",
            )
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load config file: {e}", config_file=str(config_path), error_code="E0001"
            )

    def save_config(self, config: Dict[str, Any], config_path: Path) -> None:
        """設定をファイルに保存

        Args:
            config: 保存する設定
            config_path: 保存先のパス

        Raises:
            ConfigurationError: ファイル保存エラー（E0003）
        """
        logger.debug(f"Saving config to: {config_path}")

        try:
            # ディレクトリが存在しない場合は作成
            config_path.parent.mkdir(parents=True, exist_ok=True)

            with open(config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    config, f, default_flow_style=False, allow_unicode=True, sort_keys=False
                )

            logger.info(f"Configuration saved successfully to {config_path}")

        except Exception as e:
            raise ConfigurationError(
                f"Failed to save config file: {e}", config_file=str(config_path), error_code="E0003"
            )

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """設定値のバリデーション

        Args:
            config: 検証する設定

        Returns:
            List[str]: エラーメッセージのリスト（空の場合は有効）
        """
        errors = []

        # generalセクションの検証
        if "general" not in config:
            errors.append("Missing required section: general")
        else:
            # 必須フィールドの確認（languageのみ必須）
            if "language" not in config["general"]:
                errors.append("Required field missing: general.language")
            elif config["general"]["language"] not in ["ja", "en"]:
                errors.append(f"Invalid language code: {config['general']['language']}")

        # sttセクションの検証（Phase 1では基本的な検証のみ）
        if "stt" in config:
            if "device" in config["stt"]:
                if config["stt"]["device"] not in ["cpu", "cuda"]:
                    errors.append(f"Invalid STT device: {config['stt']['device']}")

        # llmセクションの検証
        if "llm" in config:
            if "temperature" in config["llm"]:
                temp = config["llm"]["temperature"]
                if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
                    errors.append(f"Invalid temperature value: {temp} (must be 0.0-2.0)")

        return errors

    def _apply_env_overrides(self) -> None:
        """環境変数による設定のオーバーライド

        VIORATALK_で始まる環境変数を設定値にマッピング。
        例: VIORATALK_LLM_ENGINE → config["llm"]["engine"]
        """
        for env_key, env_value in os.environ.items():
            if env_key.startswith("VIORATALK_"):
                # VIORATALK_を除去してキーを作成
                config_key = env_key[10:].lower().replace("_", ".")

                # 値の型変換
                value = self._parse_env_value(env_value)

                # 設定に適用
                self.set(config_key, value)
                logger.debug(f"Environment override: {config_key} = {value}")

    def _parse_env_value(self, value: str) -> Any:
        """環境変数の値を適切な型に変換

        Args:
            value: 環境変数の文字列値

        Returns:
            Any: 変換後の値
        """
        # ブール値
        if value.lower() in ["true", "yes", "1"]:
            return True
        elif value.lower() in ["false", "no", "0"]:
            return False

        # 数値
        try:
            if "." in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass

        # 文字列のまま
        return value

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """辞書を再帰的にマージ

        Args:
            base: ベースとなる辞書
            override: 上書きする辞書

        Returns:
            Dict[str, Any]: マージ後の辞書
        """
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定を取得

        Returns:
            Dict[str, Any]: デフォルト設定
        """
        return {
            "general": {
                "app_name": "VioraTalk",
                "language": "ja",
                "log_level": "INFO",
                "auto_save_interval": 300,
            },
            "stt": {
                "engine": "faster-whisper",
                "model": "base",
                "language": "ja",
                "device": "cpu",
                "vad_threshold": 0.5,
                "max_recording_duration": 30,
            },
            "llm": {
                "engine": "gemini",
                "fallback": "claude",
                "max_tokens": 2000,
                "temperature": 0.7,
                "timeout": 30,
                "retry": {"max_attempts": 3, "initial_delay": 1.0, "backoff_factor": 2.0},
            },
            "tts": {
                "engine": "auto",
                "auto_mode": "speed",
                "volume": 0.9,
                "speed": 1.0,
                "pitch": 1.0,
            },
            "memory": {
                "short_term_hours": 24,
                "medium_term_hours": 168,
                "max_memories": {"short_term": 100, "medium_term": 500, "long_term": 10000},
                "importance_threshold": 0.7,
            },
            "features": {"auto_setup": True, "limited_mode": False, "debug_mode": False},
        }

    def __str__(self) -> str:
        """文字列表現"""
        return f"ConfigManager(path={self.config_path}, loaded={bool(self._config)})"

    def __repr__(self) -> str:
        """詳細な文字列表現"""
        return (
            f"ConfigManager(config_path={self.config_path}, "
            f"state={self._state}, sections={list(self._config.keys())})"
        )
