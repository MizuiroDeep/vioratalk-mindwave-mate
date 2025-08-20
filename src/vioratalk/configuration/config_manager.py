"""
VioraTalk Configuration Manager

設定ファイルの読み込み、保存、管理を行うマネージャー。
YAMLファイルによる設定管理とドット記法でのアクセスを提供。

Phase 1最小実装として基本的な設定管理機能を実装。
Phase 2以降で設定の動的リロード、設定プロファイル切り替え、
設定の暗号化などの高度な機能を追加予定。

Copyright (c) 2025 MizuiroDeep
"""

import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from vioratalk.configuration.settings import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_LANGUAGE,
    DEFAULT_LLM_ENGINE,
    DEFAULT_LOG_LEVEL,
    DEFAULT_STT_ENGINE,
    DEFAULT_TTS_ENGINE,
)
from vioratalk.core.base import ComponentState, VioraTalkComponent
from vioratalk.core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class ConfigManager(VioraTalkComponent):
    """設定管理マネージャー

    YAMLファイル形式の設定を管理し、ドット記法でのアクセスを提供。
    デフォルト設定とユーザー設定のマージ、設定の検証機能を持つ。

    Attributes:
        config_path: 設定ファイルのパス
        _config: 現在の設定内容
        _defaults: デフォルト設定

    NOTE: Phase 1最小実装のため、SHUTDOWN状態を使用。
          Phase 2でTERMINATING状態と動的リロード機能を追加予定。
    """

    def __init__(self, config_path: Optional[Path] = None):
        """ConfigManagerの初期化

        Args:
            config_path: 設定ファイルのパス（省略時はデフォルト）
        """
        super().__init__()  # VioraTalkComponentは引数を受け取らない
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self._config: Dict[str, Any] = {}
        self._defaults = self._get_default_config()
        logger.debug(f"ConfigManager initialized with path: {self.config_path}")

    async def initialize(self) -> None:
        """非同期初期化処理

        設定ファイルを読み込み、デフォルト値とマージして検証する。

        Raises:
            ConfigurationError: 設定の読み込みまたは検証に失敗した場合（E0001）
        """
        self._state = ComponentState.INITIALIZING
        logger.info("ConfigManager initialization started")

        try:
            # 設定ファイルの読み込み（同期処理）
            self._config = self.load_config(self.config_path)

            # 設定の検証
            errors = self.validate_config(self._config)
            if errors:
                logger.warning(f"Configuration validation warnings: {errors}")
                # Phase 1では警告のみ（エラーにしない）

            self._state = ComponentState.READY
            logger.info("ConfigManager initialization completed")

        except Exception as e:
            self._state = ComponentState.ERROR
            logger.error(f"ConfigManager initialization failed: {e}")
            raise ConfigurationError(
                f"ConfigManager initialization failed: {e}",
                config_file=str(self.config_path),
                error_code="E0001",
            )

    async def cleanup(self) -> None:
        """リソースのクリーンアップ

        設定を保存してからクリーンアップする。

        NOTE: Phase 1最小実装のため、SHUTDOWNを使用。
              Phase 2でTERMINATING → TERMINATEDの遷移に変更予定。
        """
        logger.info("ConfigManager cleanup started")
        self._state = ComponentState.SHUTDOWN  # Phase 1: SHUTDOWNを使用

        try:
            # 変更があれば保存（Phase 2以降で実装予定）
            # 現在は特にクリーンアップ処理なし
            pass

        except Exception as e:
            logger.error(f"Error during ConfigManager cleanup: {e}")
        finally:
            self._state = ComponentState.TERMINATED
            logger.info("ConfigManager cleanup completed")

    def is_available(self) -> bool:
        """利用可能状態の確認

        Returns:
            bool: READY状態の場合True

        NOTE: Phase 1最小実装のため、READYのみチェック。
              Phase 2でRUNNINGを追加し、is_operational()を使用予定。
        """
        return self._state == ComponentState.READY  # Phase 1: 直接チェック

    def get_status(self) -> Dict[str, Any]:
        """コンポーネントの状態を取得

        Returns:
            Dict[str, Any]: 状態情報を含む辞書
        """
        from datetime import datetime

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
        """設定を保存

        Args:
            config: 保存する設定
            config_path: 保存先のパス

        Raises:
            ConfigurationError: ファイル保存エラー（E0001）
        """
        logger.debug(f"Saving config to: {config_path}")

        try:
            # ディレクトリが存在しない場合は作成
            config_path.parent.mkdir(parents=True, exist_ok=True)

            # YAMLファイルとして保存
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    config, f, default_flow_style=False, allow_unicode=True, sort_keys=False
                )

            logger.info(f"Configuration saved successfully to {config_path}")

        except PermissionError as e:
            raise ConfigurationError(
                f"Permission denied when saving config: {e}",
                config_file=str(config_path),
                error_code="E0001",
            )
        except Exception as e:
            raise ConfigurationError(
                f"Failed to save config file: {e}", config_file=str(config_path), error_code="E0001"
            )

    def get(self, key: str, default: Any = None) -> Any:
        """設定値を取得（ドット記法対応）

        Args:
            key: 設定キー（例: "llm.model"、"general.language"）
            default: デフォルト値

        Returns:
            Any: 設定値（存在しない場合はdefault）
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """設定値を更新（ドット記法対応）

        Args:
            key: 設定キー（例: "llm.model"）
            value: 設定する値
        """
        keys = key.split(".")
        config = self._config

        # 最後のキーまで辿る（途中のキーが存在しない場合は作成）
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            elif not isinstance(config[k], dict):
                # 既存の値が辞書でない場合は上書き
                config[k] = {}
            config = config[k]

        # 最後のキーに値を設定
        config[keys[-1]] = value

        logger.debug(f"Configuration updated: {key} = {value}")

    def get_all(self) -> Dict[str, Any]:
        """すべての設定を取得

        Returns:
            Dict[str, Any]: 現在の設定全体の深いコピー
        """
        return copy.deepcopy(self._config)

    def get_config(self) -> Dict[str, Any]:
        """現在の設定を取得（互換性のため維持）

        Returns:
            Dict[str, Any]: 現在の設定のコピー
        """
        return self.get_all()

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """設定の妥当性を検証

        Args:
            config: 検証する設定

        Returns:
            List[str]: エラーメッセージのリスト（空なら正常）
        """
        errors = []

        # Phase 1では基本的な検証のみ

        # general セクションの検証
        if "general" not in config:
            errors.append("Missing required section: general")
        else:
            general = config["general"]
            if "language" in general:
                if general["language"] not in ["ja", "en"]:
                    errors.append(f"Invalid language: {general['language']} (must be 'ja' or 'en')")

        # STT設定の検証
        if "stt" in config:
            stt = config["stt"]
            if "device" in stt:
                if stt["device"] not in ["default", "cuda", "cpu"]:
                    errors.append(
                        f"Invalid STT device: {stt['device']} "
                        f"(must be 'default', 'cuda', or 'cpu')"
                    )

        # LLM設定の検証
        if "llm" in config:
            llm = config["llm"]
            if "temperature" in llm:
                temp = llm["temperature"]
                if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
                    errors.append(f"Invalid LLM temperature: {temp} (must be 0.0-2.0)")

        return errors

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """辞書を再帰的にマージ

        Args:
            base: ベースとなる辞書
            override: 上書きする辞書

        Returns:
            Dict[str, Any]: マージされた辞書
        """
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # 両方が辞書の場合は再帰的にマージ
                result[key] = self._deep_merge(result[key], value)
            else:
                # それ以外は上書き
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
                "version": "0.0.1",
                "language": DEFAULT_LANGUAGE,
                "theme": "light",
                "log_level": DEFAULT_LOG_LEVEL,
            },
            "stt": {
                "engine": DEFAULT_STT_ENGINE,
                "model": "base",
                "device": "default",
                "language": DEFAULT_LANGUAGE,
            },
            "llm": {
                "provider": DEFAULT_LLM_ENGINE,
                "model": "llama3",
                "temperature": 0.7,
                "max_tokens": 2048,
                "timeout": 30,
            },
            "tts": {
                "engine": DEFAULT_TTS_ENGINE,
                "speaker_id": 1,
                "speed": 1.0,
                "pitch": 0,
                "volume": 1.0,
            },
            "features": {
                "auto_setup": True,
                "background_service": False,
                "limited_mode": False,
                "voice_activation": False,
            },
            "paths": {
                "data": "data",
                "logs": "logs",
                "models": "models",
                "cache": "cache",
            },
        }
