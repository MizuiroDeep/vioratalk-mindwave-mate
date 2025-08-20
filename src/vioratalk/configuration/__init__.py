"""
VioraTalk Configuration Package

設定管理機能を提供するパッケージ。
ConfigManagerによる設定ファイル管理と、
settingsモジュールによる定数管理を行う。

Copyright (c) 2025 MizuiroDeep
"""

from vioratalk.configuration.config_manager import ConfigManager
from vioratalk.configuration.settings import (  # バージョン情報; 主要パス; デフォルト値; Phase機能; ヘルパー関数
    AUTHOR,
    BUILD_DATE,
    CACHE_DIR,
    DATA_DIR,
    DEFAULT_CONFIG_PATH,
    DEFAULT_LANGUAGE,
    DEFAULT_LLM_ENGINE,
    DEFAULT_LOG_LEVEL,
    DEFAULT_STT_ENGINE,
    DEFAULT_TTS_ENGINE,
    LOGS_DIR,
    MESSAGES_DIR,
    MODELS_DIR,
    PHASE,
    PHASE_1_FEATURES,
    PROJECT_ROOT,
    SETUP_COMPLETED_MARKER,
    USER_SETTINGS_DIR,
    VERSION,
    ensure_directories,
    get_phase_features,
    is_feature_enabled,
    mask_api_key,
)

__all__ = [
    # クラス
    "ConfigManager",
    # バージョン情報
    "VERSION",
    "PHASE",
    "BUILD_DATE",
    "AUTHOR",
    # パス定義
    "PROJECT_ROOT",
    "DATA_DIR",
    "USER_SETTINGS_DIR",
    "MESSAGES_DIR",
    "LOGS_DIR",
    "MODELS_DIR",
    "CACHE_DIR",
    "DEFAULT_CONFIG_PATH",
    "SETUP_COMPLETED_MARKER",
    # デフォルト値
    "DEFAULT_LANGUAGE",
    "DEFAULT_LOG_LEVEL",
    "DEFAULT_STT_ENGINE",
    "DEFAULT_LLM_ENGINE",
    "DEFAULT_TTS_ENGINE",
    # Phase機能
    "PHASE_1_FEATURES",
    # ヘルパー関数
    "ensure_directories",
    "get_phase_features",
    "is_feature_enabled",
    "mask_api_key",
]
