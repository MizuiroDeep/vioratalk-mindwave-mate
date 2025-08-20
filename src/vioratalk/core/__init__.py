"""
VioraTalk Core Package

コア機能を提供するパッケージ。
基底クラス、例外、エンジン、国際化などの中核機能を含む。

Copyright (c) 2025 MizuiroDeep
"""

from vioratalk.core.base import ComponentState, VioraTalkComponent
from vioratalk.core.exceptions import (
    BackgroundServiceError,
    ComponentError,
    ConfigurationError,
    InitializationError,
    ServiceCommunicationError,
    ServiceStartupError,
    SetupError,
    VioraTalkError,
)
from vioratalk.core.i18n_manager import I18nManager
from vioratalk.core.vioratalk_engine import VioraTalkEngine

__all__ = [
    # 基底クラス
    "VioraTalkComponent",
    "ComponentState",
    # 例外クラス
    "VioraTalkError",
    "ConfigurationError",
    "InitializationError",
    "ComponentError",
    "BackgroundServiceError",
    "ServiceStartupError",
    "ServiceCommunicationError",
    "SetupError",
    # コアコンポーネント
    "VioraTalkEngine",
    "I18nManager",
]
