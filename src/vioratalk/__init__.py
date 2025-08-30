"""
VioraTalk - MindWave Mate

AI対話アシスタントのコアパッケージ。
音声認識、言語モデル、音声合成を統合した対話システム。

Copyright (c) 2025 MizuiroDeep
License: MIT
"""

__version__ = "0.4.0"
__author__ = "MizuiroDeep"
__email__ = "36126374+MizuiroDeep@users.noreply.github.com"
__license__ = "MIT"

# Phase 1で公開するクラス
from vioratalk.core.base import ComponentState, VioraTalkComponent
from vioratalk.core.exceptions import ConfigurationError, InitializationError, VioraTalkError
from vioratalk.core.i18n_manager import I18nManager

# Phase 1の主要コンポーネント
from vioratalk.core.vioratalk_engine import VioraTalkEngine
from vioratalk.utils.logger_manager import LoggerManager

__all__ = [
    # バージョン情報
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    # 基底クラス
    "VioraTalkComponent",
    "ComponentState",
    # 例外クラス
    "VioraTalkError",
    "ConfigurationError",
    "InitializationError",
    # 主要コンポーネント
    "VioraTalkEngine",
    "I18nManager",
    "LoggerManager",
]
