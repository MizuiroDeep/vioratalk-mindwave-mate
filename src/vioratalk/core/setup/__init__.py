"""
VioraTalk Setup Package

自動セットアップ機能を提供するパッケージ。
初回起動時の環境構築と依存ソフトウェアの管理。

Copyright (c) 2025 MizuiroDeep
"""

from vioratalk.core.setup.auto_setup_manager import AutoSetupManager
from vioratalk.core.setup.types import ComponentStatus, SetupResult, SetupStatus

__all__ = [
    # 型定義
    "SetupStatus",
    "ComponentStatus",
    "SetupResult",
    # マネージャー
    "AutoSetupManager",
]
