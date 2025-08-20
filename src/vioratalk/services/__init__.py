"""
VioraTalk Services Package

バックグラウンドサービス管理機能を提供するパッケージ。
Ollama、AivisSpeechなどの外部サービスの管理。

Copyright (c) 2025 MizuiroDeep
"""

from vioratalk.services.background_service_manager import (
    BackgroundServiceManager,
    BaseBackgroundService,
    ServiceInfo,
    ServiceStatus,
)

__all__ = [
    # マネージャー
    "BackgroundServiceManager",
    # Enum & データクラス
    "ServiceStatus",
    "ServiceInfo",
    # 基底クラス
    "BaseBackgroundService",
]
