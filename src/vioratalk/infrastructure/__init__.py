"""
VioraTalk Infrastructure Package

基盤層の機能を提供するパッケージ。
モデルダウンロード、ハードウェア制御、外部リソース管理、
APIキー管理などを行う。

インポート規約 v1.1準拠
開発規約書 v1.12準拠

Copyright (c) 2025 MizuiroDeep
"""

# プロジェクト内インポート（アルファベット順）
from vioratalk.infrastructure.credential_manager import APIKeyManager, get_api_key_manager
from vioratalk.infrastructure.model_download_manager import ModelDownloadManager

# 公開APIを明示
__all__ = [
    # APIキー管理
    "APIKeyManager",
    "get_api_key_manager",
    # モデルダウンロード管理
    "ModelDownloadManager",
]
