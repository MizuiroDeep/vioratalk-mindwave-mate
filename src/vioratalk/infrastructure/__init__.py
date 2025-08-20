"""
VioraTalk Infrastructure Package

基盤層の機能を提供するパッケージ。
モデルダウンロード、ハードウェア制御、外部リソース管理などを行う。

Copyright (c) 2025 MizuiroDeep
"""

from vioratalk.infrastructure.model_download_manager import ModelDownloadManager

__all__ = [
    "ModelDownloadManager",
]
