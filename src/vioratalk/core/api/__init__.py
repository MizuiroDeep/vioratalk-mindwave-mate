"""API通信関連モジュール

Phase 4で作成
- base.py: BaseAPIClient（API通信の基底クラス）
- （将来）rate_limiter.py: レート制限
- （将来）api_manager.py: API統合管理

開発規約書 v1.12 セクション11準拠
"""

from .base import BaseAPIClient

__all__ = [
    "BaseAPIClient",
]
