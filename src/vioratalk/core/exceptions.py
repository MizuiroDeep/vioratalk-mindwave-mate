"""VioraTalk 例外クラス階層

プロジェクト全体で使用される例外クラスを定義。
エラーコード体系に基づいた構造化されたエラーハンドリングを提供。

エラーハンドリング指針 v1.20準拠
開発規約書 v1.12準拠
"""

from datetime import datetime
from typing import Dict, Any, Optional


class VioraTalkError(Exception):
    """VioraTalkプロジェクトの基底例外クラス
    
    すべてのカスタム例外の親クラス。
    エラーコード、詳細情報、原因となった例外の連鎖をサポート。
    
    Attributes:
        message: ユーザー向けエラーメッセージ
        error_code: エラーコード（E0001など）
        details: 詳細情報の辞書
        cause: 原因となった例外
        timestamp: エラー発生時刻
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """例外を初期化
        
        Args:
            message: エラーメッセージ
            error_code: エラーコード（E0001-E9999）
            details: 追加の詳細情報
            cause: 原因となった例外
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.cause = cause
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """エラー情報を辞書形式で取得
        
        Returns:
            Dict[str, Any]: エラー情報の辞書
        """
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "cause": str(self.cause) if self.cause else None,
            "timestamp": self.timestamp.isoformat()
        }
    
    def __str__(self) -> str:
        """文字列表現"""
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


# ============================================================================
# 初期化・設定関連エラー (E0001-E0999)
# ============================================================================

class ConfigurationError(VioraTalkError):
    """設定関連エラー (E0001-E0099)
    
    設定ファイルの読み込み、パース、検証に関するエラー
    """
    
    def __init__(self, message: str, config_file: Optional[str] = None, **kwargs):
        if "error_code" not in kwargs:
            kwargs["error_code"] = "E0001"
        super().__init__(message, **kwargs)
        if config_file:
            self.details["config_file"] = config_file


class InitializationError(VioraTalkError):
    """初期化エラー (E0100-E0199)
    
    コンポーネントの初期化時に発生するエラー
    """
    
    def __init__(self, message: str, component: Optional[str] = None, **kwargs):
        if "error_code" not in kwargs:
            kwargs["error_code"] = "E0100"
        super().__init__(message, **kwargs)
        if component:
            self.details["component"] = component


# ============================================================================
# コンポーネント関連エラー (E1000-E1999)
# ============================================================================

class ComponentError(VioraTalkError):
    """コンポーネント関連エラー (E1000-E1099)
    
    コンポーネントの一般的なエラー
    """
    
    def __init__(self, message: str, component_name: Optional[str] = None, **kwargs):
        if "error_code" not in kwargs:
            kwargs["error_code"] = "E1000"
        super().__init__(message, **kwargs)
        if component_name:
            self.details["component_name"] = component_name


# ============================================================================
# STT（音声認識）関連エラー (E1000-E1999) - 将来の実装用
# ============================================================================

class STTError(VioraTalkError):
    """音声認識エラーの基底クラス
    
    Phase 3-4で実装予定
    """
    pass


class AudioError(STTError):
    """音声データ関連エラー
    
    Phase 3-4で実装予定
    """
    pass


# ============================================================================
# LLM（言語モデル）関連エラー (E2000-E2999) - 将来の実装用
# ============================================================================

class LLMError(VioraTalkError):
    """言語モデルエラーの基底クラス
    
    Phase 3-4で実装予定
    """
    pass


class APIError(LLMError):
    """API関連エラー
    
    Phase 3-4で実装予定
    """
    pass


# ============================================================================
# TTS（音声合成）関連エラー (E3000-E3999) - 将来の実装用
# ============================================================================

class TTSError(VioraTalkError):
    """音声合成エラーの基底クラス
    
    Phase 3-4で実装予定
    """
    pass


# ============================================================================
# キャラクター・記憶関連エラー (E4000-E4999) - 将来の実装用
# ============================================================================

class CharacterError(VioraTalkError):
    """キャラクター関連エラー
    
    Phase 7で実装予定
    """
    pass


class MemoryError(VioraTalkError):
    """記憶システム関連エラー
    
    Phase 6で実装予定
    """
    pass


class EmotionError(VioraTalkError):
    """感情分析関連エラー
    
    Phase 8で実装予定
    """
    pass


# ============================================================================
# ライセンス関連エラー (E4200-E4299) - 将来の実装用
# ============================================================================

class LicenseError(VioraTalkError):
    """ライセンス関連エラー
    
    Pro版機能で使用予定（E4201: Pro版限定機能へのアクセス）
    """
    
    def __init__(self, message: str, feature: Optional[str] = None, **kwargs):
        if "error_code" not in kwargs:
            kwargs["error_code"] = "E4201"
        super().__init__(message, **kwargs)
        if feature:
            self.details["required_feature"] = feature


# ============================================================================
# システム・リソース関連エラー (E5000-E5999)
# ============================================================================

class ResourceError(VioraTalkError):
    """リソース関連エラー
    
    メモリ、ディスク、CPU等のリソース不足
    """
    
    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        available: Optional[int] = None,
        required: Optional[int] = None,
        **kwargs
    ):
        if "error_code" not in kwargs:
            kwargs["error_code"] = "E5300"
        super().__init__(message, **kwargs)
        if resource_type:
            self.details["resource_type"] = resource_type
        if available is not None:
            self.details["available"] = available
        if required is not None:
            self.details["required"] = required


class NetworkError(VioraTalkError):
    """ネットワーク関連エラー
    
    通信エラー、タイムアウト等
    """
    
    def __init__(self, message: str, url: Optional[str] = None, **kwargs):
        if "error_code" not in kwargs:
            kwargs["error_code"] = "E5200"
        super().__init__(message, **kwargs)
        if url:
            self.details["url"] = url


# ============================================================================
# バックグラウンドサービス関連エラー (E7000-E7999) - 将来の実装用
# ============================================================================

class BackgroundServiceError(VioraTalkError):
    """バックグラウンドサービス関連エラー
    
    Phase 1 Part 2で実装予定
    """
    
    def __init__(self, message: str, service_name: Optional[str] = None, **kwargs):
        if "error_code" not in kwargs:
            kwargs["error_code"] = "E7000"
        super().__init__(message, **kwargs)
        if service_name:
            self.details["service_name"] = service_name


class ServiceStartupError(BackgroundServiceError):
    """サービス起動エラー"""
    
    def __init__(self, message: str, **kwargs):
        if "error_code" not in kwargs:
            kwargs["error_code"] = "E7001"
        super().__init__(message, **kwargs)


class ServiceCommunicationError(BackgroundServiceError):
    """サービス間通信エラー"""
    
    def __init__(self, message: str, **kwargs):
        if "error_code" not in kwargs:
            kwargs["error_code"] = "E7005"
        super().__init__(message, **kwargs)


# ============================================================================
# セットアップ関連エラー (E8000-E8999) - 将来の実装用
# ============================================================================

class SetupError(VioraTalkError):
    """セットアップ関連エラー
    
    Phase 2で実装予定
    """
    
    def __init__(self, message: str, step: Optional[str] = None, **kwargs):
        if "error_code" not in kwargs:
            kwargs["error_code"] = "E8000"
        super().__init__(message, **kwargs)
        if step:
            self.details["setup_step"] = step


# ============================================================================
# 人間らしさ実装関連エラー (E9000-E9999) - 将来の実装用
# ============================================================================

class HumanLikeError(VioraTalkError):
    """人間らしさ実装関連エラー
    
    Phase 8-9で実装予定
    """
    
    def __init__(self, message: str, **kwargs):
        if "error_code" not in kwargs:
            kwargs["error_code"] = "E9000"
        super().__init__(message, **kwargs)