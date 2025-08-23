"""VioraTalk 例外クラス階層

プロジェクト全体で使用される例外クラスを定義。
エラーコード体系に基づいた構造化されたエラーハンドリングを提供。

エラーハンドリング指針 v1.20準拠
開発規約書 v1.12準拠
インターフェース定義書 v1.34準拠
"""

from datetime import datetime
from typing import Any, Dict, Optional


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
        cause: Optional[Exception] = None,
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
            "timestamp": self.timestamp.isoformat(),
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
        # デフォルトエラーコードを設定
        if "error_code" not in kwargs:
            kwargs["error_code"] = "E0001"
        super().__init__(message, **kwargs)
        if config_file:
            self.details["config_file"] = config_file


class InitializationError(VioraTalkError):
    """初期化関連エラー (E0100-E0199)

    コンポーネントの初期化、依存関係チェックに関するエラー
    """

    def __init__(self, message: str, component: Optional[str] = None, **kwargs):
        # デフォルトエラーコードを設定
        if "error_code" not in kwargs:
            kwargs["error_code"] = "E0100"
        super().__init__(message, **kwargs)
        if component:
            self.details["component"] = component


class ComponentError(VioraTalkError):
    """コンポーネント関連エラー (E1000-E1099)

    個別コンポーネントの動作エラー
    """

    def __init__(self, message: str, state: Optional[str] = None, **kwargs):
        # デフォルトエラーコードを設定
        if "error_code" not in kwargs:
            kwargs["error_code"] = "E1000"
        super().__init__(message, **kwargs)
        if state:
            self.details["state"] = state


# ============================================================================
# 音声認識（STT）関連エラー (E1000-E1999)
# ============================================================================


class STTError(VioraTalkError):
    """音声認識関連エラーの基底クラス (E1000-E1999)

    音声認識エンジンのエラー全般
    """

    def __init__(self, message: str, **kwargs):
        # デフォルトエラーコードを設定
        if "error_code" not in kwargs:
            kwargs["error_code"] = "E1000"
        super().__init__(message, **kwargs)


class AudioError(STTError):
    """音声データ関連エラー (E1001-E1099)

    Phase 3で実装
    音声データの処理、フォーマット、品質に関するエラー

    エラーコード:
        E1001: マイクアクセスエラー
        E1002: 音声フォーマットエラー
        E1003: 音声が小さい
        E1004: 音声認識タイムアウト
    """

    def __init__(self, message: str, audio_file: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        if audio_file:
            self.details["audio_file"] = audio_file


# ============================================================================
# 言語モデル（LLM）関連エラー (E2000-E2999)
# ============================================================================


class LLMError(VioraTalkError):
    """言語モデル関連エラーの基底クラス (E2000-E2999)

    LLMエンジンのエラー全般
    """

    def __init__(self, message: str, **kwargs):
        # デフォルトエラーコードを設定
        if "error_code" not in kwargs:
            kwargs["error_code"] = "E2000"
        super().__init__(message, **kwargs)


class APIError(LLMError):
    """API通信関連エラー (E2001-E2099)

    Phase 3で実装
    APIとの通信、認証、レスポンスに関するエラー

    エラーコード:
        E2001: API認証エラー
        E2002: API利用制限
        E2003: APIタイムアウト
        E2004: LLMリクエストタイムアウト
    """

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        api_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        if status_code:
            self.details["status_code"] = status_code
        if api_name:
            self.details["api_name"] = api_name


class RateLimitError(APIError):
    """レート制限エラー (E2002)

    API利用制限に達した場合のエラー
    """

    def __init__(self, message: str, retry_after: Optional[int] = None, **kwargs):
        super().__init__(message, error_code="E2002", **kwargs)
        self.retry_after = retry_after
        if retry_after:
            self.details["retry_after"] = retry_after


# ============================================================================
# 音声合成（TTS）関連エラー (E3000-E3999)
# ============================================================================


class TTSError(VioraTalkError):
    """音声合成関連エラーの基底クラス (E3000-E3999)

    TTSエンジンのエラー全般
    """

    def __init__(self, message: str, **kwargs):
        # デフォルトエラーコードを設定
        if "error_code" not in kwargs:
            kwargs["error_code"] = "E3000"
        super().__init__(message, **kwargs)


class InvalidVoiceError(TTSError):
    """無効な音声設定エラー (E3001-E3099)

    Phase 3で実装
    音声ID、パラメータ、スタイルが無効な場合のエラー

    エラーコード:
        E3001: 無効な音声ID
        E3002: 無効な音声パラメータ
        E3003: 音声スタイル未対応
    """

    def __init__(self, message: str, voice_id: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        if voice_id:
            self.details["voice_id"] = voice_id


# ============================================================================
# パーソナライゼーション関連エラー (E4000-E4999)
# ============================================================================


class CharacterError(VioraTalkError):
    """キャラクター関連エラー (E4000-E4099)

    キャラクターのロード、設定、切り替えに関するエラー
    """

    def __init__(self, message: str, character_id: Optional[str] = None, **kwargs):
        # デフォルトエラーコードを設定
        if "error_code" not in kwargs:
            kwargs["error_code"] = "E4000"
        super().__init__(message, **kwargs)
        if character_id:
            self.details["character_id"] = character_id


class MemoryError(VioraTalkError):
    """記憶システム関連エラー (E4100-E4199)

    会話履歴、長期記憶、コンテキストに関するエラー
    """

    def __init__(self, message: str, memory_type: Optional[str] = None, **kwargs):
        # デフォルトエラーコードを設定
        if "error_code" not in kwargs:
            kwargs["error_code"] = "E4100"
        super().__init__(message, **kwargs)
        if memory_type:
            self.details["memory_type"] = memory_type


class EmotionError(VioraTalkError):
    """感情分析関連エラー (E4300-E4399)

    感情認識、分析、応答生成に関するエラー
    """

    def __init__(self, message: str, emotion_data: Optional[Dict[str, Any]] = None, **kwargs):
        # デフォルトエラーコードを設定
        if "error_code" not in kwargs:
            kwargs["error_code"] = "E4300"
        super().__init__(message, **kwargs)
        if emotion_data:
            self.details["emotion_data"] = emotion_data


# ============================================================================
# システム関連エラー (E5000-E5999)
# ============================================================================


class ResourceError(VioraTalkError):
    """リソース関連エラー (E5300-E5399)

    メモリ、CPU、ディスク容量などのリソースエラー
    """

    def __init__(self, message: str, resource_type: Optional[str] = None, **kwargs):
        # デフォルトエラーコードを設定
        if "error_code" not in kwargs:
            kwargs["error_code"] = "E5300"
        super().__init__(message, **kwargs)
        if resource_type:
            self.details["resource_type"] = resource_type


class NetworkError(VioraTalkError):
    """ネットワーク関連エラー (E5200-E5299)

    ネットワーク接続、通信エラー
    """

    def __init__(self, message: str, url: Optional[str] = None, **kwargs):
        # デフォルトエラーコードを設定
        if "error_code" not in kwargs:
            kwargs["error_code"] = "E5200"
        super().__init__(message, **kwargs)
        if url:
            self.details["url"] = url


class FileSystemError(VioraTalkError):
    """ファイルシステム関連エラー (E5100-E5199)

    ファイルI/O、ディレクトリ操作に関するエラー
    """

    pass


class TimeoutError(VioraTalkError):
    """タイムアウトエラー (E5400-E5499)

    各種処理のタイムアウト
    """

    def __init__(self, message: str, timeout_seconds: Optional[float] = None, **kwargs):
        super().__init__(message, **kwargs)
        if timeout_seconds:
            self.details["timeout_seconds"] = timeout_seconds


# ============================================================================
# モデル関連エラー (E2100-E2499, E2400-E2499)
# ============================================================================


class ModelError(VioraTalkError):
    """モデル関連エラーの基底クラス

    モデルのロード、管理、選択に関するエラー
    """

    pass


class ModelNotFoundError(ModelError):
    """モデルが見つからないエラー (E2100, E2400)

    Phase 3で実装
    指定されたモデルが存在しない場合のエラー

    エラーコード:
        E2100: LLMモデルが見つからない
        E2400: モデル選択失敗
    """

    def __init__(self, message: str, model_name: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        if model_name:
            self.details["model_name"] = model_name


class ModelLoadError(ModelError):
    """モデルロードエラー (E2101-E2199, E2401-E2499)

    モデルのロード、初期化に失敗した場合のエラー
    """

    pass


# ============================================================================
# バックグラウンドサービス関連エラー (E7000-E7999)
# ============================================================================


class BackgroundServiceError(VioraTalkError):
    """バックグラウンドサービス関連エラーの基底クラス (E7000-E7999)

    Phase 1で実装
    サービスの起動、停止、管理に関するエラー
    """

    def __init__(self, message: str, service_name: Optional[str] = None, **kwargs):
        # デフォルトエラーコードを設定
        if "error_code" not in kwargs:
            kwargs["error_code"] = "E7000"
        super().__init__(message, **kwargs)
        if service_name:
            self.details["service_name"] = service_name


class ServiceStartupError(BackgroundServiceError):
    """サービス起動エラー (E7001)

    サービスの起動に失敗した場合のエラー
    """

    def __init__(self, message: str, service_name: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="E7001", **kwargs)
        if service_name:
            self.details["service_name"] = service_name


class ServiceCommunicationError(BackgroundServiceError):
    """サービス通信エラー (E7005)

    サービス間の通信に失敗した場合のエラー
    """

    def __init__(self, message: str, service_name: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="E7005", **kwargs)
        if service_name:
            self.details["service_name"] = service_name


# ============================================================================
# 自動セットアップ関連エラー (E8000-E8999)
# ============================================================================


class SetupError(VioraTalkError):
    """セットアップ関連エラーの基底クラス (E8000-E8999)

    Phase 2で実装
    自動セットアップ、環境構築に関するエラー
    """

    def __init__(self, message: str, step: Optional[str] = None, **kwargs):
        # デフォルトエラーコードを設定
        if "error_code" not in kwargs:
            kwargs["error_code"] = "E8000"
        super().__init__(message, **kwargs)
        if step:
            self.details["setup_step"] = step


# ============================================================================
# 人間らしさ実装関連エラー (E9000-E9999)
# ============================================================================


class HumanLikeError(VioraTalkError):
    """人間らしさ関連エラーの基底クラス (E9000-E9999)

    Phase 8-9で実装
    性格理論、感情記憶、会話スタイルに関するエラー
    """

    def __init__(self, message: str, **kwargs):
        # デフォルトエラーコードを設定
        if "error_code" not in kwargs:
            kwargs["error_code"] = "E9000"
        super().__init__(message, **kwargs)
