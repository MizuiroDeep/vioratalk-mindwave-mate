"""error_handler.py - エラーハンドリング統一実装

VioraTalkの統一的なエラー処理メカニズム。
エラー情報の構造化、ログ出力、ユーザーメッセージの国際化対応を提供。

エラー処理実装ガイド v1.0準拠
エラーハンドリング指針 v1.20準拠
開発規約書 v1.12準拠
データフォーマット仕様書 v1.5準拠
"""

# 標準ライブラリ
import asyncio
import logging
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

# プロジェクト内インポート


# ============================================================================
# データクラス定義
# ============================================================================


@dataclass
class ErrorContext:
    """エラーコンテキスト

    エラーが発生した状況を記録するためのコンテキスト情報。
    データフォーマット仕様書 v1.5準拠。
    """

    phase: str  # 発生Phase（例: "Phase 4", "initialization"）
    component: str  # コンポーネント名（例: "DialogueManager"）
    operation: str  # 実行中の操作（例: "process_audio_input"）
    user_id: Optional[str] = None  # ユーザーID
    session_id: Optional[str] = None  # セッションID

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "phase": self.phase,
            "component": self.component,
            "operation": self.operation,
            "user_id": self.user_id,
            "session_id": self.session_id,
        }


@dataclass
class ErrorInfo:
    """エラー情報

    Phase間でのエラー伝播に使用される構造化エラー情報。
    エラー処理実装ガイド v1.0およびデータフォーマット仕様書 v1.5準拠。
    """

    # エラー識別
    error_code: str  # E1001形式のエラーコード
    error_type: str  # 例外クラス名（例: "ValidationError"）

    # エラー詳細
    message: str  # エラーメッセージ
    details: Optional[Dict[str, Any]] = None  # 追加の詳細情報

    # コンテキスト
    context: Optional[ErrorContext] = None
    timestamp: datetime = field(default_factory=datetime.now)

    # トレース情報
    traceback: Optional[str] = None
    cause: Optional["ErrorInfo"] = None  # 原因となったエラー

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換（ログ出力用）"""
        result = {
            "error_code": self.error_code,
            "error_type": self.error_type,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details or {},
        }

        if self.context:
            result["context"] = self.context.to_dict()

        if self.traceback:
            result["traceback"] = self.traceback

        if self.cause:
            result["cause"] = self.cause.to_dict()

        return result

    def to_user_message(self, character_id: Optional[str] = None) -> str:
        """ユーザー向けメッセージに変換

        Phase 7-9でキャラクター対応メッセージ生成に拡張予定。
        現在は基本的なメッセージ生成のみ。
        """
        # Phase 7以降でI18nManagerとCharacterManagerを使用してメッセージ生成
        return f"エラーが発生しました。({self.error_code})"


# ============================================================================
# エラーハンドラー実装
# ============================================================================


class ErrorHandler:
    """エラーハンドリングの統一実装

    Phase 1: 基礎実装（同期メソッド、基本的なログ記録）
    Phase 3-4: I18nManager連携、非同期化、エラーメッセージ国際化
    Phase 7-9: キャラクター対応エラーメッセージ
    """

    def __init__(self, i18n_manager: Optional[Any] = None):
        """初期化

        Args:
            i18n_manager: I18nManagerインスタンス（Phase 3-4で必須）
        """
        self.i18n = i18n_manager
        self.logger = logging.getLogger(__name__)
        self._error_listeners: List[Callable] = []
        self._error_history: List[ErrorInfo] = []
        self._max_history_size = 100

    # ========================================================================
    # Phase 1: 同期版実装
    # ========================================================================

    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        character_id: Optional[str] = None,  # Phase 7-9準備
    ) -> ErrorInfo:
        """エラー処理（Phase 1: 同期版）

        Args:
            error: 発生した例外
            context: エラーコンテキスト情報
            character_id: キャラクターID（Phase 7-9で使用）

        Returns:
            ErrorInfo: 構造化されたエラー情報
        """
        # 1. ErrorInfo作成
        error_info = self._create_error_info(error, context)

        # 2. ログ記録
        self.log_error(error_info)

        # 3. エラー履歴に追加
        self._add_to_history(error_info)

        # 4. ユーザーメッセージ取得
        user_message = self._get_user_message(error_info, character_id=character_id)

        # 5. リスナーに通知
        self._notify_listeners(error_info, user_message)

        return error_info

    # ========================================================================
    # Phase 3-4: 非同期版実装
    # ========================================================================

    async def handle_error_async(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        character_id: Optional[str] = None,
    ) -> ErrorInfo:
        """エラー処理（Phase 3-4: 非同期版）

        非同期処理中のエラーハンドリング。
        I18nManager連携とユーザー通知を含む。

        Args:
            error: 発生した例外
            context: エラーコンテキスト情報
            character_id: キャラクターID（Phase 7-9で使用）

        Returns:
            ErrorInfo: 構造化されたエラー情報
        """
        # 1. ErrorInfo作成
        error_info = self._create_error_info(error, context)

        # 2. ログ記録（非同期）
        await self._log_error_async(error_info)

        # 3. エラー履歴に追加
        self._add_to_history(error_info)

        # 4. ユーザーメッセージ取得（I18nManager使用）
        user_message = await self._get_user_message_async(error_info, character_id=character_id)

        # 5. ユーザーへの通知（非同期）
        await self._notify_user_async(user_message)

        # 6. 非同期リスナーに通知
        await self._notify_listeners_async(error_info, user_message)

        return error_info

    # ========================================================================
    # 内部メソッド
    # ========================================================================

    def _create_error_info(
        self, error: Exception, context: Optional[Dict[str, Any]] = None
    ) -> ErrorInfo:
        """ErrorInfo作成

        例外からErrorInfoを生成する。
        """
        # エラーコード取得
        error_code = self._get_error_code(error)

        # エラーコンテキスト作成
        error_context = None
        if context:
            error_context = ErrorContext(
                phase=context.get("phase", "unknown"),
                component=context.get("component", "unknown"),
                operation=context.get("operation", "unknown"),
                user_id=context.get("user_id"),
                session_id=context.get("session_id"),
            )

        # 詳細情報取得
        details = {}
        if hasattr(error, "details") and isinstance(error.details, dict):
            details = error.details

        # トレースバック取得（デバッグ時のみ）
        tb = None
        if self.logger.isEnabledFor(logging.DEBUG):
            tb = traceback.format_exc()

        # ErrorInfo作成
        return ErrorInfo(
            error_code=error_code,
            error_type=type(error).__name__,
            message=str(error),
            details=details,
            context=error_context,
            traceback=tb,
        )

    def _get_error_code(self, error: Exception) -> str:
        """エラーコード取得

        エラーハンドリング指針 v1.20のエラーコード体系に準拠。
        """
        if hasattr(error, "error_code"):
            return error.error_code

        # デフォルトエラーコード（カテゴリ別）
        error_type = type(error).__name__

        # E0xxx: 初期化関連
        if "Initialization" in error_type or "Setup" in error_type:
            return "E0001"
        # E1xxx: STT関連
        elif "STT" in error_type or "Audio" in error_type:
            return "E1000"
        # E2xxx: LLM関連
        elif "LLM" in error_type or "API" in error_type:
            return "E2000"
        # E3xxx: TTS関連
        elif "TTS" in error_type or "Voice" in error_type:
            return "E3000"
        # E4xxx: パーソナライゼーション
        elif "Character" in error_type or "Memory" in error_type:
            return "E4000"
        # E5xxx: システムエラー（デフォルト）
        else:
            return "E5000"

    def log_error(self, error_info: ErrorInfo) -> None:
        """エラーログ記録

        開発規約書 v1.12のログ規約に準拠。
        構造化ログ（extraフィールド）を使用。
        """
        # ログレベル判定（エラーハンドリング指針 v1.20準拠）
        log_message = f"[{error_info.error_code}] {error_info.message}"

        # 構造化ログのextraフィールド
        extra = {
            "error_code": error_info.error_code,
            "error_type": error_info.error_type,
            "error_details": error_info.to_dict(),
        }

        # エラーコードによってログレベルを決定
        if error_info.error_code.startswith("E5"):
            # システムエラーはERROR
            self.logger.error(log_message, extra=extra)
        elif error_info.error_code in ["E2002", "E2003"]:  # Rate limit, Timeout
            # リトライ可能なエラーはWARNING
            self.logger.warning(log_message, extra=extra)
        else:
            # その他はERROR
            self.logger.error(log_message, extra=extra)

    async def _log_error_async(self, error_info: ErrorInfo) -> None:
        """非同期エラーログ記録

        非同期版のログ記録。基本的には同期版と同じ。
        """
        # 非同期I/Oを使う場合はここで実装
        # 現在は同期版を呼び出し
        self.log_error(error_info)

    def _get_user_message(self, error_info: ErrorInfo, character_id: Optional[str] = None) -> str:
        """ユーザー向けメッセージ取得（同期版）"""
        if self.i18n:
            try:
                # I18nManager使用（Phase 3-4）
                return self.i18n.get_error_message(
                    error_info.error_code, character_id=character_id, **(error_info.details or {})
                )
            except Exception as e:
                self.logger.warning(f"Failed to get i18n message: {e}")

        # フォールバック
        return error_info.to_user_message(character_id)

    async def _get_user_message_async(
        self, error_info: ErrorInfo, character_id: Optional[str] = None
    ) -> str:
        """ユーザー向けメッセージ取得（非同期版）"""
        if self.i18n:
            try:
                # I18nManagerが非同期メソッドを持つ場合
                if hasattr(self.i18n, "get_error_message_async"):
                    return await self.i18n.get_error_message_async(
                        error_info.error_code,
                        character_id=character_id,
                        **(error_info.details or {}),
                    )
                else:
                    # 同期版を使用
                    return self.i18n.get_error_message(
                        error_info.error_code,
                        character_id=character_id,
                        **(error_info.details or {}),
                    )
            except Exception as e:
                self.logger.warning(f"Failed to get i18n message: {e}")

        # フォールバック
        return error_info.to_user_message(character_id)

    async def _notify_user_async(self, message: str) -> None:
        """ユーザーへの通知（Phase 3-4）

        実際の通知メカニズムは別途実装。
        Phase 3-4では基本的な実装のみ。
        """
        # TODO: 実際の通知メカニズムを実装
        # 現在は標準出力に出力
        print(f"[エラー] {message}")

    def _add_to_history(self, error_info: ErrorInfo) -> None:
        """エラー履歴に追加"""
        self._error_history.append(error_info)

        # 履歴サイズ制限
        if len(self._error_history) > self._max_history_size:
            self._error_history = self._error_history[-self._max_history_size :]

    # ========================================================================
    # リスナー管理
    # ========================================================================

    def add_error_listener(self, listener: Callable) -> None:
        """エラーリスナー追加

        エラー発生時に通知を受けるリスナーを追加。
        """
        if listener not in self._error_listeners:
            self._error_listeners.append(listener)

    def remove_error_listener(self, listener: Callable) -> None:
        """エラーリスナー削除"""
        if listener in self._error_listeners:
            self._error_listeners.remove(listener)

    def _notify_listeners(self, error_info: ErrorInfo, user_message: str) -> None:
        """リスナーに通知（同期版）"""
        for listener in self._error_listeners:
            try:
                listener(error_info, user_message)
            except Exception as e:
                self.logger.warning(f"Error listener failed: {e}")

    async def _notify_listeners_async(self, error_info: ErrorInfo, user_message: str) -> None:
        """リスナーに通知（非同期版）"""
        tasks = []
        for listener in self._error_listeners:
            if asyncio.iscoroutinefunction(listener):
                tasks.append(listener(error_info, user_message))
            else:
                # 同期リスナーは別スレッドで実行
                loop = asyncio.get_event_loop()
                tasks.append(loop.run_in_executor(None, listener, error_info, user_message))

        if tasks:
            # すべてのリスナーに並行通知
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # エラーをログ
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.warning(f"Error listener {i} failed: {result}")

    # ========================================================================
    # ユーティリティメソッド
    # ========================================================================

    def get_error_history(
        self, limit: Optional[int] = None, error_code_filter: Optional[str] = None
    ) -> List[ErrorInfo]:
        """エラー履歴取得

        Args:
            limit: 取得する履歴の最大数
            error_code_filter: フィルタするエラーコード（前方一致）

        Returns:
            エラー履歴のリスト
        """
        history = self._error_history

        # エラーコードでフィルタ
        if error_code_filter:
            history = [e for e in history if e.error_code.startswith(error_code_filter)]

        # 件数制限
        if limit:
            history = history[-limit:]

        return history

    def clear_error_history(self) -> None:
        """エラー履歴クリア"""
        self._error_history.clear()

    def get_error_statistics(self) -> Dict[str, Any]:
        """エラー統計取得

        エラーコード別の発生回数などの統計情報を返す。
        """
        stats = {
            "total_errors": len(self._error_history),
            "by_error_code": {},
            "by_error_type": {},
            "recent_errors": [],
        }

        # エラーコード別集計
        for error in self._error_history:
            code = error.error_code
            if code not in stats["by_error_code"]:
                stats["by_error_code"][code] = 0
            stats["by_error_code"][code] += 1

            # エラータイプ別集計
            error_type = error.error_type
            if error_type not in stats["by_error_type"]:
                stats["by_error_type"][error_type] = 0
            stats["by_error_type"][error_type] += 1

        # 最近のエラー（最大5件）
        stats["recent_errors"] = [
            {"error_code": e.error_code, "message": e.message, "timestamp": e.timestamp.isoformat()}
            for e in self._error_history[-5:]
        ]

        return stats


# ============================================================================
# グローバルインスタンス
# ============================================================================

# デフォルトのエラーハンドラーインスタンス
# 各コンポーネントから簡単にアクセスできるように
_default_handler: Optional[ErrorHandler] = None


def get_default_error_handler() -> ErrorHandler:
    """デフォルトエラーハンドラー取得

    シングルトンパターンでグローバルなエラーハンドラーを提供。
    """
    global _default_handler
    if _default_handler is None:
        _default_handler = ErrorHandler()
    return _default_handler


def set_default_error_handler(handler: ErrorHandler) -> None:
    """デフォルトエラーハンドラー設定

    I18nManager連携時などに使用。
    """
    global _default_handler
    _default_handler = handler
