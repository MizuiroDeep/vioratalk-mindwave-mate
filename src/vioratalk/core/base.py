"""VioraTalk コンポーネント基底クラス

すべてのVioraTalkコンポーネントが継承する抽象基底クラスを定義。
ライフサイクル管理と状態遷移を提供する。

インターフェース定義書 v1.34準拠
開発規約書 v1.12準拠
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class ComponentState(Enum):
    """コンポーネントの状態を表すEnum

    すべてのVioraTalkComponentはこの状態のいずれかを持つ。
    状態遷移は以下の順序で行われる：
    NOT_INITIALIZED → INITIALIZING → READY → RUNNING
                    ↓                ↓         ↓
                    → ERROR ←--------┴---------┘
                    ↓
                    TERMINATING → TERMINATED

    インターフェース定義書 v1.34準拠
    """

    NOT_INITIALIZED = "not_initialized"  # 未初期化（作成直後）
    INITIALIZING = "initializing"  # 初期化中
    READY = "ready"  # 準備完了（旧INITIALIZED）
    RUNNING = "running"  # 実行中（アクティブ処理中）
    ERROR = "error"  # エラー状態
    TERMINATING = "terminating"  # 終了処理中
    TERMINATED = "terminated"  # 終了済み

    @classmethod
    def is_operational(cls, state: "ComponentState") -> bool:
        """動作可能な状態か判定

        Args:
            state: 判定する状態

        Returns:
            bool: READYまたはRUNNING状態の場合True
        """
        return state in [cls.READY, cls.RUNNING]

    def can_transition_to(self, target: "ComponentState") -> bool:
        """指定の状態に遷移可能か判定

        Args:
            target: 遷移先の状態

        Returns:
            bool: 遷移可能な場合True
        """
        valid_transitions = {
            ComponentState.NOT_INITIALIZED: [ComponentState.INITIALIZING],
            ComponentState.INITIALIZING: [ComponentState.READY, ComponentState.ERROR],
            ComponentState.READY: [
                ComponentState.RUNNING,
                ComponentState.ERROR,
                ComponentState.TERMINATING,
            ],
            ComponentState.RUNNING: [
                ComponentState.READY,
                ComponentState.ERROR,
                ComponentState.TERMINATING,
            ],
            ComponentState.ERROR: [ComponentState.INITIALIZING, ComponentState.TERMINATING],
            ComponentState.TERMINATING: [ComponentState.TERMINATED],
            ComponentState.TERMINATED: [],
        }
        return target in valid_transitions.get(self, [])


class VioraTalkComponent(ABC):
    """VioraTalkコンポーネントの抽象基底クラス

    すべてのコンポーネント（Engine、Manager、Service等）の基底クラス。
    統一されたライフサイクル管理とロギング機能を提供する。

    インターフェース定義書 v1.34準拠：
    - ComponentStateで定義された状態を持つ
    - get_status()メソッドで現在の状態を取得
    - is_available()メソッドで利用可能状態を確認

    Attributes:
        _state: コンポーネントの現在の状態
        _logger: コンポーネント専用のロガー
        _initialized_at: 初期化完了時刻
        _error_info: エラー情報（エラー状態の場合）
    """

    def __init__(self):
        """基底クラスの初期化

        Note:
            サブクラスは必ずsuper().__init__()を呼び出すこと
        """
        self._state: ComponentState = ComponentState.NOT_INITIALIZED
        self._logger: logging.Logger = logging.getLogger(self.__class__.__module__)
        self._initialized_at: Optional[datetime] = None
        self._error_info: Optional[Dict[str, Any]] = None
        self._error: Optional[Exception] = None

        # 初期化ログ（開発規約書v1.12 セクション6準拠）
        self._logger.debug(
            f"{self.__class__.__name__} インスタンス作成", extra={"component": self.__class__.__name__}
        )

    @abstractmethod
    async def initialize(self) -> None:
        """非同期初期化処理

        初期化が必要なケース:
        - ファイル/モデルのロード
        - ネットワーク接続
        - ハードウェアアクセス
        - 外部プロセス起動
        - 認証処理

        状態遷移:
        - 成功時: NOT_INITIALIZED → INITIALIZING → READY
        - 失敗時: NOT_INITIALIZED → INITIALIZING → ERROR

        Raises:
            InitializationError: 初期化に失敗した場合

        Note:
            - 初期化は冪等でなければならない
            - エラー時は自動的にERROR状態に遷移
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """リソースのクリーンアップ

        状態遷移:
        - 正常時: READY/RUNNING/ERROR → TERMINATING → TERMINATED

        Note:
            - クリーンアップは常に成功すべき
            - エラーが発生してもログ記録のみで継続
        """
        pass

    async def safe_initialize(self) -> None:
        """安全な初期化処理（状態チェック、エラーハンドリング付き）

        状態遷移を管理し、エラーハンドリングを行う初期化ラッパー。

        Raises:
            RuntimeError: 不適切な状態での初期化の場合
            InitializationError: 初期化処理が失敗した場合
        """
        # 状態チェック
        if self._state not in [ComponentState.NOT_INITIALIZED, ComponentState.ERROR]:
            raise RuntimeError(f"Cannot initialize component in state: {self._state.value}")

        # エラー情報をクリア
        if self._state == ComponentState.ERROR:
            self._error_info = None
            self._error = None

        # 初期化開始
        self._set_state(ComponentState.INITIALIZING)
        self._logger.info(f"{self.__class__.__name__} 初期化開始")

        try:
            # 実際の初期化処理を呼び出し
            await self.initialize()

            # 成功時
            self._set_state(ComponentState.READY)
            if not self._initialized_at:
                self._initialized_at = datetime.now()
            self._logger.info(f"{self.__class__.__name__} 初期化完了")

        except Exception as e:
            # エラー時
            self._handle_error(
                e,
                {
                    "error_type": type(e).__name__,
                    "message": str(e),
                    "timestamp": datetime.now().isoformat(),
                },
            )
            self._logger.error(f"{self.__class__.__name__} 初期化失敗: {e}", exc_info=True)
            raise

    async def safe_cleanup(self) -> None:
        """安全なクリーンアップ処理

        状態遷移を管理し、エラーハンドリングを行うクリーンアップラッパー。
        エラーが発生しても処理を継続する。
        """
        # すでに終了済みの場合は何もしない
        if self._state == ComponentState.TERMINATED:
            return

        # クリーンアップ開始
        self._set_state(ComponentState.TERMINATING)
        self._logger.info(f"{self.__class__.__name__} クリーンアップ開始")

        try:
            # 実際のクリーンアップ処理を呼び出し
            await self.cleanup()
            self._logger.info(f"{self.__class__.__name__} クリーンアップ完了")

        except Exception as e:
            # エラーが発生してもログ記録のみで継続
            self._logger.error(f"{self.__class__.__name__} クリーンアップ中にエラー: {e}", exc_info=True)

        finally:
            # 最終的に必ずTERMINATED状態にする
            self._set_state(ComponentState.TERMINATED)

    def get_state(self) -> ComponentState:
        """現在の状態を取得

        Returns:
            ComponentState: 現在の状態
        """
        return self._state

    def get_error_info(self) -> Optional[Dict[str, Any]]:
        """エラー情報を取得

        Returns:
            Optional[Dict[str, Any]]: エラー情報（エラーがない場合はNone）
        """
        return self._error_info

    def is_available(self) -> bool:
        """利用可能状態の確認

        Returns:
            bool: READYまたはRUNNING状態の場合True
        """
        return ComponentState.is_operational(self._state)

    @property
    def state(self) -> ComponentState:
        """現在の状態（プロパティ）

        Returns:
            ComponentState: 現在の状態
        """
        return self._state

    def get_status(self) -> Dict[str, Any]:
        """コンポーネントの詳細ステータス取得

        Returns:
            Dict[str, Any]: 状態情報を含む辞書
            必須フィールド:
            - state: ComponentState
            - is_available: bool (is_available()の結果)
            - error: Optional[str]
            - last_used: Optional[datetime]

        Note:
            サブクラスで拡張可能（super().get_status()を呼び出し、
            追加情報を辞書に追加する）
        """
        status = {
            "component": self.__class__.__name__,
            "state": self._state.value,
            "is_available": self.is_available(),
            "initialized_at": self._initialized_at.isoformat() if self._initialized_at else None,
            "error": None,
            "last_used": None,  # サブクラスで実装
        }

        # エラー情報の追加
        if self._error_info:
            status["error"] = self._error_info
        elif self._error:
            status["error"] = str(self._error)

        return status

    def _set_state(self, new_state: ComponentState) -> None:
        """内部用：状態を変更

        Args:
            new_state: 新しい状態

        Note:
            状態遷移の妥当性をチェックし、ログを記録する
        """
        old_state = self._state

        # 状態遷移の妥当性チェック
        if not old_state.can_transition_to(new_state):
            self._logger.warning(
                f"非推奨の状態遷移: {old_state.value} → {new_state.value}",
                extra={
                    "component": self.__class__.__name__,
                    "old_state": old_state.value,
                    "new_state": new_state.value,
                },
            )

        self._state = new_state

        # 状態遷移のログ記録
        self._logger.info(
            f"状態遷移: {old_state.value} → {new_state.value}",
            extra={
                "component": self.__class__.__name__,
                "old_state": old_state.value,
                "new_state": new_state.value,
            },
        )

        # READY状態になったら初期化時刻を記録
        if new_state == ComponentState.READY and not self._initialized_at:
            self._initialized_at = datetime.now()

    def _handle_error(self, error: Exception, error_info: Optional[Dict[str, Any]] = None) -> None:
        """内部用：エラー処理

        Args:
            error: 発生したエラー
            error_info: 追加のエラー情報
        """
        self._error = error
        self._error_info = error_info or {"message": str(error)}
        self._set_state(ComponentState.ERROR)

        # エラーログ記録
        self._logger.error(
            f"エラー発生: {error}",
            extra={
                "component": self.__class__.__name__,
                "error_type": type(error).__name__,
                "error_info": self._error_info,
            },
            exc_info=True,
        )

    def __str__(self) -> str:
        """文字列表現

        Returns:
            str: コンポーネント名と状態
        """
        return f"{self.__class__.__name__}({self._state.value})"

    def __repr__(self) -> str:
        """開発用詳細文字列表現

        Returns:
            str: デバッグ用の詳細情報
        """
        return (
            f"<{self.__class__.__name__} "
            f"state={self._state.value}, "
            f"initialized_at={self._initialized_at.isoformat() if self._initialized_at else 'None'}, "
            f"has_error={self._error_info is not None}>"
        )
