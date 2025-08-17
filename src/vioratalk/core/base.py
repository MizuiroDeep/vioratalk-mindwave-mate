"""VioraTalk コンポーネント基底クラス

すべてのVioraTalkコンポーネントが継承する抽象基底クラスを定義。
ライフサイクル管理と状態遷移を提供する。

インターフェース定義書 v1.33準拠
開発規約書 v1.12準拠
"""

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Any, Dict
from datetime import datetime


class ComponentState(Enum):
    """コンポーネントの状態を表すEnum
    
    エンジン初期化仕様書v1.4、非同期処理実装ガイドv1.1準拠
    """
    NOT_INITIALIZED = "not_initialized"  # 初期化前
    INITIALIZING = "initializing"        # 初期化中
    READY = "ready"                      # 使用可能
    ERROR = "error"                      # エラー状態
    SHUTDOWN = "shutdown"                # シャットダウン中
    TERMINATED = "terminated"            # 完全終了


class VioraTalkComponent(ABC):
    """VioraTalkコンポーネントの抽象基底クラス
    
    すべてのコンポーネント（Engine、Manager、Service等）の基底クラス。
    統一されたライフサイクル管理とロギング機能を提供する。
    
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
        
        # 初期化ログ（開発規約書v1.12 セクション6準拠）
        self._logger.debug(
            f"{self.__class__.__name__} インスタンス作成",
            extra={"component": self.__class__.__name__}
        )
    
    @abstractmethod
    async def initialize(self) -> None:
        """非同期初期化処理
        
        コンポーネントの初期化を行う。
        必要なリソースの確保、設定の読み込み、依存関係の解決などを実行。
        
        Raises:
            InitializationError: 初期化に失敗した場合
            
        Note:
            - サブクラスで実装必須
            - 初期化前の状態チェックは基底クラスで行うため不要
            - エラー時は自動的にERROR状態に遷移
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """非同期クリーンアップ処理
        
        コンポーネントのクリーンアップを行う。
        リソースの解放、接続のクローズ、一時ファイルの削除などを実行。
        
        Note:
            - サブクラスで実装必須
            - エラーが発生しても処理を継続すること
            - 最終的にTERMINATED状態に遷移
        """
        pass
    
    def is_available(self) -> bool:
        """コンポーネントが利用可能かを確認
        
        Returns:
            bool: READY状態の場合True、それ以外はFalse
        """
        return self._state == ComponentState.READY
    
    async def safe_initialize(self) -> None:
        """安全な初期化処理のラッパー
        
        状態遷移とエラーハンドリングを含む初期化処理。
        サブクラスはinitialize()を実装し、このメソッドを呼び出す。
        
        Raises:
            RuntimeError: 既に初期化済みまたは初期化中の場合
            InitializationError: 初期化に失敗した場合
        """
        # 状態チェック
        if self._state not in [ComponentState.NOT_INITIALIZED, ComponentState.ERROR]:
            raise RuntimeError(
                f"Cannot initialize component in state: {self._state.value}"
            )
        
        self._state = ComponentState.INITIALIZING
        self._logger.info(
            f"{self.__class__.__name__} 初期化開始",
            extra={"component": self.__class__.__name__}
        )
        
        try:
            # サブクラスの初期化処理を実行
            await self.initialize()
            
            # 成功時の処理
            self._state = ComponentState.READY
            self._initialized_at = datetime.now()
            self._error_info = None
            
            self._logger.info(
                f"{self.__class__.__name__} 初期化完了",
                extra={
                    "component": self.__class__.__name__,
                    "initialized_at": self._initialized_at.isoformat()
                }
            )
            
        except Exception as e:
            # エラー時の処理
            self._state = ComponentState.ERROR
            self._error_info = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
            self._logger.error(
                f"{self.__class__.__name__} 初期化失敗",
                extra={
                    "component": self.__class__.__name__,
                    "error_type": type(e).__name__
                },
                exc_info=True
            )
            raise
    
    async def safe_cleanup(self) -> None:
        """安全なクリーンアップ処理のラッパー
        
        状態遷移とエラーハンドリングを含むクリーンアップ処理。
        エラーが発生しても処理を継続し、最終的にTERMINATED状態にする。
        """
        # 既に終了済みの場合は何もしない
        if self._state == ComponentState.TERMINATED:
            self._logger.debug(
                f"{self.__class__.__name__} 既に終了済み",
                extra={"component": self.__class__.__name__}
            )
            return
        
        self._state = ComponentState.SHUTDOWN
        self._logger.info(
            f"{self.__class__.__name__} クリーンアップ開始",
            extra={"component": self.__class__.__name__}
        )
        
        try:
            # サブクラスのクリーンアップ処理を実行
            await self.cleanup()
            
            self._logger.info(
                f"{self.__class__.__name__} クリーンアップ完了",
                extra={"component": self.__class__.__name__}
            )
            
        except Exception as e:
            # エラーが発生してもログを記録して続行
            self._logger.error(
                f"{self.__class__.__name__} クリーンアップ中にエラー発生",
                extra={
                    "component": self.__class__.__name__,
                    "error_type": type(e).__name__
                },
                exc_info=True
            )
        
        finally:
            # 最終的に必ずTERMINATED状態にする
            self._state = ComponentState.TERMINATED
            self._logger.debug(
                f"{self.__class__.__name__} 終了",
                extra={"component": self.__class__.__name__}
            )
    
    def get_state(self) -> ComponentState:
        """現在の状態を取得
        
        Returns:
            ComponentState: 現在の状態
        """
        return self._state
    
    def get_error_info(self) -> Optional[Dict[str, Any]]:
        """エラー情報を取得
        
        Returns:
            Optional[Dict[str, Any]]: エラー情報（エラー状態でない場合はNone）
        """
        return self._error_info
    
    def __str__(self) -> str:
        """文字列表現"""
        return f"{self.__class__.__name__}(state={self._state.value})"
    
    def __repr__(self) -> str:
        """詳細な文字列表現"""
        return (
            f"{self.__class__.__name__}("
            f"state={self._state.value}, "
            f"initialized_at={self._initialized_at}, "
            f"has_error={self._error_info is not None})"
        )