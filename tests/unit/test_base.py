"""VioraTalkComponent基底クラスのテスト

ComponentStateの状態遷移とライフサイクルメソッドの動作を検証。
テスト戦略ガイドライン v1.7準拠
テスト実装ガイド v1.3準拠
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch
import logging

from vioratalk.core.base import VioraTalkComponent, ComponentState
from vioratalk.core.exceptions import InitializationError


# ============================================================================
# テスト用の具象クラス
# ============================================================================

class ConcreteTestComponent(VioraTalkComponent):
    """テスト用の具象コンポーネント（Test接頭辞を避けてpytestの誤認識を防ぐ）"""
    
    def __init__(self, should_fail_init: bool = False, should_fail_cleanup: bool = False):
        super().__init__()
        self.should_fail_init = should_fail_init
        self.should_fail_cleanup = should_fail_cleanup
        self.initialize_called = False
        self.cleanup_called = False
    
    async def initialize(self) -> None:
        """初期化処理（テスト用）"""
        self.initialize_called = True
        if self.should_fail_init:
            raise InitializationError("Test initialization error")
        await asyncio.sleep(0.01)  # 非同期処理をシミュレート
    
    async def cleanup(self) -> None:
        """クリーンアップ処理（テスト用）"""
        self.cleanup_called = True
        if self.should_fail_cleanup:
            raise RuntimeError("Test cleanup error")
        await asyncio.sleep(0.01)  # 非同期処理をシミュレート


# ============================================================================
# ComponentStateのテスト
# ============================================================================

@pytest.mark.unit
@pytest.mark.phase(1)
class TestComponentState:
    """ComponentState Enumのテスト"""
    
    def test_state_values(self):
        """状態値が正しく定義されているか"""
        assert ComponentState.NOT_INITIALIZED.value == "not_initialized"
        assert ComponentState.INITIALIZING.value == "initializing"
        assert ComponentState.READY.value == "ready"
        assert ComponentState.ERROR.value == "error"
        assert ComponentState.SHUTDOWN.value == "shutdown"
        assert ComponentState.TERMINATED.value == "terminated"
    
    def test_state_count(self):
        """状態の数が6つであることを確認"""
        assert len(ComponentState) == 6


# ============================================================================
# VioraTalkComponentのテスト
# ============================================================================

@pytest.mark.unit
@pytest.mark.phase(1)
class TestVioraTalkComponent:
    """VioraTalkComponent基底クラスのテスト"""
    
    @pytest.fixture
    def component(self):
        """テスト用コンポーネントのフィクスチャ"""
        return ConcreteTestComponent()
    
    @pytest.fixture
    def failing_init_component(self):
        """初期化に失敗するコンポーネント"""
        return ConcreteTestComponent(should_fail_init=True)
    
    @pytest.fixture
    def failing_cleanup_component(self):
        """クリーンアップに失敗するコンポーネント"""
        return ConcreteTestComponent(should_fail_cleanup=True)
    
    # ------------------------------------------------------------------------
    # 初期状態のテスト
    # ------------------------------------------------------------------------
    
    def test_initial_state(self, component):
        """初期状態が正しく設定されているか"""
        assert component.get_state() == ComponentState.NOT_INITIALIZED
        assert component.is_available() is False
        assert component.get_error_info() is None
        assert component._initialized_at is None
    
    def test_logger_initialization(self, component):
        """ロガーが正しく初期化されているか"""
        assert component._logger is not None
        assert isinstance(component._logger, logging.Logger)
        # モジュール名がロガー名に含まれているか
        assert "test_base" in component._logger.name
    
    # ------------------------------------------------------------------------
    # 初期化処理のテスト
    # ------------------------------------------------------------------------
    
    @pytest.mark.asyncio
    async def test_successful_initialization(self, component):
        """正常な初期化処理"""
        # 初期化実行
        await component.safe_initialize()
        
        # 状態確認
        assert component.get_state() == ComponentState.READY
        assert component.is_available() is True
        assert component.initialize_called is True
        assert component._initialized_at is not None
        assert isinstance(component._initialized_at, datetime)
        assert component.get_error_info() is None
    
    @pytest.mark.asyncio
    async def test_failed_initialization(self, failing_init_component):
        """初期化失敗時の処理"""
        component = failing_init_component
        
        # 初期化が例外を発生させることを確認
        with pytest.raises(InitializationError) as exc_info:
            await component.safe_initialize()
        
        # 状態確認
        assert component.get_state() == ComponentState.ERROR
        assert component.is_available() is False
        assert component.initialize_called is True
        
        # エラー情報の確認
        error_info = component.get_error_info()
        assert error_info is not None
        assert error_info["error_type"] == "InitializationError"
        assert "timestamp" in error_info
    
    @pytest.mark.asyncio
    async def test_double_initialization(self, component):
        """二重初期化の防止"""
        # 最初の初期化
        await component.safe_initialize()
        assert component.get_state() == ComponentState.READY
        
        # 二回目の初期化は例外を発生させる
        with pytest.raises(RuntimeError) as exc_info:
            await component.safe_initialize()
        assert "Cannot initialize component in state: ready" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_reinitialize_after_error(self, component):
        """エラー後の再初期化"""
        # 最初にエラー状態にする
        component._state = ComponentState.ERROR
        component._error_info = {"test": "error"}
        
        # 再初期化
        await component.safe_initialize()
        
        # 状態確認
        assert component.get_state() == ComponentState.READY
        assert component.is_available() is True
        assert component.get_error_info() is None
    
    # ------------------------------------------------------------------------
    # クリーンアップ処理のテスト
    # ------------------------------------------------------------------------
    
    @pytest.mark.asyncio
    async def test_successful_cleanup(self, component):
        """正常なクリーンアップ処理"""
        # 初期化してから
        await component.safe_initialize()
        assert component.get_state() == ComponentState.READY
        
        # クリーンアップ実行
        await component.safe_cleanup()
        
        # 状態確認
        assert component.get_state() == ComponentState.TERMINATED
        assert component.is_available() is False
        assert component.cleanup_called is True
    
    @pytest.mark.asyncio
    async def test_cleanup_from_not_initialized(self, component):
        """未初期化状態からのクリーンアップ"""
        assert component.get_state() == ComponentState.NOT_INITIALIZED
        
        # クリーンアップ実行
        await component.safe_cleanup()
        
        # 状態確認
        assert component.get_state() == ComponentState.TERMINATED
        assert component.cleanup_called is True
    
    @pytest.mark.asyncio
    async def test_cleanup_with_error(self, failing_cleanup_component):
        """クリーンアップ中のエラー処理"""
        component = failing_cleanup_component
        
        # 初期化してから
        await component.safe_initialize()
        
        # クリーンアップ実行（エラーが発生するが処理は継続）
        await component.safe_cleanup()
        
        # エラーが発生してもTERMINATED状態になる
        assert component.get_state() == ComponentState.TERMINATED
        assert component.cleanup_called is True
    
    @pytest.mark.asyncio
    async def test_double_cleanup(self, component):
        """二重クリーンアップの処理"""
        # 最初のクリーンアップ
        await component.safe_cleanup()
        assert component.get_state() == ComponentState.TERMINATED
        assert component.cleanup_called is True
        
        # クリーンアップ呼び出し回数をリセット
        component.cleanup_called = False
        
        # 二回目のクリーンアップ（何もしない）
        await component.safe_cleanup()
        assert component.get_state() == ComponentState.TERMINATED
        assert component.cleanup_called is False  # 二回目は呼ばれない
    
    # ------------------------------------------------------------------------
    # 状態遷移のテスト
    # ------------------------------------------------------------------------
    
    @pytest.mark.asyncio
    async def test_state_transitions(self, component):
        """状態遷移の完全なフロー"""
        # NOT_INITIALIZED -> INITIALIZING -> READY
        assert component.get_state() == ComponentState.NOT_INITIALIZED
        
        init_task = asyncio.create_task(component.safe_initialize())
        await asyncio.sleep(0.001)  # 少し待つ
        # 初期化中の状態（タイミングによってはREADYになっている可能性もある）
        assert component.get_state() in [ComponentState.INITIALIZING, ComponentState.READY]
        
        await init_task
        assert component.get_state() == ComponentState.READY
        
        # READY -> SHUTDOWN -> TERMINATED
        cleanup_task = asyncio.create_task(component.safe_cleanup())
        await asyncio.sleep(0.001)  # 少し待つ
        # クリーンアップ中の状態
        assert component.get_state() in [ComponentState.SHUTDOWN, ComponentState.TERMINATED]
        
        await cleanup_task
        assert component.get_state() == ComponentState.TERMINATED
    
    # ------------------------------------------------------------------------
    # ユーティリティメソッドのテスト
    # ------------------------------------------------------------------------
    
    def test_string_representation(self, component):
        """文字列表現のテスト"""
        str_repr = str(component)
        assert "ConcreteTestComponent" in str_repr
        assert "not_initialized" in str_repr
        
        repr_str = repr(component)
        assert "ConcreteTestComponent" in repr_str
        assert "state=not_initialized" in repr_str
        assert "initialized_at=None" in repr_str
        assert "has_error=False" in repr_str
    
    @pytest.mark.asyncio
    async def test_string_after_initialization(self, component):
        """初期化後の文字列表現"""
        await component.safe_initialize()
        
        str_repr = str(component)
        assert "ConcreteTestComponent" in str_repr
        assert "ready" in str_repr
        
        repr_str = repr(component)
        assert "state=ready" in repr_str
        assert "initialized_at=" in repr_str
        assert "None" not in repr_str.split("initialized_at=")[1].split(",")[0]
    
    # ------------------------------------------------------------------------
    # ログ出力のテスト
    # ------------------------------------------------------------------------
    
    @pytest.mark.asyncio
    async def test_logging_during_initialization(self, component, caplog):
        """初期化時のログ出力"""
        with caplog.at_level(logging.DEBUG):
            await component.safe_initialize()
        
        # ログメッセージの確認
        log_messages = [record.message for record in caplog.records]
        assert any("初期化開始" in msg for msg in log_messages)
        assert any("初期化完了" in msg for msg in log_messages)
    
    @pytest.mark.asyncio
    async def test_logging_during_error(self, failing_init_component, caplog):
        """エラー時のログ出力"""
        with caplog.at_level(logging.ERROR):
            try:
                await failing_init_component.safe_initialize()
            except InitializationError:
                pass
        
        # エラーログの確認
        error_logs = [r for r in caplog.records if r.levelname == "ERROR"]
        assert len(error_logs) > 0
        assert any("初期化失敗" in record.message for record in error_logs)