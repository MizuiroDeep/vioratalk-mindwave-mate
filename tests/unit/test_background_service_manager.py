"""BackgroundServiceManagerの単体テスト

バックグラウンドサービス管理機能の動作を検証。
Phase 1の最小実装として、遅延起動原則の遵守を重点的にテスト。

テスト戦略ガイドライン v1.7準拠
テスト実装ガイド v1.3準拠
バックグラウンドサービス管理仕様書 v1.3準拠
開発規約書 v1.12準拠
"""

import asyncio
from datetime import datetime
from typing import Any, Dict
from unittest.mock import patch

import pytest

from vioratalk.core.base import ComponentState
from vioratalk.core.exceptions import InitializationError
from vioratalk.services.background_service_manager import (
    BackgroundServiceManager,
    BaseBackgroundService,
    ServiceInfo,
    ServiceStatus,
)

# ============================================================================
# テスト用のモックサービス
# ============================================================================


class MockBackgroundService(BaseBackgroundService):
    """テスト用のモックバックグラウンドサービス"""

    def __init__(self, service_name: str, config: Dict[str, Any]):
        super().__init__(service_name, config)
        self.install_called = False
        self.start_called = False
        self.stop_called = False
        self.health_check_called = False
        self.should_fail = config.get("should_fail", False)

    async def install(self) -> bool:
        """インストール（モック）"""
        self.install_called = True
        if self.should_fail:
            return False
        self._status = ServiceStatus.STOPPED
        return True

    async def start(self) -> bool:
        """起動（モック）"""
        self.start_called = True
        if self.should_fail:
            self._status = ServiceStatus.ERROR
            return False
        self._status = ServiceStatus.RUNNING
        self._info.status = ServiceStatus.RUNNING
        return True

    async def stop(self) -> bool:
        """停止（モック）"""
        self.stop_called = True
        self._status = ServiceStatus.STOPPED
        self._info.status = ServiceStatus.STOPPED
        return True

    async def health_check(self) -> bool:
        """ヘルスチェック（モック）"""
        self.health_check_called = True
        return not self.should_fail


# ============================================================================
# ServiceStatus Enumのテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(1)
class TestServiceStatus:
    """ServiceStatus Enumのテスト"""

    def test_enum_values(self):
        """Enum値の確認"""
        assert ServiceStatus.NOT_INSTALLED.value == "not_installed"
        assert ServiceStatus.STOPPED.value == "stopped"
        assert ServiceStatus.STARTING.value == "starting"
        assert ServiceStatus.RUNNING.value == "running"
        assert ServiceStatus.STOPPING.value == "stopping"
        assert ServiceStatus.ERROR.value == "error"
        assert ServiceStatus.DEGRADED.value == "degraded"

    def test_enum_count(self):
        """Enumの個数確認"""
        assert len(ServiceStatus) == 7


# ============================================================================
# ServiceInfoクラスのテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(1)
class TestServiceInfo:
    """ServiceInfoクラスのテスト"""

    def test_initialization(self):
        """基本的な初期化"""
        info = ServiceInfo(name="test_service", status=ServiceStatus.RUNNING)

        assert info.name == "test_service"
        assert info.status == ServiceStatus.RUNNING
        assert info.version is None
        assert info.port is None
        assert info.process_id is None
        assert info.last_health_check is None
        assert info.error_message is None

    def test_initialization_with_all_fields(self):
        """全フィールド指定での初期化"""
        now = datetime.now()
        info = ServiceInfo(
            name="test_service",
            status=ServiceStatus.ERROR,
            version="1.0.0",
            port=8080,
            process_id=12345,
            last_health_check=now,
            error_message="Test error",
        )

        assert info.name == "test_service"
        assert info.status == ServiceStatus.ERROR
        assert info.version == "1.0.0"
        assert info.port == 8080
        assert info.process_id == 12345
        assert info.last_health_check == now
        assert info.error_message == "Test error"


# ============================================================================
# BackgroundServiceManagerのテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(1)
class TestBackgroundServiceManager:
    """BackgroundServiceManagerクラスのテスト"""

    @pytest.fixture
    def manager(self):
        """BackgroundServiceManagerのフィクスチャ"""
        return BackgroundServiceManager()

    def test_initialization(self):
        """初期化のテスト（Phase 1最小実装）"""
        manager = BackgroundServiceManager()

        assert manager._state == ComponentState.NOT_INITIALIZED
        assert manager._services == {}
        assert manager._health_tasks == {}
        assert hasattr(manager, "_logger")

    @pytest.mark.asyncio
    async def test_initialize(self):
        """initialize()メソッドのテスト"""
        manager = BackgroundServiceManager()

        await manager.initialize()

        assert manager._state == ComponentState.READY

    @pytest.mark.asyncio
    async def test_initialize_failure(self):
        """初期化失敗のテスト"""
        manager = BackgroundServiceManager()

        with patch.object(manager, "initialize_services", side_effect=RuntimeError("Test error")):
            with pytest.raises(InitializationError) as exc_info:
                await manager.initialize()

            assert "E7000" in str(exc_info.value)
            assert manager._state == ComponentState.ERROR

    @pytest.mark.asyncio
    async def test_no_automatic_start_on_initialize(self):
        """初期化時にサービスが起動しないことを確認（遅延起動原則）"""
        manager = BackgroundServiceManager()
        mock_service = MockBackgroundService("test", {})
        manager.register_service(mock_service)

        await manager.initialize()

        # 仕様書v1.3: 初期化時は絶対にサービスを起動しない
        assert mock_service.start_called is False
        assert manager._state == ComponentState.READY

    def test_no_start_all_services_method(self):
        """start_all_services()メソッドが存在しないことを確認"""
        manager = BackgroundServiceManager()

        # 仕様書v1.3: start_all_services()は存在してはならない
        assert not hasattr(manager, "start_all_services")

    @pytest.mark.asyncio
    async def test_lazy_loading_with_start_service_if_needed(self):
        """start_service_if_needed()での遅延起動を確認"""
        manager = BackgroundServiceManager()
        mock_service = MockBackgroundService("ollama", {})
        manager.register_service(mock_service)

        # 初期化
        await manager.initialize()
        assert mock_service.start_called is False

        # 必要時にのみ起動
        status = await manager.start_service_if_needed("ollama")

        # Phase 1ではスタブ実装なので実際には起動しない
        assert status == ServiceStatus.RUNNING

    def test_register_service(self):
        """サービス登録のテスト"""
        manager = BackgroundServiceManager()
        mock_service = MockBackgroundService("test", {})

        manager.register_service(mock_service)

        assert "test" in manager._services
        assert manager._services["test"] == mock_service

    def test_register_duplicate_service(self):
        """重複サービス登録のテスト"""
        manager = BackgroundServiceManager()
        service1 = MockBackgroundService("test", {})
        service2 = MockBackgroundService("test", {})

        # 最初の登録は成功
        manager.register_service(service1)
        assert manager._services["test"] == service1

        # 重複登録は例外を発生させる
        with pytest.raises(ValueError) as exc_info:
            manager.register_service(service2)

        assert "already registered" in str(exc_info.value)
        # 最初のサービスが保持されている
        assert manager._services["test"] == service1

    @pytest.mark.asyncio
    async def test_start_service_if_needed_stub(self):
        """start_service_if_needed()のスタブ動作確認"""
        manager = BackgroundServiceManager()

        # Phase 1: 登録されていないサービスでも成功
        status = await manager.start_service_if_needed("unregistered")
        assert status == ServiceStatus.RUNNING

    @pytest.mark.asyncio
    async def test_start_service_if_needed_unknown_service(self):
        """未登録サービスのstart_service_if_needed()"""
        manager = BackgroundServiceManager()

        # Phase 1では未登録でもRUNNINGを返す（スタブ）
        status = await manager.start_service_if_needed("unknown")
        assert status == ServiceStatus.RUNNING

    @pytest.mark.asyncio
    async def test_start_service_if_needed_with_mock(self):
        """モックサービスでのstart_service_if_needed()"""
        manager = BackgroundServiceManager()
        mock_service = MockBackgroundService("test", {})
        manager.register_service(mock_service)

        status = await manager.start_service_if_needed("test")

        # Phase 1ではスタブなので実際には起動しない
        assert status == ServiceStatus.RUNNING

    @pytest.mark.asyncio
    async def test_stop_service_stub(self):
        """stop_service()のスタブ動作確認"""
        manager = BackgroundServiceManager()

        # 未登録サービスの停止
        success = await manager.stop_service("unregistered")
        assert success is False

    @pytest.mark.asyncio
    async def test_stop_service_with_mock(self):
        """モックサービスでのstop_service()"""
        manager = BackgroundServiceManager()
        mock_service = MockBackgroundService("test", {})
        manager.register_service(mock_service)

        success = await manager.stop_service("test")

        # Phase 1ではスタブ実装
        assert success is True

    def test_get_service_status_not_registered(self):
        """未登録サービスの状態取得（同期メソッド）"""
        manager = BackgroundServiceManager()

        # get_service_statusは同期メソッド
        status = manager.get_service_status("unknown_service")

        assert status == ServiceStatus.NOT_INSTALLED

    def test_get_service_status_registered(self):
        """登録済みサービスの状態取得（同期メソッド）"""
        manager = BackgroundServiceManager()
        mock_service = MockBackgroundService("test_service", {})
        manager.register_service(mock_service)

        # get_service_statusは同期メソッド
        status = manager.get_service_status("test_service")

        assert status == ServiceStatus.STOPPED

    @pytest.mark.asyncio
    async def test_cleanup(self):
        """cleanup()メソッドのテスト"""
        manager = BackgroundServiceManager()
        mock_service = MockBackgroundService("test", {})
        manager.register_service(mock_service)

        await manager.initialize()
        await manager.cleanup()

        # Phase 1: cleanup後はTERMINATED状態
        assert manager._state == ComponentState.TERMINATED


# ============================================================================
# Phase 1最小実装の確認
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(1)
class TestPhase1MinimalImplementation:
    """Phase 1最小実装の動作確認"""

    @pytest.mark.asyncio
    async def test_stub_behavior(self):
        """Phase 1スタブ実装の動作確認"""
        manager = BackgroundServiceManager()

        # 初期化
        await manager.initialize()

        # サービス起動（スタブ）
        status = await manager.start_service_if_needed("any_service")
        assert status == ServiceStatus.RUNNING

        # サービス停止（スタブ）- 未登録はFalse
        success = await manager.stop_service("any_service")
        assert success is False  # 仕様書準拠

        # クリーンアップ
        await manager.cleanup()
        assert manager._state == ComponentState.TERMINATED

    def test_no_actual_process_spawn(self):
        """実際のプロセス起動が行われないことを確認"""
        manager = BackgroundServiceManager()

        # Phase 1では実際のOllama/AivisSpeechプロセスは起動しない
        with patch("subprocess.Popen") as mock_popen:
            asyncio.run(manager.start_service_if_needed("ollama"))
            mock_popen.assert_not_called()


# ============================================================================
# BaseBackgroundServiceのテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(1)
class TestBaseBackgroundService:
    """BaseBackgroundService抽象クラスのテスト"""

    def test_initialization(self):
        """基底クラスの初期化"""
        service = MockBackgroundService("test", {"key": "value"})

        assert service.service_name == "test"
        assert service.config == {"key": "value"}
        assert service._status == ServiceStatus.STOPPED
        assert isinstance(service._info, ServiceInfo)
        assert service._info.name == "test"

    def test_get_info(self):
        """get_info()メソッド"""
        service = MockBackgroundService("test", {})

        info = service.get_info()

        assert isinstance(info, ServiceInfo)
        assert info.name == "test"
        assert info.status == ServiceStatus.STOPPED

    @pytest.mark.asyncio
    async def test_abstract_methods_implementation(self):
        """抽象メソッドが実装されていることを確認"""
        service = MockBackgroundService("test", {})

        # すべての抽象メソッドが実装されている
        assert await service.install() is True
        assert await service.start() is True
        assert await service.stop() is True
        assert await service.health_check() is True

        # フラグの確認
        assert service.install_called is True
        assert service.start_called is True
        assert service.stop_called is True
        assert service.health_check_called is True


# ============================================================================
# エラーハンドリングのテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(1)
class TestErrorHandling:
    """エラーハンドリングのテスト"""

    @pytest.mark.asyncio
    async def test_initialization_error_code(self):
        """初期化エラーのエラーコード"""
        manager = BackgroundServiceManager()

        with patch.object(manager, "initialize_services", side_effect=RuntimeError("Test")):
            with pytest.raises(InitializationError) as exc_info:
                await manager.initialize()

            assert exc_info.value.error_code == "E7000"

    @pytest.mark.asyncio
    async def test_service_error_handling(self):
        """サービスエラーの処理"""
        manager = BackgroundServiceManager()

        # 失敗するサービスを登録
        failing_service = MockBackgroundService("failing", {"should_fail": True})
        manager.register_service(failing_service)

        # Phase 1ではスタブなので実際にはエラーにならない
        status = await manager.start_service_if_needed("failing")
        assert status == ServiceStatus.RUNNING  # スタブは常に成功
