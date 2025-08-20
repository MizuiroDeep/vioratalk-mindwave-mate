"""
バックグラウンドサービス管理モジュール

外部サービス（Ollama、AivisSpeech等）の起動・停止・監視を統括管理する。
遅延起動（Lazy Loading）により、必要時まで絶対にサービスを起動しない。

関連ドキュメント:
    - バックグラウンドサービス管理仕様書 v1.3
    - エンジン初期化仕様書 v1.4
    - エラーハンドリング指針 v1.20（E7xxx系エラー）
    - インターフェース定義書 v1.34
"""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from vioratalk.core.base import ComponentState, VioraTalkComponent
from vioratalk.core.exceptions import BackgroundServiceError, InitializationError
from vioratalk.utils.logger_manager import LoggerManager

logger = LoggerManager.get_logger(__name__)


class ServiceStatus(Enum):
    """
    サービスの状態を表すEnum

    バックグラウンドサービスの詳細な状態を表現し、
    適切な起動・停止・フォールバック処理を可能にする。
    """

    NOT_INSTALLED = "not_installed"  # 未インストール
    STOPPED = "stopped"  # 停止中
    STARTING = "starting"  # 起動中
    RUNNING = "running"  # 実行中
    STOPPING = "stopping"  # 停止処理中
    ERROR = "error"  # エラー状態
    DEGRADED = "degraded"  # 機能制限状態


class ServiceInfo:
    """
    サービスの情報を保持するクラス

    Attributes:
        name: サービス名
        status: 現在の状態
        version: インストールされているバージョン
        port: 使用ポート番号
        process_id: プロセスID
        last_health_check: 最後のヘルスチェック時刻
        error_message: エラーメッセージ（エラー時のみ）
    """

    def __init__(
        self,
        name: str,
        status: ServiceStatus,
        version: Optional[str] = None,
        port: Optional[int] = None,
        process_id: Optional[int] = None,
        last_health_check: Optional[datetime] = None,
        error_message: Optional[str] = None,
    ):
        self.name = name
        self.status = status
        self.version = version
        self.port = port
        self.process_id = process_id
        self.last_health_check = last_health_check
        self.error_message = error_message


class BaseBackgroundService(ABC):
    """
    バックグラウンドサービスの基底クラス

    各サービス（Ollama、AivisSpeech等）はこのクラスを継承して実装する。
    Phase 1では抽象クラスとして定義のみ。
    """

    def __init__(self, service_name: str, config: Dict[str, Any]):
        """
        バックグラウンドサービスの初期化

        Args:
            service_name: サービス名
            config: サービス固有の設定
        """
        self.service_name = service_name
        self.config = config
        self._status = ServiceStatus.STOPPED
        self._info = ServiceInfo(name=service_name, status=self._status)

    def get_info(self) -> ServiceInfo:
        """サービス情報を取得"""
        return self._info

    @abstractmethod
    async def install(self) -> bool:
        """サービスのインストール（Phase 2以降で実装）"""
        pass

    @abstractmethod
    async def start(self) -> bool:
        """サービスの起動"""
        pass

    @abstractmethod
    async def stop(self) -> bool:
        """サービスの停止"""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """ヘルスチェック"""
        pass


class BackgroundServiceManager(VioraTalkComponent):
    """
    バックグラウンドサービス管理クラス

    主な責務:
    1. サービスの登録と管理
    2. 遅延起動の制御（start_service_if_needed()のみ）
    3. サービス状態の監視
    4. エラーハンドリングとフォールバック

    仕様書準拠ポイント:
    - start_all_services()メソッドは実装しない（遅延起動原則）
    - initialize()では絶対にサービスを起動しない
    - start_service_if_needed()を通じてのみサービスを起動
    """

    def __init__(self):
        """BackgroundServiceManagerの初期化"""
        super().__init__()
        self._services: Dict[str, BaseBackgroundService] = {}
        self._health_tasks: Dict[str, asyncio.Task] = {}
        self._health_check_interval = 300  # 5分（Phase 1はスタブ）

    async def initialize(self) -> None:
        """
        初期化処理

        重要: この段階では絶対にサービスを起動しない。
        サービスの登録のみを行い、実際の起動はstart_service_if_needed()で行う。

        Raises:
            InitializationError: 初期化に失敗した場合
        """
        try:
            logger.info("Initializing BackgroundServiceManager")

            # サービスの登録（Phase 1では基本的にスタブ）
            await self.initialize_services()

            # 状態を更新
            self._state = ComponentState.READY

            logger.info(
                "BackgroundServiceManager initialized successfully",
                extra={"registered_services": list(self._services.keys()), "phase": "Phase 1"},
            )

        except Exception as e:
            self._state = ComponentState.ERROR
            logger.error(f"Failed to initialize BackgroundServiceManager: {e}")
            raise InitializationError(
                f"BackgroundServiceManager initialization failed: {e}", error_code="E7000"
            )

    async def initialize_services(self) -> None:
        """
        サービスの登録（Phase 1はスタブ実装）

        Phase 2以降で実際のサービス登録を実装。
        現在は登録のみで、起動は行わない。
        """
        logger.debug(
            "Service registration completed (Phase 1 stub)",
            extra={"phase": "Phase 1", "services_count": 0},
        )

    def register_service(self, service: BaseBackgroundService) -> None:
        """
        サービスを登録

        Args:
            service: 登録するサービスインスタンス

        Raises:
            ValueError: 同名のサービスが既に登録されている場合
        """
        if service.service_name in self._services:
            raise ValueError(f"Service '{service.service_name}' is already registered")

        self._services[service.service_name] = service
        logger.info(f"Service registered: {service.service_name}")

    async def start_service_if_needed(self, service_name: str) -> ServiceStatus:
        """
        必要に応じてサービスを起動

        これが唯一のサービス起動メソッド。
        サービスが必要になった時点で初めて起動する（遅延起動）。

        Args:
            service_name: 起動するサービス名

        Returns:
            ServiceStatus: 起動後のサービス状態

        Raises:
            BackgroundServiceError: サービス起動に失敗した場合
        """
        logger.info(f"Service start requested: {service_name}")

        # Phase 1: スタブ実装
        # 実際の起動処理

        service = self._services.get(service_name)
        if not service:
            # Phase 1では登録されていないサービスもOKとする
            logger.warning(
                f"Service {service_name} not found (Phase 1 stub)",
                extra={"phase": "Phase 1", "action": "returning_mock_status"},
            )
            return ServiceStatus.RUNNING  # Phase 1: モック応答

        # 実際のサービスがある場合の処理
        info = service.get_info()

        # すでに実行中の場合
        if info.status == ServiceStatus.RUNNING:
            if await service.health_check():
                logger.debug(f"Service {service_name} is already running")
                return ServiceStatus.RUNNING
            else:
                # ヘルスチェック失敗時は再起動
                logger.warning(f"Service {service_name} health check failed, restarting")
                await service.stop()

        # サービスの起動
        try:
            logger.info(f"Starting service: {service_name}")

            # Phase 1: スタブ実装
            # Phase 2以降: await service.start()

            # ヘルスチェックタスクの開始
            await self._start_health_monitoring(service_name)

            return ServiceStatus.RUNNING

        except Exception as e:
            logger.error(
                f"Failed to start service {service_name}: {e}", extra={"error_code": "E7003"}
            )
            raise BackgroundServiceError(
                f"Failed to start service {service_name}: {e}", error_code="E7003"
            )

    async def stop_service(self, service_name: str) -> bool:
        """
        サービスを停止

        Args:
            service_name: 停止するサービス名

        Returns:
            成功した場合True
        """
        logger.info(f"Stopping service: {service_name}")

        # ヘルスチェックタスクの停止
        await self._stop_health_monitoring(service_name)

        # サービスの停止
        service = self._services.get(service_name)
        if service:
            # Phase 1: スタブ実装
            # Phase 2以降: return await service.stop()
            logger.debug(f"Service {service_name} stopped (Phase 1 stub)")
            return True

        return False

    def get_service_status(self, service_name: str) -> ServiceStatus:
        """
        サービスの状態を取得

        Args:
            service_name: サービス名

        Returns:
            ServiceStatus: サービスの現在の状態
        """
        service = self._services.get(service_name)
        if service:
            return service.get_info().status
        return ServiceStatus.NOT_INSTALLED

    def get_all_services(self) -> List[BaseBackgroundService]:
        """
        登録されているすべてのサービスを取得

        Returns:
            登録されているサービスのリスト
        """
        return list(self._services.values())

    async def _start_health_monitoring(self, service_name: str) -> None:
        """
        ヘルスチェックタスクを開始

        Args:
            service_name: 監視対象のサービス名
        """
        # Phase 1: スタブ実装
        logger.debug(f"Health monitoring started for {service_name} (Phase 1 stub)")

    async def _stop_health_monitoring(self, service_name: str) -> None:
        """
        ヘルスチェックタスクを停止

        Args:
            service_name: 監視停止するサービス名
        """
        task = self._health_tasks.get(service_name)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            del self._health_tasks[service_name]
            logger.debug(f"Health monitoring stopped for {service_name}")

    async def cleanup(self) -> None:
        """
        クリーンアップ処理

        すべてのサービスを停止し、リソースを解放する。
        """
        logger.info("Cleaning up BackgroundServiceManager")

        # すべてのヘルスチェックタスクを停止
        for service_name in list(self._health_tasks.keys()):
            await self._stop_health_monitoring(service_name)

        # すべてのサービスを停止
        for service_name in list(self._services.keys()):
            await self.stop_service(service_name)

        # 状態を更新
        self._state = ComponentState.TERMINATED

        logger.info("BackgroundServiceManager cleaned up successfully")
