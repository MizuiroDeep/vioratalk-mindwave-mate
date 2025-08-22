"""AutoSetupManager - 自動セットアップ管理

初回起動時の自動セットアップを管理するコンポーネント。
必要なコンポーネントのチェックとインストールを行う。

インターフェース定義書 v1.34準拠
自動セットアップガイド v1.2準拠
エラーハンドリング指針 v1.20準拠
バージョン管理仕様書 v1.1準拠
開発規約書 v1.12準拠

Phase 1: 最小実装（スタブとして動作）
Phase 2-3: 実際のチェックとインストール機能
"""

import asyncio
import logging
import time
from datetime import datetime
from platform import python_version
from typing import Any, Dict, Optional

from vioratalk.core.base import ComponentState, VioraTalkComponent
from vioratalk.core.exceptions import InitializationError
from vioratalk.core.i18n_manager import I18nManager
from vioratalk.core.setup.types import ComponentStatus, SetupResult, SetupStatus

# ログ設定（開発規約書v1.12準拠）
logger = logging.getLogger(__name__)


class AutoSetupManager(VioraTalkComponent):
    """
    自動セットアップ管理クラス

    初回起動時に必要なコンポーネントをチェックし、
    不足しているものをインストールする。

    Attributes:
        ui_mode: UI表示モード（cli/gui）
        i18n: 国際化マネージャー
        _setup_result: セットアップ結果
        _last_setup_result: 最後のセットアップ結果
        REQUIRED_COMPONENTS: 必須コンポーネントリスト
    """

    # 必須コンポーネントリスト（自動セットアップガイド v1.2準拠）
    REQUIRED_COMPONENTS = [
        {
            "name": "Python",
            "min_version": "3.11.0",
            "required": True,
            "install_command": None,  # Pythonは手動インストール
        },
        {
            "name": "FFmpeg",
            "min_version": "4.4.0",
            "required": False,  # Phase 1では必須ではない
            "install_command": "winget install ffmpeg",
        },
        {
            "name": "CUDA",
            "min_version": "11.8",
            "required": False,  # GPU利用は必須ではない
            "install_command": None,  # 手動インストール推奨
        },
    ]

    def __init__(self, ui_mode: str = "cli"):
        """
        AutoSetupManagerの初期化

        Args:
            ui_mode: UI表示モード（"cli" または "gui"）
        """
        super().__init__()

        # UI モードの設定
        self.ui_mode = ui_mode

        # I18nManagerのインスタンス取得（Phase 1では最小実装）
        self.i18n = I18nManager()

        # セットアップ結果の初期化
        self._setup_result = SetupResult()
        self._last_setup_result = None  # 最後の実行結果を保存

        logger.info("AutoSetupManager initialized", extra={"ui_mode": ui_mode, "phase": "Phase 1"})

    async def initialize(self) -> None:
        """
        コンポーネントの初期化

        Raises:
            InitializationError: 初期化に失敗した場合（E8001）
        """
        try:
            self._state = ComponentState.INITIALIZING

            # Phase 1: 基本的な初期化のみ
            # Phase 2以降: 実際のチェック処理
            logger.info("AutoSetupManager initialization started")

            # 初期化処理（Phase 1では最小実装）
            await self._initialize_components()

            self._state = ComponentState.READY
            logger.info("AutoSetupManager initialization completed")

        except Exception as e:
            self._state = ComponentState.ERROR
            logger.error(f"AutoSetupManager initialization failed: {e}")
            raise InitializationError(
                f"AutoSetupManager initialization failed: {e}", error_code="E8001"
            ) from e

    async def _initialize_components(self) -> None:
        """内部コンポーネントの初期化

        Phase 2-3で依存性注入パターンに変更予定
        """
        # Phase 1では特に処理なし
        pass

    async def run_initial_setup(self) -> SetupResult:
        """
        初回セットアップを実行

        Phase 1では最小実装として、基本的なチェックのみ実行。
        実際のインストール処理はPhase 2以降で実装。

        Returns:
            SetupResult: セットアップ結果

        Raises:
            InitializationError: セットアップ中にエラーが発生した場合（E8002）
        """
        try:
            start_time = time.time()
            logger.info("Initial setup started")

            # 新しいSetupResultインスタンスを作成（タイムスタンプの重複を防ぐ）
            self._setup_result = SetupResult()

            # Phase 1: スタブ実装
            # 各コンポーネントのチェック（実際のインストールは行わない）
            for component_info in self.REQUIRED_COMPONENTS:
                component_status = await self._check_component(component_info)
                self._setup_result.add_component(component_status)

                # 必須コンポーネントが失敗した場合のみエラー
                if component_info["required"] and not component_status.installed:
                    self._setup_result.status = SetupStatus.FAILED
                    self._setup_result.can_continue = False
                    logger.error(f"Required component {component_info['name']} check failed")
                    break

            # 結果の集計
            self._calculate_setup_result()

            # 実行時間の記録（time.sleep(0.001)で確実に0より大きくする）
            await asyncio.sleep(0.001)  # 非同期でわずかな待機時間を入れる
            end_time = time.time()
            self._setup_result.duration_seconds = end_time - start_time
            self._setup_result.timestamp = datetime.now()

            # 最後の実行結果を保存
            self._last_setup_result = self._setup_result

            logger.info(
                f"Initial setup completed with status: {self._setup_result.status.value}",
                extra={
                    "success_rate": self._setup_result.success_rate,
                    "duration": self._setup_result.duration_seconds,
                },
            )

            return self._setup_result

        except Exception as e:
            logger.error(f"Initial setup failed: {e}")
            raise InitializationError(f"Initial setup failed: {e}", error_code="E8002") from e

    async def _check_component(self, component_info: Dict[str, Any]) -> ComponentStatus:
        """
        コンポーネントの状態をチェック

        Phase 1: スタブ実装（Pythonのみ実際にチェック）
        Phase 2-3: 実際のチェック処理

        Args:
            component_info: コンポーネント情報

        Returns:
            ComponentStatus: コンポーネントの状態
        """
        name = component_info["name"]

        # Phase 1: Pythonのみ実際にチェック
        if name == "Python":
            return self._check_python_version(component_info)

        # その他はスタブとして成功を返す（Phase 2以降で実装）
        return ComponentStatus(name=name, installed=True, version="stub", error=None)

    def _check_python_version(self, component_info: Dict[str, Any]) -> ComponentStatus:
        """
        Pythonバージョンをチェック

        Args:
            component_info: Pythonコンポーネント情報

        Returns:
            ComponentStatus: Pythonの状態
        """
        try:
            current_version = python_version()
            min_version = component_info["min_version"]

            # バージョン比較（簡易実装）
            # packagingが利用可能な場合は使用、なければ文字列比較
            try:
                from packaging import version

                is_valid = version.parse(current_version) >= version.parse(min_version)
            except ImportError:
                # packagingがない場合は単純な文字列比較
                current_parts = current_version.split(".")
                min_parts = min_version.split(".")
                is_valid = current_parts >= min_parts

            if is_valid:
                return ComponentStatus(
                    name="Python", installed=True, version=current_version, error=None
                )
            else:
                return ComponentStatus(
                    name="Python",
                    installed=False,
                    version=current_version,
                    error=f"Version {current_version} is below minimum {min_version}",
                )

        except Exception as e:
            return ComponentStatus(name="Python", installed=False, version=None, error=str(e))

    def _calculate_setup_result(self) -> None:
        """セットアップ結果を集計"""
        total = len(self._setup_result.components)
        if total == 0:
            self._setup_result.status = SetupStatus.FAILED
            return

        success_count = sum(1 for c in self._setup_result.components if c.installed)

        # 成功率を計算
        success_rate = success_count / total if total > 0 else 0.0

        # warningsがある場合はPARTIAL_SUCCESSにする
        has_warnings = len(self._setup_result.warnings) > 0

        # ステータスの判定
        if success_rate == 1.0 and not has_warnings:
            self._setup_result.status = SetupStatus.SUCCESS
        elif success_rate > 0:
            self._setup_result.status = SetupStatus.PARTIAL_SUCCESS
        else:
            self._setup_result.status = SetupStatus.FAILED
            self._setup_result.can_continue = False

    async def check_component_status(self, component_name: str) -> Optional[ComponentStatus]:
        """
        特定のコンポーネントの状態を確認

        Args:
            component_name: コンポーネント名

        Returns:
            ComponentStatus: コンポーネントの状態（見つからない場合はNone）
        """
        # 最新のセットアップ結果から検索
        for component in self._setup_result.components:
            if component.name == component_name:
                return component
        return None

    async def cleanup(self) -> None:
        """
        クリーンアップ処理

        Raises:
            InitializationError: クリーンアップに失敗した場合（E8099）
        """
        try:
            self._state = ComponentState.TERMINATING

            # Phase 1では特に処理なし
            # Phase 2-3: リソースの解放処理
            logger.info("AutoSetupManager cleanup started")

            self._state = ComponentState.TERMINATED
            logger.info("AutoSetupManager cleanup completed")

        except Exception as e:
            self._state = ComponentState.ERROR
            logger.error(f"AutoSetupManager cleanup failed: {e}")
            raise InitializationError(
                f"AutoSetupManager cleanup failed: {e}", error_code="E8099"
            ) from e

    def get_status(self) -> Dict[str, Any]:
        """
        現在のステータスを取得

        Returns:
            Dict[str, Any]: ステータス情報
        """
        status = {
            "component": "AutoSetupManager",
            "state": self._state.value,
            "is_available": self.is_available(),
            "ui_mode": self.ui_mode,
            "components_checked": len(self._setup_result.components),
        }

        # 最後のセットアップ結果を含める
        if self._last_setup_result is not None:
            status["last_setup_result"] = {
                "status": self._last_setup_result.status.value,
                "success_rate": self._last_setup_result.success_rate,
                "duration": self._last_setup_result.duration_seconds,
                "timestamp": self._last_setup_result.timestamp.isoformat()
                if self._last_setup_result.timestamp
                else None,
            }

        return status

    def is_available(self) -> bool:
        """
        利用可能かどうかを確認

        Returns:
            bool: 利用可能な場合True
        """
        return self._state in [ComponentState.READY, ComponentState.RUNNING]

    def get_setup_result(self) -> SetupResult:
        """
        最新のセットアップ結果を取得

        Returns:
            SetupResult: セットアップ結果
        """
        return self._setup_result

    def __str__(self) -> str:
        """文字列表現"""
        return f"AutoSetupManager(" f"ui_mode={self.ui_mode}, " f"state={self._state.value})"

    def __repr__(self) -> str:
        """詳細表現"""
        return (
            f"AutoSetupManager(ui_mode={self.ui_mode}, "
            f"state={self._state.value}, "
            f"setup_result={self._setup_result})"
        )


# Phase 1での利用例（開発/テスト用）
if __name__ == "__main__":

    async def test_setup():
        """自動セットアップのテスト"""
        from vioratalk.utils.logger_manager import LoggerManager

        # LoggerManagerの初期化
        LoggerManager()

        # AutoSetupManagerの作成
        setup_manager = AutoSetupManager(ui_mode="cli")

        # 初期化
        await setup_manager.initialize()

        # セットアップ実行
        result = await setup_manager.run_initial_setup()

        # 結果の表示
        print("\n=== Setup Result ===")
        print(f"Status: {result.status.value}")
        print(f"Success rate: {result.success_rate:.1%}")
        print(f"Can continue: {result.can_continue}")
        print(f"Duration: {result.duration_seconds:.2f} seconds")

        print("\n=== Components ===")
        for component in result.components:
            status = "✓" if component.installed else "✗"
            print(f"{status} {component.name}: {component.version or component.error}")

        if result.warnings:
            print("\n=== Warnings ===")
            for warning in result.warnings:
                print(f"⚠ {warning}")

        # クリーンアップ
        await setup_manager.cleanup()

    # テスト実行
    asyncio.run(test_setup())
