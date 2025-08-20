"""
自動セットアップ管理モジュール

初回起動時の自動セットアップ処理を管理する。
Phase 1では最小実装とし、実際のインストール処理はPhase 2以降で実装。

関連ドキュメント:
    - エンジン初期化仕様書 v1.4
    - 自動セットアップガイド v1.2
    - エラーハンドリング指針 v1.20（E8xxx系エラー）
    - 非同期プログラミングガイドライン v1.3
"""

import asyncio
import platform
from datetime import datetime
from typing import Any, Dict, Optional

from vioratalk.core.base import ComponentState, VioraTalkComponent
from vioratalk.core.exceptions import InitializationError
from vioratalk.core.i18n_manager import I18nManager
from vioratalk.core.setup.types import ComponentStatus, SetupResult, SetupStatus
from vioratalk.utils.logger_manager import LoggerManager

logger = LoggerManager.get_logger(__name__)


class AutoSetupManager(VioraTalkComponent):
    """
    自動セットアップマネージャー

    初回起動時に必要なコンポーネントの自動セットアップを行う。
    SetupResult型で詳細な結果を返し、部分的成功にも対応。

    Phase 1: 最小実装（実際のインストールはスタブ）
    Phase 2以降: 実際のインストール処理を実装

    Attributes:
        ui_mode: UI表示モード（"cli" または "gui"）
        i18n: 国際化マネージャー
    """

    # セットアップが必要なコンポーネントのリスト（Phase 1では定義のみ）
    REQUIRED_COMPONENTS = [
        {
            "name": "Python",
            "min_version": "3.11.0",
            "check_command": ["python", "--version"],
            "required": True,
        },
        {
            "name": "Ollama",
            "min_version": "0.1.0",
            "check_command": ["ollama", "--version"],
            "required": False,  # オプション
        },
        {
            "name": "FasterWhisper",
            "min_version": "0.10.0",
            "check_command": None,  # Pythonパッケージ
            "required": False,
        },
        {
            "name": "AivisSpeech",
            "min_version": "1.0.0",
            "check_command": None,  # 外部ツール
            "required": False,
        },
    ]

    def __init__(self, ui_mode: str = "cli"):
        """
        AutoSetupManagerの初期化

        Args:
            ui_mode: UI表示モード（"cli" または "gui"）
                     Phase 5まではCLI、Phase 10以降でGUI対応
        """
        super().__init__()
        self.ui_mode = ui_mode
        self.i18n = I18nManager()
        self._setup_result = SetupResult()

        # テスト用フラグ
        self._simulate_init_error = False
        self._simulate_setup_error = False
        self._simulate_exception_handling = False

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

            # テスト用：エラーシミュレーション
            if self._simulate_init_error:
                raise Exception("Simulated initialization error")

            # stateの設定（エラーが発生する可能性のある処理）
            self._state = ComponentState.READY
            logger.info("AutoSetupManager initialization completed")

        except Exception as e:
            self._state = ComponentState.ERROR
            # loggerのエラー記録（エラーが発生する可能性のある処理）
            try:
                logger.error(f"AutoSetupManager initialization failed: {e}")
            except Exception:
                pass  # ログ出力のエラーは無視
            # 必ずInitializationErrorでラップして投げる
            raise InitializationError(
                f"AutoSetupManager initialization failed: {e}", error_code="E8001"
            ) from e

    async def run_initial_setup(self) -> SetupResult:
        """
        初回セットアップを実行

        Phase 1では最小実装として、基本的なチェックのみ実行。
        実際のインストール処理はPhase 2以降で実装。

        Returns:
            SetupResult: セットアップの詳細な結果

        Raises:
            InitializationError: セットアップ中の致命的エラー（E8002）
        """
        logger.info("Starting initial setup")
        start_time = datetime.now()

        try:
            # テスト用：セットアップエラーシミュレーション
            if self._simulate_setup_error:
                raise RuntimeError("Simulated setup error")

            # テスト用：例外ハンドリングのシミュレーション
            if self._simulate_exception_handling:
                raise RuntimeError("Unexpected error")

            self._setup_result = SetupResult(status=SetupStatus.NOT_STARTED, timestamp=start_time)

            # Phase 1: システム環境の基本チェック
            await self._check_system_requirements()

            # Phase 1: コンポーネントのスタブセットアップ
            for component_info in self.REQUIRED_COMPONENTS:
                component_status = await self._stub_component_setup(component_info)

                # モックが返したComponentStatusを処理
                # （通常は_stub_component_setup内でadd_componentが呼ばれるが、
                # モックの場合は手動で追加する必要がある）
                if component_status and not any(
                    c.name == component_status.name for c in self._setup_result.components
                ):
                    self._setup_result.add_component(component_status)

                if component_status:  # Noneでない場合のみ待機
                    # タイムスタンプを変えるため少し待機
                    await asyncio.sleep(0.001)

            # セットアップ結果の集計
            self._determine_final_status()

            # 実行時間の記録
            end_time = datetime.now()
            self._setup_result.duration_seconds = (end_time - start_time).total_seconds()

            logger.info(
                f"Initial setup completed with status: {self._setup_result.status.value}",
                extra={
                    "success_rate": self._setup_result.success_rate,
                    "can_continue": self._setup_result.can_continue,
                },
            )

            return self._setup_result

        except Exception as e:
            # エラー発生時も結果を記録
            self._setup_result.status = SetupStatus.FAILED
            self._setup_result.add_warning(f"Setup failed: {e}")

            end_time = datetime.now()
            self._setup_result.duration_seconds = (end_time - start_time).total_seconds()

            logger.error(f"Initial setup failed: {e}")

            # 必ずInitializationErrorでラップして投げる
            raise InitializationError(f"Initial setup failed: {e}", error_code="E8002")

    async def _check_system_requirements(self) -> None:
        """
        システム要件をチェック

        Phase 1: 基本的なPythonバージョンチェックのみ
        Phase 2以降: より詳細なシステム要件チェック
        """
        logger.debug("Checking system requirements")

        # Python バージョンチェック
        python_status = self._check_python_version()

        if not python_status.installed:
            self._setup_result.add_warning(
                f"Python version {python_status.version} is below minimum requirement (3.11.0)"
            )

        self._setup_result.add_component(python_status)

        logger.debug(
            f"Python version check: {python_status.version}",
            extra={"meets_requirement": python_status.installed},
        )

    def _check_python_version(self) -> ComponentStatus:
        """
        Pythonバージョンをチェックして結果を返す

        Returns:
            ComponentStatus: Pythonのインストール状態
        """
        python_version = platform.python_version()
        major, minor, patch = map(int, python_version.split("."))
        installed = major > 3 or (major == 3 and minor >= 11)

        return ComponentStatus(
            name="Python",
            installed=installed,
            version=python_version,
            error=None if installed else f"Python 3.11+ required, found {python_version}",
        )

    async def _stub_component_setup(self, component_info: Dict[str, Any]) -> ComponentStatus:
        """
        コンポーネントのスタブセットアップ

        Phase 1では実際のインストールは行わず、
        チェックとスタブの返却のみ行う。

        Args:
            component_info: コンポーネント情報

        Returns:
            ComponentStatus: セットアップ結果
        """
        name = component_info["name"]
        required = component_info["required"]

        # Phase 1: スタブ実装（常に未インストールとして扱う）
        # ただしPythonは除く（すでにチェック済み）
        if name == "Python":
            return None  # すでに_check_system_requirementsで処理済み

        # Phase 1では他のコンポーネントは「未インストール」として記録
        component_status = ComponentStatus(
            name=name,
            installed=False,
            version=None,
            error=f"{name} installation check skipped in Phase 1",
        )

        self._setup_result.add_component(component_status)

        logger.debug(
            f"Component setup stub: {name}", extra={"required": required, "phase": "Phase 1"}
        )

        return component_status

    def _determine_final_status(self) -> None:
        """
        セットアップの最終ステータスを決定

        SetupResult型の仕様に基づき、
        成功率と必須コンポーネントの状態から
        適切なステータスを設定する。
        """
        success_rate = self._setup_result.success_rate

        # 必須コンポーネント（Python）のチェック
        python_status = next((c for c in self._setup_result.components if c.name == "Python"), None)

        if not python_status or not python_status.installed:
            # Pythonが使えない場合は致命的
            self._setup_result.status = SetupStatus.FAILED
            self._setup_result.can_continue = False
            self._setup_result.add_warning("Python is not available. Cannot continue.")
            return

        # 成功率に基づくステータス決定
        if success_rate >= 1.0:
            # 完全成功（すべてのコンポーネントが成功）
            self._setup_result.status = SetupStatus.SUCCESS
        elif success_rate > 0:  # 1つでも成功していれば部分的成功
            # 部分的成功
            self._setup_result.status = SetupStatus.PARTIAL_SUCCESS
            if success_rate <= 0.25:
                self._setup_result.add_warning(
                    "Most components failed to install. Very limited functionality available."
                )
            elif success_rate <= 0.5:
                self._setup_result.add_warning(
                    "Half of the components failed to install. Limited functionality available."
                )
            else:
                self._setup_result.add_warning("Some optional components were not installed.")
        else:
            # 完全失敗（すべて失敗 - 通常はPythonが失敗した場合のみ）
            self._setup_result.status = SetupStatus.FAILED
            self._setup_result.can_continue = False

    async def check_component_status(self, component_name: str) -> Optional[ComponentStatus]:
        """
        特定のコンポーネントの状態を確認

        Args:
            component_name: 確認するコンポーネント名

        Returns:
            コンポーネントの状態、見つからない場合はNone
        """
        for component in self._setup_result.components:
            if component.name == component_name:
                return component
        return None

    async def cleanup(self) -> None:
        """クリーンアップ処理"""
        logger.info("AutoSetupManager cleanup")
        self._state = ComponentState.SHUTDOWN

    def get_setup_result(self) -> SetupResult:
        """
        最後のセットアップ結果を取得

        Returns:
            SetupResult: 最後に実行したセットアップの結果
        """
        return self._setup_result


# Phase 1での利用例
if __name__ == "__main__":

    async def test_setup():
        """自動セットアップのテスト"""
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
