"""
VioraTalk Engine - Core System Manager

VioraTalkシステム全体を管理するメインエンジンクラス。
8段階の初期化フローに従い、各コンポーネントを順序立てて初期化する。
遅延起動原則に基づき、必要になるまでサービスは起動しない。

エンジン初期化仕様書 v1.4準拠
バックグラウンドサービス管理仕様書 v1.3準拠
自動セットアップガイド v1.2準拠
インターフェース定義書 v1.34準拠

Copyright (c) 2025 MizuiroDeep
"""

import copy
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from vioratalk.configuration.config_manager import ConfigManager
from vioratalk.core.base import ComponentState, VioraTalkComponent
from vioratalk.core.exceptions import ConfigurationError, InitializationError
from vioratalk.core.i18n_manager import I18nManager
from vioratalk.core.setup.auto_setup_manager import AutoSetupManager
from vioratalk.core.setup.types import SetupResult, SetupStatus
from vioratalk.services.background_service_manager import BackgroundServiceManager

logger = logging.getLogger(__name__)


class VioraTalkEngine(VioraTalkComponent):
    """VioraTalkシステムのメインエンジン

    全体を統括し、8段階の初期化フローに従って各コンポーネントを管理。
    遅延起動原則により、サービスは必要になるまで起動しない。

    Attributes:
        config_path: 設定ファイルのパス
        i18n_manager: 国際化マネージャー
        auto_setup_manager: 自動セットアップマネージャー
        background_service_manager: バックグラウンドサービスマネージャー
        config_manager: 設定管理マネージャー
        model_selector: モデル選択マネージャー（Phase 2）
        prompt_engine: プロンプトエンジン（Phase 2）
        dialogue_manager: 対話マネージャー（Phase 3）
    """

    def __init__(self, config_path: Optional[Path] = None, first_run: Optional[bool] = None):
        """VioraTalkEngineの初期化

        Args:
            config_path: 設定ファイルのパス（省略時はデフォルト）
            first_run: 初回起動フラグ（None=自動判定、True=強制初回、False=強制非初回）
        """
        super().__init__()  # VioraTalkComponentは引数を受け取らない
        self.config_path = config_path or Path("user_settings/config.yaml")

        # 初期化順序に従ったコンポーネント
        self.config_manager: Optional[ConfigManager] = None
        self.i18n_manager: Optional[I18nManager] = None
        self.auto_setup_manager: Optional[AutoSetupManager] = None
        self.background_service_manager: Optional[BackgroundServiceManager] = None

        # Phase 2以降のコンポーネント（プレースホルダー）
        self.model_selector = None
        self.prompt_engine = None
        self.dialogue_manager = None

        # 内部状態
        self._config: Dict[str, Any] = {}
        self._first_run_override = first_run
        self._initialized_at: Optional[datetime] = None

        logger.info(
            f"VioraTalkEngine created with config_path={config_path}, " f"first_run={first_run}"
        )

    async def initialize(self) -> None:
        """8段階初期化フロー

        エンジン初期化仕様書 v1.4に基づく初期化処理。
        エラーが発生した場合でも可能な限り継続する。

        Raises:
            InitializationError: 致命的なエラーが発生した場合（E0000）
        """
        logger.info("Starting VioraTalk Engine initialization (8-stage flow)")
        self._state = ComponentState.INITIALIZING

        try:
            # Stage 1: 設定ファイル読み込み
            await self._load_configuration()

            # Stage 2: ロガー設定（LoggerManagerは既に起動時に設定済み）
            logger.info("Stage 2: Logger already configured at startup")

            # Stage 3: I18nManager初期化
            await self._initialize_i18n_manager()

            # Stage 4: 初回起動判定
            is_first_run = await self._check_first_run()

            # Stage 5: 自動セットアップ（初回のみ）
            if is_first_run:
                await self._run_auto_setup()

            # Stage 6: BackgroundServiceManager初期化（遅延起動）
            await self._initialize_background_service_manager()

            # Stage 7: その他のマネージャー初期化（Phase 2以降）
            # Phase 1ではスキップ
            logger.info("Stage 7: Other managers (Phase 2+) - skipped")

            # Stage 8: 初期化完了処理
            await self._finalize_initialization()

            self._state = ComponentState.READY
            self._initialized_at = datetime.now()
            logger.info("VioraTalk Engine initialization completed successfully")

        except Exception as e:
            self._state = ComponentState.ERROR
            logger.error(f"Fatal error during initialization: {e}")
            raise InitializationError(
                f"VioraTalk Engine initialization failed: {e}", error_code="E0000"
            )

    async def _load_configuration(self) -> None:
        """Stage 1: 設定ファイル読み込み

        エラーハンドリング指針 v1.20準拠:
        設定ファイルエラー（E0001）はデフォルト設定で継続
        """
        logger.info("Stage 1: Loading configuration")

        try:
            # ConfigManagerの初期化
            self.config_manager = ConfigManager(self.config_path)
            await self.config_manager.initialize()

            # 設定を取得
            self._config = self.config_manager.get_all()

            logger.info(f"Configuration loaded from {self.config_path}")

        except ConfigurationError as e:
            # E0001: 設定エラーはデフォルト設定で継続
            logger.warning(f"Configuration error (E0001): {e}, using defaults")
            self._config = self._get_default_config()

        except Exception as e:
            # 予期しないエラーもデフォルト設定で継続
            logger.warning(f"Unexpected error loading configuration: {e}, using defaults")
            self._config = self._get_default_config()

    async def _initialize_i18n_manager(self) -> None:
        """Stage 3: I18nManager初期化"""
        logger.info("Stage 3: Initializing I18nManager")

        # 設定から言語を取得（デフォルト: ja）
        language = self._config.get("general", {}).get("language", "ja")

        self.i18n_manager = I18nManager(language=language)
        await self.i18n_manager.initialize()

        logger.info(f"I18nManager initialized with language={language}")

    async def _check_first_run(self) -> bool:
        """Stage 4: 初回起動判定

        Returns:
            bool: 初回起動の場合True
        """
        logger.info("Stage 4: Checking first run status")

        # オーバーライドが設定されている場合はそれを使用
        if self._first_run_override is not None:
            logger.info(f"First run override: {self._first_run_override}")
            return self._first_run_override

        # マーカーファイルのパスを決定
        if self.config_path:
            # 設定ファイルがある場合は、その相対的な位置にデータディレクトリを配置
            data_dir = self.config_path.parent.parent / "data"
        else:
            # デフォルト
            data_dir = Path("data")

        # マーカーファイルで判定
        marker_file = data_dir / ".setup_completed"
        is_first_run = not marker_file.exists()

        logger.info(f"First run: {is_first_run} (marker: {marker_file})")
        return is_first_run

    async def _run_auto_setup(self) -> None:
        """Stage 5: 自動セットアップ（初回のみ）

        エラーハンドリング指針 v1.20準拠:
        セットアップエラー（E0002）でも継続
        """
        logger.info("Stage 5: Running auto setup")

        try:
            self.auto_setup_manager = AutoSetupManager()
            await self.auto_setup_manager.initialize()

            result: SetupResult = await self.auto_setup_manager.run_initial_setup()

            if result.status == SetupStatus.SUCCESS:
                logger.info("Auto setup completed successfully")
            elif result.status == SetupStatus.PARTIAL_SUCCESS:
                logger.warning(f"Auto setup partially succeeded with warnings: {result.warnings}")
            else:
                logger.error(f"Auto setup failed: {result.error}")

        except Exception as e:
            # E0002: セットアップエラーでも継続
            logger.error(f"Auto setup error (E0002): {e}, continuing with limited functionality")

    async def _initialize_background_service_manager(self) -> None:
        """Stage 6: BackgroundServiceManager初期化

        バックグラウンドサービス管理仕様書 v1.3準拠:
        初期化時にサービスは起動しない（遅延起動原則）
        """
        logger.info("Stage 6: Initializing BackgroundServiceManager")

        self.background_service_manager = BackgroundServiceManager()
        await self.background_service_manager.initialize()

        logger.info("BackgroundServiceManager initialized (services not started)")

    async def _finalize_initialization(self) -> None:
        """Stage 8: 初期化完了処理"""
        logger.info("Stage 8: Finalizing initialization")

        # マーカーファイルのパスを決定
        if self.config_path:
            # 設定ファイルがある場合は、その相対的な位置にデータディレクトリを配置
            data_dir = self.config_path.parent.parent / "data"
        else:
            # デフォルト
            data_dir = Path("data")

        # 初回起動の場合はマーカーファイル作成
        marker_file = data_dir / ".setup_completed"
        if not marker_file.exists():
            marker_file.parent.mkdir(parents=True, exist_ok=True)
            marker_file.touch()
            logger.info(f"Setup completion marker created at {marker_file}")

    async def cleanup(self) -> None:
        """リソースのクリーンアップ

        各コンポーネントを逆順でクリーンアップする。
        エラーが発生しても他のコンポーネントのクリーンアップは続行。
        ComponentState仕様書準拠でTERMINATINGを使用。
        """
        logger.info("Starting VioraTalk Engine cleanup")
        self._state = ComponentState.TERMINATING  # 仕様書準拠

        # 逆順でクリーンアップ
        components = [
            ("BackgroundServiceManager", self.background_service_manager),
            ("AutoSetupManager", self.auto_setup_manager),
            ("I18nManager", self.i18n_manager),
            ("ConfigManager", self.config_manager),
        ]

        for name, component in components:
            if component:
                try:
                    await component.cleanup()
                    logger.info(f"{name} cleaned up")
                except Exception as e:
                    logger.error(f"Error cleaning up {name}: {e}")

        self._state = ComponentState.TERMINATED
        logger.info("VioraTalk Engine cleanup completed")

    def get_state(self) -> ComponentState:
        """現在の状態を取得

        Returns:
            ComponentState: 現在の状態
        """
        return self._state

    def get_config(self) -> Dict[str, Any]:
        """現在の設定を取得（コピーを返す）

        Returns:
            Dict[str, Any]: 現在の設定のコピー
        """
        return copy.deepcopy(self._config)

    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        """デフォルト設定を取得

        Returns:
            Dict[str, Any]: デフォルト設定
        """
        return {
            "general": {"app_name": "VioraTalk", "version": "0.0.1", "language": "ja"},
            "features": {"auto_setup": True, "limited_mode": False},
            "paths": {"data": "data", "logs": "logs", "models": "models"},
        }
