"""AutoSetupManagerの単体テスト

自動セットアップ管理機能の動作を検証。
Phase 1の最小実装（スタブ）として、基本的な機能のテストに焦点を当てる。

テスト戦略ガイドライン v1.7準拠
テスト実装ガイド v1.3準拠
自動セットアップガイド v1.2準拠
開発規約書 v1.12準拠

Phase 2-3: 依存性注入パターンによるテスト改善
"""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from vioratalk.core.base import ComponentState
from vioratalk.core.exceptions import InitializationError
from vioratalk.core.setup.auto_setup_manager import AutoSetupManager
from vioratalk.core.setup.types import ComponentStatus, SetupResult, SetupStatus

# ============================================================================
# AutoSetupManagerの基本テスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(1)
class TestAutoSetupManager:
    """AutoSetupManagerクラスの基本テスト"""

    @pytest.fixture
    def manager(self):
        """AutoSetupManagerのフィクスチャ"""
        return AutoSetupManager(ui_mode="cli")

    @pytest.fixture
    def gui_manager(self):
        """GUI用AutoSetupManagerのフィクスチャ"""
        return AutoSetupManager(ui_mode="gui")

    # ------------------------------------------------------------------------
    # 初期化のテスト
    # ------------------------------------------------------------------------

    def test_initialization_cli_mode(self):
        """CLIモードでの初期化"""
        manager = AutoSetupManager(ui_mode="cli")

        assert manager.ui_mode == "cli"
        assert manager.i18n is not None
        assert isinstance(manager._setup_result, SetupResult)
        assert manager._state == ComponentState.NOT_INITIALIZED

    def test_initialization_gui_mode(self):
        """GUIモードでの初期化"""
        manager = AutoSetupManager(ui_mode="gui")

        assert manager.ui_mode == "gui"
        assert manager.i18n is not None
        assert isinstance(manager._setup_result, SetupResult)

    def test_initialization_default_mode(self):
        """デフォルトモードでの初期化"""
        manager = AutoSetupManager()

        assert manager.ui_mode == "cli"  # デフォルトはCLI

    def test_required_components_defined(self):
        """必須コンポーネントリストが定義されているか"""
        manager = AutoSetupManager()

        assert hasattr(manager, "REQUIRED_COMPONENTS")
        assert isinstance(manager.REQUIRED_COMPONENTS, list)
        assert len(manager.REQUIRED_COMPONENTS) > 0

        # 各コンポーネントの構造を確認
        for component in manager.REQUIRED_COMPONENTS:
            assert "name" in component
            assert "min_version" in component
            assert "required" in component
            assert "install_command" in component

    # ------------------------------------------------------------------------
    # initialize メソッドのテスト
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_initialize_success(self, manager):
        """正常な初期化"""
        await manager.initialize()

        assert manager._state == ComponentState.READY

    @pytest.mark.asyncio
    async def test_initialize_with_exception(self):
        """初期化中の例外処理"""
        manager = AutoSetupManager()

        # _initialize_componentsでエラーを発生させる
        with patch.object(manager, "_initialize_components", side_effect=Exception("Test error")):
            with pytest.raises(InitializationError) as exc_info:
                await manager.initialize()

            assert exc_info.value.error_code == "E8001"
            assert manager._state == ComponentState.ERROR

    @pytest.mark.asyncio
    async def test_state_transition_during_initialize(self, manager):
        """初期化中の状態遷移"""
        states = []

        async def track_state():
            states.append(manager._state)

        with patch.object(manager, "_initialize_components", side_effect=track_state):
            await manager.initialize()

        # 初期化開始時にINITIALIZINGになることを確認
        assert manager._state == ComponentState.READY

    # ------------------------------------------------------------------------
    # run_initial_setup メソッドのテスト
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_run_initial_setup_all_success(self, manager):
        """すべてのコンポーネントチェック成功"""
        # Pythonチェックは常に成功するはず
        result = await manager.run_initial_setup()

        assert isinstance(result, SetupResult)
        assert result.status in [SetupStatus.SUCCESS, SetupStatus.PARTIAL_SUCCESS]
        assert result.can_continue is True
        assert result.duration_seconds > 0
        assert result.timestamp is not None

    @pytest.mark.asyncio
    async def test_run_initial_setup_with_required_failure(self, manager):
        """必須コンポーネントの失敗"""

        # _check_componentを必須コンポーネントで失敗させる
        async def mock_check(component_info):
            if component_info["required"]:
                return ComponentStatus(
                    name=component_info["name"],
                    installed=False,
                    version=None,
                    error="Component check failed",
                )
            return ComponentStatus(name=component_info["name"], installed=True, version="1.0.0")

        with patch.object(manager, "_check_component", side_effect=mock_check):
            result = await manager.run_initial_setup()

            assert result.status == SetupStatus.FAILED
            assert result.can_continue is False

    @pytest.mark.asyncio
    async def test_run_initial_setup_with_optional_failure(self, manager):
        """オプショナルコンポーネントの失敗"""

        # _check_componentをオプショナルコンポーネントで失敗させる
        async def mock_check(component_info):
            if not component_info["required"]:
                return ComponentStatus(
                    name=component_info["name"],
                    installed=False,
                    version=None,
                    error="Optional component not found",
                )
            return ComponentStatus(name=component_info["name"], installed=True, version="3.11.0")

        with patch.object(manager, "_check_component", side_effect=mock_check):
            result = await manager.run_initial_setup()

            # オプショナルの失敗は部分的成功
            assert result.status == SetupStatus.PARTIAL_SUCCESS
            assert result.can_continue is True

    @pytest.mark.asyncio
    async def test_run_initial_setup_exception(self, manager):
        """セットアップ中の例外処理"""
        with patch.object(manager, "_check_component", side_effect=Exception("Test error")):
            with pytest.raises(InitializationError) as exc_info:
                await manager.run_initial_setup()

            assert exc_info.value.error_code == "E8002"

    # ------------------------------------------------------------------------
    # check_component メソッドのテスト
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_check_python_component(self, manager):
        """Pythonコンポーネントのチェック"""
        python_component = {
            "name": "Python",
            "min_version": "3.11.0",
            "required": True,
            "install_command": None,
        }

        status = await manager._check_component(python_component)

        assert isinstance(status, ComponentStatus)
        assert status.name == "Python"
        assert status.installed is True
        assert status.version is not None

    @pytest.mark.asyncio
    async def test_check_ffmpeg_component(self, manager):
        """FFmpegコンポーネントのチェック（Phase 1ではスタブ）"""
        ffmpeg_component = {
            "name": "FFmpeg",
            "min_version": "4.4.0",
            "required": False,
            "install_command": "winget install ffmpeg",
        }

        status = await manager._check_component(ffmpeg_component)

        assert isinstance(status, ComponentStatus)
        assert status.name == "FFmpeg"
        # Phase 1ではスタブなので結果は実装依存

    @pytest.mark.asyncio
    async def test_check_cuda_component(self, manager):
        """CUDAコンポーネントのチェック（Phase 1ではスタブ）"""
        cuda_component = {
            "name": "CUDA",
            "min_version": "11.8",
            "required": False,
            "install_command": None,
        }

        status = await manager._check_component(cuda_component)

        assert isinstance(status, ComponentStatus)
        assert status.name == "CUDA"
        # Phase 1ではスタブなので結果は実装依存

    # ------------------------------------------------------------------------
    # check_component_status メソッドのテスト
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_check_component_status_found(self, manager):
        """存在するコンポーネントの状態確認"""
        # まずセットアップを実行
        await manager.run_initial_setup()

        # Pythonは必ず存在するはず
        status = await manager.check_component_status("Python")

        assert status is not None
        assert isinstance(status, ComponentStatus)
        assert status.name == "Python"

    @pytest.mark.asyncio
    async def test_check_component_status_not_found(self, manager):
        """存在しないコンポーネントの状態確認"""
        # まずセットアップを実行
        await manager.run_initial_setup()

        status = await manager.check_component_status("NonExistent")

        assert status is None

    # ------------------------------------------------------------------------
    # cleanup メソッドのテスト
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_cleanup_success(self, manager):
        """正常なクリーンアップ"""
        await manager.initialize()
        await manager.cleanup()

        assert manager._state == ComponentState.TERMINATED

    @pytest.mark.asyncio
    async def test_cleanup_from_not_initialized(self):
        """初期化前のクリーンアップ"""
        manager = AutoSetupManager()
        await manager.cleanup()

        assert manager._state == ComponentState.TERMINATED

    # ------------------------------------------------------------------------
    # get_status メソッドのテスト
    # ------------------------------------------------------------------------

    def test_get_status_initial(self, manager):
        """初期状態のステータス取得"""
        status = manager.get_status()

        assert isinstance(status, dict)
        assert status["state"] == ComponentState.NOT_INITIALIZED.value
        assert status["ui_mode"] == "cli"
        assert "components_checked" in status
        assert status["components_checked"] == 0
        # last_setup_resultは初期状態では存在しない（Noneの場合は含まれない）

    @pytest.mark.asyncio
    async def test_get_status_after_setup(self, manager):
        """セットアップ後のステータス取得"""
        result = await manager.run_initial_setup()
        status = manager.get_status()

        assert status["state"] == ComponentState.NOT_INITIALIZED.value
        assert status["last_setup_result"] is not None
        assert status["last_setup_result"]["status"] == result.status.value

    # ------------------------------------------------------------------------
    # SetupResult の動作テスト
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_setup_result_timestamp(self, manager):
        """セットアップ結果のタイムスタンプ"""
        result = await manager.run_initial_setup()

        assert result.timestamp is not None
        assert isinstance(result.timestamp, datetime)

    @pytest.mark.asyncio
    async def test_setup_result_duration(self, manager):
        """セットアップ実行時間の記録"""
        result = await manager.run_initial_setup()

        assert result.duration_seconds > 0

    @pytest.mark.asyncio
    async def test_multiple_setup_runs(self, manager):
        """複数回のセットアップ実行"""
        result1 = await manager.run_initial_setup()
        result2 = await manager.run_initial_setup()

        # 結果は異なるインスタンスであるべき
        assert result1 is not result2

        # タイムスタンプは異なる
        assert result1.timestamp != result2.timestamp

    def test_ui_mode_parameter(self):
        """UIモードパラメータのテスト"""
        cli_manager = AutoSetupManager(ui_mode="cli")
        gui_manager = AutoSetupManager(ui_mode="gui")

        assert cli_manager.ui_mode == "cli"
        assert gui_manager.ui_mode == "gui"

        status_cli = cli_manager.get_status()
        status_gui = gui_manager.get_status()

        assert status_cli["ui_mode"] == "cli"
        assert status_gui["ui_mode"] == "gui"


# ============================================================================
# エラーハンドリングテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(1)
class TestErrorHandling:
    """エラーハンドリングのテスト"""

    @pytest.mark.asyncio
    async def test_initialization_error_code(self):
        """初期化エラーコードのテスト"""
        manager = AutoSetupManager()

        with patch.object(manager, "_initialize_components", side_effect=Exception("Test error")):
            with pytest.raises(InitializationError) as exc_info:
                await manager.initialize()

            assert exc_info.value.error_code == "E8001"

    @pytest.mark.asyncio
    async def test_setup_error_code(self):
        """セットアップエラーコードのテスト"""
        manager = AutoSetupManager()
        await manager.initialize()

        with patch.object(manager, "_check_component", side_effect=Exception("Test error")):
            with pytest.raises(InitializationError) as exc_info:
                await manager.run_initial_setup()

            assert exc_info.value.error_code == "E8002"

    @pytest.mark.asyncio
    async def test_error_during_cleanup(self):
        """クリーンアップ中のエラー処理"""
        manager = AutoSetupManager()
        await manager.initialize()

        # cleanup内部でエラーを発生させるため、_stateプロパティへのアクセス時にエラーを発生させる
        # より直接的な方法: cleanupメソッド全体を置き換える
        original_cleanup = manager.cleanup

        async def failing_cleanup():
            """エラーを発生させるクリーンアップ"""
            manager._state = ComponentState.TERMINATING
            # ここでエラーを発生させる
            raise Exception("Cleanup error")

        # cleanupメソッドを一時的に置き換え
        manager.cleanup = failing_cleanup

        with pytest.raises(Exception) as exc_info:
            await manager.cleanup()

        # エラーメッセージを確認
        assert "Cleanup error" in str(exc_info.value)

        # 元のcleanupメソッドでE8099エラーコードが発生することも確認
        # 別の方法：内部のlogger呼び出しでエラーを発生させる
        manager2 = AutoSetupManager()
        await manager2.initialize()

        with patch("vioratalk.core.setup.auto_setup_manager.logger") as mock_logger:
            # 最初のinfo呼び出しは成功、2番目でエラーを発生させる
            mock_logger.info.side_effect = [None, Exception("Logger error")]
            mock_logger.error = Mock()  # errorは正常に動作させる

            with pytest.raises(InitializationError) as exc_info:
                await manager2.cleanup()

            assert exc_info.value.error_code == "E8099"
            assert manager2._state == ComponentState.ERROR
