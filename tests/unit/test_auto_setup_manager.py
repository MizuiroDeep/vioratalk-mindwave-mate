"""AutoSetupManagerの単体テスト

自動セットアップ管理機能の動作を検証。
Phase 1の最小実装（スタブ）として、基本的な機能のテストに焦点を当てる。

テスト戦略ガイドライン v1.7準拠
テスト実装ガイド v1.3準拠
自動セットアップガイド v1.2準拠
開発規約書 v1.12準拠
"""

import sys
from unittest.mock import patch

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

        # Pythonが含まれているか
        python_component = next(
            (c for c in manager.REQUIRED_COMPONENTS if c["name"] == "Python"), None
        )
        assert python_component is not None
        assert python_component["required"] is True

    # ------------------------------------------------------------------------
    # initialize()メソッドのテスト
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_initialize_success(self, manager):
        """正常な初期化"""
        await manager.initialize()

        assert manager._state == ComponentState.READY

    @pytest.mark.asyncio
    async def test_initialize_failure(self, manager):
        """初期化失敗のシミュレーション"""
        # _simulate_init_errorフラグを使用してエラーをシミュレート
        manager._simulate_init_error = True

        # エラーが発生してInitializationErrorが投げられる
        with pytest.raises(InitializationError) as exc_info:
            await manager.initialize()

        assert "E8001" in str(exc_info.value)
        assert manager._state == ComponentState.ERROR

    # ------------------------------------------------------------------------
    # run_initial_setup()メソッドのテスト
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_run_initial_setup_basic(self, manager):
        """基本的なセットアップ実行"""
        await manager.initialize()

        result = await manager.run_initial_setup()

        assert isinstance(result, SetupResult)
        assert result.status in [
            SetupStatus.SUCCESS,
            SetupStatus.PARTIAL_SUCCESS,
            SetupStatus.FAILED,
        ]
        assert result.timestamp is not None
        assert result.duration_seconds is not None
        assert len(result.components) > 0

    @pytest.mark.asyncio
    async def test_run_initial_setup_python_check(self, manager):
        """Pythonバージョンチェックが含まれる"""
        await manager.initialize()

        result = await manager.run_initial_setup()

        # Pythonコンポーネントが含まれている
        python_component = next((c for c in result.components if c.name == "Python"), None)
        assert python_component is not None
        assert python_component.installed is True
        assert python_component.version is not None

    @pytest.mark.asyncio
    async def test_run_initial_setup_stub_components(self, manager):
        """Phase 1でのスタブコンポーネント処理"""
        await manager.initialize()

        result = await manager.run_initial_setup()

        # Phase 1ではPython以外は未インストール扱い
        non_python_components = [c for c in result.components if c.name != "Python"]

        for component in non_python_components:
            assert component.installed is False
            assert "Phase 1" in component.error

    @pytest.mark.asyncio
    async def test_run_initial_setup_partial_success(self, manager):
        """部分的成功のテスト"""
        await manager.initialize()

        # いくつかのコンポーネントを成功にする
        with patch.object(manager, "_stub_component_setup") as mock_setup:
            # Ollamaは成功、他は失敗
            async def mock_stub(component_info):
                if component_info["name"] == "Ollama":
                    return ComponentStatus(name="Ollama", installed=True, version="0.1.24")
                else:
                    return ComponentStatus(
                        name=component_info["name"],
                        installed=False,
                        error="Simulated failure for test",
                    )

            mock_setup.side_effect = mock_stub
            result = await manager.run_initial_setup()

        assert result.status == SetupStatus.PARTIAL_SUCCESS
        assert result.can_continue is True
        assert result.has_failures is True

    @pytest.mark.asyncio
    async def test_run_initial_setup_all_success(self, manager):
        """全コンポーネント成功時"""
        await manager.initialize()

        # すべてのコンポーネントを成功にする
        with patch.object(manager, "_check_python_version") as mock_python:
            with patch.object(manager, "_stub_component_setup") as mock_setup:
                mock_python.return_value = ComponentStatus(
                    name="Python", installed=True, version="3.11.9"
                )

                async def mock_all_success(component_info):
                    return ComponentStatus(
                        name=component_info["name"], installed=True, version="1.0.0"
                    )

                mock_setup.side_effect = mock_all_success
                result = await manager.run_initial_setup()

        assert result.status == SetupStatus.SUCCESS
        assert result.has_failures is False
        assert result.success_rate == 1.0

    @pytest.mark.asyncio
    async def test_run_initial_setup_python_failure(self, manager):
        """Python失敗時の処理（継続不可）"""
        await manager.initialize()

        # Pythonチェックを失敗させる
        with patch.object(manager, "_check_python_version") as mock_python:
            mock_python.return_value = ComponentStatus(
                name="Python", installed=False, error="Python version too old: 2.7"
            )

            result = await manager.run_initial_setup()

        # Pythonが使えない場合は継続不可
        assert result.status == SetupStatus.FAILED
        assert result.can_continue is False

    @pytest.mark.asyncio
    async def test_run_initial_setup_exception_handling(self, manager):
        """セットアップ中の例外処理"""
        await manager.initialize()

        # _simulate_exception_handlingフラグを使用
        manager._simulate_exception_handling = True

        # 例外が発生
        with pytest.raises(InitializationError) as exc_info:
            await manager.run_initial_setup()

        assert "E8002" in str(exc_info.value)

    # ------------------------------------------------------------------------
    # _check_python_version()メソッドのテスト
    # ------------------------------------------------------------------------

    def test_check_python_version_current(self, manager):
        """現在のPythonバージョンチェック"""
        result = manager._check_python_version()

        assert result.name == "Python"
        assert result.installed is True
        assert result.version is not None

        # バージョン文字列の形式確認

        expected_version = f"{sys.version_info.major}.{sys.version_info.minor}."
        assert expected_version in result.version

    def test_check_python_version_with_mock(self, manager):
        """モックを使用したバージョンチェック"""
        with patch("platform.python_version", return_value="3.11.9"):
            result = manager._check_python_version()

            assert result.installed is True
            assert result.version == "3.11.9"

        # 古いバージョンの場合
        with patch("platform.python_version", return_value="3.10.0"):
            result = manager._check_python_version()

            assert result.installed is False
            assert result.error is not None

    # ------------------------------------------------------------------------
    # _stub_component_setup()メソッドのテスト
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_stub_component_setup(self, manager):
        """スタブセットアップのテスト"""
        component_info = {
            "name": "Ollama",
            "min_version": "0.1.0",
            "check_command": ["ollama", "--version"],
            "required": False,
        }

        result = await manager._stub_component_setup(component_info)

        assert result.name == "Ollama"
        assert result.installed is False
        assert "Phase 1" in result.error

    # ------------------------------------------------------------------------
    # _determine_final_status()メソッドのテスト
    # ------------------------------------------------------------------------

    def test_determine_final_status_all_success(self, manager):
        """全成功時のステータス決定"""
        manager._setup_result = SetupResult()

        # すべて成功
        manager._setup_result.add_component(
            ComponentStatus("Python", installed=True, version="3.11.9")
        )
        manager._setup_result.add_component(
            ComponentStatus("Ollama", installed=True, version="0.1.24")
        )

        manager._determine_final_status()

        assert manager._setup_result.status == SetupStatus.SUCCESS
        assert manager._setup_result.can_continue is True

    def test_determine_final_status_partial_success(self, manager):
        """部分的成功時のステータス決定"""
        manager._setup_result = SetupResult()

        # Pythonは成功、他は混在
        manager._setup_result.add_component(
            ComponentStatus("Python", installed=True, version="3.11.9")
        )
        manager._setup_result.add_component(
            ComponentStatus("Ollama", installed=True, version="0.1.24")
        )
        manager._setup_result.add_component(
            ComponentStatus("AivisSpeech", installed=False, error="Failed")
        )

        manager._determine_final_status()

        assert manager._setup_result.status == SetupStatus.PARTIAL_SUCCESS
        assert manager._setup_result.can_continue is True
        assert len(manager._setup_result.warnings) > 0

    def test_determine_final_status_python_failure(self, manager):
        """Python失敗時のステータス決定"""
        manager._setup_result = SetupResult()

        # Pythonが失敗
        manager._setup_result.add_component(
            ComponentStatus("Python", installed=False, error="Not found")
        )
        manager._setup_result.add_component(
            ComponentStatus("Ollama", installed=True, version="0.1.24")
        )

        manager._determine_final_status()

        assert manager._setup_result.status == SetupStatus.FAILED
        assert manager._setup_result.can_continue is False

    # ------------------------------------------------------------------------
    # check_component_status()メソッドのテスト
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_check_component_status_existing(self, manager):
        """存在するコンポーネントのステータス確認"""
        await manager.initialize()
        await manager.run_initial_setup()

        # Pythonのステータスを確認
        status = await manager.check_component_status("Python")

        assert status is not None
        assert status.name == "Python"
        assert status.installed is True

    @pytest.mark.asyncio
    async def test_check_component_status_not_found(self, manager):
        """存在しないコンポーネントのステータス確認"""
        await manager.initialize()
        await manager.run_initial_setup()

        # 存在しないコンポーネント
        status = await manager.check_component_status("NonExistent")

        assert status is None

    # ------------------------------------------------------------------------
    # その他のメソッドのテスト
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_cleanup(self, manager):
        """クリーンアップ処理"""
        await manager.initialize()
        await manager.cleanup()

        assert manager._state == ComponentState.SHUTDOWN

    @pytest.mark.asyncio
    async def test_get_setup_result(self, manager):
        """セットアップ結果の取得"""
        await manager.initialize()

        # セットアップ前
        result_before = manager.get_setup_result()
        assert result_before is not None

        # セットアップ実行
        result = await manager.run_initial_setup()

        # セットアップ後
        result_after = manager.get_setup_result()
        assert result_after == result


# ============================================================================
# Phase 1実装の確認テスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(1)
class TestPhase1Implementation:
    """Phase 1実装の確認テスト"""

    @pytest.mark.asyncio
    async def test_no_actual_installation(self):
        """Phase 1では実際のインストールを行わない"""
        manager = AutoSetupManager()
        await manager.initialize()

        result = await manager.run_initial_setup()

        # Python以外はインストールされない
        for component in result.components:
            if component.name != "Python":
                assert component.installed is False
                assert "Phase 1" in component.error

    @pytest.mark.asyncio
    async def test_stub_behavior(self):
        """スタブ動作の確認"""
        manager = AutoSetupManager()
        await manager.initialize()

        # 複数回実行してもエラーにならない
        result1 = await manager.run_initial_setup()
        result2 = await manager.run_initial_setup()

        assert isinstance(result1, SetupResult)
        assert isinstance(result2, SetupResult)

        # タイムスタンプは異なる
        assert result1.timestamp != result2.timestamp

    @pytest.mark.asyncio
    async def test_ui_mode_parameter(self):
        """ui_modeパラメータが保持されることを確認"""
        cli_manager = AutoSetupManager(ui_mode="cli")
        gui_manager = AutoSetupManager(ui_mode="gui")

        assert cli_manager.ui_mode == "cli"
        assert gui_manager.ui_mode == "gui"

        # Phase 5まではCLI、Phase 10以降でGUI対応
        # 現時点では両方とも同じ動作


# ============================================================================
# エラーハンドリングのテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(1)
class TestErrorHandling:
    """エラーハンドリングのテスト"""

    @pytest.mark.asyncio
    async def test_initialization_error_code(self):
        """初期化エラーのエラーコード確認"""
        manager = AutoSetupManager()

        # _simulate_init_errorフラグを使用してエラーをシミュレート
        manager._simulate_init_error = True

        with pytest.raises(InitializationError) as exc_info:
            await manager.initialize()

        # E8001エラーコード
        assert exc_info.value.error_code == "E8001"

    @pytest.mark.asyncio
    async def test_setup_error_code(self):
        """セットアップエラーのエラーコード確認"""
        manager = AutoSetupManager()
        await manager.initialize()

        # _simulate_setup_errorフラグを使用してエラーをシミュレート
        manager._simulate_setup_error = True

        with pytest.raises(InitializationError) as exc_info:
            await manager.run_initial_setup()

        # E8002エラーコード
        assert exc_info.value.error_code == "E8002"
