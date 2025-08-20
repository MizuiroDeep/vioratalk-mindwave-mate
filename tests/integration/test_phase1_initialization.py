"""Phase 1統合テスト - 初期化フロー

VioraTalkEngine全体の初期化フローを検証する統合テスト。
8段階初期化、遅延起動原則、エラーハンドリングを確認。

テスト戦略ガイドライン v1.7準拠
エンジン初期化仕様書 v1.4準拠
バックグラウンドサービス管理仕様書 v1.3準拠
"""

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from vioratalk.core.base import ComponentState
from vioratalk.core.setup.types import ComponentStatus, SetupResult, SetupStatus
from vioratalk.core.vioratalk_engine import VioraTalkEngine
from vioratalk.services.background_service_manager import ServiceStatus

# ============================================================================
# フィクスチャ
# ============================================================================


@pytest.fixture
def temp_test_dir(tmp_path):
    """テスト用一時ディレクトリ"""
    # 必要なディレクトリを作成
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "logs").mkdir(exist_ok=True)
    (tmp_path / "models").mkdir(exist_ok=True)
    (tmp_path / "user_settings").mkdir(exist_ok=True)
    return tmp_path


@pytest.fixture
def valid_config_file(temp_test_dir):
    """有効な設定ファイル"""
    config_path = temp_test_dir / "user_settings" / "config.yaml"
    config_data = {
        "general": {"app_name": "VioraTalk Integration Test", "version": "0.0.1", "language": "ja"},
        "features": {"auto_setup": True, "limited_mode": False},
        "paths": {
            "data": str(temp_test_dir / "data"),
            "logs": str(temp_test_dir / "logs"),
            "models": str(temp_test_dir / "models"),
        },
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f)
    return config_path


@pytest.fixture
def marker_file(temp_test_dir):
    """セットアップ完了マーカーファイル"""
    return temp_test_dir / "data" / ".setup_completed"


# ============================================================================
# 初期化フローテスト
# ============================================================================


@pytest.mark.integration
@pytest.mark.phase(1)
class TestPhase1InitializationFlow:
    """Phase 1初期化フローの統合テスト"""

    @pytest.fixture(autouse=True)
    def setup(self, temp_test_dir, marker_file):
        """各テストの前処理"""
        self.temp_dir = temp_test_dir
        self.marker_file = marker_file

        # 古いプロセスディレクトリをモック
        with patch("pathlib.Path.cwd", return_value=self.temp_dir):
            yield

    @pytest.mark.asyncio
    async def test_complete_initialization_flow(self, valid_config_file):
        """完全な初期化フローのテスト"""
        # マーカーファイルを作成（初回起動ではない）
        self.marker_file.parent.mkdir(parents=True, exist_ok=True)
        self.marker_file.touch()

        # エンジンを初期化
        engine = VioraTalkEngine(config_path=valid_config_file, first_run=False)

        # 初期化実行
        await engine.initialize()

        # 状態確認
        assert engine.get_state() == ComponentState.READY

        # コンポーネントが初期化されていることを確認
        assert engine.config_manager is not None
        assert engine.i18n_manager is not None
        assert engine.auto_setup_manager is None  # 初回起動ではないのでNone
        assert engine.background_service_manager is not None

        # 設定が読み込まれていることを確認
        config = engine.get_config()
        assert config["general"]["app_name"] == "VioraTalk Integration Test"

        # クリーンアップ
        await engine.cleanup()
        assert engine.get_state() == ComponentState.TERMINATED

    @pytest.mark.asyncio
    async def test_first_run_complete_flow(self, valid_config_file):
        """初回起動時の完全フローテスト"""
        # マーカーファイルが存在しないことを確認
        if self.marker_file.exists():
            self.marker_file.unlink()

        # エンジンを初期化（初回起動）
        engine = VioraTalkEngine(config_path=valid_config_file, first_run=True)

        # AutoSetupManagerのモック設定
        with patch("vioratalk.core.vioratalk_engine.AutoSetupManager") as MockASM:
            mock_asm = AsyncMock()
            MockASM.return_value = mock_asm
            mock_asm.initialize = AsyncMock()
            mock_asm.run_initial_setup = AsyncMock(
                return_value=SetupResult(
                    status=SetupStatus.SUCCESS,
                    components=[
                        ComponentStatus("Ollama", installed=True),
                        ComponentStatus("FasterWhisper", installed=True),
                        ComponentStatus("AivisSpeech", installed=True),
                    ],
                )
            )

            # 初期化実行
            await engine.initialize()

            # AutoSetupManagerが呼ばれたことを確認
            assert engine.auto_setup_manager is not None
            mock_asm.run_initial_setup.assert_called_once()

        # マーカーファイルが作成されたことを確認
        assert self.marker_file.exists()

        # クリーンアップ
        await engine.cleanup()

    @pytest.mark.asyncio
    async def test_lazy_loading_principle(self, valid_config_file):
        """遅延起動原則のテスト"""
        # エンジンを初期化
        engine = VioraTalkEngine(config_path=valid_config_file, first_run=False)
        await engine.initialize()

        # BackgroundServiceManagerは初期化されているが
        assert engine.background_service_manager is not None

        # サービスは起動していない（遅延起動原則）
        # get_service_statusは同期メソッド（awaitなし）
        service_info = engine.background_service_manager.get_service_status("ollama")
        assert service_info != ServiceStatus.RUNNING

        # クリーンアップ
        await engine.cleanup()

    @pytest.mark.asyncio
    async def test_no_automatic_service_start(self, valid_config_file):
        """サービスが自動起動しないことを確認"""
        engine = VioraTalkEngine(config_path=valid_config_file, first_run=False)

        # サービスの起動をモニタリング
        with patch(
            "vioratalk.services.background_service_manager.BackgroundServiceManager.start_service_if_needed"
        ) as mock_start:
            await engine.initialize()

            # start_service_if_neededが呼ばれていないことを確認
            mock_start.assert_not_called()

        await engine.cleanup()

    @pytest.mark.asyncio
    async def test_partial_success_handling(self, valid_config_file):
        """部分的成功の処理テスト"""
        engine = VioraTalkEngine(config_path=valid_config_file, first_run=True)

        with patch("vioratalk.core.vioratalk_engine.AutoSetupManager") as MockASM:
            mock_asm = AsyncMock()
            MockASM.return_value = mock_asm
            mock_asm.initialize = AsyncMock()
            mock_asm.run_initial_setup = AsyncMock(
                return_value=SetupResult(
                    status=SetupStatus.PARTIAL_SUCCESS,
                    components=[
                        ComponentStatus("Ollama", installed=True),
                        ComponentStatus("FasterWhisper", installed=False),
                        ComponentStatus("AivisSpeech", installed=False),
                    ],
                    warnings=["Some components failed to install"],
                )
            )

            # 初期化は成功するはず
            await engine.initialize()
            assert engine.get_state() == ComponentState.READY

        await engine.cleanup()

    @pytest.mark.asyncio
    async def test_complete_failure_recovery(self, valid_config_file):
        """完全失敗からの回復テスト"""
        engine = VioraTalkEngine(config_path=valid_config_file, first_run=True)

        with patch("vioratalk.core.vioratalk_engine.AutoSetupManager") as MockASM:
            mock_asm = AsyncMock()
            MockASM.return_value = mock_asm
            mock_asm.initialize = AsyncMock()
            mock_asm.run_initial_setup = AsyncMock(side_effect=Exception("Setup failed completely"))

            # エラーハンドリング指針 v1.20準拠: エラーでも継続
            await engine.initialize()
            assert engine.get_state() == ComponentState.READY

        await engine.cleanup()

    @pytest.mark.asyncio
    async def test_config_error_fallback(self):
        """設定エラー時のフォールバック"""
        # 存在しない設定ファイル
        nonexistent_config = Path("/nonexistent/config.yaml")

        engine = VioraTalkEngine(config_path=nonexistent_config)

        # デフォルト設定で初期化されるはず
        await engine.initialize()

        assert engine.get_state() == ComponentState.READY

        # デフォルト設定が使用されていることを確認
        config = engine.get_config()
        assert "general" in config
        assert config["general"]["app_name"] == "VioraTalk"

        await engine.cleanup()

    @pytest.mark.asyncio
    async def test_component_initialization_error_handling(self, valid_config_file):
        """コンポーネント初期化エラーの処理"""
        engine = VioraTalkEngine(config_path=valid_config_file)

        # I18nManagerのエラーをシミュレート
        with patch("vioratalk.core.vioratalk_engine.I18nManager") as MockI18n:
            MockI18n.side_effect = Exception("I18n initialization failed")

            # 致命的エラーとして扱われる
            with pytest.raises(Exception):
                await engine.initialize()

            assert engine.get_state() == ComponentState.ERROR

    @pytest.mark.asyncio
    async def test_logger_manager_integration(self, valid_config_file):
        """LoggerManagerとの統合テスト"""
        # ログ出力を捕捉
        with patch("logging.Logger.info") as mock_log:
            engine = VioraTalkEngine(config_path=valid_config_file)
            await engine.initialize()

            # ログが出力されていることを確認
            assert mock_log.called
            log_messages = [call[0][0] for call in mock_log.call_args_list]
            assert any("Stage 1:" in msg for msg in log_messages)
            assert any("Stage 3:" in msg for msg in log_messages)
            assert any("Stage 6:" in msg for msg in log_messages)

            await engine.cleanup()

    @pytest.mark.asyncio
    async def test_i18n_manager_integration(self, valid_config_file):
        """I18nManagerとの統合テスト"""
        engine = VioraTalkEngine(config_path=valid_config_file)
        await engine.initialize()

        # I18nManagerが初期化されていることを確認
        assert engine.i18n_manager is not None

        # 言語設定が反映されていることを確認
        # （Phase 1では実際のメッセージ取得はスタブ）

        await engine.cleanup()

    @pytest.mark.asyncio
    async def test_all_components_cooperation(self, valid_config_file):
        """全コンポーネントの協調動作テスト"""
        engine = VioraTalkEngine(config_path=valid_config_file, first_run=False)

        # 初期化
        await engine.initialize()

        # すべてのPhase 1コンポーネントが初期化されている
        assert engine.config_manager is not None
        assert engine.i18n_manager is not None
        assert engine.background_service_manager is not None

        # 各コンポーネントの状態を確認
        assert engine.config_manager.is_available()
        assert engine.i18n_manager._state == ComponentState.READY
        assert engine.background_service_manager._state == ComponentState.READY

        # クリーンアップも正常に動作
        await engine.cleanup()
        assert engine.get_state() == ComponentState.TERMINATED


# ============================================================================
# パフォーマンステスト
# ============================================================================


@pytest.mark.integration
@pytest.mark.phase(1)
class TestPerformance:
    """パフォーマンス関連のテスト"""

    @pytest.mark.asyncio
    async def test_initialization_time(self, valid_config_file):
        """初期化時間のテスト"""
        engine = VioraTalkEngine(config_path=valid_config_file, first_run=False)

        start_time = time.time()
        await engine.initialize()
        init_time = time.time() - start_time

        # Phase 1では5秒以内に初期化完了を期待
        assert init_time < 5.0

        await engine.cleanup()

    @pytest.mark.asyncio
    async def test_memory_usage(self, valid_config_file):
        """メモリ使用量のテスト"""
        try:
            import psutil

            process = psutil.Process()

            # 初期メモリ使用量
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            engine = VioraTalkEngine(config_path=valid_config_file)
            await engine.initialize()

            # 初期化後のメモリ使用量
            after_init_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = after_init_memory - initial_memory

            # Phase 1では100MB以下の増加を期待
            assert memory_increase < 100

            await engine.cleanup()

        except ImportError:
            pytest.skip("psutil not installed, skipping memory test")


# ============================================================================
# エッジケーステスト
# ============================================================================


@pytest.mark.integration
@pytest.mark.phase(1)
class TestEdgeCases:
    """エッジケースのテスト"""

    @pytest.mark.asyncio
    async def test_multiple_engines_initialization(self, valid_config_file):
        """複数エンジンの同時初期化"""
        engines = []

        # 3つのエンジンを同時に初期化
        for i in range(3):
            engine = VioraTalkEngine(config_path=valid_config_file, first_run=False)
            engines.append(engine)

        # 並行初期化
        await asyncio.gather(*[e.initialize() for e in engines])

        # すべてのエンジンが正常に初期化されている
        for engine in engines:
            assert engine.get_state() == ComponentState.READY

        # クリーンアップ
        await asyncio.gather(*[e.cleanup() for e in engines])

    @pytest.mark.asyncio
    async def test_repeated_initialization(self, valid_config_file):
        """同じエンジンの再初期化防止"""
        engine = VioraTalkEngine(config_path=valid_config_file)

        # 初回初期化
        await engine.initialize()
        assert engine.get_state() == ComponentState.READY

        # 再初期化は何もしない（すでにREADY状態）
        await engine.initialize()
        assert engine.get_state() == ComponentState.READY

        await engine.cleanup()

    @pytest.mark.asyncio
    async def test_cleanup_without_initialization(self):
        """初期化なしでのクリーンアップ"""
        engine = VioraTalkEngine()

        # 初期化せずにクリーンアップ（エラーにならない）
        await engine.cleanup()
        assert engine.get_state() == ComponentState.TERMINATED
