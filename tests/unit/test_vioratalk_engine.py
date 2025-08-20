"""VioraTalkEngineの単体テスト

VioraTalkシステム全体を管理するメインエンジンクラスの動作を検証。
8段階初期化フロー、遅延起動原則、SetupResult型による結果管理をテスト。

テスト戦略ガイドライン v1.7準拠
テスト実装ガイド v1.3準拠
開発規約書 v1.12準拠（単体テストでのモック使用）
"""

from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
import yaml

from vioratalk.core.base import ComponentState
from vioratalk.core.exceptions import ConfigurationError, InitializationError
from vioratalk.core.setup.types import ComponentStatus, SetupResult, SetupStatus
from vioratalk.core.vioratalk_engine import VioraTalkEngine

# ============================================================================
# フィクスチャ
# ============================================================================


@pytest.fixture
def temp_config_dir(tmp_path):
    """一時設定ディレクトリ"""
    config_dir = tmp_path / "user_settings"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


@pytest.fixture
def valid_config(temp_config_dir):
    """有効な設定ファイル"""
    config_path = temp_config_dir / "config.yaml"
    config_data = {
        "general": {"app_name": "VioraTalk", "version": "0.0.1", "language": "ja"},
        "features": {"limited_mode": False, "auto_setup": True},
    }
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f)
    return config_path


@pytest.fixture
def engine():
    """標準のVioraTalkEngineインスタンス"""
    return VioraTalkEngine()


@pytest.fixture
def first_run_engine():
    """初回起動モードのエンジン"""
    return VioraTalkEngine(first_run=True)


@pytest.fixture
def not_first_run_engine():
    """初回起動ではないモードのエンジン"""
    return VioraTalkEngine(first_run=False)


# ============================================================================
# 初期化テスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(1)
class TestVioraTalkEngine:
    """VioraTalkEngineの基本動作テスト"""

    def test_initialization(self):
        """コンストラクタのテスト"""
        engine = VioraTalkEngine()

        assert engine.config_path == Path("user_settings/config.yaml")
        assert engine._state == ComponentState.NOT_INITIALIZED
        assert engine.i18n_manager is None
        assert engine.auto_setup_manager is None
        assert engine.background_service_manager is None
        assert engine._config == {}

    def test_initialization_with_first_run(self):
        """初回起動フラグ付きの初期化"""
        engine = VioraTalkEngine(first_run=True)
        assert engine._first_run_override is True

        engine = VioraTalkEngine(first_run=False)
        assert engine._first_run_override is False

    def test_initialization_default_path(self):
        """デフォルトパスの確認"""
        engine = VioraTalkEngine()
        assert engine.config_path == Path("user_settings/config.yaml")

    @pytest.mark.asyncio
    async def test_initialize_complete_flow(self):
        """完全な初期化フローのテスト"""
        engine = VioraTalkEngine(first_run=False)

        with patch("vioratalk.core.vioratalk_engine.ConfigManager") as MockCM, patch(
            "vioratalk.core.vioratalk_engine.BackgroundServiceManager"
        ) as MockBSM, patch("vioratalk.core.vioratalk_engine.I18nManager") as MockI18n, patch(
            "vioratalk.core.vioratalk_engine.AutoSetupManager"
        ) as MockASM:
            # ConfigManagerのモック
            mock_cm = AsyncMock()
            MockCM.return_value = mock_cm
            mock_cm.initialize = AsyncMock()
            mock_cm.get_all = Mock(return_value={"general": {"language": "ja"}})  # 同期メソッド

            # BackgroundServiceManagerのモック
            mock_bsm = AsyncMock()
            MockBSM.return_value = mock_bsm
            mock_bsm.initialize = AsyncMock()

            # I18nManagerのモック
            mock_i18n = AsyncMock()
            MockI18n.return_value = mock_i18n
            mock_i18n.initialize = AsyncMock()

            # AutoSetupManagerのモック（初回起動ではないので呼ばれない）
            mock_asm = AsyncMock()
            MockASM.return_value = mock_asm

            # 初期化実行
            await engine.initialize()

            # 状態確認
            assert engine._state == ComponentState.READY
            assert engine.config_manager is not None
            assert engine.i18n_manager is not None
            assert engine.background_service_manager is not None

            # 初期化メソッドが呼ばれたことを確認
            mock_cm.initialize.assert_called_once()
            mock_bsm.initialize.assert_called_once()
            mock_i18n.initialize.assert_called_once()


# ============================================================================
# 初回起動判定テスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(1)
class TestFirstRunDetection:
    """初回起動判定のテスト"""

    @pytest.mark.asyncio
    async def test_first_run_override_true(self, first_run_engine):
        """初回起動フラグがTrueの場合"""
        result = await first_run_engine._check_first_run()
        assert result is True

    @pytest.mark.asyncio
    async def test_first_run_override_false(self, not_first_run_engine):
        """初回起動フラグがFalseの場合"""
        result = await not_first_run_engine._check_first_run()
        assert result is False

    @pytest.mark.asyncio
    async def test_first_run_marker_file_exists(self, engine):
        """マーカーファイルが存在する場合"""
        with patch("pathlib.Path.exists", return_value=True):
            result = await engine._check_first_run()
            assert result is False

    @pytest.mark.asyncio
    async def test_first_run_marker_file_not_exists(self, engine):
        """マーカーファイルが存在しない場合"""
        with patch("pathlib.Path.exists", return_value=False):
            result = await engine._check_first_run()
            assert result is True


# ============================================================================
# I18nManager初期化テスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(1)
class TestI18nInitialization:
    """I18nManager初期化のテスト"""

    @pytest.mark.asyncio
    async def test_initialize_i18n_with_config(self, engine):
        """設定に基づいたI18nManager初期化"""
        engine._config = {"general": {"language": "en"}}

        with patch("vioratalk.core.vioratalk_engine.I18nManager") as MockI18n:
            mock_i18n = AsyncMock()
            MockI18n.return_value = mock_i18n
            mock_i18n.initialize = AsyncMock()

            await engine._initialize_i18n_manager()

            # 設定で指定された言語で初期化されることを確認
            MockI18n.assert_called_once_with(language="en")

    @pytest.mark.asyncio
    async def test_initialize_i18n_default_language(self, engine):
        """デフォルト言語でのI18nManager初期化"""
        engine._config = {}  # 空の設定

        with patch("vioratalk.core.vioratalk_engine.I18nManager") as MockI18n:
            mock_i18n = AsyncMock()
            MockI18n.return_value = mock_i18n
            mock_i18n.initialize = AsyncMock()

            await engine._initialize_i18n_manager()

            # デフォルトで日本語で初期化されることを確認
            MockI18n.assert_called_once_with(language="ja")

    @pytest.mark.asyncio
    async def test_initialize_i18n_partial_config(self, engine):
        """部分的な設定の場合"""
        engine._config = {"general": {}}  # languageキーなし

        with patch("vioratalk.core.vioratalk_engine.I18nManager") as MockI18n:
            mock_i18n = AsyncMock()
            MockI18n.return_value = mock_i18n
            mock_i18n.initialize = AsyncMock()

            await engine._initialize_i18n_manager()

            # デフォルトで日本語で初期化されることを確認
            MockI18n.assert_called_once_with(language="ja")


# ============================================================================
# 設定管理テスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(1)
class TestConfiguration:
    """設定管理のテスト"""

    @pytest.mark.asyncio
    async def test_load_configuration_success(self, engine, valid_config):
        """設定ファイル読み込み成功"""
        engine.config_path = valid_config

        with patch("vioratalk.core.vioratalk_engine.ConfigManager") as MockCM:
            mock_cm = AsyncMock()
            MockCM.return_value = mock_cm
            mock_cm.initialize = AsyncMock()
            mock_cm.get_all = Mock(
                return_value={"general": {"language": "ja"}, "features": {"limited_mode": False}}
            )  # 同期メソッド

            await engine._load_configuration()

            assert engine.config_manager is not None
            assert engine._config["general"]["language"] == "ja"
            assert engine._config["features"]["limited_mode"] is False

    @pytest.mark.asyncio
    async def test_load_configuration_file_not_found(self, engine):
        """設定ファイルが見つからない場合"""
        engine.config_path = Path("nonexistent/config.yaml")

        with patch("vioratalk.core.vioratalk_engine.ConfigManager") as MockCM:
            MockCM.side_effect = ConfigurationError("File not found", error_code="E0001")

            # エラーハンドリング指針 v1.20準拠: デフォルト設定で継続
            await engine._load_configuration()

            # デフォルト設定が使用されることを確認
            assert engine._config["general"]["app_name"] == "VioraTalk"

    @pytest.mark.asyncio
    async def test_load_configuration_invalid_yaml(self, temp_config_dir):
        """不正なYAMLファイルの場合"""
        # 不正なYAMLファイルを作成
        invalid_config = temp_config_dir / "invalid.yaml"
        with open(invalid_config, "w", encoding="utf-8") as f:
            f.write("invalid: yaml: content")

        engine = VioraTalkEngine(config_path=invalid_config)

        with patch("vioratalk.core.vioratalk_engine.ConfigManager") as MockCM:
            MockCM.side_effect = ConfigurationError("Parse error", error_code="E0001")

            # エラーハンドリング指針 v1.20準拠: 例外を投げずにデフォルト設定で継続
            await engine._load_configuration()

            # デフォルト設定が使用されることを確認
            assert engine._config["general"]["app_name"] == "VioraTalk"
            assert engine._config["features"]["limited_mode"] is False


# ============================================================================
# 自動セットアップテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(1)
class TestAutoSetup:
    """自動セットアップのテスト"""

    @pytest.mark.asyncio
    async def test_auto_setup_success(self, first_run_engine):
        """自動セットアップ成功"""
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

            await first_run_engine._run_auto_setup()

            assert first_run_engine.auto_setup_manager is not None
            mock_asm.run_initial_setup.assert_called_once()

    @pytest.mark.asyncio
    async def test_auto_setup_partial_success(self, first_run_engine):
        """自動セットアップ部分的成功"""
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
                    ],
                    warnings=["Some components failed"],
                )
            )

            await first_run_engine._run_auto_setup()

            # 部分的成功でも続行
            assert first_run_engine.auto_setup_manager is not None

    @pytest.mark.asyncio
    async def test_auto_setup_error(self, first_run_engine):
        """自動セットアップエラー（E0002）"""
        with patch("vioratalk.core.vioratalk_engine.AutoSetupManager") as MockASM:
            MockASM.side_effect = Exception("Setup failed")

            # エラーでも継続（E0002）
            await first_run_engine._run_auto_setup()

            # エラーが発生してもクラッシュしない
            assert first_run_engine.auto_setup_manager is None


# ============================================================================
# BackgroundServiceManager初期化テスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(1)
class TestBackgroundServiceInitialization:
    """BackgroundServiceManager初期化のテスト"""

    @pytest.mark.asyncio
    async def test_background_service_manager_initialization(self, engine):
        """BackgroundServiceManager初期化（遅延起動）"""
        with patch("vioratalk.core.vioratalk_engine.BackgroundServiceManager") as MockBSM:
            mock_bsm = AsyncMock()
            MockBSM.return_value = mock_bsm
            mock_bsm.initialize = AsyncMock()

            await engine._initialize_background_service_manager()

            assert engine.background_service_manager is not None
            mock_bsm.initialize.assert_called_once()

            # サービスは起動されていない（遅延起動原則）
            assert (
                not hasattr(mock_bsm, "start_service_if_needed")
                or not mock_bsm.start_service_if_needed.called
            )


# ============================================================================
# 状態管理テスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(1)
class TestStateManagement:
    """状態管理のテスト"""

    @pytest.mark.asyncio
    async def test_state_transitions(self, engine):
        """状態遷移のテスト"""
        assert engine.get_state() == ComponentState.NOT_INITIALIZED

        with patch("vioratalk.core.vioratalk_engine.ConfigManager") as MockCM, patch(
            "vioratalk.core.vioratalk_engine.BackgroundServiceManager"
        ) as MockBSM, patch("vioratalk.core.vioratalk_engine.I18nManager") as MockI18n:
            # モックの設定
            mock_cm = AsyncMock()
            MockCM.return_value = mock_cm
            mock_cm.initialize = AsyncMock()
            mock_cm.get_all = Mock(return_value={"general": {"language": "ja"}})  # 同期メソッド

            mock_bsm = AsyncMock()
            MockBSM.return_value = mock_bsm
            mock_bsm.initialize = AsyncMock()

            mock_i18n = AsyncMock()
            MockI18n.return_value = mock_i18n
            mock_i18n.initialize = AsyncMock()

            # 初期化
            await engine.initialize()
            assert engine._state == ComponentState.READY

            # 状態取得
            state = engine.get_state()
            assert state == ComponentState.READY

            # クリーンアップ
            await engine.cleanup()
            assert engine._state == ComponentState.TERMINATED


# ============================================================================
# エラーコードのテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(1)
class TestErrorCodes:
    """エラーコード体系のテスト"""

    @pytest.mark.asyncio
    async def test_general_init_error_E0000(self, engine):
        """E0000: 一般的な初期化エラー"""
        with patch.object(engine, "_load_configuration", side_effect=RuntimeError("Unexpected")):
            with pytest.raises(InitializationError) as exc_info:
                await engine.initialize()

            assert "E0000" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_config_load_error_E0001(self, temp_config_dir):
        """E0001: 設定ファイル読み込みエラー（デフォルト設定で継続）"""
        engine = VioraTalkEngine(config_path=temp_config_dir / "config.yaml")

        with patch("vioratalk.core.vioratalk_engine.ConfigManager") as MockCM:
            MockCM.side_effect = ConfigurationError("Parse error", error_code="E0001")

            # エラーハンドリング指針 v1.20準拠: 例外を投げずにデフォルト設定で継続
            await engine._load_configuration()

            # デフォルト設定が使用されることを確認
            assert engine._config["general"]["app_name"] == "VioraTalk"
            assert engine._config["general"]["version"] == "0.0.1"
            assert engine._config["general"]["language"] == "ja"

    @pytest.mark.asyncio
    async def test_auto_setup_error_E0002(self, first_run_engine):
        """E0002: 自動セットアップエラー（継続）"""
        with patch("vioratalk.core.vioratalk_engine.AutoSetupManager") as MockASM:
            MockASM.side_effect = Exception("Setup error")

            # エラーでも継続
            await first_run_engine._run_auto_setup()

            # エラーコードE0002でログが記録されているはず
            assert first_run_engine.auto_setup_manager is None


# ============================================================================
# クリーンアップテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(1)
class TestCleanup:
    """クリーンアップ処理のテスト"""

    @pytest.mark.asyncio
    async def test_cleanup(self, engine):
        """正常なクリーンアップ"""
        # モックコンポーネントを設定
        engine.config_manager = AsyncMock()
        engine.i18n_manager = AsyncMock()
        engine.auto_setup_manager = AsyncMock()
        engine.background_service_manager = AsyncMock()

        engine._state = ComponentState.READY

        await engine.cleanup()

        # 状態がTERMINATEDになることを確認
        assert engine._state == ComponentState.TERMINATED

        # 各コンポーネントのcleanupが呼ばれたことを確認
        engine.config_manager.cleanup.assert_called_once()
        engine.i18n_manager.cleanup.assert_called_once()
        engine.auto_setup_manager.cleanup.assert_called_once()
        engine.background_service_manager.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_with_errors(self, engine):
        """エラーが発生してもクリーンアップが続行される"""
        # エラーを発生させるモックコンポーネント
        engine.config_manager = AsyncMock()
        engine.config_manager.cleanup.side_effect = RuntimeError("Cleanup error")

        engine.i18n_manager = AsyncMock()
        engine.auto_setup_manager = AsyncMock()
        engine.background_service_manager = AsyncMock()

        engine._state = ComponentState.READY

        await engine.cleanup()

        # エラーが発生しても他のコンポーネントのcleanupが呼ばれる
        assert engine._state == ComponentState.TERMINATED
        engine.i18n_manager.cleanup.assert_called_once()
        engine.auto_setup_manager.cleanup.assert_called_once()
        engine.background_service_manager.cleanup.assert_called_once()


# ============================================================================
# ヘルパーメソッドテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(1)
class TestHelperMethods:
    """ヘルパーメソッドのテスト"""

    def test_get_state(self, engine):
        """状態取得メソッド"""
        assert engine.get_state() == ComponentState.NOT_INITIALIZED

        engine._state = ComponentState.READY
        assert engine.get_state() == ComponentState.READY

    def test_get_config(self, engine):
        """設定取得メソッド"""
        engine._config = {"general": {"app_name": "VioraTalk"}, "features": {"limited_mode": False}}

        config = engine.get_config()
        assert config["general"]["app_name"] == "VioraTalk"
        assert config["features"]["limited_mode"] is False

        # 深いコピーが返されることを確認
        config["general"]["app_name"] = "Modified"
        assert engine._config["general"]["app_name"] == "VioraTalk"

        # ネストされた辞書も深いコピーされていることを確認
        config["features"]["limited_mode"] = True
        assert engine._config["features"]["limited_mode"] is False


# ============================================================================
# Phase 1実装範囲テスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(1)
class TestPhase1Implementation:
    """Phase 1実装範囲のテスト"""

    def test_phase1_components_only(self, engine):
        """Phase 1コンポーネントのみが実装されている"""
        # Phase 1コンポーネント
        assert hasattr(engine, "config_manager")
        assert hasattr(engine, "i18n_manager")
        assert hasattr(engine, "auto_setup_manager")
        assert hasattr(engine, "background_service_manager")

        # Phase 2以降のコンポーネント（プレースホルダー）
        assert engine.model_selector is None
        assert engine.prompt_engine is None
        assert engine.dialogue_manager is None

    @pytest.mark.asyncio
    async def test_minimal_functionality(self, engine):
        """Phase 1の最小機能が動作する"""
        with patch("vioratalk.core.vioratalk_engine.ConfigManager") as MockCM, patch(
            "vioratalk.core.vioratalk_engine.BackgroundServiceManager"
        ) as MockBSM, patch("vioratalk.core.vioratalk_engine.I18nManager") as MockI18n:
            # モックの設定
            mock_cm = AsyncMock()
            MockCM.return_value = mock_cm
            mock_cm.initialize = AsyncMock()
            mock_cm.get_all = Mock(return_value={"general": {"language": "ja"}})  # 同期メソッド

            mock_bsm = AsyncMock()
            MockBSM.return_value = mock_bsm
            mock_bsm.initialize = AsyncMock()

            mock_i18n = AsyncMock()
            MockI18n.return_value = mock_i18n
            mock_i18n.initialize = AsyncMock()

            # 初期化
            await engine.initialize()
            assert engine._state == ComponentState.READY

            # 状態取得
            state = engine.get_state()
            assert state == ComponentState.READY

            # クリーンアップ
            await engine.cleanup()
            assert engine._state == ComponentState.TERMINATED
