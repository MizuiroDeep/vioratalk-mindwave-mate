"""ConfigManagerの単体テスト

設定管理機能の動作を検証する。
Phase 1の基本実装として、設定の読み込み、保存、検証機能に焦点を当てる。

テスト戦略ガイドライン v1.7準拠
テスト実装ガイド v1.3準拠
開発規約書 v1.12準拠

NOTE: Phase 1最小実装のため、ComponentState.SHUTDOWNを使用。
      Phase 2でTERMINATING、is_operational()のテストを追加予定。
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from vioratalk.configuration.config_manager import ConfigManager
from vioratalk.core.base import ComponentState
from vioratalk.core.exceptions import ConfigurationError

# ============================================================================
# ConfigManagerの基本テスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(1)
class TestConfigManager:
    """ConfigManagerクラスの基本テスト"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """各テストの前後処理"""
        # 一時ディレクトリとファイルの作成
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_path = self.temp_dir / "test_config.yaml"

        yield

        # クリーンアップ
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @pytest.fixture
    def config_manager(self):
        """ConfigManagerのフィクスチャ"""
        return ConfigManager(self.config_path)

    @pytest.fixture
    def sample_config(self):
        """サンプル設定のフィクスチャ"""
        return {
            "general": {"app_name": "VioraTalk Test", "language": "ja", "theme": "light"},
            "stt": {"engine": "faster-whisper", "model": "large", "device": "cuda"},
            "llm": {"provider": "ollama", "model": "llama3", "temperature": 0.8},
        }

    # ------------------------------------------------------------------------
    # 初期化のテスト
    # ------------------------------------------------------------------------

    def test_initialization(self):
        """ConfigManagerの初期化"""
        manager = ConfigManager(self.config_path)

        assert manager.config_path == self.config_path
        assert manager._config == {}
        assert manager._state == ComponentState.NOT_INITIALIZED

    def test_initialization_with_default_path(self):
        """デフォルトパスでの初期化"""
        manager = ConfigManager()

        # DEFAULT_CONFIG_PATHは絶対パスを返すため、
        # config_pathも絶対パスになることを確認
        from vioratalk.configuration.settings import DEFAULT_CONFIG_PATH

        assert manager.config_path == DEFAULT_CONFIG_PATH

    @pytest.mark.asyncio
    async def test_async_initialize_success(self, sample_config):
        """非同期初期化の成功"""
        # 設定ファイルを作成
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(sample_config, f)

        manager = ConfigManager(self.config_path)
        await manager.initialize()

        assert manager._state == ComponentState.READY
        assert manager._config["general"]["app_name"] == "VioraTalk Test"

    @pytest.mark.asyncio
    async def test_async_initialize_with_missing_file(self):
        """設定ファイルが存在しない場合の初期化"""
        manager = ConfigManager(Path("nonexistent/config.yaml"))
        await manager.initialize()

        # デフォルト設定が使用される
        assert manager._state == ComponentState.READY
        assert manager._config["general"]["app_name"] == "VioraTalk"

    @pytest.mark.asyncio
    async def test_async_initialize_with_invalid_yaml(self):
        """無効なYAMLファイルでの初期化エラー"""
        # 無効なYAMLを作成
        self.config_path.write_text("invalid: yaml: content:", encoding="utf-8")

        manager = ConfigManager(self.config_path)

        with pytest.raises(ConfigurationError) as exc_info:
            await manager.initialize()

        assert manager._state == ComponentState.ERROR
        assert "E0001" in str(exc_info.value)

    # ------------------------------------------------------------------------
    # 設定ファイル読み込みのテスト
    # ------------------------------------------------------------------------

    def test_load_config_success(self, config_manager, sample_config):
        """設定ファイルの正常読み込み"""
        # 設定ファイルを作成
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(sample_config, f)

        config = config_manager.load_config(self.config_path)

        assert config["general"]["app_name"] == "VioraTalk Test"
        assert config["stt"]["engine"] == "faster-whisper"

    def test_load_config_with_defaults(self, config_manager):
        """デフォルト設定とのマージ"""
        # 部分的な設定ファイルを作成
        partial_config = {"general": {"language": "en"}}  # languageのみ変更
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(partial_config, f)

        config = config_manager.load_config(self.config_path)

        # デフォルト値が保持される
        assert config["general"]["app_name"] == "VioraTalk"
        # 上書きされた値
        assert config["general"]["language"] == "en"
        # デフォルトのまま
        assert config["stt"]["engine"] == "faster-whisper"

    def test_load_config_file_not_found(self, config_manager):
        """存在しないファイルの読み込み"""
        config = config_manager.load_config(Path("nonexistent.yaml"))

        # デフォルト設定が返される
        assert config["general"]["app_name"] == "VioraTalk"

    def test_load_config_empty_file(self, config_manager):
        """空のYAMLファイルの読み込み"""
        self.config_path.write_text("", encoding="utf-8")

        config = config_manager.load_config(self.config_path)

        # デフォルト設定が返される
        assert config["general"]["app_name"] == "VioraTalk"

    def test_load_config_yaml_error(self, config_manager):
        """不正なYAMLファイルの読み込み"""
        self.config_path.write_text("invalid: yaml: content:", encoding="utf-8")

        with pytest.raises(ConfigurationError) as exc_info:
            config_manager.load_config(self.config_path)

        assert "E0001" in str(exc_info.value)
        assert "Failed to parse YAML" in str(exc_info.value)

    # ------------------------------------------------------------------------
    # 設定ファイル保存のテスト
    # ------------------------------------------------------------------------

    def test_save_config_success(self, config_manager, sample_config):
        """設定ファイルの正常保存"""
        config_manager.save_config(sample_config, self.config_path)

        # ファイルが作成されている
        assert self.config_path.exists()

        # 内容を確認
        with open(self.config_path, "r", encoding="utf-8") as f:
            saved_config = yaml.safe_load(f)

        assert saved_config["general"]["app_name"] == "VioraTalk Test"

    def test_save_config_creates_directory(self, config_manager, sample_config):
        """ディレクトリが存在しない場合の作成"""
        new_path = self.temp_dir / "subdir" / "config.yaml"

        config_manager.save_config(sample_config, new_path)

        assert new_path.exists()
        assert new_path.parent.exists()

    def test_save_config_overwrite(self, config_manager, sample_config):
        """既存ファイルの上書き"""
        # 最初の保存
        config_manager.save_config({"old": "data"}, self.config_path)

        # 上書き保存
        config_manager.save_config(sample_config, self.config_path)

        # 新しい内容になっている
        with open(self.config_path, "r", encoding="utf-8") as f:
            saved_config = yaml.safe_load(f)

        assert "old" not in saved_config
        assert saved_config["general"]["app_name"] == "VioraTalk Test"

    def test_save_config_write_error(self, config_manager, sample_config):
        """書き込みエラーの処理"""
        # 書き込み権限エラーをシミュレート
        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            with pytest.raises(ConfigurationError) as exc_info:
                config_manager.save_config(sample_config, self.config_path)

            assert "E0001" in str(exc_info.value)
            assert "Permission denied" in str(exc_info.value)

    # ------------------------------------------------------------------------
    # 設定値取得/更新のテスト
    # ------------------------------------------------------------------------

    def test_get_single_key(self, config_manager):
        """単一キーの取得"""
        config_manager._config = {"key": "value"}

        assert config_manager.get("key") == "value"
        assert config_manager.get("missing") is None
        assert config_manager.get("missing", "default") == "default"

    def test_get_nested_key(self, config_manager):
        """ネストされたキーの取得（ドット記法）"""
        config_manager._config = {"general": {"language": "ja", "theme": "dark"}}

        assert config_manager.get("general.language") == "ja"
        assert config_manager.get("general.theme") == "dark"
        assert config_manager.get("general.missing") is None

    def test_get_deep_nested_key(self, config_manager):
        """深くネストされたキーの取得"""
        config_manager._config = {"level1": {"level2": {"level3": {"value": "deep"}}}}

        assert config_manager.get("level1.level2.level3.value") == "deep"

    def test_get_nonexistent_key(self, config_manager):
        """存在しないキーの取得"""
        config_manager._config = {"key": "value"}

        assert config_manager.get("nonexistent") is None
        assert config_manager.get("nonexistent", "default") == "default"

    def test_get_nonexistent_nested_key(self, config_manager):
        """存在しないネストキーの取得"""
        config_manager._config = {"general": {}}

        assert config_manager.get("general.nonexistent") is None
        assert config_manager.get("general.nonexistent.deep") is None

    def test_set_single_key(self, config_manager):
        """単一キーの設定"""
        config_manager.set("key", "value")

        assert config_manager._config["key"] == "value"

    def test_set_nested_key(self, config_manager):
        """ネストされたキーの設定（ドット記法）"""
        config_manager.set("general.language", "en")

        assert config_manager._config["general"]["language"] == "en"

    def test_set_deep_nested_key(self, config_manager):
        """深くネストされたキーの設定"""
        config_manager.set("level1.level2.level3.value", "deep")

        assert config_manager._config["level1"]["level2"]["level3"]["value"] == "deep"

    def test_set_creates_missing_structure(self, config_manager):
        """存在しない構造の自動作成"""
        config_manager._config = {}

        config_manager.set("new.path.to.value", "created")

        assert config_manager._config["new"]["path"]["to"]["value"] == "created"

    def test_set_overwrites_non_dict(self, config_manager):
        """辞書でない値の上書き"""
        config_manager._config = {"path": "not_a_dict"}

        config_manager.set("path.to.value", "new")

        assert isinstance(config_manager._config["path"], dict)
        assert config_manager._config["path"]["to"]["value"] == "new"

    # ------------------------------------------------------------------------
    # get_all()メソッドのテスト（新規追加）
    # ------------------------------------------------------------------------

    def test_get_all(self, config_manager):
        """get_all()メソッドの基本テスト"""
        test_config = {
            "general": {"app_name": "Test", "language": "ja"},
            "features": {"limited_mode": False},
        }
        config_manager._config = test_config

        result = config_manager.get_all()

        # 内容が同じであることを確認
        assert result == test_config
        # 別のオブジェクトであることを確認
        assert result is not test_config

    def test_get_all_returns_deep_copy(self, config_manager):
        """get_all()が深いコピーを返すことを確認"""
        test_config = {"general": {"app_name": "Original", "settings": {"nested": "value"}}}
        config_manager._config = test_config

        result = config_manager.get_all()

        # 返された設定を変更
        result["general"]["app_name"] = "Modified"
        result["general"]["settings"]["nested"] = "changed"

        # 元の設定が変更されていないことを確認
        assert config_manager._config["general"]["app_name"] == "Original"
        assert config_manager._config["general"]["settings"]["nested"] == "value"

    def test_get_all_with_empty_config(self, config_manager):
        """空の設定でのget_all()"""
        config_manager._config = {}

        result = config_manager.get_all()

        assert result == {}
        assert result is not config_manager._config

    def test_get_config_compatibility(self, config_manager):
        """get_config()メソッドの互換性確認"""
        test_config = {"test": "data"}
        config_manager._config = test_config

        # get_config()とget_all()が同じ動作をすることを確認
        result1 = config_manager.get_config()
        result2 = config_manager.get_all()

        assert result1 == result2
        assert result1 is not test_config
        assert result2 is not test_config

    # ------------------------------------------------------------------------
    # 設定検証のテスト
    # ------------------------------------------------------------------------

    def test_validate_config_valid(self, config_manager):
        """有効な設定の検証"""
        config = {"general": {"language": "ja"}}

        errors = config_manager.validate_config(config)

        assert len(errors) == 0

    def test_validate_config_missing_general(self, config_manager):
        """generalセクションが欠落"""
        config = {"stt": {}}

        errors = config_manager.validate_config(config)

        assert len(errors) == 1
        assert "Missing required section: general" in errors[0]

    def test_validate_config_invalid_language(self, config_manager):
        """無効な言語設定"""
        config = {"general": {"language": "invalid"}}

        errors = config_manager.validate_config(config)

        assert len(errors) == 1
        assert "Invalid language: invalid" in errors[0]

    def test_validate_config_invalid_stt_device(self, config_manager):
        """無効なSTTデバイス設定"""
        config = {"general": {"language": "ja"}, "stt": {"device": "invalid"}}

        errors = config_manager.validate_config(config)

        assert len(errors) == 1
        assert "Invalid STT device: invalid" in errors[0]

    def test_validate_config_invalid_temperature(self, config_manager):
        """無効なLLM temperature設定"""
        config = {"general": {"language": "ja"}, "llm": {"temperature": 3.0}}  # 範囲外

        errors = config_manager.validate_config(config)

        assert len(errors) == 1
        assert "Invalid LLM temperature: 3.0" in errors[0]

    def test_validate_config_multiple_errors(self, config_manager):
        """複数のエラーがある場合"""
        config = {"stt": {"device": "invalid"}, "llm": {"temperature": -1}}

        errors = config_manager.validate_config(config)

        assert len(errors) >= 2  # generalセクション欠如 + その他のエラー

    # ------------------------------------------------------------------------
    # その他のメソッドのテスト
    # ------------------------------------------------------------------------

    def test_is_available(self, config_manager):
        """利用可能状態の確認

        NOTE: Phase 1最小実装のため、READYのみチェック。
              Phase 2でRUNNINGとis_operational()のテストを追加予定。
        """
        # 初期状態
        assert not config_manager.is_available()

        # READY状態
        config_manager._state = ComponentState.READY
        assert config_manager.is_available()

        # ERROR状態
        config_manager._state = ComponentState.ERROR
        assert not config_manager.is_available()

        # Phase 2で追加予定:
        # config_manager._state = ComponentState.RUNNING
        # assert config_manager.is_available()

    def test_get_status(self, config_manager):
        """ステータス情報の取得"""
        config_manager._state = ComponentState.READY
        config_manager._config = {"test": "data"}

        status = config_manager.get_status()

        assert status["state"] == ComponentState.READY
        assert status["is_available"] is True
        assert status["config_loaded"] is True
        assert str(self.config_path) in status["config_path"]

    @pytest.mark.asyncio
    async def test_cleanup(self, config_manager):
        """クリーンアップ処理

        NOTE: Phase 1最小実装のため、SHUTDOWNを使用。
              Phase 2でTERMINATINGのテストを追加予定。
        """
        config_manager._state = ComponentState.READY

        await config_manager.cleanup()

        assert config_manager._state == ComponentState.TERMINATED

    # ------------------------------------------------------------------------
    # プライベートメソッドのテスト
    # ------------------------------------------------------------------------

    def test_deep_merge(self, config_manager):
        """辞書の再帰的マージ"""
        base = {"level1": {"a": 1, "b": 2, "nested": {"x": 10}}, "level2": "value"}

        override = {
            "level1": {"b": 20, "c": 3, "nested": {"y": 20}},  # 上書き  # 追加  # 追加
            "level3": "new",  # 追加
        }

        result = config_manager._deep_merge(base, override)

        assert result["level1"]["a"] == 1  # 保持
        assert result["level1"]["b"] == 20  # 上書き
        assert result["level1"]["c"] == 3  # 追加
        assert result["level1"]["nested"]["x"] == 10  # 保持
        assert result["level1"]["nested"]["y"] == 20  # 追加
        assert result["level2"] == "value"  # 保持
        assert result["level3"] == "new"  # 追加

    def test_get_default_config(self, config_manager):
        """デフォルト設定の取得"""
        defaults = config_manager._get_default_config()

        # 必須フィールドの確認
        assert "general" in defaults
        assert "stt" in defaults
        assert "llm" in defaults
        assert "tts" in defaults
        assert "features" in defaults
        assert "paths" in defaults

        # デフォルト値の確認
        assert defaults["general"]["language"] == "ja"
        assert defaults["features"]["background_service"] is False
