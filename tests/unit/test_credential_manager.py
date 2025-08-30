"""credential_manager (APIKeyManager)の単体テスト

APIキー管理機能の動作を検証する。
Phase 4の実装として、BYOK方式、優先順位、バリデーション機能をテスト。

テスト戦略ガイドライン v1.7準拠（Phase 4カバレッジ目標: 65%以上）
テスト実装ガイド v1.3準拠
開発規約書 v1.12準拠
エラーハンドリング指針 v1.20準拠（E2003）
セキュリティ実装ガイド v1.5準拠
"""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from vioratalk.core.exceptions import ConfigurationError
from vioratalk.infrastructure.credential_manager import APIKeyManager, get_api_key_manager

# ============================================================================
# フィクスチャ
# ============================================================================


@pytest.fixture(autouse=True)
def reset_singleton():
    """各テストの前後でシングルトンをリセット

    LoggerManagerのテストパターンを参考に実装
    """
    # テスト前にグローバル変数をリセット
    import vioratalk.infrastructure.credential_manager as cm

    cm._api_key_manager = None

    yield

    # テスト後にもリセット
    cm._api_key_manager = None


@pytest.fixture(autouse=True)
def clean_environment():
    """環境変数をクリーンな状態に保つ

    実環境のAPIキーがテストに影響しないよう、
    各テストの前後で環境変数をリセット
    """
    # 既存の環境変数を保存
    original_env = os.environ.copy()

    # APIキー関連の環境変数をクリア
    for key in [
        "CLAUDE_API_KEY",
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "OLLAMA_BASE_URL",
        "FASTER_WHISPER_API_KEY",
        "EDGE_TTS_API_KEY",
        "AIVISSPEECH_API_KEY",
    ]:
        os.environ.pop(key, None)

    yield

    # 環境変数を復元
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def temp_home_dir():
    """一時的なホームディレクトリ"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_project_dir():
    """一時的なプロジェクトディレクトリ"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_yaml_content():
    """テスト用YAMLコンテンツ"""
    return {
        "llm": {
            "claude_api_key": "sk-claude-test-key",
            "gemini_api_key": "gemini-test-key",
            "openai_api_key": "sk-openai-test-key",
            "ollama_base_url": "http://localhost:11434",
        },
        "stt": {"faster_whisper_api_key": "whisper-test-key"},
        "tts": {"edge_tts_api_key": "edge-test-key", "aivisspeech_api_key": "aivis-test-key"},
    }


@pytest.fixture
def home_yaml_file(temp_home_dir, sample_yaml_content):
    """ホームディレクトリのYAMLファイル"""
    vioratalk_dir = temp_home_dir / ".vioratalk"
    vioratalk_dir.mkdir(parents=True, exist_ok=True)
    yaml_file = vioratalk_dir / "api_keys.yaml"

    with open(yaml_file, "w", encoding="utf-8") as f:
        yaml.dump(sample_yaml_content, f)

    return yaml_file


@pytest.fixture
def project_yaml_file(temp_project_dir, sample_yaml_content):
    """プロジェクトディレクトリのYAMLファイル"""
    settings_dir = temp_project_dir / "user_settings"
    settings_dir.mkdir(parents=True, exist_ok=True)
    yaml_file = settings_dir / "api_keys.yaml"

    # プロジェクト用に少し異なるキーを設定
    project_content = sample_yaml_content.copy()
    project_content["llm"]["claude_api_key"] = "sk-claude-project-key"

    with open(yaml_file, "w", encoding="utf-8") as f:
        yaml.dump(project_content, f)

    return yaml_file


# ============================================================================
# APIKeyManagerの基本テスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestAPIKeyManager:
    """APIKeyManagerクラスの基本テスト"""

    def test_initialization(self):
        """初期化のテスト"""
        with patch.dict(os.environ, {}, clear=True):
            with patch("pathlib.Path.home", return_value=Path("/tmp/nonexistent")):
                with patch("pathlib.Path.exists", return_value=False):
                    manager = APIKeyManager()

                    assert isinstance(manager.keys, dict)
                    assert hasattr(manager, "_cache_valid_until")
                    assert manager.SERVICE_NAME == "VioraTalk"

    def test_supported_services(self):
        """サポートサービス一覧の確認"""
        with patch("pathlib.Path.home", return_value=Path("/tmp/nonexistent")):
            with patch("pathlib.Path.exists", return_value=False):
                manager = APIKeyManager()

                # LLMサービスが含まれているか
                assert "claude" in manager.SUPPORTED_SERVICES
                assert "gemini" in manager.SUPPORTED_SERVICES
                assert "openai" in manager.SUPPORTED_SERVICES
                assert "chatgpt" in manager.SUPPORTED_SERVICES

                # その他のサービス
                assert "faster_whisper" in manager.SUPPORTED_SERVICES
                assert "ollama_base_url" in manager.SUPPORTED_SERVICES

    def test_env_mapping(self):
        """環境変数マッピングの確認"""
        with patch("pathlib.Path.home", return_value=Path("/tmp/nonexistent")):
            with patch("pathlib.Path.exists", return_value=False):
                manager = APIKeyManager()

                assert manager.ENV_MAPPING["claude"] == "CLAUDE_API_KEY"
                assert manager.ENV_MAPPING["gemini"] == "GEMINI_API_KEY"
                assert manager.ENV_MAPPING["openai"] == "OPENAI_API_KEY"
                assert manager.ENV_MAPPING["chatgpt"] == "OPENAI_API_KEY"  # OpenAIと同じ
                assert manager.ENV_MAPPING["ollama_base_url"] == "OLLAMA_BASE_URL"


# ============================================================================
# 環境変数からの読み込みテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestEnvironmentVariableLoading:
    """環境変数からのAPIキー読み込みテスト"""

    def test_load_from_environment(self):
        """環境変数からの読み込み（最優先）"""
        with patch.dict(
            os.environ,
            {
                "CLAUDE_API_KEY": "env-claude-key",
                "GEMINI_API_KEY": "env-gemini-key",
                "OPENAI_API_KEY": "env-openai-key",
                "OLLAMA_BASE_URL": "http://env-ollama:11434",
            },
            clear=True,
        ):
            with patch("pathlib.Path.home", return_value=Path("/tmp/nonexistent")):
                with patch("pathlib.Path.exists", return_value=False):
                    manager = APIKeyManager()

                    assert manager.get_api_key("claude") == "env-claude-key"
                    assert manager.get_api_key("gemini") == "env-gemini-key"
                    assert manager.get_api_key("openai") == "env-openai-key"
                    assert manager.get_api_key("chatgpt") == "env-openai-key"  # OpenAIと同じ
                    assert manager.get_api_key("ollama_base_url") == "http://env-ollama:11434"

    def test_empty_environment_variable(self):
        """空の環境変数は無視される"""
        with patch.dict(
            os.environ,
            {
                "CLAUDE_API_KEY": "",  # 空文字列
                "GEMINI_API_KEY": "   ",  # 空白のみ
            },
            clear=True,
        ):
            with patch("pathlib.Path.home", return_value=Path("/tmp/nonexistent")):
                with patch("pathlib.Path.exists", return_value=False):
                    manager = APIKeyManager()

                    assert manager.get_api_key("claude") is None
                    assert manager.get_api_key("gemini") is None

    def test_environment_variable_trimming(self):
        """環境変数の前後の空白は削除される"""
        with patch.dict(
            os.environ,
            {
                "CLAUDE_API_KEY": "  sk-claude-key-with-spaces  ",
            },
            clear=True,
        ):
            with patch("pathlib.Path.home", return_value=Path("/tmp/nonexistent")):
                with patch("pathlib.Path.exists", return_value=False):
                    manager = APIKeyManager()

                    assert manager.get_api_key("claude") == "sk-claude-key-with-spaces"


# ============================================================================
# YAMLファイルからの読み込みテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestYAMLFileLoading:
    """YAMLファイルからのAPIキー読み込みテスト"""

    def test_load_from_home_yaml(self, home_yaml_file):
        """ホームディレクトリのYAMLから読み込み"""
        with patch.dict(os.environ, {}, clear=True):
            with patch("pathlib.Path.home", return_value=home_yaml_file.parent.parent):
                # home_yaml_fileが存在するので、存在チェックはTrueを返す
                manager = APIKeyManager()

                assert manager.get_api_key("claude") == "sk-claude-test-key"
                assert manager.get_api_key("gemini") == "gemini-test-key"
                assert manager.get_api_key("ollama_base_url") == "http://localhost:11434"

    def test_load_from_project_yaml(self, project_yaml_file):
        """プロジェクトディレクトリのYAMLから読み込み"""
        with patch.dict(os.environ, {}, clear=True):
            # ホームディレクトリは存在しない設定
            with patch("pathlib.Path.home", return_value=Path("/tmp/nonexistent")):
                # project_yaml_fileの親ディレクトリを作業ディレクトリとして設定
                project_root = project_yaml_file.parent.parent
                original_cwd = os.getcwd()

                try:
                    os.chdir(project_root)
                    manager = APIKeyManager()

                    # プロジェクトディレクトリの値が読み込まれる
                    assert manager.get_api_key("claude") == "sk-claude-project-key"
                finally:
                    os.chdir(original_cwd)

    def test_invalid_yaml_format(self, temp_home_dir):
        """不正なYAML形式の処理"""
        vioratalk_dir = temp_home_dir / ".vioratalk"
        vioratalk_dir.mkdir(parents=True, exist_ok=True)
        yaml_file = vioratalk_dir / "api_keys.yaml"

        # 不正なYAMLを作成
        with open(yaml_file, "w") as f:
            f.write("invalid: yaml: content: :")

        with patch.dict(os.environ, {}, clear=True):
            with patch("pathlib.Path.home", return_value=temp_home_dir):
                # エラーが発生してもクラッシュしない
                manager = APIKeyManager()
                assert manager.keys == {}

    def test_permission_error_handling(self, temp_home_dir):
        """権限エラーの処理"""
        vioratalk_dir = temp_home_dir / ".vioratalk"
        vioratalk_dir.mkdir(parents=True, exist_ok=True)
        yaml_file = vioratalk_dir / "api_keys.yaml"
        yaml_file.touch()

        with patch.dict(os.environ, {}, clear=True):
            with patch("pathlib.Path.home", return_value=temp_home_dir):
                with patch("builtins.open", side_effect=PermissionError("Access denied")):
                    # 権限エラーが発生してもクラッシュしない
                    manager = APIKeyManager()
                    assert manager.keys == {}


# ============================================================================
# 優先順位テスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestPriorityOrder:
    """APIキー読み込みの優先順位テスト"""

    def test_environment_overrides_yaml(self, temp_home_dir, temp_project_dir):
        """環境変数がYAMLより優先される"""
        # ホームディレクトリのYAML準備
        home_vioratalk = temp_home_dir / ".vioratalk"
        home_vioratalk.mkdir(parents=True, exist_ok=True)
        home_yaml = home_vioratalk / "api_keys.yaml"
        with open(home_yaml, "w") as f:
            yaml.dump(
                {"llm": {"claude_api_key": "home-claude", "gemini_api_key": "home-gemini"}}, f
            )

        # プロジェクトディレクトリのYAML準備
        project_settings = temp_project_dir / "user_settings"
        project_settings.mkdir(parents=True, exist_ok=True)
        project_yaml = project_settings / "api_keys.yaml"
        with open(project_yaml, "w") as f:
            yaml.dump({"llm": {"claude_api_key": "project-claude"}}, f)

        with patch.dict(
            os.environ,
            {
                "CLAUDE_API_KEY": "env-claude-override",
            },
            clear=True,
        ):
            with patch("pathlib.Path.home", return_value=temp_home_dir):
                original_cwd = os.getcwd()
                try:
                    os.chdir(temp_project_dir)
                    manager = APIKeyManager()

                    # 環境変数の値が優先される
                    assert manager.get_api_key("claude") == "env-claude-override"
                    # 環境変数にないものはYAMLから
                    assert manager.get_api_key("gemini") == "home-gemini"
                finally:
                    os.chdir(original_cwd)

    def test_home_overrides_project(self, temp_home_dir, temp_project_dir):
        """ホームディレクトリがプロジェクトより優先される"""
        # ホームディレクトリのYAML
        home_vioratalk = temp_home_dir / ".vioratalk"
        home_vioratalk.mkdir(parents=True, exist_ok=True)
        home_yaml = home_vioratalk / "api_keys.yaml"
        with open(home_yaml, "w") as f:
            yaml.dump({"llm": {"claude_api_key": "home-claude"}}, f)

        # プロジェクトディレクトリのYAML
        project_settings = temp_project_dir / "user_settings"
        project_settings.mkdir(parents=True, exist_ok=True)
        project_yaml = project_settings / "api_keys.yaml"
        with open(project_yaml, "w") as f:
            yaml.dump(
                {"llm": {"claude_api_key": "project-claude", "gemini_api_key": "project-gemini"}}, f
            )

        with patch.dict(os.environ, {}, clear=True):
            with patch("pathlib.Path.home", return_value=temp_home_dir):
                original_cwd = os.getcwd()
                try:
                    os.chdir(temp_project_dir)
                    manager = APIKeyManager()

                    # ホームディレクトリの値が優先される
                    assert manager.get_api_key("claude") == "home-claude"
                    # ホームにないものはプロジェクトから
                    assert manager.get_api_key("gemini") == "project-gemini"
                finally:
                    os.chdir(original_cwd)


# ============================================================================
# APIキー取得・要求テスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestAPIKeyRetrieval:
    """APIキー取得機能のテスト"""

    def test_get_api_key_exists(self):
        """存在するAPIキーの取得"""
        with patch.dict(os.environ, {"CLAUDE_API_KEY": "test-key"}, clear=True):
            with patch("pathlib.Path.home", return_value=Path("/tmp/nonexistent")):
                with patch("pathlib.Path.exists", return_value=False):
                    manager = APIKeyManager()

                    key = manager.get_api_key("claude")
                    assert key == "test-key"

    def test_get_api_key_not_exists(self):
        """存在しないAPIキーの取得"""
        with patch.dict(os.environ, {}, clear=True):
            with patch("pathlib.Path.home", return_value=Path("/tmp/nonexistent")):
                with patch("pathlib.Path.exists", return_value=False):
                    manager = APIKeyManager()

                    key = manager.get_api_key("claude")
                    assert key is None

    def test_require_api_key_exists(self):
        """require_api_key: キーが存在する場合"""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "required-key"}, clear=True):
            with patch("pathlib.Path.home", return_value=Path("/tmp/nonexistent")):
                with patch("pathlib.Path.exists", return_value=False):
                    manager = APIKeyManager()

                    key = manager.require_api_key("gemini")
                    assert key == "required-key"

    def test_require_api_key_not_exists(self):
        """require_api_key: キーが存在しない場合（E2003エラー）"""
        with patch.dict(os.environ, {}, clear=True):
            with patch("pathlib.Path.home", return_value=Path("/tmp/nonexistent")):
                with patch("pathlib.Path.exists", return_value=False):
                    manager = APIKeyManager()

                    with pytest.raises(ConfigurationError) as exc_info:
                        manager.require_api_key("claude")

                    # エラーコードE2003の確認
                    assert exc_info.value.error_code == "E2003"
                    assert "CLAUDE_API_KEY" in str(exc_info.value)
                    assert "api_keys.yaml" in str(exc_info.value)

    def test_has_service_key(self):
        """has_service_key メソッドのテスト"""
        with patch.dict(os.environ, {"CLAUDE_API_KEY": "test-key"}, clear=True):
            with patch("pathlib.Path.home", return_value=Path("/tmp/nonexistent")):
                with patch("pathlib.Path.exists", return_value=False):
                    manager = APIKeyManager()

                    assert manager.has_service_key("claude") is True
                    assert manager.has_service_key("gemini") is False

    def test_get_available_services(self):
        """get_available_services メソッドのテスト"""
        with patch.dict(
            os.environ, {"CLAUDE_API_KEY": "claude-key", "GEMINI_API_KEY": "gemini-key"}, clear=True
        ):
            with patch("pathlib.Path.home", return_value=Path("/tmp/nonexistent")):
                with patch("pathlib.Path.exists", return_value=False):
                    manager = APIKeyManager()

                    services = manager.get_available_services()
                    assert "claude" in services
                    assert "gemini" in services
                    assert len(services) == 2


# ============================================================================
# バリデーションテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestValidation:
    """APIキーバリデーション機能のテスト"""

    def test_validate_with_claude_key(self):
        """Claudeキーがある場合のバリデーション"""
        with patch.dict(os.environ, {"CLAUDE_API_KEY": "valid-key"}, clear=True):
            with patch("pathlib.Path.home", return_value=Path("/tmp/nonexistent")):
                with patch("pathlib.Path.exists", return_value=False):
                    manager = APIKeyManager()

                    results = manager.validate_api_keys()

                    assert results["claude"] is True
                    assert results["gemini"] is False
                    assert results["openai"] is False

    def test_validate_with_multiple_keys(self):
        """複数のLLMキーがある場合"""
        with patch.dict(
            os.environ,
            {
                "CLAUDE_API_KEY": "claude-key",
                "GEMINI_API_KEY": "gemini-key",
                "OPENAI_API_KEY": "openai-key",
            },
            clear=True,
        ):
            with patch("pathlib.Path.home", return_value=Path("/tmp/nonexistent")):
                with patch("pathlib.Path.exists", return_value=False):
                    manager = APIKeyManager()

                    results = manager.validate_api_keys()

                    assert results["claude"] is True
                    assert results["gemini"] is True
                    assert results["openai"] is True
                    assert results["chatgpt"] is True  # OpenAIと同じ

    def test_validate_with_ollama_only(self):
        """Ollamaのみの場合もバリデーション成功"""
        with patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://localhost:11434"}, clear=True):
            with patch("pathlib.Path.home", return_value=Path("/tmp/nonexistent")):
                with patch("pathlib.Path.exists", return_value=False):
                    manager = APIKeyManager()

                    results = manager.validate_api_keys()

                    assert results["ollama"] is True
                    assert results["claude"] is False
                    assert results["gemini"] is False

    def test_validate_no_llm_keys(self):
        """LLMキーが1つもない場合（E2003エラー）"""
        with patch.dict(os.environ, {}, clear=True):
            with patch("pathlib.Path.home", return_value=Path("/tmp/nonexistent")):
                with patch("pathlib.Path.exists", return_value=False):
                    manager = APIKeyManager()

                    with pytest.raises(ConfigurationError) as exc_info:
                        manager.validate_api_keys()

                    # エラーコードE2003の確認
                    assert exc_info.value.error_code == "E2003"
                    assert "LLM APIキー" in str(exc_info.value)
                    assert "環境変数" in str(exc_info.value)


# ============================================================================
# ユーティリティメソッドテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestUtilityMethods:
    """ユーティリティメソッドのテスト"""

    def test_mask_api_key_normal(self):
        """通常のAPIキーマスキング"""
        with patch("pathlib.Path.home", return_value=Path("/tmp/nonexistent")):
            with patch("pathlib.Path.exists", return_value=False):
                manager = APIKeyManager()

                # 長いキー
                masked = manager.mask_api_key("sk-1234567890abcdefghijklmnop")
                assert masked == "sk-1...mnop"

                # 中程度のキー
                masked = manager.mask_api_key("gemini-api-key-123")
                assert masked == "ge...23"

    def test_mask_api_key_short(self):
        """短いAPIキーのマスキング"""
        with patch("pathlib.Path.home", return_value=Path("/tmp/nonexistent")):
            with patch("pathlib.Path.exists", return_value=False):
                manager = APIKeyManager()

                # 8文字未満
                masked = manager.mask_api_key("short")
                assert masked == "***"

                # 空文字列
                masked = manager.mask_api_key("")
                assert masked == "***"

                # None
                masked = manager.mask_api_key(None)
                assert masked == "***"

    def test_get_summary(self):
        """get_summary メソッドのテスト"""
        with patch.dict(
            os.environ,
            {
                "CLAUDE_API_KEY": "sk-claude-very-long-key-12345",
                "OLLAMA_BASE_URL": "http://localhost:11434",
            },
            clear=True,
        ):
            with patch("pathlib.Path.home", return_value=Path("/tmp/nonexistent")):
                with patch("pathlib.Path.exists", return_value=False):
                    manager = APIKeyManager()

                    summary = manager.get_summary()

                    # Claudeキーはマスキングされる
                    assert summary["claude"] == "sk-c...2345"
                    # OllamaのURLはマスキングされない
                    assert summary["ollama_base_url"] == "http://localhost:11434"

    def test_refresh(self):
        """refresh メソッドのテスト"""
        # 初期状態
        with patch.dict(os.environ, {"CLAUDE_API_KEY": "initial-key"}, clear=True):
            with patch("pathlib.Path.home", return_value=Path("/tmp/nonexistent")):
                with patch("pathlib.Path.exists", return_value=False):
                    manager = APIKeyManager()
                    assert manager.get_api_key("claude") == "initial-key"

                    # 環境変数を変更
                    os.environ["CLAUDE_API_KEY"] = "updated-key"
                    manager.refresh()
                    assert manager.get_api_key("claude") == "updated-key"


# ============================================================================
# シングルトンパターンテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestSingletonPattern:
    """シングルトンパターンのテスト"""

    def test_get_api_key_manager_singleton(self):
        """get_api_key_manager がシングルトンを返す"""
        with patch("pathlib.Path.home", return_value=Path("/tmp/nonexistent")):
            with patch("pathlib.Path.exists", return_value=False):
                manager1 = get_api_key_manager()
                manager2 = get_api_key_manager()

                assert manager1 is manager2
                assert id(manager1) == id(manager2)

    def test_singleton_persistence(self):
        """シングルトンが値を保持する"""
        with patch.dict(os.environ, {"CLAUDE_API_KEY": "singleton-key"}, clear=True):
            with patch("pathlib.Path.home", return_value=Path("/tmp/nonexistent")):
                with patch("pathlib.Path.exists", return_value=False):
                    manager1 = get_api_key_manager()
                    key1 = manager1.get_api_key("claude")

                    # 環境変数をクリアしても、シングルトンは値を保持
                    os.environ.pop("CLAUDE_API_KEY", None)
                    manager2 = get_api_key_manager()
                    key2 = manager2.get_api_key("claude")

                    assert key1 == key2 == "singleton-key"


# ============================================================================
# 統合テスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestIntegration:
    """統合的な動作テスト"""

    def test_complete_priority_chain(self, temp_home_dir, temp_project_dir):
        """完全な優先順位チェーンのテスト"""
        # 1. プロジェクトYAML（最低優先度）
        project_settings = temp_project_dir / "user_settings"
        project_settings.mkdir(parents=True, exist_ok=True)
        project_yaml = project_settings / "api_keys.yaml"
        with open(project_yaml, "w") as f:
            yaml.dump(
                {
                    "llm": {
                        "claude_api_key": "project-claude",
                        "gemini_api_key": "project-gemini",
                        "openai_api_key": "project-openai",
                    }
                },
                f,
            )

        # 2. ホームYAML（中優先度）
        home_vioratalk = temp_home_dir / ".vioratalk"
        home_vioratalk.mkdir(parents=True, exist_ok=True)
        home_yaml = home_vioratalk / "api_keys.yaml"
        with open(home_yaml, "w") as f:
            yaml.dump(
                {"llm": {"claude_api_key": "home-claude", "gemini_api_key": "home-gemini"}}, f
            )

        # 3. 環境変数（最高優先度）
        with patch.dict(os.environ, {"CLAUDE_API_KEY": "env-claude"}, clear=True):
            with patch("pathlib.Path.home", return_value=temp_home_dir):
                original_cwd = os.getcwd()
                try:
                    os.chdir(temp_project_dir)
                    manager = APIKeyManager()

                    # 優先順位の確認
                    assert manager.get_api_key("claude") == "env-claude"  # 環境変数
                    assert manager.get_api_key("gemini") == "home-gemini"  # ホーム
                    assert manager.get_api_key("openai") == "project-openai"  # プロジェクト
                finally:
                    os.chdir(original_cwd)

    def test_real_world_scenario(self):
        """実際の使用シナリオのテスト"""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "real-gemini-key"}, clear=True):
            with patch("pathlib.Path.home", return_value=Path("/tmp/nonexistent")):
                with patch("pathlib.Path.exists", return_value=False):
                    manager = APIKeyManager()

                    # 1. バリデーション
                    results = manager.validate_api_keys()
                    assert results["gemini"] is True

                    # 2. APIキー取得（エンジン使用時）
                    api_key = manager.require_api_key("gemini")
                    assert api_key == "real-gemini-key"

                    # 3. デバッグ用サマリー取得
                    summary = manager.get_summary()
                    assert "gemini" in summary
                    assert summary["gemini"] == "re...ey"  # マスキングされている（末尾2文字）

                    # 4. 利用可能サービス確認
                    services = manager.get_available_services()
                    assert "gemini" in services
