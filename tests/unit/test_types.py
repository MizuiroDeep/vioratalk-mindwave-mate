"""SetupResult関連の型定義テスト

SetupStatus, ComponentStatus, SetupResult型の動作を検証。
自動セットアップガイド v1.2の仕様に完全準拠。

テスト戦略ガイドライン v1.7準拠
テスト実装ガイド v1.3準拠
開発規約書 v1.12準拠
"""

from datetime import datetime

import pytest

from vioratalk.core.setup.types import ComponentStatus, SetupResult, SetupStatus

# ============================================================================
# SetupStatus Enumのテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(1)
class TestSetupStatus:
    """SetupStatus Enumのテスト"""

    def test_enum_values(self):
        """Enum値が仕様書通りに定義されているか"""
        # 自動セットアップガイド v1.2準拠の値
        assert SetupStatus.SUCCESS.value == "success"
        assert SetupStatus.PARTIAL_SUCCESS.value == "partial_success"
        assert SetupStatus.SKIPPED.value == "skipped"
        assert SetupStatus.FAILED.value == "failed"
        assert SetupStatus.NOT_STARTED.value == "not_started"

    def test_enum_count(self):
        """Enumの要素数が正しいか"""
        assert len(SetupStatus) == 5

    def test_enum_members(self):
        """すべてのメンバーが存在するか"""
        members = [status.name for status in SetupStatus]
        expected = ["SUCCESS", "PARTIAL_SUCCESS", "SKIPPED", "FAILED", "NOT_STARTED"]
        assert set(members) == set(expected)

    def test_no_in_progress_status(self):
        """IN_PROGRESSが存在しないことを確認（仕様書準拠）"""
        # IN_PROGRESSは存在してはいけない
        with pytest.raises(AttributeError):
            _ = SetupStatus.IN_PROGRESS


# ============================================================================
# ComponentStatusのテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(1)
class TestComponentStatus:
    """ComponentStatusクラスのテスト"""

    def test_basic_initialization(self):
        """基本的な初期化"""
        component = ComponentStatus(name="Ollama", installed=True, version="0.1.24")

        assert component.name == "Ollama"
        assert component.installed is True
        assert component.version == "0.1.24"
        assert component.error is None
        assert component.install_path is None
        assert component.metadata == {}

    def test_initialization_with_all_fields(self):
        """全フィールドを指定した初期化"""
        component = ComponentStatus(
            name="AivisSpeech",
            installed=False,
            version=None,
            error="Installation failed: Network error",
            install_path="/opt/aivisspeech",
            metadata={"retry_count": 3, "last_attempt": "2025-08-18"},
        )

        assert component.name == "AivisSpeech"
        assert component.installed is False
        assert component.version is None
        assert component.error == "Installation failed: Network error"
        assert component.install_path == "/opt/aivisspeech"
        assert component.metadata["retry_count"] == 3

    def test_installed_field_is_bool(self):
        """installedフィールドがbool型であることを確認"""
        # 仕様書準拠：installed: bool（status: SetupStatusではない！）
        component = ComponentStatus(name="Test", installed=True)
        assert isinstance(component.installed, bool)

        component2 = ComponentStatus(name="Test2", installed=False)
        assert isinstance(component2.installed, bool)

    def test_post_init_validation_success(self):
        """__post_init__での検証 - 成功時"""
        # インストール成功時、versionがNoneなら"unknown"に設定
        component = ComponentStatus(name="Python", installed=True, version=None)
        assert component.version == "unknown"

    def test_post_init_validation_failure(self):
        """__post_init__での検証 - 失敗時"""
        # インストール失敗時、errorがNoneならデフォルトメッセージ設定
        component = ComponentStatus(name="Ollama", installed=False, error=None)
        assert component.error == "Unknown installation error"

    def test_metadata_default_factory(self):
        """metadataのdefault_factoryが正しく動作するか"""
        component1 = ComponentStatus(name="Test1", installed=True)
        component2 = ComponentStatus(name="Test2", installed=True)

        # 各インスタンスが独立したdictを持つ
        component1.metadata["key"] = "value1"
        component2.metadata["key"] = "value2"

        assert component1.metadata["key"] == "value1"
        assert component2.metadata["key"] == "value2"


# ============================================================================
# SetupResultのテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(1)
class TestSetupResult:
    """SetupResultクラスのテスト"""

    def test_default_initialization(self):
        """デフォルト初期化"""
        result = SetupResult()

        # 仕様書準拠のフィールド名とデフォルト値
        assert result.status == SetupStatus.NOT_STARTED
        assert result.components == []
        assert result.warnings == []
        assert result.errors == []
        assert result.can_continue is True
        assert result.timestamp is None
        assert result.duration_seconds is None

    def test_initialization_with_values(self):
        """値を指定した初期化"""
        now = datetime.now()
        result = SetupResult(
            status=SetupStatus.PARTIAL_SUCCESS,
            components=[],
            warnings=["Warning 1"],
            errors=["Error 1"],
            can_continue=False,
            timestamp=now,
            duration_seconds=45.3,
        )

        assert result.status == SetupStatus.PARTIAL_SUCCESS
        assert result.warnings == ["Warning 1"]
        assert result.errors == ["Error 1"]
        assert result.can_continue is False
        assert result.timestamp == now
        assert result.duration_seconds == 45.3

    def test_add_component(self):
        """コンポーネント追加"""
        result = SetupResult()

        # 成功したコンポーネント
        component1 = ComponentStatus(name="Python", installed=True, version="3.11.9")
        result.add_component(component1)

        assert len(result.components) == 1
        assert result.components[0].name == "Python"
        assert len(result.warnings) == 0  # 成功時は警告なし

    def test_add_component_with_error(self):
        """エラーがあるコンポーネント追加時の警告"""
        result = SetupResult()

        # 失敗したコンポーネント
        component = ComponentStatus(
            name="Ollama", installed=False, error="Network connection failed"
        )
        result.add_component(component)

        assert len(result.components) == 1
        assert len(result.warnings) == 1
        assert "Ollama: Network connection failed" in result.warnings[0]

    def test_add_warning(self):
        """警告メッセージ追加"""
        result = SetupResult()

        result.add_warning("Warning message 1")
        result.add_warning("Warning message 2")
        result.add_warning("Warning message 1")  # 重複

        # 重複は追加されない
        assert len(result.warnings) == 2
        assert "Warning message 1" in result.warnings
        assert "Warning message 2" in result.warnings

    def test_add_error(self):
        """エラーメッセージ追加"""
        result = SetupResult()

        result.add_error("Error message 1")
        result.add_error("Error message 2")
        result.add_error("Error message 1")  # 重複

        # 重複は追加されない
        assert len(result.errors) == 2
        assert "Error message 1" in result.errors
        assert "Error message 2" in result.errors

    def test_is_success_property(self):
        """is_successプロパティ"""
        # 完全成功
        result = SetupResult(status=SetupStatus.SUCCESS)
        assert result.is_success is True

        # 部分的成功
        result = SetupResult(status=SetupStatus.PARTIAL_SUCCESS)
        assert result.is_success is True

        # 失敗
        result = SetupResult(status=SetupStatus.FAILED)
        assert result.is_success is False

        # スキップ
        result = SetupResult(status=SetupStatus.SKIPPED)
        assert result.is_success is False

        # 未開始
        result = SetupResult(status=SetupStatus.NOT_STARTED)
        assert result.is_success is False

    def test_has_failures_property(self):
        """has_failuresプロパティ"""
        result = SetupResult()

        # 全て成功
        result.add_component(ComponentStatus("Python", installed=True))
        result.add_component(ComponentStatus("Ollama", installed=True))
        assert result.has_failures is False

        # 一部失敗
        result.add_component(ComponentStatus("AivisSpeech", installed=False))
        assert result.has_failures is True

    def test_success_rate_property(self):
        """success_rateプロパティ"""
        result = SetupResult()

        # コンポーネントなし
        assert result.success_rate == 0.0

        # 全て成功（2/2）
        result.add_component(ComponentStatus("Python", installed=True))
        result.add_component(ComponentStatus("Ollama", installed=True))
        assert result.success_rate == 1.0

        # 一部成功（2/3）
        result.add_component(ComponentStatus("AivisSpeech", installed=False))
        assert abs(result.success_rate - 0.6667) < 0.001  # 約66.67%

        # 全て失敗
        result2 = SetupResult()
        result2.add_component(ComponentStatus("Test1", installed=False))
        result2.add_component(ComponentStatus("Test2", installed=False))
        assert result2.success_rate == 0.0

    def test_get_installed_components(self):
        """インストール成功コンポーネントの取得"""
        result = SetupResult()

        comp1 = ComponentStatus("Python", installed=True)
        comp2 = ComponentStatus("Ollama", installed=False)
        comp3 = ComponentStatus("FasterWhisper", installed=True)

        result.add_component(comp1)
        result.add_component(comp2)
        result.add_component(comp3)

        installed = result.get_installed_components()
        assert len(installed) == 2
        assert installed[0].name == "Python"
        assert installed[1].name == "FasterWhisper"

    def test_get_failed_components(self):
        """インストール失敗コンポーネントの取得"""
        result = SetupResult()

        comp1 = ComponentStatus("Python", installed=True)
        comp2 = ComponentStatus("Ollama", installed=False)
        comp3 = ComponentStatus("AivisSpeech", installed=False)

        result.add_component(comp1)
        result.add_component(comp2)
        result.add_component(comp3)

        failed = result.get_failed_components()
        assert len(failed) == 2
        assert failed[0].name == "Ollama"
        assert failed[1].name == "AivisSpeech"

    def test_to_dict(self):
        """辞書形式への変換"""
        now = datetime.now()
        result = SetupResult(
            status=SetupStatus.PARTIAL_SUCCESS, timestamp=now, duration_seconds=30.5
        )

        # コンポーネント追加
        result.add_component(
            ComponentStatus(
                name="Python", installed=True, version="3.11.9", install_path="/usr/bin/python3"
            )
        )
        result.add_component(ComponentStatus(name="Ollama", installed=False, error="Network error"))

        result.add_warning("Some components failed")
        result.add_error("Critical error occurred")

        # 辞書に変換
        data = result.to_dict()

        # 基本フィールドの確認
        assert data["status"] == "partial_success"
        assert data["can_continue"] is True
        assert data["duration_seconds"] == 30.5
        assert data["timestamp"] == now.isoformat()
        assert data["success_rate"] == 0.5

        # コンポーネントの確認
        assert len(data["components"]) == 2
        assert data["components"][0]["name"] == "Python"
        assert data["components"][0]["installed"] is True
        assert data["components"][1]["name"] == "Ollama"
        assert data["components"][1]["installed"] is False

        # 警告とエラーの確認
        assert len(data["warnings"]) == 2  # コンポーネント追加時の警告 + 手動追加
        assert len(data["errors"]) == 1

    def test_fields_default_factory(self):
        """フィールドのdefault_factoryが独立して動作するか"""
        result1 = SetupResult()
        result2 = SetupResult()

        # 各インスタンスが独立したリストを持つ
        result1.components.append(ComponentStatus("Test1", installed=True))
        result2.components.append(ComponentStatus("Test2", installed=True))

        assert len(result1.components) == 1
        assert len(result2.components) == 1
        assert result1.components[0].name == "Test1"
        assert result2.components[0].name == "Test2"


# ============================================================================
# SetupResult統合テスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(1)
class TestSetupResultIntegration:
    """SetupResult型の統合的な動作テスト"""

    def test_typical_partial_success_scenario(self):
        """典型的な部分的成功のシナリオ"""
        result = SetupResult()
        start_time = datetime.now()

        # Pythonは成功（必須）
        result.add_component(
            ComponentStatus(
                name="Python", installed=True, version="3.11.9", install_path="C:\\Python311"
            )
        )

        # Ollamaは成功（オプション）
        result.add_component(
            ComponentStatus(
                name="Ollama",
                installed=True,
                version="0.1.24",
                install_path="C:\\Users\\user\\AppData\\Local\\Programs\\Ollama",
            )
        )

        # FasterWhisperは成功（オプション）
        result.add_component(
            ComponentStatus(
                name="FasterWhisper",
                installed=True,
                version="0.10.0",
                install_path="models/faster-whisper",
            )
        )

        # AivisSpeechは失敗（オプション）
        result.add_component(
            ComponentStatus(
                name="AivisSpeech", installed=False, error="Download failed: Connection timeout"
            )
        )

        # 結果の設定
        result.status = SetupStatus.PARTIAL_SUCCESS
        result.can_continue = True
        result.timestamp = start_time
        result.duration_seconds = 45.3

        # 検証
        assert result.is_success is True  # 部分的成功も成功扱い
        assert result.has_failures is True
        assert result.success_rate == 0.75  # 3/4 = 75%

        installed = result.get_installed_components()
        assert len(installed) == 3

        failed = result.get_failed_components()
        assert len(failed) == 1
        assert failed[0].name == "AivisSpeech"

        # 警告が追加されている
        assert len(result.warnings) == 1
        assert "AivisSpeech" in result.warnings[0]

    def test_complete_failure_scenario(self):
        """完全失敗のシナリオ"""
        result = SetupResult()

        # Pythonすら使えない（仮想的なシナリオ）
        result.add_component(
            ComponentStatus(name="Python", installed=False, error="Python not found in PATH")
        )

        # 他も全て失敗
        result.add_component(
            ComponentStatus(name="Ollama", installed=False, error="Cannot install without Python")
        )

        result.status = SetupStatus.FAILED
        result.can_continue = False  # 継続不可

        # 検証
        assert result.is_success is False
        assert result.has_failures is True
        assert result.success_rate == 0.0
        assert len(result.get_failed_components()) == 2
        assert len(result.warnings) == 2  # 各失敗で警告

    def test_user_skipped_scenario(self):
        """ユーザーがスキップしたシナリオ"""
        result = SetupResult(status=SetupStatus.SKIPPED)
        result.add_warning("User skipped the setup process")
        result.can_continue = True  # スキップしても継続可能（最小限の機能で）

        assert result.is_success is False
        assert result.has_failures is False  # コンポーネントがないので失敗もない
        assert result.success_rate == 0.0
        assert len(result.warnings) == 1
