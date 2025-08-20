"""VioraTalk例外クラスのテスト

例外階層とエラーコード体系の動作を検証。
エラーハンドリング指針 v1.20準拠
テスト戦略ガイドライン v1.7準拠
テスト実装ガイド v1.3準拠

Note: Phase 1では基底クラスのみ実装。
      AudioError、APIError等の具体的な例外クラスはPhase 3-4で実装予定。
"""

from datetime import datetime

import pytest

from vioratalk.core.exceptions import LLMError  # Phase 1では基底クラスのみ
from vioratalk.core.exceptions import ModelError  # Phase 1 Part 20で追加
from vioratalk.core.exceptions import STTError  # Phase 1では基底クラスのみ
from vioratalk.core.exceptions import TTSError  # Phase 1では基底クラスのみ
from vioratalk.core.exceptions import (  # LicenseError,  # Phase 7で実装予定（Pro版機能）
    BackgroundServiceError,
    CharacterError,
    ComponentError,
    ConfigurationError,
    EmotionError,
    HumanLikeError,
    InitializationError,
    MemoryError,
    NetworkError,
    ResourceError,
    ServiceCommunicationError,
    ServiceStartupError,
    SetupError,
    VioraTalkError,
)

# ============================================================================
# VioraTalkError基底クラスのテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(1)
class TestVioraTalkError:
    """VioraTalkError基底例外クラスのテスト"""

    def test_basic_initialization(self):
        """基本的な初期化"""
        error = VioraTalkError("Test error message")

        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.error_code is None
        assert error.details == {}
        assert error.cause is None
        assert isinstance(error.timestamp, datetime)

    def test_initialization_with_error_code(self):
        """エラーコード付きの初期化"""
        error = VioraTalkError("Test error", error_code="E5000")

        assert str(error) == "[E5000] Test error"
        assert error.error_code == "E5000"

    def test_initialization_with_details(self):
        """詳細情報付きの初期化"""
        details = {"file": "test.yaml", "line": 42}
        error = VioraTalkError("Test error", details=details)

        assert error.details == details

    def test_initialization_with_cause(self):
        """原因例外付きの初期化"""
        cause = ValueError("Original error")
        error = VioraTalkError("Wrapped error", cause=cause)

        assert error.cause is cause

    def test_to_dict(self):
        """辞書形式への変換"""
        cause = ValueError("Original")
        details = {"key": "value"}
        error = VioraTalkError("Test error", error_code="E5000", details=details, cause=cause)

        result = error.to_dict()

        assert result["error_type"] == "VioraTalkError"
        assert result["error_code"] == "E5000"
        assert result["message"] == "Test error"
        assert result["details"] == details
        assert result["cause"] == "Original"
        assert "timestamp" in result

    def test_inheritance(self):
        """Exception継承の確認"""
        error = VioraTalkError("Test")
        assert isinstance(error, Exception)


# ============================================================================
# 初期化・設定関連エラーのテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(1)
class TestConfigurationError:
    """ConfigurationErrorのテスト"""

    def test_default_error_code(self):
        """デフォルトエラーコードの確認"""
        error = ConfigurationError("Config error")
        assert error.error_code == "E0001"

    def test_with_config_file(self):
        """設定ファイル情報の保存"""
        error = ConfigurationError("Config error", config_file="config.yaml")
        assert error.details["config_file"] == "config.yaml"

    def test_custom_error_code(self):
        """カスタムエラーコードの指定"""
        error = ConfigurationError("Config error", error_code="E0099")
        assert error.error_code == "E0099"

    def test_inheritance(self):
        """継承関係の確認"""
        error = ConfigurationError("Test")
        assert isinstance(error, VioraTalkError)


@pytest.mark.unit
@pytest.mark.phase(1)
class TestInitializationError:
    """InitializationErrorのテスト"""

    def test_default_error_code(self):
        """デフォルトエラーコードの確認"""
        error = InitializationError("Init error")
        assert error.error_code == "E0100"

    def test_with_component(self):
        """コンポーネント情報の保存"""
        error = InitializationError("Init error", component="DialogueManager")
        assert error.details["component"] == "DialogueManager"

    def test_custom_error_code(self):
        """カスタムエラーコードの指定"""
        error = InitializationError("Init error", error_code="E0199")
        assert error.error_code == "E0199"


# ============================================================================
# コンポーネント関連エラーのテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(1)
class TestComponentError:
    """ComponentErrorのテスト"""

    def test_default_error_code(self):
        """デフォルトエラーコードの確認"""
        error = ComponentError("Component error")
        assert error.error_code == "E1000"

    def test_with_component_name(self):
        """コンポーネント名の保存"""
        error = ComponentError("Error", state="INITIALIZING")
        assert error.details["state"] == "INITIALIZING"


# ============================================================================
# エンジン関連エラーのテスト（Phase 1では基底クラスのみ）
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(1)
class TestEngineErrors:
    """エンジン関連エラーの基底クラステスト

    Note: Phase 1では基底クラスのみテスト。
          AudioError、APIError等の具体的な例外クラスはPhase 3-4で実装予定。
    """

    def test_stt_error_base(self):
        """STTError基底クラスのテスト"""
        stt_error = STTError("STT error")

        assert isinstance(stt_error, VioraTalkError)
        assert stt_error.error_code == "E1000"  # デフォルトコード
        assert str(stt_error) == "[E1000] STT error"

    def test_llm_error_base(self):
        """LLMError基底クラスのテスト"""
        llm_error = LLMError("LLM error")

        assert isinstance(llm_error, VioraTalkError)
        assert llm_error.error_code == "E2000"  # デフォルトコード
        assert str(llm_error) == "[E2000] LLM error"

    def test_tts_error_base(self):
        """TTSError基底クラスのテスト"""
        tts_error = TTSError("TTS error")

        assert isinstance(tts_error, VioraTalkError)
        assert tts_error.error_code == "E3000"  # デフォルトコード
        assert str(tts_error) == "[E3000] TTS error"

    # Phase 3-4で以下の具体的な例外クラスを実装予定:
    # - AudioError (STTError): 音声入力エラー
    # - TranscriptionError (STTError): 文字起こしエラー
    # - APIError (LLMError): API通信エラー
    # - RateLimitError (APIError): レート制限エラー
    # - TimeoutError (APIError): タイムアウトエラー
    # - VoiceModelError (TTSError): 音声モデルエラー
    # - SynthesisError (TTSError): 音声生成エラー


# ============================================================================
# キャラクター・記憶関連エラーのテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(1)
class TestCharacterMemoryErrors:
    """キャラクター・記憶関連エラーのテスト"""

    def test_character_error(self):
        """CharacterErrorのテスト"""
        error = CharacterError("Character not found")
        assert isinstance(error, VioraTalkError)
        assert error.error_code == "E4000"

    def test_character_error_with_id(self):
        """CharacterErrorにcharacter_id付きのテスト"""
        error = CharacterError("Character not found", character_id="001_aoi")
        assert error.details["character_id"] == "001_aoi"

    def test_memory_error(self):
        """MemoryErrorのテスト"""
        error = MemoryError("Memory corrupted")
        assert isinstance(error, VioraTalkError)
        assert error.error_code == "E4100"

    def test_emotion_error(self):
        """EmotionErrorのテスト"""
        error = EmotionError("Emotion analysis failed")
        assert isinstance(error, VioraTalkError)
        assert error.error_code == "E4300"


# ============================================================================
# ライセンス関連エラーのテスト（Phase 7で実装予定）
# ============================================================================

# Phase 7でLicenseErrorクラスが実装される際に以下のテストを追加予定:
# - LicenseError基底クラス
# - デフォルトエラーコード（E4201）
# - Pro版機能の制限チェック
# - 機能名の保存


# ============================================================================
# システム・リソース関連エラーのテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(1)
class TestResourceError:
    """ResourceErrorのテスト"""

    def test_default_error_code(self):
        """デフォルトエラーコードの確認"""
        error = ResourceError("Out of memory")
        assert error.error_code == "E5300"

    def test_with_resource_details(self):
        """リソース詳細情報の保存"""
        error = ResourceError("Insufficient memory", resource_type="memory")
        assert error.details["resource_type"] == "memory"

    def test_inheritance(self):
        """継承関係の確認"""
        error = ResourceError("Test")
        assert isinstance(error, VioraTalkError)


@pytest.mark.unit
@pytest.mark.phase(1)
class TestNetworkError:
    """NetworkErrorのテスト"""

    def test_default_error_code(self):
        """デフォルトエラーコードの確認"""
        error = NetworkError("Connection failed")
        assert error.error_code == "E5200"

    def test_with_url(self):
        """URL情報の保存"""
        error = NetworkError("Connection failed", url="https://api.example.com")
        assert error.details["url"] == "https://api.example.com"


# ============================================================================
# モデル関連エラーのテスト（Phase 1 Part 20で追加）
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(1)
class TestModelError:
    """ModelErrorのテスト（Phase 1最小実装）"""

    def test_basic_initialization(self):
        """基本的な初期化"""
        error = ModelError("Model error")
        assert isinstance(error, VioraTalkError)
        assert str(error) == "Model error"

    def test_with_error_code(self):
        """エラーコード付きの初期化"""
        error = ModelError("Model error", error_code="E5500")
        assert error.error_code == "E5500"

    # Phase 2以降で以下の派生クラスを実装予定:
    # - DownloadError: モデルダウンロードエラー
    # - ModelLoadError: モデル読み込みエラー


# ============================================================================
# バックグラウンドサービス関連エラーのテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(1)
class TestBackgroundServiceError:
    """BackgroundServiceErrorのテスト"""

    def test_default_error_code(self):
        """デフォルトエラーコードの確認"""
        error = BackgroundServiceError("Service error")
        assert error.error_code == "E7000"

    def test_with_service_name(self):
        """サービス名の保存"""
        error = BackgroundServiceError("Service failed", service_name="WhisperService")
        assert error.details["service_name"] == "WhisperService"

    def test_service_startup_error(self):
        """ServiceStartupErrorのテスト"""
        error = ServiceStartupError("Failed to start service")
        assert isinstance(error, BackgroundServiceError)
        assert error.error_code == "E7001"

    def test_service_communication_error(self):
        """ServiceCommunicationErrorのテスト"""
        error = ServiceCommunicationError("IPC failed")
        assert isinstance(error, BackgroundServiceError)
        assert error.error_code == "E7005"


# ============================================================================
# セットアップ関連エラーのテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(1)
class TestSetupError:
    """SetupErrorのテスト"""

    def test_default_error_code(self):
        """デフォルトエラーコードの確認"""
        error = SetupError("Setup failed")
        assert error.error_code == "E8000"

    def test_with_step(self):
        """セットアップステップの保存"""
        error = SetupError("Installation failed", step="dependency_install")
        assert error.details["setup_step"] == "dependency_install"


# ============================================================================
# 人間らしさ実装関連エラーのテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(1)
class TestHumanLikeError:
    """HumanLikeErrorのテスト"""

    def test_default_error_code(self):
        """デフォルトエラーコードの確認"""
        error = HumanLikeError("Humanlike feature failed")
        assert error.error_code == "E9000"

    def test_inheritance(self):
        """継承関係の確認"""
        error = HumanLikeError("Test")
        assert isinstance(error, VioraTalkError)


# ============================================================================
# エラーコード体系のテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(1)
class TestErrorCodeSystem:
    """エラーコード体系の整合性テスト"""

    def test_error_code_ranges(self):
        """エラーコード範囲の確認"""
        # E0001-E0099: 設定エラー
        config_error = ConfigurationError("Test")
        assert config_error.error_code.startswith("E00")

        # E0100-E0199: 初期化エラー
        init_error = InitializationError("Test")
        assert init_error.error_code.startswith("E01")

        # E1000-E1099: コンポーネントエラー
        component_error = ComponentError("Test")
        assert component_error.error_code == "E1000"

        # E4201: ライセンスエラー（Phase 7で実装予定）
        # license_error = LicenseError("Test")
        # assert license_error.error_code == "E4201"

        # E5200番台: ネットワークエラー
        network_error = NetworkError("Test")
        assert network_error.error_code.startswith("E52")

        # E5300番台: リソースエラー
        resource_error = ResourceError("Test")
        assert resource_error.error_code.startswith("E53")

        # E7000番台: バックグラウンドサービス
        service_error = BackgroundServiceError("Test")
        assert service_error.error_code.startswith("E70")

        # E8000番台: セットアップ
        setup_error = SetupError("Test")
        assert setup_error.error_code.startswith("E80")

        # E9000番台: 人間らしさ
        humanlike_error = HumanLikeError("Test")
        assert humanlike_error.error_code.startswith("E90")

    def test_custom_error_codes(self):
        """カスタムエラーコードの上書き"""
        # すべての例外クラスでカスタムエラーコードを指定可能
        error = ConfigurationError("Test", error_code="E0099")
        assert error.error_code == "E0099"

        error = InitializationError("Test", error_code="E0150")
        assert error.error_code == "E0150"

        error = ComponentError("Test", error_code="E1050")
        assert error.error_code == "E1050"
