"""VioraTalk例外クラスのテスト

例外階層とエラーコード体系の動作を検証。
エラーハンドリング指針 v1.20準拠
テスト戦略ガイドライン v1.7準拠
テスト実装ガイド v1.3準拠
"""

import pytest
from datetime import datetime
from typing import Dict, Any

from vioratalk.core.exceptions import (
    VioraTalkError,
    ConfigurationError,
    InitializationError,
    ComponentError,
    STTError,
    AudioError,
    LLMError,
    APIError,
    TTSError,
    CharacterError,
    MemoryError,
    EmotionError,
    LicenseError,
    ResourceError,
    NetworkError,
    BackgroundServiceError,
    ServiceStartupError,
    ServiceCommunicationError,
    SetupError,
    HumanLikeError
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
        error = VioraTalkError(
            "Test error",
            error_code="E5000",
            details=details,
            cause=cause
        )
        
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

class TestComponentError:
    """ComponentErrorのテスト"""
    
    def test_default_error_code(self):
        """デフォルトエラーコードの確認"""
        error = ComponentError("Component error")
        assert error.error_code == "E1000"
    
    def test_with_component_name(self):
        """コンポーネント名の保存"""
        error = ComponentError("Error", component_name="TestComponent")
        assert error.details["component_name"] == "TestComponent"


# ============================================================================
# エンジン関連エラーのテスト（将来実装用）
# ============================================================================

class TestEngineErrors:
    """エンジン関連エラーのテスト"""
    
    def test_stt_error_hierarchy(self):
        """STTエラーの継承関係"""
        stt_error = STTError("STT error")
        audio_error = AudioError("Audio error")
        
        assert isinstance(stt_error, VioraTalkError)
        assert isinstance(audio_error, STTError)
        assert isinstance(audio_error, VioraTalkError)
    
    def test_llm_error_hierarchy(self):
        """LLMエラーの継承関係"""
        llm_error = LLMError("LLM error")
        api_error = APIError("API error")
        
        assert isinstance(llm_error, VioraTalkError)
        assert isinstance(api_error, LLMError)
        assert isinstance(api_error, VioraTalkError)
    
    def test_tts_error_hierarchy(self):
        """TTSエラーの継承関係"""
        tts_error = TTSError("TTS error")
        assert isinstance(tts_error, VioraTalkError)


# ============================================================================
# キャラクター・記憶関連エラーのテスト
# ============================================================================

class TestCharacterMemoryErrors:
    """キャラクター・記憶関連エラーのテスト"""
    
    def test_character_error(self):
        """CharacterErrorのテスト"""
        error = CharacterError("Character not found")
        assert isinstance(error, VioraTalkError)
    
    def test_memory_error(self):
        """MemoryErrorのテスト"""
        error = MemoryError("Memory corrupted")
        assert isinstance(error, VioraTalkError)
    
    def test_emotion_error(self):
        """EmotionErrorのテスト"""
        error = EmotionError("Emotion analysis failed")
        assert isinstance(error, VioraTalkError)


# ============================================================================
# ライセンス関連エラーのテスト
# ============================================================================

class TestLicenseError:
    """LicenseErrorのテスト"""
    
    def test_default_error_code(self):
        """デフォルトエラーコード（E4201）の確認"""
        error = LicenseError("Pro feature required")
        assert error.error_code == "E4201"
    
    def test_with_feature(self):
        """機能名の保存"""
        error = LicenseError("Pro required", feature="advanced_voice")
        assert error.details["required_feature"] == "advanced_voice"


# ============================================================================
# システム・リソース関連エラーのテスト
# ============================================================================

class TestResourceError:
    """ResourceErrorのテスト"""
    
    def test_default_error_code(self):
        """デフォルトエラーコードの確認"""
        error = ResourceError("Out of memory")
        assert error.error_code == "E5300"
    
    def test_with_resource_details(self):
        """リソース詳細情報の保存"""
        error = ResourceError(
            "Insufficient memory",
            resource_type="memory",
            available=1024,
            required=2048
        )
        
        assert error.details["resource_type"] == "memory"
        assert error.details["available"] == 1024
        assert error.details["required"] == 2048
    
    def test_partial_resource_details(self):
        """部分的なリソース情報"""
        error = ResourceError("Low disk", resource_type="disk")
        assert error.details["resource_type"] == "disk"
        assert "available" not in error.details
        assert "required" not in error.details


class TestNetworkError:
    """NetworkErrorのテスト"""
    
    def test_default_error_code(self):
        """デフォルトエラーコードの確認"""
        error = NetworkError("Connection failed")
        assert error.error_code == "E5200"
    
    def test_with_url(self):
        """URL情報の保存"""
        error = NetworkError("Timeout", url="https://api.example.com")
        assert error.details["url"] == "https://api.example.com"


# ============================================================================
# バックグラウンドサービス関連エラーのテスト
# ============================================================================

class TestBackgroundServiceErrors:
    """バックグラウンドサービス関連エラーのテスト"""
    
    def test_background_service_error(self):
        """BackgroundServiceErrorのテスト"""
        error = BackgroundServiceError("Service failed")
        assert error.error_code == "E7000"
        assert isinstance(error, VioraTalkError)
    
    def test_with_service_name(self):
        """サービス名の保存"""
        error = BackgroundServiceError("Failed", service_name="OllamaService")
        assert error.details["service_name"] == "OllamaService"
    
    def test_service_startup_error(self):
        """ServiceStartupErrorのテスト"""
        error = ServiceStartupError("Failed to start")
        assert error.error_code == "E7001"
        assert isinstance(error, BackgroundServiceError)
        assert isinstance(error, VioraTalkError)
    
    def test_service_communication_error(self):
        """ServiceCommunicationErrorのテスト"""
        error = ServiceCommunicationError("Connection lost")
        assert error.error_code == "E7005"
        assert isinstance(error, BackgroundServiceError)


# ============================================================================
# セットアップ関連エラーのテスト
# ============================================================================

class TestSetupError:
    """SetupErrorのテスト"""
    
    def test_default_error_code(self):
        """デフォルトエラーコードの確認"""
        error = SetupError("Setup failed")
        assert error.error_code == "E8000"
    
    def test_with_step(self):
        """セットアップステップの保存"""
        error = SetupError("Failed at step 3", step="dependency_check")
        assert error.details["setup_step"] == "dependency_check"


# ============================================================================
# 人間らしさ実装関連エラーのテスト
# ============================================================================

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
        
        # E4201: ライセンスエラー
        license_error = LicenseError("Test")
        assert license_error.error_code == "E4201"
        
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
        
        error = ResourceError("Test", error_code="E5399")
        assert error.error_code == "E5399"


# ============================================================================
# エラーの実用的な使用例のテスト
# ============================================================================

class TestPracticalUsage:
    """実際の使用シナリオのテスト"""
    
    def test_error_chaining(self):
        """エラーの連鎖"""
        try:
            # 元のエラー
            raise FileNotFoundError("config.yaml not found")
        except FileNotFoundError as e:
            # ラップして再送出
            error = ConfigurationError(
                "Failed to load configuration",
                config_file="config.yaml",
                cause=e
            )
            assert error.cause is e
            assert "config.yaml" in error.details["config_file"]
    
    def test_error_logging_info(self):
        """ログ出力用の情報取得"""
        error = NetworkError(
            "API request failed",
            error_code="E5201",
            url="https://api.example.com/v1/chat",
            details={"status_code": 500, "retry_count": 3}
        )
        
        # to_dict()でログ用の情報を取得
        log_info = error.to_dict()
        
        assert log_info["error_type"] == "NetworkError"
        assert log_info["error_code"] == "E5201"
        assert log_info["message"] == "API request failed"
        assert log_info["details"]["status_code"] == 500
        assert log_info["details"]["retry_count"] == 3
        assert log_info["details"]["url"] == "https://api.example.com/v1/chat"
    
    def test_user_friendly_message(self):
        """ユーザー向けメッセージ"""
        # 技術的な詳細は内部に保持し、ユーザーには簡潔なメッセージを表示
        technical_error = ResourceError(
            "メモリが不足しています。他のアプリケーションを終了してください。",
            error_code="E5301",
            resource_type="memory",
            available=512,
            required=2048
        )
        
        # ユーザー向け
        user_message = str(technical_error)
        assert "メモリが不足" in user_message
        
        # 開発者向け（詳細情報）
        dev_info = technical_error.to_dict()
        assert dev_info["details"]["available"] == 512
        assert dev_info["details"]["required"] == 2048