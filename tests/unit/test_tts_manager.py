"""TTSManager単体テスト

TTSManagerの包括的な単体テスト。
エンジン管理、音声合成、フォールバック機能、統計情報などをテスト。

テスト戦略ガイドライン v1.7準拠
開発規約書 v1.12準拠
データフォーマット仕様書 v1.5準拠
"""

from unittest.mock import AsyncMock, MagicMock, Mock, create_autospec

import pytest

from vioratalk.core.base import ComponentState
from vioratalk.core.exceptions import InvalidVoiceError, TTSError
from vioratalk.core.tts.base import BaseTTSEngine, SynthesisResult, TTSConfig, VoiceInfo
from vioratalk.core.tts.tts_manager import TTSManager

# =============================================================================
# フィクスチャ
# =============================================================================


@pytest.fixture
def mock_tts_engine():
    """モックTTSエンジンを作成（仕様書準拠）"""
    # BaseTTSEngineのspecを使用してモック作成
    # spec_set=Falseにして属性の追加を許可
    engine = create_autospec(BaseTTSEngine, spec_set=False)

    # 状態設定（configure_mockを使用）
    engine.configure_mock(_state=ComponentState.NOT_INITIALIZED)  # 仕様書準拠：初期状態
    engine.is_available.return_value = True

    # 非同期メソッドをAsyncMockに
    engine.initialize = AsyncMock()
    engine.cleanup = AsyncMock()

    # SynthesisResultに仕様書準拠のformat追加
    engine.synthesize = AsyncMock(
        return_value=SynthesisResult(
            audio_data=b"test_audio", sample_rate=16000, duration=1.0, format="wav"  # 仕様書準拠：必須フィールド
        )
    )
    engine.test_availability = AsyncMock(return_value=True)

    # 同期メソッド（仕様書準拠のVoiceInfo定義）
    engine.get_available_voices = Mock(
        return_value=[
            VoiceInfo(
                id="voice1",  # 仕様書準拠：idフィールド
                name="Test Voice 1",
                language="ja",  # 仕様書準拠：必須フィールド
                gender="female",  # 仕様書準拠：必須フィールド
            ),
            VoiceInfo(
                id="voice2",  # 仕様書準拠：idフィールド
                name="Test Voice 2",
                language="ja",  # 仕様書準拠：必須フィールド
                gender="male",  # 仕様書準拠：必須フィールド
            ),
        ]
    )
    engine.set_voice = Mock()

    return engine


@pytest.fixture
def tts_manager():
    """TTSManagerインスタンスを作成"""
    return TTSManager(max_fallback_attempts=3)


@pytest.fixture
def mock_error_handler():
    """モックエラーハンドラーを作成"""
    handler = MagicMock()
    handler.handle_error_async = AsyncMock(
        return_value=MagicMock(error_code="E3000", message="Test error")
    )
    return handler


# =============================================================================
# 初期化とクリーンアップ
# =============================================================================


class TestInitializationAndCleanup:
    """初期化とクリーンアップのテスト"""

    def test_initialization_default(self, tts_manager):
        """デフォルト設定での初期化"""
        assert tts_manager._engines == {}
        assert tts_manager._priorities == {}
        assert tts_manager._active_engine is None
        assert tts_manager._max_fallback_attempts == 3
        # 仕様書準拠：初期状態はNOT_INITIALIZED
        assert tts_manager._state == ComponentState.NOT_INITIALIZED
        assert tts_manager._stats["total_requests"] == 0

    def test_initialization_with_config(self):
        """カスタム設定での初期化"""
        # 仕様書準拠：speedフィールドを使用（有効範囲: 0.1-3.0）
        config = TTSConfig(voice_id="custom_voice", speed=2.0, volume=0.8)  # 仕様書準拠：有効範囲内の値（0.1-3.0）
        manager = TTSManager(config=config, max_fallback_attempts=5)

        assert manager.config == config
        assert manager._max_fallback_attempts == 5

    @pytest.mark.asyncio
    async def test_initialize_with_engines(self, tts_manager, mock_tts_engine):
        """エンジン登録後の初期化"""
        tts_manager.register_engine("test", mock_tts_engine)

        await tts_manager.initialize()

        assert tts_manager._state == ComponentState.READY
        mock_tts_engine.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup(self, tts_manager, mock_tts_engine):
        """クリーンアップ処理"""
        tts_manager.register_engine("test", mock_tts_engine)
        await tts_manager.initialize()

        await tts_manager.cleanup()

        assert tts_manager._state == ComponentState.TERMINATED
        assert tts_manager._engines == {}
        assert tts_manager._active_engine is None
        mock_tts_engine.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_double_initialization(self, tts_manager, mock_tts_engine):
        """二重初期化の防止"""
        tts_manager.register_engine("test", mock_tts_engine)

        await tts_manager.initialize()
        await tts_manager.initialize()  # 2回目

        # 初期化は1回のみ実行される
        mock_tts_engine.initialize.assert_called_once()


# =============================================================================
# エンジン登録
# =============================================================================


class TestEngineRegistration:
    """エンジン登録のテスト"""

    def test_register_engine_basic(self, tts_manager, mock_tts_engine):
        """基本的なエンジン登録"""
        tts_manager.register_engine("test", mock_tts_engine, priority=10)

        assert "test" in tts_manager._engines
        assert tts_manager._engines["test"] == mock_tts_engine
        assert tts_manager._priorities["test"] == 10
        assert tts_manager._active_engine == "test"  # 最初のエンジンがアクティブ

    def test_register_multiple_engines(self, tts_manager, mock_tts_engine):
        """複数エンジンの登録"""
        # 2つ目のエンジンもBaseTTSEngineのspecで作成（spec_set=False）
        engine2 = create_autospec(BaseTTSEngine, spec_set=False)
        engine2.configure_mock(_state=ComponentState.READY)

        tts_manager.register_engine("engine1", mock_tts_engine, priority=5)
        tts_manager.register_engine("engine2", engine2, priority=10)

        assert len(tts_manager._engines) == 2
        assert tts_manager._active_engine == "engine1"  # 最初に登録したものがアクティブ

    def test_register_invalid_engine(self, tts_manager):
        """無効なエンジンの登録"""
        with pytest.raises(ValueError, match="must be BaseTTSEngine"):
            tts_manager.register_engine("invalid", "not_an_engine")

    def test_unregister_engine(self, tts_manager, mock_tts_engine):
        """エンジンの登録解除"""
        tts_manager.register_engine("test", mock_tts_engine)

        tts_manager.unregister_engine("test")

        assert "test" not in tts_manager._engines
        assert tts_manager._active_engine is None

    def test_unregister_nonexistent_engine(self, tts_manager):
        """存在しないエンジンの登録解除"""
        with pytest.raises(ValueError, match="is not registered"):
            tts_manager.unregister_engine("nonexistent")

    def test_unregister_active_engine_with_fallback(self, tts_manager, mock_tts_engine):
        """アクティブエンジン削除時の自動切り替え"""
        engine2 = create_autospec(BaseTTSEngine, spec_set=False)
        engine2.configure_mock(_state=ComponentState.READY)

        tts_manager.register_engine("engine1", mock_tts_engine, priority=5)
        tts_manager.register_engine("engine2", engine2, priority=10)
        tts_manager.set_active_engine("engine1")

        tts_manager.unregister_engine("engine1")

        # 優先順位が最も高いエンジンに切り替わる
        assert tts_manager._active_engine == "engine2"


# =============================================================================
# 音声合成
# =============================================================================


class TestSynthesis:
    """音声合成のテスト"""

    @pytest.mark.asyncio
    async def test_synthesize_basic(self, tts_manager, mock_tts_engine):
        """基本的な音声合成"""
        tts_manager.register_engine("test", mock_tts_engine)

        result = await tts_manager.synthesize("テストテキスト")

        assert isinstance(result, SynthesisResult)
        assert result.audio_data == b"test_audio"
        assert result.format == "wav"  # 仕様書準拠
        mock_tts_engine.synthesize.assert_called_once_with("テストテキスト", None, None)

    @pytest.mark.asyncio
    async def test_synthesize_with_voice_id(self, tts_manager, mock_tts_engine):
        """音声ID指定での合成"""
        tts_manager.register_engine("test", mock_tts_engine)

        await tts_manager.synthesize("テスト", voice_id="voice1")

        mock_tts_engine.synthesize.assert_called_once_with("テスト", "voice1", None)

    @pytest.mark.asyncio
    async def test_synthesize_no_engines(self, tts_manager):
        """エンジン未登録での合成"""
        with pytest.raises(RuntimeError, match="No TTS engine registered"):
            await tts_manager.synthesize("テスト")

    @pytest.mark.asyncio
    async def test_synthesize_error_handling(
        self, tts_manager, mock_tts_engine, mock_error_handler
    ):
        """合成エラーのハンドリング"""
        tts_manager._error_handler = mock_error_handler
        tts_manager.register_engine("test", mock_tts_engine)

        # エラーを発生させる
        mock_tts_engine.synthesize.side_effect = TTSError("Synthesis failed", error_code="E3000")

        with pytest.raises(TTSError):
            await tts_manager.synthesize("テスト")

        # エラーハンドラーが呼ばれたことを確認
        mock_error_handler.handle_error_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_synthesize_stats_update(self, tts_manager, mock_tts_engine):
        """統計情報の更新"""
        tts_manager.register_engine("test", mock_tts_engine)

        await tts_manager.synthesize("テスト")

        stats = tts_manager.get_stats()
        assert stats["total_requests"] == 1
        assert stats["success_count"] == 1
        assert stats["engine_usage"]["test"] == 1

    @pytest.mark.asyncio
    async def test_synthesize_with_custom_params(self, tts_manager, mock_tts_engine):
        """カスタムパラメータでの合成"""
        tts_manager.register_engine("test", mock_tts_engine)

        await tts_manager.synthesize("テスト", voice_id="voice1", style="happy", speed=150, volume=0.8)

        mock_tts_engine.synthesize.assert_called_once_with(
            "テスト", "voice1", "happy", speed=150, volume=0.8
        )


# =============================================================================
# フォールバック機能
# =============================================================================


class TestFallbackFunctionality:
    """フォールバック機能のテスト"""

    @pytest.mark.asyncio
    async def test_fallback_on_engine_failure(self, tts_manager):
        """エンジン失敗時のフォールバック"""
        # 失敗するエンジンと成功するエンジンを作成（仕様書準拠）
        failing_engine = create_autospec(BaseTTSEngine, spec_set=False)
        failing_engine.configure_mock(_state=ComponentState.READY)
        failing_engine.synthesize = AsyncMock(
            side_effect=TTSError("Engine failed", error_code="E3000")
        )

        success_engine = create_autospec(BaseTTSEngine, spec_set=False)
        success_engine.configure_mock(_state=ComponentState.READY)
        success_engine.synthesize = AsyncMock(
            return_value=SynthesisResult(
                audio_data=b"fallback_audio", sample_rate=16000, duration=1.0, format="wav"  # 仕様書準拠
            )
        )

        tts_manager.register_engine("failing", failing_engine, priority=10)
        tts_manager.register_engine("success", success_engine, priority=5)

        result = await tts_manager.synthesize_with_fallback("テスト")

        assert result.audio_data == b"fallback_audio"
        assert result.format == "wav"
        assert tts_manager._stats["fallback_count"] == 1

    @pytest.mark.asyncio
    async def test_all_engines_fail(self, tts_manager):
        """すべてのエンジンが失敗"""
        for i in range(3):
            engine = create_autospec(BaseTTSEngine, spec_set=False)
            engine.configure_mock(_state=ComponentState.READY)
            engine.synthesize = AsyncMock(
                side_effect=TTSError(f"Engine {i} failed", error_code="E3000")
            )
            tts_manager.register_engine(f"engine{i}", engine, priority=i)

        with pytest.raises(TTSError, match="All TTS engines failed"):
            await tts_manager.synthesize_with_fallback("テスト")

    @pytest.mark.asyncio
    async def test_invalid_voice_no_fallback(self, tts_manager):
        """InvalidVoiceErrorではフォールバックしない"""
        engine = create_autospec(BaseTTSEngine, spec_set=False)
        engine.configure_mock(_state=ComponentState.READY)
        engine.synthesize = AsyncMock(
            side_effect=InvalidVoiceError("Invalid voice", error_code="E3001")
        )

        tts_manager.register_engine("test", engine)

        with pytest.raises(InvalidVoiceError):
            await tts_manager.synthesize_with_fallback("テスト")

        # フォールバックカウントは増えない
        assert tts_manager._stats["fallback_count"] == 0

    @pytest.mark.asyncio
    async def test_preferred_engine_priority(self, tts_manager):
        """優先エンジン指定"""
        engine1 = create_autospec(BaseTTSEngine, spec_set=False)
        engine1.configure_mock(_state=ComponentState.READY)
        engine1.synthesize = AsyncMock(
            return_value=SynthesisResult(
                audio_data=b"engine1", sample_rate=16000, duration=1.0, format="wav"  # 仕様書準拠
            )
        )

        engine2 = create_autospec(BaseTTSEngine, spec_set=False)
        engine2.configure_mock(_state=ComponentState.READY)
        engine2.synthesize = AsyncMock(
            return_value=SynthesisResult(
                audio_data=b"engine2", sample_rate=16000, duration=1.0, format="wav"  # 仕様書準拠
            )
        )

        tts_manager.register_engine("engine1", engine1, priority=10)
        tts_manager.register_engine("engine2", engine2, priority=5)

        # engine2を優先指定
        result = await tts_manager.synthesize_with_fallback("テスト", preferred_engine="engine2")

        assert result.audio_data == b"engine2"
        assert result.format == "wav"
        engine2.synthesize.assert_called_once()
        engine1.synthesize.assert_not_called()

    @pytest.mark.asyncio
    async def test_max_fallback_attempts(self, tts_manager):
        """最大フォールバック試行回数"""
        # 5つのエンジンを登録（max_fallback_attempts=3）
        for i in range(5):
            engine = create_autospec(BaseTTSEngine, spec_set=False)
            engine.configure_mock(_state=ComponentState.READY)
            engine.synthesize = AsyncMock(
                side_effect=TTSError(f"Engine {i} failed", error_code="E3000")
            )
            tts_manager.register_engine(f"engine{i}", engine, priority=10 - i)

        with pytest.raises(TTSError):
            await tts_manager.synthesize_with_fallback("テスト")

        # 最初の3つのエンジンのみ試行される
        for i in range(3):
            tts_manager._engines[f"engine{i}"].synthesize.assert_called_once()
        for i in range(3, 5):
            tts_manager._engines[f"engine{i}"].synthesize.assert_not_called()


# =============================================================================
# 統計情報
# =============================================================================


class TestStatistics:
    """統計情報のテスト"""

    @pytest.mark.asyncio
    async def test_stats_tracking_success(self, tts_manager, mock_tts_engine):
        """成功時の統計追跡"""
        tts_manager.register_engine("test", mock_tts_engine)

        # 3回合成
        for _ in range(3):
            await tts_manager.synthesize("テスト")

        stats = tts_manager.get_stats()
        assert stats["total_requests"] == 3
        assert stats["success_count"] == 3
        assert stats["success_rate"] == 1.0
        assert stats["engine_usage"]["test"] == 3

    @pytest.mark.asyncio
    async def test_stats_tracking_failure(self, tts_manager, mock_tts_engine):
        """失敗時の統計追跡"""
        tts_manager.register_engine("test", mock_tts_engine)
        mock_tts_engine.synthesize.side_effect = TTSError("Failed", error_code="E3000")

        # 2回失敗
        for _ in range(2):
            with pytest.raises(TTSError):
                await tts_manager.synthesize("テスト")

        stats = tts_manager.get_stats()
        assert stats["total_requests"] == 2
        assert stats["success_count"] == 0
        assert stats["engine_errors"]["test"] == 2
        # エラーハンドリング指針v1.20準拠：エラーコード付き形式
        assert "[E3000]" in stats["last_error"]
        assert "Failed" in stats["last_error"]

    def test_reset_stats(self, tts_manager):
        """統計情報のリセット"""
        # 統計を変更
        tts_manager._stats["total_requests"] = 10
        tts_manager._stats["success_count"] = 8

        tts_manager.reset_stats()

        stats = tts_manager.get_stats()
        assert stats["total_requests"] == 0
        assert stats["success_count"] == 0
        assert stats["success_rate"] == 0.0


# =============================================================================
# エンジン制御
# =============================================================================


class TestEngineControl:
    """エンジン制御のテスト"""

    def test_set_active_engine(self, tts_manager, mock_tts_engine):
        """アクティブエンジンの設定"""
        engine2 = create_autospec(BaseTTSEngine, spec_set=False)
        engine2.configure_mock(_state=ComponentState.READY)

        tts_manager.register_engine("engine1", mock_tts_engine)
        tts_manager.register_engine("engine2", engine2)

        tts_manager.set_active_engine("engine2")

        assert tts_manager.get_active_engine() == "engine2"

    def test_set_active_engine_invalid(self, tts_manager):
        """無効なエンジンへの切り替え"""
        with pytest.raises(ValueError, match="is not registered"):
            tts_manager.set_active_engine("nonexistent")

    def test_get_available_engines(self, tts_manager, mock_tts_engine):
        """利用可能エンジンの取得"""
        engine2 = create_autospec(BaseTTSEngine, spec_set=False)
        engine2.configure_mock(_state=ComponentState.READY)

        tts_manager.register_engine("engine1", mock_tts_engine)
        tts_manager.register_engine("engine2", engine2)

        engines = tts_manager.get_available_engines()

        assert set(engines) == {"engine1", "engine2"}

    def test_set_engine_priority(self, tts_manager, mock_tts_engine):
        """エンジン優先順位の設定"""
        engine2 = create_autospec(BaseTTSEngine, spec_set=False)
        engine2.configure_mock(_state=ComponentState.READY)

        tts_manager.register_engine("engine1", mock_tts_engine, priority=5)
        tts_manager.register_engine("engine2", engine2, priority=10)

        # 優先順位を変更
        tts_manager.set_engine_priority({"engine1": 15, "engine2": 3})

        priorities = tts_manager.get_engine_priorities()
        assert priorities["engine1"] == 15
        assert priorities["engine2"] == 3


# =============================================================================
# エッジケース
# =============================================================================


class TestEdgeCases:
    """エッジケースのテスト"""

    def test_get_available_voices_delegated(self, tts_manager, mock_tts_engine):
        """get_available_voicesの委譲"""
        tts_manager.register_engine("test", mock_tts_engine)

        voices = tts_manager.get_available_voices()

        assert len(voices) == 2
        assert voices[0].id == "voice1"  # 仕様書準拠：idフィールド
        mock_tts_engine.get_available_voices.assert_called_once()

    def test_set_voice_delegated(self, tts_manager, mock_tts_engine):
        """set_voiceの委譲"""
        tts_manager.register_engine("test", mock_tts_engine)

        tts_manager.set_voice("voice1")

        mock_tts_engine.set_voice.assert_called_once_with("voice1")

    def test_repr_method(self, tts_manager, mock_tts_engine):
        """__repr__メソッド"""
        tts_manager.register_engine("test", mock_tts_engine)

        repr_str = repr(tts_manager)

        assert "TTSManager" in repr_str
        assert "state=" in repr_str  # 初期状態を問わず存在を確認
        assert "engines=['test']" in repr_str
        assert "active=test" in repr_str
