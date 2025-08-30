"""FasterWhisperEngineの単体テスト

音声認識エンジンの動作を検証する単体テスト。
モックを活用して外部依存を排除。

テスト戦略ガイドライン v1.7準拠
テスト実装ガイド v1.3準拠
開発規約書 v1.12準拠
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# プロジェクト内インポート
from vioratalk.core.base import ComponentState
from vioratalk.core.exceptions import InitializationError, ModelNotFoundError, STTError
from vioratalk.core.stt.base import AudioData, AudioMetadata, STTConfig, TranscriptionResult

# ============================================================================
# フィクスチャ
# ============================================================================


@pytest.fixture
def stt_config():
    """STT設定のフィクスチャ"""
    return STTConfig(
        engine="faster-whisper", model="base", language="ja", device="cpu", compute_type="int8"
    )


@pytest.fixture
def valid_audio_data():
    """有効な音声データのフィクスチャ"""
    # 1秒分のサンプル音声データ（サイン波）
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_array = np.sin(2 * np.pi * 440 * t).astype(np.float32)  # 440Hz

    return AudioData(
        raw_data=audio_array,
        metadata=AudioMetadata(
            sample_rate=sample_rate, channels=1, bit_depth=16, duration=duration, format="pcm"
        ),
        source="test",
    )


@pytest.fixture
def mock_whisper_model():
    """WhisperModelのモック"""
    mock_model = MagicMock()

    # transcribeメソッドのモック
    mock_segment = MagicMock()
    mock_segment.text = "これはテスト音声です"
    mock_segment.start = 0.0
    mock_segment.end = 1.0
    mock_segment.avg_logprob = -0.5  # np.exp(-0.5) ≈ 0.6

    mock_info = MagicMock()
    mock_info.language = "ja"

    mock_model.transcribe.return_value = ([mock_segment], mock_info)

    return mock_model


@pytest.fixture
def mock_model_download_manager():
    """ModelDownloadManagerのモック"""
    mock_manager = AsyncMock()
    # download_whisper_modelメソッドをモック（Part 24-25での変更に対応）
    mock_manager.download_whisper_model.return_value = Path("/models/whisper-base")
    return mock_manager


# ============================================================================
# 初期化テスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestFasterWhisperEngineInitialization:
    """FasterWhisperEngineの初期化テスト"""

    def test_initialization_without_faster_whisper(self):
        """faster-whisperが利用不可の場合のテスト"""
        with patch("vioratalk.core.stt.faster_whisper_engine.FASTER_WHISPER_AVAILABLE", False):
            from vioratalk.core.stt.faster_whisper_engine import FasterWhisperEngine

            with pytest.raises(InitializationError) as exc_info:
                engine = FasterWhisperEngine()

            assert exc_info.value.error_code == "E0007"
            assert "faster-whisper is not installed" in str(exc_info.value)

    @patch("vioratalk.core.stt.faster_whisper_engine.FASTER_WHISPER_AVAILABLE", True)
    @patch("vioratalk.core.stt.faster_whisper_engine.WhisperModel")
    def test_successful_initialization(self, mock_whisper_class, stt_config):
        """正常な初期化のテスト"""
        from vioratalk.core.stt.faster_whisper_engine import FasterWhisperEngine

        engine = FasterWhisperEngine(stt_config)

        assert engine._state == ComponentState.NOT_INITIALIZED
        assert engine.current_model == "base"
        assert engine.config.language == "ja"
        assert engine.config.device == "cpu"
        assert engine.supported_languages == ["ja", "en"]

    @patch("vioratalk.core.stt.faster_whisper_engine.FASTER_WHISPER_AVAILABLE", True)
    @patch("vioratalk.core.stt.faster_whisper_engine.WhisperModel")
    def test_device_auto_selection_cpu(self, mock_whisper_class):
        """デバイス自動選択（CPU）のテスト"""
        from vioratalk.core.stt.faster_whisper_engine import FasterWhisperEngine

        # torchが利用不可またはCUDA利用不可の場合（修正: sys.modulesでモック）
        with patch.dict("sys.modules", {"torch": None}):
            config = STTConfig(device="auto")
            engine = FasterWhisperEngine(config)

            assert engine.config.device == "cpu"
            assert engine.config.compute_type == "int8"

    @patch("vioratalk.core.stt.faster_whisper_engine.FASTER_WHISPER_AVAILABLE", True)
    @patch("vioratalk.core.stt.faster_whisper_engine.WhisperModel")
    def test_device_auto_selection_cuda(self, mock_whisper_class):
        """デバイス自動選択（CUDA）のテスト"""
        from vioratalk.core.stt.faster_whisper_engine import FasterWhisperEngine

        # CUDA利用可能な場合のモック
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        with patch.dict("sys.modules", {"torch": mock_torch}):
            config = STTConfig(device="auto")
            engine = FasterWhisperEngine(config)

            assert engine.config.device == "cuda"
            assert engine.config.compute_type == "float16"


# ============================================================================
# 非同期初期化テスト
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.phase(4)
class TestAsyncInitialization:
    """非同期初期化のテスト"""

    @patch("vioratalk.core.stt.faster_whisper_engine.FASTER_WHISPER_AVAILABLE", True)
    @patch("vioratalk.core.stt.faster_whisper_engine.WhisperModel")
    async def test_async_initialize(self, mock_whisper_class, mock_model_download_manager):
        """非同期初期化の正常系テスト"""
        from vioratalk.core.stt.faster_whisper_engine import FasterWhisperEngine

        # モックを設定
        mock_whisper_class.return_value = MagicMock()

        engine = FasterWhisperEngine()
        engine.model_manager = mock_model_download_manager

        # 初期化実行
        await engine.initialize()

        assert engine._state == ComponentState.READY
        assert engine._initialized_at is not None
        assert engine._model_loaded is True
        assert engine.model is not None

        # ModelDownloadManagerが呼ばれたことを確認
        # download_whisper_modelメソッドが呼ばれたことを確認（Part 24-25での変更）
        mock_model_download_manager.download_whisper_model.assert_called_once()

    @patch("vioratalk.core.stt.faster_whisper_engine.FASTER_WHISPER_AVAILABLE", True)
    @patch("vioratalk.core.stt.faster_whisper_engine.WhisperModel")
    async def test_initialize_twice(self, mock_whisper_class, mock_model_download_manager):
        """二重初期化のテスト"""
        from vioratalk.core.stt.faster_whisper_engine import FasterWhisperEngine

        mock_whisper_class.return_value = MagicMock()

        engine = FasterWhisperEngine()
        engine.model_manager = mock_model_download_manager

        # 1回目の初期化
        await engine.initialize()
        first_init_time = engine._initialized_at

        # 2回目の初期化（警告が出るが処理は続行）
        await engine.initialize()

        # 初期化時刻が変わっていないことを確認
        assert engine._initialized_at == first_init_time
        assert engine._state == ComponentState.READY

    @patch("vioratalk.core.stt.faster_whisper_engine.FASTER_WHISPER_AVAILABLE", True)
    @patch("vioratalk.core.stt.faster_whisper_engine.WhisperModel")
    async def test_initialize_with_model_error(self, mock_whisper_class):
        """モデルロードエラーのテスト"""
        from vioratalk.core.stt.faster_whisper_engine import FasterWhisperEngine

        # WhisperModelのロードでエラーを発生させる
        mock_whisper_class.side_effect = Exception("Model load failed")

        engine = FasterWhisperEngine()

        with pytest.raises(InitializationError) as exc_info:
            await engine.initialize()

        assert engine._state == ComponentState.ERROR
        assert exc_info.value.error_code == "E0007"
        assert "Failed to initialize FasterWhisperEngine" in str(exc_info.value)


# ============================================================================
# 音声認識テスト
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.phase(4)
class TestTranscription:
    """音声認識機能のテスト"""

    @patch("vioratalk.core.stt.faster_whisper_engine.FASTER_WHISPER_AVAILABLE", True)
    async def test_transcribe_success(self, mock_whisper_model, valid_audio_data):
        """正常な音声認識のテスト"""
        from vioratalk.core.stt.faster_whisper_engine import FasterWhisperEngine

        with patch(
            "vioratalk.core.stt.faster_whisper_engine.WhisperModel", return_value=mock_whisper_model
        ):
            engine = FasterWhisperEngine()
            engine._state = ComponentState.READY
            engine._model_loaded = True
            engine.model = mock_whisper_model

            # 音声認識実行
            result = await engine.transcribe(valid_audio_data, language="ja")

            assert isinstance(result, TranscriptionResult)
            assert result.text == "これはテスト音声です"
            assert result.language == "ja"
            assert 0.0 <= result.confidence <= 1.0
            assert result.duration >= 0  # 処理時間は0以上（テストでは瞬時に完了）

    @patch("vioratalk.core.stt.faster_whisper_engine.FASTER_WHISPER_AVAILABLE", True)
    async def test_transcribe_not_ready(self, valid_audio_data):
        """エンジン未準備時のテスト"""
        from vioratalk.core.stt.faster_whisper_engine import FasterWhisperEngine

        with patch("vioratalk.core.stt.faster_whisper_engine.WhisperModel"):
            engine = FasterWhisperEngine()
            # 初期化していない状態

            with pytest.raises(STTError) as exc_info:
                await engine.transcribe(valid_audio_data)

            assert exc_info.value.error_code == "E1000"
            assert "STT engine is not ready" in str(exc_info.value)

    @patch("vioratalk.core.stt.faster_whisper_engine.FASTER_WHISPER_AVAILABLE", True)
    async def test_transcribe_invalid_audio(self):
        """無効な音声データのテスト"""
        from vioratalk.core.stt.faster_whisper_engine import FasterWhisperEngine

        with patch("vioratalk.core.stt.faster_whisper_engine.WhisperModel"):
            engine = FasterWhisperEngine()
            engine._state = ComponentState.READY
            engine._model_loaded = True
            engine.model = MagicMock()

            # 無効な音声データ（空のデータ）
            invalid_audio = AudioData(
                raw_data=np.array([]),  # 空の配列
                metadata=AudioMetadata(sample_rate=16000, channels=1),  # 有効な値
            )

            # 空の配列はSTTError（内部でAudioError）として処理される
            with pytest.raises(STTError) as exc_info:
                await engine.transcribe(invalid_audio)

            # エラーメッセージに "empty" が含まれることを確認
            assert "empty" in str(exc_info.value).lower()

    @patch("vioratalk.core.stt.faster_whisper_engine.FASTER_WHISPER_AVAILABLE", True)
    async def test_transcribe_with_alternatives(self, valid_audio_data):
        """代替候補生成のテスト"""
        from vioratalk.core.stt.faster_whisper_engine import FasterWhisperEngine

        # 低信頼度のモック
        mock_model = MagicMock()
        mock_segment = MagicMock()
        mock_segment.text = "これはテスト音声です"
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.avg_logprob = -1.0  # 低信頼度

        mock_info = MagicMock()
        mock_info.language = "ja"

        mock_model.transcribe.return_value = ([mock_segment], mock_info)

        with patch(
            "vioratalk.core.stt.faster_whisper_engine.WhisperModel", return_value=mock_model
        ):
            engine = FasterWhisperEngine()
            engine._state = ComponentState.READY
            engine._model_loaded = True
            engine.model = mock_model

            result = await engine.transcribe(valid_audio_data)

            # 信頼度が低い場合は代替候補が生成される
            assert len(result.alternatives) > 0
            assert result.alternatives[0]["text"] == "これはテスト音声です？"


# ============================================================================
# モデル管理テスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestModelManagement:
    """モデル管理機能のテスト"""

    @patch("vioratalk.core.stt.faster_whisper_engine.FASTER_WHISPER_AVAILABLE", True)
    @patch("vioratalk.core.stt.faster_whisper_engine.WhisperModel")
    def test_set_model_valid(self, mock_whisper_class):
        """有効なモデル設定のテスト"""
        from vioratalk.core.stt.faster_whisper_engine import FasterWhisperEngine

        engine = FasterWhisperEngine()

        # 有効なモデルに変更
        engine.set_model("small")

        assert engine.current_model == "small"
        assert engine.config.model == "small"
        assert engine._model_loaded is False  # 再ロードが必要
        assert engine._state == ComponentState.NOT_INITIALIZED

    @patch("vioratalk.core.stt.faster_whisper_engine.FASTER_WHISPER_AVAILABLE", True)
    @patch("vioratalk.core.stt.faster_whisper_engine.WhisperModel")
    def test_set_model_invalid(self, mock_whisper_class):
        """無効なモデル設定のテスト"""
        from vioratalk.core.stt.faster_whisper_engine import FasterWhisperEngine

        engine = FasterWhisperEngine()

        with pytest.raises(ModelNotFoundError) as exc_info:
            engine.set_model("invalid_model")

        assert exc_info.value.error_code == "E2100"
        assert "Unsupported model" in str(exc_info.value)

    @patch("vioratalk.core.stt.faster_whisper_engine.FASTER_WHISPER_AVAILABLE", True)
    @patch("vioratalk.core.stt.faster_whisper_engine.WhisperModel")
    def test_set_model_while_processing(self, mock_whisper_class):
        """処理中のモデル変更テスト"""
        from vioratalk.core.stt.faster_whisper_engine import FasterWhisperEngine

        engine = FasterWhisperEngine()
        engine._processing = True  # 処理中フラグを立てる

        with pytest.raises(RuntimeError) as exc_info:
            engine.set_model("small")

        assert "Cannot change model while processing" in str(exc_info.value)

    @patch("vioratalk.core.stt.faster_whisper_engine.FASTER_WHISPER_AVAILABLE", True)
    @patch("vioratalk.core.stt.faster_whisper_engine.WhisperModel")
    def test_get_supported_languages(self, mock_whisper_class):
        """サポート言語取得のテスト"""
        from vioratalk.core.stt.faster_whisper_engine import FasterWhisperEngine

        engine = FasterWhisperEngine()
        languages = engine.get_supported_languages()

        assert isinstance(languages, list)
        assert "ja" in languages
        assert "en" in languages
        assert len(languages) == 2  # Phase 4では2言語のみ

    @patch("vioratalk.core.stt.faster_whisper_engine.FASTER_WHISPER_AVAILABLE", True)
    @patch("vioratalk.core.stt.faster_whisper_engine.WhisperModel")
    def test_get_model_info(self, mock_whisper_class):
        """モデル情報取得のテスト"""
        from vioratalk.core.stt.faster_whisper_engine import FasterWhisperEngine

        engine = FasterWhisperEngine()
        info = engine.get_model_info()

        assert info["name"] == "base"
        assert info["engine"] == "faster-whisper"
        assert info["device"] == "cpu"
        assert info["size_mb"] == 74
        assert info["parameters"] == "74M"
        assert info["compute_type"] == "int8"


# ============================================================================
# クリーンアップテスト
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.phase(4)
class TestCleanup:
    """クリーンアップ機能のテスト"""

    @patch("vioratalk.core.stt.faster_whisper_engine.FASTER_WHISPER_AVAILABLE", True)
    @patch("vioratalk.core.stt.faster_whisper_engine.WhisperModel")
    async def test_cleanup_success(self, mock_whisper_class):
        """正常なクリーンアップのテスト"""
        from vioratalk.core.stt.faster_whisper_engine import FasterWhisperEngine

        mock_model = MagicMock()
        mock_whisper_class.return_value = mock_model

        engine = FasterWhisperEngine()
        engine._state = ComponentState.READY
        engine._model_loaded = True
        engine.model = mock_model

        # クリーンアップ実行
        await engine.cleanup()

        assert engine._state == ComponentState.TERMINATED
        assert engine.model is None
        assert engine._model_loaded is False

    @patch("vioratalk.core.stt.faster_whisper_engine.FASTER_WHISPER_AVAILABLE", True)
    @patch("vioratalk.core.stt.faster_whisper_engine.WhisperModel")
    async def test_cleanup_while_processing(self, mock_whisper_class):
        """処理中のクリーンアップテスト"""
        from vioratalk.core.stt.faster_whisper_engine import FasterWhisperEngine

        engine = FasterWhisperEngine()
        engine._state = ComponentState.READY
        engine._processing = True

        # 処理中フラグを一定時間後にFalseにする
        async def clear_processing():
            await asyncio.sleep(0.2)
            engine._processing = False

        # 並行してクリーンアップを実行
        task = asyncio.create_task(clear_processing())
        await engine.cleanup()
        await task

        assert engine._state == ComponentState.TERMINATED
        assert not engine._processing


# ============================================================================
# ユーティリティメソッドテスト
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.phase(4)
class TestUtilityMethods:
    """ユーティリティメソッドのテスト"""

    @patch("vioratalk.core.stt.faster_whisper_engine.FASTER_WHISPER_AVAILABLE", True)
    @patch("vioratalk.core.stt.faster_whisper_engine.WhisperModel")
    async def test_is_available(self, mock_whisper_class):
        """利用可能性チェックのテスト"""
        from vioratalk.core.stt.faster_whisper_engine import FasterWhisperEngine

        engine = FasterWhisperEngine()

        # 初期化前
        assert not await engine.is_available()

        # 初期化後
        engine._state = ComponentState.READY
        engine._model_loaded = True
        engine.model = MagicMock()

        assert await engine.is_available()

        # エラー状態
        engine._state = ComponentState.ERROR

        assert not await engine.is_available()

    @patch("vioratalk.core.stt.faster_whisper_engine.FASTER_WHISPER_AVAILABLE", True)
    @patch("vioratalk.core.stt.faster_whisper_engine.WhisperModel")
    async def test_validate_audio(self, mock_whisper_class, valid_audio_data):
        """音声データ検証のテスト"""
        from vioratalk.core.stt.faster_whisper_engine import FasterWhisperEngine

        engine = FasterWhisperEngine()

        # 有効なデータ
        assert await engine.validate_audio(valid_audio_data)

        # Noneデータ
        assert not await engine.validate_audio(None)

        # 無効なメタデータ（修正: sample_rateを有効な値に）
        invalid_audio = AudioData(
            raw_data=np.array([1, 2, 3]),
            metadata=AudioMetadata(sample_rate=8000, channels=0),  # 有効だが低い値  # 無効な値
        )
        assert not await engine.validate_audio(invalid_audio)


# ============================================================================
# エンコード済み音声データテスト
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.phase(4)
class TestEncodedAudioData:
    """エンコード済み音声データのテスト"""

    @patch("vioratalk.core.stt.faster_whisper_engine.FASTER_WHISPER_AVAILABLE", True)
    async def test_transcribe_with_encoded_data(self, mock_whisper_model):
        """エンコード済みデータの処理テスト"""
        from vioratalk.core.stt.faster_whisper_engine import FasterWhisperEngine

        # WAVヘッダー付きのダミーデータ
        wav_header = b"RIFF" + b"\x00" * 40  # 簡易WAVヘッダー
        encoded_audio = AudioData(
            encoded_data=wav_header + b"\x00" * 1000, metadata=AudioMetadata(format="wav")
        )

        with patch(
            "vioratalk.core.stt.faster_whisper_engine.WhisperModel", return_value=mock_whisper_model
        ):
            engine = FasterWhisperEngine()
            engine._state = ComponentState.READY
            engine._model_loaded = True
            engine.model = mock_whisper_model

            # 一時ファイル作成を適切にモック
            mock_temp_file = MagicMock()
            mock_temp_file.name = "/tmp/test_audio.wav"
            mock_temp_file.write = MagicMock()

            with patch("tempfile.NamedTemporaryFile", return_value=mock_temp_file):
                with patch("os.path.exists", return_value=True):
                    with patch("os.unlink"):
                        result = await engine.transcribe(encoded_audio)

            assert isinstance(result, TranscriptionResult)
            assert result.text == "これはテスト音声です"


# ============================================================================
# ファクトリ関数テスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestFactoryFunction:
    """ファクトリ関数のテスト"""

    @patch("vioratalk.core.stt.faster_whisper_engine.FASTER_WHISPER_AVAILABLE", True)
    @patch("vioratalk.core.stt.faster_whisper_engine.WhisperModel")
    def test_create_faster_whisper_engine(self, mock_whisper_class):
        """ファクトリ関数のテスト"""
        from vioratalk.core.stt.faster_whisper_engine import create_faster_whisper_engine

        # 設定なしで作成
        engine = create_faster_whisper_engine()
        assert engine.config.engine == "faster-whisper"
        assert engine.config.model == "base"

        # 設定ありで作成
        config = {"model": "large", "language": "en", "device": "cuda"}
        engine = create_faster_whisper_engine(config)
        assert engine.config.model == "large"
        assert engine.config.language == "en"
        assert engine.config.device == "cuda"


# ============================================================================
# エッジケーステスト
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.phase(4)
class TestEdgeCases:
    """エッジケースのテスト"""

    @patch("vioratalk.core.stt.faster_whisper_engine.FASTER_WHISPER_AVAILABLE", True)
    async def test_transcribe_silent_audio(self, mock_whisper_model):
        """無音音声のテスト"""
        from vioratalk.core.stt.faster_whisper_engine import FasterWhisperEngine

        # 無音のモック応答
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], MagicMock(language="ja"))

        with patch(
            "vioratalk.core.stt.faster_whisper_engine.WhisperModel", return_value=mock_model
        ):
            engine = FasterWhisperEngine()
            engine._state = ComponentState.READY
            engine._model_loaded = True
            engine.model = mock_model

            # 無音データ
            silent_audio = AudioData(
                raw_data=np.zeros(16000, dtype=np.float32),
                metadata=AudioMetadata(sample_rate=16000),
            )

            result = await engine.transcribe(silent_audio)

            assert result.text == ""
            assert result.confidence == 0.0

    @patch("vioratalk.core.stt.faster_whisper_engine.FASTER_WHISPER_AVAILABLE", True)
    async def test_transcribe_very_short_audio(self):
        """非常に短い音声のテスト"""
        from vioratalk.core.stt.faster_whisper_engine import FasterWhisperEngine

        mock_model = MagicMock()
        mock_segment = MagicMock()
        mock_segment.text = "あ"
        mock_segment.start = 0.0
        mock_segment.end = 0.1
        mock_segment.avg_logprob = -0.3

        mock_model.transcribe.return_value = ([mock_segment], MagicMock(language="ja"))

        with patch(
            "vioratalk.core.stt.faster_whisper_engine.WhisperModel", return_value=mock_model
        ):
            engine = FasterWhisperEngine()
            engine._state = ComponentState.READY
            engine._model_loaded = True
            engine.model = mock_model

            # 0.1秒の音声
            short_audio = AudioData(
                raw_data=np.random.randn(1600).astype(np.float32),
                metadata=AudioMetadata(sample_rate=16000, duration=0.1),
            )

            result = await engine.transcribe(short_audio)

            assert result.text == "あ"
            assert result.duration >= 0  # 処理時間は0以上


# ============================================================================
# パフォーマンステスト（マーク付き）
# ============================================================================


@pytest.mark.slow
@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.phase(4)
class TestPerformance:
    """パフォーマンス関連のテスト（通常はスキップ）"""

    @patch("vioratalk.core.stt.faster_whisper_engine.FASTER_WHISPER_AVAILABLE", True)
    async def test_transcribe_performance(self, mock_whisper_model):
        """音声認識のパフォーマンステスト"""
        import time

        from vioratalk.core.stt.faster_whisper_engine import FasterWhisperEngine

        with patch(
            "vioratalk.core.stt.faster_whisper_engine.WhisperModel", return_value=mock_whisper_model
        ):
            engine = FasterWhisperEngine()
            engine._state = ComponentState.READY
            engine._model_loaded = True
            engine.model = mock_whisper_model

            # 10秒の音声データ
            long_audio = AudioData(
                raw_data=np.random.randn(160000).astype(np.float32),
                metadata=AudioMetadata(sample_rate=16000, duration=10.0),
            )

            start_time = time.time()
            result = await engine.transcribe(long_audio)
            elapsed = time.time() - start_time

            # 処理時間が音声長の50%以下であることを確認（リアルタイム処理）
            assert elapsed < 5.0, f"Processing took {elapsed:.2f}s for 10s audio"


# ============================================================================
# エクスポート
# ============================================================================

__all__ = [
    "TestFasterWhisperEngineInitialization",
    "TestAsyncInitialization",
    "TestTranscription",
    "TestModelManagement",
    "TestCleanup",
    "TestUtilityMethods",
    "TestEncodedAudioData",
    "TestFactoryFunction",
    "TestEdgeCases",
    "TestPerformance",
]
