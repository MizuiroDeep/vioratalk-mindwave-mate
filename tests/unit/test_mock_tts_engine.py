"""test_mock_tts_engine.py - MockTTSEngineテスト

MockTTSEngineの単体テスト。
音声合成エンジンのモック実装が正しく動作することを確認。

テスト実装ガイド v1.3準拠
テスト戦略ガイドライン v1.7準拠
開発規約書 v1.12準拠
データフォーマット仕様書 v1.5準拠
"""

import asyncio
import struct
import time

import pytest

# テスト対象のインポート
from tests.mocks.mock_tts_engine import (
    MockTTSEngine,
    SynthesisResult,
    VoiceInfo,
    VoiceParameters,
)
from vioratalk.core.base import ComponentState
from vioratalk.core.exceptions import AudioError, InvalidVoiceError, TTSError

# ============================================================================
# フィクスチャ
# ============================================================================


@pytest.fixture
async def mock_tts_engine():
    """MockTTSEngineのフィクスチャ"""
    engine = MockTTSEngine()
    await engine.initialize()
    yield engine
    await engine.cleanup()


@pytest.fixture
def config_with_custom_voice():
    """カスタム音声設定"""
    return {"delay": 0.05, "voice_id": "ja-JP-Male-1", "sample_rate": 44100}


@pytest.fixture
def voice_parameters_basic():
    """基本的なVoiceParameters"""
    return VoiceParameters(voice_id="ja-JP-Female-1", speed=1.0, pitch=1.0, volume=0.8)


@pytest.fixture
def voice_parameters_advanced():
    """拡張VoiceParameters（Phase 7-8準備）"""
    return VoiceParameters(
        voice_id="ja-JP-Female-1",
        speed=1.2,
        pitch=0.9,
        volume=0.7,
        style_id="happy",
        intonation_scale=1.5,
        pre_phoneme_length=0.1,
        post_phoneme_length=0.1,
    )


# ============================================================================
# 初期化・終了テスト
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.phase(3)
class TestInitializationAndCleanup:
    """初期化と終了処理のテスト"""

    async def test_initialization_success(self):
        """正常な初期化の確認"""
        engine = MockTTSEngine()

        # 初期状態の確認
        assert engine.state == ComponentState.NOT_INITIALIZED
        assert engine.current_voice_id == "ja-JP-Female-1"
        assert engine.sample_rate == 22050

        # 初期化
        await engine.initialize()

        # 初期化後の状態確認
        assert engine.state == ComponentState.READY
        assert engine.synthesis_delay == 0.1
        assert engine.error_mode is False

        await engine.cleanup()

    async def test_initialization_with_config(self, config_with_custom_voice):
        """設定付き初期化の確認"""
        engine = MockTTSEngine(config=config_with_custom_voice)

        # 設定が反映されているか確認
        assert engine.synthesis_delay == 0.05
        assert engine.current_voice_id == "ja-JP-Male-1"
        assert engine.sample_rate == 44100

        await engine.initialize()
        assert engine.state == ComponentState.READY

        await engine.cleanup()

    async def test_cleanup(self, mock_tts_engine):
        """クリーンアップ処理の確認"""
        # 初期化済みの状態から
        assert mock_tts_engine.state == ComponentState.READY

        # クリーンアップ実行
        await mock_tts_engine.cleanup()

        # 終了状態の確認
        assert mock_tts_engine.state == ComponentState.TERMINATED


# ============================================================================
# synthesizeメソッドテスト
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.phase(3)
class TestSynthesize:
    """synthesizeメソッドのテスト"""

    async def test_synthesize_basic(self, mock_tts_engine):
        """基本的な音声合成"""
        text = "こんにちは、世界"
        result = await mock_tts_engine.synthesize(text)

        assert isinstance(result, SynthesisResult)
        assert result.audio_data is not None
        assert len(result.audio_data) > 44  # WAVヘッダー分以上
        assert result.sample_rate == 22050
        assert result.format == "wav"
        assert result.duration > 0

    async def test_synthesize_with_voice_id(self, mock_tts_engine):
        """音声ID指定での合成"""
        text = "Hello, world"
        result = await mock_tts_engine.synthesize(text, voice_id="en-US-Female-1")

        assert result.metadata["voice_id"] == "en-US-Female-1"
        assert result.audio_data is not None

    async def test_synthesize_with_style(self, mock_tts_engine):
        """スタイル指定での合成"""
        text = "楽しいですね"
        result = await mock_tts_engine.synthesize(text, voice_id="ja-JP-Female-1", style="happy")

        assert result.metadata["style"] == "happy"
        assert result.audio_data is not None

    async def test_synthesize_invalid_voice_id(self, mock_tts_engine):
        """無効な音声IDの処理"""
        text = "テスト"

        # 無効な音声IDを指定してもデフォルトにフォールバック
        result = await mock_tts_engine.synthesize(text, voice_id="invalid-voice")

        # デフォルト音声が使用される
        assert result.metadata["voice_id"] == "ja-JP-Female-1"

    async def test_synthesize_not_ready_state(self):
        """未初期化状態での合成エラー"""
        engine = MockTTSEngine()

        with pytest.raises(TTSError) as exc_info:
            await engine.synthesize("test")

        assert exc_info.value.error_code == "E3000"
        assert "not ready" in str(exc_info.value)

    async def test_synthesize_error_mode(self, mock_tts_engine):
        """エラーモードでの動作"""
        mock_tts_engine.set_error_mode(True)

        with pytest.raises(AudioError) as exc_info:
            await mock_tts_engine.synthesize("test")

        assert exc_info.value.error_code == "E3001"


# ============================================================================
# VoiceParameters テスト
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.phase(3)
class TestVoiceParameters:
    """VoiceParametersを使用した音声合成のテスト"""

    async def test_synthesize_with_basic_parameters(self, mock_tts_engine, voice_parameters_basic):
        """基本パラメータでの音声合成"""
        result = await mock_tts_engine.synthesize_with_parameters(
            "パラメータテスト", voice_parameters_basic
        )

        assert isinstance(result, SynthesisResult)
        assert result.metadata["voice_id"] == "ja-JP-Female-1"
        assert result.metadata["style"] is None

        # パラメータはキャッシュされる
        assert mock_tts_engine.last_parameters is not None
        assert mock_tts_engine.last_parameters.speed == 1.0
        assert mock_tts_engine.last_parameters.pitch == 1.0
        assert mock_tts_engine.last_parameters.volume == 0.8

    async def test_synthesize_with_advanced_parameters(
        self, mock_tts_engine, voice_parameters_advanced
    ):
        """拡張パラメータでの音声合成（Phase 7-8準備）"""
        result = await mock_tts_engine.synthesize_with_parameters(
            "拡張パラメータテスト", voice_parameters_advanced
        )

        assert isinstance(result, SynthesisResult)
        assert result.metadata["voice_id"] == "ja-JP-Female-1"
        assert result.metadata["style"] == "happy"

        # パラメータがキャッシュされる
        params = mock_tts_engine.last_parameters
        assert params.speed == 1.2
        assert params.pitch == 0.9
        assert params.volume == 0.7
        assert params.style_id == "happy"
        assert params.intonation_scale == 1.5

    async def test_parameter_validation(self, mock_tts_engine):
        """パラメータ検証のテスト"""
        # 無効な音声IDのパラメータ
        invalid_params = VoiceParameters(voice_id="invalid-voice", speed=1.0)

        # デフォルト音声にフォールバック
        result = await mock_tts_engine.synthesize_with_parameters("検証テスト", invalid_params)

        # デフォルト音声が使用される
        assert result.metadata["voice_id"] == "ja-JP-Female-1"

    async def test_parameters_caching(self, mock_tts_engine):
        """パラメータキャッシングのテスト"""
        # 初期状態
        assert mock_tts_engine.last_parameters is None

        # パラメータで合成
        params = VoiceParameters(voice_id="ja-JP-Female-2", speed=0.8)

        await mock_tts_engine.synthesize_with_parameters("test", params)

        # キャッシュされている
        assert mock_tts_engine.last_parameters is not None
        assert mock_tts_engine.last_parameters.voice_id == "ja-JP-Female-2"
        assert mock_tts_engine.last_parameters.speed == 0.8


# ============================================================================
# 音声管理テスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(3)
class TestVoiceManagement:
    """音声管理機能のテスト"""

    def test_get_available_voices(self, mock_tts_engine):
        """利用可能な音声リストの取得"""
        voices = mock_tts_engine.get_available_voices()

        assert isinstance(voices, list)
        assert len(voices) == 5

        # 音声情報の確認
        first_voice = voices[0]
        assert isinstance(first_voice, VoiceInfo)
        assert first_voice.id == "ja-JP-Female-1"
        assert first_voice.name == "日本語女性1"
        assert first_voice.language == "ja"
        assert first_voice.gender == "female"

    def test_set_voice(self, mock_tts_engine):
        """音声の切り替え"""
        # 初期音声
        assert mock_tts_engine.current_voice_id == "ja-JP-Female-1"

        # 音声を変更
        mock_tts_engine.set_voice("ja-JP-Male-1")
        assert mock_tts_engine.current_voice_id == "ja-JP-Male-1"

    def test_set_invalid_voice(self, mock_tts_engine):
        """無効な音声IDの設定"""
        with pytest.raises(InvalidVoiceError) as exc_info:
            mock_tts_engine.set_voice("invalid-voice-id")

        assert exc_info.value.error_code == "E3001"
        assert "invalid-voice-id" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_test_availability(self, mock_tts_engine):
        """利用可能性テスト"""
        # 正常時
        result = await mock_tts_engine.test_availability()
        assert result is True

        # エラーモード時
        mock_tts_engine.set_error_mode(True)
        result = await mock_tts_engine.test_availability()
        assert result is False


# ============================================================================
# スタイル管理テスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(3)
class TestStyleManagement:
    """スタイル管理機能のテスト（Phase 7-8準備）"""

    def test_get_available_styles(self, mock_tts_engine):
        """利用可能なスタイルの取得"""
        # ja-JP-Female-1のスタイル
        styles = mock_tts_engine.get_available_styles("ja-JP-Female-1")

        assert isinstance(styles, list)
        assert len(styles) == 2  # 実装では2個のスタイル

        # スタイルIDの確認
        style_ids = [s.id for s in styles]
        assert "happy" in style_ids
        assert "sad" in style_ids

        # StyleInfo構造の確認
        happy_style = next(s for s in styles if s.id == "happy")
        assert happy_style.name == "楽しい"
        assert happy_style.description == "明るく楽しげな声"
        assert happy_style.voice_id == "ja-JP-Female-1"

    def test_get_styles_for_voice_without_styles(self, mock_tts_engine):
        """スタイルがない音声のスタイル取得"""
        # en-US-Male-1にはスタイルがない
        styles = mock_tts_engine.get_available_styles("en-US-Male-1")

        assert isinstance(styles, list)
        assert len(styles) == 0

    def test_get_styles_for_invalid_voice(self, mock_tts_engine):
        """無効な音声IDでのスタイル取得"""
        styles = mock_tts_engine.get_available_styles("invalid-voice")

        assert isinstance(styles, list)
        assert len(styles) == 0


# ============================================================================
# WAVデータ生成テスト
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.phase(3)
class TestWAVGeneration:
    """WAVデータ生成のテスト"""

    async def test_wav_header_structure(self, mock_tts_engine):
        """WAVヘッダー構造の確認"""
        result = await mock_tts_engine.synthesize("test")
        audio_data = result.audio_data

        # WAVヘッダーの確認（最初の44バイト）
        assert len(audio_data) >= 44

        # "RIFF"チャンク
        assert audio_data[0:4] == b"RIFF"

        # "WAVE"フォーマット
        assert audio_data[8:12] == b"WAVE"

        # "fmt "サブチャンク
        assert audio_data[12:16] == b"fmt "

        # "data"サブチャンク
        assert audio_data[36:40] == b"data"

    async def test_wav_sample_rate(self, mock_tts_engine):
        """サンプルレートの確認"""
        result = await mock_tts_engine.synthesize("test")
        audio_data = result.audio_data

        # サンプルレート（バイト24-27）
        sample_rate_bytes = audio_data[24:28]
        sample_rate = struct.unpack("<I", sample_rate_bytes)[0]

        assert sample_rate == 22050  # デフォルトサンプルレート

    async def test_wav_audio_content(self, mock_tts_engine):
        """音声データ内容の確認"""
        result = await mock_tts_engine.synthesize("テスト音声")
        audio_data = result.audio_data

        # データ部分（44バイト目以降）
        audio_content = audio_data[44:]

        # データが存在する
        assert len(audio_content) > 0

        # サイン波データとして妥当な範囲（-32768～32767）
        # 2バイトずつ読み取ってチェック
        for i in range(0, min(100, len(audio_content) - 1), 2):
            sample = struct.unpack("<h", audio_content[i : i + 2])[0]
            assert -32768 <= sample <= 32767

    async def test_duration_calculation(self, mock_tts_engine):
        """音声長さ計算の確認"""
        # 短いテキスト
        short_result = await mock_tts_engine.synthesize("短い")
        assert short_result.duration == 0.2  # 2文字 × 0.1秒

        # 長いテキスト（11文字）
        long_result = await mock_tts_engine.synthesize("これは長いテキストです")
        assert long_result.duration == 1.1  # 11文字 × 0.1秒


# ============================================================================
# ユーティリティメソッドテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(3)
class TestUtilityMethods:
    """ユーティリティメソッドのテスト"""

    def test_set_error_mode(self, mock_tts_engine):
        """エラーモード設定のテスト"""
        # 初期状態
        assert mock_tts_engine.error_mode is False

        # エラーモードを有効化
        mock_tts_engine.set_error_mode(True)
        assert mock_tts_engine.error_mode is True

        # エラーモードを無効化
        mock_tts_engine.set_error_mode(False)
        assert mock_tts_engine.error_mode is False

    def test_set_synthesis_delay(self, mock_tts_engine):
        """合成遅延設定のテスト"""
        # 初期値
        assert mock_tts_engine.synthesis_delay == 0.1

        # 遅延を変更
        mock_tts_engine.set_synthesis_delay(0.5)
        assert mock_tts_engine.synthesis_delay == 0.5

        # 負の値は0にクリップ
        mock_tts_engine.set_synthesis_delay(-0.1)
        assert mock_tts_engine.synthesis_delay == 0

    def test_get_statistics(self, mock_tts_engine):
        """統計情報取得のテスト"""
        # 初期状態
        stats = mock_tts_engine.get_statistics()

        assert stats["current_voice"] == "ja-JP-Female-1"
        assert stats["available_voices_count"] == 5
        assert stats["sample_rate"] == 22050
        assert stats["error_mode"] is False
        assert stats["synthesis_delay"] == 0.1
        assert stats["last_parameters"] is None

        # 音声を変更して確認
        mock_tts_engine.set_voice("ja-JP-Male-1")
        mock_tts_engine.set_error_mode(True)

        stats = mock_tts_engine.get_statistics()
        assert stats["current_voice"] == "ja-JP-Male-1"
        assert stats["error_mode"] is True


# ============================================================================
# 非同期動作テスト
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.phase(3)
class TestAsyncBehavior:
    """非同期動作のテスト"""

    async def test_synthesis_delay(self, mock_tts_engine):
        """合成遅延のテスト"""
        # 遅延を設定
        mock_tts_engine.set_synthesis_delay(0.05)

        # 時間を測定
        start_time = time.time()
        await mock_tts_engine.synthesize("test")
        elapsed_time = time.time() - start_time

        # 遅延が適用されている（誤差考慮）
        assert elapsed_time >= 0.04

    async def test_concurrent_synthesis(self, mock_tts_engine):
        """並行合成のテスト"""
        # 複数の合成を並行実行
        tasks = [
            mock_tts_engine.synthesize("テスト1"),
            mock_tts_engine.synthesize("テスト2"),
            mock_tts_engine.synthesize("テスト3"),
        ]

        results = await asyncio.gather(*tasks)

        # すべて成功
        assert len(results) == 3
        for result in results:
            assert isinstance(result, SynthesisResult)
            assert result.audio_data is not None


# ============================================================================
# エッジケーステスト
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.phase(3)
class TestEdgeCases:
    """エッジケースのテスト"""

    async def test_empty_text(self, mock_tts_engine):
        """空文字列の処理"""
        result = await mock_tts_engine.synthesize("")

        assert isinstance(result, SynthesisResult)
        assert result.duration == 0.0
        assert len(result.audio_data) >= 44  # ヘッダーは存在（修正: >= に変更）

    async def test_very_long_text(self, mock_tts_engine):
        """非常に長いテキストの処理"""
        long_text = "あ" * 1000
        result = await mock_tts_engine.synthesize(long_text)

        assert isinstance(result, SynthesisResult)
        assert result.duration == 100.0  # 1000文字 × 0.1秒

    async def test_special_characters(self, mock_tts_engine):
        """特殊文字を含むテキスト"""
        text = "こんにちは！🎵 #VioraTalk @test"
        result = await mock_tts_engine.synthesize(text)

        assert isinstance(result, SynthesisResult)
        assert result.audio_data is not None

    async def test_multiple_voice_changes(self, mock_tts_engine):
        """複数回の音声切り替え"""
        voices = ["ja-JP-Female-1", "en-US-Male-1", "ja-JP-Female-2", "en-US-Female-1"]

        for voice_id in voices:
            mock_tts_engine.set_voice(voice_id)
            result = await mock_tts_engine.synthesize("test")
            assert result.metadata["voice_id"] == voice_id
