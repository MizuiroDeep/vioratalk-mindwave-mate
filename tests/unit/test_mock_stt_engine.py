"""test_mock_stt_engine.py - MockSTTEngineテスト

MockSTTEngineの単体テスト。
音声認識エンジンのモック実装が正しく動作することを確認。

テスト実装ガイド v1.3準拠
テスト戦略ガイドライン v1.7準拠
開発規約書 v1.12準拠
"""

import asyncio

import pytest

# テスト対象のインポート
from tests.mocks.mock_stt_engine import AudioData, MockSTTEngine, TranscriptionResult
from vioratalk.core.base import ComponentState
from vioratalk.core.exceptions import AudioError, ModelNotFoundError, STTError

# ============================================================================
# フィクスチャ
# ============================================================================


@pytest.fixture
async def mock_stt_engine():
    """MockSTTEngineのフィクスチャ"""
    engine = MockSTTEngine()
    await engine.initialize()
    yield engine
    await engine.cleanup()


@pytest.fixture
def sample_audio_data():
    """サンプル音声データのフィクスチャ"""
    return AudioData(
        data=b"sample_audio_data",
        sample_rate=16000,
        duration=3.0,
        metadata={"filename": "test.wav"},
    )


@pytest.fixture
def config_with_delay():
    """遅延設定付きの設定"""
    return {"delay": 0.5, "model": "large"}


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
        engine = MockSTTEngine()

        # 初期状態の確認
        assert engine.state == ComponentState.NOT_INITIALIZED
        assert engine.current_model == "base"

        # 初期化
        await engine.initialize()

        # 初期化後の状態確認
        assert engine.state == ComponentState.READY
        assert engine.transcription_delay == 0.1
        assert engine.error_mode is False

        await engine.cleanup()

    async def test_initialization_with_config(self, config_with_delay):
        """設定付き初期化の確認"""
        engine = MockSTTEngine(config=config_with_delay)

        # 設定が反映されているか確認
        assert engine.transcription_delay == 0.5
        assert engine.current_model == "large"

        await engine.initialize()
        assert engine.state == ComponentState.READY

        await engine.cleanup()

    async def test_cleanup(self, mock_stt_engine):
        """クリーンアップ処理の確認"""
        # 初期化済みの状態から
        assert mock_stt_engine.state == ComponentState.READY

        # クリーンアップ実行
        await mock_stt_engine.cleanup()

        # 終了状態の確認
        assert mock_stt_engine.state == ComponentState.TERMINATED


# ============================================================================
# transcribeメソッドテスト
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.phase(3)
class TestTranscribe:
    """transcribeメソッドのテスト"""

    async def test_transcribe_default_response(self, mock_stt_engine, sample_audio_data):
        """デフォルトレスポンスの確認"""
        # メタデータをdefault用に変更
        sample_audio_data.metadata = {"filename": "unknown.wav"}

        result = await mock_stt_engine.transcribe(sample_audio_data)

        # 結果の検証
        assert isinstance(result, TranscriptionResult)
        assert result.text == "これはテスト用の音声認識結果です。"
        assert result.confidence == 0.95
        assert result.language == "ja"
        assert result.duration == 3.0
        assert len(result.alternatives) == 2

    async def test_transcribe_custom_response(self, mock_stt_engine, sample_audio_data):
        """カスタムレスポンスの確認"""
        # greetingレスポンスを取得
        sample_audio_data.metadata = {"filename": "greeting.wav"}

        result = await mock_stt_engine.transcribe(sample_audio_data)

        assert result.text == "こんにちは、今日はいい天気ですね。"
        assert result.confidence == 0.95

    async def test_transcribe_with_language(self, mock_stt_engine, sample_audio_data):
        """言語指定付きtranscribeの確認"""
        result = await mock_stt_engine.transcribe(sample_audio_data, language="en")

        assert result.language == "en"
        assert result.confidence == 0.95

    async def test_transcribe_unsupported_language(self, mock_stt_engine, sample_audio_data):
        """サポートされていない言語の処理"""
        # サポートされていない言語を指定
        result = await mock_stt_engine.transcribe(sample_audio_data, language="fr")

        # 日本語にフォールバック
        assert result.language == "ja"

    async def test_transcribe_not_ready_state(self, sample_audio_data):
        """初期化前のtranscribe呼び出し"""
        engine = MockSTTEngine()

        # 初期化せずにtranscribeを呼び出し
        with pytest.raises(STTError) as exc_info:
            await engine.transcribe(sample_audio_data)

        assert exc_info.value.error_code == "E1000"
        assert "not ready" in str(exc_info.value)

    async def test_transcribe_error_mode(self, mock_stt_engine, sample_audio_data):
        """エラーモード時の動作確認"""
        # エラーモードを有効化
        mock_stt_engine.set_error_mode(True)

        with pytest.raises(AudioError) as exc_info:
            await mock_stt_engine.transcribe(sample_audio_data)

        assert exc_info.value.error_code == "E1001"
        assert "Mock audio processing error" in str(exc_info.value)


# ============================================================================
# 言語サポートテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(3)
class TestLanguageSupport:
    """言語サポート機能のテスト"""

    def test_get_supported_languages(self, mock_stt_engine):
        """サポート言語リストの取得"""
        languages = mock_stt_engine.get_supported_languages()

        assert isinstance(languages, list)
        assert len(languages) == 4
        assert "ja" in languages
        assert "en" in languages
        assert "zh" in languages
        assert "ko" in languages

    def test_supported_languages_immutable(self, mock_stt_engine):
        """サポート言語リストの不変性確認"""
        languages = mock_stt_engine.get_supported_languages()
        original_len = len(languages)

        # リストを変更しても元のデータに影響しない
        languages.append("fr")

        new_languages = mock_stt_engine.get_supported_languages()
        assert len(new_languages) == original_len


# ============================================================================
# モデル設定テスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(3)
class TestModelSettings:
    """モデル設定機能のテスト"""

    def test_set_model_valid(self, mock_stt_engine):
        """有効なモデル名の設定"""
        # 初期モデル確認
        assert mock_stt_engine.current_model == "base"

        # モデル変更
        mock_stt_engine.set_model("large")
        assert mock_stt_engine.current_model == "large"

        # 別のモデルに変更
        mock_stt_engine.set_model("tiny")
        assert mock_stt_engine.current_model == "tiny"

    def test_set_model_invalid(self, mock_stt_engine):
        """無効なモデル名の設定"""
        with pytest.raises(ModelNotFoundError) as exc_info:
            mock_stt_engine.set_model("invalid_model")

        assert exc_info.value.error_code == "E2100"
        assert "Model 'invalid_model' not found" in str(exc_info.value)

        # モデルは変更されていない
        assert mock_stt_engine.current_model == "base"

    def test_available_models(self, mock_stt_engine):
        """利用可能なモデルの確認"""
        available_models = ["tiny", "base", "small", "medium", "large"]

        for model in available_models:
            # エラーが発生しないことを確認
            mock_stt_engine.set_model(model)
            assert mock_stt_engine.current_model == model


# ============================================================================
# テスト用メソッドテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(3)
class TestUtilityMethods:
    """テスト用ユーティリティメソッドのテスト"""

    def test_set_error_mode(self, mock_stt_engine):
        """エラーモード設定の確認"""
        # 初期状態
        assert mock_stt_engine.error_mode is False

        # エラーモード有効化
        mock_stt_engine.set_error_mode(True)
        assert mock_stt_engine.error_mode is True

        # エラーモード無効化
        mock_stt_engine.set_error_mode(False)
        assert mock_stt_engine.error_mode is False

    async def test_set_custom_response(self, mock_stt_engine):
        """カスタムレスポンス設定の確認"""
        # カスタムレスポンスを設定
        custom_text = "カスタムテキスト応答"
        mock_stt_engine.set_custom_response("custom.wav", custom_text)

        # カスタムレスポンスが返されることを確認
        audio_data = AudioData(data=b"test", metadata={"filename": "custom.wav"})

        result = await mock_stt_engine.transcribe(audio_data)
        assert result.text == custom_text

    def test_set_transcription_delay(self, mock_stt_engine):
        """音声認識遅延設定の確認"""
        # 初期値確認
        assert mock_stt_engine.transcription_delay == 0.1

        # 遅延設定
        mock_stt_engine.set_transcription_delay(0.5)
        assert mock_stt_engine.transcription_delay == 0.5

        # 負の値は0にクリップ
        mock_stt_engine.set_transcription_delay(-1.0)
        assert mock_stt_engine.transcription_delay == 0.0

    def test_get_statistics(self, mock_stt_engine):
        """統計情報取得の確認"""
        # カスタム設定を追加
        mock_stt_engine.set_custom_response("test1.wav", "テスト1")
        mock_stt_engine.set_custom_response("test2.wav", "テスト2")
        mock_stt_engine.set_error_mode(True)
        mock_stt_engine.set_model("large")

        stats = mock_stt_engine.get_statistics()

        assert stats["model"] == "large"
        assert stats["supported_languages"] == ["ja", "en", "zh", "ko"]
        assert stats["error_mode"] is True
        assert stats["custom_responses_count"] == 7  # デフォルト5 + カスタム2
        assert stats["transcription_delay"] == 0.1


# ============================================================================
# 非同期処理テスト
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.phase(3)
class TestAsyncBehavior:
    """非同期処理の動作テスト"""

    async def test_transcription_delay(self, mock_stt_engine, sample_audio_data):
        """音声認識遅延のテスト"""
        # 遅延を0.2秒に設定
        mock_stt_engine.set_transcription_delay(0.2)

        # 処理時間を計測
        import time

        start_time = time.time()

        result = await mock_stt_engine.transcribe(sample_audio_data)

        elapsed_time = time.time() - start_time

        # 遅延が適用されていることを確認（誤差を考慮）
        assert elapsed_time >= 0.2
        assert elapsed_time < 0.3
        assert result.text is not None

    async def test_concurrent_transcriptions(self, mock_stt_engine):
        """並行処理のテスト"""
        # 複数の音声データを準備
        audio_data_list = [
            AudioData(data=b"data1", metadata={"filename": "greeting.wav"}),
            AudioData(data=b"data2", metadata={"filename": "question.wav"}),
            AudioData(data=b"data3", metadata={"filename": "command.wav"}),
        ]

        # 並行実行
        results = await asyncio.gather(
            *[mock_stt_engine.transcribe(audio) for audio in audio_data_list]
        )

        # 結果の検証
        assert len(results) == 3
        assert results[0].text == "こんにちは、今日はいい天気ですね。"
        assert results[1].text == "何かお困りのことはありますか？"
        assert results[2].text == "音楽を再生してください。"


# ============================================================================
# エッジケーステスト
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.phase(3)
class TestEdgeCases:
    """エッジケースのテスト"""

    async def test_empty_audio_data(self, mock_stt_engine):
        """空の音声データの処理"""
        audio_data = AudioData(data=b"", duration=0.0, metadata={})

        result = await mock_stt_engine.transcribe(audio_data)

        # デフォルトレスポンスが返される
        assert result.text == "これはテスト用の音声認識結果です。"
        assert result.duration == 3.0  # デフォルト値

    async def test_large_audio_data(self, mock_stt_engine):
        """大きな音声データの処理"""
        # 1MBの音声データ
        large_data = b"x" * 1024 * 1024
        audio_data = AudioData(data=large_data, duration=60.0, metadata={"filename": "large.wav"})

        result = await mock_stt_engine.transcribe(audio_data)

        assert result.text is not None
        assert result.duration == 60.0

    async def test_special_characters_in_filename(self, mock_stt_engine):
        """ファイル名に特殊文字が含まれる場合"""
        audio_data = AudioData(data=b"test", metadata={"filename": "テスト_音声(1).wav"})

        result = await mock_stt_engine.transcribe(audio_data)

        # デフォルトレスポンスが返される
        assert result.text == "これはテスト用の音声認識結果です。"


# ============================================================================
# 代替候補テスト
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.phase(3)
class TestAlternatives:
    """代替候補生成のテスト"""

    async def test_alternatives_generation(self, mock_stt_engine, sample_audio_data):
        """代替候補が生成されることを確認"""
        result = await mock_stt_engine.transcribe(sample_audio_data)

        # 代替候補が存在する
        assert len(result.alternatives) > 0

        # 代替候補の形式確認
        for alt in result.alternatives:
            assert "text" in alt
            assert "confidence" in alt
            assert isinstance(alt["confidence"], float)
            assert 0.0 <= alt["confidence"] <= 1.0

    async def test_alternatives_confidence_order(self, mock_stt_engine, sample_audio_data):
        """代替候補の信頼度順序確認"""
        result = await mock_stt_engine.transcribe(sample_audio_data)

        if len(result.alternatives) > 1:
            # 信頼度が降順になっていることを確認
            for i in range(len(result.alternatives) - 1):
                assert (
                    result.alternatives[i]["confidence"] >= result.alternatives[i + 1]["confidence"]
                )

    async def test_short_text_no_alternatives(self, mock_stt_engine):
        """短いテキストでは代替候補が生成されない"""
        # 短いレスポンスを設定
        mock_stt_engine.set_custom_response("short.wav", "はい")

        audio_data = AudioData(data=b"test", metadata={"filename": "short.wav"})

        result = await mock_stt_engine.transcribe(audio_data)

        # 短いテキストなので代替候補なし
        assert len(result.alternatives) == 0


# ============================================================================
# 状態管理テスト
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.phase(3)
class TestStateManagement:
    """状態管理のテスト"""

    async def test_state_transitions(self):
        """状態遷移の確認"""
        engine = MockSTTEngine()

        # 初期状態
        assert engine.state == ComponentState.NOT_INITIALIZED

        # 初期化
        await engine.initialize()
        assert engine.state == ComponentState.READY

        # クリーンアップ
        await engine.cleanup()
        assert engine.state == ComponentState.TERMINATED

    async def test_multiple_cleanup_calls(self, mock_stt_engine):
        """複数回のクリーンアップ呼び出し"""
        # 1回目のクリーンアップ
        await mock_stt_engine.cleanup()
        assert mock_stt_engine.state == ComponentState.TERMINATED

        # 2回目のクリーンアップ（エラーが発生しないことを確認）
        await mock_stt_engine.cleanup()
        assert mock_stt_engine.state == ComponentState.TERMINATED
