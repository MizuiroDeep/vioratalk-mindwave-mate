"""
Pyttsx3Engineの単体テスト

Part 29調査結果の検証を含む包括的なテスト。
エンジン再利用不可問題の確認テストも実施。
VioraTalkComponent準拠で初期化を適切に実行。

テスト戦略ガイドライン v1.7準拠
開発規約書 v1.12準拠
"""

import os
from unittest.mock import MagicMock, Mock, patch

import pytest

from vioratalk.core.base import ComponentState
from vioratalk.core.exceptions import InvalidVoiceError, TTSError
from vioratalk.core.tts.base import SynthesisResult, TTSConfig, VoiceInfo
from vioratalk.core.tts.pyttsx3_engine import Pyttsx3Engine

# ============================================================================
# フィクスチャ
# ============================================================================


@pytest.fixture
def mock_pyttsx3():
    """pyttsx3モジュールのモック（イテレーティブ方式対応版）"""
    with patch("vioratalk.core.tts.pyttsx3_engine.pyttsx3") as mock:
        # エンジンのモック
        mock_engine = MagicMock()

        # 音声オブジェクトの作成（適切な属性を持つ）
        mock_voice_ja = Mock()
        mock_voice_ja.name = "Microsoft Haruka Desktop - Japanese"
        mock_voice_ja.id = "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_JA-JP_HARUKA_11.0"

        mock_voice_en = Mock()
        mock_voice_en.name = "Microsoft Zira Desktop - English (United States)"
        mock_voice_en.id = "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_ZIRA_11.0"

        mock_voices = [mock_voice_ja, mock_voice_en]

        # getPropertyが呼ばれたときに音声リストを返す
        mock_engine.getProperty.return_value = mock_voices
        mock_engine.stop = Mock()
        mock_engine.say = Mock()
        mock_engine.runAndWait = Mock()
        mock_engine.save_to_file = Mock()
        mock_engine.setProperty = Mock()

        # イテレーティブ方式のメソッド
        mock_engine.startLoop = Mock()
        mock_engine.iterate = Mock()
        mock_engine.endLoop = Mock()
        # isBusyは3回目でFalseを返す（短いテキストの場合）
        mock_engine.isBusy = Mock(side_effect=[True, True, False])

        mock.init.return_value = mock_engine
        yield mock


@pytest.fixture
def mock_pyttsx3_no_iterative():
    """イテレーティブ方式をサポートしないpyttsx3のモック"""
    with patch("vioratalk.core.tts.pyttsx3_engine.pyttsx3") as mock:
        mock_engine = MagicMock()

        # 音声オブジェクトの作成
        mock_voice_ja = Mock()
        mock_voice_ja.name = "Microsoft Haruka Desktop - Japanese"
        mock_voice_ja.id = "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_JA-JP_HARUKA_11.0"

        mock_voices = [mock_voice_ja]

        mock_engine.getProperty.return_value = mock_voices
        mock_engine.stop = Mock()
        mock_engine.say = Mock()
        mock_engine.runAndWait = Mock()
        mock_engine.save_to_file = Mock()
        mock_engine.setProperty = Mock()

        # イテレーティブ方式のメソッドがない（AttributeError発生）
        mock_engine.startLoop.side_effect = AttributeError(
            "'pyttsx3.Engine' object has no attribute 'startLoop'"
        )

        mock.init.return_value = mock_engine
        yield mock


@pytest.fixture
def default_config():
    """デフォルトのTTS設定"""
    return TTSConfig(
        engine="pyttsx3",
        language="ja",
        save_audio_data=False,  # デフォルトは直接出力
        speed=1.0,
        volume=0.9,
    )


@pytest.fixture
def save_audio_config():
    """音声データ保存モードの設定"""
    return TTSConfig(
        engine="pyttsx3",
        language="ja",
        save_audio_data=True,  # WAV取得モード
        speed=1.0,
        volume=0.9,
    )


# ============================================================================
# 初期化テスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestInitialization:
    """初期化関連のテスト"""

    def test_initialization_without_pyttsx3(self):
        """pyttsx3がインストールされていない場合のテスト"""
        with patch("vioratalk.core.tts.pyttsx3_engine.pyttsx3", None):
            with pytest.raises(TTSError) as exc_info:
                Pyttsx3Engine()
            assert exc_info.value.error_code == "E3004"
            assert "pyttsx3 is not installed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_initialization_with_default_config(self, mock_pyttsx3, default_config):
        """デフォルト設定での初期化テスト"""
        engine = Pyttsx3Engine(config=default_config)

        # 初期化前の状態確認
        assert engine._state == ComponentState.NOT_INITIALIZED

        # 初期化実行
        await engine.initialize()

        # 初期化後の状態確認
        assert engine._state == ComponentState.READY
        assert engine.config.engine == "pyttsx3"
        assert engine.config.language == "ja"
        assert engine.config.save_audio_data is False
        assert len(engine.available_voices) == 2
        assert engine.current_voice_id == "ja"  # 日本語が優先される

    @pytest.mark.asyncio
    async def test_initialization_with_save_audio_config(self, mock_pyttsx3, save_audio_config):
        """音声データ保存モードでの初期化テスト"""
        engine = Pyttsx3Engine(config=save_audio_config)
        await engine.initialize()

        assert engine._state == ComponentState.READY
        assert engine.config.save_audio_data is True

    @pytest.mark.asyncio
    async def test_initialization_voice_detection(self, mock_pyttsx3):
        """音声検出機能のテスト"""
        engine = Pyttsx3Engine()
        await engine.initialize()

        # 利用可能な音声が正しく検出されている
        assert len(engine.available_voices) == 2

        # 日本語音声
        ja_voice = next((v for v in engine.available_voices if v.id == "ja"), None)
        assert ja_voice is not None
        assert ja_voice.language == "ja"
        assert "Japanese" in ja_voice.name

        # 英語音声
        en_voice = next((v for v in engine.available_voices if v.id == "en"), None)
        assert en_voice is not None
        assert en_voice.language == "en"
        assert "English" in en_voice.name

    @pytest.mark.asyncio
    async def test_initialization_error_handling(self, mock_pyttsx3):
        """初期化エラーのハンドリングテスト"""
        # エンジン初期化時にエラーが発生
        mock_pyttsx3.init.side_effect = [Exception("Init failed"), MagicMock()]

        # エラーが発生してもフォールバック音声情報が設定される
        engine = Pyttsx3Engine()
        assert len(engine.available_voices) == 2
        assert engine.available_voices[0].id == "ja"
        assert engine.available_voices[1].id == "en"

        # initialize()でもエラーハンドリング（フォールバック用途なのでエラーにしない）
        mock_pyttsx3.init.side_effect = None  # エラーを解除
        await engine.initialize()
        assert engine._state == ComponentState.READY

    @pytest.mark.asyncio
    async def test_initialization_no_voices_detected(self, mock_pyttsx3):
        """音声が検出されない場合のテスト"""
        # 空の音声リストを返す
        mock_engine = mock_pyttsx3.init.return_value
        mock_engine.getProperty.return_value = []

        engine = Pyttsx3Engine()
        await engine.initialize()

        # フォールバック音声が設定される
        assert len(engine.available_voices) == 2
        assert engine.available_voices[0].name == "Japanese (Fallback)"
        assert engine.available_voices[1].name == "English (Fallback)"
        assert engine._state == ComponentState.READY


# ============================================================================
# synthesizeメソッドのテスト（直接出力モード）
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestSynthesizeDirect:
    """直接出力モードのsynthesizeテスト"""

    @pytest.mark.asyncio
    async def test_synthesize_direct_mode_iterative(self, mock_pyttsx3, default_config):
        """直接出力モード（イテレーティブ方式）の基本動作テスト"""
        engine = Pyttsx3Engine(config=default_config)
        await engine.initialize()

        # isBusyのモックをリセット（3回目でFalse）
        mock_engine = mock_pyttsx3.init.return_value
        mock_engine.isBusy.reset_mock()
        mock_engine.isBusy.side_effect = [True, True, False]

        # テキストを合成
        result = await engine.synthesize("こんにちは")

        assert isinstance(result, SynthesisResult)
        assert result.audio_data == b""  # 直接出力なのでデータは空
        assert result.format == "direct_output"
        assert result.duration > 0
        assert result.sample_rate == 22050
        assert result.metadata["mode"] == "speaker"

        # イテレーティブ方式のメソッドが正しく呼ばれたか確認
        mock_engine.say.assert_called_with("こんにちは")
        mock_engine.startLoop.assert_called_with(False)
        mock_engine.iterate.assert_called()
        mock_engine.endLoop.assert_called()
        mock_engine.stop.assert_called()

    @pytest.mark.asyncio
    async def test_synthesize_direct_mode_fallback(self, mock_pyttsx3_no_iterative, default_config):
        """直接出力モード（フォールバック：runAndWait）のテスト"""
        engine = Pyttsx3Engine(config=default_config)
        await engine.initialize()

        # テキストを合成
        result = await engine.synthesize("こんにちは")

        assert isinstance(result, SynthesisResult)
        assert result.audio_data == b""
        assert result.format == "direct_output"

        # フォールバックでrunAndWaitが呼ばれる
        mock_engine = mock_pyttsx3_no_iterative.init.return_value
        mock_engine.runAndWait.assert_called()
        mock_engine.stop.assert_called()

    @pytest.mark.asyncio
    async def test_synthesize_empty_text(self, mock_pyttsx3, default_config):
        """空文字列の処理テスト（修正版）"""
        engine = Pyttsx3Engine(config=default_config)
        await engine.initialize()

        # 空文字列
        result = await engine.synthesize("")
        assert result.duration == 0.0
        assert result.format == "direct_output"  # "empty"ではなく"direct_output"
        assert result.metadata["reason"] == "empty_text"
        assert len(result.audio_data) >= 44  # WAVヘッダー分

        # 空白のみ
        result = await engine.synthesize("   ")
        assert result.duration == 0.0
        assert result.format == "direct_output"
        assert len(result.audio_data) >= 44

    @pytest.mark.asyncio
    async def test_synthesize_with_voice_id(self, mock_pyttsx3, default_config):
        """音声ID指定でのsynthesizeテスト"""
        engine = Pyttsx3Engine(config=default_config)
        await engine.initialize()

        # モックをリセット
        mock_engine = mock_pyttsx3.init.return_value
        mock_engine.isBusy.reset_mock()
        mock_engine.isBusy.side_effect = [True, False]

        # 英語音声を指定
        result = await engine.synthesize("Hello", voice_id="en")
        assert result.metadata["voice_id"] == "en"

        # モックをリセット
        mock_engine.isBusy.reset_mock()
        mock_engine.isBusy.side_effect = [True, False]

        # 日本語音声を指定
        result = await engine.synthesize("こんにちは", voice_id="ja")
        assert result.metadata["voice_id"] == "ja"

    @pytest.mark.asyncio
    async def test_synthesize_invalid_voice_id(self, mock_pyttsx3, default_config):
        """無効な音声IDでのエラーテスト"""
        engine = Pyttsx3Engine(config=default_config)
        await engine.initialize()

        with pytest.raises(InvalidVoiceError) as exc_info:
            await engine.synthesize("テスト", voice_id="invalid")

        assert exc_info.value.error_code == "E3001"
        assert "invalid" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_synthesize_direct_error_handling(self, mock_pyttsx3, default_config):
        """直接出力モードでのエラーハンドリング"""
        engine = Pyttsx3Engine(config=default_config)
        await engine.initialize()

        # 新しいモックエンジンを作成してエラーを発生させる
        mock_engine = MagicMock()
        mock_engine.say = Mock()
        mock_engine.startLoop = Mock()
        mock_engine.iterate.side_effect = Exception("Audio output failed")
        mock_engine.stop = Mock()
        mock_engine.setProperty = Mock()
        mock_engine.getProperty.return_value = []

        # synthesizeが呼ばれたときに新しいモックを返す
        mock_pyttsx3.init.return_value = mock_engine

        with pytest.raises(TTSError) as exc_info:
            await engine.synthesize("テスト")

        assert exc_info.value.error_code == "E3000"
        assert "Failed to synthesize speech" in str(exc_info.value)


# ============================================================================
# synthesizeメソッドのテスト（WAVデータ取得モード）
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestSynthesizeWithData:
    """WAVデータ取得モードのsynthesizeテスト"""

    @pytest.mark.asyncio
    async def test_synthesize_save_audio_mode(self, mock_pyttsx3, save_audio_config):
        """WAVデータ取得モードの基本動作テスト"""
        engine = Pyttsx3Engine(config=save_audio_config)
        await engine.initialize()

        # モックをリセット
        mock_engine = mock_pyttsx3.init.return_value
        mock_engine.isBusy.reset_mock()
        mock_engine.isBusy.side_effect = [True, True, False]

        # ファイル読み込みをモック
        test_wav_data = b"RIFF....WAVEfmt ...."  # ダミーWAVデータ

        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = test_wav_data

            result = await engine.synthesize("テスト音声")

            assert isinstance(result, SynthesisResult)
            assert result.audio_data == test_wav_data
            assert result.format == "wav"
            assert result.metadata["mode"] == "file"
            assert result.metadata["file_size"] == len(test_wav_data)

        # save_to_fileが呼ばれたか確認
        mock_engine.save_to_file.assert_called()

    @pytest.mark.asyncio
    async def test_synthesize_runtime_override(self, mock_pyttsx3, default_config):
        """実行時のsave_audioパラメータによるオーバーライドテスト"""
        engine = Pyttsx3Engine(config=default_config)  # デフォルトは直接出力
        await engine.initialize()

        # モックをリセット
        mock_engine = mock_pyttsx3.init.return_value
        mock_engine.isBusy.reset_mock()
        mock_engine.isBusy.side_effect = [True, False]

        # 実行時にWAVデータ取得を指定
        test_wav_data = b"RIFF....WAVEfmt ...."

        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = test_wav_data

            result = await engine.synthesize("テスト", save_audio=True)
            assert result.audio_data == test_wav_data
            assert result.format == "wav"

    @pytest.mark.asyncio
    async def test_synthesize_temp_file_cleanup(self, mock_pyttsx3, save_audio_config):
        """一時ファイルのクリーンアップテスト"""
        engine = Pyttsx3Engine(config=save_audio_config)
        await engine.initialize()

        # モックをリセット
        mock_engine = mock_pyttsx3.init.return_value
        mock_engine.isBusy.reset_mock()
        mock_engine.isBusy.side_effect = [True, False]

        temp_file_path = None

        # save_to_fileの呼び出しを監視
        def capture_temp_file(text, path):
            nonlocal temp_file_path
            temp_file_path = path
            # 実際にファイルを作成
            with open(path, "wb") as f:
                f.write(b"RIFF....WAVEfmt ....")

        mock_engine.save_to_file.side_effect = capture_temp_file

        await engine.synthesize("テスト")

        # 一時ファイルが削除されているか確認
        assert temp_file_path is not None
        assert not os.path.exists(temp_file_path)

    @pytest.mark.asyncio
    async def test_synthesize_file_error_handling(self, mock_pyttsx3, save_audio_config):
        """ファイル操作エラーのハンドリングテスト"""
        engine = Pyttsx3Engine(config=save_audio_config)
        await engine.initialize()

        # save_to_fileでエラーを発生させる
        mock_engine = MagicMock()
        mock_engine.save_to_file.side_effect = Exception("File write failed")
        mock_engine.stop = Mock()
        mock_engine.setProperty = Mock()
        mock_engine.getProperty.return_value = []
        mock_engine.startLoop = Mock()
        mock_engine.iterate = Mock()
        mock_engine.endLoop = Mock()
        mock_engine.isBusy = Mock(return_value=False)

        mock_pyttsx3.init.return_value = mock_engine

        with pytest.raises(TTSError) as exc_info:
            await engine.synthesize("テスト")

        assert exc_info.value.error_code == "E3000"
        assert "Failed to synthesize speech to file" in str(exc_info.value)


# ============================================================================
# Part 29調査結果の検証テスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestEnginReuseProblem:
    """Part 29で発見されたエンジン再利用不可問題の検証"""

    @pytest.mark.asyncio
    async def test_multiple_synthesis(self, mock_pyttsx3, default_config):
        """複数回の音声合成が正常に動作することを確認"""
        engine = Pyttsx3Engine(config=default_config)
        await engine.initialize()

        # 複数回synthesizeを呼ぶ
        texts = ["こんにちは", "今日は良い天気です", "さようなら"]

        for i, text in enumerate(texts):
            # 各呼び出しのためにモックをリセット
            mock_engine = mock_pyttsx3.init.return_value
            mock_engine.isBusy.reset_mock()
            mock_engine.isBusy.side_effect = [True, False]

            result = await engine.synthesize(text)
            assert result is not None
            assert result.duration > 0

        # 毎回新しいエンジンが作成されていることを確認
        # 初期化時の呼び出し（音声検出用）+ 各synthesizeでの呼び出し
        assert mock_pyttsx3.init.call_count >= len(texts)

    @pytest.mark.asyncio
    async def test_engine_cleanup_always_happens(self, mock_pyttsx3, default_config):
        """エラー時でも必ずエンジンがクリーンアップされることを確認（修正版）"""
        engine = Pyttsx3Engine(config=default_config)
        await engine.initialize()

        # synthesize内でエラーが発生する設定
        mock_engine = MagicMock()
        mock_engine.say.side_effect = Exception("Say failed")
        mock_engine.stop = Mock()
        mock_engine.runAndWait = Mock()
        mock_engine.setProperty = Mock()
        mock_engine.getProperty.return_value = []
        mock_engine.startLoop = Mock()
        mock_engine.iterate = Mock()
        mock_engine.endLoop = Mock()
        mock_engine.isBusy = Mock(return_value=False)

        # 次のinit呼び出しで新しいモックエンジンを返す
        mock_pyttsx3.init.return_value = mock_engine

        try:
            await engine.synthesize("テスト")
        except TTSError:
            pass  # エラーは予期される

        # finallyブロックでstop()が呼ばれている
        mock_engine.stop.assert_called()

    @pytest.mark.asyncio
    async def test_no_engine_reuse(self, mock_pyttsx3, default_config):
        """エンジンが再利用されないことを確認"""
        engine = Pyttsx3Engine(config=default_config)
        await engine.initialize()

        # 各合成で新しいエンジンが作成されることを確認
        call_count_before = mock_pyttsx3.init.call_count

        # モックをリセット
        mock_engine = mock_pyttsx3.init.return_value
        mock_engine.isBusy.reset_mock()
        mock_engine.isBusy.side_effect = [True, False]

        await engine.synthesize("テスト1")
        call_count_after_1 = mock_pyttsx3.init.call_count

        # モックをリセット
        mock_engine.isBusy.reset_mock()
        mock_engine.isBusy.side_effect = [True, False]

        await engine.synthesize("テスト2")
        call_count_after_2 = mock_pyttsx3.init.call_count

        # 各synthesizeで新しいエンジンが作成されている
        assert call_count_after_1 > call_count_before
        assert call_count_after_2 > call_count_after_1


# ============================================================================
# その他のメソッドのテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestOtherMethods:
    """その他のメソッドのテスト"""

    @pytest.mark.asyncio
    async def test_get_available_voices(self, mock_pyttsx3):
        """get_available_voicesメソッドのテスト"""
        engine = Pyttsx3Engine()
        await engine.initialize()

        voices = engine.get_available_voices()

        assert isinstance(voices, list)
        assert len(voices) == 2
        assert all(isinstance(v, VoiceInfo) for v in voices)

    @pytest.mark.asyncio
    async def test_set_voice_valid(self, mock_pyttsx3):
        """set_voiceメソッドの正常系テスト"""
        engine = Pyttsx3Engine()
        await engine.initialize()

        # 英語に切り替え
        engine.set_voice("en")
        assert engine.current_voice_id == "en"

        # 日本語に戻す
        engine.set_voice("ja")
        assert engine.current_voice_id == "ja"

    @pytest.mark.asyncio
    async def test_set_voice_invalid(self, mock_pyttsx3):
        """set_voiceメソッドの異常系テスト"""
        engine = Pyttsx3Engine()
        await engine.initialize()

        with pytest.raises(InvalidVoiceError) as exc_info:
            engine.set_voice("invalid_voice")

        assert exc_info.value.error_code == "E3001"

    @pytest.mark.asyncio
    async def test_test_availability_success(self, mock_pyttsx3):
        """test_availabilityメソッドの正常系テスト"""
        engine = Pyttsx3Engine()
        await engine.initialize()

        # モックをリセット
        mock_engine = mock_pyttsx3.init.return_value
        mock_engine.isBusy.reset_mock()
        mock_engine.isBusy.side_effect = [True, False]

        result = await engine.test_availability()
        assert result is True

    @pytest.mark.asyncio
    async def test_test_availability_failure(self, mock_pyttsx3):
        """test_availabilityメソッドの異常系テスト"""
        engine = Pyttsx3Engine()
        await engine.initialize()

        # synthesizeでエラーを発生させる
        mock_engine = MagicMock()
        mock_engine.startLoop.side_effect = Exception("Not available")
        mock_engine.stop = Mock()
        mock_engine.say = Mock()
        mock_engine.setProperty = Mock()
        mock_engine.getProperty.return_value = []
        mock_pyttsx3.init.return_value = mock_engine

        result = await engine.test_availability()
        assert result is False


# ============================================================================
# 設定関連のテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestConfiguration:
    """設定関連のテスト"""

    @pytest.mark.asyncio
    async def test_config_speed_adjustment(self, mock_pyttsx3):
        """話速設定のテスト"""
        config = TTSConfig(speed=1.5)  # 1.5倍速
        engine = Pyttsx3Engine(config=config)
        await engine.initialize()

        # モックをリセット
        mock_engine = mock_pyttsx3.init.return_value
        mock_engine.isBusy.reset_mock()
        mock_engine.isBusy.side_effect = [True, False]

        # synthesizeを実行して設定が適用されることを確認
        await engine.synthesize("テスト")

        # setPropertyが呼ばれたことを確認（最新のモックエンジンで）
        mock_engine.setProperty.assert_any_call("rate", 225)  # 150 * 1.5

    @pytest.mark.asyncio
    async def test_config_volume_adjustment(self, mock_pyttsx3):
        """音量設定のテスト"""
        config = TTSConfig(volume=0.5)  # 50%音量
        engine = Pyttsx3Engine(config=config)
        await engine.initialize()

        # モックをリセット
        mock_engine = mock_pyttsx3.init.return_value
        mock_engine.isBusy.reset_mock()
        mock_engine.isBusy.side_effect = [True, False]

        # synthesizeを実行して設定が適用されることを確認
        await engine.synthesize("テスト")

        # setPropertyが呼ばれたことを確認
        mock_engine.setProperty.assert_any_call("volume", 0.5)

    @pytest.mark.asyncio
    async def test_duration_estimation(self, mock_pyttsx3):
        """音声時間推定のテスト"""
        config = TTSConfig(speed=2.0)  # 2倍速
        engine = Pyttsx3Engine(config=config)
        await engine.initialize()

        # モックをリセット
        mock_engine = mock_pyttsx3.init.return_value
        mock_engine.isBusy.reset_mock()
        mock_engine.isBusy.side_effect = [True, False]

        # 10文字のテキスト（基本: 1文字0.1秒 = 1秒）
        result = await engine.synthesize("あいうえおかきくけこ")

        # 2倍速なので0.5秒になるはず
        assert result.duration == pytest.approx(0.5, rel=0.01)


# ============================================================================
# フォールバック機能のテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestFallbackFunctionality:
    """フォールバック機能のテスト（Part 29調査結果反映）"""

    @pytest.mark.asyncio
    async def test_voice_detection_failure_fallback(self):
        """音声検出失敗時のフォールバックテスト"""
        with patch("vioratalk.core.tts.pyttsx3_engine.pyttsx3") as mock:
            # getPropertyがNoneを返す
            mock_engine = MagicMock()
            mock_engine.getProperty.return_value = None
            mock_engine.stop = Mock()
            mock.init.return_value = mock_engine

            engine = Pyttsx3Engine()
            await engine.initialize()

            # フォールバック音声が設定される
            assert len(engine.available_voices) == 2
            assert "Fallback" in engine.available_voices[0].name
            assert engine._state == ComponentState.READY

    @pytest.mark.asyncio
    async def test_voice_with_no_name_attribute(self):
        """name属性がない音声オブジェクトの処理"""
        with patch("vioratalk.core.tts.pyttsx3_engine.pyttsx3") as mock:
            # name属性がないオブジェクト
            mock_voice = Mock(spec=[])  # 属性なし
            mock_engine = MagicMock()
            mock_engine.getProperty.return_value = [mock_voice]
            mock_engine.stop = Mock()
            mock.init.return_value = mock_engine

            engine = Pyttsx3Engine()
            await engine.initialize()

            # フォールバック音声が設定される
            assert len(engine.available_voices) == 2
            assert "Fallback" in engine.available_voices[0].name

    @pytest.mark.asyncio
    async def test_iterative_timeout_handling(self, mock_pyttsx3, default_config):
        """イテレーティブ方式のタイムアウト処理テスト"""
        engine = Pyttsx3Engine(config=default_config)
        await engine.initialize()

        # isBusyが常にTrueを返す（タイムアウトシミュレーション）
        mock_engine = mock_pyttsx3.init.return_value
        mock_engine.isBusy.reset_mock()
        mock_engine.isBusy.return_value = True  # 常にTrue

        # タイムアウトは発生するが、エラーにはならない
        # （ログに警告は出るが、処理は継続）
        result = await engine.synthesize("タイムアウトテスト")

        assert result is not None
        assert result.duration > 0
        # endLoopは呼ばれる（タイムアウト後もクリーンアップは実行）
        mock_engine.endLoop.assert_called()
        mock_engine.stop.assert_called()
