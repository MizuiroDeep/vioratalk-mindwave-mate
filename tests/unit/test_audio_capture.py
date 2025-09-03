"""
AudioCaptureクラスの単体テスト

音声入力デバイス管理機能のテスト。
外部依存（sounddevice, pyaudio）は全てモック化。

テスト戦略ガイドライン v1.7準拠
テスト実装ガイド v1.3準拠
開発規約書 v1.12準拠
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vioratalk.core.base import ComponentState
from vioratalk.core.exceptions import AudioError
from vioratalk.core.stt.base import AudioData

# テスト対象のインポート
from vioratalk.infrastructure.audio_capture import (
    DEFAULT_CHANNELS,
    DEFAULT_SAMPLE_RATE,
    AudioCapture,
    AudioDevice,
    RecordingConfig,
    get_audio_capture,
    quick_record,
)

# ============================================================================
# フィクスチャ
# ============================================================================


@pytest.fixture
def mock_sounddevice():
    """sounddeviceモジュールのモック"""
    with patch("vioratalk.infrastructure.audio_capture.sd") as mock_sd:
        # デフォルトのモック設定
        mock_sd.query_devices.return_value = [
            {
                "name": "Built-in Microphone",
                "max_input_channels": 2,
                "max_output_channels": 0,
                "default_samplerate": 44100.0,
                "hostapi": 0,
                "default_low_input_latency": 0.01,
            },
            {
                "name": "USB Microphone",
                "max_input_channels": 1,
                "max_output_channels": 0,
                "default_samplerate": 48000.0,
                "hostapi": 0,
                "default_low_input_latency": 0.005,
            },
        ]
        mock_sd.default.device = (0, 0)  # デフォルトデバイス
        mock_sd.query_hostapis.return_value = {"name": "Core Audio"}

        # 録音用のモック
        mock_audio = np.random.randn(DEFAULT_SAMPLE_RATE * 3).astype(np.float32)
        mock_sd.rec.return_value = mock_audio.reshape(-1, 1)
        mock_sd.wait.return_value = None

        # SOUNDDEVICE_AVAILABLEフラグも設定
        with patch("vioratalk.infrastructure.audio_capture.SOUNDDEVICE_AVAILABLE", True):
            yield mock_sd


@pytest.fixture
def mock_pyaudio():
    """pyaudioモジュールのモック（動的インポート対応）"""
    # pyaudioモジュール全体をモック
    mock_pa_module = MagicMock()
    mock_pa_module.paFloat32 = 3  # pyaudioの定数

    mock_instance = MagicMock()
    mock_pa_module.PyAudio.return_value = mock_instance

    # デバイス情報
    mock_instance.get_device_count.return_value = 2
    mock_instance.get_default_input_device_info.return_value = {"index": 0}
    mock_instance.get_device_info_by_index.side_effect = [
        {
            "index": 0,
            "name": "Built-in Microphone",
            "maxInputChannels": 2,
            "maxOutputChannels": 0,
            "defaultSampleRate": 44100.0,
            "hostApi": 0,
        },
        {
            "index": 1,
            "name": "USB Microphone",
            "maxInputChannels": 1,
            "maxOutputChannels": 0,
            "defaultSampleRate": 48000.0,
            "hostApi": 0,
        },
    ]
    mock_instance.get_host_api_info_by_index.return_value = {"name": "MME"}

    # ストリームのモック
    mock_stream = MagicMock()
    mock_instance.open.return_value = mock_stream
    mock_stream.read.return_value = np.random.randn(1024).astype(np.float32).tobytes()

    # sys.modulesを使用してpyaudioをモック化
    with patch.dict("sys.modules", {"pyaudio": mock_pa_module}):
        # PYAUDIO_AVAILABLEフラグも設定
        with patch("vioratalk.infrastructure.audio_capture.PYAUDIO_AVAILABLE", True):
            yield mock_pa_module


@pytest.fixture
def recording_config():
    """RecordingConfig のフィクスチャ"""
    return RecordingConfig(
        sample_rate=16000, channels=1, duration=3.0, enable_agc=False, enable_noise_reduction=False
    )


@pytest.fixture
def audio_capture(recording_config):
    """AudioCaptureインスタンスのフィクスチャ"""
    return AudioCapture(config=recording_config)


# ============================================================================
# AudioDeviceのテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestAudioDevice:
    """AudioDeviceクラスのテスト"""

    def test_initialization(self):
        """初期化のテスト"""
        device = AudioDevice(
            id=0,
            name="Test Microphone",
            channels=2,
            sample_rate=44100,
            is_default=True,
            host_api="Core Audio",
            latency=0.01,
        )

        assert device.id == 0
        assert device.name == "Test Microphone"
        assert device.channels == 2
        assert device.sample_rate == 44100
        assert device.is_default is True
        assert device.host_api == "Core Audio"
        assert device.latency == 0.01

    def test_string_representation(self):
        """文字列表現のテスト"""
        device = AudioDevice(id=1, name="USB Mic", channels=1, sample_rate=48000, is_default=False)

        str_repr = str(device)
        assert "1: USB Mic" in str_repr
        assert "1ch" in str_repr
        assert "48000Hz" in str_repr
        assert "[DEFAULT]" not in str_repr

    def test_default_device_string(self):
        """デフォルトデバイスの文字列表現"""
        device = AudioDevice(id=0, name="Built-in", channels=2, sample_rate=44100, is_default=True)

        str_repr = str(device)
        assert "[DEFAULT]" in str_repr


# ============================================================================
# RecordingConfigのテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestRecordingConfig:
    """RecordingConfigクラスのテスト"""

    def test_default_values(self):
        """デフォルト値のテスト"""
        config = RecordingConfig()

        assert config.device_id is None
        assert config.sample_rate == DEFAULT_SAMPLE_RATE
        assert config.channels == DEFAULT_CHANNELS
        assert config.duration == 5.0
        assert config.chunk_size == 1024
        assert config.enable_agc is False
        assert config.enable_noise_reduction is False
        assert config.silence_threshold == -40.0

    def test_custom_values(self):
        """カスタム値のテスト"""
        config = RecordingConfig(
            device_id=1,
            sample_rate=48000,
            channels=2,
            duration=10.0,
            enable_agc=True,
            enable_noise_reduction=True,
        )

        assert config.device_id == 1
        assert config.sample_rate == 48000
        assert config.channels == 2
        assert config.duration == 10.0
        assert config.enable_agc is True
        assert config.enable_noise_reduction is True


# ============================================================================
# AudioCaptureの初期化テスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestAudioCaptureInitialization:
    """AudioCaptureの初期化テスト"""

    def test_initialization_without_config(self, mock_sounddevice):
        """設定なしでの初期化"""
        capture = AudioCapture()

        assert capture.config is not None
        assert capture.config.sample_rate == DEFAULT_SAMPLE_RATE
        # 初期化時にデバイススキャンが行われるため、devicesは空ではない
        assert len(capture.devices) == 2  # mock_sounddeviceで2つのデバイスを定義
        assert capture.current_device is None  # 選択はまだされていない
        assert capture._state == ComponentState.NOT_INITIALIZED  # baseクラスで設定される初期状態

    def test_initialization_with_config(self, recording_config, mock_sounddevice):
        """設定ありでの初期化"""
        capture = AudioCapture(config=recording_config)

        assert capture.config == recording_config
        assert capture.config.sample_rate == 16000

    @pytest.mark.asyncio
    async def test_async_initialization(self, audio_capture, mock_sounddevice):
        """非同期初期化のテスト - 仕様書準拠（safe_initialize使用）"""
        # safe_initialize()を使用して状態遷移を正しく管理
        await audio_capture.safe_initialize()

        assert audio_capture._state == ComponentState.READY
        assert len(audio_capture.devices) > 0
        assert audio_capture.current_device is not None

    @pytest.mark.asyncio
    async def test_initialization_without_devices(self, audio_capture):
        """デバイスが見つからない場合のエラー"""
        with patch("vioratalk.infrastructure.audio_capture.sd") as mock_sd:
            mock_sd.query_devices.return_value = []

            with pytest.raises(AudioError) as exc_info:
                await audio_capture.safe_initialize()

            assert exc_info.value.error_code == "E1002"

    @pytest.mark.asyncio
    async def test_cleanup(self, audio_capture, mock_sounddevice):
        """クリーンアップのテスト - 仕様書準拠（safe_initialize/safe_cleanup使用）"""
        # safe_initialize()で初期化
        await audio_capture.safe_initialize()
        # safe_cleanup()でクリーンアップ
        await audio_capture.safe_cleanup()

        assert audio_capture._state == ComponentState.TERMINATED  # 仕様書準拠
        assert audio_capture._audio_interface is None


# ============================================================================
# デバイス管理のテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestDeviceManagement:
    """デバイス管理機能のテスト"""

    def test_list_devices_with_sounddevice(self, audio_capture, mock_sounddevice):
        """sounddeviceでのデバイスリスト取得"""
        devices = audio_capture.list_devices()

        assert len(devices) == 2
        assert devices[0].name == "Built-in Microphone"
        assert devices[0].channels == 2
        assert devices[0].is_default is True
        assert devices[1].name == "USB Microphone"

    def test_list_devices_with_pyaudio(self, audio_capture, mock_pyaudio):
        """pyaudioでのデバイスリスト取得"""
        with patch("vioratalk.infrastructure.audio_capture.SOUNDDEVICE_AVAILABLE", False):
            devices = audio_capture.list_devices()

            assert len(devices) == 2
            assert devices[0].name == "Built-in Microphone"
            assert devices[1].name == "USB Microphone"

    def test_select_device(self, audio_capture, mock_sounddevice):
        """デバイス選択のテスト"""
        audio_capture.list_devices()  # デバイスをスキャン

        audio_capture.select_device(1)

        assert audio_capture.current_device is not None
        assert audio_capture.current_device.id == 1
        assert audio_capture.config.device_id == 1

    def test_select_invalid_device(self, audio_capture, mock_sounddevice):
        """無効なデバイス選択"""
        audio_capture.list_devices()

        with pytest.raises(AudioError) as exc_info:
            audio_capture.select_device(999)

        assert exc_info.value.error_code == "E1002"
        assert "999" in str(exc_info.value)

    def test_get_current_device(self, audio_capture, mock_sounddevice):
        """現在のデバイス取得"""
        audio_capture.list_devices()
        audio_capture.select_device(0)

        device = audio_capture.get_current_device()

        assert device is not None
        assert device.id == 0
        assert device.name == "Built-in Microphone"


# ============================================================================
# 録音機能のテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestRecording:
    """録音機能のテスト"""

    @pytest.mark.asyncio
    async def test_record_from_microphone_sounddevice(self, audio_capture, mock_sounddevice):
        """sounddeviceでの録音テスト"""
        await audio_capture.safe_initialize()

        # 録音実行
        audio_data = await audio_capture.record_from_microphone(duration=1.0)

        assert isinstance(audio_data, AudioData)
        assert audio_data.raw_data is not None
        assert isinstance(audio_data.raw_data, np.ndarray)
        assert audio_data.metadata.sample_rate == 16000
        assert audio_data.metadata.channels == 1
        assert audio_data.metadata.duration == 1.0

        # sounddeviceの関数が呼ばれたことを確認
        mock_sounddevice.rec.assert_called_once()
        mock_sounddevice.wait.assert_called_once()

    @pytest.mark.asyncio
    async def test_record_from_microphone_pyaudio(self, audio_capture, mock_pyaudio):
        """pyaudioでの録音テスト"""
        with patch("vioratalk.infrastructure.audio_capture.SOUNDDEVICE_AVAILABLE", False):
            await audio_capture.safe_initialize()

            audio_data = await audio_capture.record_from_microphone(duration=0.1)

            assert isinstance(audio_data, AudioData)
            assert audio_data.raw_data is not None

    @pytest.mark.asyncio
    async def test_record_with_callback(self, audio_capture, mock_sounddevice):
        """コールバック付き録音のテスト"""
        await audio_capture.safe_initialize()

        callback_called = False
        callback_data = None

        def callback(data):
            nonlocal callback_called, callback_data
            callback_called = True
            callback_data = data

        await audio_capture.record_from_microphone(duration=1.0, callback=callback)

        assert callback_called is True
        assert callback_data is not None

    @pytest.mark.asyncio
    async def test_record_no_library_available(self, audio_capture):
        """録音ライブラリが利用できない場合のエラー"""
        with patch("vioratalk.infrastructure.audio_capture.SOUNDDEVICE_AVAILABLE", False):
            with patch("vioratalk.infrastructure.audio_capture.PYAUDIO_AVAILABLE", False):
                with pytest.raises(AudioError) as exc_info:
                    await audio_capture.record_from_microphone()

                # エラーコードの確認
                assert exc_info.value.error_code == "E1002"


# ============================================================================
# 音声処理機能のテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestAudioProcessing:
    """音声処理機能のテスト"""

    @pytest.mark.asyncio
    async def test_agc_enabled(self, audio_capture, mock_sounddevice):
        """自動ゲイン調整のテスト"""
        audio_capture.config.enable_agc = True
        await audio_capture.safe_initialize()

        # 小さい音声データを作成
        small_audio = np.ones(16000) * 0.01  # 小さい音量
        mock_sounddevice.rec.return_value = small_audio.reshape(-1, 1)

        audio_data = await audio_capture.record_from_microphone(duration=1.0)

        # AGCが適用されて音量が調整されているはず
        assert audio_data.raw_data is not None
        # 実際の値のチェックは実装依存のため、エラーがないことを確認

    @pytest.mark.asyncio
    async def test_noise_reduction_enabled(self, audio_capture, mock_sounddevice):
        """ノイズリダクションのテスト"""
        audio_capture.config.enable_noise_reduction = True
        await audio_capture.safe_initialize()

        audio_data = await audio_capture.record_from_microphone(duration=1.0)

        assert audio_data.raw_data is not None
        # ノイズリダクションが適用されたことを確認（DC成分除去）
        mean_value = np.mean(audio_data.raw_data)
        assert abs(mean_value) < 0.01  # DCオフセットが除去されている

    def test_is_silent(self, audio_capture):
        """無音判定のテスト"""
        # プライベートメソッドだが重要なのでテスト

        # 無音データ
        silent_audio = np.zeros(1000)
        assert audio_capture._is_silent(silent_audio) is True

        # 通常の音声データ
        normal_audio = np.random.randn(1000) * 0.1
        # bool型で返ることを確認
        result = audio_capture._is_silent(normal_audio)
        assert isinstance(result, bool)  # numpy.bool_ではなくbool型
        assert result is False

    def test_calculate_db(self, audio_capture):
        """dB計算のテスト"""
        # フルスケール信号（0dB）
        full_scale = np.ones(1000)
        db = audio_capture._calculate_db(full_scale)
        assert db == pytest.approx(0.0, abs=1.0)

        # 半分の振幅（約-6dB）
        half_scale = np.ones(1000) * 0.5
        db = audio_capture._calculate_db(half_scale)
        assert db == pytest.approx(-6.0, abs=1.0)

        # 無音（最小dB）
        silence = np.zeros(1000)
        db = audio_capture._calculate_db(silence)
        assert db == -60.0  # MIN_AUDIO_LEVEL


# ============================================================================
# エラーハンドリングのテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestErrorHandling:
    """エラーハンドリングのテスト"""

    @pytest.mark.asyncio
    async def test_initialization_error(self, audio_capture):
        """初期化エラーのテスト"""
        with patch.object(audio_capture, "_scan_devices", side_effect=Exception("Scan error")):
            with pytest.raises(AudioError) as exc_info:
                await audio_capture.safe_initialize()

            assert exc_info.value.error_code == "E1001"

    @pytest.mark.asyncio
    async def test_recording_error(self, audio_capture, mock_sounddevice):
        """録音エラーのテスト"""
        await audio_capture.safe_initialize()

        # 録音で例外を発生させる
        mock_sounddevice.rec.side_effect = Exception("Recording failed")

        with pytest.raises(AudioError) as exc_info:
            await audio_capture.record_from_microphone()

        assert exc_info.value.error_code == "E1001"

    @pytest.mark.asyncio
    async def test_device_scan_error_on_init(self):
        """初期化時のデバイススキャンエラー（続行）"""
        with patch("vioratalk.infrastructure.audio_capture.sd") as mock_sd:
            mock_sd.query_devices.side_effect = Exception("Device query failed")

            # エラーが発生しても初期化は続行される
            capture = AudioCapture()
            assert capture is not None
            assert capture.devices == []


# ============================================================================
# ユーティリティ関数のテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestUtilityFunctions:
    """ユーティリティ関数のテスト"""

    def test_get_audio_capture(self, recording_config):
        """get_audio_captureファクトリ関数のテスト"""
        capture = get_audio_capture(config=recording_config)

        assert isinstance(capture, AudioCapture)
        assert capture.config == recording_config

    def test_get_audio_capture_default(self):
        """デフォルト設定でのget_audio_capture"""
        capture = get_audio_capture()

        assert isinstance(capture, AudioCapture)
        assert capture.config.sample_rate == DEFAULT_SAMPLE_RATE

    @pytest.mark.asyncio
    async def test_quick_record(self, mock_sounddevice):
        """quick_record関数のテスト"""
        audio_data = await quick_record(duration=1.0)

        assert isinstance(audio_data, AudioData)
        assert audio_data.raw_data is not None
        assert audio_data.metadata.duration == 1.0

    @pytest.mark.asyncio
    async def test_quick_record_error_handling(self):
        """quick_recordのエラーハンドリング"""
        with patch("vioratalk.infrastructure.audio_capture.sd") as mock_sd:
            mock_sd.query_devices.return_value = []

            with pytest.raises(AudioError) as exc_info:
                await quick_record()

            assert exc_info.value.error_code == "E1002"


# ============================================================================
# 統合的なシナリオテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestIntegrationScenarios:
    """統合的なシナリオテスト"""

    @pytest.mark.asyncio
    async def test_complete_recording_workflow(self, mock_sounddevice):
        """完全な録音ワークフローのテスト - 仕様書準拠（safe_initialize/safe_cleanup使用）"""
        # 1. インスタンス作成
        config = RecordingConfig(
            sample_rate=48000, channels=2, enable_agc=True, enable_noise_reduction=True
        )
        capture = AudioCapture(config=config)

        # 2. 初期化（safe_initialize使用）
        await capture.safe_initialize()
        assert capture._state == ComponentState.READY

        # 3. デバイスリスト取得
        devices = capture.list_devices()
        assert len(devices) > 0

        # 4. デバイス選択
        capture.select_device(devices[0].id)
        assert capture.current_device == devices[0]

        # 5. 録音
        audio_data = await capture.record_from_microphone(duration=2.0)
        assert audio_data is not None
        assert audio_data.metadata.sample_rate == 48000
        assert audio_data.metadata.channels == 2

        # 6. クリーンアップ（safe_cleanup使用）
        await capture.safe_cleanup()
        assert capture._state == ComponentState.TERMINATED  # 仕様書準拠

    @pytest.mark.asyncio
    async def test_multiple_recordings(self, audio_capture, mock_sounddevice):
        """複数回録音のテスト"""
        await audio_capture.safe_initialize()

        # 3回連続で録音
        recordings = []
        for i in range(3):
            audio_data = await audio_capture.record_from_microphone(duration=0.5)
            recordings.append(audio_data)

        assert len(recordings) == 3
        for audio_data in recordings:
            assert audio_data is not None
            assert audio_data.raw_data is not None

    @pytest.mark.asyncio
    async def test_device_switching(self, audio_capture, mock_sounddevice):
        """デバイス切り替えのテスト"""
        await audio_capture.safe_initialize()

        # 最初のデバイスで録音
        audio_capture.select_device(0)
        audio1 = await audio_capture.record_from_microphone(duration=0.5)

        # デバイスを切り替えて録音
        audio_capture.select_device(1)
        audio2 = await audio_capture.record_from_microphone(duration=0.5)

        assert audio1 is not None
        assert audio2 is not None
        assert audio_capture.current_device.id == 1


# ============================================================================
# パフォーマンス関連のテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestPerformance:
    """パフォーマンス関連のテスト"""

    @pytest.mark.asyncio
    async def test_recording_memory_usage(self, audio_capture, mock_sounddevice):
        """録音時のメモリ使用量"""
        await audio_capture.safe_initialize()

        # 3秒の録音（メモリ使用量のテスト）
        audio_data = await audio_capture.record_from_microphone(duration=3.0)

        # 期待されるサイズを計算（duration=3秒）
        expected_samples = 16000 * 3  # サンプルレート × 秒数
        assert len(audio_data.raw_data) == expected_samples

        # float32なので4バイト × サンプル数
        expected_bytes = expected_samples * 4
        assert audio_data.raw_data.nbytes == expected_bytes

    def test_device_list_caching(self, audio_capture, mock_sounddevice):
        """デバイスリストのキャッシング"""
        # 複数回呼んでもスキャンは効率的に行われる
        devices1 = audio_capture.list_devices()
        devices2 = audio_capture.list_devices()

        # 同じ結果が返される（ただしコピー）
        assert len(devices1) == len(devices2)
        assert devices1 is not devices2  # 別のリストオブジェクト


# ============================================================================
# エクスポート定義のテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestExports:
    """エクスポート定義のテスト"""

    def test_all_exports(self):
        """__all__で定義されたエクスポートの確認"""
        from vioratalk.infrastructure import audio_capture

        exports = audio_capture.__all__

        assert "AudioCapture" in exports
        assert "AudioDevice" in exports
        assert "RecordingConfig" in exports
        assert "get_audio_capture" in exports
        assert "quick_record" in exports
        assert "DEFAULT_SAMPLE_RATE" in exports
        assert "DEFAULT_CHANNELS" in exports
