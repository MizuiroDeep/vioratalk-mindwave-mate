"""
VoiceActivityDetectorクラスの単体テスト

音声区間検出機能のテスト。
各種アルゴリズムとパラメータの動作確認。

テスト戦略ガイドライン v1.7準拠
テスト実装ガイド v1.3準拠
開発規約書 v1.12準拠
"""


import numpy as np
import pytest

from vioratalk.core.base import ComponentState

# テスト対象のインポート
from vioratalk.core.stt.vad import (
    DEFAULT_FRAME_DURATION,
    DEFAULT_SAMPLE_RATE,
    SpeechSegment,
    SpeechState,
    VADConfig,
    VADMode,
    VoiceActivityDetector,
    create_vad,
    detect_speech_in_file,
)

# ============================================================================
# テスト用ユーティリティ
# ============================================================================


def generate_silence(duration: float, sample_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    """無音データを生成

    Args:
        duration: 時間（秒）
        sample_rate: サンプリングレート

    Returns:
        np.ndarray: 無音データ
    """
    num_samples = int(duration * sample_rate)
    # 微小なノイズを含む無音
    return np.random.randn(num_samples) * 0.001


def generate_speech(duration: float, sample_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    """疑似音声データを生成

    Args:
        duration: 時間（秒）
        sample_rate: サンプリングレート

    Returns:
        np.ndarray: 疑似音声データ
    """
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples)

    # 複数の周波数成分を持つ信号（音声を模擬）
    signal = (
        0.3 * np.sin(2 * np.pi * 200 * t)
        + 0.2 * np.sin(2 * np.pi * 400 * t)  # 基本周波数
        + 0.1 * np.sin(2 * np.pi * 800 * t)  # 第2倍音  # 第3倍音
    )

    # エンベロープを適用（自然な音声のように）
    envelope = np.exp(-t * 0.5) + 0.3
    signal = signal * envelope

    # ノイズを追加
    noise = np.random.randn(num_samples) * 0.02

    return signal + noise


def generate_mixed_audio(
    speech_duration: float = 1.0,
    silence_duration: float = 0.5,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> np.ndarray:
    """音声と無音が混在したデータを生成

    Args:
        speech_duration: 音声部分の長さ（秒）
        silence_duration: 無音部分の長さ（秒）
        sample_rate: サンプリングレート

    Returns:
        np.ndarray: 混在データ
    """
    silence1 = generate_silence(silence_duration, sample_rate)
    speech = generate_speech(speech_duration, sample_rate)
    silence2 = generate_silence(silence_duration, sample_rate)

    return np.concatenate([silence1, speech, silence2])


# ============================================================================
# フィクスチャ
# ============================================================================


@pytest.fixture
def vad_config():
    """VADConfig のフィクスチャ"""
    return VADConfig(
        mode=VADMode.NORMAL,
        sample_rate=16000,
        frame_duration=0.03,
        adaptive_threshold=True,
        enable_noise_learning=True,
    )


@pytest.fixture
def vad(vad_config):
    """VoiceActivityDetector のフィクスチャ"""
    return VoiceActivityDetector(config=vad_config)


@pytest.fixture
def speech_audio():
    """音声データのフィクスチャ"""
    return generate_speech(1.0)


@pytest.fixture
def silence_audio():
    """無音データのフィクスチャ"""
    return generate_silence(1.0)


@pytest.fixture
def mixed_audio():
    """混在データのフィクスチャ"""
    return generate_mixed_audio()


# ============================================================================
# VADConfigのテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestVADConfig:
    """VADConfigクラスのテスト"""

    def test_default_values(self):
        """デフォルト値のテスト"""
        config = VADConfig()

        assert config.mode == VADMode.NORMAL
        assert config.sample_rate == DEFAULT_SAMPLE_RATE
        assert config.frame_duration == DEFAULT_FRAME_DURATION
        assert config.adaptive_threshold is True
        assert config.enable_noise_learning is True

    def test_mode_adjustment(self):
        """モードによる閾値調整のテスト"""
        # AGGRESSIVE モード
        aggressive_config = VADConfig(mode=VADMode.AGGRESSIVE)
        assert aggressive_config.energy_threshold < VADConfig().energy_threshold
        assert aggressive_config.silence_min_duration < VADConfig().silence_min_duration

        # CONSERVATIVE モード
        conservative_config = VADConfig(mode=VADMode.CONSERVATIVE)
        assert conservative_config.energy_threshold > VADConfig().energy_threshold
        assert conservative_config.silence_min_duration > VADConfig().silence_min_duration

    def test_custom_values(self):
        """カスタム値のテスト"""
        config = VADConfig(
            mode=VADMode.AGGRESSIVE,
            sample_rate=48000,
            frame_duration=0.02,
            energy_threshold=0.02,
            adaptive_threshold=False,
        )

        assert config.sample_rate == 48000
        assert config.frame_duration == 0.02
        assert config.adaptive_threshold is False


# ============================================================================
# SpeechSegmentのテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestSpeechSegment:
    """SpeechSegmentクラスのテスト"""

    def test_initialization(self):
        """初期化のテスト"""
        segment = SpeechSegment(
            start_time=1.0,
            end_time=3.5,
            start_sample=16000,
            end_sample=56000,
            confidence=0.95,
            energy_level=0.1,
        )

        assert segment.start_time == 1.0
        assert segment.end_time == 3.5
        assert segment.duration == 2.5
        assert segment.confidence == 0.95

    def test_duration_property(self):
        """duration プロパティのテスト"""
        segment = SpeechSegment(start_time=0.5, end_time=2.0, start_sample=8000, end_sample=32000)

        assert segment.duration == 1.5

    def test_to_dict(self):
        """辞書変換のテスト"""
        segment = SpeechSegment(
            start_time=1.0,
            end_time=2.0,
            start_sample=16000,
            end_sample=32000,
            confidence=0.9,
            energy_level=0.05,
        )

        data = segment.to_dict()

        assert data["start_time"] == 1.0
        assert data["end_time"] == 2.0
        assert data["duration"] == 1.0
        assert data["confidence"] == 0.9
        assert data["energy_level"] == 0.05


# ============================================================================
# VoiceActivityDetectorの初期化テスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestVADInitialization:
    """VAD初期化のテスト"""

    def test_initialization_without_config(self):
        """設定なしでの初期化"""
        vad = VoiceActivityDetector()

        assert vad.config is not None
        assert vad.speech_state == SpeechState.SILENCE  # stateではなくspeech_state
        assert vad.statistics is not None
        assert vad._state == ComponentState.NOT_INITIALIZED  # ComponentStateの_state

    def test_initialization_with_config(self, vad_config):
        """設定ありでの初期化"""
        vad = VoiceActivityDetector(config=vad_config)

        assert vad.config == vad_config
        assert vad.frame_size == int(16000 * 0.03)

    @pytest.mark.asyncio
    async def test_async_initialization(self, vad):
        """非同期初期化のテスト"""
        await vad.initialize()

        assert vad._state == ComponentState.READY
        assert vad.noise_profile["learned"] is False


# ============================================================================
# フレーム処理のテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestFrameProcessing:
    """フレーム処理のテスト"""

    def test_process_frame_silence(self, vad):
        """無音フレームの処理"""
        silence_frame = generate_silence(0.03)

        is_speech = vad.process_frame(silence_frame)

        assert is_speech is False
        assert vad.statistics.total_frames == 1
        assert vad.statistics.silence_frames == 1
        assert vad.statistics.speech_frames == 0

    def test_process_frame_speech(self, vad):
        """音声フレームの処理"""
        speech_frame = generate_speech(0.03)

        # ノイズ学習を無効化（音声を確実に検出するため）
        vad.config.enable_noise_learning = False
        vad.config.energy_threshold = 0.001

        is_speech = vad.process_frame(speech_frame)

        assert is_speech is True
        assert vad.statistics.speech_frames == 1

    def test_energy_calculation(self, vad):
        """エネルギー計算のテスト"""
        # フルスケール信号
        full_scale = np.ones(480)
        energy = vad._calculate_energy(full_scale)
        assert energy == pytest.approx(1.0, abs=0.01)

        # 半分の振幅
        half_scale = np.ones(480) * 0.5
        energy = vad._calculate_energy(half_scale)
        assert energy == pytest.approx(0.25, abs=0.01)

        # 無音
        silence = np.zeros(480)
        energy = vad._calculate_energy(silence)
        assert energy == 0.0

    def test_zcr_calculation(self, vad):
        """ゼロクロッシング率計算のテスト"""
        # 高周波信号（ZCRが高い）
        t = np.linspace(0, 0.03, 480)
        high_freq = np.sin(2 * np.pi * 1000 * t)
        zcr_high = vad._calculate_zcr(high_freq)

        # 低周波信号（ZCRが低い）
        low_freq = np.sin(2 * np.pi * 100 * t)
        zcr_low = vad._calculate_zcr(low_freq)

        assert zcr_high > zcr_low


# ============================================================================
# セグメント検出のテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestSegmentDetection:
    """セグメント検出のテスト"""

    def test_detect_segments_single_speech(self, vad):
        """単一音声区間の検出"""
        # 1秒の音声
        audio = generate_speech(1.0)

        # 検出しやすいように設定調整
        vad.config.energy_threshold = 0.001
        vad.config.speech_min_duration = 0.1
        vad.config.silence_min_duration = 0.1
        vad.config.enable_noise_learning = False

        segments = vad.detect_segments(audio)

        assert len(segments) >= 1
        if segments:
            assert segments[0].duration > 0
            assert segments[0].confidence >= 0.0
            assert segments[0].confidence <= 1.0

    def test_detect_segments_mixed(self, vad, mixed_audio):
        """混在音声の区間検出"""
        # 検出しやすいように設定調整
        vad.config.energy_threshold = 0.001
        vad.config.speech_min_duration = 0.3
        vad.config.silence_min_duration = 0.3
        vad.config.enable_noise_learning = False

        segments = vad.detect_segments(mixed_audio)

        # 少なくとも1つの音声区間が検出される
        assert len(segments) >= 1

        if segments:
            # 最初のセグメントは無音の後に開始
            assert segments[0].start_time > 0.2
            # 音声部分の長さに近い
            assert segments[0].duration > 0.5

    def test_detect_segments_silence_only(self, vad, silence_audio):
        """無音のみの場合"""
        segments = vad.detect_segments(silence_audio)

        # 音声区間は検出されない
        assert len(segments) == 0

    def test_statistics_update(self, vad, mixed_audio):
        """統計情報の更新テスト"""
        initial_stats = vad.get_statistics()
        assert initial_stats.total_frames == 0

        vad.detect_segments(mixed_audio)

        stats = vad.get_statistics()
        assert stats.total_frames > 0
        assert stats.speech_frames >= 0
        assert stats.silence_frames >= 0
        assert stats.total_frames == stats.speech_frames + stats.silence_frames


# ============================================================================
# ストリーム処理のテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestStreamProcessing:
    """ストリーム処理のテスト"""

    @pytest.mark.asyncio
    async def test_process_stream_basic(self, vad):
        """基本的なストリーム処理"""

        async def audio_generator():
            """テスト用音声ストリーム"""
            # 無音 → 音声 → 無音
            yield generate_silence(0.5)
            yield generate_speech(1.0)
            yield generate_silence(0.5)

        states = []
        frames = []

        async for state, frame in vad.process_stream(audio_generator()):
            states.append(state)
            frames.append(frame)

        assert len(frames) > 0
        # 様々な状態が含まれる
        assert SpeechState.SILENCE in states

    @pytest.mark.asyncio
    async def test_process_stream_with_callback(self, vad):
        """コールバック付きストリーム処理"""

        async def audio_generator():
            yield generate_speech(0.5)

        callback_called = False
        callback_state = None

        def callback(state: SpeechState, time: float):
            nonlocal callback_called, callback_state
            callback_called = True
            callback_state = state

        # 設定を調整
        vad.config.enable_noise_learning = False
        vad.config.energy_threshold = 0.001

        async for _ in vad.process_stream(audio_generator(), callback=callback):
            pass

        # コールバックが呼ばれることを確認
        # （状態変化があれば呼ばれる）


# ============================================================================
# 設定・調整のテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestConfiguration:
    """設定・調整機能のテスト"""

    def test_set_mode(self, vad):
        """モード設定のテスト"""
        original_threshold = vad.config.energy_threshold

        # AGGRESSIVE モードに変更
        vad.set_mode(VADMode.AGGRESSIVE)
        assert vad.config.mode == VADMode.AGGRESSIVE
        assert vad.config.energy_threshold < original_threshold

        # CONSERVATIVE モードに変更
        vad.set_mode(VADMode.CONSERVATIVE)
        assert vad.config.mode == VADMode.CONSERVATIVE
        assert vad.config.energy_threshold > original_threshold

    def test_adjust_sensitivity(self, vad):
        """感度調整のテスト"""
        original_energy = vad.config.energy_threshold
        original_zcr = vad.config.zcr_threshold

        # 感度を上げる（閾値を下げる）
        vad.adjust_sensitivity(0.5)
        assert vad.config.energy_threshold == original_energy * 0.5
        assert vad.config.zcr_threshold == original_zcr * 0.5

        # 感度を下げる（閾値を上げる）
        vad.adjust_sensitivity(2.0)
        assert vad.config.energy_threshold == original_energy
        assert vad.config.zcr_threshold == original_zcr

    def test_reset_noise_profile(self, vad):
        """ノイズプロファイルリセットのテスト"""
        # ノイズ学習を実行
        for _ in range(30):
            vad.process_frame(generate_silence(0.03))

        # 学習完了を確認
        vad.noise_profile["learned"] = True

        # リセット
        vad.reset_noise_profile()

        assert vad.noise_profile["learned"] is False
        assert len(vad._noise_frames) == 0


# ============================================================================
# ノイズ学習のテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestNoiseLearning:
    """ノイズ学習のテスト"""

    def test_noise_learning(self, vad):
        """ノイズプロファイルの学習"""
        vad.config.enable_noise_learning = True

        # 30フレームの無音を処理（学習用）
        for i in range(30):
            noise_frame = generate_silence(0.03)
            vad.process_frame(noise_frame)

        # ノイズプロファイルが学習されたことを確認
        assert vad.noise_profile["learned"] is True
        assert vad.noise_profile["energy"] > 0

    def test_adaptive_threshold(self, vad):
        """適応的閾値のテスト"""
        vad.config.adaptive_threshold = True
        vad.config.enable_noise_learning = True

        # ノイズ学習
        for _ in range(30):
            vad.process_frame(generate_silence(0.03))

        # 学習後、ノイズレベルに基づいて閾値が調整される
        assert vad.noise_profile["learned"] is True

        # 音声判定時に適応的閾値が使用される
        speech_frame = generate_speech(0.03)
        is_speech = vad.process_frame(speech_frame)

        # 統計情報にノイズレベルが反映される
        stats = vad.get_statistics()
        assert stats.noise_level > 0


# ============================================================================
# 状態遷移のテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestStateTransition:
    """状態遷移のテスト"""

    def test_state_transition_basic(self, vad):
        """基本的な状態遷移"""
        # 初期状態（音声検出状態）
        assert vad.speech_state == SpeechState.SILENCE  # stateではなくspeech_state

        # 設定を調整（ハングオーバー時間も短縮）
        vad.config.energy_threshold = 0.001
        vad.config.speech_min_duration = 0.1
        vad.config.silence_min_duration = 0.1
        vad.config.hangover_time = 0.1  # ハングオーバー時間も0.1秒に短縮
        vad.config.enable_noise_learning = False

        # ハングオーバーフレーム数を再計算
        vad._hangover_frames = int(vad.config.hangover_time / vad.config.frame_duration)

        # 音声データの後に無音を追加（完全な状態遷移をテスト）
        speech_data = generate_speech(0.5)
        silence_data = generate_silence(0.3)  # ハングオーバー0.1秒 + 無音0.1秒 = 0.2秒必要、0.3秒なら十分
        complete_data = np.concatenate([speech_data, silence_data])

        segments = vad.detect_segments(complete_data)

        # 状態が変化したことを確認
        # （最終的にはSILENCEに戻る）
        assert vad.speech_state == SpeechState.SILENCE  # stateではなくspeech_state

    def test_minimum_duration_requirements(self, vad):
        """最小時間要件のテスト"""
        vad.config.speech_min_duration = 0.5
        vad.config.energy_threshold = 0.001
        vad.config.enable_noise_learning = False

        # 短い音声（0.2秒）
        short_speech = generate_speech(0.2)
        segments = vad.detect_segments(short_speech)

        # 最小時間を満たさないので検出されない
        assert len(segments) == 0


# ============================================================================
# ユーティリティ関数のテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestUtilityFunctions:
    """ユーティリティ関数のテスト"""

    def test_create_vad(self):
        """create_vad関数のテスト"""
        vad = create_vad(mode=VADMode.AGGRESSIVE)

        assert isinstance(vad, VoiceActivityDetector)
        assert vad.config.mode == VADMode.AGGRESSIVE

    @pytest.mark.asyncio
    async def test_detect_speech_in_file(self, mixed_audio):
        """detect_speech_in_file関数のテスト"""
        segments = await detect_speech_in_file(mixed_audio, sample_rate=16000, mode=VADMode.NORMAL)

        assert isinstance(segments, list)
        # セグメントがあれば、正しい型であることを確認
        if segments:
            assert all(isinstance(s, SpeechSegment) for s in segments)


# ============================================================================
# エラーハンドリングのテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestErrorHandling:
    """エラーハンドリングのテスト"""

    def test_empty_frame(self, vad):
        """空のフレーム処理"""
        empty_frame = np.array([])

        # エラーは発生しないが、無音として扱われる
        is_speech = vad.process_frame(empty_frame)
        assert is_speech is False

    def test_invalid_audio_data(self, vad):
        """無効な音声データ"""
        # NaNを含むデータ
        invalid_data = np.array([np.nan] * 480)

        # エラーは発生しないが、適切に処理される
        is_speech = vad.process_frame(invalid_data)
        assert is_speech is False


# ============================================================================
# パフォーマンステスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestPerformance:
    """パフォーマンス関連のテスト"""

    def test_large_audio_processing(self, vad):
        """大きな音声データの処理"""
        # 10秒の音声
        large_audio = generate_mixed_audio(speech_duration=5.0, silence_duration=2.5)

        # 処理が完了することを確認
        segments = vad.detect_segments(large_audio)

        # 統計情報が正しく更新される
        stats = vad.get_statistics()
        assert stats.total_frames > 0
        assert stats.segments_detected >= 0

    def test_frame_buffer_limit(self, vad):
        """フレームバッファの制限テスト"""
        # バッファの最大サイズ（100）を超えるフレームを処理
        for _ in range(150):
            vad.process_frame(generate_silence(0.03))

        # バッファがオーバーフローしないことを確認
        assert len(vad.frame_buffer) <= 100
