"""
Voice Activity Detection (VAD) Module

音声区間検出機能を提供。
発話の開始・終了を自動検出し、効率的な音声処理を実現。

開発規約書 v1.12準拠
エラーハンドリング指針 v1.20準拠
インターフェース定義書 v1.34準拠
"""

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple

import numpy as np

# プロジェクト内インポート
from vioratalk.core.base import ComponentState, VioraTalkComponent
from vioratalk.core.i18n_manager import I18nManager

# ============================================================================
# 定数定義
# ============================================================================

# VADパラメータのデフォルト値
DEFAULT_FRAME_DURATION = 0.03  # 30ms per frame
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_ENERGY_THRESHOLD = 0.01  # エネルギー閾値
DEFAULT_ZCR_THRESHOLD = 0.1  # ゼロクロッシング率閾値
DEFAULT_SPEECH_MIN_DURATION = 0.3  # 最小発話時間（秒）
DEFAULT_SILENCE_MIN_DURATION = 0.5  # 最小無音時間（秒）
DEFAULT_HANGOVER_TIME = 0.3  # ハングオーバー時間（秒）

# 適応的閾値調整
NOISE_LEARNING_FRAMES = 30  # ノイズ学習用フレーム数
THRESHOLD_MULTIPLIER = 1.5  # ノイズレベルに対する倍率


# ============================================================================
# 列挙型定義
# ============================================================================


class VADMode(Enum):
    """VAD動作モード"""

    AGGRESSIVE = "aggressive"  # 高感度（ノイズが少ない環境）
    NORMAL = "normal"  # 標準
    CONSERVATIVE = "conservative"  # 低感度（ノイズが多い環境）


class SpeechState(Enum):
    """発話状態"""

    SILENCE = "silence"  # 無音
    SPEECH_START = "speech_start"  # 発話開始
    SPEAKING = "speaking"  # 発話中
    SPEECH_END = "speech_end"  # 発話終了


# ============================================================================
# データクラス定義
# ============================================================================


@dataclass
class VADConfig:
    """VAD設定

    Attributes:
        mode: 動作モード
        sample_rate: サンプリングレート
        frame_duration: フレーム長（秒）
        energy_threshold: エネルギー閾値
        zcr_threshold: ゼロクロッシング率閾値
        speech_min_duration: 最小発話時間
        silence_min_duration: 最小無音時間
        hangover_time: ハングオーバー時間
        adaptive_threshold: 適応的閾値調整の有効化
        enable_noise_learning: ノイズ学習の有効化
    """

    mode: VADMode = VADMode.NORMAL
    sample_rate: int = DEFAULT_SAMPLE_RATE
    frame_duration: float = DEFAULT_FRAME_DURATION
    energy_threshold: float = DEFAULT_ENERGY_THRESHOLD
    zcr_threshold: float = DEFAULT_ZCR_THRESHOLD
    speech_min_duration: float = DEFAULT_SPEECH_MIN_DURATION
    silence_min_duration: float = DEFAULT_SILENCE_MIN_DURATION
    hangover_time: float = DEFAULT_HANGOVER_TIME
    adaptive_threshold: bool = True
    enable_noise_learning: bool = True

    def __post_init__(self):
        """初期化後の調整"""
        # モードに応じた閾値調整
        if self.mode == VADMode.AGGRESSIVE:
            self.energy_threshold *= 0.7
            self.silence_min_duration *= 0.7
        elif self.mode == VADMode.CONSERVATIVE:
            self.energy_threshold *= 1.5
            self.silence_min_duration *= 1.3


@dataclass
class SpeechSegment:
    """音声区間情報

    Attributes:
        start_time: 開始時刻（秒）
        end_time: 終了時刻（秒）
        start_sample: 開始サンプル位置
        end_sample: 終了サンプル位置
        confidence: 信頼度（0.0-1.0）
        energy_level: 平均エネルギーレベル
    """

    start_time: float
    end_time: float
    start_sample: int
    end_sample: int
    confidence: float = 0.0
    energy_level: float = 0.0

    @property
    def duration(self) -> float:
        """区間の長さ（秒）"""
        return self.end_time - self.start_time

    def to_dict(self) -> dict:
        """辞書形式に変換"""
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "confidence": self.confidence,
            "energy_level": self.energy_level,
        }


@dataclass
class VADStatistics:
    """VAD統計情報

    Attributes:
        total_frames: 処理したフレーム総数
        speech_frames: 音声フレーム数
        silence_frames: 無音フレーム数
        noise_level: 推定ノイズレベル
        average_energy: 平均エネルギー
        segments_detected: 検出されたセグメント数
    """

    total_frames: int = 0
    speech_frames: int = 0
    silence_frames: int = 0
    noise_level: float = 0.0
    average_energy: float = 0.0
    segments_detected: int = 0


# ============================================================================
# VoiceActivityDetectorクラス
# ============================================================================


class VoiceActivityDetector(VioraTalkComponent):
    """音声区間検出クラス

    エネルギーベースとゼロクロッシング率を組み合わせた
    VADアルゴリズムを実装。

    Note:
        プッシュトゥトークモードとの併用も可能。
        自動検出モードとマニュアルモードの切り替えをサポート。

    Attributes:
        config: VAD設定
        speech_state: 現在の発話状態（SpeechState）
        statistics: 統計情報
        noise_profile: ノイズプロファイル
        frame_buffer: フレームバッファ
    """

    def __init__(self, config: Optional[VADConfig] = None):
        """初期化

        Args:
            config: VAD設定（Noneの場合はデフォルト使用）
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.config = config or VADConfig()
        self.i18n = I18nManager()

        # 音声検出状態管理（基底クラスのstateプロパティと区別）
        self.speech_state = SpeechState.SILENCE
        self.statistics = VADStatistics()

        # ノイズプロファイル
        self.noise_profile = {
            "energy": self.config.energy_threshold,
            "zcr": self.config.zcr_threshold,
            "learned": False,
        }

        # バッファ管理
        self.frame_size = int(self.config.sample_rate * self.config.frame_duration)
        self.frame_buffer = deque(maxlen=100)  # 最大100フレーム保持

        # 状態遷移管理
        self._speech_start_time: Optional[float] = None
        self._silence_start_time: Optional[float] = None
        self._hangover_frames = int(self.config.hangover_time / self.config.frame_duration)
        self._hangover_counter = 0

        # ノイズ学習用バッファ
        self._noise_frames: List[np.ndarray] = []
        self._energy_history = deque(maxlen=50)  # エネルギー履歴

        # インターフェース定義書v1.34準拠
        self._last_used: Optional[datetime] = None
        # _stateは基底クラスで初期化済み（ComponentState.NOT_INITIALIZED）

    async def initialize(self) -> None:
        """非同期初期化"""
        # 基底クラスの初期化は不要（abstractメソッドのため）

        # VAD固有の初期化処理
        self._state = ComponentState.INITIALIZING
        try:
            # 初期化処理（必要に応じて）
            self._last_used = datetime.now()
            self._state = ComponentState.READY

            self.logger.info(
                "VAD initialized",
                extra={
                    "mode": self.config.mode.value,
                    "sample_rate": self.config.sample_rate,
                    "adaptive": self.config.adaptive_threshold,
                },
            )
        except Exception as e:
            self._state = ComponentState.ERROR
            self._error = e
            raise

    async def cleanup(self) -> None:
        """リソースのクリーンアップ

        インターフェース定義書v1.34準拠の実装。
        状態遷移: READY/RUNNING/ERROR → TERMINATING → TERMINATED
        """
        if self._state == ComponentState.TERMINATED:
            return

        self._state = ComponentState.TERMINATING

        try:
            # バッファのクリア
            self.frame_buffer.clear()
            self._noise_frames.clear()
            self._energy_history.clear()

            # 統計情報のリセット
            self.statistics = VADStatistics()

            # 音声検出状態のリセット
            self.speech_state = SpeechState.SILENCE
            self._speech_start_time = None
            self._silence_start_time = None
            self._hangover_counter = 0

            self.logger.info("VAD cleanup completed")

        except Exception as e:
            # エラーが発生してもログ記録のみで継続
            self.logger.error(f"Error during VAD cleanup: {e}")
        finally:
            self._state = ComponentState.TERMINATED

    def is_available(self) -> bool:
        """利用可能状態の確認

        インターフェース定義書v1.34準拠の実装。

        Returns:
            bool: READYまたはRUNNING状態の場合True
        """
        return ComponentState.is_operational(self._state)

    def get_status(self) -> Dict[str, Any]:
        """コンポーネントの状態を取得

        インターフェース定義書v1.34準拠の実装。

        Returns:
            Dict[str, Any]: 状態情報を含む辞書
        """
        return {
            "state": self._state,
            "is_available": self.is_available(),
            "error": str(self._error) if self._error else None,
            "last_used": self._last_used,
            # VAD固有の情報
            "speech_state": self.speech_state.value,
            "mode": self.config.mode.value,
            "statistics": {
                "total_frames": self.statistics.total_frames,
                "speech_frames": self.statistics.speech_frames,
                "silence_frames": self.statistics.silence_frames,
                "segments_detected": self.statistics.segments_detected,
                "noise_level": self.statistics.noise_level,
                "average_energy": self.statistics.average_energy,
            },
            "noise_learned": self.noise_profile["learned"],
        }

    # ========================================================================
    # メインAPI
    # ========================================================================

    def process_frame(self, audio_frame: np.ndarray) -> bool:
        """単一フレームの音声判定

        Args:
            audio_frame: 音声フレーム（numpy配列）

        Returns:
            bool: 音声の場合True、無音の場合False
        """
        # 使用時刻を更新
        self._last_used = datetime.now()

        # 状態をRUNNINGに更新
        if self._state == ComponentState.READY:
            self._state = ComponentState.RUNNING

        # 統計更新
        self.statistics.total_frames += 1

        # 特徴量計算
        energy = self._calculate_energy(audio_frame)
        zcr = self._calculate_zcr(audio_frame)

        # エネルギー履歴更新
        self._energy_history.append(energy)

        # ノイズ学習（初期フレーム）
        if self.config.enable_noise_learning and not self.noise_profile["learned"]:
            self._learn_noise(audio_frame, energy, zcr)

        # 音声判定
        is_speech = self._is_speech(energy, zcr)

        # 統計更新
        if is_speech:
            self.statistics.speech_frames += 1
        else:
            self.statistics.silence_frames += 1

        # 平均エネルギー更新
        if self._energy_history:
            self.statistics.average_energy = np.mean(self._energy_history)

        return is_speech

    def detect_segments(
        self, audio_data: np.ndarray, return_audio: bool = False
    ) -> List[SpeechSegment]:
        """音声データから音声区間を検出

        Args:
            audio_data: 音声データ全体
            return_audio: 音声データも返すかどうか

        Returns:
            List[SpeechSegment]: 検出された音声区間のリスト
        """
        # 使用時刻を更新
        self._last_used = datetime.now()

        # 状態をRUNNINGに更新
        if self._state == ComponentState.READY:
            self._state = ComponentState.RUNNING

        segments = []
        current_segment = None

        # フレーム単位で処理
        num_frames = len(audio_data) // self.frame_size

        for i in range(num_frames):
            start_idx = i * self.frame_size
            end_idx = start_idx + self.frame_size
            frame = audio_data[start_idx:end_idx]

            # フレーム処理
            is_speech = self.process_frame(frame)

            # 状態遷移処理
            new_state = self._update_state(is_speech, i * self.config.frame_duration)

            # セグメント管理
            if new_state == SpeechState.SPEECH_START:
                # 新しいセグメント開始
                current_segment = SpeechSegment(
                    start_time=i * self.config.frame_duration,
                    end_time=0,
                    start_sample=start_idx,
                    end_sample=0,
                )

            elif new_state == SpeechState.SPEECH_END and current_segment:
                # セグメント終了
                current_segment.end_time = i * self.config.frame_duration
                current_segment.end_sample = end_idx
                current_segment.confidence = self._calculate_confidence()
                current_segment.energy_level = float(np.mean(self._energy_history))
                segments.append(current_segment)
                current_segment = None
                self.statistics.segments_detected += 1

        # 最後のセグメントが終了していない場合
        if current_segment and self.speech_state == SpeechState.SPEAKING:
            current_segment.end_time = num_frames * self.config.frame_duration
            current_segment.end_sample = len(audio_data)
            current_segment.confidence = self._calculate_confidence()
            current_segment.energy_level = float(np.mean(self._energy_history))
            segments.append(current_segment)
            self.statistics.segments_detected += 1

        # 処理完了後、状態をREADYに戻す
        if self._state == ComponentState.RUNNING:
            self._state = ComponentState.READY

        return segments

    async def process_stream(
        self,
        audio_stream: AsyncGenerator[np.ndarray, None],
        callback: Optional[Callable[[SpeechState, float], None]] = None,
    ) -> AsyncGenerator[Tuple[SpeechState, np.ndarray], None]:
        """ストリーミング音声の処理

        リアルタイムで音声区間を検出し、状態変化を通知。

        Args:
            audio_stream: 音声ストリーム（非同期ジェネレータ）
            callback: 状態変化時のコールバック

        Yields:
            Tuple[SpeechState, np.ndarray]: (状態, 音声フレーム)
        """
        # 使用時刻を更新
        self._last_used = datetime.now()

        # 状態をRUNNINGに更新
        if self._state == ComponentState.READY:
            self._state = ComponentState.RUNNING

        frame_count = 0

        try:
            async for audio_chunk in audio_stream:
                # チャンクをフレームに分割
                frames = self._split_to_frames(audio_chunk)

                for frame in frames:
                    is_speech = self.process_frame(frame)
                    current_time = frame_count * self.config.frame_duration
                    new_state = self._update_state(is_speech, current_time)

                    # 状態変化時のコールバック
                    if callback and new_state != self.speech_state:
                        callback(new_state, current_time)

                    # 状態とフレームを返す
                    yield (new_state, frame)

                    frame_count += 1
        finally:
            # 処理完了後、状態をREADYに戻す
            if self._state == ComponentState.RUNNING:
                self._state = ComponentState.READY

    # ========================================================================
    # 設定・調整メソッド
    # ========================================================================

    def set_mode(self, mode: VADMode) -> None:
        """動作モードを設定

        Args:
            mode: VADモード
        """
        self.config.mode = mode
        self.config.__post_init__()  # 閾値を再調整
        self.logger.info(f"VAD mode changed to: {mode.value}")

    def adjust_sensitivity(self, factor: float) -> None:
        """感度を調整

        Args:
            factor: 調整係数（1.0が基準、小さいほど高感度）
        """
        self.config.energy_threshold *= factor
        self.config.zcr_threshold *= factor
        self.logger.info(f"VAD sensitivity adjusted by factor: {factor}")

    def reset_noise_profile(self) -> None:
        """ノイズプロファイルをリセット"""
        self.noise_profile["learned"] = False
        self._noise_frames.clear()
        self.logger.info("Noise profile reset")

    def get_statistics(self) -> VADStatistics:
        """統計情報を取得

        Returns:
            VADStatistics: 統計情報
        """
        self.statistics.noise_level = self.noise_profile["energy"]
        return self.statistics

    # ========================================================================
    # プライベートメソッド
    # ========================================================================

    def _calculate_energy(self, frame: np.ndarray) -> float:
        """フレームのエネルギーを計算

        Args:
            frame: 音声フレーム

        Returns:
            float: エネルギー値
        """
        if len(frame) == 0:
            return 0.0
        return float(np.sum(frame**2) / len(frame))

    def _calculate_zcr(self, frame: np.ndarray) -> float:
        """ゼロクロッシング率を計算

        Args:
            frame: 音声フレーム

        Returns:
            float: ゼロクロッシング率
        """
        if len(frame) == 0:
            return 0.0
        signs = np.sign(frame)
        signs[signs == 0] = -1  # ゼロを-1として扱う
        zcr = np.sum(signs[:-1] != signs[1:]) / (2 * len(frame))
        return float(zcr)

    def _is_speech(self, energy: float, zcr: float) -> bool:
        """音声かどうかを判定

        Args:
            energy: エネルギー値
            zcr: ゼロクロッシング率

        Returns:
            bool: 音声の場合True
        """
        # 適応的閾値を使用
        if self.config.adaptive_threshold and self.noise_profile["learned"]:
            energy_threshold = self.noise_profile["energy"] * THRESHOLD_MULTIPLIER
        else:
            energy_threshold = self.config.energy_threshold

        # エネルギーとZCRの両方を考慮
        energy_check = energy > energy_threshold
        zcr_check = zcr < self.config.zcr_threshold * 2  # ZCRが高すぎない

        return energy_check and zcr_check

    def _update_state(self, is_speech: bool, current_time: float) -> SpeechState:
        """音声検出状態を更新

        Args:
            is_speech: 現在のフレームが音声かどうか
            current_time: 現在時刻（秒）

        Returns:
            SpeechState: 新しい音声検出状態
        """
        previous_state = self.speech_state

        if self.speech_state == SpeechState.SILENCE:
            if is_speech:
                # 発話開始の可能性
                if self._speech_start_time is None:
                    self._speech_start_time = current_time
                elif current_time - self._speech_start_time >= self.config.speech_min_duration:
                    # 最小発話時間を満たした
                    self.speech_state = SpeechState.SPEECH_START
                    self._speech_start_time = None
            else:
                # 無音継続
                self._speech_start_time = None

        elif self.speech_state == SpeechState.SPEECH_START:
            # すぐにSPEAKING状態へ
            self.speech_state = SpeechState.SPEAKING

        elif self.speech_state == SpeechState.SPEAKING:
            if not is_speech:
                # 無音検出（ハングオーバー開始）
                if self._hangover_counter < self._hangover_frames:
                    self._hangover_counter += 1
                else:
                    # ハングオーバー終了
                    if self._silence_start_time is None:
                        self._silence_start_time = current_time
                    elif (
                        current_time - self._silence_start_time >= self.config.silence_min_duration
                    ):
                        # 最小無音時間を満たした
                        self.speech_state = SpeechState.SPEECH_END
                        self._silence_start_time = None
                        self._hangover_counter = 0
            else:
                # 発話継続
                self._hangover_counter = 0
                self._silence_start_time = None

        elif self.speech_state == SpeechState.SPEECH_END:
            # すぐにSILENCE状態へ
            self.speech_state = SpeechState.SILENCE

        return self.speech_state

    def _learn_noise(self, frame: np.ndarray, energy: float, zcr: float) -> None:
        """ノイズプロファイルを学習

        Args:
            frame: 音声フレーム
            energy: エネルギー値
            zcr: ゼロクロッシング率
        """
        self._noise_frames.append(frame)

        if len(self._noise_frames) >= NOISE_LEARNING_FRAMES:
            # ノイズレベルを計算
            noise_energies = [self._calculate_energy(f) for f in self._noise_frames]
            noise_zcrs = [self._calculate_zcr(f) for f in self._noise_frames]

            # 中央値を使用（外れ値に強い）
            self.noise_profile["energy"] = float(np.median(noise_energies))
            self.noise_profile["zcr"] = float(np.median(noise_zcrs))
            self.noise_profile["learned"] = True

            self.logger.info(
                "Noise profile learned",
                extra={"energy": self.noise_profile["energy"], "zcr": self.noise_profile["zcr"]},
            )

    def _calculate_confidence(self) -> float:
        """信頼度を計算（改善版）

        開発規約書 v1.12準拠：実動作を重視した実装
        テスト戦略ガイドライン v1.7準拠：実用性を優先

        Returns:
            float: 信頼度（0.1-1.0）最小値0.1を保証
        """
        if not self._energy_history:
            return 0.5  # デフォルト中間値

        energies = list(self._energy_history)
        mean_energy = np.mean(energies)

        # 1. エネルギーレベルベースの基本信頼度
        if mean_energy <= 0:
            return 0.1  # 最小信頼度を保証

        # 閾値との比較で基本信頼度を計算
        # 適応的閾値を使用している場合は、それを基準にする
        if self.config.adaptive_threshold and self.noise_profile["learned"]:
            effective_threshold = self.noise_profile["energy"] * THRESHOLD_MULTIPLIER
        else:
            effective_threshold = self.config.energy_threshold

        # エネルギーが閾値の何倍かで信頼度を決定
        if effective_threshold > 0:
            threshold_ratio = mean_energy / effective_threshold
            # 閾値の1-3倍を0.3-1.0にマッピング
            base_confidence = min(1.0, max(0.3, (threshold_ratio - 1.0) / 2.0 + 0.5))
        else:
            base_confidence = 0.5

        # 2. 安定性による補正（補助的な指標）
        std_energy = np.std(energies)
        if mean_energy > 0:
            cv = std_energy / mean_energy  # 変動係数

            # 変動係数による安定性評価
            if cv < 0.3:  # 非常に安定
                stability_factor = 1.0
            elif cv < 0.6:  # 安定
                stability_factor = 0.9
            elif cv < 1.0:  # やや不安定
                stability_factor = 0.8
            elif cv < 1.5:  # 不安定
                stability_factor = 0.7
            else:  # 非常に不安定
                stability_factor = 0.6
        else:
            stability_factor = 0.8

        # 3. 最終的な信頼度を計算
        confidence = base_confidence * stability_factor

        # 最小値0.1を保証（完全に信頼度0にならない）
        return max(0.1, min(1.0, confidence))

    def _split_to_frames(self, audio_chunk: np.ndarray) -> List[np.ndarray]:
        """音声チャンクをフレームに分割

        Args:
            audio_chunk: 音声チャンク

        Returns:
            List[np.ndarray]: フレームのリスト
        """
        frames = []
        num_frames = len(audio_chunk) // self.frame_size

        for i in range(num_frames):
            start = i * self.frame_size
            end = start + self.frame_size
            frames.append(audio_chunk[start:end])

        # 残りのサンプルがある場合
        remainder = len(audio_chunk) % self.frame_size
        if remainder > 0:
            # パディングして最後のフレームを作成
            last_frame = np.zeros(self.frame_size)
            last_frame[:remainder] = audio_chunk[-remainder:]
            frames.append(last_frame)

        return frames


# ============================================================================
# ユーティリティ関数
# ============================================================================


def create_vad(mode: VADMode = VADMode.NORMAL) -> VoiceActivityDetector:
    """VADインスタンスを作成

    Args:
        mode: 動作モード

    Returns:
        VoiceActivityDetector: VADインスタンス
    """
    config = VADConfig(mode=mode)
    return VoiceActivityDetector(config=config)


async def detect_speech_in_file(
    audio_data: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE, mode: VADMode = VADMode.NORMAL
) -> List[SpeechSegment]:
    """ファイル内の音声区間を検出

    Args:
        audio_data: 音声データ
        sample_rate: サンプリングレート
        mode: VADモード

    Returns:
        List[SpeechSegment]: 検出された音声区間
    """
    config = VADConfig(mode=mode, sample_rate=sample_rate)
    vad = VoiceActivityDetector(config=config)
    await vad.initialize()

    segments = vad.detect_segments(audio_data)

    return segments


# ============================================================================
# エクスポート定義
# ============================================================================

__all__ = [
    "VoiceActivityDetector",
    "VADConfig",
    "VADMode",
    "SpeechState",
    "SpeechSegment",
    "VADStatistics",
    "create_vad",
    "detect_speech_in_file",
]
