"""FasterWhisperEngine実装

Faster-Whisperを使用した音声認識エンジンの実装。
高速かつ高精度な音声認識を提供。

インターフェース定義書 v1.34準拠
エンジン初期化仕様書 v1.4準拠
エラーハンドリング指針 v1.20準拠
開発規約書 v1.12準拠
"""

import asyncio
import os
import tempfile
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

# faster-whisperインポート（エラーハンドリング付き）
try:
    from faster_whisper import WhisperModel

    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    WhisperModel = None
    warnings.warn(
        "faster-whisper is not installed. FasterWhisperEngine will not be available.", ImportWarning
    )

# プロジェクト内インポート（絶対インポート）
from vioratalk.core.base import ComponentState
from vioratalk.core.error_handler import get_default_error_handler
from vioratalk.core.exceptions import (
    AudioError,
    FileSystemError,
    InitializationError,
    ModelNotFoundError,
    NetworkError,
    STTError,
)
from vioratalk.core.stt.base import AudioData, BaseSTTEngine, STTConfig, TranscriptionResult
from vioratalk.infrastructure.model_download_manager import ModelDownloadManager
from vioratalk.utils.logger_manager import LoggerManager
from vioratalk.utils.progress import ProgressBar

# ============================================================================
# 定数定義
# ============================================================================

# サポートするモデルサイズ
SUPPORTED_MODELS = {
    "tiny": {"size_mb": 39, "parameters": "39M", "languages": 99},
    "base": {"size_mb": 74, "parameters": "74M", "languages": 99},
    "small": {"size_mb": 244, "parameters": "244M", "languages": 99},
    "medium": {"size_mb": 769, "parameters": "769M", "languages": 99},
    "large": {"size_mb": 1550, "parameters": "1550M", "languages": 99},
    "large-v2": {"size_mb": 1550, "parameters": "1550M", "languages": 99},
    "large-v3": {"size_mb": 1550, "parameters": "1550M", "languages": 99},
}

# デフォルト設定
DEFAULT_MODEL = "base"
DEFAULT_COMPUTE_TYPE = "int8"  # CPU向け最適化
DEFAULT_DEVICE = "auto"
DEFAULT_BEAM_SIZE = 5
DEFAULT_VAD_FILTER = True

# 主要言語コード（Phase 4では日本語と英語のみ）
PRIMARY_LANGUAGES = ["ja", "en"]

# 全サポート言語（将来的な拡張用）
ALL_LANGUAGES = [
    "ja",
    "en",
    "zh",
    "ko",
    "es",
    "fr",
    "de",
    "ru",
    "ar",
    "hi",
    "pt",
    "it",
    "tr",
    "pl",
    "nl",
    "sv",
]


# ============================================================================
# FasterWhisperEngine実装
# ============================================================================


class FasterWhisperEngine(BaseSTTEngine):
    """Faster-Whisperを使用した音声認識エンジン

    高速かつ高精度な音声認識を提供。
    CPUでも実用的な速度で動作し、GPUがあれば自動的に利用。

    Attributes:
        model: WhisperModelインスタンス
        model_manager: モデルダウンロード管理
        error_handler: エラーハンドリング
        logger: ロガー
        model_path: ダウンロードしたモデルのパス
        _model_loaded: モデルロード状態
        _processing: 処理中フラグ
    """

    def __init__(self, config: Optional[STTConfig] = None):
        """初期化

        Args:
            config: STTエンジン設定（Noneの場合はデフォルト使用）

        Raises:
            InitializationError: faster-whisperが利用不可の場合
        """
        # 基底クラスの初期化
        super().__init__(config)

        # faster-whisper利用可能性チェック
        if not FASTER_WHISPER_AVAILABLE:
            raise InitializationError(
                "faster-whisper is not installed",
                error_code="E0007",
                component="FasterWhisperEngine",
            )

        # 内部状態の初期化
        self._state = ComponentState.NOT_INITIALIZED
        self._initialized_at: Optional[datetime] = None
        self._error: Optional[Exception] = None
        self._model_loaded = False
        self._processing = False

        # コンポーネントの初期化
        self.logger = LoggerManager.get_logger(self.__class__.__name__)
        self.error_handler = get_default_error_handler()
        self.model_manager = ModelDownloadManager()

        # Whisperモデル（遅延ロード）
        self.model: Optional[WhisperModel] = None
        self.model_path: Optional[Path] = None  # ダウンロードしたモデルのパス

        # サポート言語の設定（Phase 4では限定的）
        self.supported_languages = PRIMARY_LANGUAGES.copy()

        # 現在のモデル名
        self.current_model = self.config.model or DEFAULT_MODEL

        # デバイス設定の検証と調整
        self._setup_device()

        self.logger.info(
            f"FasterWhisperEngine initialized with model: {self.current_model}, "
            f"device: {self.config.device}"
        )

    def _setup_device(self) -> None:
        """デバイス設定のセットアップ

        GPUの利用可能性をチェックし、適切なデバイスを設定。
        """
        if self.config.device == "auto":
            # 自動選択モード
            try:
                import torch

                if torch.cuda.is_available():
                    self.config.device = "cuda"
                    self.logger.info("CUDA is available. Using GPU acceleration.")
                else:
                    self.config.device = "cpu"
                    self.logger.info("CUDA not available. Using CPU.")
            except ImportError:
                self.config.device = "cpu"
                self.logger.info("PyTorch not installed. Using CPU.")

        # compute_typeの調整
        if self.config.device == "cpu":
            self.config.compute_type = "int8"  # CPU最適化
        elif self.config.device == "cuda":
            self.config.compute_type = "float16"  # GPU最適化

    async def initialize(self) -> None:
        """非同期初期化

        モデルのダウンロードとロードを実行。

        Raises:
            InitializationError: 初期化失敗
        """
        if self._state != ComponentState.NOT_INITIALIZED:
            self.logger.warning(f"Already initialized (state: {self._state})")
            return

        try:
            self._state = ComponentState.INITIALIZING
            self.logger.info("Starting FasterWhisperEngine initialization...")

            # モデルのダウンロード確認
            await self._ensure_model_available()

            # モデルのロード
            await self._load_model()

            self._state = ComponentState.READY
            self._initialized_at = datetime.utcnow()
            self.logger.info("FasterWhisperEngine initialization completed")

        except Exception as e:
            self._state = ComponentState.ERROR
            self._error = e
            self.logger.error(f"Initialization failed: {e}")
            raise InitializationError(
                f"Failed to initialize FasterWhisperEngine: {str(e)}",
                error_code="E0007",
                component="FasterWhisperEngine",
            ) from e

    async def _ensure_model_available(self) -> None:
        """モデルの利用可能性を確保

        ModelDownloadManagerを使用してWhisperモデルをダウンロード。
        統一プログレスバーで進捗表示。
        エラーハンドリング指針 v1.20準拠。
        """
        # モデル情報の検証
        model_info = SUPPORTED_MODELS.get(self.current_model)
        if not model_info:
            # ModelNotFoundError（E2100）を使用
            error = ModelNotFoundError(
                f"Unsupported Whisper model: {self.current_model}",
                error_code="E2100",  # LLMモデルが見つからない
                model_name=self.current_model,
            )

            # エラーハンドラーで処理
            await self.error_handler.handle_error_async(
                error,
                context={
                    "phase": "initialization",
                    "component": "FasterWhisperEngine",
                    "operation": "model_validation",
                },
            )
            raise error

        # プログレスバー設定
        progress_bar = ProgressBar(
            prefix=f"Downloading Whisper {self.current_model}",
            suffix=f"({model_info['size_mb']}MB)",
            show_time=True,
        )

        try:
            # ModelDownloadManagerでダウンロード
            self.logger.info(
                f"Ensuring Whisper model '{self.current_model}' is available...",
                extra={
                    "component": "FasterWhisperEngine",
                    "model": self.current_model,
                    "size_mb": model_info["size_mb"],
                },
            )

            # download_whisper_modelを使用（ModelDownloadManagerの新しいメソッド）
            self.model_path = await self.model_manager.download_whisper_model(
                model_size=self.current_model, progress_callback=progress_bar.create_callback()
            )

            # 成功ログ
            self.logger.info(
                f"Whisper model '{self.current_model}' ready at {self.model_path}",
                extra={
                    "component": "FasterWhisperEngine",
                    "model": self.current_model,
                    "path": str(self.model_path),
                },
            )

            # 完了メッセージ（close()メソッドを使用）
            progress_bar.close(f"Model {self.current_model} downloaded successfully")

        except FileSystemError as e:
            # ファイルシステムエラーはそのまま伝播
            self.logger.error(
                f"File system error during model download: {e}",
                extra={
                    "error_code": e.error_code,
                    "model": self.current_model,
                    "component": "FasterWhisperEngine",
                },
            )

            await self.error_handler.handle_error_async(
                e,
                context={
                    "phase": "initialization",
                    "component": "FasterWhisperEngine",
                    "operation": "model_download",
                    "model": self.current_model,
                },
            )
            raise

        except NetworkError as e:
            # ネットワークエラーはそのまま伝播
            self.logger.error(
                f"Network error during model download: {e}",
                extra={
                    "error_code": e.error_code,
                    "model": self.current_model,
                    "component": "FasterWhisperEngine",
                },
            )

            await self.error_handler.handle_error_async(
                e,
                context={
                    "phase": "initialization",
                    "component": "FasterWhisperEngine",
                    "operation": "model_download",
                    "model": self.current_model,
                },
            )
            raise

        except Exception as e:
            # 予期しないエラーはModelNotFoundErrorに変換
            self.logger.error(
                f"Unexpected error during model download: {e}",
                extra={
                    "model": self.current_model,
                    "component": "FasterWhisperEngine",
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )

            error = ModelNotFoundError(
                f"Failed to download Whisper model '{self.current_model}': {str(e)}",
                error_code="E2101",  # モデルロード失敗
                model_name=self.current_model,
            )
            error.details["original_error"] = str(e)
            error.details["error_type"] = type(e).__name__

            await self.error_handler.handle_error_async(
                error,
                context={
                    "phase": "initialization",
                    "component": "FasterWhisperEngine",
                    "operation": "model_download",
                    "model": self.current_model,
                },
            )
            raise error from e

    async def _load_model(self) -> None:
        """モデルをメモリにロード

        別スレッドで実行してブロッキングを回避。
        """
        loop = asyncio.get_event_loop()

        def load_model_sync():
            """同期的なモデルロード"""
            try:
                # モデルパスの決定（ダウンロード済みのパスを優先）
                if self.model_path:
                    model_size_or_path = str(self.model_path)
                elif self.config.model_path:
                    model_size_or_path = self.config.model_path
                else:
                    model_size_or_path = self.current_model

                # WhisperModelのロード
                self.model = WhisperModel(
                    model_size_or_path=model_size_or_path,
                    device=self.config.device,
                    compute_type=self.config.compute_type,
                    cpu_threads=0,  # 自動設定
                    num_workers=1,
                    download_root=None,  # デフォルトキャッシュ使用
                    local_files_only=False,
                )
                self._model_loaded = True
                self.logger.info(
                    f"Model '{self.current_model}' loaded successfully",
                    extra={
                        "component": "FasterWhisperEngine",
                        "model": self.current_model,
                        "device": self.config.device,
                        "compute_type": self.config.compute_type,
                    },
                )

            except Exception as e:
                self.logger.error(
                    f"Failed to load model: {e}",
                    extra={
                        "component": "FasterWhisperEngine",
                        "model": self.current_model,
                        "error_type": type(e).__name__,
                    },
                    exc_info=True,
                )
                raise

        # 別スレッドで実行
        await loop.run_in_executor(None, load_model_sync)

    async def cleanup(self) -> None:
        """リソースのクリーンアップ

        モデルのアンロードとメモリ解放。
        """
        if self._state == ComponentState.TERMINATED:
            return

        try:
            self._state = ComponentState.TERMINATING
            self.logger.info("Cleaning up FasterWhisperEngine...")

            # 処理中の場合は待機
            while self._processing:
                await asyncio.sleep(0.1)

            # モデルの解放
            if self.model:
                del self.model
                self.model = None
                self._model_loaded = False

            self._state = ComponentState.TERMINATED
            self.logger.info("FasterWhisperEngine cleanup completed")

        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
            self._state = ComponentState.ERROR
            self._error = e

    async def transcribe(
        self, audio_data: AudioData, language: Optional[str] = None
    ) -> TranscriptionResult:
        """音声をテキストに変換

        Args:
            audio_data: 音声データ
            language: 認識言語（Noneの場合は自動検出）

        Returns:
            TranscriptionResult: 認識結果

        Raises:
            STTError: エンジン未準備（E1000）
            AudioError: 音声データエラー（E1001）
        """
        # 状態チェック（修正: クラスメソッドとして呼び出し）
        if not ComponentState.is_operational(self._state):
            raise STTError(
                "STT engine is not ready", error_code="E1000", details={"state": str(self._state)}
            )

        # モデルロードチェック
        if not self._model_loaded or self.model is None:
            raise STTError(
                "Model is not loaded", error_code="E1000", details={"model": self.current_model}
            )

        # 音声データ検証
        if not await self.validate_audio(audio_data):
            raise AudioError(
                "Invalid audio data", error_code="E1001", audio_file=audio_data.metadata.filename
            )

        # 処理フラグ設定
        self._processing = True
        start_time = datetime.utcnow()

        try:
            # 音声データの準備
            audio_array = await self._prepare_audio(audio_data)

            # 言語の決定
            target_language = language or self.config.language
            if target_language not in self.supported_languages:
                self.logger.warning(f"Unsupported language '{target_language}', using 'ja'")
                target_language = "ja"

            # Whisperで音声認識（別スレッドで実行）
            result = await self._run_transcription(audio_array, target_language)

            # 処理時間計算
            duration = (datetime.utcnow() - start_time).total_seconds()

            # TranscriptionResult作成
            return TranscriptionResult(
                text=result["text"],
                confidence=result["confidence"],
                language=result["language"],
                duration=duration,
                alternatives=result.get("alternatives", []),
            )

        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")

            # エラーハンドリング
            if "timeout" in str(e).lower():
                raise STTError(
                    "Transcription timeout",
                    error_code="E1004",
                    details={"duration": audio_data.metadata.duration},
                ) from e
            elif "format" in str(e).lower():
                raise AudioError(
                    f"Unsupported audio format: {audio_data.metadata.format}", error_code="E1002"
                ) from e
            else:
                raise STTError(f"Transcription failed: {str(e)}", error_code="E1000") from e

        finally:
            self._processing = False

    async def _prepare_audio(self, audio_data: AudioData) -> np.ndarray:
        """音声データを処理用に準備

        Args:
            audio_data: 入力音声データ

        Returns:
            np.ndarray: 処理用音声配列
        """
        # raw_dataがある場合はそのまま使用
        if audio_data.raw_data is not None:
            audio_array = audio_data.raw_data

            # 空の配列チェック
            if len(audio_array) == 0:
                raise AudioError("Audio data is empty", error_code="E1001")

            # float32に変換（必要な場合）
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)

            # 正規化（-1.0 to 1.0）
            if np.abs(audio_array).max() > 1.0:
                audio_array = audio_array / np.abs(audio_array).max()

        # encoded_dataの場合はデコード
        elif audio_data.encoded_data is not None:
            # 一時ファイルに書き出し（faster-whisperが直接読み込むため）
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(audio_data.encoded_data)
                tmp_path = tmp_file.name

            try:
                # faster-whisperで直接読み込み
                # （内部でffmpegを使用してデコード）
                audio_array = tmp_path  # パスをそのまま渡す
            finally:
                # 一時ファイル削除
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        else:
            raise AudioError("No audio data available", error_code="E1001")

        return audio_array

    async def _run_transcription(
        self, audio: Union[np.ndarray, str], language: str
    ) -> Dict[str, Any]:
        """実際の音声認識処理

        Args:
            audio: 音声データまたはファイルパス
            language: 言語コード

        Returns:
            Dict[str, Any]: 認識結果
        """
        loop = asyncio.get_event_loop()

        def transcribe_sync():
            """同期的な音声認識"""
            # transcribeメソッドを呼び出し
            segments, info = self.model.transcribe(
                audio=audio,
                language=language,
                beam_size=DEFAULT_BEAM_SIZE,
                best_of=5,
                patience=1.0,
                length_penalty=1.0,
                temperature=0.0,  # グリーディーデコード
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                condition_on_previous_text=True,
                initial_prompt=None,
                word_timestamps=False,
                prepend_punctuations="\"'¿([{-",  # 特殊文字を正しくエスケープ
                append_punctuations="\"'.。,，!！?？:：)]}、",  # 括弧を引用符内に移動
                vad_filter=DEFAULT_VAD_FILTER,
                vad_parameters=None,
            )

            # セグメントを結合してテキスト化
            full_text = ""
            all_segments = []
            total_log_prob = 0.0
            segment_count = 0

            for segment in segments:
                full_text += segment.text
                all_segments.append(
                    {
                        "text": segment.text,
                        "start": segment.start,
                        "end": segment.end,
                        "confidence": np.exp(segment.avg_logprob),
                    }
                )
                total_log_prob += segment.avg_logprob
                segment_count += 1

            # 平均信頼度計算
            avg_confidence = np.exp(total_log_prob / segment_count) if segment_count > 0 else 0.0

            # 代替候補生成（信頼度が低い場合）
            alternatives = []
            if avg_confidence < 0.7 and all_segments:
                # 最初のセグメントの別バージョン
                alternatives.append(
                    {"text": full_text + "？", "confidence": avg_confidence * 0.8}  # 疑問形
                )

            return {
                "text": full_text.strip(),
                "confidence": float(avg_confidence),
                "language": info.language or language,
                "segments": all_segments,
                "alternatives": alternatives,
            }

        # 別スレッドで実行
        return await loop.run_in_executor(None, transcribe_sync)

    def get_supported_languages(self) -> List[str]:
        """サポートする言語のリスト取得

        Returns:
            List[str]: 言語コードのリスト
        """
        return self.supported_languages.copy()

    def set_model(self, model_name: str) -> None:
        """使用するモデルを設定

        Args:
            model_name: モデル名

        Raises:
            ModelNotFoundError: サポートされていないモデル
            RuntimeError: 処理中の変更
        """
        # 処理中チェック
        if self._processing:
            raise RuntimeError("Cannot change model while processing")

        # モデル名検証
        if model_name not in SUPPORTED_MODELS:
            # 修正: available_models引数を削除
            raise ModelNotFoundError(
                f"Unsupported model: {model_name}", error_code="E2100", model_name=model_name
            )

        # モデル変更が必要か確認
        if model_name == self.current_model:
            self.logger.info(f"Model '{model_name}' is already loaded")
            return

        # モデル変更
        old_model = self.current_model
        self.current_model = model_name
        self.config.model = model_name
        self._model_loaded = False

        self.logger.info(f"Model changed from '{old_model}' to '{model_name}'")

        # 再初期化が必要
        if self._state == ComponentState.READY:
            self._state = ComponentState.NOT_INITIALIZED

    def get_model_info(self) -> Dict[str, Any]:
        """現在のモデル情報を取得

        Returns:
            Dict[str, Any]: モデル情報
        """
        base_info = super().get_model_info()

        # FasterWhisper固有の情報を追加
        if self.current_model in SUPPORTED_MODELS:
            model_details = SUPPORTED_MODELS[self.current_model]
            base_info.update(
                {
                    "size_mb": model_details["size_mb"],
                    "parameters": model_details["parameters"],
                    "compute_type": self.config.compute_type,
                    "beam_size": DEFAULT_BEAM_SIZE,
                    "vad_filter": DEFAULT_VAD_FILTER,
                }
            )

        return base_info

    async def is_available(self) -> bool:
        """エンジンが利用可能かチェック

        Returns:
            bool: 利用可能な場合True
        """
        return (
            FASTER_WHISPER_AVAILABLE
            and ComponentState.is_operational(self._state)
            and self._model_loaded  # 修正: クラスメソッドとして呼び出し
            and self.model is not None
        )

    @property
    def state(self) -> ComponentState:
        """現在の状態を取得

        Returns:
            ComponentState: 現在の状態
        """
        return self._state


# ============================================================================
# ヘルパー関数
# ============================================================================


def create_faster_whisper_engine(config: Optional[Dict[str, Any]] = None) -> FasterWhisperEngine:
    """FasterWhisperEngineのファクトリ関数

    Args:
        config: 設定辞書

    Returns:
        FasterWhisperEngine: エンジンインスタンス
    """
    stt_config = STTConfig.from_dict(config) if config else STTConfig()
    return FasterWhisperEngine(stt_config)


# ============================================================================
# エクスポート定義
# ============================================================================

__all__ = [
    "FasterWhisperEngine",
    "create_faster_whisper_engine",
    "SUPPORTED_MODELS",
    "FASTER_WHISPER_AVAILABLE",
]
