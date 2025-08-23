"""MockSTTEngine実装

音声認識エンジンのモック実装。
テスト用に固定値やカスタムレスポンスを返す。

インターフェース定義書 v1.34準拠
テストデータ・モック完全仕様書 v1.1準拠
エラーハンドリング指針 v1.20準拠
開発規約書 v1.12準拠
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

# プロジェクト内インポート（絶対インポート）
from vioratalk.core.base import ComponentState, VioraTalkComponent
from vioratalk.core.exceptions import AudioError, ModelNotFoundError, STTError


@dataclass
class TranscriptionResult:
    """音声認識結果

    データフォーマット仕様書 v1.5準拠
    """

    text: str
    confidence: float
    language: str
    duration: float
    alternatives: List[Dict[str, float]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AudioData:
    """音声データ（簡易版）

    実際のAudioDataクラスがPhase 3で実装されるまでの代替
    """

    data: bytes
    sample_rate: int = 16000
    duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class MockSTTEngine(VioraTalkComponent):
    """音声認識エンジンのモック実装

    Phase 3のテスト用に固定値を返すSTTエンジン。
    エラーモードやカスタムレスポンス設定機能を提供。

    Attributes:
        transcription_delay: 音声認識の遅延シミュレーション（秒）
        error_mode: エラーモードの有効/無効
        responses: ファイル名とレスポンステキストのマッピング
        supported_languages: サポートする言語のリスト
        current_model: 現在使用中のモデル名
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初期化

        Args:
            config: 設定辞書（オプション）
        """
        # VioraTalkComponentは引数を受け取らない
        super().__init__()

        # configはインスタンス変数として保持
        self.config = config or {}

        # 設定の初期化
        self.transcription_delay = self.config.get("delay", 0.1)
        self.error_mode = False
        self.current_model = self.config.get("model", "base")

        # サポート言語
        self.supported_languages = ["ja", "en", "zh", "ko"]

        # デフォルトレスポンス（テストデータ・モック完全仕様書 v1.1準拠）
        self.responses: Dict[str, str] = {
            "default": "これはテスト用の音声認識結果です。",
            "greeting.wav": "こんにちは、今日はいい天気ですね。",
            "question.wav": "何かお困りのことはありますか？",
            "command.wav": "音楽を再生してください。",
            "error.wav": "すみません、もう一度お願いします。",
        }

        # 統計情報
        self.transcription_count = 0
        self.total_duration = 0.0
        self.error_count = 0

        # 利用可能なモデル
        self.available_models = ["tiny", "base", "small", "medium", "large"]

    async def initialize(self) -> None:
        """非同期初期化処理

        Phase 3準拠：MockエンジンとしてSTT機能をシミュレート
        """
        self._state = ComponentState.INITIALIZING

        # 初期化処理のシミュレーション
        await asyncio.sleep(0.1)

        # 初期化完了
        self._state = ComponentState.READY
        self._initialized_at = datetime.now()

    async def cleanup(self) -> None:
        """リソースのクリーンアップ

        使用したリソースを解放する
        """
        self._state = ComponentState.TERMINATING

        # クリーンアップ処理のシミュレーション
        await asyncio.sleep(0.05)

        # クリーンアップ完了
        self._state = ComponentState.TERMINATED

    def is_available(self) -> bool:
        """利用可能状態の確認

        Returns:
            bool: READYまたはRUNNING状態の場合True
        """
        return self._state in [ComponentState.READY, ComponentState.RUNNING]

    def get_status(self) -> Dict[str, Any]:
        """コンポーネントの状態を取得

        Returns:
            Dict[str, Any]: 状態情報を含む辞書
        """
        return {
            "state": self._state.value,
            "is_available": self.is_available(),
            "error": self._error_info if hasattr(self, "_error_info") else None,
            "last_used": getattr(self, "_last_used", None),
            "model": self.current_model,
            "transcription_count": self.transcription_count,
            "error_count": self.error_count,
            "supported_languages": self.supported_languages,
        }

    async def transcribe(
        self, audio_data: AudioData, language: Optional[str] = None
    ) -> TranscriptionResult:
        """音声データをテキストに変換（モック実装）

        Args:
            audio_data: 音声データ
            language: 言語コード（オプション、Noneの場合は自動判定）

        Returns:
            TranscriptionResult: 音声認識結果

        Raises:
            STTError: エンジンが準備できていない場合
            AudioError: エラーモードが有効な場合
        """
        # 状態チェック
        if self._state != ComponentState.READY:
            raise STTError("STT engine is not ready", error_code="E1000")

        # エラーモードのチェック
        if self.error_mode:
            self.error_count += 1
            raise AudioError("Mock audio processing error", error_code="E1001")

        # 状態を更新（並行実行対応のため、変更しない）
        # self._state = ComponentState.RUNNING
        self._last_used = datetime.now()

        # 遅延のシミュレーション
        await asyncio.sleep(self.transcription_delay)

        # ファイル名からレスポンステキストを決定
        filename = audio_data.metadata.get("filename", "default")

        # ファイル名が見つからない場合はdefaultを使用
        if filename not in self.responses:
            filename = "default"

        text = self.responses[filename]

        # 言語を判定（引数優先、なければ自動判定）
        if language and language in self.supported_languages:
            result_language = language
        elif language and language not in self.supported_languages:
            # サポートされていない言語はjaにフォールバック
            result_language = "ja"
        elif any(ord(c) > 0x3000 for c in text):
            result_language = "ja"
        else:
            result_language = "en"

        # 代替候補を生成（10文字以上の場合）
        alternatives = []
        if len(text) > 10:
            alternatives = [
                {"text": text[: len(text) // 2] + "...", "confidence": 0.7},
                {"text": text + "か？", "confidence": 0.6},
            ]

        # 統計情報を更新
        self.transcription_count += 1
        self.total_duration += audio_data.duration if audio_data.duration > 0 else 3.0

        # 状態を戻す（変更しないので不要）
        # self._state = ComponentState.READY

        return TranscriptionResult(
            text=text,
            confidence=0.95,
            language=result_language,
            duration=audio_data.duration if audio_data.duration > 0 else 3.0,
            alternatives=alternatives,
        )

    def get_supported_languages(self) -> List[str]:
        """サポートする言語のリストを取得

        Returns:
            List[str]: 言語コードのリスト
        """
        return self.supported_languages.copy()

    def set_model(self, model_name: str) -> None:
        """使用するモデルを設定

        Args:
            model_name: モデル名

        Raises:
            ModelNotFoundError: 指定されたモデルが存在しない場合
        """
        if model_name not in self.available_models:
            raise ModelNotFoundError(
                f"Model '{model_name}' not found", error_code="E2100", model_name=model_name
            )

        self.current_model = model_name

    # ============================================================================
    # テスト用メソッド
    # ============================================================================

    def set_error_mode(self, enabled: bool) -> None:
        """エラーモードを設定（テスト用）

        Args:
            enabled: エラーモードの有効/無効
        """
        self.error_mode = enabled

    def set_custom_response(self, filename: str, text: str) -> None:
        """カスタムレスポンスを設定（テスト用）

        Args:
            filename: ファイル名
            text: レスポンステキスト
        """
        self.responses[filename] = text

    def set_transcription_delay(self, delay: float) -> None:
        """音声認識の遅延を設定（テスト用）

        Args:
            delay: 遅延時間（秒）、負の値は0にクリップ
        """
        self.transcription_delay = max(0.0, delay)

    def get_statistics(self) -> Dict[str, Any]:
        """統計情報を取得（テスト用）

        Returns:
            Dict[str, Any]: 統計情報
        """
        return {
            "transcription_count": self.transcription_count,
            "total_duration": self.total_duration,
            "error_count": self.error_count,
            "average_duration": (
                self.total_duration / self.transcription_count
                if self.transcription_count > 0
                else 0
            ),
            "model": self.current_model,
            "supported_languages": self.supported_languages,
            "error_mode": self.error_mode,
            "custom_responses_count": len(self.responses),
            "transcription_delay": self.transcription_delay,
        }

    def reset_statistics(self) -> None:
        """統計情報をリセット（テスト用）"""
        self.transcription_count = 0
        self.total_duration = 0.0
        self.error_count = 0
