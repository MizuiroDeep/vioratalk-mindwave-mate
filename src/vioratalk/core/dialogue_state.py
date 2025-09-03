"""対話状態管理モジュール

DialogueTurnデータクラスを定義し、対話の1ターンを表現する。
Phase 2の対話システム構築で使用。
Phase 4拡張: audio_responseフィールド追加（音声入出力対応）

インターフェース定義書 v1.34準拠
データフォーマット仕様書 v1.5準拠
開発規約書 v1.12準拠
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class DialogueTurn:
    """対話の1ターンを表すデータクラス

    ユーザー入力とアシスタント応答のペアを保持。
    Phase 2以降の対話管理で使用。
    Phase 4拡張: 音声応答データ（audio_response）を追加。

    インターフェース定義書 v1.34 セクション2.4準拠

    Attributes:
        user_input: ユーザーからの入力テキスト
        assistant_response: アシスタントの応答テキスト
        timestamp: このターンの開始時刻
        turn_number: 会話内でのターン番号（1から開始）
        character_id: 使用されたキャラクターのID
        emotion: 応答時の感情（オプション）
        confidence: 応答の確信度（0.0-1.0、オプション）
        processing_time: 処理時間（秒、オプション）
        audio_response: 音声応答データ（Phase 4追加、オプション）
        metadata: 追加のメタデータ（オプション）

    Example:
        >>> from datetime import datetime
        >>> turn = DialogueTurn(
        ...     user_input="こんにちは",
        ...     assistant_response="こんにちは！今日はどんなお話をしましょうか？",
        ...     timestamp=datetime.now(),
        ...     turn_number=1,
        ...     character_id="001_aoi",
        ...     emotion="happy",
        ...     confidence=0.95,
        ...     audio_response=b"audio_data"  # Phase 4
        ... )
        >>> data = turn.to_dict()
        >>> print(data["turn_number"])
        1
    """

    # 必須フィールド
    user_input: str
    assistant_response: str
    timestamp: datetime
    turn_number: int
    character_id: str

    # オプションフィールド（Phase 2）
    emotion: Optional[str] = None
    confidence: Optional[float] = None
    processing_time: Optional[float] = None

    # Phase 4追加フィールド
    audio_response: Optional[bytes] = None  # 音声応答データ

    # メタデータ
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """データクラス初期化後の検証

        入力値の妥当性をチェックする。

        Raises:
            ValueError: 不正な値が設定された場合
        """
        # ターン番号の検証
        if self.turn_number < 1:
            raise ValueError(f"turn_number must be >= 1, got {self.turn_number}")

        # 確信度の検証
        if self.confidence is not None:
            if not 0.0 <= self.confidence <= 1.0:
                raise ValueError(f"confidence must be between 0.0 and 1.0, got {self.confidence}")

        # 処理時間の検証
        if self.processing_time is not None:
            if self.processing_time < 0:
                raise ValueError(f"processing_time must be >= 0, got {self.processing_time}")

        # キャラクターIDの検証（空文字列を許可しない）
        if not self.character_id or not self.character_id.strip():
            raise ValueError("character_id cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換

        ログ出力やAPI通信用に辞書形式に変換する。
        timestampはISO形式の文字列に変換される。
        audio_responseはPhase 4で追加されたフィールド。

        Returns:
            Dict[str, Any]: DialogueTurnの全フィールドを含む辞書

        Example:
            >>> turn = DialogueTurn(...)
            >>> data = turn.to_dict()
            >>> print(data.keys())
            dict_keys(['user_input', 'assistant_response', 'timestamp', ...])
        """
        return {
            "user_input": self.user_input,
            "assistant_response": self.assistant_response,
            "timestamp": self.timestamp.isoformat(),
            "turn_number": self.turn_number,
            "character_id": self.character_id,
            "emotion": self.emotion,
            "confidence": self.confidence,
            "processing_time": self.processing_time,
            "audio_response": self.audio_response,  # Phase 4追加
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DialogueTurn":
        """辞書形式から生成

        to_dict()で出力した辞書からDialogueTurnインスタンスを復元する。
        Phase 4でaudio_responseフィールドに対応。

        Args:
            data: DialogueTurnの辞書表現

        Returns:
            DialogueTurn: 復元されたインスタンス

        Raises:
            KeyError: 必須フィールドが不足している場合
            ValueError: フィールドの値が不正な場合

        Example:
            >>> data = {"user_input": "hello", ...}
            >>> turn = DialogueTurn.from_dict(data)
        """
        # timestampの文字列をdatetimeに変換
        if isinstance(data.get("timestamp"), str):
            timestamp = datetime.fromisoformat(data["timestamp"])
        else:
            timestamp = data["timestamp"]

        return cls(
            user_input=data["user_input"],
            assistant_response=data["assistant_response"],
            timestamp=timestamp,
            turn_number=data["turn_number"],
            character_id=data["character_id"],
            emotion=data.get("emotion"),
            confidence=data.get("confidence"),
            processing_time=data.get("processing_time"),
            audio_response=data.get("audio_response"),  # Phase 4追加
            metadata=data.get("metadata", {}),
        )

    def get_summary(self) -> str:
        """ターンの概要を取得

        ログ出力用の簡潔な文字列表現を返す。
        音声データの有無もPhase 4で追加表示。

        Returns:
            str: ターンの概要文字列

        Example:
            >>> turn = DialogueTurn(...)
            >>> print(turn.get_summary())
            Turn #1 [001_aoi]: "こんにちは" -> "こんにちは！今日は..." [audio:yes]
        """
        # 長いテキストは省略
        user_preview = (
            self.user_input[:30] + "..." if len(self.user_input) > 30 else self.user_input
        )
        response_preview = (
            self.assistant_response[:30] + "..."
            if len(self.assistant_response) > 30
            else self.assistant_response
        )

        # Phase 4: 音声データの有無を表示
        audio_status = " [audio:yes]" if self.audio_response else ""

        return (
            f"Turn #{self.turn_number} [{self.character_id}]: "
            f'"{user_preview}" -> "{response_preview}"{audio_status}'
        )

    def __str__(self) -> str:
        """文字列表現

        Returns:
            str: DialogueTurnの簡潔な文字列表現
        """
        return self.get_summary()

    def __repr__(self) -> str:
        """開発用詳細文字列表現

        音声データは大きいため、有無のみ表示。

        Returns:
            str: デバッグ用の詳細情報
        """
        audio_info = ", has_audio=True" if self.audio_response else ""

        return (
            f"DialogueTurn(turn_number={self.turn_number}, "
            f"character_id='{self.character_id}', "
            f"timestamp={self.timestamp.isoformat()}, "
            f"emotion={self.emotion}, "
            f"confidence={self.confidence}"
            f"{audio_info})"
        )
