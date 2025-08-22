"""対話システム設定モジュール

DialogueConfigクラスを定義し、対話システムの設定を管理する。
Phase 2の対話システム構築で使用。

設定ファイル完全仕様書 v1.2準拠
DialogueManager統合ガイド v1.2準拠
開発規約書 v1.12準拠
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


class ResponseMode(Enum):
    """応答生成モード

    Phase 2では最小実装としてSIMPLEのみ実装。
    将来的にCREATIVE、BALANCEDを追加予定。
    """

    SIMPLE = "simple"  # シンプルな応答
    CREATIVE = "creative"  # 創造的な応答（Phase 3以降）
    BALANCED = "balanced"  # バランス型（Phase 3以降）


@dataclass
class DialogueConfig:
    """対話システムの設定クラス

    DialogueManagerで使用される設定を管理。
    Phase 2では最小限の設定項目のみ実装。

    設定ファイル完全仕様書 v1.2準拠
    DialogueManager統合ガイド v1.2準拠

    Attributes:
        max_turns: 最大対話ターン数（デフォルト: 100）
        turn_timeout: 1ターンのタイムアウト時間（秒）
        response_mode: 応答生成モード
        temperature: 応答のランダム性（0.0-2.0）
        max_response_length: 最大応答文字数
        enable_memory: 記憶システムの有効化（Phase 7以降）
        enable_emotion: 感情分析の有効化（Phase 7以降）
        default_character_id: デフォルトキャラクターID
        language: 対話言語
        debug_mode: デバッグモードの有効化
        log_conversation: 対話ログの保存
        use_mock_engines: モックエンジンの使用（Phase 2ではTrue）
        auto_save_interval: 自動保存間隔（秒）
        metadata: 追加のメタデータ

    Example:
        >>> config = DialogueConfig()
        >>> print(config.max_turns)
        100
        >>> config = DialogueConfig(debug_mode=True, temperature=0.8)
        >>> print(config.temperature)
        0.8
    """

    # 基本的な対話設定
    max_turns: int = 100
    turn_timeout: float = 30.0
    response_mode: ResponseMode = ResponseMode.SIMPLE
    temperature: float = 0.7
    max_response_length: int = 500

    # パーソナライゼーション設定（Phase 7以降）
    enable_memory: bool = False
    enable_emotion: bool = False
    default_character_id: str = "001_aoi"

    # 言語設定
    language: str = "ja"

    # デバッグとログ設定
    debug_mode: bool = False
    log_conversation: bool = True

    # Phase 2特有の設定
    use_mock_engines: bool = True  # Phase 2ではモックエンジンを使用

    # システム設定
    auto_save_interval: int = 300  # 5分ごとに自動保存

    # 拡張用メタデータ
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """データクラス初期化後の検証

        設定値の妥当性をチェックする。

        Raises:
            ValueError: 不正な設定値が指定された場合
        """
        # max_turnsの検証
        if self.max_turns < 1:
            raise ValueError(f"max_turns must be >= 1, got {self.max_turns}")

        # turn_timeoutの検証
        if self.turn_timeout <= 0:
            raise ValueError(f"turn_timeout must be > 0, got {self.turn_timeout}")

        # temperatureの検証
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be between 0.0 and 2.0, got {self.temperature}")

        # max_response_lengthの検証
        if self.max_response_length < 1:
            raise ValueError(f"max_response_length must be >= 1, got {self.max_response_length}")

        # languageの検証（Phase 2では日本語と英語のみサポート）
        if self.language not in ["ja", "en"]:
            raise ValueError(f"language must be 'ja' or 'en', got {self.language}")

        # auto_save_intervalの検証
        if self.auto_save_interval < 0:
            raise ValueError(f"auto_save_interval must be >= 0, got {self.auto_save_interval}")

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換

        設定を辞書形式に変換する。
        ログ出力やファイル保存用。

        Returns:
            Dict[str, Any]: 設定の辞書表現

        Example:
            >>> config = DialogueConfig(debug_mode=True)
            >>> data = config.to_dict()
            >>> print(data["debug_mode"])
            True
        """
        return {
            "max_turns": self.max_turns,
            "turn_timeout": self.turn_timeout,
            "response_mode": self.response_mode.value,
            "temperature": self.temperature,
            "max_response_length": self.max_response_length,
            "enable_memory": self.enable_memory,
            "enable_emotion": self.enable_emotion,
            "default_character_id": self.default_character_id,
            "language": self.language,
            "debug_mode": self.debug_mode,
            "log_conversation": self.log_conversation,
            "use_mock_engines": self.use_mock_engines,
            "auto_save_interval": self.auto_save_interval,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DialogueConfig":
        """辞書形式から生成

        辞書形式のデータからDialogueConfigインスタンスを生成する。
        設定ファイルからの読み込み用。

        Args:
            data: 設定データの辞書

        Returns:
            DialogueConfig: 生成されたインスタンス

        Raises:
            KeyError: 必須フィールドが不足している場合
            ValueError: 不正な値が含まれている場合

        Example:
            >>> data = {"max_turns": 50, "debug_mode": True}
            >>> config = DialogueConfig.from_dict(data)
            >>> print(config.max_turns)
            50
        """
        # ResponseModeの変換
        if "response_mode" in data and isinstance(data["response_mode"], str):
            try:
                data["response_mode"] = ResponseMode(data["response_mode"])
            except ValueError:
                # 未知のモードの場合はSIMPLEにフォールバック
                data["response_mode"] = ResponseMode.SIMPLE

        # 既知のフィールドのみを抽出
        known_fields = {
            "max_turns",
            "turn_timeout",
            "response_mode",
            "temperature",
            "max_response_length",
            "enable_memory",
            "enable_emotion",
            "default_character_id",
            "language",
            "debug_mode",
            "log_conversation",
            "use_mock_engines",
            "auto_save_interval",
            "metadata",
        }

        # 既知のフィールドのみを使用してインスタンス化
        filtered_data = {k: v for k, v in data.items() if k in known_fields}

        return cls(**filtered_data)

    def validate(self) -> List[str]:
        """設定の妥当性を検証

        すべての設定項目の妥当性をチェックし、
        問題がある場合は警告メッセージのリストを返す。

        Returns:
            List[str]: 警告メッセージのリスト（問題がない場合は空リスト）

        Example:
            >>> config = DialogueConfig(max_turns=1000)
            >>> warnings = config.validate()
            >>> if warnings:
            ...     for warning in warnings:
            ...         print(f"Warning: {warning}")
        """
        warnings = []

        # max_turnsが多すぎる場合の警告
        if self.max_turns > 1000:
            warnings.append(
                f"max_turns ({self.max_turns}) is very high, " "may cause memory issues"
            )

        # turn_timeoutが長すぎる場合の警告
        if self.turn_timeout > 300:
            warnings.append(
                f"turn_timeout ({self.turn_timeout}s) is very long, "
                "may cause poor user experience"
            )

        # temperatureが極端な場合の警告
        if self.temperature < 0.1:
            warnings.append(
                f"temperature ({self.temperature}) is very low, "
                "responses may be too deterministic"
            )
        elif self.temperature > 1.5:
            warnings.append(
                f"temperature ({self.temperature}) is very high, " "responses may be too random"
            )

        # Phase 2での機能制限の警告
        if self.enable_memory:
            warnings.append("enable_memory is True but memory system is not implemented in Phase 2")

        if self.enable_emotion:
            warnings.append(
                "enable_emotion is True but emotion analysis is not implemented in Phase 2"
            )

        if not self.use_mock_engines:
            warnings.append(
                "use_mock_engines is False but real engines are not implemented in Phase 2"
            )

        return warnings

    def get_summary(self) -> str:
        """設定の概要を取得

        主要な設定項目の概要を文字列で返す。
        ログ出力やデバッグ用。

        Returns:
            str: 設定の概要

        Example:
            >>> config = DialogueConfig(debug_mode=True)
            >>> print(config.get_summary())
            DialogueConfig: max_turns=100, timeout=30.0s, mode=simple, ...
        """
        return (
            f"DialogueConfig: "
            f"max_turns={self.max_turns}, "
            f"timeout={self.turn_timeout}s, "
            f"mode={self.response_mode.value}, "
            f"temp={self.temperature}, "
            f"lang={self.language}, "
            f"debug={self.debug_mode}, "
            f"mock={self.use_mock_engines}"
        )

    def copy_with(self, **kwargs) -> "DialogueConfig":
        """部分的に設定を変更したコピーを作成

        指定された項目のみを変更した新しいインスタンスを作成する。

        Args:
            **kwargs: 変更したい設定項目と値

        Returns:
            DialogueConfig: 変更を適用した新しいインスタンス

        Example:
            >>> config = DialogueConfig()
            >>> new_config = config.copy_with(debug_mode=True, max_turns=50)
            >>> print(new_config.debug_mode)
            True
            >>> print(new_config.max_turns)
            50
        """
        # 現在の設定を辞書化
        current_data = self.to_dict()

        # ResponseModeの処理
        if "response_mode" in kwargs and isinstance(kwargs["response_mode"], ResponseMode):
            kwargs["response_mode"] = kwargs["response_mode"].value

        # 変更を適用
        current_data.update(kwargs)

        # 新しいインスタンスを作成
        return DialogueConfig.from_dict(current_data)

    def __str__(self) -> str:
        """文字列表現

        Returns:
            str: 設定の概要
        """
        return self.get_summary()

    def __repr__(self) -> str:
        """開発用詳細文字列表現

        Returns:
            str: デバッグ用の詳細情報
        """
        return (
            f"DialogueConfig("
            f"max_turns={self.max_turns}, "
            f"turn_timeout={self.turn_timeout}, "
            f"response_mode={self.response_mode}, "
            f"temperature={self.temperature}, "
            f"language='{self.language}', "
            f"debug_mode={self.debug_mode})"
        )
