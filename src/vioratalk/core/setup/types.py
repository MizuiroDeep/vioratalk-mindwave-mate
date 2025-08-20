"""
セットアップ関連の型定義

自動セットアップの結果と状態を表現する型を定義する。
SetupResult型による詳細な状態管理により、部分的成功を適切に処理可能。

関連ドキュメント:
    - エンジン初期化仕様書 v1.4
    - 自動セットアップガイド v1.2
    - エラーハンドリング指針 v1.20
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class SetupStatus(Enum):
    """
    セットアップの状態を表すEnum

    セットアップの結果を詳細に表現し、
    部分的成功や機能制限モードの判定に使用する。
    """

    SUCCESS = "success"  # 完全成功
    PARTIAL_SUCCESS = "partial_success"  # 部分的成功（一部コンポーネント失敗）
    SKIPPED = "skipped"  # ユーザーによるスキップ
    FAILED = "failed"  # 失敗（継続不可）
    NOT_STARTED = "not_started"  # 未実行


@dataclass
class ComponentStatus:
    """
    個々のコンポーネントのセットアップ状態

    Attributes:
        name: コンポーネント名（例: "Ollama", "AivisSpeech"）
        installed: インストール成功フラグ
        version: インストールされたバージョン
        error: エラーメッセージ（失敗時のみ）
        install_path: インストール先パス
        metadata: その他のメタデータ
    """

    name: str
    installed: bool
    version: Optional[str] = None
    error: Optional[str] = None
    install_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """バリデーション"""
        if self.installed and not self.version:
            # インストール成功時はバージョン必須
            self.version = "unknown"
        if not self.installed and not self.error:
            # インストール失敗時はエラーメッセージ必須
            self.error = "Unknown installation error"


@dataclass
class SetupResult:
    """
    自動セットアップの詳細な結果

    SetupResult型により、部分的成功や機能制限モードを
    適切に管理可能。VioraTalkEngineが結果に応じて
    動作モードを決定する。

    Attributes:
        status: セットアップ全体の状態
        components: 各コンポーネントの状態リスト
        warnings: 警告メッセージのリスト
        errors: エラーメッセージのリスト
        can_continue: 継続可能かどうか
        timestamp: セットアップ実行時刻
        duration_seconds: 実行時間（秒）
    """

    status: SetupStatus = SetupStatus.NOT_STARTED
    components: List[ComponentStatus] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    can_continue: bool = True
    timestamp: Optional[datetime] = None
    duration_seconds: Optional[float] = None

    def add_component(self, component: ComponentStatus) -> None:
        """
        コンポーネントの状態を追加

        Args:
            component: 追加するコンポーネントの状態
        """
        self.components.append(component)

        # 失敗したコンポーネントがある場合は警告を追加
        if not component.installed and component.error:
            self.add_warning(f"{component.name}: {component.error}")

    def add_warning(self, message: str) -> None:
        """
        警告メッセージを追加

        Args:
            message: 警告メッセージ
        """
        if message not in self.warnings:
            self.warnings.append(message)

    def add_error(self, message: str) -> None:
        """
        エラーメッセージを追加

        Args:
            message: エラーメッセージ
        """
        if message not in self.errors:
            self.errors.append(message)

    @property
    def is_success(self) -> bool:
        """完全成功または部分的成功かどうか"""
        return self.status in [SetupStatus.SUCCESS, SetupStatus.PARTIAL_SUCCESS]

    @property
    def has_failures(self) -> bool:
        """失敗したコンポーネントがあるかどうか"""
        return any(not c.installed for c in self.components)

    @property
    def success_rate(self) -> float:
        """
        成功率を計算（0.0～1.0）

        Returns:
            成功したコンポーネントの割合
        """
        if not self.components:
            return 0.0

        success_count = sum(1 for c in self.components if c.installed)
        return success_count / len(self.components)

    def get_installed_components(self) -> List[ComponentStatus]:
        """
        インストール成功したコンポーネントのリストを取得

        Returns:
            成功したコンポーネントのリスト
        """
        return [c for c in self.components if c.installed]

    def get_failed_components(self) -> List[ComponentStatus]:
        """
        インストール失敗したコンポーネントのリストを取得

        Returns:
            失敗したコンポーネントのリスト
        """
        return [c for c in self.components if not c.installed]

    def to_dict(self) -> Dict[str, Any]:
        """
        辞書形式に変換（ログやファイル保存用）

        Returns:
            SetupResultの辞書表現
        """
        return {
            "status": self.status.value,
            "components": [
                {
                    "name": c.name,
                    "installed": c.installed,
                    "version": c.version,
                    "error": c.error,
                    "install_path": c.install_path,
                    "metadata": c.metadata,
                }
                for c in self.components
            ],
            "warnings": self.warnings,
            "errors": self.errors,
            "can_continue": self.can_continue,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "duration_seconds": self.duration_seconds,
            "success_rate": self.success_rate,
        }


# Phase 1での利用例
if __name__ == "__main__":
    # セットアップ結果の作成例
    result = SetupResult(status=SetupStatus.PARTIAL_SUCCESS)

    # 成功したコンポーネント
    result.add_component(
        ComponentStatus(
            name="Python", installed=True, version="3.11.9", install_path="C:\\Python311"
        )
    )

    # 失敗したコンポーネント
    result.add_component(
        ComponentStatus(name="Ollama", installed=False, error="Network connection failed")
    )

    # 成功したコンポーネント
    result.add_component(
        ComponentStatus(
            name="FasterWhisper",
            installed=True,
            version="0.10.0",
            install_path="models/faster-whisper",
        )
    )

    # 結果の確認
    print(f"Status: {result.status.value}")
    print(f"Success rate: {result.success_rate:.1%}")
    print(f"Can continue: {result.can_continue}")
    print(f"Warnings: {result.warnings}")
    print(f"Installed: {[c.name for c in result.get_installed_components()]}")
    print(f"Failed: {[c.name for c in result.get_failed_components()]}")
