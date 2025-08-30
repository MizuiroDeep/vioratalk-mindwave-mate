"""
VioraTalk プログレスバー表示ユーティリティ

アプリケーション全体で統一的なプログレスバー表示を提供。
ダウンロード、処理進捗など様々な場面で使用可能。

開発規約書 v1.12準拠
"""

from datetime import datetime, timedelta
from typing import Callable, Optional


class ProgressBar:
    """統一的なプログレスバー表示クラス

    コンソールに表示するプログレスバーを管理。
    ModelDownloadManager、各種エンジンなどで共通使用。

    使用例:
        # 基本的な使用
        progress = ProgressBar(prefix="Downloading")
        for i in range(100):
            progress.update(i / 100)
        progress.close()

        # コールバックとして使用
        progress = ProgressBar(prefix="Processing")
        await some_async_function(progress_callback=progress.create_callback())

    Attributes:
        prefix: プログレスバーの前に表示するテキスト
        bar_length: バーの長さ（文字数）
        filled_char: 完了部分の文字
        empty_char: 未完了部分の文字
        show_time: 経過時間と残り時間を表示するか
        min_update_interval: 最小更新間隔（秒）
    """

    def __init__(
        self,
        prefix: str = "Progress",
        suffix: str = "",
        bar_length: int = 40,
        filled_char: str = "█",
        empty_char: str = "░",
        show_time: bool = False,
        min_update_interval: float = 0.1,
    ):
        """初期化

        Args:
            prefix: プログレスバーの前に表示するテキスト
            suffix: プログレスバーの後に表示するテキスト（ファイル名など）
            bar_length: バーの長さ（文字数）
            filled_char: 完了部分の文字
            empty_char: 未完了部分の文字
            show_time: 経過時間と残り時間を表示
            min_update_interval: 最小更新間隔（秒）
        """
        self.prefix = prefix
        self.suffix = suffix
        self.bar_length = bar_length
        self.filled_char = filled_char
        self.empty_char = empty_char
        self.show_time = show_time
        self.min_update_interval = min_update_interval

        # 内部状態
        self._last_progress = -1
        self._last_update_time = None
        self._start_time = None
        self._completed = False

    def update(self, progress: float, suffix: Optional[str] = None) -> None:
        """プログレスバー更新

        Args:
            progress: 進捗率（0.0 ～ 1.0）
            suffix: 一時的な suffix（ファイル名など）
        """
        # 完了済みなら何もしない
        if self._completed:
            return

        # 範囲制限
        progress = max(0.0, min(1.0, progress))

        # 更新頻度の制限（パフォーマンス対策）
        current_time = datetime.now()
        if self._last_update_time:
            if (current_time - self._last_update_time).total_seconds() < self.min_update_interval:
                # ただし100%の時は必ず更新
                if progress < 1.0:
                    return

        # 初回実行時
        if self._start_time is None:
            self._start_time = current_time

        self._last_progress = progress
        self._last_update_time = current_time

        # プログレスバー生成
        filled = int(self.bar_length * progress)
        bar = self.filled_char * filled + self.empty_char * (self.bar_length - filled)

        # 表示テキスト構築
        display_suffix = suffix if suffix is not None else self.suffix
        display_parts = [self.prefix]

        if display_suffix:
            display_parts[0] = f"{self.prefix} {display_suffix}"

        display_parts.append(f": [{bar}] {progress*100:.1f}%")

        # 時間表示（オプション）
        if self.show_time and self._start_time:
            elapsed = current_time - self._start_time
            display_parts.append(f" [{self._format_time(elapsed)}")

            # 残り時間推定（進捗が0より大きい場合）
            if progress > 0 and progress < 1.0:
                total_estimated = elapsed.total_seconds() / progress
                remaining = total_estimated - elapsed.total_seconds()
                if remaining > 0:
                    display_parts.append(
                        f" / ETA: {self._format_time(timedelta(seconds=remaining))}"
                    )

            display_parts.append("]")

        # 表示
        display_text = "".join(display_parts)

        # 同じ行で更新（\r で行頭に戻る）
        print(f"\r{display_text}", end="", flush=True)

        # 完了時の処理
        if progress >= 1.0:
            self._completed = True
            print()  # 改行

    def _format_time(self, td: timedelta) -> str:
        """時間をフォーマット

        Args:
            td: timedelta オブジェクト

        Returns:
            str: フォーマット済み時間（例: "1:23", "45s"）
        """
        total_seconds = int(td.total_seconds())

        if total_seconds < 60:
            return f"{total_seconds}s"
        elif total_seconds < 3600:
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            return f"{minutes}:{seconds:02d}"
        else:
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            return f"{hours}:{minutes:02d}:{seconds:02d}"

    def create_callback(self, suffix: Optional[str] = None) -> Callable[[float], None]:
        """コールバック関数を生成

        ModelDownloadManagerなどのprogress_callbackに渡せる関数を生成。

        Args:
            suffix: 固定のsuffix

        Returns:
            Callable[[float], None]: プログレスコールバック関数
        """

        def callback(progress: float):
            self.update(progress, suffix)

        return callback

    def reset(self):
        """プログレスバーをリセット

        同じインスタンスを再利用する場合に使用。
        """
        self._last_progress = -1
        self._last_update_time = None
        self._start_time = None
        self._completed = False

    def close(self, message: Optional[str] = None):
        """プログレスバーをクローズする

        Python標準の慣習に従い、close()メソッドを提供。
        with文での自動クリーンアップにも対応可能。

        Args:
            message: 完了メッセージ（Noneの場合は通常の100%表示）
        """
        if message:
            # カスタムメッセージで完了
            print(f"\r{self.prefix}: {message}")
        else:
            # 100%で完了
            self.update(1.0)

        self._completed = True


class MultiProgressBar:
    """複数のプログレスバーを管理するクラス

    複数のダウンロードや処理を同時に表示する場合に使用。

    使用例:
        multi_progress = MultiProgressBar()

        # タスク1
        task1 = multi_progress.add_task("Download Model A")
        await download_model_a(progress_callback=task1.create_callback())

        # タスク2（並行実行）
        task2 = multi_progress.add_task("Download Model B")
        await download_model_b(progress_callback=task2.create_callback())
    """

    def __init__(self):
        """初期化"""
        self.tasks = {}
        self._next_id = 0

    def add_task(self, prefix: str = "Task", **kwargs) -> ProgressBar:
        """タスクを追加

        Args:
            prefix: タスクのプレフィックス
            **kwargs: ProgressBarに渡す追加引数

        Returns:
            ProgressBar: 作成されたプログレスバー
        """
        task_id = self._next_id
        self._next_id += 1

        progress_bar = ProgressBar(prefix=f"[{task_id}] {prefix}", **kwargs)
        self.tasks[task_id] = progress_bar

        return progress_bar

    def remove_task(self, task_id: int):
        """タスクを削除

        Args:
            task_id: タスクID
        """
        if task_id in self.tasks:
            del self.tasks[task_id]

    def clear(self):
        """すべてのタスクをクリア"""
        self.tasks.clear()
        self._next_id = 0


def create_simple_progress(prefix: str = "Progress") -> Callable[[float], None]:
    """シンプルなプログレスコールバックを作成

    ワンライナーでプログレスバーを使いたい場合に便利。

    使用例:
        await download_something(
            progress_callback=create_simple_progress("Downloading")
        )

    Args:
        prefix: プレフィックス

    Returns:
        Callable[[float], None]: プログレスコールバック関数
    """
    progress_bar = ProgressBar(prefix=prefix)
    return progress_bar.create_callback()


# エクスポート定義
__all__ = ["ProgressBar", "MultiProgressBar", "create_simple_progress"]
