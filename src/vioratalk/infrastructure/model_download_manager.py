"""
VioraTalk モデルダウンロード管理

モデルファイルのダウンロード、キャッシュ管理、チェックサム検証を行う。
Phase 1では基本的なインターフェースとスタブ実装のみ提供。

インターフェース定義書 v1.33準拠
エラーハンドリング指針 v1.20準拠
開発規約書 v1.12準拠
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

from vioratalk.configuration.settings import MODELS_DIR
from vioratalk.utils.logger_manager import LoggerManager

# ロガーの取得
logger = LoggerManager.get_logger(__name__)


class ModelDownloadManager:
    """モデルダウンロード管理クラス

    モデルファイルのダウンロード、進捗管理、チェックサム検証、
    およびキャッシュ管理を統一的に行う。

    Phase 1実装内容:
    - 基本的なインターフェース定義
    - キャッシュパス管理
    - スタブ実装（実際のダウンロードは行わない）

    Phase 2以降の拡張予定:
    - httpxを使用した実際のダウンロード実装
    - 並列ダウンロード
    - レジューム機能
    - ミラーサイト対応
    - 圧縮ファイルの展開

    Attributes:
        cache_dir: モデルキャッシュディレクトリ
        _download_tasks: 実行中のダウンロードタスク
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """ModelDownloadManagerの初期化

        Args:
            cache_dir: キャッシュディレクトリ（デフォルト: data/models）
        """
        self.cache_dir = cache_dir or MODELS_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._download_tasks: List[asyncio.Task] = []

        logger.info(f"ModelDownloadManager initialized with cache_dir: {self.cache_dir}")

    async def download_with_progress(
        self,
        url: str,
        destination: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> None:
        """進捗表示付きダウンロード

        Phase 1ではスタブ実装。実際のダウンロードは行わず、
        プログレスコールバックのシミュレーションのみ行う。

        Args:
            url: ダウンロードURL
            destination: 保存先パス
            progress_callback: 進捗コールバック（0.0-1.0）

        Raises:
            ModelError: ダウンロード失敗（Phase 2以降で実装）
        """
        logger.info(f"Download requested: {url} -> {destination}")

        # Phase 1: スタブ実装
        # 進捗のシミュレーション
        if progress_callback:
            for i in range(11):  # 0%, 10%, 20%, ..., 100%
                progress = i / 10.0
                progress_callback(progress)
                await asyncio.sleep(0.1)  # シミュレーション用の遅延

        # ディレクトリが存在しない場合は作成
        destination.parent.mkdir(parents=True, exist_ok=True)

        # Phase 1では空ファイルを作成（実際のダウンロードは行わない）
        if not destination.exists():
            destination.touch()
            logger.info(f"Stub file created at: {destination}")
        else:
            logger.info(f"File already exists: {destination}")

        # Phase 2以降の実装メモ:
        # - httpx.AsyncClientを使用
        # - チャンクごとにダウンロード
        # - Content-Lengthから総サイズを取得
        # - 進捗計算とコールバック呼び出し
        # - エラーハンドリング（ネットワークエラー、タイムアウト等）

    def verify_checksum(
        self, file_path: Path, expected_checksum: str, algorithm: str = "sha256"
    ) -> bool:
        """チェックサム検証

        Phase 1では常にTrueを返すスタブ実装。
        Phase 2以降で実際のチェックサム検証を実装。

        Args:
            file_path: 検証するファイル
            expected_checksum: 期待されるチェックサム
            algorithm: ハッシュアルゴリズム（sha256/md5）

        Returns:
            bool: 検証成功ならTrue（Phase 1では常にTrue）
        """
        logger.debug(
            f"Checksum verification requested: {file_path} "
            f"(expected: {expected_checksum[:8]}...)"
        )

        # Phase 1: スタブ実装（常に成功）
        if not file_path.exists():
            logger.warning(f"File not found for checksum: {file_path}")
            return False

        # Phase 2以降の実装メモ:
        # 実際のチェックサム計算
        # if algorithm.lower() == "sha256":
        #     hash_func = hashlib.sha256()
        # elif algorithm.lower() == "md5":
        #     hash_func = hashlib.md5()
        # else:
        #     raise ValueError(f"Unsupported algorithm: {algorithm}")
        #
        # with open(file_path, 'rb') as f:
        #     for chunk in iter(lambda: f.read(8192), b''):
        #         hash_func.update(chunk)
        #
        # actual_checksum = hash_func.hexdigest()
        # return actual_checksum.lower() == expected_checksum.lower()

        logger.info(f"Checksum verification passed (stub): {file_path}")
        return True

    def get_cache_path(self, model_name: str) -> Path:
        """キャッシュパスを取得

        モデル名から適切なキャッシュディレクトリ内のパスを生成。

        Args:
            model_name: モデル名（例: "whisper-base", "aivisspeech-kana"）

        Returns:
            Path: キャッシュディレクトリ内のパス
        """
        # モデル名を正規化（特殊文字を除去）
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in model_name)

        # カテゴリごとにディレクトリを分ける
        if "whisper" in model_name.lower():
            category = "stt"
        elif "aivisspeech" in model_name.lower() or "tts" in model_name.lower():
            category = "tts"
        elif any(llm in model_name.lower() for llm in ["llama", "mistral", "phi"]):
            category = "llm"
        else:
            category = "misc"

        cache_path = self.cache_dir / category / safe_name
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Cache path for '{model_name}': {cache_path}")
        return cache_path

    async def cleanup_cache(self, max_size_gb: float = 10.0) -> None:
        """キャッシュクリーンアップ

        古いモデルファイルを削除してキャッシュサイズを制限内に収める。
        Phase 1では基本的な実装のみ。

        Args:
            max_size_gb: 最大キャッシュサイズ（GB）
        """
        logger.info(f"Cache cleanup started (max size: {max_size_gb}GB)")

        try:
            # キャッシュディレクトリ内のファイルを取得
            cache_files = self._get_cache_files()

            # 総サイズを計算
            total_size = sum(f.stat().st_size for f in cache_files)
            total_size_gb = total_size / (1024**3)

            logger.info(f"Current cache size: {total_size_gb:.2f}GB " f"({len(cache_files)} files)")

            if total_size_gb <= max_size_gb:
                logger.info("Cache size is within limit")
                return

            # アクセス時刻でソート（古い順）
            cache_files.sort(key=lambda f: f.stat().st_atime)

            # 古いファイルから削除
            deleted_size = 0
            deleted_count = 0
            target_size = int(max_size_gb * 1024**3)

            for file_path in cache_files:
                if total_size - deleted_size <= target_size:
                    break

                file_size = file_path.stat().st_size

                # Phase 1では実際には削除しない（ログのみ）
                logger.info(f"Would delete: {file_path} ({file_size / 1024 ** 2:.2f}MB)")

                # Phase 2以降では実際に削除
                # file_path.unlink()

                deleted_size += file_size
                deleted_count += 1

            logger.info(
                f"Cache cleanup completed: {deleted_count} files, "
                f"{deleted_size / 1024 ** 2:.2f}MB would be freed"
            )

        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
            # Phase 1ではエラーを無視（クリーンアップは必須ではない）

    def is_model_cached(self, model_name: str) -> bool:
        """モデルがキャッシュされているか確認

        Args:
            model_name: モデル名

        Returns:
            bool: キャッシュされている場合True
        """
        cache_path = self.get_cache_path(model_name)

        # ディレクトリが存在し、何かファイルが含まれているか確認
        if cache_path.exists():
            if cache_path.is_file():
                return True
            elif cache_path.is_dir():
                # ディレクトリ内にファイルがあるか確認
                return any(cache_path.iterdir())

        return False

    def get_model_info(self, model_name: str) -> Optional[dict]:
        """モデル情報を取得

        Phase 1では基本情報のみ返す。
        Phase 2以降でメタデータファイルから詳細情報を読み込む。

        Args:
            model_name: モデル名

        Returns:
            Optional[dict]: モデル情報（存在しない場合None）
        """
        cache_path = self.get_cache_path(model_name)

        if not self.is_model_cached(model_name):
            return None

        # Phase 1: 基本情報のみ
        info = {
            "name": model_name,
            "path": str(cache_path),
            "cached": True,
            "size": 0,
            "last_accessed": None,
        }

        # ファイルサイズと最終アクセス時刻を取得
        try:
            if cache_path.is_file():
                stat = cache_path.stat()
                info["size"] = stat.st_size
                info["last_accessed"] = datetime.fromtimestamp(stat.st_atime)
            elif cache_path.is_dir():
                # ディレクトリの場合は合計サイズを計算
                total_size = sum(f.stat().st_size for f in cache_path.rglob("*") if f.is_file())
                info["size"] = total_size
        except Exception as e:
            logger.warning(f"Failed to get model info for {model_name}: {e}")

        return info

    async def download_if_needed(
        self,
        model_name: str,
        url: str,
        checksum: Optional[str] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Path:
        """必要に応じてモデルをダウンロード

        キャッシュに存在しない場合のみダウンロードを実行。

        Args:
            model_name: モデル名
            url: ダウンロードURL
            checksum: チェックサム（オプション）
            progress_callback: 進捗コールバック

        Returns:
            Path: モデルファイルのパス
        """
        cache_path = self.get_cache_path(model_name)

        if self.is_model_cached(model_name):
            logger.info(f"Model already cached: {model_name}")
            return cache_path

        logger.info(f"Downloading model: {model_name}")
        await self.download_with_progress(url, cache_path, progress_callback)

        if checksum:
            if not self.verify_checksum(cache_path, checksum):
                logger.warning(f"Checksum verification failed for {model_name}")
                # Phase 2以降ではModelErrorを発生させる

        return cache_path

    def _get_cache_files(self) -> List[Path]:
        """キャッシュディレクトリ内のファイルを取得

        Returns:
            List[Path]: キャッシュファイルのリスト
        """
        cache_files = []

        if self.cache_dir.exists():
            for file_path in self.cache_dir.rglob("*"):
                if file_path.is_file():
                    cache_files.append(file_path)

        return cache_files

    def get_cache_stats(self) -> dict:
        """キャッシュ統計情報を取得

        Returns:
            dict: キャッシュの統計情報
        """
        cache_files = self._get_cache_files()

        total_size = sum(f.stat().st_size for f in cache_files)

        stats = {
            "cache_dir": str(self.cache_dir),
            "total_files": len(cache_files),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024**2),
            "total_size_gb": total_size / (1024**3),
            "models": {},
        }

        # モデルごとの統計
        for category in ["stt", "tts", "llm", "misc"]:
            category_dir = self.cache_dir / category
            if category_dir.exists():
                models = list(category_dir.iterdir())
                stats["models"][category] = len(models)

        return stats
