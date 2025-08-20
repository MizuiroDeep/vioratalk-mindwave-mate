"""
ログ管理モジュール

VioraTalkプロジェクト全体で統一されたログ出力を提供する。
シングルトンパターンで実装され、環境変数による設定が可能。

関連ドキュメント:
    - ログ実装ガイド v1.1
    - エラー処理実装ガイド v1.0 セクション4
    - 開発規約書 v1.11 セクション6
"""

import json
import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


class LoggerManager:
    """
    統一されたログ管理クラス（シングルトン）

    環境変数:
        VIORATALK_ENV: 実行環境 (development/test/production)
        VIORATALK_LOG_LEVEL: ログレベル (DEBUG/INFO/WARNING/ERROR/CRITICAL)
        VIORATALK_LOG_DIR: ログ出力先ディレクトリ

    Attributes:
        log_dir: ログファイルの出力ディレクトリ
        log_level: ログレベル
        debug_mode: デバッグモードフラグ
    """

    _instance: Optional["LoggerManager"] = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs):
        """シングルトンインスタンスの取得"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        log_dir: Optional[Path] = None,
        log_level: Optional[str] = None,
        debug_mode: bool = False,
    ):
        """
        LoggerManagerの初期化

        Args:
            log_dir: ログディレクトリ（None時は環境変数から取得）
            log_level: ログレベル（None時は環境変数から取得）
            debug_mode: デバッグモードフラグ
        """
        # 既に初期化済みなら何もしない
        if LoggerManager._initialized:
            return

        # 環境設定の取得
        env = os.getenv("VIORATALK_ENV", "development")

        # ログディレクトリの決定
        if log_dir is None:
            log_dir = self._get_default_log_dir(env)

        self.log_dir = Path(log_dir)

        # ログレベルの決定
        if log_level is None:
            log_level = self._get_default_log_level(env)

        self.log_level = log_level if not debug_mode else "DEBUG"
        self.debug_mode = debug_mode
        self.env = env

        # ログディレクトリ作成
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # ルートロガーの設定
        self._setup_root_logger()

        LoggerManager._initialized = True

        # 初期化完了ログ
        logger = self.get_logger("LoggerManager")
        logger.info(
            "LoggerManager initialized",
            extra={"env": self.env, "log_dir": str(self.log_dir), "log_level": self.log_level},
        )

    def _get_default_log_dir(self, env: str) -> Path:
        """
        環境に応じたデフォルトログディレクトリを取得

        Args:
            env: 実行環境

        Returns:
            ログディレクトリのPath
        """
        # 環境変数で指定されている場合はそれを使用
        if env_dir := os.getenv("VIORATALK_LOG_DIR"):
            return Path(env_dir)

        # 環境別のデフォルト
        if env == "production":
            # 本番環境: ユーザーホーム
            return Path.home() / ".vioratalk" / "logs"
        elif env == "test":
            # テスト環境: 一時ディレクトリ
            import tempfile

            return Path(tempfile.gettempdir()) / "vioratalk_test_logs"
        else:
            # 開発環境: プロジェクトルート
            return Path("logs")

    def _get_default_log_level(self, env: str) -> str:
        """
        環境に応じたデフォルトログレベルを取得

        Args:
            env: 実行環境

        Returns:
            ログレベル文字列
        """
        # 環境変数で指定されている場合はそれを使用
        if env_level := os.getenv("VIORATALK_LOG_LEVEL"):
            return env_level

        # 環境別のデフォルト
        if env == "production":
            return "WARNING"
        elif env == "test":
            return "INFO"
        else:
            return "DEBUG"

    def _setup_root_logger(self):
        """ルートロガーの設定"""
        root_logger = logging.getLogger("vioratalk")
        root_logger.setLevel(logging.DEBUG)  # ハンドラーで制御

        # 既存のハンドラーをクリア
        root_logger.handlers.clear()

        # コンソールハンドラー
        console_handler = self._create_console_handler()
        root_logger.addHandler(console_handler)

        # ファイルハンドラー
        file_handler = self._create_file_handler()
        root_logger.addHandler(file_handler)

        # エラー専用ファイルハンドラー
        error_handler = self._create_error_handler()
        root_logger.addHandler(error_handler)

    def _create_console_handler(self) -> logging.StreamHandler:
        """
        コンソールハンドラーの作成

        Returns:
            設定済みのStreamHandler
        """
        handler = logging.StreamHandler()
        handler.setLevel(getattr(logging, self.log_level))

        # 開発環境では詳細フォーマット、本番環境では簡潔フォーマット
        if self.env == "development" or self.debug_mode:
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        else:
            formatter = logging.Formatter("%(levelname)s - %(message)s")

        handler.setFormatter(formatter)
        return handler

    def _create_file_handler(self) -> logging.handlers.RotatingFileHandler:
        """
        ファイルハンドラーの作成（ローテーション付き）

        Returns:
            設定済みのRotatingFileHandler
        """
        log_file = self.log_dir / "vioratalk.log"

        handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10_485_760, backupCount=5, encoding="utf-8"  # 10MB
        )

        handler.setLevel(logging.DEBUG)  # ファイルには全てのログを記録

        # JSON形式のフォーマッター
        handler.setFormatter(self._create_json_formatter())

        return handler

    def _create_error_handler(self) -> logging.handlers.RotatingFileHandler:
        """
        エラー専用ファイルハンドラーの作成

        Returns:
            設定済みのRotatingFileHandler
        """
        error_log_file = self.log_dir / "errors.log"

        handler = logging.handlers.RotatingFileHandler(
            error_log_file, maxBytes=10_485_760, backupCount=5, encoding="utf-8"  # 10MB
        )

        handler.setLevel(logging.ERROR)  # エラー以上のみ

        # JSON形式のフォーマッター
        handler.setFormatter(self._create_json_formatter())

        return handler

    def _create_json_formatter(self) -> logging.Formatter:
        """
        JSON形式のフォーマッターを作成

        Returns:
            JSONLogFormatter
        """
        return JSONLogFormatter()

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        指定された名前のロガーを取得

        Args:
            name: ロガー名（通常は__name__を使用）

        Returns:
            設定済みのLogger
        """
        # "vioratalk"プレフィックスを追加（重複しない場合のみ）
        if not name.startswith("vioratalk"):
            full_name = f"vioratalk.{name}"
        else:
            full_name = name

        return logging.getLogger(full_name)

    @classmethod
    def reset(cls):
        """
        シングルトンをリセット（テスト用）

        警告:
            本番環境では使用しないこと
        """
        cls._instance = None
        cls._initialized = False


class JSONLogFormatter(logging.Formatter):
    """
    JSON形式でログを出力するフォーマッター

    extraフィールドを含めてJSON形式で出力する。
    エラーハンドリング指針 v1.20に準拠。
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        ログレコードをJSON形式にフォーマット

        Args:
            record: ログレコード

        Returns:
            JSON形式の文字列
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # extraフィールドを追加
        # 標準フィールドを除外
        standard_fields = {
            "name",
            "msg",
            "args",
            "created",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "message",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "thread",
            "threadName",
            "exc_info",
            "exc_text",
            "stack_info",
        }

        for key, value in record.__dict__.items():
            if key not in standard_fields and not key.startswith("_"):
                log_data[key] = value

        # 例外情報がある場合は追加
        if record.exc_info:
            import traceback

            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        return json.dumps(log_data, ensure_ascii=False, default=str)


# Phase 1での利用例（他のコンポーネントから使用）
if __name__ == "__main__":
    # 開発環境での動作確認用
    import asyncio

    async def test_logging():
        """ログ出力のテスト"""
        # LoggerManagerの初期化（通常はmain.pyで実行）
        logger_manager = LoggerManager()

        # 各コンポーネントでのロガー取得
        logger = LoggerManager.get_logger(__name__)

        # 各レベルでのログ出力
        logger.debug("Debug message")
        logger.info("Info message", extra={"component": "test"})
        logger.warning("Warning message", extra={"user_id": 123})
        logger.error(
            "Error message", extra={"error_code": "E0001", "details": {"reason": "Test error"}}
        )

        # 例外のログ
        try:
            raise ValueError("Test exception")
        except Exception:
            logger.error("Exception occurred", exc_info=True, extra={"error_code": "E0002"})

    # テスト実行
    asyncio.run(test_logging())
