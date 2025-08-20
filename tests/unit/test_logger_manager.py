"""LoggerManagerの単体テスト

ログ管理機能の動作を検証。
Phase 1の最小実装として、基本的な機能のテストに焦点を当てる。

テスト戦略ガイドライン v1.7準拠
テスト実装ガイド v1.3準拠
開発規約書 v1.12準拠
"""

import json
import logging
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from vioratalk.utils.logger_manager import JSONLogFormatter, LoggerManager

# ============================================================================
# LoggerManagerのテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(1)
class TestLoggerManager:
    """LoggerManagerクラスのテスト"""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """各テストの前後でシングルトンをリセット"""
        # テスト前にリセット
        LoggerManager._instance = None
        LoggerManager._initialized = False

        yield

        # テスト後にもリセット
        LoggerManager._instance = None
        LoggerManager._initialized = False

    @pytest.fixture
    def temp_log_dir(self):
        """一時ログディレクトリのフィクスチャ"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        # クリーンアップ
        shutil.rmtree(temp_dir, ignore_errors=True)

    # ------------------------------------------------------------------------
    # シングルトンパターンのテスト
    # ------------------------------------------------------------------------

    def test_singleton_pattern(self, temp_log_dir):
        """シングルトンパターンが正しく動作するか"""
        # 複数回インスタンス化しても同じオブジェクトが返される
        manager1 = LoggerManager(log_dir=temp_log_dir)
        manager2 = LoggerManager(log_dir=temp_log_dir)

        assert manager1 is manager2
        assert id(manager1) == id(manager2)

    def test_singleton_initialization_once(self, temp_log_dir):
        """初期化が一度だけ行われるか"""
        # 最初のインスタンスで設定
        manager1 = LoggerManager(log_dir=temp_log_dir, log_level="INFO", debug_mode=False)

        # 2回目は異なるパラメータを渡しても無視される
        temp_dir2 = tempfile.mkdtemp()
        try:
            manager2 = LoggerManager(log_dir=Path(temp_dir2), log_level="DEBUG", debug_mode=True)

            # 最初の設定が保持されている
            assert manager2.log_dir == temp_log_dir
            assert manager2.log_level == "INFO"
            assert manager2.debug_mode is False
        finally:
            shutil.rmtree(temp_dir2, ignore_errors=True)

    # ------------------------------------------------------------------------
    # 環境変数による設定のテスト
    # ------------------------------------------------------------------------

    def test_environment_variable_config(self, temp_log_dir):
        """環境変数からの設定読み込み"""
        with patch.dict(
            os.environ,
            {
                "VIORATALK_ENV": "production",
                "VIORATALK_LOG_LEVEL": "ERROR",
                "VIORATALK_LOG_DIR": str(temp_log_dir),
            },
        ):
            manager = LoggerManager()

            assert manager.env == "production"
            assert manager.log_level == "ERROR"
            assert manager.log_dir == temp_log_dir

    def test_default_log_level_by_environment(self, temp_log_dir):
        """環境別のデフォルトログレベル"""
        # 開発環境
        with patch.dict(os.environ, {"VIORATALK_ENV": "development"}, clear=True):
            manager = LoggerManager(log_dir=temp_log_dir)
            assert manager.log_level == "DEBUG"
            LoggerManager._instance = None
            LoggerManager._initialized = False

        # テスト環境
        with patch.dict(os.environ, {"VIORATALK_ENV": "test"}, clear=True):
            manager = LoggerManager(log_dir=temp_log_dir)
            assert manager.log_level == "INFO"
            LoggerManager._instance = None
            LoggerManager._initialized = False

        # 本番環境
        with patch.dict(os.environ, {"VIORATALK_ENV": "production"}, clear=True):
            manager = LoggerManager(log_dir=temp_log_dir)
            assert manager.log_level == "WARNING"

    # ------------------------------------------------------------------------
    # ログディレクトリ作成のテスト
    # ------------------------------------------------------------------------

    def test_log_directory_creation(self):
        """ログディレクトリが自動作成されるか"""
        temp_base = tempfile.mkdtemp()
        try:
            log_dir = Path(temp_base) / "nested" / "log" / "dir"
            assert not log_dir.exists()

            manager = LoggerManager(log_dir=log_dir)

            assert log_dir.exists()
            assert log_dir.is_dir()
        finally:
            shutil.rmtree(temp_base, ignore_errors=True)

    # ------------------------------------------------------------------------
    # get_loggerメソッドのテスト
    # ------------------------------------------------------------------------

    def test_get_logger_with_prefix(self, temp_log_dir):
        """get_loggerでvioratalkプレフィックスが追加されるか"""
        LoggerManager(log_dir=temp_log_dir)

        # プレフィックスなしの場合は追加される
        logger1 = LoggerManager.get_logger("module.submodule")
        assert logger1.name == "vioratalk.module.submodule"

        # 既にプレフィックスがある場合は追加されない
        logger2 = LoggerManager.get_logger("vioratalk.core.engine")
        assert logger2.name == "vioratalk.core.engine"

    def test_get_logger_returns_logger_instance(self, temp_log_dir):
        """get_loggerがLoggerインスタンスを返すか"""
        LoggerManager(log_dir=temp_log_dir)

        logger = LoggerManager.get_logger(__name__)
        assert isinstance(logger, logging.Logger)

    # ------------------------------------------------------------------------
    # ハンドラー設定のテスト
    # ------------------------------------------------------------------------

    def test_handlers_are_configured(self, temp_log_dir):
        """ハンドラーが正しく設定されているか"""
        manager = LoggerManager(log_dir=temp_log_dir)

        root_logger = logging.getLogger("vioratalk")
        handlers = root_logger.handlers

        # 3つのハンドラーが設定されている
        assert len(handlers) >= 3

        # ハンドラーの種類を確認
        handler_types = [type(h).__name__ for h in handlers]
        assert "StreamHandler" in handler_types
        assert handler_types.count("RotatingFileHandler") >= 2  # 通常ログとエラーログ

    def test_log_files_created(self, temp_log_dir):
        """ログファイルが作成されるか"""
        manager = LoggerManager(log_dir=temp_log_dir)
        logger = LoggerManager.get_logger("test")

        # ログ出力
        logger.info("Test info message")
        logger.error("Test error message")

        # ハンドラーを明示的にフラッシュ
        for handler in logging.getLogger("vioratalk").handlers:
            handler.flush()

        # ログファイルの存在確認
        assert (temp_log_dir / "vioratalk.log").exists()
        assert (temp_log_dir / "errors.log").exists()

    # ------------------------------------------------------------------------
    # resetメソッドのテスト
    # ------------------------------------------------------------------------

    def test_reset_method(self, temp_log_dir):
        """resetメソッドがシングルトンをリセットするか"""
        manager1 = LoggerManager(log_dir=temp_log_dir)
        LoggerManager.reset()

        assert LoggerManager._instance is None
        assert LoggerManager._initialized is False

        # リセット後は新しいインスタンスが作成される
        temp_dir2 = tempfile.mkdtemp()
        try:
            manager2 = LoggerManager(log_dir=Path(temp_dir2))
            assert manager1 is not manager2
            assert manager2.log_dir == Path(temp_dir2)
        finally:
            shutil.rmtree(temp_dir2, ignore_errors=True)


# ============================================================================
# JSONLogFormatterのテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(1)
class TestJSONLogFormatter:
    """JSONLogFormatterクラスのテスト"""

    @pytest.fixture
    def formatter(self):
        """JSONLogFormatterのフィクスチャ"""
        return JSONLogFormatter()

    @pytest.fixture
    def log_record(self):
        """基本的なLogRecordのフィクスチャ"""
        record = logging.LogRecord(
            name="test.module",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Test message with %s",
            args=("argument",),
            exc_info=None,
        )
        return record

    def test_basic_json_format(self, formatter, log_record):
        """基本的なJSON形式でフォーマットされるか"""
        formatted = formatter.format(log_record)

        # JSON として解析可能か
        data = json.loads(formatted)

        # 必須フィールドの確認
        assert "timestamp" in data
        assert "level" in data
        assert "logger" in data
        assert "message" in data
        assert "module" in data
        assert "function" in data
        assert "line" in data

        # 値の確認
        assert data["level"] == "INFO"
        assert data["logger"] == "test.module"
        assert data["message"] == "Test message with argument"
        assert data["line"] == 42

    def test_extra_fields_included(self, formatter):
        """extraフィールドが含まれるか"""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test",
            args=(),
            exc_info=None,
        )

        # extraフィールドを追加
        record.error_code = "E0001"
        record.user_id = 12345
        record.component = "TestComponent"

        formatted = formatter.format(record)
        data = json.loads(formatted)

        # extraフィールドが含まれているか
        assert data["error_code"] == "E0001"
        assert data["user_id"] == 12345
        assert data["component"] == "TestComponent"

    def test_exception_formatting(self, formatter):
        """例外情報が正しくフォーマットされるか"""
        try:
            raise ValueError("Test exception")
        except ValueError:
            import sys

            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="",
                lineno=0,
                msg="Error occurred",
                args=(),
                exc_info=sys.exc_info(),
            )

        formatted = formatter.format(record)
        data = json.loads(formatted)

        # 例外情報が含まれているか
        assert "exception" in data
        assert data["exception"]["type"] == "ValueError"
        assert data["exception"]["message"] == "Test exception"
        assert "traceback" in data["exception"]
        assert isinstance(data["exception"]["traceback"], list)

    def test_timestamp_format(self, formatter, log_record):
        """タイムスタンプがISO形式か"""
        formatted = formatter.format(log_record)
        data = json.loads(formatted)

        # ISO形式の確認（末尾にZ）
        assert data["timestamp"].endswith("Z")

        # パース可能か確認

        timestamp = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
        assert isinstance(timestamp, datetime)

    def test_unicode_handling(self, formatter):
        """Unicode文字が正しく処理されるか"""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="日本語メッセージ: %s",
            args=("テスト",),
            exc_info=None,
        )

        formatted = formatter.format(record)
        data = json.loads(formatted)

        assert data["message"] == "日本語メッセージ: テスト"

        # ensure_ascii=Falseなので、エスケープされていない
        assert "\\u" not in formatted


# ============================================================================
# 統合的な動作テスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(1)
class TestLoggerManagerIntegration:
    """LoggerManagerの統合的な動作テスト"""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """シングルトンのリセット"""
        LoggerManager._instance = None
        LoggerManager._initialized = False
        yield
        LoggerManager._instance = None
        LoggerManager._initialized = False

    @pytest.fixture
    def temp_log_dir(self):
        """一時ログディレクトリ"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_multiple_loggers_work_together(self, temp_log_dir):
        """複数のロガーが協調動作するか"""
        manager = LoggerManager(log_dir=temp_log_dir, log_level="DEBUG")

        # 異なるモジュールのロガーを取得
        logger1 = LoggerManager.get_logger("core.engine")
        logger2 = LoggerManager.get_logger("services.background")
        logger3 = LoggerManager.get_logger("utils.helper")

        # それぞれでログ出力
        logger1.info("Engine started")
        logger2.warning("Service warning")
        logger3.error("Helper error")

        # ハンドラーをフラッシュ
        for handler in logging.getLogger("vioratalk").handlers:
            handler.flush()

        # ログファイルが作成されている
        log_file = temp_log_dir / "vioratalk.log"
        error_file = temp_log_dir / "errors.log"

        assert log_file.exists()
        assert error_file.exists()

        # エラーログにはERROR以上のみ
        if error_file.stat().st_size > 0:
            with open(error_file, "r", encoding="utf-8") as f:
                content = f.read()
                assert "Helper error" in content
                assert "Engine started" not in content

    def test_debug_mode_changes_console_format(self, temp_log_dir, capsys):
        """デバッグモードでコンソール出力形式が変わるか"""
        # 通常モード
        manager1 = LoggerManager(log_dir=temp_log_dir, debug_mode=False)
        logger1 = LoggerManager.get_logger("test1")
        logger1.info("Normal mode message")

        LoggerManager._instance = None
        LoggerManager._initialized = False

        # デバッグモード
        manager2 = LoggerManager(log_dir=temp_log_dir, debug_mode=True)
        logger2 = LoggerManager.get_logger("test2")
        logger2.debug("Debug mode message")

        # デバッグモードではDEBUGレベルも出力される
        captured = capsys.readouterr()
        assert "Debug mode message" in captured.err or "Debug mode message" in captured.out
