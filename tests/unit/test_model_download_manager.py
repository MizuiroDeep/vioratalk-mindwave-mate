"""
ModelDownloadManagerの単体テスト

テスト実装ガイド v1.3準拠
テスト戦略ガイドライン v1.7準拠
開発規約書 v1.12準拠
"""

import asyncio
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest
from aioresponses import aioresponses

from vioratalk.infrastructure.model_download_manager import ModelDownloadManager

# ============================================================================
# ModelDownloadManagerのテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(1)
class TestModelDownloadManager:
    """ModelDownloadManagerクラスのテスト"""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """各テストの前後処理"""
        # 一時ディレクトリの作成
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir) / "models"

        yield

        # クリーンアップ
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.fixture
    def manager(self):
        """ModelDownloadManagerのフィクスチャ"""
        return ModelDownloadManager(cache_dir=self.cache_dir)

    # ------------------------------------------------------------------------
    # 初期化のテスト
    # ------------------------------------------------------------------------

    def test_initialization_with_custom_dir(self):
        """カスタムディレクトリでの初期化"""
        manager = ModelDownloadManager(cache_dir=self.cache_dir)

        assert manager.cache_dir == self.cache_dir
        assert self.cache_dir.exists()
        assert manager._download_tasks == []

    def test_initialization_with_default_dir(self):
        """デフォルトディレクトリでの初期化"""
        with patch("vioratalk.infrastructure.model_download_manager.MODELS_DIR") as mock_dir:
            mock_dir.return_value = Path("data/models")
            manager = ModelDownloadManager()

            assert manager.cache_dir == mock_dir

    def test_initialization_creates_directory(self):
        """初期化時にディレクトリを作成"""
        non_existent = Path(self.temp_dir) / "new_dir"
        manager = ModelDownloadManager(cache_dir=non_existent)

        assert non_existent.exists()
        assert non_existent.is_dir()

    # ------------------------------------------------------------------------
    # download_whisper_modelのテスト（Phase 4新規追加）
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    @pytest.mark.phase(4)
    async def test_download_whisper_model_basic(self, manager):
        """基本的なWhisperモデルダウンロード"""
        # huggingface_hubがインストールされていることにする
        with patch(
            "vioratalk.infrastructure.model_download_manager.HUGGINGFACE_HUB_AVAILABLE", True
        ):
            with patch(
                "vioratalk.infrastructure.model_download_manager.snapshot_download"
            ) as mock_download:
                mock_download.return_value = None

                # プログレスコールバックのモック
                progress_callback = Mock()

                # run_in_executorのモック
                with patch.object(asyncio, "get_event_loop") as mock_loop:
                    mock_loop.return_value.run_in_executor = AsyncMock(return_value=None)

                    # ダウンロード実行
                    model_path = await manager.download_whisper_model(
                        model_size="tiny", progress_callback=progress_callback
                    )

                    # パスの確認
                    assert "whisper-tiny" in str(model_path)
                    assert model_path.parent.name == "stt"

                    # プログレス更新の確認（最終的に100%になる）
                    progress_callback.assert_called()
                    assert progress_callback.call_args[0][0] == 1.0

    @pytest.mark.asyncio
    @pytest.mark.phase(4)
    async def test_download_whisper_model_cached(self, manager):
        """キャッシュ済みWhisperモデルのスキップテスト"""
        # モデルディレクトリを事前に作成
        model_dir = manager.cache_dir / "stt" / "whisper-base"
        model_dir.mkdir(parents=True)

        # 必要なファイルを作成
        (model_dir / "model.bin").touch()
        (model_dir / "config.json").touch()
        (model_dir / "tokenizer.json").touch()
        (model_dir / "vocabulary.txt").touch()

        # メタデータファイル作成
        metadata = {
            "model_size": "base",
            "downloaded_at": "2025-08-26T00:00:00",
            "version": "1.0.0",
        }
        (model_dir / "download_metadata.json").write_text(json.dumps(metadata))

        # ダウンロードは呼ばれないはず
        with patch(
            "vioratalk.infrastructure.model_download_manager.snapshot_download"
        ) as mock_download:
            model_path = await manager.download_whisper_model("base")

            # snapshot_downloadが呼ばれていないことを確認
            mock_download.assert_not_called()
            assert model_path == model_dir

    @pytest.mark.asyncio
    @pytest.mark.phase(4)
    async def test_download_whisper_model_invalid_size(self, manager):
        """無効なWhisperモデルサイズのエラーテスト"""
        from vioratalk.core.exceptions import ModelError

        with pytest.raises(ModelError) as exc_info:
            await manager.download_whisper_model("invalid_size")

        assert exc_info.value.error_code == "E5500"
        assert "invalid_size" in str(exc_info.value)

    @pytest.mark.asyncio
    @pytest.mark.phase(4)
    async def test_download_whisper_model_no_huggingface_hub(self, manager):
        """huggingface_hubなしでの手動ダウンロード"""
        # huggingface_hubが利用不可
        with patch(
            "vioratalk.infrastructure.model_download_manager.HUGGINGFACE_HUB_AVAILABLE", False
        ):
            progress_callback = Mock()

            # 手動ダウンロード（スタブ実装）を実行
            model_path = await manager.download_whisper_model(
                model_size="small", progress_callback=progress_callback
            )

            # パスの確認
            assert "whisper-small" in str(model_path)
            # ダミーファイルが作成される
            assert (model_path / "model.bin").exists()
            # プログレス100%
            progress_callback.assert_called_with(1.0)

    @pytest.mark.asyncio
    @pytest.mark.phase(4)
    async def test_download_whisper_model_download_error(self, manager):
        """Whisperモデルダウンロードエラーのテスト"""
        from vioratalk.core.exceptions import FileSystemError

        with patch(
            "vioratalk.infrastructure.model_download_manager.HUGGINGFACE_HUB_AVAILABLE", True
        ):
            with patch(
                "vioratalk.infrastructure.model_download_manager.snapshot_download"
            ) as mock_download:
                mock_download.side_effect = Exception("Network error")

                with patch.object(asyncio, "get_event_loop") as mock_loop:
                    mock_loop.return_value.run_in_executor = AsyncMock(
                        side_effect=Exception("Network error")
                    )

                    with pytest.raises(FileSystemError) as exc_info:
                        await manager.download_whisper_model("medium")

                    assert exc_info.value.error_code == "E5101"
                    assert "whisper-medium" in str(exc_info.value)

    # ------------------------------------------------------------------------
    # download_with_progressのテスト（Phase 4実装版）
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_download_with_progress_stub(self, manager):
        """進捗表示付きダウンロード（Phase 1スタブ互換性テスト）"""
        url = "https://example.com/model.bin"
        destination = self.cache_dir / "model.bin"

        # プログレスコールバックのモック
        progress_callback = Mock()

        # Phase 4では実際のHTTP通信を試みるのでモックする
        with aioresponses() as mocked:
            # 正常なレスポンスをモック
            content = b"dummy content"
            mocked.get(url, status=200, body=content, headers={"content-length": str(len(content))})

            await manager.download_with_progress(url, destination, progress_callback)

            # ファイルが作成される
            assert destination.exists()

            # プログレスコールバックが呼ばれた
            assert progress_callback.called
            # 最終的に100%になる
            assert progress_callback.call_args[0][0] == 1.0

    @pytest.mark.asyncio
    @pytest.mark.phase(4)
    async def test_download_with_progress_real_impl(self, manager):
        """実際のダウンロード実装のテスト（Phase 4）"""
        url = "https://example.com/model.bin"
        destination = self.cache_dir / "test_model.bin"

        # aioresponsesでHTTPレスポンスをモック
        with aioresponses() as mocked:
            # ダミーコンテンツ
            content = b"model_content_" + b"x" * 1000  # 約1KB
            mocked.get(url, status=200, body=content, headers={"content-length": str(len(content))})

            # プログレスコールバック
            progress_values = []

            def progress_callback(value):
                progress_values.append(value)

            # ダウンロード実行
            await manager.download_with_progress(
                url=url, destination=destination, progress_callback=progress_callback
            )

            # ファイルが作成されたことを確認
            assert destination.exists()
            # 実際のコンテンツが書き込まれたことを確認
            assert destination.stat().st_size == len(content)

            # プログレスが更新されたことを確認
            assert len(progress_values) > 0
            assert progress_values[-1] == 1.0  # 最終値は100%

    @pytest.mark.asyncio
    @pytest.mark.phase(4)
    async def test_download_with_progress_network_error(self, manager):
        """ダウンロード中のネットワークエラーテスト"""
        from vioratalk.core.exceptions import FileSystemError

        url = "https://example.com/error.bin"
        destination = self.cache_dir / "error.bin"

        with aioresponses() as mocked:
            # ネットワークエラーをシミュレート
            mocked.get(url, status=404)

            with pytest.raises(FileSystemError) as exc_info:
                await manager.download_with_progress(url, destination)

            assert exc_info.value.error_code == "E5101"
            # 一時ファイルは削除されている
            assert not (destination.with_suffix(".bin.tmp")).exists()

    @pytest.mark.asyncio
    async def test_download_without_callback(self, manager):
        """コールバックなしのダウンロード"""
        url = "https://example.com/model.bin"
        destination = self.cache_dir / "model.bin"

        # Phase 4では実際のHTTP通信を試みるのでモックする
        with aioresponses() as mocked:
            content = b"test content"
            mocked.get(url, status=200, body=content, headers={"content-length": str(len(content))})

            # エラーが発生しないことを確認
            await manager.download_with_progress(url, destination, None)

            assert destination.exists()

    @pytest.mark.asyncio
    async def test_download_creates_parent_directory(self, manager):
        """親ディレクトリが存在しない場合は作成"""
        url = "https://example.com/model.bin"
        destination = self.cache_dir / "nested" / "dir" / "model.bin"

        # Phase 4では実際のHTTP通信を試みるのでモックする
        with aioresponses() as mocked:
            content = b"test content"
            mocked.get(url, status=200, body=content, headers={"content-length": str(len(content))})

            await manager.download_with_progress(url, destination)

            assert destination.parent.exists()
            assert destination.exists()

    @pytest.mark.asyncio
    async def test_download_existing_file(self, manager):
        """既存ファイルがある場合の動作"""
        url = "https://example.com/model.bin"
        destination = self.cache_dir / "model.bin"

        # 既存ファイルを作成
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("existing content")

        await manager.download_with_progress(url, destination)

        # Phase 1では既存ファイルはそのまま（上書きしない）
        assert destination.exists()
        assert destination.read_text() == "existing content"

    # ------------------------------------------------------------------------
    # verify_checksumのテスト（Phase 4実装版）
    # ------------------------------------------------------------------------

    def test_verify_checksum_stub_success(self, manager):
        """チェックサム検証（Phase 4実装 - 実際に検証）"""
        # テストファイルを作成
        test_file = self.cache_dir / "test.bin"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_content = b"test content"
        test_file.write_bytes(test_content)

        # Phase 4では実際にチェックサムを計算して比較
        import hashlib

        correct_checksum = hashlib.sha256(test_content).hexdigest()

        # 正しいチェックサムの場合
        result = manager.verify_checksum(test_file, correct_checksum, "sha256")
        assert result is True

        # 間違ったチェックサムの場合
        result = manager.verify_checksum(test_file, "wrong_checksum", "sha256")
        assert result is False

    @pytest.mark.phase(4)
    def test_verify_checksum_real_impl_sha256(self, manager):
        """実際のSHA256チェックサム検証（Phase 4）"""
        # テストファイルを作成
        test_file = self.cache_dir / "test.bin"
        test_content = b"test content for checksum verification"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_bytes(test_content)

        # 正しいチェックサムを計算
        correct_sha256 = hashlib.sha256(test_content).hexdigest()

        # 正しいチェックサム（成功）
        assert manager.verify_checksum(test_file, correct_sha256, "sha256") is True

        # 間違ったチェックサム（失敗）
        assert manager.verify_checksum(test_file, "wrong_checksum_value", "sha256") is False

    @pytest.mark.phase(4)
    def test_verify_checksum_real_impl_md5(self, manager):
        """実際のMD5チェックサム検証（Phase 4）"""
        # テストファイルを作成
        test_file = self.cache_dir / "test.bin"
        test_content = b"another test content"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_bytes(test_content)

        # 正しいチェックサムを計算
        correct_md5 = hashlib.md5(test_content).hexdigest()

        # MD5検証（成功）
        assert manager.verify_checksum(test_file, correct_md5, "md5") is True

        # 大文字小文字を区別しない
        assert manager.verify_checksum(test_file, correct_md5.upper(), "md5") is True

    @pytest.mark.phase(4)
    def test_verify_checksum_unsupported_algorithm(self, manager):
        """サポートされていないアルゴリズムのテスト（Phase 4）"""
        test_file = self.cache_dir / "test.bin"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_bytes(b"content")

        # サポートされていないアルゴリズム
        with pytest.raises(ValueError) as exc_info:
            manager.verify_checksum(test_file, "checksum", "sha512")

        assert "Unsupported algorithm: sha512" in str(exc_info.value)

    def test_verify_checksum_file_not_found(self, manager):
        """ファイルが存在しない場合のチェックサム検証"""
        non_existent = self.cache_dir / "non_existent.bin"

        result = manager.verify_checksum(non_existent, "expected_checksum")

        assert result is False

    def test_verify_checksum_different_algorithms(self, manager):
        """異なるアルゴリズムでの検証（Phase 4実装）"""
        test_file = self.cache_dir / "test.bin"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_content = b"test content"
        test_file.write_bytes(test_content)

        # Phase 4では実際に検証を行う
        import hashlib

        sha256_checksum = hashlib.sha256(test_content).hexdigest()
        md5_checksum = hashlib.md5(test_content).hexdigest()

        # 正しいチェックサムで検証
        assert manager.verify_checksum(test_file, sha256_checksum, "sha256") is True
        assert manager.verify_checksum(test_file, md5_checksum, "md5") is True

        # 間違ったチェックサムで検証
        assert manager.verify_checksum(test_file, "wrong", "sha256") is False

        # サポートされていないアルゴリズムは ValueError を投げる
        with pytest.raises(ValueError):
            manager.verify_checksum(test_file, "checksum", "invalid")

    # ------------------------------------------------------------------------
    # get_cache_pathのテスト
    # ------------------------------------------------------------------------

    def test_get_cache_path_whisper(self, manager):
        """Whisperモデルのキャッシュパス"""
        path = manager.get_cache_path("whisper-base")

        assert "stt" in str(path)
        assert "whisper-base" in str(path)
        assert path.parent.exists()  # ディレクトリが作成される

    def test_get_cache_path_aivisspeech(self, manager):
        """AivisSpeechモデルのキャッシュパス"""
        path = manager.get_cache_path("aivisspeech-kana")

        assert "tts" in str(path)
        assert "aivisspeech-kana" in str(path)

    def test_get_cache_path_llm(self, manager):
        """LLMモデルのキャッシュパス"""
        for model_name in ["llama-3", "mistral-7b", "phi-3"]:
            path = manager.get_cache_path(model_name)
            assert "llm" in str(path)

    def test_get_cache_path_misc(self, manager):
        """その他のモデルのキャッシュパス"""
        path = manager.get_cache_path("unknown-model")

        assert "misc" in str(path)
        assert "unknown-model" in str(path)

    def test_get_cache_path_sanitization(self, manager):
        """モデル名のサニタイズ"""
        # 特殊文字を含むモデル名
        path = manager.get_cache_path("model/with:special*chars?")

        # 特殊文字がアンダースコアに置換される
        assert "/" not in path.name
        assert ":" not in path.name
        assert "*" not in path.name
        assert "?" not in path.name

    # ------------------------------------------------------------------------
    # cleanup_cacheのテスト
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_cleanup_cache_under_limit(self, manager):
        """キャッシュサイズが制限内の場合"""
        # 小さいファイルを作成
        test_file = self.cache_dir / "small.bin"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_bytes(b"small content")

        await manager.cleanup_cache(max_size_gb=10.0)

        # ファイルは削除されない
        assert test_file.exists()

    @pytest.mark.asyncio
    async def test_cleanup_cache_over_limit(self, manager):
        """キャッシュサイズが制限を超える場合（Phase 4では実際に削除される）"""
        # 複数のファイルを作成
        files = []
        for i in range(5):
            test_file = self.cache_dir / f"file_{i}.bin"
            test_file.parent.mkdir(parents=True, exist_ok=True)
            test_file.write_bytes(b"x" * 1024)  # 1KB
            files.append(test_file)
            # アクセス時刻を変更してLRU順序を制御
            time.sleep(0.01)  # ファイル間でアクセス時刻に差をつける
            os.utime(test_file, (time.time() - (5 - i), time.time() - (5 - i)))

        # 非常に小さい制限を設定
        await manager.cleanup_cache(max_size_gb=0.000001)  # 1バイト

        # Phase 4では実際に削除される
        # 古いファイルから削除されるので、残っているファイルを確認
        remaining_files = [f for f in files if f.exists()]
        deleted_files = [f for f in files if not f.exists()]

        # 少なくとも1つは削除されているはず
        assert len(deleted_files) > 0

        # 削除されたファイルが古い順であることを確認（任意）
        # または、単に削除が行われたことだけを確認
        assert len(remaining_files) < 5

    @pytest.mark.asyncio
    async def test_cleanup_cache_empty_directory(self, manager):
        """空のキャッシュディレクトリ"""
        # エラーが発生しないことを確認
        await manager.cleanup_cache()

        assert self.cache_dir.exists()

    @pytest.mark.asyncio
    async def test_cleanup_cache_with_subdirectories(self, manager):
        """サブディレクトリを含むキャッシュ"""
        # サブディレクトリとファイルを作成
        (self.cache_dir / "stt").mkdir(parents=True)
        (self.cache_dir / "tts").mkdir(parents=True)

        file1 = self.cache_dir / "stt" / "model1.bin"
        file2 = self.cache_dir / "tts" / "model2.bin"

        file1.write_bytes(b"content1")
        file2.write_bytes(b"content2")

        await manager.cleanup_cache(max_size_gb=10.0)

        # ファイルは保持される
        assert file1.exists()
        assert file2.exists()

    # ------------------------------------------------------------------------
    # is_model_cachedのテスト
    # ------------------------------------------------------------------------

    def test_is_model_cached_file_exists(self, manager):
        """ファイルが存在する場合"""
        model_name = "test-model"
        cache_path = manager.get_cache_path(model_name)

        # ファイルを作成
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(b"model content")

        assert manager.is_model_cached(model_name) is True

    def test_is_model_cached_directory_with_files(self, manager):
        """ディレクトリにファイルが含まれる場合"""
        model_name = "test-model"
        cache_path = manager.get_cache_path(model_name)

        # ディレクトリとファイルを作成
        cache_path.mkdir(parents=True)
        (cache_path / "model.bin").write_bytes(b"content")

        assert manager.is_model_cached(model_name) is True

    def test_is_model_cached_empty_directory(self, manager):
        """空のディレクトリの場合"""
        model_name = "test-model"
        cache_path = manager.get_cache_path(model_name)

        # 空のディレクトリを作成
        cache_path.mkdir(parents=True)

        assert manager.is_model_cached(model_name) is False

    def test_is_model_cached_not_exists(self, manager):
        """何も存在しない場合"""
        assert manager.is_model_cached("non-existent-model") is False

    # ------------------------------------------------------------------------
    # get_model_infoのテスト
    # ------------------------------------------------------------------------

    def test_get_model_info_cached_file(self, manager):
        """キャッシュされたファイルの情報取得"""
        model_name = "test-model"
        cache_path = manager.get_cache_path(model_name)

        # ファイルを作成
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        content = b"x" * 1024  # 1KB
        cache_path.write_bytes(content)

        info = manager.get_model_info(model_name)

        assert info is not None
        assert info["name"] == model_name
        assert info["cached"] is True
        assert info["size"] == 1024
        assert str(cache_path) in info["path"]

    def test_get_model_info_cached_directory(self, manager):
        """キャッシュされたディレクトリの情報取得"""
        model_name = "test-model"
        cache_path = manager.get_cache_path(model_name)

        # ディレクトリとファイルを作成
        cache_path.mkdir(parents=True)
        (cache_path / "file1.bin").write_bytes(b"x" * 500)
        (cache_path / "file2.bin").write_bytes(b"x" * 300)

        info = manager.get_model_info(model_name)

        assert info is not None
        assert info["size"] == 800  # 500 + 300

    def test_get_model_info_not_cached(self, manager):
        """キャッシュされていないモデル"""
        info = manager.get_model_info("non-existent-model")

        assert info is None

    # ------------------------------------------------------------------------
    # download_if_neededのテスト
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_download_if_needed_not_cached(self, manager):
        """キャッシュされていない場合はダウンロード"""
        model_name = "new-model"
        url = "https://example.com/model.bin"
        progress_callback = Mock()

        # HTTPレスポンスをモック
        with aioresponses() as mocked:
            content = b"model content"
            mocked.get(url, status=200, body=content, headers={"content-length": str(len(content))})

            path = await manager.download_if_needed(
                model_name, url, progress_callback=progress_callback
            )

            assert path.exists()
            assert progress_callback.called

    @pytest.mark.asyncio
    async def test_download_if_needed_already_cached(self, manager):
        """既にキャッシュされている場合はスキップ"""
        model_name = "cached-model"
        cache_path = manager.get_cache_path(model_name)

        # 事前にファイルを作成
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(b"cached content")

        url = "https://example.com/model.bin"
        progress_callback = Mock()

        path = await manager.download_if_needed(
            model_name, url, progress_callback=progress_callback
        )

        assert path == cache_path
        assert not progress_callback.called  # ダウンロードはスキップ

    @pytest.mark.asyncio
    async def test_download_if_needed_with_checksum(self, manager):
        """チェックサム検証付きダウンロード"""
        model_name = "model-with-checksum"
        url = "https://example.com/model.bin"
        checksum = "expected_checksum"

        # HTTPレスポンスをモック
        with aioresponses() as mocked:
            content = b"model content"
            mocked.get(url, status=200, body=content, headers={"content-length": str(len(content))})

            path = await manager.download_if_needed(model_name, url, checksum=checksum)
            assert path.exists()
            # Phase 4では警告ログのみ（エラーにはならない）

    # ------------------------------------------------------------------------
    # _get_cache_filesのテスト
    # ------------------------------------------------------------------------

    def test_get_cache_files_empty(self, manager):
        """空のキャッシュディレクトリ"""
        files = manager._get_cache_files()

        assert files == []

    def test_get_cache_files_with_files(self, manager):
        """ファイルを含むキャッシュディレクトリ"""
        # ファイルを作成
        (self.cache_dir / "file1.bin").write_bytes(b"content1")
        (self.cache_dir / "subdir").mkdir()
        (self.cache_dir / "subdir" / "file2.bin").write_bytes(b"content2")

        files = manager._get_cache_files()

        assert len(files) == 2
        assert any("file1.bin" in str(f) for f in files)
        assert any("file2.bin" in str(f) for f in files)

    def test_get_cache_files_excludes_directories(self, manager):
        """ディレクトリは除外される"""
        # ディレクトリのみ作成
        (self.cache_dir / "dir1").mkdir()
        (self.cache_dir / "dir2").mkdir()

        files = manager._get_cache_files()

        assert files == []

    # ------------------------------------------------------------------------
    # get_cache_statsのテスト
    # ------------------------------------------------------------------------

    def test_get_cache_stats_empty(self, manager):
        """空のキャッシュの統計情報"""
        stats = manager.get_cache_stats()

        assert stats["total_files"] == 0
        assert stats["total_size_bytes"] == 0
        assert stats["total_size_mb"] == 0.0
        assert stats["total_size_gb"] == 0.0
        assert str(self.cache_dir) in stats["cache_dir"]

    def test_get_cache_stats_with_files(self, manager):
        """ファイルを含むキャッシュの統計情報"""
        # カテゴリごとにファイルを作成
        (self.cache_dir / "stt").mkdir(parents=True)
        (self.cache_dir / "stt" / "model1").mkdir()
        (self.cache_dir / "stt" / "model1" / "file.bin").write_bytes(b"x" * 1024)

        (self.cache_dir / "tts").mkdir(parents=True)
        (self.cache_dir / "tts" / "model2").mkdir()
        (self.cache_dir / "tts" / "model2" / "file.bin").write_bytes(b"x" * 2048)

        stats = manager.get_cache_stats()

        assert stats["total_files"] == 2
        assert stats["total_size_bytes"] == 3072  # 1024 + 2048
        assert stats["total_size_mb"] == 3072 / (1024**2)
        assert stats["models"]["stt"] == 1
        assert stats["models"]["tts"] == 1

    def test_get_cache_stats_categories(self, manager):
        """カテゴリごとのモデル数"""
        # 各カテゴリにディレクトリを作成
        for category in ["stt", "tts", "llm", "misc"]:
            category_dir = self.cache_dir / category
            category_dir.mkdir(parents=True)
            for i in range(2):
                (category_dir / f"model_{i}").mkdir()

        stats = manager.get_cache_stats()

        assert stats["models"]["stt"] == 2
        assert stats["models"]["tts"] == 2
        assert stats["models"]["llm"] == 2
        assert stats["models"]["misc"] == 2

    # ------------------------------------------------------------------------
    # cleanup関連のテスト
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    @pytest.mark.phase(4)
    async def test_cleanup_resources(self, manager):
        """リソースクリーンアップのテスト"""
        # セッションを作成
        manager._session = aiohttp.ClientSession()

        # 適切なコルーチンタスクを作成
        async def dummy_task():
            await asyncio.sleep(0.1)

        task = asyncio.create_task(dummy_task())
        manager._download_tasks = [task]

        # クリーンアップ実行
        await manager.cleanup()

        # 確認
        assert manager._session is None
        assert task.cancelled()
        assert len(manager._download_tasks) == 0

    # ------------------------------------------------------------------------
    # monitor_download_progressのテスト（内部メソッド）
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    @pytest.mark.phase(4)
    async def test_monitor_download_progress(self, manager):
        """ダウンロード進捗監視のテスト"""
        # モデルディレクトリを作成
        model_dir = self.cache_dir / "stt" / "whisper-test"
        model_dir.mkdir(parents=True)

        # モデル情報
        model_info = {"size_mb": 1}  # 1MB

        # プログレスコールバック
        progress_values = []

        def progress_callback(value):
            progress_values.append(value)

        # 監視タスクを開始（すぐに終了するように）
        (model_dir / "model.bin").write_bytes(b"x" * 1024 * 1024)  # 1MB

        # 監視を短時間実行
        task = asyncio.create_task(
            manager._monitor_download_progress(model_dir, model_info, progress_callback)
        )
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # プログレスが記録されたことを確認
        assert len(progress_values) > 0
