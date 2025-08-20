"""
ModelDownloadManagerの単体テスト

テスト実装ガイド v1.3準拠
テスト戦略ガイドライン v1.7準拠
開発規約書 v1.12準拠
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

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
    # download_with_progressのテスト
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_download_with_progress_stub(self, manager):
        """進捗表示付きダウンロード（Phase 1スタブ）"""
        url = "https://example.com/model.bin"
        destination = self.cache_dir / "model.bin"

        # プログレスコールバックのモック
        progress_callback = Mock()

        await manager.download_with_progress(url, destination, progress_callback)

        # スタブ実装の確認
        assert destination.exists()  # 空ファイルが作成される

        # プログレスコールバックが呼ばれた
        assert progress_callback.call_count == 11  # 0%, 10%, ..., 100%

        # 進捗値の確認
        progress_values = [call[0][0] for call in progress_callback.call_args_list]
        assert progress_values[0] == 0.0
        assert progress_values[-1] == 1.0

    @pytest.mark.asyncio
    async def test_download_without_callback(self, manager):
        """コールバックなしのダウンロード"""
        url = "https://example.com/model.bin"
        destination = self.cache_dir / "model.bin"

        # エラーが発生しないことを確認
        await manager.download_with_progress(url, destination, None)

        assert destination.exists()

    @pytest.mark.asyncio
    async def test_download_creates_parent_directory(self, manager):
        """親ディレクトリが存在しない場合は作成"""
        url = "https://example.com/model.bin"
        destination = self.cache_dir / "nested" / "dir" / "model.bin"

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
    # verify_checksumのテスト
    # ------------------------------------------------------------------------

    def test_verify_checksum_stub_success(self, manager):
        """チェックサム検証（Phase 1スタブ - 常に成功）"""
        # テストファイルを作成
        test_file = self.cache_dir / "test.bin"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_bytes(b"test content")

        # Phase 1では常にTrue
        result = manager.verify_checksum(test_file, "expected_checksum", "sha256")

        assert result is True

    def test_verify_checksum_file_not_found(self, manager):
        """ファイルが存在しない場合のチェックサム検証"""
        non_existent = self.cache_dir / "non_existent.bin"

        result = manager.verify_checksum(non_existent, "expected_checksum")

        assert result is False

    def test_verify_checksum_different_algorithms(self, manager):
        """異なるアルゴリズムでの検証（Phase 1では無視）"""
        test_file = self.cache_dir / "test.bin"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_bytes(b"test content")

        # Phase 1では常にTrue（アルゴリズムは無視）
        assert manager.verify_checksum(test_file, "checksum", "sha256") is True
        assert manager.verify_checksum(test_file, "checksum", "md5") is True
        assert manager.verify_checksum(test_file, "checksum", "invalid") is True

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
        """キャッシュサイズが制限を超える場合（Phase 1では削除しない）"""
        # 複数のファイルを作成
        for i in range(5):
            test_file = self.cache_dir / f"file_{i}.bin"
            test_file.parent.mkdir(parents=True, exist_ok=True)
            test_file.write_bytes(b"x" * 1024)  # 1KB

        # 非常に小さい制限を設定
        await manager.cleanup_cache(max_size_gb=0.000001)  # 1バイト

        # Phase 1では実際には削除しない（ログのみ）
        for i in range(5):
            test_file = self.cache_dir / f"file_{i}.bin"
            assert test_file.exists()

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

        # モックコールバック
        progress_callback = Mock()

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

        path = await manager.download_if_needed(model_name, url, checksum=checksum)

        assert path.exists()
        # Phase 1では検証は常に成功（警告ログのみ）

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
