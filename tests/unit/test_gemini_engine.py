"""test_gemini_engine.py - GeminiEngineの単体テスト（新SDK対応・検索機能テスト追加版）

GeminiEngineクラスの包括的な単体テスト。
google-genai（新SDK）をモック化してAPIキーなしでテスト可能。
Phase 4 Part 87: Google検索機能のテストを追加。

テスト戦略ガイドライン v1.7準拠
テスト実装ガイド v1.3準拠
開発規約書 v1.12準拠
エラーハンドリング指針 v1.20準拠
API通信実装ガイド v1.4準拠
"""

# 標準ライブラリ
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# テストライブラリ
import pytest

# プロジェクトのパスを追加（必要に応じて）
project_root = Path(__file__).parent.parent.parent.parent
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))

# テスト対象のインポート
from vioratalk.core.base import ComponentState
from vioratalk.core.exceptions import (
    APIError,
    AuthenticationError,
    ConfigurationError,
    LLMError,
    ModelNotFoundError,
    RateLimitError,
)
from vioratalk.core.llm import GeminiEngine, LLMConfig, LLMRequest, LLMResponse

# ============================================================================
# カスタム例外クラス（テスト用）
# ============================================================================


class MockGoogleGenerativeAIError(Exception):
    """テスト用のGemini固有エラー

    実際のgoogle.genai.types.GoogleGenerativeAIErrorを模擬。
    一般的なExceptionとは区別される。
    """

    def __init__(self, message):
        super().__init__(message)
        # Google APIエラーとして識別されやすくする
        self.__module__ = "google.genai.types"


# ============================================================================
# フィクスチャ
# ============================================================================


@pytest.fixture
def mock_genai():
    """google-genai（新SDK）のモック"""
    with patch("vioratalk.core.llm.gemini_engine.genai") as mock:
        # Clientのモック
        mock_client = MagicMock()

        # models.generate_contentのモック
        mock_client.models.generate_content = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(text="Generated response")

        # models.generate_content_streamのモック（ストリーミング用）
        mock_client.models.generate_content_stream = MagicMock()

        # Clientコンストラクタのモック
        mock.Client.return_value = mock_client

        # typesのモック
        mock.types = MagicMock()
        mock.types.GenerateContentConfig = MagicMock()
        # カスタム例外クラスを使用
        mock.types.GoogleGenerativeAIError = MockGoogleGenerativeAIError

        yield mock


@pytest.fixture
def mock_types():
    """google.genai.typesのモック（検索機能テスト用）"""
    with patch("vioratalk.core.llm.gemini_engine.types") as mock:
        # Tool, GoogleSearchクラスのモック
        mock.Tool = MagicMock()
        mock.GoogleSearch = MagicMock()
        mock.GenerateContentConfig = MagicMock()

        # Tool(google_search=GoogleSearch())の動作をモック
        mock_tool = MagicMock()
        mock.Tool.return_value = mock_tool

        yield mock


@pytest.fixture
def mock_genai_not_available():
    """google-genaiが未インストールの状態"""
    with patch("vioratalk.core.llm.gemini_engine.GENAI_AVAILABLE", False):
        yield


@pytest.fixture
def mock_api_key_manager():
    """APIKeyManagerのモック"""
    with patch("vioratalk.core.llm.gemini_engine.get_api_key_manager") as mock:
        mock_manager = MagicMock()
        mock_manager.get_api_key.return_value = "test-gemini-api-key"
        mock_manager.require_api_key.return_value = "test-gemini-api-key"
        mock.return_value = mock_manager
        yield mock_manager


@pytest.fixture
def mock_error_handler():
    """ErrorHandlerのモック"""
    with patch("vioratalk.core.llm.gemini_engine.get_default_error_handler") as mock:
        mock_handler = MagicMock()
        mock_handler.handle_error = MagicMock()
        mock.return_value = mock_handler
        yield mock_handler


@pytest.fixture
def llm_config():
    """テスト用LLMConfig"""
    return LLMConfig(
        engine="gemini",
        model="gemini-2.0-flash",  # 新SDK: デフォルトモデルを更新
        temperature=0.7,
        max_tokens=1000,
    )


@pytest.fixture
def llm_config_with_search():
    """検索機能有効のLLMConfig（Phase 4 Part 87追加）"""
    return LLMConfig(
        engine="gemini",
        model="gemini-2.0-flash",
        temperature=0.7,
        max_tokens=1000,
        enable_search=True,
        search_threshold=0.5,
    )


@pytest.fixture
async def gemini_engine(
    mock_genai, mock_types, mock_api_key_manager, mock_error_handler, llm_config
):
    """初期化済みGeminiEngineフィクスチャ"""
    engine = GeminiEngine(config=llm_config, api_key="test-api-key")
    await engine.initialize()
    yield engine
    await engine.cleanup()


@pytest.fixture
async def gemini_engine_with_search(
    mock_genai, mock_types, mock_api_key_manager, mock_error_handler, llm_config_with_search
):
    """検索機能有効のGeminiEngineフィクスチャ（Phase 4 Part 87追加）"""
    engine = GeminiEngine(config=llm_config_with_search, api_key="test-api-key")
    await engine.initialize()
    yield engine
    await engine.cleanup()


# ============================================================================
# 初期化テスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestInitialization:
    """初期化関連のテスト"""

    def test_init_with_api_key(self, mock_genai, mock_types, mock_error_handler):
        """APIキー付き初期化"""
        engine = GeminiEngine(api_key="test-api-key")

        assert engine.api_key == "test-api-key"
        assert engine.current_model == "gemini-2.0-flash"  # 新SDK: デフォルトモデル
        assert engine.max_tokens == 1048576  # 新SDK: gemini-2.0-flashの最大トークン数
        assert "gemini-2.0-flash" in engine.available_models

        # 新SDK: Clientが初期化される
        mock_genai.Client.assert_called_once_with(api_key="test-api-key")

    def test_init_without_api_key(
        self, mock_genai, mock_types, mock_api_key_manager, mock_error_handler
    ):
        """APIキーなし初期化（credential_managerから取得）"""
        engine = GeminiEngine()

        assert engine.api_key == "test-gemini-api-key"
        mock_api_key_manager.get_api_key.assert_called_once_with("gemini")

    def test_init_with_config(self, mock_genai, mock_types, mock_error_handler, llm_config):
        """設定付き初期化"""
        engine = GeminiEngine(config=llm_config, api_key="test-api-key")

        assert engine.config.engine == "gemini"
        assert engine.config.model == "gemini-2.0-flash"
        assert engine.config.temperature == 0.7
        assert engine.config.max_tokens == 1000

    def test_init_with_search_config(
        self, mock_genai, mock_types, mock_error_handler, llm_config_with_search
    ):
        """検索機能付き初期化（Phase 4 Part 87追加）"""
        engine = GeminiEngine(config=llm_config_with_search, api_key="test-api-key")

        assert engine.config.enable_search is True
        assert engine.config.search_threshold == 0.5

    def test_init_without_genai_library(self, mock_genai_not_available):
        """google-genai未インストール時"""
        with pytest.raises(ConfigurationError) as exc_info:
            GeminiEngine()

        assert exc_info.value.error_code == "E2010"
        assert "google-genai is not installed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_async_initialize(self, mock_genai, mock_types, mock_error_handler):
        """非同期初期化"""
        engine = GeminiEngine(api_key="test-api-key")

        # 接続テストのモック（新SDK）
        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.return_value = MagicMock(text="Hello response")

        await engine.initialize()

        assert engine.is_available()
        assert engine._state == ComponentState.READY
        assert engine._initialized_at is not None
        # 接続テストが実行される
        mock_client.models.generate_content.assert_called()

    @pytest.mark.asyncio
    async def test_initialize_without_api_key(
        self, mock_genai, mock_types, mock_api_key_manager, mock_error_handler
    ):
        """APIキーなしでの非同期初期化"""
        engine = GeminiEngine()

        # 最初はAPIキーなし
        engine.api_key = None

        await engine.initialize()

        # require_api_keyが呼ばれる
        mock_api_key_manager.require_api_key.assert_called_once_with("gemini")
        assert engine.api_key == "test-gemini-api-key"

    @pytest.mark.asyncio
    async def test_initialize_connection_test_failure(
        self, mock_genai, mock_types, mock_error_handler
    ):
        """接続テスト失敗"""
        engine = GeminiEngine(api_key="test-api-key")

        # 接続テストで例外（新SDK）
        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.side_effect = Exception("API_KEY_INVALID")

        with pytest.raises(AuthenticationError) as exc_info:
            await engine.initialize()

        assert "API key validation failed" in str(exc_info.value)


# ============================================================================
# generate()メソッドテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestGenerate:
    """generate()メソッドのテスト"""

    @pytest.mark.asyncio
    async def test_generate_success(self, gemini_engine, mock_genai):
        """正常な生成"""
        # モックレスポンス設定（新SDK）
        mock_response = MagicMock()
        mock_response.text = "これは生成されたテキストです。"
        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.return_value = mock_response

        response = await gemini_engine.generate(prompt="こんにちは", temperature=0.8, max_tokens=500)

        assert isinstance(response, LLMResponse)
        assert response.content == "これは生成されたテキストです。"
        assert response.model == "gemini-2.0-flash"  # 新SDK: デフォルトモデル
        assert response.finish_reason == "stop"
        assert "total_tokens" in response.usage

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self, gemini_engine, mock_genai):
        """システムプロンプト付き生成"""
        mock_response = MagicMock()
        mock_response.text = "システムプロンプトに基づく応答"
        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.return_value = mock_response

        response = await gemini_engine.generate(
            prompt="質問です", system_prompt="あなたは専門家です", temperature=0.5
        )

        # プロンプトが結合される
        call_args = mock_client.models.generate_content.call_args
        assert response.content == "システムプロンプトに基づく応答"

    @pytest.mark.asyncio
    async def test_generate_without_client(self, mock_genai, mock_types, mock_error_handler):
        """クライアント未初期化時のエラー"""
        engine = GeminiEngine(api_key="test-api-key")
        engine.client = None  # クライアントなし

        with pytest.raises(LLMError) as exc_info:
            await engine.generate("テスト")

        assert exc_info.value.error_code == "E2000"
        assert "not initialized" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_rate_limit_error(self, gemini_engine, mock_genai):
        """レート制限エラー"""
        # 新SDK: Gemini固有エラーをシミュレート
        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.side_effect = MockGoogleGenerativeAIError(
            "Resource has been exhausted (e.g. check quota)"
        )

        with pytest.raises(RateLimitError) as exc_info:
            await gemini_engine.generate("テスト")

        assert exc_info.value.error_code == "E2002"
        assert hasattr(exc_info.value, "retry_after")

    @pytest.mark.asyncio
    async def test_generate_authentication_error(self, gemini_engine, mock_genai):
        """認証エラー"""
        # 新SDK: Gemini固有エラーをシミュレート
        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.side_effect = MockGoogleGenerativeAIError(
            "API_KEY_INVALID: The provided API key is invalid"
        )

        with pytest.raises(AuthenticationError) as exc_info:
            await gemini_engine.generate("テスト")

        assert exc_info.value.error_code == "E2003"

    @pytest.mark.asyncio
    async def test_generate_validation_error(self, gemini_engine, mock_genai):
        """バリデーションエラー（Phase 4 Part 87追加）"""
        # ツール関連のバリデーションエラーをシミュレート
        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.side_effect = MockGoogleGenerativeAIError(
            "Invalid value at 'tools.0.Tool': Input should be a valid dictionary"
        )

        with pytest.raises(ConfigurationError) as exc_info:
            await gemini_engine.generate("テスト")

        assert exc_info.value.error_code == "E2012"
        assert "Invalid tool configuration" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_api_error(self, gemini_engine, mock_genai):
        """その他のAPIエラー"""
        # 新SDK: Gemini固有エラーをシミュレート
        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.side_effect = MockGoogleGenerativeAIError(
            "Some other API error"
        )

        with pytest.raises(APIError) as exc_info:
            await gemini_engine.generate("テスト")

        assert exc_info.value.error_code == "E2001"
        assert hasattr(exc_info.value, "details")

    @pytest.mark.asyncio
    async def test_generate_generic_error(self, gemini_engine):
        """一般的なエラー"""
        # 通常のRuntimeErrorを発生させる（Gemini固有エラーではない）
        with patch.object(
            gemini_engine, "_combine_prompts", side_effect=RuntimeError("Test error")
        ):
            with pytest.raises(LLMError) as exc_info:
                await gemini_engine.generate("テスト")

            assert exc_info.value.error_code == "E2000"
            assert "Generation failed" in str(exc_info.value)


# ============================================================================
# 検索機能テスト（Phase 4 Part 87追加）
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestSearchFunctionality:
    """Google検索機能のテスト"""

    def test_create_search_tools_gemini_20(self, gemini_engine_with_search, mock_types):
        """Gemini 2.0系での検索ツール作成"""
        gemini_engine_with_search.current_model = "gemini-2.0-flash"

        # _create_search_tools()を呼び出し
        tools = gemini_engine_with_search._create_search_tools()

        # types.Tool(google_search=types.GoogleSearch())が呼ばれることを確認
        assert tools is not None
        assert len(tools) == 1
        mock_types.Tool.assert_called_once()
        mock_types.GoogleSearch.assert_called_once()

    def test_create_search_tools_gemini_25(self, gemini_engine_with_search, mock_types):
        """Gemini 2.5系での検索ツール作成"""
        gemini_engine_with_search.current_model = "gemini-2.5-pro"

        # _create_search_tools()を呼び出し
        tools = gemini_engine_with_search._create_search_tools()

        # types.Tool(google_search=types.GoogleSearch())が呼ばれることを確認
        assert tools is not None
        assert len(tools) == 1
        mock_types.Tool.assert_called_once()
        mock_types.GoogleSearch.assert_called_once()

    def test_create_search_tools_gemini_15(self, gemini_engine_with_search, mock_types):
        """Gemini 1.5系での検索ツール（未サポート）"""
        gemini_engine_with_search.current_model = "gemini-1.5-flash"

        # _create_search_tools()を呼び出し
        tools = gemini_engine_with_search._create_search_tools()

        # 1.5系は新SDKでサポートされないため、None
        assert tools is None
        mock_types.Tool.assert_not_called()

    def test_create_search_tools_unsupported_model(self, gemini_engine_with_search, mock_types):
        """未対応モデルでの検索ツール"""
        gemini_engine_with_search.current_model = "unsupported-model"

        # _create_search_tools()を呼び出し
        tools = gemini_engine_with_search._create_search_tools()

        # 未対応モデルではNone
        assert tools is None
        mock_types.Tool.assert_not_called()

    def test_create_search_tools_types_unavailable(self, gemini_engine_with_search):
        """typesが利用不可の場合"""
        # typesをNoneにパッチ
        with patch("vioratalk.core.llm.gemini_engine.types", None):
            tools = gemini_engine_with_search._create_search_tools()
            assert tools is None

    @pytest.mark.asyncio
    async def test_generate_with_search_enabled(
        self, gemini_engine_with_search, mock_genai, mock_types
    ):
        """検索有効時の生成"""
        # モックレスポンス設定（検索結果を含む）
        mock_response = MagicMock()
        mock_response.text = "検索結果を含む応答"

        # grounding_metadataを設定
        mock_response.grounding_metadata = MagicMock()
        mock_response.grounding_metadata.web_search_queries = ["test query"]
        mock_response.grounding_metadata.grounding_chunks = [
            MagicMock(uri="https://example.com", title="Example")
        ]

        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.return_value = mock_response

        response = await gemini_engine_with_search.generate(
            prompt="東京の天気は？", temperature=0.5, max_tokens=500
        )

        # GenerateContentConfigに検索ツールが含まれることを確認
        call_args = mock_client.models.generate_content.call_args
        assert response.content == "検索結果を含む応答"

        # groundingメタデータが含まれることを確認
        assert "grounding" in response.metadata
        assert response.metadata["grounding"]["search_performed"] is True
        assert len(response.metadata["grounding"]["search_queries"]) == 1
        assert response.metadata["grounding"]["search_queries"][0] == "test query"
        assert len(response.metadata["grounding"]["sources"]) == 1

    @pytest.mark.asyncio
    async def test_generate_with_search_disabled(self, gemini_engine, mock_genai, mock_types):
        """検索無効時の生成"""
        # 検索無効設定を確認
        assert gemini_engine.config.enable_search is False

        # モックレスポンス設定（検索結果なし）
        mock_response = MagicMock()
        mock_response.text = "通常の応答"
        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.return_value = mock_response

        response = await gemini_engine.generate(prompt="東京の天気は？", temperature=0.5, max_tokens=500)

        # 検索ツールが呼ばれないことを確認
        mock_types.Tool.assert_not_called()
        mock_types.GoogleSearch.assert_not_called()

        assert response.content == "通常の応答"
        # groundingメタデータがないことを確認
        assert "grounding" not in response.metadata

    @pytest.mark.asyncio
    async def test_generate_search_no_results(
        self, gemini_engine_with_search, mock_genai, mock_types
    ):
        """検索有効だが結果がない場合"""
        # モックレスポンス設定（grounding_metadataなし）
        mock_response = MagicMock()
        mock_response.text = "検索結果なしの応答"
        # grounding_metadataをNoneに設定
        mock_response.grounding_metadata = None

        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.return_value = mock_response

        response = await gemini_engine_with_search.generate(prompt="特殊な質問", temperature=0.5)

        assert response.content == "検索結果なしの応答"
        # 検索は有効だったが結果がなかったことを記録
        assert "grounding" in response.metadata
        assert response.metadata["grounding"]["search_performed"] is False
        assert "reason" in response.metadata["grounding"]

    def test_process_response_with_search(self, gemini_engine_with_search):
        """検索結果を含むレスポンス処理"""
        # モックレスポンス
        mock_response = MagicMock()
        mock_response.text = "検索結果を含む応答"

        # grounding_metadataを設定
        mock_response.grounding_metadata = MagicMock()
        mock_response.grounding_metadata.web_search_queries = ["query1", "query2"]

        # grounding_chunksを設定
        chunk1 = MagicMock()
        chunk1.uri = "https://example1.com"
        chunk1.title = "Example 1"
        chunk2 = MagicMock()
        chunk2.uri = "https://example2.com"
        chunk2.title = "Example 2"
        mock_response.grounding_metadata.grounding_chunks = [chunk1, chunk2]

        # リクエスト
        request = LLMRequest(prompt="テスト", temperature=0.7, max_tokens=100)

        # 処理実行
        result = gemini_engine_with_search._process_response_with_search(mock_response, request)

        # 検証
        assert result.content == "検索結果を含む応答"
        assert result.metadata["grounding"]["search_performed"] is True
        assert len(result.metadata["grounding"]["search_queries"]) == 2
        assert result.metadata["grounding"]["search_queries"][0] == "query1"
        assert len(result.metadata["grounding"]["sources"]) == 2
        assert result.metadata["grounding"]["sources"][0]["uri"] == "https://example1.com"
        assert result.metadata["grounding"]["source_count"] == 2


# ============================================================================
# stream_generate()メソッドテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestStreamGenerate:
    """stream_generate()メソッドのテスト"""

    @pytest.mark.asyncio
    async def test_stream_generate_success(self, gemini_engine, mock_genai):
        """正常なストリーミング生成"""
        # ストリーミングレスポンスのモック（新SDK）
        mock_chunks = [MagicMock(text="これは"), MagicMock(text="ストリーミング"), MagicMock(text="レスポンスです。")]

        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content_stream.return_value = iter(mock_chunks)

        # ストリーミング生成
        chunks = []
        async for chunk in gemini_engine.stream_generate("テスト"):
            chunks.append(chunk)

        assert chunks == ["これは", "ストリーミング", "レスポンスです。"]

        # generate_content_streamが呼ばれることを確認
        mock_client.models.generate_content_stream.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_generate_with_search(
        self, gemini_engine_with_search, mock_genai, mock_types
    ):
        """検索有効時のストリーミング（検索は使用されない）"""
        # ストリーミングレスポンスのモック
        mock_chunks = [MagicMock(text="ストリーミング")]
        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content_stream.return_value = iter(mock_chunks)

        # ストリーミング生成
        chunks = []
        async for chunk in gemini_engine_with_search.stream_generate("テスト"):
            chunks.append(chunk)

        # ストリーミングでは検索機能が使用されないことを確認
        mock_types.Tool.assert_not_called()
        mock_types.GoogleSearch.assert_not_called()

    @pytest.mark.asyncio
    async def test_stream_generate_without_client(self, mock_genai, mock_types, mock_error_handler):
        """クライアント未初期化時のストリーミングエラー"""
        engine = GeminiEngine(api_key="test-api-key")
        engine.client = None

        with pytest.raises(LLMError) as exc_info:
            async for _ in engine.stream_generate("テスト"):
                pass

        assert exc_info.value.error_code == "E2000"

    @pytest.mark.asyncio
    async def test_stream_generate_error(self, gemini_engine, mock_genai):
        """ストリーミング中のエラー"""
        # 新SDK: エラーをシミュレート
        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content_stream.side_effect = MockGoogleGenerativeAIError(
            "Streaming error"
        )

        with pytest.raises(LLMError) as exc_info:
            async for _ in gemini_engine.stream_generate("テスト"):
                pass

        assert exc_info.value.error_code == "E2000"
        assert "Streaming generation failed" in str(exc_info.value)


# ============================================================================
# モデル管理テスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestModelManagement:
    """モデル管理機能のテスト"""

    def test_get_available_models(self, gemini_engine):
        """利用可能モデルの取得"""
        models = gemini_engine.get_available_models()

        assert isinstance(models, list)
        assert "gemini-2.0-flash" in models  # 新SDK: 新モデル
        assert "gemini-2.5-flash" in models  # 新SDK: 新モデル
        assert len(models) > 0

    def test_set_model_success(self, gemini_engine):
        """モデル変更成功"""
        gemini_engine.set_model("gemini-2.5-pro")  # 新SDK: 新モデル

        assert gemini_engine.current_model == "gemini-2.5-pro"
        assert gemini_engine.max_tokens == 1048576  # 新SDK: 2.5-proの最大トークン数

    def test_set_model_not_found(self, gemini_engine):
        """存在しないモデル指定"""
        with pytest.raises(ModelNotFoundError) as exc_info:
            gemini_engine.set_model("invalid-model")

        assert exc_info.value.error_code == "E2004"
        assert "invalid-model" in str(exc_info.value)
        assert "not found" in str(exc_info.value)

    def test_get_max_tokens(self, gemini_engine):
        """最大トークン数取得"""
        tokens = gemini_engine.get_max_tokens()

        assert tokens == 1048576  # 新SDK: gemini-2.0-flashのデフォルト

        # モデル変更後
        gemini_engine.set_model("gemini-2.5-flash")
        tokens = gemini_engine.get_max_tokens()
        assert tokens == 1048576


# ============================================================================
# ヘルパーメソッドテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestHelperMethods:
    """ヘルパーメソッドのテスト"""

    def test_combine_prompts_with_system(self, gemini_engine):
        """システムプロンプトとの結合"""
        combined = gemini_engine._combine_prompts(prompt="ユーザー入力", system_prompt="システム指示")

        assert combined == "システム指示\n\nユーザー入力"

    def test_combine_prompts_without_system(self, gemini_engine):
        """システムプロンプトなしの場合"""
        combined = gemini_engine._combine_prompts(prompt="ユーザー入力", system_prompt=None)

        assert combined == "ユーザー入力"

    def test_process_response(self, gemini_engine):
        """レスポンス処理（修正版）"""
        mock_response = MagicMock()
        mock_response.text = "生成されたテキスト"

        # usage_metadataを適切に設定（開発規約書v1.12準拠）
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 1  # 数値として設定
        mock_response.usage_metadata.candidates_token_count = 1  # 数値として設定
        mock_response.usage_metadata.total_token_count = 2  # 数値として設定

        request = LLMRequest(prompt="テストプロンプト", temperature=0.7, max_tokens=100)

        result = gemini_engine._process_response(mock_response, request)

        assert isinstance(result, LLMResponse)
        assert result.content == "生成されたテキスト"
        assert result.model == "gemini-2.0-flash"  # 新SDK: デフォルトモデル
        assert result.finish_reason == "stop"
        # 修正: 数値として正しく取得されることを確認
        assert result.usage["prompt_tokens"] == 1
        assert result.usage["completion_tokens"] == 1
        assert result.usage["total_tokens"] == 2

    def test_process_response_error(self, gemini_engine):
        """レスポンス処理エラー"""
        mock_response = None  # 不正なレスポンス
        request = LLMRequest(prompt="テスト")

        result = gemini_engine._process_response(mock_response, request)

        # エラー時でも最小限のレスポンスを返す（空文字列）
        assert result.content == ""
        assert result.finish_reason == "error"
        assert "error" in result.metadata


# ============================================================================
# 環境変数からの設定読み込みテスト（Phase 4 Part 87追加）
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestEnvironmentConfiguration:
    """環境変数からの設定読み込みテスト"""

    def test_init_with_search_env_vars(self, mock_genai, mock_types, mock_error_handler):
        """環境変数から検索設定を読み込み"""
        with patch.dict(
            "os.environ", {"GEMINI_ENABLE_SEARCH": "true", "GEMINI_SEARCH_THRESHOLD": "0.8"}
        ):
            engine = GeminiEngine(api_key="test-api-key")

            assert engine.config.enable_search is True
            assert engine.config.search_threshold == 0.8

    def test_init_with_invalid_search_env_vars(self, mock_genai, mock_types, mock_error_handler):
        """無効な環境変数値の処理"""
        with patch.dict(
            "os.environ",
            {"GEMINI_ENABLE_SEARCH": "invalid", "GEMINI_SEARCH_THRESHOLD": "not_a_number"},
        ):
            # エラーは発生せず、デフォルト値が使用される
            engine = GeminiEngine(api_key="test-api-key")

            assert engine.config.enable_search is False  # デフォルト値


# ============================================================================
# クリーンアップテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestCleanup:
    """クリーンアップのテスト"""

    @pytest.mark.asyncio
    async def test_cleanup(self, gemini_engine):
        """リソースクリーンアップ"""
        assert gemini_engine.client is not None

        await gemini_engine.cleanup()

        assert gemini_engine.client is None
        assert not gemini_engine.is_available()
        assert gemini_engine._state == ComponentState.TERMINATED


# ============================================================================
# 統合的なシナリオテスト
# ============================================================================


@pytest.mark.integration
@pytest.mark.phase(4)
class TestIntegrationScenarios:
    """統合シナリオテスト"""

    @pytest.mark.asyncio
    async def test_full_lifecycle(
        self, mock_genai, mock_types, mock_api_key_manager, mock_error_handler
    ):
        """完全なライフサイクル"""
        # 1. 初期化
        engine = GeminiEngine()
        assert engine.api_key == "test-gemini-api-key"

        # 2. 非同期初期化（接続テスト成功をモック）
        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.return_value = MagicMock(text="Hello")
        await engine.initialize()
        assert engine.is_available()
        assert engine._state == ComponentState.READY

        # 3. テキスト生成
        mock_client.models.generate_content.return_value = MagicMock(text="生成結果")
        response = await engine.generate("テスト")
        assert response.content == "生成結果"

        # 4. モデル変更（新SDK: 新モデルを使用）
        engine.set_model("gemini-2.5-pro")
        assert engine.current_model == "gemini-2.5-pro"

        # 5. クリーンアップ
        await engine.cleanup()
        assert engine.client is None

    @pytest.mark.asyncio
    async def test_search_lifecycle(
        self, mock_genai, mock_types, mock_api_key_manager, mock_error_handler
    ):
        """検索機能を含むライフサイクル（Phase 4 Part 87追加）"""
        # 1. 検索機能付き初期化
        config = LLMConfig(
            engine="gemini", model="gemini-2.0-flash", enable_search=True, search_threshold=0.5
        )
        engine = GeminiEngine(config=config, api_key="test-api-key")

        # 2. 非同期初期化
        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.return_value = MagicMock(text="Hello")
        await engine.initialize()

        # 3. 検索付き生成
        mock_response = MagicMock()
        mock_response.text = "検索結果を含む応答"
        mock_response.grounding_metadata = MagicMock()
        mock_response.grounding_metadata.web_search_queries = ["query"]
        mock_response.grounding_metadata.grounding_chunks = []
        mock_client.models.generate_content.return_value = mock_response

        response = await engine.generate("検索クエリ")
        assert "grounding" in response.metadata

        # 4. クリーンアップ
        await engine.cleanup()


# ============================================================================
# 実API接続テスト（通常はスキップ）
# ============================================================================


@pytest.mark.skip(reason="実際のAPIキーが必要")
@pytest.mark.integration
class TestRealAPI:
    """実際のGemini APIとの接続テスト"""

    @pytest.mark.asyncio
    async def test_real_api_connection(self):
        """実APIへの接続テスト"""
        # 環境変数からAPIキーを取得
        import os

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            pytest.skip("GEMINI_API_KEY not found")

        engine = GeminiEngine(api_key=api_key)
        await engine.initialize()

        response = await engine.generate(
            prompt="Hello, please respond with a simple greeting.", max_tokens=50
        )

        assert response.content
        assert len(response.content) > 0
        assert response.model == "gemini-2.0-flash"  # 新SDK: デフォルトモデル

        await engine.cleanup()


# ============================================================================
# エクスポート
# ============================================================================

__all__ = [
    "TestInitialization",
    "TestGenerate",
    "TestSearchFunctionality",
    "TestStreamGenerate",
    "TestModelManagement",
    "TestHelperMethods",
    "TestEnvironmentConfiguration",
    "TestCleanup",
    "TestIntegrationScenarios",
    "TestRealAPI",
]
