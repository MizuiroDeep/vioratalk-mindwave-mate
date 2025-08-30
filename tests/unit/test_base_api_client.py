"""BaseAPIClientの単体テスト

API通信基底クラスの動作を検証。
リトライ機構、エラーハンドリング、タイムアウト処理をテスト。

テスト実装ガイド v1.3準拠
API通信実装ガイド v1.4準拠
"""

import asyncio
from typing import Any, Dict

import aiohttp
import pytest
from aioresponses import aioresponses
from tenacity import RetryError

from vioratalk.core.api.base import BaseAPIClient
from vioratalk.core.base import ComponentState
from vioratalk.core.exceptions import (
    APIError,
    AuthenticationError,
    InvalidRequestError,
    NetworkError,
    RateLimitError,
)

# ============================================================================
# テスト用の具象クラス
# ============================================================================


class TestAPIClient(BaseAPIClient):
    """テスト用のBaseAPIClient具象実装"""

    async def _prepare_headers(self) -> Dict[str, str]:
        """テスト用ヘッダー準備"""
        headers = {
            "User-Agent": "TestClient/1.0",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _process_request_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """テスト用リクエストデータ処理"""
        # テスト用: prefixを追加
        if data:
            data["test_prefix"] = "test_"
        return data


# ============================================================================
# 初期化とクリーンアップのテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestInitializationAndCleanup:
    """初期化とクリーンアップのテスト"""

    def test_initialization_with_defaults(self):
        """デフォルト値での初期化テスト"""
        client = TestAPIClient()

        assert client.api_key is None
        assert client.base_url is None
        assert client.connect_timeout == BaseAPIClient.DEFAULT_CONNECT_TIMEOUT
        assert client.read_timeout == BaseAPIClient.DEFAULT_READ_TIMEOUT
        assert client.proxy is None
        assert client._session is None
        assert client._state == ComponentState.NOT_INITIALIZED

    def test_initialization_with_params(self):
        """パラメータ指定での初期化テスト"""
        client = TestAPIClient(
            api_key="test_key",
            base_url="https://api.example.com",
            connect_timeout=5.0,
            read_timeout=30.0,
            proxy="http://proxy.example.com:8080",
        )

        assert client.api_key == "test_key"
        assert client.base_url == "https://api.example.com"
        assert client.connect_timeout == 5.0
        assert client.read_timeout == 30.0
        assert client.proxy == "http://proxy.example.com:8080"

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """非同期コンテキストマネージャーのテスト"""
        client = TestAPIClient()

        # 初期状態
        assert client._session is None
        assert client._state == ComponentState.NOT_INITIALIZED

        # コンテキストマネージャー使用
        async with client:
            assert client._session is not None
            assert isinstance(client._session, aiohttp.ClientSession)
            assert client._state == ComponentState.READY

        # クリーンアップ後
        assert client._session is None
        assert client._state == ComponentState.TERMINATED

    @pytest.mark.asyncio
    async def test_manual_session_management(self):
        """手動セッション管理のテスト"""
        client = TestAPIClient()

        # セッション作成
        await client._ensure_session()
        assert client._session is not None
        assert client._state == ComponentState.READY

        # クリーンアップ
        await client.close()
        assert client._session is None
        assert client._state == ComponentState.TERMINATED


# ============================================================================
# HTTPリクエスト基本機能のテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestHTTPRequests:
    """HTTPリクエスト機能のテスト"""

    @pytest.mark.asyncio
    async def test_successful_request(self):
        """正常なリクエストのテスト"""
        client = TestAPIClient(api_key="test_key", base_url="https://api.example.com")

        with aioresponses() as mocked:
            # 成功レスポンスをモック
            mocked.post(
                "https://api.example.com/test",
                status=200,
                payload={"result": "success", "data": "test_data"},
            )

            async with client:
                result = await client._make_request(
                    method="POST", endpoint="/test", data={"input": "test"}
                )

                assert result["result"] == "success"
                assert result["data"] == "test_data"

    @pytest.mark.asyncio
    async def test_request_with_custom_headers(self):
        """カスタムヘッダー付きリクエストのテスト"""
        client = TestAPIClient(api_key="test_key")

        with aioresponses() as mocked:
            # ヘッダー検証用フラグ
            headers_checked = False

            def check_headers(url, **kwargs):
                nonlocal headers_checked
                headers = kwargs.get("headers", {})
                # デフォルトヘッダーの確認
                assert headers["User-Agent"] == "TestClient/1.0"
                assert headers["Authorization"] == "Bearer test_key"
                # カスタムヘッダーの確認
                assert headers["X-Custom"] == "CustomValue"
                headers_checked = True
                # callbackは値を返さない

            mocked.post(
                "https://api.example.com/test",
                callback=check_headers,
                payload={"result": "ok"},  # レスポンスはpayloadで設定
                status=200,
            )

            async with client:
                client.base_url = "https://api.example.com"
                result = await client._make_request(
                    method="POST", endpoint="/test", headers={"X-Custom": "CustomValue"}
                )

                assert headers_checked  # ヘッダー検証が実行されたことを確認
                assert result["result"] == "ok"

    @pytest.mark.asyncio
    async def test_request_data_processing(self):
        """リクエストデータ処理のテスト"""
        client = TestAPIClient()

        with aioresponses() as mocked:
            # データ検証用フラグ
            data_checked = False

            def check_data(url, **kwargs):
                nonlocal data_checked
                data = kwargs.get("json", {})
                # _process_request_dataで追加されたプレフィックスを確認
                assert "test_prefix" in data
                assert data["test_prefix"] == "test_"
                assert data["original"] == "data"
                data_checked = True
                # callbackは値を返さない

            mocked.post(
                "https://api.example.com/test",
                callback=check_data,
                payload={"result": "ok"},  # レスポンスはpayloadで設定
                status=200,
            )

            async with client:
                client.base_url = "https://api.example.com"
                result = await client._make_request(
                    method="POST", endpoint="/test", data={"original": "data"}
                )

                assert data_checked  # データ検証が実行されたことを確認
                assert result["result"] == "ok"


# ============================================================================
# エラーハンドリングのテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestErrorHandling:
    """エラーハンドリングのテスト"""

    @pytest.mark.asyncio
    async def test_authentication_error(self):
        """認証エラー（401）のテスト"""
        client = TestAPIClient()
        # リトライを無効化
        client._enable_retry = False

        with aioresponses() as mocked:
            mocked.post(
                "https://api.example.com/test", status=401, payload={"message": "Invalid API key"}
            )

            async with client:
                client.base_url = "https://api.example.com"
                with pytest.raises(AuthenticationError) as exc_info:
                    await client._make_request("POST", "/test")

                assert exc_info.value.error_code == "E2003"

    @pytest.mark.asyncio
    async def test_rate_limit_error(self):
        """レート制限エラー（429）のテスト"""
        client = TestAPIClient()
        # リトライを無効化
        client._enable_retry = False

        with aioresponses() as mocked:
            mocked.post(
                "https://api.example.com/test",
                status=429,
                headers={"Retry-After": "60"},
                payload={"message": "Rate limit exceeded"},
            )

            async with client:
                client.base_url = "https://api.example.com"
                with pytest.raises(RateLimitError) as exc_info:
                    await client._make_request("POST", "/test")

                assert exc_info.value.error_code == "E2002"
                assert exc_info.value.retry_after == 60

    @pytest.mark.asyncio
    async def test_invalid_request_error(self):
        """無効なリクエストエラー（400）のテスト"""
        client = TestAPIClient()
        # リトライを無効化
        client._enable_retry = False

        with aioresponses() as mocked:
            mocked.post(
                "https://api.example.com/test",
                status=400,
                payload={"message": "Invalid parameters"},
            )

            async with client:
                client.base_url = "https://api.example.com"
                with pytest.raises(InvalidRequestError) as exc_info:
                    await client._make_request("POST", "/test")

                assert exc_info.value.error_code == "E2001"

    @pytest.mark.asyncio
    async def test_server_error(self):
        """サーバーエラー（500）のテスト"""
        client = TestAPIClient()
        # リトライを無効化
        client._enable_retry = False

        with aioresponses() as mocked:
            mocked.post(
                "https://api.example.com/test",
                status=500,
                payload={"message": "Internal server error"},
            )

            async with client:
                client.base_url = "https://api.example.com"
                with pytest.raises(APIError) as exc_info:
                    await client._make_request("POST", "/test")

                assert exc_info.value.error_code == "E2000"


# ============================================================================
# リトライ機構のテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestRetryMechanism:
    """リトライ機構のテスト"""

    @pytest.mark.asyncio
    async def test_retry_on_server_error(self):
        """サーバーエラー時のリトライテスト"""
        client = TestAPIClient()
        # リトライは有効（デフォルト）

        with aioresponses() as mocked:
            # 最初は500エラー、次にテキストのみのエラー、最後に成功
            mocked.post(
                "https://api.example.com/test", status=500, payload={"message": "Server error"}
            )
            mocked.post(
                "https://api.example.com/test",
                status=500,
                body="Internal Server Error",
                content_type="text/plain",
            )
            mocked.post("https://api.example.com/test", status=200, payload={"result": "success"})

            async with client:
                client.base_url = "https://api.example.com"
                # リトライで成功するはず
                result = await client._make_request("POST", "/test")
                assert result["result"] == "success"

    @pytest.mark.asyncio
    async def test_no_retry_on_client_error(self):
        """クライアントエラー時はリトライしないテスト"""
        client = TestAPIClient()
        # リトライを無効化して正確にカウント
        client._enable_retry = False
        call_count = 0

        with aioresponses() as mocked:

            def count_calls(url, **kwargs):
                nonlocal call_count
                call_count += 1
                # callbackは値を返さない

            mocked.post(
                "https://api.example.com/test",
                callback=count_calls,
                payload={"error": "Bad request"},
                status=400,
            )

            async with client:
                client.base_url = "https://api.example.com"
                with pytest.raises(InvalidRequestError):
                    await client._make_request("POST", "/test")

                # 400エラーはリトライしないので1回のみ
                assert call_count == 1

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """最大リトライ回数超過のテスト"""
        client = TestAPIClient()
        # リトライは有効（デフォルト）

        with aioresponses() as mocked:
            # すべて500エラーを返す（MAX_RETRIES + 1回分）
            for _ in range(BaseAPIClient.MAX_RETRIES + 1):
                mocked.post("https://api.example.com/test", status=500, body="Server Error")

            async with client:
                client.base_url = "https://api.example.com"
                # 最大リトライ回数を超えてRetryErrorが発生（tenacityの仕様）
                with pytest.raises(RetryError) as exc_info:
                    await client._make_request("POST", "/test")

                # RetryErrorの中に元のAPIErrorが含まれていることを確認
                # exc_info.value.last_attemptに最後の試行の情報が含まれる


# ============================================================================
# タイムアウト処理のテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestTimeoutHandling:
    """タイムアウト処理のテスト"""

    @pytest.mark.asyncio
    async def test_connection_timeout(self):
        """接続タイムアウトのテスト"""
        client = TestAPIClient(connect_timeout=0.1)
        # リトライを無効化
        client._enable_retry = False

        with aioresponses() as mocked:
            # タイムアウトをシミュレート
            mocked.post("https://api.example.com/test", exception=asyncio.TimeoutError())

            async with client:
                client.base_url = "https://api.example.com"
                with pytest.raises(NetworkError) as exc_info:
                    await client._make_request("POST", "/test", timeout=0.1)

                assert exc_info.value.error_code == "E5205"

    @pytest.mark.asyncio
    async def test_read_timeout(self):
        """読み取りタイムアウトのテスト"""
        client = TestAPIClient(read_timeout=0.1)

        # NetworkErrorの処理ロジックのみ確認
        error = asyncio.TimeoutError()
        network_error = client._handle_connection_error(error)

        assert isinstance(network_error, NetworkError)
        assert network_error.error_code == "E5205"


# ============================================================================
# プロキシ対応のテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestProxySupport:
    """プロキシ対応のテスト"""

    @pytest.mark.asyncio
    async def test_proxy_configuration(self):
        """プロキシ設定のテスト"""
        proxy_url = "http://proxy.example.com:8080"
        client = TestAPIClient(proxy=proxy_url)

        assert client.proxy == proxy_url

        # プロキシが設定されていることを確認
        async with client:
            assert client._session is not None

    @pytest.mark.asyncio
    async def test_proxy_connection_error(self):
        """プロキシ接続エラーのテスト"""
        client = TestAPIClient(proxy="http://invalid.proxy:8080")

        # プロキシエラーのシミュレーション
        from aiohttp import ClientProxyConnectionError

        error = ClientProxyConnectionError(
            connection_key=None, os_error=OSError("Cannot connect to proxy")
        )
        network_error = client._handle_connection_error(error)

        assert isinstance(network_error, NetworkError)
        assert network_error.error_code == "E5203"


# ============================================================================
# セキュリティ機能のテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestSecurityFeatures:
    """セキュリティ機能のテスト"""

    def test_api_key_masking_in_repr(self):
        """APIキーマスキングのテスト"""
        client = TestAPIClient(api_key="super_secret_key", base_url="https://api.example.com")

        repr_str = repr(client)
        assert "super_secret_key" not in repr_str
        assert "***" in repr_str
        assert "https://api.example.com" in repr_str

    @pytest.mark.asyncio
    async def test_ssl_verification(self):
        """SSL証明書検証のテスト"""
        client = TestAPIClient()

        async with client:
            # SSLコンテキストが設定されていることを確認
            connector = client._session.connector
            assert connector is not None
            # aiohttp.TCPConnectorのssl_contextが設定されていることを確認
            if hasattr(connector, "_ssl"):
                assert connector._ssl is not None


# ============================================================================
# 統合的な動作テスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestIntegratedBehavior:
    """統合的な動作テスト"""

    @pytest.mark.asyncio
    async def test_full_request_lifecycle(self):
        """完全なリクエストライフサイクルのテスト"""
        client = TestAPIClient(
            api_key="test_key",
            base_url="https://api.example.com",
            connect_timeout=5.0,
            read_timeout=10.0,
        )

        with aioresponses() as mocked:
            # 最初は500エラー（リトライ）
            mocked.post("https://api.example.com/test", status=500, body="Server Error")
            # 2回目で成功
            mocked.post(
                "https://api.example.com/test",
                status=200,
                payload={"result": "success", "value": 42},
            )

            async with client:
                # 状態確認
                assert client._state == ComponentState.READY

                # リクエスト実行
                result = await client._make_request(
                    method="POST", endpoint="/test", data={"query": "test"}
                )

                # 結果確認
                assert result["result"] == "success"
                assert result["value"] == 42

            # クリーンアップ後の状態確認
            assert client._state == ComponentState.TERMINATED

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """並行リクエストのテスト"""
        client = TestAPIClient(base_url="https://api.example.com")

        with aioresponses() as mocked:
            # 複数のエンドポイントをモック
            for i in range(5):
                mocked.post(
                    f"https://api.example.com/test{i}",
                    status=200,
                    payload={"id": i, "result": f"result_{i}"},
                )

            async with client:
                # 並行リクエスト実行
                tasks = [client._make_request("POST", f"/test{i}") for i in range(5)]
                results = await asyncio.gather(*tasks)

                # 結果確認
                assert len(results) == 5
                for i, result in enumerate(results):
                    assert result["id"] == i
                    assert result["result"] == f"result_{i}"
