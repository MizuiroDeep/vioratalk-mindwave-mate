"""API通信基底クラス

すべてのAPI通信クライアントの基底となるクラス。
セキュリティ、タイムアウト、リトライ、エラー処理を統合実装。

API通信実装ガイド v1.4準拠
インターフェース定義書 v1.34準拠
開発規約書 v1.12準拠
"""

# 標準ライブラリ
import asyncio
import logging
import ssl
from abc import abstractmethod
from datetime import datetime
from typing import Any, Callable, Dict, Optional, TypeVar

# サードパーティ
import aiohttp
import certifi
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# プロジェクト内
from vioratalk.core.base import ComponentState, VioraTalkComponent
from vioratalk.core.exceptions import (
    APIError,
    AuthenticationError,
    InvalidRequestError,
    NetworkError,
    RateLimitError,
)
from vioratalk.core.i18n_manager import I18nManager

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BaseAPIClient(VioraTalkComponent):
    """API通信の基底クラス

    セキュリティ、タイムアウト、リトライ、エラーハンドリングを
    統合実装。すべてのAPIクライアントはこのクラスを継承する。

    Attributes:
        api_key: APIキー（BYOK対応）
        base_url: APIベースURL
        connect_timeout: 接続タイムアウト（秒）
        read_timeout: 読み取りタイムアウト（秒）
        proxy: プロキシURL
        _session: aiohttp ClientSession
        _connector: 接続プール管理
        _i18n: 国際化マネージャー

    API通信実装ガイド v1.4準拠
    """

    # デフォルトタイムアウト設定（秒）
    DEFAULT_CONNECT_TIMEOUT = 10.0
    DEFAULT_READ_TIMEOUT = 60.0

    # リトライ設定
    MAX_RETRIES = 3
    INITIAL_WAIT = 1.0  # 初期待機時間（秒）
    MAX_WAIT = 60.0  # 最大待機時間（秒）
    EXPONENTIAL_BASE = 2.0  # 指数バックオフの基数

    # 接続プール設定
    CONNECTION_LIMIT = 100  # 同時接続数制限
    CONNECTION_LIMIT_PER_HOST = 50  # ホストごとの接続数制限
    DNS_TTL = 300  # DNSキャッシュ（秒）

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        connect_timeout: Optional[float] = None,
        read_timeout: Optional[float] = None,
        proxy: Optional[str] = None,
    ):
        """BaseAPIClientの初期化

        Args:
            api_key: APIキー（BYOK）
            base_url: APIベースURL
            connect_timeout: 接続タイムアウト（秒）
            read_timeout: 読み取りタイムアウト（秒）
            proxy: プロキシURL
        """
        super().__init__()
        self.api_key = api_key
        self.base_url = base_url
        self.connect_timeout = connect_timeout or self.DEFAULT_CONNECT_TIMEOUT
        self.read_timeout = read_timeout or self.DEFAULT_READ_TIMEOUT
        self.proxy = proxy

        # セッション関連
        self._session: Optional[aiohttp.ClientSession] = None
        self._connector: Optional[aiohttp.TCPConnector] = None

        # 国際化マネージャー
        self._i18n = I18nManager()

        # 状態管理
        self._state = ComponentState.NOT_INITIALIZED
        self._error = None
        self._initialized_at: Optional[datetime] = None

        # リトライ有効フラグ（テスト用）
        self._enable_retry = True

    # ------------------------------------------------------------------------
    # 非同期コンテキストマネージャー（API通信実装ガイド v1.4準拠）
    # ------------------------------------------------------------------------

    async def __aenter__(self):
        """非同期コンテキストマネージャーのエントリー

        セッションを初期化し、自身を返す。

        Returns:
            BaseAPIClient: 初期化済みのクライアント
        """
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """非同期コンテキストマネージャーの終了

        セッションとコネクターをクリーンアップする。

        Args:
            exc_type: 例外タイプ
            exc_val: 例外値
            exc_tb: トレースバック
        """
        await self.close()

    # ------------------------------------------------------------------------
    # セッション管理
    # ------------------------------------------------------------------------

    async def _ensure_session(self) -> None:
        """セッションが初期化されていることを確認

        セッションが存在しない場合は新規作成する。
        SSL/TLS設定、接続プール、タイムアウトを適切に設定。

        API通信実装ガイド v1.4準拠
        """
        if self._session is None or self._session.closed:
            # タイムアウト設定
            timeout = aiohttp.ClientTimeout(connect=self.connect_timeout, total=self.read_timeout)

            # SSL/TLS設定
            ssl_context = self._create_ssl_context()

            # コネクター設定（SSL/TLS、接続プール）
            self._connector = aiohttp.TCPConnector(
                ssl=ssl_context,
                limit=self.CONNECTION_LIMIT,
                limit_per_host=self.CONNECTION_LIMIT_PER_HOST,
                ttl_dns_cache=self.DNS_TTL,
                force_close=True,  # Keep-Aliveを無効化（安定性優先）
            )

            # デフォルトヘッダー
            headers = self._get_default_headers()

            # セッション作成
            self._session = aiohttp.ClientSession(
                connector=self._connector,
                timeout=timeout,
                headers=headers,
                trust_env=True,  # 環境変数のプロキシ設定を信頼
            )

            # プロキシ設定（必要な場合）
            if self.proxy:
                # プロキシ設定はリクエスト時に個別に適用
                pass

            # 状態を更新
            self._state = ComponentState.READY
            self._initialized_at = datetime.utcnow()

            logger.info(
                f"{self.__class__.__name__} session initialized",
                extra={"initialized_at": self._initialized_at.isoformat()},
            )

    def _create_ssl_context(self) -> ssl.SSLContext:
        """SSL/TLSコンテキストを作成

        Returns:
            ssl.SSLContext: 設定済みのSSLコンテキスト
        """
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        ssl_context.check_hostname = True
        ssl_context.verify_mode = ssl.CERT_REQUIRED
        ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
        return ssl_context

    def _get_default_headers(self) -> Dict[str, str]:
        """デフォルトヘッダーを取得

        Returns:
            Dict[str, str]: デフォルトヘッダー
        """
        return {"User-Agent": "VioraTalk/1.0", "X-Requested-With": "VioraTalk"}

    async def close(self) -> None:
        """セッションとコネクターをクローズ

        リソースを適切にクリーンアップし、状態を更新する。
        """
        if self._session:
            await self._session.close()
            self._session = None

        if self._connector:
            await self._connector.close()
            self._connector = None

        self._state = ComponentState.TERMINATED
        logger.info(f"{self.__class__.__name__} session closed")

    # ------------------------------------------------------------------------
    # リクエスト処理
    # ------------------------------------------------------------------------

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """HTTPリクエストを実行（リトライ機能付き）

        Args:
            method: HTTPメソッド（GET, POST, etc.）
            endpoint: エンドポイントパス
            data: リクエストデータ
            headers: 追加ヘッダー
            timeout: タイムアウト（秒）

        Returns:
            Dict[str, Any]: レスポンスデータ

        Raises:
            APIError: API関連のエラー
            NetworkError: ネットワークエラー
            AuthenticationError: 認証エラー
            RateLimitError: レート制限エラー
            InvalidRequestError: 無効なリクエスト
        """
        # リトライが有効な場合はデコレータを適用
        if self._enable_retry:
            retry_decorator = self._get_retry_decorator()
            request_func = retry_decorator(self._make_request_internal)
        else:
            request_func = self._make_request_internal

        return await request_func(method, endpoint, data, headers, timeout)

    async def _make_request_internal(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """HTTPリクエストの内部実装（リトライなし）

        Args:
            method: HTTPメソッド（GET, POST, etc.）
            endpoint: エンドポイントパス
            data: リクエストデータ
            headers: 追加ヘッダー
            timeout: タイムアウト（秒）

        Returns:
            Dict[str, Any]: レスポンスデータ

        Raises:
            APIError: API関連のエラー
            NetworkError: ネットワークエラー
            AuthenticationError: 認証エラー
            RateLimitError: レート制限エラー
            InvalidRequestError: 無効なリクエスト
        """
        await self._ensure_session()

        # URL構築
        url = f"{self.base_url}{endpoint}" if self.base_url else endpoint

        # ヘッダー準備
        request_headers = await self._prepare_headers()
        if headers:
            request_headers.update(headers)

        # データ処理
        if data:
            data = await self._process_request_data(data)

        # タイムアウト設定
        if timeout:
            timeout_obj = aiohttp.ClientTimeout(total=timeout)
        else:
            timeout_obj = None

        try:
            async with self._session.request(
                method=method,
                url=url,
                headers=request_headers,
                json=data,
                timeout=timeout_obj,
                proxy=self.proxy,
            ) as response:
                # エラーチェック
                if response.status >= 400:
                    await self._handle_response_error(response)

                # 成功レスポンス
                return await response.json()

        except aiohttp.ClientError as e:
            # 接続エラーの処理
            error = self._handle_connection_error(e)
            raise error from e
        except asyncio.TimeoutError as e:
            # タイムアウトエラー
            error_msg = self._i18n.get_error_message("E5205")
            error = NetworkError(error_msg, error_code="E5205")
            raise error from e

    async def _handle_response_error(self, response: aiohttp.ClientResponse) -> None:
        """HTTPエラーレスポンスの処理

        Args:
            response: エラーレスポンス

        Raises:
            適切な例外クラス
        """
        status = response.status

        try:
            error_data = await response.json()
        except:
            error_data = {"message": await response.text()}

        # 4xx系エラー
        if 400 <= status < 500:
            if status == 401:
                error_msg = self._i18n.get_error_message("E2003")
                raise AuthenticationError(error_msg)
            elif status == 429:
                retry_after = int(response.headers.get("Retry-After", 60))
                error_msg = self._i18n.get_error_message("E2002", retry_after=retry_after)
                raise RateLimitError(error_msg, retry_after=retry_after)
            else:
                error_detail = (
                    error_data.get("message", "Unknown error") if error_data else "Unknown error"
                )
                error_msg = self._i18n.get_error_message("E2001", error=error_detail)
                raise InvalidRequestError(error_msg)

        # 5xx系エラー（リトライ可能）
        elif 500 <= status < 600:
            error_detail = (
                error_data.get("message", "Unknown error") if error_data else "Unknown error"
            )
            error_msg = self._i18n.get_error_message("E2000", error=error_detail)
            raise APIError(error_msg, status_code=status, error_code="E2000")

    def _handle_connection_error(self, error: aiohttp.ClientError) -> NetworkError:
        """接続エラーのハンドリング

        Args:
            error: aiohttp例外

        Returns:
            NetworkError: 適切なNetworkError

        Note:
            API通信実装ガイド v1.4準拠
        """
        if isinstance(error, aiohttp.ClientProxyConnectionError):
            error_msg = self._i18n.get_error_message("E5203")
            return NetworkError(error_msg, error_code="E5203")
        elif isinstance(error, aiohttp.ClientConnectorError):
            error_msg = self._i18n.get_error_message("E5502")
            return NetworkError(error_msg, error_code="E5502")
        elif isinstance(error, asyncio.TimeoutError):
            error_msg = self._i18n.get_error_message("E5205")
            return NetworkError(error_msg, error_code="E5205")
        else:
            error_msg = self._i18n.get_error_message("E5201", error=str(error))
            return NetworkError(error_msg, error_code="E5201")

    # ------------------------------------------------------------------------
    # VioraTalkComponent必須メソッド
    # ------------------------------------------------------------------------

    async def initialize(self) -> None:
        """コンポーネントの初期化

        HTTPセッションと接続プールを初期化する。
        """
        if self._state not in [ComponentState.NOT_INITIALIZED, ComponentState.ERROR]:
            raise RuntimeError(f"Cannot initialize component in state: {self._state.value}")

        self._state = ComponentState.INITIALIZING

        try:
            await self._ensure_session()

            self._state = ComponentState.READY
            self._initialized_at = datetime.utcnow()

            logger.info(
                f"{self.__class__.__name__} initialization completed",
                extra={"initialized_at": self._initialized_at.isoformat()},
            )

        except Exception as e:
            self._state = ComponentState.ERROR
            self._error = e
            logger.error(
                f"[E5000] Initialization error: {e}",
                extra={"error_type": type(e).__name__},
                exc_info=True,
            )
            raise

    async def cleanup(self) -> None:
        """リソースのクリーンアップ

        HTTPセッションと接続プールを適切にクローズする。
        """
        if self._state == ComponentState.TERMINATED:
            return

        self._state = ComponentState.TERMINATING

        try:
            await self.close()

            self._state = ComponentState.TERMINATED
            logger.info(f"{self.__class__.__name__} cleanup completed")

        except Exception as e:
            logger.error(f"Cleanup error: {e}", extra={"error_type": type(e).__name__})

    def _get_retry_decorator(self) -> Callable:
        """リトライデコレータを取得

        Returns:
            tenacityのretryデコレータ
        """
        return retry(
            stop=stop_after_attempt(self.MAX_RETRIES),
            wait=wait_exponential(
                multiplier=self.INITIAL_WAIT,
                max=self.MAX_WAIT,
                exp_base=self.EXPONENTIAL_BASE,
            ),
            retry=retry_if_exception_type((APIError, NetworkError)),
            before_sleep=before_sleep_log(logger, logging.WARNING),
        )

    # ------------------------------------------------------------------------
    # 抽象メソッド（サブクラスで実装）
    # ------------------------------------------------------------------------

    @abstractmethod
    async def _prepare_headers(self) -> Dict[str, str]:
        """サービス固有のヘッダーを準備

        サブクラスで実装する必要がある。

        Returns:
            Dict[str, str]: リクエストヘッダー
        """
        pass

    @abstractmethod
    async def _process_request_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """リクエストデータを処理

        サブクラスで実装する必要がある。

        Args:
            data: 元のリクエストデータ

        Returns:
            Dict[str, Any]: 処理済みのリクエストデータ
        """
        pass

    def __repr__(self) -> str:
        """文字列表現（APIキーをマスク）"""
        masked_key = "***" if self.api_key else "None"
        return (
            f"<{self.__class__.__name__} "
            f"api_key={masked_key} "
            f"base_url={self.base_url} "
            f"state={self._state.value}>"
        )
