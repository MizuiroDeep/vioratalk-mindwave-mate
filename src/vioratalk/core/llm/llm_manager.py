"""LLMエンジン統合管理モジュール

複数のLLMエンジンを統一的に管理し、フォールバックや統計情報収集を提供。
Phase 4実装：基本的なエンジン管理とフォールバック機能を実装。

インターフェース定義書 v1.34準拠
API通信実装ガイド v1.4準拠
エラーハンドリング指針 v1.20準拠
開発規約書 v1.12準拠
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from vioratalk.core.base import ComponentState, VioraTalkComponent
from vioratalk.core.error_handler import ErrorHandler
from vioratalk.core.exceptions import LLMError

# LLMResponseをbaseからインポート（仕様書準拠）
from vioratalk.core.llm.base import LLMResponse

logger = logging.getLogger(__name__)


class LLMEngineStatus(Enum):
    """LLMエンジンの状態"""

    READY = "ready"  # 使用可能
    BUSY = "busy"  # 処理中
    ERROR = "error"  # エラー状態
    DISABLED = "disabled"  # 無効化


@dataclass
class LLMEngineInfo:
    """LLMエンジン情報"""

    name: str
    engine: Any  # BaseLLMEngineインスタンス
    priority: int = 0
    status: LLMEngineStatus = LLMEngineStatus.READY
    enabled: bool = True
    max_retries: int = 3
    timeout: float = 30.0


@dataclass
class LLMStats:
    """LLM使用統計情報"""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_tokens_input: int = 0
    total_tokens_output: int = 0
    total_cost: float = 0.0
    average_response_time: float = 0.0
    last_used: Optional[datetime] = None
    error_history: List[Tuple[datetime, str]] = field(default_factory=list)


class LLMManager(VioraTalkComponent):
    """LLMエンジン統合管理クラス

    複数のLLMエンジンを統一的に管理し、自動切り替えや
    フォールバック機能を提供する。

    Phase 4実装：
    - エンジン登録・管理
    - 優先順位ベースの自動選択
    - フォールバック機能
    - 統計情報収集
    - エラーハンドリング

    Phase 6以降で追加予定：
    - A/Bテスト機能
    - コスト最適化
    - 動的優先順位調整

    Attributes:
        engines: 登録されたエンジン情報
        stats: エンジン別統計情報
        priority_order: 優先順位順のエンジン名リスト
        current_engine: 現在選択中のエンジン
        error_handler: エラーハンドラー
        _initialized: 初期化済みフラグ

    Example:
        >>> llm_manager = LLMManager()
        >>> llm_manager.register_engine("gemini", gemini_engine, priority=1)
        >>> await llm_manager.initialize()
        >>> response = await llm_manager.generate("こんにちは")
        >>> print(response.content)
    """

    def __init__(
        self,
        error_handler: Optional[ErrorHandler] = None,
        default_timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        """コンストラクタ

        Args:
            error_handler: エラーハンドラー（Noneの場合は新規作成）
            default_timeout: デフォルトタイムアウト（秒）
            max_retries: デフォルト最大リトライ回数
        """
        super().__init__()  # VioraTalkComponentの初期化
        self.engines: Dict[str, LLMEngineInfo] = {}
        self.stats: Dict[str, LLMStats] = {}
        self.priority_order: List[str] = []
        self.current_engine: Optional[str] = None
        self.error_handler = error_handler or ErrorHandler()
        self.default_timeout = default_timeout
        self.max_retries = max_retries
        self._initialized = False

    async def initialize(self) -> None:
        """非同期初期化

        登録されたエンジンを初期化し、使用可能な状態にする。

        Raises:
            InitializationError: 初期化に失敗した場合
        """
        if self._state == ComponentState.READY:
            logger.warning("LLMManager is already initialized")
            return

        self._state = ComponentState.INITIALIZING
        logger.info("Initializing LLMManager")

        # 登録されたエンジンを初期化
        init_tasks = []
        for engine_info in self.engines.values():
            if engine_info.enabled and hasattr(engine_info.engine, "initialize"):
                init_tasks.append(self._initialize_engine(engine_info))

        if init_tasks:
            results = await asyncio.gather(*init_tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    engine_name = list(self.engines.keys())[i]
                    logger.error(f"Failed to initialize engine {engine_name}: {result}")
                    self.engines[engine_name].status = LLMEngineStatus.ERROR

        # 優先順位順を更新
        self._update_priority_order()

        # 利用可能なエンジンがあるかチェック
        if not self.priority_order:
            self._state = ComponentState.ERROR
            raise LLMError("No LLM engines available", error_code="E2000")

        self._initialized = True
        self._state = ComponentState.READY
        logger.info(f"LLMManager initialized with {len(self.priority_order)} engines")

    async def cleanup(self) -> None:
        """リソースのクリーンアップ

        使用したリソースを解放する。
        """
        if self._state == ComponentState.TERMINATED:
            return

        self._state = ComponentState.TERMINATING
        logger.info("Cleaning up LLMManager")

        # 各エンジンをクリーンアップ
        cleanup_tasks = []
        for engine_info in self.engines.values():
            if hasattr(engine_info.engine, "cleanup"):
                cleanup_tasks.append(engine_info.engine.cleanup())

        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)

        self.engines.clear()
        self.stats.clear()
        self.priority_order.clear()
        self.current_engine = None
        self._initialized = False
        self._state = ComponentState.TERMINATED

    def is_available(self) -> bool:
        """利用可能状態の確認

        Returns:
            bool: READYまたはRUNNING状態の場合True
        """
        return ComponentState.is_operational(self._state)

    def get_status(self) -> Dict[str, Any]:
        """コンポーネントの状態を取得

        Returns:
            Dict[str, Any]: 状態情報を含む辞書
        """
        return {
            "state": self._state,
            "is_available": self.is_available(),
            "error": str(self._error) if self._error else None,
            "last_used": None,  # 統計情報から取得可能
            "engines": list(self.engines.keys()),
            "current_engine": self.current_engine,
            "priority_order": self.priority_order,
        }

    def register_engine(
        self,
        name: str,
        engine: Any,
        priority: int = 0,
        enabled: bool = True,
        max_retries: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> None:
        """エンジンを登録

        Args:
            name: エンジン名（一意）
            engine: LLMエンジンインスタンス
            priority: 優先度（大きいほど優先）
            enabled: 有効/無効フラグ
            max_retries: 最大リトライ回数
            timeout: タイムアウト（秒）

        Raises:
            ValueError: 既に同名のエンジンが登録されている場合
        """
        if name in self.engines:
            raise ValueError(f"Engine '{name}' is already registered")

        engine_info = LLMEngineInfo(
            name=name,
            engine=engine,
            priority=priority,
            enabled=enabled,
            max_retries=max_retries or self.max_retries,
            timeout=timeout or self.default_timeout,
        )

        self.engines[name] = engine_info
        self.stats[name] = LLMStats()

        # 優先順位順を更新
        if self._initialized:
            self._update_priority_order()

        logger.info(f"Registered LLM engine: {name} (priority={priority})")

    def unregister_engine(self, name: str) -> None:
        """エンジンの登録を解除

        Args:
            name: エンジン名

        Raises:
            KeyError: 指定されたエンジンが存在しない場合
        """
        if name not in self.engines:
            raise KeyError(f"Engine '{name}' not found")

        del self.engines[name]
        del self.stats[name]

        # 現在のエンジンだった場合はクリア
        if self.current_engine == name:
            self.current_engine = None

        # 優先順位順を更新
        self._update_priority_order()

        logger.info(f"Unregistered LLM engine: {name}")

    async def generate(
        self,
        prompt: str,
        engine_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """テキスト生成

        指定されたエンジンまたは自動選択されたエンジンで
        テキストを生成する。失敗時は自動的にフォールバック。

        Args:
            prompt: プロンプトテキスト
            engine_name: 使用するエンジン名（Noneで自動選択）
            temperature: 生成温度
            max_tokens: 最大トークン数
            **kwargs: エンジン固有のパラメータ

        Returns:
            LLMResponse: 生成結果

        Raises:
            LLMError: すべてのエンジンで生成に失敗した場合
            RuntimeError: 未初期化の場合
        """
        if not self._initialized:
            raise RuntimeError("LLMManager is not initialized")

        # エンジンリストを準備
        if engine_name:
            # 指定エンジンのみ試行
            if engine_name not in self.engines:
                raise ValueError(f"Engine '{engine_name}' not found")
            engine_list = [engine_name]
        else:
            # 優先順位順に試行
            engine_list = self.priority_order

        if not engine_list:
            raise LLMError("No LLM engines available", error_code="E2000")

        # 各エンジンで試行
        last_error = None
        for engine_name in engine_list:
            engine_info = self.engines[engine_name]

            # 無効化されているエンジンはスキップ
            if not engine_info.enabled or engine_info.status == LLMEngineStatus.DISABLED:
                continue

            try:
                # エンジンで生成
                response = await self._generate_with_engine(
                    engine_info, prompt, temperature, max_tokens, **kwargs
                )

                # 統計を更新
                self._update_stats(engine_name, success=True, response=response)

                # 現在のエンジンを更新
                self.current_engine = engine_name

                return response

            except Exception as e:
                # エラーを記録
                last_error = e
                self._update_stats(engine_name, success=False, error=str(e))

                # エラーハンドラーで処理
                self.error_handler.handle_error(
                    e,
                    context={
                        "component": "LLMManager",
                        "operation": "generate",
                        "engine": engine_name,
                        "prompt_length": len(prompt),
                    },
                )

                # 次のエンジンを試す
                logger.warning(f"Engine '{engine_name}' failed: {e}")
                continue

        # すべて失敗
        error_msg = f"All LLM engines failed. Last error: {last_error}"
        raise LLMError(error_msg, error_code="E2001")

    async def _generate_with_engine(
        self,
        engine_info: LLMEngineInfo,
        prompt: str,
        temperature: float,
        max_tokens: Optional[int],
        **kwargs,
    ) -> LLMResponse:
        """エンジンでテキスト生成（内部メソッド）

        Args:
            engine_info: エンジン情報
            prompt: プロンプト
            temperature: 生成温度
            max_tokens: 最大トークン数
            **kwargs: エンジン固有パラメータ

        Returns:
            LLMResponse: 生成結果

        Raises:
            TimeoutError: タイムアウトした場合
            LLMError: 生成に失敗した場合
        """
        engine_info.status = LLMEngineStatus.BUSY
        start_time = datetime.now()

        try:
            # タイムアウト付きで生成
            response = await asyncio.wait_for(
                engine_info.engine.generate(
                    prompt=prompt, temperature=temperature, max_tokens=max_tokens, **kwargs
                ),
                timeout=engine_info.timeout,
            )

            # レスポンスをLLMResponse形式に変換
            if isinstance(response, LLMResponse):
                return response
            elif isinstance(response, str):
                # 文字列の場合はLLMResponseに変換
                return LLMResponse(
                    content=response,  # contentフィールド（仕様書準拠）
                    usage={},
                    model=engine_info.name,
                    finish_reason="stop",
                    metadata={"engine": engine_info.name},
                    timestamp=datetime.now(),
                )
            elif hasattr(response, "content"):  # contentフィールドをチェック
                # content属性がある場合
                return LLMResponse(
                    content=response.content,  # contentフィールド（仕様書準拠）
                    usage=getattr(response, "usage", {}),
                    model=getattr(response, "model", engine_info.name),
                    finish_reason=getattr(response, "finish_reason", "stop"),
                    metadata=getattr(response, "metadata", {}),
                    timestamp=getattr(response, "timestamp", datetime.now()),
                )
            else:
                raise LLMError(
                    f"Invalid response type from engine: {type(response)}", error_code="E2000"
                )

        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Engine '{engine_info.name}' timed out after {engine_info.timeout}s"
            )

        finally:
            engine_info.status = LLMEngineStatus.READY
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.debug(f"Engine '{engine_info.name}' took {elapsed:.2f}s")

    async def _initialize_engine(self, engine_info: LLMEngineInfo) -> None:
        """エンジンを初期化（内部メソッド）

        Args:
            engine_info: エンジン情報
        """
        try:
            await engine_info.engine.initialize()
            engine_info.status = LLMEngineStatus.READY
            logger.info(f"Engine '{engine_info.name}' initialized successfully")
        except Exception as e:
            engine_info.status = LLMEngineStatus.ERROR
            logger.error(f"Failed to initialize engine '{engine_info.name}': {e}")
            raise

    def _update_priority_order(self) -> None:
        """優先順位順を更新（内部メソッド）"""
        # 有効なエンジンのみを優先度順にソート
        enabled_engines = [
            (name, info.priority)
            for name, info in self.engines.items()
            if info.enabled and info.status != LLMEngineStatus.ERROR
        ]

        # 優先度でソート（降順）
        enabled_engines.sort(key=lambda x: x[1], reverse=True)

        # エンジン名のリストを更新
        self.priority_order = [name for name, _ in enabled_engines]

        logger.debug(f"Priority order updated: {self.priority_order}")

    def _update_stats(
        self,
        engine_name: str,
        success: bool,
        response: Optional[LLMResponse] = None,
        error: Optional[str] = None,
    ) -> None:
        """統計情報を更新（内部メソッド）

        Args:
            engine_name: エンジン名
            success: 成功/失敗
            response: 応答（成功時）
            error: エラーメッセージ（失敗時）
        """
        if engine_name not in self.stats:
            return

        stats = self.stats[engine_name]
        stats.total_calls += 1
        stats.last_used = datetime.now()

        if success:
            stats.successful_calls += 1
            if response and response.usage:
                stats.total_tokens_input += response.usage.get("input_tokens", 0)
                stats.total_tokens_output += response.usage.get("output_tokens", 0)
        else:
            stats.failed_calls += 1
            if error:
                # エラー履歴に追加（最大10件）
                stats.error_history.append((datetime.now(), error))
                if len(stats.error_history) > 10:
                    stats.error_history.pop(0)

    def get_stats(self, engine_name: Optional[str] = None) -> Dict[str, Any]:
        """統計情報を取得

        Args:
            engine_name: エンジン名（Noneですべて）

        Returns:
            統計情報の辞書

        Example:
            >>> stats = llm_manager.get_stats("gemini")
            >>> print(f"Success rate: {stats['success_rate']:.2%}")
        """
        if engine_name:
            if engine_name not in self.stats:
                return {}

            stats = self.stats[engine_name]
            success_rate = (
                stats.successful_calls / stats.total_calls if stats.total_calls > 0 else 0
            )

            return {
                "engine": engine_name,
                "total_calls": stats.total_calls,
                "successful_calls": stats.successful_calls,
                "failed_calls": stats.failed_calls,
                "success_rate": success_rate,
                "total_tokens_input": stats.total_tokens_input,
                "total_tokens_output": stats.total_tokens_output,
                "total_cost": stats.total_cost,
                "last_used": stats.last_used.isoformat() if stats.last_used else None,
                "recent_errors": [
                    {"timestamp": ts.isoformat(), "error": err} for ts, err in stats.error_history
                ],
            }
        else:
            # 全エンジンの統計
            return {name: self.get_stats(name) for name in self.stats.keys()}

    def get_available_engines(self) -> List[str]:
        """利用可能なエンジンのリストを取得

        Returns:
            エンジン名のリスト
        """
        return [
            name
            for name, info in self.engines.items()
            if info.enabled and info.status == LLMEngineStatus.READY
        ]

    def set_engine_priority(self, engine_name: str, priority: int) -> None:
        """エンジンの優先度を変更

        Args:
            engine_name: エンジン名
            priority: 新しい優先度

        Raises:
            KeyError: エンジンが存在しない場合
        """
        if engine_name not in self.engines:
            raise KeyError(f"Engine '{engine_name}' not found")

        self.engines[engine_name].priority = priority
        self._update_priority_order()

    def enable_engine(self, engine_name: str) -> None:
        """エンジンを有効化

        Args:
            engine_name: エンジン名

        Raises:
            KeyError: エンジンが存在しない場合
        """
        if engine_name not in self.engines:
            raise KeyError(f"Engine '{engine_name}' not found")

        self.engines[engine_name].enabled = True
        self._update_priority_order()
        logger.info(f"Engine '{engine_name}' enabled")

    def disable_engine(self, engine_name: str) -> None:
        """エンジンを無効化

        Args:
            engine_name: エンジン名

        Raises:
            KeyError: エンジンが存在しない場合
        """
        if engine_name not in self.engines:
            raise KeyError(f"Engine '{engine_name}' not found")

        self.engines[engine_name].enabled = False
        self._update_priority_order()
        logger.info(f"Engine '{engine_name}' disabled")

    def get_current_engine(self) -> Optional[str]:
        """現在選択中のエンジン名を取得

        Returns:
            エンジン名（未選択の場合None）
        """
        return self.current_engine

    def __repr__(self) -> str:
        """開発用の詳細文字列表現

        Returns:
            str: オブジェクトの詳細情報
        """
        engine_list = ", ".join(self.priority_order) if self.priority_order else "none"
        return (
            f"LLMManager("
            f"state={self._state.value}, "
            f"engines=[{engine_list}], "
            f"current={self.current_engine or 'none'})"
        )
