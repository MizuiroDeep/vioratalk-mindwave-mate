"""TTSエンジン統合管理モジュール

複数のTTSエンジンを統合管理し、優先順位に基づく自動選択、
フォールバック機能、手動切り替え機能を提供する。

インターフェース定義書 v1.34準拠
エラーハンドリング指針 v1.20準拠
開発規約書 v1.12準拠

Phase 4: 基本実装（単一エンジン管理）
Phase 6: 完全実装（複数エンジン対応予定）
"""

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from vioratalk.core.base import ComponentState
from vioratalk.core.error_handler import ErrorHandler, get_default_error_handler
from vioratalk.core.exceptions import InvalidVoiceError, TTSError
from vioratalk.core.tts.base import BaseTTSEngine, SynthesisResult, TTSConfig, VoiceInfo

logger = logging.getLogger(__name__)


class TTSManager(BaseTTSEngine):
    """複数TTSエンジン統合管理クラス

    LLMManagerと同様の設計で、優先順位に基づく自動選択、
    フォールバック機能、手動切り替え機能を提供する。

    Attributes:
        _engines: 登録されたエンジンの辞書
        _priorities: エンジンの優先順位
        _active_engine: 現在のアクティブエンジン名
        _error_handler: エラーハンドラー
        _stats: 使用統計情報
        _max_fallback_attempts: 最大フォールバック試行回数

    Example:
        >>> # Phase 4: 単一エンジン
        >>> manager = TTSManager()
        >>> engine = Pyttsx3Engine()
        >>> manager.register_engine("pyttsx3", engine, priority=0)
        >>>
        >>> # 音声合成
        >>> result = await manager.synthesize("こんにちは")
        >>>
        >>> # Phase 6: 複数エンジン（将来）
        >>> manager.register_engine("sapi", WindowsSAPIEngine(), priority=1)
        >>> manager.register_engine("edge-tts", EdgeTTSEngine(), priority=2)
        >>> manager.set_active_engine("edge-tts")
    """

    # OS別デフォルト優先順位（Phase 6で使用予定）
    DEFAULT_PRIORITIES = {
        "win32": {
            "speed": ["sapi", "edge-tts", "aivisspeech", "pyttsx3"],
            "quality": ["aivisspeech", "sapi", "edge-tts", "pyttsx3"],
        },
        "darwin": {
            "speed": ["edge-tts", "aivisspeech", "pyttsx3"],
            "quality": ["aivisspeech", "edge-tts", "pyttsx3"],
        },
        "linux": {
            "speed": ["edge-tts", "aivisspeech", "pyttsx3"],
            "quality": ["aivisspeech", "edge-tts", "pyttsx3"],
        },
    }

    def __init__(
        self,
        config: Optional[TTSConfig] = None,
        error_handler: Optional[ErrorHandler] = None,
        max_fallback_attempts: int = 3,
    ) -> None:
        """初期化

        Args:
            config: TTS設定
            error_handler: エラーハンドラー（None時はデフォルト使用）
            max_fallback_attempts: 最大フォールバック試行回数
        """
        super().__init__(config)

        # エンジン管理
        self._engines: Dict[str, BaseTTSEngine] = {}
        self._priorities: Dict[str, int] = {}
        self._active_engine: Optional[str] = None

        # エラーハンドリング
        self._error_handler = error_handler or get_default_error_handler()
        self._max_fallback_attempts = max_fallback_attempts

        # 統計情報
        self._stats: Dict[str, Any] = {
            "total_requests": 0,
            "success_count": 0,
            "fallback_count": 0,
            "engine_usage": defaultdict(int),
            "engine_errors": defaultdict(int),
            "total_synthesis_time": 0.0,
            "last_error": None,
            "session_start": datetime.now(),
        }

        logger.info(
            "TTSManager initialized",
            extra={
                "max_fallback_attempts": max_fallback_attempts,
                "config": config.__dict__ if config else None,
            },
        )

    # =========================================================================
    # エンジン管理
    # =========================================================================

    def register_engine(self, name: str, engine: BaseTTSEngine, priority: int = 0) -> None:
        """エンジンを登録

        Args:
            name: エンジン識別名
            engine: TTSエンジンインスタンス
            priority: 優先順位（大きいほど高優先）

        Raises:
            ValueError: 無効な引数
        """
        if not name:
            raise ValueError("Engine name cannot be empty")
        if not isinstance(engine, BaseTTSEngine):
            raise ValueError(f"Engine must be BaseTTSEngine instance, got {type(engine)}")

        self._engines[name] = engine
        self._priorities[name] = priority

        # 初めてのエンジンならアクティブに設定
        if self._active_engine is None:
            self._active_engine = name

        logger.info(
            f"TTS engine registered: {name}",
            extra={
                "priority": priority,
                "engine_type": type(engine).__name__,
                "total_engines": len(self._engines),
            },
        )

    def unregister_engine(self, name: str) -> None:
        """エンジンの登録解除

        Args:
            name: 登録解除するエンジン名

        Raises:
            ValueError: 指定エンジンが未登録
        """
        if name not in self._engines:
            raise ValueError(f"Engine '{name}' is not registered")

        # クリーンアップ処理は削除（同期関数内では実行不可）
        # 注：エンジンのクリーンアップはTTSManager.cleanup()で一括実行される
        engine = self._engines[name]

        # 削除
        del self._engines[name]
        del self._priorities[name]

        # アクティブエンジンだった場合は別のエンジンに切り替え
        if self._active_engine == name:
            if self._engines:
                self._active_engine = self._get_highest_priority_engine()
            else:
                self._active_engine = None

        logger.info(f"TTS engine unregistered: {name}")

    # =========================================================================
    # 手動切り替え機能（LLMManagerと同様）
    # =========================================================================

    def set_active_engine(self, engine_name: str) -> None:
        """アクティブエンジンを手動で設定

        永続的にエンジンを切り替える。
        フォールバック時もこのエンジンから開始される。

        Args:
            engine_name: 切り替え先エンジン名

        Raises:
            ValueError: 指定エンジンが未登録
        """
        if engine_name not in self._engines:
            raise ValueError(f"Engine '{engine_name}' is not registered")

        old_engine = self._active_engine
        self._active_engine = engine_name

        logger.info(
            f"Active TTS engine changed: {old_engine} -> {engine_name}",
            extra={"available_engines": list(self._engines.keys())},
        )

        # Phase 4では実質的に1つのエンジンのみ
        if len(self._engines) == 1:
            logger.info("Only one TTS engine available, switching has no effect")

    def get_active_engine(self) -> str:
        """現在のアクティブエンジン名を取得

        Returns:
            str: アクティブエンジン名

        Raises:
            RuntimeError: エンジンが登録されていない
        """
        if self._active_engine is None:
            raise RuntimeError("No TTS engine registered")
        return self._active_engine

    def get_available_engines(self) -> List[str]:
        """利用可能なエンジン名のリストを取得

        Returns:
            List[str]: 登録済みエンジン名のリスト
        """
        return list(self._engines.keys())

    # =========================================================================
    # 優先順位管理（フォールバック用）
    # =========================================================================

    def set_engine_priority(self, priorities: Dict[str, int]) -> None:
        """エンジン優先順位を設定

        フォールバック時の試行順序を決定する。
        手動切り替えには影響しない。

        Args:
            priorities: エンジン名と優先順位のマッピング

        Example:
            >>> manager.set_engine_priority({
            ...     "aivisspeech": 3,  # 最高優先
            ...     "edge-tts": 2,
            ...     "pyttsx3": 1       # 最低優先
            ... })
        """
        for name, priority in priorities.items():
            if name in self._engines:
                self._priorities[name] = priority
                logger.debug(f"Priority updated for {name}: {priority}")
            else:
                logger.warning(f"Cannot set priority for unregistered engine: {name}")

    def get_engine_priorities(self) -> Dict[str, int]:
        """現在の優先順位設定を取得

        Returns:
            Dict[str, int]: エンジン名と優先順位のマッピング
        """
        return self._priorities.copy()

    # =========================================================================
    # 音声合成（BaseTTSEngine準拠 + 拡張）
    # =========================================================================

    async def synthesize(
        self, text: str, voice_id: Optional[str] = None, style: Optional[str] = None, **kwargs
    ) -> SynthesisResult:
        """アクティブエンジンで音声合成

        BaseTTSEngine準拠のインターフェース。
        現在のアクティブエンジンを使用する。

        Args:
            text: 合成するテキスト
            voice_id: 音声ID（エンジン依存）
            style: スタイル（AivisSpeech用）
            **kwargs: エンジン固有パラメータ

        Returns:
            SynthesisResult: 音声合成結果

        Raises:
            TTSError: 合成失敗（フォールバックなし）
            RuntimeError: エンジン未登録
        """
        if not self._engines:
            raise RuntimeError("No TTS engine registered")

        if self._active_engine is None:
            self._active_engine = self._get_highest_priority_engine()

        engine = self._engines[self._active_engine]

        # 統計更新
        self._stats["total_requests"] += 1
        self._stats["engine_usage"][self._active_engine] += 1

        start_time = datetime.now()

        try:
            # エンジンで合成
            result = await engine.synthesize(text, voice_id, style, **kwargs)

            # 成功統計
            synthesis_time = (datetime.now() - start_time).total_seconds()
            self._stats["success_count"] += 1
            self._stats["total_synthesis_time"] += synthesis_time

            logger.debug(
                f"Synthesis successful with {self._active_engine}",
                extra={
                    "engine": self._active_engine,
                    "text_length": len(text),
                    "synthesis_time": synthesis_time,
                },
            )

            return result

        except Exception as e:
            # エラー統計
            self._stats["engine_errors"][self._active_engine] += 1
            self._stats["last_error"] = str(e)

            # エラーハンドリング
            error_info = await self._error_handler.handle_error_async(
                e,
                context={
                    "component": "TTSManager",
                    "engine": self._active_engine,
                    "operation": "synthesize",
                },
            )

            logger.error(
                f"Synthesis failed with {self._active_engine}",
                extra={
                    "error": str(e),
                    "error_code": error_info.error_code,
                    "engine": self._active_engine,
                },
            )

            raise

    async def synthesize_with_fallback(
        self, text: str, preferred_engine: Optional[str] = None, **kwargs
    ) -> SynthesisResult:
        """フォールバック付き音声合成

        優先順位に従って自動的にフォールバックする。

        Args:
            text: 合成するテキスト
            preferred_engine: この合成のみ使用するエンジン（一時的）
            **kwargs: エンジン固有パラメータ

        Returns:
            SynthesisResult: 音声合成結果

        Raises:
            TTSError: すべてのエンジンで失敗（E3000）
        """
        if not self._engines:
            raise RuntimeError("No TTS engine registered")

        # エンジン試行順序を決定
        engines_to_try = self._get_engine_order(preferred_engine)

        # 統計更新
        self._stats["total_requests"] += 1

        last_error = None
        attempts = 0

        for engine_name in engines_to_try:
            if attempts >= self._max_fallback_attempts:
                logger.warning(f"Max fallback attempts ({self._max_fallback_attempts}) reached")
                break

            attempts += 1
            engine = self._engines.get(engine_name)
            if not engine:
                continue

            try:
                # 統計更新
                self._stats["engine_usage"][engine_name] += 1
                start_time = datetime.now()

                # エンジンで合成試行
                logger.debug(f"Attempting synthesis with {engine_name}")
                result = await engine.synthesize(text, **kwargs)

                # 成功
                synthesis_time = (datetime.now() - start_time).total_seconds()
                self._stats["success_count"] += 1
                self._stats["total_synthesis_time"] += synthesis_time

                # フォールバックが発生した場合
                if attempts > 1:
                    self._stats["fallback_count"] += 1
                    logger.info(
                        f"Synthesis succeeded with fallback engine: {engine_name}",
                        extra={"attempt": attempts, "original_error": str(last_error)},
                    )

                return result

            except InvalidVoiceError as e:
                # E3001: 音声ID無効（リトライ不可）
                logger.error(f"Invalid voice ID: {e}")
                await self._error_handler.handle_error_async(e)
                raise  # フォールバック不要

            except TTSError as e:
                # E3000/E3004: 一般的なTTSエラー（フォールバック試行）
                last_error = e
                self._stats["engine_errors"][engine_name] += 1

                # エラーコードによる判定
                if hasattr(e, "error_code"):
                    if e.error_code in ["E3001", "E3002", "E3003"]:
                        # フォールバック不要なエラー
                        logger.error(f"Non-recoverable TTS error: {e}")
                        await self._error_handler.handle_error_async(e)
                        raise

                # フォールバック可能なエラー
                logger.warning(
                    f"TTS engine {engine_name} failed, trying next",
                    extra={"error": str(e), "attempt": attempts},
                )

                await self._error_handler.handle_error_async(e)
                continue  # 次のエンジンを試す

            except Exception as e:
                # 予期しないエラー
                last_error = TTSError(f"Unexpected error in {engine_name}: {e}", error_code="E3000")
                self._stats["engine_errors"][engine_name] += 1
                logger.error(f"Unexpected error with {engine_name}: {e}")
                continue

        # すべて失敗
        self._stats["last_error"] = str(last_error) if last_error else "All engines failed"

        error_msg = f"All TTS engines failed after {attempts} attempts"
        if last_error:
            error_msg += f": {last_error}"

        raise TTSError(error_msg, error_code="E3000")

    # =========================================================================
    # 統計情報
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """エンジン使用統計を取得

        Returns:
            Dict containing:
            - total_requests: 総リクエスト数
            - success_count: 成功数
            - success_rate: 成功率
            - fallback_count: フォールバック発生回数
            - engine_usage: エンジン別使用回数
            - engine_errors: エンジン別エラー回数
            - average_synthesis_time: 平均合成時間
            - last_error: 最後のエラー
            - session_duration: セッション時間
        """
        stats = self._stats.copy()

        # 成功率計算
        if stats["total_requests"] > 0:
            stats["success_rate"] = stats["success_count"] / stats["total_requests"]
        else:
            stats["success_rate"] = 0.0

        # 平均合成時間
        if stats["success_count"] > 0:
            stats["average_synthesis_time"] = stats["total_synthesis_time"] / stats["success_count"]
        else:
            stats["average_synthesis_time"] = 0.0

        # セッション時間
        stats["session_duration"] = (datetime.now() - stats["session_start"]).total_seconds()

        # dict型に変換（defaultdictを通常のdictに）
        stats["engine_usage"] = dict(stats["engine_usage"])
        stats["engine_errors"] = dict(stats["engine_errors"])

        return stats

    def reset_stats(self) -> None:
        """統計情報をリセット"""
        self._stats = {
            "total_requests": 0,
            "success_count": 0,
            "fallback_count": 0,
            "engine_usage": defaultdict(int),
            "engine_errors": defaultdict(int),
            "total_synthesis_time": 0.0,
            "last_error": None,
            "session_start": datetime.now(),
        }
        logger.info("TTS statistics reset")

    # =========================================================================
    # BaseTTSEngine準拠メソッド（内部でアクティブエンジンに委譲）
    # =========================================================================

    def get_available_voices(self) -> List[VoiceInfo]:
        """アクティブエンジンの利用可能な音声リストを取得

        Returns:
            List[VoiceInfo]: 音声情報リスト

        Raises:
            RuntimeError: エンジン未登録
        """
        if not self._engines or not self._active_engine:
            raise RuntimeError("No TTS engine registered")

        engine = self._engines[self._active_engine]
        return engine.get_available_voices()

    def set_voice(self, voice_id: str) -> None:
        """アクティブエンジンの音声を設定

        Args:
            voice_id: 音声ID

        Raises:
            RuntimeError: エンジン未登録
            InvalidVoiceError: 無効な音声ID（E3001）
        """
        if not self._engines or not self._active_engine:
            raise RuntimeError("No TTS engine registered")

        engine = self._engines[self._active_engine]

        try:
            engine.set_voice(voice_id)
            logger.info(f"Voice set to {voice_id} on {self._active_engine}")
        except InvalidVoiceError as e:
            logger.error(f"Failed to set voice: {e}")
            raise

    async def test_availability(self) -> bool:
        """アクティブエンジンの利用可能性をテスト

        Returns:
            bool: 利用可能な場合True
        """
        if not self._engines or not self._active_engine:
            return False

        engine = self._engines[self._active_engine]

        try:
            return await engine.test_availability()
        except Exception as e:
            logger.warning(f"Availability test failed for {self._active_engine}: {e}")
            return False

    # =========================================================================
    # VioraTalkComponent準拠メソッド
    # =========================================================================

    async def initialize(self) -> None:
        """非同期初期化

        登録されたすべてのエンジンを初期化する。
        """
        if self._state == ComponentState.READY:
            return

        self._state = ComponentState.INITIALIZING
        logger.info("Initializing TTSManager")

        # 各エンジンの初期化
        init_errors = []
        for name, engine in self._engines.items():
            try:
                if hasattr(engine, "initialize") and engine._state != ComponentState.READY:
                    await engine.initialize()
                    logger.info(f"Initialized TTS engine: {name}")
            except Exception as e:
                logger.error(f"Failed to initialize {name}: {e}")
                init_errors.append((name, e))

        # 一部でも成功していればREADY
        if len(init_errors) < len(self._engines):
            self._state = ComponentState.READY
            logger.info(
                "TTSManager initialization completed",
                extra={
                    "initialized": len(self._engines) - len(init_errors),
                    "failed": len(init_errors),
                },
            )
        else:
            self._state = ComponentState.ERROR
            raise TTSError(
                f"Failed to initialize any TTS engine: {init_errors}", error_code="E3004"
            )

    async def cleanup(self) -> None:
        """リソースのクリーンアップ

        すべてのエンジンをクリーンアップする。
        """
        if self._state == ComponentState.TERMINATED:
            return

        self._state = ComponentState.TERMINATING
        logger.info("Cleaning up TTSManager")

        # 各エンジンのクリーンアップ
        for name, engine in self._engines.items():
            try:
                if hasattr(engine, "cleanup"):
                    await engine.cleanup()
                    logger.debug(f"Cleaned up TTS engine: {name}")
            except Exception as e:
                logger.error(f"Error cleaning up {name}: {e}")

        self._engines.clear()
        self._priorities.clear()
        self._active_engine = None

        self._state = ComponentState.TERMINATED
        logger.info("TTSManager cleanup completed")

    def is_available(self) -> bool:
        """利用可能状態の確認

        Returns:
            bool: READYまたはRUNNING状態の場合True
        """
        return self._state in [ComponentState.READY, ComponentState.RUNNING]

    def get_status(self) -> Dict[str, Any]:
        """ステータス情報の取得

        Returns:
            Dict[str, Any]: ステータス情報
        """
        return {
            "state": self._state.value,
            "is_available": self.is_available(),
            "active_engine": self._active_engine,
            "registered_engines": list(self._engines.keys()),
            "priorities": self._priorities.copy(),
            "stats": self.get_stats(),
        }

    # =========================================================================
    # 内部ヘルパーメソッド
    # =========================================================================

    def _get_highest_priority_engine(self) -> str:
        """最高優先順位のエンジンを取得

        Returns:
            str: エンジン名
        """
        if not self._engines:
            raise RuntimeError("No engines registered")

        return max(self._priorities.items(), key=lambda x: x[1])[0]

    def _get_engine_order(self, preferred_engine: Optional[str] = None) -> List[str]:
        """エンジン試行順序を取得

        Args:
            preferred_engine: 優先エンジン（指定時は最初に試行）

        Returns:
            List[str]: エンジン名のリスト（試行順）
        """
        if preferred_engine and preferred_engine in self._engines:
            # 指定エンジンを最初に
            engines = [preferred_engine]
            # 残りを優先順位順に追加
            others = sorted(
                [(name, pri) for name, pri in self._priorities.items() if name != preferred_engine],
                key=lambda x: x[1],
                reverse=True,
            )
            engines.extend([name for name, _ in others])
            return engines
        else:
            # 優先順位順
            return [
                name
                for name, _ in sorted(self._priorities.items(), key=lambda x: x[1], reverse=True)
            ]

    def __repr__(self) -> str:
        """開発用の詳細文字列表現

        Returns:
            str: オブジェクトの詳細情報
        """
        return (
            f"TTSManager("
            f"state={self._state.value}, "
            f"engines={list(self._engines.keys())}, "
            f"active={self._active_engine}, "
            f"stats={self._stats['total_requests']} requests)"
        )
