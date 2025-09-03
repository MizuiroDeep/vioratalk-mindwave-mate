"""LLMManager単体テストモジュール

LLMManagerの全機能を網羅的にテストする。
32個のテストケースで、エンジン管理、フォールバック、統計情報収集を確認。

テスト戦略ガイドライン v1.7準拠
開発規約書 v1.12準拠
テスト実装ガイド v1.3準拠
"""

import asyncio
from typing import Optional
from unittest.mock import AsyncMock, Mock, patch

import pytest

from vioratalk.core.base import ComponentState
from vioratalk.core.error_handler import ErrorHandler
from vioratalk.core.exceptions import LLMError
from vioratalk.core.llm.llm_manager import (
    LLMEngineStatus,
    LLMManager,
    LLMResponse,
)

# ============================================================================
# モッククラス
# ============================================================================


class MockLLMEngine:
    """LLMエンジンのモック実装

    開発規約書 v1.12 セクション12.3.2準拠
    """

    def __init__(self, name: str = "mock", should_fail: bool = False, fail_init: bool = False):
        """コンストラクタ

        Args:
            name: エンジン名
            should_fail: generate時に失敗をシミュレートするか
            fail_init: 初期化時に失敗をシミュレートするか（分離）
        """
        self.name = name
        self.should_fail = should_fail
        self.fail_init = fail_init  # 初期化失敗を別フラグで制御
        self._initialized = False
        self.generate_count = 0

    async def initialize(self) -> None:
        """初期化"""
        if self.fail_init:  # fail_initフラグで制御
            raise RuntimeError(f"Mock engine {self.name} initialization failed")
        self._initialized = True

    async def cleanup(self) -> None:
        """クリーンアップ"""
        self._initialized = False

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """テキスト生成（モック）"""
        if not self._initialized:
            raise RuntimeError("Engine not initialized")
        if self.should_fail:  # should_failはgenerate時のみ影響
            raise LLMError(f"Mock engine {self.name} generation failed", error_code="E2001")

        self.generate_count += 1

        # 修正: text → content
        return LLMResponse(
            content=f"Response from {self.name}: {prompt[:50]}",
            model=self.name,
            finish_reason="stop",
            usage={"input_tokens": len(prompt), "output_tokens": 50},
            metadata={"engine": self.name, "temperature": temperature},
        )


# ============================================================================
# 初期化・クリーンアップのテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestInitializationAndCleanup:
    """初期化とクリーンアップのテスト"""

    @pytest.mark.asyncio
    async def test_initialization_success(self):
        """正常な初期化のテスト"""
        manager = LLMManager()
        engine = MockLLMEngine("test_engine")

        manager.register_engine("test", engine, priority=1)

        assert manager._state == ComponentState.NOT_INITIALIZED
        await manager.initialize()

        assert manager._state == ComponentState.READY
        assert manager._initialized is True
        assert "test" in manager.priority_order

    @pytest.mark.asyncio
    async def test_initialization_no_engines(self):
        """エンジンなしでの初期化（エラー）"""
        manager = LLMManager()

        with pytest.raises(LLMError) as exc_info:
            await manager.initialize()

        assert "No LLM engines available" in str(exc_info.value)
        assert manager._state == ComponentState.ERROR

    @pytest.mark.asyncio
    async def test_initialization_with_failed_engine(self):
        """初期化失敗エンジンがある場合"""
        manager = LLMManager()

        good_engine = MockLLMEngine("good")
        bad_engine = MockLLMEngine("bad", fail_init=True)  # fail_initフラグを使用

        manager.register_engine("good", good_engine)
        manager.register_engine("bad", bad_engine)

        await manager.initialize()

        # 正常なエンジンのみが利用可能
        assert manager._state == ComponentState.READY
        assert "good" in manager.priority_order
        assert "bad" not in manager.priority_order
        assert manager.engines["bad"].status == LLMEngineStatus.ERROR

    @pytest.mark.asyncio
    async def test_cleanup(self):
        """クリーンアップのテスト"""
        manager = LLMManager()
        engine = MockLLMEngine("test")

        manager.register_engine("test", engine)
        await manager.initialize()
        await manager.cleanup()

        assert manager._state == ComponentState.TERMINATED
        assert len(manager.engines) == 0
        assert len(manager.stats) == 0
        assert len(manager.priority_order) == 0
        assert manager.current_engine is None

    @pytest.mark.asyncio
    async def test_double_initialization(self):
        """二重初期化の防止テスト"""
        manager = LLMManager()
        engine = MockLLMEngine("test")

        manager.register_engine("test", engine)
        await manager.initialize()

        # 警告ログが出力されるが、エラーにはならない
        with patch("vioratalk.core.llm.llm_manager.logger") as mock_logger:
            await manager.initialize()
            mock_logger.warning.assert_called_once()


# ============================================================================
# エンジン登録・管理のテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestEngineRegistration:
    """エンジン登録と管理のテスト"""

    def test_register_engine_basic(self):
        """基本的なエンジン登録"""
        manager = LLMManager()
        engine = MockLLMEngine("test")

        manager.register_engine("test", engine, priority=5)

        assert "test" in manager.engines
        assert manager.engines["test"].priority == 5
        assert manager.engines["test"].enabled is True
        assert "test" in manager.stats

    def test_register_duplicate_engine(self):
        """重複エンジン登録（エラー）"""
        manager = LLMManager()
        engine1 = MockLLMEngine("test1")
        engine2 = MockLLMEngine("test2")

        manager.register_engine("test", engine1)

        with pytest.raises(ValueError) as exc_info:
            manager.register_engine("test", engine2)

        assert "already registered" in str(exc_info.value)

    def test_unregister_engine(self):
        """エンジンの登録解除"""
        manager = LLMManager()
        engine = MockLLMEngine("test")

        manager.register_engine("test", engine)
        manager.unregister_engine("test")

        assert "test" not in manager.engines
        assert "test" not in manager.stats

    def test_unregister_nonexistent_engine(self):
        """存在しないエンジンの登録解除（エラー）"""
        manager = LLMManager()

        with pytest.raises(KeyError) as exc_info:
            manager.unregister_engine("nonexistent")

        assert "not found" in str(exc_info.value)

    def test_priority_order_update(self):
        """優先順位順の更新テスト"""
        manager = LLMManager()

        engine1 = MockLLMEngine("low")
        engine2 = MockLLMEngine("high")
        engine3 = MockLLMEngine("medium")

        manager.register_engine("low", engine1, priority=1)
        manager.register_engine("high", engine2, priority=10)
        manager.register_engine("medium", engine3, priority=5)

        manager._update_priority_order()

        assert manager.priority_order == ["high", "medium", "low"]

    def test_register_with_custom_params(self):
        """カスタムパラメータでの登録"""
        manager = LLMManager(default_timeout=60.0, max_retries=5)
        engine = MockLLMEngine("test")

        manager.register_engine(
            "test",
            engine,
            priority=3,
            enabled=False,
            max_retries=10,
            timeout=120.0,
        )

        info = manager.engines["test"]
        assert info.priority == 3
        assert info.enabled is False
        assert info.max_retries == 10
        assert info.timeout == 120.0

    @pytest.mark.asyncio
    async def test_register_after_initialization(self):
        """初期化後のエンジン登録"""
        manager = LLMManager()
        engine1 = MockLLMEngine("first")

        manager.register_engine("first", engine1)
        await manager.initialize()

        # 初期化後に新しいエンジンを登録
        engine2 = MockLLMEngine("second")
        manager.register_engine("second", engine2, priority=10)

        # 優先順位が自動更新される
        assert manager.priority_order[0] == "second"

    def test_unregister_current_engine(self):
        """現在使用中のエンジンを登録解除"""
        manager = LLMManager()
        engine = MockLLMEngine("current")

        manager.register_engine("current", engine)
        manager.current_engine = "current"

        manager.unregister_engine("current")

        assert manager.current_engine is None


# ============================================================================
# テキスト生成のテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestTextGeneration:
    """テキスト生成機能のテスト"""

    @pytest.mark.asyncio
    async def test_generate_with_auto_selection(self):
        """自動エンジン選択での生成"""
        manager = LLMManager()
        engine = MockLLMEngine("auto")

        manager.register_engine("auto", engine)
        await manager.initialize()

        response = await manager.generate("Test prompt")

        assert isinstance(response, LLMResponse)
        assert "Response from auto" in response.content
        assert manager.current_engine == "auto"

    @pytest.mark.asyncio
    async def test_generate_with_specific_engine(self):
        """特定エンジン指定での生成"""
        manager = LLMManager()

        engine1 = MockLLMEngine("engine1")
        engine2 = MockLLMEngine("engine2")

        manager.register_engine("engine1", engine1)
        manager.register_engine("engine2", engine2)
        await manager.initialize()

        response = await manager.generate("Test", engine_name="engine2")

        assert "Response from engine2" in response.content
        assert manager.current_engine == "engine2"

    @pytest.mark.asyncio
    async def test_generate_with_invalid_engine(self):
        """存在しないエンジン指定（エラー）"""
        manager = LLMManager()
        engine = MockLLMEngine("valid")

        manager.register_engine("valid", engine)
        await manager.initialize()

        with pytest.raises(ValueError) as exc_info:
            await manager.generate("Test", engine_name="invalid")

        assert "not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_without_initialization(self):
        """未初期化での生成（エラー）"""
        manager = LLMManager()
        engine = MockLLMEngine("test")

        manager.register_engine("test", engine)

        with pytest.raises(RuntimeError) as exc_info:
            await manager.generate("Test")

        assert "not initialized" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_with_disabled_engine(self):
        """無効化されたエンジンのスキップ"""
        manager = LLMManager()

        disabled_engine = MockLLMEngine("disabled")
        enabled_engine = MockLLMEngine("enabled")

        manager.register_engine("disabled", disabled_engine, priority=10, enabled=False)
        manager.register_engine("enabled", enabled_engine, priority=1)
        await manager.initialize()

        response = await manager.generate("Test")

        # 優先度は低いが有効なエンジンが使われる
        assert "Response from enabled" in response.content

    @pytest.mark.asyncio
    async def test_generate_with_custom_params(self):
        """カスタムパラメータでの生成"""
        manager = LLMManager()
        engine = MockLLMEngine("custom")

        manager.register_engine("custom", engine)
        await manager.initialize()

        response = await manager.generate(
            "Test",
            temperature=0.9,
            max_tokens=100,
            custom_param="value",
        )

        assert response.metadata["temperature"] == 0.9

    @pytest.mark.asyncio
    async def test_generate_with_timeout(self):
        """タイムアウト処理のテスト"""
        manager = LLMManager(default_timeout=0.1)

        # 遅延するエンジンをモック
        engine = MockLLMEngine("slow")

        async def slow_generate(*args, **kwargs):
            await asyncio.sleep(1.0)
            return LLMResponse(content="Late", model="slow", finish_reason="stop")

        engine.generate = slow_generate

        manager.register_engine("slow", engine)
        await manager.initialize()

        with pytest.raises(LLMError) as exc_info:
            await manager.generate("Test")

        assert "failed" in str(exc_info.value)


# ============================================================================
# フォールバック機能のテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestFallbackFunctionality:
    """フォールバック機能のテスト"""

    @pytest.mark.asyncio
    async def test_fallback_on_engine_failure(self):
        """エンジン失敗時のフォールバック"""
        manager = LLMManager()

        failing_engine = MockLLMEngine("failing", should_fail=True)
        working_engine = MockLLMEngine("working")

        manager.register_engine("failing", failing_engine, priority=10)
        manager.register_engine("working", working_engine, priority=1)
        await manager.initialize()

        response = await manager.generate("Test")

        # フォールバックして動作するエンジンが使われる
        assert "Response from working" in response.content
        assert manager.current_engine == "working"

    @pytest.mark.asyncio
    async def test_all_engines_fail(self):
        """全エンジン失敗時のエラー"""
        manager = LLMManager()

        # generateで失敗するエンジン（初期化は成功）
        engine1 = MockLLMEngine("fail1", should_fail=True)
        engine2 = MockLLMEngine("fail2", should_fail=True)

        manager.register_engine("fail1", engine1)
        manager.register_engine("fail2", engine2)
        await manager.initialize()

        with pytest.raises(LLMError) as exc_info:
            await manager.generate("Test")

        assert "All LLM engines failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_fallback_priority_order(self):
        """優先順位に従ったフォールバック"""
        manager = LLMManager()

        # 優先度順: high(失敗) -> medium(成功) -> low(未試行)
        high_engine = MockLLMEngine("high", should_fail=True)
        medium_engine = MockLLMEngine("medium")
        low_engine = MockLLMEngine("low")

        manager.register_engine("high", high_engine, priority=10)
        manager.register_engine("medium", medium_engine, priority=5)
        manager.register_engine("low", low_engine, priority=1)
        await manager.initialize()

        response = await manager.generate("Test")

        assert "Response from medium" in response.content
        # lowエンジンは試行されない
        assert low_engine.generate_count == 0

    @pytest.mark.asyncio
    async def test_error_handler_integration(self):
        """エラーハンドラーとの統合テスト"""
        error_handler = ErrorHandler()
        manager = LLMManager(error_handler=error_handler)

        # generateで失敗するエンジン（初期化は成功）
        failing_engine = MockLLMEngine("fail", should_fail=True)
        manager.register_engine("fail", failing_engine)
        await manager.initialize()

        with patch.object(error_handler, "handle_error") as mock_handle:
            with pytest.raises(LLMError):
                await manager.generate("Test")

            # エラーハンドラーが呼ばれたことを確認
            mock_handle.assert_called()

            # call_argsの構造を正しく取得
            # 第1引数は例外、contextはキーワード引数
            call_args = mock_handle.call_args
            called_error = call_args.args[0]  # 位置引数の最初（例外）
            call_context = call_args.kwargs.get("context")  # キーワード引数のcontext

            assert isinstance(called_error, LLMError)
            assert call_context["component"] == "LLMManager"
            assert call_context["operation"] == "generate"

    @pytest.mark.asyncio
    async def test_partial_initialization_fallback(self):
        """部分的初期化失敗でのフォールバック"""
        manager = LLMManager()

        # 初期化失敗するエンジン（fail_initフラグ使用）
        bad_init_engine = MockLLMEngine("bad_init", fail_init=True)

        good_engine = MockLLMEngine("good")

        manager.register_engine("bad_init", bad_init_engine, priority=10)
        manager.register_engine("good", good_engine, priority=1)

        await manager.initialize()

        # 初期化失敗したエンジンは使用不可
        assert "bad_init" not in manager.priority_order
        assert "good" in manager.priority_order

        response = await manager.generate("Test")
        assert "Response from good" in response.content


# ============================================================================
# 統計情報のテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestStatistics:
    """統計情報収集のテスト"""

    @pytest.mark.asyncio
    async def test_stats_tracking_success(self):
        """成功時の統計情報記録"""
        manager = LLMManager()
        engine = MockLLMEngine("stats")

        manager.register_engine("stats", engine)
        await manager.initialize()

        # 3回生成を実行
        for _ in range(3):
            await manager.generate("Test")

        stats = manager.get_stats("stats")

        assert stats["total_calls"] == 3
        assert stats["successful_calls"] == 3
        assert stats["failed_calls"] == 0
        assert stats["success_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_stats_tracking_failure(self):
        """失敗時の統計情報記録"""
        manager = LLMManager()

        # generateで失敗するエンジンとフォールバック用のエンジン
        failing_engine = MockLLMEngine("fail", should_fail=True)
        working_engine = MockLLMEngine("work")

        manager.register_engine("fail", failing_engine, priority=10)
        manager.register_engine("work", working_engine, priority=1)
        await manager.initialize()

        # フォールバックが発生する
        await manager.generate("Test")

        fail_stats = manager.get_stats("fail")
        assert fail_stats["total_calls"] == 1
        assert fail_stats["successful_calls"] == 0
        assert fail_stats["failed_calls"] == 1
        assert len(fail_stats["recent_errors"]) == 1

        work_stats = manager.get_stats("work")
        assert work_stats["total_calls"] == 1
        assert work_stats["successful_calls"] == 1

    @pytest.mark.asyncio
    async def test_get_all_stats(self):
        """全エンジンの統計情報取得"""
        manager = LLMManager()

        engine1 = MockLLMEngine("engine1")
        engine2 = MockLLMEngine("engine2")

        manager.register_engine("engine1", engine1)
        manager.register_engine("engine2", engine2)
        await manager.initialize()

        await manager.generate("Test", engine_name="engine1")
        await manager.generate("Test", engine_name="engine2")

        all_stats = manager.get_stats()

        assert "engine1" in all_stats
        assert "engine2" in all_stats
        assert all_stats["engine1"]["total_calls"] == 1
        assert all_stats["engine2"]["total_calls"] == 1


# ============================================================================
# エンジン制御のテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestEngineControl:
    """エンジン制御機能のテスト"""

    def test_enable_disable_engine(self):
        """エンジンの有効/無効切り替え"""
        manager = LLMManager()
        engine = MockLLMEngine("toggle")

        manager.register_engine("toggle", engine)
        assert manager.engines["toggle"].enabled is True

        manager.disable_engine("toggle")
        assert manager.engines["toggle"].enabled is False

        manager.enable_engine("toggle")
        assert manager.engines["toggle"].enabled is True

    def test_set_engine_priority(self):
        """エンジン優先度の変更"""
        manager = LLMManager()

        engine1 = MockLLMEngine("e1")
        engine2 = MockLLMEngine("e2")

        manager.register_engine("e1", engine1, priority=1)
        manager.register_engine("e2", engine2, priority=2)

        manager._update_priority_order()
        assert manager.priority_order == ["e2", "e1"]

        # 優先度を変更
        manager.set_engine_priority("e1", 10)
        assert manager.priority_order == ["e1", "e2"]

    def test_get_available_engines(self):
        """利用可能エンジンのリスト取得"""
        manager = LLMManager()

        enabled_engine = MockLLMEngine("enabled")
        disabled_engine = MockLLMEngine("disabled")
        error_engine = MockLLMEngine("error")

        manager.register_engine("enabled", enabled_engine)
        manager.register_engine("disabled", disabled_engine, enabled=False)
        manager.register_engine("error", error_engine)

        # エラー状態を設定
        manager.engines["error"].status = LLMEngineStatus.ERROR

        available = manager.get_available_engines()

        assert "enabled" in available
        assert "disabled" not in available
        assert "error" not in available

    def test_get_current_engine(self):
        """現在のエンジン取得"""
        manager = LLMManager()

        assert manager.get_current_engine() is None

        manager.current_engine = "test"
        assert manager.get_current_engine() == "test"


# ============================================================================
# エッジケースのテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(4)
class TestEdgeCases:
    """エッジケースのテスト"""

    @pytest.mark.asyncio
    async def test_response_type_conversion(self):
        """異なる応答型の変換テスト"""
        manager = LLMManager()

        # 文字列を返すエンジン
        string_engine = Mock()
        string_engine.initialize = AsyncMock()
        string_engine.cleanup = AsyncMock()
        string_engine.generate = AsyncMock(return_value="Simple string response")

        manager.register_engine("string", string_engine)
        await manager.initialize()

        response = await manager.generate("Test")

        assert isinstance(response, LLMResponse)
        # 修正: text → content
        assert response.content == "Simple string response"
        assert response.model == "string"

    @pytest.mark.asyncio
    async def test_error_history_limit(self):
        """エラー履歴の上限テスト"""
        manager = LLMManager()
        # generateで失敗するエンジン（初期化は成功）
        failing_engine = MockLLMEngine("fail", should_fail=True)

        manager.register_engine("fail", failing_engine)
        await manager.initialize()

        # 15回失敗させる
        for i in range(15):
            with pytest.raises(LLMError):
                await manager.generate(f"Test {i}")

        stats = manager.get_stats("fail")

        # エラー履歴は最大10件
        assert len(stats["recent_errors"]) == 10
        assert stats["failed_calls"] == 15

    def test_repr_method(self):
        """__repr__メソッドのテスト"""
        manager = LLMManager()

        engine1 = MockLLMEngine("alpha")
        engine2 = MockLLMEngine("beta")

        manager.register_engine("alpha", engine1, priority=1)
        manager.register_engine("beta", engine2, priority=2)
        manager._update_priority_order()
        manager.current_engine = "beta"

        repr_str = repr(manager)

        assert "LLMManager" in repr_str
        assert "beta, alpha" in repr_str  # 優先順位順
        assert "current=beta" in repr_str
