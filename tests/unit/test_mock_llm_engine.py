"""test_mock_llm_engine.py - MockLLMEngineテスト

MockLLMEngineの単体テスト。
大規模言語モデルエンジンのモック実装が正しく動作することを確認。

テスト実装ガイド v1.3準拠
テスト戦略ガイドライン v1.7準拠
開発規約書 v1.12準拠
非同期処理実装ガイド v1.1準拠
エラーハンドリング指針 v1.20準拠
"""

import asyncio
import time

import pytest

# テスト対象のインポート
from tests.mocks.mock_llm_engine import LLMResponse, Message, MockLLMEngine
from vioratalk.core.base import ComponentState
from vioratalk.core.exceptions import APIError, LLMError, ModelNotFoundError

# ============================================================================
# フィクスチャ
# ============================================================================


@pytest.fixture
async def mock_llm_engine():
    """MockLLMEngineのフィクスチャ"""
    engine = MockLLMEngine()
    await engine.initialize()
    yield engine
    await engine.cleanup()


@pytest.fixture
def config_with_streaming():
    """ストリーミング有効の設定"""
    return {"delay": 0.05, "streaming": True, "model": "mock-gpt-4", "max_tokens": 1000}


@pytest.fixture
def sample_messages():
    """サンプルメッセージリスト"""
    return [
        Message(role="user", content="こんにちは"),
        Message(role="assistant", content="こんにちは！碧衣です。"),
        Message(role="user", content="今日の天気はどうですか？"),
    ]


# ============================================================================
# 初期化・終了テスト
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.phase(3)
class TestInitializationAndCleanup:
    """初期化と終了処理のテスト"""

    async def test_initialization_success(self):
        """正常な初期化の確認"""
        engine = MockLLMEngine()

        # 初期状態の確認
        assert engine.state == ComponentState.NOT_INITIALIZED
        assert engine.current_model == "mock-gpt-3.5"
        assert engine.max_tokens == 2048

        # 初期化
        await engine.initialize()

        # 初期化後の状態確認
        assert engine.state == ComponentState.READY
        assert engine.response_delay == 0.1
        assert engine.error_mode is False
        assert engine.streaming_enabled is False

        await engine.cleanup()

    async def test_initialization_with_config(self, config_with_streaming):
        """設定付き初期化の確認"""
        engine = MockLLMEngine(config=config_with_streaming)

        # 設定が反映されているか確認
        assert engine.response_delay == 0.05
        assert engine.streaming_enabled is True
        assert engine.current_model == "mock-gpt-4"
        assert engine.max_tokens == 1000

        await engine.initialize()
        assert engine.state == ComponentState.READY

        await engine.cleanup()

    async def test_cleanup(self, mock_llm_engine):
        """クリーンアップ処理の確認"""
        # 初期化済みの状態から
        assert mock_llm_engine.state == ComponentState.READY

        # クリーンアップ実行
        await mock_llm_engine.cleanup()

        # 終了状態の確認
        assert mock_llm_engine.state == ComponentState.TERMINATED


# ============================================================================
# generateメソッドテスト
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.phase(3)
class TestGenerate:
    """generateメソッドのテスト"""

    async def test_generate_simple_prompt(self, mock_llm_engine):
        """シンプルなプロンプトでの生成"""
        response = await mock_llm_engine.generate("こんにちは")

        assert isinstance(response, LLMResponse)
        assert len(response.content) > 0
        assert response.model == "mock-gpt-3.5"
        assert response.finish_reason == "stop"
        assert "total_tokens" in response.usage

    async def test_generate_with_system_prompt(self, mock_llm_engine):
        """システムプロンプト付き生成"""
        response = await mock_llm_engine.generate(prompt="質問があります", system_prompt="あなたは親切なアシスタントです")

        assert response.content is not None
        assert response.metadata["response_type"] == "default"  # "質問"だけでは質問として判定されない

    async def test_generate_with_temperature(self, mock_llm_engine):
        """温度パラメータ付き生成"""
        response = await mock_llm_engine.generate(prompt="創造的な話をして", temperature=0.9)

        assert response.content is not None
        assert response.metadata["temperature"] == 0.9

    async def test_generate_with_max_tokens(self, mock_llm_engine):
        """最大トークン数指定"""
        response = await mock_llm_engine.generate(prompt="長い説明をして", max_tokens=100)

        assert response.content is not None
        assert response.usage["total_tokens"] <= 100

    async def test_generate_not_ready_state(self):
        """初期化前のgenerate呼び出し"""
        engine = MockLLMEngine()

        # 初期化せずにgenerateを呼び出し
        with pytest.raises(LLMError) as exc_info:
            await engine.generate("test")

        assert exc_info.value.error_code == "E2000"
        assert "not ready" in str(exc_info.value)

    async def test_generate_error_mode(self, mock_llm_engine):
        """エラーモード時の動作確認"""
        # エラーモードを有効化
        mock_llm_engine.set_error_mode(True)

        with pytest.raises(APIError) as exc_info:
            await mock_llm_engine.generate("test")

        assert exc_info.value.error_code == "E2001"
        assert "Mock API error" in str(exc_info.value)


# ============================================================================
# キャラクター別応答テスト
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.phase(3)
class TestCharacterResponses:
    """キャラクター別応答のテスト"""

    async def test_aoi_responses(self, mock_llm_engine):
        """碧衣（001_aoi）の応答テスト"""
        # 挨拶
        response = await mock_llm_engine.generate(
            prompt="こんにちは", system_prompt="character_id:001_aoi"
        )
        assert "碧衣" in response.content
        assert response.metadata["character_id"] == "001_aoi"

        # 質問
        response = await mock_llm_engine.generate(
            prompt="これは何ですか？", system_prompt="character_id:001_aoi"
        )
        assert "質問" in response.content or "興味深い" in response.content

        # コマンド
        response = await mock_llm_engine.generate(
            prompt="音楽を再生してください", system_prompt="character_id:001_aoi"
        )
        assert "承知" in response.content or "対応" in response.content

    async def test_haru_responses(self, mock_llm_engine):
        """春人（002_haru）の応答テスト"""
        # 挨拶
        response = await mock_llm_engine.generate(
            prompt="やあ", system_prompt="character_id:002_haru"
        )
        assert "春人" in response.content or "やあ" in response.content
        assert response.metadata["character_id"] == "002_haru"

        # デフォルト応答
        response = await mock_llm_engine.generate(
            prompt="今日は忙しかった", system_prompt="character_id:002_haru"
        )
        assert "そうなんだ" in response.content or "おもしろい" in response.content

    async def test_yui_responses(self, mock_llm_engine):
        """結衣（003_yui）の応答テスト"""
        # 挨拶
        response = await mock_llm_engine.generate(
            prompt="初めまして", system_prompt="character_id:003_yui"
        )
        # 「初めまして」は挨拶として判定されないのでdefault応答
        assert response.metadata["character_id"] == "003_yui"

        # コマンド形式（明確にコマンドとして判定されるもの）
        response = await mock_llm_engine.generate(
            prompt="これをやってください", system_prompt="character_id:003_yui"
        )
        assert "やってみますね" in response.content or "はい" in response.content

    async def test_unknown_character(self, mock_llm_engine):
        """未知のキャラクターの応答テスト"""
        response = await mock_llm_engine.generate(
            prompt="こんにちは", system_prompt="character_id:999_unknown"
        )

        # デフォルトキャラクターの応答が返される
        assert response.metadata["character_id"] == "default"
        assert "こんにちは" in response.content or "お手伝い" in response.content


# ============================================================================
# ストリーミング生成テスト
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.phase(3)
class TestStreamGenerate:
    """ストリーミング生成のテスト"""

    async def test_stream_generate_success(self, mock_llm_engine):
        """正常なストリーミング生成"""
        mock_llm_engine.streaming_enabled = True

        chunks = []
        async for chunk in mock_llm_engine.stream_generate("こんにちは"):
            chunks.append(chunk)

        full_response = "".join(chunks)
        assert len(full_response) > 0
        assert len(chunks) == len(full_response)  # 1文字ずつストリーミング

    async def test_stream_generate_character_based(self, mock_llm_engine):
        """キャラクター別ストリーミング生成"""
        mock_llm_engine.streaming_enabled = True
        mock_llm_engine.set_custom_response("test", "ABC")

        chunks = []
        async for chunk in mock_llm_engine.stream_generate("test"):
            chunks.append(chunk)

        assert chunks == ["A", "B", "C"]

    async def test_stream_generate_with_delay(self, mock_llm_engine):
        """遅延付きストリーミング生成"""
        mock_llm_engine.streaming_enabled = True
        mock_llm_engine.set_custom_response("test", "12345")

        start_time = time.time()
        chunks = []

        async for chunk in mock_llm_engine.stream_generate("test"):
            chunks.append(chunk)

        elapsed_time = time.time() - start_time

        # 各文字に0.01秒の遅延があるので、5文字で約0.05秒以上
        assert elapsed_time >= 0.04  # 誤差を考慮
        assert len(chunks) == 5

    async def test_stream_generate_error_mode(self, mock_llm_engine):
        """エラーモード時のストリーミング"""
        mock_llm_engine.streaming_enabled = True
        mock_llm_engine.set_error_mode(True)

        with pytest.raises(APIError) as exc_info:
            async for _ in mock_llm_engine.stream_generate("test"):
                pass

        # エラーハンドリング指針 v1.20準拠: E2001（応答生成失敗）
        assert exc_info.value.error_code == "E2001"
        assert "Mock API error" in str(exc_info.value)

    async def test_stream_generate_with_system_prompt(self, mock_llm_engine):
        """システムプロンプト付きストリーミング"""
        mock_llm_engine.streaming_enabled = True

        chunks = []
        async for chunk in mock_llm_engine.stream_generate(
            prompt="こんにちは", system_prompt="character_id:002_haru"
        ):
            chunks.append(chunk)

        full_response = "".join(chunks)
        # 春人の応答パターンが含まれる
        assert "春人" in full_response or "やあ" in full_response


# ============================================================================
# モデル管理テスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.phase(3)
class TestModelManagement:
    """モデル管理機能のテスト"""

    def test_get_available_models(self, mock_llm_engine):
        """利用可能なモデルリストの取得"""
        models = mock_llm_engine.get_available_models()

        assert isinstance(models, list)
        assert len(models) == 4
        assert "mock-gpt-3.5" in models
        assert "mock-gpt-4" in models
        assert "mock-claude-3" in models
        assert "mock-gemini-pro" in models

    def test_set_model_valid(self, mock_llm_engine):
        """有効なモデルの設定"""
        # 初期モデル確認
        assert mock_llm_engine.current_model == "mock-gpt-3.5"

        # モデル変更
        mock_llm_engine.set_model("mock-gpt-4")
        assert mock_llm_engine.current_model == "mock-gpt-4"

        # 別のモデルに変更
        mock_llm_engine.set_model("mock-claude-3")
        assert mock_llm_engine.current_model == "mock-claude-3"

    def test_set_model_invalid(self, mock_llm_engine):
        """無効なモデルの設定"""
        with pytest.raises(ModelNotFoundError) as exc_info:
            mock_llm_engine.set_model("invalid-model")

        # エラーハンドリング指針 v1.20準拠: E2404（モデル選択ロジックエラー）
        assert exc_info.value.error_code == "E2404"
        assert "Model 'invalid-model' not found" in str(exc_info.value)

        # モデルは変更されていない
        assert mock_llm_engine.current_model == "mock-gpt-3.5"


# ============================================================================
# 会話履歴管理テスト
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.phase(3)
class TestConversationHistory:
    """会話履歴管理のテスト"""

    async def test_add_message(self, mock_llm_engine):
        """メッセージ追加のテスト"""
        # 初期状態
        assert len(mock_llm_engine.conversation_history) == 0

        # メッセージ追加
        mock_llm_engine.add_message("user", "こんにちは")
        assert len(mock_llm_engine.conversation_history) == 1
        assert mock_llm_engine.conversation_history[0].role == "user"
        assert mock_llm_engine.conversation_history[0].content == "こんにちは"

        # アシスタントメッセージ追加
        mock_llm_engine.add_message("assistant", "こんにちは！")
        assert len(mock_llm_engine.conversation_history) == 2

    async def test_clear_history(self, mock_llm_engine):
        """履歴クリアのテスト"""
        # メッセージを追加
        mock_llm_engine.add_message("user", "test1")
        mock_llm_engine.add_message("assistant", "response1")
        mock_llm_engine.add_message("user", "test2")

        assert len(mock_llm_engine.conversation_history) == 3

        # 履歴をクリア
        mock_llm_engine.clear_history()
        assert len(mock_llm_engine.conversation_history) == 0

    async def test_get_history(self, mock_llm_engine):
        """履歴取得のテスト"""
        # メッセージを追加
        mock_llm_engine.add_message("user", "質問1")
        mock_llm_engine.add_message("assistant", "回答1")

        history = mock_llm_engine.get_history()

        assert len(history) == 2
        assert history[0].role == "user"
        assert history[0].content == "質問1"
        assert history[1].role == "assistant"
        assert history[1].content == "回答1"

        # 取得した履歴は独立していることを確認
        history.append(Message(role="system", content="test"))
        assert len(mock_llm_engine.conversation_history) == 2  # 元の履歴は変更されない

    async def test_history_with_generation(self, mock_llm_engine):
        """生成時の履歴追加テスト"""
        # 初期状態
        assert len(mock_llm_engine.conversation_history) == 0

        # 生成実行
        await mock_llm_engine.generate("質問です")

        # 履歴が追加されていることを確認
        assert len(mock_llm_engine.conversation_history) == 2
        assert mock_llm_engine.conversation_history[0].role == "user"
        assert mock_llm_engine.conversation_history[0].content == "質問です"
        assert mock_llm_engine.conversation_history[1].role == "assistant"


# ============================================================================
# テスト用ユーティリティメソッドテスト
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.phase(3)
class TestUtilityMethods:
    """テスト用ユーティリティメソッドのテスト"""

    async def test_set_error_mode(self, mock_llm_engine):
        """エラーモード設定の確認"""
        # 初期状態
        assert mock_llm_engine.error_mode is False

        # エラーモード有効化
        mock_llm_engine.set_error_mode(True)
        assert mock_llm_engine.error_mode is True

        # エラーモード無効化
        mock_llm_engine.set_error_mode(False)
        assert mock_llm_engine.error_mode is False

    async def test_set_custom_response(self, mock_llm_engine):
        """カスタムレスポンス設定の確認"""
        custom_text = "カスタム応答テキスト"
        mock_llm_engine.set_custom_response("test_prompt", custom_text)

        # カスタムレスポンスが返されることを確認
        response = await mock_llm_engine.generate("test_prompt")
        assert response.content == custom_text

    async def test_custom_response_in_generation(self, mock_llm_engine):
        """生成時のカスタムレスポンス使用"""
        custom_text = "特別な応答"
        mock_llm_engine.set_custom_response("特別な質問", custom_text)

        # 通常の応答
        normal_response = await mock_llm_engine.generate("通常の質問")
        assert normal_response.content != custom_text

        # カスタム応答
        custom_response = await mock_llm_engine.generate("特別な質問")
        assert custom_response.content == custom_text

    async def test_set_response_delay(self, mock_llm_engine):
        """応答遅延設定の確認"""
        # 初期値確認
        assert mock_llm_engine.response_delay == 0.1

        # 遅延設定
        mock_llm_engine.set_response_delay(0.5)
        assert mock_llm_engine.response_delay == 0.5

        # 負の値は0にクリップ
        mock_llm_engine.set_response_delay(-1.0)
        assert mock_llm_engine.response_delay == 0.0

    async def test_get_statistics(self, mock_llm_engine):
        """統計情報取得の確認"""
        # カスタムレスポンスを設定
        mock_llm_engine.set_custom_response("test1", "response1")
        mock_llm_engine.set_custom_response("test2", "response2")

        # 生成を実行
        await mock_llm_engine.generate("test1")
        await mock_llm_engine.generate("test2")

        stats = mock_llm_engine.get_statistics()

        assert stats["total_requests"] == 2
        assert stats["total_tokens"] > 0
        assert stats["average_tokens_per_request"] > 0
        assert stats["conversation_length"] == 4  # 2つの生成で4メッセージ


# ============================================================================
# 非同期処理テスト
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.phase(3)
class TestAsyncBehavior:
    """非同期処理の動作テスト"""

    async def test_response_delay(self, mock_llm_engine):
        """応答遅延のテスト"""
        # 遅延を0.2秒に設定
        mock_llm_engine.set_response_delay(0.2)

        # 処理時間を計測
        start_time = time.time()

        response = await mock_llm_engine.generate("test")

        elapsed_time = time.time() - start_time

        # 遅延が適用されていることを確認（誤差を考慮）
        assert elapsed_time >= 0.2
        assert elapsed_time < 0.3
        assert response.content is not None

    async def test_concurrent_generations(self, mock_llm_engine):
        """並行生成のテスト"""
        # 複数のプロンプトを準備
        prompts = ["質問1", "質問2", "質問3"]

        # 並行実行
        tasks = [mock_llm_engine.generate(prompt) for prompt in prompts]
        responses = await asyncio.gather(*tasks)

        # すべて成功していることを確認
        assert len(responses) == 3
        for response in responses:
            assert isinstance(response, LLMResponse)
            assert response.content is not None


# ============================================================================
# エッジケーステスト
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.phase(3)
class TestEdgeCases:
    """エッジケースのテスト"""

    async def test_empty_prompt(self, mock_llm_engine):
        """空のプロンプト"""
        response = await mock_llm_engine.generate("")

        assert response.content is not None
        assert len(response.content) > 0

    async def test_very_long_prompt(self, mock_llm_engine):
        """非常に長いプロンプト"""
        long_prompt = "これは" * 1000  # 3000文字
        response = await mock_llm_engine.generate(long_prompt)

        assert response.content is not None
        assert response.usage["prompt_tokens"] > 100

    async def test_special_characters_in_prompt(self, mock_llm_engine):
        """特殊文字を含むプロンプト"""
        special_prompt = "これは\n改行と\tタブと😀絵文字を含むプロンプトです！"
        response = await mock_llm_engine.generate(special_prompt)

        assert response.content is not None
        assert len(response.content) > 0

    async def test_max_tokens_exceeded(self, mock_llm_engine):
        """最大トークン数超過のテスト"""
        response = await mock_llm_engine.generate(prompt="短い", max_tokens=1)

        # finish_reasonがlengthになる
        assert response.finish_reason == "length"
        assert response.usage["completion_tokens"] <= 1


# ============================================================================
# トークン計算テスト
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.phase(3)
class TestTokenCalculation:
    """トークン計算のテスト"""

    async def test_token_usage_calculation(self, mock_llm_engine):
        """トークン使用量計算の確認"""
        response = await mock_llm_engine.generate("これはテストです")

        usage = response.usage
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage

        # 合計が正しいことを確認
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

        # トークン数が妥当な範囲内であることを確認
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] > 0

    async def test_token_approximation(self, mock_llm_engine):
        """トークン数の近似計算確認"""
        # 日本語のプロンプト（1文字≒2トークン）
        japanese_prompt = "こんにちは"  # 5文字
        response = await mock_llm_engine.generate(japanese_prompt)

        # 日本語は1文字2トークンで計算されるため、約10トークン
        assert response.usage["prompt_tokens"] >= 8  # 誤差を考慮
        assert response.usage["prompt_tokens"] <= 12
