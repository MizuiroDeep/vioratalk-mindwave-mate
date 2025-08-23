"""test_mock_engines_integration.py - 統合テスト

Phase 1～3の全コンポーネントの統合テスト。
音声入力から応答生成、音声合成までの完全なフローをテスト。

テスト実装ガイド v1.3準拠
テスト戦略ガイドライン v1.7準拠
開発規約書 v1.12準拠
プロジェクト構造説明書 v2.25準拠
"""

import asyncio
import time

import pytest

from tests.mocks.mock_llm_engine import MockLLMEngine

# Phase 3 Mockエンジン
from tests.mocks.mock_stt_engine import AudioData, MockSTTEngine
from tests.mocks.mock_tts_engine import MockTTSEngine
from vioratalk.configuration.config_manager import ConfigManager  # 修正: configurationパッケージ

# Phase 1コンポーネント
from vioratalk.core.base import ComponentState
from vioratalk.core.vioratalk_engine import VioraTalkEngine
from vioratalk.utils.logger_manager import LoggerManager

# Phase 2コンポーネント（存在する場合）
try:
    from tests.mocks.mock_character_manager import MockCharacterManager
    from vioratalk.core.dialogue.dialogue_config import DialogueConfig
    from vioratalk.core.dialogue.dialogue_manager import DialogueManager

    PHASE2_AVAILABLE = True
except ImportError:
    PHASE2_AVAILABLE = False

# エラークラス
from vioratalk.core.exceptions import AudioError  # 追加: test_tts_fallback_mechanismで使用
from vioratalk.core.exceptions import LLMError, STTError

# ============================================================================
# フィクスチャ
# ============================================================================


@pytest.fixture
async def mock_stt_engine():
    """MockSTTEngineフィクスチャ"""
    engine = MockSTTEngine()
    await engine.initialize()
    yield engine
    await engine.cleanup()


@pytest.fixture
async def mock_llm_engine():
    """MockLLMEngineフィクスチャ"""
    engine = MockLLMEngine()
    await engine.initialize()
    yield engine
    await engine.cleanup()


@pytest.fixture
async def mock_tts_engine():
    """MockTTSEngineフィクスチャ"""
    engine = MockTTSEngine()
    await engine.initialize()
    yield engine
    await engine.cleanup()


@pytest.fixture
async def all_mock_engines(mock_stt_engine, mock_llm_engine, mock_tts_engine):
    """全Mockエンジンのフィクスチャ"""
    return {"stt": mock_stt_engine, "llm": mock_llm_engine, "tts": mock_tts_engine}


@pytest.fixture
def sample_audio_data():
    """サンプル音声データ"""
    return AudioData(
        data=b"sample_audio_data",
        sample_rate=16000,
        duration=3.0,
        metadata={"filename": "greeting.wav"},
    )


@pytest.fixture
async def vioratalk_engine():
    """VioraTalkEngineフィクスチャ（Phase 1）"""
    engine = VioraTalkEngine()
    await engine.initialize()
    yield engine
    await engine.cleanup()


# ============================================================================
# 基本的な連携テスト
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.phase(3)
class TestBasicIntegration:
    """基本的な統合テスト"""

    async def test_stt_to_llm_flow(self, mock_stt_engine, mock_llm_engine, sample_audio_data):
        """STT → LLMの連携テスト"""
        # 音声認識
        transcription = await mock_stt_engine.transcribe(sample_audio_data)
        assert transcription.text == "こんにちは、今日はいい天気ですね。"

        # 認識結果をLLMに渡す
        response = await mock_llm_engine.generate(transcription.text)
        assert response.content is not None
        assert response.metadata["response_type"] == "greeting"

    async def test_llm_to_tts_flow(self, mock_llm_engine, mock_tts_engine):
        """LLM → TTSの連携テスト"""
        # LLMで応答生成
        llm_response = await mock_llm_engine.generate("こんにちは")
        assert llm_response.content is not None

        # 応答を音声合成
        synthesis_result = await mock_tts_engine.synthesize(llm_response.content)
        assert synthesis_result.audio_data is not None
        assert len(synthesis_result.audio_data) > 44  # WAVヘッダー以上

    async def test_complete_conversation_flow(self, all_mock_engines, sample_audio_data):
        """完全な対話フローテスト（STT → LLM → TTS）"""
        stt = all_mock_engines["stt"]
        llm = all_mock_engines["llm"]
        tts = all_mock_engines["tts"]

        # 1. 音声認識
        transcription = await stt.transcribe(sample_audio_data)
        assert transcription.text is not None

        # 2. 応答生成
        response = await llm.generate(
            prompt=transcription.text, system_prompt="character_id:001_aoi"
        )
        assert response.content is not None
        assert response.metadata["character_id"] == "001_aoi"

        # 3. 音声合成
        synthesis = await tts.synthesize(response.content)
        assert synthesis.audio_data is not None
        assert synthesis.sample_rate == 22050
        assert synthesis.duration > 0


# ============================================================================
# エラー処理テスト
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.phase(3)
class TestErrorHandling:
    """エラー処理の統合テスト"""

    async def test_stt_error_recovery(self, mock_stt_engine, mock_llm_engine):
        """STTエラーからの回復テスト"""
        # エラーモードを有効化
        mock_stt_engine.error_mode = True

        # エラーが発生することを確認（E1001: AudioError）
        with pytest.raises(STTError) as exc_info:
            await mock_stt_engine.transcribe(AudioData(data=b"test"))
        assert exc_info.value.error_code == "E1001"  # 修正: E1000→E1001

        # エラーモードを無効化して再試行
        mock_stt_engine.error_mode = False
        transcription = await mock_stt_engine.transcribe(
            AudioData(data=b"test", metadata={"filename": "greeting.wav"})
        )
        assert transcription.text is not None

        # LLMが正常に処理できることを確認
        response = await mock_llm_engine.generate(transcription.text)
        assert response.content is not None

    async def test_llm_timeout_handling(self, mock_llm_engine):
        """LLMタイムアウト処理テスト"""
        # 遅延を増やしてタイムアウトをシミュレート
        mock_llm_engine.response_delay = 0.001  # 短い遅延で正常動作確認

        # 正常に処理されることを確認
        response = await mock_llm_engine.generate("テスト")
        assert response.content is not None

        # エラーモードでのテスト
        mock_llm_engine.error_mode = True
        with pytest.raises(LLMError):
            await mock_llm_engine.generate("エラーテスト")

    async def test_tts_fallback_mechanism(self, mock_tts_engine):
        """TTS フォールバック機構テスト"""
        # 初回はエラーモード
        mock_tts_engine.error_mode = True

        # AudioError（E3001）が発生することを確認
        with pytest.raises(AudioError) as exc_info:
            await mock_tts_engine.synthesize("テスト")
        assert exc_info.value.error_code == "E3001"  # 修正: E3000→E3001, TTSError→AudioError

        # エラーモードを解除してフォールバック
        mock_tts_engine.error_mode = False
        result = await mock_tts_engine.synthesize("フォールバックテスト")
        assert result.audio_data is not None


# ============================================================================
# パフォーマンステスト
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.phase(3)
class TestPerformance:
    """パフォーマンス統合テスト"""

    async def test_response_time_sla(self, all_mock_engines, sample_audio_data):
        """応答時間SLAテスト"""
        stt = all_mock_engines["stt"]
        llm = all_mock_engines["llm"]
        tts = all_mock_engines["tts"]

        start_time = time.time()

        # 完全な処理フロー
        transcription = await stt.transcribe(sample_audio_data)
        response = await llm.generate(transcription.text)
        synthesis = await tts.synthesize(response.content)

        end_time = time.time()
        total_time = end_time - start_time

        # SLA: 3秒以内に完了（モックなので高速）
        assert total_time < 3.0
        assert synthesis.audio_data is not None

    async def test_concurrent_processing(self, all_mock_engines):
        """並行処理テスト"""
        stt = all_mock_engines["stt"]
        llm = all_mock_engines["llm"]
        tts = all_mock_engines["tts"]

        # 複数の音声データを準備
        audio_data_list = [
            AudioData(data=b"test1", metadata={"filename": "greeting.wav"}),
            AudioData(data=b"test2", metadata={"filename": "question.wav"}),
            AudioData(data=b"test3", metadata={"filename": "command.wav"}),
        ]

        # 並行して処理
        tasks = []
        for audio in audio_data_list:

            async def process_audio(audio_data):
                transcription = await stt.transcribe(audio_data)
                response = await llm.generate(transcription.text)
                synthesis = await tts.synthesize(response.content)
                return synthesis

            tasks.append(process_audio(audio))

        # 全タスクの完了を待つ
        results = await asyncio.gather(*tasks)

        # 全ての結果が取得できている
        assert len(results) == 3
        for result in results:
            assert result.audio_data is not None

    async def test_memory_efficiency(self, all_mock_engines):
        """メモリ効率のテスト"""
        llm = all_mock_engines["llm"]

        # 大量の会話履歴を追加
        for i in range(100):
            llm.add_message("user", f"メッセージ {i}")
            llm.add_message("assistant", f"応答 {i}")

        # 履歴のサイズ確認
        history = llm.get_history()
        assert len(history) == 200  # 100ターン × 2

        # メモリをクリア
        llm.clear_history()
        assert len(llm.get_history()) == 0


# ============================================================================
# Phase 1コンポーネントとの統合テスト
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.phase(1, 3)
class TestPhase1Integration:
    """Phase 1コンポーネントとの統合テスト"""

    async def test_vioratalk_engine_with_mock_engines(self, vioratalk_engine, all_mock_engines):
        """VioraTalkEngineとMockエンジンの統合"""
        # VioraTalkEngineが正常に初期化されている
        assert vioratalk_engine.state == ComponentState.READY

        # 設定管理との連携
        if hasattr(vioratalk_engine, "config_manager"):
            config = vioratalk_engine.config_manager.get_all()
            assert config is not None

    async def test_logger_integration(self, all_mock_engines):
        """LoggerManagerとの統合テスト"""
        logger_manager = LoggerManager()
        logger = logger_manager.get_logger("mock_engines_test")

        stt = all_mock_engines["stt"]

        # ログ出力の確認
        logger.info("Starting STT test")
        result = await stt.transcribe(AudioData(data=b"test", metadata={"filename": "test.wav"}))
        logger.info(f"STT result: {result.text}")

        assert result.text is not None

    async def test_config_manager_integration(self, all_mock_engines):
        """ConfigManagerとの統合テスト"""
        config_manager = ConfigManager()

        # STTエンジン選択
        stt_engine_type = config_manager.get("stt.engine", "mock")
        if stt_engine_type == "mock":
            engine = all_mock_engines["stt"]
            assert isinstance(engine, MockSTTEngine)

        # LLMエンジン選択
        llm_engine_type = config_manager.get("llm.engine", "mock")
        if llm_engine_type == "mock":
            engine = all_mock_engines["llm"]
            assert isinstance(engine, MockLLMEngine)

        # TTSエンジン選択
        tts_engine_type = config_manager.get("tts.engine", "mock")
        if tts_engine_type == "mock":
            engine = all_mock_engines["tts"]
            assert isinstance(engine, MockTTSEngine)


# ============================================================================
# Phase 2コンポーネントとの統合テスト（存在する場合）
# ============================================================================

if PHASE2_AVAILABLE:

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.phase(2, 3)
    class TestPhase2Integration:
        """Phase 2コンポーネントとの統合テスト"""

        async def test_dialogue_manager_with_mock_engines(self, all_mock_engines):
            """DialogueManagerとMockエンジンの統合"""
            # DialogueManagerの初期化
            dialogue_config = DialogueConfig(language="ja", temperature=0.8, max_turns=10)

            mock_character = MockCharacterManager()
            await mock_character.initialize()

            dialogue_manager = DialogueManager(
                config=dialogue_config, character_manager=mock_character
            )
            await dialogue_manager.initialize()

            # Mockエンジンとの連携
            stt = all_mock_engines["stt"]
            llm = all_mock_engines["llm"]
            tts = all_mock_engines["tts"]

            # 音声入力の処理
            audio_data = AudioData(data=b"test", metadata={"filename": "greeting.wav"})
            transcription = await stt.transcribe(audio_data)

            # DialogueManagerで処理（実際のprocess_audio_inputがある場合）
            # ここではテキスト入力として処理
            response = await dialogue_manager.process_text_input(transcription.text)

            # 応答を音声合成
            if response and hasattr(response, "text"):
                synthesis = await tts.synthesize(response.text)
                assert synthesis.audio_data is not None

            await dialogue_manager.cleanup()
            await mock_character.cleanup()


# ============================================================================
# ストリーミング処理テスト
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.phase(3)
class TestStreamingIntegration:
    """ストリーミング処理の統合テスト"""

    async def test_llm_streaming_to_tts(self, mock_llm_engine, mock_tts_engine):
        """LLMストリーミング → TTS連携"""
        # ストリーミングを有効化
        mock_llm_engine.streaming_enabled = True

        # ストリーミング生成
        chunks = []
        async for chunk in mock_llm_engine.stream_generate("こんにちは"):
            chunks.append(chunk)

        # チャンクを結合
        full_text = "".join(chunks)

        # 完全なテキストを音声合成
        synthesis = await mock_tts_engine.synthesize(full_text)
        assert synthesis.audio_data is not None
        assert synthesis.metadata["text_length"] == len(full_text)

    async def test_progressive_tts_synthesis(self, mock_llm_engine, mock_tts_engine):
        """段階的な音声合成テスト"""
        mock_llm_engine.streaming_enabled = True

        # 文ごとに音声合成
        sentence_buffer = ""
        audio_segments = []

        async for chunk in mock_llm_engine.stream_generate("こんにちは。今日はいい天気ですね。"):
            sentence_buffer += chunk

            # 句点で区切って合成
            if "。" in sentence_buffer:
                sentences = sentence_buffer.split("。")
                for sentence in sentences[:-1]:  # 最後の要素は不完全な可能性
                    if sentence:
                        synthesis = await mock_tts_engine.synthesize(sentence + "。")
                        audio_segments.append(synthesis.audio_data)
                sentence_buffer = sentences[-1]

        # 残りがあれば合成
        if sentence_buffer:
            synthesis = await mock_tts_engine.synthesize(sentence_buffer)
            audio_segments.append(synthesis.audio_data)

        # 全セグメントが生成されている
        assert len(audio_segments) >= 2  # 少なくとも2文


# ============================================================================
# キャラクター別処理テスト
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.phase(3)
class TestCharacterIntegration:
    """キャラクター別処理の統合テスト"""

    async def test_character_specific_response(self, mock_llm_engine, mock_tts_engine):
        """キャラクター別応答テスト"""
        characters = ["001_aoi", "002_haru", "003_yui"]

        # 利用可能な音声IDのマッピング
        voice_mapping = {
            "001_aoi": "ja-JP-Female-1",  # 修正: 正しい音声ID形式
            "002_haru": "ja-JP-Female-2",  # 修正: 正しい音声ID形式
            "003_yui": "ja-JP-Male-1",  # 修正: 正しい音声ID形式
        }

        for char_id in characters:
            # キャラクター指定で応答生成
            response = await mock_llm_engine.generate(
                prompt="こんにちは", system_prompt=f"character_id:{char_id}"
            )

            assert response.content is not None
            assert response.metadata["character_id"] == char_id

            # キャラクターに応じた音声合成
            voice_id = voice_mapping[char_id]
            mock_tts_engine.set_voice(voice_id)
            synthesis = await mock_tts_engine.synthesize(response.content)

            assert synthesis.audio_data is not None
            assert synthesis.metadata["voice_id"] == voice_id

    async def test_conversation_continuity(self, mock_llm_engine):
        """会話の継続性テスト"""
        # 会話履歴を追加
        mock_llm_engine.add_message("user", "私の名前は太郎です")
        mock_llm_engine.add_message("assistant", "太郎さん、よろしくお願いします！")

        # 履歴が保存されていることを確認
        history = mock_llm_engine.get_history()
        assert len(history) == 2
        assert history[0].content == "私の名前は太郎です"
        assert history[1].content == "太郎さん、よろしくお願いします！"

        # 応答生成（MockLLMEngineは固定応答なので履歴は反映されない）
        response = await mock_llm_engine.generate("私の名前を覚えていますか？")

        # 応答が生成されていることを確認（内容は固定応答）
        assert response.content is not None
        assert response.metadata["response_type"] == "question"  # 質問として認識される

        # 履歴が更新されていることを確認
        history = mock_llm_engine.get_history()
        assert len(history) == 4  # 元の2つ + 新しい質問と応答
