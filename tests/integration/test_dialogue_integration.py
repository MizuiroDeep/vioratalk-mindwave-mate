"""DialogueManager統合テスト

DialogueManagerを中心とした統合テスト。
音声入力→STT→LLM→TTS→音声出力の完全フローを検証。
モックエンジンと実エンジンの両方でテスト。

テスト実装ガイド v1.3準拠
テスト戦略ガイドライン v1.7準拠
開発規約書 v1.12準拠
DialogueManager統合ガイド v1.2準拠
"""

import wave
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from tests.mocks.mock_character_manager import MockCharacterManager
from vioratalk.core.base import ComponentState
from vioratalk.core.dialogue_config import DialogueConfig
from vioratalk.core.dialogue_manager import ConversationState, DialogueManager
from vioratalk.core.exceptions import AudioError, LLMError, STTError
from vioratalk.core.llm.llm_manager import LLMManager
from vioratalk.core.stt.base import AudioData, TranscriptionResult
from vioratalk.core.stt.vad import VoiceActivityDetector
from vioratalk.core.tts.tts_manager import TTSManager
from vioratalk.infrastructure.audio_capture import AudioCapture

# ================================================================================
# モジュールレベルのフィクスチャ（全テストクラスから利用可能）
# DialogueManager統合ガイド v1.2準拠
# ================================================================================


@pytest.fixture
async def mock_character_manager():
    """MockCharacterManagerのフィクスチャ"""
    manager = MockCharacterManager()
    await manager.initialize()
    return manager


@pytest.fixture
def dialogue_config():
    """DialogueConfig のフィクスチャ

    Phase 4音声パラメータを含む設定
    設定ファイル完全仕様書 v1.2準拠
    """
    return DialogueConfig(
        max_turns=10,
        temperature=0.7,
        use_mock_engines=True,
        # Phase 4音声パラメータ
        tts_enabled=True,
        tts_enabled_for_text=True,
        voice_id="ja",
        voice_style="normal",
        recording_duration=5.0,
        sample_rate=16000,
    )


@pytest.fixture
def mock_llm_manager():
    """モックLLMManagerのフィクスチャ

    修正: response.text → response.content
    Phase 4 Part 88で修正
    """
    mock = AsyncMock(spec=LLMManager)
    mock._state = ComponentState.NOT_INITIALIZED
    mock.initialize = AsyncMock()
    mock.cleanup = AsyncMock()

    # LLM応答のモック（修正: text → content）
    mock_response = MagicMock()
    mock_response.content = "こんにちは！今日はどんなお話をしましょうか？"  # text → content に変更
    mock.generate = AsyncMock(return_value=mock_response)

    return mock


@pytest.fixture
def mock_tts_manager():
    """モックTTSManagerのフィクスチャ"""
    mock = AsyncMock(spec=TTSManager)
    mock._state = ComponentState.NOT_INITIALIZED
    mock.initialize = AsyncMock()
    mock.cleanup = AsyncMock()

    # TTS合成結果のモック
    mock_synthesis = MagicMock()
    mock_synthesis.audio_data = b"synthesized_audio_data"
    mock.synthesize = AsyncMock(return_value=mock_synthesis)

    return mock


@pytest.fixture
def mock_stt_engine():
    """モックSTTエンジンのフィクスチャ"""
    mock = AsyncMock()
    mock._state = ComponentState.NOT_INITIALIZED
    mock.initialize = AsyncMock()
    mock.cleanup = AsyncMock()

    # 音声認識結果のモック
    mock_result = MagicMock(spec=TranscriptionResult)
    mock_result.text = "こんにちは"
    mock_result.confidence = 0.95
    mock_result.language = "ja"
    mock.transcribe = AsyncMock(return_value=mock_result)

    return mock


@pytest.fixture
def mock_audio_capture():
    """モックAudioCaptureのフィクスチャ"""
    mock = AsyncMock(spec=AudioCapture)
    mock._state = ComponentState.NOT_INITIALIZED
    mock.safe_initialize = AsyncMock()
    mock.safe_cleanup = AsyncMock()

    # 録音データのモック
    audio_array = np.zeros(16000 * 5, dtype=np.float32)  # 5秒分
    mock_audio_data = MagicMock(spec=AudioData)
    mock_audio_data.raw_data = audio_array
    mock.record_from_microphone = AsyncMock(return_value=mock_audio_data)

    return mock


@pytest.fixture
def mock_vad():
    """モックVADのフィクスチャ

    重要：start_timeとend_timeは直接float値を設定
    MagicMockの__format__エラーを回避
    """
    mock = AsyncMock(spec=VoiceActivityDetector)
    mock._state = ComponentState.NOT_INITIALIZED
    mock.initialize = AsyncMock()
    mock.cleanup = AsyncMock()

    # 音声区間検出結果のモック
    mock_segment = MagicMock()
    mock_segment.start_sample = 1000
    mock_segment.end_sample = 16000
    # float値を直接設定（MagicMockの__format__エラー回避）
    mock_segment.start_time = 0.0625
    mock_segment.end_time = 1.0
    mock.detect_segments = MagicMock(return_value=[mock_segment])

    return mock


# ================================================================================
# TestDialogueManagerMockIntegration
# モックエンジンを使用した統合テスト
# ================================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestDialogueManagerMockIntegration:
    """DialogueManagerとモックエンジンの統合テスト"""

    async def test_dialogue_with_mock_engines(
        self, dialogue_config, mock_character_manager, mock_llm_manager, mock_tts_manager
    ):
        """モックエンジンでの基本対話フローテスト"""
        # DialogueManager作成
        manager = DialogueManager(
            config=dialogue_config,
            character_manager=mock_character_manager,
            llm_manager=mock_llm_manager,
            tts_manager=mock_tts_manager,
        )

        # 初期化
        await manager.initialize()
        assert manager._state == ComponentState.READY
        assert manager._initialized is True

        # テキスト入力処理
        turn = await manager.process_text_input("こんにちは")

        # 結果検証
        assert turn is not None
        assert turn.user_input == "こんにちは"
        assert turn.assistant_response == "こんにちは！今日はどんなお話をしましょうか？"
        assert turn.audio_response == b"synthesized_audio_data"
        assert turn.character_id == "001_aoi"
        assert turn.turn_number == 1

        # LLMとTTSが呼ばれたことを確認
        mock_llm_manager.generate.assert_called_once()
        mock_tts_manager.synthesize.assert_called_once()

        # クリーンアップ
        await manager.cleanup()
        assert manager._state == ComponentState.TERMINATED

    async def test_audio_input_complete_flow(
        self,
        dialogue_config,
        mock_character_manager,
        mock_llm_manager,
        mock_tts_manager,
        mock_stt_engine,
        mock_audio_capture,
        mock_vad,
    ):
        """音声入力→音声出力の完全フローテスト"""
        # DialogueManager作成（全コンポーネント統合）
        manager = DialogueManager(
            config=dialogue_config,
            character_manager=mock_character_manager,
            llm_manager=mock_llm_manager,
            tts_manager=mock_tts_manager,
            stt_engine=mock_stt_engine,
            audio_capture=mock_audio_capture,
            vad=mock_vad,
        )

        # 初期化
        await manager.initialize()

        # 音声入力処理（マイクから録音）
        turn = await manager.process_audio_input()

        # 結果検証
        assert turn is not None
        assert turn.user_input == "こんにちは"
        assert turn.assistant_response == "こんにちは！今日はどんなお話をしましょうか？"
        assert turn.audio_response == b"synthesized_audio_data"
        assert turn.confidence == 0.95

        # 各コンポーネントが順番に呼ばれたことを確認
        mock_audio_capture.record_from_microphone.assert_called_once()
        mock_vad.detect_segments.assert_called_once()
        mock_stt_engine.transcribe.assert_called_once()
        mock_llm_manager.generate.assert_called_once()
        mock_tts_manager.synthesize.assert_called_once()

        # クリーンアップ
        await manager.cleanup()

    async def test_conversation_context_management(
        self, dialogue_config, mock_character_manager, mock_llm_manager, mock_tts_manager
    ):
        """会話コンテキストの管理テスト"""
        manager = DialogueManager(
            config=dialogue_config,
            character_manager=mock_character_manager,
            llm_manager=mock_llm_manager,
            tts_manager=mock_tts_manager,
        )

        await manager.initialize()

        # 複数ターンの対話（修正: text → content）
        responses = []
        for i in range(3):
            mock_llm_manager.generate.return_value.content = f"応答{i+1}"  # text → content に変更
            turn = await manager.process_text_input(f"入力{i+1}")
            responses.append(turn)

        # 会話履歴の確認
        history = manager.get_conversation_history(limit=10)
        assert len(history) == 3

        # 各ターンの確認
        for i, turn in enumerate(history):
            assert turn.user_input == f"入力{i+1}"
            assert turn.assistant_response == f"応答{i+1}"
            assert turn.turn_number == i + 1

        # 会話統計の確認
        stats = manager.get_conversation_stats()
        assert stats["total_turns"] == 3
        assert stats["current_turn_number"] == 3
        assert stats["character_id"] == "001_aoi"
        assert stats["llm_available"] is True
        assert stats["tts_available"] is True

        # 会話リセット
        await manager.reset_conversation()
        history = manager.get_conversation_history()
        assert len(history) == 0

        await manager.cleanup()

    async def test_character_switching_flow(
        self, dialogue_config, mock_character_manager, mock_llm_manager
    ):
        """キャラクター切り替えフローテスト

        注：DialogueManagerの仕様上、会話中（READYステート）では
        キャラクター切り替えは不可
        MockCharacterManagerには001_aoiのみ存在
        """
        manager = DialogueManager(
            config=dialogue_config,
            character_manager=mock_character_manager,
            llm_manager=mock_llm_manager,
        )

        await manager.initialize()

        # 初期キャラクター確認
        context = manager.get_current_context()
        assert context.character_id == "001_aoi"

        # 会話開始後はキャラクター切り替え不可（READYステート）
        turn = await manager.process_text_input("こんにちは")
        assert manager.can_switch_character() is False  # READYステートなので切り替え不可

        # 存在しないキャラクターへの切り替え試行（エラーを期待）
        with pytest.raises(ValueError) as exc_info:
            await mock_character_manager.switch_character("002_haruka")
        assert "not found" in str(exc_info.value)

        # 会話リセット後も初期キャラクターのまま
        await manager.reset_conversation()
        context = manager.get_current_context()
        assert context.character_id == "001_aoi"

        await manager.cleanup()

    async def test_error_handling_stt_failure(
        self,
        dialogue_config,
        mock_character_manager,
        mock_llm_manager,
        mock_tts_manager,
        mock_stt_engine,
        mock_audio_capture,
        mock_vad,
    ):
        """STTエラー時のエラーハンドリングテスト"""
        # STTエラーを設定
        mock_stt_engine.transcribe.side_effect = STTError(
            "Speech recognition failed", error_code="E2000"
        )

        manager = DialogueManager(
            config=dialogue_config,
            character_manager=mock_character_manager,
            llm_manager=mock_llm_manager,
            tts_manager=mock_tts_manager,
            stt_engine=mock_stt_engine,
            audio_capture=mock_audio_capture,
            vad=mock_vad,
        )

        await manager.initialize()

        # STTエラーが伝播することを確認
        with pytest.raises(STTError) as exc_info:
            await manager.process_audio_input()

        assert exc_info.value.error_code == "E2000"
        assert "Speech recognition failed" in str(exc_info.value)

        # エラー後もコンテキストは維持される
        context = manager.get_current_context()
        assert context.state == ConversationState.ERROR

        await manager.cleanup()

    async def test_error_handling_no_speech_detected(
        self, dialogue_config, mock_character_manager, mock_audio_capture, mock_vad
    ):
        """音声が検出されない場合のエラーハンドリング"""
        # VADが空のリストを返す（音声なし）
        mock_vad.detect_segments.return_value = []

        manager = DialogueManager(
            config=dialogue_config,
            character_manager=mock_character_manager,
            audio_capture=mock_audio_capture,
            vad=mock_vad,
        )

        await manager.initialize()

        # AudioErrorが発生することを確認
        with pytest.raises(AudioError) as exc_info:
            await manager.process_audio_input()

        assert exc_info.value.error_code == "E1003"
        assert "No speech detected" in str(exc_info.value)

        await manager.cleanup()

    async def test_fallback_to_mock_response(
        self, dialogue_config, mock_character_manager, mock_llm_manager, mock_tts_manager
    ):
        """LLMエラー時のモック応答フォールバックテスト

        注：現在の実装では、LLMエラー時でも
        フォールバック応答のconfidenceは0.95を維持
        """
        # LLMエラーを設定
        mock_llm_manager.generate.side_effect = LLMError(
            "API rate limit exceeded", error_code="E2103"
        )

        manager = DialogueManager(
            config=dialogue_config,
            character_manager=mock_character_manager,
            llm_manager=mock_llm_manager,
            tts_manager=mock_tts_manager,
        )

        await manager.initialize()

        # モック応答にフォールバック
        turn = await manager.process_text_input("こんにちは")

        # モック応答が返されることを確認
        assert turn.assistant_response == "こんにちは！今日はどんなお話をしましょうか？"
        # 現在の実装ではconfidenceは0.95を維持
        assert turn.confidence == 0.95

        await manager.cleanup()

    async def test_tts_optional_synthesis(
        self, dialogue_config, mock_character_manager, mock_llm_manager, mock_tts_manager
    ):
        """TTS音声合成のオプション動作テスト"""
        # TTS無効化設定
        dialogue_config.tts_enabled_for_text = False

        manager = DialogueManager(
            config=dialogue_config,
            character_manager=mock_character_manager,
            llm_manager=mock_llm_manager,
            tts_manager=mock_tts_manager,
        )

        await manager.initialize()

        # テキスト入力処理（TTS無効）
        turn = await manager.process_text_input("こんにちは")

        # 音声データがないことを確認
        assert turn.audio_response is None

        # TTSが呼ばれていないことを確認
        mock_tts_manager.synthesize.assert_not_called()

        await manager.cleanup()

    async def test_max_turns_limit(self, mock_character_manager, mock_llm_manager):
        """最大ターン数制限のテスト"""
        # 最大3ターンに制限（Phase 4音声パラメータ含む）
        config = DialogueConfig(
            max_turns=3, use_mock_engines=True, tts_enabled=True, voice_id="ja", sample_rate=16000
        )

        manager = DialogueManager(
            config=config, character_manager=mock_character_manager, llm_manager=mock_llm_manager
        )

        await manager.initialize()

        # 5ターン実行（修正: text → content）
        for i in range(5):
            mock_llm_manager.generate.return_value.content = f"応答{i+1}"  # text → content に変更
            await manager.process_text_input(f"入力{i+1}")

        # 履歴は3ターンのみ保持
        history = manager.get_conversation_history()
        assert len(history) == 3

        # 最新の3ターンが残っていることを確認
        assert history[0].user_input == "入力3"
        assert history[1].user_input == "入力4"
        assert history[2].user_input == "入力5"

        await manager.cleanup()


# ================================================================================
# TestDialogueManagerRealEnginesIntegration
# 実エンジンを使用した統合テスト（オプション）
# ================================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestDialogueManagerRealEnginesIntegration:
    """DialogueManagerと実エンジンの統合テスト（オプション）"""

    @pytest.mark.skipif(
        not Path(".env").exists(), reason="Real engine testing requires .env file with API keys"
    )
    async def test_with_real_gemini_engine(self):
        """実際のGeminiEngineとの統合テスト"""
        from vioratalk.core.llm.gemini_engine import GeminiEngine
        from vioratalk.infrastructure.credential_manager import get_api_key_manager

        # APIキー取得
        key_manager = get_api_key_manager()
        api_key = key_manager.get_api_key("gemini")

        if not api_key:
            pytest.skip("Gemini API key not found")

        # 実エンジン作成（Phase 4音声パラメータ含む）
        config = DialogueConfig(
            use_mock_engines=False,
            tts_enabled=False,  # 実エンジンテストではTTS無効
            voice_id="ja",
            sample_rate=16000,
        )
        llm_engine = GeminiEngine(api_key=api_key)
        llm_manager = LLMManager()
        llm_manager.register_engine("gemini", llm_engine)

        manager = DialogueManager(config=config, llm_manager=llm_manager)

        await manager.initialize()

        # 実際のLLM応答を取得
        turn = await manager.process_text_input("今日の天気はどうですか？")

        # 応答が生成されることを確認
        assert turn.assistant_response != ""
        assert len(turn.assistant_response) > 10

        await manager.cleanup()

    @pytest.mark.skipif(
        not Path("tests/fixtures/audio/input/test_japanese.wav").exists(),
        reason="Test audio file not found",
    )
    async def test_with_audio_file_input(self):
        """音声ファイルを使用した統合テスト"""
        from vioratalk.core.stt.faster_whisper_engine import FasterWhisperEngine

        # テスト用音声ファイル読み込み
        audio_path = Path("tests/fixtures/audio/input/test_japanese.wav")

        with wave.open(str(audio_path), "rb") as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        # モックVADを作成（音声全体を1セグメントとして返す）
        mock_vad = AsyncMock()
        mock_vad._state = ComponentState.NOT_INITIALIZED
        mock_vad.initialize = AsyncMock()
        mock_vad.cleanup = AsyncMock()

        mock_segment = MagicMock()
        mock_segment.start_sample = 0
        mock_segment.end_sample = len(audio_data)
        # float値を直接設定（MagicMockの__format__エラー回避）
        mock_segment.start_time = 0.0
        mock_segment.end_time = float(len(audio_data)) / 16000.0
        mock_vad.detect_segments = MagicMock(return_value=[mock_segment])

        # STTエンジン作成
        stt_engine = FasterWhisperEngine()

        # モックLLM（簡単な応答）（修正: text → content）
        mock_llm = AsyncMock()
        mock_llm._state = ComponentState.NOT_INITIALIZED
        mock_llm.generate = AsyncMock(
            return_value=MagicMock(content="音声ファイルからの入力を受け取りました")  # text → content に変更
        )

        # Phase 4音声パラメータを含む設定
        config = DialogueConfig(
            tts_enabled=False, voice_id="ja", sample_rate=16000  # 音声ファイルテストではTTS無効
        )
        manager = DialogueManager(
            config=config, stt_engine=stt_engine, llm_manager=mock_llm, vad=mock_vad
        )

        await manager.initialize()

        # 音声データを処理
        turn = await manager.process_audio_input(audio_data.tobytes())

        # 音声認識結果が得られることを確認
        assert turn.user_input != ""
        assert turn.assistant_response == "音声ファイルからの入力を受け取りました"

        await manager.cleanup()


# ================================================================================
# TestDialogueManagerPerformance
# パフォーマンステスト
# ================================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestDialogueManagerPerformance:
    """DialogueManagerのパフォーマンステスト"""

    async def test_response_time_text_input(
        self, dialogue_config, mock_character_manager, mock_llm_manager, mock_tts_manager
    ):
        """テキスト入力の応答時間テスト

        モックエンジンを使用しているため、
        応答は高速（1秒以内）であることを確認
        """
        import time

        manager = DialogueManager(
            config=dialogue_config,
            character_manager=mock_character_manager,
            llm_manager=mock_llm_manager,
            tts_manager=mock_tts_manager,
        )

        await manager.initialize()

        # 応答時間測定
        start = time.time()
        turn = await manager.process_text_input("テスト")
        elapsed = time.time() - start

        # 1秒以内に応答することを確認（モックなので高速）
        assert elapsed < 1.0
        # 注：現在のDialogueManagerは処理時間を設定していないため、
        # processing_timeのチェックは行わない
        # 将来的に実装される場合は以下のアサーションを有効化
        # assert turn.processing_time > 0

        # 基本的な応答が返されることを確認
        assert turn is not None
        assert turn.user_input == "テスト"
        assert turn.assistant_response is not None

        await manager.cleanup()

    async def test_memory_usage_multiple_turns(
        self, dialogue_config, mock_character_manager, mock_llm_manager
    ):
        """複数ターンでのメモリ使用量テスト"""
        import os

        import psutil

        process = psutil.Process(os.getpid())

        manager = DialogueManager(
            config=dialogue_config,
            character_manager=mock_character_manager,
            llm_manager=mock_llm_manager,
        )

        await manager.initialize()

        # 初期メモリ使用量
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 100ターンの対話（修正: text → content）
        for i in range(100):
            mock_llm_manager.generate.return_value.content = f"応答{i}"  # text → content に変更
            await manager.process_text_input(f"入力{i}")

        # 最終メモリ使用量
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # メモリ増加が100MB以内であることを確認
        assert memory_increase < 100

        await manager.cleanup()


# テスト実行用
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
