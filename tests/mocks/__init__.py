"""VioraTalk Test Mocks Package"""

# Mockエンジンとデータクラスをインポート
from .mock_llm_engine import LLMResponse, Message, MockLLMEngine
from .mock_stt_engine import AudioData, MockSTTEngine, TranscriptionResult
from .mock_tts_engine import MockTTSEngine, StyleInfo, SynthesisResult, VoiceInfo, VoiceParameters

# Phase 2のMockも含める（存在する場合）
try:
    from .mock_character_manager import Character, MockCharacterManager  # Characterを追加

    _has_character_manager = True
except ImportError:
    _has_character_manager = False

# 公開APIを定義
__all__ = [
    # STT関連
    "MockSTTEngine",
    "AudioData",
    "TranscriptionResult",
    # LLM関連
    "MockLLMEngine",
    "LLMResponse",
    "Message",
    # TTS関連
    "MockTTSEngine",
    "SynthesisResult",
    "VoiceInfo",
    "StyleInfo",
    "VoiceParameters",
]

# MockCharacterManagerとCharacterが存在する場合は追加
if _has_character_manager:
    __all__.extend(["MockCharacterManager", "Character"])  # 両方追加
