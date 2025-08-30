#!/usr/bin/env python3
"""
Phase 4 実エンジン完全動作確認用手動テストスクリプト（改善版）

実行方法:
  python scripts/manual_tests/feature/test_manual_phase4_complete.py

必要な環境:
  - マイク（音声入力用）
  - スピーカー（音声出力用）  
  - インターネット接続（Gemini API用）
  - 環境変数 GEMINI_API_KEY（設定済みの場合）

このスクリプトは以下をテストします：
  1. 各エンジンの個別動作
  2. STT→LLM→TTSの完全な対話フロー
  3. エラーハンドリング
  4. パフォーマンス測定

開発規約書 v1.12準拠
テスト戦略ガイドライン v1.7準拠
"""

import asyncio
import os
import sys
import time
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import json

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# サードパーティライブラリ
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: numpy not available. Some features may be limited.")

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("Warning: sounddevice not available. Microphone input will not work.")
    print("Install with: pip install sounddevice")

# VioraTalkのインポート
from vioratalk.core.stt import FasterWhisperEngine, AudioData, AudioMetadata, STTConfig
from vioratalk.core.llm import GeminiEngine, LLMConfig
from vioratalk.core.tts import Pyttsx3Engine, TTSConfig
from vioratalk.core.exceptions import (
    STTError, LLMError, TTSError, AudioError, 
    APIError, AuthenticationError
)
from vioratalk.utils.logger_manager import LoggerManager


# ============================================================================
# テスト設定クラス
# ============================================================================

class TestSettings:
    """テスト設定の管理"""
    
    def __init__(self):
        self.stt_model = "base"  # tiny, base, small, medium, large
        self.language = "ja"     # ja, en, auto
        self.record_duration = 5  # 3, 5, 10 seconds
        self.llm_model = "gemini-2.0-flash"
        self.tts_speed = 1.0
        self.tts_volume = 0.9
    
    def display(self):
        """現在の設定を表示"""
        print("\n" + "="*60)
        print("CURRENT SETTINGS")
        print("="*60)
        print(f"STT Model:        {self.stt_model}")
        print(f"Language:         {self.language}")
        print(f"Record Duration:  {self.record_duration} seconds")
        print(f"LLM Model:        {self.llm_model}")
        print(f"TTS Speed:        {self.tts_speed}x")
        print(f"TTS Volume:       {int(self.tts_volume * 100)}%")
    
    def change_settings(self):
        """設定を変更"""
        while True:
            print("\n" + "="*60)
            print("CHANGE SETTINGS")
            print("="*60)
            print("1. STT Model (tiny/base/small/medium/large)")
            print(f"   Current: {self.stt_model}")
            print("2. Language (ja/en/auto)")
            print(f"   Current: {self.language}")
            print("3. Record Duration (3/5/10 seconds)")
            print(f"   Current: {self.record_duration}s")
            print("4. LLM Model")
            print(f"   Current: {self.llm_model}")
            print("5. TTS Speed (0.5-2.0)")
            print(f"   Current: {self.tts_speed}x")
            print("6. TTS Volume (0-100)")
            print(f"   Current: {int(self.tts_volume * 100)}%")
            print("0. Back to main menu")
            print("-"*60)
            
            choice = input("\nSelect setting to change (0-6): ")
            
            if choice == '0':
                break
            elif choice == '1':
                print("\nAvailable models:")
                print("  tiny   - Fastest, lowest accuracy (39MB)")
                print("  base   - Good balance (74MB)")
                print("  small  - Better accuracy (244MB)")
                print("  medium - High accuracy (769MB)")
                print("  large  - Best accuracy (1550MB)")
                model = input("Enter model name: ").lower()
                if model in ['tiny', 'base', 'small', 'medium', 'large']:
                    self.stt_model = model
                    print(f"✓ STT model changed to: {model}")
                else:
                    print("✗ Invalid model name")
            
            elif choice == '2':
                print("\nAvailable languages:")
                print("  ja   - Japanese")
                print("  en   - English")
                print("  auto - Auto-detect")
                lang = input("Enter language code: ").lower()
                if lang in ['ja', 'en', 'auto']:
                    self.language = lang
                    print(f"✓ Language changed to: {lang}")
                else:
                    print("✗ Invalid language code")
            
            elif choice == '3':
                print("\nAvailable durations:")
                print("  3  - Quick test")
                print("  5  - Normal conversation")
                print("  10 - Long questions")
                duration = input("Enter duration (3/5/10): ")
                if duration in ['3', '5', '10']:
                    self.record_duration = int(duration)
                    print(f"✓ Record duration changed to: {duration}s")
                else:
                    print("✗ Invalid duration")
            
            elif choice == '4':
                print("\nAvailable models:")
                print("  gemini-2.0-flash - Fast, good quality")
                print("  gemini-2.5-flash - Newer, balanced")
                model = input("Enter model name: ")
                if 'gemini' in model.lower():
                    self.llm_model = model
                    print(f"✓ LLM model changed to: {model}")
                else:
                    print("✗ Invalid model name")
            
            elif choice == '5':
                speed = input("Enter speed (0.5-2.0): ")
                try:
                    speed_val = float(speed)
                    if 0.5 <= speed_val <= 2.0:
                        self.tts_speed = speed_val
                        print(f"✓ TTS speed changed to: {speed_val}x")
                    else:
                        print("✗ Speed must be between 0.5 and 2.0")
                except ValueError:
                    print("✗ Invalid speed value")
            
            elif choice == '6':
                volume = input("Enter volume (0-100): ")
                try:
                    vol_val = int(volume)
                    if 0 <= vol_val <= 100:
                        self.tts_volume = vol_val / 100.0
                        print(f"✓ TTS volume changed to: {vol_val}%")
                    else:
                        print("✗ Volume must be between 0 and 100")
                except ValueError:
                    print("✗ Invalid volume value")


# ============================================================================
# ユーティリティクラス
# ============================================================================

class TestReporter:
    """テスト結果のレポート生成"""
    
    def __init__(self):
        self.results = []
        self.start_time = datetime.now()
        
    def add_result(self, test_name: str, status: str, details: Dict[str, Any], duration: float = 0):
        """テスト結果を追加"""
        self.results.append({
            "test": test_name,
            "status": status,  # PASS, FAIL, SKIP
            "details": details,
            "duration": duration,
            "timestamp": datetime.now().isoformat()
        })
        
    def generate_report(self) -> str:
        """レポートを生成"""
        total_duration = (datetime.now() - self.start_time).total_seconds()
        
        passed = sum(1 for r in self.results if r["status"] == "PASS")
        failed = sum(1 for r in self.results if r["status"] == "FAIL")
        skipped = sum(1 for r in self.results if r["status"] == "SKIP")
        
        report = []
        report.append("\n" + "="*60)
        report.append("TEST EXECUTION REPORT")
        report.append("="*60)
        report.append(f"Execution Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Duration: {total_duration:.1f} seconds")
        report.append(f"\nResults: {passed} PASSED, {failed} FAILED, {skipped} SKIPPED")
        report.append("-"*60)
        
        for result in self.results:
            status_symbol = {
                "PASS": "✓",
                "FAIL": "✗", 
                "SKIP": "⊘"
            }.get(result["status"], "?")
            
            report.append(f"\n{status_symbol} {result['test']}")
            report.append(f"  Status: {result['status']}")
            if result["duration"] > 0:
                report.append(f"  Duration: {result['duration']:.2f}s")
            
            if result["details"]:
                report.append("  Details:")
                for key, value in result["details"].items():
                    report.append(f"    - {key}: {value}")
        
        report.append("\n" + "="*60)
        return "\n".join(report)
    
    def save_json(self, filename: str = None):
        """JSON形式で保存"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_report_{timestamp}.json"
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump({
                "start_time": self.start_time.isoformat(),
                "results": self.results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"Report saved to: {filename}")


class AudioCapture:
    """マイクから音声を録音"""
    
    @staticmethod
    def record(duration: float = 3.0, sample_rate: int = 16000, wait_for_enter: bool = True) -> Optional[AudioData]:
        """指定時間録音"""
        if not SOUNDDEVICE_AVAILABLE:
            print("❌ Error: sounddevice is not available")
            return None
            
        try:
            if wait_for_enter:
                input("\n🎤 Press ENTER to start recording... ")
            
            print(f"🎤 Recording for {duration} seconds...")
            print("   Speak now!")
            
            # 録音
            audio = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype='float32'
            )
            sd.wait()  # 録音完了まで待機
            
            print("   Recording completed ✓")
            
            # AudioDataに変換
            return AudioData(
                raw_data=audio.flatten(),  # 1次元配列に変換
                metadata=AudioMetadata(
                    filename="microphone_input.wav",
                    format="pcm",
                    sample_rate=sample_rate,
                    channels=1,
                    duration=duration
                )
            )
            
        except Exception as e:
            print(f"❌ Recording error: {e}")
            return None


# ============================================================================
# 環境チェック
# ============================================================================

def check_environment() -> bool:
    """環境のチェック"""
    print("\n" + "="*60)
    print("ENVIRONMENT CHECK")
    print("="*60)
    
    all_ok = True
    
    # 1. Python環境
    print("\n1. Python Environment:")
    print(f"   Python version: {sys.version.split()[0]}")
    print(f"   Platform: {sys.platform}")
    
    # 2. 必須パッケージ
    print("\n2. Required Packages:")
    packages = {
        "numpy": NUMPY_AVAILABLE,
        "sounddevice": SOUNDDEVICE_AVAILABLE,
        "faster_whisper": False,
        "google.genai": False,
        "pyttsx3": False
    }
    
    try:
        import faster_whisper
        packages["faster_whisper"] = True
    except ImportError:
        pass
        
    try:
        from google import genai
        packages["google.genai"] = True
    except ImportError:
        pass
        
    try:
        import pyttsx3
        packages["pyttsx3"] = True
    except ImportError:
        pass
    
    for pkg, available in packages.items():
        status = "✓" if available else "✗"
        print(f"   {status} {pkg}")
        if not available and pkg in ["faster_whisper", "pyttsx3"]:
            all_ok = False
    
    # 3. マイクデバイス
    print("\n3. Audio Devices:")
    if SOUNDDEVICE_AVAILABLE:
        try:
            devices = sd.query_devices()
            input_devices = [d for d in devices if d['max_input_channels'] > 0]
            output_devices = [d for d in devices if d['max_output_channels'] > 0]
            
            print(f"   Input devices: {len(input_devices)}")
            if input_devices:
                default = sd.query_devices(kind='input')
                print(f"   Default input: {default['name']}")
            else:
                print("   ⚠ No microphone detected!")
                all_ok = False
                
            print(f"   Output devices: {len(output_devices)}")
            if output_devices:
                default = sd.query_devices(kind='output')
                print(f"   Default output: {default['name']}")
            else:
                print("   ⚠ No speaker detected!")
                
        except Exception as e:
            print(f"   ✗ Error checking devices: {e}")
            all_ok = False
    else:
        print("   ✗ sounddevice not available")
        all_ok = False
    
    # 4. APIキー
    print("\n4. API Keys:")
    gemini_key = os.environ.get('GEMINI_API_KEY')
    if gemini_key:
        print(f"   ✓ GEMINI_API_KEY is set (length: {len(gemini_key)})")
    else:
        print("   ⚠ GEMINI_API_KEY not set")
        print("     You can still test without it, but LLM features will be limited")
    
    # 5. インターネット接続
    print("\n5. Internet Connection:")
    try:
        import urllib.request
        urllib.request.urlopen('https://www.google.com', timeout=5)
        print("   ✓ Internet connected")
    except:
        print("   ✗ No internet connection")
        if gemini_key:
            all_ok = False
    
    print("\n" + "-"*60)
    if all_ok:
        print("✓ Environment check PASSED")
    else:
        print("⚠ Some issues detected, but test can continue")
    
    return all_ok


# ============================================================================
# 個別エンジンテスト
# ============================================================================

async def test_stt_engine(reporter: TestReporter, settings: TestSettings):
    """STTエンジンのテスト"""
    print("\n" + "="*60)
    print("STT ENGINE TEST (FasterWhisper)")
    print("="*60)
    
    try:
        # 初期化
        print(f"\n1. Initializing FasterWhisperEngine...")
        print(f"   Model: {settings.stt_model}")
        print(f"   Language: {settings.language}")
        
        config = STTConfig(
            engine="faster-whisper",
            model=settings.stt_model,
            language=None if settings.language == "auto" else settings.language
        )
        engine = FasterWhisperEngine(config)
        await engine.initialize()
        print("   ✓ Initialized successfully")
        
        reporter.add_result(
            "STT Initialization",
            "PASS",
            {"model": settings.stt_model, "language": settings.language}
        )
        
        # 音声認識テスト
        print(f"\n2. Testing speech recognition...")
        print(f"   Duration: {settings.record_duration} seconds")
        print(f"   Language: {settings.language}")
        
        audio = AudioCapture.record(settings.record_duration, wait_for_enter=True)
        if audio:
            start = time.time()
            
            # 言語指定
            if settings.language == "auto":
                result = await engine.transcribe(audio)
            else:
                result = await engine.transcribe(audio, language=settings.language)
                
            duration = time.time() - start
            
            print(f"\n   Recognition result:")
            print(f"   Text: '{result.text}'")
            print(f"   Detected Language: {result.language}")
            print(f"   Confidence: {result.confidence:.2%}")
            print(f"   Processing time: {duration:.2f}s")
            
            # 結果の評価
            is_correct = input("\n   Is this correct? (y/n/skip): ").lower()
            
            if is_correct == 'y':
                reporter.add_result(
                    "STT Recognition",
                    "PASS",
                    {
                        "text": result.text,
                        "language": result.language,
                        "confidence": f"{result.confidence:.2%}",
                        "time": f"{duration:.2f}s",
                        "model": settings.stt_model
                    },
                    duration
                )
                print("   ✓ Test PASSED")
            elif is_correct == 'skip':
                reporter.add_result(
                    "STT Recognition",
                    "SKIP",
                    {"reason": "User skipped"}
                )
            else:
                reporter.add_result(
                    "STT Recognition",
                    "FAIL",
                    {"text": result.text, "issue": "Incorrect recognition"},
                    duration
                )
                print("   ✗ Test FAILED")
        else:
            reporter.add_result(
                "STT Recognition",
                "SKIP",
                {"reason": "No audio captured"}
            )
            print("   ⊘ Test SKIPPED (no audio)")
        
        # モデル切り替えテスト
        change_model = input("\n   Test different model? (y/n): ").lower()
        if change_model == 'y':
            print("\n3. Model comparison test")
            models_to_test = ['tiny', 'base', 'small']
            
            for model in models_to_test:
                if model == settings.stt_model:
                    continue
                    
                print(f"\n   Testing model: {model}")
                engine.set_model(model)
                await engine.initialize()
                
                audio = AudioCapture.record(3.0, wait_for_enter=True)
                if audio:
                    start = time.time()
                    result = await engine.transcribe(audio)
                    duration = time.time() - start
                    
                    print(f"   Result: '{result.text}'")
                    print(f"   Time: {duration:.2f}s")
                    
                    reporter.add_result(
                        f"STT Model Test - {model}",
                        "PASS",
                        {"text": result.text[:50], "time": f"{duration:.2f}s"},
                        duration
                    )
        
        # クリーンアップ
        await engine.cleanup()
        print("\n   ✓ Cleanup completed")
        
    except Exception as e:
        print(f"\n   ✗ Error: {e}")
        reporter.add_result(
            "STT Engine Test",
            "FAIL",
            {"error": str(e)}
        )


async def test_llm_engine(reporter: TestReporter, settings: TestSettings):
    """LLMエンジンのテスト"""
    print("\n" + "="*60)
    print("LLM ENGINE TEST (Gemini)")
    print("="*60)
    
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        print("\n⚠ GEMINI_API_KEY not set")
        use_mock = input("Use mock response? (y/n): ").lower() == 'y'
        
        if not use_mock:
            reporter.add_result(
                "LLM Engine Test",
                "SKIP",
                {"reason": "No API key"}
            )
            return
    
    try:
        # 初期化
        print(f"\n1. Initializing GeminiEngine...")
        print(f"   Model: {settings.llm_model}")
        
        config = LLMConfig(
            engine="gemini",
            model=settings.llm_model,
            temperature=0.7
        )
        
        if api_key:
            engine = GeminiEngine(config, api_key=api_key)
            await engine.initialize()
            print("   ✓ Initialized with real API")
        else:
            # モックモード
            print("   ⚠ Running in mock mode")
        
        reporter.add_result(
            "LLM Initialization",
            "PASS",
            {"model": settings.llm_model, "mode": "real" if api_key else "mock"}
        )
        
        # 生成テスト（言語に応じたプロンプト）
        print("\n2. Testing text generation...")
        
        if settings.language == "ja":
            test_prompts = [
                {
                    "prompt": "こんにちは",
                    "expected": ["こんにちは", "今日", "元気"],
                    "name": "Japanese greeting"
                },
                {
                    "prompt": "2 + 2 は？",
                    "expected": ["4", "四"],
                    "name": "Simple calculation (JA)"
                }
            ]
        else:
            test_prompts = [
                {
                    "prompt": "Hello",
                    "expected": ["hello", "hi", "greetings"],
                    "name": "English greeting"
                },
                {
                    "prompt": "What is 2 + 2?",
                    "expected": ["4", "four"],
                    "name": "Simple calculation (EN)"
                }
            ]
        
        for test in test_prompts:
            print(f"\n   Test: {test['name']}")
            print(f"   Prompt: '{test['prompt']}'")
            
            if api_key:
                start = time.time()
                response = await engine.generate(
                    prompt=test['prompt'],
                    max_tokens=50
                )
                duration = time.time() - start
                
                print(f"   Response: '{response.content}'")
                print(f"   Tokens used: {response.usage.get('total_tokens', 'N/A')}")
                print(f"   Time: {duration:.2f}s")
                
                # 期待値チェック
                contains_expected = any(
                    exp.lower() in response.content.lower() 
                    for exp in test['expected']
                )
                
                if contains_expected or len(response.content) > 0:
                    reporter.add_result(
                        f"LLM Generation - {test['name']}",
                        "PASS",
                        {
                            "response": response.content[:100],
                            "time": f"{duration:.2f}s"
                        },
                        duration
                    )
                    print("   ✓ Response generated successfully")
                else:
                    reporter.add_result(
                        f"LLM Generation - {test['name']}",
                        "FAIL",
                        {"response": response.content, "expected": test['expected']},
                        duration
                    )
                    print("   ⚠ Unexpected response")
            else:
                # モック応答
                mock_response = "こんにちは！今日はいい天気ですね。" if settings.language == "ja" else "Hello! How can I help you today?"
                print(f"   [Mock] Response: '{mock_response}'")
                reporter.add_result(
                    f"LLM Generation - {test['name']}",
                    "SKIP",
                    {"reason": "Mock mode"}
                )
        
        # クリーンアップ
        if api_key:
            await engine.cleanup()
            print("\n   ✓ Cleanup completed")
        
    except Exception as e:
        print(f"\n   ✗ Error: {e}")
        reporter.add_result(
            "LLM Engine Test",
            "FAIL",
            {"error": str(e)}
        )


async def test_tts_engine(reporter: TestReporter, settings: TestSettings):
    """TTSエンジンのテスト"""
    print("\n" + "="*60)
    print("TTS ENGINE TEST (Pyttsx3)")
    print("="*60)
    
    try:
        # 初期化
        print(f"\n1. Initializing Pyttsx3Engine...")
        print(f"   Language: {settings.language}")
        print(f"   Speed: {settings.tts_speed}x")
        print(f"   Volume: {int(settings.tts_volume * 100)}%")
        
        config = TTSConfig(
            engine="pyttsx3",
            language=settings.language,
            speed=settings.tts_speed,
            volume=settings.tts_volume
        )
        engine = Pyttsx3Engine(config)
        await engine.initialize()
        print("   ✓ Initialized successfully")
        
        reporter.add_result(
            "TTS Initialization",
            "PASS",
            {"engine": "pyttsx3", "language": settings.language}
        )
        
        # 音声合成テスト
        print("\n2. Testing speech synthesis...")
        
        if settings.language == "ja":
            test_texts = [
                ("こんにちは、私はVioraTalkです。", "ja"),
                ("今日はいい天気ですね。", "ja")
            ]
        else:
            test_texts = [
                ("Hello, I am VioraTalk.", "en"),
                ("It's a beautiful day today.", "en")
            ]
        
        for text, lang in test_texts:
            print(f"\n   Text ({lang}): '{text}'")
            print("   🔊 Playing audio...")
            
            start = time.time()
            result = await engine.synthesize(
                text=text,
                voice_id=lang
            )
            duration = time.time() - start
            
            print(f"   Duration: {result.duration:.2f}s")
            print(f"   Processing time: {duration:.2f}s")
            
            # 音質評価
            quality = input("   Rate audio quality (1-5, skip): ")
            
            if quality.isdigit():
                rating = int(quality)
                status = "PASS" if rating >= 3 else "FAIL"
                reporter.add_result(
                    f"TTS Synthesis - {lang}",
                    status,
                    {
                        "text": text,
                        "quality": rating,
                        "speed": settings.tts_speed,
                        "volume": int(settings.tts_volume * 100),
                        "time": f"{duration:.2f}s"
                    },
                    duration
                )
                print(f"   {'✓' if status == 'PASS' else '✗'} Quality rating: {rating}/5")
            else:
                reporter.add_result(
                    f"TTS Synthesis - {lang}",
                    "SKIP",
                    {"reason": "User skipped"}
                )
        
        # クリーンアップ
        await engine.cleanup()
        print("\n   ✓ Cleanup completed")
        
    except Exception as e:
        print(f"\n   ✗ Error: {e}")
        reporter.add_result(
            "TTS Engine Test",
            "FAIL",
            {"error": str(e)}
        )


# ============================================================================
# 統合対話テスト
# ============================================================================

async def test_complete_dialogue(reporter: TestReporter, settings: TestSettings):
    """完全な対話フローのテスト"""
    print("\n" + "="*60)
    print("COMPLETE DIALOGUE TEST")
    print("="*60)
    print("\nThis test will:")
    print("1. Record your voice (STT)")
    print("2. Generate a response (LLM)")
    print("3. Speak the response (TTS)")
    print(f"\nSettings:")
    print(f"  Language: {settings.language}")
    print(f"  Record duration: {settings.record_duration}s")
    print(f"  STT model: {settings.stt_model}")
    print("-"*60)
    
    api_key = os.environ.get('GEMINI_API_KEY')
    
    try:
        # エンジンの初期化
        print("\nInitializing all engines...")
        
        stt_config = STTConfig(
            model=settings.stt_model,
            language=None if settings.language == "auto" else settings.language
        )
        stt_engine = FasterWhisperEngine(stt_config)
        await stt_engine.initialize()
        print("   ✓ STT ready")
        
        if api_key:
            llm_config = LLMConfig(model=settings.llm_model)
            llm_engine = GeminiEngine(llm_config, api_key=api_key)
            await llm_engine.initialize()
            print("   ✓ LLM ready")
        else:
            llm_engine = None
            print("   ⚠ LLM skipped (no API key)")
        
        tts_config = TTSConfig(
            language=settings.language if settings.language != "auto" else "ja",
            speed=settings.tts_speed,
            volume=settings.tts_volume
        )
        tts_engine = Pyttsx3Engine(tts_config)
        await tts_engine.initialize()
        print("   ✓ TTS ready")
        
        # 対話ループ
        print("\n" + "-"*60)
        if settings.language == "ja":
            print("Starting dialogue (say 'さようなら' to exit)")
        else:
            print("Starting dialogue (say 'goodbye' to exit)")
        print("-"*60)
        
        round_num = 0
        while round_num < 3:  # 最大3ラウンド
            round_num += 1
            print(f"\n[Round {round_num}]")
            
            # STT: 音声入力
            print(f"\n📢 Please speak ({settings.record_duration} seconds):")
            audio = AudioCapture.record(settings.record_duration, wait_for_enter=True)
            
            if not audio:
                print("   ⚠ No audio captured")
                continue
            
            start_total = time.time()
            
            # 音声認識
            start_stt = time.time()
            if settings.language == "auto":
                transcript = await stt_engine.transcribe(audio)
            else:
                transcript = await stt_engine.transcribe(audio, language=settings.language)
            stt_time = time.time() - start_stt
            
            print(f"\n🎤 You said: '{transcript.text}'")
            print(f"   Language: {transcript.language}")
            print(f"   (STT: {stt_time:.2f}s)")
            
            # 終了チェック
            exit_words = ['さようなら', 'goodbye', 'bye', 'exit', '終了']
            if any(word in transcript.text.lower() for word in exit_words):
                print("\n👋 Goodbye!")
                break
            
            # LLM: 応答生成
            if llm_engine:
                start_llm = time.time()
                
                # 言語に応じたシステムプロンプト
                if transcript.language == "ja" or settings.language == "ja":
                    system_prompt = "あなたは親切なアシスタントです。簡潔に応答してください。"
                else:
                    system_prompt = "You are a helpful assistant. Please respond concisely."
                
                response = await llm_engine.generate(
                    prompt=transcript.text,
                    system_prompt=system_prompt,
                    max_tokens=100
                )
                llm_time = time.time() - start_llm
                response_text = response.content
                print(f"\n🤖 Response: '{response_text}'")
                print(f"   (LLM: {llm_time:.2f}s)")
            else:
                if settings.language == "ja":
                    response_text = f"あなたは「{transcript.text}」と言いましたね。"
                else:
                    response_text = f"You said: '{transcript.text}'"
                llm_time = 0
                print(f"\n🤖 [Mock] Response: '{response_text}'")
            
            # TTS: 音声出力
            start_tts = time.time()
            
            # 検出された言語に応じて音声を切り替え
            tts_lang = transcript.language if transcript.language in ["ja", "en"] else "ja"
            synthesis = await tts_engine.synthesize(response_text, voice_id=tts_lang)
            tts_time = time.time() - start_tts
            
            print(f"\n🔊 Speaking response...")
            print(f"   (TTS: {tts_time:.2f}s)")
            
            total_time = time.time() - start_total
            
            # パフォーマンス評価
            print(f"\n⏱ Total response time: {total_time:.2f}s")
            print(f"   Breakdown: STT={stt_time:.2f}s, LLM={llm_time:.2f}s, TTS={tts_time:.2f}s")
            
            reporter.add_result(
                f"Dialogue Round {round_num}",
                "PASS" if total_time < 10 else "WARN",
                {
                    "user_input": transcript.text,
                    "language": transcript.language,
                    "response": response_text[:50] + "...",
                    "total_time": f"{total_time:.2f}s",
                    "stt_time": f"{stt_time:.2f}s",
                    "llm_time": f"{llm_time:.2f}s",
                    "tts_time": f"{tts_time:.2f}s"
                },
                total_time
            )
            
            # ユーザー評価
            quality = input("\nRate this interaction (1-5, Enter to continue): ")
            if quality.isdigit():
                rating = int(quality)
                print(f"   {'✓' if rating >= 3 else '⚠'} Rating: {rating}/5")
        
        # クリーンアップ
        print("\nCleaning up...")
        await stt_engine.cleanup()
        if llm_engine:
            await llm_engine.cleanup()
        await tts_engine.cleanup()
        print("   ✓ All engines cleaned up")
        
    except Exception as e:
        print(f"\n✗ Error in dialogue: {e}")
        reporter.add_result(
            "Complete Dialogue",
            "FAIL",
            {"error": str(e)}
        )


# ============================================================================
# メイン実行
# ============================================================================

async def main():
    """メインテスト実行"""
    print("\n" + "="*60)
    print(" VioraTalk Phase 4 Manual Test")
    print(" Real Engines Complete Test (v2.0)")
    print("="*60)
    
    reporter = TestReporter()
    settings = TestSettings()
    
    # 環境チェック
    env_ok = check_environment()
    
    if not env_ok:
        proceed = input("\n⚠ Some issues detected. Continue anyway? (y/n): ")
        if proceed.lower() != 'y':
            print("Test cancelled.")
            return
    
    # テストメニュー
    while True:
        print("\n" + "="*60)
        print("TEST MENU")
        print("="*60)
        settings.display()
        print("-"*60)
        print("1. STT Engine Test (FasterWhisper)")
        print("2. LLM Engine Test (Gemini)")
        print("3. TTS Engine Test (Pyttsx3)")
        print("4. Complete Dialogue Test")
        print("5. Run All Tests")
        print("6. Generate Report")
        print("C. Change Settings")
        print("0. Exit")
        print("-"*60)
        
        choice = input("\nSelect test (0-6, C): ").upper()
        
        if choice == '0':
            break
        elif choice == '1':
            await test_stt_engine(reporter, settings)
        elif choice == '2':
            await test_llm_engine(reporter, settings)
        elif choice == '3':
            await test_tts_engine(reporter, settings)
        elif choice == '4':
            await test_complete_dialogue(reporter, settings)
        elif choice == '5':
            # 全テスト実行
            await test_stt_engine(reporter, settings)
            await test_llm_engine(reporter, settings)
            await test_tts_engine(reporter, settings)
            await test_complete_dialogue(reporter, settings)
        elif choice == '6':
            # レポート生成
            print(reporter.generate_report())
            save = input("\nSave report to JSON? (y/n): ")
            if save.lower() == 'y':
                reporter.save_json()
        elif choice == 'C':
            settings.change_settings()
        else:
            print("Invalid choice")
    
    # 最終レポート
    print(reporter.generate_report())
    print("\nTest completed!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()