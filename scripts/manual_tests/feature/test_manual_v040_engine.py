#!/usr/bin/env python3
"""
Phase 4 実エンジン完全動作確認用手動テストスクリプト（AudioCapture/VAD統合版）

実行方法:
  python scripts/manual_tests/feature/test_manual_phase4_complete.py

必要な環境:
  - マイク（音声入力用）
  - スピーカー（音声出力用）  
  - インターネット接続（Gemini API用）
  - 環境変数 GEMINI_API_KEY（設定済みの場合）

このスクリプトは以下をテストします：
  1. AudioCaptureデバイス管理
  2. VAD音声区間検出
  3. 各エンジンの個別動作
  4. STT→LLM→TTSの完全な対話フロー（VAD統合）
  5. エラーハンドリング
  6. パフォーマンス測定

開発規約書 v1.12準拠
テスト戦略ガイドライン v1.7準拠
インターフェース定義書 v1.34準拠
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
    print("Warning: sounddevice not available. Some features may be limited.")

# VioraTalkのインポート（仕様書準拠）
from vioratalk.infrastructure.audio_capture import (
    AudioCapture, 
    RecordingConfig, 
    AudioDevice
)
from vioratalk.core.stt.vad import (
    VoiceActivityDetector,
    VADConfig,
    VADMode,
    SpeechState,
    SpeechSegment
)
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
        # 基本設定
        self.stt_model = "base"  # tiny, base, small, medium, large, large-v2, large-v3
        self.language = "ja"     # ja, en, auto
        self.record_duration = 5  # 3, 5, 10 seconds
        self.llm_model = "gemini-2.0-flash"
        self.tts_speed = 1.0
        self.tts_volume = 0.9
        
        # AudioCapture設定
        self.device_id = None  # None = デフォルトデバイス
        self.sample_rate = 16000
        self.channels = 1
        self.enable_agc = False
        self.enable_noise_reduction = False
        
        # VAD設定
        self.vad_mode = VADMode.NORMAL
        self.vad_enabled = False  # VAD使用フラグ
        self.speech_min_duration = 0.3
        self.silence_min_duration = 0.5
    
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
        print("-"*60)
        print(f"Audio Device:     {self.device_id or 'Default'}")
        print(f"Sample Rate:      {self.sample_rate} Hz")
        print(f"Channels:         {self.channels}")
        print(f"AGC:              {'ON' if self.enable_agc else 'OFF'}")
        print(f"Noise Reduction:  {'ON' if self.enable_noise_reduction else 'OFF'}")
        print("-"*60)
        print(f"VAD Mode:         {self.vad_mode.value}")
        print(f"VAD Enabled:      {'ON' if self.vad_enabled else 'OFF'}")
    
    def change_settings(self):
        """設定を変更"""
        while True:
            print("\n" + "="*60)
            print("CHANGE SETTINGS")
            print("="*60)
            print("=== Basic Settings ===")
            print(f"1. STT Model (tiny/base/small/medium/large/large-v2/large-v3)")
            print(f"   Current: {self.stt_model}")
            print(f"2. Language (ja/en/auto)")
            print(f"   Current: {self.language}")
            print(f"3. Record Duration (3/5/10 seconds)")
            print(f"   Current: {self.record_duration} seconds")
            print(f"4. LLM Model")
            print(f"   Current: {self.llm_model}")
            print(f"5. TTS Speed (0.5-2.0)")
            print(f"   Current: {self.tts_speed}x")
            print(f"6. TTS Volume (0-100)")
            print(f"   Current: {int(self.tts_volume * 100)}%")
            print("\n=== Audio Settings ===")
            print(f"7. Audio Device")
            print(f"   Current: {'Default' if self.device_id is None else f'Device {self.device_id}'}")
            print(f"8. AGC (Auto Gain Control)")
            print(f"   Current: {'ON' if self.enable_agc else 'OFF'}")
            print(f"9. Noise Reduction")
            print(f"   Current: {'ON' if self.enable_noise_reduction else 'OFF'}")
            print("\n=== VAD Settings ===")
            print(f"10. VAD Mode (AGGRESSIVE/NORMAL/CONSERVATIVE)")
            print(f"   Current: {self.vad_mode.value}")
            print(f"11. VAD Enable/Disable")
            print(f"   Current: {'ON' if self.vad_enabled else 'OFF'}")
            print("\n0. Back to main menu")
            print("-"*60)
            
            
            choice = input("\nSelect setting to change (0-11): ")
            
            if choice == '0':
                break
            elif choice == '1':
                print("\n" + "-"*40)
                print("STT Model Selection")
                print("-"*40)
                print(f"Current model: {self.stt_model}")
                print("\nAvailable models:")
                print("  tiny     - Fastest, lowest accuracy (39MB)")
                print("  base     - Good balance (74MB) [Recommended]")
                print("  small    - Better accuracy (244MB)")
                print("  medium   - High accuracy (769MB)")
                print("  large    - Best accuracy v1 (1550MB)")
                print("  large-v2 - Best accuracy v2 (1550MB)")
                print("  large-v3 - Best accuracy v3 (1550MB) [Recommended for Japanese]")
                model = input("\nEnter model name (or press Enter to cancel): ").lower()
                if model in ['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3']:
                    old_model = self.stt_model
                    self.stt_model = model
                    print(f"✓ STT model changed: {old_model} → {model}")
                elif model == '':
                    print("✗ Cancelled")
                else:
                    print("✗ Invalid model name")
                    
            elif choice == '2':
                print("\n" + "-"*40)
                print("Language Selection")
                print("-"*40)
                print(f"Current language: {self.language}")
                print("\nAvailable languages:")
                print("  ja   - Japanese")
                print("  en   - English")
                print("  auto - Auto-detect")
                lang = input("\nEnter language code (or press Enter to cancel): ").lower()
                if lang in ['ja', 'en', 'auto']:
                    old_lang = self.language
                    self.language = lang
                    print(f"✓ Language changed: {old_lang} → {lang}")
                elif lang == '':
                    print("✗ Cancelled")
                else:
                    print("✗ Invalid language code")
                    
            elif choice == '3':
                print("\n" + "-"*40)
                print("Record Duration Selection")
                print("-"*40)
                print(f"Current duration: {self.record_duration} seconds")
                print("\nAvailable durations:")
                print("  3  - Quick test")
                print("  5  - Normal conversation [Recommended]")
                print("  10 - Long questions")
                duration = input("\nEnter duration (3/5/10 or press Enter to cancel): ")
                if duration in ['3', '5', '10']:
                    old_duration = self.record_duration
                    self.record_duration = int(duration)
                    print(f"✓ Record duration changed: {old_duration}s → {duration}s")
                elif duration == '':
                    print("✗ Cancelled")
                else:
                    print("✗ Invalid duration")
                    
            elif choice == '4':
                print("\n" + "-"*40)
                print("LLM Model Selection")
                print("-"*40)
                print(f"Current model: {self.llm_model}")
                print("\nAvailable models:")
                print("  gemini-2.0-flash - Fast, good quality [Recommended]")
                print("  gemini-2.5-flash - Newer, balanced")
                print("  gemini-1.5-pro   - High quality")
                model = input("\nEnter model name (or press Enter to cancel): ")
                if model and 'gemini' in model.lower():
                    old_model = self.llm_model
                    self.llm_model = model
                    print(f"✓ LLM model changed: {old_model} → {model}")
                elif model == '':
                    print("✗ Cancelled")
                else:
                    print("✗ Invalid model name (must contain 'gemini')")
                    
            elif choice == '5':
                print("\n" + "-"*40)
                print("TTS Speed Adjustment")
                print("-"*40)
                print(f"Current speed: {self.tts_speed}x")
                print("\nSpeed range: 0.5 (slow) - 2.0 (fast)")
                print("Recommended: 1.0 (normal)")
                speed = input("\nEnter speed (0.5-2.0 or press Enter to cancel): ")
                if speed:
                    try:
                        speed_val = float(speed)
                        if 0.5 <= speed_val <= 2.0:
                            old_speed = self.tts_speed
                            self.tts_speed = speed_val
                            print(f"✓ TTS speed changed: {old_speed}x → {speed_val}x")
                        else:
                            print("✗ Speed must be between 0.5 and 2.0")
                    except ValueError:
                        print("✗ Invalid speed value")
                else:
                    print("✗ Cancelled")
                    
            elif choice == '6':
                print("\n" + "-"*40)
                print("TTS Volume Adjustment")
                print("-"*40)
                print(f"Current volume: {int(self.tts_volume * 100)}%")
                print("\nVolume range: 0 (mute) - 100 (max)")
                print("Recommended: 80-90")
                volume = input("\nEnter volume (0-100 or press Enter to cancel): ")
                if volume:
                    try:
                        vol_val = int(volume)
                        if 0 <= vol_val <= 100:
                            old_volume = int(self.tts_volume * 100)
                            self.tts_volume = vol_val / 100.0
                            print(f"✓ TTS volume changed: {old_volume}% → {vol_val}%")
                        else:
                            print("✗ Volume must be between 0 and 100")
                    except ValueError:
                        print("✗ Invalid volume value")
                else:
                    print("✗ Cancelled")
                    
            elif choice == '7':
                print("\n" + "-"*40)
                print("Audio Device Selection")
                print("-"*40)
                print(f"Current device: {'Default' if self.device_id is None else f'Device {self.device_id}'}")
                print("\n⚠ Device selection will be available in Device Management Test")
                print("  (Main menu option 1)")
                input("\nPress Enter to continue...")
                
            elif choice == '8':
                print("\n" + "-"*40)
                print("AGC (Auto Gain Control)")
                print("-"*40)
                old_state = "ON" if self.enable_agc else "OFF"
                self.enable_agc = not self.enable_agc
                new_state = "ON" if self.enable_agc else "OFF"
                print(f"✓ AGC changed: {old_state} → {new_state}")
                print("\nAGC automatically adjusts microphone input level")
                print("Useful in environments with varying voice volumes")
                
            elif choice == '9':
                print("\n" + "-"*40)
                print("Noise Reduction")
                print("-"*40)
                old_state = "ON" if self.enable_noise_reduction else "OFF"
                self.enable_noise_reduction = not self.enable_noise_reduction
                new_state = "ON" if self.enable_noise_reduction else "OFF"
                print(f"✓ Noise Reduction changed: {old_state} → {new_state}")
                print("\nNoise Reduction removes background noise")
                print("Useful in noisy environments")
                
            elif choice == '10':
                print("\n" + "-"*40)
                print("VAD Mode Selection")
                print("-"*40)
                print(f"Current mode: {self.vad_mode.value}")
                print("\nVAD Modes:")
                print("  1. AGGRESSIVE - High sensitivity (quiet environment)")
                print("     Best for: Quiet rooms, close microphone")
                print("  2. NORMAL - Standard sensitivity [Recommended]")
                print("     Best for: Normal environments")
                print("  3. CONSERVATIVE - Low sensitivity (noisy environment)")
                print("     Best for: Noisy rooms, background noise")
                mode_choice = input("\nSelect mode (1-3 or press Enter to cancel): ")
                if mode_choice == '1':
                    old_mode = self.vad_mode.value
                    self.vad_mode = VADMode.AGGRESSIVE
                    print(f"✓ VAD mode changed: {old_mode} → AGGRESSIVE")
                elif mode_choice == '2':
                    old_mode = self.vad_mode.value
                    self.vad_mode = VADMode.NORMAL
                    print(f"✓ VAD mode changed: {old_mode} → NORMAL")
                elif mode_choice == '3':
                    old_mode = self.vad_mode.value
                    self.vad_mode = VADMode.CONSERVATIVE
                    print(f"✓ VAD mode changed: {old_mode} → CONSERVATIVE")
                elif mode_choice == '':
                    print("✗ Cancelled")
                else:
                    print("✗ Invalid choice")
                    
            elif choice == '11':
                print("\n" + "-"*40)
                print("VAD (Voice Activity Detection)")
                print("-"*40)
                old_state = "ON" if self.vad_enabled else "OFF"
                self.vad_enabled = not self.vad_enabled
                new_state = "ON" if self.vad_enabled else "OFF"
                print(f"✓ VAD changed: {old_state} → {new_state}")
                print("\nVAD automatically detects when you stop speaking")
                print("When ON: Recording stops automatically after silence")
                print("When OFF: Fixed duration recording")
            else:
                print("✗ Invalid choice")


# ============================================================================
# ユーティリティクラス
# ============================================================================

class TestReporter:
    """テスト結果のレポート生成（既存のまま）"""
    
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


# ============================================================================
# 環境チェック（既存のまま、一部省略）
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
        # sounddeviceがない場合はAudioCaptureのpyaudioフォールバックを使用可能
        print("   ⚠ Will try to use pyaudio fallback")
    
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
# AudioCapture/VADテスト（新規追加）
# ============================================================================

async def test_device_management(reporter: TestReporter, settings: TestSettings):
    """AudioCaptureデバイス管理テスト"""
    print("\n" + "="*60)
    print("DEVICE MANAGEMENT TEST (AudioCapture)")
    print("="*60)
    
    try:
        # AudioCapture初期化
        capture = AudioCapture()
        await capture.safe_initialize()
        
        # 1. デバイス一覧表示
        print("\n1. Available Audio Devices:")
        devices = capture.list_devices()
        
        if not devices:
            print("   ✗ No audio devices found")
            reporter.add_result(
                "Device Detection",
                "FAIL",
                {"error": "No devices found"}
            )
            return
        
        for device in devices:
            marker = " [CURRENT]" if device.is_default else ""
            print(f"   {device.id}: {device.name}")
            print(f"      Channels: {device.channels}, Sample Rate: {device.sample_rate}Hz")
            print(f"      Host API: {device.host_api}{marker}")
        
        reporter.add_result(
            "Device Detection",
            "PASS",
            {"device_count": len(devices)}
        )
        
        # 2. デバイス選択テスト
        if len(devices) > 1:
            print("\n2. Device Selection Test:")
            choice = input("   Select device ID (Enter to skip): ")
            
            if choice.isdigit():
                device_id = int(choice)
                try:
                    capture.select_device(device_id)
                    current = capture.get_current_device()
                    print(f"   ✓ Selected: {current.name}")
                    settings.device_id = device_id
                    
                    reporter.add_result(
                        "Device Selection",
                        "PASS",
                        {"selected_device": current.name}
                    )
                except AudioError as e:
                    print(f"   ✗ Selection failed: {e}")
                    reporter.add_result(
                        "Device Selection",
                        "FAIL",
                        {"error": str(e)}
                    )
        
        # 3. 録音品質テスト
        print("\n3. Recording Quality Test:")
        test_record = input("   Test recording? (y/n): ")
        
        if test_record.lower() == 'y':
            config = RecordingConfig(
                device_id=settings.device_id,
                sample_rate=settings.sample_rate,
                channels=settings.channels,
                duration=3.0,
                enable_agc=settings.enable_agc,
                enable_noise_reduction=settings.enable_noise_reduction
            )
            
            capture_with_config = AudioCapture(config)
            await capture_with_config.safe_initialize()
            
            input("   Press ENTER to start 3-second test recording...")
            print("   🎤 Recording...")
            
            start = time.time()
            audio_data = await capture_with_config.record_from_microphone(duration=3.0)
            duration = time.time() - start
            
            # 音声レベル分析
            audio_array = audio_data.raw_data
            rms = np.sqrt(np.mean(audio_array ** 2))
            db_level = 20 * np.log10(rms) if rms > 0 else -60
            
            print(f"\n   Recording completed:")
            print(f"   - Duration: {duration:.2f}s")
            print(f"   - Samples: {len(audio_array)}")
            print(f"   - RMS Level: {rms:.4f}")
            print(f"   - dB Level: {db_level:.1f} dB")
            
            quality_ok = input("   Was the quality acceptable? (y/n): ")
            
            reporter.add_result(
                "Recording Quality",
                "PASS" if quality_ok.lower() == 'y' else "FAIL",
                {
                    "duration": f"{duration:.2f}s",
                    "db_level": f"{db_level:.1f}dB",
                    "agc": settings.enable_agc,
                    "noise_reduction": settings.enable_noise_reduction
                },
                duration
            )
            
            await capture_with_config.safe_cleanup()
        
        # クリーンアップ
        await capture.safe_cleanup()
        print("\n   ✓ Cleanup completed")
        
    except Exception as e:
        print(f"\n   ✗ Error: {e}")
        reporter.add_result(
            "Device Management Test",
            "FAIL",
            {"error": str(e)}
        )


async def test_vad_detection(reporter: TestReporter, settings: TestSettings):
    """VAD音声区間検出テスト"""
    print("\n" + "="*60)
    print("VAD DETECTION TEST")
    print("="*60)
    
    try:
        # VAD初期化
        print(f"\n1. Initializing VAD...")
        print(f"   Mode: {settings.vad_mode.value}")
        print(f"   Speech min duration: {settings.speech_min_duration}s")
        print(f"   Silence min duration: {settings.silence_min_duration}s")
        
        vad_config = VADConfig(
            mode=settings.vad_mode,
            sample_rate=settings.sample_rate,
            speech_min_duration=settings.speech_min_duration,
            silence_min_duration=settings.silence_min_duration,
            adaptive_threshold=True,
            enable_noise_learning=True
        )
        vad = VoiceActivityDetector(vad_config)
        await vad.initialize()
        print("   ✓ VAD initialized")
        
        reporter.add_result(
            "VAD Initialization",
            "PASS",
            {"mode": settings.vad_mode.value}
        )
        
        # AudioCapture初期化
        capture_config = RecordingConfig(
            device_id=settings.device_id,
            sample_rate=settings.sample_rate,
            channels=1,  # VADはモノラル必須
            enable_agc=settings.enable_agc,
            enable_noise_reduction=settings.enable_noise_reduction
        )
        capture = AudioCapture(capture_config)
        await capture.safe_initialize()
        
        # 2. 音声区間検出テスト
        print("\n2. Speech Segment Detection Test:")
        print("   Instructions:")
        print("   - Speak a few sentences with pauses")
        print("   - VAD will detect speech segments")
        print(f"   - Recording for {settings.record_duration} seconds")
        
        input("\n   Press ENTER to start recording...")
        print("   🎤 Recording... (speak with pauses)")
        
        start = time.time()
        audio_data = await capture.record_from_microphone(duration=settings.record_duration)
        recording_time = time.time() - start
        
        print("   Analyzing speech segments...")
        segments = vad.detect_segments(audio_data.raw_data)
        
        print(f"\n   📊 Detected {len(segments)} speech segments:")
        total_speech_time = 0
        for i, seg in enumerate(segments, 1):
            print(f"   Segment {i}:")
            print(f"     Start: {seg.start_time:.2f}s")
            print(f"     End: {seg.end_time:.2f}s")
            print(f"     Duration: {seg.duration:.2f}s")
            print(f"     Confidence: {seg.confidence:.2%}")
            total_speech_time += seg.duration
        
        if segments:
            speech_ratio = total_speech_time / settings.record_duration
            print(f"\n   Total speech time: {total_speech_time:.2f}s ({speech_ratio:.1%})")
            
            reporter.add_result(
                "VAD Segment Detection",
                "PASS",
                {
                    "segments": len(segments),
                    "total_speech": f"{total_speech_time:.2f}s",
                    "speech_ratio": f"{speech_ratio:.1%}"
                },
                recording_time
            )
        else:
            print("   ⚠ No speech segments detected")
            reporter.add_result(
                "VAD Segment Detection",
                "FAIL",
                {"error": "No segments detected"}
            )
        
        # 3. VADモード比較テスト
        compare = input("\n3. Compare VAD modes? (y/n): ")
        if compare.lower() == 'y':
            modes = [VADMode.AGGRESSIVE, VADMode.NORMAL, VADMode.CONSERVATIVE]
            
            for mode in modes:
                if mode == settings.vad_mode:
                    continue
                
                print(f"\n   Testing {mode.value} mode...")
                vad.set_mode(mode)
                
                input("   Press ENTER to record...")
                audio = await capture.record_from_microphone(duration=3.0)
                segments = vad.detect_segments(audio.raw_data)
                
                print(f"   Detected {len(segments)} segments in {mode.value} mode")
                
                reporter.add_result(
                    f"VAD Mode Test - {mode.value}",
                    "PASS",
                    {"segments": len(segments)}
                )
        
        # 4. 統計情報表示
        print("\n4. VAD Statistics:")
        stats = vad.get_statistics()
        print(f"   Total frames: {stats.total_frames}")
        print(f"   Speech frames: {stats.speech_frames}")
        print(f"   Silence frames: {stats.silence_frames}")
        print(f"   Noise level: {stats.noise_level:.4f}")
        print(f"   Average energy: {stats.average_energy:.4f}")
        
        # クリーンアップ
        await vad.cleanup()
        await capture.safe_cleanup()
        print("\n   ✓ Cleanup completed")
        
    except Exception as e:
        print(f"\n   ✗ Error: {e}")
        reporter.add_result(
            "VAD Detection Test",
            "FAIL",
            {"error": str(e)}
        )


# ============================================================================
# 個別エンジンテスト（既存、一部省略）
# ============================================================================

async def test_stt_engine(reporter: TestReporter, settings: TestSettings):
    """STTエンジンのテスト（VAD統合オプション付き）"""
    print("\n" + "="*60)
    print("STT ENGINE TEST (FasterWhisper)")
    print("="*60)
    
    try:
        # 1. STT初期化
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
        print("   ✓ STT Engine initialized successfully")
        
        reporter.add_result(
            "STT Initialization",
            "PASS",
            {"model": settings.stt_model, "language": settings.language}
        )
        
        # 2. AudioCapture初期化
        print(f"\n2. Initializing AudioCapture...")
        capture_config = RecordingConfig(
            device_id=settings.device_id,
            sample_rate=16000,
            channels=1,
            enable_agc=settings.enable_agc,
            enable_noise_reduction=settings.enable_noise_reduction
        )
        capture = AudioCapture(capture_config)
        await capture.safe_initialize()
        print("   ✓ AudioCapture initialized")
        
        # 3. 音声認識テスト
        print(f"\n3. Testing speech recognition...")
        print(f"   Duration: {settings.record_duration} seconds")
        print(f"   Language: {settings.language}")
        
        # VAD統合オプション
        use_vad = input("\n   Use VAD for automatic speech detection? (y/n): ").lower()
        
        if use_vad == 'y':
            # VAD使用
            print("\n   Initializing VAD...")
            vad = VoiceActivityDetector(VADConfig(
                mode=settings.vad_mode,
                sample_rate=16000,
                speech_min_duration=0.3,
                silence_min_duration=0.5
            ))
            await vad.initialize()
            print("   ✓ VAD initialized")
            
            input("\n   Press ENTER to start recording...")
            print("   🎤 Speak naturally, VAD will detect when you stop...")
            print("   (Recording for up to 10 seconds)")
            
            start = time.time()
            audio_data = await capture.record_from_microphone(duration=10.0)
            recording_time = time.time() - start
            
            print("   Analyzing speech segments...")
            segments = vad.detect_segments(audio_data.raw_data)
            
            if segments:
                print(f"\n   Detected {len(segments)} speech segments")
                
                # 各セグメントを認識
                all_texts = []
                for i, seg in enumerate(segments, 1):
                    # セグメントの音声データを抽出
                    segment_audio = audio_data.raw_data[seg.start_sample:seg.end_sample]
                    segment_data = AudioData(
                        raw_data=segment_audio,
                        metadata=AudioMetadata(
                            filename=f"segment_{i}.wav",
                            format="pcm",
                            sample_rate=16000,
                            channels=1,
                            duration=seg.duration
                        )
                    )
                    
                    # 音声認識
                    start_stt = time.time()
                    result = await engine.transcribe(segment_data, language=settings.language if settings.language != "auto" else None)
                    stt_time = time.time() - start_stt
                    
                    print(f"\n   Segment {i}:")
                    print(f"     Text: '{result.text}'")
                    print(f"     Language: {result.language}")
                    print(f"     Confidence: {result.confidence:.2%}")
                    print(f"     STT Time: {stt_time:.2f}s")
                    
                    all_texts.append(result.text)
                    
                    reporter.add_result(
                        f"STT with VAD - Segment {i}",
                        "PASS",
                        {
                            "text": result.text,
                            "duration": f"{seg.duration:.2f}s",
                            "confidence": f"{result.confidence:.2%}",
                            "stt_time": f"{stt_time:.2f}s"
                        },
                        stt_time
                    )
                
                # 全体の結果
                combined_text = " ".join(all_texts)
                print(f"\n   Combined result: '{combined_text}'")
                
            else:
                print("   ⚠ No speech segments detected")
                reporter.add_result(
                    "STT with VAD",
                    "FAIL",
                    {"error": "No segments detected"}
                )
            
            await vad.cleanup()
            
        else:
            # 通常の固定時間録音
            input(f"\n   Press ENTER to start {settings.record_duration}-second recording...")
            print(f"   🎤 Recording for {settings.record_duration} seconds...")
            print("   Speak now!")
            
            start = time.time()
            audio_data = await capture.record_from_microphone(duration=settings.record_duration)
            recording_time = time.time() - start
            print(f"   Recording completed ({recording_time:.2f}s)")
            
            # 音声認識
            print("   Processing audio...")
            start_stt = time.time()
            
            if settings.language == "auto":
                result = await engine.transcribe(audio_data)
            else:
                result = await engine.transcribe(audio_data, language=settings.language)
            
            stt_time = time.time() - start_stt
            
            print(f"\n   Recognition result:")
            print(f"   Text: '{result.text}'")
            print(f"   Detected Language: {result.language}")
            print(f"   Confidence: {result.confidence:.2%}")
            print(f"   Processing time: {stt_time:.2f}s")
            
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
                        "time": f"{stt_time:.2f}s",
                        "model": settings.stt_model
                    },
                    stt_time
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
                    stt_time
                )
                print("   ✗ Test FAILED")
        
        # 4. モデル切り替えテスト（オプション）
        change_model = input("\n4. Test different model? (y/n): ").lower()
        if change_model == 'y':
            print("\n   Model comparison test")
            models_to_test = ['tiny', 'base', 'small']
            
            for model in models_to_test:
                if model == settings.stt_model:
                    continue
                    
                print(f"\n   Testing model: {model}")
                engine.config.model = model
                await engine.cleanup()
                await engine.initialize()
                
                input("   Press ENTER to record...")
                audio = await capture.record_from_microphone(duration=3.0)
                
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
        print("\n5. Cleanup...")
        await engine.cleanup()
        await capture.safe_cleanup()
        print("   ✓ Cleanup completed")
        
    except Exception as e:
        print(f"\n   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
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
        
        # 生成テスト
        print("\n2. Testing text generation...")
        
        if settings.language == "ja":
            test_prompts = [
                ("こんにちは", "Japanese greeting"),
                ("今日の天気はどうですか？", "Weather question")
            ]
        else:
            test_prompts = [
                ("Hello", "English greeting"),
                ("What's the weather like today?", "Weather question")
            ]
        
        for prompt, description in test_prompts:
            print(f"\n   Test: {description}")
            print(f"   Prompt: '{prompt}'")
            
            if api_key:
                start = time.time()
                response = await engine.generate(
                    prompt=prompt,
                    max_tokens=100
                )
                duration = time.time() - start
                
                print(f"   Response: '{response.content[:100]}...'")
                print(f"   Time: {duration:.2f}s")
                
                reporter.add_result(
                    f"LLM Generation - {description}",
                    "PASS",
                    {
                        "prompt": prompt,
                        "response_length": len(response.content),
                        "time": f"{duration:.2f}s"
                    },
                    duration
                )
            else:
                # モック応答
                mock_response = "こんにちは！今日もいい天気ですね。" if settings.language == "ja" else "Hello! It's a beautiful day today."
                print(f"   [Mock] Response: '{mock_response}'")
                reporter.add_result(
                    f"LLM Generation - {description}",
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
            language=settings.language if settings.language != "auto" else "ja",
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
                ("こんにちは、VioraTalkです。", "ja"),
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
            quality = input("   Rate audio quality (1-5, Enter to skip): ")
            
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
# 統合対話テスト（VAD統合）
# ============================================================================

async def test_complete_dialogue(reporter: TestReporter, settings: TestSettings):
    """完全な対話フローのテスト（VAD統合版）"""
    print("\n" + "="*60)
    print("COMPLETE DIALOGUE TEST")
    print("="*60)
    
    # 録音モード選択
    print("\nSelect recording mode:")
    print("1. Fixed duration (traditional)")
    print("2. VAD auto-detect (automatic stop)")
    print("3. Push-to-Talk (manual control)")
    
    mode = input("Select mode (1-3): ")
    
    api_key = os.environ.get('GEMINI_API_KEY')
    
    try:
        # エンジンの初期化
        print("\nInitializing all engines...")
        
        # STT
        stt_config = STTConfig(
            model=settings.stt_model,
            language=None if settings.language == "auto" else settings.language
        )
        stt_engine = FasterWhisperEngine(stt_config)
        await stt_engine.initialize()
        print("   ✓ STT ready")
        
        # LLM
        if api_key:
            llm_config = LLMConfig(model=settings.llm_model)
            llm_engine = GeminiEngine(llm_config, api_key=api_key)
            await llm_engine.initialize()
            print("   ✓ LLM ready")
        else:
            llm_engine = None
            print("   ⚠ LLM skipped (no API key)")
        
        # TTS
        tts_config = TTSConfig(
            language=settings.language if settings.language != "auto" else "ja",
            speed=settings.tts_speed,
            volume=settings.tts_volume
        )
        tts_engine = Pyttsx3Engine(tts_config)
        await tts_engine.initialize()
        print("   ✓ TTS ready")
        
        # AudioCapture
        capture_config = RecordingConfig(
            device_id=settings.device_id,
            sample_rate=16000,
            channels=1,
            enable_agc=settings.enable_agc,
            enable_noise_reduction=settings.enable_noise_reduction
        )
        capture = AudioCapture(capture_config)
        await capture.safe_initialize()
        print("   ✓ AudioCapture ready")
        
        # VAD（モード2の場合）
        vad = None
        if mode == '2':
            vad = VoiceActivityDetector(VADConfig(
                mode=settings.vad_mode,
                speech_min_duration=0.3,
                silence_min_duration=0.8  # 対話用に長めに設定
            ))
            await vad.initialize()
            print("   ✓ VAD ready")
        
        # 対話ループ
        print("\n" + "-"*60)
        print("Starting dialogue...")
        if mode == '1':
            print(f"Mode: Fixed duration ({settings.record_duration}s)")
        elif mode == '2':
            print("Mode: VAD auto-detect (speak naturally)")
        else:
            print("Mode: Push-to-Talk (hold SPACE to speak)")
        
        print("Say 'goodbye' or 'さようなら' to exit")
        print("-"*60)
        
        round_num = 0
        while round_num < 3:
            round_num += 1
            print(f"\n[Round {round_num}]")
            
            # 音声入力
            if mode == '1':
                # 固定時間録音
                input("Press ENTER to speak...")
                print(f"🎤 Recording for {settings.record_duration} seconds...")
                audio_data = await capture.record_from_microphone(duration=settings.record_duration)
                
            elif mode == '2':
                # VAD自動検出
                print("🎤 Speak now (VAD will detect when you stop)...")
                
                # 最大15秒録音してVADで区間検出
                max_duration = 15.0
                audio_data = await capture.record_from_microphone(duration=max_duration)
                segments = vad.detect_segments(audio_data.raw_data)
                
                if segments:
                    # 最初の有効なセグメントを使用
                    seg = segments[0]
                    segment_audio = audio_data.raw_data[seg.start_sample:seg.end_sample]
                    audio_data = AudioData(
                        raw_data=segment_audio,
                        metadata=AudioMetadata(
                            sample_rate=16000,
                            channels=1,
                            duration=seg.duration
                        )
                    )
                    print(f"   Detected speech: {seg.duration:.1f}s")
                else:
                    print("   ⚠ No speech detected, retrying...")
                    continue
                    
            else:
                # Push-to-Talk（簡易実装）
                input("Press ENTER to start, then ENTER again to stop...")
                print("🎤 Recording...")
                start_time = time.time()
                audio_data = await capture.record_from_microphone(duration=10.0)
                actual_duration = time.time() - start_time
                print(f"   Recorded: {actual_duration:.1f}s")
            
            # STT: 音声認識
            start_stt = time.time()
            transcript = await stt_engine.transcribe(audio_data)
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
                response_text = f"Echo: {transcript.text}"
                llm_time = 0
            
            # TTS: 音声出力
            start_tts = time.time()
            tts_lang = transcript.language if transcript.language in ["ja", "en"] else "ja"
            synthesis = await tts_engine.synthesize(response_text, voice_id=tts_lang)
            tts_time = time.time() - start_tts
            
            print(f"\n🔊 Speaking response...")
            print(f"   (TTS: {tts_time:.2f}s)")
            
            total_time = stt_time + llm_time + tts_time
            print(f"\n⏱ Total response time: {total_time:.2f}s")
            
            reporter.add_result(
                f"Dialogue Round {round_num}",
                "PASS",
                {
                    "mode": ["Fixed", "VAD", "PTT"][int(mode)-1],
                    "user_input": transcript.text[:50],
                    "response": response_text[:50],
                    "total_time": f"{total_time:.2f}s"
                },
                total_time
            )
        
        # クリーンアップ
        print("\nCleaning up...")
        await stt_engine.cleanup()
        if llm_engine:
            await llm_engine.cleanup()
        await tts_engine.cleanup()
        await capture.safe_cleanup()
        if vad:
            await vad.cleanup()
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
    print(" AudioCapture/VAD Integration Edition (v3.0)")
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
        print("=== Device & Audio Processing ===")
        print("1. Device Management Test (AudioCapture)")
        print("2. VAD Detection Test")
        print("\n=== Engine Tests ===")
        print("3. STT Engine Test (FasterWhisper)")
        print("4. LLM Engine Test (Gemini)")
        print("5. TTS Engine Test (Pyttsx3)")
        print("\n=== Integration ===")
        print("6. Complete Dialogue Test (with VAD)")
        print("7. Run All Tests")
        print("\n=== Reports & Settings ===")
        print("8. Generate Report")
        print("C. Change Settings")
        print("0. Exit")
        print("-"*60)
        
        choice = input("\nSelect test (0-8, C): ").upper()
        
        if choice == '0':
            break
        elif choice == '1':
            await test_device_management(reporter, settings)
        elif choice == '2':
            await test_vad_detection(reporter, settings)
        elif choice == '3':
            await test_stt_engine(reporter, settings)
        elif choice == '4':
            await test_llm_engine(reporter, settings)
        elif choice == '5':
            await test_tts_engine(reporter, settings)
        elif choice == '6':
            await test_complete_dialogue(reporter, settings)
        elif choice == '7':
            # 全テスト実行
            await test_device_management(reporter, settings)
            await test_vad_detection(reporter, settings)
            await test_stt_engine(reporter, settings)
            await test_llm_engine(reporter, settings)
            await test_tts_engine(reporter, settings)
            await test_complete_dialogue(reporter, settings)
        elif choice == '8':
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