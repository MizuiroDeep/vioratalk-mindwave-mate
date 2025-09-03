#!/usr/bin/env python3
"""
VioraTalk v0.5.0 DialogueManager統合版 手動テストスクリプト

実行方法:
  python scripts/manual_tests/feature/test_manual_v050_dialogue.py

必要な環境:
  - マイク（音声入力用）
  - スピーカー（音声出力用）  
  - インターネット接続（Gemini API用）
  - 環境変数 GEMINI_API_KEY（設定済みの場合）

このスクリプトは以下をテストします：
  1. DialogueManager統合動作
  2. AudioCapture/VAD統合
  3. STT→LLM→TTSの完全フロー（DialogueManager経由）
  4. エラーハンドリング
  5. パフォーマンス測定

開発規約書 v1.12準拠 
DialogueManager統合ガイド v1.2準拠
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
from vioratalk.core.dialogue_manager import DialogueManager
from vioratalk.core.dialogue_config import DialogueConfig
from vioratalk.core.dialogue_state import DialogueTurn

# Managers
from vioratalk.core.llm.llm_manager import LLMManager
from vioratalk.core.tts.tts_manager import TTSManager

# Engines
from vioratalk.core.stt import FasterWhisperEngine, STTConfig
from vioratalk.core.llm import GeminiEngine, LLMConfig
from vioratalk.core.tts import Pyttsx3Engine, TTSConfig

# Audio関連
from vioratalk.infrastructure.audio_capture import (
    AudioCapture, 
    RecordingConfig, 
    AudioDevice
)
from vioratalk.core.stt.vad import (
    VoiceActivityDetector,
    VADConfig,
    VADMode
)

# エラーハンドリング
from vioratalk.core.exceptions import (
    STTError, LLMError, TTSError, AudioError, 
    APIError, AuthenticationError
)
from vioratalk.utils.logger_manager import LoggerManager


# ============================================================================
# テスト設定クラス（既存と同じインターフェース）
# ============================================================================

class TestSettings:
    """テスト設定の管理（ユーザー視点は変更なし）"""
    
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
        """現在の設定を表示（表示は既存と同じ）"""
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
        """設定を変更（UIは既存と同じ）"""
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
                self._change_stt_model()
            elif choice == '2':
                self._change_language()
            elif choice == '3':
                self._change_record_duration()
            elif choice == '4':
                self._change_llm_model()
            elif choice == '5':
                self._change_tts_speed()
            elif choice == '6':
                self._change_tts_volume()
            elif choice == '7':
                print("\n⚠ Device selection will be available in Device Management Test")
                input("\nPress Enter to continue...")
            elif choice == '8':
                self.enable_agc = not self.enable_agc
                print(f"✔ AGC changed: {'ON' if self.enable_agc else 'OFF'}")
            elif choice == '9':
                self.enable_noise_reduction = not self.enable_noise_reduction
                print(f"✔ Noise Reduction changed: {'ON' if self.enable_noise_reduction else 'OFF'}")
            elif choice == '10':
                self._change_vad_mode()
            elif choice == '11':
                self.vad_enabled = not self.vad_enabled
                print(f"✔ VAD changed: {'ON' if self.vad_enabled else 'OFF'}")
            else:
                print("✗ Invalid choice")
    
    def _change_stt_model(self):
        """STTモデル変更（内部メソッド）"""
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
            print(f"✔ STT model changed: {old_model} → {model}")
        elif model == '':
            print("✗ Cancelled")
        else:
            print("✗ Invalid model name")
    
    def _change_language(self):
        """言語変更（内部メソッド）"""
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
            print(f"✔ Language changed: {old_lang} → {lang}")
        elif lang == '':
            print("✗ Cancelled")
        else:
            print("✗ Invalid language code")
    
    def _change_record_duration(self):
        """録音時間変更（内部メソッド）"""
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
            print(f"✔ Record duration changed: {old_duration}s → {duration}s")
        elif duration == '':
            print("✗ Cancelled")
        else:
            print("✗ Invalid duration")
    
    def _change_llm_model(self):
        """LLMモデル変更（内部メソッド）"""
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
            print(f"✔ LLM model changed: {old_model} → {model}")
        elif model == '':
            print("✗ Cancelled")
        else:
            print("✗ Invalid model name (must contain 'gemini')")
    
    def _change_tts_speed(self):
        """TTS速度変更（内部メソッド）"""
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
                    print(f"✔ TTS speed changed: {old_speed}x → {speed_val}x")
                else:
                    print("✗ Speed must be between 0.5 and 2.0")
            except ValueError:
                print("✗ Invalid speed value")
        else:
            print("✗ Cancelled")
    
    def _change_tts_volume(self):
        """TTS音量変更（内部メソッド）"""
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
                    print(f"✔ TTS volume changed: {old_volume}% → {vol_val}%")
                else:
                    print("✗ Volume must be between 0 and 100")
            except ValueError:
                print("✗ Invalid volume value")
        else:
            print("✗ Cancelled")
    
    def _change_vad_mode(self):
        """VADモード変更（内部メソッド）"""
        print("\n" + "-"*40)
        print("VAD Mode Selection")
        print("-"*40)
        print(f"Current mode: {self.vad_mode.value}")
        print("\nVAD Modes:")
        print("  1. AGGRESSIVE - High sensitivity (quiet environment)")
        print("  2. NORMAL - Standard sensitivity [Recommended]")
        print("  3. CONSERVATIVE - Low sensitivity (noisy environment)")
        mode_choice = input("\nSelect mode (1-3 or press Enter to cancel): ")
        if mode_choice == '1':
            old_mode = self.vad_mode.value
            self.vad_mode = VADMode.AGGRESSIVE
            print(f"✔ VAD mode changed: {old_mode} → AGGRESSIVE")
        elif mode_choice == '2':
            old_mode = self.vad_mode.value
            self.vad_mode = VADMode.NORMAL
            print(f"✔ VAD mode changed: {old_mode} → NORMAL")
        elif mode_choice == '3':
            old_mode = self.vad_mode.value
            self.vad_mode = VADMode.CONSERVATIVE
            print(f"✔ VAD mode changed: {old_mode} → CONSERVATIVE")
        elif mode_choice == '':
            print("✗ Cancelled")
        else:
            print("✗ Invalid choice")


# ============================================================================
# テストレポータークラス（既存と同じ）
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
                "PASS": "✔",
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
# DialogueManager初期化（v0.5.0の核心部分）
# ============================================================================

class DialogueManagerSetup:
    """DialogueManagerのセットアップ管理"""
    
    def __init__(self, settings: TestSettings):
        self.settings = settings
        self.dialogue_manager = None
        self.llm_manager = None
        self.tts_manager = None
        self.stt_engine = None
        self.audio_capture = None
        self.vad = None
        
    async def initialize(self) -> DialogueManager:
        """DialogueManagerを初期化（全エンジン統合）"""
        print("\n" + "="*60)
        print("INITIALIZING DIALOGUE MANAGER")
        print("="*60)
        
        try:
            # 1. DialogueConfig作成
            print("\n1. Creating DialogueConfig...")
            config = DialogueConfig(
                max_turns=100,
                turn_timeout=30.0,
                temperature=0.7,
                language=self.settings.language if self.settings.language != "auto" else "ja",
                use_mock_engines=False,  # 実エンジンを使用
                tts_enabled=True,
                tts_enabled_for_text=True,
                recording_duration=self.settings.record_duration,
                sample_rate=self.settings.sample_rate
            )
            print("   ✔ DialogueConfig created")
            
            # 2. STTエンジン初期化
            print("\n2. Initializing STT Engine...")
            stt_config = STTConfig(
                engine="faster-whisper",
                model=self.settings.stt_model,
                language=None if self.settings.language == "auto" else self.settings.language
            )
            self.stt_engine = FasterWhisperEngine(stt_config)
            await self.stt_engine.initialize()
            print(f"   ✔ STT Engine initialized (model: {self.settings.stt_model})")
            
            # 3. LLMManager初期化
            print("\n3. Initializing LLM Manager...")
            self.llm_manager = LLMManager()
            
            api_key = os.environ.get('GEMINI_API_KEY')
            if api_key:
                llm_config = LLMConfig(
                    engine="gemini",
                    model=self.settings.llm_model,
                    temperature=0.7
                )
                gemini_engine = GeminiEngine(llm_config, api_key=api_key)
                self.llm_manager.register_engine("gemini", gemini_engine, priority=1)
                print(f"   ✔ Gemini Engine registered (model: {self.settings.llm_model})")
            else:
                print("   ⚠ No GEMINI_API_KEY found, LLM will use mock responses")
            
            await self.llm_manager.initialize()
            print("   ✔ LLM Manager initialized")
            
            # 4. TTSManager初期化
            print("\n4. Initializing TTS Manager...")
            self.tts_manager = TTSManager()
            
            tts_config = TTSConfig(
                engine="pyttsx3",
                language=self.settings.language if self.settings.language != "auto" else "ja",
                speed=self.settings.tts_speed,
                volume=self.settings.tts_volume
            )
            pyttsx3_engine = Pyttsx3Engine(tts_config)
            self.tts_manager.register_engine("pyttsx3", pyttsx3_engine, priority=1)
            await self.tts_manager.initialize()
            print("   ✔ TTS Manager initialized")
            
            # 5. AudioCapture初期化
            print("\n5. Initializing AudioCapture...")
            recording_config = RecordingConfig(
                device_id=self.settings.device_id,
                sample_rate=self.settings.sample_rate,
                channels=self.settings.channels,
                enable_agc=self.settings.enable_agc,
                enable_noise_reduction=self.settings.enable_noise_reduction
            )
            self.audio_capture = AudioCapture(recording_config)
            await self.audio_capture.safe_initialize()
            print("   ✔ AudioCapture initialized")
            
            # 6. VAD初期化（オプション）
            if self.settings.vad_enabled:
                print("\n6. Initializing VAD...")
                vad_config = VADConfig(
                    mode=self.settings.vad_mode,
                    sample_rate=self.settings.sample_rate,
                    speech_min_duration=self.settings.speech_min_duration,
                    silence_min_duration=self.settings.silence_min_duration
                )
                self.vad = VoiceActivityDetector(vad_config)
                await self.vad.initialize()
                print("   ✔ VAD initialized")
            else:
                print("\n6. VAD is disabled")
                self.vad = None
            
            # 7. DialogueManager作成
            print("\n7. Creating DialogueManager...")
            self.dialogue_manager = DialogueManager(
                config=config,
                llm_manager=self.llm_manager,
                stt_engine=self.stt_engine,
                tts_manager=self.tts_manager,
                audio_capture=self.audio_capture,
                vad=self.vad
            )
            
            # 8. DialogueManager初期化
            print("\n8. Initializing DialogueManager...")
            await self.dialogue_manager.initialize()
            print("   ✔ DialogueManager initialized successfully")
            
            # 統計情報表示
            stats = self.dialogue_manager.get_conversation_stats()
            print("\n" + "-"*60)
            print("DialogueManager Status:")
            print(f"  - LLM Available: {stats['llm_available']}")
            print(f"  - TTS Available: {stats['tts_available']}")
            print(f"  - STT Available: {stats['stt_available']}")
            print(f"  - Audio Capture: {stats['audio_capture_available']}")
            print(f"  - VAD Available: {stats['vad_available']}")
            print("-"*60)
            
            return self.dialogue_manager
            
        except Exception as e:
            print(f"\n✗ Failed to initialize DialogueManager: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    async def cleanup(self):
        """リソースのクリーンアップ"""
        if self.dialogue_manager:
            await self.dialogue_manager.cleanup()


# ============================================================================
# 環境チェック（既存と同じ）
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
        status = "✔" if available else "✗"
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
        print("   ⚠ Will try to use pyaudio fallback")
    
    # 4. APIキー
    print("\n4. API Keys:")
    gemini_key = os.environ.get('GEMINI_API_KEY')
    if gemini_key:
        print(f"   ✔ GEMINI_API_KEY is set (length: {len(gemini_key)})")
    else:
        print("   ⚠ GEMINI_API_KEY not set")
        print("     You can still test without it, but LLM features will be limited")
    
    # 5. インターネット接続
    print("\n5. Internet Connection:")
    try:
        import urllib.request
        urllib.request.urlopen('https://www.google.com', timeout=5)
        print("   ✔ Internet connected")
    except:
        print("   ✗ No internet connection")
        if gemini_key:
            all_ok = False
    
    print("\n" + "-"*60)
    if all_ok:
        print("✔ Environment check PASSED")
    else:
        print("⚠ Some issues detected, but test can continue")
    
    return all_ok


# ============================================================================
# DialogueManager統合テスト（v0.5.0の主要テスト）
# ============================================================================

async def test_dialogue_manager_info(dm_setup: DialogueManagerSetup, reporter: TestReporter):
    """DialogueManager情報表示"""
    print("\n" + "="*60)
    print("DIALOGUE MANAGER INFORMATION")
    print("="*60)
    
    try:
        dm = dm_setup.dialogue_manager
        
        # 統計情報取得
        stats = dm.get_conversation_stats()
        
        print("\n1. DialogueManager Status:")
        print(f"   State: {dm._state.value}")
        print(f"   Initialized: {dm._initialized}")
        print(f"   Conversation ID: {stats['conversation_id']}")
        
        print("\n2. Component Status:")
        print(f"   LLM Manager: {'Available' if stats['llm_available'] else 'Not Available'}")
        print(f"   TTS Manager: {'Available' if stats['tts_available'] else 'Not Available'}")
        print(f"   STT Engine: {'Available' if stats['stt_available'] else 'Not Available'}")
        print(f"   Audio Capture: {'Available' if stats['audio_capture_available'] else 'Not Available'}")
        print(f"   VAD: {'Available' if stats['vad_available'] else 'Not Available'}")
        
        print("\n3. Configuration:")
        print(f"   Max Turns: {dm.config.max_turns}")
        print(f"   Turn Timeout: {dm.config.turn_timeout}s")
        print(f"   Language: {dm.config.language}")
        print(f"   TTS Enabled: {dm.config.tts_enabled}")
        
        reporter.add_result(
            "DialogueManager Info",
            "PASS",
            {
                "state": dm._state.value,
                "components": sum([
                    stats['llm_available'],
                    stats['tts_available'],
                    stats['stt_available'],
                    stats['audio_capture_available'],
                    stats['vad_available']
                ])
            }
        )
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        reporter.add_result(
            "DialogueManager Info",
            "FAIL",
            {"error": str(e)}
        )


async def test_text_dialogue(dm_setup: DialogueManagerSetup, reporter: TestReporter, settings: TestSettings):
    """テキスト対話テスト（DialogueManager.process_text_input）"""
    print("\n" + "="*60)
    print("TEXT DIALOGUE TEST (DialogueManager)")
    print("="*60)
    
    try:
        dm = dm_setup.dialogue_manager
        
        print("\nEnter text messages to test dialogue.")
        print("Type 'exit' to finish this test.")
        print("-"*60)
        
        turn_count = 0
        while True:
            # ユーザー入力を受け付ける
            user_message = input(f"\n[Turn {turn_count + 1}] You: ")
            
            if user_message.lower() in ['exit', 'quit', '終了']:
                print("Ending text dialogue test.")
                break
            
            if not user_message.strip():
                print("Please enter a message.")
                continue
            
            turn_count += 1
            
            # DialogueManagerで処理
            start_time = time.time()
            turn = await dm.process_text_input(user_message)
            process_time = time.time() - start_time
            
            # 応答表示
            print(f"Assistant: {turn.assistant_response}")
            print(f"Processing time: {process_time:.2f}s")
            
            # 音声再生がある場合
            if turn.audio_response:
                print("🔊 Playing audio response...")
            
            # 統計情報
            print(f"Turn number: {turn.turn_number}, Confidence: {turn.confidence:.2%}")
            
            reporter.add_result(
                f"Text Dialogue Turn {turn_count}",
                "PASS",
                {
                    "user_input": user_message[:50],
                    "response": turn.assistant_response[:50],
                    "has_audio": turn.audio_response is not None,
                    "time": f"{process_time:.2f}s"
                },
                process_time
            )
            
            # 5ターンごとに履歴確認
            if turn_count % 5 == 0:
                history = dm.get_conversation_history(5)
                print(f"\n📝 Recent history: {len(history)} turns")
        
        # 最終的な会話履歴確認
        history = dm.get_conversation_history(10)
        print(f"\n✔ Total conversation: {len(history)} turns")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        reporter.add_result(
            "Text Dialogue Test",
            "FAIL",
            {"error": str(e)}
        )


async def test_audio_dialogue(dm_setup: DialogueManagerSetup, reporter: TestReporter, settings: TestSettings):
    """音声対話テスト（DialogueManager.process_audio_input）"""
    print("\n" + "="*60)
    print("AUDIO DIALOGUE TEST (DialogueManager)")
    print("="*60)
    
    try:
        dm = dm_setup.dialogue_manager
        
        # 録音モード選択
        print("\nSelect recording mode:")
        print("1. Fixed duration (traditional)")
        print("2. VAD auto-detect (automatic stop)")
        mode = input("Select mode (1-2): ")
        
        rounds = 3
        for round_num in range(1, rounds + 1):
            print(f"\n[Round {round_num}]")
            
            if mode == '1':
                # 固定時間録音
                input("Press ENTER to speak...")
                print(f"🎤 Recording for {settings.record_duration} seconds...")
                
                start_time = time.time()
                # DialogueManagerが内部でAudioCaptureを使用
                turn = await dm.process_audio_input()
                total_time = time.time() - start_time
                
            else:
                # VAD自動検出（DialogueManagerが内部で処理）
                print("🎤 Speak now (VAD will detect when you stop)...")
                
                start_time = time.time()
                turn = await dm.process_audio_input()
                total_time = time.time() - start_time
            
            # 結果表示
            print(f"\n🎤 You said: '{turn.user_input}'")
            print(f"🤖 Response: '{turn.assistant_response}'")
            
            if turn.audio_response:
                print("🔊 Playing response...")
            
            print(f"\n⏱ Total response time: {total_time:.2f}s")
            print(f"   Turn number: {turn.turn_number}")
            print(f"   Confidence: {turn.confidence:.2%}")
            
            reporter.add_result(
                f"Audio Dialogue Round {round_num}",
                "PASS",
                {
                    "mode": "Fixed" if mode == '1' else "VAD",
                    "user_input": turn.user_input[:50],
                    "response": turn.assistant_response[:50],
                    "total_time": f"{total_time:.2f}s",
                    "turn_number": turn.turn_number
                },
                total_time
            )
            
            # 終了チェック
            exit_words = ['さようなら', 'goodbye', 'bye', 'exit', '終了']
            if any(word in turn.user_input.lower() for word in exit_words):
                print("\n👋 Goodbye!")
                break
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        reporter.add_result(
            "Audio Dialogue Test",
            "FAIL",
            {"error": str(e)}
        )


async def test_conversation_management(dm_setup: DialogueManagerSetup, reporter: TestReporter):
    """会話管理テスト"""
    print("\n" + "="*60)
    print("CONVERSATION MANAGEMENT TEST")
    print("="*60)
    
    try:
        dm = dm_setup.dialogue_manager
        
        print("\nThis test demonstrates conversation management features.")
        print("You can test history, reset, and statistics functions.")
        print("-"*60)
        
        while True:
            print("\n=== Conversation Management Menu ===")
            print("1. Add a message to conversation")
            print("2. View conversation history")
            print("3. Reset conversation")
            print("4. View statistics")
            print("5. Clear conversation (keep session)")
            print("0. Exit test")
            
            choice = input("\nSelect option (0-5): ")
            
            if choice == '0':
                break
                
            elif choice == '1':
                # ユーザー入力で会話追加
                user_message = input("Enter your message: ")
                if user_message.strip():
                    print("\nProcessing...")
                    start_time = time.time()
                    turn = await dm.process_text_input(user_message)
                    process_time = time.time() - start_time
                    
                    print(f"You: {user_message}")
                    print(f"Assistant: {turn.assistant_response}")
                    print(f"Turn #{turn.turn_number} completed in {process_time:.2f}s")
                    
                    reporter.add_result(
                        f"Conversation Management - Add Turn",
                        "PASS",
                        {
                            "turn_number": turn.turn_number,
                            "time": f"{process_time:.2f}s"
                        }
                    )
                
            elif choice == '2':
                # 履歴表示
                limit = input("How many recent turns to view? (default: 5): ")
                limit = int(limit) if limit.isdigit() else 5
                
                history = dm.get_conversation_history(limit)
                print(f"\n=== Last {limit} turns ===")
                if history:
                    for turn in history:
                        print(f"\nTurn #{turn.turn_number}:")
                        print(f"  You: {turn.user_input[:50]}...")
                        print(f"  Assistant: {turn.assistant_response[:50]}...")
                        print(f"  Time: {turn.timestamp.strftime('%H:%M:%S')}")
                else:
                    print("No conversation history yet.")
                
                reporter.add_result(
                    "View History",
                    "PASS",
                    {"turns_retrieved": len(history)}
                )
                
            elif choice == '3':
                # 会話リセット
                confirm = input("Reset conversation? This will clear all history (y/n): ")
                if confirm.lower() == 'y':
                    old_stats = dm.get_conversation_stats()
                    old_id = old_stats['conversation_id']
                    old_turns = old_stats['total_turns']
                    
                    await dm.reset_conversation()
                    
                    new_stats = dm.get_conversation_stats()
                    new_id = new_stats['conversation_id']
                    
                    print(f"\n✔ Conversation reset")
                    print(f"  Old ID: {old_id} ({old_turns} turns)")
                    print(f"  New ID: {new_id} (0 turns)")
                    
                    reporter.add_result(
                        "Conversation Reset",
                        "PASS",
                        {
                            "old_id": old_id,
                            "new_id": new_id,
                            "old_turns": old_turns
                        }
                    )
                    
            elif choice == '4':
                # 統計情報表示
                stats = dm.get_conversation_stats()
                print("\n=== Conversation Statistics ===")
                print(f"Conversation ID: {stats['conversation_id']}")
                print(f"Total turns: {stats['total_turns']}")
                print(f"Current turn: {stats['current_turn_number']}")
                print(f"State: {stats['state']}")
                print(f"Duration: {stats['duration_seconds']:.1f} seconds")
                print(f"Started at: {stats['started_at']}")
                print(f"Last activity: {stats['last_activity']}")
                
                reporter.add_result(
                    "View Statistics",
                    "PASS",
                    {
                        "total_turns": stats['total_turns'],
                        "state": stats['state']
                    }
                )
                
            elif choice == '5':
                # 履歴クリア（セッション維持）
                confirm = input("Clear history but keep session? (y/n): ")
                if confirm.lower() == 'y':
                    dm.clear_conversation()
                    print("✔ Conversation history cleared (session maintained)")
                    
                    reporter.add_result(
                        "Clear History",
                        "PASS",
                        {}
                    )
            else:
                print("Invalid choice")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        reporter.add_result(
            "Conversation Management",
            "FAIL",
            {"error": str(e)}
        )


# ============================================================================
# メイン実行
# ============================================================================

async def main():
    """メインテスト実行"""
    print("\n" + "="*60)
    print(" VioraTalk v0.5.0 DialogueManager Integration Test")
    print(" Version: 0.5.0 - DialogueManager統合版")
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
    
    # DialogueManagerセットアップ
    dm_setup = DialogueManagerSetup(settings)
    dialogue_manager = None
    
    try:
        # 初期化
        dialogue_manager = await dm_setup.initialize()
        
        # テストメニュー
        while True:
            print("\n" + "="*60)
            print("TEST MENU")
            print("="*60)
            settings.display()
            print("-"*60)
            print("=== DialogueManager Tests ===")
            print("1. DialogueManager Information")
            print("2. Text Dialogue Test")
            print("3. Audio Dialogue Test")
            print("4. Conversation Management Test")
            print("\n=== Integration ===")
            print("5. Run All Tests")
            print("\n=== Reports & Settings ===")
            print("6. Generate Report")
            print("C. Change Settings")
            print("0. Exit")
            print("-"*60)
            
            choice = input("\nSelect test (0-6, C): ").upper()
            
            if choice == '0':
                break
            elif choice == '1':
                await test_dialogue_manager_info(dm_setup, reporter)
            elif choice == '2':
                await test_text_dialogue(dm_setup, reporter, settings)
            elif choice == '3':
                await test_audio_dialogue(dm_setup, reporter, settings)
            elif choice == '4':
                await test_conversation_management(dm_setup, reporter)
            elif choice == '5':
                # 全テスト実行
                await test_dialogue_manager_info(dm_setup, reporter)
                await test_text_dialogue(dm_setup, reporter, settings)
                await test_audio_dialogue(dm_setup, reporter, settings)
                await test_conversation_management(dm_setup, reporter)
            elif choice == '6':
                # レポート生成
                print(reporter.generate_report())
                save = input("\nSave report to JSON? (y/n): ")
                if save.lower() == 'y':
                    reporter.save_json()
            elif choice == 'C':
                settings.change_settings()
                # 設定変更後は再初期化が必要
                print("\n⚠ Settings changed. Reinitializing DialogueManager...")
                await dm_setup.cleanup()
                dialogue_manager = await dm_setup.initialize()
            else:
                print("Invalid choice")
        
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # クリーンアップ
        print("\nCleaning up...")
        if dm_setup:
            await dm_setup.cleanup()
        print("✔ Cleanup completed")
    
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
