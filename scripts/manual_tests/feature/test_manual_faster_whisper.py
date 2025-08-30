#!/usr/bin/env python3
"""
FasterWhisperEngine 手動テストスクリプト

音声認識エンジンの実動作確認を行うための手動テストツール。
音声ファイルによる基本動作確認とリアルタイム音声入力テストの両方をサポート。

実行方法:
    # 基本動作確認（音声ファイル）
    python scripts/manual_tests/feature/test_manual_faster_whisper.py --mode file
    
    # リアルタイムテスト（マイク入力）
    python scripts/manual_tests/feature/test_manual_faster_whisper.py --mode realtime
    
    # 両方実行（ベンチマーク含む）
    python scripts/manual_tests/feature/test_manual_faster_whisper.py --mode all

必要環境:
    - faster-whisper インストール済み
    - マイク（リアルタイムテストの場合）
    - テスト用音声ファイル（tests/fixtures/audio/）

開発規約書 v1.12準拠
テスト戦略ガイドライン v1.7準拠
"""

import asyncio
import argparse
import sys
import time
import json
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from dataclasses import dataclass, asdict

# NumPy（音声処理用）
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    warnings.warn("NumPy not installed. Some features may be limited.")

# sounddevice（マイク入力用）
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    warnings.warn("sounddevice not installed. Realtime mode will not be available.")

# keyboard（Push-to-Talk用）
try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    if SOUNDDEVICE_AVAILABLE:
        warnings.warn("keyboard not installed. Using simple input instead of Push-to-Talk.")

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# VioraTalkインポート
try:
    from vioratalk.core.stt.faster_whisper_engine import FasterWhisperEngine
    from vioratalk.core.stt.base import AudioData, AudioMetadata, STTConfig
    from vioratalk.core.exceptions import STTError, AudioError, ModelNotFoundError
    VIORATALK_AVAILABLE = True
except ImportError as e:
    print(f"Error: Failed to import VioraTalk modules: {e}")
    print("Make sure the project is properly set up.")
    VIORATALK_AVAILABLE = False
    sys.exit(1)


# ============================================================================
# テスト結果記録用
# ============================================================================

@dataclass
class TestResult:
    """テスト結果を記録するデータクラス"""
    test_name: str
    status: str  # "PASS", "FAIL", "SKIP"
    duration: float
    details: Dict[str, Any]
    error: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class TestReporter:
    """テスト結果のレポート管理"""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.start_time = time.time()
    
    def add_result(self, result: TestResult):
        """結果を追加"""
        self.results.append(result)
        self._print_result(result)
    
    def _print_result(self, result: TestResult):
        """結果を即座に表示"""
        status_symbol = {
            "PASS": "✓",
            "FAIL": "✗", 
            "SKIP": "○"
        }.get(result.status, "?")
        
        print(f"\n[{status_symbol}] {result.test_name}")
        print(f"    Duration: {result.duration:.2f}s")
        
        if result.details:
            for key, value in result.details.items():
                print(f"    {key}: {value}")
        
        if result.error:
            print(f"    Error: {result.error}")
    
    def generate_summary(self):
        """サマリーレポートを生成"""
        total_time = time.time() - self.start_time
        passed = sum(1 for r in self.results if r.status == "PASS")
        failed = sum(1 for r in self.results if r.status == "FAIL")
        skipped = sum(1 for r in self.results if r.status == "SKIP")
        
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Total Tests: {len(self.results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Skipped: {skipped}")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Success Rate: {passed/len(self.results)*100:.1f}%")
        
        if failed > 0:
            print("\nFailed Tests:")
            for r in self.results:
                if r.status == "FAIL":
                    print(f"  - {r.test_name}: {r.error}")
    
    def save_report(self, filepath: Path):
        """レポートをJSONファイルに保存"""
        report = {
            "summary": {
                "total": len(self.results),
                "passed": sum(1 for r in self.results if r.status == "PASS"),
                "failed": sum(1 for r in self.results if r.status == "FAIL"),
                "skipped": sum(1 for r in self.results if r.status == "SKIP"),
                "duration": time.time() - self.start_time
            },
            "results": [asdict(r) for r in self.results]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)
        
        print(f"\nReport saved to: {filepath}")


# ============================================================================
# FasterWhisperEngine テストクラス
# ============================================================================

class FasterWhisperManualTest:
    """FasterWhisperEngineの手動テスト"""
    
    def __init__(self, model_size: str = "base", device: str = "auto"):
        self.model_size = model_size
        self.device = device
        self.engine: Optional[FasterWhisperEngine] = None
        self.reporter = TestReporter()
        self.fixtures_dir = project_root / "tests" / "fixtures" / "audio"
        
        # テスト用音声ファイル
        self.test_files = {
            "japanese": self.fixtures_dir / "hello_japanese.wav",
            "english": self.fixtures_dir / "hello_english.wav",
            "silence": self.fixtures_dir / "silence.wav",
            "noisy": self.fixtures_dir / "noisy.wav"
        }
    
    async def setup(self):
        """エンジンのセットアップ"""
        print("\n" + "="*60)
        print("SETUP: Initializing FasterWhisperEngine")
        print("="*60)
        
        start_time = time.time()
        
        try:
            # STT設定
            config = STTConfig(
                engine="faster-whisper",
                model=self.model_size,
                language="ja",
                device=self.device
            )
            
            # エンジン作成
            print(f"Creating engine with model='{self.model_size}', device='{self.device}'...")
            self.engine = FasterWhisperEngine(config)
            
            # 非同期初期化
            print("Initializing engine...")
            await self.engine.initialize()
            
            duration = time.time() - start_time
            print(f"✓ Engine initialized successfully in {duration:.2f}s")
            
            # デバイス情報を表示
            print(f"  - Model: {self.engine.current_model}")
            print(f"  - Device: {self.engine.config.device}")
            print(f"  - Supported Languages: {self.engine.supported_languages}")
            
            self.reporter.add_result(TestResult(
                test_name="Engine Setup",
                status="PASS",
                duration=duration,
                details={
                    "model": self.model_size,
                    "device": self.engine.config.device
                }
            ))
            
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"✗ Setup failed: {e}")
            self.reporter.add_result(TestResult(
                test_name="Engine Setup",
                status="FAIL",
                duration=duration,
                details={
                    "model": self.model_size,
                    "device": self.device,
                    "error_type": type(e).__name__
                },
                error=str(e)
            ))
            return False
    
    async def cleanup(self):
        """エンジンのクリーンアップ"""
        if self.engine:
            print("\nCleaning up engine...")
            await self.engine.cleanup()
            print("✓ Cleanup completed")
    
    # ========================================================================
    # 音声ファイルテスト
    # ========================================================================
    
    def load_audio_file(self, filepath: Path) -> Optional[AudioData]:
        """音声ファイルを読み込んでAudioDataに変換"""
        try:
            import wave
            
            with wave.open(str(filepath), 'rb') as wf:
                frames = wf.readframes(wf.getnframes())
                sample_rate = wf.getframerate()
                channels = wf.getnchannels()
                
                # バイト列をNumPy配列に変換
                audio_array = np.frombuffer(frames, dtype=np.int16)
                # -1.0 to 1.0に正規化
                audio_array = audio_array.astype(np.float32) / 32768.0
                
                return AudioData(
                    raw_data=audio_array,
                    metadata=AudioMetadata(
                        sample_rate=sample_rate,
                        channels=channels,
                        duration=len(audio_array) / sample_rate,
                        format="wav",
                        filename=filepath.name
                    )
                )
                
        except Exception as e:
            print(f"Error loading audio file {filepath}: {e}")
            return None
    
    async def test_file_basic(self):
        """基本的な音声ファイル認識テスト"""
        print("\n" + "="*60)
        print("TEST: Basic File Recognition")
        print("="*60)
        
        for name, filepath in self.test_files.items():
            if not filepath.exists():
                print(f"⚠ Skipping {name}: File not found ({filepath})")
                self.reporter.add_result(TestResult(
                    test_name=f"File Recognition - {name}",
                    status="SKIP",
                    duration=0,
                    details={
                        "file": filepath.name,
                        "path": str(filepath)
                    },
                    error="File not found"
                ))
                continue
            
            print(f"\nTesting: {name} ({filepath.name})")
            start_time = time.time()
            
            try:
                # 音声ファイル読み込み
                audio_data = self.load_audio_file(filepath)
                if not audio_data:
                    raise Exception("Failed to load audio file")
                
                # 認識実行
                result = await self.engine.transcribe(audio_data)
                
                duration = time.time() - start_time
                
                print(f"  Text: '{result.text}'")
                print(f"  Confidence: {result.confidence:.2f}")
                print(f"  Language: {result.language}")
                print(f"  Processing Time: {duration:.2f}s")
                
                self.reporter.add_result(TestResult(
                    test_name=f"File Recognition - {name}",
                    status="PASS",
                    duration=duration,
                    details={
                        "text": result.text,
                        "confidence": result.confidence,
                        "language": result.language
                    }
                ))
                
            except Exception as e:
                duration = time.time() - start_time
                print(f"  ✗ Error: {e}")
                self.reporter.add_result(TestResult(
                    test_name=f"File Recognition - {name}",
                    status="FAIL",
                    duration=duration,
                    details={
                        "file": filepath.name,
                        "error_type": type(e).__name__
                    },
                    error=str(e)
                ))
    
    async def test_model_switching(self):
        """モデル切り替えテスト"""
        print("\n" + "="*60)
        print("TEST: Model Switching")
        print("="*60)
        
        # 全モデルサイズをテスト（サイズ順）
        models_to_test = ["tiny", "base", "small", "medium", "large-v3"]  # 各モデルで精度比較
        
        for model in models_to_test:
            print(f"\nSwitching to model: {model}")
            start_time = time.time()
            
            try:
                # set_modelは同期メソッドなのでawaitしない
                self.engine.set_model(model)
                
                # モデル変更後は再初期化が必要
                print(f"  Re-initializing engine with {model} model...")
                await self.engine.initialize()
                
                duration = time.time() - start_time
                
                print(f"  ✓ Successfully switched to {model} in {duration:.2f}s")
                
                # 簡単な認識テストで動作確認
                if self.test_files["japanese"].exists():
                    audio_data = self.load_audio_file(self.test_files["japanese"])
                    if audio_data:
                        result = await self.engine.transcribe(audio_data)
                        print(f"  Test recognition: '{result.text}'")
                
                self.reporter.add_result(TestResult(
                    test_name=f"Model Switch - {model}",
                    status="PASS",
                    duration=duration,
                    details={"model": model}
                ))
                
            except Exception as e:
                duration = time.time() - start_time
                print(f"  ✗ Error: {e}")
                self.reporter.add_result(TestResult(
                    test_name=f"Model Switch - {model}",
                    status="FAIL",
                    duration=duration,
                    details={
                        "model": model,
                        "error_type": type(e).__name__
                    },
                    error=str(e)
                ))
    
    async def test_error_handling(self):
        """エラーハンドリングテスト"""
        print("\n" + "="*60)
        print("TEST: Error Handling")
        print("="*60)
        
        # テストケース
        test_cases = [
            ("Empty Audio", AudioData(
                raw_data=np.array([], dtype=np.float32),
                metadata=AudioMetadata()
            ), STTError),  # FasterWhisperEngineはSTTErrorでラップする
            ("Invalid Format", AudioData(
                raw_data=np.array([0.1, 0.2], dtype=np.float32),
                metadata=AudioMetadata(sample_rate=8000)  # 低すぎるサンプルレート
            ), None),
            ("Too Quiet", AudioData(
                raw_data=np.zeros(16000, dtype=np.float32),  # 1秒の無音
                metadata=AudioMetadata(sample_rate=16000)
            ), None)
        ]
        
        for name, audio_data, expected_error in test_cases:
            print(f"\nTesting: {name}")
            start_time = time.time()
            
            try:
                result = await self.engine.transcribe(audio_data)
                duration = time.time() - start_time
                
                if expected_error:
                    print(f"  ⚠ Expected error but got result: '{result.text}'")
                    status = "FAIL"
                else:
                    print(f"  ✓ Handled correctly: '{result.text}'")
                    status = "PASS"
                
                self.reporter.add_result(TestResult(
                    test_name=f"Error Handling - {name}",
                    status=status,
                    duration=duration,
                    details={"result": result.text if result else None}
                ))
                
            except Exception as e:
                duration = time.time() - start_time
                
                if expected_error and isinstance(e, expected_error):
                    print(f"  ✓ Correctly raised {expected_error.__name__}: {e}")
                    status = "PASS"
                else:
                    print(f"  ✗ Unexpected error: {e}")
                    status = "FAIL"
                
                self.reporter.add_result(TestResult(
                    test_name=f"Error Handling - {name}",
                    status=status,
                    duration=duration,
                    details={
                        "expected_error": expected_error.__name__ if expected_error else "None",
                        "actual_error": type(e).__name__
                    },
                    error=str(e)
                ))
    
    # ========================================================================
    # リアルタイムテスト
    # ========================================================================
    
    def record_audio(self, duration: float = 3.0, sample_rate: int = 16000) -> Optional[AudioData]:
        """マイクから音声を録音"""
        if not SOUNDDEVICE_AVAILABLE:
            print("Error: sounddevice not available")
            return None
        
        try:
            print(f"Recording for {duration} seconds...")
            print("Speak now!")
            
            # 録音
            audio = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype='float32'
            )
            sd.wait()  # 録音完了まで待機
            
            print("Recording completed")
            
            # AudioDataに変換
            return AudioData(
                raw_data=audio.flatten(),
                metadata=AudioMetadata(
                    sample_rate=sample_rate,
                    channels=1,
                    duration=duration,
                    format="pcm"
                )
            )
            
        except Exception as e:
            print(f"Recording error: {e}")
            return None
    
    def record_push_to_talk(self, sample_rate: int = 16000) -> Optional[AudioData]:
        """Push-to-Talk方式で録音"""
        if not SOUNDDEVICE_AVAILABLE:
            print("Error: sounddevice not available")
            return None
        
        print("\n" + "="*40)
        print("Push-to-Talk Recording")
        print("="*40)
        
        if KEYBOARD_AVAILABLE:
            print("Hold SPACE key to record, release to stop")
            print("Press ESC to cancel")
        else:
            print("Press ENTER to start recording")
            print("Press ENTER again to stop")
        
        frames = []
        
        try:
            with sd.InputStream(samplerate=sample_rate, 
                              channels=1, 
                              dtype='float32') as stream:
                
                if KEYBOARD_AVAILABLE:
                    # keyboard使用可能な場合
                    recording = False
                    while True:
                        if keyboard.is_pressed('esc'):
                            print("\nCancelled")
                            return None
                        
                        if keyboard.is_pressed('space'):
                            if not recording:
                                print("Recording...", end="", flush=True)
                                recording = True
                            
                            frame, _ = stream.read(1024)
                            frames.append(frame)
                            print(".", end="", flush=True)
                        
                        elif recording:
                            # スペースキーを離したら録音終了
                            print("\nRecording stopped")
                            break
                else:
                    # keyboard使用不可の場合（簡易版）
                    input("Press ENTER to start recording...")
                    print("Recording... Press ENTER to stop")
                    
                    # 別スレッドで録音（簡易実装）
                    import threading
                    stop_flag = threading.Event()
                    
                    def record_thread():
                        while not stop_flag.is_set():
                            frame, _ = stream.read(1024)
                            frames.append(frame)
                    
                    thread = threading.Thread(target=record_thread)
                    thread.start()
                    
                    input()  # ENTERを待つ
                    stop_flag.set()
                    thread.join()
                    print("Recording stopped")
            
            if not frames:
                print("No audio recorded")
                return None
            
            # 結合してAudioDataに変換
            audio = np.concatenate(frames)
            duration = len(audio) / sample_rate
            
            print(f"Recorded {duration:.1f} seconds")
            
            return AudioData(
                raw_data=audio,
                metadata=AudioMetadata(
                    sample_rate=sample_rate,
                    channels=1,
                    duration=duration,
                    format="pcm"
                )
            )
            
        except Exception as e:
            print(f"Recording error: {e}")
            return None
    
    async def test_realtime_basic(self):
        """基本的なリアルタイム認識テスト（指定文章版）"""
        print("\n" + "="*60)
        print("TEST: Realtime Recognition with Specified Phrases")
        print("="*60)
        
        if not SOUNDDEVICE_AVAILABLE:
            print("⚠ Skipping: sounddevice not available")
            self.reporter.add_result(TestResult(
                test_name="Realtime Recognition",
                status="SKIP",
                duration=0,
                details={
                    "reason": "dependency_missing",
                    "dependency": "sounddevice"
                },
                error="sounddevice not installed"
            ))
            return
        
        # テストシナリオと期待される発話内容
        test_scenarios = [
            ("Japanese Simple", "こんにちは、音声認識のテストです。", 5.0),
            ("Japanese Complex", "本日は晴天なり、マイクのテスト中です。", 5.0),
            ("Numbers and Units", "今日の気温は25度で、湿度は60パーセントです。", 6.0),
            ("Technical Terms", "人工知能と機械学習の違いについて説明します。", 6.0),
            ("English Simple", "Hello, this is a speech recognition test.", 5.0),
            ("Mixed Language", "これはAIによるspeech recognitionのテストです。", 6.0)
        ]
        
        results = []
        
        for scenario_name, expected_text, duration in test_scenarios:
            print(f"\n{'='*40}")
            print(f"Scenario: {scenario_name}")
            print('='*40)
            print(f"Please say EXACTLY this phrase:")
            print(f"「{expected_text}」")
            print(f"\nYou have {duration} seconds.")
            input("Press ENTER when ready to start recording...")
            
            # 録音
            audio_data = self.record_audio(duration=duration)
            
            if not audio_data:
                self.reporter.add_result(TestResult(
                    test_name=f"Realtime - {scenario_name}",
                    status="SKIP",
                    duration=0,
                    details={
                        "scenario": scenario_name,
                        "reason": "recording_failed"
                    },
                    error="Recording failed"
                ))
                continue
            
            # 音声レベルチェック
            audio_level = np.abs(audio_data.raw_data).max()
            print(f"Audio level: {audio_level:.3f}")
            
            if audio_level < 0.01:
                print("⚠ Warning: Very low audio level detected")
            
            # 認識
            start_time = time.time()
            
            try:
                result = await self.engine.transcribe(audio_data)
                processing_time = time.time() - start_time
                
                # 精度計算
                accuracy = self._calculate_accuracy(result.text, expected_text)
                
                print(f"\n  Expected: '{expected_text}'")
                print(f"  Result:   '{result.text}'")
                print(f"  Accuracy: {accuracy:.1%}")
                print(f"  Confidence: {result.confidence:.3f}")
                print(f"  Processing Time: {processing_time:.2f}s")
                
                # 結果を保存
                results.append({
                    "scenario": scenario_name,
                    "expected": expected_text,
                    "recognized": result.text,
                    "accuracy": accuracy,
                    "confidence": result.confidence,
                    "processing_time": processing_time,
                    "audio_level": audio_level
                })
                
                self.reporter.add_result(TestResult(
                    test_name=f"Realtime - {scenario_name}",
                    status="PASS" if accuracy > 0.8 else "FAIL",
                    duration=processing_time,
                    details={
                        "text": result.text,
                        "expected": expected_text,
                        "accuracy": accuracy,
                        "confidence": result.confidence,
                        "language": result.language,
                        "audio_level": audio_level
                    }
                ))
                
            except Exception as e:
                processing_time = time.time() - start_time
                print(f"  ✗ Error: {e}")
                self.reporter.add_result(TestResult(
                    test_name=f"Realtime - {scenario_name}",
                    status="FAIL",
                    duration=processing_time,
                    details={
                        "scenario": scenario_name,
                        "error_type": type(e).__name__
                    },
                    error=str(e)
                ))
        
        # 分析レポート
        if results:
            self._print_realtime_analysis(results)
    
    def _calculate_accuracy(self, recognized: str, expected: str) -> float:
        """文字列の精度を計算（レーベンシュタイン距離ベース）"""
        if not recognized and not expected:
            return 1.0
        if not recognized or not expected:
            return 0.0
        
        # 簡易的な文字一致率計算
        # より正確にはdifflib.SequenceMatcherを使用可能
        import difflib
        return difflib.SequenceMatcher(None, recognized, expected).ratio()
    
    def _print_realtime_analysis(self, results):
        """リアルタイム認識結果の分析レポート"""
        print("\n" + "="*60)
        print("REALTIME RECOGNITION ANALYSIS")
        print("="*60)
        
        # 全体統計
        avg_accuracy = np.mean([r["accuracy"] for r in results])
        avg_confidence = np.mean([r["confidence"] for r in results])
        avg_time = np.mean([r["processing_time"] for r in results])
        avg_audio_level = np.mean([r["audio_level"] for r in results])
        
        print(f"\nOverall Statistics:")
        print(f"  Average Accuracy: {avg_accuracy:.1%}")
        print(f"  Average Confidence: {avg_confidence:.3f}")
        print(f"  Average Processing Time: {avg_time:.2f}s")
        print(f"  Average Audio Level: {avg_audio_level:.3f}")
        
        # カテゴリ別分析
        print("\n" + "-"*60)
        print("Detailed Results:")
        print("-"*60)
        
        for res in results:
            status = "✓" if res["accuracy"] > 0.8 else "✗"
            print(f"\n[{status}] {res['scenario']}")
            print(f"    Accuracy: {res['accuracy']:.1%}")
            print(f"    Confidence: {res['confidence']:.3f}")
            print(f"    Audio Level: {res['audio_level']:.3f}")
            
            # エラー分析
            if res["accuracy"] < 1.0:
                print(f"    Expected:   '{res['expected']}'")
                print(f"    Recognized: '{res['recognized']}'")
        
        # 問題の特定
        print("\n" + "-"*60)
        print("Issues Identified:")
        print("-"*60)
        
        low_accuracy = [r for r in results if r["accuracy"] < 0.8]
        if low_accuracy:
            print(f"  Low Accuracy ({len(low_accuracy)} scenarios):")
            for r in low_accuracy:
                print(f"    - {r['scenario']}: {r['accuracy']:.1%}")
        
        low_confidence = [r for r in results if r["confidence"] < 0.5]
        if low_confidence:
            print(f"  Low Confidence ({len(low_confidence)} scenarios):")
            for r in low_confidence:
                print(f"    - {r['scenario']}: {r['confidence']:.3f}")
        
        low_audio = [r for r in results if r["audio_level"] < 0.05]
        if low_audio:
            print(f"  Low Audio Level ({len(low_audio)} scenarios):")
            for r in low_audio:
                print(f"    - {r['scenario']}: {r['audio_level']:.3f}")
        
        if not (low_accuracy or low_confidence or low_audio):
            print("  No significant issues detected!")
    
    async def test_push_to_talk(self):
        """Push-to-Talk テスト（改善版）"""
        print("\n" + "="*60)
        print("TEST: Push-to-Talk Mode with Specified Phrases")
        print("="*60)
        
        if not SOUNDDEVICE_AVAILABLE:
            print("⚠ Skipping: sounddevice not available")
            self.reporter.add_result(TestResult(
                test_name="Push-to-Talk",
                status="SKIP",
                duration=0,
                details={
                    "reason": "dependency_missing",
                    "dependency": "sounddevice"
                },
                error="sounddevice not installed"
            ))
            return
        
        # テストフレーズ
        test_phrases = [
            "音声認識のテストをしています。",
            "プッシュトゥトークのテスト中です。",
            "今日は良い天気ですね。"
        ]
        
        print("\nThis test will run 3 times with different phrases")
        
        for i, expected_text in enumerate(test_phrases):
            print(f"\n--- Round {i+1}/3 ---")
            print(f"Please say: 「{expected_text}」")
            
            # Push-to-Talk録音（改善版）
            audio_data = self.record_push_to_talk_improved()
            
            if not audio_data:
                self.reporter.add_result(TestResult(
                    test_name=f"Push-to-Talk #{i+1}",
                    status="SKIP",
                    duration=0,
                    details={
                        "round": i+1,
                        "reason": "recording_cancelled"
                    },
                    error="Recording cancelled or failed"
                ))
                continue
            
            # 音声レベル確認
            audio_level = np.abs(audio_data.raw_data).max()
            print(f"Audio level: {audio_level:.3f}")
            print(f"Duration: {audio_data.metadata.duration:.1f}s")
            
            # 認識
            start_time = time.time()
            
            try:
                result = await self.engine.transcribe(audio_data)
                duration = time.time() - start_time
                
                # 精度計算
                accuracy = self._calculate_accuracy(result.text, expected_text)
                
                print(f"\n  Expected: '{expected_text}'")
                print(f"  Result:   '{result.text}'")
                print(f"  Accuracy: {accuracy:.1%}")
                print(f"  Confidence: {result.confidence:.3f}")
                print(f"  Processing Time: {duration:.2f}s")
                
                self.reporter.add_result(TestResult(
                    test_name=f"Push-to-Talk #{i+1}",
                    status="PASS" if accuracy > 0.7 else "FAIL",
                    duration=duration,
                    details={
                        "text": result.text,
                        "expected": expected_text,
                        "accuracy": accuracy,
                        "confidence": result.confidence,
                        "audio_duration": audio_data.metadata.duration,
                        "audio_level": audio_level
                    }
                ))
                
            except Exception as e:
                duration = time.time() - start_time
                print(f"  ✗ Error: {e}")
                self.reporter.add_result(TestResult(
                    test_name=f"Push-to-Talk #{i+1}",
                    status="FAIL",
                    duration=duration,
                    details={
                        "round": i+1,
                        "error_type": type(e).__name__
                    },
                    error=str(e)
                ))
    
    def record_push_to_talk_improved(self, sample_rate: int = 16000) -> Optional[AudioData]:
        """改善版Push-to-Talk録音"""
        if not SOUNDDEVICE_AVAILABLE:
            print("Error: sounddevice not available")
            return None
        
        print("\n" + "="*40)
        print("Push-to-Talk Recording")
        print("="*40)
        
        if KEYBOARD_AVAILABLE:
            print("Hold SPACE key to record, release to stop")
            print("Press ESC to cancel")
        else:
            print("Press ENTER to start recording")
            print("Press ENTER again to stop")
        
        frames = []
        
        try:
            # より大きなバッファサイズを使用
            buffer_size = 4096
            
            with sd.InputStream(samplerate=sample_rate, 
                              channels=1, 
                              dtype='float32',
                              blocksize=buffer_size) as stream:
                
                if KEYBOARD_AVAILABLE:
                    # keyboard使用可能な場合
                    recording = False
                    while True:
                        if keyboard.is_pressed('esc'):
                            print("\nCancelled")
                            return None
                        
                        if keyboard.is_pressed('space'):
                            if not recording:
                                print("Recording...", end="", flush=True)
                                recording = True
                            
                            frame, _ = stream.read(buffer_size)
                            frames.append(frame)
                            print(".", end="", flush=True)
                        
                        elif recording:
                            # スペースキーを離したら録音終了
                            print("\nRecording stopped")
                            break
                else:
                    # keyboard使用不可の場合（改善版）
                    input("Press ENTER to start recording...")
                    print("Recording... Press ENTER to stop")
                    
                    # 別スレッドで録音
                    import threading
                    import queue
                    
                    q = queue.Queue()
                    stop_flag = threading.Event()
                    
                    def record_thread():
                        while not stop_flag.is_set():
                            frame, _ = stream.read(buffer_size)
                            q.put(frame)
                    
                    thread = threading.Thread(target=record_thread)
                    thread.start()
                    
                    input()  # ENTERを待つ
                    stop_flag.set()
                    thread.join()
                    
                    # キューからフレームを取得
                    while not q.empty():
                        frames.append(q.get())
                    
                    print("Recording stopped")
            
            if not frames:
                print("No audio recorded")
                return None
            
            # 結合してAudioDataに変換
            audio = np.concatenate(frames)
            audio = audio.flatten()  # 1次元配列に変換（重要）
            
            # デバッグ情報を表示
            print(f"\nDebug info:")
            print(f"  Total frames: {len(frames)}")
            print(f"  Audio shape: {audio.shape}")
            print(f"  Audio dtype: {audio.dtype}")
            print(f"  Audio range: [{audio.min():.3f}, {audio.max():.3f}]")
            print(f"  Audio mean: {audio.mean():.3f}")
            print(f"  Audio std: {audio.std():.3f}")
            
            duration = len(audio) / sample_rate
            
            print(f"Recorded {duration:.1f} seconds")
            
            return AudioData(
                raw_data=audio,
                metadata=AudioMetadata(
                    sample_rate=sample_rate,
                    channels=1,
                    duration=duration,
                    format="pcm"
                )
            )
            
        except Exception as e:
            print(f"Recording error: {e}")
            return None
    
    # ========================================================================
    # ベンチマーク
    # ========================================================================
    
    async def test_performance_benchmark(self):
        """パフォーマンスベンチマーク"""
        print("\n" + "="*60)
        print("TEST: Performance Benchmark")
        print("="*60)
        
        # テスト音声を用意（1秒の音声をシミュレート）
        test_audio = AudioData(
            raw_data=np.random.randn(16000).astype(np.float32) * 0.1,
            metadata=AudioMetadata(
                sample_rate=16000,
                channels=1,
                duration=1.0
            )
        )
        
        # 10回連続で認識を実行
        times = []
        
        print("\nRunning 10 iterations...")
        for i in range(10):
            start_time = time.time()
            
            try:
                result = await self.engine.transcribe(test_audio)
                elapsed = time.time() - start_time
                times.append(elapsed)
                print(f"  Iteration {i+1}: {elapsed:.3f}s")
                
            except Exception as e:
                print(f"  Iteration {i+1}: Failed - {e}")
        
        if times:
            avg_time = np.mean(times)
            min_time = np.min(times)
            max_time = np.max(times)
            std_time = np.std(times)
            
            print(f"\nResults:")
            print(f"  Average: {avg_time:.3f}s")
            print(f"  Min: {min_time:.3f}s")
            print(f"  Max: {max_time:.3f}s")
            print(f"  Std Dev: {std_time:.3f}s")
            
            self.reporter.add_result(TestResult(
                test_name="Performance Benchmark",
                status="PASS",
                duration=sum(times),
                details={
                    "iterations": len(times),
                    "avg_time": f"{avg_time:.3f}s",
                    "min_time": f"{min_time:.3f}s",
                    "max_time": f"{max_time:.3f}s"
                }
            ))
        else:
            self.reporter.add_result(TestResult(
                test_name="Performance Benchmark",
                status="FAIL",
                duration=0,
                details={
                    "iterations": 0,
                    "reason": "no_successful_iterations"
                },
                error="No successful iterations"
            ))
    
    # ========================================================================
    # メイン実行メソッド
    # ========================================================================
    
    async def test_model_comparison(self):
        """全モデルの精度比較テスト"""
        print("\n" + "="*60)
        print("TEST: Model Accuracy Comparison")
        print("="*60)
        
        # テストするモデル（サイズ順）
        models = {
            "tiny": 39,      # 39MB
            "base": 74,      # 74MB
            "small": 244,    # 244MB
            "medium": 769,   # 769MB
            "large-v3": 1550 # 1550MB
        }
        
        # 比較結果を保存
        comparison_results = []
        
        # 日本語音声でテスト
        test_file = self.test_files["japanese"]
        if not test_file.exists():
            print("⚠ Test file not found, skipping comparison")
            return
        
        audio_data = self.load_audio_file(test_file)
        if not audio_data:
            print("⚠ Failed to load audio file")
            return
        
        print(f"\nTesting with: {test_file.name}")
        print(f"Expected text: 'こんにちは、音声認識のテストです。'\n")
        
        for model_name, size_mb in models.items():
            print(f"\n{'='*40}")
            print(f"Model: {model_name} ({size_mb}MB)")
            print('='*40)
            
            try:
                # モデル切り替え
                self.engine.set_model(model_name)
                print(f"Initializing {model_name} model...")
                await self.engine.initialize()
                
                # 認識実行
                start_time = time.time()
                result = await self.engine.transcribe(audio_data)
                duration = time.time() - start_time
                
                # 結果を保存
                comparison_results.append({
                    "model": model_name,
                    "size_mb": size_mb,
                    "text": result.text,
                    "confidence": result.confidence,
                    "duration": duration
                })
                
                print(f"Text: '{result.text}'")
                print(f"Confidence: {result.confidence:.3f}")
                print(f"Processing Time: {duration:.2f}s")
                
                # 精度評価（簡易的な文字一致率）
                expected = "こんにちは、音声認識のテストです。"
                if result.text == expected:
                    print("Accuracy: Perfect match! ✓")
                else:
                    # 文字単位での類似度を計算
                    matching_chars = sum(1 for a, b in zip(result.text, expected) if a == b)
                    accuracy = matching_chars / max(len(result.text), len(expected))
                    print(f"Accuracy: {accuracy:.1%}")
                
            except Exception as e:
                print(f"✗ Error: {e}")
                comparison_results.append({
                    "model": model_name,
                    "size_mb": size_mb,
                    "error": str(e)
                })
        
        # 比較サマリー
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        print(f"{'Model':<12} {'Size':<8} {'Confidence':<12} {'Time':<8} {'Text'}")
        print("-"*60)
        
        for res in comparison_results:
            if "error" not in res:
                text_preview = res['text'][:30] + "..." if len(res['text']) > 30 else res['text']
                print(f"{res['model']:<12} {res['size_mb']:<8} {res['confidence']:<12.3f} {res['duration']:<8.2f} {text_preview}")
            else:
                print(f"{res['model']:<12} {res['size_mb']:<8} Error: {res['error'][:40]}")
        
        # 最高精度のモデルを特定
        valid_results = [r for r in comparison_results if "error" not in r]
        if valid_results:
            best_confidence = max(valid_results, key=lambda x: x['confidence'])
            fastest = min(valid_results, key=lambda x: x['duration'])
            
            print("\n" + "="*60)
            print("ANALYSIS")
            print("="*60)
            print(f"Highest Confidence: {best_confidence['model']} ({best_confidence['confidence']:.3f})")
            print(f"Fastest Processing: {fastest['model']} ({fastest['duration']:.2f}s)")
            
            # モデルサイズと精度の相関を分析
            print("\nModel Size vs Confidence:")
            for res in sorted(valid_results, key=lambda x: x['size_mb']):
                bar_length = int(res['confidence'] * 50)
                bar = "█" * bar_length
                print(f"{res['model']:10} ({res['size_mb']:4}MB): {bar} {res['confidence']:.3f}")
        
        # レポートとして結果を記録
        self.reporter.add_result(TestResult(
            test_name="Model Comparison",
            status="PASS" if valid_results else "FAIL",
            duration=sum(r.get('duration', 0) for r in comparison_results),
            details={
                "models_tested": len(comparison_results),
                "successful": len(valid_results),
                "best_model": best_confidence['model'] if valid_results else None
            }
        ))
    
    async def run_file_tests(self):
        """音声ファイルテストを実行"""
        print("\n" + "#"*60)
        print("# FILE-BASED TESTS")
        print("#"*60)
        
        await self.test_file_basic()
        await self.test_model_switching()
        await self.test_error_handling()
        await self.test_model_comparison()  # 精度比較テストを追加
    
    async def test_realtime_model_comparison(self):
        """リアルタイム音声での全モデル比較テスト"""
        print("\n" + "="*60)
        print("TEST: Realtime Model Comparison")
        print("="*60)
        
        if not SOUNDDEVICE_AVAILABLE:
            print("⚠ Skipping: sounddevice not available")
            return
        
        # テストフレーズ
        test_phrase = "人工知能による音声認識技術のテストをしています。"
        
        print(f"\nYou will say the SAME phrase for each model:")
        print(f"「{test_phrase}」")
        print("\nThis allows us to compare model accuracy on human speech.")
        
        # テストするモデル
        models = ["tiny", "base", "small", "medium", "large-v3"]
        
        comparison_results = []
        
        for model_name in models:
            print(f"\n{'='*40}")
            print(f"Testing with {model_name} model")
            print('='*40)
            
            try:
                # モデル切り替え
                self.engine.set_model(model_name)
                print(f"Initializing {model_name} model...")
                await self.engine.initialize()
                
                # 録音準備
                print(f"\nPlease say: 「{test_phrase}」")
                input(f"Press ENTER when ready to record with {model_name}...")
                
                # 録音
                audio_data = self.record_audio(duration=6.0)
                
                if not audio_data:
                    print(f"Recording failed for {model_name}")
                    continue
                
                # 音声レベル確認
                audio_level = np.abs(audio_data.raw_data).max()
                
                # 認識実行
                start_time = time.time()
                result = await self.engine.transcribe(audio_data)
                processing_time = time.time() - start_time
                
                # 精度計算
                accuracy = self._calculate_accuracy(result.text, test_phrase)
                
                # 結果を保存
                comparison_results.append({
                    "model": model_name,
                    "text": result.text,
                    "expected": test_phrase,
                    "accuracy": accuracy,
                    "confidence": result.confidence,
                    "processing_time": processing_time,
                    "audio_level": audio_level
                })
                
                print(f"Result:     '{result.text}'")
                print(f"Accuracy:   {accuracy:.1%}")
                print(f"Confidence: {result.confidence:.3f}")
                print(f"Time:       {processing_time:.2f}s")
                print(f"Audio Level: {audio_level:.3f}")
                
            except Exception as e:
                print(f"✗ Error with {model_name}: {e}")
                comparison_results.append({
                    "model": model_name,
                    "error": str(e)
                })
        
        # 比較分析レポート
        self._print_model_comparison_analysis(comparison_results, test_phrase)
        
        # レポートとして記録
        valid_results = [r for r in comparison_results if "error" not in r]
        if valid_results:
            best = max(valid_results, key=lambda x: x['accuracy'])
            self.reporter.add_result(TestResult(
                test_name="Realtime Model Comparison",
                status="PASS",
                duration=sum(r.get('processing_time', 0) for r in comparison_results),
                details={
                    "models_tested": len(comparison_results),
                    "successful": len(valid_results),
                    "best_model": best['model'],
                    "best_accuracy": best['accuracy']
                }
            ))
    
    def _print_model_comparison_analysis(self, results, expected_text):
        """モデル比較の詳細分析"""
        print("\n" + "="*60)
        print("MODEL COMPARISON ON HUMAN SPEECH")
        print("="*60)
        print(f"Expected: 「{expected_text}」")
        
        print("\n" + "-"*60)
        print("Results by Model:")
        print("-"*60)
        
        valid_results = [r for r in results if "error" not in r]
        
        # テーブル形式で表示
        print(f"\n{'Model':<12} {'Accuracy':<12} {'Confidence':<12} {'Time(s)':<10} {'Audio Level'}")
        print("-"*60)
        
        for res in valid_results:
            accuracy_str = f"{res['accuracy']:.1%}"
            confidence_str = f"{res['confidence']:.3f}"
            time_str = f"{res['processing_time']:.2f}"
            audio_str = f"{res['audio_level']:.3f}"
            
            # 最高精度にマーク
            mark = " ⭐" if res['accuracy'] == max(r['accuracy'] for r in valid_results) else ""
            
            print(f"{res['model']:<12} {accuracy_str:<12} {confidence_str:<12} {time_str:<10} {audio_str}{mark}")
        
        print("\n" + "-"*60)
        print("Transcription Details:")
        print("-"*60)
        
        for res in valid_results:
            status = "✓" if res['accuracy'] > 0.8 else "✗"
            print(f"\n[{status}] {res['model']}:")
            print(f"    '{res['text']}'")
            if res['accuracy'] < 1.0:
                # エラー箇所を特定
                expected_chars = list(expected_text)
                recognized_chars = list(res['text'])
                diff = []
                for i, (e, r) in enumerate(zip(expected_chars, recognized_chars)):
                    if e != r:
                        diff.append(f"位置{i}: '{e}'→'{r}'")
                if diff and len(diff) <= 3:
                    print(f"    Errors: {', '.join(diff)}")
        
        # 分析サマリー
        if valid_results:
            print("\n" + "="*60)
            print("ANALYSIS SUMMARY")
            print("="*60)
            
            best_accuracy = max(valid_results, key=lambda x: x['accuracy'])
            fastest = min(valid_results, key=lambda x: x['processing_time'])
            best_confidence = max(valid_results, key=lambda x: x['confidence'])
            
            print(f"Best Accuracy:   {best_accuracy['model']} ({best_accuracy['accuracy']:.1%})")
            print(f"Highest Confidence: {best_confidence['model']} ({best_confidence['confidence']:.3f})")
            print(f"Fastest:         {fastest['model']} ({fastest['processing_time']:.2f}s)")
            
            # 推奨事項
            print("\n" + "-"*60)
            print("RECOMMENDATION FOR HUMAN SPEECH:")
            print("-"*60)
            
            # baseモデルが80%以上の精度なら推奨
            base_result = next((r for r in valid_results if r['model'] == 'base'), None)
            if base_result and base_result['accuracy'] >= 0.8:
                print("✓ 'base' model provides good balance of speed and accuracy")
                print(f"  ({base_result['accuracy']:.1%} accuracy in {base_result['processing_time']:.2f}s)")
            elif best_accuracy['accuracy'] >= 0.9:
                print(f"✓ '{best_accuracy['model']}' model recommended for highest accuracy")
                print(f"  ({best_accuracy['accuracy']:.1%} accuracy)")
            else:
                print("⚠ Consider environmental factors (noise, microphone quality)")
                print(f"  Best achieved accuracy: {best_accuracy['accuracy']:.1%}")
    
    async def run_realtime_tests(self):
        """リアルタイムテストを実行"""
        print("\n" + "#"*60)
        print("# REALTIME TESTS")
        print("#"*60)
        
        await self.test_realtime_basic()
        await self.test_push_to_talk()
        await self.test_realtime_model_comparison()  # モデル比較を追加
    
    async def run_all_tests(self):
        """すべてのテストを実行"""
        # セットアップ
        if not await self.setup():
            print("\n⚠ Setup failed. Aborting tests.")
            return
        
        try:
            # ファイルベーステスト
            await self.run_file_tests()
            
            # リアルタイムテスト
            await self.run_realtime_tests()
            
            # ベンチマーク
            await self.test_performance_benchmark()
            
        finally:
            # クリーンアップ
            await self.cleanup()
            
            # サマリー表示
            self.reporter.generate_summary()
            
            # レポート保存
            report_path = project_root / "logs" / f"whisper_test_{datetime.now():%Y%m%d_%H%M%S}.json"
            report_path.parent.mkdir(exist_ok=True)
            self.reporter.save_report(report_path)


# ============================================================================
# メイン関数
# ============================================================================

async def main():
    """メインエントリーポイント"""
    parser = argparse.ArgumentParser(
        description="FasterWhisperEngine Manual Test Tool"
    )
    parser.add_argument(
        "--mode",
        choices=["file", "realtime", "all"],
        default="all",
        help="Test mode (default: all)"
    )
    parser.add_argument(
        "--model",
        default="base",
        help="Whisper model size (default: base)"
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use (auto/cpu/cuda)"
    )
    
    args = parser.parse_args()
    
    # ヘッダー表示
    print("="*60)
    print("FasterWhisperEngine Manual Test Tool")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Time: {datetime.now()}")
    
    # テスト実行
    tester = FasterWhisperManualTest(
        model_size=args.model,
        device=args.device
    )
    
    if args.mode == "file":
        if await tester.setup():
            await tester.run_file_tests()
            await tester.cleanup()
            tester.reporter.generate_summary()
    
    elif args.mode == "realtime":
        if await tester.setup():
            await tester.run_realtime_tests()
            await tester.cleanup()
            tester.reporter.generate_summary()
    
    else:  # all
        await tester.run_all_tests()


def check_dependencies():
    """依存関係をチェック"""
    print("\nChecking dependencies...")
    
    deps = {
        "numpy": NUMPY_AVAILABLE,
        "sounddevice": SOUNDDEVICE_AVAILABLE,
        "keyboard": KEYBOARD_AVAILABLE,
        "vioratalk": VIORATALK_AVAILABLE
    }
    
    all_ok = True
    for name, available in deps.items():
        status = "✓" if available else "✗"
        print(f"  {status} {name}")
        if not available and name in ["numpy", "vioratalk"]:
            all_ok = False
    
    if not all_ok:
        print("\n⚠ Critical dependencies are missing!")
        print("Please install required packages:")
        print("  pip install numpy")
        print("  pip install sounddevice  # for realtime tests")
        print("  pip install keyboard     # for push-to-talk (optional)")
        return False
    
    return True


if __name__ == "__main__":
    # 依存関係チェック
    if not check_dependencies():
        sys.exit(1)
    
    # 非同期実行
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
