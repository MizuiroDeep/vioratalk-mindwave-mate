#!/usr/bin/env python3
"""
Pyttsx3Engine手動テストスクリプト

実際の音声出力を確認するための包括的なテストスクリプト。
Part 29調査結果の実環境検証を含む。

実行方法:
    poetry run python scripts/manual_tests/feature/test_manual_pyttsx3.py

オプション:
    --skip-audio: 音声出力をスキップ（CI環境用）
    --save-wav: WAVファイルを保存
    --stress-test: ストレステストを実行
    --verbose: 詳細ログを出力

テスト戦略ガイドライン v1.7準拠
開発規約書 v1.12準拠
"""

import argparse
import asyncio
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from vioratalk.core.tts.base import TTSConfig
from vioratalk.core.tts.pyttsx3_engine import Pyttsx3Engine
from vioratalk.utils.logger_manager import LoggerManager


class Pyttsx3ManualTester:
    """Pyttsx3Engineの手動テストクラス"""
    
    def __init__(self, args):
        """初期化
        
        Args:
            args: コマンドライン引数
        """
        self.args = args
        self.logger = LoggerManager().get_logger(self.__class__.__name__)
        self.results = []
        self.engine = None
        
        # テスト用ディレクトリ作成
        self.test_dir = Path("test_output/pyttsx3")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
    async def setup(self) -> bool:
        """テスト環境のセットアップ
        
        Returns:
            bool: セットアップ成功時True
        """
        try:
            self.logger.info("=" * 60)
            self.logger.info("Pyttsx3Engine Manual Test - Setup")
            self.logger.info("=" * 60)
            
            # エンジン作成
            config = TTSConfig(
                engine="pyttsx3",
                language="ja",
                save_audio_data=False,  # デフォルトは直接出力
                speed=1.0,
                volume=0.9
            )
            
            self.engine = Pyttsx3Engine(config=config)
            await self.engine.initialize()
            
            # 利用可能な音声を表示
            voices = self.engine.get_available_voices()
            self.logger.info(f"Available voices: {len(voices)}")
            for voice in voices:
                self.logger.info(f"  - {voice.id}: {voice.name} ({voice.language})")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Setup failed: {e}")
            traceback.print_exc()
            return False
    
    async def test_basic_synthesis(self) -> Dict:
        """基本的な音声合成テスト"""
        test_name = "Basic Synthesis"
        self.logger.info(f"\n[TEST] {test_name}")
        
        try:
            texts = [
                ("こんにちは、今日は良い天気ですね。", "ja"),
                ("Hello, how are you today?", "en"),
                ("VioraTalk音声合成エンジンのテストです。", "ja"),
            ]
            
            for text, lang in texts:
                self.logger.info(f"  Synthesizing ({lang}): {text[:30]}...")
                start_time = time.time()
                
                result = await self.engine.synthesize(text, voice_id=lang)
                
                elapsed = time.time() - start_time
                self.logger.info(f"    Duration: {result.duration:.2f}s, Elapsed: {elapsed:.3f}s")
                
                if not self.args.skip_audio:
                    await asyncio.sleep(result.duration + 0.5)
            
            return {"status": "PASS", "test": test_name}
            
        except Exception as e:
            self.logger.error(f"Test failed: {e}")
            return {"status": "FAIL", "test": test_name, "error": str(e)}
    
    async def test_multiple_synthesis(self) -> Dict:
        """Part 29問題：複数回合成の安定性テスト"""
        test_name = "Multiple Synthesis (Part 29 Issue)"
        self.logger.info(f"\n[TEST] {test_name}")
        
        try:
            # 10回連続で合成を行う
            test_count = 10
            success_count = 0
            
            for i in range(test_count):
                text = f"テスト{i + 1}回目です。"
                self.logger.info(f"  Attempt {i + 1}/{test_count}: {text}")
                
                try:
                    result = await self.engine.synthesize(text)
                    success_count += 1
                    
                    if not self.args.skip_audio:
                        await asyncio.sleep(0.5)  # 短い間隔を空ける
                        
                except Exception as e:
                    self.logger.error(f"    Failed at attempt {i + 1}: {e}")
            
            success_rate = (success_count / test_count) * 100
            self.logger.info(f"  Success rate: {success_count}/{test_count} ({success_rate:.1f}%)")
            
            if success_rate >= 90:
                return {"status": "PASS", "test": test_name, "success_rate": success_rate}
            else:
                return {"status": "FAIL", "test": test_name, "success_rate": success_rate}
                
        except Exception as e:
            self.logger.error(f"Test failed: {e}")
            return {"status": "FAIL", "test": test_name, "error": str(e)}
    
    async def test_wav_generation(self) -> Dict:
        """WAVファイル生成モードのテスト"""
        test_name = "WAV File Generation"
        self.logger.info(f"\n[TEST] {test_name}")
        
        try:
            texts = [
                "WAVファイル生成テストです。",
                "This is a WAV file generation test.",
            ]
            
            for i, text in enumerate(texts):
                self.logger.info(f"  Generating WAV for: {text[:30]}...")
                
                # save_audioパラメータで一時的にWAVモードに切り替え
                result = await self.engine.synthesize(text, save_audio=True)
                
                if result.audio_data:
                    file_path = self.test_dir / f"test_{i + 1}.wav"
                    with open(file_path, "wb") as f:
                        f.write(result.audio_data)
                    
                    file_size = len(result.audio_data)
                    self.logger.info(f"    Saved: {file_path.name} ({file_size:,} bytes)")
                else:
                    self.logger.warning("    No audio data generated")
            
            return {"status": "PASS", "test": test_name}
            
        except Exception as e:
            self.logger.error(f"Test failed: {e}")
            return {"status": "FAIL", "test": test_name, "error": str(e)}
    
    async def test_special_cases(self) -> Dict:
        """特殊ケースのテスト"""
        test_name = "Special Cases"
        self.logger.info(f"\n[TEST] {test_name}")
        
        special_texts = [
            ("", "Empty text"),
            ("   ", "Whitespace only"),
            ("あ", "Single character"),
            ("🎵♪🎶", "Emoji/symbols"),
            ("「こんにちは」と言いました。", "Japanese quotes"),
            ("100円のコーヒー", "Numbers in text"),
            ("a" * 500, "Long text (500 chars)"),
            ("改行を\n含む\nテキスト", "Text with newlines"),
            ("タブ\tを含む\tテキスト", "Text with tabs"),
        ]
        
        failures = []
        
        for text, description in special_texts:
            try:
                self.logger.info(f"  Testing: {description}")
                result = await self.engine.synthesize(text)
                
                if text.strip():
                    assert result.duration > 0, "Duration should be positive for non-empty text"
                else:
                    assert result.duration == 0, "Duration should be 0 for empty text"
                
                self.logger.info(f"    ✓ Duration: {result.duration:.2f}s")
                
            except Exception as e:
                self.logger.error(f"    ✗ Failed: {e}")
                failures.append((description, str(e)))
        
        if failures:
            return {"status": "PARTIAL", "test": test_name, "failures": failures}
        else:
            return {"status": "PASS", "test": test_name}
    
    async def test_voice_switching(self) -> Dict:
        """音声切り替えテスト"""
        test_name = "Voice Switching"
        self.logger.info(f"\n[TEST] {test_name}")
        
        try:
            # 日本語 → 英語 → 日本語の切り替え
            test_sequences = [
                ("ja", "こんにちは、日本語の音声です。"),
                ("en", "Hello, this is English voice."),
                ("ja", "日本語に戻りました。"),
                ("en", "Back to English again."),
            ]
            
            for voice_id, text in test_sequences:
                self.logger.info(f"  Voice: {voice_id}, Text: {text[:30]}...")
                
                # 音声を切り替え
                self.engine.set_voice(voice_id)
                result = await self.engine.synthesize(text)
                
                assert result.metadata["voice_id"] == voice_id
                self.logger.info(f"    ✓ Voice ID confirmed: {voice_id}")
                
                if not self.args.skip_audio:
                    await asyncio.sleep(result.duration + 0.5)
            
            return {"status": "PASS", "test": test_name}
            
        except Exception as e:
            self.logger.error(f"Test failed: {e}")
            return {"status": "FAIL", "test": test_name, "error": str(e)}
    
    async def test_parameters(self) -> Dict:
        """音声パラメータ調整テスト"""
        test_name = "Voice Parameters"
        self.logger.info(f"\n[TEST] {test_name}")
        
        try:
            base_text = "パラメータテストです。速度と音量を変更します。"
            
            # 異なるパラメータでテスト
            parameter_sets = [
                (1.0, 0.9, "Normal"),
                (1.5, 0.9, "Fast (1.5x)"),
                (0.7, 0.9, "Slow (0.7x)"),
                (1.0, 0.5, "Low volume"),
                (2.0, 1.0, "Very fast, max volume"),
            ]
            
            for speed, volume, description in parameter_sets:
                self.logger.info(f"  Testing: {description}")
                
                # 新しい設定でエンジンを再作成
                config = TTSConfig(
                    engine="pyttsx3",
                    language="ja",
                    speed=speed,
                    volume=volume
                )
                temp_engine = Pyttsx3Engine(config=config)
                await temp_engine.initialize()
                
                result = await temp_engine.synthesize(base_text)
                
                # 推定時間の検証
                expected_duration = len(base_text) * 0.1 / speed
                self.logger.info(f"    Duration: {result.duration:.2f}s (expected: ~{expected_duration:.2f}s)")
                
                await temp_engine.cleanup()
                
                if not self.args.skip_audio:
                    await asyncio.sleep(result.duration + 0.5)
            
            return {"status": "PASS", "test": test_name}
            
        except Exception as e:
            self.logger.error(f"Test failed: {e}")
            return {"status": "FAIL", "test": test_name, "error": str(e)}
    
    async def test_concurrent_requests(self) -> Dict:
        """並行リクエストテスト"""
        test_name = "Concurrent Requests"
        self.logger.info(f"\n[TEST] {test_name}")
        
        try:
            # 5つの並行リクエスト
            tasks = []
            texts = [
                "並行リクエスト1",
                "並行リクエスト2", 
                "並行リクエスト3",
                "並行リクエスト4",
                "並行リクエスト5",
            ]
            
            self.logger.info(f"  Sending {len(texts)} concurrent requests...")
            start_time = time.time()
            
            for text in texts:
                task = self.engine.synthesize(text)
                tasks.append(task)
            
            # すべてのタスクを同時に実行
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            elapsed = time.time() - start_time
            
            # 結果の検証
            success_count = sum(1 for r in results if not isinstance(r, Exception))
            self.logger.info(f"  Completed in {elapsed:.2f}s")
            self.logger.info(f"  Success: {success_count}/{len(texts)}")
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"    Task {i + 1}: Failed - {result}")
                else:
                    self.logger.info(f"    Task {i + 1}: Success - Duration {result.duration:.2f}s")
            
            if success_count == len(texts):
                return {"status": "PASS", "test": test_name}
            else:
                return {"status": "PARTIAL", "test": test_name, "success_rate": success_count / len(texts)}
                
        except Exception as e:
            self.logger.error(f"Test failed: {e}")
            return {"status": "FAIL", "test": test_name, "error": str(e)}
    
    async def test_stress_test(self) -> Dict:
        """ストレステスト（オプション）"""
        test_name = "Stress Test"
        self.logger.info(f"\n[TEST] {test_name}")
        
        if not self.args.stress_test:
            self.logger.info("  Skipped (use --stress-test to enable)")
            return {"status": "SKIP", "test": test_name}
        
        try:
            iterations = 50
            success_count = 0
            total_time = 0
            
            self.logger.info(f"  Running {iterations} iterations...")
            
            for i in range(iterations):
                text = f"ストレステスト {i + 1}/{iterations}"
                
                try:
                    start = time.time()
                    result = await self.engine.synthesize(text, save_audio=False)
                    elapsed = time.time() - start
                    
                    total_time += elapsed
                    success_count += 1
                    
                    if (i + 1) % 10 == 0:
                        self.logger.info(f"    Progress: {i + 1}/{iterations}")
                        
                except Exception as e:
                    self.logger.error(f"    Failed at iteration {i + 1}: {e}")
                    
                # 少し待機（エンジンの負荷を下げる）
                await asyncio.sleep(0.1)
            
            avg_time = total_time / iterations if iterations > 0 else 0
            success_rate = (success_count / iterations) * 100
            
            self.logger.info(f"  Results:")
            self.logger.info(f"    Success rate: {success_count}/{iterations} ({success_rate:.1f}%)")
            self.logger.info(f"    Average time: {avg_time:.3f}s")
            
            if success_rate >= 95:
                return {"status": "PASS", "test": test_name, "metrics": {
                    "success_rate": success_rate,
                    "avg_time": avg_time
                }}
            else:
                return {"status": "FAIL", "test": test_name, "success_rate": success_rate}
                
        except Exception as e:
            self.logger.error(f"Test failed: {e}")
            return {"status": "FAIL", "test": test_name, "error": str(e)}
    
    async def test_error_recovery(self) -> Dict:
        """エラーリカバリーテスト"""
        test_name = "Error Recovery"
        self.logger.info(f"\n[TEST] {test_name}")
        
        try:
            # 無効な音声IDでエラーを発生させる
            self.logger.info("  Testing invalid voice ID...")
            try:
                await self.engine.synthesize("テスト", voice_id="invalid_voice")
                self.logger.error("    Expected error did not occur")
                return {"status": "FAIL", "test": test_name, "reason": "No error for invalid voice"}
            except Exception as e:
                self.logger.info(f"    ✓ Error caught: {e}")
            
            # エラー後も正常に動作することを確認
            self.logger.info("  Testing recovery after error...")
            result = await self.engine.synthesize("エラー後の正常動作テスト")
            assert result is not None
            assert result.duration > 0
            self.logger.info("    ✓ Recovery successful")
            
            return {"status": "PASS", "test": test_name}
            
        except Exception as e:
            self.logger.error(f"Test failed: {e}")
            return {"status": "FAIL", "test": test_name, "error": str(e)}
    
    async def test_resource_management(self) -> Dict:
        """リソース管理テスト"""
        test_name = "Resource Management"
        self.logger.info(f"\n[TEST] {test_name}")
        
        try:
            # 複数のエンジンを作成・破棄
            self.logger.info("  Creating and destroying multiple engines...")
            
            for i in range(5):
                config = TTSConfig(engine="pyttsx3", language="ja")
                temp_engine = Pyttsx3Engine(config=config)
                await temp_engine.initialize()
                
                result = await temp_engine.synthesize(f"エンジン{i + 1}")
                assert result is not None
                
                await temp_engine.cleanup()
                self.logger.info(f"    Engine {i + 1}: Created, used, and cleaned up")
            
            self.logger.info("  ✓ All engines properly managed")
            
            return {"status": "PASS", "test": test_name}
            
        except Exception as e:
            self.logger.error(f"Test failed: {e}")
            return {"status": "FAIL", "test": test_name, "error": str(e)}
    
    async def run_all_tests(self) -> None:
        """すべてのテストを実行"""
        if not await self.setup():
            self.logger.error("Setup failed, aborting tests")
            return
        
        # テストリスト
        tests = [
            self.test_basic_synthesis,
            self.test_multiple_synthesis,
            self.test_wav_generation,
            self.test_special_cases,
            self.test_voice_switching,
            self.test_parameters,
            self.test_concurrent_requests,
            self.test_error_recovery,
            self.test_resource_management,
            self.test_stress_test,
        ]
        
        # 各テストを実行
        for test_func in tests:
            try:
                result = await test_func()
                self.results.append(result)
            except Exception as e:
                self.logger.error(f"Unexpected error in {test_func.__name__}: {e}")
                self.results.append({
                    "status": "ERROR",
                    "test": test_func.__name__,
                    "error": str(e)
                })
        
        # クリーンアップ
        if self.engine:
            await self.engine.cleanup()
        
        # 結果サマリー
        self.print_summary()
    
    def print_summary(self) -> None:
        """テスト結果のサマリーを表示"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("TEST SUMMARY")
        self.logger.info("=" * 60)
        
        pass_count = sum(1 for r in self.results if r["status"] == "PASS")
        fail_count = sum(1 for r in self.results if r["status"] == "FAIL")
        partial_count = sum(1 for r in self.results if r["status"] == "PARTIAL")
        skip_count = sum(1 for r in self.results if r["status"] == "SKIP")
        error_count = sum(1 for r in self.results if r["status"] == "ERROR")
        
        for result in self.results:
            status = result["status"]
            test_name = result["test"]
            
            if status == "PASS":
                symbol = "✓"
            elif status == "FAIL":
                symbol = "✗"
            elif status == "PARTIAL":
                symbol = "△"
            elif status == "SKIP":
                symbol = "○"
            else:
                symbol = "!"
            
            self.logger.info(f"{symbol} {test_name}: {status}")
            
            if "error" in result:
                self.logger.info(f"    Error: {result['error']}")
            if "success_rate" in result:
                self.logger.info(f"    Success rate: {result['success_rate']:.1f}%")
        
        self.logger.info("-" * 60)
        self.logger.info(f"Total: {len(self.results)} tests")
        self.logger.info(f"  PASS: {pass_count}")
        self.logger.info(f"  FAIL: {fail_count}")
        self.logger.info(f"  PARTIAL: {partial_count}")
        self.logger.info(f"  SKIP: {skip_count}")
        self.logger.info(f"  ERROR: {error_count}")
        
        if fail_count == 0 and error_count == 0:
            self.logger.info("\n✅ All tests completed successfully!")
        else:
            self.logger.warning(f"\n⚠️ {fail_count + error_count} tests failed")


async def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="Pyttsx3Engine Manual Test Script"
    )
    parser.add_argument(
        "--skip-audio",
        action="store_true",
        help="Skip actual audio output (for CI environments)"
    )
    parser.add_argument(
        "--save-wav",
        action="store_true",
        help="Save generated WAV files"
    )
    parser.add_argument(
        "--stress-test",
        action="store_true",
        help="Run stress test (50 iterations)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # ログレベル設定
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    # テスター実行
    tester = Pyttsx3ManualTester(args)
    await tester.run_all_tests()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        traceback.print_exc()
        sys.exit(1)
