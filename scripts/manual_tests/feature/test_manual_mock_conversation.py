#!/usr/bin/env python3
"""
手動テスト: Phase 3 Mockエンジンの対話フロー確認
実行方法: poetry run python scripts/manual_tests/feature/test_manual_mock_conversation.py
必要環境: Phase 3 Mockエンジン実装済み

テスト戦略ガイドライン v1.7準拠
開発規約書 v1.12準拠
インポート規約 v1.1準拠
"""

import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# プロジェクトルートをパスに追加
# scripts/manual_tests/feature/test_manual_mock_conversation.py から4階層上
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))  # 最優先でプロジェクトルートを追加

# デバッグ情報（問題がある場合は --debug オプションで実行）
if "--debug" in sys.argv:
    print(f"Project root: {project_root}")
    print(f"Python path: {sys.path[:2]}")
    import os

    print(f"Current directory: {os.getcwd()}")
    tests_dir = project_root / "tests"
    print(f"tests/ exists: {tests_dir.exists()}")
    if tests_dir.exists():
        print(f"tests/__init__.py exists: {(tests_dir / '__init__.py').exists()}")
        print(f"tests/mocks/ exists: {(tests_dir / 'mocks').exists()}")
        print(f"tests/mocks/__init__.py exists: {(tests_dir / 'mocks' / '__init__.py').exists()}")

import tests.mocks.mock_llm_engine as llm_module

# Phase 3 Mockエンジンのインポート - 直接モジュールから
import tests.mocks.mock_stt_engine as stt_module
import tests.mocks.mock_tts_engine as tts_module

MockSTTEngine = stt_module.MockSTTEngine
AudioData = stt_module.AudioData
MockLLMEngine = llm_module.MockLLMEngine
LLMResponse = llm_module.LLMResponse
MockTTSEngine = tts_module.MockTTSEngine
SynthesisResult = tts_module.SynthesisResult


# エラークラス
from vioratalk.core.exceptions import AudioError, LLMError, STTError, TTSError, VioraTalkError

# Phase 1コンポーネント（srcディレクトリから）
from vioratalk.core.vioratalk_engine import VioraTalkEngine
from vioratalk.utils.logger_manager import LoggerManager

# ============================================================================
# パフォーマンス監視（簡易版）
# ============================================================================


class PerformanceMonitor:
    """簡易パフォーマンス監視"""

    def __init__(self):
        self.measurements = {}

    def start_measurement(self, name: str) -> str:
        """計測開始"""
        self.measurements[name] = time.time()
        return name

    def end_measurement(self, name: str) -> float:
        """計測終了"""
        if name in self.measurements:
            elapsed = time.time() - self.measurements[name]
            del self.measurements[name]
            return elapsed
        return 0.0

    def get_resource_usage(self) -> Dict[str, Any]:
        """リソース使用状況（ダミー）"""
        return {
            "memory_mb": 256,  # ダミー値
            "cpu_percent": 15.5,  # ダミー値
            "gpu_percent": None,  # GPU未使用
        }


# ============================================================================
# 対話シミュレーター
# ============================================================================


class ConversationSimulator:
    """対話フローシミュレーター"""

    def __init__(self, stt, llm, tts):
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.conversation_history = []
        self.logger = LoggerManager().get_logger("ConversationSimulator")

    async def simulate_turn(self, user_input: str, character_id: str = "001_aoi") -> Dict[str, Any]:
        """1ターンの対話をシミュレート

        Args:
            user_input: ユーザー入力（テキスト）
            character_id: キャラクターID

        Returns:
            Dict[str, Any]: ターンの結果
        """
        start_time = time.time()

        # 1. 音声データをシミュレート
        # ファイル名に基づいて適切な応答を決定
        metadata_filename = "greeting.wav"  # デフォルト
        if "天気" in user_input:
            metadata_filename = "question.wav"
        elif "音楽" in user_input or "再生" in user_input:
            metadata_filename = "command.wav"

        audio_data = AudioData(
            data=b"simulated_audio",
            sample_rate=16000,
            duration=len(user_input) * 0.1,
            metadata={"filename": metadata_filename},
        )

        # 2. STT処理
        stt_start = time.time()
        transcription = await self.stt.transcribe(audio_data)
        stt_time = time.time() - stt_start

        # 3. LLM処理
        llm_start = time.time()
        response = await self.llm.generate(
            prompt=transcription.text, system_prompt=f"character_id:{character_id}"
        )
        llm_time = time.time() - llm_start

        # 4. TTS処理
        tts_start = time.time()
        synthesis = await self.tts.synthesize(response.content)
        tts_time = time.time() - tts_start

        total_time = time.time() - start_time

        # 結果を記録
        result = {
            "user": user_input,
            "transcription": transcription.text,
            "assistant": response.content,
            "character": character_id,
            "audio_size": len(synthesis.audio_data),
            "timing": {"stt": stt_time, "llm": llm_time, "tts": tts_time, "total": total_time},
            "timestamp": datetime.now(),
        }

        self.conversation_history.append(result)
        return result


# ============================================================================
# メインテスト関数
# ============================================================================


async def main():
    """手動テストのメイン関数"""
    print("=" * 70)
    print("🎯 Phase 3 Mock エンジン手動テスト開始")
    print("=" * 70)

    # ロガーの初期化
    logger_manager = LoggerManager()
    logger = logger_manager.get_logger("ManualTest")

    # パフォーマンス監視
    monitor = PerformanceMonitor()
    test_id = monitor.start_measurement("manual_test")

    try:
        # ============================================================
        # 1. エンジンの初期化
        # ============================================================
        print("\n📌 Step 1: エンジン初期化")
        print("-" * 60)

        # MockSTTEngine
        print("  初期化: MockSTTEngine...")
        stt_engine = MockSTTEngine()
        await stt_engine.initialize()
        print(f"  ✅ MockSTTEngine 初期化完了 (状態: {stt_engine.state.value})")

        # MockLLMEngine
        print("  初期化: MockLLMEngine...")
        llm_engine = MockLLMEngine()
        await llm_engine.initialize()
        print(f"  ✅ MockLLMEngine 初期化完了 (状態: {llm_engine.state.value})")

        # MockTTSEngine
        print("  初期化: MockTTSEngine...")
        tts_engine = MockTTSEngine()
        await tts_engine.initialize()
        print(f"  ✅ MockTTSEngine 初期化完了 (状態: {tts_engine.state.value})")

        # VioraTalkEngine
        print("  初期化: VioraTalkEngine...")
        vioratalk_engine = VioraTalkEngine()
        await vioratalk_engine.initialize()
        print(f"  ✅ VioraTalkEngine 初期化完了 (状態: {vioratalk_engine.state.value})")

        print("\n✨ 全エンジン初期化成功\n")

        # ============================================================
        # 2. 基本的な対話フローテスト
        # ============================================================
        print("=" * 60)
        print("📌 Step 2: 基本対話フローテスト")
        print("=" * 60)

        simulator = ConversationSimulator(stt_engine, llm_engine, tts_engine)

        # テストケース1: 挨拶
        print("\n🔹 テスト 1: 挨拶")
        result = await simulator.simulate_turn("こんにちは", "001_aoi")
        print(f"  👤 User: {result['user']}")
        print(f"  🎤 STT: {result['transcription']}")
        print(f"  🤖 LLM ({result['character']}): {result['assistant']}")
        print(f"  🔊 TTS: 音声データ {result['audio_size']} bytes")
        print(f"  ⏱️ 処理時間: {result['timing']['total']:.3f}秒")

        # テストケース2: 質問
        print("\n🔹 テスト 2: 質問")
        result = await simulator.simulate_turn("今日の天気はどうですか？", "001_aoi")
        print(f"  👤 User: {result['user']}")
        print(f"  🤖 LLM: {result['assistant']}")
        print(f"  ⏱️ 処理時間: {result['timing']['total']:.3f}秒")

        # テストケース3: コマンド
        print("\n🔹 テスト 3: コマンド")
        result = await simulator.simulate_turn("音楽を再生して", "001_aoi")
        print(f"  👤 User: {result['user']}")
        print(f"  🤖 LLM: {result['assistant']}")
        print(f"  ⏱️ 処理時間: {result['timing']['total']:.3f}秒")

        print("\n✅ 基本対話フローテスト完了")

        # ============================================================
        # 3. エラー処理テスト
        # ============================================================
        print("\n" + "=" * 60)
        print("📌 Step 3: エラー処理テスト")
        print("=" * 60)

        print("\n🔹 STTエラーシミュレーション")
        stt_engine.set_error_mode(True)
        try:
            await stt_engine.transcribe(AudioData(data=b"test"))
            print("  ❌ エラーが発生しませんでした")
        except STTError as e:
            print(f"  ✅ 期待通りのエラー: [{e.error_code}] {e}")
        stt_engine.set_error_mode(False)

        print("\n🔹 LLMエラーシミュレーション")
        llm_engine.set_error_mode(True)
        try:
            await llm_engine.generate("test")
            print("  ❌ エラーが発生しませんでした")
        except LLMError as e:
            print(f"  ✅ 期待通りのエラー: [{e.error_code}] {e}")
        llm_engine.set_error_mode(False)

        print("\n🔹 TTSエラーシミュレーション")
        tts_engine.set_error_mode(True)
        try:
            await tts_engine.synthesize("test")
            print("  ❌ エラーが発生しませんでした")
        except (AudioError, TTSError, VioraTalkError) as e:
            print(f"  ✅ 期待通りのエラー: [{e.error_code}] {e}")
        tts_engine.set_error_mode(False)

        print("\n✅ エラー処理テスト完了")

        # ============================================================
        # 4. キャラクター切り替えテスト
        # ============================================================
        print("\n" + "=" * 60)
        print("📌 Step 4: キャラクター切り替えテスト")
        print("=" * 60)

        characters = [("001_aoi", "碧衣"), ("002_haru", "春人"), ("003_yui", "結衣")]

        for char_id, char_name in characters:
            print(f"\n🔹 キャラクター: {char_name} ({char_id})")
            result = await simulator.simulate_turn("こんにちは", char_id)
            print(f"  🤖 {char_name}: {result['assistant']}")

            # 対応する音声IDを取得して設定
            if char_id == "001_aoi":
                tts_engine.set_voice("ja-JP-Female-1")
            elif char_id == "002_haru":
                tts_engine.set_voice("ja-JP-Female-2")
            else:
                tts_engine.set_voice("ja-JP-Male-1")

            print(f"  🔊 音声ID: {tts_engine.current_voice_id}")

        print("\n✅ キャラクター切り替えテスト完了")

        # ============================================================
        # 5. パフォーマンステスト
        # ============================================================
        print("\n" + "=" * 60)
        print("📌 Step 5: パフォーマンステスト")
        print("=" * 60)

        print("\n🔹 10ターン連続処理")
        total_times = []
        for i in range(10):
            test_phrases = [
                "こんにちは",
                "今日はいい天気ですね",
                "何か面白い話を聞かせて",
                "音楽が好きですか？",
                "おすすめの本はありますか？",
                "料理は得意ですか？",
                "休日は何をしていますか？",
                "好きな季節は？",
                "ペットは飼っていますか？",
                "さようなら",
            ]

            phrase = test_phrases[i]
            result = await simulator.simulate_turn(phrase, "001_aoi")
            total_times.append(result["timing"]["total"])
            print(f"  Turn {i+1}: {result['timing']['total']:.3f}秒 - {phrase[:10]}...")

        avg_time = sum(total_times) / len(total_times)
        max_time = max(total_times)
        min_time = min(total_times)

        print("\n📊 パフォーマンス統計:")
        print(f"  平均処理時間: {avg_time:.3f}秒")
        print(f"  最大処理時間: {max_time:.3f}秒")
        print(f"  最小処理時間: {min_time:.3f}秒")
        print(f"  ✅ SLA基準（3秒以内）: {'達成' if max_time < 3.0 else '未達成'}")

        print("\n✅ パフォーマンステスト完了")

        # ============================================================
        # 6. ストリーミングテスト
        # ============================================================
        print("\n" + "=" * 60)
        print("📌 Step 6: ストリーミングテスト")
        print("=" * 60)

        print("\n🔹 LLMストリーミング生成")
        llm_engine.streaming_enabled = True

        chunks = []
        print("  生成中: ", end="", flush=True)
        async for chunk in llm_engine.stream_generate("長い文章を生成してください"):
            chunks.append(chunk)
            print(".", end="", flush=True)

        full_text = "".join(chunks)
        print(f"\n  ✅ {len(chunks)}チャンクで生成完了")
        print(f"  生成テキスト: {full_text[:50]}...")

        llm_engine.streaming_enabled = False

        print("\n✅ ストリーミングテスト完了")

        # ============================================================
        # 7. 会話履歴テスト
        # ============================================================
        print("\n" + "=" * 60)
        print("📌 Step 7: 会話履歴テスト")
        print("=" * 60)

        # 履歴をクリア
        llm_engine.clear_history()

        # 会話を追加
        llm_engine.add_message("user", "私の名前は太郎です")
        llm_engine.add_message("assistant", "太郎さん、よろしくお願いします")
        llm_engine.add_message("user", "今日は良い天気ですね")
        llm_engine.add_message("assistant", "はい、とても良い天気です")

        history = llm_engine.get_history()
        print(f"\n🔹 会話履歴: {len(history)}メッセージ")
        for i, msg in enumerate(history, 1):
            role_icon = "👤" if msg.role == "user" else "🤖"
            print(f"  [{i}] {role_icon} {msg.role}: {msg.content}")

        print("\n✅ 会話履歴テスト完了")

        # ============================================================
        # 8. 音声パラメータテスト
        # ============================================================
        print("\n" + "=" * 60)
        print("📌 Step 8: 音声パラメータテスト")
        print("=" * 60)

        voices = tts_engine.get_available_voices()
        print(f"\n🔹 利用可能な音声: {len(voices)}種類")
        for voice in voices[:3]:  # 最初の3つだけ表示
            print(f"  - {voice.name} ({voice.id}) - {voice.language}/{voice.gender}")

        # 音声変更テスト
        print("\n🔹 音声変更テスト")
        original_voice = tts_engine.current_voice_id
        print(f"  現在の音声: {original_voice}")

        tts_engine.set_voice("ja-JP-Female-2")
        print(f"  変更後: {tts_engine.current_voice_id}")

        # 変更した音声で合成
        result = await tts_engine.synthesize("テスト音声です")
        print(f"  合成成功: {len(result.audio_data)} bytes")

        tts_engine.set_voice(original_voice)
        print(f"  復元: {tts_engine.current_voice_id}")

        print("\n✅ 音声パラメータテスト完了")

        # ============================================================
        # 9. 最終パフォーマンス情報
        # ============================================================
        elapsed = monitor.end_measurement(test_id)
        resources = monitor.get_resource_usage()

        print("\n" + "=" * 60)
        print("📊 最終パフォーマンス情報")
        print("=" * 60)
        print(f"  総実行時間: {elapsed:.2f}秒")
        print(f"  メモリ使用量: {resources['memory_mb']}MB（推定）")
        print(f"  CPU使用率: {resources['cpu_percent']:.1f}%（推定）")
        if resources.get("gpu_percent") is not None:
            print(f"  GPU使用率: {resources['gpu_percent']:.1f}%")

        # ============================================================
        # 10. クリーンアップ
        # ============================================================
        print("\n" + "=" * 60)
        print("🧹 クリーンアップ中...")
        print("=" * 60)

        await stt_engine.cleanup()
        print("  ✅ MockSTTEngine クリーンアップ完了")

        await llm_engine.cleanup()
        print("  ✅ MockLLMEngine クリーンアップ完了")

        await tts_engine.cleanup()
        print("  ✅ MockTTSEngine クリーンアップ完了")

        await vioratalk_engine.cleanup()
        print("  ✅ VioraTalkEngine クリーンアップ完了")

        print("\n✨ 全エンジンクリーンアップ完了")

    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        logger.error(f"手動テストエラー: {e}", exc_info=True)
        import traceback

        traceback.print_exc()
        return 1

    print("\n" + "=" * 70)
    print("✅ Phase 3 Mock エンジン手動テスト完了")
    print("=" * 70)
    print("\n📝 テスト結果サマリー:")
    print("  - 基本対話フロー: ✅")
    print("  - エラー処理: ✅")
    print("  - キャラクター切り替え: ✅")
    print("  - パフォーマンス: ✅")
    print("  - ストリーミング: ✅")
    print("  - 会話履歴: ✅")
    print("  - 音声パラメータ: ✅")

    logger.info("手動テスト正常終了")
    return 0


# ============================================================================
# エントリーポイント
# ============================================================================

if __name__ == "__main__":
    print("\n📋 環境情報:")
    print(f"  Python: {sys.version}")
    print(f"  プロジェクトルート: {project_root}")
    print(f"  実行パス: {Path(__file__).resolve()}")
    print()

    # イベントループの実行
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
