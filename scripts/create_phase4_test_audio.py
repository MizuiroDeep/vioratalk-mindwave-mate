#!/usr/bin/env python3
"""
Phase 4用テスト音声ファイル生成スクリプト

Pyttsx3を使用してSTT→LLM→TTS統合テスト用の音声ファイルを生成します。
各音声は音声認識の精度とLLMの応答品質を同時にテストできる内容になっています。

実行方法:
    python scripts/create_phase4_test_audio.py

生成ファイル:
    - tests/fixtures/audio/input/test_japanese.wav: 数字認識＋事実確認
    - tests/fixtures/audio/input/test_english.wav: 英語＋計算要求
    - tests/fixtures/audio/input/test_long_japanese.wav: 時刻認識＋推論
    - tests/fixtures/audio/input/test_mixed.wav: 技術用語＋列挙要求
    - tests/fixtures/audio/input/test_complex.wav: 複雑な数値と単位変換
    - tests/fixtures/audio/input/silence.wav: 無音ファイル

テスト項目:
    STT: 数字、時刻、金額、単位、英語混在、専門用語
    LLM: 事実確認、計算、推論、技術的回答、単位変換

注意:
    以前のgenerate_test_audio.pyとは別のスクリプトです。
    Phase 4専用のテストファイルを生成します。
"""

import sys
import wave
import struct
import tempfile
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pyttsx3


def create_silence_wav(output_path: Path, duration: float = 3.0) -> None:
    """無音のWAVファイルを作成
    
    Args:
        output_path: 出力ファイルパス
        duration: 無音の長さ（秒）
    """
    print(f"Creating silence WAV: {output_path}")
    
    sample_rate = 16000
    num_samples = int(sample_rate * duration)
    
    # 完全な無音データ（若干のノイズを加える）
    samples = np.random.normal(0, 0.001, num_samples)
    
    # WAVファイルとして保存
    with wave.open(str(output_path), 'wb') as wav_file:
        wav_file.setnchannels(1)  # モノラル
        wav_file.setsampwidth(2)   # 16bit
        wav_file.setframerate(sample_rate)
        
        # int16に変換
        samples_int16 = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
        wav_file.writeframes(samples_int16.tobytes())
    
    print(f"  ✓ Created: {duration}秒の無音ファイル")


def create_test_audio_with_pyttsx3(
    text: str,
    output_path: Path,
    language: str = "ja",
    rate: int = 150
) -> None:
    """Pyttsx3でテスト音声を生成
    
    Args:
        text: 読み上げるテキスト
        output_path: 出力ファイルパス
        language: 言語（ja/en）
        rate: 読み上げ速度
    """
    print(f"Creating {language} audio: {output_path}")
    print(f"  Text: {text}")
    
    try:
        # Pyttsx3エンジンの初期化
        engine = pyttsx3.init()
        
        # 設定
        engine.setProperty('rate', rate)  # 読み上げ速度
        engine.setProperty('volume', 0.9)  # 音量
        
        # 利用可能な音声を取得して表示
        voices = engine.getProperty('voices')
        print(f"  利用可能な音声: {len(voices)}個")
        
        # デバッグ用：すべての音声を表示
        for i, voice in enumerate(voices):
            print(f"    [{i}] {voice.name}")
            print(f"        ID: {voice.id}")
        
        # 言語に応じて音声を選択
        selected_voice = None
        if language == 'en':
            # 英語音声を探す（改善版）
            for voice in voices:
                voice_id_lower = voice.id.lower()
                voice_name_lower = voice.name.lower() if voice.name else ""
                
                # より広い条件で英語音声を検出
                if any(keyword in voice_id_lower for keyword in ['english', 'en-us', 'en_us', 'david', 'zira']):
                    selected_voice = voice
                    print(f"  → 英語音声を選択: {voice.name}")
                    break
                elif any(keyword in voice_name_lower for keyword in ['english', 'david', 'zira']):
                    selected_voice = voice
                    print(f"  → 英語音声を選択: {voice.name}")
                    break
            
            if not selected_voice and voices:
                print(f"  ⚠ 英語音声が見つかりません。手動で選択してください。")
                print(f"  使用したい音声の番号を入力 (0-{len(voices)-1}): ", end="")
                try:
                    choice = int(input())
                    if 0 <= choice < len(voices):
                        selected_voice = voices[choice]
                        print(f"  → 手動選択: {selected_voice.name}")
                except:
                    selected_voice = voices[0]
                    print(f"  → デフォルト音声を使用: {selected_voice.name}")
        else:
            # 日本語音声を探す
            for voice in voices:
                voice_id_lower = voice.id.lower()
                voice_name_lower = voice.name.lower() if voice.name else ""
                
                if any(keyword in voice_id_lower for keyword in ['japanese', 'ja-jp', 'ja_jp', 'haruka', 'sayaka']):
                    selected_voice = voice
                    print(f"  → 日本語音声を選択: {voice.name}")
                    break
                elif any(keyword in voice_name_lower for keyword in ['japanese', 'haruka', 'sayaka']):
                    selected_voice = voice
                    print(f"  → 日本語音声を選択: {voice.name}")
                    break
            
            if not selected_voice and voices:
                selected_voice = voices[0]
                print(f"  → デフォルト音声を使用: {selected_voice.name}")
        
        if selected_voice:
            engine.setProperty('voice', selected_voice.id)
        
        # ファイルに保存
        engine.save_to_file(text, str(output_path))
        engine.runAndWait()
        
        print(f"  ✓ Created: {output_path.name}")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        print(f"  代替として簡易的な音声ファイルを生成します...")
        
        # エラー時は簡易的な音声ファイル（トーン）を生成
        create_tone_wav(output_path, duration=2.0)


def create_tone_wav(output_path: Path, duration: float = 2.0, frequency: float = 440) -> None:
    """トーン音声のWAVファイルを作成（フォールバック用）
    
    Args:
        output_path: 出力ファイルパス
        duration: 音声の長さ（秒）
        frequency: 周波数（Hz）
    """
    sample_rate = 16000
    num_samples = int(sample_rate * duration)
    
    # サイン波を生成
    t = np.linspace(0, duration, num_samples)
    samples = np.sin(2 * np.pi * frequency * t) * 0.3  # 音量を抑える
    
    # エンベロープを適用（フェードイン・フェードアウト）
    envelope = np.ones_like(samples)
    fade_samples = int(sample_rate * 0.1)  # 0.1秒のフェード
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
    samples *= envelope
    
    # WAVファイルとして保存
    with wave.open(str(output_path), 'wb') as wav_file:
        wav_file.setnchannels(1)  # モノラル
        wav_file.setsampwidth(2)   # 16bit
        wav_file.setframerate(sample_rate)
        
        # int16に変換
        samples_int16 = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
        wav_file.writeframes(samples_int16.tobytes())
    
    print(f"  ✓ Created tone WAV as fallback")


def main():
    """メイン処理"""
    print("="*60)
    print("Phase 4 STT→LLM→TTS統合テスト音声ファイル生成")
    print("="*60)
    
    # 出力ディレクトリの作成（正しいパス：fixtures/audio/input/）
    audio_dir = project_root / "tests" / "fixtures" / "audio" / "input"
    audio_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n出力ディレクトリ: {audio_dir}")
    print("(Phase 4 STT→LLM→TTS統合テスト用)\n")
    
    # テスト音声の定義（Phase 4 STT+LLM統合テスト用）
    test_cases = [
        {
            "name": "test_japanese.wav",
            "text": "2024年の日本の人口は約1億2千万人ですが、これは正しいですか？",
            "language": "ja",
            "description": "数字認識＋事実確認"
        },
        {
            "name": "test_english.wav",
            "text": "Please calculate 15 percent of 240 dollars",
            "language": "en",
            "description": "英語＋計算要求"
        },
        {
            "name": "test_long_japanese.wav",
            "text": "私は毎朝6時30分に起きて、7時15分の電車に乗ります。通勤時間はどのくらいか計算できますか？",
            "language": "ja",
            "description": "時刻認識＋推論"
        },
        {
            "name": "test_mixed.wav",
            "text": "APIのresponse timeが500msを超えた場合の対処法を3つ教えて",
            "language": "ja",
            "description": "技術用語＋数値＋列挙要求"
        },
        {
            "name": "test_complex.wav",
            "text": "東京から大阪まで新幹線で2時間30分、料金は13,320円です。これをマイルに換算すると何マイルですか？",
            "language": "ja",
            "description": "複雑な数値と単位変換"
        }
    ]
    
    # 各テスト音声を生成
    print("テスト音声を生成中...")
    print("(STTとLLMの統合テスト用に最適化された内容)\n")
    
    for test_case in test_cases:
        output_path = audio_dir / test_case["name"]
        print(f"[{test_case['description']}]")
        create_test_audio_with_pyttsx3(
            text=test_case["text"],
            output_path=output_path,
            language=test_case["language"]
        )
        print()
    
    # 無音ファイルの生成
    print("[無音ファイル（エラー処理テスト用）]")
    create_silence_wav(audio_dir / "silence.wav", duration=3.0)
    
    print("\n" + "="*60)
    print("✅ Phase 4用テスト音声ファイルを生成しました！")
    print("="*60)
    
    # 生成されたファイルのリスト
    print("\n生成されたファイル:")
    expected_files = [
        "test_japanese.wav",
        "test_english.wav", 
        "test_long_japanese.wav",
        "test_mixed.wav",
        "test_complex.wav",
        "silence.wav"
    ]
    
    for filename in expected_files:
        file_path = audio_dir / filename
        if file_path.exists():
            size_kb = file_path.stat().st_size / 1024
            print(f"  ✓ {filename} ({size_kb:.1f} KB)")
        else:
            print(f"  ✗ {filename} (生成失敗)")
    
    print("\n次のステップ:")
    print("1. 生成された音声ファイルを確認してください")
    print("   場所: tests/fixtures/audio/input/")
    print("2. 必要に応じて手動で録音した音声と差し替えてください")
    print("3. 統合テストを実行してください:")
    print("   pytest tests/integration/test_phase4_real_engines.py -v")


if __name__ == "__main__":
    main()
